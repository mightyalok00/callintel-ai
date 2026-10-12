"""
Integration tests for real-time live WebSocket audio streaming.
Verifies:
1. Real audio chunks streaming (PCM/WAV)
2. Transcript timing and monotonic timestamp advancement
3. Connection interruption handling & abrupt disconnects
4. Error handling with corrupted / malformed payloads
5. Clean session cleanup and resource management
6. Bounded buffer backpressure limits (512KB chunk limit)
7. State lifecycle transitions (CONNECTED -> RECORDING -> COMPLETED)
"""

from pathlib import Path
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient

from src.callintel.api.app import app
from src.callintel.api.websocket_live import manager, StreamSessionStatus

client = TestClient(app)
SAMPLE_WAV = Path("data/samples/sample_customer_call.wav")


def test_real_audio_chunk_streaming_and_timing():
    """Streams real audio chunks from sample WAV and verifies monotonic transcript timing."""
    assert SAMPLE_WAV.exists(), "Sample WAV missing."

    from src.callintel.speech.audio_utils import load_and_normalize_audio
    audio_float, sr, _ = load_and_normalize_audio(SAMPLE_WAV, target_sample_rate=16000)
    audio_pcm16 = (audio_float * 32767).astype(np.int16)

    # Slice into two exact 1.0 second chunks (16000 samples each)
    chunk_1 = audio_pcm16[:16000].tobytes()
    chunk_2 = audio_pcm16[16000:32000].tobytes()

    session_id = "test_live_audio_session_01"

    with client.websocket_connect(f"/api/v1/stream/{session_id}") as ws:
        assert manager.has_session(session_id) is True

        init_frame = ws.receive_json()
        assert init_frame["type"] == "SESSION_INITIALIZED"
        assert init_frame["status"] in ["CONNECTED", "READY_FOR_AUDIO_STREAM"]
        assert "limits" in init_frame

        # Send chunk 1 (0.0s to 1.0s)
        ws.send_bytes(chunk_1)
        resp_1 = ws.receive_json()

        assert resp_1["type"] == "STREAM_AUDIO_PROCESSED"
        assert resp_1["turn_index"] == 1
        assert resp_1["start_seconds"] == 0.0
        assert resp_1["end_seconds"] == 1.0
        assert resp_1["duration_seconds"] == 1.0
        assert "emotion" in resp_1
        assert "sentiment" in resp_1
        assert "latency_ms" in resp_1

        # Send chunk 2 (1.0s to 2.0s)
        ws.send_bytes(chunk_2)
        resp_2 = ws.receive_json()

        assert resp_2["type"] == "STREAM_AUDIO_PROCESSED"
        assert resp_2["turn_index"] == 2
        # Monotonic timestamp check: chunk 2 starts at or after chunk 1 end
        assert resp_2["start_seconds"] == resp_1["end_seconds"]
        assert resp_2["end_seconds"] == 2.0
        assert resp_2["end_seconds"] > resp_2["start_seconds"]

    # Verify session cleanup immediately upon exiting websocket context
    assert manager.has_session(session_id) is False


def test_bounded_buffer_backpressure():
    """Sends an oversized chunk (>512KB) and asserts recoverable backpressure error and dropped counter."""
    session_id = "test_bounded_buffer_session"

    with client.websocket_connect(f"/api/v1/stream/{session_id}") as ws:
        ws.receive_json()  # SESSION_INITIALIZED

        # 600 KB payload exceeding 512 KB limit
        oversized_chunk = b"\x00" * (600 * 1024)
        ws.send_bytes(oversized_chunk)

        err_resp = ws.receive_json()
        assert err_resp["type"] == "STREAM_ERROR"
        assert err_resp["error_code"] == "CHUNK_SIZE_LIMIT_EXCEEDED"
        assert err_resp["recoverable"] is True
        assert err_resp["dropped_chunks"] == 1

    assert manager.has_session(session_id) is False


def test_connection_interruption_and_session_cleanup():
    """Simulates an abrupt connection interruption and verifies complete session cleanup."""
    session_id = "test_abrupt_disconnect_session_02"

    with client.websocket_connect(f"/api/v1/stream/{session_id}") as ws:
        assert manager.has_session(session_id) is True
        ws.receive_json()  # SESSION_INITIALIZED

        # Send a text turn
        ws.send_text("Hello agent, can you hear me?")
        resp = ws.receive_json()
        assert resp["type"] in ["TEXT_TURN_SCORED", "STREAM_TURN_EMITTED"]

    # Context exited without explicit CLOSE_SESSION - simulates socket disconnect
    assert manager.has_session(session_id) is False


def test_error_handling_and_recovery():
    """Sends invalid/corrupt audio frames and verifies server returns structured recoverable errors."""
    session_id = "test_error_handling_session_03"

    with client.websocket_connect(f"/api/v1/stream/{session_id}") as ws:
        ws.receive_json()  # SESSION_INITIALIZED

        # Send malformed base64 audio payload
        ws.send_json({
            "type": "AUDIO_CHUNK",
            "audio_base64": "!!!INVALID_BASE64_BYTES###",
        })
        err_resp = ws.receive_json()
        assert err_resp["type"] == "STREAM_ERROR"
        assert err_resp["recoverable"] is True

        # Verify session is still alive and can process subsequent valid frames
        ws.send_text("I am following up on my previous billing question.")
        valid_resp = ws.receive_json()
        assert valid_resp["type"] in ["TEXT_TURN_SCORED", "STREAM_TURN_EMITTED"]
        assert "billing" in valid_resp["text"].lower()

    assert manager.has_session(session_id) is False


def test_heartbeat_and_graceful_close():
    """Tests ping/pong heartbeat, graceful session closure, and latency reporting."""
    session_id = "test_graceful_session_04"

    with client.websocket_connect(f"/api/v1/stream/{session_id}") as ws:
        ws.receive_json()  # SESSION_INITIALIZED

        # Ping
        ws.send_json({"type": "PING"})
        pong = ws.receive_json()
        assert pong["type"] == "PONG"

        # Graceful close
        ws.send_json({"type": "CLOSE_SESSION"})
        close_resp = ws.receive_json()
        assert close_resp["type"] == "SESSION_CLOSED"
        assert close_resp["session_id"] == session_id
        assert close_resp["session_state"] == "COMPLETED"
        assert "total_latency_ms" in close_resp
        assert "dropped_chunks" in close_resp

    assert manager.has_session(session_id) is False
