"""
Integration tests for FastAPI REST Endpoints and WebSocket streaming.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.callintel.api.app import app

client = TestClient(app)
SAMPLE_WAV = Path("data/samples/sample_customer_call.wav")


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "models_loaded" in data


def test_analytics_overview_endpoint():
    response = client.get("/api/v1/analytics/overview")
    assert response.status_code == 200
    data = response.json()
    assert "total_calls_analyzed" in data
    assert "saved_arr_estimate" in data


def test_call_upload_lifecycle():
    assert SAMPLE_WAV.exists(), "Sample WAV missing for integration test."

    with open(SAMPLE_WAV, "rb") as f:
        response = client.post(
            "/api/v1/calls/upload",
            files={"file": ("test_call.wav", f, "audio/wav")},
        )

    assert response.status_code == 202
    data = response.json()
    assert "call_id" in data
    assert data["status"] == "PENDING"
    assert "poll_url" in data

    # Poll status immediately
    poll_resp = client.get(data["poll_url"])
    assert poll_resp.status_code == 200
    assert poll_resp.json()["call_id"] == data["call_id"]


def test_websocket_streaming_session():
    with client.websocket_connect("/api/v1/stream/test_session_101") as websocket:
        init_data = websocket.receive_json()
        assert init_data["type"] == "SESSION_INITIALIZED"
        assert init_data["status"] == "READY_FOR_AUDIO_STREAM"

        # Stream a frustrated turn
        websocket.send_text("Your product is way too expensive, I might cancel.")
        turn_data = websocket.receive_json()

        assert turn_data["type"] == "STREAM_TURN_EMITTED"
        assert "emotion" in turn_data
        assert "sentiment" in turn_data
        assert turn_data["in_call_alert"]["flagged"] is True


def test_upload_rejects_corrupted_magic_bytes():
    fake_audio = b"FAKE_CONTENT_NOT_WAVE_HEADER_DATA"
    response = client.post(
        "/api/v1/calls/upload",
        files={"file": ("fake.wav", fake_audio, "audio/wav")},
    )
    assert response.status_code == 400
    assert "Invalid audio content signature" in response.json()["detail"]


def test_upload_rejects_oversized_payload(monkeypatch):
    from src.callintel.core.config import settings
    # Set limit to 200 bytes for test
    monkeypatch.setattr(settings, "max_upload_size_bytes", 200)

    # Valid wav header but oversized
    wav_header = b"RIFF" + (300).to_bytes(4, "little") + b"WAVE" + (b"\x00" * 300)
    response = client.post(
        "/api/v1/calls/upload",
        files={"file": ("oversized.wav", wav_header, "audio/wav")},
    )
    assert response.status_code == 413
    assert "exceeds maximum allowed upload limit" in response.json()["detail"]


def test_upload_sanitizes_path_traversal():
    assert SAMPLE_WAV.exists()
    with open(SAMPLE_WAV, "rb") as f:
        response = client.post(
            "/api/v1/calls/upload",
            files={"file": ("../../../../etc/malicious_call.wav", f, "audio/wav")},
        )
    assert response.status_code == 202
    filename = response.json()["filename"]
    assert ".." not in filename
    assert "/" not in filename
    assert "\\" not in filename
    assert "malicious_call.wav" in filename


def test_upload_enforces_api_key_when_required(monkeypatch):
    from src.callintel.core.config import settings
    monkeypatch.setattr(settings, "require_auth", True)
    monkeypatch.setattr(settings, "api_key", "secret_test_key_123")

    with open(SAMPLE_WAV, "rb") as f:
        # Request without key
        fail_resp = client.post(
            "/api/v1/calls/upload",
            files={"file": ("valid.wav", f, "audio/wav")},
        )
    assert fail_resp.status_code == 401

    with open(SAMPLE_WAV, "rb") as f:
        # Request with valid key
        success_resp = client.post(
            "/api/v1/calls/upload",
            files={"file": ("valid.wav", f, "audio/wav")},
            headers={"X-API-Key": "secret_test_key_123"},
        )
    assert success_resp.status_code == 202


def test_delete_call_and_clear_history():
    # 1. Upload a call
    with open(SAMPLE_WAV, "rb") as f:
        resp = client.post(
            "/api/v1/calls/upload",
            files={"file": ("test_del.wav", f, "audio/wav")},
        )
    assert resp.status_code == 202
    call_id = resp.json()["call_id"]

    # 2. Delete this specific call
    del_resp = client.delete(f"/api/v1/calls/{call_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "SUCCESS"

    # 3. Deleting non-existent call returns 404
    del_404 = client.delete(f"/api/v1/calls/{call_id}")
    assert del_404.status_code == 404

    # 4. Clear all call history
    clear_resp = client.delete("/api/v1/calls/history")
    assert clear_resp.status_code == 200
    assert clear_resp.json()["status"] == "SUCCESS"


