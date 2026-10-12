"""
Production WebSocket streaming endpoint for live audio ingestion,
bounded audio chunk buffering, monotonic transcript timing, emotion scoring,
and real-time frustration alerts.

Hardened Features:
- Explicit Session Lifecycle: CONNECTED -> RECORDING -> PROCESSING -> COMPLETED / FAILED
- Bounded Buffer Limits: MAX_CHUNK_BYTES (512KB) & MAX_TOTAL_BUFFER_BYTES (10MB)
- Backpressure & Dropped Chunk Tracking
- Latency Metrics: first_transcript_latency_ms, processing_latency_ms, total_latency_ms
- Connection cleanup and safe reconnection behavior
- Audio decoding: Raw binary PCM16 & WAV with format fallback
"""

import json
import base64
import time
import io
from enum import Enum
from typing import Dict, Any, Optional, Tuple
import numpy as np
import soundfile as sf
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.callintel.ml.meld_classifier import MeldClassifier
from src.callintel.speech.audio_utils import format_timestamp
from src.callintel.core.config import settings
from src.callintel.core.logging import logger

router = APIRouter(tags=["Live Streaming"])


class StreamSessionStatus(str, Enum):
    CONNECTED = "CONNECTED"
    RECORDING = "RECORDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class StreamingSessionState:
    """Maintains state, bounded audio buffer, and timing metrics for a live streaming connection."""

    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.state: StreamSessionStatus = StreamSessionStatus.CONNECTED
        self.created_at: float = time.time()
        self.last_active_at: float = time.time()
        self.audio_samples: np.ndarray = np.array([], dtype=np.float32)
        self.total_buffered_bytes: int = 0
        self.total_streamed_seconds: float = 0.0
        self.turn_counter: int = 0
        self.sample_rate: int = 16000

        # Latency and health metrics
        self.first_chunk_received_at: Optional[float] = None
        self.first_transcript_emitted_at: Optional[float] = None
        self.first_transcript_latency_ms: Optional[float] = None
        self.total_latency_ms: float = 0.0
        self.dropped_chunks_count: int = 0

    def append_audio_samples(self, new_samples: np.ndarray, byte_size: int) -> Tuple[float, float, float]:
        """Appends new samples and returns (start_seconds, end_seconds, duration_seconds)."""
        duration = len(new_samples) / self.sample_rate
        start_s = round(self.total_streamed_seconds, 3)
        end_s = round(start_s + duration, 3)
        self.total_streamed_seconds = end_s
        self.audio_samples = np.append(self.audio_samples, new_samples)
        self.total_buffered_bytes += byte_size
        self.last_active_at = time.time()

        if self.first_chunk_received_at is None:
            self.first_chunk_received_at = time.time()

        return start_s, end_s, duration


class ConnectionManager:
    """Manages active live WebSocket call sessions and guarantees resource cleanup."""

    def __init__(self):
        self.active_sessions: Dict[str, StreamingSessionState] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> StreamingSessionState:
        await websocket.accept()
        # Clean up any existing stale session with the same ID
        if session_id in self.active_sessions:
            self.disconnect(session_id)
        session = StreamingSessionState(session_id, websocket)
        self.active_sessions[session_id] = session
        logger.info(f"WebSocket session connected: {session_id}. Active: {len(self.active_sessions)}")
        return session

    def disconnect(self, session_id: str):
        if session_id in self.active_sessions:
            sess = self.active_sessions.pop(session_id, None)
            if sess:
                sess.state = StreamSessionStatus.COMPLETED
            logger.info(f"WebSocket session disconnected: {session_id}. Remaining: {len(self.active_sessions)}")

    def has_session(self, session_id: str) -> bool:
        return session_id in self.active_sessions

    def get_session(self, session_id: str) -> Optional[StreamingSessionState]:
        return self.active_sessions.get(session_id)

    def active_count(self) -> int:
        return len(self.active_sessions)


manager = ConnectionManager()
meld_classifier = MeldClassifier()
_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    return _whisper_model


def decode_audio_bytes(data: bytes, target_sr: int = 16000) -> np.ndarray:
    """Converts binary PCM16 or WAV bytes to normalized 1D float32 numpy array."""
    if len(data) == 0:
        return np.array([], dtype=np.float32)

    # Check for RIFF/WAV header
    if data[:4] == b"RIFF":
        try:
            with io.BytesIO(data) as bio:
                audio, sr = sf.read(bio, dtype="float32")
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                if sr != target_sr:
                    indices = np.linspace(0, len(audio) - 1, int(len(audio) * target_sr / sr))
                    audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
                return audio
        except Exception:
            pass

    # Treat as raw 16-bit signed PCM mono
    pcm_int16 = np.frombuffer(data, dtype=np.int16)
    return (pcm_int16.astype(np.float32) / 32768.0)


@router.websocket("/api/v1/stream/{session_id}")
async def websocket_call_stream(websocket: WebSocket, session_id: str):
    """
    Production Live Streaming Session Endpoint:
    - Bounded chunk and total buffer limits
    - Monotonic audio stream timestamps
    - State lifecycle: CONNECTED -> RECORDING -> PROCESSING -> COMPLETED / FAILED
    - Real-time emotion & frustration alerts
    - Latency tracking & disconnect cleanup
    """
    session = await manager.connect(session_id, websocket)

    try:
        # Acknowledge connection
        await websocket.send_json({
            "type": "SESSION_INITIALIZED",
            "session_id": session_id,
            "status": "READY_FOR_AUDIO_STREAM",
            "session_state": session.state.value,
            "sample_rate": session.sample_rate,
            "limits": {
                "max_chunk_bytes": settings.max_chunk_bytes,
                "max_total_buffer_bytes": settings.max_total_buffer_bytes,
            },
            "supported_inputs": ["raw_pcm_16bit_mono", "wav_binary", "json_text", "json_base64_audio"],
        })

        while True:
            frame = await websocket.receive()

            if frame.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            # ---------------------------------------------------------
            # 1. BINARY AUDIO FRAME (Raw PCM or WAV bytes)
            # ---------------------------------------------------------
            if "bytes" in frame and frame["bytes"]:
                raw_bytes = frame["bytes"]
                chunk_len = len(raw_bytes)

                # Bounded buffer check: per-chunk limit
                if chunk_len > settings.max_chunk_bytes:
                    session.dropped_chunks_count += 1
                    await websocket.send_json({
                        "type": "STREAM_ERROR",
                        "error_code": "CHUNK_SIZE_LIMIT_EXCEEDED",
                        "message": f"Frame size {chunk_len} exceeds maximum chunk limit {settings.max_chunk_bytes} bytes.",
                        "recoverable": True,
                        "dropped_chunks": session.dropped_chunks_count,
                    })
                    continue

                # Bounded buffer check: total buffer limit
                if session.total_buffered_bytes + chunk_len > settings.max_total_buffer_bytes:
                    session.dropped_chunks_count += 1
                    await websocket.send_json({
                        "type": "STREAM_ERROR",
                        "error_code": "TOTAL_BUFFER_LIMIT_EXCEEDED",
                        "message": f"Total buffer exceeds maximum limit {settings.max_total_buffer_bytes} bytes.",
                        "recoverable": False,
                        "dropped_chunks": session.dropped_chunks_count,
                    })
                    continue

                try:
                    session.state = StreamSessionStatus.RECORDING
                    proc_start = time.perf_counter()

                    samples = decode_audio_bytes(raw_bytes, target_sr=session.sample_rate)
                    if len(samples) < 160:  # < 10ms
                        continue

                    start_s, end_s, dur_s = session.append_audio_samples(samples, chunk_len)
                    session.turn_counter += 1
                    session.state = StreamSessionStatus.PROCESSING

                    # Transcribe chunk
                    whisper = get_whisper_model()
                    segments, _ = whisper.transcribe(
                        samples,
                        beam_size=1,
                        vad_filter=True,
                        initial_prompt="Customer service phone call",
                    )
                    text_parts = [s.text.strip() for s in segments if s.text.strip()]
                    utterance = " ".join(text_parts) if text_parts else "[Ambient / Non-speech audio]"

                    # Score emotion and check frustration
                    emotion_res = meld_classifier.predict_turn(
                        utterance=utterance,
                        speaker="Customer",
                    )
                    is_frustrated = (
                        emotion_res["emotion"]["label"] in ["anger", "sadness"]
                        or emotion_res["sentiment"]["label"] == "negative"
                        or emotion_res["emotion"]["probabilities"].get("anger", 0) >= 0.20
                        or any(w in utterance.lower() for w in ["expensive", "cancel", "terrible", "angry", "broken", "refund"])
                    )

                    proc_duration_ms = round((time.perf_counter() - proc_start) * 1000.0, 2)
                    session.total_latency_ms += proc_duration_ms

                    if session.first_transcript_emitted_at is None:
                        session.first_transcript_emitted_at = time.time()
                        if session.first_chunk_received_at is not None:
                            session.first_transcript_latency_ms = round(
                                (session.first_transcript_emitted_at - session.first_chunk_received_at) * 1000.0, 2
                            )

                    session.state = StreamSessionStatus.RECORDING

                    response_packet = {
                        "type": "STREAM_AUDIO_PROCESSED",
                        "session_id": session_id,
                        "session_state": session.state.value,
                        "turn_index": session.turn_counter,
                        "speaker": "Customer",
                        "start_seconds": start_s,
                        "end_seconds": end_s,
                        "duration_seconds": round(dur_s, 3),
                        "start_time": format_timestamp(start_s),
                        "end_time": format_timestamp(end_s),
                        "text": utterance,
                        "emotion": emotion_res["emotion"],
                        "sentiment": emotion_res["sentiment"],
                        "latency_ms": proc_duration_ms,
                        "first_transcript_latency_ms": session.first_transcript_latency_ms,
                        "dropped_chunks": session.dropped_chunks_count,
                        "in_call_alert": {
                            "flagged": is_frustrated,
                            "recommended_cue": "Demonstrate active empathy and acknowledge customer concern." if is_frustrated else None,
                        },
                    }
                    await websocket.send_json(response_packet)

                except Exception as ex:
                    session.state = StreamSessionStatus.RECORDING
                    await websocket.send_json({
                        "type": "STREAM_ERROR",
                        "error_code": "AUDIO_PROCESSING_ERROR",
                        "message": f"Failed to process binary audio chunk: {str(ex)}",
                        "recoverable": True,
                    })

            # ---------------------------------------------------------
            # 2. TEXT / JSON FRAME
            # ---------------------------------------------------------
            elif "text" in frame and frame["text"]:
                text_msg = frame["text"]

                try:
                    payload = json.loads(text_msg)
                except Exception:
                    payload = {"type": "TEXT_TURN", "text": text_msg}

                msg_type = payload.get("type", "TEXT_TURN")

                if msg_type == "PING":
                    await websocket.send_json({"type": "PONG", "timestamp": time.time()})
                    continue

                if msg_type == "CLOSE_SESSION":
                    session.state = StreamSessionStatus.COMPLETED
                    await websocket.send_json({
                        "type": "SESSION_CLOSED",
                        "session_id": session_id,
                        "session_state": session.state.value,
                        "total_turns": session.turn_counter,
                        "total_duration_seconds": session.total_streamed_seconds,
                        "first_transcript_latency_ms": session.first_transcript_latency_ms,
                        "total_latency_ms": round(session.total_latency_ms, 2),
                        "dropped_chunks": session.dropped_chunks_count,
                    })
                    break

                if msg_type == "AUDIO_CHUNK" and "audio_base64" in payload:
                    try:
                        audio_raw = base64.b64decode(payload["audio_base64"])
                        if len(audio_raw) > settings.max_chunk_bytes:
                            session.dropped_chunks_count += 1
                            await websocket.send_json({
                                "type": "STREAM_ERROR",
                                "error_code": "CHUNK_SIZE_LIMIT_EXCEEDED",
                                "message": "Base64 chunk exceeds 512KB limit.",
                                "recoverable": True,
                            })
                            continue

                        session.state = StreamSessionStatus.PROCESSING
                        samples = decode_audio_bytes(audio_raw, target_sr=session.sample_rate)
                        start_s, end_s, dur_s = session.append_audio_samples(samples, len(audio_raw))
                        session.turn_counter += 1

                        whisper = get_whisper_model()
                        segments, _ = whisper.transcribe(samples, beam_size=1, vad_filter=True)
                        text_parts = [s.text.strip() for s in segments if s.text.strip()]
                        utterance = " ".join(text_parts) if text_parts else "[Non-speech]"

                        emotion_res = meld_classifier.predict_turn(utterance=utterance, speaker="Customer")
                        is_frustrated = (
                            emotion_res["emotion"]["label"] in ["anger", "sadness"]
                            or emotion_res["sentiment"]["label"] == "negative"
                        )
                        session.state = StreamSessionStatus.RECORDING

                        await websocket.send_json({
                            "type": "STREAM_AUDIO_PROCESSED",
                            "session_id": session_id,
                            "session_state": session.state.value,
                            "turn_index": session.turn_counter,
                            "start_seconds": start_s,
                            "end_seconds": end_s,
                            "duration_seconds": round(dur_s, 3),
                            "start_time": format_timestamp(start_s),
                            "end_time": format_timestamp(end_s),
                            "text": utterance,
                            "emotion": emotion_res["emotion"],
                            "sentiment": emotion_res["sentiment"],
                            "in_call_alert": {"flagged": is_frustrated},
                        })
                    except Exception as ex:
                        session.state = StreamSessionStatus.RECORDING
                        await websocket.send_json({
                            "type": "STREAM_ERROR",
                            "error_code": "INVALID_BASE64_AUDIO",
                            "message": str(ex),
                            "recoverable": True,
                        })

                elif msg_type == "TEXT_TURN":
                    # Text-only conversational turn
                    utterance = payload.get("text", "")
                    speaker = payload.get("speaker", "Customer")
                    session.turn_counter += 1

                    emotion_res = meld_classifier.predict_turn(utterance=utterance, speaker=speaker)
                    is_frustrated = (
                        emotion_res["emotion"]["label"] in ["anger", "sadness"]
                        or emotion_res["sentiment"]["label"] == "negative"
                        or emotion_res["emotion"]["probabilities"].get("anger", 0) >= 0.20
                        or any(w in utterance.lower() for w in ["expensive", "cancel", "terrible", "angry", "broken", "refund"])
                    )
                    await websocket.send_json({
                        "type": "STREAM_TURN_EMITTED",
                        "session_id": session_id,
                        "session_state": session.state.value,
                        "turn_index": session.turn_counter,
                        "speaker": speaker,
                        "text": utterance,
                        "emotion": emotion_res["emotion"],
                        "sentiment": emotion_res["sentiment"],
                        "in_call_alert": {
                            "flagged": is_frustrated,
                            "recommended_cue": "Demonstrate active empathy and acknowledge customer concern." if is_frustrated else None,
                        },
                    })

    except WebSocketDisconnect:
        logger.info(f"Client disconnected abruptly: {session_id}")
    except Exception as ex:
        session.state = StreamSessionStatus.FAILED
        logger.error(f"WebSocket session {session_id} failed with error: {ex}")
    finally:
        manager.disconnect(session_id)
