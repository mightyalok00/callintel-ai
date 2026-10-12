"""
Production WebSocket streaming endpoint for live audio ingestion,
audio chunk buffering, monotonic transcript timing, emotion scoring, and real-time frustration alerts.
Supports:
- Raw binary PCM/WAV audio frames (16kHz 16-bit mono)
- Base64 encoded audio payloads in JSON frames
- Text turn messages
- Ping/pong heartbeats
- Robust connection interruption recovery and clean session cleanup
- Graceful error handling for corrupted or invalid payloads
"""

import json
import base64
import time
import io
from typing import Dict, Any, Optional
import numpy as np
import soundfile as sf
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.callintel.ml.meld_classifier import MeldClassifier
from src.callintel.speech.audio_utils import format_timestamp

router = APIRouter(tags=["Live Streaming"])


class StreamingSessionState:
    """Maintains state, buffered audio, and timing for a live streaming connection."""

    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.created_at = time.time()
        self.last_active_at = time.time()
        self.audio_samples: np.ndarray = np.array([], dtype=np.float32)
        self.total_streamed_seconds: float = 0.0
        self.turn_counter: int = 0
        self.sample_rate: int = 16000

    def append_audio_samples(self, new_samples: np.ndarray) -> tuple[float, float, float]:
        """Appends new samples and returns (start_seconds, end_seconds, duration_seconds)."""
        duration = len(new_samples) / self.sample_rate
        start_s = round(self.total_streamed_seconds, 3)
        end_s = round(start_s + duration, 3)
        self.total_streamed_seconds = end_s
        self.audio_samples = np.append(self.audio_samples, new_samples)
        self.last_active_at = time.time()
        return start_s, end_s, duration


class ConnectionManager:
    """Manages active live WebSocket call sessions and resources."""

    def __init__(self):
        self.active_sessions: Dict[str, StreamingSessionState] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> StreamingSessionState:
        await websocket.accept()
        # Clean up any existing stale session with the same ID
        if session_id in self.active_sessions:
            self.disconnect(session_id)
        session = StreamingSessionState(session_id, websocket)
        self.active_sessions[session_id] = session
        return session

    def disconnect(self, session_id: str):
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

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
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
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
    Live streaming session endpoint:
    - Accepts binary PCM/WAV chunks or JSON payload messages
    - Tracks audio stream timestamps monotonically
    - Emits transcribed turns with conversational emotion/sentiment and frustration alerts
    - Handles connection drops and guarantees session cleanup
    """
    session = await manager.connect(session_id, websocket)

    try:
        # Acknowledge connection
        await websocket.send_json({
            "type": "SESSION_INITIALIZED",
            "session_id": session_id,
            "status": "READY_FOR_AUDIO_STREAM",
            "sample_rate": session.sample_rate,
            "supported_inputs": ["raw_pcm_16bit_mono", "wav_binary", "json_text", "json_base64_audio"],
        })

        while True:
            # Wait for frame (can be bytes or text)
            frame = await websocket.receive()

            if frame.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            # ---------------------------------------------------------
            # 1. BINARY AUDIO FRAME (Raw PCM or WAV bytes)
            # ---------------------------------------------------------
            if "bytes" in frame and frame["bytes"]:
                raw_bytes = frame["bytes"]
                try:
                    samples = decode_audio_bytes(raw_bytes, target_sr=session.sample_rate)
                    if len(samples) < 160:  # < 10ms
                        continue

                    start_s, end_s, dur_s = session.append_audio_samples(samples)
                    session.turn_counter += 1

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

                    response_packet = {
                        "type": "STREAM_AUDIO_PROCESSED",
                        "session_id": session_id,
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
                        "in_call_alert": {
                            "flagged": is_frustrated,
                            "recommended_cue": "Demonstrate active empathy and acknowledge customer concern." if is_frustrated else None,
                        },
                    }
                    await websocket.send_json(response_packet)

                except Exception as ex:
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

                # Handle JSON control frames or base64 audio
                try:
                    payload = json.loads(text_msg)
                except Exception:
                    # Treat plain string as raw text turn
                    payload = {"type": "TEXT_TURN", "text": text_msg}

                msg_type = payload.get("type", "TEXT_TURN")

                if msg_type == "PING":
                    await websocket.send_json({"type": "PONG", "timestamp": time.time()})
                    continue

                if msg_type == "CLOSE_SESSION":
                    await websocket.send_json({
                        "type": "SESSION_CLOSED",
                        "session_id": session_id,
                        "total_turns": session.turn_counter,
                        "total_duration_seconds": session.total_streamed_seconds,
                    })
                    break

                if msg_type == "AUDIO_CHUNK" and "audio_base64" in payload:
                    try:
                        audio_raw = base64.b64decode(payload["audio_base64"])
                        samples = decode_audio_bytes(audio_raw, target_sr=session.sample_rate)
                        start_s, end_s, dur_s = session.append_audio_samples(samples)
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
                        await websocket.send_json({
                            "type": "STREAM_AUDIO_PROCESSED",
                            "session_id": session_id,
                            "turn_index": session.turn_counter,
                            "start_seconds": start_s,
                            "end_seconds": end_s,
                            "text": utterance,
                            "emotion": emotion_res["emotion"],
                            "sentiment": emotion_res["sentiment"],
                            "in_call_alert": {"flagged": is_frustrated},
                        })
                    except Exception as err:
                        await websocket.send_json({
                            "type": "STREAM_ERROR",
                            "error_code": "BASE64_AUDIO_ERROR",
                            "message": str(err),
                            "recoverable": True,
                        })
                    continue

                # Standard text turn
                session.turn_counter += 1
                utterance = payload.get("text", text_msg)
                speaker = payload.get("speaker", "Customer")
                start_s = round(session.total_streamed_seconds, 3)
                estimated_duration = max(1.0, len(utterance.split()) * 0.4)
                end_s = round(start_s + estimated_duration, 3)
                session.total_streamed_seconds = end_s

                emotion_res = meld_classifier.predict_turn(utterance=utterance, speaker=speaker)
                is_frustrated = (
                    emotion_res["emotion"]["label"] in ["anger", "sadness"]
                    or emotion_res["sentiment"]["label"] == "negative"
                    or emotion_res["emotion"]["probabilities"].get("anger", 0) >= 0.20
                    or any(w in utterance.lower() for w in ["expensive", "cancel", "terrible", "angry", "broken", "refund", "unfair"])
                )

                response_packet = {
                    "type": "STREAM_TURN_EMITTED",
                    "session_id": session_id,
                    "turn_index": session.turn_counter,
                    "speaker": speaker,
                    "start_seconds": start_s,
                    "end_seconds": end_s,
                    "start_time": format_timestamp(start_s),
                    "end_time": format_timestamp(end_s),
                    "text": utterance,
                    "emotion": emotion_res["emotion"],
                    "sentiment": emotion_res["sentiment"],
                    "in_call_alert": {
                        "flagged": is_frustrated,
                        "recommended_cue": "Demonstrate active empathy and acknowledge customer concern." if is_frustrated else None,
                    },
                }
                await websocket.send_json(response_packet)

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        manager.disconnect(session_id)
    finally:
        manager.disconnect(session_id)
