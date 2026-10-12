"""
WebSocket streaming endpoint for live microphone audio ingestion and real-time turn emission.
"""

import json
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.callintel.ml.meld_classifier import MeldClassifier

router = APIRouter(tags=["Live Streaming"])


class ConnectionManager:
    """Manages active live WebSocket call sessions."""

    def __init__(self):
        self.active_sessions: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_sessions[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

    async def send_json(self, session_id: str, data: Dict[str, Any]):
        if session_id in self.active_sessions:
            await self.active_sessions[session_id].send_json(data)


manager = ConnectionManager()
meld_classifier = MeldClassifier()


@router.websocket("/api/v1/stream/{session_id}")
async def websocket_call_stream(websocket: WebSocket, session_id: str):
    """
    Live audio streaming session endpoint:
    - Accepts real-time audio chunk packets or speech turns
    - Returns real-time transcribed turns with MELD sentiment/emotion scoring
    - Emits live de-escalation alerts when customer frustration is detected.
    """
    await manager.connect(session_id, websocket)
    turn_counter = 0

    try:
        # Acknowledge connection
        await websocket.send_json({
            "type": "SESSION_INITIALIZED",
            "session_id": session_id,
            "status": "READY_FOR_AUDIO_STREAM",
        })

        while True:
            # Receive text turn or serialized audio packet
            message = await websocket.receive_text()
            turn_counter += 1

            try:
                payload = json.loads(message)
                utterance = payload.get("text", message)
                speaker = payload.get("speaker", "Customer")
            except Exception:
                utterance = message
                speaker = "Customer"

            # Run low-latency conversational emotion inference
            emotion_res = meld_classifier.predict_turn(
                utterance=utterance,
                speaker=speaker,
            )

            # Detect real-time frustration trigger (emotion label, anger probability, or explicit cancellation/objection)
            is_frustrated = (
                emotion_res["emotion"]["label"] in ["anger", "sadness"]
                or emotion_res["sentiment"]["label"] == "negative"
                or emotion_res["emotion"]["probabilities"].get("anger", 0) >= 0.20
                or any(w in utterance.lower() for w in ["expensive", "cancel", "terrible", "angry", "broken", "refund", "unfair"])
            )

            response_packet = {
                "type": "STREAM_TURN_EMITTED",
                "session_id": session_id,
                "turn_index": turn_counter,
                "speaker": speaker,
                "text": utterance,
                "emotion": emotion_res["emotion"],
                "sentiment": emotion_res["sentiment"],
                "in_call_alert": {
                    "flagged": is_frustrated,
                    "recommended_cue": "Demonstrate active empathy and acknowledge budget constraints." if is_frustrated else None,
                },
            }

            await websocket.send_json(response_packet)

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        manager.disconnect(session_id)
