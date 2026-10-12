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
