"""
Unit tests for operational monitoring, liveness, readiness, and telemetry endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.callintel.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_legacy_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "models_loaded" in data


def test_liveness_probe(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ALIVE"
    assert "uptime_seconds" in data
    assert "pid" in data
    assert "timestamp_utc" in data
    assert data["uptime_seconds"] >= 0


def test_readiness_probe(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "READY"
    assert "checks" in data
    assert data["checks"]["database"]["status"] == "HEALTHY"
    assert data["checks"]["models"]["churn_xgb_pipeline"] == "LOADABLE"
    assert data["checks"]["storage"]["data/uploads"] == "WRITABLE"


def test_metrics_json_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "traffic" in data
    assert "total_requests" in data["traffic"]
    assert "status_codes" in data["traffic"]
    assert "business_telemetry" in data


def test_metrics_prometheus_format(client):
    response = client.get("/metrics", headers={"Accept": "text/plain"})
    assert response.status_code == 200
    content = response.text
    assert "# HELP app_uptime_seconds" in content
    assert "# HELP http_requests_total" in content
    assert "http_requests_total" in content


def test_request_id_and_duration_headers(client):
    custom_id = "test-custom-trace-id-12345"
    response = client.get("/health/live", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id
    assert "X-Response-Time-Ms" in response.headers
    duration_ms = float(response.headers["X-Response-Time-Ms"])
    assert duration_ms >= 0.0


def test_auto_generated_request_id(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    req_id = response.headers.get("X-Request-ID")
    assert req_id is not None
    assert req_id.startswith("req_")
