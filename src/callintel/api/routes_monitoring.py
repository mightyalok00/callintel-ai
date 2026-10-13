"""
Operational Monitoring, Health Probes, and Telemetry Endpoints for Production Readiness.
Implements Kubernetes/Docker Liveness (/health/live), Readiness (/health/ready), and Observability (/metrics).
"""

import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, Response, Request, status
from sqlalchemy import text

from src.callintel.core.logging import logger
from src.callintel.db.models import SessionLocal

router = APIRouter(tags=["Operational Monitoring"])

_START_TIME = time.time()

# In-memory metrics counters for telemetry
_METRICS = {
    "total_requests": 0,
    "active_requests": 0,
    "status_2xx": 0,
    "status_4xx": 0,
    "status_5xx": 0,
    "total_latency_ms": 0.0,
}


def record_request_start() -> None:
    _METRICS["total_requests"] += 1
    _METRICS["active_requests"] += 1


def record_request_end(status_code: int, duration_ms: float) -> None:
    _METRICS["active_requests"] = max(0, _METRICS["active_requests"] - 1)
    _METRICS["total_latency_ms"] += duration_ms
    if 200 <= status_code < 300:
        _METRICS["status_2xx"] += 1
    elif 400 <= status_code < 500:
        _METRICS["status_4xx"] += 1
    elif status_code >= 500:
        _METRICS["status_5xx"] += 1


@router.get("/health/live", summary="Liveness Probe", status_code=status.HTTP_200_OK)
async def liveness_probe() -> Dict[str, Any]:
    """
    Kubernetes/container liveness probe.
    Confirms the HTTP process is responsive and running.
    """
    uptime_sec = round(time.time() - _START_TIME, 2)
    return {
        "status": "ALIVE",
        "service": "AI Call Intelligence Engine",
        "version": "1.0.0",
        "uptime_seconds": uptime_sec,
        "pid": os.getpid(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/ready", summary="Readiness Probe")
async def readiness_probe(response: Response) -> Dict[str, Any]:
    """
    Kubernetes/container readiness probe.
    Validates that:
    1. Database connection is active.
    2. ML model artifacts exist on disk.
    3. Storage directories are writable.
    Returns HTTP 200 when ready, HTTP 503 when degraded or not ready.
    """
    checks = {}
    is_ready = True

    # 1. Database Check
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            checks["database"] = {"status": "HEALTHY", "message": "SQL query executed successfully"}
        finally:
            db.close()
    except Exception as exc:
        is_ready = False
        checks["database"] = {"status": "UNHEALTHY", "error": str(exc)}

    # 2. Model Artifacts Check
    models_dir = Path("models")
    required_models = {
        "churn_xgb_pipeline": models_dir / "churn_xgb_pipeline.joblib",
        "emotion_pipeline": models_dir / "emotion_pipeline.joblib",
        "sentiment_pipeline": models_dir / "sentiment_pipeline.joblib",
    }
    model_status = {}
    for name, path in required_models.items():
        if path.exists():
            model_status[name] = "LOADABLE"
        else:
            is_ready = False
            model_status[name] = "MISSING_ARTIFACT"
    checks["models"] = model_status

    # 3. Storage Writability Check
    storage_status = {}
    for dir_name in ["data/uploads", "reports"]:
        p = Path(dir_name)
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / f".health_write_test_{uuid.uuid4().hex[:6]}"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            storage_status[dir_name] = "WRITABLE"
        except Exception as exc:
            is_ready = False
            storage_status[dir_name] = f"WRITE_ERROR: {exc}"
    checks["storage"] = storage_status

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "NOT_READY",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        }

    return {
        "status": "READY",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": round(time.time() - _START_TIME, 2),
        "checks": checks,
    }


@router.get("/metrics", summary="Telemetry & Prometheus-Compatible Metrics")
async def metrics_endpoint(request: Request, response: Response):
    """
    Observability endpoint providing operational metrics.
    Supports Prometheus format (when Accept: text/plain) or JSON format.
    """
    uptime_sec = max(0.001, time.time() - _START_TIME)
    total_reqs = _METRICS["total_requests"]
    avg_latency = (
        round(_METRICS["total_latency_ms"] / total_reqs, 2) if total_reqs > 0 else 0.0
    )

    # Optional: fetch total calls from DB
    total_calls_in_db = 0
    try:
        db = SessionLocal()
        try:
            res = db.execute(text("SELECT count(*) FROM call_records")).scalar()
            total_calls_in_db = int(res or 0)
        finally:
            db.close()
    except Exception:
        total_calls_in_db = 0

    accept_header = request.headers.get("accept", "")
    format_param = request.query_params.get("format", "")

    if "text/plain" in accept_header or format_param == "prometheus":
        # Prometheus plaintext exposition format
        prom_lines = [
            "# HELP app_uptime_seconds Application uptime in seconds",
            "# TYPE app_uptime_seconds gauge",
            f"app_uptime_seconds {uptime_sec:.2f}",
            "# HELP http_requests_total Total HTTP requests processed",
            "# TYPE http_requests_total counter",
            f"http_requests_total {total_reqs}",
            f'http_requests_total_by_status{{status="2xx"}} {_METRICS["status_2xx"]}',
            f'http_requests_total_by_status{{status="4xx"}} {_METRICS["status_4xx"]}',
            f'http_requests_total_by_status{{status="5xx"}} {_METRICS["status_5xx"]}',
            "# HELP http_requests_active Active concurrent requests",
            "# TYPE http_requests_active gauge",
            f'http_requests_active {_METRICS["active_requests"]}',
            "# HELP http_request_duration_ms_avg Average request latency in milliseconds",
            "# TYPE http_request_duration_ms_avg gauge",
            f"http_request_duration_ms_avg {avg_latency:.2f}",
            "# HELP callintel_calls_stored_total Total audio calls in database",
            "# TYPE callintel_calls_stored_total counter",
            f"callintel_calls_stored_total {total_calls_in_db}",
        ]
        return Response(content="\n".join(prom_lines) + "\n", media_type="text/plain; version=0.0.4")

    return {
        "service": "AI Call Intelligence Engine",
        "uptime_seconds": round(uptime_sec, 2),
        "traffic": {
            "total_requests": total_reqs,
            "active_requests": _METRICS["active_requests"],
            "requests_per_second": round(total_reqs / uptime_sec, 4),
            "status_codes": {
                "2xx": _METRICS["status_2xx"],
                "4xx": _METRICS["status_4xx"],
                "5xx": _METRICS["status_5xx"],
            },
            "average_latency_ms": avg_latency,
        },
        "business_telemetry": {
            "total_calls_stored": total_calls_in_db,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
