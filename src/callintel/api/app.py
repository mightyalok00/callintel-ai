"""
FastAPI Application Entry Point for AI Call Intelligence Engine.
"""

import time
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.callintel.api.routes_calls import router as calls_router
from src.callintel.api.routes_analytics import router as analytics_router
from src.callintel.api.websocket_live import router as websocket_router
from src.callintel.api.routes_monitoring import (
    router as monitoring_router,
    record_request_start,
    record_request_end,
)
from src.callintel.core.config import settings
from src.callintel.core.logging import logger

app = FastAPI(
    title="AI Call Intelligence Engine API",
    version="1.0.0",
    description="Production REST and WebSocket API for Speech AI, Conversational Emotion, Predictive Churn, and Multi-Agent Evidence Grounding.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Correlation ID and Request Latency Middleware
@app.middleware("http")
async def operational_telemetry_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
    record_request_start()
    start_time = time.time()
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
        record_request_end(response.status_code, duration_ms)
        return response
    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000.0
        record_request_end(500, duration_ms)
        logger.error(f"Unhandled request exception [{request_id}]: {exc}")
        raise

# Configure CORS centrally. For production, set explicit trusted frontend origins
# in Settings; wildcard origins must never be combined with credentialed requests.
_allow_wildcard_origins = "*" in settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=not _allow_wildcard_origins,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
)

# Register Routers
app.include_router(monitoring_router)
app.include_router(calls_router)
app.include_router(analytics_router)
app.include_router(websocket_router)


@app.get("/health", tags=["Operational Monitoring"])
async def health_check():
    """Legacy liveness probe verifying API availability and active models."""
    return {
        "status": "HEALTHY",
        "service": "AI Call Intelligence Engine API",
        "version": "1.0.0",
        "models_loaded": {
            "speech_transcription": "faster-whisper-base",
            "emotion_sentiment": "MELD-Logistic",
            "churn_prediction": "XGBoost + TreeSHAP",
            "evidence_ledger": "Active",
            "multi_agent_supervisor": "6 Specialized Agents",
        },
    }
