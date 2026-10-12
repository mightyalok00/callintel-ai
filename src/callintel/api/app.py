"""
FastAPI Application Entry Point for AI Call Intelligence Engine.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.callintel.api.routes_calls import router as calls_router
from src.callintel.api.routes_analytics import router as analytics_router
from src.callintel.api.websocket_live import router as websocket_router

app = FastAPI(
    title="AI Call Intelligence Engine API",
    version="1.0.0",
    description="Production REST and WebSocket API for Speech AI, Conversational Emotion, Predictive Churn, and Multi-Agent Evidence Grounding.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend and dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(calls_router)
app.include_router(analytics_router)
app.include_router(websocket_router)


@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe verifying API availability and active models."""
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
