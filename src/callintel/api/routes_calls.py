"""
REST Endpoints for Call Audio Ingestion, Transcript Retrieval, and Insights.
Built with clean Presentation Layer architecture and Dependency Injection.
"""

import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, status, Depends
from pydantic import BaseModel

from src.callintel.core.config import settings
from src.callintel.core.logging import logger
from src.callintel.services.call_analysis_service import CallAnalysisService
from src.callintel.api.dependencies import (
    get_call_analysis_service,
    get_transcriber,
    get_meld_classifier,
    get_churn_predictor,
    get_multi_agent_supervisor,
)
from src.callintel.db.repository import CallRepository

router = APIRouter(prefix="/api/v1/calls", tags=["Calls & Intelligence"])


class CallUploadResponse(BaseModel):
    call_id: str
    status: str
    filename: str
    poll_url: str


def run_pipeline_worker(call_id: str, audio_path: Path):
    """Background worker invoking CallAnalysisService with injected singletons."""
    try:
        repo = CallRepository()
        service = CallAnalysisService(
            repo=repo,
            transcriber=get_transcriber(),
            meld=get_meld_classifier(),
            churn_predictor=get_churn_predictor(),
            supervisor=get_multi_agent_supervisor(),
        )
        service.process_call_audio(call_id=call_id, audio_path=audio_path)
    except Exception as ex:
        logger.error(f"Error processing call {call_id}: {ex}")
        repo = CallRepository()
        repo.set_failed(call_id, str(ex))


@router.post("/upload", response_model=CallUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_call_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    service: CallAnalysisService = Depends(get_call_analysis_service),
):
    """
    Uploads a recorded customer call (WAV/MP3) for asynchronous intelligence processing.
    Returns 202 Accepted with poll URL.
    """
    if not file.filename or not file.filename.lower().endswith((".wav", ".mp3", ".ogg", ".flac")):
        raise HTTPException(status_code=400, detail="Invalid audio format. Supported: .wav, .mp3, .ogg, .flac")

    safe_filename: str = file.filename
    call_id = f"CALL_{uuid.uuid4().hex[:10].upper()}"
    save_path = settings.upload_dir / f"{call_id}_{safe_filename}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    service.create_pending_call(call_id=call_id, filename=safe_filename)
    background_tasks.add_task(run_pipeline_worker, call_id, save_path)

    return CallUploadResponse(
        call_id=call_id,
        status="PENDING",
        filename=safe_filename,
        poll_url=f"/api/v1/calls/{call_id}",
    )


@router.get("/{call_id}")
async def get_call_status(
    call_id: str,
    service: CallAnalysisService = Depends(get_call_analysis_service),
):
    """Returns processing status, metadata, and high-level KPIs for a specific call."""
    record = service.get_call(call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call record not found")

    return {
        "call_id": record.call_id,
        "filename": record.filename,
        "status": record.status,
        "duration_seconds": record.duration_seconds,
        "language": record.language,
        "overall_sentiment": record.overall_sentiment,
        "sentiment_score": record.sentiment_score,
        "churn_risk_level": record.churn_risk_level,
        "churn_risk_score": record.churn_risk_score,
        "primary_intent": record.primary_intent,
        "qa_score": record.qa_score,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


@router.get("/{call_id}/transcript")
async def get_call_transcript(
    call_id: str,
    service: CallAnalysisService = Depends(get_call_analysis_service),
):
    """Returns timestamped dialogue turns with speaker attribution for a call."""
    record = service.get_call(call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call record not found")

    return {
        "call_id": record.call_id,
        "status": record.status,
        "total_turns": len(record.transcript_turns or []),
        "turns": record.transcript_turns or [],
    }


@router.get("/{call_id}/insights")
async def get_call_insights(
    call_id: str,
    service: CallAnalysisService = Depends(get_call_analysis_service),
):
    """Returns the Auditable Evidence Ledger, SHAP drivers, and action recommendations."""
    record = service.get_call(call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call record not found")

    if record.status != "COMPLETED":
        return {
            "call_id": record.call_id,
            "status": record.status,
            "message": "Intelligence processing still in progress.",
        }

    return {
        "call_id": record.call_id,
        "overall_sentiment": record.overall_sentiment,
        "churn_risk_level": record.churn_risk_level,
        "churn_risk_score": record.churn_risk_score,
        "primary_intent": record.primary_intent,
        "qa_score": record.qa_score,
        "evidence_ledger": record.evidence_ledger or [],
        "recommendations": record.recommendations or [],
        "metadata": record.metadata_payload or {},
    }
