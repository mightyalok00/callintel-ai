"""
REST Endpoints for Call Audio Ingestion, Transcript Retrieval, and Insights.
"""

import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, status
from pydantic import BaseModel

from src.callintel.db.repository import CallRepository
from src.callintel.speech.transcriber import WhisperTranscriber
from src.callintel.ml.meld_classifier import MeldClassifier
from src.callintel.ml.churn_model import ChurnRiskPredictor
from src.callintel.agents.multi_agents import MultiAgentSupervisor

router = APIRouter(prefix="/api/v1/calls", tags=["Calls & Intelligence"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

repo = CallRepository()


class CallUploadResponse(BaseModel):
    call_id: str
    status: str
    filename: str
    poll_url: str


def process_audio_background(call_id: str, audio_path: Path):
    """Background worker pipeline executing ASR -> ML -> Agents -> DB."""
    try:
        # 1. ASR Transcription
        transcriber = WhisperTranscriber(model_size="base")
        asr_res = transcriber.transcribe_file(audio_path)

        # 2. Emotion & Sentiment
        meld = MeldClassifier()
        turn_emotions = meld.analyze_call_timeline(asr_res["turns"])

        # 3. Churn Predictor
        churn_pred = ChurnRiskPredictor()
        anger_peak = max(
            (t["emotion"]["probabilities"]["anger"] for t in turn_emotions), default=0.0
        )
        neg_ratio = sum(
            1 for t in turn_emotions if t["sentiment"]["label"] == "negative"
        ) / max(1, len(turn_emotions))

        churn_res = churn_pred.predict_churn(
            customer_profile={"CreditScore": 650.0, "Age": 45.0, "Balance": 60000.0},
            call_signals={
                "customer_frustration_score": anger_peak,
                "negative_sentiment_ratio": neg_ratio,
                "has_pricing_objection": any("expensive" in t["utterance"].lower() for t in asr_res["turns"]),
                "has_cancellation_warning": any("cancel" in t["utterance"].lower() for t in asr_res["turns"]),
            },
        )

        # 4. Multi-Agent Supervisor
        supervisor = MultiAgentSupervisor()
        report = supervisor.process_call(
            call_id=call_id,
            turns=asr_res["turns"],
            turn_emotions=turn_emotions,
            churn_prediction=churn_res,
        )

        # 5. Persist to DB
        repo.update_call_result(
            call_id=call_id,
            duration=asr_res["duration_seconds"],
            language=asr_res["language"],
            turns=asr_res["turns"],
            insight_report=report.model_dump(),
        )
    except Exception as e:
        repo.set_failed(call_id, str(e))


@router.post("/upload", response_model=CallUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_call_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Uploads a recorded customer call (WAV/MP3) for asynchronous intelligence processing.
    Returns 202 Accepted with poll URL.
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".ogg", ".flac")):
        raise HTTPException(status_code=400, detail="Invalid audio format. Supported: .wav, .mp3, .ogg, .flac")

    call_id = f"CALL_{uuid.uuid4().hex[:10].upper()}"
    save_path = UPLOAD_DIR / f"{call_id}_{file.filename}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    repo.create_call(call_id=call_id, filename=file.filename)
    background_tasks.add_task(process_audio_background, call_id, save_path)

    return CallUploadResponse(
        call_id=call_id,
        status="PENDING",
        filename=file.filename,
        poll_url=f"/api/v1/calls/{call_id}",
    )


@router.get("/{call_id}")
async def get_call_status(call_id: str):
    """Returns processing status, metadata, and high-level KPIs for a specific call."""
    record = repo.get_by_id(call_id)
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
async def get_call_transcript(call_id: str):
    """Returns timestamped dialogue turns with speaker attribution for a call."""
    record = repo.get_by_id(call_id)
    if not record:
        raise HTTPException(status_code=404, detail="Call record not found")

    return {
        "call_id": record.call_id,
        "status": record.status,
        "total_turns": len(record.transcript_turns or []),
        "turns": record.transcript_turns or [],
    }


@router.get("/{call_id}/insights")
async def get_call_insights(call_id: str):
    """Returns the Auditable Evidence Ledger, SHAP drivers, and action recommendations."""
    record = repo.get_by_id(call_id)
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
