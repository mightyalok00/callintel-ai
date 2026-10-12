"""
REST Endpoints for Call Audio Ingestion, Transcript Retrieval, and Insights.
Built with clean Presentation Layer architecture and Dependency Injection.
"""

import re
import time
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
    verify_api_key,
)
from src.callintel.db.repository import CallRepository

router = APIRouter(prefix="/api/v1/calls", tags=["Calls & Intelligence"])


def sanitize_filename(raw_filename: Optional[str]) -> str:
    """
    Sanitizes upload filenames by stripping directory traversals and dangerous characters.
    """
    if not raw_filename:
        return f"audio_{uuid.uuid4().hex[:8]}.wav"
    # Extract only basename (defense against path traversal attacks)
    base_name = Path(raw_filename).name
    # Keep only alphanumeric, dot, underscore, dash
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", base_name)
    sanitized = re.sub(r"^\.+", "", sanitized)  # strip leading dots
    if not sanitized or len(sanitized) < 4:
        return f"audio_{uuid.uuid4().hex[:8]}.wav"
    return sanitized[:128]  # cap length


def validate_audio_signature(header: bytes, ext: str) -> bool:
    """
    Validates audio file magic bytes against declared extension.
    Guards against disguised malicious payloads.
    """
    ext = ext.lower()
    if ext == ".wav":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    elif ext == ".mp3":
        # ID3 container or raw syncword frame (\xff\xfb, \xff\xf3, \xff\xf2)
        if header.startswith(b"ID3"):
            return True
        if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
            return True
        return False
    elif ext == ".ogg":
        return header.startswith(b"OggS")
    elif ext == ".flac":
        return header.startswith(b"fLaC")
    return False


def cleanup_stale_uploads(max_age_hours: int = 24) -> int:
    """Purges uploaded files older than the retention threshold."""
    purged = 0
    now = time.time()
    cutoff = now - (max_age_hours * 3600)
    try:
        for f in settings.upload_dir.glob("*"):
            if f.is_file() and f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                    purged += 1
                except Exception as e:
                    logger.warning(f"Failed to delete stale upload {f}: {e}")
    except Exception as ex:
        logger.error(f"Error during upload directory cleanup: {ex}")
    return purged


class CallUploadResponse(BaseModel):
    call_id: str
    status: str
    filename: str
    poll_url: str


def run_pipeline_worker(call_id: str, audio_path: Path):
    """Background worker invoking CallAnalysisService with injected singletons and temp cleanup."""
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
    finally:
        # Cleanup temporary upload artifact if configured
        if settings.cleanup_temp_files and audio_path.exists():
            try:
                audio_path.unlink()
                logger.info(f"Cleaned up temporary audio upload artifact: {audio_path.name}")
            except Exception as clean_err:
                logger.warning(f"Could not remove temporary file {audio_path}: {clean_err}")


@router.post("/upload", response_model=CallUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_call_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    service: CallAnalysisService = Depends(get_call_analysis_service),
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Hardened upload endpoint for recorded customer calls.
    Enforces:
    - Authentication verification via X-API-Key / Bearer token
    - Sanitized filename parsing (anti-path-traversal)
    - File extension and magic bytes header validation
    - Enforced maximum upload payload size limits (HTTP 413)
    - Safe temporary file storage and automated cleanup
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing filename in upload payload.",
        )

    safe_name = sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lower()
    allowed_exts = {".wav", ".mp3", ".ogg", ".flac"}

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audio format '{ext}'. Supported formats: {', '.join(sorted(allowed_exts))}",
        )

    call_id = f"CALL_{uuid.uuid4().hex[:10].upper()}"
    save_path = settings.upload_dir / f"{call_id}_{safe_name}"

    # Stream file in chunks while enforcing size limits and validating magic bytes
    chunk_size = 64 * 1024  # 64 KB
    total_bytes = 0
    header_checked = False

    try:
        with open(save_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break

                if not header_checked:
                    if not validate_audio_signature(chunk, ext):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid audio content signature. File header does not match declared format '{ext}'.",
                        )
                    header_checked = True

                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_size_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=(
                            f"File size exceeds maximum allowed upload limit of "
                            f"{settings.max_upload_size_bytes // (1024 * 1024)} MB."
                        ),
                    )
                buffer.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )

    except Exception:
        # Clean up partial / corrupted file on any error
        if save_path.exists():
            save_path.unlink()
        raise

    # Trigger background cleanup of any stale temporary files
    background_tasks.add_task(cleanup_stale_uploads, settings.upload_retention_hours)

    service.create_pending_call(call_id=call_id, filename=safe_name)
    background_tasks.add_task(run_pipeline_worker, call_id, save_path)

    return CallUploadResponse(
        call_id=call_id,
        status="PENDING",
        filename=safe_name,
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


@router.delete("/history")
async def delete_all_call_history(
    service: CallAnalysisService = Depends(get_call_analysis_service),
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Purges all call records and temporary staged upload files from the system.
    """
    deleted_count = service.clear_history()
    # Clean up staged files
    purged_files = cleanup_stale_uploads(max_age_hours=0)
    return {
        "status": "SUCCESS",
        "message": f"Successfully deleted call history. Cleared {deleted_count} record(s) and {purged_files} file(s).",
        "deleted_records": deleted_count,
        "purged_files": purged_files,
    }


@router.delete("/{call_id}")
async def delete_call_record(
    call_id: str,
    service: CallAnalysisService = Depends(get_call_analysis_service),
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Deletes a specific call record by ID."""
    success = service.delete_call(call_id)
    if not success:
        raise HTTPException(status_code=404, detail="Call record not found.")
    return {
        "status": "SUCCESS",
        "message": f"Call record {call_id} deleted successfully.",
        "call_id": call_id,
    }
