"""
Repository pattern for CallRecord persistence and querying.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from src.callintel.db.models import CallRecord, SessionLocal


class CallRepository:
    """Manages database operations for call transcripts and intelligence reports."""

    def __init__(self, db_session: Optional[Session] = None):
        self._session_factory = SessionLocal
        self._db = db_session

    def _get_db(self) -> Session:
        return self._db or self._session_factory()

    def create_call(self, call_id: str, filename: str) -> CallRecord:
        db = self._get_db()
        record = CallRecord(call_id=call_id, filename=filename, status="PENDING")
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    def get_by_id(self, call_id: str) -> Optional[CallRecord]:
        db = self._get_db()
        return db.query(CallRecord).filter(CallRecord.call_id == call_id).first()

    def list_calls(self, limit: int = 50, offset: int = 0) -> List[CallRecord]:
        db = self._get_db()
        return db.query(CallRecord).order_by(CallRecord.created_at.desc()).offset(offset).limit(limit).all()

    def update_call_result(
        self,
        call_id: str,
        duration: float,
        language: str,
        turns: List[Dict[str, Any]],
        insight_report: Dict[str, Any],
    ) -> Optional[CallRecord]:
        db = self._get_db()
        record = db.query(CallRecord).filter(CallRecord.call_id == call_id).first()
        if not record:
            return None

        record.status = "COMPLETED"
        record.duration_seconds = duration
        record.language = language
        record.overall_sentiment = insight_report.get("overall_sentiment")
        record.sentiment_score = insight_report.get("sentiment_score")
        record.churn_risk_level = insight_report.get("churn_risk_level")
        record.churn_risk_score = insight_report.get("churn_risk_score")
        record.primary_intent = insight_report.get("primary_intent")
        record.qa_score = insight_report.get("metadata", {}).get("qa_score")
        record.qa_status = insight_report.get("metadata", {}).get("qa_status")
        record.transcript_turns = turns
        record.evidence_ledger = insight_report.get("evidence_ledger", [])
        record.recommendations = insight_report.get("recommendations", [])
        record.metadata_payload = insight_report.get("metadata", {})

        db.commit()
        db.refresh(record)
        return record

    def set_failed(self, call_id: str, error_msg: str):
        db = self._get_db()
        record = db.query(CallRecord).filter(CallRecord.call_id == call_id).first()
        if record:
            record.status = "FAILED"
            record.metadata_payload = {"error": error_msg}
            db.commit()
