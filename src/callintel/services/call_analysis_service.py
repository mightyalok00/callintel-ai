"""
Application Service Layer: CallAnalysisService.
Coordinates speech transcription, acoustic diarization, emotion classification,
statistically validated churn risk prediction, auditable evidence ledger compilation,
and database persistence.
Completely decouples business logic from presentation routers (FastAPI, Streamlit, WebSockets).
"""

from pathlib import Path
from typing import Dict, Any, Optional, List

from src.callintel.db.repository import CallRepository
from src.callintel.speech.transcriber import WhisperTranscriber
from src.callintel.ml.meld_classifier import MeldClassifier
from src.callintel.ml.churn_model import ChurnRiskPredictor
from src.callintel.agents.multi_agents import MultiAgentSupervisor
from src.callintel.schemas.insight import CallInsightReport
from src.callintel.core.logging import logger


class CallAnalysisService:
    """
    Application Layer Service orchestrating the end-to-end Call Intelligence pipeline.
    Uses Dependency Injection for all model and persistence components.
    """

    def __init__(
        self,
        repo: CallRepository,
        transcriber: WhisperTranscriber,
        meld: MeldClassifier,
        churn_predictor: ChurnRiskPredictor,
        supervisor: MultiAgentSupervisor,
    ):
        self.repo = repo
        self.transcriber = transcriber
        self.meld = meld
        self.churn_predictor = churn_predictor
        self.supervisor = supervisor

    def process_call_audio(
        self,
        call_id: str,
        audio_path: Path,
        customer_profile: Optional[Dict[str, Any]] = None,
    ) -> CallInsightReport:
        """
        Executes the full pipeline:
        ASR & Diarization -> Emotion Dynamics -> Churn Risk Model -> Multi-Agent Evidence Grounding -> ORM Persistence.
        """
        logger.info(f"Starting analysis for call {call_id} from {audio_path}")

        # 1. ASR & Acoustic Voice Diarization
        asr_res = self.transcriber.transcribe_file(audio_path)
        turns = asr_res.get("turns", [])
        duration_s = asr_res.get("duration_seconds", 0.0)
        language = asr_res.get("language", "en")

        # 2. Emotion Timeline Analysis
        turn_emotions = self.meld.analyze_call_timeline(turns)

        # 3. Churn Prediction with decoupled operational modifier
        profile = customer_profile or {"CreditScore": 650.0, "Age": 45.0, "Balance": 60000.0}
        anger_peak = max(
            (t["emotion"]["probabilities"]["anger"] for t in turn_emotions), default=0.0
        )
        neg_ratio = sum(
            1 for t in turn_emotions if t["sentiment"]["label"] == "negative"
        ) / max(1, len(turn_emotions))

        call_signals = {
            "customer_frustration_score": anger_peak,
            "negative_sentiment_ratio": neg_ratio,
            "has_pricing_objection": any("expensive" in t["utterance"].lower() for t in turns),
            "has_cancellation_warning": any("cancel" in t["utterance"].lower() for t in turns),
        }
        churn_res = self.churn_predictor.predict_churn(
            customer_profile=profile,
            call_signals=call_signals,
        )

        # 4. Multi-Agent Supervisor & Evidence Grounding
        insight_report = self.supervisor.process_call(
            call_id=call_id,
            turns=turns,
            turn_emotions=turn_emotions,
            churn_prediction=churn_res,
        )

        # 5. Persist to Repository
        self.repo.update_call_result(
            call_id=call_id,
            duration=duration_s,
            language=language,
            turns=turns,
            insight_report=insight_report.model_dump(),
        )

        logger.info(f"Completed analysis for call {call_id}: Churn={insight_report.churn_risk_level.value}, Sentiment={insight_report.overall_sentiment}")
        return insight_report

    def get_call(self, call_id: str):
        return self.repo.get_call_by_id(call_id)

    def list_calls(self, limit: int = 50, offset: int = 0):
        return self.repo.list_calls(limit=limit, offset=offset)

    def create_pending_call(self, call_id: str, filename: str):
        return self.repo.create_call(call_id=call_id, filename=filename)

    def get_analytics_summary(self) -> Dict[str, Any]:
        calls = self.repo.list_calls(limit=1000)
        total = len(calls)
        completed = sum(1 for c in calls if c.status == "COMPLETED")
        failed = sum(1 for c in calls if c.status == "FAILED")
        pending = sum(1 for c in calls if c.status in ["PENDING", "PROCESSING"])

        avg_churn = 0.0
        completed_scores: List[float] = [
            float(c.churn_risk_score)
            for c in calls
            if c.status == "COMPLETED" and c.churn_risk_score is not None
        ]
        if completed_scores:
            avg_churn = round(sum(completed_scores) / len(completed_scores), 3)

        return {
            "total_calls": total,
            "completed_calls": completed,
            "failed_calls": failed,
            "pending_calls": pending,
            "mean_churn_probability": avg_churn,
        }

    def delete_call(self, call_id: str) -> bool:
        """Deletes a call from database records."""
        return self.repo.delete_call(call_id)

    def clear_history(self) -> int:
        """Purges all call records from database."""
        return self.repo.clear_history()
