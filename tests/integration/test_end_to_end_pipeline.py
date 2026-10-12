"""
End-to-End Integration Tests for Call Intelligence Pipeline.
Validates the complete workflow on real audio files without mocks:
1. Audio loading & acoustic normalization
2. faster-whisper ASR + Acoustic Speaker Diarization
3. MELD Conversational Emotion & Sentiment scoring
4. Account-level tabular churn baseline + In-call operational modifier
5. Multi-Agent Supervisor & Auditable Evidence Ledger
6. SQLite ORM persistence & schema validation
"""

from pathlib import Path
import pytest

from src.callintel.speech.transcriber import WhisperTranscriber
from src.callintel.ml.meld_classifier import MeldClassifier
from src.callintel.ml.churn_model import ChurnRiskPredictor
from src.callintel.agents.multi_agents import MultiAgentSupervisor
from src.callintel.db.models import SessionLocal
from src.callintel.db.repository import CallRepository
from src.callintel.schemas.insight import CallInsightReport

SAMPLE_CALL = Path("data/samples/sample_customer_call.wav")
SAMPLE_SALES = Path("data/samples/sample_sales_expansion.wav")


@pytest.fixture(scope="module")
def setup_services():
    transcriber = WhisperTranscriber(model_size="base", device="cpu", compute_type="int8")
    meld = MeldClassifier()
    churn_pred = ChurnRiskPredictor()
    supervisor = MultiAgentSupervisor()
    return transcriber, meld, churn_pred, supervisor


def test_end_to_end_audio_to_insight_pipeline(setup_services):
    transcriber, meld, churn_pred, supervisor = setup_services

    assert SAMPLE_CALL.exists(), "Sample customer call audio file missing."

    # Step 1 & 2: Real Audio Transcription & Acoustic Speaker Diarization
    asr_result = transcriber.transcribe_file(SAMPLE_CALL)

    assert asr_result["total_turns"] >= 6
    assert asr_result["duration_seconds"] > 30.0
    turns = asr_result["turns"]

    # Verify acoustic speaker diarization attribution
    speakers = {t["speaker"] for t in turns}
    assert "Agent" in speakers
    assert "Customer" in speakers

    clusters = {t["speaker_cluster_id"] for t in turns}
    assert len(clusters) == 2

    for t in turns:
        assert t["attribution_method"] == "acoustic_voice_clustering"
        assert 0.70 <= t["attribution_confidence"] <= 0.99
        assert "mean_pitch_hz" in t["acoustic_metrics"]
        assert t["acoustic_metrics"]["mean_pitch_hz"] > 0.0

    # Step 3: MELD Emotion Timeline Analysis
    turn_emotions = meld.analyze_call_timeline(turns)
    assert len(turn_emotions) == len(turns)
    for te in turn_emotions:
        assert "emotion" in te
        assert "sentiment" in te
        assert te["emotion"]["label"] in ["neutral", "joy", "anger", "surprise", "sadness", "disgust", "fear"]
        assert te["sentiment"]["label"] in ["positive", "neutral", "negative"]

    # Step 4: Account Churn Baseline + In-Call Operational Modifier
    customer_profile = {
        "CreditScore": 640.0,
        "Age": 49.0,
        "Tenure": 2.0,
        "Balance": 88000.0,
        "NumOfProducts": 1.0,
        "HasCrCard": 1.0,
        "IsActiveMember": 0.0,
        "EstimatedSalary": 92000.0,
        "Rating": -2,
        "Geography": "Berlin-Germany",
        "Gender": "Female",
    }
    call_signals = {
        "customer_frustration_score": 0.45,
        "has_pricing_objection": True,
        "has_cancellation_warning": True,
        "unresolved_issues_count": 0,
    }
    churn_result = churn_pred.predict_churn(customer_profile, call_signals)

    assert "account_baseline_churn_probability" in churn_result
    assert 0.0 <= churn_result["account_baseline_churn_probability"] <= 1.0
    assert "in_call_operational_modifier" in churn_result
    assert churn_result["in_call_operational_modifier"]["multiplier"] >= 1.0
    assert len(churn_result["top_shap_drivers"]) > 0

    # Verify non-audio disclaimer in model output
    assert "tabular" in churn_result["statistical_validation"]["training_domain"].lower()

    # Step 5: Multi-Agent Processing & Evidence Ledger Generation
    import uuid
    call_id = f"INTEGRATION_TEST_{uuid.uuid4().hex[:8]}"
    insight_report = supervisor.process_call(
        call_id=call_id,
        turns=turns,
        turn_emotions=turn_emotions,
        churn_prediction=churn_result,
    )

    assert isinstance(insight_report, CallInsightReport)
    assert insight_report.call_id == call_id
    assert insight_report.churn_risk_score is not None and insight_report.churn_risk_score > 0.0
    assert len(insight_report.evidence_ledger) > 0

    # Verify that evidence claims link to exact verbatim quotes and speaker attributions
    for claim in insight_report.evidence_ledger:
        assert claim.speaker in ["Agent", "Customer"]
        assert len(claim.verbatim_quote) > 0
        assert ":" in claim.start_time

    # Step 6: ORM Database Persistence & Integrity
    db = SessionLocal()
    repo = CallRepository(db)
    try:
        call_rec = repo.create_call(call_id=call_id, filename="sample_customer_call.wav")
        assert call_rec.status == "PENDING"

        transcript_data = {
            "total_turns": len(turns),
            "speakers": list(speakers),
            "turns": turns,
        }
        repo.update_call_result(
            call_id=call_id,
            duration=asr_result["duration_seconds"],
            language=asr_result.get("language", "en"),
            turns=turns,
            insight_report=insight_report.model_dump(),
        )

        persisted = repo.get_by_id(call_id)
        assert persisted is not None
        assert persisted.status == "COMPLETED"
        assert persisted.duration_seconds == asr_result["duration_seconds"]
        assert len(persisted.transcript_turns) == len(turns)
        assert persisted.churn_risk_score == insight_report.churn_risk_score
    finally:
        db.close()
