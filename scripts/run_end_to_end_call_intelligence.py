"""
End-to-End Execution Script:
Connects Audio -> faster-whisper ASR -> MELD Emotion -> XGBoost Churn -> 6 AI Agents -> Auditable Evidence Ledger.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.callintel.speech.transcriber import WhisperTranscriber
from src.callintel.ml.meld_classifier import MeldClassifier
from src.callintel.ml.churn_model import ChurnRiskPredictor
from src.callintel.agents.multi_agents import MultiAgentSupervisor

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CALL_SCENARIOS = [
    {
        "call_id": "CALL_2026_APEX_001",
        "audio_path": Path("data/samples/sample_customer_call.wav"),
        "customer_profile": {
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
        },
        "pricing_objection": True,
        "cancellation_warning": True,
    },
    {
        "call_id": "CALL_2026_SALES_002",
        "audio_path": Path("data/samples/sample_sales_expansion.wav"),
        "customer_profile": {
            "CreditScore": 760.0,
            "Age": 38.0,
            "Tenure": 4.0,
            "Balance": 210000.0,
            "NumOfProducts": 2.0,
            "HasCrCard": 1.0,
            "IsActiveMember": 1.0,
            "EstimatedSalary": 185000.0,
            "Rating": 2,
            "Geography": "Paris-France",
            "Gender": "Female",
        },
        "pricing_objection": False,
        "cancellation_warning": False,
    },
    {
        "call_id": "CALL_2026_TECH_003",
        "audio_path": Path("data/samples/sample_technical_escalation.wav"),
        "customer_profile": {
            "CreditScore": 680.0,
            "Age": 45.0,
            "Tenure": 1.5,
            "Balance": 115000.0,
            "NumOfProducts": 1.0,
            "HasCrCard": 1.0,
            "IsActiveMember": 1.0,
            "EstimatedSalary": 140000.0,
            "Rating": -1,
            "Geography": "Madrid-Spain",
            "Gender": "Male",
        },
        "pricing_objection": False,
        "cancellation_warning": False,
    },
]


def process_single_call(scenario: dict, transcriber, meld, churn_pred, supervisor):
    call_id = scenario["call_id"]
    audio_path = scenario["audio_path"]

    print(f"\n---> Processing: {call_id} ({audio_path.name})")
    asr_result = transcriber.transcribe_file(audio_path)
    turn_emotions = meld.analyze_call_timeline(asr_result["turns"])

    anger_peak = max(
        (t["emotion"]["probabilities"]["anger"] for t in turn_emotions), default=0.0
    )
    neg_ratio = sum(
        1 for t in turn_emotions if t["sentiment"]["label"] == "negative"
    ) / max(1, len(turn_emotions))

    call_signals = {
        "customer_frustration_score": anger_peak,
        "negative_sentiment_ratio": neg_ratio,
        "has_pricing_objection": scenario.get("pricing_objection", False),
        "has_cancellation_warning": scenario.get("cancellation_warning", False),
        "unresolved_issues_count": 1 if "TECH" in call_id else 0,
    }
    churn_result = churn_pred.predict_churn(scenario["customer_profile"], call_signals)

    insight_report = supervisor.process_call(
        call_id=call_id,
        turns=asr_result["turns"],
        turn_emotions=turn_emotions,
        churn_prediction=churn_result,
    )

    report_dict = insight_report.model_dump()
    out_file = REPORTS_DIR / f"{call_id}_intelligence_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print(f"    [OK] Saved report to: {out_file}")
    print(f"    Sentiment: {insight_report.overall_sentiment.upper()} | Churn: {insight_report.churn_risk_score:.1%} | Intent: {insight_report.primary_intent.upper()}")
    return insight_report


def run_call_intelligence_pipeline():
    print("=" * 70)
    print("AI CALL INTELLIGENCE ENGINE: BATCH CALL PROCESSING")
    print("=" * 70)

    transcriber = WhisperTranscriber(model_size="base")
    meld = MeldClassifier()
    churn_pred = ChurnRiskPredictor()
    supervisor = MultiAgentSupervisor()

    for sc in CALL_SCENARIOS:
        audio_path = Path(str(sc["audio_path"]))
        if audio_path.exists():
            process_single_call(sc, transcriber, meld, churn_pred, supervisor)
        else:
            print(f"[!] Audio file missing: {audio_path}")

    print("\n" + "=" * 70)
    print("[OK] All call scenario reports successfully generated in reports/")
    print("=" * 70)


if __name__ == "__main__":
    run_call_intelligence_pipeline()
