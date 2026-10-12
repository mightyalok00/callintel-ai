"""
End-to-End Execution Script:
Connects Audio -> faster-whisper ASR -> MELD Emotion -> XGBoost Churn -> 6 AI Agents -> Auditable Evidence Ledger.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.callintel.speech.transcriber import WhisperTranscriber
from src.callintel.ml.meld_classifier import MeldClassifier
from src.callintel.ml.churn_model import ChurnRiskPredictor
from src.callintel.agents.multi_agents import MultiAgentSupervisor

AUDIO_PATH = Path("data/samples/sample_customer_call.wav")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_call_intelligence_pipeline():
    print("=" * 70)
    print("AI CALL INTELLIGENCE ENGINE: END-TO-END PIPELINE EXECUTION")
    print("=" * 70)

    # 1. SPEECH-TO-TEXT PIPELINE
    print("\n[Step 1/5] Transcribing Audio with faster-whisper...")
    transcriber = WhisperTranscriber(model_size="base")
    asr_result = transcriber.transcribe_file(AUDIO_PATH)
    print(f"[+] Audio Duration: {asr_result['duration_seconds']}s")
    print(f"[+] Total Turns Extracted: {asr_result['total_turns']}")

    # 2. MELD EMOTION & SENTIMENT INTELLIGENCE
    print("\n[Step 2/5] Running MELD Conversational Emotion & Sentiment Engine...")
    meld = MeldClassifier()
    turn_emotions = meld.analyze_call_timeline(asr_result["turns"])
    print(f"[+] Analyzed emotion trajectory across {len(turn_emotions)} turns.")

    # 3. PREDICTIVE CHURN MODEL & SHAP EXPLAINABILITY
    print("\n[Step 3/5] Computing Churn Risk with XGBoost & SHAP Explainability...")
    churn_pred = ChurnRiskPredictor()
    
    # Calculate aggregate conversation signals
    anger_peak = max(
        (t["emotion"]["probabilities"]["anger"] for t in turn_emotions), default=0.0
    )
    neg_ratio = sum(
        1 for t in turn_emotions if t["sentiment"]["label"] == "negative"
    ) / max(1, len(turn_emotions))

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
        "customer_frustration_score": anger_peak,
        "negative_sentiment_ratio": neg_ratio,
        "has_pricing_objection": True,
        "has_cancellation_warning": True,
        "unresolved_issues_count": 0,
    }
    churn_result = churn_pred.predict_churn(customer_profile, call_signals)
    print(f"[+] Baseline Churn Risk: {churn_result['baseline_probability']:.1%}")
    print(f"[+] Call Escalation Multiplier: {churn_result['call_multiplier']}x")
    print(f"[+] Final Churn Probability: {churn_result['final_churn_probability']:.1%} ({churn_result['risk_tier']})")

    # 4. MULTI-AGENT ORCHESTRATION & EVIDENCE LEDGER
    print("\n[Step 4/5] Orchestrating 6 Specialized AI Agents...")
    supervisor = MultiAgentSupervisor()
    call_id = "CALL_2026_APEX_001"
    
    insight_report = supervisor.process_call(
        call_id=call_id,
        turns=asr_result["turns"],
        turn_emotions=turn_emotions,
        churn_prediction=churn_result,
    )

    # 5. GENERATE FINAL REPORT
    print("\n[Step 5/5] Compiling Auditable Executive Report & Evidence Ledger...")
    report_dict = insight_report.model_dump()
    out_file = REPORTS_DIR / f"{call_id}_intelligence_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\n[OK] Intelligence Report successfully generated and saved to: {out_file}")
    print("\n" + "=" * 70)
    print("EXECUTIVE CALL REPORT SUMMARY")
    print("=" * 70)
    print(f"Call ID:           {insight_report.call_id}")
    print(f"Overall Sentiment: {insight_report.overall_sentiment.upper()} (Score: {insight_report.sentiment_score})")
    print(f"Churn Risk:        {insight_report.churn_risk_level.value} ({insight_report.churn_risk_score:.1%})")
    print(f"Primary Intent:    {insight_report.primary_intent.upper()}")
    print(f"QA Score:          {insight_report.metadata['qa_score']}% ({insight_report.metadata['qa_status']})")
    print(f"Evidence Items:    {len(insight_report.evidence_ledger)} verified quotes in ledger")
    print(f"Recommendations:   {len(insight_report.recommendations)} evidence-grounded action items")

    print("\nEVIDENCE LEDGER (Auditable Excerpts):")
    for item in insight_report.evidence_ledger:
        signals = ", ".join(f"{k}={v}" for k, v in item.supporting_signals.items() if v)
        print(f"  [{item.start_time} -> {item.end_time}] ({item.speaker}): \"{item.verbatim_quote}\"")
        print(f"    --> Signals: {signals}")

    print("\nACTION RECOMMENDATIONS:")
    for rec in insight_report.recommendations:
        print(f"  [{rec.priority.value}] {rec.title} (Owner: {rec.owner_role})")
        print(f"    Action Code: {rec.action_code}")
        print(f"    Description: {rec.description}")
        if rec.evidence:
            print(f"    Verified Evidence: \"{rec.evidence[0].verbatim_quote}\" [{rec.evidence[0].start_time}]")
    print("=" * 70)


if __name__ == "__main__":
    run_call_intelligence_pipeline()
