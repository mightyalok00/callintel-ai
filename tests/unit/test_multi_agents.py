"""
Unit tests for the Evidence Ledger and the 6 Specialized AI Agents.
"""

import pytest
from src.callintel.agents.evidence_ledger import EvidenceLedgerEngine
from src.callintel.agents.multi_agents import MultiAgentSupervisor


def test_evidence_ledger_extraction():
    engine = EvidenceLedgerEngine()
    turns = [
        {"turn_index": 0, "speaker": "Agent", "utterance": "Thank you for calling. How can I help?", "start_time": "00:00:00.000", "end_time": "00:00:03.000"},
        {"turn_index": 1, "speaker": "Customer", "utterance": "Your product is too expensive, I might cancel my subscription.", "start_time": "00:00:03.500", "end_time": "00:00:08.200"},
        {"turn_index": 2, "speaker": "Agent", "utterance": "I can apply a 15% discount for you.", "start_time": "00:00:08.500", "end_time": "00:00:12.000"},
    ]
    ledger = engine.build_ledger(turns)
    assert len(ledger) == 2  # Turn 1 (objection + cancellation) and Turn 2 (discount commitment)

    cancellation_items = engine.filter_by_signal(ledger, "is_cancellation_warning")
    assert len(cancellation_items) == 1
    assert "cancel" in cancellation_items[0].verbatim_quote.lower()
    assert cancellation_items[0].start_time == "00:00:03.500"


def test_multi_agent_supervisor_pipeline():
    supervisor = MultiAgentSupervisor()
    turns = [
        {"turn_index": 0, "speaker": "Agent", "utterance": "Hello, thank you for calling support. My name is Alex."},
        {"turn_index": 1, "speaker": "Customer", "utterance": "Your service is too expensive. I might cancel if it goes up."},
        {"turn_index": 2, "speaker": "Agent", "utterance": "I understand. I can apply a fifteen percent discount."},
        {"turn_index": 3, "speaker": "Customer", "utterance": "That sounds reasonable, please go ahead. Thank you."},
    ]
    churn_info = {
        "risk_tier": "CRITICAL",
        "final_churn_probability": 0.88,
        "top_shap_drivers": [{"feature": "Age", "impact": 1.2}],
    }

    report = supervisor.process_call("test_call_001", turns, churn_prediction=churn_info)

    assert report.call_id == "test_call_001"
    assert report.churn_risk_level.value == "CRITICAL"
    assert len(report.recommendations) > 0

    # Ensure all recommendations cite evidence
    retention_rec = [r for r in report.recommendations if r.action_code == "ACT_RETENTION_LOCK_DISCOUNT"][0]
    assert retention_rec.priority.value == "P1_URGENT"
    assert len(retention_rec.evidence) > 0
    assert "cancel" in retention_rec.evidence[0].verbatim_quote.lower()
