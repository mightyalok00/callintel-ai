"""
Unit tests for EvidenceValidator and AI Agent Evidence Grounding.
Verifies citation grounding, nonexistent segment rejection, fabricated quote detection,
explicit abstention, semantic paraphrases, subtle contradictions, misleading quotations,
and claims that cite valid segments but are not actually supported by them.
"""

import pytest
from src.callintel.schemas.insight import (
    AgentFinding,
    EvidenceStatus,
    FindingCategory,
)
from src.callintel.agents.evidence_validator import EvidenceValidator


@pytest.fixture
def sample_turns():
    return [
        {
            "turn_index": 0,
            "speaker": "Agent",
            "utterance": "Thank you for calling customer care, my name is Alex. How can I help you today?",
            "start_time": "00:00:01.000",
            "end_time": "00:00:04.500",
        },
        {
            "turn_index": 1,
            "speaker": "Customer",
            "utterance": "I saw my bill went up by twenty dollars and I cannot afford this. I want to cancel my subscription.",
            "start_time": "00:00:05.000",
            "end_time": "00:00:10.200",
        },
        {
            "turn_index": 2,
            "speaker": "Agent",
            "utterance": "I understand your concern completely. I can apply a fifteen percent loyalty discount to your plan.",
            "start_time": "00:00:11.000",
            "end_time": "00:00:16.800",
        },
        {
            "turn_index": 3,
            "speaker": "Customer",
            "utterance": "That sounds reasonable, go ahead and apply it. Thank you.",
            "start_time": "00:00:17.500",
            "end_time": "00:00:20.000",
        },
    ]


@pytest.fixture
def technical_outage_turns():
    return [
        {
            "turn_index": 0,
            "speaker": "Agent",
            "utterance": "Apex Cloud Technical Support line, my name is David. What issue are you experiencing today?",
            "start_time": "00:00:01.000",
            "end_time": "00:00:05.000",
        },
        {
            "turn_index": 1,
            "speaker": "Customer",
            "utterance": "David, our production database synchronization stopped working two hours ago and our transactions are failing.",
            "start_time": "00:00:06.000",
            "end_time": "00:00:14.000",
        },
        {
            "turn_index": 2,
            "speaker": "Customer",
            "utterance": "We are losing revenue every minute this persists. We need this escalated to tier three engineering right away.",
            "start_time": "00:00:15.000",
            "end_time": "00:00:23.000",
        },
    ]


def test_validator_supported_claim(sample_turns):
    validator = EvidenceValidator()
    finding = AgentFinding(
        claim_id="FND_TEST_01",
        claim="Customer requested cancellation due to bill increase.",
        category=FindingCategory.CHURN_RISK,
        evidence_status=EvidenceStatus.SUPPORTED,
        supporting_segment_ids=[1],
        supporting_timestamps=["00:00:05.000 - 00:00:10.200"],
        verbatim_citations=["I want to cancel my subscription"],
        confidence_score=0.92,
        confidence_rationale="Verbatim cancellation request in turn 1.",
        recommended_action="Offer retention discount.",
    )

    validated = validator.validate_finding(finding, sample_turns)
    assert validated.evidence_status == EvidenceStatus.SUPPORTED
    assert validated.confidence_score == 0.92
    assert 1 in validated.supporting_segment_ids
    assert "Verified" in (validated.validator_notes or "")


def test_validator_rejects_nonexistent_segment_id(sample_turns):
    validator = EvidenceValidator()
    finding = AgentFinding(
        claim_id="FND_TEST_BAD_SEG",
        claim="Customer complained about website downtime.",
        category=FindingCategory.GENERAL,
        evidence_status=EvidenceStatus.SUPPORTED,
        supporting_segment_ids=[99],  # Does not exist
        supporting_timestamps=["00:05:00.000"],
        verbatim_citations=["The website was down all morning"],
        confidence_score=0.88,
        confidence_rationale="Claim from non-existent segment.",
        recommended_action="File technical ticket.",
    )

    validated = validator.validate_finding(finding, sample_turns)
    assert validated.evidence_status == EvidenceStatus.INSUFFICIENT
    assert validated.confidence_score == 0.0
    assert "Abstain" in (validated.recommended_action or "")
    assert "Rejected nonexistent segment IDs: [99]" in (validated.validator_notes or "")


def test_validator_partial_nonexistent_segment(sample_turns):
    validator = EvidenceValidator()
    finding = AgentFinding(
        claim_id="FND_TEST_PARTIAL",
        claim="Agent offered loyalty discount.",
        category=FindingCategory.AGENT_COMMITMENT,
        evidence_status=EvidenceStatus.SUPPORTED,
        supporting_segment_ids=[2, 88],  # 2 is valid, 88 is nonexistent
        supporting_timestamps=["00:00:11.000 - 00:00:16.800"],
        verbatim_citations=["fifteen percent loyalty discount"],
        confidence_score=0.90,
        confidence_rationale="Turn 2 contains discount offer.",
        recommended_action="Confirm discount.",
    )

    validated = validator.validate_finding(finding, sample_turns)
    assert validated.evidence_status == EvidenceStatus.SUPPORTED
    assert 2 in validated.supporting_segment_ids
    assert 88 not in validated.supporting_segment_ids
    assert validated.confidence_score < 0.90
    assert "Rejected nonexistent segment IDs: [88]" in (validated.validator_notes or "")


def test_validator_fabricated_citation_abstains(sample_turns):
    validator = EvidenceValidator()
    finding = AgentFinding(
        claim_id="FND_TEST_FABRICATED",
        claim="Customer demanded to speak with the CEO directly.",
        category=FindingCategory.ESCALATION,
        evidence_status=EvidenceStatus.SUPPORTED,
        supporting_segment_ids=[1],
        supporting_timestamps=["00:00:05.000 - 00:00:10.200"],
        verbatim_citations=["Let me talk to your CEO right now or I sue"],
        confidence_score=0.85,
        confidence_rationale="Fabricated quote claim.",
        recommended_action="Escalate to executive team.",
    )

    validated = validator.validate_finding(finding, sample_turns)
    assert validated.evidence_status == EvidenceStatus.INSUFFICIENT
    assert "Abstain" in (validated.recommended_action or "")
    assert "failed transcript verification" in (validated.confidence_rationale or "")


def test_validator_empty_evidence_forces_abstention(sample_turns):
    validator = EvidenceValidator()
    finding = AgentFinding(
        claim_id="FND_TEST_NO_EVIDENCE",
        claim="Customer has severe security concerns.",
        category=FindingCategory.GENERAL,
        evidence_status=EvidenceStatus.SUPPORTED,
        supporting_segment_ids=[],
        supporting_timestamps=[],
        verbatim_citations=[],
        confidence_score=0.75,
        confidence_rationale="Model hallucination without supporting turns.",
        recommended_action="Schedule security audit.",
    )

    validated = validator.validate_finding(finding, sample_turns)
    assert validated.evidence_status == EvidenceStatus.INSUFFICIENT
    assert validated.confidence_score == 0.0
    assert "Abstain" in (validated.recommended_action or "")
    assert "zero supporting segment references" in (validated.validator_notes or "")


def test_validator_claim_cites_valid_segment_without_factual_support(sample_turns):
    """
    Critical Test: Finding cites a valid segment ID (turn 0: greeting), but the claim
    asserts an entirely unsupported fact (database data loss / billing fraud).
    Must detect segment support failure and abstain.
    """
    validator = EvidenceValidator()
    finding = AgentFinding(
        claim_id="FND_UNSUPPORTED_SEG",
        claim="Customer reported severe database data loss and demanded regulatory investigation.",
        category=FindingCategory.ESCALATION,
        evidence_status=EvidenceStatus.SUPPORTED,
        supporting_segment_ids=[0],  # Valid segment ID, but turn 0 is just "Thank you for calling..."
        supporting_timestamps=["00:00:01.000 - 00:00:04.500"],
        verbatim_citations=[],
        confidence_score=0.85,
        confidence_rationale="Cites turn 0.",
        recommended_action="File regulatory report.",
    )

    validated = validator.validate_finding(finding, sample_turns)
    assert validated.evidence_status == EvidenceStatus.INSUFFICIENT
    assert validated.confidence_score == 0.0
    assert "Abstain" in (validated.recommended_action or "")
    assert "Segment Support Failure" in (validated.validator_notes or "")


def test_validator_paraphrase_support(sample_turns):
    """
    Verifies that a genuine semantic paraphrase is recognized as SUPPORTED even without verbatim quotes.
    Turn 1 states: "I saw my bill went up by twenty dollars and I cannot afford this. I want to cancel my subscription."
    Paraphrase: "Customer indicated churn risk due to unexpected billing increases."
    """
    validator = EvidenceValidator()
    finding = AgentFinding(
        claim_id="FND_PARAPHRASE",
        claim="Customer indicated churn risk due to unexpected billing increases.",
        category=FindingCategory.CHURN_RISK,
        evidence_status=EvidenceStatus.SUPPORTED,
        supporting_segment_ids=[1],
        supporting_timestamps=["00:00:05.000 - 00:00:10.200"],
        verbatim_citations=[],
        confidence_score=0.88,
        confidence_rationale="Derived from customer turn 1.",
        recommended_action="Offer billing adjustment.",
    )

    validated = validator.validate_finding(finding, sample_turns)
    assert validated.evidence_status == EvidenceStatus.SUPPORTED
    assert validated.confidence_score >= 0.80
    assert "semantic concept alignment" in (validated.validator_notes or "")


def test_validator_misleading_quotation_negation(sample_turns):
    """
    Tests detection of misleading quotation where a phrase is quoted out of context,
    cherry-picking words negated in the original turn ("cannot afford this").
    """
    validator = EvidenceValidator()
    finding = AgentFinding(
        claim_id="FND_MISLEADING_QUOTE",
        claim="Customer stated they afford this pricing structure.",
        category=FindingCategory.PRICING_OBJECTION,
        evidence_status=EvidenceStatus.SUPPORTED,
        supporting_segment_ids=[1],
        supporting_timestamps=["00:00:05.000 - 00:00:10.200"],
        verbatim_citations=["afford this"],  # Original context: "I cannot afford this"
        confidence_score=0.80,
        confidence_rationale="Quoted 'afford this' from turn 1.",
        recommended_action="Proceed to higher tier.",
    )

    validated = validator.validate_finding(finding, sample_turns)
    assert validated.evidence_status == EvidenceStatus.CONTRADICTED
    assert validated.confidence_score == 0.0
    assert "Misleading quotation" in (validated.validator_notes or "")
    assert "Contradiction alert" in (validated.recommended_action or "")


def test_validator_subtle_outage_contradiction(technical_outage_turns):
    """
    Tests subtle contradiction detection: Claim asserts operational stability and uptime,
    while cited turns document database synchronization failure and transaction stoppage.
    """
    validator = EvidenceValidator()
    finding = AgentFinding(
        claim_id="FND_CONTRADICT_OUTAGE",
        claim="Customer confirmed operational stability and zero outage on production cluster.",
        category=FindingCategory.GENERAL,
        evidence_status=EvidenceStatus.SUPPORTED,
        supporting_segment_ids=[1, 2],
        supporting_timestamps=["00:00:06.000 - 00:00:23.000"],
        verbatim_citations=[],
        confidence_score=0.90,
        confidence_rationale="System audit.",
        recommended_action="Log stable operational status.",
    )

    validated = validator.validate_finding(finding, technical_outage_turns)
    assert validated.evidence_status == EvidenceStatus.CONTRADICTED
    assert validated.confidence_score == 0.0
    assert "Contradiction detected" in (validated.validator_notes or "")
    assert "operational stability" in (validated.validator_notes or "")


def test_validator_discount_numerical_contradiction(sample_turns):
    """
    Tests numerical discrepancy contradiction: Agent offered 15% discount,
    but claim asserts agent offered a 50% discount.
    """
    validator = EvidenceValidator()
    finding = AgentFinding(
        claim_id="FND_CONTRADICT_NUMERICAL",
        claim="Agent agreed and applied a fifty percent loyalty discount.",
        category=FindingCategory.AGENT_COMMITMENT,
        evidence_status=EvidenceStatus.SUPPORTED,
        supporting_segment_ids=[2],
        supporting_timestamps=["00:00:11.000 - 00:00:16.800"],
        verbatim_citations=[],
        confidence_score=0.88,
        confidence_rationale="Turn 2 contains discount.",
        recommended_action="Verify discount rate in billing system.",
    )

    validated = validator.validate_finding(finding, sample_turns)
    assert validated.evidence_status == EvidenceStatus.CONTRADICTED
    assert validated.confidence_score == 0.0
    assert "fifty percent discount" in (validated.validator_notes or "")


def test_validator_audit_summary(sample_turns):
    validator = EvidenceValidator()
    findings = [
        AgentFinding(
            claim_id="F1",
            claim="Customer wants to cancel subscription due to cost.",
            category=FindingCategory.CHURN_RISK,
            evidence_status=EvidenceStatus.SUPPORTED,
            supporting_segment_ids=[1],
            supporting_timestamps=["00:00:05.000"],
            verbatim_citations=["cancel my subscription"],
            confidence_score=0.90,
            confidence_rationale="Valid",
        ),
        AgentFinding(
            claim_id="F2",
            claim="Unsupported claim with no segments",
            category=FindingCategory.GENERAL,
            evidence_status=EvidenceStatus.SUPPORTED,
            supporting_segment_ids=[],
            supporting_timestamps=[],
            verbatim_citations=[],
            confidence_score=0.50,
            confidence_rationale="None",
        ),
    ]

    validated_list = validator.validate_findings(findings, sample_turns)
    audit = validator.audit_summary(validated_list)

    assert audit["total_findings"] == 2
    assert audit["supported_count"] == 1
    assert audit["insufficient_count"] == 1
    assert audit["grounding_pass_rate"] == 0.5
