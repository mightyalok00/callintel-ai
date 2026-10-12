"""
Data contracts for Evidence Ledger items, AI insights, and executive business reports.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    MEDIUM = "MEDIUM"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ActionPriority(str, Enum):
    P1_URGENT = "P1_URGENT"
    P2_HIGH = "P2_HIGH"
    P3_MEDIUM = "P3_MEDIUM"
    P4_ROUTINE = "P4_ROUTINE"


class EvidenceStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


class FindingCategory(str, Enum):
    CHURN_RISK = "CHURN_RISK"
    PRICING_OBJECTION = "PRICING_OBJECTION"
    AGENT_COMMITMENT = "AGENT_COMMITMENT"
    COMPLIANCE_GAP = "COMPLIANCE_GAP"
    ESCALATION = "ESCALATION"
    CUSTOMER_SENTIMENT = "CUSTOMER_SENTIMENT"
    GENERAL = "GENERAL"


class AgentFinding(BaseModel):
    """
    Structured, evidence-grounded finding produced by specialist AI agents.
    Every claim must be traceable to explicit transcript segment IDs and timestamps.
    """
    claim_id: str
    claim: str
    category: FindingCategory = FindingCategory.GENERAL
    evidence_status: EvidenceStatus = EvidenceStatus.SUPPORTED
    supporting_segment_ids: List[int] = Field(default_factory=list)
    supporting_timestamps: List[str] = Field(default_factory=list)
    verbatim_citations: List[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    confidence_rationale: str = Field(..., description="Defined explanation for the assigned confidence score")
    recommended_action: Optional[str] = None
    validator_notes: Optional[str] = None


class EvidenceLedgerItem(BaseModel):
    """
    Signature Capability: Links every insight, score, or recommendation back to
    exact timestamped turns and verbatim quotes.
    """
    turn_id: int
    speaker: str
    start_time: str
    end_time: str
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    verbatim_quote: str
    supporting_signals: Dict[str, Any] = Field(default_factory=dict)


class ActionRecommendation(BaseModel):
    action_code: str
    title: str
    description: str
    priority: ActionPriority = ActionPriority.P2_HIGH
    owner_role: str = "Customer Success / Retention"
    requires_human_approval: bool = True
    evidence: List[EvidenceLedgerItem] = Field(default_factory=list)


class CallInsightReport(BaseModel):
    call_id: str
    summary: str
    overall_sentiment: str
    sentiment_score: float
    churn_risk_level: SeverityLevel
    churn_risk_score: Optional[float] = None
    primary_intent: str
    unresolved_issues: List[str] = Field(default_factory=list)
    buying_signals: List[str] = Field(default_factory=list)
    objections: List[str] = Field(default_factory=list)
    findings: List[AgentFinding] = Field(default_factory=list)
    evidence_ledger: List[EvidenceLedgerItem] = Field(default_factory=list)
    recommendations: List[ActionRecommendation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
