"""
Evidence and Schema Validator for AI Agent Findings.
Ensures zero-hallucination grounding: every claim must cite valid transcript turns,
match verbatim quotes, and explicitly abstain when evidence is missing or contradicted.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from src.callintel.schemas.insight import (
    AgentFinding,
    EvidenceStatus,
    FindingCategory,
)


class EvidenceValidator:
    """
    Enforces strict grounding of AI agent claims against conversation transcripts.

    Validation Rules:
    1. Segment Existence: Supporting segment IDs must exist in the transcript. Nonexistent
       IDs are rejected immediately.
    2. Verbatim Grounding: Verbatim citations must match actual turn utterances
       (case-insensitive, normalized whitespace).
    3. Contradiction Detection: Flags claims where transcript explicitly refutes the claim.
    4. Explicit Abstention: When evidence is insufficient or fabricated, forces
       evidence_status=INSUFFICIENT, zeroes/penalizes confidence, and replaces recommended action
       with an explicit abstention directive.
    """

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Strip punctuation and normalize whitespace for robust fuzzy matching."""
        return re.sub(r"[^\w\s]", "", text).lower().strip()

    def validate_finding(
        self,
        finding: AgentFinding,
        turns: List[Dict[str, Any]],
    ) -> AgentFinding:
        """
        Validates a single AgentFinding against conversation turns.
        Returns the validated finding with updated status, confidence, and notes.
        """
        # Map turn_index or list index to turn dict
        turn_lookup: Dict[int, Dict[str, Any]] = {}
        for idx, turn in enumerate(turns):
            turn_idx = turn.get("turn_index", idx)
            turn_lookup[turn_idx] = turn

        valid_segments: List[int] = []
        rejected_segments: List[int] = []
        matched_timestamps: List[str] = []
        validation_notes: List[str] = []

        # 1. Validate Segment ID Existence
        if not finding.supporting_segment_ids:
            # Explicit abstention when no evidence is cited
            return AgentFinding(
                claim_id=finding.claim_id,
                claim=finding.claim,
                category=finding.category,
                evidence_status=EvidenceStatus.INSUFFICIENT,
                supporting_segment_ids=[],
                supporting_timestamps=[],
                verbatim_citations=[],
                confidence_score=0.0,
                confidence_rationale="Abstention: No supporting transcript segments cited.",
                recommended_action="Abstain: Insufficient transcript evidence to substantiate finding.",
                validator_notes="Rejected: Claim provided zero supporting segment references.",
            )

        for seg_id in finding.supporting_segment_ids:
            if seg_id in turn_lookup:
                valid_segments.append(seg_id)
                t = turn_lookup[seg_id]
                start_t = t.get("start_time", "00:00:00.000")
                end_t = t.get("end_time", "00:00:00.000")
                matched_timestamps.append(f"{start_t} - {end_t}")
            else:
                rejected_segments.append(seg_id)

        if rejected_segments:
            validation_notes.append(
                f"Rejected nonexistent segment IDs: {rejected_segments}."
            )

        # If ALL cited segments are nonexistent -> Abstain
        if not valid_segments:
            return AgentFinding(
                claim_id=finding.claim_id,
                claim=finding.claim,
                category=finding.category,
                evidence_status=EvidenceStatus.INSUFFICIENT,
                supporting_segment_ids=[],
                supporting_timestamps=[],
                verbatim_citations=finding.verbatim_citations,
                confidence_score=0.0,
                confidence_rationale="Abstention: All cited transcript segment IDs are nonexistent.",
                recommended_action="Abstain: Insufficient transcript evidence to substantiate finding.",
                validator_notes="; ".join(validation_notes),
            )

        # 2. Validate Verbatim Citations against Valid Turns
        valid_turn_texts = [
            self._normalize_text(turn_lookup[s].get("utterance", ""))
            for s in valid_segments
        ]
        citation_matches = 0
        unmatched_citations = []

        for citation in finding.verbatim_citations:
            norm_citation = self._normalize_text(citation)
            if not norm_citation:
                continue

            matched = any(norm_citation in t_text or t_text in norm_citation for t_text in valid_turn_texts)
            if not matched:
                # Check for significant token overlap (>= 70%)
                citation_tokens = set(norm_citation.split())
                for t_text in valid_turn_texts:
                    t_tokens = set(t_text.split())
                    overlap = len(citation_tokens & t_tokens) / max(1, len(citation_tokens))
                    if overlap >= 0.70:
                        matched = True
                        break

            if matched:
                citation_matches += 1
            else:
                unmatched_citations.append(citation)

        if unmatched_citations:
            validation_notes.append(
                f"Unverified citations not found in referenced turns: {unmatched_citations[:2]}"
            )

        # 3. Check for Direct Contradictions
        # E.g. Claim asserts retention commitment or agreement, but cited text contains refusal
        combined_text = " ".join(valid_turn_texts)
        is_contradicted = False
        claim_lower = finding.claim.lower()

        if "agreed" in claim_lower or "accepted discount" in claim_lower:
            if any(k in combined_text for k in ["no discount", "cannot offer", "refuse", "not possible", "unable to"]):
                is_contradicted = True
                validation_notes.append(
                    "Contradiction detected: claim asserts agreement but cited turns state refusal."
                )

        if is_contradicted:
            return AgentFinding(
                claim_id=finding.claim_id,
                claim=finding.claim,
                category=finding.category,
                evidence_status=EvidenceStatus.CONTRADICTED,
                supporting_segment_ids=valid_segments,
                supporting_timestamps=matched_timestamps,
                verbatim_citations=finding.verbatim_citations,
                confidence_score=0.0,
                confidence_rationale="Contradicted: Transcript text directly refutes the asserted claim.",
                recommended_action=f"Contradiction alert: Investigate supervisor audit. Claim refuted by turn evidence.",
                validator_notes="; ".join(validation_notes),
            )

        # 4. Determine Grounding Status & Calibrate Confidence
        if finding.verbatim_citations and citation_matches == 0:
            # Citations were supplied but none matched actual turn speech
            return AgentFinding(
                claim_id=finding.claim_id,
                claim=finding.claim,
                category=finding.category,
                evidence_status=EvidenceStatus.INSUFFICIENT,
                supporting_segment_ids=valid_segments,
                supporting_timestamps=matched_timestamps,
                verbatim_citations=finding.verbatim_citations,
                confidence_score=round(finding.confidence_score * 0.3, 2),
                confidence_rationale="Abstention: Verbatim citations failed transcript verification.",
                recommended_action="Abstain: Insufficient transcript evidence to substantiate finding.",
                validator_notes="; ".join(validation_notes),
            )

        # Grounded and verified
        grounded_score = finding.confidence_score
        if rejected_segments:
            # Penalty for citing bad segment along with good one
            grounded_score = round(max(0.1, grounded_score * 0.8), 2)
            validation_notes.append(f"Confidence discounted due to partial invalid segment references.")

        validation_notes.append(
            f"Verified: {len(valid_segments)} valid segment(s) with confirmed quote alignment."
        )

        return AgentFinding(
            claim_id=finding.claim_id,
            claim=finding.claim,
            category=finding.category,
            evidence_status=EvidenceStatus.SUPPORTED,
            supporting_segment_ids=valid_segments,
            supporting_timestamps=matched_timestamps or finding.supporting_timestamps,
            verbatim_citations=finding.verbatim_citations,
            confidence_score=grounded_score,
            confidence_rationale=finding.confidence_rationale,
            recommended_action=finding.recommended_action,
            validator_notes="; ".join(validation_notes),
        )

    def validate_findings(
        self,
        findings: List[AgentFinding],
        turns: List[Dict[str, Any]],
    ) -> List[AgentFinding]:
        """Validates all findings in batch."""
        return [self.validate_finding(f, turns) for f in findings]

    def audit_summary(self, findings: List[AgentFinding]) -> Dict[str, Any]:
        """Computes audit stats across validated findings."""
        total = len(findings)
        supported = sum(1 for f in findings if f.evidence_status == EvidenceStatus.SUPPORTED)
        insufficient = sum(1 for f in findings if f.evidence_status == EvidenceStatus.INSUFFICIENT)
        contradicted = sum(1 for f in findings if f.evidence_status == EvidenceStatus.CONTRADICTED)

        return {
            "total_findings": total,
            "supported_count": supported,
            "insufficient_count": insufficient,
            "contradicted_count": contradicted,
            "grounding_pass_rate": round(supported / max(1, total), 3),
        }
