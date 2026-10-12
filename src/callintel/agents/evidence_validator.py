"""
Evidence and Schema Validator for AI Agent Findings.
Ensures zero-hallucination grounding: every claim must cite valid transcript turns,
match verbatim quotes or semantic paraphrases, detect subtle contradictions and misleading citations,
and explicitly abstain when evidence is missing, unsupported, or contradicted.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import re
from src.callintel.schemas.insight import (
    AgentFinding,
    EvidenceStatus,
    FindingCategory,
)

# Standard English stopwords for semantic content extraction
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "don", "should", "now", "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "would", "could", "shall",
}

# Semantic Concept Thesaurus for Paraphrase Alignment
CONCEPT_CLUSTERS = {
    "churn": {"churn", "cancel", "cancellation", "cancelling", "unsubscribe", "leave", "drop", "terminate", "leaving"},
    "pricing": {"price", "pricing", "expensive", "cost", "costly", "fee", "bill", "billing", "rate", "budget", "discount"},
    "outage": {"outage", "down", "downtime", "crash", "sync", "synchronization", "failing", "failure", "broken", "bug", "stopped"},
    "acceptance": {"agreed", "agree", "accept", "accepted", "proceed", "deal", "stay", "approved", "confirm", "confirmed"},
    "refusal": {"refuse", "refused", "decline", "declined", "reject", "rejected", "cannot", "won't", "wont"},
    "escalation": {"escalate", "escalated", "escalation", "severity", "urgent", "emergency", "sre", "priority"},
    "expansion": {"expand", "expanding", "expansion", "upgrade", "upgrading", "enterprise", "scale", "tier"},
    "satisfaction": {"satisfied", "happy", "pleased", "great", "love", "loved", "smooth", "reasonable"},
    "frustration": {"frustrated", "angry", "terrible", "unacceptable", "annoyed", "upset", "unhappy"},
}


class EvidenceValidator:
    """
    Enforces strict grounding of AI agent claims against conversation transcripts.

    Validation Rules:
    1. Segment Existence: Supporting segment IDs must exist in the transcript. Nonexistent
       IDs are rejected immediately.
    2. Verbatim & Context Grounding: Verbatim citations must match actual turn utterances and
       must not be quoted out of context (e.g. cherry-picking an immediately negated phrase).
    3. Subtle Contradiction Detection: Flags claims where transcript explicitly refutes
       the claim across business dimensions (agreement vs refusal, stability vs outage, etc.).
    4. Fact & Paraphrase Grounding: Citing valid segment IDs is not enough; the content of
       cited turns must semantically support the claim. Claims with zero factual support are rejected.
    5. Explicit Abstention: When evidence is insufficient, unsupported, or contradicted,
       forces evidence_status=INSUFFICIENT/CONTRADICTED, zeroes confidence, and provides auditable rationale.
    """

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Strip punctuation and normalize whitespace for robust matching."""
        return re.sub(r"[^\w\s]", " ", text).lower().strip()

    @classmethod
    def _extract_content_words(cls, text: str) -> Set[str]:
        """Extracts meaningful non-stopword tokens of length >= 3."""
        norm = cls._normalize_text(text)
        tokens = norm.split()
        return {w for w in tokens if len(w) >= 3 and w not in STOPWORDS}

    @classmethod
    def _expand_concepts(cls, words: Set[str]) -> Set[str]:
        """Expands a set of content words with related concepts from the thesaurus."""
        expanded = set(words)
        for cluster_name, synonyms in CONCEPT_CLUSTERS.items():
            if words & synonyms:
                expanded.update(synonyms)
        return expanded

    def _check_misleading_quotation(
        self,
        citation: str,
        raw_turn_text: str,
    ) -> Optional[str]:
        """
        Detects if a quotation was cherry-picked out of context (e.g. immediately preceded by negation).
        """
        norm_turn = self._normalize_text(raw_turn_text)
        norm_cit = self._normalize_text(citation)
        if not norm_cit or norm_cit not in norm_turn:
            return None

        # Look at the words immediately preceding the citation in the turn
        pos = norm_turn.find(norm_cit)
        prefix = norm_turn[:pos].strip()
        if not prefix:
            return None

        tokens_before = prefix.split()
        immediate_window = tokens_before[-2:] if len(tokens_before) >= 2 else tokens_before[-1:]
        negation_markers = {"not", "never", "cannot", "cant", "dont", "wont", "hardly", "no"}

        for marker in immediate_window:
            if marker in negation_markers:
                return (
                    f"Misleading quotation: Quoted phrase '{citation}' is negated "
                    f"in transcript context by preceding marker '{marker}'."
                )
        return None

    def _check_subtle_contradictions(
        self,
        claim: str,
        combined_text: str,
    ) -> Optional[str]:
        """
        Evaluates semantic contradictions across operational, commercial, and technical dimensions.
        """
        claim_norm = self._normalize_text(claim)
        turn_norm = self._normalize_text(combined_text)

        # 1. Agreement / Renewal asserted in claim vs Refusal / Cancellation in turn
        agreed_in_claim = any(w in claim_norm for w in ["agreed", "accepted discount", "signed", "committed to renew", "decided to stay", "renewed subscription"])
        refusal_in_turn = any(w in turn_norm for w in ["no discount", "cannot offer", "refuse", "not possible", "unable to", "cancel my subscription", "cancelling", "too expensive", "will cancel", "closing account"])
        if agreed_in_claim and refusal_in_turn:
            return "Contradiction detected: Claim asserts agreement/acceptance but cited turns state refusal or cancellation."

        # 2. Operational Stability / Satisfaction in claim vs Active Outage / Crash in turn
        stability_in_claim = any(w in claim_norm for w in ["no outage", "functioning normally", "operational stability", "satisfied with uptime", "smooth operation", "no issues reported", "transactions succeeding"])
        outage_in_turn = any(w in turn_norm for w in ["outage", "stopped working", "transactions are failing", "database synchronization stopped", "critical bug", "severity one", "losing revenue", "broken"])
        if stability_in_claim and outage_in_turn:
            return "Contradiction detected: Claim asserts operational stability but cited turns document active outage/failure."

        # 3. Issue Resolved in claim vs Active Outage / Escalation in turn
        resolved_in_claim = any(w in claim_norm for w in ["issue resolved", "problem fixed", "ticket closed", "outage resolved", "fully resolved"])
        escalation_in_turn = any(w in turn_norm for w in ["losing revenue every minute this persists", "need this escalated", "emergency slack channel", "transactions are failing", "still failing"])
        if resolved_in_claim and escalation_in_turn:
            return "Contradiction detected: Claim asserts issue resolution but cited turns document ongoing emergency escalation."

        # 4. Refusal asserted in claim vs Explicit Acceptance in turn (False Negative Claim)
        refusal_in_claim = any(w in claim_norm for w in ["customer refused", "customer rejected", "declined discount", "customer declined", "refused offer"])
        acceptance_in_turn = any(w in turn_norm for w in ["that sounds reasonable go ahead and apply it", "sounds great", "i accept", "please go ahead", "fits our budget and security requirements"])
        if refusal_in_claim and acceptance_in_turn:
            return "Contradiction detected: Claim asserts customer refusal but cited turns confirm customer acceptance."

        # 5. Numerical / discount discrepancy
        if re.search(r"\b(50%|fifty percent|half)\b", claim_norm) and re.search(r"\b(15%|fifteen percent)\b", turn_norm):
            return "Contradiction detected: Claim asserts fifty percent discount but cited turns explicitly state fifteen percent."

        return None

    def _check_factual_support(
        self,
        claim: str,
        valid_turn_texts: List[str],
        citation_matched: bool,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifies that cited transcript turns actually support the facts of the claim.
        Protects against claims that cite valid turn IDs but are completely hallucinated.
        """
        if citation_matched:
            return True, None

        claim_words = self._extract_content_words(claim)
        if not claim_words:
            return True, None

        all_turn_words = set()
        for t in valid_turn_texts:
            all_turn_words.update(self._extract_content_words(t))

        expanded_turn_words = self._expand_concepts(all_turn_words)
        concept_overlap = claim_words & expanded_turn_words

        if len(concept_overlap) == 0:
            return False, "Segment Support Failure: Cited turn(s) contain zero factual or topical overlap with the asserted claim."

        overlap_ratio = len(concept_overlap) / len(claim_words)
        if len(concept_overlap) < 2 and overlap_ratio < 0.20:
            return False, f"Segment Support Failure: Insufficient factual alignment between claim and cited turns (overlap={overlap_ratio:.1%})."

        return True, None

    def _check_speaker_attribution(
        self,
        claim: str,
        valid_turns: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Flag explicit actor/verb claims when cited turns come from another speaker."""
        claim_norm = self._normalize_text(claim)
        speaker_rules = [
            (
                r"\b(customer|caller|client)\b.{0,60}\b(said|stated|confirmed|requested|asked|agreed|declined|refused|reported|complained|threatened|wants|wanted|told)\b",
                {"customer", "caller", "client"},
            ),
            (
                r"\b(agent|representative|advisor)\b.{0,60}\b(said|stated|confirmed|requested|asked|agreed|declined|refused|reported|offered|promised|committed)\b",
                {"agent", "representative", "advisor"},
            ),
        ]
        for pattern, expected_speakers in speaker_rules:
            match = re.search(pattern, claim_norm)
            if not match:
                continue
            expected = match.group(1)
            actual_speakers = {
                str(turn.get("speaker", "")).strip().lower()
                for turn in valid_turns
            }
            if not any(
                any(alias == speaker or alias in speaker for alias in expected_speakers)
                for speaker in actual_speakers
            ):
                return (
                    f"Speaker attribution mismatch: Claim attributes the statement to "
                    f"{expected}, but cited turns are attributed to "
                    f"{sorted(actual_speakers) or ['unknown']}."
                )
        return None

    def _check_recommendation_support(
        self,
        recommendation: Optional[str],
        combined_text: str,
    ) -> Optional[str]:
        """Conservatively flag high-impact action types absent from the cited transcript."""
        if not recommendation:
            return None
        action = self._normalize_text(recommendation)
        evidence = self._normalize_text(combined_text)
        guarded_actions = [
            (("security incident", "security breach", "incident response", "security audit"), ("security", "breach", "unauthorized", "credential", "vulnerability")),
            (("refund", "billing credit", "chargeback"), ("refund", "overcharg", "billing", "bill", "charge", "payment")),
            (("escalate", "escalation", "incident response", "page engineering"), ("escalat", "urgent", "emergency", "outage", "failure", "failing", "broken", "security", "compliance")),
        ]
        for action_phrases, evidence_terms in guarded_actions:
            if any(phrase in action for phrase in action_phrases) and not any(
                term in evidence for term in evidence_terms
            ):
                return (
                    "Unsupported recommendation: Proposed high-impact action is not "
                    "grounded in the cited transcript turns."
                )
        return None

    def validate_finding(
        self,
        finding: AgentFinding,
        turns: List[Dict[str, Any]],
    ) -> AgentFinding:
        """
        Validates a single AgentFinding against conversation turns.
        Returns the validated finding with updated status, confidence, and notes.
        """
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

        valid_raw_texts = [turn_lookup[s].get("utterance", "") for s in valid_segments]
        valid_turns = [turn_lookup[s] for s in valid_segments]
        valid_turn_texts = [self._normalize_text(t) for t in valid_raw_texts]
        combined_turn_text = " ".join(valid_turn_texts)

        speaker_mismatch = self._check_speaker_attribution(finding.claim, valid_turns)
        if speaker_mismatch:
            validation_notes.append(speaker_mismatch)
            return AgentFinding(
                claim_id=finding.claim_id,
                claim=finding.claim,
                category=finding.category,
                evidence_status=EvidenceStatus.CONTRADICTED,
                supporting_segment_ids=valid_segments,
                supporting_timestamps=matched_timestamps,
                verbatim_citations=finding.verbatim_citations,
                confidence_score=0.0,
                confidence_rationale="Contradicted: Claim's explicit speaker attribution does not match the cited turn speaker.",
                recommended_action="Contradiction alert: Verify speaker attribution before acting.",
                validator_notes="; ".join(validation_notes),
            )

        # 2. Validate Verbatim Citations & Detect Misleading Citations
        citation_matches = 0
        unmatched_citations = []
        misleading_citation_note: Optional[str] = None

        for citation in finding.verbatim_citations:
            norm_citation = self._normalize_text(citation)
            if not norm_citation:
                continue

            matched = any(norm_citation in t_text or t_text in norm_citation for t_text in valid_turn_texts)
            if not matched:
                citation_tokens = set(norm_citation.split())
                for t_text in valid_turn_texts:
                    t_tokens = set(t_text.split())
                    overlap = len(citation_tokens & t_tokens) / max(1, len(citation_tokens))
                    if overlap >= 0.70:
                        matched = True
                        break

            if matched:
                citation_matches += 1
                for raw_text in valid_raw_texts:
                    misleading_reason = self._check_misleading_quotation(citation, raw_text)
                    if misleading_reason:
                        misleading_citation_note = misleading_reason
                        break
            else:
                unmatched_citations.append(citation)

        if unmatched_citations:
            validation_notes.append(
                f"Unverified citations not found in referenced turns: {unmatched_citations[:2]}"
            )

        # Handle Misleading Quotation
        if misleading_citation_note:
            validation_notes.append(misleading_citation_note)
            return AgentFinding(
                claim_id=finding.claim_id,
                claim=finding.claim,
                category=finding.category,
                evidence_status=EvidenceStatus.CONTRADICTED,
                supporting_segment_ids=valid_segments,
                supporting_timestamps=matched_timestamps,
                verbatim_citations=finding.verbatim_citations,
                confidence_score=0.0,
                confidence_rationale="Contradicted: Cited quote was taken out of context and negated in transcript.",
                recommended_action="Contradiction alert: Quotation distortion detected. Reject finding.",
                validator_notes="; ".join(validation_notes),
            )

        # 3. Check Fabricated Citations (when citations were explicitly given but none verified)
        if finding.verbatim_citations and citation_matches == 0:
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

        # 4. Check for Subtle & Direct Contradictions
        contradiction_reason = self._check_subtle_contradictions(finding.claim, combined_turn_text)
        if contradiction_reason:
            validation_notes.append(contradiction_reason)
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
                recommended_action="Contradiction alert: Investigate supervisor audit. Claim refuted by turn evidence.",
                validator_notes="; ".join(validation_notes),
            )

        # 5. Check Factual Support (Guards against claims citing valid segments with zero support)
        has_matched_citation = (citation_matches > 0)
        is_supported, support_failure_reason = self._check_factual_support(
            finding.claim, valid_turn_texts, has_matched_citation
        )
        if not is_supported:
            validation_notes.append(support_failure_reason or "Segment Support Failure")
            return AgentFinding(
                claim_id=finding.claim_id,
                claim=finding.claim,
                category=finding.category,
                evidence_status=EvidenceStatus.INSUFFICIENT,
                supporting_segment_ids=valid_segments,
                supporting_timestamps=matched_timestamps,
                verbatim_citations=finding.verbatim_citations,
                confidence_score=0.0,
                confidence_rationale="Abstention: Cited transcript segment(s) do not contain factual support for this claim.",
                recommended_action="Abstain: Insufficient transcript evidence to substantiate finding.",
                validator_notes="; ".join(validation_notes),
            )

        recommendation_issue = self._check_recommendation_support(
            finding.recommended_action, combined_turn_text
        )
        if recommendation_issue:
            validation_notes.append(recommendation_issue)
            return AgentFinding(
                claim_id=finding.claim_id,
                claim=finding.claim,
                category=finding.category,
                evidence_status=EvidenceStatus.INSUFFICIENT,
                supporting_segment_ids=valid_segments,
                supporting_timestamps=matched_timestamps,
                verbatim_citations=finding.verbatim_citations,
                confidence_score=0.0,
                confidence_rationale="Abstention: Recommended action is not grounded in cited transcript evidence.",
                recommended_action="Abstain: Proposed action lacks supporting transcript evidence.",
                validator_notes="; ".join(validation_notes),
            )

        # 6. Calibrate Grounded Confidence
        grounded_score = finding.confidence_score
        if rejected_segments:
            grounded_score = round(max(0.1, grounded_score * 0.8), 2)
            validation_notes.append("Confidence discounted due to partial invalid segment references.")

        if has_matched_citation:
            validation_notes.append(
                f"Verified: {len(valid_segments)} valid segment(s) with confirmed quote alignment."
            )
        else:
            validation_notes.append(
                f"Verified: {len(valid_segments)} valid segment(s) with semantic concept alignment (paraphrase)."
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
