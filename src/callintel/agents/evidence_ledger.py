"""
Auditable Evidence Ledger Engine.
Extracts and validates timestamped evidence items from call transcripts and ML predictions.
Ensures zero-hallucination grounding for all downstream business decisions.
"""

from typing import List, Dict, Any, Optional
import re
from src.callintel.schemas.insight import EvidenceLedgerItem


class EvidenceLedgerEngine:
    """Builds and queries the Evidence Ledger connecting insights to exact audio timestamps."""

    # Keywords and phrases that indicate critical business events
    OBJECTION_PATTERNS = [
        r"too expensive",
        r"price increase",
        r"pricing",
        r"cost is too high",
        r"cannot afford",
        r"cheaper",
        r"budget",
    ]
    CANCELLATION_PATTERNS = [
        r"cancel",
        r"stop my subscription",
        r"terminate",
        r"close my account",
        r"switch to",
        r"leave",
    ]
    COMMITMENT_PATTERNS = [
        r"discount",
        r"will apply",
        r"fifteen percent",
        r"ten percent",
        r"refund",
        r"loyalty",
        r"credit",
    ]

    def build_ledger(
        self,
        turns: List[Dict[str, Any]],
        turn_emotions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[EvidenceLedgerItem]:
        """
        Inspects all turns and extracts timestamped evidence items with supporting signals.
        """
        ledger = []
        turn_emotions = turn_emotions or []

        for i, turn in enumerate(turns):
            utterance = turn.get("utterance", "")
            speaker = turn.get("speaker", "Unknown")
            start_time = turn.get("start_time", "00:00:00.000")
            end_time = turn.get("end_time", "00:00:00.000")
            start_sec = turn.get("start_seconds")
            end_sec = turn.get("end_seconds")

            lower = utterance.lower()
            detected_signals = {}

            # 1. Objection detection
            if any(re.search(p, lower) for p in self.OBJECTION_PATTERNS):
                detected_signals["is_pricing_objection"] = True

            # 2. Cancellation detection
            if any(re.search(p, lower) for p in self.CANCELLATION_PATTERNS):
                detected_signals["is_cancellation_warning"] = True

            # 3. Commitment detection
            if any(re.search(p, lower) for p in self.COMMITMENT_PATTERNS):
                detected_signals["is_agent_commitment"] = True

            # 4. Integrate emotion signals if available
            if i < len(turn_emotions):
                emo_info = turn_emotions[i]
                detected_signals["emotion"] = emo_info.get("emotion", {}).get("label")
                detected_signals["sentiment"] = emo_info.get("sentiment", {}).get("label")

            # Only add to ledger if it contains a meaningful business signal
            if detected_signals:
                item = EvidenceLedgerItem(
                    turn_id=turn.get("turn_index", i),
                    speaker=speaker,
                    start_time=start_time,
                    end_time=end_time,
                    start_seconds=start_sec,
                    end_seconds=end_sec,
                    verbatim_quote=utterance,
                    supporting_signals=detected_signals,
                )
                ledger.append(item)

        return ledger

    def filter_by_signal(
        self, ledger: List[EvidenceLedgerItem], signal_key: str
    ) -> List[EvidenceLedgerItem]:
        """Returns evidence items matching a specific signal key."""
        return [
            item for item in ledger if item.supporting_signals.get(signal_key) is True
        ]
