"""
Inference engine for Emotion and Sentiment classification.
Loads trained models from models/emotion_pipeline.joblib and models/sentiment_pipeline.joblib.
Provides single-turn and multi-turn batch predictions with calibrated probabilities.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib

MODELS_DIR = Path("models")
EMOTION_MODEL_PATH = MODELS_DIR / "emotion_pipeline.joblib"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"


class MeldClassifier:
    """Inference service for conversational emotion and sentiment."""

    def __init__(self, models_dir: Optional[Path] = None):
        base_dir = models_dir or MODELS_DIR
        self.emotion_path = base_dir / "emotion_pipeline.joblib"
        self.sentiment_path = base_dir / "sentiment_pipeline.joblib"
        
        self.emotion_model = None
        self.sentiment_model = None
        self._load_models()

    def _load_models(self):
        if self.emotion_path.exists():
            self.emotion_model = joblib.load(self.emotion_path)
        if self.sentiment_path.exists():
            self.sentiment_model = joblib.load(self.sentiment_path)

    @property
    def is_ready(self) -> bool:
        return self.emotion_model is not None and self.sentiment_model is not None

    def predict_turn(
        self,
        utterance: str,
        speaker: str = "Customer",
        prev_speaker: Optional[str] = None,
        prev_utterance: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Classify a single conversational turn with context.
        Returns:
            {
                "emotion": {"label": str, "confidence": float, "all_probs": dict},
                "sentiment": {"label": str, "confidence": float, "all_probs": dict}
            }
        """
        if not self.is_ready:
            self._load_models()
            if not self.is_ready:
                raise RuntimeError("Models not loaded. Please run train_emotion.py first.")

        # Build contextual representation
        if prev_speaker and prev_utterance:
            enriched_text = f"[{prev_speaker}: {prev_utterance[:60]}] {speaker}: {utterance}"
        else:
            enriched_text = f"{speaker}: {utterance}"

        # Emotion prediction
        emo_classes = list(self.emotion_model.classes_)
        emo_probs = self.emotion_model.predict_proba([enriched_text])[0]
        top_emo_idx = emo_probs.argmax()
        top_emo = emo_classes[top_emo_idx]
        top_emo_conf = float(emo_probs[top_emo_idx])

        # Sentiment prediction
        sent_classes = list(self.sentiment_model.classes_)
        sent_probs = self.sentiment_model.predict_proba([enriched_text])[0]
        top_sent_idx = sent_probs.argmax()
        top_sent = sent_classes[top_sent_idx]
        top_sent_conf = float(sent_probs[top_sent_idx])

        return {
            "text": utterance,
            "speaker": speaker,
            "emotion": {
                "label": top_emo,
                "confidence": round(top_emo_conf, 4),
                "probabilities": {cls: round(float(prob), 4) for cls, prob in zip(emo_classes, emo_probs)},
            },
            "sentiment": {
                "label": top_sent,
                "confidence": round(top_sent_conf, 4),
                "probabilities": {cls: round(float(prob), 4) for cls, prob in zip(sent_classes, sent_probs)},
            },
        }

    def analyze_call_timeline(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze a full chronological list of call turns.
        Each turn dict must have 'utterance' and 'speaker' (and optionally timestamps).
        """
        timeline = []
        for i, turn in enumerate(turns):
            prev_speaker = turns[i - 1].get("speaker") if i > 0 else None
            prev_text = turns[i - 1].get("utterance") if i > 0 else None

            res = self.predict_turn(
                utterance=turn.get("utterance", ""),
                speaker=turn.get("speaker", "Customer"),
                prev_speaker=prev_speaker,
                prev_utterance=prev_text,
            )
            # Retain original turn metadata (e.g. timestamps)
            res["turn_index"] = turn.get("turn_index", i)
            res["start_time"] = turn.get("start_time", "")
            res["end_time"] = turn.get("end_time", "")
            timeline.append(res)

        return timeline
