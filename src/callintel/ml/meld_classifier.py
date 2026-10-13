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
            if not self.is_ready or self.emotion_model is None or self.sentiment_model is None:
                raise RuntimeError("Models not loaded. Please run train_emotion.py first.")
        if self.emotion_model is None or self.sentiment_model is None:
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

    def predict_turn_multimodal(
        self,
        utterance: str,
        speaker: str = "Customer",
        acoustic_metrics: Optional[Dict[str, float]] = None,
        prev_speaker: Optional[str] = None,
        prev_utterance: Optional[str] = None,
        text_weight: float = 0.75,
        acoustic_weight: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Multimodal acoustic-lexical emotion prediction.
        Combines contextual lexical TF-IDF with prosodic acoustic cues
        (pitch, spectral centroid, voiced ratio, RMS energy) via calibrated late fusion.
        """
        base_res = self.predict_turn(
            utterance=utterance,
            speaker=speaker,
            prev_speaker=prev_speaker,
            prev_utterance=prev_utterance,
        )

        if not acoustic_metrics:
            base_res["fusion_metadata"] = {
                "fusion_applied": False,
                "reason": "No acoustic metrics provided; defaulted to lexical model",
            }
            return base_res

        pitch = float(acoustic_metrics.get("mean_pitch_hz", 0.0))
        centroid = float(acoustic_metrics.get("spectral_centroid_hz", 0.0))
        voiced = float(acoustic_metrics.get("voiced_ratio", 0.0))
        energy = float(acoustic_metrics.get("rms_energy", 0.0))

        # Vocal Arousal Index: maps pitch [100..300Hz] and centroid [1000..3500Hz] to [0..1]
        norm_pitch = min(1.0, max(0.0, (pitch - 100.0) / 200.0)) if pitch > 0 else 0.4
        norm_centroid = min(1.0, max(0.0, (centroid - 1000.0) / 2500.0)) if centroid > 0 else 0.4
        arousal = 0.6 * norm_pitch + 0.4 * norm_centroid

        # Construct prosodic acoustic prior distribution over classes
        emo_probs = base_res["emotion"]["probabilities"].copy()
        emo_classes = list(emo_probs.keys())

        # Acoustic prior mapping
        # High arousal boosts anger, surprise, joy; low arousal / low voiced ratio boosts sadness, neutral
        ac_priors: Dict[str, float] = {}
        for c in emo_classes:
            c_lower = c.lower()
            if c_lower in ["anger", "angry"]:
                ac_priors[c] = 0.10 + 0.45 * arousal
            elif c_lower in ["surprise", "surprised"]:
                ac_priors[c] = 0.05 + 0.25 * arousal
            elif c_lower in ["joy", "happy"]:
                ac_priors[c] = 0.08 + 0.20 * arousal
            elif c_lower in ["sadness", "sad"]:
                ac_priors[c] = 0.10 + 0.35 * (1.0 - arousal) * (1.0 - voiced * 0.5)
            elif c_lower in ["disgust"]:
                ac_priors[c] = 0.05 + 0.15 * arousal
            elif c_lower in ["fear"]:
                ac_priors[c] = 0.05 + 0.20 * arousal
            else:  # neutral
                ac_priors[c] = 0.40 * (1.0 - abs(arousal - 0.45))

        # Normalize acoustic prior
        sum_ac = sum(ac_priors.values()) or 1.0
        ac_priors = {k: v / sum_ac for k, v in ac_priors.items()}

        # Calibrated Late Fusion
        fused_probs: Dict[str, float] = {}
        for c in emo_classes:
            fused_p = text_weight * emo_probs[c] + acoustic_weight * ac_priors.get(c, 0.0)
            fused_probs[c] = float(fused_p)

        # Normalize fused distribution
        total_p = sum(fused_probs.values()) or 1.0
        fused_probs = {k: round(v / total_p, 4) for k, v in fused_probs.items()}

        top_emo = max(fused_probs.keys(), key=lambda k: fused_probs[k])
        top_conf = fused_probs[top_emo]

        # Update emotion section
        base_res["emotion"] = {
            "label": top_emo,
            "confidence": top_conf,
            "probabilities": fused_probs,
        }

        # Align sentiment to fused emotion if sentiment shifts
        sent_probs = base_res["sentiment"]["probabilities"].copy()
        if top_emo in ["anger", "disgust", "sadness", "fear"]:
            sent_probs["negative"] = min(0.98, max(sent_probs.get("negative", 0.0), top_conf * 0.9))
            sum_sent = sum(sent_probs.values()) or 1.0
            sent_probs = {k: round(v / sum_sent, 4) for k, v in sent_probs.items()}
            top_sent = max(sent_probs.keys(), key=lambda k: sent_probs[k])
            base_res["sentiment"] = {
                "label": top_sent,
                "confidence": sent_probs[top_sent],
                "probabilities": sent_probs,
            }

        base_res["fusion_metadata"] = {
            "fusion_applied": True,
            "fusion_method": "calibrated_late_fusion",
            "text_weight": text_weight,
            "acoustic_weight": acoustic_weight,
            "vocal_arousal_score": round(arousal, 4),
            "acoustic_metrics": {
                "mean_pitch_hz": pitch,
                "spectral_centroid_hz": centroid,
                "voiced_ratio": voiced,
            },
        }

        return base_res

    def analyze_call_timeline(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze a full chronological list of call turns.
        Automatically leverages acoustic_metrics when present on turns for multimodal fusion.
        """
        timeline = []
        for i, turn in enumerate(turns):
            prev_speaker = turns[i - 1].get("speaker") if i > 0 else None
            prev_text = turns[i - 1].get("utterance") if i > 0 else None
            acoustic = turn.get("acoustic_metrics")

            if acoustic:
                res = self.predict_turn_multimodal(
                    utterance=turn.get("utterance", ""),
                    speaker=turn.get("speaker", "Customer"),
                    acoustic_metrics=acoustic,
                    prev_speaker=prev_speaker,
                    prev_utterance=prev_text,
                )
            else:
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

