"""
Inference service for Customer Churn Prediction and SHAP Explainability.
Integrates baseline customer profile features with conversational call signals.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path("models")
CHURN_MODEL_PATH = MODELS_DIR / "churn_xgb_pipeline.joblib"
SHAP_EXPLAINER_PATH = MODELS_DIR / "churn_shap_explainer.joblib"


class ChurnRiskPredictor:
    """Predicts customer churn risk and computes SHAP feature attributions."""

    def __init__(self, models_dir: Optional[Path] = None):
        base_dir = models_dir or MODELS_DIR
        self.model_path = base_dir / "churn_xgb_pipeline.joblib"
        self.explainer_path = base_dir / "churn_shap_explainer.joblib"
        
        self.pipeline = None
        self.explainer = None
        self._load_artifacts()

    def _load_artifacts(self):
        if self.model_path.exists():
            self.pipeline = joblib.load(self.model_path)
        if self.explainer_path.exists():
            self.explainer = joblib.load(self.explainer_path)

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None and self.explainer is not None

    def predict_churn(
        self,
        customer_profile: Dict[str, Any],
        call_signals: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates customer baseline churn risk, SHAP feature drivers, and adjusts
        risk dynamically using conversational call signals.

        customer_profile keys:
          CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard,
          IsActiveMember, EstimatedSalary, Rating, Geography, Gender

        call_signals keys:
          customer_frustration_score (0.0 to 1.0)
          negative_sentiment_ratio (0.0 to 1.0)
          has_pricing_objection (bool)
          has_cancellation_warning (bool)
          unresolved_issues_count (int)
        """
        if not self.is_ready:
            self._load_artifacts()
            if not self.is_ready:
                raise RuntimeError("Churn model artifacts not found. Run train_churn.py first.")

        # Default fallback values for customer profile
        defaults = {
            "CreditScore": 650.0,
            "Age": 40.0,
            "Tenure": 3.0,
            "Balance": 50000.0,
            "NumOfProducts": 1.0,
            "HasCrCard": 1.0,
            "IsActiveMember": 0.0,
            "EstimatedSalary": 100000.0,
            "Rating": 0.0,
            "Geography": "Paris-France",
            "Gender": "Male",
        }
        profile_data = {**defaults, **customer_profile}
        df_input = pd.DataFrame([profile_data])

        preprocessor = self.pipeline["preprocessor"]
        model = self.pipeline["model"]
        feature_names = self.pipeline["feature_names"]

        # Transform features
        X_trans = preprocessor.transform(df_input)
        X_df = pd.DataFrame(X_trans, columns=feature_names)

        # Baseline model probability
        base_prob = float(model.predict_proba(X_df)[0, 1])

        # Compute SHAP values for explainability
        shap_values = self.explainer(X_df)
        raw_shaps = shap_values.values[0]

        top_drivers = []
        for feat_name, shap_val, feat_val in zip(feature_names, raw_shaps, X_trans[0]):
            top_drivers.append({
                "feature": feat_name,
                "impact": round(float(shap_val), 4),
                "value": round(float(feat_val), 2),
                "direction": "increases_risk" if shap_val > 0 else "reduces_risk"
            })
        top_drivers.sort(key=lambda x: abs(x["impact"]), reverse=True)

        # Apply Conversational Call Multiplier
        call_signals = call_signals or {}
        frustration = float(call_signals.get("customer_frustration_score", 0.0))
        neg_ratio = float(call_signals.get("negative_sentiment_ratio", 0.0))
        pricing_objection = bool(call_signals.get("has_pricing_objection", False))
        cancellation_warning = bool(call_signals.get("has_cancellation_warning", False))
        unresolved_count = int(call_signals.get("unresolved_issues_count", 0))

        multiplier = 1.0
        applied_modifiers = []

        if cancellation_warning:
            multiplier += 0.45
            applied_modifiers.append({"signal": "cancellation_warning", "effect": "+45% risk multiplier"})
        if pricing_objection:
            multiplier += 0.20
            applied_modifiers.append({"signal": "pricing_objection", "effect": "+20% risk multiplier"})
        if frustration > 0.40:
            boost = round(frustration * 0.35, 3)
            multiplier += boost
            applied_modifiers.append({"signal": "high_frustration", "effect": f"+{boost*100:.1f}% risk multiplier"})
        if unresolved_count > 0:
            boost = round(min(0.30, unresolved_count * 0.15), 3)
            multiplier += boost
            applied_modifiers.append({"signal": "unresolved_issues", "effect": f"+{boost*100:.1f}% risk multiplier"})

        final_churn_prob = min(0.99, round(base_prob * multiplier, 4))

        # Assign Risk Tier
        if final_churn_prob >= 0.75:
            risk_tier = "CRITICAL"
        elif final_churn_prob >= 0.50:
            risk_tier = "ELEVATED"
        elif final_churn_prob >= 0.25:
            risk_tier = "MODERATE"
        else:
            risk_tier = "LOW"

        return {
            "baseline_probability": round(base_prob, 4),
            "final_churn_probability": final_churn_prob,
            "risk_tier": risk_tier,
            "call_multiplier": round(multiplier, 3),
            "applied_call_modifiers": applied_modifiers,
            "top_shap_drivers": top_drivers[:5],
        }
