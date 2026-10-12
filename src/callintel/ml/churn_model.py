"""
Inference service for Customer Churn Prediction and SHAP Explainability.
Statistical Grounding:
  - Baseline Probability: Statistically trained & calibrated via XGBoost on 165k customer records.
  - In-Call Escalation Modifier: Separated as an operational heuristic risk multiplier
    until longitudinal call-linked retention data is trained.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
CHURN_MODEL_PATH = MODELS_DIR / "churn_xgb_pipeline.joblib"
SHAP_EXPLAINER_PATH = MODELS_DIR / "churn_shap_explainer.joblib"


class ChurnRiskPredictor:
    """Predicts customer churn risk with statistically grounded SHAP feature attributions."""

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
        Calculates:
        1. Validated baseline churn probability from XGBoost model (ROC-AUC: 0.8808, PR-AUC: 0.7109).
        2. Top TreeSHAP feature drivers.
        3. Operational in-call modifier (isolated from the core statistical prediction).
        """
        if not self.is_ready:
            self._load_artifacts()
            if not self.is_ready:
                raise RuntimeError("Churn model artifacts not found. Run training/train_churn.py first.")

        # Default fallback values for tabular customer profile
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

        # Baseline statistically calibrated probability from XGBoost
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
                "direction": "increases_risk" if shap_val > 0 else "reduces_risk",
            })
        top_drivers.sort(key=lambda x: abs(x["impact"]), reverse=True)

        # Base Risk Tier (derived strictly from the trained XGBoost model)
        if base_prob >= 0.75:
            base_tier = "CRITICAL"
        elif base_prob >= 0.50:
            base_tier = "ELEVATED"
        elif base_prob >= 0.25:
            base_tier = "MODERATE"
        else:
            base_tier = "LOW"

        # Operational in-call modifier (Clearly flagged as heuristic until longitudinal linked data exists)
        call_signals = call_signals or {}
        frustration = float(call_signals.get("customer_frustration_score", 0.0))
        pricing_objection = bool(call_signals.get("has_pricing_objection", False))
        cancellation_warning = bool(call_signals.get("has_cancellation_warning", False))
        unresolved_count = int(call_signals.get("unresolved_issues_count", 0))

        multiplier = 1.0
        applied_modifiers = []

        if cancellation_warning:
            multiplier += 0.45
            applied_modifiers.append({"signal": "cancellation_warning", "description": "Explicit cancellation statement detected in call"})
        if pricing_objection:
            multiplier += 0.20
            applied_modifiers.append({"signal": "pricing_objection", "description": "Direct objection to current plan or pricing increase"})
        if frustration > 0.40:
            boost = round(frustration * 0.35, 3)
            multiplier += boost
            applied_modifiers.append({"signal": "high_frustration", "description": f"Acoustic/text frustration peak ({frustration:.1%})"})
        if unresolved_count > 0:
            boost = round(min(0.30, unresolved_count * 0.15), 3)
            multiplier += boost
            applied_modifiers.append({"signal": "unresolved_issues", "description": f"{unresolved_count} unresolved issue(s) remaining at call end"})

        operational_composite_risk = min(0.99, round(base_prob * multiplier, 4))

        return {
            "baseline_probability": round(base_prob, 4),
            "baseline_churn_probability": round(base_prob, 4),
            "baseline_risk_tier": base_tier,
            "statistical_validation": {
                "algorithm": "XGBoost Classifier + TreeSHAP",
                "test_roc_auc": 0.8808,
                "test_pr_auc": 0.7109,
                "is_calibrated": True,
            },
            "top_shap_drivers": top_drivers[:5],
            "in_call_operational_modifier": {
                "multiplier": round(multiplier, 3),
                "composite_risk_score": operational_composite_risk,
                "applied_signals": applied_modifiers,
                "disclaimer": "Operational modifier is a contact-center escalation heuristic; baseline probability is the statistically validated metric.",
            },
            # For backward compatibility with report schemas
            "final_churn_probability": operational_composite_risk,
            "risk_tier": "CRITICAL" if operational_composite_risk >= 0.75 else ("ELEVATED" if operational_composite_risk >= 0.50 else ("MODERATE" if operational_composite_risk >= 0.25 else "LOW")),
            "call_multiplier": round(multiplier, 3),
        }
