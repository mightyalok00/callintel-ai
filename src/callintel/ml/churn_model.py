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
            if self.pipeline is None or self.explainer is None:
                raise RuntimeError("Churn model artifacts not found. Run training/train_churn.py first.")

        assert self.pipeline is not None and self.explainer is not None

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
        top_drivers.sort(key=lambda x: abs(float(x["impact"])), reverse=True)

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
            "account_baseline_churn_probability": round(base_prob, 4),
            "account_risk_tier": base_tier,
            "baseline_probability": round(base_prob, 4),
            "baseline_churn_probability": round(base_prob, 4),
            "baseline_risk_tier": base_tier,
            "statistical_validation": {
                "algorithm": "XGBoost Classifier + TreeSHAP",
                "training_domain": "Tabular Banking Account Demographics (Non-Audio)",
                "dataset": "Bank Customer Churn (165,031 records)",
                "test_roc_auc": 0.8808,
                "test_pr_auc": 0.7109,
                "is_calibrated": True,
                "disclaimer": "This baseline probability is strictly an account-level propensity model trained on tabular banking profile data, NOT trained on call speech or transcripts.",
            },
            "top_shap_drivers": top_drivers[:5],
            "in_call_operational_modifier": {
                "multiplier": round(multiplier, 3),
                "composite_risk_score": operational_composite_risk,
                "applied_signals": applied_modifiers,
                "disclaimer": "Operational modifier is a contact-center escalation heuristic combining real-time frustration and dialogue objections; it is not a call-trained machine learning model.",
            },
            # For backward compatibility with report schemas
            "final_churn_probability": operational_composite_risk,
            "risk_tier": "CRITICAL" if operational_composite_risk >= 0.75 else ("ELEVATED" if operational_composite_risk >= 0.50 else ("MODERATE" if operational_composite_risk >= 0.25 else "LOW")),
            "call_multiplier": round(multiplier, 3),
        }

    @staticmethod
    def optimize_business_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        cost_fn: float = 1200.0,
        cost_fp: float = 80.0,
        save_rate: float = 0.65,
        n_thresholds: int = 91,
    ) -> Dict[str, Any]:
        """Exposes static method to compute optimal business threshold."""
        return compute_optimal_business_threshold(
            y_true=y_true,
            y_prob=y_prob,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            save_rate=save_rate,
            n_thresholds=n_thresholds,
        )


def compute_optimal_business_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 1200.0,
    cost_fp: float = 80.0,
    save_rate: float = 0.65,
    n_thresholds: int = 91,
) -> Dict[str, Any]:
    """
    Computes Cost-Utility & Expected Monetary Value (EMV) curve across decision cutoffs.

    Decision Economics:
      - Cost of False Negative (FN): Losing an at-risk customer without proactive outreach = $LTV ($1,200).
      - Cost of False Positive (FP): Outbound incentive concession to a non-churner = $Offer ($80).
      - True Positive (TP): Customer flagged & outreach deployed. Successfully retained with probability
        save_rate (0.65). Net benefit = save_rate * cost_fn - cost_fp = 0.65 * $1200 - $80 = $700.
      - False Negative (FN): At-risk customer unflagged, churns = -$1,200.
      - False Positive (FP): Offer concession wasted = -$80.
      - True Negative (TN): Retained naturally with zero cost = $0.

    Returns full threshold curve, optimal threshold t*, and business lift vs naive 0.50 cutoff.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    total_samples = len(y_true_arr)
    if total_samples == 0:
        raise ValueError("Cannot compute threshold optimization on empty sample arrays.")

    total_positives = int(np.sum(y_true_arr == 1))
    total_negatives = int(np.sum(y_true_arr == 0))
    do_nothing_loss = total_positives * cost_fn

    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    curve_points = []
    best_emv = -float("inf")
    best_threshold = 0.50
    best_metrics: Dict[str, Any] = {}

    naive_metrics: Dict[str, Any] = {}

    for t in thresholds:
        pred_pos = y_prob_arr >= t
        tp = int(np.sum(pred_pos & (y_true_arr == 1)))
        fp = int(np.sum(pred_pos & (y_true_arr == 0)))
        fn = int(np.sum((~pred_pos) & (y_true_arr == 1)))
        tn = int(np.sum((~pred_pos) & (y_true_arr == 0)))

        # Intervention cost model
        intervention_cost = (tp + fp) * cost_fp
        unprevented_churn_loss = fn * cost_fn + tp * (1.0 - save_rate) * cost_fn
        total_cost = intervention_cost + unprevented_churn_loss

        # Net monetary savings relative to zero-action baseline
        net_savings = do_nothing_loss - total_cost
        savings_per_1000 = (net_savings / total_samples) * 1000.0

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        point = {
            "threshold": round(float(t), 3),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "total_cost": round(float(total_cost), 2),
            "net_savings": round(float(net_savings), 2),
            "savings_per_1000": round(float(savings_per_1000), 2),
        }
        curve_points.append(point)

        if net_savings > best_emv:
            best_emv = net_savings
            best_threshold = float(t)
            best_metrics = point

        # Track naive 0.50
        if abs(t - 0.50) < 0.006:
            naive_metrics = point

    if not naive_metrics and curve_points:
        naive_metrics = curve_points[len(curve_points) // 2]

    lift_over_naive_total = best_metrics["net_savings"] - naive_metrics["net_savings"]
    lift_per_1000 = (lift_over_naive_total / total_samples) * 1000.0

    return {
        "financial_parameters": {
            "cost_fn_ltv": cost_fn,
            "cost_fp_retention_offer": cost_fp,
            "save_rate_probability": save_rate,
            "total_evaluated_accounts": total_samples,
            "churn_prevalence": round(float(total_positives / total_samples), 4),
            "do_nothing_baseline_loss": round(float(do_nothing_loss), 2),
        },
        "optimal_business_cutoff": {
            "threshold": round(best_threshold, 3),
            "net_savings": best_metrics["net_savings"],
            "savings_per_1000_accounts": best_metrics["savings_per_1000"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1_score": best_metrics["f1"],
            "true_positives": best_metrics["tp"],
            "false_positives": best_metrics["fp"],
            "false_negatives": best_metrics["fn"],
        },
        "naive_cutoff_0_50": {
            "threshold": naive_metrics["threshold"],
            "net_savings": naive_metrics["net_savings"],
            "savings_per_1000_accounts": naive_metrics["savings_per_1000"],
            "precision": naive_metrics["precision"],
            "recall": naive_metrics["recall"],
            "f1_score": naive_metrics["f1"],
            "true_positives": naive_metrics["tp"],
            "false_positives": naive_metrics["fp"],
            "false_negatives": naive_metrics["fn"],
        },
        "business_lift_analysis": {
            "net_monetary_lift_total": round(lift_over_naive_total, 2),
            "net_monetary_lift_per_1000_accounts": round(lift_per_1000, 2),
            "recall_gain_percentage": round((best_metrics["recall"] - naive_metrics["recall"]) * 100.0, 2),
            "recommendation": (
                f"Deploy operational threshold at t*={best_threshold:.2f} rather than naive 0.50. "
                f"Yields an extra ${lift_per_1000:,.2f} in net retained LTV per 1,000 customer accounts "
                f"by capturing {best_metrics['recall']*100:.1f}% of churners (vs {naive_metrics['recall']*100:.1f}% at 0.50)."
            ),
        },
        "emv_curve": curve_points,
    }

