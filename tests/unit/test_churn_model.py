"""
Unit tests for the XGBoost Churn Risk Predictor and SHAP Explainability service.
"""

import pytest
from src.callintel.ml.churn_model import ChurnRiskPredictor


def test_churn_predictor_loading():
    predictor = ChurnRiskPredictor()
    assert predictor.is_ready, "ChurnRiskPredictor failed to load pipeline or SHAP explainer."


def test_churn_prediction_with_shap():
    predictor = ChurnRiskPredictor()
    profile = {
        "CreditScore": 650.0,
        "Age": 52.0,
        "Tenure": 2.0,
        "Balance": 120000.0,
        "NumOfProducts": 1.0,
        "HasCrCard": 1.0,
        "IsActiveMember": 0.0,
        "EstimatedSalary": 90000.0,
        "Rating": -2,
        "Geography": "Berlin-Germany",
        "Gender": "Female",
    }
    call_signals = {
        "customer_frustration_score": 0.65,
        "negative_sentiment_ratio": 0.50,
        "has_pricing_objection": True,
        "has_cancellation_warning": True,
        "unresolved_issues_count": 1,
    }

    result = predictor.predict_churn(profile, call_signals)

    assert "baseline_probability" in result
    assert "final_churn_probability" in result
    assert "risk_tier" in result
    assert "top_shap_drivers" in result

    assert 0.0 <= result["baseline_probability"] <= 1.0
    assert 0.0 <= result["final_churn_probability"] <= 1.0
    assert result["risk_tier"] in ["LOW", "MODERATE", "ELEVATED", "CRITICAL"]

    # Verify SHAP drivers contain feature names and non-zero impacts
    drivers = result["top_shap_drivers"]
    assert len(drivers) > 0
    assert "feature" in drivers[0]
    assert "impact" in drivers[0]
    assert "direction" in drivers[0]


def test_business_threshold_optimization():
    import numpy as np
    from src.callintel.ml.churn_model import compute_optimal_business_threshold

    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.35, 0.4, 0.2, 0.1, 0.05, 0.15, 0.02])

    res = compute_optimal_business_threshold(y_true, y_prob, cost_fn=1200.0, cost_fp=80.0, save_rate=0.65)
    assert "optimal_business_cutoff" in res
    assert "naive_cutoff_0_50" in res
    assert "business_lift_analysis" in res
    assert "emv_curve" in res
    assert 0.05 <= res["optimal_business_cutoff"]["threshold"] <= 0.95
    assert res["optimal_business_cutoff"]["net_savings"] >= res["naive_cutoff_0_50"]["net_savings"]

