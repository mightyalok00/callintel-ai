r"""
Production Training Script for Customer Churn Prediction and SHAP Explainability.
Dataset: data/churn/bank_churn_cleaned.csv (165,031 customer records)

Outputs:
  - models/churn_xgb_pipeline.joblib (Pretrained XGBoost model with preprocessor)
  - models/churn_shap_explainer.joblib (TreeSHAP explainer object)
  - reports/churn_benchmark_report.json (PR-AUC, ROC-AUC, classification metrics)
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    classification_report,
    confusion_matrix,
    brier_score_loss,
    f1_score,
    accuracy_score,
)
import xgboost as xgb
import shap

DATA_PATH = Path("data/churn/bank_churn_cleaned.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def train_churn_model():
    print(f"[*] Loading churn dataset from {DATA_PATH}...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"[+] Loaded {len(df):,} customer records with {df.shape[1]} columns.")

    target_col = "Exited"
    features = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Rating",
        "Geography",
        "Gender",
    ]

    X = df[features]
    y = df[target_col].astype(int)

    # Calculate class weight ratio
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = float(neg_count / pos_count)
    print(f"[+] Class distribution: Non-churn={neg_count:,} ({neg_count/len(y):.1%}), Churn={pos_count:,} ({pos_count/len(y):.1%})")
    print(f"[+] scale_pos_weight = {scale_pos_weight:.3f}")

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Train split: {len(X_train):,} samples | Test split: {len(X_test):,} samples")

    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Rating",
    ]
    cat_cols = ["Geography", "Gender"]

    # Preprocessing pipelines
    num_transformer = SimpleImputer(strategy="median")
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    print("[*] Fitting preprocessing transformations...")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_names
    print(f"[+] Total transformed features: {len(all_feature_names)} -> {all_feature_names}")

    # Convert to DataFrame for XGBoost with proper feature names
    X_train_df = pd.DataFrame(X_train_trans, columns=all_feature_names)
    X_test_df = pd.DataFrame(X_test_trans, columns=all_feature_names)

    print("\n" + "="*50)
    print("[*] Training XGBoost Classifier with Early Stopping...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric=["aucpr", "auc"],
        early_stopping_rounds=25,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )

    xgb_model.fit(
        X_train_df,
        y_train,
        eval_set=[(X_test_df, y_test)],
        verbose=50,
    )

    # Predictions & Probabilities
    y_pred = xgb_model.predict(X_test_df)
    y_probs = xgb_model.predict_proba(X_test_df)[:, 1]

    # Metrics
    roc_auc = roc_auc_score(y_test, y_probs)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall_curve, precision_curve)
    brier = brier_score_loss(y_test, y_probs)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*50)
    print("XGBoost Test Performance:")
    print(f"  - ROC-AUC:  {roc_auc:.4f}")
    print(f"  - PR-AUC:   {pr_auc:.4f}  (Baseline = {pos_count/len(y):.3f})")
    print(f"  - Brier:    {brier:.4f}")
    print(f"  - F1 Score: {f1:.4f}")
    print(f"  - Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained (0)", "Churned (1)"]))

    # ==========================================
    # FIT SHAP TREE EXPLAINER
    # ==========================================
    print("\n" + "="*50)
    print("[*] Fitting SHAP TreeExplainer on background sample...")
    background_sample = X_train_df.sample(n=1000, random_state=42)
    explainer = shap.TreeExplainer(xgb_model, data=background_sample)
    print("[+] SHAP TreeExplainer successfully initialized.")

    # Save Pipeline and Explainer
    pipeline_artifact = {
        "preprocessor": preprocessor,
        "feature_names": all_feature_names,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "model": xgb_model,
    }
    model_path = MODELS_DIR / "churn_xgb_pipeline.joblib"
    joblib.dump(pipeline_artifact, model_path)
    print(f"[OK] Saved XGBoost Churn Pipeline to {model_path}")

    explainer_path = MODELS_DIR / "churn_shap_explainer.joblib"
    joblib.dump(explainer, explainer_path)
    print(f"[OK] Saved SHAP Explainer to {explainer_path}")

    # Benchmark report
    report_data = {
        "model_type": "XGBoost Classifier + TreeSHAP",
        "dataset_rows": len(df),
        "test_samples": len(X_test),
        "scale_pos_weight": round(scale_pos_weight, 3),
        "metrics": {
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "brier_score": round(float(brier), 4),
            "f1_score": round(float(f1), 4),
            "accuracy": round(float(acc), 4),
        },
        "feature_importance": {
            name: round(float(imp), 4)
            for name, imp in zip(all_feature_names, xgb_model.feature_importances_)
        },
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    report_path = REPORTS_DIR / "churn_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[OK] Saved Benchmark Report to {report_path}")

    print("\n" + "="*50)
    print("[OK] Churn model training & explainability export complete!")


if __name__ == "__main__":
    train_churn_model()
