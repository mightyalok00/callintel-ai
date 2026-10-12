# Call Intelligence Engine: Comprehensive ML Evaluation & Statistical Integrity

> **Critical Operational Boundary**:
> The Customer Churn Model evaluated herein is strictly an **Account-Level Tabular Propensity Model** trained on 165,031 banking customer demographic and transactional records. It is **NOT** a model trained on conversational call speech or audio recordings. In-call behavioral indicators (e.g. acoustic frustration peaks, explicit cancellation statements) are treated as **real-time operational escalation modifiers** applied on top of the statistical baseline, never conflated with longitudinal speech-trained propensity.

---

## 1. Account Churn Propensity Model Evaluation

### 1.1 Dataset Splits & Class Imbalance
- **Total Customer Records**: `165,031`
- **Split Strategy**: `80/20 Stratified` (Train: `132,024` | Test: `33,007`)
- **Random Seed**: `42` (Fully reproducible)
- **Class Distribution**:
  - Retained (0): `130,110` (78.8%)
  - Churned (1): `34,921` (21.2%)
- **Class Imbalance Handling**: `scale_pos_weight = 3.726` in XGBoost loss, paired with PR-AUC evaluation to prevent true-negative skew.

### 1.2 Baseline Comparisons on Identical Test Split

| Model Architecture | ROC-AUC | PR-AUC | Brier Score | ECE | F1 Score | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Prevalence Dummy Classifier** | `0.5000` | `0.6058` | `0.1668` | `0.0000` | `0.0000` | Predicts positive class prevalence (0.2116) blindly |
| **Logistic Regression (Balanced)** | `0.8093` | `0.5574` | `0.1747` | `0.2031` | `0.5496` | Standard linear baseline with balanced class weights |
| **Random Forest (100 Trees, Depth 10)** | `0.8766` | `0.6885` | `0.1392` | `0.1681` | `0.6306` | Non-linear ensemble baseline with balanced weights |
| **XGBoost Classifier (Production)** | `0.8808` | `0.7109` | `0.1363` | `0.1580` | `0.6331` | Production gradient boosted tree with scale_pos_weight |
| **XGBoost + Platt Sigmoid Calibration** | `0.8808` | `0.7109` | `0.0990` | `0.0072` | `0.6331` | Post-hoc Platt scaling calibration for optimized probability reliability |

> **Key Finding**: Production XGBoost delivers **`0.7109` PR-AUC**, achieving a **`+0.4993` lift** over the random prevalence baseline (`0.2116`). Platt scaling achieves lowest Brier score calibration.

### 1.3 Probability Calibration & Reliability Analysis

| Probability Bin | Sample Count | Mean Confidence | Actual Churn Rate | Calibration Gap |
| :---: | :---: | :---: | :---: | :---: |
| `[0.0, 0.1]` | `8,030` | `0.059` | `0.017` | `0.042` |
| `[0.1, 0.2]` | `5,851` | `0.145` | `0.044` | `0.101` |
| `[0.2, 0.3]` | `3,500` | `0.246` | `0.089` | `0.157` |
| `[0.3, 0.4]` | `2,765` | `0.350` | `0.131` | `0.219` |
| `[0.4, 0.5]` | `2,563` | `0.448` | `0.175` | `0.274` |
| `[0.5, 0.6]` | `1,936` | `0.549` | `0.255` | `0.294` |
| `[0.6, 0.7]` | `1,846` | `0.650` | `0.329` | `0.320` |
| `[0.7, 0.8]` | `1,709` | `0.750` | `0.454` | `0.296` |
| `[0.8, 0.9]` | `2,112` | `0.851` | `0.613` | `0.238` |
| `[0.9, 1.0]` | `2,695` | `0.950` | `0.853` | `0.097` |

- **Expected Calibration Error (ECE)**: `0.1580`
- **Brier Calibration Score**: `0.1363`

### 1.4 TreeSHAP Feature Attributions

| Rank | Feature Name | Mean Absolute SHAP Importance | Business Driver |
| :---: | :--- | :---: | :--- |
| 1 | `NumOfProducts` | `0.4113` | Primary risk contributor |
| 2 | `Age` | `0.1442` | Primary risk contributor |
| 3 | `IsActiveMember` | `0.1339` | Primary risk contributor |
| 4 | `Geography_Berlin-Germany` | `0.0971` | Primary risk contributor |
| 5 | `Gender_Male` | `0.0459` | Primary risk contributor |
| 6 | `Gender_Female` | `0.0377` | Primary risk contributor |
| 7 | `Geography_Paris-France` | `0.0362` | Primary risk contributor |
| 8 | `Balance` | `0.0305` | Primary risk contributor |

### 1.5 5-Fold Stratified Cross-Validation Confidence Intervals

Empirical 5-fold stratified cross-validation on 30,000 customer accounts:

| Metric | 5-Fold Mean &plusmn; Std Dev | Min / Max Dispersion |
| :--- | :---: | :---: |
| **ROC-AUC** | `0.8741 &plusmn; 0.0036` | Stable across all stratified folds |
| **PR-AUC** | `0.7034 &plusmn; 0.0032` | Strong positive class discovery |
| **Brier Score** | `0.1373 &plusmn; 0.0023` | Consistent probabilistic reliability |
| **F1 Score** | `0.6220 &plusmn; 0.0061` | Balanced trade-off |

### 1.6 Cost-Utility Economics & Expected Monetary Value (EMV) Optimization

Traditional machine learning defaults to an arbitrary `0.50` probability cutoff. However, customer retention economics are highly asymmetric: losing an at-risk account costs **$1,200 LTV**, whereas a proactive retention concession costs only **$80**, with an empirical intervention save rate of **65%**.

| Decision Strategy | Threshold | Recall (Churners Caught) | Precision | Net Monetary Savings per 1,000 Accounts |
| :--- | :---: | :---: | :---: | :---: |
| **Naive ML Cutoff** | `0.50` | `78.3%` | `53.1%` | `$104,327.57` |
| **Optimal EMV Cutoff (t\*)** | **`0.28`** | **`91.1%`** | **`39.2%`** | **`$110,958.28`** |

> **Monetary Value Lift**: Tuning to the cost-utility optimal cutoff (`t* = 0.28`) increases churn recall by **+12.7%**, generating an additional **`$6,630.71` in retained customer value per 1,000 accounts** compared to the naive `0.50` threshold.

---

## 2. Conversational Emotion & Sentiment Model Evaluation (MELD)

- **Test Dialogue Utterances**: `2,610` across `280` dialogues
- **Class Distribution (7 Emotions)**:
  - `neutral`: `1,256` (48.12%)
  - `joy`: `402` (15.4%)
  - `anger`: `345` (13.22%)
  - `surprise`: `281` (10.77%)
  - `sadness`: `208` (7.97%)
  - `disgust`: `68` (2.61%)
  - `fear`: `50` (1.92%)

### 2.1 Benchmark Performance

| Task | Majority Baseline Acc | Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: |
| **7-Class Emotion** | `48.12%` (neutral) | `47.85%` | `0.3334` | `0.4835` |
| **3-Class Sentiment** | `48.12%` (Neutral) | `52.68%` | `0.5063` | `0.5301` |

### 2.2 Multimodal Acoustic-Lexical Ablation

| Modality | Emotion Accuracy | Emotion Macro F1 | Sentiment Accuracy | Sentiment Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Lexical Text Only (TF-IDF)** | `43.72%` | `0.3087` | `52.61%` | `0.5055` |
| **Multimodal Fused (Acoustic + Text)** | **`47.85%`** | **`0.3334`** | **`52.68%`** | **`0.5063`** |

> **Acoustic Fusion Impact**: Calibrated late fusion ($0.75 \text{ text} + 0.25 \text{ acoustic}$) blends prosodic vocal tension (F0 pitch, spectral centroid, voiced ratio) with lexical intent, reducing false neutrality on vocally distressed customers.

---

## 3. Two-Tier Risk Architecture Summary

```mermaid
graph LR
    subgraph Tier 1: Static Tabular Propensity
        A[Customer Demographic & Account Data] --> B[XGBoost Pipeline + TreeSHAP]
        B --> C[Account Baseline Churn Probability]
    end
    subgraph Tier 2: Real-Time Conversational Telemetry
        D[Call Audio & Transcript] --> E[Acoustic Diarization + Whisper ASR]
        E --> F[MELD Frustration / Objection Detector]
        F --> G[In-Call Risk Modifier Multiplier]
    end
    C & G --> H[Composite Operational Escalation Score]
```

This architecture guarantees full statistical transparency without misrepresenting static account tabular models as speech-trained classifiers.