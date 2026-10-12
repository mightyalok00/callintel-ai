# Reproducible Evaluation Report — CallIntel AI

**Evaluation artifact snapshot:** 2026-10-12  
**Repository:** [mightyalok00/callintel-ai](https://github.com/mightyalok00/callintel-ai)  
**Primary machine-readable results:** [`ml_evaluation_benchmark.json`](ml_evaluation_benchmark.json) · [`speech_benchmark_report.json`](speech_benchmark_report.json)  
**Detailed existing reports:** [ML evaluation](ML_EVALUATION.md) · [Speech benchmark](SPEECH_BENCHMARK.md)

> **Scope and evidence note.** This report summarizes the checked-in benchmark artifacts; it does not claim a fresh training run was executed as part of this documentation change. The scores below are reproducible only when the dataset files, dependency environment, model/configuration, and split procedure match the repository snapshot. The call audio fixtures are controlled/synthetic, not a representative sample of production calls.

## Executive summary

| Task | Evaluation data | Primary result | Baseline / interpretation |
|---|---|---|---|
| Account-level bank churn | 33,007-row stratified test split from 165,031 account records | XGBoost ROC-AUC **0.8808**, PR-AUC **0.7109**, F1 **0.6331** | Beats the 0.5 ROC-AUC ranking baseline and the churn prevalence of **0.2116** as a PR reference; see the PR-AUC note below |
| MELD 7-class emotion | 2,610 utterances / 280 dialogues | Accuracy **0.4785**, macro-F1 **0.3334**, weighted-F1 **0.4835** | Accuracy is **below** the neutral-majority baseline of 0.4812; macro-F1 improves the view beyond majority accuracy but minority classes remain weak |
| MELD 3-class sentiment | Same 2,610 utterance test set | Accuracy **0.5268**, macro-F1 **0.5063**, weighted-F1 **0.5301** | Accuracy is 4.56 percentage points above the 0.4812 majority baseline |
| Speech transcription / diarization | 4 controlled sample calls | Mean WER **1.03%**, reported DER **0.00%**, mean RTF **0.090x** | Fixture-only result. Automatic speaker-count discovery was correct on only **2 of 4** calls (50%); do not infer real-call performance |

## 1. Reproduction

Use Python 3.12 and the repository's pinned/compatible dependencies. Run from the repository root.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python training/evaluate_ml_models.py
python training/benchmark_speech.py
```

Expected outputs:

- `reports/ml_evaluation_benchmark.json`
- `reports/ML_EVALUATION.md`
- `reports/speech_benchmark_report.json`
- `reports/SPEECH_BENCHMARK.md`

The churn evaluation uses an 80/20 stratified split with random seed 42. MELD metrics use the held-out test partition recorded in the JSON artifact (2,610 turns across 280 dialogues). Speech metrics use the four reference fixtures and their hard-coded reference transcripts/turns in `training/benchmark_speech.py`.

For a traceable result, record the Git commit SHA, Python version, OS/CPU, installed dependency versions, dataset checksums, and the exact command output alongside any newly generated report. Dependency lower bounds in `requirements.txt` are not a fully locked environment; scores may vary across package/model versions and hardware. Compare generated JSON against the checked-in artifacts rather than silently overwriting a difference.

## 2. Churn model: baselines and class-level results

The model is a **static account-level bank churn propensity model** trained on tabular customer attributes. It is **not** trained on call audio or transcripts, and its probability must not be described as call-derived churn prediction.

### Dataset and split

- Records: **165,031**
- Train/test: **132,024 / 33,007**, stratified 80/20, seed **42**
- Retained: **130,110 (78.84%)**
- Churned: **34,921 (21.16%)**
- Test confusion matrix at the recorded model threshold, rows = actual `[retained, churned]`, columns = predicted `[retained, churned]`:

```text
[[21196, 4827],
 [ 1513, 5471]]
```

### Model comparison

| Model | ROC-AUC | PR-AUC in saved artifact | Brier score | ECE | F1 |
|---|---:|---:|---:|---:|---:|
| Prevalence dummy | 0.5000 | 0.6058* | 0.1668 | 0.0000 | 0.0000 |
| Logistic regression (balanced) | 0.8093 | 0.5574 | 0.1747 | 0.2031 | 0.5496 |
| Random forest | 0.8766 | 0.6885 | 0.1392 | 0.1681 | 0.6306 |
| XGBoost | **0.8808** | **0.7109** | 0.1363 | 0.1580 | **0.6331** |
| XGBoost + Platt calibration | 0.8808 | 0.7109 | **0.0990** | **0.0072** | 0.6331 |

`*` **Baseline caveat:** the saved dummy PR-AUC of 0.6058 is generated from a constant-score precision-recall curve and is not a meaningful random-ranking PR baseline. For an imbalanced binary task, the positive prevalence **0.2116** is the appropriate PR-AUC reference level for a random ranking. The report's phrase “+0.4993 lift” is arithmetic against prevalence (0.7109 − 0.2116), not a measured improvement against the dummy row's 0.6058. Interpret PR-AUC using the prevalence reference and rerun the benchmark before making comparative claims.

### Per-class metrics (XGBoost)

Metrics derived from the saved confusion matrix and the recorded churn-class scores; rounding may cause small differences.

| Actual class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Retained (0) | 0.9334 | 0.8145 | 0.8699 | 26,023 |
| Churned (1) | 0.5313 | 0.7834 | 0.6331 | 6,984 |

The model catches about 78% of churners at this threshold, while roughly 47% of predicted churn cases are false positives (precision 53.1%). A lower business threshold can increase recall, but it also increases outreach volume and false positives. The EMV threshold in the existing report depends on explicit illustrative financial assumptions ($1,200 lost LTV, $80 concession, 65% save rate); these are scenario inputs, not measured realized savings.

### Cross-validation artifact

Five-fold stratified cross-validation on a 30,000-account sample records ROC-AUC **0.8741 ± 0.0036**, PR-AUC **0.7034 ± 0.0032**, Brier **0.1373 ± 0.0023**, and F1 **0.6220 ± 0.0061**. These are fold mean ± standard deviation, not confidence intervals.

## 3. MELD emotion: per-class metrics and failure analysis

The test set contains 2,610 utterances across 280 dialogues. The seven emotion classes are imbalanced: neutral 1,256; joy 402; anger 345; surprise 281; sadness 208; disgust 68; fear 50.

### Overall comparison

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Neutral-majority baseline | 0.4812 | — | — |
| Lexical text-only | 0.4372 | 0.3087 | 0.4461 |
| Acoustic + text fusion | 0.4785 | 0.3334 | 0.4835 |

The fused model improves on text-only macro-F1 by **0.0247**, but its accuracy (47.85%) remains slightly below the 48.12% majority baseline. That is a warning against describing the emotion classifier as broadly reliable.

### Fused model per-class report

| Emotion | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Anger | 0.3077 | 0.3246 | 0.3159 | 345 |
| Disgust | 0.1500 | 0.0882 | 0.1111 | 68 |
| Fear | 0.0725 | 0.1000 | 0.0840 | 50 |
| Joy | 0.4065 | 0.4378 | 0.4216 | 402 |
| Neutral | 0.6571 | 0.5812 | 0.6168 | 1,256 |
| Sadness | 0.3080 | 0.3317 | 0.3194 | 208 |
| Surprise | 0.4092 | 0.5374 | 0.4646 | 281 |

### Confusion and failure patterns

- **Neutral absorbs many non-neutral utterances.** The confusion matrix shows 143 anger, 162 joy, 74 sadness, and 104 surprise examples predicted as neutral. This is consistent with the majority class dominating predictions.
- **Fear and disgust are not dependable.** Fear F1 is 0.0840 (50 examples) and disgust F1 is 0.1111 (68 examples). Do not use these classes alone to trigger consequential actions.
- **Acoustic fusion helps modestly, not decisively.** Accuracy improves from 43.72% to 47.85% and macro-F1 from 0.3087 to 0.3334, but the fused model still does not exceed majority-baseline accuracy.
- **Sentiment remains imperfect.** The positive class has precision 0.3833 and F1 0.4247; positive/negative customer intent should be checked against transcript evidence and not inferred from emotion alone.

### Sentiment per-class report

| Sentiment | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.5152 | 0.4874 | 0.5009 | 833 |
| Neutral | 0.6136 | 0.5740 | 0.5932 | 1,256 |
| Positive | 0.3833 | 0.4760 | 0.4247 | 521 |

## 4. Speech benchmark: per-call results and limitations

| Scenario | WER | DER (reported) | Discovered speakers | Expected speakers | RTF |
|---|---:|---:|---:|---:|---:|
| Retention call | 1.0% | 0.0% | 2 | 2 | 0.130 |
| Sales expansion | 0.0% | 0.0% | 3 | 2 | 0.072 |
| Technical escalation | 2.11% | 0.0% | 3 | 2 | 0.081 |
| Additive-noise stress | 1.0% | 0.0% | 2 | 2 | 0.075 |

The perfect reported DER should be interpreted cautiously: two calls produce three speaker clusters when the reference expects two. The reported DER implementation appears to allow optimal mapping without fully reflecting extra discovered clusters in the headline value. Speaker-count accuracy is 2/4 (50%), so the headline DER alone is insufficient. Review the DER implementation and add tests that penalize over-segmentation/extra speakers before using DER as a release gate.

The synthetic fixtures also have clean, scripted reference speech and do not cover representative accent, overlap, telephony codec, or noise distributions. A real-call evaluation requires consent, de-identification, independently human-verified transcripts/speaker turns, a held-out split, and results segmented by relevant acoustic conditions.

## 5. Release gates and next experiments

1. **Fix evaluation accounting:** replace the constant-score dummy PR-AUC interpretation with a prevalence reference and document average-precision vs trapezoidal PR-AUC consistently.
2. **Strengthen emotion performance:** report per-class confusion changes, tune class weighting only on training/validation data, and evaluate calibration and abstention for low-confidence minority classes.
3. **Audit diarization scoring:** verify the DER implementation against known over-clustering/under-clustering fixtures and a trusted scorer; publish the scoring code/version and collar.
4. **Collect representative data:** evaluate only on consented, de-identified calls with independent annotation and a held-out set.
5. **Make runs fully reproducible:** lock dependencies, hash datasets and fixtures, record hardware/software versions and random seeds, and archive generated JSON/Markdown as CI artifacts.

Until those gates pass, treat this as a strong research/portfolio prototype, not a validated production call-center scoring system.
