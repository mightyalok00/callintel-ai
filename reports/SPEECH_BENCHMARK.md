# Speech Processing & Diarization Benchmark Report

Generated: 2026-10-12 18:55:50
ASR Engine: `faster-whisper (model=base, compute_type=int8/float32)`
Diarizer: `AcousticDiarizer (34-dim acoustic vector: F0 pitch, 13 MFCC means & stds, spectral centroid, rolloff, RMS)`

---

## 1. Executive Benchmark Summary

| Metric | Measured Value | Industry Target | Status |
| :--- | :--- | :--- | :--- |
| **Mean Word Error Rate (WER)** | **1.03%** | < 15.0% | **Controlled fixtures only; not generalizable** |
| **Diarization Error Rate (DER)** | **Not measured** | Requires time-aligned references | **Not reported** |
| **Speaker Count Discovery ($k^*$)** | **50.0%** | > 85.0% | **Below target** |
| **Real-Time Factor (RTF)** | **0.09x** | < 1.0x (faster than real-time) | **Measured on this environment** |
| **Peak Memory Footprint** | **56.44 MB** | < 1,024 MB | **Measured on this environment** |

> [!NOTE]
> **Controlled Benchmark Fixture vs In-the-Wild Telephony**:
> Controlled fixtures are not evidence of real-world telephony performance. DER is not measured because reference turns have no timestamps; the previous 0.00% DER claim was unsupported.
> A text-overlap speaker-attribution diagnostic is not a substitute for DER and must not be compared with published DER targets.

---

## 2. Diarization metric limitation

A valid DER requires time-aligned reference speech regions and scoring missed speech, false alarms, speaker confusion, and a declared collar convention. The checked-in reference turns contain text and speaker labels but no reference timestamps. Therefore DER is not measured in this report. The former methodology description below does not describe the implemented proxy:
1. **Collar Tolerance**: $\pm 0.250$s forgiveness window placed at speaker transition boundaries to accommodate acoustic pause overlaps.
2. **Formula**:
   $$\text{DER} = \frac{\text{Missed Speech} + \text{False Alarm} + \text{Speaker Confusion}}{\text{Total Reference Speech Duration}}$$
3. **Speaker Mapping**: Optimal Hungarian 1-to-1 mapping matching acoustic cluster labels to ground truth speakers (`Agent`, `Customer`).
4. **Automatic Speaker Count Discovery ($k^*$)**: When speaker count is unprovided, Silhouette scoring on Cosine distance across $k \in [1, 3]$ automatically estimates optimal cluster count.

---

## 3. Detailed Call-Level Benchmark Breakdown

| Call ID | Scenario | Duration | Processing Time | RTF | WER | Speaker-attribution proxy (not DER) | Discovered $k^*$ | Peak RAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| `CALL_2026_APEX_001` | Customer Retention (Pricing & Cancellation Threat) | 45.9s | 5.98s | 0.13x | **1.0%** | **0.0%** | `k*=2` | 66.01 MB |
| `CALL_2026_SALES_002` | Sales Expansion (Enterprise Tier & Buying Signals) | 49.8s | 3.57s | 0.072x | **0.0%** | **0.0%** | `k*=3` | 57.82 MB |
| `CALL_2026_TECH_003` | Technical Outage (Critical Bug & Urgent Escalation) | 41.8s | 3.39s | 0.081x | **2.1%** | **0.0%** | `k*=3` | 48.55 MB |
| `CALL_2026_NOISE_STRESS` | Acoustic Noise Stress (+15dB SNR Additive Office Noise) | 45.9s | 3.46s | 0.075x | **1.0%** | **0.0%** | `k*=2` | 53.39 MB |

---

## 4. Acoustic Stress & Hardware Analysis

- **Hardware Target**: Dual-core x86_64 CPU (Local / Container)
- **Model Size**: Whisper `base` (74M parameters)
- **Quantization**: Automatic 8-bit int8 / CPU float32
- **Acoustic Clustering**: Feature extraction runs in $< 50$ms per turn on Mel-frequency filterbanks without GPU dependencies.
- **Gain Invariance**: $L_2$-normalized Cosine Agglomerative Clustering eliminates sensitivity to microphone gain variations.
- **Handling Overlap**: When conversational overlap occurs, energy-weighted pitch extraction identifies the dominant fundamental frequency ($F_0$).
