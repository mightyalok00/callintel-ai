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

A valid diarization error rate (DER) requires time-aligned reference speech regions and scoring for missed speech, false alarms, speaker confusion, and the selected collar convention. The checked-in reference turns contain text and speaker labels but no reference timestamps. DER is therefore **not measured** in this benchmark. The former 0.00% DER claim was unsupported and has been removed.

The code now reports `der: null` and a separate text-overlap speaker-attribution diagnostic. That diagnostic is not DER and must not be compared with published DER targets. The checked-in numeric table below is a historical snapshot; rerun `python training/benchmark_speech.py` to regenerate measurements using the corrected metric labels.

## 3. Detailed Call-Level Benchmark Breakdown

| Call ID | Scenario | Duration | Processing Time | RTF | WER | Speaker-attribution proxy (not DER) | Discovered $k^*$ | Peak RAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| `CALL_2026_APEX_001` | Customer Retention (Pricing & Cancellation Threat) | 45.9s | 5.98s | 0.13x | **1.0%** | Not measured | `k*=2` | 66.01 MB |
| `CALL_2026_SALES_002` | Sales Expansion (Enterprise Tier & Buying Signals) | 49.8s | 3.57s | 0.072x | **0.0%** | Not measured | `k*=3` | 57.82 MB |
| `CALL_2026_TECH_003` | Technical Outage (Critical Bug & Urgent Escalation) | 41.8s | 3.39s | 0.081x | **2.11%** | Not measured | `k*=3` | 48.55 MB |
| `CALL_2026_NOISE_STRESS` | Acoustic Noise Stress (+15dB SNR Additive Office Noise) | 45.9s | 3.46s | 0.075x | **1.0%** | Not measured | `k*=2` | 53.39 MB |

---

## 4. Acoustic Stress & Hardware Analysis

- **Hardware Target**: Dual-core x86_64 CPU (Local / Container)
- **Model Size**: Whisper `base` (74M parameters)
- **Quantization**: Automatic 8-bit int8 / CPU float32
- **Acoustic Clustering**: Feature extraction runs in $< 50$ms per turn on Mel-frequency filterbanks without GPU dependencies.
- **Gain Invariance**: $L_2$-normalized Cosine Agglomerative Clustering eliminates sensitivity to microphone gain variations.
- **Handling Overlap**: When conversational overlap occurs, energy-weighted pitch extraction identifies the dominant fundamental frequency ($F_0$).
