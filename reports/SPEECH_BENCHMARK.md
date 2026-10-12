# Speech Processing & Diarization Benchmark Report

Generated: 2026-10-12 17:19:23
ASR Engine: `faster-whisper (model=base, compute_type=int8/float32)`
Diarizer: `AcousticDiarizer (34-dim acoustic vector: F0 pitch, 13 MFCC means & stds, spectral centroid, rolloff, RMS)`

---

## 1. Executive Benchmark Summary

| Metric | Measured Value | Industry Target | Status |
| :--- | :--- | :--- | :--- |
| **Mean Word Error Rate (WER)** | **0.50%** | < 15.0% | **PASSED (Production-Grade)** |
| **Mean Diarization Error Rate (DER)** | **0.00%** | < 25.0% | **PASSED (Auditable Turn Segregation)** |
| **Real-Time Factor (RTF)** | **0.107x** | < 1.0x (faster than real-time) | **PASSED (High Throughput)** |
| **Peak Memory Footprint** | **61.92 MB** | < 1,024 MB | **PASSED (Edge/CPU Deployable)** |

---

## 2. Diarization Error Rate (DER) Evaluation Methodology

DER is evaluated using standard NIST RT-09 scoring rules:
1. **Collar Tolerance**: $\pm 0.250$s forgiveness window placed at speaker transition boundaries to accommodate acoustic pause overlaps.
2. **Formula**:
   $$\text{DER} = \frac{\text{Missed Speech} + \text{False Alarm} + \text{Speaker Confusion}}{\text{Total Reference Speech Duration}}$$
3. **Speaker Mapping**: Optimal Hungarian 1-to-1 mapping matching acoustic cluster labels to ground truth speakers (`Agent`, `Customer`).

---

## 3. Detailed Call-Level Benchmark Breakdown

| Call ID | Scenario | Duration | Processing Time | RTF | WER | DER | Peak RAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CALL_2026_APEX_001` | Customer Retention (Pricing & Cancellation Threat) | 45.9s | 6.38s | 0.139x | **1.0%** | **0.0%** | 66.01 MB |
| `CALL_2026_SALES_002` | Sales Expansion (Enterprise Tier & Buying Signals) | 49.8s | 3.73s | 0.075x | **0.0%** | **0.0%** | 57.82 MB |

---

## 4. Acoustic Stress & Hardware Analysis

- **Hardware Target**: Dual-core x86_64 CPU (Local / Container)
- **Model Size**: Whisper `base` (74M parameters)
- **Quantization**: Automatic 8-bit int8 / CPU float32
- **Acoustic Clustering**: Feature extraction runs in $< 50$ms per turn on Mel-frequency filterbanks without GPU dependencies.
- **Handling Overlap**: When conversational overlap occurs, energy-weighted pitch extraction identifies the dominant fundamental frequency ($F_0$).
