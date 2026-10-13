# Evaluation protocol: emotion, diarization, and real calls

## Emotion (MELD)

Run `python training/evaluate_ml_models.py`. The evaluator trains TF-IDF + class-weighted logistic regression using only the official MELD train dialogue split and scores the official held-out test dialogues once. It reports majority-class accuracy and macro-F1, model macro/weighted F1, per-class precision/recall/F1, confusion matrix, and a 95% percentile bootstrap interval that resamples whole dialogues. It asserts no dialogue-ID overlap across splits. This measures scripted TV-dialogue emotion classification, not call-center performance. Acoustic fusion is not claimed.

CI executes this evaluation. A failed macro-F1 baseline comparison is a failed model objective, not something to hide with accuracy.

## Diarization ground truth

Synthetic sample-call fixtures are regression fixtures, not independent ground truth. Do not report their WER as a real-world estimate, and do not report DER without reference speaker turns with timestamps.

Prepare independently human-annotated reference RTTM and system hypothesis RTTM using identical recording IDs. Annotators must not see system predictions; document annotation instructions and adjudicate disagreements. Install optional dependencies with `python -m pip install -r requirements-speech-eval.txt`, then run:

```bash
python training/score_diarization.py --reference data/real_calls/reference.rttm --hypothesis reports/real_calls/hypothesis.rttm --collar 0.25
```

The scorer uses pyannote.metrics DER and records the collar/overlap policy. The repository currently does not contain a licensed, independently timestamp-annotated real-call evaluation corpus, so no real-call DER result is claimed until one is added.

## Realistic call-center validation

Use consented, de-identified calls covering accents, channels/codecs, noise, overlap, agent/customer turns, call reasons, and outcomes. Freeze call-level train/dev/test splits before tuning; segments from one call must never cross splits. Independently verify transcripts against audio and annotate speaker turn start/end times. Report WER by condition, DER with scoring convention, emotion macro-F1/per-class metrics with call-level uncertainty, and subgroup breakdowns. Keep identifying data out of Git and document consent, retention, and access controls. Public MELD plus generated fixtures cannot establish production call-center performance.
