# Validation, Evidence Quality, Retention & Production Safety

This document distinguishes **implemented controls** from **validation still required**. A passing CI run or a synthetic benchmark is not evidence that the system is ready for unrestricted production use.

## 1. Speech quality: benchmark limits and real-call validation

The current speech benchmark uses four controlled, synthesized sample calls. The reported WER of 1.03% and DER of 0% describe those fixtures only; they must not be presented as expected accuracy for real customer calls.

Before making a real-world accuracy claim, build a consented, de-identified evaluation set that is separate from training data and representative of the intended deployment:

- Multiple accents and speaking rates, including code-switching and non-native speech.
- Interruptions, short backchannels, overlapping speech, crosstalk, and speaker changes.
- Telephone-bandwidth codecs, packet loss, clipping, quiet speech, reverberation, and background noise at several measured SNR levels.
- Different microphones, devices, call lengths, and agent/customer voice characteristics.
- A documented human-verified transcript and speaker-turn reference for every evaluated recording.

Report WER/CER and diarization error rate (DER) with the scoring convention, sample count, confidence intervals, language/accent breakdowns, and failure cases. Keep a held-out test set untouched during tuning. Compare against a baseline and run the same benchmark whenever the ASR/diarization model, preprocessing, or decoding configuration changes. Do not claim that this evaluation has happened until the dataset exists and the benchmark artifacts are published.

## 2. ML dataset scope and business claims

- **MELD** supports conversational emotion/sentiment research. It is not a telephony customer-support dataset and does not by itself validate call-center performance across accents, channels, or industries.
- **Bank churn data** supports an account-level tabular churn baseline. It is not a call-audio or transcript-derived churn model. A score from this dataset must not be described as churn predicted from the call.
- **In-call signals** such as an explicit cancellation statement, frustration, or unresolved escalation should be treated as separately evaluated conversational indicators or rules unless trained and validated with appropriate call-outcome labels.

For any future call-based churn model, create a cohort with consent and an explicit outcome window, prevent post-outcome leakage, split by customer (and preferably time), report prevalence and calibration as well as ROC-AUC/PR-AUC, and evaluate performance by relevant customer and channel segments. Document whether the output is a propensity score, a rule-based escalation flag, or a validated probability. Never conflate these outputs.

## 3. Evidence-ledger adversarial validation

Every user-facing recommendation should be traceable to specific transcript turns and should be withheld or marked uncertain when its supporting evidence is missing, contradicted, or attributed to the wrong speaker.

The evidence validator should be exercised with an adversarial regression suite covering at least:

| Case | Example failure to catch | Expected behavior |
|---|---|---|
| Paraphrased contradiction | “I will cancel” summarized as “the customer agreed to stay” | Reject or mark contradicted |
| Negation / reversal | “The outage is not fixed” summarized as “the outage is fixed” | Reject or mark contradicted |
| Unsupported recommendation | Recommends a discount although no price concern is stated | Mark unsupported / abstain |
| Wrong speaker | Agent's promise attributed to the customer | Reject attribution or flag ambiguity |
| Invalid citation | Cites a missing turn ID | Reject |
| Misleading quote | Quotes a phrase that is immediately negated in the same turn/context | Reject or mark contradicted |
| Weak semantic overlap | A valid turn ID is cited but does not support the actual claim | Mark unsupported |
| Multiple evidence turns | One cited turn supports a claim but another directly refutes it | Surface conflict; do not silently select one |

These examples define the expected test coverage; they are not a claim that all cases already pass. Add each discovered failure as a deterministic regression test, and retain a reviewed set of false positives/false negatives for validator tuning. For high-impact recommendations, require a human reviewer until evaluation demonstrates an acceptable error rate.

## 4. Deletion and data lifecycle

The delete-call endpoint now removes the database record through the repository/service and best-effort deletes runtime files whose names include the call ID from the configured upload and report directories, plus the known data/cache and .cache/callintel locations. The history-delete endpoint clears call records, purges files in the upload staging directory, and removes report/cache files only when their names identify a call ID. Shared benchmark reports and artifacts without a call ID are intentionally retained.

Deletion is best-effort, not a cryptographic-erasure guarantee. A successful response reports how many files were removed, but filesystem errors are logged and should be monitored. Before claiming complete deletion, verify where each component stores data, including database backups, external object storage, logs, temporary directories, browser caches, and any model/provider retention. Those external systems are not controlled by this cleanup helper.

Retention controls currently include temporary-upload cleanup and a configurable upload age threshold. Set an explicit retention period for transcripts, reports, and backups as well; confirm whether each data store has an independent TTL. Do not retain raw audio longer than necessary.

## 5. Authentication, access control, and privacy

**Important current limitation:** authentication defaults to disabled and the API key defaults to unset. This is suitable only for local development. Runtime overrides are read from environment variables. Before deployment, set CALLINTEL_REQUIRE_AUTH=true and CALLINTEL_API_KEY to a unique high-entropy secret. If authentication is enabled without a key, HTTP routes fail closed with a configuration error and WebSocket connections are rejected. Configure CALLINTEL_CORS_ORIGINS as a comma-separated list of exact trusted origins. Never use a demo/default key, commit secrets, or put credentials in client-side code.

PowerShell example:

```powershell
$env:CALLINTEL_REQUIRE_AUTH = "true"
$env:CALLINTEL_API_KEY = "replace-with-a-long-random-secret"
$env:CALLINTEL_CORS_ORIGINS = "https://your-dashboard.example.com"
```

Linux/macOS example:

```bash
export CALLINTEL_REQUIRE_AUTH=true
export CALLINTEL_API_KEY="replace-with-a-long-random-secret"
export CALLINTEL_CORS_ORIGINS="https://your-dashboard.example.com"
```

The environment loader also supports CALLINTEL_DATABASE_URL, CALLINTEL_UPLOAD_DIR, CALLINTEL_REPORTS_DIR, CALLINTEL_UPLOAD_RETENTION_HOURS, and audio upload/buffer limits. See src/callintel/core/config.py for the complete supported list.

Before deployment:

1. Enable authentication and test missing, invalid, and valid credentials on upload, read, delete, analytics, and WebSocket routes.
2. Restrict CORS to the exact trusted frontend origins. Avoid wildcard origins for production.
3. Put the API behind HTTPS/TLS and a trusted reverse proxy; restrict network ingress and rate-limit upload and analysis endpoints.
4. Apply tenant/user authorization to each call ID. An API key alone is not row-level authorization for a multi-user deployment.
5. Redact phone numbers, names, credentials, and other PII from logs. Encrypt database volumes, audio, and backups; restrict filesystem permissions.
6. Define recording consent, purpose limitation, access logging, deletion SLAs, backup expiry, and incident-response procedures appropriate to the jurisdiction and customer contracts.
7. Review the privacy and retention terms of any external speech or LLM provider. Local inference does not remove the application's own access-control and retention obligations.

## 6. Release checklist

- [ ] Evaluate on a held-out, consented real-call dataset and publish a stratified scorecard.
- [ ] Keep synthetic benchmark results explicitly labeled as fixture-only.
- [ ] Add regression tests for all evidence-validator adversarial cases above.
- [ ] Verify per-call and history deletion against DB rows, uploads, generated reports, and configured caches.
- [ ] Test auth on every API and WebSocket route; use explicit CORS origins.
- [ ] Document retention, consent, PII handling, backup lifecycle, and provider data handling.
- [ ] Do not call the system production-ready until the above controls have been tested in the deployment environment.
