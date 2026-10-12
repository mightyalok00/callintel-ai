# AI Call Intelligence Engine

[![CallIntel AI CI Pipeline](https://github.com/mightyalok00/callintel-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/mightyalok00/callintel-ai/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Turn customer conversations into actionable business intelligence — not just transcripts.**

---

## 1. Executive Summary & Design Scope

The **AI Call Intelligence Engine** is a comprehensive Data Science, Speech AI, Predictive ML, and Multi-Agent platform designed to analyze recorded and live customer calls across **Customer Support**, **Sales**, and **Retention**.

### Core Choices

- **Input Channels**: Both recorded call audio files (`.wav`, `.mp3`) and live microphone/browser streaming via WebSocket.
- **Target Domains**: Customer Support (ticket categorization, escalation, QA), Sales (buying signals, objections, competitor tracking), Retention (cancellation risk, churn mitigation).
- **Deployment**: Hybrid architecture — local-first models (`faster-whisper`, scikit-learn, XGBoost, local Ollama) with seamless optional cloud LLM API fallback.
- **Deliverable**: Full portfolio project with FastAPI backend, WebSocket streaming, PostgreSQL/SQLite database, and an executive Streamlit Analytics Dashboard with an auditable **Evidence Ledger**.
- **Speech Emotion & Conversation Dataset**: Ingested directly from **`D:\MELD-master`** (13,708 multi-turn conversation utterances across 1,432 dialogues).

---

## 2. System Architecture

```mermaid
graph TB
    subgraph Ingestion["1. Ingestion Layer"]
        A1[Recorded Call Audio Files] --> B1[Validation & Audio Normalization 16kHz Mono]
        A2[Live Microphone Audio Stream] --> B2[WebSocket Streaming Server & Audio Chunks]
    end

    subgraph Speech["2. Speech Intelligence Layer"]
        B1 & B2 --> C1[VAD: Voice Activity Detection]
        C1 --> C2[ASR: faster-whisper / Whisper Base]
        C2 --> C3[Speaker Diarization / Turn Segmentation]
    end

    subgraph NLP_ML["3. NLP & Predictive ML Pipeline"]
        C3 --> D1[Emotion & Sentiment Classifier\nTrained on MELD: D:\MELD-master]
        C3 --> D2[Customer Intent & Objection Classifier\n10 Commercial Call Classes]
        C3 --> D3[Topic & Unmet Needs Extractor\nSentence Embeddings + Clustering]
        C3 --> D4[Predictive Churn Risk Model\nXGBoost + SHAP Explainability]
    end

    subgraph MultiAgent["4. Multi-Agent Intelligence & Evidence Ledger"]
        D1 & D2 & D3 & D4 --> E1[Evidence Ledger: Timestamp & Turn Attribution]
        E1 --> E2[Agent 1: Call Analyst Agent]
        E1 --> E3[Agent 2: Customer Emotion Agent]
        E1 --> E4[Agent 3: Sales Intelligence Agent]
        E1 --> E5[Agent 4: Retention Risk Agent]
        E1 --> E6[Agent 5: QA & Compliance Agent]
        E2 & E3 & E4 & E5 & E6 --> E7[Agent 6: Decision & Action Agent]
        E7 --> E8[Hybrid Provider: Ollama / Local / Cloud API / Rule Fallback]
    end

    subgraph Serving["5. Serving & UI Layer"]
        E8 --> F1[(SQLite / PostgreSQL DB)]
        E8 --> F2[FastAPI REST API + WebSocket Server]
        F2 --> G1[Streamlit Executive Intelligence Dashboard]
    end
```

---

## 3. Dataset Architecture

### Primary Dataset: MELD (`D:\MELD-master`)

- **Location**: `D:\MELD-master\data\MELD`
- **Volume**: 13,708 utterances across 1,432 dialogues.
  - **Train**: 9,989 utterances (1,038 dialogues)
  - **Dev**: 1,109 utterances (114 dialogues)
  - **Test**: 2,610 utterances (280 dialogues)
- **Annotations**:
  - `Emotion`: 7 discrete emotions (`neutral`, `joy`, `surprise`, `anger`, `sadness`, `disgust`, `fear`)
  - `Sentiment`: 3 polarity classes (`positive`, `neutral`, `negative`)
  - `Speaker`, `StartTime`, `EndTime` for turn-by-turn conversation dynamics.
- **Ingestion**: Automated via `training/ingest_meld.py` into `data/meld/`.

### Supplementary Datasets

1. **Commercial Call Intent Dataset**: 10-class intent taxonomy (Pricing Objections, Cancellation Warnings, Feature Requests, Bug Reports, Account Access, Upgrade Inquiries, Competitor Comparisons, Service Complaints, Product Questions, General Inquiries).
2. **Customer Churn Dataset**: Supervised tabular dataset (`bank_churn_cleaned.csv`) for training XGBoost risk scoring with SHAP feature attributions.

---

## 4. Repository & Directory Structure

```text
e:\ai\
├── README.md                           # Project documentation & architecture
├── requirements.txt                    # Pinned Python dependencies
├── pyproject.toml                      # Package build configuration
├── configs/
│   ├── app_config.yaml                 # Application & server configuration
│   └── models_config.yaml              # Model hyperparameters & paths
├── data/
│   ├── meld/                           # Processed MELD dataset from D:\MELD-master
│   │   ├── meld_train_dialogues.json   # 1,038 dialogues (9,989 turns)
│   │   ├── meld_dev_dialogues.json     # 114 dialogues (1,109 turns)
│   │   ├── meld_test_dialogues.json    # 280 dialogues (2,610 turns)
│   │   └── meld_summary_metadata.json  # Class distributions & metadata
│   ├── churn/                          # Tabular churn data for predictive ML
│   └── samples/                        # Synthetic sample audio (.wav) for demos
├── src/
│   └── callintel/
│       ├── core/                       # App settings, logging, PII security
│       ├── schemas/                    # Pydantic schemas (Transcript, Turn, Evidence)
│       ├── speech/                     # faster-whisper, VAD, audio normalization
│       ├── ml/                         # Emotion, Intent, Churn & Topic models
│       ├── agents/                     # 6 specialized agents + supervisor
│       ├── providers/                  # Hybrid LLM interface (Ollama / Cloud / Rule Fallback)
│       ├── db/                         # SQLAlchemy ORM models & repository
│       ├── api/                        # FastAPI routes & WebSocket streamer
│       └── services/                   # Audio pipeline & live session manager
├── dashboard/
│   ├── app.py                          # Streamlit application entrypoint
│   ├── components/                     # Audio player, Evidence Ledger widget
│   └── pages/                          # Multi-page dashboard views
├── training/
│   ├── ingest_meld.py                  # Ingestion script for D:\MELD-master
│   ├── train_emotion.py                # Emotion classifier training
│   ├── train_intent.py                 # Intent classifier training
│   └── train_churn.py                  # XGBoost churn model + SHAP explainer
└── tests/
    ├── unit/                           # Audio processing & schema tests
    ├── integration/                    # Pipeline end-to-end tests
    └── api/                            # REST & WebSocket tests
```

---

## 5. Technology Stack & Dependencies

- **Language**: Python 3.12 (64-bit)
- **Audio & Speech**: `faster-whisper`, `soundfile`, `webrtcvad-wheels`
- **Data Science & ML**: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`
- **NLP & Transformers**: `torch`, `transformers`, `sentence-transformers`
- **API & Streaming**: `fastapi`, `uvicorn[standard]`, `websockets`, `pydantic`
- **Dashboard**: `streamlit`, `plotly`
- **Database**: `sqlalchemy`, `sqlite3` (development) / `postgresql` (production)
- **Containerization & CI/CD**: `Docker`, `docker-compose`, `GitHub Actions`
- **Testing**: `pytest`, `pytest-asyncio`, `httpx`

---

## 6. Model Benchmarks & Statistical Integrity

### MELD Emotion & Sentiment Classifiers

- **Dataset**: MELD (13,708 conversational turns across 1,432 multi-turn dialogues).
- **Architecture**: TF-IDF + Calibrated Linear Support Vector Machine (`SGDClassifier(loss='log_loss')`) with class balancing.
- **Emotion Performance (7-class)**:
  - Neutral F1: **0.78**
  - Joy F1: **0.55**
  - Anger F1: **0.44**
  - Surprise F1: **0.51**
  - Overall Weighted F1: **0.6558**
- **Sentiment Performance (3-class polarity)**:
  - Neutral / Positive / Negative Macro F1: **0.6406**

### Customer Churn Risk Model & SHAP Explainability

- **Dataset**: Bank Customer Churn dataset (165,000+ customer records).
- **Architecture**: XGBoost Gradient Boosted Decision Trees + TreeSHAP Explainer.
- **Statistical Benchmark**:
  - **Test ROC-AUC**: `0.8808`
  - **Test PR-AUC**: `0.7109`
  - **Brier Calibration Score**: `0.0984`
- **Statistical Separation Guarantee**:
  - The model outputs a statistically verified `baseline_churn_probability` (0.0 to 1.0) derived strictly from the trained XGBoost model and explained via TreeSHAP feature contributions.
  - In-call conversational modifiers (e.g., explicit cancellation threats, extreme frustration peaks) are computed separately and reported under `in_call_operational_modifier` as contact-center operational escalation signals, never misrepresented as learned longitudinal probabilities.

---

## 7. Verified API & WebSocket Reference

The backend is built with **FastAPI** (`src/callintel/api/app.py`) and backed by SQLAlchemy ORM persistence.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health check and model readiness status |
| `POST` | `/api/v1/calls/upload` | Multipart audio file upload, async processing & persistence |
| `GET` | `/api/v1/calls/{call_id}` | Retrieve call metadata, duration, and processing status |
| `GET` | `/api/v1/calls/{call_id}/transcript` | Retrieve timestamped transcript turns and speaker attributions |
| `GET` | `/api/v1/calls/{call_id}/insights` | Retrieve ML predictions, SHAP drivers, and Evidence Ledger claims |
| `GET` | `/api/v1/analytics/overview` | Aggregated contact center KPIs, sentiment ratios, and risk counts |
| `WS` | `/api/v1/stream/{session_id}` | Real-time WebSocket streaming audio chunk processing & frustration alerts |

---

## 8. Quickstart & Verification Guide

### 1. Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/mightyalok00/callintel-ai.git
cd callintel-ai

# Create and activate Python 3.12 virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Test Suite

```bash
pytest tests/unit/ tests/api/ -v
```

*Expected: 12 passed tests across speech, ML, agents, and FastAPI REST/WebSocket endpoints.*

### 3. Launch the FastAPI Service

```bash
uvicorn src.callintel.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger API documentation is available at `http://localhost:8000/docs`.

### 4. Launch the Executive Streamlit Dashboard

```bash
streamlit run dashboard/app.py --server.port 8501
```

Open `http://localhost:8501` to access:

- **Call Explorer**: Audio player with synchronized transcript highlighting.
- **Evidence Ledger**: Auditable claims linked directly to conversation turn IDs and timestamps.
- **SHAP Retention Waterfall**: Interactive Plotly waterfall charts visualizing customer risk factors.
- **Action Dispatcher**: One-click escalation to CRM, Slack, and Zendesk.

### 5. Run with Docker Compose

```bash
docker-compose up --build
```

This boots both the FastAPI backend (`port 8000`) and the Streamlit dashboard (`port 8501`) concurrently.

---

## 9. CI/CD & Automated Safeguards

- **GitHub Actions Workflow** (`.github/workflows/ci.yml`): Runs on every pull request, push to `main`, and manual dispatch (`workflow_dispatch`) across Ubuntu and Windows runners.
- **Automated Validation**:
  - Python 3.12 syntax and unit test verification.
  - REST & WebSocket API lifecycle test coverage.
  - Secret scanning and git hygiene verification.
