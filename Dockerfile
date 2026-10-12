# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install audio processing system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, models, configs, and dashboard
COPY src/ ./src/
COPY configs/ ./configs/
COPY dashboard/ ./dashboard/
COPY models/ ./models/
COPY reports/ ./reports/
COPY data/samples/ ./data/samples/

EXPOSE 8000 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: launch FastAPI server
CMD ["uvicorn", "src.callintel.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
