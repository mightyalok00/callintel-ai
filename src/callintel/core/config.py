"""
Centralized Configuration for CallIntel AI.
Provides typed, validated configuration parameters across presentation, application,
and infrastructure layers.
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "CallIntel AI - Call Intelligence Engine"
    environment: str = "production"
    debug: bool = False

    # Storage paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent.parent
    upload_dir: Path = Path("data/uploads")
    samples_dir: Path = Path("data/samples")
    reports_dir: Path = Path("reports")

    # Database
    database_url: str = "sqlite:///./callintel.db"

    # Audio & Streaming
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    max_chunk_bytes: int = 512 * 1024  # 512 KB per frame
    max_total_buffer_bytes: int = 10 * 1024 * 1024  # 10 MB per stream
    max_upload_size_bytes: int = 25 * 1024 * 1024  # 25 MB max upload
    cleanup_temp_files: bool = True  # Automatically clean up temporary upload artifacts
    upload_retention_hours: int = 24  # Stale file purge threshold

    # Security & API
    api_key: Optional[str] = None
    require_auth: bool = False
    cors_origins: list[str] = ["*"]

    # Logging
    log_level: str = "INFO"


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.reports_dir.mkdir(parents=True, exist_ok=True)
