"""
Centralized Configuration for CallIntel AI.
Provides typed, validated configuration parameters across presentation, application,
and infrastructure layers.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel


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


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable without treating 'false' as truthy."""
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false, 1/0, yes/no, or on/off")


def load_settings_from_environment() -> Settings:
    """Load supported runtime overrides from environment variables."""
    cors_raw = os.getenv("CALLINTEL_CORS_ORIGINS", "*")
    cors_origins = [origin.strip() for origin in cors_raw.split(",") if origin.strip()]
    return Settings(
        app_name=os.getenv("CALLINTEL_APP_NAME", "CallIntel AI - Call Intelligence Engine"),
        environment=os.getenv("CALLINTEL_ENVIRONMENT", "production"),
        debug=_env_bool("CALLINTEL_DEBUG", False),
        upload_dir=Path(os.getenv("CALLINTEL_UPLOAD_DIR", "data/uploads")),
        samples_dir=Path(os.getenv("CALLINTEL_SAMPLES_DIR", "data/samples")),
        reports_dir=Path(os.getenv("CALLINTEL_REPORTS_DIR", "reports")),
        database_url=os.getenv("CALLINTEL_DATABASE_URL", "sqlite:///./callintel.db"),
        whisper_model_size=os.getenv("CALLINTEL_WHISPER_MODEL_SIZE", "base"),
        whisper_device=os.getenv("CALLINTEL_WHISPER_DEVICE", "cpu"),
        whisper_compute_type=os.getenv("CALLINTEL_WHISPER_COMPUTE_TYPE", "int8"),
        max_chunk_bytes=int(os.getenv("CALLINTEL_MAX_CHUNK_BYTES", str(512 * 1024))),
        max_total_buffer_bytes=int(os.getenv("CALLINTEL_MAX_TOTAL_BUFFER_BYTES", str(10 * 1024 * 1024))),
        max_upload_size_bytes=int(os.getenv("CALLINTEL_MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024))),
        cleanup_temp_files=_env_bool("CALLINTEL_CLEANUP_TEMP_FILES", True),
        upload_retention_hours=int(os.getenv("CALLINTEL_UPLOAD_RETENTION_HOURS", "24")),
        api_key=os.getenv("CALLINTEL_API_KEY") or None,
        require_auth=_env_bool("CALLINTEL_REQUIRE_AUTH", False),
        cors_origins=cors_origins or ["*"],
        log_level=os.getenv("CALLINTEL_LOG_LEVEL", "INFO"),
    )


settings = load_settings_from_environment()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.reports_dir.mkdir(parents=True, exist_ok=True)
