"""
Database models and storage schemas for Call Intelligence Engine.
Supports SQLite / PostgreSQL with SQLAlchemy 2.0 typed mappings.
"""

from datetime import datetime, timezone
from typing import Optional, Any
from sqlalchemy import (
    String,
    Float,
    DateTime,
    JSON,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class CallRecord(Base):
    """Stores metadata, transcripts, and intelligence reports for a processed call."""
    __tablename__ = "call_records"

    call_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)  # PENDING, PROCESSING, COMPLETED, FAILED
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    language: Mapped[str] = mapped_column(String(16), default="en")

    # Intelligence Metrics
    overall_sentiment: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    churn_risk_level: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    churn_risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    primary_intent: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    qa_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    qa_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # Structured JSON Payloads
    transcript_turns: Mapped[Any] = mapped_column(JSON, default=list)
    evidence_ledger: Mapped[Any] = mapped_column(JSON, default=list)
    recommendations: Mapped[Any] = mapped_column(JSON, default=list)
    metadata_payload: Mapped[Any] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


DATABASE_URL = "sqlite:///./callintel.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)
