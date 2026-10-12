"""
Database models and storage schemas for Call Intelligence Engine.
Supports SQLite / PostgreSQL with SQLAlchemy.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    DateTime,
    JSON,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class CallRecord(Base):
    """Stores metadata, transcripts, and intelligence reports for a processed call."""
    __tablename__ = "call_records"

    call_id = Column(String(64), primary_key=True, index=True)
    filename = Column(String(256), nullable=False)
    status = Column(String(32), default="PENDING", index=True)  # PENDING, PROCESSING, COMPLETED, FAILED
    duration_seconds = Column(Float, nullable=True)
    language = Column(String(16), default="en")
    
    # Intelligence Metrics
    overall_sentiment = Column(String(32), nullable=True)
    sentiment_score = Column(Float, nullable=True)
    churn_risk_level = Column(String(32), nullable=True)
    churn_risk_score = Column(Float, nullable=True)
    primary_intent = Column(String(64), nullable=True)
    qa_score = Column(Float, nullable=True)
    qa_status = Column(String(32), nullable=True)
    
    # Structured JSON Payloads
    transcript_turns = Column(JSON, default=list)
    evidence_ledger = Column(JSON, default=list)
    recommendations = Column(JSON, default=list)
    metadata_payload = Column(JSON, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


DATABASE_URL = "sqlite:///./callintel.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)
