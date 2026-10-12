"""
Dependency Injection providers for FastAPI.
Provides managed, shared instances of database sessions, ML models, and application services.
"""

from typing import Generator, Optional
from functools import lru_cache
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from src.callintel.db.models import SessionLocal
from src.callintel.db.repository import CallRepository
from src.callintel.speech.transcriber import WhisperTranscriber
from src.callintel.ml.meld_classifier import MeldClassifier
from src.callintel.ml.churn_model import ChurnRiskPredictor
from src.callintel.agents.multi_agents import MultiAgentSupervisor
from src.callintel.services.call_analysis_service import CallAnalysisService


def get_db() -> Generator[Session, None, None]:
    """Yields a database session and guarantees closure upon request completion."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_call_repository(db: Session = Depends(get_db)) -> CallRepository:
    """Provides a request-scoped CallRepository with active session."""
    return CallRepository(db=db)


@lru_cache(maxsize=1)
def get_transcriber() -> WhisperTranscriber:
    """Provides cached singleton WhisperTranscriber."""
    return WhisperTranscriber(model_size="base")


@lru_cache(maxsize=1)
def get_meld_classifier() -> MeldClassifier:
    """Provides cached singleton MeldClassifier."""
    return MeldClassifier()


@lru_cache(maxsize=1)
def get_churn_predictor() -> ChurnRiskPredictor:
    """Provides cached singleton ChurnRiskPredictor."""
    return ChurnRiskPredictor()


@lru_cache(maxsize=1)
def get_multi_agent_supervisor() -> MultiAgentSupervisor:
    """Provides cached singleton MultiAgentSupervisor."""
    return MultiAgentSupervisor()


def get_call_analysis_service(
    repo: CallRepository = Depends(get_call_repository),
    transcriber: WhisperTranscriber = Depends(get_transcriber),
    meld: MeldClassifier = Depends(get_meld_classifier),
    churn_predictor: ChurnRiskPredictor = Depends(get_churn_predictor),
    supervisor: MultiAgentSupervisor = Depends(get_multi_agent_supervisor),
) -> CallAnalysisService:
    """Constructs application service with fully injected domain and infrastructure components."""
    return CallAnalysisService(
        repo=repo,
        transcriber=transcriber,
        meld=meld,
        churn_predictor=churn_predictor,
        supervisor=supervisor,
    )


def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Optional[str]:
    """
    Validates API Key if authentication is enabled in configuration.
    Permits unauthenticated requests when require_auth is False and api_key is None.
    Supports either 'X-API-Key' header or 'Authorization: Bearer <token>'.
    """
    from src.callintel.core.config import settings
    if settings.require_auth or settings.api_key:
        expected = settings.api_key
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication is enabled but no API key is configured.",
            )
        token = x_api_key
        if not token and authorization:
            if authorization.startswith("Bearer "):
                token = authorization[7:].strip()
            else:
                token = authorization.strip()

        if not token or token != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Missing or invalid API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return token
    return x_api_key
