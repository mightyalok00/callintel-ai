"""Application service layer orchestrating speech processing, ML pipelines, and persistence."""

from src.callintel.services.call_analysis_service import CallAnalysisService

__all__ = ["CallAnalysisService"]
