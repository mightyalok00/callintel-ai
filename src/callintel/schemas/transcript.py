"""
Data contracts and Pydantic schemas for conversation transcripts, turns, and speakers.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SpeakerTurn(BaseModel):
    turn_index: int = Field(..., description="Zero-indexed chronological turn number")
    speaker: str = Field(..., description="Speaker identifier e.g. Customer, Agent, Chandler")
    utterance: str = Field(..., description="Spoken text content")
    start_time: Optional[str] = Field(None, description="HH:MM:SS.mmm start timestamp")
    end_time: Optional[str] = Field(None, description="HH:MM:SS.mmm end timestamp")
    start_seconds: Optional[float] = Field(None, description="Audio offset in seconds")
    end_seconds: Optional[float] = Field(None, description="Audio offset in seconds")


class EnrichedTurn(SpeakerTurn):
    emotion_label: Optional[str] = None
    emotion_confidence: Optional[float] = None
    sentiment_label: Optional[str] = None
    sentiment_confidence: Optional[float] = None
    intent_label: Optional[str] = None
    intent_confidence: Optional[float] = None
    is_complaint_or_objection: bool = False


class CallTranscript(BaseModel):
    call_id: str
    total_turns: int
    duration_seconds: Optional[float] = None
    speakers: List[str] = Field(default_factory=list)
    turns: List[EnrichedTurn] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
