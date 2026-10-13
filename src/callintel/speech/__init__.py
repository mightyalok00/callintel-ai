"""Speech processing modules: Whisper ASR transcription and acoustic diarization."""

from src.callintel.speech.transcriber import WhisperTranscriber
from src.callintel.speech.diarizer import AcousticDiarizer

__all__ = ["WhisperTranscriber", "AcousticDiarizer"]
