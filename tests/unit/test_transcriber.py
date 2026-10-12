"""
Unit tests for the faster-whisper transcription service.
"""

from pathlib import Path
import pytest
from src.callintel.speech.transcriber import WhisperTranscriber

SAMPLE_WAV = Path("data/samples/sample_customer_call.wav")


def test_whisper_transcription():
    assert SAMPLE_WAV.exists(), "Sample WAV file missing."

    transcriber = WhisperTranscriber(model_size="base", device="cpu", compute_type="int8")
    result = transcriber.transcribe_file(SAMPLE_WAV)

    assert "turns" in result
    assert result["total_turns"] > 0
    assert result["duration_seconds"] > 0

    first_turn = result["turns"][0]
    assert "utterance" in first_turn
    assert "start_time" in first_turn
    assert "end_time" in first_turn
    assert first_turn["speaker"] in ["Agent", "Customer"]
