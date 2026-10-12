"""
Unit tests for the Acoustic Speaker Diarizer.
Verifies acoustic feature extraction, pitch estimation, speaker clustering,
and calibrated speaker confidence without conversational pause heuristics.
"""

from pathlib import Path
import numpy as np
import pytest

from src.callintel.speech.audio_utils import load_and_normalize_audio
from src.callintel.speech.diarizer import AcousticDiarizer

SAMPLE_WAV = Path("data/samples/sample_customer_call.wav")


def test_acoustic_feature_extraction():
    diarizer = AcousticDiarizer(target_sample_rate=16000)

    # Synthetic tone: 150 Hz sine wave (approx male voice pitch)
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    synthetic_male = (0.5 * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32)

    feat_male = diarizer.extract_segment_features(synthetic_male)
    assert feat_male["embedding"].shape == (34,)
    assert feat_male["pitch_mean"] > 130.0 and feat_male["pitch_mean"] < 170.0
    assert feat_male["voiced_ratio"] > 0.5
    assert feat_male["spectral_centroid"] > 0.0

    # Synthetic tone: 220 Hz sine wave (approx female voice pitch)
    synthetic_female = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    feat_female = diarizer.extract_segment_features(synthetic_female)
    assert feat_female["pitch_mean"] > 200.0 and feat_female["pitch_mean"] < 240.0
    assert feat_female["pitch_mean"] > feat_male["pitch_mean"]


def test_diarize_segments_multi_speaker():
    assert SAMPLE_WAV.exists(), "Sample WAV file missing."
    audio, sr, dur = load_and_normalize_audio(SAMPLE_WAV)

    diarizer = AcousticDiarizer(target_sample_rate=sr)

    # Define test turns spanning the audio
    mock_segments = [
        {"start_seconds": 0.0, "end_seconds": 3.5, "start_time": "00:00:00.000", "end_time": "00:00:03.500", "utterance": "Hello, thank you for calling Apex Cloud Support. My name is Alex."},
        {"start_seconds": 7.5, "end_seconds": 11.5, "start_time": "00:00:07.500", "end_time": "00:00:11.500", "utterance": "Hi Alex. Your product is good, but it is too expensive."},
        {"start_seconds": 17.0, "end_seconds": 21.0, "start_time": "00:00:17.000", "end_time": "00:00:21.000", "utterance": "I completely understand your pricing concerns."},
        {"start_seconds": 27.0, "end_seconds": 30.5, "start_time": "00:00:27.000", "end_time": "00:00:30.500", "utterance": "Okay, please do. If you have an annual discount, I would like to consider staying."},
    ]

    diarized = diarizer.diarize_segments(audio, mock_segments, expected_speakers=2)

    assert len(diarized) == 4
    speakers = {t["speaker"] for t in diarized}
    assert "Agent" in speakers
    assert "Customer" in speakers

    clusters = {t["speaker_cluster_id"] for t in diarized}
    assert len(clusters) == 2

    # Check acoustic metrics and confidence
    for turn in diarized:
        assert turn["attribution_method"] == "acoustic_voice_clustering"
        assert 0.70 <= turn["attribution_confidence"] <= 0.99
        assert "mean_pitch_hz" in turn["acoustic_metrics"]
        assert "spectral_centroid_hz" in turn["acoustic_metrics"]


def test_single_speaker_handling():
    diarizer = AcousticDiarizer(target_sample_rate=16000)
    audio = np.random.randn(16000 * 2).astype(np.float32) * 0.1

    mock_segments = [
        {"start_seconds": 0.0, "end_seconds": 1.5, "start_time": "00:00:00.000", "end_time": "00:00:01.500", "utterance": "Testing monologue."},
    ]

    diarized = diarizer.diarize_segments(audio, mock_segments)
    assert len(diarized) == 1
    assert diarized[0]["speaker"] == "Agent"
    assert diarized[0]["speaker_cluster_id"] == "SPEAKER_00"
