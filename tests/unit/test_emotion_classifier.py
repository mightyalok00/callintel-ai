"""
Unit tests for the MELD Emotion and Sentiment Classifier.
"""

import pytest
from pathlib import Path
from src.callintel.ml.meld_classifier import MeldClassifier


def test_meld_classifier_loading():
    clf = MeldClassifier()
    assert clf.is_ready, "MeldClassifier models failed to load."


def test_single_turn_prediction():
    clf = MeldClassifier()
    res = clf.predict_turn(
        utterance="Your product is good, but it is too expensive.",
        speaker="Customer",
        prev_speaker="Agent",
        prev_utterance="How are things going?",
    )
    assert "emotion" in res
    assert "sentiment" in res
    assert "label" in res["emotion"]
    assert "confidence" in res["emotion"]
    assert 0.0 <= res["emotion"]["confidence"] <= 1.0
    assert 0.0 <= res["sentiment"]["confidence"] <= 1.0


def test_timeline_analysis():
    clf = MeldClassifier()
    turns = [
        {"turn_index": 0, "speaker": "Agent", "utterance": "Thank you for calling support. How can I help?"},
        {"turn_index": 1, "speaker": "Customer", "utterance": "I have a billing issue with my invoice."},
        {"turn_index": 2, "speaker": "Customer", "utterance": "This is unacceptable, I was charged twice!"},
    ]
    timeline = clf.analyze_call_timeline(turns)
    assert len(timeline) == 3
    assert timeline[2]["sentiment"]["label"] == "negative" or timeline[2]["emotion"]["label"] in ["anger", "sadness", "neutral"]
