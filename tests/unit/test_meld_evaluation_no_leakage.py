"""Regression guard: MELD evaluation must not derive features from target labels."""

import inspect

from training.evaluate_ml_models import evaluate_meld_models


def test_meld_evaluation_does_not_synthesize_acoustics_from_ground_truth():
    source = inspect.getsource(evaluate_meld_models)
    assert "mock_pitch" not in source
    assert "mock_centroid" not in source
    assert "mock_voiced" not in source
    assert "predict_turn_multimodal" not in source


def test_meld_evaluation_discloses_text_only_mode():
    source = inspect.getsource(evaluate_meld_models)
    assert "text_only_no_label_derived_acoustic_features" in source
    assert "No aligned audio features are loaded" in source
