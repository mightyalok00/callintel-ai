import json
from pathlib import Path
import pytest
import training.validate_meld_splits as validator
from training.score_asr import score

def _turn(text="hello", speaker="A", emotion="neutral", utterance_id=0):
    return {"utterance_id": utterance_id, "speaker": speaker, "utterance": text, "emotion": emotion}

def test_split_local_dialogue_ids_are_not_compared_across_splits(tmp_path, monkeypatch):
    data = tmp_path / "meld"
    raw = tmp_path / "raw"
    data.mkdir(); raw.mkdir()
    monkeypatch.setattr(validator, "EXPECTED", {s: {"utterances": 1, "dialogues": 1} for s in ("train", "dev", "test")})
    for split in ("train", "dev", "test"):
        (data / f"meld_{split}_dialogues.json").write_text(json.dumps({"0": [_turn(text=split)]}), encoding="utf-8")
    result = validator.validate_meld_splits(data, raw)
    assert result["splits"]["train"]["dialogues"] == 1
    assert result["raw_csv_provenance"].startswith("partial")

def test_validator_fails_closed_on_missing_official_partition(tmp_path):
    data = tmp_path / "meld"; data.mkdir()
    with pytest.raises(ValueError, match="Missing official train partition"):
        validator.validate_meld_splits(data)

def test_validator_detects_exact_duplicate_dialogue_content_across_splits(tmp_path, monkeypatch):
    data = tmp_path / "meld"; raw = tmp_path / "raw"; data.mkdir(); raw.mkdir()
    monkeypatch.setattr(validator, "EXPECTED", {"train": {"utterances": 3, "dialogues": 1}, "dev": {"utterances": 3, "dialogues": 1}, "test": {"utterances": 1, "dialogues": 1}})
    duplicate = [_turn("hello", utterance_id=0), _turn("there", utterance_id=1), _turn("again", utterance_id=2)]
    (data / "meld_train_dialogues.json").write_text(json.dumps({"0": duplicate}), encoding="utf-8")
    (data / "meld_dev_dialogues.json").write_text(json.dumps({"0": duplicate}), encoding="utf-8")
    (data / "meld_test_dialogues.json").write_text(json.dumps({"0": [_turn("different")]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Exact duplicate multi-turn dialogue"):
        validator.validate_meld_splits(data, raw)

def test_asr_scorer_reports_call_level_ci_and_rejects_empty_reference():
    result = score([
        {"call_id": "a", "reference": "hello world", "hypothesis": "hello world"},
        {"call_id": "b", "reference": "good morning", "hypothesis": "good evening"},
    ], bootstrap=100, seed=3)
    assert result["wer"] == 0.25
    assert result["confidence_interval_95"] is not None
    with pytest.raises(ValueError, match="reference has no words"):
        score([{"call_id": "x", "reference": "!!!", "hypothesis": ""}])
