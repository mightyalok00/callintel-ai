"""Validate MELD split structure and, when raw CSVs exist, processed/raw provenance.

Dialogue_ID is split-local in MELD; never compare raw IDs across partitions.
"""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "train": {"utterances": 9989, "dialogues": 1038},
    "dev": {"utterances": 1109, "dialogues": 114},
    "test": {"utterances": 2610, "dialogues": 280},
}
EMOTIONS = {"neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"}

def _canonical(turn):
    return (
        str(turn.get("utterance_id", turn.get("Utterance_ID", ""))),
        str(turn.get("speaker", turn.get("Speaker", ""))).strip(),
        str(turn.get("utterance", turn.get("Utterance", ""))).strip(),
        str(turn.get("emotion", turn.get("Emotion", ""))).strip().lower(),
    )

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def validate_meld_splits(data_dir: Path, raw_dir: Path | None = None) -> dict:
    data_dir = Path(data_dir)
    raw_dir = Path(raw_dir) if raw_dir else data_dir.parent / "meld_raw"
    splits = {}
    exact_dialogue_hashes = {}
    result = {"validator": "meld_split_integrity_v1", "checks": [], "splits": {}, "raw_csv_provenance": "not_checked"}
    for split, expected in EXPECTED.items():
        path = data_dir / f"meld_{split}_dialogues.json"
        if not path.is_file():
            raise ValueError(f"Missing official {split} partition file: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data:
            raise ValueError(f"{split}: expected non-empty dialogue mapping")
        utterance_count = 0
        canonical_dialogues = {}
        emotion_counts = {}
        for dialogue_id, turns in data.items():
            if not isinstance(turns, list) or not turns:
                raise ValueError(f"{split}/{dialogue_id}: dialogue must have non-empty turn list")
            seen_turn_ids = set()
            normalized = []
            for turn in turns:
                if not isinstance(turn, dict):
                    raise ValueError(f"{split}/{dialogue_id}: malformed turn")
                turn_id, speaker, text, emotion = _canonical(turn)
                if not speaker or not text or emotion not in EMOTIONS:
                    raise ValueError(f"{split}/{dialogue_id}: missing speaker/text or unknown emotion {emotion!r}")
                if turn_id in seen_turn_ids:
                    raise ValueError(f"{split}/{dialogue_id}: duplicate utterance_id {turn_id}")
                seen_turn_ids.add(turn_id)
                normalized.append((turn_id, speaker, text, emotion))
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                utterance_count += 1
            canonical_dialogues[str(dialogue_id)] = normalized
            signature = hashlib.sha256(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            # One- and two-turn utterances often repeat naturally ("yes", "yeah").
            # Only flag exact multi-turn duplicate sequences; this is a diagnostic,
            # not a substitute for comparing the official source partitions.
            if len(normalized) >= 3:
                if signature in exact_dialogue_hashes:
                    other_split, other_id = exact_dialogue_hashes[signature]
                    raise ValueError(f"Exact duplicate multi-turn dialogue across splits: {other_split}/{other_id} and {split}/{dialogue_id}")
                exact_dialogue_hashes[signature] = (split, str(dialogue_id))
        if len(data) != expected["dialogues"] or utterance_count != expected["utterances"]:
            raise ValueError(f"{split}: expected {expected['dialogues']} dialogues/{expected['utterances']} utterances; got {len(data)}/{utterance_count}")
        splits[split] = {"dialogues": len(data), "utterances": utterance_count, "sha256": _sha256(path), "emotion_counts": dict(sorted(emotion_counts.items()))}
        raw_path = raw_dir / f"{split}_sent_emo.csv"
        if raw_path.is_file():
            with raw_path.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            raw_grouped = {}
            for row in rows:
                raw_grouped.setdefault(str(row["Dialogue_ID"]), []).append(_canonical(row))
            if len(rows) != utterance_count or set(raw_grouped) != set(canonical_dialogues):
                raise ValueError(f"{split}: raw CSV partition does not match processed JSON IDs/counts")
            for did, turns in raw_grouped.items():
                if turns != canonical_dialogues[did]:
                    raise ValueError(f"{split}/{did}: raw CSV turns differ from processed JSON")
        result["checks"].append(f"{split}: expected counts and schema passed; dialogue IDs treated as split-local")
    raw_paths = [raw_dir / f"{s}_sent_emo.csv" for s in EXPECTED]
    if all(p.is_file() for p in raw_paths):
        result["raw_csv_provenance"] = "verified: all processed partitions match raw split CSV rows"
    else:
        result["raw_csv_provenance"] = "partial/unavailable: raw official CSVs not all present; structural checks and exact-content duplicate check only"
    result["splits"] = splits
    result["scope_note"] = "Structural consistency is not cryptographic proof of upstream origin. To verify source provenance, provide the original official train/dev/test CSVs; cross-split raw Dialogue_ID overlap is not a valid leakage test."
    return result

def main():
    result = validate_meld_splits(ROOT / "data" / "meld")
    print(json.dumps(result, indent=2))
    if result["raw_csv_provenance"].startswith("partial"):
        print("[INFO] " + result["raw_csv_provenance"])

if __name__ == "__main__":
    main()
