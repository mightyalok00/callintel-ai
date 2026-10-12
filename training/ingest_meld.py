"""
Dataset ingestion and verification script for MELD (Multimodal EmotionLines Dataset).
Default source directory is relative to workspace: data/meld_raw or env var MELD_DATA_DIR.
Bundled processed files are in data/meld/.
"""

import os
import csv
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MELD_SOURCE_DIR = Path(os.getenv("MELD_DATA_DIR", PROJECT_ROOT / "data" / "meld_raw"))
DEST_DIR = Path(os.getenv("DEST_MELD_DIR", PROJECT_ROOT / "data" / "meld"))

def inspect_and_ingest():
    # If already ingested and bundled in repo, verify and summarize
    bundled_train = DEST_DIR / "meld_train_dialogues.json"
    if not MELD_SOURCE_DIR.exists() and bundled_train.exists():
        print(f"[+] Processed MELD dataset already bundled in repository at: {DEST_DIR}")
        print("    No external ingestion required. To re-ingest from raw CSVs, set MELD_DATA_DIR.")
        return

    print(f"[*] Checking MELD source directory at: {MELD_SOURCE_DIR}")
    if not MELD_SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source directory not found: {MELD_SOURCE_DIR}. "
            "Download raw MELD CSV files (train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv) "
            "from https://github.com/declare-lab/MELD and place them in data/meld_raw/ "
            "or set the MELD_DATA_DIR environment variable."
        )
    
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    splits = {
        "train": MELD_SOURCE_DIR / "train_sent_emo.csv",
        "dev": MELD_SOURCE_DIR / "dev_sent_emo.csv",
        "test": MELD_SOURCE_DIR / "test_sent_emo.csv",
    }
    
    summary_stats = {}
    
    for split_name, filepath in splits.items():
        if not filepath.exists():
            print(f"[!] Warning: Missing {filepath}")
            continue
            
        print(f"[*] Processing {split_name} from {filepath}...")
        
        dialogues = {}
        emotions = Counter()
        sentiments = Counter()
        total_rows = 0
        
        with open(filepath, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                d_id = row.get("Dialogue_ID", "0")
                emo = row.get("Emotion", "neutral").lower()
                sent = row.get("Sentiment", "neutral").lower()
                
                emotions[emo] += 1
                sentiments[sent] += 1
                
                if d_id not in dialogues:
                    dialogues[d_id] = []
                    
                dialogues[d_id].append({
                    "utterance_id": int(row.get("Utterance_ID", 0)),
                    "speaker": row.get("Speaker", "Unknown"),
                    "utterance": row.get("Utterance", ""),
                    "emotion": emo,
                    "sentiment": sent,
                    "start_time": row.get("StartTime", ""),
                    "end_time": row.get("EndTime", "")
                })
        
        summary_stats[split_name] = {
            "total_utterances": total_rows,
            "total_dialogues": len(dialogues),
            "emotions": dict(emotions),
            "sentiments": dict(sentiments)
        }
        
        # Save structured JSON for direct call intelligence pipeline ingestion
        output_json = DEST_DIR / f"meld_{split_name}_dialogues.json"
        with open(output_json, "w", encoding="utf-8") as out_f:
            json.dump(dialogues, out_f, indent=2)
            
        print(f"[+] Saved {len(dialogues)} structured dialogues ({total_rows} turns) to {output_json}")

    meta_file = DEST_DIR / "meld_summary_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
        
    print(f"\n[OK] Ingestion complete. Summary written to {meta_file}")
    print(json.dumps(summary_stats, indent=2))

if __name__ == "__main__":
    inspect_and_ingest()
