"""WER scorer for independently human-annotated ASR evaluation JSONL."""
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path

def words(text: str) -> list[str]:
    return re.findall(r"\b[\w']+\b", text.casefold())

def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, token in enumerate(ref, 1):
        cur = [i]
        for j, other in enumerate(hyp, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (token != other)))
        prev = cur
    return prev[-1]

def score(records: list[dict], bootstrap: int = 2000, seed: int = 42) -> dict:
    if not records:
        raise ValueError("No evaluation records")
    per_call = []
    for i, row in enumerate(records):
        if not row.get("call_id") or not isinstance(row.get("reference"), str) or not isinstance(row.get("hypothesis"), str):
            raise ValueError(f"row {i+1}: requires call_id, reference, hypothesis strings")
        ref, hyp = words(row["reference"]), words(row["hypothesis"])
        if not ref:
            raise ValueError(f"row {i+1} ({row['call_id']}): reference has no words")
        per_call.append({"call_id": str(row["call_id"]), "errors": edit_distance(ref, hyp), "reference_words": len(ref)})
    def corpus_wer(sample):
        den = sum(per_call[i]["reference_words"] for i in sample)
        return sum(per_call[i]["errors"] for i in sample) / den if den else float("nan")
    overall = corpus_wer(list(range(len(per_call))))
    ci = None
    if len(per_call) >= 2 and bootstrap > 0:
        rng = random.Random(seed)
        vals = [corpus_wer([rng.randrange(len(per_call)) for _ in per_call]) for _ in range(bootstrap)]
        vals.sort()
        ci = {"lower_95": vals[int(0.025 * (len(vals)-1))], "upper_95": vals[int(0.975 * (len(vals)-1))], "method": "percentile bootstrap resampling calls", "replicates": bootstrap, "seed": seed}
    return {"metric": "WER", "wer": overall, "wer_percent": 100*overall, "reference_words": sum(x["reference_words"] for x in per_call), "edit_errors": sum(x["errors"] for x in per_call), "calls": len(per_call), "confidence_interval_95": ci, "normalization": "casefold; Unicode word tokens; punctuation excluded", "note": "Valid only when references are independently human-verified and hypotheses were generated without reference access."}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    rows = [json.loads(line) for line in a.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(score(rows, a.bootstrap, a.seed), indent=2))
if __name__ == "__main__":
    main()
