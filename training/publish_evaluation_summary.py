"""Publish current ML evaluation metrics to the GitHub Actions job summary."""
from __future__ import annotations
import json, os
from pathlib import Path

def main():
    report_path = Path("reports/ml_evaluation_benchmark.json")
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing generated report: {report_path}")
    data = json.loads(report_path.read_text(encoding="utf-8"))
    emotion = data.get("meld_emotion_evaluation", data.get("emotion_evaluation", {}))
    if isinstance(emotion, dict) and isinstance(emotion.get("emotion_evaluation"), dict):
        emotion = emotion["emotion_evaluation"]
    if not emotion:
        raise ValueError("Could not find held-out MELD emotion metrics in report JSON")
    lines = [
        "# CallIntel AI — held-out emotion evaluation",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Test utterances | {data.get('meld_emotion_evaluation', {}).get('dataset_split', {}).get('test_turns', 'see JSON artifact')} |",
        f"| Accuracy | {emotion.get('accuracy', 'n/a')} |",
        f"| Majority baseline accuracy | {emotion.get('majority_baseline_accuracy', 'n/a')} |",
        f"| Macro-F1 | {emotion.get('macro_f1', 'n/a')} |",
        f"| Majority baseline macro-F1 | {emotion.get('majority_baseline_macro_f1', 'n/a')} |",
        f"| Macro-F1 lift | {emotion.get('macro_f1_lift_over_majority', 'n/a')} |",
        f"| 95% dialogue-bootstrap CI | {emotion.get('macro_f1_uncertainty', {})} |",
        "",
        "## Per-class F1",
        "",
        "| Emotion | Support | Precision | Recall | F1 | Majority baseline F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, m in sorted(emotion.get("per_class", {}).items()):
        lines.append(f"| {label} | {m.get('support','')} | {m.get('precision','')} | {m.get('recall','')} | {m.get('f1','')} | {m.get('majority_baseline_f1','')} |")
    lines += ["", "> MELD is scripted television dialogue. These results are not evidence of real-call emotion accuracy. See the uploaded JSON/Markdown artifacts for complete results."]
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
