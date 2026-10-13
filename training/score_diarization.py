"""Score diarization against independently annotated RTTM with recording-level uncertainty."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

def read_rttm(path: Path):
    from pyannote.core import Annotation, Segment
    annotations = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 8 or fields[0] != "SPEAKER":
                raise ValueError(f"{path}:{line_number}: malformed RTTM SPEAKER row")
            recording, start, duration, speaker = fields[1], float(fields[3]), float(fields[4]), fields[7]
            if start < 0 or duration <= 0:
                raise ValueError(f"{path}:{line_number}: invalid segment bounds")
            annotation = annotations.setdefault(recording, Annotation(uri=recording))
            annotation[Segment(start, start + duration), speaker] = speaker
    if not annotations:
        raise ValueError(f"No SPEAKER segments found in {path}")
    return annotations

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Human-verified reference RTTM")
    parser.add_argument("--hypothesis", required=True, type=Path, help="System hypothesis RTTM")
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--skip-overlap", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for label, path in (("reference", args.reference), ("hypothesis", args.hypothesis)):
        if not path.is_file() or path.stat().st_size == 0:
            parser.error(f"{label} RTTM missing or empty: {path}")
    if args.collar < 0 or args.bootstrap < 0:
        parser.error("--collar and --bootstrap must be >= 0")
    try:
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        parser.error("Install optional scorer dependencies: python -m pip install -r requirements-speech-eval.txt")
    reference, hypothesis = read_rttm(args.reference), read_rttm(args.hypothesis)
    if set(reference) != set(hypothesis):
        raise ValueError(f"Recording IDs differ: reference={sorted(reference)}, hypothesis={sorted(hypothesis)}")
    metric = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
    per_recording = []
    for recording in sorted(reference):
        value = float(metric(reference[recording], hypothesis[recording]))
        per_recording.append({"recording_id": recording, "der": value})
    # Report macro-average recording DER. The CI uses the same estimand and
    # resamples whole recordings, preserving within-recording dependence.
    overall = sum(row["der"] for row in per_recording) / len(per_recording)
    ci = None
    if len(per_recording) >= 2 and args.bootstrap:
        vals = [row["der"] for row in per_recording]
        rng = random.Random(args.seed)
        samples = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals) for _ in range(args.bootstrap))
        ci = {"lower_95": samples[int(0.025 * (len(samples)-1))], "upper_95": samples[int(0.975 * (len(samples)-1))], "method": "percentile bootstrap resampling recordings; macro-average DER", "replicates": args.bootstrap, "seed": args.seed}
    result = {
        "metric": "DER", "macro_der": overall, "macro_der_percent": overall * 100,
        "recordings": len(per_recording), "collar_seconds": args.collar, "skip_overlap": args.skip_overlap,
        "per_recording_macro_der": sum(x["der"] for x in per_recording) / len(per_recording),
        "macro_der_confidence_interval_95": ci, "per_recording": per_recording,
        "note": "Requires independently human-verified reference RTTM. Both DER and CI use macro-average recording DER; this is not duration-weighted micro DER."
    }
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
