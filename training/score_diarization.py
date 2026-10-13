"""Score diarization using pyannote.metrics on independently annotated RTTM files."""
from __future__ import annotations
import argparse
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Human-verified reference RTTM")
    parser.add_argument("--hypothesis", required=True, type=Path, help="System hypothesis RTTM")
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--skip-overlap", action="store_true")
    args = parser.parse_args()
    for label, path in (("reference", args.reference), ("hypothesis", args.hypothesis)):
        if not path.is_file() or path.stat().st_size == 0:
            parser.error(f"{label} RTTM missing or empty: {path}")
    if args.collar < 0:
        parser.error("--collar must be >= 0")
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        parser.error("Install optional scorer dependencies: python -m pip install -r requirements-speech-eval.txt")

    def read_rttm(path: Path) -> Annotation:
        annotation = Annotation()
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
                annotation[Segment(start, start + duration), (recording, speaker)] = speaker
        if not annotation:
            raise ValueError(f"No SPEAKER segments found in {path}")
        return annotation

    reference, hypothesis = read_rttm(args.reference), read_rttm(args.hypothesis)
    ref_ids = {track[0][0] for track in reference.itertracks(yield_label=True)}
    hyp_ids = {track[0][0] for track in hypothesis.itertracks(yield_label=True)}
    if ref_ids != hyp_ids:
        raise ValueError(f"Recording IDs differ: reference={sorted(ref_ids)}, hypothesis={sorted(hyp_ids)}")
    metric = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
    der = metric(reference, hypothesis)
    print("Diarization evaluation (pyannote.metrics)")
    print(f"Reference RTTM: {args.reference}; hypothesis RTTM: {args.hypothesis}")
    print(f"Recordings: {len(ref_ids)}; collar: {args.collar}s; skip overlap: {args.skip_overlap}")
    print(f"DER: {der:.4f} ({der * 100:.2f}%)")
    print(f"Components: {metric.compute_components(reference, hypothesis)}")
    print("Validity depends on independently verified reference annotations.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
