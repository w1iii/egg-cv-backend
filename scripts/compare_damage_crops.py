"""Compare two damage checkpoints on labeled crops; never trains or tunes.

Manifest needs filename and label_name columns. This evaluates supplied crops,
not detector behavior or hidden shell surfaces. Labels must be verified separately.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.damage_inference import DamageClassifier  # noqa: E402
from app.ml.preprocessing import file_sha256  # noqa: E402


def compare(rows, root, baseline, candidate):
    results = []
    for row in rows:
        label = row.get("label_name", "").strip()
        if label not in ("Damaged", "Not Damaged"):
            raise ValueError(f"Invalid label: {label!r}")
        filename = row.get("filename", "").strip()
        if not filename:
            raise ValueError("Missing filename")
        image = cv2.imread(str(root / filename))
        if image is None:
            raise ValueError(f"Cannot read image: {filename}")
        old, new = baseline.classify_crop(image), candidate.classify_crop(image)
        result = dict(row)
        for prefix, prediction in (("baseline", old), ("candidate", new)):
            result[f"{prefix}_status"] = prediction.status
            result[f"{prefix}_damage_probability"] = prediction.probability
            result[f"{prefix}_review_reason"] = prediction.review_reason
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--review-threshold", type=float, default=0.75)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (0 < args.threshold < 1 and 0 < args.review_threshold <= 1):
        parser.error("thresholds must be between 0 and 1")
    if args.out.exists():
        parser.error("output already exists; refusing to overwrite")
    classifiers = [DamageClassifier(checkpoint_path=str(path), damaged_threshold=args.threshold,
                                    review_threshold=args.review_threshold, device="cpu")
                   for path in (args.baseline, args.candidate)]
    for path, classifier in zip((args.baseline, args.candidate), classifiers):
        if not classifier.load_model():
            raise SystemExit(f"Could not load checkpoint: {path}")
        print(f"{path.name}: {file_sha256(path)}")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"filename", "label_name"}.issubset(reader.fieldnames or []):
            parser.error("manifest requires filename and label_name columns")
        rows = list(reader)
    if not rows:
        parser.error("manifest is empty")
    results = compare(rows, args.data_root, *classifiers)
    with args.out.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} comparisons to {args.out}; threshold={args.threshold}")


if __name__ == "__main__":
    main()
