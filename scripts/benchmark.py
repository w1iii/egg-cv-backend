"""Reproducible benchmark harness for the egg inspection pipeline.

Each mode evaluates one component on a frozen, labeled set and writes a JSON
report (plus Markdown) that records the manifest hash, code revision,
effective thresholds and checkpoint hashes, so before/after runs can be
compared on identical inputs.

Modes
-----
detector    YOLO-format images + labels -> detection P/R/F1, count errors.
classifier  Crop manifest (path,label) -> damage or cleanliness metrics
            through the app's own classifier wrapper and thresholds.
end-to-end  Full images with per-egg boxes and labels -> decision confusion,
            review rate, and (with physical ground truth) sizing error.
            See docs/evaluation/end_to_end_manifest_template.csv.

``--backend-root`` imports the app from another checkout (e.g. the
pre-change ``main``) so the same harness measures both versions.

Nothing here makes accuracy claims by itself: a report is only as
independent as the manifest it is run on, and the report says which
manifest that was.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_bench_metrics", HERE.parent / "src" / "evaluation_metrics.py")
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)

CROP_REFUSALS = {"empty_crop", "invalid_crop", "small_crop"}


WORKSPACE = HERE.parents[1]


def display_path(path) -> str:
    """Workspace-relative path for reports, so they carry no machine paths."""
    resolved = Path(path).resolve()
    for base, label in ((WORKSPACE, "<workspace>"),):
        try:
            return f"{label}/{resolved.relative_to(base).as_posix()}"
        except ValueError:
            pass
    return f"<external>/{resolved.name}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> dict[str, Any]:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None}


def import_app(backend_root: Path):
    os.environ.setdefault("PROJECT_ROOT", str(backend_root))
    os.environ.setdefault("DATABASE_URL", "sqlite://")
    sys.path.insert(0, str(backend_root))
    from app.config import settings  # noqa: E402

    return settings


def environment(backend_root: Path, settings) -> dict[str, Any]:
    info: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend_root": display_path(backend_root),
        "code": git_revision(backend_root),
    }
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:
        pass
    keys = ["CONFIDENCE_THRESHOLD", "DAMAGE_DECISION_THRESHOLD", "DAMAGE_REVIEW_THRESHOLD",
            "CLEANLINESS_DECISION_THRESHOLD", "CLEANLINESS_REVIEW_THRESHOLD",
            "CLEANLINESS_RELEASE_APPROVED", "ARUCO_GRADES_APPROVED", "DETECTION_CROP_PADDING",
            "CLEANLINESS_CROP_PADDING", "DETECTION_DUPLICATE_IOU", "MAX_TRAY_EGGS"]
    info["settings"] = {key: getattr(settings, key) for key in keys if hasattr(settings, key)}
    info["checkpoints"] = {}
    for key in ("MODEL_PATH", "DAMAGE_MODEL_PATH", "CLEANLINESS_MODEL_PATH"):
        path = Path(getattr(settings, key))
        info["checkpoints"][key] = {"file": path.name, "sha256": sha256_file(path) if path.is_file() else None}
    return info


# --------------------------------------------------------------------- detector

def read_yolo_labels(path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
    return boxes


def run_detector(args, settings) -> dict[str, Any]:
    import cv2
    from app.ml.yolo_inference import YOLOPredictor

    predictor = YOLOPredictor()
    if not predictor.load_model():
        raise SystemExit("detector checkpoint could not be loaded")
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    tp = fp = fn = exact = 0
    abs_count_error = []
    latencies = []
    per_image = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        truth = read_yolo_labels(Path(args.labels) / f"{image_path.stem}.txt", width, height)
        start = time.perf_counter()
        result = predictor.predict_frame(image, confidence=args.confidence)
        latencies.append(time.perf_counter() - start)
        predictions = [(det["bbox"], det["confidence"]) for det in result["detections"]]
        matches, missed, extra = metrics.match_boxes(truth, predictions, args.iou)
        tp, fp, fn = tp + len(matches), fp + len(extra), fn + len(missed)
        exact += len(predictions) == len(truth)
        abs_count_error.append(abs(len(predictions) - len(truth)))
        per_image.append({"image": image_path.name, "truth": len(truth), "predicted": len(predictions),
                          "matched": len(matches), "missed": len(missed), "false_positives": len(extra),
                          "duplicates_removed": result.get("duplicate_boxes_removed", 0),
                          "invalid_removed": result.get("invalid_boxes_removed")})
    return {
        "mode": "detector",
        "inputs": {"images": display_path(args.images), "labels": display_path(args.labels), "image_count": len(images),
                   "confidence": args.confidence, "iou": args.iou,
                   "manifest_sha256": hashlib.sha256("".join(
                       sha256_file(p) + (sha256_file(Path(args.labels) / f"{p.stem}.txt")
                                         if (Path(args.labels) / f"{p.stem}.txt").is_file() else "-")
                       for p in images).encode()).hexdigest()},
        "metrics": {**metrics.detection_metrics(tp, fp, fn),
                    "exact_count_accuracy": metrics.safe_div(exact, len(images)),
                    "mean_abs_count_error": metrics.safe_div(sum(abs_count_error), len(images)),
                    "latency": metrics.latency_summary(latencies)},
        "per_image": per_image,
    }


# ------------------------------------------------------------------- classifier

TASKS = {
    "damage": ("Damaged", "Not Damaged", ("Damaged", "Not Damaged")),
    "cleanliness": ("Stained", "Clean", ("Clean", "Stained")),
}


def positive_probability(result, positive: str) -> Optional[float]:
    probability = getattr(result, "probability", None)
    if probability is not None:
        return probability
    if result.review_reason in CROP_REFUSALS:
        return None
    return result.confidence if result.status == positive else 1.0 - result.confidence


def run_classifier(args, settings) -> dict[str, Any]:
    import cv2

    positive, negative, class_order = TASKS[args.task]
    if args.task == "damage":
        from app.ml.damage_inference import DamageClassifier as Classifier

        clf = Classifier(damaged_threshold=settings.DAMAGE_DECISION_THRESHOLD,
                         review_threshold=settings.DAMAGE_REVIEW_THRESHOLD)
    else:
        from app.ml.cleanliness_inference import CleanlinessClassifier as Classifier

        clf = Classifier(stained_threshold=settings.CLEANLINESS_DECISION_THRESHOLD,
                         review_threshold=settings.CLEANLINESS_REVIEW_THRESHOLD)
    if not clf.load_model():
        raise SystemExit(f"{args.task} checkpoint could not be loaded")

    manifest = Path(args.manifest)
    root = Path(args.data_root) if args.data_root else manifest.parent
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    truth, predicted, probabilities, latencies, missing = [], [], [], [], 0
    for row in rows:
        path = root / row["filename"]
        image = cv2.imread(str(path))
        if image is None:
            missing += 1
            continue
        label = row.get("label_name") or class_order[int(row["label"])]
        start = time.perf_counter()
        result = clf.classify_crop(image)
        latencies.append(time.perf_counter() - start)
        truth.append(label)
        predicted.append(result.status)
        probabilities.append(positive_probability(result, positive))

    report = {
        "mode": "classifier",
        "task": args.task,
        "inputs": {"manifest": display_path(manifest), "manifest_sha256": sha256_file(manifest), "rows": len(rows),
                   "unreadable": missing},
        "app_policy": {
            "description": "Statuses produced by the app wrapper at the app's configured thresholds, including Needs Review.",
            "decision_threshold": getattr(clf, "damaged_threshold", getattr(clf, "stained_threshold", None)),
            "review_threshold": clf.review_threshold,
            **metrics.binary_metrics(truth, predicted, positive, negative),
        },
        "latency": metrics.latency_summary(latencies),
    }
    if args.reference_threshold is not None:
        ref = [positive if p is not None and p >= args.reference_threshold else negative
               for p in probabilities]
        report["reference_threshold"] = {
            "description": "Raw P(positive) thresholded without a review band, to compare with the research repo's reports.",
            "threshold": args.reference_threshold,
            **metrics.binary_metrics(truth, ref, positive, negative),
        }
    return report


# ------------------------------------------------------------------- end-to-end

def run_end_to_end(args, settings) -> dict[str, Any]:
    import cv2
    from app.ml.model_registry import ModelRegistry
    from src.grading import size_category

    try:
        from app.services.inspection_pipeline import inspect_image
    except ImportError as exc:  # pre-change checkout has no decision pipeline
        raise SystemExit(f"end-to-end mode needs the decision pipeline: {exc}")

    registry = ModelRegistry()
    registry.load()
    if not (registry.detector_loaded and registry.damage_loaded):
        raise SystemExit("required models could not be loaded")
    manifest = Path(args.manifest)
    root = Path(args.data_root) if args.data_root else manifest.parent
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    by_image: dict[str, list[dict]] = {}
    for row in rows:
        by_image.setdefault(row["image"], []).append(row)

    eggs, latencies, misses, false_positives = [], [], 0, 0
    calibration_states = []
    for image_name, egg_rows in by_image.items():
        image = cv2.imread(str(root / image_name))
        if image is None:
            raise SystemExit(f"cannot read {image_name}")
        start = time.perf_counter()
        result = inspect_image(image, predictor=registry.detector, damage_classifier=registry.damage,
                               cleanliness_classifier=registry.cleanliness if registry.cleanliness_loaded else None,
                               detection_threshold=args.confidence)
        latencies.append(time.perf_counter() - start)
        calibration_states.append(result["calibration"]["status"])
        truth_boxes = [[float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])] for r in egg_rows]
        predictions = [(d["bbox"], d["confidence"]) for d in result["detections"]]
        matches, missed, extra = metrics.match_boxes(truth_boxes, predictions, args.iou)
        misses += len(missed)
        false_positives += len(extra)
        matched = {t: p for t, p, _ in matches}
        for index, row in enumerate(egg_rows):
            damaged = row["damage_label"].strip() == "Damaged"
            stained = row.get("cleanliness_label", "").strip() == "Stained"
            expected = "Reject" if damaged or stained else "Acceptable"
            det = result["detections"][matched[index]] if index in matched else None
            egg = {"image": image_name, "expected_decision": expected,
                   "decision": det["decision"] if det else "Missed",
                   "decision_reason": det["decision_reason"] if det else None,
                   "damage_label": row["damage_label"], "damage_status": det["damage_status"] if det else None,
                   "cleanliness_label": row.get("cleanliness_label"),
                   "cleanliness_status": det["cleanliness_status"] if det else None,
                   "groups": {k[6:]: v for k, v in row.items() if k.startswith("group_") and v}}
            true_length = row.get("true_length_mm")
            if true_length and det and det.get("length_mm") is not None:
                egg["true_length_mm"] = float(true_length)
                egg["pred_length_mm"] = det["length_mm"]
                egg["true_size"] = size_category(float(true_length))
                egg["pred_size"] = det["size"]
            if row.get("true_breadth_mm") and det and det.get("breadth_mm") is not None:
                egg["true_breadth_mm"] = float(row["true_breadth_mm"])
                egg["pred_breadth_mm"] = det["breadth_mm"]
            eggs.append(egg)

    def summarize(subset: list[dict]) -> dict[str, Any]:
        detected = [e for e in subset if e["decision"] != "Missed"]
        decided = [e for e in detected if e["decision"] != "Needs Review"]
        return {
            "eggs": len(subset),
            "missed": len(subset) - len(detected),
            "decision_confusion": metrics.multiclass_confusion(
                (e["expected_decision"] for e in subset), (e["decision"] for e in subset)),
            "review_rate": metrics.safe_div(len(detected) - len(decided), len(detected)),
            "decided_accuracy": metrics.safe_div(sum(e["decision"] == e["expected_decision"] for e in decided), len(decided)),
            "reject_recall_counting_review_as_miss": metrics.safe_div(
                sum(e["decision"] == "Reject" for e in subset if e["expected_decision"] == "Reject"),
                sum(e["expected_decision"] == "Reject" for e in subset)),
            "acceptable_precision": metrics.safe_div(
                sum(e["expected_decision"] == "Acceptable" for e in detected if e["decision"] == "Acceptable"),
                sum(e["decision"] == "Acceptable" for e in detected)),
        }

    groups: dict[str, dict[str, list]] = {}
    for egg in eggs:
        for key, value in egg["groups"].items():
            groups.setdefault(key, {}).setdefault(value, []).append(egg)
    sized = [e for e in eggs if "pred_length_mm" in e]
    breadth = [e for e in eggs if "pred_breadth_mm" in e]
    return {
        "mode": "end-to-end",
        "inputs": {"manifest": display_path(manifest), "manifest_sha256": sha256_file(manifest),
                   "images": len(by_image), "eggs": len(eggs), "confidence": args.confidence, "iou": args.iou},
        "metrics": {
            **summarize(eggs),
            "detection": metrics.detection_metrics(len(eggs) - misses, false_positives, misses),
            "latency": metrics.latency_summary(latencies),
            "by_group": {key: {value: summarize(items) for value, items in sorted(values.items())}
                         for key, values in sorted(groups.items())},
        },
        "sizing": {
            "invalid_calibration_rate": metrics.safe_div(
                sum(state != "available" for state in calibration_states), len(calibration_states)),
            "length_error_mm": metrics.error_summary([e["pred_length_mm"] for e in sized], [e["true_length_mm"] for e in sized]),
            "breadth_error_mm": metrics.error_summary([e["pred_breadth_mm"] for e in breadth], [e["true_breadth_mm"] for e in breadth]),
            "size_confusion": metrics.multiclass_confusion((e["true_size"] for e in sized), (e["pred_size"] for e in sized)),
            "note": "Only eggs with physical ground truth AND a valid marker are measured.",
        },
        "eggs": eggs,
    }


# --------------------------------------------------------------------- reports

def flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)


def compare(before: dict, after: dict) -> list[dict[str, Any]]:
    a, b = {}, {}
    for section in ("metrics", "app_policy", "reference_threshold", "latency", "sizing"):
        flatten(section, before.get(section, {}), b)
        flatten(section, after.get(section, {}), a)
    rows = []
    for key in sorted(set(a) | set(b)):
        if ".by_group." in key or key.endswith(".count"):
            continue
        old, new = b.get(key), a.get(key)
        if old == new:
            status = "unchanged"
        elif old is None or new is None:
            status = "added" if old is None else "removed"
        else:
            status = "changed"
        rows.append({"metric": key, "before": old, "after": new,
                     "delta": None if old is None or new is None else new - old, "status": status})
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown(report: dict) -> str:
    lines = [f"# Benchmark: {report['mode']}" + (f" ({report['task']})" if report.get("task") else ""), ""]
    env = report["environment"]
    lines += [f"- Generated: {env['generated_at']}",
              f"- Code: {env['code'].get('branch')} @ {env['code'].get('commit')} (dirty: {env['code'].get('dirty')})",
              f"- Python {env['python']}, torch {env.get('torch', 'n/a')}",
              f"- Inputs: `{json.dumps(report['inputs'])}`", ""]
    lines += ["| Checkpoint | SHA-256 |", "|---|---|"]
    lines += [f"| {v['file']} | `{v['sha256']}` |" for v in env["checkpoints"].values()]
    lines += ["", "| Setting | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in env["settings"].items()]
    flat: dict[str, float] = {}
    for section in ("metrics", "app_policy", "reference_threshold", "latency", "sizing"):
        flatten(section, report.get(section, {}), flat)
    lines += ["", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {fmt(v)} |" for k, v in flat.items() if ".by_group." not in k]
    if report.get("comparison"):
        lines += ["", f"## Comparison with `{report['comparison_source']}`", "",
                  "| Metric | Before | After | Delta | Status |", "|---|---|---|---|---|"]
        lines += [f"| {r['metric']} | {fmt(r['before'])} | {fmt(r['after'])} | {fmt(r['delta'])} | {r['status']} |"
                  for r in report["comparison"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["detector", "classifier", "end-to-end"])
    parser.add_argument("--backend-root", type=Path, default=HERE.parent)
    parser.add_argument("--out", type=Path, required=True, help="JSON report path; a .md is written next to it")
    parser.add_argument("--compare", type=Path, help="earlier JSON report of the same mode and manifest")
    parser.add_argument("--images")
    parser.add_argument("--labels")
    parser.add_argument("--manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--reference-threshold", type=float)
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    backend_root = args.backend_root.resolve()
    settings = import_app(backend_root)
    runner = {"detector": run_detector, "classifier": run_classifier, "end-to-end": run_end_to_end}[args.mode]
    if args.mode == "detector" and not (args.images and args.labels):
        parser.error("detector mode needs --images and --labels")
    if args.mode in {"classifier", "end-to-end"} and not args.manifest:
        parser.error(f"{args.mode} mode needs --manifest")
    if args.mode == "classifier" and not args.task:
        parser.error("classifier mode needs --task")

    report = runner(args, settings)
    report["environment"] = environment(backend_root, settings)
    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        if before.get("inputs", {}).get("manifest_sha256") != report["inputs"].get("manifest_sha256"):
            raise SystemExit("refusing to compare runs on different inputs (manifest hash differs)")
        report["comparison_source"] = display_path(args.compare)
        report["comparison"] = compare(before, report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
