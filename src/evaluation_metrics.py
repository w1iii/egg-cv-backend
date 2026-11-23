"""Pure metric helpers for the benchmark harness.

Deliberately dependency-free (no app or ``src`` imports) so the harness can
load it by path while importing the app under test from another checkout.
"""

from collections import Counter
from statistics import mean
from typing import Any, Iterable, Optional, Sequence


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def match_boxes(
    truth: Sequence[Sequence[float]],
    predictions: Sequence[tuple[Sequence[float], float]],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy one-to-one matching, most confident prediction first.

    Returns ``(matches, unmatched_truth, unmatched_predictions)`` where each
    match is ``(truth_index, prediction_index, iou)``.
    """
    order = sorted(range(len(predictions)), key=lambda i: predictions[i][1], reverse=True)
    used_truth: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    unmatched_predictions: list[int] = []
    for pred_index in order:
        box = list(predictions[pred_index][0])
        best, best_iou = None, iou_threshold
        for truth_index, truth_box in enumerate(truth):
            if truth_index in used_truth:
                continue
            iou = box_iou(box, list(truth_box))
            if iou >= best_iou:
                best, best_iou = truth_index, iou
        if best is None:
            unmatched_predictions.append(pred_index)
        else:
            used_truth.add(best)
            matches.append((best, pred_index, best_iou))
    unmatched_truth = [i for i in range(len(truth)) if i not in used_truth]
    return matches, unmatched_truth, sorted(unmatched_predictions)


def detection_metrics(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def binary_metrics(
    truth: Sequence[str],
    predicted: Sequence[str],
    positive: str,
    negative: str,
    abstain: str = "Needs Review",
) -> dict[str, Any]:
    """Metrics for a classifier that may abstain.

    The full truth x prediction table includes the abstain column. Precision,
    recall, F1 and balanced accuracy are computed on decided items only, and
    ``review_rate`` reports how many items were deferred; the two must be
    read together.
    """
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted differ in length")
    labels_out = [positive, negative, abstain]
    table = {t: {p: 0 for p in labels_out} for t in (positive, negative)}
    other = 0
    for t, p in zip(truth, predicted):
        if t not in table:
            continue
        if p not in labels_out:
            other += 1
            continue
        table[t][p] += 1
    tp, fn = table[positive][positive], table[positive][negative]
    fp, tn = table[negative][positive], table[negative][negative]
    reviewed = table[positive][abstain] + table[negative][abstain]
    total = sum(sum(row.values()) for row in table.values())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    balanced = (
        (recall + specificity) / 2 if recall is not None and specificity is not None else None
    )
    return {
        "samples": total,
        "confusion": table,
        "unrecognized_predictions": other,
        "decided": total - reviewed,
        "review_rate": safe_div(reviewed, total),
        "accuracy_decided": safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced,
        # Counting every abstention as an error gives a pessimistic bound.
        "positive_recall_counting_review_as_miss": safe_div(tp, tp + fn + table[positive][abstain]),
    }


def multiclass_confusion(truth: Iterable[str], predicted: Iterable[str]) -> dict[str, dict[str, int]]:
    table: dict[str, Counter] = {}
    for t, p in zip(truth, predicted):
        table.setdefault(t, Counter())[p] += 1
    return {t: dict(sorted(row.items())) for t, row in sorted(table.items())}


def latency_summary(seconds: Sequence[float]) -> dict[str, Optional[float]]:
    if not seconds:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(seconds)

    def percentile(q: float) -> float:
        index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
        return ordered[index] * 1000

    return {
        "count": len(ordered),
        "mean_ms": mean(ordered) * 1000,
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "max_ms": ordered[-1] * 1000,
    }


def error_summary(predicted: Sequence[float], truth: Sequence[float]) -> dict[str, Optional[float]]:
    errors = [p - t for p, t in zip(predicted, truth)]
    if not errors:
        return {"count": 0, "mean_error": None, "mae": None, "rmse": None, "mean_abs_pct_error": None}
    return {
        "count": len(errors),
        "mean_error": mean(errors),
        "mae": mean(abs(e) for e in errors),
        "rmse": mean(e * e for e in errors) ** 0.5,
        "mean_abs_pct_error": mean(abs(e) / t * 100 for e, t in zip(errors, truth) if t),
    }
