"""Detector post-processing shared by inference and evaluation.

There is no fixed egg count: every valid detection in an image is kept and
graded individually. Duplicates are suppressed, coordinates are clipped to
the image, and degenerate boxes are dropped and counted.
"""

from typing import Any, Iterable, Optional


DEFAULT_DUPLICATE_IOU = 0.85
DEFAULT_CONTAINMENT = 0.95
DEFAULT_CENTER_DISTANCE = 0.25


def box_area(box: list[int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_intersection(box_a: list[int], box_b: list[int]) -> int:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))
    return width * height


def box_iou(box_a: list[int], box_b: list[int]) -> float:
    intersection = box_intersection(box_a, box_b)
    union = box_area(box_a) + box_area(box_b) - intersection
    return intersection / union if union > 0 else 0.0


def overlap_ratio(box_a: list[int], box_b: list[int]) -> float:
    """Intersection over the smaller box: catches a small egg hidden under a
    large one, which plain IoU understates."""
    smaller = min(box_area(box_a), box_area(box_b))
    if smaller <= 0:
        return 0.0
    return box_intersection(box_a, box_b) / smaller


def clip_box(
    box: Iterable[float], width: Optional[int], height: Optional[int]
) -> Optional[list[int]]:
    """Truncate and clip a box to the image; ``None`` if nothing valid is left.

    Truncation (``int()``) matches both the original inference code and the
    research repo's crop builder, so classifier inputs are pixel-identical.
    """
    x1, y1, x2, y2 = (int(float(value)) for value in box)
    if width is not None:
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    if height is not None:
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def deduplicate_boxes(
    candidates: Iterable[dict[str, Any]],
    iou_threshold: float = DEFAULT_DUPLICATE_IOU,
) -> tuple[list[dict[str, Any]], int]:
    """Suppress near-identical boxes and strongly contained, centered fragments.

    Containment alone is insufficient: a smaller real egg can overlap a larger
    egg. Require centers to be close in both axes as well. Keep the larger box
    for contained fragments so damage inference sees the whole egg. Remaining
    ambiguous overlaps are retained for the quality-review policy.
    """
    def centered_containment(a: list[int], b: list[int]) -> bool:
        if overlap_ratio(a, b) < DEFAULT_CONTAINMENT:
            return False
        large, small = (a, b) if box_area(a) >= box_area(b) else (b, a)
        width, height = large[2] - large[0], large[3] - large[1]
        return (
            abs((small[0] + small[2]) - (large[0] + large[2])) / (2 * width)
            <= DEFAULT_CENTER_DISTANCE
            and abs((small[1] + small[3]) - (large[1] + large[3])) / (2 * height)
            <= DEFAULT_CENTER_DISTANCE
        )

    kept: list[dict[str, Any]] = []
    removed = 0
    for candidate in sorted(
        candidates, key=lambda item: (box_area(item["bbox"]), item["confidence"]), reverse=True
    ):
        duplicate = next((existing for existing in kept if
                          box_iou(candidate["bbox"], existing["bbox"]) >= iou_threshold
                          or centered_containment(candidate["bbox"], existing["bbox"])), None)
        if duplicate is not None:
            if (box_iou(candidate["bbox"], duplicate["bbox"]) >= iou_threshold
                    and candidate["confidence"] > duplicate["confidence"]):
                kept[kept.index(duplicate)] = candidate
            removed += 1
            continue
        kept.append(candidate)
    kept.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return kept, removed


def detection_count_status(total: int) -> dict[str, Any]:
    return {"total": total, "count_status": "empty" if total == 0 else "ok"}
