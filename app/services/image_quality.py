"""Image- and crop-level input checks that route unreliable inputs to review.

These are heuristic guards, not calibrated quality models. Their thresholds
live in ``app.config.Settings`` and were chosen only so that the bundled
sample images are not flagged; they have not been tuned on labeled data.
Every flag adds a review reason, so a failing check can only move an egg to
``Needs Review``, never to ``Acceptable`` or ``Reject``.
"""

from typing import Any, Optional

import cv2
import numpy as np

from app.config import settings
from src.detection_postprocess import overlap_ratio


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def assess_image(image: np.ndarray, config: Any = settings) -> dict[str, Any]:
    """Whole-image checks; any reason applies to every egg in the image."""
    height, width = image.shape[:2]
    gray = _gray(image)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons: list[str] = []
    if min(height, width) < config.IMAGE_MIN_SIDE_PX:
        reasons.append("image_low_resolution")
    if sharpness < config.IMAGE_MIN_SHARPNESS:
        reasons.append("image_blurry")
    return {
        "width": int(width),
        "height": int(height),
        "sharpness": round(sharpness, 2),
        "mean_brightness": round(float(gray.mean()), 2),
        "reasons": reasons,
    }


def assess_crop(
    image: np.ndarray,
    bbox: list[int],
    other_boxes: list[list[int]],
    config: Any = settings,
) -> dict[str, Any]:
    """Per-detection checks: partial visibility, overlap, exposure and glare."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    margin = config.EDGE_MARGIN_PX
    reasons: list[str] = []

    touches_edge = (
        x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin
    )
    if touches_edge:
        reasons.append("partial_visibility")

    max_overlap = max(
        (overlap_ratio(bbox, other) for other in other_boxes), default=0.0
    )
    if max_overlap >= config.DETECTION_OVERLAP_REVIEW:
        reasons.append("overlapping_detection")

    crop = image[y1:y2, x1:x2]
    mean_brightness: Optional[float] = None
    saturated_fraction: Optional[float] = None
    if crop.size:
        gray = _gray(crop)
        mean_brightness = float(gray.mean())
        saturated_fraction = float((gray >= 250).mean())
        if mean_brightness < config.CROP_MIN_BRIGHTNESS:
            reasons.append("crop_underexposed")
        if saturated_fraction > config.CROP_MAX_SATURATED_FRACTION:
            reasons.append("crop_glare")

    return {
        "touches_edge": bool(touches_edge),
        "max_overlap": round(float(max_overlap), 3),
        "mean_brightness": (
            None if mean_brightness is None else round(mean_brightness, 2)
        ),
        "saturated_fraction": (
            None if saturated_fraction is None else round(saturated_fraction, 4)
        ),
        "reasons": reasons,
    }
