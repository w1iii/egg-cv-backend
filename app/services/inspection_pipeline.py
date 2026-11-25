"""Image -> per-egg inspection result, without any database or file I/O.

Shared by the upload API and the offline benchmark so both run exactly the
same detection, crop, classification, review and grading logic.
"""

from typing import Any, Optional

import numpy as np

from app.config import settings
from app.services.aruco_calibration import detect_scale
from app.services.grading_service import (
    DECISIONS,
    REASON_DETECTION_LOW_CONFIDENCE,
    calculate_commercial_grade,
    calculate_quality_decision,
)
from app.services.image_quality import assess_crop, assess_image

CALIBRATION_VERSION_ARUCO = "aruco-static-v1"
CALIBRATION_VERSION_NONE = "uncalibrated"


def crop_detection(
    image: np.ndarray,
    bbox: list[int],
    padding_ratio: float,
    min_crop_size: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    box_width, box_height = x2 - x1, y2 - y1
    if box_width <= 0 or box_height <= 0:
        return image[0:0, 0:0]
    pad_x = int(round(box_width * padding_ratio))
    pad_y = int(round(box_height * padding_ratio))
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)
    if right - left < min_crop_size or bottom - top < min_crop_size:
        return image[0:0, 0:0]
    return image[top:bottom, left:right]


def inspect_image(
    image: np.ndarray,
    *,
    predictor: Any,
    damage_classifier: Any,
    cleanliness_classifier: Optional[Any],
    detection_threshold: float,
    config: Any = settings,
) -> dict[str, Any]:
    """Detect, classify, decide and grade every egg in one still image."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Uploaded image could not be decoded")
    if damage_classifier is None:
        raise RuntimeError("Damage classifier is not configured")

    cleanliness_approved = bool(config.CLEANLINESS_RELEASE_APPROVED)
    calibration = detect_scale(image, config.ARUCO_MARKER_ID, config.ARUCO_SIDE_MM)
    sizing_available = calibration["status"] == "available"
    size_grades_enabled = sizing_available and bool(config.ARUCO_GRADES_APPROVED)
    image_quality = assess_image(image, config)

    result = predictor.predict_frame(
        image,
        confidence=detection_threshold,
        mm_per_pixel=calibration["mm_per_pixel"] if sizing_available else 0.0,
    )
    result["calibration"] = calibration
    result["image_quality"] = image_quality
    result["size_grades_enabled"] = size_grades_enabled
    result["cleanliness_release_approved"] = cleanliness_approved

    detections = result["detections"]
    boxes = [det["bbox"] for det in detections]
    for index, det in enumerate(detections):
        others = boxes[:index] + boxes[index + 1:]
        crop_quality = assess_crop(image, det["bbox"], others, config)
        input_reasons = list(image_quality["reasons"]) + list(crop_quality["reasons"])
        if det["confidence"] < detection_threshold:
            input_reasons.insert(0, REASON_DETECTION_LOW_CONFIDENCE)
        det["crop_quality"] = crop_quality

        damage = damage_classifier.classify_crop(
            crop_detection(
                image, det["bbox"], config.DETECTION_CROP_PADDING, config.MIN_CROP_SIZE
            )
        )
        det["damage_status"] = damage.status
        det["damage_confidence"] = damage.confidence
        det["damage_probability"] = getattr(damage, "probability", None)
        det["damage_model_version"] = damage.model_version
        # Experimental provider predictions are visible, but cannot make a
        # final accept/reject decision until independently validated.
        if getattr(damage_classifier, "experimental", False):
            input_reasons.append("damage_unvalidated")

        if cleanliness_classifier is None:
            cleanliness_status = "Unavailable"
            cleanliness_confidence = None
            cleanliness_probability = None
            cleanliness_version = None
            cleanliness_reason = None
        else:
            cleanliness = cleanliness_classifier.classify_crop(
                crop_detection(
                    image,
                    det["bbox"],
                    config.CLEANLINESS_CROP_PADDING,
                    config.MIN_CROP_SIZE,
                )
            )
            cleanliness_status = cleanliness.status
            cleanliness_confidence = cleanliness.confidence
            cleanliness_probability = getattr(cleanliness, "probability", None)
            cleanliness_version = cleanliness.model_version
            cleanliness_reason = cleanliness.review_reason
        det["cleanliness_status"] = cleanliness_status
        det["cleanliness_confidence"] = cleanliness_confidence
        det["cleanliness_probability"] = cleanliness_probability
        det["cleanliness_model_version"] = cleanliness_version

        if not sizing_available:
            # Millimetres are meaningless without a valid marker; publish none.
            det["size"] = "Unknown"
            det["length_mm"] = None
            det["breadth_mm"] = None
        # Weight estimation has no calibration dataset; it stays disabled.
        det["weight_g"] = None
        det["calibration_version"] = (
            CALIBRATION_VERSION_ARUCO if sizing_available else CALIBRATION_VERSION_NONE
        )

        decision = calculate_quality_decision(
            damage_status=damage.status,
            cleanliness_status=cleanliness_status,
            cleanliness_release_approved=cleanliness_approved,
            damage_reason=damage.review_reason,
            cleanliness_reason=cleanliness_reason,
            input_review_reasons=input_reasons,
        )
        det["decision"] = decision.decision
        det["decision_reason"] = decision.reason_text
        det["requires_review"] = decision.requires_review
        det["review_reason"] = decision.reason_text if decision.requires_review else None
        det["grade"] = calculate_commercial_grade(
            decision.decision,
            det.get("size"),
            size_grades_enabled=size_grades_enabled,
        )

    result["summary"] = summarize(detections)
    return result


def summarize(detections: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = {decision: 0 for decision in DECISIONS}
    reason_counts: dict[str, int] = {}
    for det in detections:
        decision = det.get("decision")
        if decision in decisions:
            decisions[decision] += 1
        for reason in (det.get("decision_reason") or "").split(";"):
            if reason and decision != "Reject":
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "total": len(detections),
        "acceptable": decisions["Acceptable"],
        "reject": decisions["Reject"],
        "needs_review": decisions["Needs Review"],
        "damaged": sum(det.get("damage_status") == "Damaged" for det in detections),
        "not_damaged": sum(
            det.get("damage_status") == "Not Damaged" for det in detections
        ),
        "review_reason_counts": reason_counts,
    }
