"""Per-image scale estimates for the fixed ArUco camera setup."""
import cv2
import numpy as np


def detect_scale(image: np.ndarray, marker_id: int = 0, side_mm: float = 50.0) -> dict:
    result = {"status": "unavailable", "reason": "marker_not_found", "marker_id": marker_id,
              "dictionary": "DICT_4X4_50", "side_mm": side_mm, "mm_per_pixel": None}
    if image is None or image.size == 0 or side_mm <= 0:
        result["reason"] = "invalid_image_or_configuration"
        return result
    if not hasattr(cv2, "aruco"):
        result["reason"] = "aruco_unavailable"
        return result
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
    if ids is None:
        return result
    matches = [i for i, value in enumerate(ids.flatten()) if int(value) == marker_id]
    if len(matches) != 1:
        result["reason"] = "wrong_marker" if not matches else "duplicate_marker"
        return result
    points = corners[matches[0]].reshape(4, 2)
    h, w = image.shape[:2]
    if (points[:, 0].min() <= 1 or points[:, 1].min() <= 1 or
            points[:, 0].max() >= w - 2 or points[:, 1].max() >= h - 2):
        result["reason"] = "marker_clipped"
        return result
    sides = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    if sides.min() < 40 or sides.max() / sides.min() > 1.2:
        result["reason"] = "marker_too_small_or_distorted"
        return result
    result.update(status="available", reason=None, mm_per_pixel=side_mm / float(sides.mean()))
    return result
