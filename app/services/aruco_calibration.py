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
    if not _has_black_border(image, points):
        # A thin white margin on a darker surface can be decoded as the
        # marker itself, inflating the measured side by the margin width.
        result["reason"] = "marker_border_invalid"
        return result
    result.update(status="available", reason=None, mm_per_pixel=side_mm / float(sides.mean()))
    return result


# DICT_4X4: 4 data cells plus a one-cell black border on each side.
_MARKER_CELLS = 6
_CELL_PX = 16


def _has_black_border(image: np.ndarray, points: np.ndarray) -> bool:
    """True if the detected quad's outer cell ring is dark, as a real marker's is."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = _MARKER_CELLS * _CELL_PX
    target = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    warp = cv2.warpPerspective(
        gray, cv2.getPerspectiveTransform(points.astype(np.float32), target), (size, size)
    ).astype(np.float32)
    lo, hi = np.percentile(warp, 5), np.percentile(warp, 95)
    if hi - lo < 30:
        return False
    # Sample the middle of the border cells so slight corner error is tolerated.
    inset, width = _CELL_PX // 4, _CELL_PX // 2
    ring = np.concatenate([
        warp[inset:inset + width, :].ravel(),
        warp[-inset - width:-inset, :].ravel(),
        warp[:, inset:inset + width].ravel(),
        warp[:, -inset - width:-inset].ravel(),
    ])
    return float(ring.mean()) <= lo + 0.35 * (hi - lo)
