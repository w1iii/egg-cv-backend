import unittest

import cv2
import numpy as np

from app.services.aruco_calibration import detect_scale
from src.geometry import bbox_geometry
from src.grading import size_category


class ArucoCalibrationTests(unittest.TestCase):
    def image(self, marker_id=0, size=200):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
        image = np.full((400, 400), 255, dtype=np.uint8)
        image[100:100 + size, 100:100 + size] = marker
        return image

    def test_scale(self):
        result = detect_scale(self.image())
        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["mm_per_pixel"], 50 / 199, places=4)

    def test_wrong_marker(self):
        self.assertEqual(detect_scale(self.image(1))["reason"], "wrong_marker")

    def test_missing(self):
        self.assertEqual(detect_scale(np.full((200, 200), 255, dtype=np.uint8))["reason"], "marker_not_found")

    def test_empty(self):
        self.assertEqual(detect_scale(np.array([]))["reason"], "invalid_image_or_configuration")

    def test_duplicate_marker(self):
        marker = self.image()[100:300, 100:300]
        image = np.full((250, 500), 255, dtype=np.uint8)
        image[25:225, 25:225] = marker
        image[25:225, 275:475] = marker
        self.assertEqual(detect_scale(image)["reason"], "duplicate_marker")

    def test_thin_quiet_zone_on_dark_surface_is_rejected(self):
        # Regression: a white margin narrower than one marker cell on a
        # darker surface used to be measured as the marker (230 px instead
        # of 199 px), silently shrinking mm-per-pixel by ~13%.
        marker = self.image()[100:300, 100:300]
        for background in (60, 140):
            image = np.full((500, 500), background, dtype=np.uint8)
            image[85:315, 85:315] = 255
            image[100:300, 100:300] = marker
            result = detect_scale(image)
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["reason"], "marker_border_invalid")

    def test_full_quiet_zone_on_dark_surface_is_measured(self):
        marker = self.image()[100:300, 100:300]
        image = np.full((500, 500), 60, dtype=np.uint8)
        image[60:340, 60:340] = 255
        image[100:300, 100:300] = marker
        result = detect_scale(image)
        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["mm_per_pixel"], 50 / 199, places=4)

    def test_scale_drives_bbox_geometry(self):
        scale = detect_scale(self.image())["mm_per_pixel"]
        geometry = bbox_geometry(220, 160, scale)
        self.assertAlmostEqual(geometry.length_mm, 220 * 50 / 199)
        self.assertEqual(size_category(geometry.length_mm), "Medium")


if __name__ == "__main__":
    unittest.main()
