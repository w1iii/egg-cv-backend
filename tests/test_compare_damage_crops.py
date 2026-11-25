from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts.compare_damage_crops import compare


class Classifier:
    def __init__(self, status, probability):
        self.status, self.probability = status, probability

    def classify_crop(self, image):
        return SimpleNamespace(status=self.status, probability=self.probability,
                               review_reason=None)


def test_compare_preserves_labels_and_scores(tmp_path):
    cv2.imwrite(str(tmp_path / "egg.png"), np.zeros((60, 60, 3), dtype=np.uint8))
    rows = [{"filename": "egg.png", "label_name": "Not Damaged", "source_file": "sheet"}]
    result = compare(rows, tmp_path, Classifier("Not Damaged", 0.1),
                     Classifier("Damaged", 0.9))[0]
    assert result["source_file"] == "sheet"
    assert result["baseline_status"] == "Not Damaged"
    assert result["candidate_status"] == "Damaged"
    assert result["candidate_damage_probability"] == 0.9


def test_compare_rejects_bad_data(tmp_path):
    fake = Classifier("Damaged", 0.9)
    with pytest.raises(ValueError, match="Invalid label"):
        compare([{"filename": "x", "label_name": "Uncertain"}], tmp_path, fake, fake)
    with pytest.raises(ValueError, match="Cannot read"):
        compare([{"filename": "missing.png", "label_name": "Damaged"}], tmp_path, fake, fake)
