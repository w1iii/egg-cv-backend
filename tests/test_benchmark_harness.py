"""Harness mechanics only, on synthetic data with fake models.

These numbers are test fixtures, not evaluation results.
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import settings
from tests.fakes import cleanliness, damage, make_predictor

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("benchmark_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_end_to_end_scores_decisions_and_misses(harness, tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    image = (np.full((480, 640, 3), 120, np.uint8) + rng.integers(0, 40, (480, 640, 3), dtype=np.uint8))
    cv2.imwrite(str(tmp_path / "a.png"), image)
    boxes = [[100, 100, 180, 180], [300, 100, 380, 180]]

    class FakeRegistry:
        def load(self):
            self.detector = make_predictor(boxes)
            self.damage = damage(queue=[("Damaged", 0.9, None), ("Not Damaged", 0.95, None)])
            self.cleanliness = cleanliness()
            self.detector_loaded = self.damage_loaded = self.cleanliness_loaded = True

    import app.ml.model_registry as registry_module

    monkeypatch.setattr(registry_module, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(settings, "CLEANLINESS_RELEASE_APPROVED", True)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "x1", "y1", "x2", "y2", "damage_label", "cleanliness_label", "group_lighting"])
        writer.writerow(["a.png", 100, 100, 180, 180, "Damaged", "Clean", "bright"])
        writer.writerow(["a.png", 300, 100, 380, 180, "Not Damaged", "Clean", "bright"])
        writer.writerow(["a.png", 500, 300, 580, 380, "Not Damaged", "Stained", "dim"])

    args = argparse.Namespace(manifest=str(manifest), data_root=None, confidence=0.75, iou=0.5)
    report = harness.run_end_to_end(args, settings)
    m = report["metrics"]
    assert m["eggs"] == 3 and m["missed"] == 1
    assert m["decision_confusion"] == {
        "Acceptable": {"Acceptable": 1},
        "Reject": {"Missed": 1, "Reject": 1},
    }
    assert m["reject_recall_counting_review_as_miss"] == 0.5
    assert m["detection"]["fn"] == 1 and m["detection"]["fp"] == 0
    assert set(m["by_group"]["lighting"]) == {"bright", "dim"}
    assert report["sizing"]["invalid_calibration_rate"] == 1.0
    assert report["sizing"]["length_error_mm"]["count"] == 0


def test_compare_flags_changed_metrics(harness):
    before = {"metrics": {"precision": 0.9, "recall": 1.0}}
    after = {"metrics": {"precision": 0.95, "recall": 1.0}}
    rows = {row["metric"]: row for row in harness.compare(before, after)}
    assert rows["metrics.precision"]["status"] == "changed"
    assert rows["metrics.precision"]["delta"] == pytest.approx(0.05)
    assert rows["metrics.recall"]["status"] == "unchanged"
