import numpy as np
import pytest

from src.detection_postprocess import (
    clip_box,
    deduplicate_boxes,
    detection_count_status,
    overlap_ratio,
)
from tests.fakes import grid_boxes, make_predictor


def test_clip_box_clamps_and_rejects_degenerate():
    assert clip_box([-10.4, -3, 50.6, 40], 100, 100) == [0, 0, 50, 40]  # truncates like training crops
    assert clip_box([90, 90, 130, 140], 100, 100) == [90, 90, 100, 100]
    assert clip_box([120, 10, 150, 40], 100, 100) is None  # fully outside
    assert clip_box([10, 10, 10, 40], 100, 100) is None  # zero width
    assert clip_box([10, 10, 40, 40], None, None) == [10, 10, 40, 40]


def test_deduplicate_keeps_most_confident_and_orders_rows():
    candidates = [
        {"bbox": [10, 10, 60, 60], "confidence": 0.8},
        {"bbox": [11, 11, 61, 61], "confidence": 0.9},
        {"bbox": [0, 100, 50, 150], "confidence": 0.7},
    ]
    kept, removed = deduplicate_boxes(candidates, 0.85)
    assert removed == 1
    assert [item["confidence"] for item in kept] == [0.9, 0.7]


def test_overlap_ratio_uses_smaller_box():
    big, small = [0, 0, 100, 100], [80, 80, 100, 100]
    assert overlap_ratio(big, small) == pytest.approx(1.0)
    assert overlap_ratio(big, [200, 200, 210, 210]) == 0.0


def test_contained_fragment_removed_even_if_more_confident():
    candidates = [
        {"bbox": [470, 75, 1036, 848], "confidence": 0.78},
        {"bbox": [460, 286, 786, 845], "confidence": 0.90},
    ]
    kept, removed = deduplicate_boxes(candidates)
    assert removed == 1
    assert kept == [candidates[0]]


def test_distinct_overlapping_or_off_center_contained_boxes_retained():
    candidates = [
        {"bbox": [0, 0, 100, 100], "confidence": 0.9},
        {"bbox": [70, 30, 130, 90], "confidence": 0.8},
        {"bbox": [75, 75, 95, 95], "confidence": 0.7},
    ]
    kept, removed = deduplicate_boxes(candidates)
    assert removed == 0
    assert len(kept) == 3


def test_predictor_discards_contained_fragment_before_assigning_egg_ids():
    image = np.zeros((1000, 1500, 3), dtype=np.uint8)
    predictor = make_predictor(
        [[470, 75, 1036, 848], [460, 286, 786, 845]], [0.78, 0.90]
    )
    result = predictor.predict_frame(image, confidence=0.75)
    assert result["total"] == 1
    assert result["duplicate_boxes_removed"] == 1
    assert result["detections"][0]["bbox"] == [470, 75, 1036, 848]


def test_count_status_has_no_capacity():
    assert detection_count_status(0) == {"total": 0, "count_status": "empty"}
    assert detection_count_status(37) == {"total": 37, "count_status": "ok"}


@pytest.mark.parametrize("count", [0, 1, 5, 6, 12, 30])
def test_arbitrary_egg_counts_are_all_kept(count):
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    predictor = make_predictor(grid_boxes(count, size=60, gap=40, columns=7))
    result = predictor.predict_frame(image, confidence=0.5)
    assert result["total"] == count
    assert len(result["detections"]) == count
    assert "over_capacity" not in result and "max_eggs" not in result
    assert [det["egg_id"] for det in result["detections"]] == list(range(1, count + 1))


def test_parser_clips_out_of_bounds_and_counts_removed_boxes():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    predictor = make_predictor(
        [[-20, -20, 60, 60], [250, 150, 400, 260], [500, 500, 600, 600], [100, 50, 150, 100], [101, 51, 151, 101]],
        [0.9, 0.9, 0.9, 0.95, 0.8],
    )
    result = predictor.predict_frame(image, confidence=0.5)
    bboxes = [det["bbox"] for det in result["detections"]]
    assert [0, 0, 60, 60] in bboxes
    assert [250, 150, 300, 200] in bboxes
    assert result["invalid_boxes_removed"] == 1
    assert result["duplicate_boxes_removed"] == 1
    assert result["raw_detection_count"] == 5
    for x1, y1, x2, y2 in bboxes:
        assert 0 <= x1 < x2 <= 300 and 0 <= y1 < y2 <= 200


def test_detector_threshold_is_passed_through():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    predictor = make_predictor([[10, 10, 60, 60], [100, 100, 150, 150]], [0.9, 0.4])
    assert predictor.predict_frame(image, confidence=0.75)["total"] == 1
    assert predictor.predict_frame(image, confidence=0.3)["total"] == 2
