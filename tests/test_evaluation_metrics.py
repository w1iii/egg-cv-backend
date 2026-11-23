import pytest

from src.evaluation_metrics import (
    binary_metrics,
    detection_metrics,
    error_summary,
    latency_summary,
    match_boxes,
)


def test_match_boxes_one_to_one_by_confidence():
    truth = [[0, 0, 10, 10], [20, 20, 30, 30]]
    predictions = [([0, 0, 10, 10], 0.6), ([1, 1, 10, 10], 0.9), ([50, 50, 60, 60], 0.8)]
    matches, missed, extra = match_boxes(truth, predictions)
    assert [(t, p) for t, p, _ in matches] == [(0, 1)]
    assert missed == [1]
    assert extra == [0, 2]


def test_detection_metrics_handle_empty():
    assert detection_metrics(0, 0, 0)["precision"] is None
    result = detection_metrics(8, 2, 0)
    assert result["precision"] == pytest.approx(0.8)
    assert result["recall"] == 1.0
    assert result["f1"] == pytest.approx(2 * 0.8 / 1.8)


def test_binary_metrics_separate_review_from_errors():
    truth = ["Damaged"] * 4 + ["Not Damaged"] * 4
    predicted = ["Damaged", "Damaged", "Not Damaged", "Needs Review",
                 "Not Damaged", "Not Damaged", "Damaged", "Needs Review"]
    m = binary_metrics(truth, predicted, "Damaged", "Not Damaged")
    assert m["samples"] == 8 and m["decided"] == 6
    assert m["review_rate"] == pytest.approx(2 / 8)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)
    assert m["positive_recall_counting_review_as_miss"] == pytest.approx(2 / 4)
    assert m["confusion"]["Damaged"] == {"Damaged": 2, "Not Damaged": 1, "Needs Review": 1}


def test_error_and_latency_summaries():
    errors = error_summary([52.0, 58.0], [50.0, 60.0])
    assert errors["mae"] == 2.0 and errors["mean_error"] == 0.0
    assert errors["mean_abs_pct_error"] == pytest.approx((4 + 10 / 3) / 2)
    assert latency_summary([])["count"] == 0
    assert latency_summary([0.1, 0.2, 0.3])["p50_ms"] == pytest.approx(200)
