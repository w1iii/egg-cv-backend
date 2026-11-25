import cv2
import numpy as np
import pytest

from app.config import settings
from app.services.inspection_pipeline import inspect_image
from tests.fakes import cleanliness, damage, grid_boxes, make_predictor


def config(**overrides):
    return settings.model_copy(update=overrides)


def textured(height=600, width=1000):
    rng = np.random.default_rng(2)
    return (np.full((height, width, 3), 120, dtype=np.uint8)
            + rng.integers(0, 40, size=(height, width, 3), dtype=np.uint8))


def with_marker(image, size=200, origin=(50, 50)):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, size)
    x, y = origin
    image[y - 40:y + size + 40, x - 40:x + size + 40] = 255  # >= one cell quiet zone
    image[y:y + size, x:x + size] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return image


def run(image, boxes, damage_clf=None, clean_clf=None, cfg=None, confs=None, threshold=0.75):
    return inspect_image(
        image,
        predictor=make_predictor(boxes, confs),
        damage_classifier=damage_clf or damage(),
        cleanliness_classifier=clean_clf,
        detection_threshold=threshold,
        config=cfg or config(),
    )


@pytest.mark.parametrize("count", [1, 5, 6, 14])
def test_every_valid_detection_is_processed(count):
    result = run(textured(), grid_boxes(count, size=90, gap=40, origin=40, columns=7), clean_clf=cleanliness())
    assert result["summary"]["total"] == count
    assert all(det["decision"] for det in result["detections"])
    assert "tray_over_capacity" not in str(result)


def test_zero_detections_complete_with_empty_status():
    result = run(textured(), [])
    assert result["count_status"] == "empty"
    assert result["summary"]["total"] == 0


def test_default_release_state_routes_clean_intact_eggs_to_review():
    result = run(textured(), grid_boxes(3, size=90, origin=40), clean_clf=cleanliness())
    for det in result["detections"]:
        assert det["decision"] == "Needs Review"
        assert det["decision_reason"] == "cleanliness_unapproved"
        assert det["grade"] == "N/A"
        assert det["requires_review"] is True


def test_missing_cleanliness_model_is_safe():
    result = run(textured(), grid_boxes(2, size=90, origin=40), clean_clf=None,
                 cfg=config(CLEANLINESS_RELEASE_APPROVED=True))
    for det in result["detections"]:
        assert det["cleanliness_status"] == "Unavailable"
        assert det["decision"] == "Needs Review"
        assert "cleanliness_unavailable" in det["decision_reason"]


def test_damaged_egg_is_rejected_and_reasons_are_separated():
    clf = damage(queue=[("Damaged", 0.9, None), ("Needs Review", 0.6, "low_confidence")])
    result = run(textured(), grid_boxes(3, size=90, origin=40), damage_clf=clf,
                 clean_clf=cleanliness(), cfg=config(CLEANLINESS_RELEASE_APPROVED=True))
    decisions = [(d["decision"], d["decision_reason"]) for d in result["detections"]]
    assert decisions == [
        ("Reject", "damaged"),
        ("Needs Review", "damage_low_confidence"),
        ("Acceptable", None),
    ]
    assert result["summary"]["acceptable"] == 1
    assert result["summary"]["reject"] == 1
    assert result["summary"]["needs_review"] == 1


def test_experimental_damage_prediction_never_finalizes_decision():
    clf = damage(queue=[("Damaged", 0.9, None)])
    clf.experimental = True
    result = run(textured(), grid_boxes(1, size=90, origin=40), damage_clf=clf,
                 clean_clf=cleanliness(), cfg=config(CLEANLINESS_RELEASE_APPROVED=True))
    det = result["detections"][0]
    assert det["damage_status"] == "Damaged"
    assert det["decision"] == "Needs Review"
    assert "damage_unvalidated" in det["decision_reason"]


def test_partial_and_overlapping_eggs_are_flagged_individually():
    boxes = [[0, 200, 90, 290], [300, 200, 400, 300], [330, 230, 430, 330], [600, 200, 690, 290]]
    result = run(textured(), boxes, clean_clf=cleanliness(), cfg=config(CLEANLINESS_RELEASE_APPROVED=True))
    by_box = {tuple(d["bbox"]): d for d in result["detections"]}
    assert "partial_visibility" in by_box[(0, 200, 90, 290)]["decision_reason"]
    assert "overlapping_detection" in by_box[(300, 200, 400, 300)]["decision_reason"]
    assert "overlapping_detection" in by_box[(330, 230, 430, 330)]["decision_reason"]
    assert by_box[(600, 200, 690, 290)]["decision"] == "Acceptable"


def test_damage_and_cleanliness_use_separate_padding():
    dmg, cln = damage(), cleanliness()
    run(textured(), [[300, 200, 400, 300]], damage_clf=dmg, clean_clf=cln)
    assert dmg.crops[0].shape[:2] == (120, 120)  # 10% padding each side
    assert cln.crops[0].shape[:2] == (100, 100)  # tight crop


def test_without_marker_no_dimensions_or_grades():
    result = run(textured(), grid_boxes(2, size=90, origin=40), clean_clf=cleanliness(),
                 cfg=config(CLEANLINESS_RELEASE_APPROVED=True, ARUCO_GRADES_APPROVED=True))
    assert result["calibration"]["status"] == "unavailable"
    for det in result["detections"]:
        assert det["size"] == "Unknown"
        assert det["length_mm"] is None and det["breadth_mm"] is None
        assert det["weight_g"] is None
        assert det["grade"] == "N/A"
        assert det["calibration_version"] == "uncalibrated"


SIZED_BOXES = [[300, 250, 480, 380], [520, 250, 740, 400], [300, 420, 560, 590]]


def test_marker_without_approval_measures_but_never_grades():
    result = run(with_marker(textured(700, 1000)), SIZED_BOXES, clean_clf=cleanliness(),
                 cfg=config(CLEANLINESS_RELEASE_APPROVED=True, ARUCO_GRADES_APPROVED=False))
    assert result["calibration"]["status"] == "available"
    assert result["size_grades_enabled"] is False
    assert [d["size"] for d in result["detections"]] == ["Small", "Medium", "Large"]
    assert all(d["length_mm"] for d in result["detections"])
    assert {d["grade"] for d in result["detections"]} == {"N/A"}
    assert {d["decision"] for d in result["detections"]} == {"Acceptable"}


def test_approved_calibration_grades_only_acceptable_eggs():
    clf = damage(queue=[("Not Damaged", 0.95, None), ("Not Damaged", 0.95, None), ("Damaged", 0.9, None)])
    result = run(with_marker(textured(700, 1000)), SIZED_BOXES, damage_clf=clf, clean_clf=cleanliness(),
                 cfg=config(CLEANLINESS_RELEASE_APPROVED=True, ARUCO_GRADES_APPROVED=True))
    assert [(d["decision"], d["grade"]) for d in result["detections"]] == [
        ("Acceptable", "C"),
        ("Acceptable", "B"),
        ("Reject", "N/A"),
    ]


def test_undecodable_image_raises():
    with pytest.raises(ValueError, match="could not be decoded"):
        run(np.zeros((0, 0, 3), dtype=np.uint8), [])
