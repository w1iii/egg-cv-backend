import itertools

import pytest

from app.services.grading_service import (
    COMMERCIAL_GRADES,
    calculate_commercial_grade,
    calculate_quality_decision,
    classifier_reason,
    public_grade,
)

DAMAGE = ["Damaged", "Not Damaged", "Needs Review", None]
CLEANLINESS = ["Clean", "Stained", "Needs Review", "Unavailable", "Unknown", None]


def expected(damage, cleanliness, approved):
    """Independent restatement of the policy table in the improvement plan."""
    if damage == "Damaged":
        return "Reject"
    if approved and cleanliness == "Stained":
        return "Reject"
    if damage != "Not Damaged":
        return "Needs Review"
    if not approved or cleanliness != "Clean":
        return "Needs Review"
    return "Acceptable"


@pytest.mark.parametrize(
    "damage,cleanliness,approved",
    list(itertools.product(DAMAGE, CLEANLINESS, [True, False])),
)
def test_every_policy_combination(damage, cleanliness, approved):
    result = calculate_quality_decision(
        damage_status=damage,
        cleanliness_status=cleanliness,
        cleanliness_release_approved=approved,
    )
    assert result.decision == expected(damage, cleanliness, approved)
    assert result.requires_review == (result.decision == "Needs Review")
    if result.decision != "Acceptable":
        assert result.reasons, "every non-acceptable decision must explain itself"
    else:
        assert result.reasons == ()


def test_unapproved_cleanliness_never_accepts_or_rejects_alone():
    for status in CLEANLINESS:
        result = calculate_quality_decision(
            damage_status="Not Damaged",
            cleanliness_status=status,
            cleanliness_release_approved=False,
        )
        assert result.decision == "Needs Review"
        assert "cleanliness_unapproved" in result.reasons


def test_damage_rejects_even_with_unapproved_cleanliness():
    result = calculate_quality_decision(
        damage_status="Damaged",
        cleanliness_status="Clean",
        cleanliness_release_approved=False,
    )
    assert (result.decision, result.reasons) == ("Reject", ("damaged",))


def test_stained_beats_uncertain_damage_when_approved():
    result = calculate_quality_decision(
        damage_status="Needs Review",
        cleanliness_status="Stained",
        cleanliness_release_approved=True,
    )
    assert (result.decision, result.reasons) == ("Reject", ("stained",))


@pytest.mark.parametrize("damage", ["Damaged", "Not Damaged"])
def test_unreliable_input_overrides_classifier(damage):
    result = calculate_quality_decision(
        damage_status=damage,
        cleanliness_status="Clean",
        cleanliness_release_approved=True,
        input_review_reasons=["partial_visibility", "overlapping_detection"],
    )
    assert result.decision == "Needs Review"
    assert result.reasons == ("partial_visibility", "overlapping_detection")


def test_refused_crop_is_an_input_reason():
    result = calculate_quality_decision(
        damage_status="Needs Review",
        cleanliness_status="Clean",
        cleanliness_release_approved=True,
        damage_reason="small_crop",
    )
    assert result.reasons == ("crop_too_small",)


def test_classifier_reasons_are_prefixed():
    result = calculate_quality_decision(
        damage_status="Needs Review",
        cleanliness_status="Needs Review",
        cleanliness_release_approved=True,
        damage_reason="low_confidence",
        cleanliness_reason="low_confidence",
    )
    assert result.reasons == ("damage_low_confidence", "cleanliness_low_confidence")
    assert classifier_reason("damage", None) is None


@pytest.mark.parametrize(
    "size,grade", [("Large", "A"), ("Medium", "B"), ("Small", "C"), ("Unknown", "N/A"), (None, "N/A")]
)
def test_grade_only_for_acceptable_and_enabled(size, grade):
    assert calculate_commercial_grade("Acceptable", size, size_grades_enabled=True) == grade
    assert calculate_commercial_grade("Acceptable", size, size_grades_enabled=False) == "N/A"
    for decision in ("Reject", "Needs Review"):
        assert calculate_commercial_grade(decision, size, size_grades_enabled=True) == "N/A"


def test_grade_vocabulary_never_contains_decisions():
    assert "Reject" not in COMMERCIAL_GRADES
    assert "Acceptable" not in COMMERCIAL_GRADES


@pytest.mark.parametrize(
    "stored,public", [("AA", "A"), ("A", "A"), ("N/A", "N/A"), ("Reject", "N/A"), (None, None)]
)
def test_legacy_grades_map_to_commercial_vocabulary(stored, public):
    assert public_grade(stored) == public
