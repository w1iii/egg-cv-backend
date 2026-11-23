"""Deterministic per-egg quality decision and commercial size grade.

Two separate concepts are produced for every detected egg:

``decision`` (external quality)
    ``Acceptable``, ``Reject`` or ``Needs Review``. It depends only on damage,
    cleanliness and the reliability of the inputs, never on size.

``grade`` (commercial size grade)
    ``A`` (Large), ``B`` (Medium), ``C`` (Small) or ``N/A``. A letter grade is
    published only for an ``Acceptable`` egg measured with a valid ArUco
    calibration *and* when size grades are release-approved
    (``ARUCO_GRADES_APPROVED``). Everything else is ``N/A``. ``Reject`` is a
    decision, never a grade.

Policy (first matching rule wins):

1. Any input-reliability problem (low detection confidence, partial or
   overlapping egg, unusable crop, poor image) -> ``Needs Review``. A
   classifier result on an unreliable crop is not trusted in either direction.
2. Damage ``Damaged`` -> ``Reject``.
3. Cleanliness ``Stained`` from a release-approved model -> ``Reject``.
4. Damage ``Needs Review``, or cleanliness ``Needs Review``/``Unavailable``/
   ``Unknown``, or a cleanliness model that is not release-approved ->
   ``Needs Review``.
5. Damage ``Not Damaged`` and approved cleanliness ``Clean`` -> ``Acceptable``.

An unapproved cleanliness model can therefore never produce ``Acceptable``
nor ``Reject`` on its own; its raw prediction is still stored and shown as
beta information.
"""

from dataclasses import dataclass
from typing import Iterable, Optional


DECISION_ACCEPTABLE = "Acceptable"
DECISION_REJECT = "Reject"
DECISION_NEEDS_REVIEW = "Needs Review"
DECISIONS = (DECISION_ACCEPTABLE, DECISION_REJECT, DECISION_NEEDS_REVIEW)

GRADE_NOT_APPLICABLE = "N/A"
COMMERCIAL_GRADES = ("A", "B", "C", GRADE_NOT_APPLICABLE)
SIZE_GRADES = {"Large": "A", "Medium": "B", "Small": "C"}

DAMAGE_STATUSES = ("Damaged", "Not Damaged", "Needs Review")
CLEANLINESS_STATUSES = ("Clean", "Stained", "Needs Review", "Unavailable", "Unknown")
CLEANLINESS_UNAVAILABLE = "Unavailable"

# Reason codes. Prefixes separate detector, crop/image and classifier causes.
REASON_DAMAGED = "damaged"
REASON_STAINED = "stained"
REASON_DETECTION_LOW_CONFIDENCE = "detection_low_confidence"
REASON_DAMAGE_NEEDS_REVIEW = "damage_needs_review"
REASON_DAMAGE_LOW_CONFIDENCE = "damage_low_confidence"
REASON_CLEANLINESS_UNAPPROVED = "cleanliness_unapproved"
REASON_CLEANLINESS_UNAVAILABLE = "cleanliness_unavailable"
REASON_CLEANLINESS_NEEDS_REVIEW = "cleanliness_needs_review"
REASON_CLEANLINESS_LOW_CONFIDENCE = "cleanliness_low_confidence"

# Raw classifier review reasons -> public reason codes.
_CROP_REASONS = {
    "empty_crop": "crop_empty",
    "invalid_crop": "crop_invalid",
    "small_crop": "crop_too_small",
}


@dataclass(frozen=True)
class QualityDecision:
    decision: str
    reasons: tuple[str, ...]

    @property
    def requires_review(self) -> bool:
        return self.decision == DECISION_NEEDS_REVIEW

    @property
    def reason_text(self) -> Optional[str]:
        return join_reasons(self.reasons)


def normalize_size(size: Optional[str]) -> str:
    if not size:
        return "Unknown"
    normalized = str(size).strip().title()
    if normalized in {"Small", "Medium", "Large"}:
        return normalized
    return "Unknown"


def normalize_damage_status(status: Optional[str]) -> str:
    if status in DAMAGE_STATUSES:
        return status
    return "Needs Review"


def normalize_cleanliness_status(status: Optional[str]) -> str:
    if status in CLEANLINESS_STATUSES:
        return status
    return "Unknown"


def classifier_reason(prefix: str, raw_reason: Optional[str]) -> Optional[str]:
    """Translate a classifier's raw review reason into a public reason code."""
    if not raw_reason:
        return None
    if raw_reason in _CROP_REASONS:
        return _CROP_REASONS[raw_reason]
    if raw_reason == "low_confidence":
        return f"{prefix}_low_confidence"
    return f"{prefix}_{raw_reason}"


def join_reasons(reasons: Iterable[Optional[str]]) -> Optional[str]:
    unique = list(dict.fromkeys(reason for reason in reasons if reason))
    return ";".join(unique) if unique else None


def split_reasons(text: Optional[str]) -> list[str]:
    if not text:
        return []
    return [part for part in text.split(";") if part]


def calculate_quality_decision(
    *,
    damage_status: Optional[str],
    cleanliness_status: Optional[str],
    cleanliness_release_approved: bool,
    damage_reason: Optional[str] = None,
    cleanliness_reason: Optional[str] = None,
    input_review_reasons: Iterable[Optional[str]] = (),
) -> QualityDecision:
    """Return the external-quality decision and the reasons that drove it."""
    input_reasons = [reason for reason in input_review_reasons if reason]
    damage_detail = classifier_reason("damage", damage_reason)
    # A crop that the damage classifier refused is an input problem too.
    if damage_detail in _CROP_REASONS.values():
        input_reasons.append(damage_detail)
    if input_reasons:
        return QualityDecision(
            DECISION_NEEDS_REVIEW, tuple(dict.fromkeys(input_reasons))
        )

    damage = normalize_damage_status(damage_status)
    cleanliness = normalize_cleanliness_status(cleanliness_status)

    if damage == "Damaged":
        return QualityDecision(DECISION_REJECT, (REASON_DAMAGED,))
    if cleanliness_release_approved and cleanliness == "Stained":
        return QualityDecision(DECISION_REJECT, (REASON_STAINED,))

    reasons: list[str] = []
    if damage == "Needs Review":
        reasons.append(damage_detail or REASON_DAMAGE_NEEDS_REVIEW)

    if not cleanliness_release_approved:
        reasons.append(REASON_CLEANLINESS_UNAPPROVED)
    elif cleanliness in {CLEANLINESS_UNAVAILABLE, "Unknown"}:
        reasons.append(REASON_CLEANLINESS_UNAVAILABLE)
    elif cleanliness == "Needs Review":
        reasons.append(
            classifier_reason("cleanliness", cleanliness_reason)
            or REASON_CLEANLINESS_NEEDS_REVIEW
        )

    if reasons:
        return QualityDecision(
            DECISION_NEEDS_REVIEW, tuple(dict.fromkeys(reasons))
        )
    return QualityDecision(DECISION_ACCEPTABLE, ())


def calculate_commercial_grade(
    decision: str,
    size: Optional[str],
    *,
    size_grades_enabled: bool,
) -> str:
    """Size grade A/B/C for an Acceptable egg with approved calibration."""
    if decision != DECISION_ACCEPTABLE or not size_grades_enabled:
        return GRADE_NOT_APPLICABLE
    return SIZE_GRADES.get(normalize_size(size), GRADE_NOT_APPLICABLE)


def public_grade(value: Optional[str]) -> Optional[str]:
    """Map stored grades, including legacy values, onto the A/B/C/N/A set.

    Rows written before the decision/grade split may hold ``AA`` (renamed to
    ``A``) or ``Reject`` (now a decision, so its commercial grade is ``N/A``).
    """
    if value is None:
        return None
    if value == "AA":
        return "A"
    if value in COMMERCIAL_GRADES:
        return value
    return GRADE_NOT_APPLICABLE
