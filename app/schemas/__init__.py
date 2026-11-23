from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator
from typing import Optional, List
from datetime import datetime

from app.services.grading_service import COMMERCIAL_GRADES, DECISIONS, public_grade


ALLOWED_GRADES = set(COMMERCIAL_GRADES)
ALLOWED_DECISIONS = set(DECISIONS)
ALLOWED_DAMAGE = {"Damaged", "Not Damaged", "Needs Review"}
ALLOWED_CLEANLINESS = {
    "Clean",
    "Stained",
    "Needs Review",
    "Unavailable",
    "Unknown",
}
ALLOWED_SIZES = {"Small", "Medium", "Large", "Unknown"}
ALLOWED_STATUSES = {"processing", "completed", "failed"}


class EggConfidence(BaseModel):
    detection: float = Field(ge=0, le=1)
    damage: Optional[float] = Field(default=None, ge=0, le=1)
    cleanliness: Optional[float] = Field(default=None, ge=0, le=1)


class EggGeometry(BaseModel):
    length_mm: Optional[float] = None
    breadth_mm: Optional[float] = None
    quality: Optional[str] = None


class ModelVersions(BaseModel):
    detector: Optional[str] = None
    damage: Optional[str] = None
    cleanliness: Optional[str] = None
    calibration: Optional[str] = None


class DetectionBoxResponse(BaseModel):
    id: int
    prediction_id: int
    egg_id: Optional[int] = None
    class_name: str
    detection_confidence: float = Field(
        validation_alias="confidence", exclude=True, ge=0, le=1
    )
    x1: int
    y1: int
    x2: int
    y2: int
    annotated: bool
    # External-quality decision (Acceptable / Reject / Needs Review). NULL
    # only for inspections recorded before decisions existed.
    decision: Optional[str] = None
    decision_reason: Optional[str] = None
    # Commercial size grade, always one of A / B / C / N/A.
    grade: Optional[str] = None
    size: Optional[str] = None
    weight_g: Optional[float] = None
    damage_status: Optional[str] = None
    damage_confidence: Optional[float] = Field(
        default=None, exclude=True, ge=0, le=1
    )
    cleanliness_status: Optional[str] = None
    cleanliness_confidence: Optional[float] = Field(
        default=None, exclude=True, ge=0, le=1
    )
    length_mm: Optional[float] = Field(default=None, exclude=True)
    breadth_mm: Optional[float] = Field(default=None, exclude=True)
    geometry_quality: Optional[str] = Field(default=None, exclude=True)
    detection_model_version: Optional[str] = Field(
        default=None, exclude=True
    )
    damage_model_version: Optional[str] = Field(
        default=None, exclude=True
    )
    cleanliness_model_version: Optional[str] = Field(
        default=None, exclude=True
    )
    calibration_version: Optional[str] = Field(
        default=None, exclude=True
    )
    requires_review: bool = False
    review_reason: Optional[str] = None

    @field_validator("grade")
    @classmethod
    def validate_grade(cls, value):
        return public_grade(value)

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, value):
        if value is not None and value not in ALLOWED_DECISIONS:
            raise ValueError(f"Unsupported decision: {value!r}")
        return value

    @field_validator("damage_status")
    @classmethod
    def validate_damage_status(cls, value):
        if value is not None and value not in ALLOWED_DAMAGE:
            raise ValueError(f"Unsupported damage status: {value!r}")
        return value

    @field_validator("cleanliness_status")
    @classmethod
    def validate_cleanliness_status(cls, value):
        if value is not None and value not in ALLOWED_CLEANLINESS:
            raise ValueError(f"Unsupported cleanliness status: {value!r}")
        return value

    @field_validator("size")
    @classmethod
    def validate_size(cls, value):
        if value is not None and value not in ALLOWED_SIZES:
            raise ValueError(f"Unsupported size: {value!r}")
        return value

    @computed_field
    @property
    def confidence(self) -> EggConfidence:
        return EggConfidence(
            detection=self.detection_confidence,
            damage=self.damage_confidence,
            cleanliness=self.cleanliness_confidence,
        )

    @computed_field
    @property
    def geometry(self) -> EggGeometry:
        return EggGeometry(
            length_mm=self.length_mm,
            breadth_mm=self.breadth_mm,
            quality=self.geometry_quality,
        )

    @computed_field
    @property
    def model_versions(self) -> ModelVersions:
        return ModelVersions(
            detector=self.detection_model_version,
            damage=self.damage_model_version,
            cleanliness=self.cleanliness_model_version,
            calibration=self.calibration_version,
        )

    class Config:
        from_attributes = True


class PredictionResponse(BaseModel):
    """One inspection: a single uploaded or captured still image."""

    id: int
    file_name: str
    file_type: str
    inspection_code: str
    captured_at: Optional[datetime] = None
    operator: Optional[str] = None
    inference_metadata: Optional[dict] = None
    status: str
    error_message: Optional[str] = None
    total_detections: int
    acceptable_count: int = 0
    reject_count: int = 0
    needs_review_count: int
    damaged_count: int
    not_damaged_count: int
    processing_time_ms: Optional[int]
    created_at: datetime
    completed_at: Optional[datetime]

    @field_validator("error_message")
    @classmethod
    def sanitize_error_message(cls, value):
        """Stored errors may hold internal detail; only allow-listed text leaves."""
        if not value:
            return None
        if value.startswith("CSV export failed"):
            return "CSV export failed"
        from app.utils.errors import public_error_message

        return public_error_message(Exception(value)) or "Prediction processing failed"

    @computed_field
    @property
    def decision_policy_version(self) -> Optional[str]:
        """``None`` marks a legacy inspection recorded before decisions."""
        return (self.inference_metadata or {}).get("decision_policy_version")

    @field_validator("status")
    @classmethod
    def validate_status(cls, value):
        if value not in ALLOWED_STATUSES:
            raise ValueError(f"Unsupported prediction status: {value!r}")
        return value

    class Config:
        from_attributes = True


class PredictionDetail(PredictionResponse):
    result_json: Optional[dict]
    annotated_path: Optional[str]
    detection_boxes: List[DetectionBoxResponse] = Field(default_factory=list)

    @computed_field
    @property
    def annotated_url(self) -> Optional[str]:
        """Browser-fetchable path for the annotated image, if one exists."""
        if not self.annotated_path:
            return None
        return f"/api/v1/predictions/{self.id}/download"

    class Config:
        from_attributes = True


class PredictionCreate(BaseModel):
    confidence_threshold: float = 0.75
    save_annotated: bool = True


class DashboardStats(BaseModel):
    total_predictions: int
    total_images: int
    total_detections: int
    acceptable_count: int = 0
    reject_count: int = 0
    needs_review_count: int = 0
    # Eggs from legacy inspections that carry no decision.
    undecided_count: int = 0
    damaged_percentage: float
    avg_processing_time_ms: float
    recent_predictions: List[PredictionResponse]


class ModelInfo(BaseModel):
    loaded: bool
    file: Optional[str] = None
    sha256: Optional[str] = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    damage_model_loaded: bool = False
    cleanliness_model_loaded: bool = False
    # Loaded is not the same as approved: an unapproved cleanliness model
    # can never produce a final Acceptable or Reject decision.
    cleanliness_release_approved: bool = False
    size_grades_approved: bool = False
    weight_estimation_enabled: bool = False
    database_connected: bool
    models: dict[str, ModelInfo] = Field(default_factory=dict)


class UserSettings(BaseModel):
    mm_per_pixel: float = Field(gt=0, le=10)

    class Config:
        from_attributes = True


class UserSettingsUpdate(BaseModel):
    mm_per_pixel: float = Field(gt=0, le=10)


class GradeDistribution(BaseModel):
    """Commercial size grades. Legacy 'Reject' grades count as N/A."""

    grade_a: int = 0
    grade_b: int = 0
    grade_c: int = 0
    grade_na: int = 0
    total: int = 0


class DecisionDistribution(BaseModel):
    acceptable: int = 0
    reject: int = 0
    needs_review: int = 0
    undecided: int = 0
    total: int = 0
