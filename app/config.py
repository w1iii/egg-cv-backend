from functools import lru_cache
import os
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings


_module_path = Path(__file__).resolve()
_repository_root = _module_path.parents[2]
PROJECT_ROOT = Path(
    os.getenv(
        "PROJECT_ROOT",
        str(_repository_root if (_repository_root / "models").exists() else _module_path.parents[1]),
    )
)


class Settings(BaseSettings):
    APP_NAME: str = "Egg-CV API"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    VERSION: str = "1.0.0"

    DATABASE_URL: str = "postgresql://wii@localhost/eggcvdatabase"

    CORS_ORIGINS: str = (
        "http://localhost:5173,http://localhost:3000,http://localhost:5174"
    )

    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 50
    # The API has no authentication (single shared local operator). A
    # production deployment must acknowledge that explicitly.
    ACCEPT_UNAUTHENTICATED_ACCESS: bool = False
    # Schema changes go through Alembic. create_all() is opt-in for throwaway
    # development databases only; it never alters existing tables.
    AUTO_CREATE_SCHEMA: bool = False

    MODEL_PATH: str = str(
        PROJECT_ROOT / "egg_detection" / "finetune_egg_v1" / "weights" / "best.pt"
    )
    DAMAGE_MODEL_PATH: str = str(
        PROJECT_ROOT / "models" / "egg_grader_crop_resnet18_v1.pth"
    )
    # v3 columns carry decision/grade separately; older files keep their
    # own header and are never appended to with the new schema.
    EGG_CSV_PATH: str = str(PROJECT_ROOT / "reports" / "egg_statistics_v3.csv")
    CONFIDENCE_THRESHOLD: float = 0.75
    # Conservative MVP defaults; the lower validation thresholds favor recall
    # but produce too many artifact-driven false positives in web images.
    DAMAGE_DECISION_THRESHOLD: float = 0.65
    DAMAGE_REVIEW_THRESHOLD: float = 0.75
    # Hosted provider is experimental and sends egg crops to Roboflow.
    DAMAGE_PROVIDER: str = "local"
    ROBOFLOW_API_KEY: str = ""
    ROBOFLOW_DAMAGE_MODEL: str = "egg_crack/2"
    ROBOFLOW_TIMEOUT_SECONDS: float = 15.0
    CLEANLINESS_MODEL_PATH: str = str(
        PROJECT_ROOT / "models" / "cleanliness_reviewed_resnet18.pth"
    )
    # The validation-selected 0.04 threshold maximizes stained recall but
    # turns minor background artifacts into stained decisions.
    CLEANLINESS_DECISION_THRESHOLD: float = 0.60
    CLEANLINESS_REVIEW_THRESHOLD: float = 0.75
    # Set true only after physical validation and release sign-off. Until then
    # the UI must surface cleanliness results as beta, never as final grades.
    CLEANLINESS_RELEASE_APPROVED: bool = False
    DETECTION_CROP_PADDING: float = 0.1
    # The cleanliness checkpoint was trained on tight annotated egg crops,
    # unlike the damage checkpoint which uses detector crops with context.
    CLEANLINESS_CROP_PADDING: float = 0.0
    MIN_CROP_SIZE: int = 8
    DETECTION_DUPLICATE_IOU: float = 0.85
    # Input-reliability heuristics (app/services/image_quality.py). They only
    # route eggs to Needs Review and are not tuned on labeled data.
    IMAGE_MIN_SIDE_PX: int = 320
    IMAGE_MIN_SHARPNESS: float = 5.0
    EDGE_MARGIN_PX: int = 2
    DETECTION_OVERLAP_REVIEW: float = 0.30
    CROP_MIN_BRIGHTNESS: float = 35.0
    CROP_MAX_SATURATED_FRACTION: float = 0.25
    MM_PER_PIXEL: float = 0.09
    # Fitted coefficient for W = k * L * B^2; 0.0 means not yet calibrated.
    WEIGHT_K: float = 0.0
    # Only a non-"uncalibrated" value enables size-derived grades and mm
    # metrics. Changing MM_PER_PIXEL alone does not count as calibration.
    CALIBRATION_VERSION: str = "uncalibrated"
    ARUCO_MARKER_ID: int = 0
    ARUCO_SIDE_MM: float = 50.0
    # Enable only after the fixed physical setup has been validated.
    ARUCO_GRADES_APPROVED: bool = False

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.CORS_ORIGINS.split(",")
            if origin.strip()
        ]

    @model_validator(mode="after")
    def validate_production_security(self):
        if self.ENVIRONMENT != "production":
            return self
        if self.DEBUG:
            raise ValueError("DEBUG must be false in production")
        if "*" in self.cors_origins:
            raise ValueError("Wildcard CORS origin is not allowed in production")
        if "localhost" in self.DATABASE_URL or "127.0.0.1" in self.DATABASE_URL:
            raise ValueError(
                "DATABASE_URL points at localhost in production; set the "
                "deployment database URL"
            )
        if not self.ACCEPT_UNAUTHENTICATED_ACCESS:
            raise ValueError(
                "The API has no authentication. Set "
                "ACCEPT_UNAUTHENTICATED_ACCESS=true to acknowledge that "
                "anyone who can reach the service can upload, read and "
                "delete inspections, or keep it on a private network."
            )
        if self.AUTO_CREATE_SCHEMA:
            raise ValueError(
                "AUTO_CREATE_SCHEMA must be false in production; run "
                "'alembic upgrade head'"
            )
        return self

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
