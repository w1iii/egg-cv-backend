from typing import Optional

from app.config import settings
from app.ml.cleanliness_inference import CleanlinessClassifier
from app.ml.damage_inference import DamageClassifier
from app.ml.roboflow_damage import RoboflowDamageClassifier
from app.ml.yolo_inference import YOLOPredictor


class ModelRegistry:
    def __init__(self) -> None:
        self.detector: Optional[YOLOPredictor] = None
        self.damage: Optional[DamageClassifier] = None
        self.cleanliness: Optional[CleanlinessClassifier] = None
        self.detector_loaded = False
        self.damage_loaded = False
        self.cleanliness_loaded = False

    def load(self) -> tuple[bool, bool]:
        self.detector = YOLOPredictor()
        detector_loaded = self.detector.load_model()
        if settings.DAMAGE_PROVIDER == "roboflow":
            self.damage = RoboflowDamageClassifier(
                api_key=settings.ROBOFLOW_API_KEY,
                model_id=settings.ROBOFLOW_DAMAGE_MODEL,
                threshold=settings.DAMAGE_DECISION_THRESHOLD,
                timeout=settings.ROBOFLOW_TIMEOUT_SECONDS,
            )
        elif settings.DAMAGE_PROVIDER == "local":
            self.damage = DamageClassifier(
                damaged_threshold=settings.DAMAGE_DECISION_THRESHOLD,
                review_threshold=settings.DAMAGE_REVIEW_THRESHOLD,
            )
        else:
            raise ValueError("Unknown DAMAGE_PROVIDER")
        damage_loaded = self.damage.load_model()
        self.cleanliness = CleanlinessClassifier(
            stained_threshold=settings.CLEANLINESS_DECISION_THRESHOLD,
            review_threshold=settings.CLEANLINESS_REVIEW_THRESHOLD,
        )
        cleanliness_loaded = self.cleanliness.load_model()
        self.detector_loaded = detector_loaded
        self.damage_loaded = damage_loaded
        self.cleanliness_loaded = cleanliness_loaded
        return detector_loaded, damage_loaded


model_registry = ModelRegistry()
