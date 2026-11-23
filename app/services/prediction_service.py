import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import cv2
from sqlalchemy import exists
from sqlalchemy.orm import Session

from app.config import settings
from app.ml.cleanliness_inference import CleanlinessClassifier
from app.ml.damage_inference import DamageClassifier
from app.ml.yolo_inference import YOLOPredictor
from app.models import DetectionBox, Prediction
from app.services.csv_service import append_prediction
from app.services.inspection_pipeline import (
    CALIBRATION_VERSION_ARUCO,
    CALIBRATION_VERSION_NONE,
    crop_detection,
    inspect_image,
)

logger = logging.getLogger(__name__)

# Bump when the decision/grade rules in grading_service change meaning.
DECISION_POLICY_VERSION = "decision-v1"

__all__ = [
    "DECISION_POLICY_VERSION",
    "PredictionService",
    "crop_detection",
    "is_within_directory",
    "prediction_service",
]


def is_within_directory(path: str | Path, directory: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except (OSError, ValueError):
        return False


def _remove_upload(path: Optional[str]) -> None:
    """Delete a stored upload, but never anything outside UPLOAD_DIR."""
    if not path:
        return
    if not is_within_directory(path, settings.UPLOAD_DIR):
        logger.warning("Refusing to delete file outside upload dir: %s", path)
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not delete upload %s", path, exc_info=True)


class PredictionService:
    def __init__(
        self,
        predictor: Optional[YOLOPredictor] = None,
        damage_classifier: Optional[DamageClassifier] = None,
        cleanliness_classifier: Optional[CleanlinessClassifier] = None,
    ):
        self.predictor = predictor or YOLOPredictor()
        self.damage_classifier = damage_classifier
        self.cleanliness_classifier = cleanliness_classifier

    def create_prediction(
        self,
        db: Session,
        user_id: int,
        file_name: str,
        file_path: str,
        file_type: str,
        operator: Optional[str] = None,
    ) -> Prediction:
        prediction = Prediction(
            user_id=user_id,
            file_name=file_name,
            file_type=file_type,
            file_path=file_path,
            inspection_code=uuid4().hex,
            captured_at=datetime.utcnow(),
            operator=operator,
            status="processing",
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return prediction

    def run_prediction(
        self,
        db: Session,
        prediction_id: int,
        confidence: float,
        save_annotated: bool,
    ) -> Prediction:
        start_time = time.time()

        # Release the pooled DB connection before CPU inference so concurrent
        # uploads never pin pool slots for the whole prediction.
        db.close()

        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not prediction:
            raise ValueError("Prediction not found")

        annotated_path: Optional[str] = None
        try:
            if prediction.file_type != "image":
                raise ValueError("Video prediction is not supported yet")

            img = cv2.imread(prediction.file_path)
            if img is None:
                raise ValueError("Uploaded image could not be decoded")
            result = inspect_image(
                img,
                predictor=self.predictor,
                damage_classifier=self.damage_classifier,
                cleanliness_classifier=self.cleanliness_classifier,
                detection_threshold=confidence,
            )

            if save_annotated and result["detections"]:
                annotated = self.predictor.draw_boxes(img, result["detections"])
                source = Path(prediction.file_path)
                candidate = str(source.parent / f"annotated_{source.stem}{source.suffix}")
                if cv2.imwrite(candidate, annotated):
                    annotated_path = candidate
                else:
                    result["annotation_error"] = "annotated_image_write_failed"

            summary = result["summary"]
            prediction.inference_metadata = self._inference_metadata(confidence, result)
            for det in result["detections"]:
                db.add(self._detection_box(prediction.id, det, annotated_path))

            prediction.status = "completed"
            prediction.error_message = None
            prediction.result_json = result
            prediction.annotated_path = annotated_path
            prediction.total_detections = summary["total"]
            prediction.acceptable_count = summary["acceptable"]
            prediction.reject_count = summary["reject"]
            prediction.needs_review_count = summary["needs_review"]
            prediction.damaged_count = summary["damaged"]
            prediction.not_damaged_count = summary["not_damaged"]
            prediction.processing_time_ms = int((time.time() - start_time) * 1000)
            prediction.completed_at = datetime.utcnow()
            # Boxes, counts and status land in one transaction.
            db.commit()
            db.refresh(prediction)
        except Exception as exc:
            # A failed commit leaves the session unusable until rolled back;
            # drop partial boxes and any annotated file, then mark failure.
            db.rollback()
            _remove_upload(annotated_path)
            failed = db.query(Prediction).filter(Prediction.id == prediction_id).first()
            if failed is not None:
                failed.status = "failed"
                failed.error_message = str(exc)[:1000]
                failed.annotated_path = None
                db.commit()
            raise

        # The CSV is a secondary export; its failure must not undo the
        # committed inspection, but it is recorded on the row.
        try:
            append_prediction(prediction)
        except Exception as exc:
            logger.warning("CSV export failed for prediction %s", prediction.id, exc_info=True)
            prediction.error_message = f"CSV export failed: {exc}"[:1000]
            db.commit()

        return prediction

    @staticmethod
    def _detection_box(
        prediction_id: int, det: dict[str, Any], annotated_path: Optional[str]
    ) -> DetectionBox:
        return DetectionBox(
            prediction_id=prediction_id,
            egg_id=det.get("egg_id"),
            class_name=det["class"],
            confidence=det["confidence"],
            x1=det["bbox"][0],
            y1=det["bbox"][1],
            x2=det["bbox"][2],
            y2=det["bbox"][3],
            annotated=annotated_path is not None,
            size=det.get("size"),
            length_mm=det.get("length_mm"),
            breadth_mm=det.get("breadth_mm"),
            geometry_quality=det.get("geometry_quality"),
            weight_g=det.get("weight_g"),
            decision=det.get("decision"),
            decision_reason=det.get("decision_reason"),
            grade=det.get("grade"),
            requires_review=bool(det.get("requires_review")),
            review_reason=det.get("review_reason"),
            damage_status=det.get("damage_status"),
            damage_confidence=det.get("damage_confidence"),
            damage_model_version=det.get("damage_model_version"),
            cleanliness_status=det.get("cleanliness_status"),
            cleanliness_confidence=det.get("cleanliness_confidence"),
            cleanliness_model_version=det.get("cleanliness_model_version"),
            detection_model_version=det.get("detection_model_version"),
            calibration_version=det.get("calibration_version"),
        )

    def _inference_metadata(self, confidence: float, result: dict) -> dict:
        damage_threshold = getattr(
            self.damage_classifier, "damaged_threshold", settings.DAMAGE_DECISION_THRESHOLD
        )
        damage_review = getattr(
            self.damage_classifier, "review_threshold", settings.DAMAGE_REVIEW_THRESHOLD
        )
        cleanliness = self.cleanliness_classifier
        sizing_available = result["calibration"]["status"] == "available"
        return {
            "decision_policy_version": DECISION_POLICY_VERSION,
            "detector_model_version": getattr(self.predictor, "model_version", None),
            "detector_sha256": getattr(self.predictor, "checkpoint_sha256", None),
            "damage_model_version": getattr(self.damage_classifier, "model_version", None),
            "damage_sha256": getattr(self.damage_classifier, "checkpoint_sha256", None),
            "cleanliness_model_available": cleanliness is not None,
            "cleanliness_model_version": getattr(cleanliness, "model_version", None),
            "cleanliness_sha256": getattr(cleanliness, "checkpoint_sha256", None),
            "cleanliness_release_approved": result["cleanliness_release_approved"],
            "calibration_version": (
                CALIBRATION_VERSION_ARUCO if sizing_available else CALIBRATION_VERSION_NONE
            ),
            "calibration": result["calibration"],
            "sizing_available": sizing_available,
            "size_calibrated": result["size_grades_enabled"],
            "size_grades_approved": settings.ARUCO_GRADES_APPROVED,
            "weight_calibrated": False,
            "detection_threshold": confidence,
            "damage_decision_threshold": damage_threshold,
            "damage_review_threshold": damage_review,
            "cleanliness_decision_threshold": (
                None
                if cleanliness is None
                else getattr(cleanliness, "stained_threshold", settings.CLEANLINESS_DECISION_THRESHOLD)
            ),
            "cleanliness_review_threshold": (
                None
                if cleanliness is None
                else getattr(cleanliness, "review_threshold", settings.CLEANLINESS_REVIEW_THRESHOLD)
            ),
            "crop_padding": settings.DETECTION_CROP_PADDING,
            "cleanliness_crop_padding": settings.CLEANLINESS_CROP_PADDING,
            "detection_count_status": result.get("count_status"),
            "raw_detection_count": result.get("raw_detection_count", 0),
            "duplicate_boxes_removed": result.get("duplicate_boxes_removed", 0),
            "invalid_boxes_removed": result.get("invalid_boxes_removed", 0),
            "image_quality": result.get("image_quality"),
            "review_reason_counts": result["summary"]["review_reason_counts"],
            "annotation_error": result.get("annotation_error"),
        }

    def get_predictions(
        self,
        db: Session,
        user_id: int,
        skip: int = 0,
        limit: int = 20,
        decision: Optional[str] = None,
        review_reason: Optional[str] = None,
    ) -> list[Prediction]:
        query = db.query(Prediction).filter(Prediction.user_id == user_id)
        if decision is not None:
            query = query.filter(
                exists().where(
                    DetectionBox.prediction_id == Prediction.id,
                    DetectionBox.decision == decision,
                )
            )
        if review_reason:
            query = query.filter(
                exists().where(
                    DetectionBox.prediction_id == Prediction.id,
                    DetectionBox.decision_reason.contains(review_reason),
                )
            )
        return (
            query.order_by(Prediction.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_prediction_detail(
        self, db: Session, prediction_id: int, user_id: int
    ) -> Optional[Prediction]:
        return (
            db.query(Prediction)
            .filter(
                Prediction.id == prediction_id,
                Prediction.user_id == user_id,
            )
            .first()
        )

    def delete_prediction(
        self, db: Session, prediction_id: int, user_id: int
    ) -> bool:
        """Delete the inspection row, its eggs, the upload and annotation.

        The CSV export is an append-only audit log and is not rewritten.
        """
        prediction = self.get_prediction_detail(db, prediction_id, user_id)
        if not prediction:
            return False
        paths = [prediction.file_path, prediction.annotated_path]
        db.delete(prediction)
        db.commit()
        for path in paths:
            _remove_upload(path)
        return True


prediction_service = PredictionService()
