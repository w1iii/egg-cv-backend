from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Literal, Optional
from pathlib import Path
import mimetypes

from app.database import get_db
from app.models import User, Prediction, BatchInspection
from app.operator import get_operator
from app.config import settings
from app.schemas import PredictionResponse, PredictionDetail
from app.services.file_service import file_service
from app.services.prediction_service import (
    PredictionService,
    is_within_directory,
    prediction_service,
)
from app.ml.model_registry import model_registry
from app.utils.errors import public_error_message

router = APIRouter(prefix="/predictions", tags=["predictions"])

DecisionFilter = Literal["Acceptable", "Reject", "Needs Review"]


@router.post("/upload", response_model=PredictionDetail)
def upload_file(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.75, ge=0.1, le=0.99),
    save_annotated: bool = Query(True),
    db: Session = Depends(get_db),
    operator: User = Depends(get_operator)
):
    if not file_service.is_allowed_file(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image type. Allowed: jpg, jpeg, png, webp, bmp"
        )

    try:
        file_path, file_type = file_service.save_file_sync(file)
        prediction = prediction_service.create_prediction(
            db,
            operator.id,
            file.filename,
            file_path,
            file_type,
            operator=operator.username,
        )
        if (
            not model_registry.detector_loaded
            or not model_registry.damage_loaded
            or model_registry.detector is None
            or model_registry.damage is None
        ):
            raise RuntimeError("Required prediction models are not available")
        cleanliness_classifier = (
            model_registry.cleanliness
            if model_registry.cleanliness_loaded
            else None
        )
        pred_service = PredictionService(
            model_registry.detector,
            model_registry.damage,
            cleanliness_classifier,
        )
        prediction = pred_service.run_prediction(
            db,
            prediction.id,
            confidence_threshold,
            save_annotated,
        )
    except ValueError as e:
        # Only remove the upload when no prediction row references it.
        if "file_path" in locals() and "prediction" not in locals():
            file_service.delete_file(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=public_error_message(e) or "Invalid request",
        ) from e
    except Exception as e:
        if "file_path" in locals() and "prediction" not in locals():
            file_service.delete_file(file_path)
        if "prediction" in locals():
            prediction.status = "failed"
            prediction.error_message = "Prediction processing failed"
            db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction processing failed",
        ) from e
    
    return prediction


@router.get("", response_model=List[PredictionResponse])
def get_predictions(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    decision: Optional[DecisionFilter] = Query(
        None, description="Only inspections with at least one egg of this decision"
    ),
    review_reason: Optional[str] = Query(
        None,
        max_length=64,
        pattern=r"^[a-z_]+$",
        description="Only inspections with an egg carrying this reason code",
    ),
    db: Session = Depends(get_db),
    operator: User = Depends(get_operator)
):
    return prediction_service.get_predictions(
        db, operator.id, skip, limit, decision=decision, review_reason=review_reason
    )


@router.get("/{prediction_id}", response_model=PredictionDetail)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    operator: User = Depends(get_operator)
):
    prediction = prediction_service.get_prediction_detail(
        db, prediction_id, operator.id
    )
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
    return prediction


@router.delete("/{prediction_id}")
def delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    operator: User = Depends(get_operator)
):
    if db.query(BatchInspection).filter_by(prediction_id=prediction_id).first():
        raise HTTPException(status_code=409, detail="Cannot delete an inspection in a saved batch")
    success = prediction_service.delete_prediction(
        db, prediction_id, operator.id
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
    return {"message": "Prediction deleted"}


@router.get("/{prediction_id}/download")
def download_annotated(
    prediction_id: int,
    db: Session = Depends(get_db),
    operator: User = Depends(get_operator)
):
    prediction = prediction_service.get_prediction_detail(
        db, prediction_id, operator.id
    )
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
    if not prediction.annotated_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotated file not found"
        )
    annotated_path = Path(prediction.annotated_path)
    if (
        not is_within_directory(annotated_path, settings.UPLOAD_DIR)
        or not annotated_path.is_file()
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotated image is no longer available"
        )
    media_type = mimetypes.guess_type(annotated_path.name)[0] or "application/octet-stream"
    return FileResponse(
        str(annotated_path),
        media_type=media_type,
        filename=f"annotated_{prediction.file_name}",
        headers={"X-Content-Type-Options": "nosniff"},
    )
