"""Save groups of existing inspections without changing their decisions."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Batch, BatchInspection, Prediction, User
from app.operator import get_operator

router = APIRouter(prefix="/batches", tags=["batches"])


class BatchRequest(BaseModel):
    inspection_ids: list[int] = Field(min_length=1, max_length=500)


def serialize(batch, db):
    ids = [item.prediction_id for item in batch.inspections]
    predictions = db.query(Prediction).filter(Prediction.id.in_(ids)).all()
    return {"id": batch.id, "created_at": batch.created_at, "inspection_ids": ids,
            "total_detections": sum(p.total_detections for p in predictions),
            "acceptable_count": sum(p.acceptable_count for p in predictions),
            "reject_count": sum(p.reject_count for p in predictions),
            "needs_review_count": sum(p.needs_review_count for p in predictions)}


@router.post("", status_code=201)
def create_batch(payload: BatchRequest, db: Session = Depends(get_db), operator: User = Depends(get_operator)):
    ids = payload.inspection_ids
    if len(set(ids)) != len(ids):
        raise HTTPException(422, "Duplicate inspection IDs")
    predictions = db.query(Prediction).filter(Prediction.id.in_(ids), Prediction.user_id == operator.id).all()
    if len(predictions) != len(ids):
        raise HTTPException(404, "Inspection not found")
    if any(p.status != "completed" for p in predictions):
        raise HTTPException(409, "Only completed inspections can be saved")
    batch = Batch(user_id=operator.id, inspections=[BatchInspection(prediction_id=i) for i in ids])
    db.add(batch)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(409, "An inspection already belongs to a batch") from exc
    db.refresh(batch)
    return serialize(batch, db)


@router.get("")
def list_batches(db: Session = Depends(get_db), operator: User = Depends(get_operator)):
    batches = db.query(Batch).filter_by(user_id=operator.id).order_by(Batch.id.desc()).limit(100).all()
    return [serialize(batch, db) for batch in batches]
