from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.ml.model_registry import model_registry
from app.models import DetectionBox, Prediction, User
from app.operator import get_operator
from app.schemas import (
    DashboardStats,
    DecisionDistribution,
    GradeDistribution,
    HealthResponse,
    ModelInfo,
)
from app.services.grading_service import public_grade

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _decision_counts(db: Session, user_id: int) -> DecisionDistribution:
    prediction_ids = select(Prediction.id).where(Prediction.user_id == user_id)
    rows = (
        db.query(DetectionBox.decision, func.count(DetectionBox.id))
        .filter(DetectionBox.prediction_id.in_(prediction_ids))
        .group_by(DetectionBox.decision)
        .all()
    )
    counts = {decision: count for decision, count in rows}
    distribution = DecisionDistribution(
        acceptable=counts.get("Acceptable", 0),
        reject=counts.get("Reject", 0),
        needs_review=counts.get("Needs Review", 0),
        undecided=counts.get(None, 0),
    )
    distribution.total = (
        distribution.acceptable
        + distribution.reject
        + distribution.needs_review
        + distribution.undecided
    )
    return distribution


@router.get("/stats", response_model=DashboardStats)
def get_dashboard_stats(
    db: Session = Depends(get_db),
    operator: User = Depends(get_operator)
):
    mine = Prediction.user_id == operator.id
    total = db.query(Prediction).filter(mine).count()
    total_images = db.query(Prediction).filter(
        mine, Prediction.file_type == "image"
    ).count()
    total_detections = db.query(func.sum(Prediction.total_detections)).filter(
        mine
    ).scalar() or 0
    damaged = db.query(func.sum(Prediction.damaged_count)).filter(
        mine
    ).scalar() or 0
    damaged_percentage = (
        damaged / total_detections * 100 if total_detections > 0 else 0
    )
    avg_time = db.query(func.avg(Prediction.processing_time_ms)).filter(
        mine, Prediction.status == "completed"
    ).scalar() or 0
    recent = (
        db.query(Prediction)
        .filter(mine)
        .order_by(Prediction.created_at.desc())
        .limit(5)
        .all()
    )
    decisions = _decision_counts(db, operator.id)

    return DashboardStats(
        total_predictions=total,
        total_images=total_images,
        total_detections=total_detections,
        acceptable_count=decisions.acceptable,
        reject_count=decisions.reject,
        needs_review_count=decisions.needs_review,
        undecided_count=decisions.undecided,
        damaged_percentage=round(damaged_percentage, 1),
        avg_processing_time_ms=round(float(avg_time), 1),
        recent_predictions=recent,
    )


@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    database_connected = False
    try:
        db.execute(text("SELECT 1"))
        database_connected = True
    except Exception:
        pass

    # Cleanliness is optional: missing it degrades results (every egg goes
    # to review), not API readiness. Detector, damage classifier and DB are
    # required.
    ready = (
        model_registry.detector_loaded
        and model_registry.damage_loaded
        and database_connected
    )

    def info(model, loaded: bool) -> ModelInfo:
        return ModelInfo(
            loaded=loaded,
            file=getattr(model, "model_version", None),
            sha256=getattr(model, "checkpoint_sha256", None),
        )

    return HealthResponse(
        status="healthy" if ready else "degraded",
        model_loaded=model_registry.detector_loaded,
        damage_model_loaded=model_registry.damage_loaded,
        cleanliness_model_loaded=model_registry.cleanliness_loaded,
        cleanliness_release_approved=settings.CLEANLINESS_RELEASE_APPROVED,
        size_grades_approved=settings.ARUCO_GRADES_APPROVED,
        weight_estimation_enabled=False,
        database_connected=database_connected,
        models={
            "detector": info(model_registry.detector, model_registry.detector_loaded),
            "damage": info(model_registry.damage, model_registry.damage_loaded),
            "cleanliness": info(
                model_registry.cleanliness, model_registry.cleanliness_loaded
            ),
        },
    )


@router.get("/grades", response_model=GradeDistribution)
def get_grade_distribution(
    db: Session = Depends(get_db),
    operator: User = Depends(get_operator)
):
    prediction_ids = select(Prediction.id).where(
        Prediction.user_id == operator.id
    )
    grade_counts = (
        db.query(DetectionBox.grade, func.count(DetectionBox.id))
        .filter(DetectionBox.prediction_id.in_(prediction_ids))
        .group_by(DetectionBox.grade)
        .all()
    )
    grades = {"A": 0, "B": 0, "C": 0, "N/A": 0}
    for grade, count in grade_counts:
        mapped = public_grade(grade)
        if mapped in grades:
            grades[mapped] += count

    return GradeDistribution(
        grade_a=grades["A"],
        grade_b=grades["B"],
        grade_c=grades["C"],
        grade_na=grades["N/A"],
        total=sum(grades.values()),
    )


@router.get("/decisions", response_model=DecisionDistribution)
def get_decision_distribution(
    db: Session = Depends(get_db),
    operator: User = Depends(get_operator)
):
    return _decision_counts(db, operator.id)
