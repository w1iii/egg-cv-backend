"""Vocabulary and configuration contracts shared across layers."""

import re
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import CheckConstraint

from app.config import Settings
from app.ml.preprocessing import IMAGENET_MEAN, IMAGENET_STD, check_crop, to_rgb
from app.models import DetectionBox
from app.schemas import ALLOWED_DECISIONS, ALLOWED_GRADES
from app.services.csv_service import CSV_FIELDS, CsvSchemaMismatch, append_prediction
from app.services.grading_service import COMMERCIAL_GRADES, DECISIONS

FRONTEND_TYPES = Path(__file__).resolve().parents[2] / "egg-cv-frontend" / "src" / "types" / "index.ts"


def check_sql(name: str) -> str:
    for constraint in DetectionBox.__table__.constraints:
        if isinstance(constraint, CheckConstraint) and constraint.name == name:
            return str(constraint.sqltext)
    raise AssertionError(name)


def test_decision_vocabulary_matches_database_constraint():
    sql = check_sql("ck_detection_boxes_decision")
    assert set(re.findall(r"'([^']+)'", sql)) == set(DECISIONS) == ALLOWED_DECISIONS


def test_grade_vocabulary_is_commercial_only():
    assert ALLOWED_GRADES == set(COMMERCIAL_GRADES) == {"A", "B", "C", "N/A"}


def test_csv_has_no_tray_columns_and_carries_decision():
    assert not any("tray" in field for field in CSV_FIELDS)
    for field in ("inspection_code", "decision", "decision_reason", "grade"):
        assert field in CSV_FIELDS
    assert "provisional_grade" not in CSV_FIELDS


@pytest.mark.skipif(not FRONTEND_TYPES.is_file(), reason="frontend repo not checked out alongside")
def test_frontend_types_use_the_same_vocabulary():
    source = FRONTEND_TYPES.read_text(encoding="utf-8")

    def union(name):
        match = re.search(rf"export type {name} =([^;]+);", source)
        assert match, name
        return set(re.findall(r"'([^']+)'", match.group(1)))

    assert union("Decision") == set(DECISIONS)
    assert union("Grade") == set(COMMERCIAL_GRADES)
    assert "tray_id" not in source and "batch_recorded" not in source
    assert "inspection_code" in source


def test_csv_header_guard(tmp_path, monkeypatch):
    path = tmp_path / "old.csv"
    path.write_text("timestamp,tray_id,grade\n", encoding="utf-8")
    from app.services import csv_service

    monkeypatch.setattr(csv_service, "rows_for_prediction", lambda _p: [{f: "" for f in CSV_FIELDS}])
    with pytest.raises(CsvSchemaMismatch):
        append_prediction(object(), str(path))
    assert path.read_text(encoding="utf-8") == "timestamp,tray_id,grade\n"


PRODUCTION = dict(
    ENVIRONMENT="production",
    DATABASE_URL="postgresql://user@db.internal/eggs",
    CORS_ORIGINS="https://eggs.example",
)


def test_production_requires_explicit_unauthenticated_acknowledgement():
    with pytest.raises(ValueError, match="ACCEPT_UNAUTHENTICATED_ACCESS"):
        Settings(**PRODUCTION)
    assert Settings(**PRODUCTION, ACCEPT_UNAUTHENTICATED_ACCESS=True)


def test_production_forbids_auto_schema_creation():
    with pytest.raises(ValueError, match="AUTO_CREATE_SCHEMA"):
        Settings(**PRODUCTION, ACCEPT_UNAUTHENTICATED_ACCESS=True, AUTO_CREATE_SCHEMA=True)


def test_crop_checks_and_color_order():
    assert check_crop(None).reason == "empty_crop"
    assert check_crop(np.zeros((0, 0, 3), np.uint8)).reason == "empty_crop"
    assert check_crop(np.zeros((40, 40, 3), np.uint8)).reason == "small_crop"
    assert check_crop(np.zeros((60, 60, 5), np.uint8)).reason == "invalid_crop"
    assert check_crop(np.zeros((60, 60, 3), np.uint8)).ok
    bgr = np.zeros((60, 60, 3), np.uint8)
    bgr[..., 0] = 255  # blue in OpenCV order
    assert tuple(to_rgb(bgr)[0, 0]) == (0, 0, 255)
    assert to_rgb(np.zeros((60, 60), np.uint8)).shape == (60, 60, 3)
    assert to_rgb(np.zeros((60, 60, 4), np.uint8)).shape == (60, 60, 3)


def test_eval_transform_shape_and_normalization():
    torch = pytest.importorskip("torch")
    from app.ml.preprocessing import build_eval_transform

    transform = build_eval_transform(224)
    white = to_rgb(np.full((90, 130, 3), 255, np.uint8))
    tensor = transform(white)
    assert tuple(tensor.shape) == (3, 224, 224)
    assert tensor.dtype == torch.float32
    expected = [(1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    for channel, value in enumerate(expected):
        assert float(tensor[channel].mean()) == pytest.approx(value, abs=1e-5)
    black = transform(to_rgb(np.zeros((90, 130, 3), np.uint8)))
    assert float(black.min()) == pytest.approx(-IMAGENET_MEAN[0] / IMAGENET_STD[0], abs=1e-5)
