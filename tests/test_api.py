"""End-to-end upload -> result -> history -> delete through the HTTP API.

Uses the real FastAPI app, SQLAlchemy models (SQLite) and file handling with
deterministic fake models.
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.database import Base, engine
from app.main import app
from app.ml.model_registry import model_registry
from app.models import DetectionBox, Prediction
from app.database import SessionLocal
from app.services import prediction_service as prediction_module
from app.services.csv_service import CSV_FIELDS
from tests.fakes import cleanliness, damage, grid_boxes, make_predictor


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    csv_path = Path(settings.EGG_CSV_PATH)
    csv_path.unlink(missing_ok=True)
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    for stale in upload_dir.iterdir():
        if stale.is_file():
            stale.unlink()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


def install_models(boxes, damage_clf=None, clean_clf=None):
    model_registry.detector = make_predictor(boxes)
    model_registry.damage = damage_clf or damage()
    model_registry.cleanliness = clean_clf
    model_registry.detector_loaded = True
    model_registry.damage_loaded = True
    model_registry.cleanliness_loaded = clean_clf is not None


def png_bytes(height=480, width=640):
    rng = np.random.default_rng(3)
    image = (np.full((height, width, 3), 120, dtype=np.uint8)
             + rng.integers(0, 40, size=(height, width, 3), dtype=np.uint8))
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def upload(client, name="eggs.png", content=None, **params):
    return client.post(
        "/api/v1/predictions/upload",
        files={"file": (name, content or png_bytes(), "image/png")},
        params=params,
    )


def test_upload_returns_decision_and_grade_separately(client):
    install_models(grid_boxes(6, size=80, origin=40), clean_clf=cleanliness())
    response = upload(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["total_detections"] == 6
    assert body["inspection_code"]
    assert body["decision_policy_version"] == "decision-v1"
    assert body["needs_review_count"] == 6
    assert body["acceptable_count"] == 0 and body["reject_count"] == 0
    for key in ("tray_id", "batch_recorded", "okay_count", "batch_damaged_count"):
        assert key not in body
    egg = body["detection_boxes"][0]
    assert egg["decision"] == "Needs Review"
    assert egg["decision_reason"] == "cleanliness_unapproved"
    assert egg["grade"] == "N/A"
    assert "provisional_grade" not in egg
    assert egg["confidence"]["detection"] == pytest.approx(0.95)
    meta = body["inference_metadata"]
    assert meta["cleanliness_release_approved"] is False
    assert meta["cleanliness_model_available"] is True
    assert meta["detection_count_status"] == "ok"
    assert not any(key.startswith("tray") or key == "max_tray_eggs" for key in meta)


def test_csv_row_per_egg_with_new_columns(client):
    install_models(grid_boxes(3, size=80, origin=40), clean_clf=cleanliness())
    body = upload(client).json()
    with open(settings.EGG_CSV_PATH, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == CSV_FIELDS
    assert len(rows) == 3
    assert {row["decision"] for row in rows} == {"Needs Review"}
    assert {row["grade"] for row in rows} == {"N/A"}
    assert {row["inspection_code"] for row in rows} == {body["inspection_code"]}


def test_zero_detections_is_a_completed_inspection(client):
    install_models([])
    body = upload(client).json()
    assert body["status"] == "completed"
    assert body["total_detections"] == 0
    assert body["inference_metadata"]["detection_count_status"] == "empty"
    assert body["annotated_url"] is None


def test_history_lists_every_inspection_and_filters(client):
    install_models(grid_boxes(2, size=80, origin=40),
                   damage_clf=damage(queue=[("Damaged", 0.9, None)]),
                   clean_clf=cleanliness())
    rejected = upload(client).json()
    install_models(grid_boxes(1, size=80, origin=40), clean_clf=cleanliness())
    reviewed = upload(client).json()

    all_items = client.get("/api/v1/predictions").json()
    assert {item["id"] for item in all_items} == {rejected["id"], reviewed["id"]}

    rejects = client.get("/api/v1/predictions", params={"decision": "Reject"}).json()
    assert [item["id"] for item in rejects] == [rejected["id"]]
    by_reason = client.get(
        "/api/v1/predictions", params={"review_reason": "cleanliness_unapproved"}
    ).json()
    assert {item["id"] for item in by_reason} == {rejected["id"], reviewed["id"]}
    assert client.get("/api/v1/predictions", params={"decision": "Bogus"}).status_code == 422
    assert client.get("/api/v1/records").status_code == 200


def test_batch_record_endpoint_is_gone(client):
    install_models(grid_boxes(1, size=80, origin=40))
    prediction_id = upload(client).json()["id"]
    response = client.post(
        f"/api/v1/predictions/{prediction_id}/record",
        json={"okay_count": 1, "stained_count": 0, "damaged_count": 0},
    )
    assert response.status_code in (404, 405)


def test_save_batch_groups_inspections_and_prevents_duplicate_membership(client):
    install_models(grid_boxes(1, size=80, origin=40), clean_clf=cleanliness())
    first = upload(client).json()
    second = upload(client).json()
    ids = [first["id"], second["id"]]
    response = client.post("/api/v1/batches", json={"inspection_ids": ids})
    assert response.status_code == 201, response.text
    batch = response.json()
    assert set(batch["inspection_ids"]) == set(ids)
    assert batch["total_detections"] == 2
    assert batch["needs_review_count"] == 2
    assert client.get("/api/v1/batches").json()[0]["id"] == batch["id"]
    assert client.post("/api/v1/batches", json={"inspection_ids": ids}).status_code == 409
    assert client.post("/api/v1/batches", json={"inspection_ids": [999999]}).status_code == 404
    assert client.delete(f"/api/v1/predictions/{ids[0]}").status_code == 409
    assert client.get(f"/api/v1/predictions/{ids[0]}").status_code == 200


def test_delete_removes_row_and_files(client):
    install_models(grid_boxes(1, size=80, origin=40))
    body = upload(client).json()
    with SessionLocal() as db:
        prediction = db.get(Prediction, body["id"])
        paths = [Path(prediction.file_path), Path(prediction.annotated_path)]
    assert all(path.is_file() for path in paths)
    assert client.get(f"/api/v1/predictions/{body['id']}/download").status_code == 200
    assert client.delete(f"/api/v1/predictions/{body['id']}").status_code == 200
    assert not any(path.exists() for path in paths)
    assert client.get(f"/api/v1/predictions/{body['id']}").status_code == 404
    with SessionLocal() as db:
        assert db.query(DetectionBox).count() == 0


def test_delete_never_touches_files_outside_upload_dir(client, tmp_path):
    outside = tmp_path / "keep.png"
    outside.write_bytes(b"x")
    with SessionLocal() as db:
        from app.operator import get_operator

        operator = get_operator(db)
        row = Prediction(user_id=operator.id, file_name="keep.png", file_path=str(outside),
                         file_type="image", status="completed")
        db.add(row)
        db.commit()
        row_id = row.id
    assert client.delete(f"/api/v1/predictions/{row_id}").status_code == 200
    assert outside.exists()


def test_model_failure_marks_inspection_failed(client):
    class Exploding:
        model_version = "boom.pth"

        def classify_crop(self, crop):
            raise RuntimeError("inference crashed at /secret/path")

    install_models(grid_boxes(2, size=80, origin=40), damage_clf=Exploding())
    response = upload(client)
    assert response.status_code == 500
    assert "secret" not in response.text
    with SessionLocal() as db:
        prediction = db.query(Prediction).one()
        assert prediction.status == "failed"
        assert prediction.annotated_path is None
        assert db.query(DetectionBox).count() == 0
        prediction_id = prediction.id
    # The stored internal message never reaches API clients.
    listed = client.get("/api/v1/predictions").json()[0]
    assert listed["status"] == "failed"
    assert listed["error_message"] == "Prediction processing failed"
    assert "secret" not in client.get(f"/api/v1/predictions/{prediction_id}").text


def test_database_commit_failure_rolls_back_and_cleans_annotation(client):
    # An out-of-vocabulary status violates a CHECK constraint at commit time.
    install_models(grid_boxes(2, size=80, origin=40),
                   damage_clf=damage(default=("Bogus", 0.9, None)))
    response = upload(client)
    assert response.status_code == 500
    upload_dir = Path(settings.UPLOAD_DIR)
    assert not list(upload_dir.glob("annotated_*"))
    with SessionLocal() as db:
        prediction = db.query(Prediction).one()
        assert prediction.status == "failed"
        assert db.query(DetectionBox).count() == 0


def test_csv_failure_keeps_inspection_and_records_error(client, monkeypatch):
    def broken(_prediction):
        raise OSError("disk full")

    monkeypatch.setattr(prediction_module, "append_prediction", broken)
    install_models(grid_boxes(2, size=80, origin=40))
    body = upload(client).json()
    assert body["status"] == "completed"
    assert body["error_message"] == "CSV export failed"  # no OS detail leaked
    assert len(body["detection_boxes"]) == 2


def test_annotation_write_failure_is_not_fatal(client, monkeypatch):
    monkeypatch.setattr(prediction_module.cv2, "imwrite", lambda *args, **kwargs: False)
    install_models(grid_boxes(1, size=80, origin=40))
    body = upload(client).json()
    assert body["status"] == "completed"
    assert body["annotated_url"] is None
    assert body["inference_metadata"]["annotation_error"] == "annotated_image_write_failed"


def test_models_unavailable_fails_cleanly(client):
    install_models(grid_boxes(1))
    model_registry.damage_loaded = False
    response = upload(client)
    assert response.status_code == 500
    with SessionLocal() as db:
        assert db.query(Prediction).one().status == "failed"


@pytest.mark.parametrize(
    "name,content,status",
    [
        ("eggs.gif", b"GIF89a", 400),
        ("eggs.png", b"not really a png", 400),
        ("eggs.webp", b"RIFF\x00\x00\x00\x00WAVEfmt ", 400),
    ],
)
def test_invalid_uploads_are_rejected(client, name, content, status):
    install_models(grid_boxes(1))
    assert upload(client, name=name, content=content).status_code == status


def test_undecodable_image_is_reported(client):
    install_models(grid_boxes(1))
    # Valid PNG signature, truncated body.
    response = upload(client, content=b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    assert response.status_code == 400
    assert "could not be decoded" in response.json()["detail"]


def test_oversized_upload_is_rejected_before_parsing(client):
    install_models(grid_boxes(1))
    size = settings.MAX_FILE_SIZE_MB * 1024 * 1024 + 200 * 1024
    response = client.post(
        "/api/v1/predictions/upload",
        content=b"0" * 16,
        headers={"content-type": "multipart/form-data; boundary=x", "content-length": str(size)},
    )
    assert response.status_code == 413


def test_uploads_are_not_served_statically(client):
    install_models(grid_boxes(1, size=80, origin=40))
    upload(client)
    stored = next(Path(settings.UPLOAD_DIR).glob("*.png"))
    assert client.get(f"/uploads/{stored.name}").status_code == 404


def test_dashboard_health_and_distributions(client):
    install_models(grid_boxes(3, size=80, origin=40),
                   damage_clf=damage(queue=[("Damaged", 0.9, None)]),
                   clean_clf=cleanliness())
    upload(client)
    stats = client.get("/api/v1/dashboard/stats").json()
    assert stats["reject_count"] == 1 and stats["needs_review_count"] == 2
    decisions = client.get("/api/v1/dashboard/decisions").json()
    assert decisions == {"acceptable": 0, "reject": 1, "needs_review": 2, "undecided": 0, "total": 3}
    grades = client.get("/api/v1/dashboard/grades").json()
    assert grades["grade_na"] == 3 and "grade_reject" not in grades
    health = client.get("/api/v1/dashboard/health").json()
    assert health["cleanliness_model_loaded"] is True
    assert health["cleanliness_release_approved"] is False
    assert health["size_grades_approved"] is False
    assert health["weight_estimation_enabled"] is False
    assert set(health["models"]) == {"detector", "damage", "cleanliness"}


def test_legacy_rows_remain_readable(client):
    """Rows written before the decision split keep their data and still load."""
    with SessionLocal() as db:
        from app.operator import get_operator

        operator = get_operator(db)
        row = Prediction(user_id=operator.id, file_name="old.png", file_path="old.png",
                         file_type="image", status="completed", total_detections=1,
                         batch_recorded=True, okay_count=1)
        db.add(row)
        db.flush()
        db.add(DetectionBox(prediction_id=row.id, class_name="egg", confidence=0.9,
                            x1=0, y1=0, x2=10, y2=10, grade="Reject",
                            provisional_grade="AA", damage_status="Damaged"))
        db.commit()
        row_id = row.id
    body = client.get(f"/api/v1/predictions/{row_id}").json()
    assert body["decision_policy_version"] is None
    egg = body["detection_boxes"][0]
    assert egg["decision"] is None
    assert egg["grade"] == "N/A"
    assert egg["damage_status"] == "Damaged"
    with SessionLocal() as db:
        assert db.get(DetectionBox, egg["id"]).grade == "Reject"  # stored value untouched
    assert client.get("/api/v1/dashboard/decisions").json()["undecided"] == 1
