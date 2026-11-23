import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app.config import settings
from app.database import Base, SessionLocal, engine
from app.models import Prediction, User

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "purge_uploads.py"


@pytest.fixture
def purge_module():
    spec = importlib.util.spec_from_file_location("purge_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rows(tmp_path):
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    old_file, old_annotated, new_file = (upload_dir / name for name in ("old.png", "annotated_old.png", "new.png"))
    outside = tmp_path / "outside.png"
    for path in (old_file, old_annotated, new_file, outside):
        path.write_bytes(b"x" * 10)
    now = datetime(2026, 11, 23)
    with SessionLocal() as db:
        user = User(email="r@localhost", username="r", password_hash="!")
        db.add(user)
        db.flush()
        db.add_all([
            Prediction(user_id=user.id, file_name="old", file_path=str(old_file), annotated_path=str(old_annotated),
                       file_type="image", status="completed", created_at=now - timedelta(days=100)),
            Prediction(user_id=user.id, file_name="new", file_path=str(new_file),
                       file_type="image", status="completed", created_at=now - timedelta(days=5)),
            Prediction(user_id=user.id, file_name="outside", file_path=str(outside),
                       file_type="image", status="completed", created_at=now - timedelta(days=100)),
        ])
        db.commit()
    yield {"now": now, "old": old_file, "annotated": old_annotated, "new": new_file, "outside": outside}
    Base.metadata.drop_all(bind=engine)


def test_dry_run_deletes_nothing(purge_module, rows):
    summary = purge_module.purge(90, apply=False, now=rows["now"])
    assert summary["files"] == 2 and summary["skipped_outside_upload_dir"] == 1
    assert all(rows[key].exists() for key in ("old", "annotated", "new", "outside"))


def test_apply_removes_only_old_files_inside_upload_dir(purge_module, rows):
    purge_module.purge(90, apply=True, now=rows["now"])
    assert not rows["old"].exists() and not rows["annotated"].exists()
    assert rows["new"].exists() and rows["outside"].exists()
    with SessionLocal() as db:
        assert db.query(Prediction).count() == 3  # records are retained
        old = db.query(Prediction).filter_by(file_name="old").one()
        assert old.annotated_path is None
