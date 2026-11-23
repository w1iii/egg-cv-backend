"""Test isolation: every run gets its own SQLite DB, upload dir and CSV.

Environment variables must be set before ``app.config`` is imported because
settings are read once at import time.
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="eggcv-tests-"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(_TMP / 'test.db').as_posix()}")
os.environ.setdefault("UPLOAD_DIR", str(_TMP / "uploads"))
os.environ.setdefault("EGG_CSV_PATH", str(_TMP / "reports" / "egg_statistics_test.csv"))
os.environ.setdefault("ENVIRONMENT", "test")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tests.fakes import FakeClassifier, make_predictor  # noqa: E402


@pytest.fixture
def tmp_root() -> Path:
    return _TMP


@pytest.fixture
def blank_image():
    """Mid-grey textured image large enough to pass image checks."""
    rng = np.random.default_rng(0)
    image = np.full((480, 640, 3), 120, dtype=np.uint8)
    noise = rng.integers(0, 40, size=image.shape, dtype=np.uint8)
    return image + noise


@pytest.fixture
def fake_predictor():
    return make_predictor


@pytest.fixture
def fake_classifier():
    return FakeClassifier
