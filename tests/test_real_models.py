"""Quality gate against the real checkpoints (skipped if they are absent).

* Checkpoint files must be byte-identical to the approved versions.
* The production pipeline must run end to end on a real photo.

This proves the integration works; it says nothing about accuracy.
"""

from pathlib import Path

import cv2
import pytest

from app.config import settings
from app.ml.preprocessing import file_sha256

# Recorded 2026-11-23 before any code change (docs/evaluation/baseline.md).
APPROVED_CHECKPOINTS = {
    "MODEL_PATH": "c755a7a6cf5ad883486e0396dc0dfda5030ada3007315e6ef479b26a99a931cf",
    "DAMAGE_MODEL_PATH": "9fe5d113a5c1c5faebec17b806a09975c50f2cac3cbf67f9a06d5432b26f1861",
    "CLEANLINESS_MODEL_PATH": "b00884a1f6b03b87e2f8b3378081cbadbbdab9e68218de082d3c624e3c866a6f",
}
SAMPLE_DIR = Path(__file__).resolve().parents[2] / "egg-cv-frontend" / "public" / "egg-images"


def _path(setting: str) -> Path:
    return Path(getattr(settings, setting))


@pytest.mark.parametrize("setting,expected", sorted(APPROVED_CHECKPOINTS.items()))
def test_checkpoint_is_unchanged(setting, expected):
    path = _path(setting)
    if not path.is_file():
        pytest.skip(f"{path} not present")
    assert file_sha256(path) == expected


@pytest.fixture(scope="module")
def registry():
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    if not all(_path(setting).is_file() for setting in APPROVED_CHECKPOINTS):
        pytest.skip("checkpoints not present")
    from app.ml.model_registry import ModelRegistry

    registry = ModelRegistry()
    registry.load()
    assert registry.detector_loaded and registry.damage_loaded and registry.cleanliness_loaded
    return registry


def test_registry_reports_hashes(registry):
    assert registry.detector.checkpoint_sha256 == APPROVED_CHECKPOINTS["MODEL_PATH"]
    assert registry.damage.checkpoint_sha256 == APPROVED_CHECKPOINTS["DAMAGE_MODEL_PATH"]
    assert registry.cleanliness.checkpoint_sha256 == APPROVED_CHECKPOINTS["CLEANLINESS_MODEL_PATH"]
    assert registry.damage.img_size == 224 and registry.cleanliness.img_size == 224


@pytest.mark.skipif(not SAMPLE_DIR.is_dir(), reason="sample images not available")
def test_pipeline_runs_on_a_real_photo(registry):
    from app.services.inspection_pipeline import inspect_image

    image_path = next(p for p in sorted(SAMPLE_DIR.iterdir()) if p.suffix.lower() == ".png")
    image = cv2.imread(str(image_path))
    result = inspect_image(
        image,
        predictor=registry.detector,
        damage_classifier=registry.damage,
        cleanliness_classifier=registry.cleanliness,
        detection_threshold=0.75,
    )
    assert result["summary"]["total"] == len(result["detections"]) > 0
    for det in result["detections"]:
        assert det["decision"] in {"Acceptable", "Reject", "Needs Review"}
        assert det["grade"] == "N/A"  # size grades are not approved
        assert 0.0 <= det["damage_probability"] <= 1.0
        if not settings.CLEANLINESS_RELEASE_APPROVED:
            assert det["decision"] != "Acceptable"
