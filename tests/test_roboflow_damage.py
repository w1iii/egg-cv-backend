import numpy as np
import requests

from app.ml.roboflow_damage import RoboflowDamageClassifier


def classifier():
    return RoboflowDamageClassifier("test-key", "egg_crack/2", 0.65)


def test_detection_and_no_detection(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"predictions": [{"class": "White cracked egg", "confidence": 0.9}]}

    monkeypatch.setattr(requests, "post", lambda *a, **kw: Response())
    crop = np.zeros((80, 80, 3), dtype=np.uint8)
    result = classifier().classify_crop(crop)
    assert result.status == "Damaged" and result.probability == 0.9
    monkeypatch.setattr(Response, "json", lambda self: {"predictions": []})
    assert classifier().classify_crop(crop).status == "Needs Review"


def test_remote_failure_and_unknown_class_are_review(monkeypatch):
    crop = np.zeros((80, 80, 3), dtype=np.uint8)
    def fail(*args, **kwargs):
        raise requests.Timeout()
    monkeypatch.setattr(requests, "post", fail)
    assert classifier().classify_crop(crop).review_reason == "provider_error"
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"predictions": [{"class": "unknown", "confidence": 0.99}]}
    monkeypatch.setattr(requests, "post", lambda *a, **kw: Response())
    assert classifier().classify_crop(crop).status == "Needs Review"


def test_missing_key_and_small_crop():
    assert not RoboflowDamageClassifier("", "egg_crack/2", 0.65).load_model()
    assert classifier().classify_crop(np.zeros((2, 2, 3), dtype=np.uint8)).review_reason == "small_crop"
