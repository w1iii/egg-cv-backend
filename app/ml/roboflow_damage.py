"""Experimental hosted crack detector. Empty detections are not proof of intact shell."""
import math

import cv2
import requests

from app.ml.damage_inference import DamageResult
from app.ml.preprocessing import check_crop


class RoboflowDamageClassifier:
    experimental = True
    checkpoint_sha256 = None

    def __init__(self, api_key, model_id, threshold, timeout=15.0):
        self.api_key = api_key
        self.model_id = model_id
        self.threshold = threshold
        self.timeout = timeout

    @property
    def model_version(self):
        return f"roboflow:{self.model_id}"

    def load_model(self):
        # Configuration check only; actual remote availability is checked per crop.
        return bool(self.api_key and self.model_id and self.timeout > 0)

    def classify_crop(self, crop):
        check = check_crop(crop)
        if not check.ok:
            return DamageResult("Needs Review", 0.0, self.model_version, check.reason)
        if not self.load_model():
            return DamageResult("Needs Review", 0.0, self.model_version, "provider_unavailable")
        ok, encoded = cv2.imencode(".jpg", crop)
        if not ok:
            return DamageResult("Needs Review", 0.0, self.model_version, "provider_error")
        try:
            response = requests.post(
                f"https://detect.roboflow.com/{self.model_id}",
                params={"api_key": self.api_key, "confidence": 1},
                files={"file": ("crop.jpg", encoded.tobytes(), "image/jpeg")},
                timeout=self.timeout,
            )
            response.raise_for_status()
            predictions = response.json()["predictions"]
            if not isinstance(predictions, list):
                raise ValueError("Invalid predictions")
            scores = []
            for prediction in predictions:
                label = str(prediction["class"]).strip().lower()
                score = float(prediction["confidence"])
                if label not in ("white cracked egg", "brown cracked egg") or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Unexpected detection")
                scores.append(score)
            score = max(scores, default=None)
            if score is not None and score >= self.threshold:
                return DamageResult("Damaged", score, self.model_version, probability=score)
            return DamageResult("Needs Review", score or 0.0, self.model_version,
                                "unconfirmed", probability=score)
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return DamageResult("Needs Review", 0.0, self.model_version, "provider_error")
