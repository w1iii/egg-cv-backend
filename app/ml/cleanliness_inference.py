from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Optional

import cv2
import numpy as np

from app.config import settings
from app.ml.preprocessing import build_eval_transform, check_crop, file_sha256, to_rgb


@dataclass(frozen=True)
class CleanlinessResult:
    status: str
    confidence: float
    model_version: str
    review_reason: Optional[str] = None
    # Raw positive-class probability (P(Damaged) / P(Stained)); None when
    # the crop was refused before inference.
    probability: Optional[float] = None


def decide_cleanliness(
    stained_probability: float,
    stained_threshold: float,
    review_threshold: float,
) -> tuple[str, float, Optional[str]]:
    """Map P(Stained) to status, confidence, and review reason.

    A stained prediction is never downgraded to review; stain recall takes
    priority. Review applies only when a clean prediction is uncertain.
    """
    if stained_probability >= stained_threshold:
        return "Stained", stained_probability, None
    confidence = 1.0 - stained_probability
    if confidence < review_threshold:
        return "Needs Review", confidence, "low_confidence"
    return "Clean", confidence, None


class CleanlinessClassifier:
    """Inference wrapper for the binary cleanliness checkpoint."""

    EXPECTED_CLASSES = ("Clean", "Stained")

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        review_threshold: Optional[float] = None,
        stained_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = Path(
            checkpoint_path or settings.CLEANLINESS_MODEL_PATH
        )
        self.review_threshold = (
            review_threshold
            if review_threshold is not None
            else settings.CLEANLINESS_REVIEW_THRESHOLD
        )
        self.stained_threshold = (
            stained_threshold
            if stained_threshold is not None
            else settings.CLEANLINESS_DECISION_THRESHOLD
        )
        self._explicit_stained_threshold = stained_threshold is not None
        self.device_name = device
        self._model = None
        self._torch = None
        self._transform = None
        self._init_lock = threading.Lock()
        self.class_names = self.EXPECTED_CLASSES
        self.arch = "unknown"
        self.img_size = 224
        self.checkpoint_sha256: Optional[str] = None

    @property
    def model_version(self) -> str:
        return self.checkpoint_path.name

    def load_model(self) -> bool:
        if not self.checkpoint_path.is_file():
            return False
        try:
            self._ensure_model()
            return True
        except (ImportError, KeyError, RuntimeError, ValueError, OSError):
            self._model = None
            return False

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._init_lock:
            if self._model is not None:
                return
            self._load_model_locked()

    def _load_model_locked(self) -> None:
        import torch

        from src.model_factory import build_model

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self._select_device(torch),
            weights_only=True,
        )
        class_names = tuple(checkpoint.get("class_names", ()))
        if class_names != self.EXPECTED_CLASSES:
            raise ValueError(
                f"Unexpected cleanliness checkpoint classes: {class_names!r}"
            )

        arch = str(checkpoint.get("arch", "custom"))
        arch_args = checkpoint.get("arch_args") or {}
        num_classes = int(arch_args.get("num_classes", len(class_names)))
        if num_classes != len(class_names):
            raise ValueError(
                f"Checkpoint num_classes={num_classes} does not match "
                f"class_names={class_names!r}"
            )
        img_size = int(checkpoint.get("img_size", 224))
        if img_size <= 0:
            raise ValueError(f"Invalid checkpoint img_size: {img_size}")
        if not self._explicit_stained_threshold:
            checkpoint_threshold = checkpoint.get("stained_threshold")
            if checkpoint_threshold is not None:
                checkpoint_threshold = float(checkpoint_threshold)
                if not 0 < checkpoint_threshold < 1:
                    raise ValueError(
                        "Invalid checkpoint stained_threshold: "
                        f"{checkpoint_threshold}"
                    )
                self.stained_threshold = checkpoint_threshold

        model = build_model(
            arch,
            num_classes=num_classes,
            pretrained=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self._select_device(torch))
        model.eval()
        transform = build_eval_transform(img_size)
        # Publish state only once every dependency is ready so concurrent
        # callers never observe a model without its transform.
        self._torch = torch
        self.arch = arch
        self.img_size = img_size
        self._transform = transform
        self.checkpoint_sha256 = file_sha256(self.checkpoint_path)
        self._model = model

    def _select_device(self, torch_module: Any):
        if self.device_name:
            return torch_module.device(self.device_name)
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
            return torch_module.device("mps")
        return torch_module.device("cpu")

    def classify_crop(self, crop: np.ndarray) -> CleanlinessResult:
        check = check_crop(crop)
        if not check.ok:
            return CleanlinessResult(
                "Needs Review", 0.0, self.model_version, check.reason
            )

        self._ensure_model()
        crop = to_rgb(crop)

        tensor = self._transform(crop).unsqueeze(0).to(
            self._model.parameters().__next__().device
        )
        with self._torch.no_grad():
            probabilities = self._torch.softmax(self._model(tensor), dim=1)[0]
        stained_probability = float(probabilities[1].item())
        status, confidence, review_reason = decide_cleanliness(
            stained_probability,
            self.stained_threshold,
            self.review_threshold,
        )
        return CleanlinessResult(
            status,
            confidence,
            self.model_version,
            review_reason,
            stained_probability,
        )
