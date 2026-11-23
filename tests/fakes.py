"""Deterministic stand-ins for the detector and classifiers.

The fake detector feeds synthetic Ultralytics-shaped results through the
real ``YOLOPredictor._parse_results`` so clipping, de-duplication and
geometry are exercised exactly as in production.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from app.ml.cleanliness_inference import CleanlinessResult
from app.ml.damage_inference import DamageResult
from app.ml.yolo_inference import YOLOPredictor


class _Tensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _Tensor(xyxy)
        self.conf = _Tensor(conf)
        self.cls = _Tensor([0] * len(conf))

    def __len__(self):
        return len(self.conf.numpy())


class FakeYoloResult:
    def __init__(self, boxes: Sequence[Sequence[float]], confs: Sequence[float], shape):
        self.boxes = _Boxes(boxes, confs) if boxes else None
        self.orig_shape = shape


class FakeYoloModel:
    names = {0: "egg"}

    def __init__(self, boxes, confs):
        self.boxes = [list(box) for box in boxes]
        self.confs = list(confs)
        self.calls = 0

    def __call__(self, frame, conf=0.75):
        self.calls += 1
        keep = [i for i, value in enumerate(self.confs) if value >= conf]
        return [
            FakeYoloResult(
                [self.boxes[i] for i in keep],
                [self.confs[i] for i in keep],
                frame.shape[:2],
            )
        ]


def make_predictor(boxes: Iterable[Sequence[float]] = (), confs: Optional[Sequence[float]] = None):
    boxes = list(boxes)
    confs = list(confs) if confs is not None else [0.95] * len(boxes)
    predictor = YOLOPredictor(model_path="fake-detector.pt")
    predictor._model = FakeYoloModel(boxes, confs)
    return predictor


def grid_boxes(count: int, size: int = 80, gap: int = 30, origin: int = 20, columns: int = 5):
    boxes = []
    for index in range(count):
        row, col = divmod(index, columns)
        x1 = origin + col * (size + gap)
        y1 = origin + row * (size + gap)
        boxes.append([x1, y1, x1 + size, y1 + size])
    return boxes


@dataclass
class FakeClassifier:
    """Returns queued (status, confidence, reason) tuples, then a default."""

    kind: str = "damage"
    default: tuple = ("Not Damaged", 0.95, None)
    queue: Optional[list] = None
    model_version: str = "fake.pth"
    checkpoint_sha256: str = "0" * 64
    damaged_threshold: float = 0.65
    stained_threshold: float = 0.60
    review_threshold: float = 0.75

    def __post_init__(self):
        self.queue = list(self.queue or [])
        self.crops = []

    def classify_crop(self, crop):
        self.crops.append(crop)
        status, confidence, reason = self.queue.pop(0) if self.queue else self.default
        if self.kind == "damage":
            return DamageResult(status, confidence, self.model_version, reason, 1 - confidence)
        return CleanlinessResult(status, confidence, self.model_version, reason, 1 - confidence)


def damage(default=("Not Damaged", 0.95, None), queue=None):
    return FakeClassifier("damage", default, queue)


def cleanliness(default=("Clean", 0.95, None), queue=None):
    return FakeClassifier("cleanliness", default, queue)
