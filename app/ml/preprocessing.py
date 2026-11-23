"""Crop preprocessing shared by the damage and cleanliness classifiers.

Mirrors the evaluation transform used in training
(research repo ``src/utils.py::get_transforms(train=False)`` and
``src/dataset.py``): OpenCV BGR -> RGB, square resize to the checkpoint's
``img_size``, ``ToTensor`` scaling to [0, 1], then ImageNet normalization.
Crop padding differs per model and is applied by the caller
(``DETECTION_CROP_PADDING`` for damage, ``CLEANLINESS_CROP_PADDING`` for
cleanliness), matching how each training set was cropped.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Optional

import cv2
import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Below ~5% of a 224x224 input the resize upsamples too much detail away.
MIN_CROP_AREA = 2500


@dataclass(frozen=True)
class CropCheck:
    ok: bool
    reason: Optional[str] = None


def check_crop(crop) -> CropCheck:
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
        return CropCheck(False, "empty_crop")
    if crop.ndim not in (2, 3) or crop.shape[0] < 2 or crop.shape[1] < 2:
        return CropCheck(False, "invalid_crop")
    if crop.ndim == 3 and crop.shape[2] not in (1, 3, 4):
        return CropCheck(False, "invalid_crop")
    if crop.shape[0] * crop.shape[1] < MIN_CROP_AREA:
        return CropCheck(False, "small_crop")
    return CropCheck(True)


def to_rgb(crop: np.ndarray) -> np.ndarray:
    """OpenCV crop (gray, BGR or BGRA) -> contiguous RGB uint8."""
    if crop.ndim == 2 or crop.shape[2] == 1:
        rgb = cv2.cvtColor(crop.reshape(crop.shape[:2]), cv2.COLOR_GRAY2RGB)
    elif crop.shape[2] == 4:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


def build_eval_transform(img_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD)),
        ]
    )


def file_sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None
