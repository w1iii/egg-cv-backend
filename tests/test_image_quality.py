import numpy as np

from app.services.image_quality import assess_crop, assess_image


def textured(height=480, width=640, base=120):
    rng = np.random.default_rng(1)
    return (np.full((height, width, 3), base, dtype=np.uint8)
            + rng.integers(0, 40, size=(height, width, 3), dtype=np.uint8))


def test_good_image_has_no_reasons():
    report = assess_image(textured())
    assert report["reasons"] == []
    assert report["width"] == 640 and report["height"] == 480


def test_low_resolution_and_blur_are_flagged():
    assert "image_low_resolution" in assess_image(textured(200, 300))["reasons"]
    flat = np.full((480, 640, 3), 128, dtype=np.uint8)
    assert "image_blurry" in assess_image(flat)["reasons"]


def test_partial_visibility_at_image_edge():
    image = textured()
    assert "partial_visibility" in assess_crop(image, [0, 100, 80, 180], [])["reasons"]
    assert "partial_visibility" in assess_crop(image, [560, 100, 640, 180], [])["reasons"]
    assert "partial_visibility" not in assess_crop(image, [100, 100, 180, 180], [])["reasons"]


def test_overlapping_detection_flag():
    image = textured()
    report = assess_crop(image, [100, 100, 200, 200], [[150, 150, 250, 250]])
    assert "overlapping_detection" not in report["reasons"]  # 25% of smaller box
    report = assess_crop(image, [100, 100, 200, 200], [[130, 130, 230, 230]])
    assert "overlapping_detection" in report["reasons"]
    assert report["max_overlap"] > 0.3


def test_exposure_and_glare_flags():
    image = textured()
    image[100:200, 100:200] = 5
    assert "crop_underexposed" in assess_crop(image, [100, 100, 200, 200], [])["reasons"]
    image[300:400, 300:400] = 255
    assert "crop_glare" in assess_crop(image, [300, 300, 400, 400], [])["reasons"]
