# Benchmark: detector

- Generated: 2026-11-23T11:06:58.543096+00:00
- Code: feat/inspection-quality-decisions @ 225d5e11e53702f5c865149d7a211def8503099e (dirty: True)
- Python 3.12.10, torch 2.3.1+cpu
- Inputs: `{"images": "<workspace>/A-CNN-BASED-Automated-System-for-Multi-Criterion-Egg-Grading-Quality-Sizing-And-Damage-Detection/data/detection/images/test", "labels": "<workspace>/A-CNN-BASED-Automated-System-for-Multi-Criterion-Egg-Grading-Quality-Sizing-And-Damage-Detection/data/detection_finetune/labels/test", "image_count": 19, "confidence": 0.75, "iou": 0.5, "manifest_sha256": "a8e8c5d9e6d9e887ca461e956aaeb3f955fc6038421983fe8f9319f196570b3d"}`

| Checkpoint | SHA-256 |
|---|---|
| best.pt | `c755a7a6cf5ad883486e0396dc0dfda5030ada3007315e6ef479b26a99a931cf` |
| egg_grader_crop_resnet18_v1.pth | `9fe5d113a5c1c5faebec17b806a09975c50f2cac3cbf67f9a06d5432b26f1861` |
| egg_grader_cleanliness_v1.pth | `b00884a1f6b03b87e2f8b3378081cbadbbdab9e68218de082d3c624e3c866a6f` |

| Setting | Value |
|---|---|
| CONFIDENCE_THRESHOLD | 0.75 |
| DAMAGE_DECISION_THRESHOLD | 0.65 |
| DAMAGE_REVIEW_THRESHOLD | 0.75 |
| CLEANLINESS_DECISION_THRESHOLD | 0.6 |
| CLEANLINESS_REVIEW_THRESHOLD | 0.75 |
| CLEANLINESS_RELEASE_APPROVED | False |
| ARUCO_GRADES_APPROVED | False |
| DETECTION_CROP_PADDING | 0.1 |
| CLEANLINESS_CROP_PADDING | 0.0 |
| DETECTION_DUPLICATE_IOU | 0.85 |

| Metric | Value |
|---|---|
| metrics.tp | 57.0000 |
| metrics.fp | 3.0000 |
| metrics.fn | 0.0000 |
| metrics.precision | 0.9500 |
| metrics.recall | 1.0000 |
| metrics.f1 | 0.9744 |
| metrics.exact_count_accuracy | 0.8947 |
| metrics.mean_abs_count_error | 0.1579 |
| metrics.latency.count | 19.0000 |
| metrics.latency.mean_ms | 321.2500 |
| metrics.latency.p50_ms | 224.1819 |
| metrics.latency.p95_ms | 235.2304 |
| metrics.latency.max_ms | 2083.5869 |

## Comparison with `docs\evaluation\benchmarks\detector_before.json`

| Metric | Before | After | Delta | Status |
|---|---|---|---|---|
| metrics.exact_count_accuracy | 0.8947 | 0.8947 | 0.0000 | unchanged |
| metrics.f1 | 0.9744 | 0.9744 | 0.0000 | unchanged |
| metrics.fn | 0.0000 | 0.0000 | 0.0000 | unchanged |
| metrics.fp | 3.0000 | 3.0000 | 0.0000 | unchanged |
| metrics.latency.max_ms | 2076.0599 | 2083.5869 | 7.5270 | changed |
| metrics.latency.mean_ms | 325.7548 | 321.2500 | -4.5048 | changed |
| metrics.latency.p50_ms | 226.4869 | 224.1819 | -2.3050 | changed |
| metrics.latency.p95_ms | 284.8734 | 235.2304 | -49.6430 | changed |
| metrics.mean_abs_count_error | 0.1579 | 0.1579 | 0.0000 | unchanged |
| metrics.precision | 0.9500 | 0.9500 | 0.0000 | unchanged |
| metrics.recall | 1.0000 | 1.0000 | 0.0000 | unchanged |
| metrics.tp | 57.0000 | 57.0000 | 0.0000 | unchanged |
