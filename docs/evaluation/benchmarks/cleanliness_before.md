# Benchmark: classifier (cleanliness)

- Generated: 2026-11-23T11:06:03.974569+00:00
- Code: main @ 225d5e11e53702f5c865149d7a211def8503099e (dirty: False)
- Python 3.12.10, torch 2.3.1+cpu
- Inputs: `{"manifest": "<workspace>/A-CNN-BASED-Automated-System-for-Multi-Criterion-Egg-Grading-Quality-Sizing-And-Damage-Detection/data/cleanliness_candidates/manifests/test.csv", "manifest_sha256": "b093e48fc103093b2a1cdaef7915add2eac037701d1fbba5e47767c591edbe6d", "rows": 91, "unreadable": 9}`

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
| MAX_TRAY_EGGS | 5 |

| Metric | Value |
|---|---|
| app_policy.decision_threshold | 0.6000 |
| app_policy.review_threshold | 0.7500 |
| app_policy.samples | 82.0000 |
| app_policy.confusion.Stained.Stained | 38.0000 |
| app_policy.confusion.Stained.Clean | 3.0000 |
| app_policy.confusion.Stained.Needs Review | 1.0000 |
| app_policy.confusion.Clean.Stained | 1.0000 |
| app_policy.confusion.Clean.Clean | 39.0000 |
| app_policy.confusion.Clean.Needs Review | 0.0000 |
| app_policy.unrecognized_predictions | 0.0000 |
| app_policy.decided | 81.0000 |
| app_policy.review_rate | 0.0122 |
| app_policy.accuracy_decided | 0.9506 |
| app_policy.precision | 0.9744 |
| app_policy.recall | 0.9268 |
| app_policy.specificity | 0.9750 |
| app_policy.f1 | 0.9500 |
| app_policy.balanced_accuracy | 0.9509 |
| app_policy.positive_recall_counting_review_as_miss | 0.9048 |
| reference_threshold.threshold | 0.0400 |
| reference_threshold.samples | 82.0000 |
| reference_threshold.confusion.Stained.Stained | 41.0000 |
| reference_threshold.confusion.Stained.Clean | 1.0000 |
| reference_threshold.confusion.Stained.Needs Review | 0.0000 |
| reference_threshold.confusion.Clean.Stained | 1.0000 |
| reference_threshold.confusion.Clean.Clean | 39.0000 |
| reference_threshold.confusion.Clean.Needs Review | 0.0000 |
| reference_threshold.unrecognized_predictions | 0.0000 |
| reference_threshold.decided | 82.0000 |
| reference_threshold.review_rate | 0.0000 |
| reference_threshold.accuracy_decided | 0.9756 |
| reference_threshold.precision | 0.9762 |
| reference_threshold.recall | 0.9762 |
| reference_threshold.specificity | 0.9750 |
| reference_threshold.f1 | 0.9762 |
| reference_threshold.balanced_accuracy | 0.9756 |
| reference_threshold.positive_recall_counting_review_as_miss | 0.9762 |
| latency.count | 82.0000 |
| latency.mean_ms | 33.0639 |
| latency.p50_ms | 32.8325 |
| latency.p95_ms | 36.9071 |
| latency.max_ms | 42.3588 |
