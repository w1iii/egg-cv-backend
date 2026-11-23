# Thesis claim -> evidence map

Use this to check every system claim in the thesis. If a claim is not in the
"Supported" table, it is not supported by the current system. Implementation
facts here are separate from academic conclusions.

## Supported (current system)

| Claim | Evidence |
|---|---|
| The system processes uploaded still images and single captured camera frames; there is no live-video inference. | `app/routers/predictions.py` (image only), `src/components/inspector/LiveCameraCapture.tsx` |
| Every valid egg detection in an image is evaluated individually; there is no fixed egg count or tray. | `src/detection_postprocess.py`, `tests/test_detection.py::test_arbitrary_egg_counts_are_all_kept` |
| Detection uses the fine-tuned Ultralytics model at confidence 0.75 with duplicate suppression (IoU 0.85). | `app/config.py`, `app/ml/yolo_inference.py`, checkpoint hash in `architecture.md` |
| Damage and cleanliness are separate ResNet18 crop classifiers with distinct crop padding (0.10 / 0.0) matching their training crops. | `app/ml/*_inference.py`, `app/ml/preprocessing.py`, `tests/test_inspection_pipeline.py::test_damage_and_cleanliness_use_separate_padding` |
| Each egg receives a quality decision (Acceptable / Reject / Needs Review) separate from a commercial size grade (A/B/C/N/A). | `app/services/grading_service.py`, `tests/test_grading_policy.py` |
| Uncertain or unusable inputs are routed to human review with stated reasons. | `app/services/image_quality.py`, `grading_service.py`, `tests/test_inspection_pipeline.py` |
| The cleanliness model is not release-approved and cannot produce a final Acceptable or Reject decision. | `CLEANLINESS_RELEASE_APPROVED=false`, `tests/test_grading_policy.py::test_unapproved_cleanliness_never_accepts_or_rejects_alone` |
| Size grades are disabled until physical validation; millimetre sizes appear only with a valid ArUco marker. | `ARUCO_GRADES_APPROVED=false`, `tests/test_inspection_pipeline.py::test_marker_without_approval_measures_but_never_grades` |
| Weight is not estimated. | `app/services/inspection_pipeline.py` (`weight_g = None`), `/dashboard/health` |
| Results are stored per egg in PostgreSQL, exported per egg to CSV, and drawn on an annotated image. | `app/models.py`, `app/services/csv_service.py`, `tests/test_api.py::test_csv_row_per_egg_with_new_columns` |
| On the 19-image detector test split, the deployed detector path found all 57 labeled eggs with 3 false positives (precision 0.95, recall 1.00). | `evaluation/benchmarks/detector_after.json` (same-source test split, not field data) |
| On 82 available cleanliness test crops the deployed classifier path reproduces the research evaluation (1 FP, 1 FN at threshold 0.04). | `evaluation/benchmarks/cleanliness_after.json` (web/stock crops, not field data) |

## Not supported: do not claim

| Claim | Why |
|---|---|
| End-to-end grading accuracy of the application | No labeled full-image evaluation set exists (`evaluation/benchmark-report.md`). |
| Field / on-farm performance of any component | All test data is same-source or web/stock. |
| Size or weight measurement accuracy | No physical ground truth collected. |
| Damage classifier accuracy at the app's thresholds (0.65 / 0.75) | Crop test set not available; not re-measured. Research report used threshold 0.18 on a different crop set. |
| Classifier validation metrics as application accuracy | Plan constraint 5. |
| That the changes in this branch improved model accuracy | Before/after benchmarks show no change; checkpoints unchanged. |
| Multi-user / authenticated operation | Single shared local operator; no authentication. |
