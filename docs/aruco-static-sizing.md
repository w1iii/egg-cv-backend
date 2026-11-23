# Static ArUco sizing implementation record

## Setup and implementation

Use a fixed overhead camera and a fully visible DICT_4X4_50 marker ID 0 with a white margin at least one marker cell wide. Print the marker at actual size and measure the outer black square: it must be 50 mm. Configure `ARUCO_MARKER_ID` and `ARUCO_SIDE_MM` if the physical marker differs. The marker and egg silhouette are at different heights; scale and bounding boxes are estimates, not physically validated dimensions.

`app/services/aruco_calibration.py` detects the marker in each image, checks size, border clearance, side consistency, duplicate ID and that the detected outline is the marker's black border (a thin white margin on a darker surface was otherwise measured as the marker; reason `marker_border_invalid`), and calculates millimetres per pixel from average side length. `inspection_pipeline.py` passes that scale to YOLO for bounding-box geometry and saves the result in prediction metadata. Missing or invalid markers suppress dimensions and size while retaining other classification. Weight is not calculated. The frontend displays calibration and dimensions on inspection and history screens. `ARUCO_GRADES_APPROVED` defaults to false; do not enable until the complete fixed setup has been physically validated.

## Test record — 2026-11-19

- `python3 -m unittest discover -s tests -v`: 6 passed. Synthetic marker scale, missing marker, wrong ID, duplicate ID, empty image and scale-to-geometry calculation.
- `python3 -m compileall -q app src`: passed.
- Frontend `npm run build`: passed; Browserslist reported outdated data.
- Physical marker, camera, egg measurements, end-to-end API and database persistence: **not yet tested**. No accuracy claims are supported by synthetic tests.

## Update — 2026-11-23

- Border check added (see above); 2 regression tests in `tests/test_aruco_calibration.py`.
- Size letters are now a commercial grade separate from the quality decision; they are published only for `Acceptable` eggs and only when `ARUCO_GRADES_APPROVED=true`.
- Physical validation is still **not done**. Protocol and harness: `docs/evaluation/benchmark-report.md` (end-to-end mode with `true_length_mm` / `true_breadth_mm`).
