# F-00 Baseline and safety snapshot

Recorded 2026-11-23, before any code change, on `main` @
`225d5e11e53702f5c865149d7a211def8503099e` (backend) and the matching
frontend `main`.

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26300 |
| Python | 3.12.10 (repo pins 3.11.9; 3.11 was not installed on this machine) |
| torch / torchvision | 2.3.1+cpu / 0.18.1+cpu |
| ultralytics | 8.1.0 |
| fastapi / pydantic / SQLAlchemy / alembic | 0.109.0 / 2.5.3 / 2.0.25 / 1.13.1 |
| numpy / opencv-python-headless | 1.26.4 / 4.9.0.80 (ultralytics also installs opencv-python 4.11.0.86) |
| Node / npm | 24.14.0 / 11.16.0 |
| Database server | none available (no PostgreSQL, no Docker) |

## Checkpoints (SHA-256)

| File | SHA-256 | Bytes |
|---|---|---|
| `egg_detection/finetune_egg_v1/weights/best.pt` | `c755a7a6cf5ad883486e0396dc0dfda5030ada3007315e6ef479b26a99a931cf` | 22,518,425 |
| `models/egg_grader_crop_resnet18_v1.pth` | `9fe5d113a5c1c5faebec17b806a09975c50f2cac3cbf67f9a06d5432b26f1861` | 44,791,758 |
| `models/egg_grader_cleanliness_v1.pth` | `b00884a1f6b03b87e2f8b3378081cbadbbdab9e68218de082d3c624e3c866a6f` | 44,792,346 |

The same hashes were re-verified after all changes (`tests/test_real_models.py`).

## Existing tests on unmodified code

| Command | Result |
|---|---|
| backend `python -m unittest discover -s tests -v` | 6 passed (ArUco synthetic tests only) |
| backend `python -m compileall -q app src` | passed |
| frontend `tsc --noEmit` | passed |
| frontend `npm run lint` | passed (0 warnings) |
| frontend `npm run build` | passed (Browserslist data 7 months old: informational) |

No failures to classify. Coverage gaps found: no tests for grading,
detection post-processing, the API, persistence, CSV, failure paths or the
frontend contract.

## Behaviour snapshot (real models, 6 bundled sample images)

The six images in `egg-cv-frontend/public/egg-images` were run through
`PredictionService.run_prediction` with the real checkpoints on both `main`
and the branch. These are web/stock images without labels; they are used only
to check that behaviour changed where intended and nowhere else.

| | `main` | branch |
|---|---|---|
| Eggs detected (per image) | 5, 5, 5, 7, 7, 7 | identical |
| Boxes, damage/cleanliness status and confidences | — | identical for all 36 eggs |
| Eggs flagged `tray_over_capacity` | 21 (every egg in the three 7-egg images) | 0 |
| Not-damaged eggs labelled grade `Reject` purely from the unapproved cleanliness model | 10 | 0 (now `Needs Review` / `cleanliness_unapproved`) |
| Decision outcome | n/a | 16 Reject (all `damaged`), 20 Needs Review (all `cleanliness_unapproved`), 0 Acceptable |
| Grades | `Reject` / `N/A` mixed with size letters | all `N/A` (size grades not approved, no marker) |

## Defects found during the baseline that were fixed

1. Every egg in an image with more than five eggs was routed to review as
   `tray_over_capacity` (fixed-tray assumption).
2. An unapproved cleanliness model could turn an intact egg into a final
   `Reject` grade.
3. ArUco: a marker whose white margin is narrower than one cell, on a darker
   surface, was measured at the margin's outline (230 px instead of 199 px
   in a synthetic test), shrinking mm-per-pixel by about 13% while reporting
   `available`. Now rejected as `marker_border_invalid`
   (`tests/test_aruco_calibration.py`).
4. A failed database commit left the session unusable; the failure path then
   raised again without marking the inspection `failed`.
5. `/uploads` served every stored image without restriction.
