# Improvement plan implementation log

Branch `feat/inspection-quality-decisions` in `egg-cv-backend` and
`egg-cv-frontend`, both from `main` @ their 2026-11-23 heads. Changes are
**uncommitted**. Owner decisions taken up front: implement the whole plan;
keep `tray_id` and batch columns as legacy (no destructive migration);
unapproved or unavailable cleanliness forces `Needs Review`; no commits.

Nothing in this log claims improved model accuracy. Model checkpoints, labels,
training data and decision thresholds were not changed. Size grades and weight
estimation stay disabled.

## Quality gates (final run)

| Gate | Command | Result |
|---|---|---|
| Backend tests | `.venv/Scripts/python -m pytest -q` | 147 passed |
| Checkpoint hashes unchanged | `tests/test_real_models.py` | passed (3/3 identical) |
| Real models end to end | `tests/test_real_models.py::test_pipeline_runs_on_a_real_photo` | passed |
| Frontend typecheck | `npx tsc --noEmit` | passed |
| Frontend lint | `npm run lint` (`--max-warnings 0`) | passed |
| Frontend build | `npm run build` | passed |
| API/DB/frontend vocabulary | `tests/test_contracts.py` | passed |
| Migration chain renders | `alembic upgrade head --sql` from empty; `upgrade/downgrade batch_recording:head --sql` | passed; single head `c3d4e5f6a7b8` |
| Migration **applies** on a clean and an existing PostgreSQL DB | — | **Not run: no PostgreSQL or Docker on this machine** (existing migrations use Postgres-only SQL, so SQLite cannot stand in) |
| Manual UI check | Real models, sample image, inspector / history / dashboard at desktop and 375 px | passed; no console errors; no horizontal scroll |

## Work packages

### F-00 Baseline
Recorded in `evaluation/baseline.md`: environment, hashes, original tests
(6 passed), frontend gates (passed), and a real-model output snapshot of
the six sample images used to verify later changes. Python 3.12 was used;
the repo pins 3.11.

### F-01 Tray and batch assumptions removed
- Backend: `MAX_TRAY_EGGS`, `tray_count_status`, `tray_over_capacity` and all
  `tray_*`/`max_tray_eggs` metadata removed; `src/tray_count.py` replaced by
  `src/detection_postprocess.py`. Any number of detections is graded.
  `POST /predictions/{id}/record` and its schemas removed; history lists every
  inspection.
- DB/API: `tray_id` column kept, mapped to ORM attribute `inspection_code` and
  exposed only under that name. Legacy `batch_*`, `okay_count`,
  `stained_count`, `provisional_grade` columns kept, no longer written or
  exposed. CSV moves to `egg_statistics_v3.csv` with `inspection_code`.
- Frontend: Batch History -> Inspection History; batch accumulation and
  "Record batch" removed; no fixed 5-card layout or "x / 5 detected"; tray
  wording removed from upload, camera and notices.
- Tests: `test_detection.py::test_arbitrary_egg_counts_are_all_kept` (0, 1, 5,
  6, 12, 30), `test_inspection_pipeline.py::test_every_valid_detection_is_processed`
  (1, 5, 6, 14), `test_api.py::test_batch_record_endpoint_is_gone`.
- Evidence: on real photos, 21 eggs previously flagged `tray_over_capacity`
  now carry no such flag (`evaluation/baseline.md`).

### F-02 Quality decision vs commercial grade
- `calculate_quality_decision()` and `calculate_commercial_grade()` in
  `app/services/grading_service.py`; `decision`, `decision_reason` on every
  egg; `acceptable_count` / `reject_count` per inspection; grade limited to
  A/B/C/N/A (legacy `Reject`/`AA` mapped on output, stored data untouched).
  `provisional_grade` dropped from the API (semantics superseded).
- Migration `alembic/versions/c3d4e5f6a7b8_add_quality_decision_fields.py`
  (additive; legacy rows keep `decision = NULL`).
- Tests: `test_grading_policy.py` covers all 48 damage x cleanliness x approval
  combinations against an independent restatement of the policy table, plus
  input-reason precedence and grade gating.

### F-03 Cleanliness release policy
- Unapproved or missing cleanliness yields `Needs Review`
  (`cleanliness_unapproved` / `cleanliness_unavailable`), never `Acceptable`
  and never a stain `Reject`. Raw output is kept and shown with a "beta" tag.
- Metadata records availability, approval, checkpoint name and SHA-256 and
  thresholds; `/dashboard/health` separates "loaded" from "approved".
- Consequence with shipped settings: **no egg can be Acceptable**; every
  undamaged egg goes to review until cleanliness is approved.

### F-04 Crop and preprocessing reliability
- `app/ml/preprocessing.py`: one crop check and transform shared by both
  classifiers, verified against the research repo's training/eval code
  (BGR->RGB, square resize, ImageNet normalization, padding 0.10 / 0.0).
- `app/services/image_quality.py`: low resolution, blur, partial (edge)
  visibility, overlap, underexposure and glare produce review reasons.
  Thresholds are untuned heuristics; on the six sample photos none fired.
- Tests: `test_contracts.py::test_eval_transform_shape_and_normalization`,
  `::test_crop_checks_and_color_order`, `test_image_quality.py`,
  `test_inspection_pipeline.py::test_damage_and_cleanliness_use_separate_padding`.
- Evidence that model inputs are unchanged: 36/36 eggs identical
  (boxes, statuses, confidences) before vs after. An intermediate version that
  rounded instead of truncating box coordinates changed several confidences;
  that was reverted to truncation.

### F-05 Detection robustness (detector unchanged)
- Coordinates clipped to the image; degenerate boxes dropped and counted;
  raw, duplicate and invalid counts plus `detection_count_status` in metadata;
  reason codes prefixed by source (`detection_*`, `crop_*`, `image_*`,
  `damage_*`, `cleanliness_*`); raw P(Damaged)/P(Stained) kept in `result_json`.
- Tests: `test_detection.py` (clipping, duplicates, zero, arbitrary counts,
  threshold pass-through), `test_inspection_pipeline.py` (partial/overlap).
- Benchmark: no metric changed on the 19-image detector test split.

### F-06 Calibration and sizing
- New: ArUco detections whose outline is not the black marker border are
  rejected (`marker_border_invalid`); this fixed a synthetic case where a
  thin white margin inflated the measured marker by about 15%.
- Grades require Acceptable + valid marker + `ARUCO_GRADES_APPROVED`.
- Physical validation **not done**; harness and protocol ready
  (`evaluation/benchmark-report.md`).

### F-07 / F-09 Review experience and frontend
- Per-egg card: decision badge (icon + text, not colour alone), "Rejected
  because / Review because" with plain-language reasons, damage and
  cleanliness with confidence, beta tag, size grade, size and dimensions.
- Notices for failed, legacy, beta/missing cleanliness, calibration state,
  image-quality problems, annotation and CSV failures; empty-result state.
- Filters: decision chips on the result and in history detail; history
  filters by decision and reason (server side).
- Camera stays a still-frame capture; wording says so.
- Accessibility: live region for analysis status, `role="alert"` errors,
  labelled inputs and filters, `aria-pressed` filter buttons, dialog
  semantics and Escape to close the history drawer.

### F-08 Consistency
- One vocabulary across SQLAlchemy check constraint, Pydantic schemas,
  CSV header and TypeScript types (`test_contracts.py`).
- Transactional writes with rollback and annotated-file cleanup; CSV
  failure recorded but not fatal; CSV header guard against old files.
- Tests for failed inference, failed commit, failed CSV, failed annotation,
  model unavailable, undecodable/forged/oversized uploads, deletion of
  linked files, legacy rows (`test_api.py`).

### F-10 Security
`security-checklist.md`. Highlights: production requires explicit
`ACCEPT_UNAUTHENTICATED_ACCESS=true`; `/uploads` no longer public;
path-confined deletes/downloads; sanitized stored errors; stricter WEBP check;
`AUTO_CREATE_SCHEMA` replaces implicit `create_all`; retention script.

### F-11 Tests
From 6 to 147 tests. See the gates table above.

### F-12 Benchmark
`scripts/benchmark.py` + `evaluation/benchmark-report.md`. Detector and
cleanliness: no change before vs after, and both reproduce the research
repo's reported test results. Damage classifier, end-to-end and sizing: not
measured (data missing), stated as such.

### F-13 Documentation
`architecture.md`, `security-checklist.md`, `evaluation/*`,
`thesis-evidence-map.md`, updated READMEs and `aruco-static-sizing.md`.
The thesis document itself was not edited.

## Remaining risks and unverified items

1. **Production start will fail** on the next deploy of this branch until the
   owner sets `ACCEPT_UNAUTHENTICATED_ACCESS=true` (or keeps the API private
   and approves authentication work).
2. Migration not applied to a real PostgreSQL database.
3. With shipped gates, 100% of undamaged eggs need review; the workflow is
   safe but not yet useful for acceptance until cleanliness is approved.
4. Damage classifier at the app thresholds is not re-measured here; crop set
   missing.
5. Image-quality thresholds are heuristics, not validated on labeled data.
6. No independent end-to-end or physical sizing evaluation exists.
7. Python 3.12 used locally; deploy target is 3.11.9.
8. The research repo working tree has ~2,650 staged deletions (including 9
   cleanliness test crops); it was only read, never modified.
