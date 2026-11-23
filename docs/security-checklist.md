# Security and deployment checklist (F-10)

Reviewed 2026-11-23 against branch `feat/inspection-quality-decisions`.
Status: **complete for the shared-local-operator design, with one open
decision that is the owner's (network exposure).**

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | Operator model | Shared local operator, no login (unchanged by design). | `app/operator.py` |
| 2 | Unauthenticated network exposure | **Open decision.** Production refuses to start unless `ACCEPT_UNAUTHENTICATED_ACCESS=true` records that it was accepted. No authentication was added. | `app/config.py`, `tests/test_contracts.py::test_production_requires_explicit_unauthenticated_acknowledgement` |
| 3 | Static upload access | Fixed. `/uploads` static mount removed; stored images are not browsable. The only read path is the annotated download of one inspection. | `app/main.py`, `tests/test_api.py::test_uploads_are_not_served_statically` |
| 4 | Path handling | Fixed. Downloads and deletions only touch files resolved inside `UPLOAD_DIR`; stored names are server-generated UUIDs. | `app/services/prediction_service.py::is_within_directory`, `tests/test_api.py::test_delete_never_touches_files_outside_upload_dir` |
| 5 | Upload size limit | In place. `MAX_FILE_SIZE_MB` (50) enforced before multipart parsing and again while streaming. | `app/main.py::UploadGuardMiddleware`, `tests/test_api.py::test_oversized_upload_is_rejected_before_parsing` |
| 6 | Content validation | Tightened. Extension allow-list + magic bytes (WEBP now checks the `WEBP` tag, not only `RIFF`) + OpenCV decode. | `app/services/file_service.py`, `tests/test_api.py::test_invalid_uploads_are_rejected` |
| 7 | Error leakage | Fixed. HTTP errors use an allow-list; stored `error_message` is sanitized before it is returned (previously not exposed, now exposed sanitized). | `app/utils/errors.py`, `app/schemas/__init__.py`, `tests/test_api.py::test_model_failure_marks_inspection_failed` |
| 8 | CORS | Tightened. Explicit origins only (wildcard rejected in production); credentials off (no cookies are used); methods limited. | `app/main.py`, `app/config.py` |
| 9 | Secrets / credentials | No secrets in the repo. `DATABASE_URL` comes from the environment; Alembic logs host/db only. Production rejects a localhost DB URL. | `.env.example`, `alembic/env.py`, `app/config.py` |
| 10 | Response headers | `X-Content-Type-Options: nosniff` on the image download. | `app/routers/predictions.py` |
| 11 | Persistent disk | `UPLOAD_DIR` must be a persistent volume (Render disk / Railway volume). | `README.md`, `render.yaml` |
| 12 | Retention and deletion | Policy: inspection images may be purged after N days; records, per-egg results and CSV rows are kept. Script is dry-run by default. Deleting an inspection through the API removes its row, eggs, upload and annotated image; the CSV is an append-only audit log and is not rewritten. | `scripts/purge_uploads.py`, `tests/test_retention.py`, `PredictionService.delete_prediction` |
| 13 | Dev schema vs production migrations | Separated. `create_all()` only when `AUTO_CREATE_SCHEMA=true`, which production rejects; Alembic owns the schema. | `app/main.py`, `app/config.py` |
| 14 | Dependency versions | Pinned in `requirements.txt` / `package-lock.json`. No audit was run in this pass. | — |

## Owner decisions still required

1. **Network exposure.** Either keep the API on a private network, or set
   `ACCEPT_UNAUTHENTICATED_ACCESS=true` on the deployment to accept the risk,
   or approve adding authentication. Until one happens, a production deploy
   from this branch will fail at startup by design. `render.yaml` and
   `railway.json` were intentionally not changed.
2. **Retention window.** No default window is applied automatically; choose N
   and schedule `scripts/purge_uploads.py --older-than-days N --apply`.

## Not covered

- No penetration test, rate limiting or dependency vulnerability audit.
- The legacy `PUT /api/v1/settings` (`mm_per_pixel`) endpoint remains; the
  pipeline ignores that value (sizing uses only the ArUco marker), so it can
  no longer influence results.
