# Egg CV Backend

FastAPI service that inspects still images of eggs. Every egg detected in an
image gets its own damage and cleanliness result, an external-quality
**decision** (`Acceptable` / `Reject` / `Needs Review`) with reasons, and a
separate commercial size **grade** (`A` / `B` / `C` / `N/A`).

- How it works, the data model and the API: [`docs/architecture.md`](docs/architecture.md)
- Decision and grade policy: [`app/services/grading_service.py`](app/services/grading_service.py)
- Security posture and retention: [`docs/security-checklist.md`](docs/security-checklist.md)
- Evidence (baseline, benchmarks, what is not yet validated): [`docs/evaluation/`](docs/evaluation/)

## Local development

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # then point DATABASE_URL at your local DB
set -a && . ./.env && set +a
.venv/bin/alembic upgrade head
.venv/bin/uvicorn app.main:app --reload --port 8000
```

`alembic upgrade head` owns the schema. For a throwaway SQLite database you
can instead set `AUTO_CREATE_SCHEMA=true` (never in production; it does not
add new columns to existing tables).

Health check: `GET /api/v1/dashboard/health` (reports loaded models, their
SHA-256 and whether each release gate is approved).

## Tests and quality gates

```bash
.venv/bin/python -m pytest -q
```

The suite needs no database server (SQLite) and covers the decision policy,
detection post-processing, image checks, the upload-to-history API, failure
paths, CSV output and vocabulary contracts with the frontend (when the
`egg-cv-frontend` repo sits next to this one). `tests/test_real_models.py`
additionally checks that the checkpoints are byte-identical to the approved
versions and runs the real models end to end; it is skipped if they are
absent.

## Benchmarks

`scripts/benchmark.py` measures the detector, either classifier, or the full
pipeline on a frozen labeled set, and records the manifest hash, code
revision, thresholds and checkpoint hashes. `--backend-root` runs it against
another checkout for before/after comparisons, and `--compare` refuses to
compare runs on different inputs. See
[`docs/evaluation/benchmark-report.md`](docs/evaluation/benchmark-report.md).

## Release gates

| Setting | Default | Effect when `false` |
| --- | --- | --- |
| `CLEANLINESS_RELEASE_APPROVED` | `false` | Cleanliness output is beta information only; eggs without damage go to `Needs Review`, never `Acceptable` or a stain `Reject`. |
| `ARUCO_GRADES_APPROVED` | `false` | Marker-based sizes are shown as estimates; every grade is `N/A`. |

Weight estimation is disabled in code until a documented calibration dataset
and error evaluation exist.

## Deploy (Railway / Render)

The repository already includes `Procfile`, `railway.json`, `render.yaml`,
`.python-version` (3.11), and the model checkpoints under `models/` and
`egg_detection/`.

Set these environment variables in the provider:

| Variable       | Value                                                        |
| -------------- | ------------------------------------------------------------ |
| `ENVIRONMENT`  | `production`                                                 |
| `DEBUG`        | `false`                                                      |
| `DATABASE_URL` | managed Postgres URL                                         |
| `CORS_ORIGINS` | your Vercel origin, e.g. `https://egg-cv.vercel.app`         |
| `UPLOAD_DIR`   | persistent volume path, e.g. `/var/data/uploads`             |
| `ACCEPT_UNAUTHENTICATED_ACCESS` | must be set to `true` to start in production; see below |

**The API has no authentication.** Anyone who can reach a deployed instance
can upload, list, download and delete inspections. Production startup fails
unless `ACCEPT_UNAUTHENTICATED_ACCESS=true` records that this was accepted.
Prefer keeping the service on a private network.

Notes:

- Migrations run automatically via the start command
  (`alembic upgrade head`), or manually with
  `DATABASE_URL=... alembic upgrade head`.
- Use one worker: each worker loads three model checkpoints.
- Attach a persistent disk for `UPLOAD_DIR`; container filesystems are
  ephemeral. Uploaded images are not served statically; only the annotated
  image of an inspection can be downloaded, through the API.
- Retention: `python scripts/purge_uploads.py --older-than-days N` lists
  images past the window; add `--apply` to delete them (records are kept).
- Give the service at least 2 GB RAM so torch + ultralytics fit comfortably.
- `PROJECT_ROOT` is auto-detected; override it only if you relocate the
  checkpoints.
- `nixpacks.toml` reinstalls `opencv-python-headless` after dependencies
  resolve. `ultralytics` pulls GUI `opencv-python`, which shadows the headless
  build and needs `libGL`, absent in the Railway runtime image.
