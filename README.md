# Egg CV Backend

FastAPI service for egg detection, damage classification, cleanliness, grading,
and dashboard APIs.

## Local development

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # then point DATABASE_URL at your local DB
set -a && . ./.env && set +a
.venv/bin/alembic upgrade head
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Health check: `GET /api/v1/dashboard/health`.

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

Notes:

- Migrations run automatically via the start command
  (`alembic upgrade head`), or manually with
  `DATABASE_URL=... alembic upgrade head`.
- Use one worker: each worker loads three model checkpoints.
- Attach a persistent disk for `UPLOAD_DIR`; container filesystems are
  ephemeral.
- Give the service at least 2 GB RAM so torch + ultralytics fit comfortably.
- `PROJECT_ROOT` is auto-detected; override it only if you relocate the
  checkpoints.
