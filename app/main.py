from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
from pathlib import Path

from app.config import settings
from app.database import engine, Base
from app.routers import batches, compat, predictions, dashboard, user_settings
from app.ml.model_registry import model_registry

model_loaded = False
damage_model_loaded = False

UPLOAD_PATHS = {"/api/v1/predictions/upload", "/api/v1/grade"}
# Multipart framing adds a few KB on top of the file bytes.
UPLOAD_OVERHEAD_BYTES = 64 * 1024


class UploadGuardMiddleware:
    """Reject oversized uploads before the multipart body is spooled."""

    def __init__(self, app):
        self.app = app
        self.max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("method") == "POST":
            path = scope.get("path", "")
            if path in UPLOAD_PATHS:
                headers = dict(scope.get("headers") or [])
                declared = headers.get(b"content-length")
                try:
                    declared_bytes = int(declared) if declared else None
                except ValueError:
                    declared_bytes = None
                if (
                    declared_bytes is not None
                    and declared_bytes > self.max_bytes + UPLOAD_OVERHEAD_BYTES
                ):
                    payload = json.dumps(
                        {
                            "detail": (
                                "File exceeds the "
                                f"{settings.MAX_FILE_SIZE_MB} MB limit"
                            )
                        }
                    ).encode()
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 413,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(payload)).encode()),
                            ],
                        }
                    )
                    await send(
                        {"type": "http.response.body", "body": payload}
                    )
                    return
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loaded, damage_model_loaded
    try:
        model_loaded, damage_model_loaded = model_registry.load()
        print(
            "Models loaded: "
            f"detector={model_loaded}, damage_classifier={damage_model_loaded}, "
            f"cleanliness_classifier={model_registry.cleanliness_loaded}"
        )
    except Exception as e:
        print(f"Warning: Could not load YOLO model: {e}")

    # Schema is owned by Alembic. create_all() only creates missing tables
    # and never adds columns, so it is opt-in for throwaway databases.
    if settings.AUTO_CREATE_SCHEMA:
        Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

app.add_middleware(UploadGuardMiddleware)
# CORS must be outermost so 413s from the upload guard still carry headers.
# No cookies or credentials are used, so credentialed CORS stays off.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "PUT"],
    allow_headers=["*"],
)

# Uploaded images are not served statically: the only way to read one back
# is the per-inspection annotated download endpoint.
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

app.include_router(predictions.router, prefix="/api/v1")
app.include_router(batches.router, prefix="/api/v1")
app.include_router(dashboard.router, prefix="/api/v1")
app.include_router(user_settings.router, prefix="/api/v1")
app.include_router(compat.router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {"message": "Egg-CV API", "version": settings.VERSION}
