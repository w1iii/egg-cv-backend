"""Apply the upload retention policy: remove old inspection images.

Policy (docs/security-checklist.md): uploaded and annotated images older
than the retention window are deleted from UPLOAD_DIR. Inspection records,
per-egg results and the CSV export are kept, so history and statistics stay
intact; the history view reports the image as no longer available.

Dry run by default. Nothing is deleted without ``--apply``.

    python scripts/purge_uploads.py --older-than-days 90           # list
    python scripts/purge_uploads.py --older-than-days 90 --apply   # delete
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import Prediction  # noqa: E402
from app.services.prediction_service import is_within_directory  # noqa: E402


def purge(older_than_days: int, apply: bool, now: datetime | None = None) -> dict:
    if older_than_days < 1:
        raise ValueError("--older-than-days must be at least 1")
    cutoff = (now or datetime.utcnow()) - timedelta(days=older_than_days)
    summary = {"inspections": 0, "files": 0, "bytes": 0, "skipped_outside_upload_dir": 0}
    with SessionLocal() as db:
        rows = db.query(Prediction).filter(Prediction.created_at < cutoff).all()
        for row in rows:
            touched = False
            for attr in ("file_path", "annotated_path"):
                path = getattr(row, attr)
                if not path or not Path(path).is_file():
                    continue
                if not is_within_directory(path, settings.UPLOAD_DIR):
                    summary["skipped_outside_upload_dir"] += 1
                    continue
                summary["files"] += 1
                summary["bytes"] += Path(path).stat().st_size
                touched = True
                print(("DELETE " if apply else "would delete ") + path)
                if apply:
                    Path(path).unlink(missing_ok=True)
            if touched:
                summary["inspections"] += 1
                if apply:
                    row.annotated_path = None
        if apply:
            db.commit()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--older-than-days", type=int, required=True)
    parser.add_argument("--apply", action="store_true", help="actually delete files")
    args = parser.parse_args()
    summary = purge(args.older_than_days, args.apply)
    mode = "deleted" if args.apply else "dry run, would delete"
    print(f"{mode}: {summary['files']} files ({summary['bytes']} bytes) from "
          f"{summary['inspections']} inspections; skipped outside upload dir: "
          f"{summary['skipped_outside_upload_dir']}")


if __name__ == "__main__":
    main()
