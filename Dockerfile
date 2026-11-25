FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.9.0.80

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY src ./src
COPY models ./models
COPY egg_detection/finetune_egg_v1/weights/best.pt ./egg_detection/finetune_egg_v1/weights/best.pt

RUN mkdir -p /data/uploads /data/reports
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
