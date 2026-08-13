FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.lock /app/requirements-runtime.lock
RUN python -m pip install --no-cache-dir -r /app/requirements-runtime.lock

COPY work /app/work
COPY assets /app/assets
COPY outputs/rag_assets /app/outputs/rag_assets

EXPOSE 8765

CMD ["python", "-m", "uvicorn", "fastapi_server:app", "--app-dir", "work", "--host", "0.0.0.0", "--port", "8765", "--no-access-log"]
