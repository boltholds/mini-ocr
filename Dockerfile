FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry>=2.0,<3.0"

COPY pyproject.toml poetry.lock README.md ./
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi --no-root

COPY src ./src

CMD ["uvicorn", "mini_ocr.main:app", "--host", "0.0.0.0", "--port", "8000"]
