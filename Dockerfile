FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG POETRY_VERSION=2.2.1

COPY pyproject.toml poetry.lock ./

RUN python -m pip install "poetry==${POETRY_VERSION}" \
    && POETRY_VIRTUALENVS_CREATE=false poetry install --only main --no-interaction --no-ansi

COPY README.md .
COPY src ./src

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

FROM base AS test

RUN POETRY_VIRTUALENVS_CREATE=false poetry install --with dev --no-interaction --no-ansi
COPY tests ./tests

USER appuser

CMD ["python", "-m", "pytest", "-q"]

FROM base AS production

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
