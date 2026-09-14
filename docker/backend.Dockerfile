# CareCopilot API.
#
# Two stages so the runtime image carries no compiler and no build cache.
# The wheels are built once against the same Python that will run them, then
# copied into a slim image that has only what a request needs.
#
# Not python:3.14-alpine: psycopg and onnxruntime (fastembed's backend) both
# ship manylinux wheels and no musl ones, so Alpine would rebuild them from
# source — a much larger image, built much more slowly, for no benefit.

# --- build ---------------------------------------------------------------- #
FROM python:3.14-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# build-essential is needed by any dependency without a wheel for this
# platform; it stays in this stage and never reaches the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY backend/requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# --- runtime -------------------------------------------------------------- #
FROM python:3.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # fastembed downloads the ONNX model on first use; pinning the cache to a
    # named volume means one download for the life of the stack rather than
    # one per container start.
    FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed

# curl is here for the container healthcheck below, not for the application.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

COPY --from=build /opt/venv /opt/venv

# The container mirrors the repository layout rather than flattening it.
# scripts/*.py resolve the backend as `parents[1]/"backend"` relative to
# themselves, so a layout where backend/ is not a sibling of scripts/ breaks
# every one of them with an ImportError at startup.
WORKDIR /srv
COPY --chown=app:app backend/ /srv/backend/
COPY --chown=app:app scripts/ /srv/scripts/
COPY --chown=app:app evaluation/ /srv/evaluation/

RUN mkdir -p /home/app/.cache/fastembed && chown -R app:app /home/app/.cache
USER app

# Both entrypoints run from here: alembic.ini lives in backend/, and
# `python run_server.py` needs backend/ as the working directory for
# `app.main:app` to import.
WORKDIR /srv/backend

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

# run_server.py rather than `uvicorn app.main:app`: uvicorn selects the event
# loop itself, and the application needs to own that choice (see app/runtime.py).
# The indirection is load-bearing on Windows and harmless here, so both
# environments start the server the same way.
#
# --host 0.0.0.0 is required, not stylistic: the script defaults to
# 127.0.0.1, which inside a container accepts only connections originating
# in that same container — the port mapping would resolve to nothing.
CMD ["python", "run_server.py", "--host", "0.0.0.0", "--port", "8000"]
