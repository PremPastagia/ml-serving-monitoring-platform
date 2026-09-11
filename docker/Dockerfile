# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Serving image for the mlserve Adult income classifier.
#
# NOT BUILT OR MEASURED. No container runtime was available on the machine this
# project was developed on, so this file has been written and statically checked but
# never built. Image size and container start-up time are therefore recorded as
# UNVERIFIED in BENCHMARKS.md and are not quoted as results anywhere. Treat this as a
# reviewed deployment artefact, not as a tested one.
#
# Two stages so that build tooling never reaches the runtime image: the wheels are
# assembled in `builder` and only the installed site-packages are copied forward.
# ---------------------------------------------------------------------------

# Pinned to an exact patch version, not `3.13-slim` and not `latest`: a floating tag
# means two builds of the same commit can produce different images, which would
# undermine every reproducibility claim this project makes.
FROM python:3.13.9-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# gcc is needed only if a wheel is unavailable for this platform; it stays in the
# builder stage and never reaches the runtime image.
RUN apt-get update \
 && apt-get install --no-install-recommends -y gcc g++ \
 && rm -rf /var/lib/apt/lists/*

# Requirements are copied before the source so that a code change does not invalidate
# the dependency layer, which is by far the slowest part of the build.
COPY requirements.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN /opt/venv/bin/pip install --no-cache-dir --no-deps .

# ---------------------------------------------------------------------------

FROM python:3.13.9-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="mlserve" \
      org.opencontainers.image.description="Adult income classifier serving API" \
      org.opencontainers.image.source="https://github.com/example/ml-serving-platform"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MLSERVE_CONFIG=/app/configs/config.yaml \
    # Single-threaded BLAS/OpenMP. For one-row online inference the thread pool costs
    # more than the work it parallelises, and oversubscription across concurrent
    # workers is a common cause of tail latency in containerised model servers.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY configs ./configs
COPY src ./src

# The image serves a model bundle mounted at /app/artifacts/current. The model is
# deliberately NOT baked in: an image that contains its weights has to be rebuilt to
# promote a model, which defeats the registry. `serving.model_source: file` reads the
# mounted bundle; `registry` reaches an MLflow server instead.
RUN mkdir -p /app/artifacts /app/data

# A non-root user, created with no login shell and owning only what it needs.
RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 mlserve \
 && chown -R mlserve:mlserve /app
USER mlserve

EXPOSE 8077

# Readiness, not liveness: /ready is 503 until a model is actually loaded, so the
# orchestrator will not route traffic to an instance that cannot answer.
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import sys,urllib.request;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8077/ready',timeout=4).status==200 else 1)"

ENTRYPOINT ["uvicorn", "mlserve.serving.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8077", "--workers", "1", "--no-access-log"]
