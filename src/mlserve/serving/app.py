"""FastAPI serving application.

Endpoints
---------
``GET  /health``              liveness + whether a model is actually loaded
``GET  /ready``               readiness: 200 only when a prediction could succeed
``GET  /model-info``          full provenance of the model currently being served
``POST /predict``             score a batch of records
``GET  /metrics``             Prometheus exposition
``GET  /monitoring/summary``  recent traffic snapshot from the prediction store
``POST /admin/reload``        re-resolve the registry alias and hot-swap the model

``/admin/reload`` is what makes promotion and rollback observable from the serving
side: the registry alias moves, the service reloads, and the next response carries the
new version. It is deliberately a separate call rather than a background poller,
because a poller would make "when did the version change?" unanswerable in a test.

Error contract
--------------
Every non-2xx response is an ``ErrorResponse`` carrying the request id, so a caller
can quote one id and the server can find the request. Four classes are distinguished:
``validation_error`` (422, the payload violates the contract), ``model_not_loaded``
(503), ``payload_too_large`` (413), and ``internal_error`` (500, never leaking a
traceback to the caller but logging it in full).
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import APIRouter, FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mlserve import __version__
from mlserve.config import Config, load_config
from mlserve.data.schema import FEATURE_NAMES, NEGATIVE_LABEL, POSITIVE_LABEL
from mlserve.logging_utils import configure_logging, get_logger, request_id_var
from mlserve.monitoring.store import PredictionStore
from mlserve.serving.metrics import ServingMetrics
from mlserve.serving.model_loader import ModelLoader, ModelNotLoadedError
from mlserve.serving.schemas import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
)

logger = get_logger("mlserve.api")

# Starlette renamed 422 to UNPROCESSABLE_CONTENT (RFC 9110); resolve it once so the
# service works on both the old and the new constant without emitting a warning.
HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", None) or status.HTTP_422_UNPROCESSABLE_ENTITY
HTTP_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", None) or status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

REQUEST_ID_HEADER = "X-Request-ID"
MODEL_VERSION_HEADER = "X-Model-Version"


class ServiceState:
    """Everything one running service instance owns.

    Held on ``app.state`` rather than in module globals so that a test can build an
    isolated instance -- separate registry, separate SQLite file -- without the
    previous one's metrics or model leaking into it.
    """

    def __init__(self, config: Config, *, loader: ModelLoader | None = None,
                 store: PredictionStore | None = None, metrics: ServingMetrics | None = None):
        self.config = config
        self.started_at = time.time()
        self.threshold = float(config.require("evaluation.decision_threshold"))
        self.max_batch_size = int(config.require("serving.max_batch_size"))
        self.loader = loader or ModelLoader(config)
        self.store = store if store is not None else PredictionStore(
            config.path("paths.prediction_db"),
            enabled=bool(config.require("serving.log_predictions")),
        )
        self.metrics = metrics or ServingMetrics(
            latency_buckets=tuple(config.require("monitoring.latency_buckets_seconds")),
            prediction_buckets=tuple(config.require("monitoring.prediction_histogram_buckets")),
            feature_sample_rate=float(config.require("serving.feature_sample_rate")),
            seed=config.seed,
        )

    @property
    def uptime(self) -> float:
        return time.time() - self.started_at

    def publish_model_identity(self) -> None:
        model = self.loader.model
        if model is None:
            self.metrics.model_loaded.set(0)
            return
        self.metrics.set_model(
            name=model.model_name, version=model.model_version, alias=model.model_alias,
            dataset_version=model.dataset_version, git_commit=model.git_commit,
        )
        self.metrics.model_loaded.set(1)


def _error_response(request_id: str, code: int, error_type: str, message: str,
                    details: list[dict] | None = None) -> JSONResponse:
    body = ErrorResponse(
        request_id=request_id,
        error=ErrorDetail(type=error_type, message=message, details=details or []),
    )
    return JSONResponse(
        status_code=code,
        content=body.model_dump(),
        headers={REQUEST_ID_HEADER: request_id},
    )


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["operations"],
            summary="Liveness and model-loaded state")
def health(request: Request) -> HealthResponse:
    """200 whenever the process is alive.

    ``status`` is ``degraded`` rather than a non-200 when no model is loaded: an
    orchestrator's liveness probe should not restart a process that is running
    correctly and merely has nothing to serve, because a restart cannot fix an empty
    registry. ``/ready`` is the probe that should gate traffic.
    """
    state: ServiceState = request.app.state.service
    model = state.loader.model
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=model.model_version if model else None,
        uptime_seconds=round(state.uptime, 3),
        version=__version__,
    )


@router.get("/ready", tags=["operations"], summary="Readiness: can this instance serve?")
def ready(request: Request) -> Response:
    state: ServiceState = request.app.state.service
    rid = request_id_var.get() or "unknown"
    if state.loader.model is None:
        return _error_response(
            rid, status.HTTP_503_SERVICE_UNAVAILABLE, "model_not_loaded",
            state.loader.last_error or "no model is loaded",
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ready", "model_version": state.loader.model.model_version},
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["model"],
            summary="Provenance of the model currently being served")
def model_info(request: Request):
    state: ServiceState = request.app.state.service
    rid = request_id_var.get() or "unknown"
    model = state.loader.model
    if model is None:
        return _error_response(
            rid, status.HTTP_503_SERVICE_UNAVAILABLE, "model_not_loaded",
            state.loader.last_error or "no model is loaded",
        )
    return ModelInfoResponse(**model.to_info())


@router.post("/predict", response_model=PredictResponse, tags=["model"],
             summary="Score one or more records",
             responses={
                 422: {"model": ErrorResponse, "description": "Payload violates the data contract"},
                 413: {"model": ErrorResponse, "description": "Batch exceeds the configured maximum"},
                 503: {"model": ErrorResponse, "description": "No model is loaded"},
             })
def predict(payload: PredictRequest, request: Request):
    state: ServiceState = request.app.state.service
    rid = request_id_var.get() or str(uuid.uuid4())
    start = time.perf_counter()

    if len(payload.records) > state.max_batch_size:
        state.metrics.observe_error("/predict", "payload_too_large")
        state.store.log_event("payload_too_large", request_id=rid, status_code=413,
                              detail=f"{len(payload.records)} records")
        return _error_response(
            rid, HTTP_413, "payload_too_large",
            f"batch of {len(payload.records)} exceeds the maximum of {state.max_batch_size}",
        )

    try:
        model = state.loader.require()
    except ModelNotLoadedError as exc:
        state.metrics.observe_error("/predict", "model_not_loaded")
        state.store.log_event("model_not_loaded", request_id=rid, status_code=503, detail=str(exc))
        return _error_response(rid, status.HTTP_503_SERVICE_UNAVAILABLE, "model_not_loaded", str(exc))

    records = [r.model_dump() for r in payload.records]
    frame = pd.DataFrame(records, columns=FEATURE_NAMES)

    try:
        probabilities = model.predict_proba(frame)
    except Exception as exc:  # pragma: no cover - exercised by failure injection
        logger.exception("prediction failed", extra={"error_type": type(exc).__name__})
        state.metrics.observe_error("/predict", "inference_error")
        state.store.log_event("inference_error", request_id=rid, status_code=500,
                              detail=f"{type(exc).__name__}: {exc}")
        return _error_response(
            rid, status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error",
            "the model failed to score this batch",
        )

    threshold = state.threshold
    probs = [float(p) for p in probabilities]
    hard = [int(p >= threshold) for p in probs]
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    state.metrics.observe_predictions(model.model_version, probs, hard)
    state.metrics.observe_features(records)
    state.store.log_predictions(
        request_id=rid, model_name=model.model_name, model_version=model.model_version,
        features=records, probabilities=probs, predictions=hard, latency_ms=elapsed_ms,
    )

    logger.info("prediction served", extra={
        "endpoint": "/predict", "n_records": len(records),
        "model_version": model.model_version, "latency_ms": round(elapsed_ms, 3),
    })

    return PredictResponse(
        request_id=rid,
        model_name=model.model_name,
        model_version=model.model_version,
        model_alias=model.model_alias,
        threshold=threshold,
        n_records=len(records),
        predictions=[
            Prediction(probability=p, prediction=h,
                       label=POSITIVE_LABEL if h else NEGATIVE_LABEL)
            for p, h in zip(probs, hard, strict=True)
        ],
        latency_ms=round(elapsed_ms, 3),
    )


@router.get("/metrics", tags=["operations"], summary="Prometheus exposition",
            response_class=Response)
def metrics(request: Request) -> Response:
    state: ServiceState = request.app.state.service
    state.metrics.refresh_resource_gauges(state.uptime)
    from mlserve.serving.metrics import CONTENT_TYPE

    return Response(content=state.metrics.render(), media_type=CONTENT_TYPE)


@router.get("/monitoring/summary", tags=["operations"],
            summary="Recent traffic and prediction distribution")
def monitoring_summary(request: Request, window_seconds: float | None = None) -> dict:
    state: ServiceState = request.app.state.service
    summary = state.store.summary(window_seconds)
    summary["uptime_seconds"] = round(state.uptime, 3)
    model = state.loader.model
    summary["served_model_version"] = model.model_version if model else None
    return summary


@router.post("/admin/reload", tags=["operations"],
             summary="Re-resolve the registry alias and hot-swap the model")
def admin_reload(request: Request) -> JSONResponse:
    """Pick up a promotion or rollback without restarting the process.

    Unauthenticated by design *for this local platform*, and called out as a known
    limitation in SYSTEM_DESIGN.md: a deployed service would put this behind
    authentication or move it off the public listener entirely.
    """
    state: ServiceState = request.app.state.service
    rid = request_id_var.get() or "unknown"
    model, details = state.loader.reload()
    if details["ok"]:
        state.publish_model_identity()
        state.metrics.record_load(outcome="success", seconds=details["seconds"])
        logger.info("model reloaded", extra=details)
    else:
        state.metrics.record_load(outcome="failure", seconds=details["seconds"])
        state.store.log_event("model_reload_failed", request_id=rid, status_code=503,
                              detail=details.get("error"))
        logger.error("model reload failed", extra=details)
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=details)
    return JSONResponse(status_code=200, content=details)


def create_app(config: Config | None = None, *, loader: ModelLoader | None = None,
               store: PredictionStore | None = None, metrics: ServingMetrics | None = None,
               load_on_startup: bool = True) -> FastAPI:
    """Build a service instance. Every collaborator is injectable, for testability."""
    config = config or load_config()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state: ServiceState = app.state.service
        if load_on_startup:
            start = time.perf_counter()
            model = state.loader.try_load()
            elapsed = time.perf_counter() - start
            if model is not None:
                state.metrics.record_load(outcome="success", seconds=elapsed)
                state.publish_model_identity()
                logger.info("model loaded at startup", extra={
                    "model_version": model.model_version, "load_seconds": round(elapsed, 4),
                    "source": model.source,
                })
            else:
                state.metrics.record_load(outcome="failure", seconds=elapsed)
                logger.error("startup model load failed", extra={"error": state.loader.last_error})
        yield
        state.store.close()

    app = FastAPI(
        title="mlserve - Adult income classifier",
        version=__version__,
        description=(
            "Serving layer for a HistGradientBoosting income classifier. Request "
            "validation is generated from the same data contract the model was "
            "trained against, and every response carries the registry version that "
            "produced it."
        ),
        lifespan=lifespan,
    )
    app.state.service = ServiceState(config, loader=loader, store=store, metrics=metrics)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Assign a request id, time the request, and record it exactly once."""
        rid = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        token = request_id_var.set(rid)
        start = time.perf_counter()
        state: ServiceState = request.app.state.service
        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - start
            state.metrics.observe_request(request.url.path, request.method, 500, elapsed)
            state.metrics.observe_error(request.url.path, "unhandled_exception")
            logger.exception("unhandled exception", extra={"endpoint": request.url.path})
            request_id_var.reset(token)
            return _error_response(rid, 500, "internal_error", "an unexpected error occurred")
        elapsed = time.perf_counter() - start
        state.metrics.observe_request(request.url.path, request.method, response.status_code, elapsed)
        response.headers[REQUEST_ID_HEADER] = rid
        model = state.loader.model
        if model is not None:
            response.headers[MODEL_VERSION_HEADER] = model.model_version
        request_id_var.reset(token)
        return response

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        """Turn Pydantic's error list into the service's own error contract."""
        rid = request_id_var.get() or "unknown"
        state: ServiceState = request.app.state.service
        details = [
            {
                "location": ".".join(str(p) for p in err.get("loc", [])),
                "message": err.get("msg", ""),
                "type": err.get("type", ""),
            }
            for err in exc.errors()[:20]
        ]
        state.metrics.observe_error(request.url.path, "validation_error")
        state.store.log_event("validation_error", request_id=rid, status_code=422,
                              detail=f"{len(exc.errors())} field error(s)")
        logger.warning("request rejected", extra={
            "endpoint": request.url.path, "n_errors": len(exc.errors()),
        })
        return _error_response(
            rid, HTTP_422, "validation_error",
            f"the request payload violates the data contract ({len(exc.errors())} error(s))",
            details,
        )

    app.include_router(router)
    return app
