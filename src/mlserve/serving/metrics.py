"""Prometheus instrumentation for the serving layer.

What is measured and why
------------------------
=========================== =========== ==================================================
Series                      Type        Question it answers
=========================== =========== ==================================================
mlserve_requests_total      Counter     Is traffic arriving, and on which endpoint/status?
mlserve_errors_total        Counter     What fraction of requests fail, and why?
mlserve_request_latency     Histogram   Are p50/p95/p99 within budget? (quantiles need a
                                        histogram, not a gauge of the last value)
mlserve_predictions_total   Counter     How many records -- not requests -- were scored?
mlserve_prediction_score    Histogram   Has the *output* distribution shifted? This is the
                                        earliest drift signal available without labels.
mlserve_predicted_class     Counter     Has the positive rate moved?
mlserve_feature_value       Histogram   Has an *input* numeric distribution shifted?
mlserve_feature_level       Counter     Has a categorical level's share shifted?
mlserve_model_info          Gauge       Which model version served this traffic? (labels
                                        carry the identity; the value is a constant 1)
mlserve_model_load          Gauge/Ctr   Cold-start cost and reload/failure counts.
mlserve_resource_*          Gauge       Process CPU and RSS, sampled on scrape.
=========================== =========== ==================================================

Sampling. Request, error, latency and prediction-count series are exact. The
per-feature distribution series are the expensive ones -- 12 features times every
record -- so they are sampled at ``serving.feature_sample_rate`` (1.0 locally). The
durable SQLite store, not Prometheus, is the authority for drift analysis; these
series exist so a dashboard can show the shift without a query.

Cardinality. ``mlserve_feature_level`` is restricted to the low-cardinality
categoricals. ``native_country`` has 41 levels and ``occupation`` 14; exporting every
level of every categorical as its own time series is how a metrics backend falls over,
so the high-cardinality ones are tracked in SQLite only.
"""

from __future__ import annotations

import os
import random
import threading

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST as OPENMETRICS_TYPE

from mlserve.data.schema import NUMERIC_FEATURES

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

#: Categoricals whose level shares are exported as counters. Chosen by cardinality:
#: every one of these has at most 8 levels.
LOW_CARDINALITY_CATEGORICALS = ["sex", "race", "workclass", "relationship"]

#: Per-feature histogram buckets. Ranges come from the contract, so a bucket edge
#: cannot drift away from the values it is meant to separate.
FEATURE_BUCKETS: dict[str, tuple[float, ...]] = {
    "age": (20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 90),
    "education_num": (4, 6, 8, 9, 10, 11, 12, 13, 14, 16),
    "capital_gain": (1, 1000, 3000, 5000, 7000, 10000, 15000, 25000, 50000, 99999),
    "capital_loss": (1, 500, 1000, 1500, 1800, 2000, 2500, 4356),
    "hours_per_week": (10, 20, 30, 35, 40, 45, 50, 60, 80, 99),
}


class ServingMetrics:
    """All Prometheus series for one service instance.

    Bound to an explicit ``CollectorRegistry`` rather than the process-global default
    so that a test can build a fresh instance without duplicate-timeseries errors --
    the usual reason metrics end up untested.
    """

    def __init__(
        self,
        *,
        registry: CollectorRegistry | None = None,
        latency_buckets: tuple[float, ...] = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        prediction_buckets: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
        feature_sample_rate: float = 1.0,
        seed: int | None = None,
    ):
        self.registry = registry if registry is not None else CollectorRegistry()
        self.feature_sample_rate = float(feature_sample_rate)
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

        self.requests = Counter(
            "mlserve_requests", "Requests handled, by endpoint, method and status class.",
            ["endpoint", "method", "status"], registry=self.registry,
        )
        self.errors = Counter(
            "mlserve_errors", "Failed requests, by endpoint and error type.",
            ["endpoint", "error_type"], registry=self.registry,
        )
        self.latency = Histogram(
            "mlserve_request_latency_seconds", "Server-side request handling time.",
            ["endpoint"], buckets=latency_buckets, registry=self.registry,
        )
        self.predictions = Counter(
            "mlserve_predictions", "Individual records scored.",
            ["model_version"], registry=self.registry,
        )
        self.prediction_score = Histogram(
            "mlserve_prediction_score", "Distribution of predicted positive-class probability.",
            ["model_version"], buckets=prediction_buckets, registry=self.registry,
        )
        self.predicted_class = Counter(
            "mlserve_predicted_class", "Predicted hard labels.",
            ["model_version", "predicted_class"], registry=self.registry,
        )
        # prometheus_client fixes bucket edges per metric, so each numeric feature gets
        # its own histogram rather than one histogram with a `feature` label: age and
        # capital_gain span completely different ranges and cannot share edges.
        self.feature_histograms: dict[str, Histogram] = {
            name: Histogram(
                f"mlserve_feature_{name}", f"Distribution of incoming feature '{name}'.",
                buckets=FEATURE_BUCKETS.get(name, (0.0, 1.0)), registry=self.registry,
            )
            for name in NUMERIC_FEATURES
        }
        self.feature_level = Counter(
            "mlserve_feature_level", "Observed levels of the low-cardinality categoricals.",
            ["feature", "level"], registry=self.registry,
        )
        self.model_info = Gauge(
            "mlserve_model_info", "Always 1; the labels carry the served model's identity.",
            ["model_name", "model_version", "model_alias", "dataset_version", "git_commit"],
            registry=self.registry,
        )
        self.model_loaded = Gauge(
            "mlserve_model_loaded", "1 when a model is loaded and able to serve.",
            registry=self.registry,
        )
        self.model_load_seconds = Gauge(
            "mlserve_model_load_seconds", "Wall-clock time of the most recent model load.",
            registry=self.registry,
        )
        self.model_loads = Counter(
            "mlserve_model_loads", "Model load attempts, by outcome.",
            ["outcome"], registry=self.registry,
        )
        self.uptime = Gauge("mlserve_uptime_seconds", "Process uptime.", registry=self.registry)
        self.cpu_percent = Gauge("mlserve_resource_cpu_percent", "Process CPU utilisation.", registry=self.registry)
        self.memory_bytes = Gauge("mlserve_resource_memory_bytes", "Process resident set size.", registry=self.registry)

        self._process = None
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
            self._process.cpu_percent(interval=None)  # prime the first delta
        except Exception:
            self._process = None

    # ------------------------------------------------------------------ recording

    def observe_request(self, endpoint: str, method: str, status_code: int, seconds: float) -> None:
        self.requests.labels(endpoint=endpoint, method=method, status=str(status_code)).inc()
        self.latency.labels(endpoint=endpoint).observe(seconds)

    def observe_error(self, endpoint: str, error_type: str) -> None:
        self.errors.labels(endpoint=endpoint, error_type=error_type).inc()

    def observe_predictions(
        self, model_version: str, probabilities: list[float], predictions: list[int]
    ) -> None:
        self.predictions.labels(model_version=model_version).inc(len(probabilities))
        for prob, pred in zip(probabilities, predictions, strict=True):
            self.prediction_score.labels(model_version=model_version).observe(prob)
            self.predicted_class.labels(
                model_version=model_version, predicted_class=str(int(pred))
            ).inc()

    def observe_features(self, records: list[dict]) -> int:
        """Record input distributions for a sampled subset. Returns rows sampled."""
        if not records or self.feature_sample_rate <= 0.0:
            return 0
        sampled = 0
        for record in records:
            if self.feature_sample_rate < 1.0:
                with self._lock:
                    keep = self._rng.random() < self.feature_sample_rate
                if not keep:
                    continue
            sampled += 1
            for name, hist in self.feature_histograms.items():
                value = record.get(name)
                if value is not None:
                    hist.observe(float(value))
            for name in LOW_CARDINALITY_CATEGORICALS:
                level = record.get(name)
                if level is not None:
                    self.feature_level.labels(feature=name, level=str(level)).inc()
        return sampled

    def set_model(self, *, name: str, version: str, alias: str | None,
                  dataset_version: str | None, git_commit: str | None) -> None:
        """Publish the served model's identity, clearing any previous one.

        The gauge is cleared first so that after a promotion or rollback exactly one
        `mlserve_model_info` series is present. Leaving the old labels behind would
        make a dashboard show two "current" versions at once.
        """
        self.model_info.clear()
        self.model_info.labels(
            model_name=name, model_version=version, model_alias=alias or "none",
            dataset_version=dataset_version or "unknown", git_commit=git_commit or "unknown",
        ).set(1)

    def record_load(self, *, outcome: str, seconds: float | None = None) -> None:
        self.model_loads.labels(outcome=outcome).inc()
        if seconds is not None:
            self.model_load_seconds.set(seconds)
        self.model_loaded.set(1 if outcome == "success" else 0)

    def refresh_resource_gauges(self, uptime_seconds: float) -> None:
        """Sampled at scrape time: polling CPU on every request would cost more than
        the prediction it is measuring."""
        self.uptime.set(uptime_seconds)
        if self._process is None:
            return
        try:
            self.cpu_percent.set(self._process.cpu_percent(interval=None))
            self.memory_bytes.set(self._process.memory_info().rss)
        except Exception:
            pass

    def render(self) -> bytes:
        return generate_latest(self.registry)


__all__ = ["CONTENT_TYPE", "OPENMETRICS_TYPE", "FEATURE_BUCKETS",
           "LOW_CARDINALITY_CATEGORICALS", "ServingMetrics"]
