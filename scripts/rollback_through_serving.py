#!/usr/bin/env python
"""Phase 7: prove rollback works *through the serving layer*, not just in the registry.

Moving a registry alias is trivial and proves nothing about the running service. This
script measures the thing that actually matters:

1. Register a second model version if the registry has only one.
2. Start the real uvicorn server and record the version it is serving.
3. Start a continuous background prediction load, so the system is *in use* throughout.
4. Promote the other version, reload, and confirm the API now reports the new version.
5. Roll the registry alias back, reload, and confirm the API reports the original.
6. Count every request issued during the switch and how many failed.

Step 6 is the point. "Zero-downtime" is a claim about requests in flight during the
change, so it can only be made by issuing requests during the change and counting the
failures. This script reports that count; it does not assume it.

    python scripts/rollback_through_serving.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from mlserve.config import load_config
from mlserve.data.schema import FEATURE_NAMES
from mlserve.models.registry import PREVIOUS, PRODUCTION, ModelRegistry
from mlserve.models.tracking import log_training_run
from mlserve.models.train import load_and_split, train_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_test import ServerProcess  # noqa: E402


@dataclass
class ContinuityProbe:
    """Issues predictions continuously from several threads and records every outcome.

    Several threads, not one: a single serial prober issues a request only every few
    hundred milliseconds under load, so a sub-second alias switch can complete with no
    request in flight at all -- and "0 of 0 requests failed" is not evidence of
    anything. Concurrent probers guarantee the switch window actually contains traffic.

    Each record is ``(request_start, request_end, status, model_version)``. Both
    timestamps are kept so a request that *spanned* the switch can be identified,
    which is the case a continuity claim is really about.
    """

    base_url: str
    payload: dict
    interval: float = 0.0
    threads: int = 4
    results: list[tuple[float, float, int, str | None]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _threads: list[threading.Thread] = field(default_factory=list)

    def _run(self) -> None:
        with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
            while not self._stop.is_set():
                start = time.perf_counter()
                try:
                    response = client.post("/predict", json=self.payload)
                    end = time.perf_counter()
                    version = (response.json().get("model_version")
                               if response.status_code == 200 else None)
                    record = (start, end, response.status_code, version)
                except httpx.HTTPError as exc:
                    record = (start, time.perf_counter(), -1, f"{type(exc).__name__}: {exc}")
                with self._lock:
                    self.results.append(record)
                if self.interval:
                    time.sleep(self.interval)

    def start(self) -> None:
        self._threads = [threading.Thread(target=self._run, daemon=True)
                         for _ in range(self.threads)]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=15)

    def window(self, start: float, end: float) -> list[tuple[float, float, int, str | None]]:
        """Every request that overlapped [start, end] -- started, finished, or spanned."""
        with self._lock:
            return [r for r in self.results if r[1] >= start and r[0] <= end]

    @staticmethod
    def summarise(rows: list[tuple[float, float, int, str | None]]) -> dict:
        total = len(rows)
        ok = sum(1 for _, _, status, _ in rows if status == 200)
        return {
            "requests": total,
            "succeeded": ok,
            "failed": total - ok,
            "failure_rate": round((total - ok) / total, 6) if total else None,
            "status_codes": {str(code): sum(1 for _, _, s, _ in rows if s == code)
                             for code in sorted({s for _, _, s, _ in rows})},
            "versions_observed": sorted({v for _, _, s, v in rows if s == 200 and v}),
        }


def ensure_two_versions(config, registry: ModelRegistry) -> None:
    """Register a second, deliberately weaker version if one does not already exist."""
    if len(registry.list_versions()) >= 2:
        return
    print("registry has fewer than two versions; training a second one")
    split = load_and_split(config)
    params = {**dict(config.require("model.params")), "learning_rate": 0.01,
              "max_leaf_nodes": 3}
    pipeline, run = train_model(config, split=split, params=params,
                                run_name="rollback-target", evaluate_test=True)
    X_val, _ = split.xy("validation")
    logged = log_training_run(pipeline, run, config, input_example=X_val,
                              tags={"purpose": "rollback-target"})
    ref = registry.register(logged.model_uri, tags={
        "purpose": "rollback-target",
        "validation_roc_auc": f"{run.metrics['validation_roc_auc']:.6f}",
        "test_roc_auc": f"{run.metrics['test_roc_auc']:.6f}",
        "dataset_version": run.dataset_version,
        "git_commit": run.git_commit,
        "training_fingerprint": run.training_fingerprint,
    })
    print(f"  registered version {ref.version}")


def served_version(base_url: str) -> str:
    return httpx.get(f"{base_url}/model-info", timeout=10.0).json()["model_version"]


def switch(base_url: str, action, label: str) -> dict:
    """Perform a registry alias change plus a reload, timing the whole switch.

    The switch window is only *recorded* here. Summarising the traffic inside it has to
    wait until the probe threads have drained, because a request that was in flight
    during the switch has not been appended to the probe's results yet -- and those are
    precisely the requests a continuity claim is about.
    """
    before = served_version(base_url)
    start = time.perf_counter()
    registry_result = action()
    registry_done = time.perf_counter()
    reload_result = httpx.post(f"{base_url}/admin/reload", timeout=60.0).json()
    end = time.perf_counter()
    after = served_version(base_url)

    return {
        "label": label,
        "served_before": before,
        "served_after": after,
        "changed": before != after,
        "registry_alias_seconds": round(registry_done - start, 6),
        "reload_seconds": reload_result.get("seconds"),
        "end_to_end_seconds": round(end - start, 6),
        "registry_result": registry_result,
        "reload_result": reload_result,
        "_window": (start, end),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--probe-interval", type=float, default=0.0)
    parser.add_argument("--probe-threads", type=int, default=4)
    parser.add_argument("--settle", type=float, default=2.0,
                        help="seconds of steady traffic before and after each switch")
    args = parser.parse_args(argv)

    config = load_config()
    registry = ModelRegistry(config)
    ensure_two_versions(config, registry)

    versions = [v.version for v in registry.list_versions()]
    if len(versions) < 2:
        print("ERROR: need at least two registered versions", file=sys.stderr)
        return 1

    if registry.production() is None:
        registry.set_alias(PRODUCTION, versions[0])
    original = registry.production().version
    other = next(v for v in versions if v != original)
    print(f"versions   : {versions}")
    print(f"production : {original} -> will switch to {other} and back\n")

    split = load_and_split(config)
    payload = {"records": split.test[FEATURE_NAMES].head(8).to_dict(orient="records")}

    host = str(config.require("serving.host"))
    port = args.port or int(config.require("serving.port"))
    results_dir = config.path("paths.results_dir") / "serving"
    results_dir.mkdir(parents=True, exist_ok=True)
    server = ServerProcess(host, port, log_path=results_dir / "rollback_server.log")
    server.start()
    print(f"server ready in {server.cold_start_seconds:.3f}s, serving version "
          f"{served_version(server.base_url)}")

    probe = ContinuityProbe(server.base_url, payload, interval=args.probe_interval,
                            threads=args.probe_threads)
    probe.start()
    time.sleep(args.settle)

    try:
        promotion = switch(server.base_url, lambda: registry.promote(other), "promote")
        time.sleep(args.settle)
        rollback = switch(server.base_url, lambda: registry.rollback(), "rollback")
        time.sleep(args.settle)
    finally:
        probe.stop()
        server.stop()

    # Now that every probe thread has finished, requests that spanned a switch are
    # present in the results and can be attributed to their window.
    for record in (promotion, rollback):
        start, end = record.pop("_window")
        record["switch_window"] = {"start": round(start, 6), "end": round(end, 6)}
        record["traffic_during_switch"] = ContinuityProbe.summarise(probe.window(start, end))
        print(f"{record['label']:9s}: {record['served_before']} -> {record['served_after']} "
              f"in {record['end_to_end_seconds']:.3f}s, "
              f"{record['traffic_during_switch']['failed']} failed request(s) of "
              f"{record['traffic_during_switch']['requests']} in flight")

    overall = ContinuityProbe.summarise(probe.results)
    success = (
        promotion["served_after"] == other
        and rollback["served_after"] == original
        and rollback["changed"]
        # A continuity claim requires traffic to have been in flight during the switch.
        and promotion["traffic_during_switch"]["requests"] > 0
        and rollback["traffic_during_switch"]["requests"] > 0
    )

    report = {
        "registry_versions": versions,
        "original_production_version": original,
        "switched_to_version": other,
        "promotion": promotion,
        "rollback": rollback,
        "overall_traffic": overall,
        "probe_interval_seconds": args.probe_interval,
        "probe_threads": args.probe_threads,
        "host_load_average": os.getloadavg(),
        "final_production_version": registry.production().version,
        "final_previous_alias": (registry.resolve_alias(PREVIOUS).version
                                 if registry.resolve_alias(PREVIOUS) else None),
        "rollback_verified_through_serving": success,
    }
    (results_dir / "rollback_through_serving.json").write_text(json.dumps(report, indent=2))

    print(f"\ntotal probe traffic: {overall['requests']} requests, "
          f"{overall['failed']} failed, versions seen {overall['versions_observed']}")
    print(f"final production version: {report['final_production_version']} "
          f"(expected {original})")

    if not success:
        print("ROLLBACK_THROUGH_SERVING_FAILED", file=sys.stderr)
        return 1
    print("ROLLBACK_THROUGH_SERVING_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
