#!/usr/bin/env python
"""Phase 3: a reproducible HTTP load test against the running service.

    python scripts/load_test.py --duration 20 --concurrency 8 --batch-size 1

Why it is built this way
------------------------
* **It starts and stops its own server.** Benchmarking whatever happens to be running
  is how a number becomes unreproducible. The script launches uvicorn, waits for
  ``/health``, runs, then shuts down, and records the exact command it used.
* **Payloads are drawn from the real test split with a fixed seed**, so two runs send
  the same bytes. Sending one repeated record would let every layer cache and would
  measure the cache.
* **A warm-up phase is excluded from the statistics.** The first requests pay lazy
  imports and first-call allocation inside scikit-learn; folding those into p99 makes
  the tail meaningless. Cold-start is measured separately and reported on its own.
* **The host load average is recorded with every result.** On a shared or busy machine
  latency percentiles are dominated by scheduler queueing rather than by the service,
  and a latency number without that context is not interpretable.

Reported: throughput, p50/p95/p99 latency, error rate, server CPU and RSS, and
cold-start time.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import psutil

from mlserve.config import environment_info, load_config, project_root
from mlserve.data.schema import FEATURE_NAMES
from mlserve.models.train import load_and_split

STARTUP_TIMEOUT = 120.0

#: Fraction of total CPU already consumed by *other* processes above which a latency
#: measurement stops describing the service and starts describing the run queue.
#: 0.35 leaves the majority of the machine free for the server and the load generator,
#: which share this host.
BUSY_CPU_FRACTION = 0.35

#: Seconds spent sampling CPU utilisation before deciding whether the host is busy.
CPU_SAMPLE_SECONDS = 3.0


def host_pressure(sample_seconds: float = CPU_SAMPLE_SECONDS) -> dict:
    """Measure how much of this machine is already in use.

    Deliberately based on *measured CPU utilisation*, not on the load average. On
    macOS the load average counts threads blocked in uninterruptible I/O as well as
    runnable ones, so it routinely reads 30-40 on a 10-core laptop that is in fact
    only ~20% busy -- which made an earlier load-average guard flag every single run
    as unusable. Utilisation answers the question that actually matters: is there a
    free core for the server to run on?

    The load average is still recorded, because it is the number a reader will expect
    to see, but it is annotated rather than used as the gate.
    """
    utilisation = psutil.cpu_percent(interval=sample_seconds) / 100.0
    cpus = psutil.cpu_count() or 1
    load = os.getloadavg()[0]
    return {
        "cpu_utilisation": round(utilisation, 4),
        "cpu_count": cpus,
        "free_cpus": round(cpus * (1.0 - utilisation), 2),
        "load_average_1m": round(load, 2),
        "busy": utilisation > BUSY_CPU_FRACTION,
    }


def wait_for_quiet_host(timeout: float, *, poll: float = 15.0) -> dict:
    """Block until the host has enough free CPU to benchmark, or give up."""
    deadline = time.monotonic() + timeout
    pressure = host_pressure()
    while time.monotonic() < deadline and pressure["busy"]:
        remaining = int(deadline - time.monotonic())
        print(f"  host busy: {pressure['cpu_utilisation']:.0%} CPU in use "
              f"({pressure['free_cpus']} of {pressure['cpu_count']} cores free, "
              f"limit {BUSY_CPU_FRACTION:.0%}); waiting up to {remaining}s")
        time.sleep(poll)
        pressure = host_pressure()
    return pressure


@dataclass
class LoadTestResult:
    scenario: str
    concurrency: int
    batch_size: int
    server_workers: int
    server_thread_pin: int | None
    duration_seconds: float
    warmup_seconds: float
    requests_total: int
    requests_ok: int
    requests_failed: int
    records_scored: int
    error_rate: float
    throughput_rps: float
    throughput_records_per_second: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float
    server_cpu_percent_mean: float | None
    server_cpu_percent_max: float | None
    server_rss_mb_mean: float | None
    server_rss_mb_max: float | None
    cold_start_seconds: float | None
    #: Measured CPU utilisation of the whole machine during this run, the server and
    #: load generator included. This is the honest context for every latency figure.
    host_cpu_utilisation: float
    host_load_1m: float
    host_cpu_count: int
    #: True when *other* work was already consuming more than BUSY_CPU_FRACTION of the
    #: machine when this run started. Latency percentiles from such a run reflect OS
    #: scheduling delay as much as the service and must not be quoted as performance.
    host_busy_before_run: bool
    model_version: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def find_free_port(host: str, preferred: int, *, attempts: int = 40) -> int:
    """Return ``preferred`` if bindable, else the next free port after it.

    A leftover server from an interrupted run holds the configured port, and uvicorn
    then exits immediately while ``/health`` still answers -- from the *old* process.
    The benchmark would silently measure a stale build. Refusing to reuse a busy port
    makes that impossible.
    """
    for offset in range(attempts):
        candidate = preferred + offset
        if port_is_free(host, candidate):
            return candidate
    raise RuntimeError(f"no free port in {preferred}..{preferred + attempts - 1} on {host}")


class ServerProcess:
    """Starts uvicorn, waits for readiness, and samples its resource usage."""

    def __init__(self, host: str, port: int, *, log_path: Path, workers: int = 1,
                 thread_pin: int | None = None):
        self.host, self.port = host, port
        self.workers = workers
        #: When set, caps the OpenMP/BLAS thread pools inside the server process.
        #: scikit-learn's HistGradientBoosting predicts through OpenMP and will fan a
        #: single-row prediction across every core by default, which saturates the
        #: machine and makes request concurrency actively counterproductive.
        self.thread_pin = thread_pin
        self.base_url = f"http://{host}:{port}"
        self.log_path = log_path
        self.process: subprocess.Popen | None = None
        self.cold_start_seconds: float | None = None
        self._samples: list[tuple[float, float]] = []
        self._sampling = False
        self._thread: threading.Thread | None = None
        self.command = [
            sys.executable, "-m", "uvicorn", "mlserve.serving.main:app",
            "--host", host, "--port", str(port), "--workers", str(workers),
            "--log-level", "warning", "--no-access-log",
        ]

    def start(self) -> None:
        if not port_is_free(self.host, self.port):
            raise RuntimeError(
                f"{self.host}:{self.port} is already in use. A previous server is still "
                f"running; stop it first, or pass --port. Refusing to start, because "
                f"/health would answer from the stale process and the benchmark would "
                f"measure it instead of this build."
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        if self.thread_pin is not None:
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[name] = str(self.thread_pin)
        self.env_overrides = {k: env[k] for k in
                              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
                              if k in env}
        started = time.perf_counter()
        with open(self.log_path, "wb") as log:
            self.process = subprocess.Popen(
                self.command, cwd=project_root(), stdout=log, stderr=subprocess.STDOUT,
                env=env,
            )
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"server exited with code {self.process.returncode}; see {self.log_path}"
                )
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=2.0)
                if response.status_code == 200 and response.json().get("model_loaded"):
                    self.cold_start_seconds = time.perf_counter() - started
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise TimeoutError(f"server not ready within {STARTUP_TIMEOUT}s; see {self.log_path}")

    def begin_sampling(self, interval: float = 0.25) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            proc = psutil.Process(self.process.pid)
            proc.cpu_percent(interval=None)
            for child in proc.children(recursive=True):
                child.cpu_percent(interval=None)
        except (psutil.Error, ProcessLookupError):
            # The server died between readiness and here. Resource sampling is
            # optional; the request loop below will surface the real failure.
            return
        self._sampling = True

        def sample():
            while self._sampling:
                try:
                    # With --workers > 1 the parent only supervises; the real CPU and
                    # memory live in the children, so the family is summed.
                    family = [proc, *proc.children(recursive=True)]
                    cpu = sum(p.cpu_percent(interval=None) for p in family)
                    rss = sum(p.memory_info().rss for p in family) / (1024 * 1024)
                    self._samples.append((cpu, rss))
                except psutil.Error:
                    return
                time.sleep(interval)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def end_sampling(self) -> dict:
        self._sampling = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self._samples:
            return {"cpu_mean": None, "cpu_max": None, "rss_mean": None, "rss_max": None}
        cpu = [c for c, _ in self._samples if c is not None]
        rss = [r for _, r in self._samples]
        return {
            "cpu_mean": round(statistics.fmean(cpu), 2) if cpu else None,
            "cpu_max": round(max(cpu), 2) if cpu else None,
            "rss_mean": round(statistics.fmean(rss), 2),
            "rss_max": round(max(rss), 2),
        }

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


def build_payloads(n: int, batch_size: int, seed: int) -> list[dict]:
    """Deterministic payloads drawn from the held-out test split."""
    split = load_and_split(load_config())
    frame = split.test[FEATURE_NAMES]
    rng = np.random.default_rng(seed)
    payloads = []
    for _ in range(n):
        idx = rng.choice(len(frame), size=batch_size, replace=False)
        payloads.append({"records": frame.iloc[idx].to_dict(orient="records")})
    return payloads


def worker(base_url: str, payloads: list[dict], stop_at: float, results: list,
           lock: threading.Lock, offset: int) -> None:
    latencies, ok, failed, records = [], 0, 0, 0
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        i = offset
        while time.perf_counter() < stop_at:
            payload = payloads[i % len(payloads)]
            i += 1
            start = time.perf_counter()
            try:
                response = client.post("/predict", json=payload)
                elapsed = (time.perf_counter() - start) * 1000.0
                latencies.append(elapsed)
                if response.status_code == 200:
                    ok += 1
                    records += len(payload["records"])
                else:
                    failed += 1
            except httpx.HTTPError:
                latencies.append((time.perf_counter() - start) * 1000.0)
                failed += 1
    with lock:
        results.append((latencies, ok, failed, records))


def run_phase(base_url: str, payloads: list[dict], *, concurrency: int,
              duration: float, label: str) -> tuple[list[float], int, int, int, float]:
    results: list = []
    lock = threading.Lock()
    stop_at = time.perf_counter() + duration
    started = time.perf_counter()
    threads = [
        threading.Thread(target=worker,
                         args=(base_url, payloads, stop_at, results, lock, t * 37))
        for t in range(concurrency)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started

    latencies = [value for batch, _, _, _ in results for value in batch]
    ok = sum(r[1] for r in results)
    failed = sum(r[2] for r in results)
    records = sum(r[3] for r in results)
    return latencies, ok, failed, records, wall


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=20.0, help="measured seconds")
    parser.add_argument("--warmup", type=float, default=5.0, help="excluded warm-up seconds")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--payloads", type=int, default=256)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--thread-pin", type=int, default=1,
                        help="cap OpenMP/BLAS threads inside the server. 1 is the "
                             "production setting and the default; pass 0 to leave the "
                             "pools uncapped, which is the documented comparison case.")
    parser.add_argument("--workers", type=int, default=1,
                        help="uvicorn worker processes; >1 tests whether the throughput "
                             "ceiling is the GIL rather than the model")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wait-for-quiet-host", type=float, default=0.0,
                        help="seconds to wait for the host load to fall below the limit")
    args = parser.parse_args(argv)

    config = load_config()
    seed = args.seed if args.seed is not None else config.seed
    host = str(config.require("serving.host"))
    port = args.port or int(config.require("serving.port"))
    results_dir = config.path("paths.results_dir") / "serving"
    results_dir.mkdir(parents=True, exist_ok=True)

    pressure = host_pressure()
    print(f"host: {pressure['cpu_count']} CPUs, {pressure['cpu_utilisation']:.0%} in use "
          f"by other work ({pressure['free_cpus']} cores free); "
          f"1m load average {pressure['load_average_1m']} "
          f"(not used as the gate -- see host_pressure())")
    if pressure["busy"] and args.wait_for_quiet_host > 0:
        pressure = wait_for_quiet_host(args.wait_for_quiet_host)
    if pressure["busy"]:
        print(f"WARNING: {pressure['cpu_utilisation']:.0%} of this machine is already "
              f"busy (limit {BUSY_CPU_FRACTION:.0%}). Latency percentiles from this run "
              f"measure OS scheduling delay as much as the service, and every affected "
              f"row is flagged host_busy_before_run=True.\n")

    chosen_port = find_free_port(host, port)
    if chosen_port != port:
        print(f"port {port} is busy; using {chosen_port} instead")
    tag = f"{'pin' if args.thread_pin else 'nopin'}_w{args.workers}"
    server = ServerProcess(host, chosen_port, workers=args.workers,
                           thread_pin=args.thread_pin or None,
                           log_path=results_dir / f"load_test_server_{tag}.log")
    print(f"starting: {' '.join(server.command)}")
    server.start()
    print(f"ready in {server.cold_start_seconds:.3f}s (cold start)\n")

    rows: list[LoadTestResult] = []
    try:
        model_version = httpx.get(f"{server.base_url}/health", timeout=5.0).json().get("model_version")
        for batch_size in args.batch_size:
            payloads = build_payloads(args.payloads, batch_size, seed)
            for concurrency in args.concurrency:
                pin = "pin" if args.thread_pin else "nopin"
                label = f"{pin}-w{args.workers}-c{concurrency}-b{batch_size}"
                # Sample the host immediately before the measured phase, while only the
                # warm-up traffic is running, so "was someone else using the machine?"
                # is answered without counting our own load.
                before = host_pressure(sample_seconds=1.5)
                run_phase(server.base_url, payloads, concurrency=concurrency,
                          duration=args.warmup, label=f"{label}-warmup")
                psutil.cpu_percent(interval=None)
                server.begin_sampling()
                latencies, ok, failed, records, wall = run_phase(
                    server.base_url, payloads, concurrency=concurrency,
                    duration=args.duration, label=label,
                )
                usage = server.end_sampling()

                during_cpu = round(psutil.cpu_percent(interval=None) / 100.0, 4)
                total = ok + failed
                array = np.array(latencies) if latencies else np.array([float("nan")])
                result = LoadTestResult(
                    scenario=label,
                    concurrency=concurrency,
                    batch_size=batch_size,
                    server_workers=args.workers,
                    server_thread_pin=args.thread_pin,
                    duration_seconds=round(wall, 3),
                    warmup_seconds=args.warmup,
                    requests_total=total,
                    requests_ok=ok,
                    requests_failed=failed,
                    records_scored=records,
                    error_rate=round(failed / total, 6) if total else 0.0,
                    throughput_rps=round(total / wall, 2),
                    throughput_records_per_second=round(records / wall, 2),
                    p50_ms=round(float(np.percentile(array, 50)), 3),
                    p90_ms=round(float(np.percentile(array, 90)), 3),
                    p95_ms=round(float(np.percentile(array, 95)), 3),
                    p99_ms=round(float(np.percentile(array, 99)), 3),
                    max_ms=round(float(array.max()), 3),
                    mean_ms=round(float(array.mean()), 3),
                    server_cpu_percent_mean=usage["cpu_mean"],
                    server_cpu_percent_max=usage["cpu_max"],
                    server_rss_mb_mean=usage["rss_mean"],
                    server_rss_mb_max=usage["rss_max"],
                    cold_start_seconds=round(server.cold_start_seconds, 3),
                    host_cpu_utilisation=during_cpu,
                    host_load_1m=round(os.getloadavg()[0], 2),
                    host_cpu_count=pressure["cpu_count"],
                    host_busy_before_run=bool(before["busy"]),
                    model_version=model_version,
                )
                rows.append(result)
                print(f"  {label:10s} rps={result.throughput_rps:8.1f} "
                      f"rec/s={result.throughput_records_per_second:9.1f} "
                      f"p50={result.p50_ms:7.2f} p95={result.p95_ms:7.2f} "
                      f"p99={result.p99_ms:8.2f} err={result.error_rate:.3f} "
                      f"rss={result.server_rss_mb_mean}MB cpu={during_cpu:.0%}"
                      + ("  [OTHER WORK ON HOST]" if before["busy"] else ""))
    finally:
        server.stop()

    frame = pd.DataFrame([r.to_dict() for r in rows])
    root = config.path("paths.results_dir").parent
    out_csv = results_dir / f"load_test_{tag}.csv"
    frame.to_csv(out_csv, index=False)
    # The top-level CSV accumulates every worker configuration measured.
    combined = pd.concat(
        [pd.read_csv(path) for path in sorted(results_dir.glob("load_test_*p*w*.csv"))],
        ignore_index=True,
    )
    combined.to_csv(root / "SERVING_BENCHMARKS.csv", index=False)
    (results_dir / f"load_test_{tag}.json").write_text(json.dumps({
        "command": server.command,
        "workers": args.workers,
        "thread_pin": args.thread_pin,
        "server_env_overrides": getattr(server, "env_overrides", {}),
        "seed": seed,
        "payloads": args.payloads,
        "duration_seconds": args.duration,
        "warmup_seconds": args.warmup,
        "environment": environment_info().to_dict(),
        "host_cpu_count": os.cpu_count(),
        "host_load_average": os.getloadavg(),
        "busy_cpu_fraction_limit": BUSY_CPU_FRACTION,
        "host_pressure_at_start": pressure,
        "any_run_on_busy_host": bool(frame["host_busy_before_run"].any()),
        "results": [r.to_dict() for r in rows],
    }, indent=2))

    print(f"\nwrote {root / 'SERVING_BENCHMARKS.csv'}")
    if bool(frame["host_busy_before_run"].any()):
        print("NOTE: at least one run started while other work was using more than "
              f"{BUSY_CPU_FRACTION:.0%} of the machine and is flagged "
              "host_busy_before_run=True. Re-run on an idle machine before quoting "
              "those latency figures.")
    print("LOAD_TEST_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
