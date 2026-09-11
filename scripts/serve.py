#!/usr/bin/env python
"""Start the serving API using the host/port from configs/config.yaml.

Thread pinning
--------------
The numeric thread-pool caps below are set *before* numpy, scikit-learn or the app are
imported, because OpenMP and the BLAS libraries read them at load time and ignore
later changes.

This matters more than it looks. scikit-learn's HistGradientBoostingClassifier
predicts through OpenMP, which by default fans work across every core -- including for
a single-row request, where the fan-out costs far more than the work. Measured on a
10-core machine with `scripts/load_test.py`, uncapped versus capped at one thread:

    concurrency 1:   76-84 rps, p50 ~11ms, p99 ~26-30ms, host CPU ~99%
                 ->  263 rps,   p50 3.7ms, p99 9.2ms,    host CPU ~14%

    concurrency 8:   39-44 rps, p50 ~180ms, p99 200-400ms, host CPU ~100%
                 ->  250 rps,   p50 31ms,   p99 54ms,      host CPU ~16%

Uncapped, request concurrency made throughput *worse*, because eight concurrent
requests each tried to use ten cores. Capped, throughput holds flat under concurrency.
Scale with worker processes, not with intra-op threads.

`--threads N` raises the cap for a batch-scoring deployment, where large matrix work
genuinely does parallelise.
"""

from __future__ import annotations

import argparse
import os
import sys

# Must run before numpy/scikit-learn are imported anywhere in this process.
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def pin_threads(n: int) -> None:
    for name in _THREAD_ENV:
        os.environ[name] = str(n)


def _preparse_threads(argv: list[str] | None) -> int:
    """Read --threads before anything heavy is imported."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--threads", type=int, default=1)
    known, _ = pre.parse_known_args(argv if argv is not None else sys.argv[1:])
    return known.threads


def main(argv: list[str] | None = None) -> int:
    pin_threads(_preparse_threads(argv))

    import uvicorn  # noqa: PLC0415 - imported after the thread caps are set

    from mlserve.config import load_config  # noqa: PLC0415

    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=str(config.require("serving.host")))
    parser.add_argument("--port", type=int, default=int(config.require("serving.port")))
    parser.add_argument("--workers", type=int, default=1,
                        help="uvicorn worker processes; scale here, not with --threads")
    parser.add_argument("--threads", type=int, default=1,
                        help="intra-op thread cap for OpenMP/BLAS (default 1; see the "
                             "module docstring for the measurement behind that default)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    print(f"thread caps: {args.threads} | workers: {args.workers} | "
          f"listening on http://{args.host}:{args.port}")
    uvicorn.run(
        "mlserve.serving.main:app",
        host=args.host, port=args.port, workers=args.workers,
        log_level=args.log_level, access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
