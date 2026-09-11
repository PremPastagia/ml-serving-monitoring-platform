#!/usr/bin/env python
"""Reproduce the recorded training metrics in a brand-new virtual environment.

Builds a fresh venv from the pinned requirements, installs the package into it, trains,
and compares the resulting training fingerprint and metrics against the ones recorded
in results/training/last_training_summary.json.

This is what makes reproducibility a tested claim rather than an assertion. It verifies
reproducibility **on this machine across clean environments** -- not across operating
systems or CPU architectures, which is a different and much stronger claim that this
project does not make.

    python scripts/verify/verify_clean_env_repro.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = 1e-9


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep the temporary venv")
    args = parser.parse_args()

    baseline_path = ROOT / "results" / "training" / "last_training_summary.json"
    if not baseline_path.exists():
        print("no baseline; run `python scripts/train.py` first", file=sys.stderr)
        return 1
    baseline = json.loads(baseline_path.read_text())
    reference = next(r for r in baseline["runs"] if r["run_name"] == baseline["selected_run"])

    workdir = Path(tempfile.mkdtemp(prefix="mlserve-cleanenv-"))
    venv = workdir / "venv"
    python = venv / "bin" / "python"
    print(f"clean environment: {venv}")

    try:
        t0 = time.perf_counter()
        result = run([sys.executable, "-m", "venv", str(venv)])
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return 1

        print("installing pinned requirements (this is the slow part)...")
        for step in (
            [str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
            [str(python), "-m", "pip", "install", "--quiet", "-r", "requirements-dev.txt"],
            [str(python), "-m", "pip", "install", "--quiet", "-e", ".", "--no-deps"],
        ):
            result = run(step)
            if result.returncode != 0:
                print(f"install failed: {' '.join(step)}\n{result.stderr[-3000:]}", file=sys.stderr)
                return 1
        install_seconds = time.perf_counter() - t0
        print(f"installed in {install_seconds:.1f}s")

        # Train into a scratch output directory, without touching the real registry.
        out = workdir / "artifacts"
        t1 = time.perf_counter()
        # --results-dir keeps this run from overwriting the very baseline it is being
        # compared against; without it the check would silently compare a run to itself
        # on the next invocation.
        result = run([str(python), "scripts/train.py", "--no-mlflow",
                      "--out", str(out), "--results-dir", str(workdir / "results")])
        train_seconds = time.perf_counter() - t1
        if result.returncode != 0:
            print(f"training failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}",
                  file=sys.stderr)
            return 1

        produced = json.loads((out / "run.json").read_text())
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)

    problems: list[str] = []
    if produced["training_fingerprint"] != reference["training_fingerprint"]:
        problems.append(
            f"fingerprint differs:\n  recorded {reference['training_fingerprint']}\n"
            f"  clean env {produced['training_fingerprint']}"
        )
    if produced["dataset_version"] != reference["dataset_version"]:
        problems.append("dataset version differs")
    if produced["split_id"] != reference["split_id"]:
        problems.append("split id differs")

    compared = 0
    for key, expected in reference["metrics"].items():
        actual = produced["metrics"].get(key)
        if actual is None:
            problems.append(f"metric {key} missing from the clean-environment run")
            continue
        compared += 1
        if abs(actual - expected) > TOLERANCE:
            problems.append(f"metric {key}: recorded {expected}, clean env {actual}")

    print(f"\ndataset version : {produced['dataset_version']}")
    print(f"split id        : {produced['split_id']}")
    print(f"fingerprint     : {produced['training_fingerprint'][:32]}")
    print(f"metrics compared: {compared}")
    print(f"train time      : {train_seconds:.1f}s (clean env) vs "
          f"{reference['total_seconds']:.1f}s (recorded)")

    if problems:
        print(f"\n{len(problems)} mismatch(es):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nVERIFY_CLEAN_ENV_REPRO_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
