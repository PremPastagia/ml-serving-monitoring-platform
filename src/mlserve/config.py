"""Typed access to configs/config.yaml plus the code-version stamp.

Configuration is loaded once and hashed. The hash goes into every MLflow run, so
"which config produced this number" is answerable without guessing.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(os.environ.get("MLSERVE_CONFIG", "configs/config.yaml"))


def project_root() -> Path:
    """Repository root, resolved from this file so scripts work from any cwd."""
    return Path(__file__).resolve().parents[2]


class Config:
    """Thin dotted-access wrapper over the parsed YAML."""

    def __init__(self, data: dict[str, Any], source: Path):
        self._data = data
        self.source = source

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"{dotted} is not defined in {self.source}")
        return value

    def path(self, dotted: str) -> Path:
        """Resolve a configured path against the repository root."""
        value = self.require(dotted)
        p = Path(value)
        return p if p.is_absolute() else project_root() / p

    @property
    def raw(self) -> dict[str, Any]:
        return self._data

    @property
    def digest(self) -> str:
        """Stable hash of the whole configuration."""
        return hashlib.sha256(json.dumps(self._data, sort_keys=True).encode()).hexdigest()[:12]

    @property
    def seed(self) -> int:
        return int(self.require("project.random_seed"))


@lru_cache(maxsize=8)
def _load(path_str: str) -> Config:
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root() / path
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return Config(data, path)


def load_config(path: str | Path | None = None) -> Config:
    return _load(str(path or DEFAULT_CONFIG_PATH))


def git_commit() -> str:
    """Short git SHA, or 'unavailable' outside a repository. Never raises."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root(), capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root(), capture_output=True, text=True, timeout=5, check=False,
            )
            suffix = "-dirty" if dirty.stdout.strip() else ""
            return out.stdout.strip() + suffix
    except Exception:
        pass
    return "unavailable"


def code_version() -> str:
    """Content hash of the source tree.

    Git is the right answer when it exists, but this project must stay traceable in a
    plain directory too, so the source tree itself is hashed as a fallback identity.
    """
    src = project_root() / "src"
    h = hashlib.sha256()
    for file in sorted(src.rglob("*.py")):
        h.update(file.relative_to(src).as_posix().encode())
        h.update(file.read_bytes())
    return h.hexdigest()[:12]


@dataclass(frozen=True)
class Environment:
    python_version: str
    platform: str
    processor: str
    packages: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "processor": self.processor,
            **{f"pkg.{k}": v for k, v in self.packages.items()},
        }


TRACKED_PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "mlflow", "fastapi", "pydantic"]


def environment_info() -> Environment:
    import importlib.metadata as md

    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = md.version(name)
        except Exception:
            packages[name] = "absent"
    return Environment(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        processor=platform.machine(),
        packages=packages,
    )
