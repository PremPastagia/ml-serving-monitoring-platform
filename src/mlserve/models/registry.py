"""Model registry operations: register, version, promote, roll back.

Aliases, not stages
-------------------
MLflow's ``Staging``/``Production`` *stages* are deprecated. This project uses
**registry aliases** instead, which is both the supported API and the better model
for what serving actually needs: an alias is a mutable pointer to an immutable
version, so promotion and rollback are a single atomic repoint rather than a
multi-step stage transition that can be observed half-applied.

Three aliases are used:

``production`` -- the version the API serves.
``previous``   -- the version ``production`` pointed at before the last promotion.
                  This is what makes rollback a one-step operation with a known
                  target, instead of "find the last good version and hope".
``candidate``  -- a freshly retrained model awaiting the acceptance decision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException, RestException
from mlflow.tracking import MlflowClient

from mlserve.config import Config, load_config
from mlserve.models.tracking import configure_mlflow

PRODUCTION = "production"
PREVIOUS = "previous"
CANDIDATE = "candidate"


class RegistryError(RuntimeError):
    """Raised for registry operations that cannot be satisfied."""


@dataclass(frozen=True)
class ModelRef:
    """A resolved pointer to one registered model version."""

    name: str
    version: str
    run_id: str
    alias: str | None
    source: str
    tags: dict[str, str]

    @property
    def uri(self) -> str:
        return f"models:/{self.name}/{self.version}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "run_id": self.run_id,
            "alias": self.alias,
            "uri": self.uri,
            "tags": self.tags,
        }


class ModelRegistry:
    """Thin, testable wrapper over the MLflow registry client."""

    def __init__(self, config: Config | None = None):
        self.config = config or load_config()
        self.client: MlflowClient = configure_mlflow(self.config)
        self.model_name = str(self.config.require("mlflow.registered_model_name"))

    # ---------------------------------------------------------------- registration

    def ensure_registered_model(self) -> None:
        try:
            self.client.get_registered_model(self.model_name)
        except (MlflowException, RestException):
            self.client.create_registered_model(self.model_name)

    def register(self, model_uri: str, *, tags: dict[str, str] | None = None) -> ModelRef:
        """Register the logged model at ``model_uri`` as a new registry version.

        ``model_uri`` comes from :class:`~mlserve.models.tracking.LoggedRun`; pass the
        logged-model URI rather than ``runs:/<id>/model`` so the version's source
        points at the artifact directly instead of through the deprecated run path.
        """
        self.ensure_registered_model()
        version = mlflow.register_model(model_uri=model_uri, name=self.model_name, tags=tags or {})
        self._await_ready(version.version)
        return self.get_version(version.version)

    def _await_ready(self, version: str, timeout: float = 30.0) -> None:
        """Block until the version leaves PENDING_REGISTRATION.

        With the SQLite backend this is effectively instant, but the state machine is
        real and a caller that promotes a PENDING version gets an unusable pointer.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            info = self.client.get_model_version(self.model_name, version)
            if info.status == "READY":
                return
            if info.status == "FAILED_REGISTRATION":
                raise RegistryError(f"version {version} failed registration")
            time.sleep(0.1)
        raise RegistryError(f"version {version} was not READY within {timeout}s")

    # -------------------------------------------------------------------- queries

    def get_version(self, version: str | int) -> ModelRef:
        info = self.client.get_model_version(self.model_name, str(version))
        return ModelRef(
            name=self.model_name,
            version=str(info.version),
            run_id=info.run_id,
            alias=(info.aliases[0] if getattr(info, "aliases", None) else None),
            source=info.source,
            tags=dict(info.tags or {}),
        )

    def list_versions(self) -> list[ModelRef]:
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
        except (MlflowException, RestException):
            return []
        refs = [
            ModelRef(
                name=self.model_name,
                version=str(v.version),
                run_id=v.run_id,
                alias=(v.aliases[0] if getattr(v, "aliases", None) else None),
                source=v.source,
                tags=dict(v.tags or {}),
            )
            for v in versions
        ]
        return sorted(refs, key=lambda r: int(r.version))

    def resolve_alias(self, alias: str) -> ModelRef | None:
        try:
            info = self.client.get_model_version_by_alias(self.model_name, alias)
        except (MlflowException, RestException):
            return None
        return ModelRef(
            name=self.model_name,
            version=str(info.version),
            run_id=info.run_id,
            alias=alias,
            source=info.source,
            tags=dict(info.tags or {}),
        )

    def production(self) -> ModelRef | None:
        return self.resolve_alias(PRODUCTION)

    # ------------------------------------------------------- promotion / rollback

    def set_alias(self, alias: str, version: str | int) -> ModelRef:
        self.client.set_registered_model_alias(self.model_name, alias, str(version))
        return self.get_version(version)

    def promote(self, version: str | int, *, alias: str = PRODUCTION, keep_previous: bool = True) -> dict:
        """Point ``alias`` at ``version``, remembering what it pointed at before.

        Returns the before/after pointers and the wall-clock cost of the repoint, so
        promotion latency is measured rather than assumed.
        """
        version = str(version)
        current = self.resolve_alias(alias)
        if current is not None and current.version == version:
            return {"changed": False, "alias": alias, "from": current.version,
                    "to": version, "seconds": 0.0}

        start = time.perf_counter()
        if keep_previous and current is not None:
            self.client.set_registered_model_alias(self.model_name, PREVIOUS, current.version)
        self.client.set_registered_model_alias(self.model_name, alias, version)
        elapsed = time.perf_counter() - start

        return {
            "changed": True,
            "alias": alias,
            "from": current.version if current else None,
            "to": version,
            "seconds": round(elapsed, 6),
        }

    def rollback(self, *, alias: str = PRODUCTION, to_version: str | int | None = None) -> dict:
        """Repoint ``alias`` back to the previous version (or an explicit one).

        Rollback is deliberately symmetric with promotion: it also updates
        ``previous`` to the version being rolled *away from*, so a rollback can itself
        be rolled back. Without that, a mistaken rollback is a dead end.
        """
        current = self.resolve_alias(alias)
        if current is None:
            raise RegistryError(f"alias {alias!r} is not set; nothing to roll back")

        if to_version is None:
            target = self.resolve_alias(PREVIOUS)
            if target is None:
                raise RegistryError(
                    f"no {PREVIOUS!r} alias recorded; supply to_version explicitly"
                )
            target_version = target.version
        else:
            target_version = str(to_version)
            self.get_version(target_version)  # raises if it does not exist

        if target_version == current.version:
            raise RegistryError(
                f"rollback target {target_version} is already the {alias!r} version"
            )

        start = time.perf_counter()
        self.client.set_registered_model_alias(self.model_name, PREVIOUS, current.version)
        self.client.set_registered_model_alias(self.model_name, alias, target_version)
        elapsed = time.perf_counter() - start

        return {
            "alias": alias,
            "from": current.version,
            "to": target_version,
            "seconds": round(elapsed, 6),
        }

    def delete_alias(self, alias: str) -> None:
        try:
            self.client.delete_registered_model_alias(self.model_name, alias)
        except (MlflowException, RestException):
            pass

    # ------------------------------------------------------------------- loading

    def load(self, ref_or_alias: ModelRef | str) -> Any:
        """Load a pyfunc-compatible sklearn pipeline from the registry."""
        if isinstance(ref_or_alias, ModelRef):
            uri = ref_or_alias.uri
        else:
            uri = f"models:/{self.model_name}@{ref_or_alias}"
        return mlflow.sklearn.load_model(uri)

    def set_version_tags(self, version: str | int, tags: dict[str, str]) -> None:
        for key, value in tags.items():
            self.client.set_model_version_tag(self.model_name, str(version), key, str(value))
