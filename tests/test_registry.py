"""Model registry: registration, versioning, promotion and rollback.

Every test runs against an isolated MLflow store under tmp_path, so the suite never
mutates the developer's real registry and tests cannot influence one another.
"""

from __future__ import annotations

import pytest
from mlflow.exceptions import MlflowException

from mlserve.models.registry import CANDIDATE, PREVIOUS, PRODUCTION, ModelRegistry, RegistryError
from mlserve.models.tracking import best_run, list_runs, log_training_run
from mlserve.models.train import train_model


@pytest.fixture(scope="module")
def fast_params():
    return {"learning_rate": 0.2, "max_iter": 15, "max_leaf_nodes": 15,
            "min_samples_leaf": 20, "l2_regularization": 1.0, "max_bins": 255,
            "early_stopping": False}


@pytest.fixture(scope="module")
def module_config(tmp_path_factory):
    """One isolated MLflow store for this module.

    Creating a fresh SQLite tracking store costs a full alembic migration (several
    seconds), so paying it per test made this file dominate the suite runtime. The
    store is therefore shared across the module and only the *mutable* state -- the
    aliases -- is reset between tests by `reset_aliases` below. Registered versions are
    append-only and identical for every test here, so sharing them is safe.
    """
    from tests.conftest import clone_config

    return clone_config(tmp_path_factory.mktemp("registry"))


@pytest.fixture(scope="module")
def _versions(module_config, small_split, fast_params):
    registry = ModelRegistry(module_config)
    X_val, _ = small_split.xy("validation")
    refs = []
    specs = [
        ("hist_gradient_boosting", fast_params, "strong"),
        ("logistic_regression", {"C": 1.0, "max_iter": 200}, "weak"),
    ]
    for estimator, params, label in specs:
        pipeline, run = train_model(
            module_config, split=small_split, estimator=estimator, params=params,
            run_name=f"{label}-model", evaluate_test=False,
        )
        logged = log_training_run(pipeline, run, module_config, input_example=X_val,
                                  tags={"label": label})
        refs.append(registry.register(logged.model_uri, tags={
            "label": label,
            "validation_roc_auc": f"{run.metrics['validation_roc_auc']:.6f}",
        }))
    return registry, refs


@pytest.fixture
def registry_with_versions(_versions):
    """The shared registry with every alias cleared, so each test starts from zero."""
    registry, refs = _versions
    for alias in (PRODUCTION, PREVIOUS, CANDIDATE):
        registry.delete_alias(alias)
    return registry, refs


# --------------------------------------------------------------------- experiments


def test_multiple_runs_are_tracked(module_config, registry_with_versions):
    runs = list_runs(module_config)
    assert len(runs) >= 2
    assert "metrics.validation_roc_auc" in runs.columns


def test_best_run_selects_the_highest_primary_metric(module_config, registry_with_versions):
    runs = list_runs(module_config)
    top = best_run(module_config)
    assert top is not None
    assert top["metrics.validation_roc_auc"] == runs["metrics.validation_roc_auc"].max()


def test_a_failed_training_run_leaves_no_registry_version(module_config, registry_with_versions, small_split):
    """A run that raises must not produce a promotable artefact."""
    registry, _ = registry_with_versions
    before = len(registry.list_versions())
    with pytest.raises(ValueError):
        train_model(module_config, split=small_split, estimator="not_an_estimator",
                    params={}, evaluate_test=False)
    assert len(registry.list_versions()) == before


# ------------------------------------------------------------------- registration


def test_registration_creates_increasing_versions(registry_with_versions):
    registry, refs = registry_with_versions
    assert [r.version for r in refs] == ["1", "2"]
    assert [v.version for v in registry.list_versions()] == ["1", "2"]


def test_registered_version_carries_provenance_tags(registry_with_versions):
    _, refs = registry_with_versions
    assert refs[0].tags["label"] == "strong"
    assert float(refs[0].tags["validation_roc_auc"]) > 0.5


def test_registered_model_loads_and_predicts(registry_with_versions, small_split):
    registry, refs = registry_with_versions
    model = registry.load(refs[0])
    X, _ = small_split.xy("validation")
    proba = model.predict_proba(X.head(20))[:, 1]
    assert proba.shape == (20,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_getting_an_unknown_version_raises(registry_with_versions):
    registry, _ = registry_with_versions
    with pytest.raises(MlflowException):
        registry.get_version("9999")


# --------------------------------------------------------------------- promotion


def test_first_promotion_sets_the_alias(registry_with_versions):
    registry, refs = registry_with_versions
    result = registry.promote(refs[0].version)
    assert result["changed"] is True
    assert result["from"] is None
    assert registry.production().version == refs[0].version


def test_promotion_records_the_previous_version(registry_with_versions):
    registry, refs = registry_with_versions
    registry.promote(refs[0].version)
    registry.promote(refs[1].version)
    assert registry.production().version == refs[1].version
    assert registry.resolve_alias(PREVIOUS).version == refs[0].version


def test_promoting_the_current_version_is_a_no_op(registry_with_versions):
    registry, refs = registry_with_versions
    registry.promote(refs[0].version)
    result = registry.promote(refs[0].version)
    assert result["changed"] is False
    assert registry.production().version == refs[0].version


def test_promotion_reports_its_duration(registry_with_versions):
    registry, refs = registry_with_versions
    result = registry.promote(refs[0].version)
    assert result["seconds"] >= 0.0


def test_candidate_alias_does_not_affect_production(registry_with_versions):
    registry, refs = registry_with_versions
    registry.promote(refs[0].version)
    registry.set_alias(CANDIDATE, refs[1].version)
    assert registry.production().version == refs[0].version
    assert registry.resolve_alias(CANDIDATE).version == refs[1].version


# ---------------------------------------------------------------------- rollback


def test_rollback_returns_to_the_previous_version(registry_with_versions):
    registry, refs = registry_with_versions
    registry.promote(refs[0].version)
    registry.promote(refs[1].version)
    result = registry.rollback()
    assert result["from"] == refs[1].version
    assert result["to"] == refs[0].version
    assert registry.production().version == refs[0].version


def test_rollback_is_itself_reversible(registry_with_versions):
    """After rolling back, `previous` points at what we rolled away from."""
    registry, refs = registry_with_versions
    registry.promote(refs[0].version)
    registry.promote(refs[1].version)
    registry.rollback()
    assert registry.resolve_alias(PREVIOUS).version == refs[1].version
    registry.rollback()
    assert registry.production().version == refs[1].version


def test_rollback_to_an_explicit_version(registry_with_versions):
    registry, refs = registry_with_versions
    registry.promote(refs[1].version)
    result = registry.rollback(to_version=refs[0].version)
    assert result["to"] == refs[0].version
    assert registry.production().version == refs[0].version


def test_rollback_without_a_production_alias_raises(registry_with_versions):
    registry, _ = registry_with_versions
    with pytest.raises(RegistryError, match="not set"):
        registry.rollback()


def test_rollback_without_a_previous_alias_raises(registry_with_versions):
    registry, refs = registry_with_versions
    registry.promote(refs[0].version)
    with pytest.raises(RegistryError, match="no 'previous'"):
        registry.rollback()


def test_rollback_to_the_current_version_is_refused(registry_with_versions):
    registry, refs = registry_with_versions
    registry.promote(refs[0].version)
    with pytest.raises(RegistryError, match="already the"):
        registry.rollback(to_version=refs[0].version)


def test_rollback_to_an_unknown_version_raises(registry_with_versions):
    registry, refs = registry_with_versions
    registry.promote(refs[0].version)
    with pytest.raises(MlflowException):
        registry.rollback(to_version="4242")


def test_rollback_reports_its_duration(registry_with_versions):
    registry, refs = registry_with_versions
    registry.promote(refs[0].version)
    registry.promote(refs[1].version)
    assert registry.rollback()["seconds"] >= 0.0


# ---------------------------------------------------------------- alias resolution


def test_resolving_an_unset_alias_returns_none(registry_with_versions):
    registry, _ = registry_with_versions
    assert registry.resolve_alias("nonexistent-alias") is None
    assert registry.production() is None


def test_loading_by_alias_matches_loading_by_version(registry_with_versions, small_split):
    registry, refs = registry_with_versions
    registry.promote(refs[0].version)
    X, _ = small_split.xy("validation")
    by_alias = registry.load(PRODUCTION).predict_proba(X.head(10))[:, 1]
    by_version = registry.load(refs[0]).predict_proba(X.head(10))[:, 1]
    assert (by_alias == by_version).all()


def test_list_versions_on_an_empty_registry(isolated_config):
    assert ModelRegistry(isolated_config).list_versions() == []
