"""The data contract is the single source of truth; these tests keep it that way."""

from __future__ import annotations

import pytest

from mlserve.data.schema import (
    BY_NAME,
    CATEGORICAL_FEATURES,
    CONTRACT,
    DROPPED_COLUMNS,
    FEATURE_NAMES,
    NUMERIC_FEATURES,
    RAW_COLUMNS,
    TARGET,
    TARGET_CLASSES,
)
from mlserve.features.pipeline import MODEL_FEATURES
from mlserve.serving.schemas import contract_field_names


def test_feature_partition_is_exact():
    assert set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) == set(FEATURE_NAMES)
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_features_are_raw_columns_minus_dropped_and_target():
    expected = [c for c in RAW_COLUMNS if c not in DROPPED_COLUMNS and c != TARGET]
    assert FEATURE_NAMES == expected


def test_numeric_specs_declare_usable_bounds():
    for name in NUMERIC_FEATURES:
        spec = BY_NAME[name]
        assert spec.minimum is not None and spec.maximum is not None, name
        assert spec.minimum < spec.maximum, name


def test_categorical_specs_declare_levels():
    for name in CATEGORICAL_FEATURES:
        spec = BY_NAME[name]
        assert len(spec.allowed) >= 2, name
        assert len(set(spec.allowed)) == len(spec.allowed), f"{name} has duplicate levels"


def test_api_request_model_matches_the_contract_exactly():
    """The generated Pydantic model must not drift from the contract."""
    assert contract_field_names() == FEATURE_NAMES


def test_model_features_extend_but_preserve_contract_features():
    """Engineering may add columns; it may never silently drop a contract feature."""
    assert set(FEATURE_NAMES).issubset(set(MODEL_FEATURES))
    assert len(MODEL_FEATURES) > len(FEATURE_NAMES)


def test_target_classes_are_binary_and_distinct():
    assert len(TARGET_CLASSES) == 2
    assert TARGET_CLASSES[0] != TARGET_CLASSES[1]


def test_contract_serialises_completely():
    payload = CONTRACT.to_dict()
    assert payload["target"] == TARGET
    assert len(payload["features"]) == len(FEATURE_NAMES)
    assert payload["dropped_columns"] == DROPPED_COLUMNS
    for feature in payload["features"]:
        assert feature["description"].strip(), f"{feature['name']} has no description"


@pytest.mark.parametrize("dropped", DROPPED_COLUMNS)
def test_dropped_columns_are_not_servable(dropped):
    """A dropped column must not reappear as an API field."""
    assert dropped not in contract_field_names()
