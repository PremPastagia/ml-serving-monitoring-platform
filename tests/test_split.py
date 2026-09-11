"""Splitting: determinism, stratification and the anti-contamination guarantees."""

from __future__ import annotations

import pandas as pd
import pytest

from mlserve.data.ingest import label_to_int
from mlserve.data.schema import FEATURE_NAMES, TARGET
from mlserve.data.split import make_split, overlap_report, row_fingerprints, save_split


def test_split_sizes_sum_to_the_source(raw_data, config):
    development, test, version = raw_data
    split = make_split(development, test, version, seed=config.seed, validation_fraction=0.2)
    assert split.sizes["train"] + split.sizes["validation"] == len(development)
    assert split.sizes["test"] == len(test)


def test_no_source_record_appears_in_two_splits(full_split):
    """The guarantee that actually matters: one record, one split."""
    report = overlap_report(full_split)
    assert report["index_overlap_train_validation"] == 0
    assert report["index_coverage_train_validation"] == full_split.sizes["train"] + full_split.sizes["validation"]


def test_test_split_comes_from_a_physically_separate_file(full_split, raw_data):
    """No code path can move a test row into the fit, because it is a different file."""
    _, test, _ = raw_data
    assert len(full_split.test) == len(test)
    assert full_split.test.equals(test.reset_index(drop=True))


def test_split_is_deterministic_for_a_fixed_seed(raw_data, config):
    development, test, version = raw_data
    a = make_split(development, test, version, seed=config.seed, validation_fraction=0.2)
    b = make_split(development, test, version, seed=config.seed, validation_fraction=0.2)
    assert a.split_id == b.split_id
    pd.testing.assert_frame_equal(a.train, b.train)
    pd.testing.assert_frame_equal(a.validation, b.validation)


def test_a_different_seed_produces_a_different_split(raw_data, config):
    development, test, version = raw_data
    a = make_split(development, test, version, seed=config.seed, validation_fraction=0.2)
    b = make_split(development, test, version, seed=config.seed + 1, validation_fraction=0.2)
    assert a.split_id != b.split_id


def test_stratification_preserves_class_balance(full_split):
    rates = {
        name: float(label_to_int(getattr(full_split, name)[TARGET]).mean())
        for name in ("train", "validation")
    }
    assert abs(rates["train"] - rates["validation"]) < 0.01


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_validation_fraction_is_rejected(raw_data, config, fraction):
    development, test, version = raw_data
    with pytest.raises(ValueError):
        make_split(development, test, version, seed=config.seed, validation_fraction=fraction)


def test_split_id_changes_when_contents_change(full_split):
    mutated = make_split(
        full_split.train.head(100), full_split.test, full_split.dataset_version,
        seed=full_split.seed, validation_fraction=0.2,
    )
    assert mutated.split_id != full_split.split_id


def test_manifest_records_everything_needed_to_reproduce(full_split):
    manifest = full_split.to_manifest()
    for key in ("dataset_version", "split_id", "seed", "validation_fraction", "sizes", "positive_rate"):
        assert key in manifest
    assert manifest["seed"] == full_split.seed


def test_xy_returns_contract_features_and_binary_labels(full_split):
    X, y = full_split.xy("train")
    assert list(X.columns) == FEATURE_NAMES
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y)


def test_overlap_report_quantifies_shared_feature_vectors(full_split):
    """Repeated respondent profiles are expected; the report must state how many."""
    report = overlap_report(full_split)
    assert report["shared_feature_vectors_train_test"] >= 0
    assert 0.0 <= report["shared_feature_rate_in_test"] < 0.5


def test_row_fingerprints_distinguish_different_rows(full_split):
    head = full_split.train.head(50)
    assert len(row_fingerprints(head)) == len(head.drop_duplicates())


def test_save_split_writes_all_three_parts(full_split, tmp_path):
    written = save_split(full_split, tmp_path)
    assert set(written) == {"train", "validation", "test"}
    for name, path in written.items():
        assert path.exists()
        assert len(pd.read_parquet(path)) == full_split.sizes[name]
