"""Deterministic, leakage-safe dataset splitting.

Three splits with three different jobs:

* **train**      -- fits the preprocessing statistics and the estimator.
* **validation** -- carved out of the same file as train; used for model selection,
                    threshold choice and the retraining acceptance decision. It is
                    seen many times, so it is *not* an unbiased estimate of
                    generalisation and is never quoted as the headline number.
* **test**       -- the official UCI `adult.test` file, a genuinely separate
                    collection. It is loaded once at the end of a run and is never
                    used to choose anything.

Keeping the holdout in a physically separate file is the cheapest structural defence
against contamination: there is no code path in which a test row can reach the fit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlserve.data.ingest import DatasetVersion, label_to_int
from mlserve.data.schema import FEATURE_NAMES, TARGET


@dataclass(frozen=True)
class Split:
    """One materialised train/validation/test partition."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    dataset_version: DatasetVersion
    seed: int
    validation_fraction: float
    #: Positional indices into the source development frame, kept so that
    #: "did the same source record land in two splits?" is answerable directly
    #: rather than inferred from row contents.
    train_source_index: tuple[int, ...] = ()
    validation_source_index: tuple[int, ...] = ()

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    def xy(self, name: str) -> tuple[pd.DataFrame, pd.Series]:
        frame = getattr(self, name)
        return frame[FEATURE_NAMES].copy(), label_to_int(frame[TARGET])

    @property
    def split_id(self) -> str:
        """Hash of the split's actual contents, so a silent repartition is detectable."""
        h = hashlib.sha256()
        h.update(self.dataset_version.dataset_id.encode())
        for name in ("train", "validation", "test"):
            frame = getattr(self, name)
            h.update(f"|{name}:{len(frame)}:".encode())
            h.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
        return h.hexdigest()[:16]

    def to_manifest(self) -> dict:
        return {
            "dataset_version": self.dataset_version.dataset_id,
            "split_id": self.split_id,
            "seed": self.seed,
            "validation_fraction": self.validation_fraction,
            "sizes": self.sizes,
            "positive_rate": {
                name: round(float(label_to_int(getattr(self, name)[TARGET]).mean()), 6)
                for name in ("train", "validation", "test")
            },
        }


def make_split(
    development: pd.DataFrame,
    test: pd.DataFrame,
    version: DatasetVersion,
    *,
    seed: int,
    validation_fraction: float,
    stratify: bool = True,
) -> Split:
    """Split ``development`` into train/validation; keep ``test`` untouched."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")

    strat = development[TARGET] if stratify else None
    train, validation = train_test_split(
        development,
        test_size=validation_fraction,
        random_state=seed,
        stratify=strat,
        shuffle=True,
    )
    return Split(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        dataset_version=version,
        seed=seed,
        validation_fraction=validation_fraction,
        train_source_index=tuple(int(i) for i in train.index),
        validation_source_index=tuple(int(i) for i in validation.index),
    )


def row_fingerprints(frame: pd.DataFrame, columns: list[str] | None = None) -> set[int]:
    """Hash every row so two frames can be compared for shared records."""
    cols = columns if columns is not None else list(frame.columns)
    return set(pd.util.hash_pandas_object(frame[cols], index=False).tolist())


def overlap_report(split: Split) -> dict:
    """Measure how much the splits share, both as records and as feature vectors.

    Two different quantities, deliberately reported separately:

    * ``shared_full_rows``      -- identical feature vector *and* label across splits.
      Between train and validation this must be interpreted carefully: the source file
      genuinely contains repeated respondents, so a nonzero count here is expected and
      is not a bug. It bounds how optimistic a score can be.
    * ``index_overlap``         -- the same source record in two splits. This must be
      zero; anything else is a real splitting defect.
    """
    train_idx = set(split.train_source_index)
    val_idx = set(split.validation_source_index)

    feature_cols = FEATURE_NAMES
    fp = {name: row_fingerprints(getattr(split, name), feature_cols + [TARGET])
          for name in ("train", "validation", "test")}
    feat_fp = {name: row_fingerprints(getattr(split, name), feature_cols)
               for name in ("train", "validation", "test")}

    def rate(a: str, b: str, table: dict[str, set[int]]) -> float:
        other = getattr(split, b)
        if len(other) == 0:
            return 0.0
        return round(len(table[a] & table[b]) / len(other), 6)

    return {
        "index_overlap_train_validation": len(train_idx & val_idx),
        "index_coverage_train_validation": len(train_idx | val_idx),
        "shared_full_rows_train_validation": len(fp["train"] & fp["validation"]),
        "shared_full_rows_train_test": len(fp["train"] & fp["test"]),
        "shared_feature_vectors_train_validation": len(feat_fp["train"] & feat_fp["validation"]),
        "shared_feature_vectors_train_test": len(feat_fp["train"] & feat_fp["test"]),
        "shared_feature_rate_in_validation": rate("train", "validation", feat_fp),
        "shared_feature_rate_in_test": rate("train", "test", feat_fp),
    }


def save_split(split: Split, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name in ("train", "validation", "test"):
        path = out / f"{name}.parquet"
        getattr(split, name).to_parquet(path, index=False)
        written[name] = path
    return written
