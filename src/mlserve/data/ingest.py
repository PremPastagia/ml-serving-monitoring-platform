"""Raw-file ingestion, cleaning and deterministic dataset versioning.

The dataset version is a content hash of the raw bytes *and* of the parsing rules,
so a change to either produces a new id. Every training run, registry entry and
drift baseline records that id, which is what makes results traceable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from mlserve.data.schema import (
    DROPPED_COLUMNS,
    MISSING_CATEGORY,
    MISSING_MARKER,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RAW_COLUMNS,
    TARGET,
)

#: Bump when the parsing/cleaning rules below change in a way that alters the frame.
INGEST_LOGIC_VERSION = "ingest-2"

#: Pinned upstream sources. Recorded so a reviewer can re-fetch the identical bytes.
SOURCES: dict[str, dict[str, str]] = {
    "adult.data": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "sha256": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
        "role": "development (train + validation)",
    },
    "adult.test": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        "sha256": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05",
        "role": "held-out test (official UCI split)",
    },
}


@dataclass(frozen=True)
class DatasetVersion:
    """Immutable identity of one materialised dataset."""

    dataset_id: str
    logic_version: str
    files: dict[str, str]
    n_development: int
    n_test: int

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ChecksumMismatch(RuntimeError):
    """Raised when a raw file's bytes differ from the pinned checksum."""


def verify_raw_files(raw_dir: str | Path, *, strict: bool = True) -> dict[str, str]:
    """Return {filename: sha256}, raising if a file is missing or altered."""
    raw_dir = Path(raw_dir)
    observed: dict[str, str] = {}
    for name, meta in SOURCES.items():
        path = raw_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run `python scripts/fetch_data.py` to download it."
            )
        digest = sha256_file(path)
        if strict and digest != meta["sha256"]:
            raise ChecksumMismatch(
                f"{name} sha256 {digest} != pinned {meta['sha256']}. "
                "The upstream file changed or the download is truncated."
            )
        observed[name] = digest
    return observed


def _read_raw(path: Path, *, skiprows: int) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        skipinitialspace=True,
        skiprows=skiprows,
        na_values=[],          # keep '?' as a literal token; we handle it explicitly
        keep_default_na=False,
        dtype=str,             # parse everything as text first, then coerce deliberately
        engine="c",
    )
    # The raw files end with a blank line, which pandas turns into an all-empty row.
    frame = frame[frame["age"].str.strip() != ""].reset_index(drop=True)
    return frame


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented cleaning rules. Deterministic and order-preserving."""
    out = frame.copy()
    for col in out.columns:
        out[col] = out[col].str.strip()

    # adult.test writes the label as '<=50K.' / '>50K.' with a trailing period.
    out[TARGET] = out[TARGET].str.rstrip(".")

    # '?' is the survey's "no answer" marker on three categoricals. Encoding it as an
    # explicit category (rather than dropping the row or imputing a mode) keeps the
    # non-response signal, which is itself predictive, and keeps train and serving
    # behaviour identical: the API accepts '?' and maps it the same way.
    for col in ("workclass", "occupation", "native_country"):
        out[col] = out[col].replace(MISSING_MARKER, MISSING_CATEGORY)

    numeric = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.drop(columns=DROPPED_COLUMNS)

    # NOTE ON DUPLICATES. Exact duplicate rows are *kept*. In a population survey two
    # different respondents can legitimately share every retained attribute, so a
    # repeated feature vector is not a data-quality defect and is not leakage --
    # leakage would be the same *record* appearing in two splits, which the split
    # step forbids by index. Silently deduplicating here would (a) discard real
    # frequency information a probability model should see and (b) change the
    # published row counts, making the dataset version incomparable with the UCI
    # benchmark. The validator instead measures and reports the duplicate rate, and
    # `scripts/verify/verify_no_leakage.py` measures how many held-out feature
    # vectors also occur in train, which is what actually bounds optimism.
    return out.reset_index(drop=True)


def load_raw(raw_dir: str | Path = "data/raw", *, strict: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, DatasetVersion]:
    """Load the development and held-out test frames plus their joint version id."""
    raw_dir = Path(raw_dir)
    checksums = verify_raw_files(raw_dir, strict=strict)

    development = _clean(_read_raw(raw_dir / "adult.data", skiprows=0))
    # adult.test's first line is the junk banner '|1x3 Cross validator'.
    test = _clean(_read_raw(raw_dir / "adult.test", skiprows=1))

    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "files": checksums,
                "logic": INGEST_LOGIC_VERSION,
                "dropped": DROPPED_COLUMNS,
                "missing_category": MISSING_CATEGORY,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    version = DatasetVersion(
        dataset_id=f"adult-{INGEST_LOGIC_VERSION}-{fingerprint[:12]}",
        logic_version=INGEST_LOGIC_VERSION,
        files=checksums,
        n_development=len(development),
        n_test=len(test),
    )
    return development, test, version


def label_to_int(series: pd.Series) -> pd.Series:
    """Map the string target onto {0, 1} with '>50K' as the positive class."""
    mapping = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}
    out = series.map(mapping)
    if out.isna().any():
        bad = sorted(set(series[out.isna()].unique()))
        raise ValueError(f"target contains values outside {sorted(mapping)}: {bad}")
    return out.astype("int64")
