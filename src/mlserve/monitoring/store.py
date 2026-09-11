"""SQLite prediction store: the durable half of monitoring.

Prometheus holds aggregates; this holds the individual scored feature vectors. Both
are needed and they answer different questions. A counter tells you *that* traffic
changed; only the retained rows let you re-run a two-sample test and say *which
feature* changed and by how much. Drift detection is a retrospective query over this
table, so the table is the thing that has to exist.

Why SQLite and not PostgreSQL: the platform must run from a clean checkout with no
services to start, which is the property the reproducibility gate checks. SQLite is
ACID, ships with Python, and handles this write rate comfortably. The schema is plain
SQL with no SQLite-only types, so moving to PostgreSQL is a connection-string change
plus a driver, and that trade-off is recorded in SYSTEM_DESIGN.md.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from mlserve.data.schema import CATEGORICAL_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES

#: Feature columns are materialised as real SQL columns rather than a JSON blob so a
#: drift query is a single scan instead of 100k JSON parses.
_FEATURE_COLUMN_SQL = ",\n    ".join(
    f"{name} {'REAL' if name in NUMERIC_FEATURES else 'TEXT'}" for name in FEATURE_NAMES
)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT NOT NULL,
    ts              REAL NOT NULL,
    ts_iso          TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    probability     REAL NOT NULL,
    prediction      INTEGER NOT NULL,
    latency_ms      REAL,
    batch_size      INTEGER,
    {_FEATURE_COLUMN_SQL}
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts);
CREATE INDEX IF NOT EXISTS idx_predictions_version ON predictions(model_version);

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,
    ts_iso        TEXT NOT NULL,
    request_id    TEXT,
    kind          TEXT NOT NULL,
    status_code   INTEGER,
    detail        TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);

CREATE TABLE IF NOT EXISTS drift_reports (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                REAL NOT NULL,
    ts_iso            TEXT NOT NULL,
    scenario          TEXT,
    reference_rows    INTEGER,
    current_rows      INTEGER,
    n_features        INTEGER,
    n_drifted         INTEGER,
    drift_detected    INTEGER,
    elapsed_seconds   REAL,
    payload           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_drift_ts ON drift_reports(ts);
"""

_INSERT_PREDICTION = (
    "INSERT INTO predictions (request_id, ts, ts_iso, model_name, model_version, "
    "probability, prediction, latency_ms, batch_size, "
    + ", ".join(FEATURE_NAMES)
    + ") VALUES ("
    + ", ".join(["?"] * (9 + len(FEATURE_NAMES)))
    + ")"
)


def _now() -> tuple[float, str]:
    ts = time.time()
    return ts, datetime.fromtimestamp(ts, UTC).isoformat()


class PredictionStore:
    """Append-only store for scored records, request events and drift reports.

    One connection guarded by a lock. FastAPI runs the sync prediction path in a
    thread pool, so concurrent writes are real; WAL plus a mutex is the simplest
    correct answer at this write rate, and it avoids the connection-per-request cost
    that would otherwise dominate a 2ms prediction.
    """

    def __init__(self, path: str | Path, *, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            # NORMAL rather than FULL: monitoring rows are observability, not the
            # system of record. Losing the last few on a hard crash is acceptable;
            # adding an fsync to every prediction is not.
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ lifecycle

    @contextmanager
    def _cursor(self):
        if not self.enabled or self._conn is None:
            yield None
            return
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            finally:
                cur.close()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # --------------------------------------------------------------------- writes

    def log_predictions(
        self,
        *,
        request_id: str,
        model_name: str,
        model_version: str,
        features: Sequence[dict],
        probabilities: Sequence[float],
        predictions: Sequence[int],
        latency_ms: float,
    ) -> int:
        """Persist one batch. Returns the number of rows written."""
        if not self.enabled or not features:
            return 0
        ts, ts_iso = _now()
        batch_size = len(features)
        rows = [
            (
                request_id, ts, ts_iso, model_name, model_version,
                float(prob), int(pred), float(latency_ms), batch_size,
                *[record.get(name) for name in FEATURE_NAMES],
            )
            for record, prob, pred in zip(features, probabilities, predictions, strict=True)
        ]
        with self._cursor() as cur:
            if cur is None:
                return 0
            cur.executemany(_INSERT_PREDICTION, rows)
        return len(rows)

    def log_event(self, kind: str, *, request_id: str | None = None,
                  status_code: int | None = None, detail: str | None = None) -> None:
        if not self.enabled:
            return
        ts, ts_iso = _now()
        with self._cursor() as cur:
            if cur is None:
                return
            cur.execute(
                "INSERT INTO events (ts, ts_iso, request_id, kind, status_code, detail) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ts, ts_iso, request_id, kind, status_code, detail),
            )

    def log_drift_report(self, report: dict, *, scenario: str | None = None) -> None:
        if not self.enabled:
            return
        ts, ts_iso = _now()
        with self._cursor() as cur:
            if cur is None:
                return
            cur.execute(
                "INSERT INTO drift_reports (ts, ts_iso, scenario, reference_rows, current_rows, "
                "n_features, n_drifted, drift_detected, elapsed_seconds, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts, ts_iso, scenario or report.get("scenario"),
                    report.get("reference_rows"), report.get("current_rows"),
                    report.get("n_features"), report.get("n_drifted"),
                    int(bool(report.get("drift_detected"))), report.get("elapsed_seconds"),
                    json.dumps(report, default=str),
                ),
            )

    # --------------------------------------------------------------------- reads

    def count(self, table: str = "predictions") -> int:
        if not self.enabled or self._conn is None:
            return 0
        if table not in {"predictions", "events", "drift_reports"}:
            raise ValueError(f"unknown table {table!r}")
        with self._lock:
            return int(self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def recent_features(self, limit: int = 2000, *, model_version: str | None = None) -> pd.DataFrame:
        """The most recent scored feature vectors -- the 'current window' for drift."""
        if not self.enabled or self._conn is None:
            return pd.DataFrame(columns=FEATURE_NAMES)
        cols = ", ".join(FEATURE_NAMES + ["probability", "prediction", "model_version", "ts"])
        sql = f"SELECT {cols} FROM predictions"
        params: list = []
        if model_version is not None:
            sql += " WHERE model_version = ?"
            params.append(model_version)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            frame = pd.read_sql_query(sql, self._conn, params=params)
        return frame.iloc[::-1].reset_index(drop=True)

    def summary(self, window_seconds: float | None = None) -> dict:
        """Operational snapshot used by GET /monitoring/summary."""
        if not self.enabled or self._conn is None:
            return {"enabled": False}
        clause, params = "", []
        if window_seconds is not None:
            clause = " WHERE ts >= ?"
            params = [time.time() - float(window_seconds)]
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), AVG(probability), MIN(probability), MAX(probability), "
                f"AVG(latency_ms) FROM predictions{clause}", params
            ).fetchone()
            versions = self._conn.execute(
                f"SELECT model_version, COUNT(*) FROM predictions{clause} GROUP BY model_version",
                params,
            ).fetchall()
            events = self._conn.execute(
                f"SELECT kind, COUNT(*) FROM events{clause} GROUP BY kind", params
            ).fetchall()
        return {
            "enabled": True,
            "window_seconds": window_seconds,
            "n_predictions": int(row[0] or 0),
            "mean_probability": round(row[1], 6) if row[1] is not None else None,
            "min_probability": round(row[2], 6) if row[2] is not None else None,
            "max_probability": round(row[3], 6) if row[3] is not None else None,
            "mean_latency_ms": round(row[4], 4) if row[4] is not None else None,
            "by_model_version": {str(v): int(c) for v, c in versions},
            "events": {str(k): int(c) for k, c in events},
        }

    def feature_frame(self, limit: int | None = None) -> pd.DataFrame:
        frame = self.recent_features(limit or 10_000_000)
        return frame[FEATURE_NAMES] if not frame.empty else pd.DataFrame(columns=FEATURE_NAMES)


def records_from_frame(frame: pd.DataFrame) -> Iterable[dict]:
    """Frame -> the record dicts :meth:`PredictionStore.log_predictions` expects."""
    return frame[FEATURE_NAMES].to_dict(orient="records")


__all__ = ["CATEGORICAL_FEATURES", "PredictionStore", "records_from_frame"]
