"""Loading, validation, and persistence.

DETERMINISTIC MODULE — no model calls happen here.

`load_dataset` reads the file once and keeps every column in one frame.
`LoadedData.ground_truth` is a derived view over the `expected_*` columns for
the evaluator; it is a convenience for scoring, not a security boundary.

What actually keeps an answer key out of a prompt is `ReturnRecord`: the only
type the model layer accepts, with eight named fields and nowhere for an
`expected_*` value to sit. `_to_record` names each one explicitly, so a column
added to the input file cannot reach a model by accident.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import config
from .config import (
    GROUND_TRUTH_COLUMNS,
    OTHER_DROPDOWN_VALUE,
    REQUIRED_COLUMNS,
)


# --- The record the model layer is allowed to see -------------------------


@dataclass(frozen=True)
class ReturnRecord:
    """One return, with only the fields a classifier may see.

    There is deliberately no field here that could hold an `expected_*` value.
    This is the enforcement mechanism for premise P6 — isolation by type, not
    by convention.
    """

    return_id: str
    sku: str
    product_name: str
    category: str
    vendor: str
    size_ordered: str | None
    dropdown_reason: str
    text: str

    @property
    def has_usable_text(self) -> bool:
        return bool(self.text and self.text.strip())


@dataclass
class ValidationReport:
    """Everything spec section 7.1 requires the upload screen to display."""

    total_rows: int = 0
    other_rows: int = 0
    other_pct: float = 0.0
    unique_skus: int = 0
    unique_categories: int = 0
    unique_vendors: int = 0
    missing_required_columns: list[str] = field(default_factory=list)
    rows_missing_required_values: int = 0
    other_rows_with_empty_text: int = 0
    non_other_rows_with_text: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """A file with any hard error must not be processed (spec 7.1)."""
        return not self.errors

    def to_dict(self) -> dict:
        return asdict(self) | {"is_valid": self.is_valid}


@dataclass
class LoadedData:
    """The result of a load: the frame and its validation report."""

    features: pd.DataFrame
    report: ValidationReport

    @property
    def ground_truth(self) -> pd.DataFrame:
        """The `expected_*` columns, for the evaluator.

        Derived on access rather than split off at load time. Empty columns
        when the file carries no answer key, which is the normal case for a
        real returns export — `has_answer_key` in the API checks for exactly
        that before offering to score anything.
        """
        if self.features.empty:
            return self.features
        present = [c for c in GROUND_TRUTH_COLUMNS if c in self.features.columns]
        if not present:
            return pd.DataFrame(columns=["return_id", *GROUND_TRUTH_COLUMNS])
        return self.features[["return_id", *present]].copy()

    def eligible_records(self) -> list[ReturnRecord]:
        """The rows in scope for classification: dropdown == 'Other' (P1)."""
        subset = self.features[
            self.features["return_reason_dropdown"] == OTHER_DROPDOWN_VALUE
        ]
        return [_to_record(row) for _, row in subset.iterrows()]

    @property
    def dropdown_distribution(self) -> pd.Series:
        """Original dropdown distribution, including the 560 non-Other rows."""
        return self.features["return_reason_dropdown"].value_counts()


def _clean_str(value: Any) -> str:
    """Normalise a cell to a stripped string; NaN/None become ''."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _to_record(row: pd.Series) -> ReturnRecord:
    size = _clean_str(row.get("size_ordered"))
    return ReturnRecord(
        return_id=_clean_str(row["return_id"]),
        sku=_clean_str(row["sku"]),
        product_name=_clean_str(row["product_name"]),
        category=_clean_str(row["category"]),
        vendor=_clean_str(row["vendor"]),
        size_ordered=size or None,
        dropdown_reason=_clean_str(row["return_reason_dropdown"]),
        text=_clean_str(row.get("return_reason_text")),
    )


def _read_workbook(source: Path | str | Any, preferred: str) -> pd.DataFrame:
    """Read `preferred` if the workbook has it, else its first sheet."""
    with pd.ExcelFile(source) as book:
        sheet = preferred if preferred in book.sheet_names else book.sheet_names[0]
        return book.parse(sheet, dtype=str)


def load_dataset(
    source: Path | str | Any,
    sheet_name: str | None = None,
) -> LoadedData:
    """Read the dataset and split ground truth away from features.

    Accepts a path to `.xlsx` or `.csv`, or a file-like object from an
    upload. Reading the workbook directly means there is no manual
    xlsx-to-csv conversion step that could silently rename a column.
    """
    # There is no default file: the dashboard is built only from what the
    # user uploads, so a caller that forgets to pass one must fail loudly
    # rather than quietly classify the bundled sample.
    sheet = sheet_name or config.DATASET_SHEET

    name = getattr(source, "name", str(source))
    report = ValidationReport()

    try:
        if str(name).lower().endswith((".xlsx", ".xlsm", ".xls")):
            raw = _read_workbook(source, sheet)
        else:
            raw = pd.read_csv(source, dtype=str)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        report.errors.append(f"Could not read the file: {exc}")
        empty = pd.DataFrame()
        return LoadedData(features=empty, report=report)

    raw.columns = [str(c).strip() for c in raw.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    report.missing_required_columns = missing
    if missing:
        report.errors.append(
            "Missing required column(s): " + ", ".join(missing) + ". "
            "Cannot process this file."
        )

    report.total_rows = int(len(raw))
    if report.total_rows == 0:
        report.errors.append("The file contains no data rows.")

    if report.errors:
        empty = pd.DataFrame()
        return LoadedData(features=empty, report=report)

    features = raw

    # --- descriptive validation ------------------------------------------
    dropdown = features["return_reason_dropdown"].fillna("")
    text = features.get("return_reason_text", pd.Series("", index=features.index))
    text = text.fillna("")
    is_other = dropdown == OTHER_DROPDOWN_VALUE
    has_text = text.astype(str).str.strip() != ""

    report.other_rows = int(is_other.sum())
    report.other_pct = round(100.0 * report.other_rows / report.total_rows, 2)
    report.unique_skus = int(features["sku"].nunique())
    report.unique_categories = int(features["category"].nunique())
    report.unique_vendors = int(features["vendor"].nunique())
    report.other_rows_with_empty_text = int((is_other & ~has_text).sum())
    report.non_other_rows_with_text = int((~is_other & has_text).sum())

    required_present = [c for c in REQUIRED_COLUMNS if c in features.columns]
    blank = features[required_present].isna() | (
        features[required_present].astype(str).apply(lambda s: s.str.strip() == "")
    )
    report.rows_missing_required_values = int(blank.any(axis=1).sum())

    if features["return_id"].duplicated().any():
        dupes = int(features["return_id"].duplicated().sum())
        report.errors.append(
            f"{dupes} duplicate return_id value(s). return_id must be unique."
        )

    return LoadedData(features=features, report=report)


# --- Run artifact ---------------------------------------------------------


def save_run(payload: dict, path: Path | None = None) -> Path:
    """Persist a completed run, asserting no ground truth reaches disk.

    Written to a temp file and renamed, so a reader never sees a half-written
    run.
    """
    target = path or config.RUN_ARTIFACT
    target.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so a reader never sees a half-written run.
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_run(path: Path | None = None) -> dict | None:
    """Load the persisted run, or None when there isn't one yet."""
    target = path or config.RUN_ARTIFACT
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def export_records(records: Iterable[dict]) -> str:
    """Build the downloadable CSV of final classifications (spec section 29).

    The records come from the run artifact, which holds predictions only.
    """
    frame = pd.DataFrame(list(records))
    # Customer-written text goes straight into this file, and a spreadsheet
    # treats a cell starting with = + - @ as a formula. Prefixing a quote
    # makes Excel and Sheets show it as text instead of running it.
    frame = frame.map(
        lambda v: "'" + v if isinstance(v, str) and v[:1] in _FORMULA_START else v
    )
    return frame.to_csv(index=False)


_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


# --- SQLite: response cache + reviewer corrections ------------------------

_DB_LOCK = threading.Lock()


def _connect(path: Path | None = None) -> sqlite3.Connection:
    target = path or config.STATE_DB
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_state_db(path: Path | None = None) -> None:
    """Create the cache and corrections tables if they do not exist."""
    with _DB_LOCK, _connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS response_cache (
                prompt      TEXT NOT NULL,
                llm_string  TEXT NOT NULL,
                response    TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                PRIMARY KEY (prompt, llm_string)
            );
            CREATE TABLE IF NOT EXISTS corrections (
                session_id     TEXT NOT NULL,
                return_id      TEXT NOT NULL,
                primary_reason TEXT,
                sub_reason     TEXT,
                body_area      TEXT,
                action         TEXT NOT NULL,
                note           TEXT,
                updated_at     TEXT NOT NULL,
                PRIMARY KEY (session_id, return_id)
            );
            CREATE INDEX IF NOT EXISTS corrections_session
                ON corrections (session_id);
            """
        )


def save_correction(
    session_id: str,
    return_id: str,
    *,
    action: str,
    primary_reason: str | None = None,
    sub_reason: str | None = None,
    body_area: str | None = None,
    note: str | None = None,
    path: Path | None = None,
) -> None:
    """Record one reviewer correction, namespaced to a browser session.

    Session-keying is what stops two people on the deployed URL from altering
    each other's dashboard numbers (decision D5).
    """
    init_state_db(path)
    with _DB_LOCK, _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO corrections (session_id, return_id, primary_reason,
                                     sub_reason, body_area, action, note, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, return_id) DO UPDATE SET
                primary_reason=excluded.primary_reason,
                sub_reason=excluded.sub_reason,
                body_area=excluded.body_area,
                action=excluded.action,
                note=excluded.note,
                updated_at=excluded.updated_at
            """,
            (
                session_id,
                return_id,
                primary_reason,
                sub_reason,
                body_area,
                action,
                note,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )


def load_corrections(session_id: str, path: Path | None = None) -> dict[str, dict]:
    """All corrections for one session, keyed by return_id."""
    init_state_db(path)
    with _DB_LOCK, _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT return_id, primary_reason, sub_reason, body_area, action, note,
                   updated_at
            FROM corrections WHERE session_id = ?
            """,
            (session_id,),
        ).fetchall()
    return {
        r[0]: {
            "primary_reason": r[1],
            "sub_reason": r[2],
            "body_area": r[3],
            "action": r[4],
            "note": r[5],
            "updated_at": r[6],
        }
        for r in rows
    }


def clear_corrections(session_id: str, path: Path | None = None) -> int:
    """Drop every correction a session made.

    Corrections are keyed by return_id, so they must not outlive the run they
    were made against: a new upload that reuses `R0012` would otherwise inherit
    the last file's answer for it.
    """
    init_state_db(path)
    with _DB_LOCK, _connect(path) as conn:
        cur = conn.execute("DELETE FROM corrections WHERE session_id = ?", (session_id,))
    return cur.rowcount


def purge_old_corrections(max_age_days: int = 30, path: Path | None = None) -> int:
    """Delete corrections from abandoned sessions (the D5 cleanup policy)."""
    init_state_db(path)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max_age_days)
    with _DB_LOCK, _connect(path) as conn:
        cur = conn.execute(
            "DELETE FROM corrections WHERE updated_at < ?", (cutoff.isoformat(),)
        )
        return int(cur.rowcount or 0)


class SQLiteResponseCache:
    """A LangChain `BaseCache` backed by SQLite.

    Why not provider prompt-caching: it is not portable across providers, and
    the minimum-cacheable-prefix rules differ per model in ways that fail
    SILENTLY (you get a zero cache-write with no error). An exact-match
    response cache is provider-neutral and makes re-running the same 440 rows
    during development free.

    Implements the three `BaseCache` abstract methods. Imported lazily by
    `models.py` so this module stays free of LangChain imports.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config.STATE_DB
        init_state_db(self.path)

    def lookup(self, prompt: str, llm_string: str):
        from langchain_core.load import loads

        with _DB_LOCK, _connect(self.path) as conn:
            row = conn.execute(
                "SELECT response FROM response_cache WHERE prompt=? AND llm_string=?",
                (prompt, llm_string),
            ).fetchone()
        if not row:
            return None
        try:
            # `allowed_objects="messages"` restricts deserialisation to chat
            # message types. The cache file is local, but a deserialiser that
            # will instantiate arbitrary classes from disk is a bad default to
            # ship regardless of current threat model.
            return [
                loads(blob, allowed_objects="messages")
                for blob in json.loads(row[0])
            ]
        except Exception:  # noqa: BLE001 - a stale cache must never break a run
            return None

    def update(self, prompt: str, llm_string: str, return_val) -> None:
        from langchain_core.load import dumps

        try:
            blob = json.dumps([dumps(g) for g in return_val])
        except Exception:  # noqa: BLE001 - caching is best effort
            return
        with _DB_LOCK, _connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO response_cache (prompt, llm_string, response, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(prompt, llm_string) DO UPDATE SET
                    response=excluded.response, created_at=excluded.created_at
                """,
                (
                    prompt,
                    llm_string,
                    blob,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )

    def clear(self, **kwargs: Any) -> None:
        with _DB_LOCK, _connect(self.path) as conn:
            conn.execute("DELETE FROM response_cache")
