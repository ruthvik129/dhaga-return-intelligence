"""Dhaga & Co. Return Reason Intelligence — JSON API for the React front end.

DETERMINISTIC MODULE — this file makes no model calls of its own. It reads the
run artifact, asks `src/metrics.py` for numbers, and asks `run_pipeline.run`
to do live work when a classification job is started. All LLM work lives in
`src/models.py` (spec section 16).

The rules the earlier Streamlit front end was built on carry over:

1. **Upload first, nothing by default.** The dashboard is built only from a
   file this browser session uploaded (or explicitly chose: the sample) and
   classified. Until then every screen answers `artifact: false` and the UI
   sends the user to Upload & Process. Each session's run is stored on its own
   (`output/runs/<session>.json`) so one person's upload never replaces
   another person's dashboard.
2. **Every aggregate drills down to verbatim comments.**
3. **Corrections never mutate the run.** They are keyed to the browser's
   `X-Session-Id` header and layered at read time (decision D5).

Everything leaving this module goes through `api/presentation.py`, which
removes model names and "Model A/B" wording before the browser sees it.

Run with:  .venv/bin/uvicorn api.server:app --port 8000
"""

from __future__ import annotations

import logging
import os
import re
import threading
import zipfile
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config, data_io, metrics
from src.models import ModelUnavailable, credential_present, probe_model
from src.router import review_budget_minutes
from src.schemas import TAXONOMY, BodyArea, PrimaryReason

from . import presentation as pres

app = FastAPI(title="Dhaga Return Reason Intelligence", docs_url="/api/docs")

WEB_DIST = config.PROJECT_ROOT / "web" / "dist"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_UPLOAD_NAME = re.compile(r"uploaded_[0-9a-f]{32}\.(xlsx|csv)")
_SESSION = re.compile(r"[A-Za-z0-9_-]{8,64}")
GROUP_COLUMNS = ("sku", "vendor", "category")

#: One job may classify at most this many rows. Classification spends money,
#: so the browser's `limit` is a request, not an authority.
MAX_JOB_ROWS = int(os.getenv("MAX_JOB_ROWS", "2000"))

#: An .xlsx is a zip. The 20 MB cap is on the compressed bytes, so the
#: uncompressed size is checked too before pandas inflates it.
MAX_XLSX_UNCOMPRESSED = 200 * 1024 * 1024

#: Uploads older than this are deleted the next time anyone uploads.
UPLOAD_TTL_SECONDS = 24 * 3600

log = logging.getLogger("dhaga.api")


@app.middleware("http")
async def _limit_body(request: Request, call_next):
    """Refuse oversized bodies before they are spooled to disk."""
    size = request.headers.get("content-length")
    if size and size.isdigit() and int(size) > MAX_UPLOAD_BYTES + 1024 * 1024:
        return JSONResponse({"detail": "Request is larger than 20 MB."}, status_code=413)
    return await call_next(request)


# --- shared state ---------------------------------------------------------


def _session(session_id: str | None) -> str:
    """Validate the browser's session id; corrections are private to it."""
    if not session_id or not _SESSION.fullmatch(session_id):
        raise HTTPException(400, "Missing or malformed X-Session-Id header.")
    return session_id


@lru_cache(maxsize=4)
def _artifact_cached(path_str: str, mtime: float) -> dict | None:
    del mtime  # in the key so a fresh run busts the cache
    return data_io.load_run(Path(path_str))


RUNS_DIR = config.OUTPUT_DIR / "runs"


def run_path(session_id: str) -> Path:
    """Where this session's run lives. `session_id` is already validated."""
    return RUNS_DIR / f"{session_id}.json"


#: On a public demo a first-time visitor would otherwise land on "upload a
#: file" and have to wait out a full run before seeing anything. With this on,
#: a session with no run of its own reads the committed one; the moment they
#: classify their own file, their run takes over.
DEMO_SEED_RUN = os.getenv("DEMO_SEED_RUN", "0").strip().lower() in {"1", "true", "yes"}


def load_artifact(session_id: str) -> dict | None:
    """This session's run, or None until it has uploaded and classified."""
    path = run_path(session_id)
    if not path.exists():
        if DEMO_SEED_RUN and config.RUN_ARTIFACT.exists():
            return _artifact_cached(
                str(config.RUN_ARTIFACT), config.RUN_ARTIFACT.stat().st_mtime
            )
        return None
    return _artifact_cached(str(path), path.stat().st_mtime)


@lru_cache(maxsize=4)
def _dataset_cached(path_str: str, mtime: float):
    del mtime
    loaded = data_io.load_dataset(Path(path_str))
    return loaded.features, loaded.ground_truth, loaded.report.to_dict()


def dataset(path: Path):
    return _dataset_cached(str(path), path.stat().st_mtime)


SAMPLE = "sample"


def resolve_source(source: str) -> Path:
    """`sample` or an upload token. Never a caller-supplied path."""
    if source == SAMPLE:
        return config.DEFAULT_DATASET
    if not _UPLOAD_NAME.fullmatch(source):
        raise HTTPException(400, "Unknown dataset.")
    path = config.OUTPUT_DIR / source
    if not path.exists():
        raise HTTPException(404, "That upload no longer exists. Upload it again.")
    return path


def require_artifact(session_id: str) -> dict:
    payload = load_artifact(session_id)
    if payload is None:
        raise HTTPException(409, "Nothing processed yet. Upload a file first.")
    return payload


def run_dataset(payload: dict):
    """The features, ground truth and report of the file this run classified."""
    source = (payload.get("run") or {}).get("source")
    try:
        return dataset(resolve_source(source or ""))
    except HTTPException:
        # The uploaded file has expired; scoring needs it, the rest does not.
        empty = pd.DataFrame()
        return empty, empty, payload["validation"]


def has_answer_key(ground_truth: pd.DataFrame) -> bool:
    """Scoring needs every expected_* column; a partial key is not a key."""
    return bool(
        not ground_truth.empty
        and all(c in ground_truth.columns for c in config.GROUND_TRUTH_COLUMNS)
        and ground_truth[list(config.GROUND_TRUTH_COLUMNS)].notna().any().any()
    )


def working_frame(payload: dict, session_id: str) -> pd.DataFrame:
    """Predictions with this session's corrections layered on top."""
    frame = metrics.results_frame(payload["records"])
    return metrics.apply_corrections(frame, data_io.load_corrections(session_id))


def records(frame: pd.DataFrame, payload: dict) -> list[dict]:
    """A frame as browser-safe JSON rows (NaN -> null, model identity removed)."""
    if frame.empty:
        return []
    clean = frame.astype(object).where(pd.notna(frame), None)
    ids = pres.run_ids(payload)
    return [pres.public_record(r, ids) for r in clean.to_dict("records")]


def _review_count(frame: pd.DataFrame) -> int:
    if frame.empty or "status" not in frame.columns:
        return 0
    return int((frame["status"] == "NEEDS_HUMAN_REVIEW").sum())


def table(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def readiness() -> dict:
    """Pre-flight: can live classification run? Names no model."""
    ok_all = True
    messages: list[str] = []
    for spec in (config.MODEL_A, config.MODEL_B):
        ok, msg = probe_model(spec)
        ok_all = ok_all and ok
        if not ok:
            if credential_present(spec) is False:
                msg = (
                    "No OpenRouter key found. Set one of "
                    f"{', '.join(config.OPENROUTER_KEY_ENV)} in .env to run live "
                    "classification."
                )
            else:
                log.warning("classifier probe failed: %s", msg)
                msg = (
                    "The classifier could not be initialised. Check the "
                    "MODEL_* settings in .env."
                )
            messages.append(msg)
    if ok_all:
        messages = ["Classifier ready: credential present."]
    return {"ready": ok_all, "messages": sorted(set(messages))}


# --- meta -----------------------------------------------------------------


def _public_job(job: dict | None) -> dict | None:
    """A copy of a job without its owner. Copied under the lock because the
    job thread updates the same dict."""
    if job is None:
        return None
    with _JOB_LOCK:
        return {k: v for k, v in job.items() if k != "session"}


def _session_job(session_id: str) -> dict | None:
    """The most recent job this session started, if any."""
    with _JOB_LOCK:
        mine = [j for j in _JOBS.values() if j["session"] == session_id]
    return max(mine, key=lambda j: j["started_at"]) if mine else None


@app.get("/api/status")
def status(x_session_id: str | None = Header(default=None)) -> dict:
    sid = _session(x_session_id)
    payload = load_artifact(sid)
    out: dict = {
        "artifact": payload is not None,
        "classifier": readiness(),
        "job": _public_job(_session_job(sid)),
    }
    if payload is not None:
        frame = working_frame(payload, sid)
        out["run"] = pres.public_run(payload)
        out["run_total_usd"] = payload["cost"].get("run_total_usd")
        out["review_pending"] = _review_count(frame)
    return out


@app.delete("/api/run")
def clear_run(x_session_id: str | None = Header(default=None)) -> dict:
    """Start over: drop this session's run so the dashboard is empty again."""
    sid = _session(x_session_id)
    run_path(sid).unlink(missing_ok=True)
    data_io.clear_corrections(sid)
    _artifact_cached.cache_clear()
    return {"ok": True}


@app.get("/api/taxonomy")
def taxonomy() -> dict:
    return {
        "primary_reasons": [p.value for p in PrimaryReason],
        "sub_reasons": {p.value: list(subs) for p, subs in TAXONOMY.items()},
        "body_areas": [b.value for b in BodyArea],
    }


# --- Overview -------------------------------------------------------------


@app.get("/api/overview")
def overview(x_session_id: str | None = Header(default=None)) -> dict:
    sid = _session(x_session_id)
    payload = load_artifact(sid)
    if payload is None:
        return {"artifact": False}
    frame = working_frame(payload, sid)
    report = payload["validation"]
    return {
        "artifact": True,
        "run": pres.public_run(payload),
        "counts": pres.public_overview(metrics.overview(frame, report)),
        "dropdown_distribution": [
            {"label": k, "rows": int(v)}
            for k, v in sorted(
                payload["dropdown_distribution"].items(), key=lambda kv: -kv[1]
            )
        ],
        "non_other_rows_with_text": report["non_other_rows_with_text"],
        "validation": report,
    }


# --- Upload & Process -----------------------------------------------------


def _report(source: str) -> dict:
    path = resolve_source(source)
    try:
        _, ground_truth, report = dataset(path)
    except Exception as exc:  # noqa: BLE001 - logged; the browser gets a fixed message
        log.exception("could not read %s", source)
        raise HTTPException(
            422, "Could not read that file. Check it is a valid .xlsx or .csv."
        ) from exc
    return {
        "source": source,
        "name": config.DEFAULT_DATASET.name if source == SAMPLE else source,
        "report": report,
        "has_answer_key": has_answer_key(ground_truth),
        "classifier": readiness(),
        "max_rows": MAX_JOB_ROWS,
    }


@app.get("/api/dataset")
def dataset_report(source: str) -> dict:
    return _report(source)


def _purge_old_uploads() -> None:
    cutoff = time.time() - UPLOAD_TTL_SECONDS
    for path in config.OUTPUT_DIR.glob("uploaded_*"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def _check_xlsx(path: Path) -> None:
    """Reject a zip whose contents would inflate past a sane size."""
    try:
        with zipfile.ZipFile(path) as z:
            total = sum(i.file_size for i in z.infolist())
    except zipfile.BadZipFile as exc:
        raise HTTPException(422, "That .xlsx file is not a valid workbook.") from exc
    if total > MAX_XLSX_UNCOMPRESSED:
        raise HTTPException(413, "That workbook is too large once uncompressed.")


def _store_upload(token: str, blob: bytes) -> dict:
    """Write, validate and report an upload. Blocking; run off the event loop."""
    _purge_old_uploads()
    path = config.OUTPUT_DIR / token
    path.write_bytes(blob)
    try:
        if path.suffix == ".xlsx":
            _check_xlsx(path)
        report = _report(token)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    if not report["report"]["is_valid"]:
        path.unlink(missing_ok=True)  # it can never be processed
    return report


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".xlsx", ".csv"}:
        raise HTTPException(400, "Upload an .xlsx or .csv file.")
    blob = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(blob) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 20 MB.")
    # A generated name, never the client's: the client's name is a path.
    token = f"uploaded_{uuid.uuid4().hex}{suffix}"
    report = await run_in_threadpool(_store_upload, token, blob)
    return {**report, "original_name": Path(file.filename or token).name}


class JobRequest(BaseModel):
    source: str
    limit: int = Field(ge=1)


_JOBS: dict[str, dict] = {}
_JOB_LOCK = threading.Lock()


def _set(job: dict, **fields) -> None:
    """Update a job under the lock; request threads read it concurrently."""
    with _JOB_LOCK:
        job.update(fields)


def _run_job(job_id: str, session_id: str, source: str, limit: int) -> None:
    """Classify `limit` rows of `source` and save them as this session's run.

    Runs on a worker thread. Failures are logged in full but reach the browser
    only as fixed sentences, because provider errors can name the model.
    """
    job = _JOBS[job_id]

    def tick(msg: str, frac: float) -> None:
        _set(job, message=pres.progress_message(msg), progress=min(max(frac, 0.0), 1.0))

    try:
        import run_pipeline

        payload = run_pipeline.run(
            limit=limit, dataset=str(resolve_source(source)), progress=tick
        )
        # Remember which file this run came from: Overview reads its
        # validation report and Evaluation scores against its answer key.
        payload["run"]["source"] = source
        data_io.save_run(payload, run_path(session_id))
        # A new run starts with a clean review: old corrections were made
        # against a different file's rows.
        data_io.clear_corrections(session_id)
        _artifact_cached.cache_clear()
        counts = (
            pd.Series([pres.public_status(r["status"]) for r in payload["records"]])
            .value_counts()
            .to_dict()
        )
        _set(
            job,
            state="done",
            progress=1.0,
            message="Finished",
            result={
                "rows": payload["run"]["eligible_rows"],
                "elapsed_seconds": payload["run"]["elapsed_seconds"],
                "run_total_usd": payload["cost"].get("run_total_usd"),
                "counts": {k: int(v) for k, v in counts.items()},
            },
        )
    except ModelUnavailable:
        log.exception("job %s: classifier unavailable", job_id)
        _set(job, state="error", error="The classifier is not available. Check the API key in .env.")
    except SystemExit as exc:  # load_dataset validation failure, already user-facing
        _set(job, state="error", error=pres.scrub(str(exc)))
    except BaseException:  # noqa: BLE001 - provider errors can name models; log, don't relay
        log.exception("job %s failed", job_id)
        _set(
            job,
            state="error",
            error=f"Classification failed (reference {job_id[:8]}). See the server log.",
        )
    finally:
        _set(job, finished_at=time.time())


@app.post("/api/jobs", status_code=202)
def start_job(req: JobRequest, x_session_id: str | None = Header(default=None)) -> dict:
    sid = _session(x_session_id)
    _, _, report = dataset(resolve_source(req.source))
    if not report["is_valid"] or not report["other_rows"]:
        raise HTTPException(400, "That file has no 'Other' rows to classify.")
    if req.limit > MAX_JOB_ROWS:
        raise HTTPException(
            400, f"One run can classify at most {MAX_JOB_ROWS:,} rows."
        )
    if not readiness()["ready"]:
        raise HTTPException(409, "Live classification is not configured.")
    with _JOB_LOCK:
        # One live run at a time across the server: it is a spend surface and
        # the classifier is rate-limited, so queued runs would only contend.
        if any(j["state"] == "running" for j in _JOBS.values()):
            raise HTTPException(
                409, "Another classification run is in progress. Try again shortly."
            )
        cutoff = time.time() - 3600
        for old in [k for k, j in _JOBS.items() if (j["finished_at"] or time.time()) < cutoff]:
            del _JOBS[old]
        job_id = uuid.uuid4().hex
        _JOBS[job_id] = {
            "id": job_id,
            "session": sid,
            "state": "running",
            "progress": 0.0,
            "message": "Starting",
            "started_at": time.time(),
            "finished_at": None,
            "result": None,
            "error": None,
        }
    threading.Thread(
        target=_run_job, args=(job_id, sid, req.source, req.limit), daemon=True
    ).start()
    return _public_job(_JOBS[job_id])


@app.get("/api/jobs/{job_id}")
def job(job_id: str, x_session_id: str | None = Header(default=None)) -> dict:
    sid = _session(x_session_id)
    with _JOB_LOCK:
        found = _JOBS.get(job_id)
    if found is None or found["session"] != sid:
        raise HTTPException(404, "No such job.")
    return _public_job(found)


# --- Insights -------------------------------------------------------------


def _check_group(group_by: str) -> str:
    if group_by not in GROUP_COLUMNS:
        raise HTTPException(400, f"group_by must be one of {', '.join(GROUP_COLUMNS)}.")
    return group_by


# --- Briefing (the home screen) -------------------------------------------


def _public_queue(summary: dict) -> dict:
    """Queue counts with public reason codes, not internal ones."""
    out = dict(summary)
    out["by_reason"] = [
        {**r, "reason": pres.public_review_reason(r["reason"])}
        for r in summary.get("by_reason", [])
    ]
    return out


@app.get("/api/briefing")
def briefing(x_session_id: str | None = Header(default=None)) -> dict:
    """The home screen: what to fix, what is waiting, and whether to trust it.

    Everything here is computed in `src/metrics.py`; this endpoint only shapes
    it for the browser and scrubs model identity. The screen leads with
    conclusions because the audience is a merchandiser deciding what to change
    this week, not an engineer auditing a pipeline.
    """
    sid = _session(x_session_id)
    payload = load_artifact(sid)
    if payload is None:
        return {"artifact": False}

    frame = working_frame(payload, sid)
    counts = pres.public_overview(metrics.overview(frame, payload["validation"]))
    findings = metrics.headline_findings(frame)
    ids = pres.run_ids(payload)
    for f in findings:
        for q in f["quotes"]:
            q["text"] = pres.scrub(q["text"], ids)

    out = {
        "artifact": True,
        "run": pres.public_run(payload),
        "counts": counts,
        "findings": findings,
        "queue": _public_queue(metrics.queue_summary(frame)),
        "cost": {
            "run_total_usd": payload["cost"].get("run_total_usd"),
            "per_1000_usd": payload["cost"].get("cost_per_1000_comments_usd"),
        },
        "non_other_rows_with_text": payload["validation"]["non_other_rows_with_text"],
        "validation": payload["validation"],
    }

    # The trust line only appears when the file carried an answer key. Without
    # one, claiming an accuracy figure would be inventing it.
    _, ground_truth, _ = run_dataset(payload)
    out["accuracy"] = None
    if has_answer_key(ground_truth):
        ev = metrics.evaluate(frame, ground_truth)
        if ev.get("available"):
            out["accuracy"] = {
                "overall_pct": ev["primary_accuracy_overall"]["accuracy_pct"],
                "auto_pct": ev["auto_accept_precision_a"]["accuracy_pct"],
                "rows": ev["primary_accuracy_overall"]["n"],
            }
    return out


@app.get("/api/insights/reasons")
def insights_reasons(
    primary: str | None = None, x_session_id: str | None = Header(default=None)
) -> dict:
    sid = _session(x_session_id)
    payload = require_artifact(sid)
    frame = working_frame(payload, sid)
    return {
        "run": pres.public_run(payload),
        "reasons": table(metrics.reason_distribution(frame)),
        "sub_reasons": table(metrics.sub_reason_distribution(frame, primary)),
        "body_areas": table(metrics.body_area_distribution(frame, primary or "fit")),
    }


@app.get("/api/insights/groups")
def insights_groups(
    group_by: str = "sku",
    reason: str = "fit",
    min_rows: int = Query(5, ge=1, le=1000),
    x_session_id: str | None = Header(default=None),
) -> dict:
    sid = _session(x_session_id)
    group_by = _check_group(group_by)
    frame = working_frame(require_artifact(sid), sid)
    groups = metrics.group_intelligence(frame, group_by, reason, min_rows=min_rows)
    rows = table(groups)
    for row in rows:
        row["group"] = row.pop(group_by)
    return {"group_by": group_by, "reason": reason, "groups": rows}


@app.get("/api/insights/group")
def insights_group(
    group_by: str,
    value: str,
    reason: str,
    x_session_id: str | None = Header(default=None),
) -> dict:
    sid = _session(x_session_id)
    group_by = _check_group(group_by)
    payload = require_artifact(sid)
    frame = working_frame(payload, sid)
    scoped = frame[frame[group_by] == value]
    rows = metrics.drill_down(
        frame, group_col=group_by, group_value=value, primary=reason
    )
    return {
        "sub_reasons": table(metrics.sub_reason_distribution(scoped, reason)),
        "body_areas": table(metrics.body_area_distribution(scoped, "fit")),
        "rows": records(rows, payload),
    }


@app.get("/api/export.csv", response_class=PlainTextResponse)
def export_csv(
    session: str,
    group_by: str | None = None,
    value: str | None = None,
    reason: str | None = None,
) -> PlainTextResponse:
    """CSV download. The session travels as a query param because a plain
    `<a download>` link cannot set headers."""
    sid = _session(session)
    payload = require_artifact(sid)
    frame = working_frame(payload, sid)
    if group_by:
        _check_group(group_by)
        frame = metrics.drill_down(
            frame, group_col=group_by, group_value=value, primary=reason
        )
        name = f"{group_by}_{value}_{reason}.csv"
    else:
        name = "dhaga_classifications.csv"
    rows = records(frame.drop(columns=["trace"], errors="ignore"), payload)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return PlainTextResponse(
        data_io.export_records(rows),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{safe}"'},
    )


# --- Review Queue ---------------------------------------------------------


@app.get("/api/review")
def review_queue(x_session_id: str | None = Header(default=None)) -> dict:
    sid = _session(x_session_id)
    payload = require_artifact(sid)
    frame = working_frame(payload, sid)
    in_run = set(frame["return_id"]) if not frame.empty else set()
    corrections = {
        k: v for k, v in data_io.load_corrections(sid).items() if k in in_run
    }
    queue = frame[frame["status"] == "NEEDS_HUMAN_REVIEW"] if not frame.empty else frame
    return {
        "queue": records(queue, payload),
        "corrections": [{"return_id": k, **v} for k, v in corrections.items()],
        "estimated_minutes": review_budget_minutes(len(queue)),
        "minutes_per_row": config.REVIEW_MINUTES_PER_ROW,
    }


class Correction(BaseModel):
    action: Literal["confirm", "change", "mark_unclear"]
    primary_reason: str
    sub_reason: str
    body_area: str
    note: str | None = Field(default=None, max_length=1000)


@app.post("/api/review/{return_id}")
def save_review(
    return_id: str,
    fix: Correction,
    x_session_id: str | None = Header(default=None),
) -> dict:
    sid = _session(x_session_id)
    payload = require_artifact(sid)
    if not any(r["return_id"] == return_id for r in payload["records"]):
        raise HTTPException(404, "No such return in this run.")
    try:
        primary = PrimaryReason(fix.primary_reason)
    except ValueError as exc:
        raise HTTPException(400, "Unknown primary reason.") from exc
    if fix.sub_reason not in TAXONOMY[primary]:
        raise HTTPException(400, "That sub-reason does not belong to that primary.")
    if fix.body_area not in {b.value for b in BodyArea}:
        raise HTTPException(400, "Unknown body area.")
    data_io.save_correction(
        sid,
        return_id,
        action=fix.action,
        primary_reason=fix.primary_reason,
        sub_reason=fix.sub_reason,
        body_area=fix.body_area,
        note=fix.note or None,
    )
    return {"ok": True, "return_id": return_id, "action": fix.action}


# --- Evaluation -----------------------------------------------------------


def _baseline(features: pd.DataFrame, ground_truth: pd.DataFrame, ids: set) -> dict:
    """The rules control arm over exactly the rows this run classified."""
    other = features[features["return_id"].isin(ids)]
    texts = other["return_reason_text"].fillna("").tolist() if "return_reason_text" in other else [""] * len(other)
    return metrics.evaluate_baseline(texts, other["return_id"].tolist(), ground_truth)


@app.get("/api/evaluation")
def evaluation(x_session_id: str | None = Header(default=None)) -> dict:
    sid = _session(x_session_id)
    payload = load_artifact(sid)
    if payload is None:
        return {"artifact": False}

    features, ground_truth, report = run_dataset(payload)
    out: dict = {
        "artifact": True,
        "run": pres.public_run(payload),
        "dataset": {
            "total_rows": report["total_rows"],
            "other_rows": report["other_rows"],
        },
        "has_answer_key": has_answer_key(ground_truth),
    }
    if not out["has_answer_key"] or features.empty:
        return out

    frame = working_frame(payload, sid)
    out["baseline"] = _baseline(features, ground_truth, set(frame["return_id"]))
    ids = pres.run_ids(payload)
    ev = pres.public_evaluation(metrics.evaluate(frame, ground_truth), ids)
    if not ev.get("available"):
        # Answer-key columns exist, but none of them line up with the rows
        # this run classified, so there is nothing to score.
        return {**out, "cascade": ev, "rows_scored": 0}
    merged = metrics._join_truth(frame, ground_truth)
    wrong = merged[merged["primary_reason"] != merged["expected_primary_reason"]]
    misses = [
        {**r, "status": pres.public_status(r["status"])}
        for r in table(
            wrong[
                [
                    "return_id",
                    "original_text",
                    "primary_reason",
                    "expected_primary_reason",
                    "confidence",
                    "status",
                ]
            ]
        )
    ]
    return {**out, "cascade": ev, "misses": misses, "rows_scored": len(merged)}


# --- Run Cost -------------------------------------------------------------


@app.get("/api/cost")
def cost(x_session_id: str | None = Header(default=None)) -> dict:
    sid = _session(x_session_id)
    payload = require_artifact(sid)
    cfg = payload["run"].get("config") or {}
    frame = working_frame(payload, sid)
    queue = _review_count(frame)
    return {
        "run": pres.public_run(payload),
        "cost": pres.public_cost(payload["cost"], cfg),
        "escalation": pres.public_escalation(payload["escalation"]),
        "usage": pres.public_usage(payload["usage"], cfg),
        "review_share": queue / len(frame) if len(frame) else 0.0,
        "minutes_per_row": config.REVIEW_MINUTES_PER_ROW,
    }


# --- errors and static files ---------------------------------------------


@app.exception_handler(HTTPException)
async def _http_error(_request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"detail": pres.scrub(str(exc.detail))}, status_code=exc.status_code)


if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(404, "Not found.")
        target = (WEB_DIST / path).resolve()
        if path and target.is_file() and WEB_DIST.resolve() in target.parents:
            return FileResponse(target)
        # index.html names the hashed bundles, so it must never be cached:
        # a stale copy keeps serving the previous build after a deploy.
        return FileResponse(
            WEB_DIST / "index.html", headers={"Cache-Control": "no-cache"}
        )
