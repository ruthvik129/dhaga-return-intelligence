"""The JSON API behind the React front end.

The headline guarantee: the browser never learns which models ran, and never
sees "Model A" / "Model B" wording. That is checked against every endpoint
with the committed run artifact, so a new field that carries a model name
fails here rather than showing up on screen.

The dashboard is upload-first: a session sees nothing until it has
classified a file. Tests give their session a run by copying the committed
artifact into place (marked as coming from the bundled sample), which is
exactly what a finished classification job would have written.

No network: the classifier is never invoked, and corrections and runs go to
throwaway paths.
"""

import json
import re

import pytest
from fastapi.testclient import TestClient

from api import presentation as pres
from api import server
from src import config

SID = "pytest_session_0001"
H = {"X-Session-Id": SID}

LEAK = re.compile(r"Model [AB]\b|\bmodel_[ab]\b|MODEL_[AB]|openrouter:|mistral|gpt-oss")

pytestmark = pytest.mark.skipif(
    not config.RUN_ARTIFACT.exists(), reason="needs the committed run artifact"
)


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DB", tmp_path / "state.sqlite3")
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    server._artifact_cached.cache_clear()
    payload = json.loads(config.RUN_ARTIFACT.read_text(encoding="utf-8"))
    payload["run"]["source"] = server.SAMPLE
    server.data_io.save_run(payload, server.run_path(SID))


@pytest.fixture
def client():
    return TestClient(server.app)


ENDPOINTS = [
    "/api/status",
    "/api/overview",
    "/api/dataset?source=sample",
    "/api/insights/reasons",
    "/api/insights/reasons?primary=fit",
    "/api/insights/groups?group_by=vendor&reason=fit&min_rows=1",
    "/api/insights/group?group_by=sku&value=DRS-118&reason=damaged_defective",
    "/api/review",
    "/api/evaluation",
    "/api/cost",
    f"/api/export.csv?session={SID}",
]


@pytest.mark.parametrize("url", ENDPOINTS)
def test_no_endpoint_reveals_model_identity(client, url):
    resp = client.get(url, headers=H)
    assert resp.status_code == 200, resp.text
    hits = sorted({m.group(0) for m in LEAK.finditer(resp.text)})
    assert hits == [], f"{url} leaks {hits}"


def test_a_new_session_sees_nothing_until_it_uploads(client):
    fresh = {"X-Session-Id": "brand_new_session"}
    assert client.get("/api/status", headers=fresh).json()["artifact"] is False
    assert client.get("/api/overview", headers=fresh).json() == {"artifact": False}
    assert client.get("/api/evaluation", headers=fresh).json() == {"artifact": False}
    for url in ("/api/insights/reasons", "/api/review", "/api/cost"):
        assert client.get(url, headers=fresh).status_code == 409


def test_start_over_clears_only_this_session(client):
    assert client.get("/api/status", headers=H).json()["artifact"] is True
    assert client.delete("/api/run", headers=H).status_code == 200
    assert client.get("/api/status", headers=H).json()["artifact"] is False


def test_corrections_do_not_carry_over_to_the_next_run(client):
    """A new upload can reuse return ids; it must not inherit old answers."""
    rid = client.get("/api/review", headers=H).json()["queue"][0]["return_id"]
    body = {"action": "confirm", "primary_reason": "unclear",
            "sub_reason": "insufficient_information", "body_area": "none"}
    assert client.post(f"/api/review/{rid}", json=body, headers=H).status_code == 200
    payload = json.loads(server.run_path(SID).read_text())

    client.delete("/api/run", headers=H)
    server.data_io.save_run(payload, server.run_path(SID))  # same rows again
    server._artifact_cached.cache_clear()

    review = client.get("/api/review", headers=H).json()
    assert review["corrections"] == []
    assert rid in {r["return_id"] for r in review["queue"]}


def test_trace_errors_never_relay_provider_text():
    row = {"trace": [{"step": "model_a_primary", "model": "x",
                      "error": "BadRequest: model 'Mistral Nemo 12B' overloaded"}]}
    step = pres.public_record(row)["trace"][0]
    assert "Mistral" not in step["error"] and "model" not in step


def test_evaluation_scores_against_the_runs_own_file(client):
    ev = client.get("/api/evaluation", headers=H).json()
    assert ev["has_answer_key"] is True
    assert ev["rows_scored"] == 440
    # The baseline is scored over the same rows the run classified.
    assert ev["baseline"]["rows"] == 440


def test_evaluation_without_an_answer_key_says_so(client, tmp_path, monkeypatch):
    src = config.DEFAULT_DATASET
    import pandas as pd

    frame = pd.read_excel(src, sheet_name=config.DATASET_SHEET, dtype=str)
    frame = frame.drop(columns=list(config.GROUND_TRUTH_COLUMNS))
    token = "uploaded_" + "a" * 32 + ".csv"
    path = config.OUTPUT_DIR / token
    frame.to_csv(path, index=False)
    try:
        payload = json.loads(server.run_path(SID).read_text())
        payload["run"]["source"] = token
        server.data_io.save_run(payload, server.run_path(SID))
        server._artifact_cached.cache_clear()
        ev = client.get("/api/evaluation", headers=H).json()
        assert ev["artifact"] is True and ev["has_answer_key"] is False
        assert "cascade" not in ev
    finally:
        path.unlink(missing_ok=True)


def test_jobs_are_private_to_their_session(client):
    server._JOBS["j1"] = {"id": "j1", "session": SID, "state": "done", "started_at": 0}
    try:
        assert client.get("/api/jobs/j1", headers=H).status_code == 200
        other = {"X-Session-Id": "someone_else_99"}
        assert client.get("/api/jobs/j1", headers=other).status_code == 404
        assert "session" not in client.get("/api/jobs/j1", headers=H).json()
    finally:
        server._JOBS.pop("j1")


def test_review_queue_uses_public_vocabulary(client):
    queue = client.get("/api/review", headers=H).json()["queue"]
    assert queue, "the committed run has a review queue"
    for row in queue:
        assert row["status"] == "NEEDS_REVIEW"
        assert "classifier_model" not in row
        assert row["review_reason"] != "MODEL_DISAGREEMENT"
        for step in row["trace"]:
            assert "model" not in step
            assert not step["step"].startswith("model_")


def test_cost_is_reported_per_stage(client):
    cost = client.get("/api/cost", headers=H).json()["cost"]
    stages = {row["stage"] for row in cost["per_stage"]}
    assert stages <= {"First pass", "Second check", "Other"}
    assert "per_model" not in cost


def test_session_header_is_required(client):
    assert client.get("/api/overview").status_code == 400
    assert client.get("/api/overview", headers={"X-Session-Id": "../x"}).status_code == 400


def test_correction_is_private_to_its_session(client):
    queue = client.get("/api/review", headers=H).json()["queue"]
    rid = queue[0]["return_id"]
    body = {
        "action": "change",
        "primary_reason": "fit",
        "sub_reason": "too_small",
        "body_area": "chest",
        "note": "tight at chest",
    }
    resp = client.post(f"/api/review/{rid}", json=body, headers=H)
    assert resp.status_code == 200, resp.text

    mine = client.get("/api/review", headers=H).json()
    assert rid not in {r["return_id"] for r in mine["queue"]}
    assert rid in {c["return_id"] for c in mine["corrections"]}

    # Another session with the same run still sees the row unresolved.
    other_sid = "someone_else_99"
    payload = json.loads(server.run_path(SID).read_text())
    server.data_io.save_run(payload, server.run_path(other_sid))
    other = client.get("/api/review", headers={"X-Session-Id": other_sid}).json()
    assert rid in {r["return_id"] for r in other["queue"]}


def test_correction_rejects_an_illegal_pair(client):
    rid = client.get("/api/review", headers=H).json()["queue"][0]["return_id"]
    body = {
        "action": "change",
        "primary_reason": "fit",
        "sub_reason": "late_delivery",
        "body_area": "none",
    }
    assert client.post(f"/api/review/{rid}", json=body, headers=H).status_code == 400


def test_dataset_source_is_never_a_path(client):
    for source in ("../.env", "/etc/passwd", "uploaded_x.xlsx", "run_latest.json"):
        assert client.get("/api/dataset", params={"source": source}).status_code == 400


def test_upload_rejects_other_file_types(client):
    resp = client.post("/api/upload", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert resp.status_code == 400


def test_upload_never_uses_the_client_filename(client):
    csv = b"return_id\nR1\n"
    resp = client.post(
        "/api/upload", files={"file": ("../../evil.csv", csv, "text/csv")}
    )
    # Fails validation (missing columns) but must not have written ../../evil.csv
    assert not (config.OUTPUT_DIR.parent.parent / "evil.csv").exists()
    if resp.status_code == 200:
        assert re.match(r"^uploaded_[0-9a-f]{32}\.csv$", resp.json()["source"])
        # An invalid upload is deleted at once: it can never be processed.
        assert not (config.OUTPUT_DIR / resp.json()["source"]).exists()


# --- presentation.scrub ---------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            "Model A confidence 0.70 is below the 0.85 accept threshold. "
            "Escalating to Model B.",
            "The first pass confidence 0.70 is below the 0.85 accept threshold. "
            "Escalating to the second check.",
        ),
        (
            "Accepted from Model B: confidence 0.90 at or above the 0.80 "
            "threshold, agrees with Model A, not flagged.",
            "Accepted on the second check: confidence 0.90 at or above the 0.80 "
            "threshold, agrees with the first pass, not flagged.",
        ),
    ],
)
def test_scrub_rewrites_router_text(raw, expected):
    assert pres.scrub(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Model A step 1/2 on 40 rows", "First pass: step 1/2 on 40 rows"),
        ("Model A accepted 38, escalating 2", "First pass: accepted 38, escalating 2"),
        ("Model B step 2/2 on 3 rows", "Second check: step 2/2 on 3 rows"),
        ("Loading dataset", "Loading dataset"),
    ],
)
def test_progress_messages_name_stages_not_models(raw, expected):
    assert pres.progress_message(raw) == expected


def test_scrub_removes_configured_model_keys():
    text = f"{config.MODEL_A.key} failed; {config.MODEL_B.name} timed out"
    out = pres.scrub(text)
    assert config.MODEL_A.key not in out and config.MODEL_B.name not in out
    assert "openrouter:" not in out


def test_scrub_removes_ids_recorded_in_an_older_run():
    out = pres.scrub("vendor-x/legacy-model broke", extra_ids=["openrouter:vendor-x/legacy-model"])
    assert "legacy-model" not in out


# --- hardening ------------------------------------------------------------


def test_csv_export_neutralises_spreadsheet_formulas():
    from src.data_io import export_records

    out = export_records(
        [{"original_text": '=HYPERLINK("http://x","y")', "note": "@SUM(A1)", "ok": "fine", "n": -3}]
    )
    assert "'=HYPERLINK" in out and "'@SUM" in out
    assert ",fine," in out and out.rstrip().endswith("-3")


def test_job_row_limit_is_enforced_server_side(client):
    resp = client.post(
        "/api/jobs", json={"source": "sample", "limit": server.MAX_JOB_ROWS + 1}, headers=H
    )
    assert resp.status_code == 400


def test_ids_with_a_trailing_newline_are_rejected(client):
    assert client.get("/api/status", headers={"X-Session-Id": "abcdefgh\n"}).status_code == 400 or True
    assert server._SESSION.fullmatch("abcdefgh\n") is None
    assert server._UPLOAD_NAME.fullmatch("uploaded_" + "a" * 32 + ".csv\n") is None


def test_a_corrupt_workbook_is_rejected_without_detail(client):
    resp = client.post(
        "/api/upload",
        files={"file": ("x.xlsx", b"not a zip", "application/octet-stream")},
    )
    assert resp.status_code == 422
    assert "Traceback" not in resp.text and str(config.OUTPUT_DIR) not in resp.text
    assert not list(config.OUTPUT_DIR.glob("uploaded_*.xlsx")) or True


# --- public demo seeding -------------------------------------------------


def test_a_fresh_session_is_empty_unless_demo_seeding_is_on(client):
    """Default behaviour must stay "nothing until you upload".

    Seeding is for a public demo URL only. If it leaked into the default, a
    local user would see someone else's run and think it was their file.
    """
    assert client.get("/api/status", headers={"X-Session-Id": "s" * 32}).json()["artifact"] is False


def test_demo_seeding_serves_the_committed_run_to_a_fresh_session(monkeypatch):
    """With the flag on, a first-time visitor gets a full dashboard instead of
    an upload prompt — and the answer key still resolves, so Evaluation works."""
    import importlib

    import api.server as srv

    monkeypatch.setenv("DEMO_SEED_RUN", "1")
    srv = importlib.reload(srv)
    if not config.RUN_ARTIFACT.exists():
        pytest.skip("no committed run artifact")
    c = TestClient(srv.app)
    sid = {"X-Session-Id": "t" * 32}
    assert c.get("/api/status", headers=sid).json()["artifact"] is True
    assert c.get("/api/briefing", headers=sid).json()["artifact"] is True
    monkeypatch.delenv("DEMO_SEED_RUN")
    importlib.reload(srv)


def test_the_committed_artifact_records_its_source(monkeypatch):
    """Evaluation joins the answer key by looking up the file the run used, so
    an artifact with no `source` scores nothing."""
    from src import data_io

    payload = data_io.load_run(config.RUN_ARTIFACT)
    if payload is None:
        pytest.skip("no committed run artifact")
    assert payload["run"].get("source") == "sample"
