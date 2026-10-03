"""The briefing is the screen a category head makes decisions from.

Its whole value is that it states a conclusion rather than handing over a
table, which means a wrong conclusion is worse here than anywhere else in the
app. These tests pin the three ways it could mislead: inventing a finding from
a handful of returns, saying the same thing twice under two labels, and
quoting a comment that does not support the headline it sits under.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.server import app, run_path
from src import config, data_io, metrics


def _frame(rows: list[dict]) -> pd.DataFrame:
    return metrics.results_frame(rows)


def _row(rid, sku, vendor, primary, sub, area="none", text="tight hai", status="CLASSIFIED_MODEL_A"):
    return {
        "return_id": rid,
        "sku": sku,
        "product_name": f"{sku} product",
        "category": "Womenswear",
        "vendor": vendor,
        "size_ordered": "M",
        "original_text": text,
        "primary_reason": primary,
        "sub_reason": sub,
        "body_area": area,
        "confidence": 0.9,
        "evidence_text": "",
        "status": status,
        "review_reason": None,
        "trace": [],
    }


# --- what counts as a finding -------------------------------------------


def test_a_handful_of_returns_is_not_a_finding():
    """Three returns at 100% is noise. The interval has to clear the baseline.

    Without this, the briefing would open with a confident sentence about a
    style that happened to be returned twice.
    """
    rows = [_row(f"R{i:03d}", "TINY", "V1", "fit", "too_small") for i in range(3)]
    rows += [_row(f"S{i:03d}", "BIG", "V2", "quality_issue", "fabric_quality") for i in range(60)]
    findings = metrics.headline_findings(_frame(rows), min_rows=10)
    assert not [f for f in findings if f["group"] == "TINY"]


def test_a_real_concentration_is_a_finding_and_carries_its_arithmetic():
    rows = [_row(f"K{i:03d}", "KRT", "V1", "fit", "too_small", "chest") for i in range(40)]
    rows += [_row(f"K9{i:02d}", "KRT", "V1", "quality_issue", "fabric_quality") for i in range(5)]
    rows += [
        _row(f"O{i:03d}", "OTH", f"V{2 + i % 3}", "quality_issue", "fabric_quality") for i in range(60)
    ]
    findings = metrics.headline_findings(_frame(rows))
    krt = next(f for f in findings if f["group"] == "KRT")
    assert krt["reason"] == "fit"
    assert krt["matching_rows"] == 40
    assert krt["classified_rows"] == 45
    assert krt["share_pct"] > krt["baseline_pct"]
    assert krt["excess_rows"] > 0
    assert krt["top_detail"] == "too_small"
    assert krt["top_body_area"] == "chest"


def test_the_same_returns_are_not_reported_twice_under_two_labels():
    """A single-style supplier produces the identical finding as a SKU and as a
    vendor. Saying it twice fills the briefing without adding information."""
    rows = [_row(f"K{i:03d}", "SOLO", "VSOLO", "fit", "too_small", "chest") for i in range(40)]
    rows += [_row(f"O{i:03d}", "OTH", "VMANY", "quality_issue", "fabric_quality") for i in range(60)]
    findings = metrics.headline_findings(_frame(rows))
    groups = [f["group"] for f in findings]
    assert not ("SOLO" in groups and "VSOLO" in groups), groups


def test_quotes_support_the_headline_they_sit_under():
    """The headline names the dominant body area, so the quotes must come from
    rows about that area, not from whichever row sorted first."""
    rows = [_row(f"C{i:03d}", "KRT", "V1", "fit", "too_small", "chest", f"chest tight {i}") for i in range(30)]
    rows.insert(0, _row("Z000", "KRT", "V1", "fit", "too_long", "length", "length lamba hai"))
    rows += [_row(f"O{i:03d}", "OTH", "V2", "quality_issue", "fabric_quality") for i in range(60)]
    finding = next(f for f in metrics.headline_findings(_frame(rows)) if f["group"] == "KRT")
    assert finding["top_body_area"] == "chest"
    assert all("chest" in q["text"] for q in finding["quotes"]), finding["quotes"]


def test_identical_comments_are_quoted_once():
    """Three rows reading the same sentence are one quote and two repeats."""
    rows = [_row(f"C{i:03d}", "KRT", "V1", "fit", "too_small", "chest", "chest bahut tight tha") for i in range(30)]
    rows += [_row(f"O{i:03d}", "OTH", "V2", "quality_issue", "fabric_quality") for i in range(60)]
    finding = next(f for f in metrics.headline_findings(_frame(rows)) if f["group"] == "KRT")
    assert len(finding["quotes"]) == 1


def test_no_findings_when_nothing_concentrates():
    """An even spread must produce an empty list, not the least-bad group.

    The screen says "nothing stands out" in that case, which is a useful
    answer. Promoting the top row of a flat distribution would not be.
    """
    rows = []
    for g in range(6):
        for i in range(20):
            primary = "fit" if i % 2 else "quality_issue"
            rows.append(_row(f"G{g}{i:03d}", f"SKU{g}", f"V{g}", primary, "too_small"))
    assert metrics.headline_findings(_frame(rows)) == []


def test_unclear_is_never_a_finding():
    """"This style gets a lot of unsorted comments" is a queue fact, not a
    merchandising one, and it belongs in the review queue instead."""
    rows = [_row(f"U{i:03d}", "VAGUE", "V1", "unclear", "insufficient_information") for i in range(40)]
    rows += [_row(f"O{i:03d}", "OTH", "V2", "fit", "too_small") for i in range(60)]
    assert all(f["reason"] != "unclear" for f in metrics.headline_findings(_frame(rows)))


# --- the queue summary ---------------------------------------------------


def test_queue_summary_counts_reasons_and_picks_a_readable_example():
    rows = [_row(f"A{i:03d}", "S", "V", "unclear", "insufficient_information", text="not good", status="NEEDS_HUMAN_REVIEW") for i in range(3)]
    rows[0]["review_reason"] = "AMBIGUOUS_TEXT"
    rows[1]["review_reason"] = "AMBIGUOUS_TEXT"
    rows[2]["review_reason"] = "LOW_CONFIDENCE"
    rows[2]["original_text"] = "no"
    summary = metrics.queue_summary(_frame(rows))
    assert summary["rows"] == 3
    assert summary["by_reason"][0] == {"reason": "AMBIGUOUS_TEXT", "rows": 2}
    # "no" is shorter but reads as nothing; prefer the shortest real phrase.
    assert summary["example"] == "not good"
    assert summary["minutes"] == round(3 * config.REVIEW_MINUTES_PER_ROW)


def test_queue_summary_on_an_empty_queue():
    rows = [_row("A001", "S", "V", "fit", "too_small")]
    assert metrics.queue_summary(_frame(rows))["rows"] == 0


# --- the endpoint --------------------------------------------------------


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    """A session whose run is the committed artifact."""
    client = TestClient(app)
    sid = "c" * 32
    payload = data_io.load_run(config.RUN_ARTIFACT)
    if payload is None:
        pytest.skip("no committed run artifact")
    payload.setdefault("run", {})["source"] = "sample"
    path = run_path(sid)
    path.parent.mkdir(parents=True, exist_ok=True)
    data_io.save_run(payload, path)
    yield client, sid
    path.unlink(missing_ok=True)


def test_briefing_endpoint_shape(seeded):
    client, sid = seeded
    body = client.get("/api/briefing", headers={"X-Session-Id": sid}).json()
    assert body["artifact"] is True
    assert body["counts"]["total_other"] > 0
    assert isinstance(body["findings"], list)
    assert body["queue"]["rows"] >= 0
    assert "run_total_usd" in body["cost"]


def test_briefing_names_no_model(seeded):
    client, sid = seeded
    raw = client.get("/api/briefing", headers={"X-Session-Id": sid}).text.lower()
    for token in ("mistral", "openai", "gpt", "anthropic", "claude", "openrouter"):
        assert token not in raw, token


def test_briefing_uses_public_review_reasons(seeded):
    client, sid = seeded
    body = client.get("/api/briefing", headers={"X-Session-Id": sid}).json()
    assert "MODEL_DISAGREEMENT" not in {r["reason"] for r in body["queue"]["by_reason"]}


def test_briefing_without_a_run_is_an_empty_state(seeded):
    client, _ = seeded
    body = client.get("/api/briefing", headers={"X-Session-Id": "d" * 32}).json()
    assert body == {"artifact": False}
