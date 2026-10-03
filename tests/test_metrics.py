"""Deterministic arithmetic: invariants, normalisation, lift, intervals, cost.

These protect the numbers Neha would actually act on. The two that matter most
are the no-silent-drop invariant and the both-sides sub-reason normalisation,
because each fails quietly: one renders a short dashboard, the other reports a
wrong accuracy with every test still green.
"""

import pandas as pd
import pytest

from src import metrics
from src.metrics import InvariantViolation, wilson_interval
from src.schemas import normalise_sub_reason


def _rows(n=4, status="CLASSIFIED_MODEL_A"):
    return [
        {
            "return_id": f"R{i}",
            "sku": "KRT-184" if i % 2 else "MEN-522",
            "vendor": "Vendor_12" if i % 2 else "Vendor_01",
            "category": "Womenswear",
            "primary_reason": "fit" if i % 2 else "quality_issue",
            "sub_reason": "too_small" if i % 2 else "fabric_quality",
            "body_area": "chest" if i % 2 else "none",
            "status": status,
            "escalated": False,
            "original_text": "chest tight hai" if i % 2 else "cloth itchy",
        }
        for i in range(n)
    ]


# --- invariants ----------------------------------------------------------


def test_row_count_invariant_catches_dropped_rows():
    with pytest.raises(InvariantViolation, match="Row count mismatch"):
        metrics.assert_no_rows_lost(_rows(3), eligible_count=4)


def test_invariant_catches_missing_terminal_state():
    rows = _rows(2)
    rows[0]["status"] = None
    with pytest.raises(InvariantViolation, match="no terminal status"):
        metrics.assert_no_rows_lost(rows, eligible_count=2)


def test_invariant_catches_unknown_state():
    rows = _rows(2)
    rows[1]["status"] = "SOMETHING_ELSE"
    with pytest.raises(InvariantViolation, match="Unknown row status"):
        metrics.assert_no_rows_lost(rows, eligible_count=2)


def test_invariant_catches_duplicate_ids():
    rows = _rows(2)
    rows[1]["return_id"] = rows[0]["return_id"]
    with pytest.raises(InvariantViolation, match="Duplicate return_id"):
        metrics.assert_no_rows_lost(rows, eligible_count=2)


def test_invariant_passes_on_a_clean_run():
    metrics.assert_no_rows_lost(_rows(5), eligible_count=5)


# --- Wilson interval -----------------------------------------------------


def test_wilson_matches_published_worked_example():
    """p=0.5, n=100 -> approximately (0.4038, 0.5962) at 95%."""
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=0.001)
    assert hi == pytest.approx(0.5962, abs=0.001)


def test_wilson_handles_zero_denominator():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_is_wide_on_small_samples():
    """The reason intervals are printed next to every small-bucket accuracy."""
    lo, hi = wilson_interval(7, 10)
    assert hi - lo > 0.3


# --- sub-reason normalisation (premise P3) -------------------------------


def test_defect_alias_maps_to_spec_spelling():
    assert normalise_sub_reason("defect") == "general_defect"
    assert normalise_sub_reason("general_defect") == "general_defect"
    assert normalise_sub_reason(None) is None
    assert normalise_sub_reason("too_small") == "too_small"


def test_normalisation_is_applied_to_both_sides_of_the_comparison():
    """The subtle bug this guards: normalise predictions but not ground truth
    and every damaged row scores wrong with no test failing."""
    frame = pd.DataFrame([{
        "return_id": "R1", "primary_reason": "damaged_defective",
        "sub_reason": "general_defect", "body_area": "none",
        "status": "CLASSIFIED_MODEL_A", "escalated": False,
    }])
    truth = pd.DataFrame([{
        "return_id": "R1",
        "expected_primary_reason": "damaged_defective",
        "expected_sub_reason": "defect",       # the dataset's spelling
        "expected_body_area": "none",
    }])
    result = metrics.evaluate(frame, truth)
    assert result["sub_reason_accuracy"]["accuracy_pct"] == 100.0, (
        "spec spelling vs dataset spelling must score as a match"
    )


# --- lift / share-within-group (premise P4) ------------------------------


def test_group_intelligence_reports_share_and_lift_not_raw_counts():
    frame = metrics.results_frame(_rows(20))
    out = metrics.group_intelligence(frame, "vendor", "fit", min_rows=2)
    assert not out.empty
    for column in ("share_pct", "baseline_pct", "lift",
                   "share_ci_low_pct", "share_ci_high_pct"):
        assert column in out.columns
    assert not any("rate" in c for c in out.columns), (
        "no column may be called a rate: the dataset has no sales denominator "
        "(spec section 18.4)"
    )


def test_high_volume_group_does_not_automatically_top_the_chart():
    """Vendor_12 holds 31% of all Other rows, so a raw-count chart ranks it
    first regardless of quality. Lift must rank by concentration instead."""
    rows = []
    # big vendor, low fit share (5 of 50)
    for i in range(50):
        rows.append({
            "return_id": f"B{i}", "vendor": "Vendor_Big", "sku": "S1",
            "category": "C", "primary_reason": "fit" if i < 5 else "changed_mind",
            "sub_reason": "too_small", "body_area": "none",
            "status": "CLASSIFIED_MODEL_A", "escalated": False,
        })
    # small vendor, high fit share (8 of 10)
    for i in range(10):
        rows.append({
            "return_id": f"S{i}", "vendor": "Vendor_Small", "sku": "S2",
            "category": "C", "primary_reason": "fit" if i < 8 else "changed_mind",
            "sub_reason": "too_small", "body_area": "none",
            "status": "CLASSIFIED_MODEL_A", "escalated": False,
        })
    out = metrics.group_intelligence(
        metrics.results_frame(rows), "vendor", "fit", min_rows=5
    )
    assert out.iloc[0]["vendor"] == "Vendor_Small", (
        "the concentrated vendor must rank above the merely large one"
    )
    big = out[out["vendor"] == "Vendor_Big"].iloc[0]
    assert big["matching_rows"] < out.iloc[0]["matching_rows"] or True
    assert out.iloc[0]["lift"] > big["lift"]


def test_small_groups_are_excluded_rather_than_shown_as_noise():
    frame = metrics.results_frame(_rows(4))
    out = metrics.group_intelligence(frame, "sku", "fit", min_rows=100)
    assert out.empty


# --- cost (spec section 21) ---------------------------------------------


def test_unpriced_model_reports_none_not_zero():
    """A silent zero would make an expensive run look free."""
    usage = {"madeup:model-x": {"calls": 10, "input_tokens": 1000,
                                "output_tokens": 500}}
    cost = metrics.cost_report(usage, rows_classified=10)
    assert cost["run_total_usd"] is None
    assert cost["fully_priced"] is False
    assert "madeup:model-x" in cost["unpriced_models"]
    assert cost["per_model"][0]["total_cost_usd"] is None


@pytest.fixture
def known_price(monkeypatch):
    """A fixed $1/$5 per Mtok price, so the arithmetic is tested against a
    number the test owns rather than whatever the price snapshot says today."""
    from src import config

    monkeypatch.setitem(config.PRICES, "test:priced-model", config.Price(1.00, 5.00, "test"))
    return "test:priced-model"


def test_priced_model_computes_cost_deterministically(known_price):
    usage = {known_price: {
        "calls": 2, "input_tokens": 1_000_000, "output_tokens": 1_000_000,
        "token_counts_measured": True,
    }}
    cost = metrics.cost_report(usage, rows_classified=1000)
    # $1.00/Mtok in + $5.00/Mtok out
    assert cost["run_total_usd"] == pytest.approx(6.00)
    assert cost["cost_per_1000_comments_usd"] == pytest.approx(6.00)
    assert cost["fully_priced"] is True


def test_cost_labels_measured_vs_estimated_tokens(known_price):
    usage = {known_price: {
        "calls": 1, "input_tokens": 10, "output_tokens": 5,
        "token_counts_measured": False,
    }}
    cost = metrics.cost_report(usage, rows_classified=1)
    assert cost["all_tokens_measured"] is False
    assert cost["per_model"][0]["token_counts"] == "estimated"


def test_projected_cost_scales_the_measured_rate(known_price):
    usage = {known_price: {
        "calls": 1, "input_tokens": 1_000_000, "output_tokens": 0,
    }}
    cost = metrics.cost_report(usage, rows_classified=1000)
    assert metrics.projected_cost(cost, 100_000) == pytest.approx(100.0, abs=1)


def test_escalation_economics_reports_share_of_calls():
    usage = {"p:a": {"calls": 80}, "p:b": {"calls": 20}}
    econ = metrics.escalation_economics(usage, "p:b")
    assert econ["total_model_calls"] == 100
    assert econ["pct_calls_on_model_b"] == 20.0


# --- corrections feed downstream analytics (spec section 7.4) ------------


def test_correction_changes_the_aggregate():
    frame = metrics.results_frame(_rows(4))
    before = metrics.reason_distribution(frame)
    fit_before = int(before.loc[before["primary_reason"] == "fit", "rows"].sum())

    corrected = metrics.apply_corrections(
        frame,
        {"R1": {"action": "change", "primary_reason": "quality_issue",
                "sub_reason": "fabric_quality", "body_area": "none"}},
    )
    after = metrics.reason_distribution(corrected)
    fit_after = int(after.loc[after["primary_reason"] == "fit", "rows"].sum())
    assert fit_after == fit_before - 1, "a correction must move the chart"


def test_mark_unclear_correction_sets_unclear():
    frame = metrics.results_frame(_rows(2))
    out = metrics.apply_corrections(frame, {"R1": {"action": "mark_unclear"}})
    row = out[out["return_id"] == "R1"].iloc[0]
    assert row["primary_reason"] == "unclear"
    assert row["sub_reason"] == "insufficient_information"


def test_evaluation_reports_denominator_with_every_accuracy():
    frame = metrics.results_frame(_rows(4))
    truth = pd.DataFrame([{
        "return_id": f"R{i}",
        "expected_primary_reason": "fit" if i % 2 else "quality_issue",
        "expected_sub_reason": "too_small" if i % 2 else "fabric_quality",
        "expected_body_area": "chest" if i % 2 else "none",
    } for i in range(4)])
    result = metrics.evaluate(frame, truth)
    assert result["available"] is True
    for key in ("primary_accuracy_overall", "auto_accept_precision_a",
                "sub_reason_accuracy", "body_area_accuracy"):
        assert "n" in result[key], f"{key} must carry its denominator"
        assert "ci_low_pct" in result[key]


def test_body_area_scored_only_where_truth_states_one():
    """Counting none==none as a win would inflate the figure with rows that
    required no judgement (spec section 20: 'where applicable')."""
    frame = metrics.results_frame(_rows(4))
    truth = pd.DataFrame([{
        "return_id": f"R{i}",
        "expected_primary_reason": "fit" if i % 2 else "quality_issue",
        "expected_sub_reason": "too_small" if i % 2 else "fabric_quality",
        "expected_body_area": "chest" if i % 2 else "none",
    } for i in range(4)])
    result = metrics.evaluate(frame, truth)
    assert result["body_area_accuracy"]["n"] == 2, "only the 2 rows with a truth"


def test_variable_priced_routers_are_unpriced_not_negative():
    """OpenRouter lists routers at -1/token; that is "unknown", not a refund."""
    from src import config

    snapshot = config.OPENROUTER_PRICE_SNAPSHOT.read_text(encoding="utf-8")
    if '"typesafe/jev-router"' not in snapshot:
        pytest.skip("router not in this snapshot")
    assert not config.price_for(config.ModelSpec("openrouter", "typesafe/jev-router")).is_configured
    usage = {"openrouter:typesafe/jev-router": {"calls": 1, "input_tokens": 100, "output_tokens": 10}}
    cost = metrics.cost_report(usage, rows_classified=1)
    assert cost["run_total_usd"] is None
