"""compare_models.py, without a network.

The scoring and the table are checked against the committed run artifact,
whose numbers are already known from the README (all 440 sample rows), so a
change in how a candidate is scored shows up here as a changed number.
"""

import json

import pytest

import compare_models as cm
from src import config
from src.data_io import load_dataset


@pytest.fixture(scope="module")
def committed():
    payload = json.loads(config.RUN_ARTIFACT.read_text(encoding="utf-8"))
    truth = load_dataset(config.DEFAULT_DATASET).ground_truth
    return payload, truth


def _cand(first="vendor/first-model", second="vendor/second-model"):
    return cm.Candidate(cm.spec_for(first, second=False), cm.spec_for(second, second=True))


def test_summary_matches_the_readme_numbers(committed):
    payload, truth = committed
    row = cm.summarise(payload, truth)
    assert row["rows"] == 440
    assert row["auto_accept_accuracy_pct"] == 94.0
    assert row["auto_accept_n"] == 416
    assert row["overall_accuracy_pct"] == 93.0
    assert row["sub_reason_accuracy_pct"] == 73.6
    assert row["second_check_pct"] == 5.5
    assert row["review_pct"] == 5.5
    assert row["cost_per_1000_usd"] == pytest.approx(0.0247)


def test_without_an_answer_key_accuracy_is_not_available(committed):
    payload, truth = committed
    row = cm.summarise(payload, truth.iloc[0:0])
    assert row["auto_accept_accuracy_pct"] is None
    assert row["cost_per_1000_usd"] is not None  # cost does not need a key


def test_recommendation_is_the_cheapest_that_clears_the_bar():
    cheap_bad = cm.Result(_cand("a/cheap"), row={"auto_accept_accuracy_pct": 90.0, "cost_per_1000_usd": 0.01})
    mid_good = cm.Result(_cand("b/mid"), row={"auto_accept_accuracy_pct": 96.0, "cost_per_1000_usd": 0.02})
    dear_good = cm.Result(_cand("c/dear"), row={"auto_accept_accuracy_pct": 98.0, "cost_per_1000_usd": 0.20})
    text = cm.recommend([cheap_bad, mid_good, dear_good], target_pct=95)
    assert "b/mid" in text and "a/cheap" not in text


def test_recommendation_says_so_when_nothing_clears_the_bar():
    r = cm.Result(_cand(), row={"auto_accept_accuracy_pct": 80.0, "cost_per_1000_usd": 0.01})
    assert cm.recommend([r], target_pct=95).startswith("No candidate reached")


def test_recommendation_without_an_answer_key():
    r = cm.Result(_cand(), row={"auto_accept_accuracy_pct": None, "cost_per_1000_usd": 0.01})
    assert cm.recommend([r], target_pct=95).startswith("No answer key")


def test_table_names_models_and_shows_failures(committed):
    payload, truth = committed
    ok = cm.Result(_cand("mistralai/mistral-nemo", "openai/gpt-oss-120b"), row=cm.summarise(payload, truth))
    bad = cm.Result(_cand("vendor/broken"), error="RuntimeError: provider down")
    table = cm.render_table([ok, bad])
    assert "mistralai/mistral-nemo → openai/gpt-oss-120b" in table
    assert "94.0%" in table and "$0.0247" in table
    assert "FAILED: RuntimeError: provider down" in table


def test_csv_has_one_row_per_candidate(tmp_path, committed):
    payload, truth = committed
    results = [cm.Result(_cand(), row=cm.summarise(payload, truth)), cm.Result(_cand("x/y"), error="boom")]
    path = tmp_path / "c.csv"
    cm.write_csv(results, path)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("first_pass,second_check,error")
    assert len(lines) == 3


def test_unknown_and_incapable_models_are_refused_before_spending():
    problems = cm.check_candidates(["no-such/model-xyz"])
    assert problems and "not in the OpenRouter price snapshot" in problems[0]


def test_routers_are_flagged_as_unpriced():
    known = cm.snapshot_models()
    if "typesafe/jev-router" not in known:
        pytest.skip("router not in this snapshot")
    assert any("variable pricing" in w for w in cm.warnings_for(["typesafe/jev-router"]))


def test_dataset_and_first_pass_are_required():
    with pytest.raises(SystemExit):
        cm.parse_args(["--first-pass", "a/b"])
    with pytest.raises(SystemExit):
        cm.parse_args(["--dataset", "x.csv"])
    args = cm.parse_args(["--dataset", "x.csv", "--first-pass", "a/b", "c/d"])
    assert args.first_pass == ["a/b", "c/d"] and args.second_check == [config.MODEL_B.name]
