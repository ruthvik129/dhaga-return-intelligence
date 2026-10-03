"""Taxonomy integrity, every routing branch, and the schema-repair path.

Spec section 31: "No unsupported label can silently enter the dataset."
The mechanism is that prompt text and validator are both generated from the
same TAXONOMY dict, so these tests assert they agree rather than trusting it.
"""

import types

import pytest

from src import models, prompts
from src.config import ACCEPT_THRESHOLD_A, ACCEPT_THRESHOLD_B
from src.router import (
    Route,
    StepOutcome,
    review_budget_minutes,
    route_after_a,
    route_after_b,
    route_no_text,
    terminal_state,
)
from src.schemas import (
    TAXONOMY,
    BodyArea,
    PrimaryReason,
    attribute_schema_for,
    is_legal_pair,
    sub_reasons_in_schema,
)


# --- taxonomy ------------------------------------------------------------


def test_every_primary_has_at_least_one_sub_reason():
    for primary in PrimaryReason:
        assert TAXONOMY[primary], primary


def test_generated_schema_permits_exactly_the_legal_sub_reasons():
    """The schema is the enforcement point, so it must match TAXONOMY exactly."""
    for primary, expected in TAXONOMY.items():
        schema = attribute_schema_for(primary)
        assert set(sub_reasons_in_schema(schema)) == set(expected), primary


def test_illegal_pair_is_structurally_rejected():
    schema = attribute_schema_for(PrimaryReason.FIT)
    with pytest.raises(Exception):
        schema(
            sub_reason="fabric_quality",   # legal for quality_issue, not fit
            body_area=BodyArea.NONE,
            evidence_text="",
            confidence=0.9,
            needs_review=False,
            explanation="x",
        )


def test_legal_pair_validates():
    schema = attribute_schema_for(PrimaryReason.FIT)
    obj = schema(
        sub_reason="too_small",
        body_area=BodyArea.CHEST,
        evidence_text="chest tight",
        confidence=0.9,
        needs_review=False,
        explanation="x",
    )
    assert obj.sub_reason == "too_small"


def test_confidence_bound_is_enforced_by_pydantic():
    """JSON Schema cannot express numeric bounds, so this layer is load-bearing
    rather than belt-and-braces (decision D4)."""
    schema = attribute_schema_for(PrimaryReason.FIT)
    for bad in (1.5, -0.1):
        with pytest.raises(Exception):
            schema(
                sub_reason="too_small",
                body_area=BodyArea.NONE,
                evidence_text="",
                confidence=bad,
                needs_review=False,
                explanation="x",
            )


def test_prompt_text_lists_only_legal_sub_reasons():
    """Prompt and validator are generated from one source; prove they agree."""
    for primary, allowed in TAXONOMY.items():
        text = prompts.step_b_system(primary)
        for sub in allowed:
            assert sub in text, (primary, sub)
        illegal = {s for p, subs in TAXONOMY.items() if p is not primary
                   for s in subs} - set(allowed)
        for sub in illegal:
            assert f"- {sub}\n" not in text, (primary, sub)


def test_is_legal_pair_rejects_unknown_primary():
    assert not is_legal_pair("not_a_reason", "too_small")


# --- routing after Model A -----------------------------------------------


def test_a_accepts_confident_unflagged_row():
    d = route_after_a(StepOutcome(True, "fit", "too_small", 0.93, False))
    assert d.route is Route.ACCEPT and d.reason is None


def test_a_escalates_below_threshold():
    below = ACCEPT_THRESHOLD_A - 0.01
    d = route_after_a(StepOutcome(True, "fit", "too_small", below, False))
    assert d.route is Route.ESCALATE and d.reason == "LOW_CONFIDENCE"


def test_a_accepts_exactly_at_threshold():
    """Boundary: the spec says `>=`, so the threshold value itself accepts."""
    d = route_after_a(StepOutcome(True, "fit", "too_small", ACCEPT_THRESHOLD_A, False))
    assert d.route is Route.ACCEPT


def test_a_escalates_when_flagged_even_at_high_confidence():
    d = route_after_a(StepOutcome(True, "unclear", None, 1.0, True))
    assert d.route is Route.ESCALATE and d.reason == "AMBIGUOUS_TEXT"


def test_a_escalates_on_schema_failure():
    d = route_after_a(StepOutcome(False, error="bad json"))
    assert d.route is Route.ESCALATE and d.reason == "SCHEMA_FAILURE"


def test_a_escalates_on_unsupported_pair():
    d = route_after_a(StepOutcome(True, "fit", "fabric_quality", 0.99, False))
    assert d.route is Route.ESCALATE and d.reason == "UNSUPPORTED_OUTPUT"


def test_threshold_is_configurable_not_hardcoded():
    outcome = StepOutcome(True, "fit", "too_small", 0.50, False)
    assert route_after_a(outcome, threshold=0.40).route is Route.ACCEPT
    assert route_after_a(outcome, threshold=0.90).route is Route.ESCALATE


# --- routing after Model B -----------------------------------------------


def _a_low():
    return StepOutcome(True, "fit", "too_small", 0.40, False)


def test_b_accepts_when_confident_and_agreeing():
    d = route_after_b(StepOutcome(True, "fit", "too_small", 0.9, False), _a_low())
    assert d.route is Route.ACCEPT


def test_b_sends_disagreement_to_human():
    d = route_after_b(
        StepOutcome(True, "quality_issue", "comfort", 0.95, False), _a_low()
    )
    assert d.route is Route.HUMAN and d.reason == "MODEL_DISAGREEMENT"


def test_b_sends_still_ambiguous_to_human():
    d = route_after_b(StepOutcome(True, "unclear", None, 0.9, True), _a_low())
    assert d.route is Route.HUMAN and d.reason == "AMBIGUOUS_TEXT"


def test_b_sends_low_confidence_to_human():
    below = ACCEPT_THRESHOLD_B - 0.01
    d = route_after_b(StepOutcome(True, "fit", "too_small", below, False), _a_low())
    assert d.route is Route.HUMAN and d.reason == "LOW_CONFIDENCE"


def test_b_schema_failure_goes_to_human_not_another_escalation():
    d = route_after_b(StepOutcome(False, error="bad"), _a_low())
    assert d.route is Route.HUMAN and d.reason == "SCHEMA_FAILURE"


def test_empty_text_never_reaches_a_model():
    d = route_no_text()
    assert d.route is Route.HUMAN and d.reason == "NO_USEFUL_TEXT"


def test_terminal_states_map_correctly():
    assert terminal_state(Route.ACCEPT, escalated=False) == "CLASSIFIED_MODEL_A"
    assert terminal_state(Route.ACCEPT, escalated=True) == "CLASSIFIED_MODEL_B"
    assert terminal_state(Route.HUMAN, escalated=True) == "NEEDS_HUMAN_REVIEW"


def test_review_budget_is_expressed_in_reviewer_time():
    assert review_budget_minutes(0) == 0
    assert review_budget_minutes(10) > 0


# --- schema repair path (decision D4) ------------------------------------


def _fake_classifier(step_a_responses):
    """A Classifier stand-in that needs no network and no credential."""
    calls = {"n": 0}

    def invoke(_messages):
        i = min(calls["n"], len(step_a_responses) - 1)
        calls["n"] += 1
        return step_a_responses[i]

    return types.SimpleNamespace(
        spec=types.SimpleNamespace(key="test:model"),
        ledger=models.UsageLedger(),
        _step_a=types.SimpleNamespace(invoke=invoke),
    ), calls


def _record(rid="R1", text="chest tight hai"):
    from src.data_io import ReturnRecord

    return ReturnRecord(rid, "SKU", "p", "c", "v", "M", "Other", text)


def test_repair_retry_recovers_a_bad_first_response():
    """A parse failure must be routed on as data, not raised. That is the whole
    reason the model layer uses include_raw=True."""
    good = types.SimpleNamespace(
        primary_reason=PrimaryReason.FIT, confidence=0.9, needs_review=False
    )
    # The responses list feeds the RETRY calls only; the first bad response is
    # handed in directly as the already-received step-A result below.
    fake, calls = _fake_classifier([
        {"parsed": good, "parsing_error": None, "raw": None},
    ])
    result = models.Classifier._finish_step_a(
        fake, _record(), {"parsed": None, "parsing_error": "bad json", "raw": None}, []
    )
    assert result.repaired is True
    assert result.outcome.schema_ok is True
    assert result.outcome.primary == "fit"
    assert fake.ledger.for_model(fake.spec).schema_repairs == 1
    assert fake.ledger.for_model(fake.spec).schema_failures == 0


def test_persistent_schema_failure_is_reported_not_raised():
    bad = {"parsed": None, "parsing_error": "still bad", "raw": None}
    fake, _ = _fake_classifier([bad, bad])
    result = models.Classifier._finish_step_a(fake, _record(), bad, [])
    assert result.outcome.schema_ok is False
    assert fake.ledger.for_model(fake.spec).schema_failures == 1
    # and the router turns that into an escalation rather than a crash
    assert route_after_a(result.outcome).reason == "SCHEMA_FAILURE"


def test_transport_exception_becomes_a_row_state_not_a_crash():
    """return_exceptions=True means a failing row is a value. Spec section 23:
    never silently drop a record — and never lose the whole batch either."""
    fake, _ = _fake_classifier([{"parsed": None, "parsing_error": "x", "raw": None}])
    result = models.Classifier._finish_step_a(
        fake, _record(), RuntimeError("connection reset"), []
    )
    assert result.outcome.schema_ok is False
    assert "RuntimeError" in result.outcome.error
    assert fake.ledger.for_model(fake.spec).errors == 1


def test_fault_injection_is_off_unless_explicitly_enabled():
    from src import config

    assert config.INJECT_SCHEMA_FAULTS is False, (
        "fault injection must never be on by default"
    )
    assert models._injected_fault("R0001", attempt=1) is False


def test_fault_injection_when_enabled_is_deterministic(monkeypatch):
    """Needed so the SCHEMA_FAILURE path and its metric are reachable at all —
    provider-side enforcement makes organic failures near-zero."""
    monkeypatch.setattr(models.config, "INJECT_SCHEMA_FAULTS", True)
    ids = [f"R{i:04d}" for i in range(400)]
    first = [models._injected_fault(i, attempt=1) for i in ids]
    assert any(first), "some rows must fail their first attempt"
    assert not all(first), "not every row may fail"
    # deterministic across calls
    assert first == [models._injected_fault(i, attempt=1) for i in ids]
    assert any(models._injected_fault(i, attempt=2) for i in ids)
