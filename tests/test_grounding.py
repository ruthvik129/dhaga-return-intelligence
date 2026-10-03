"""Grounding: the model may not invent evidence, and may not invent a body area.

These two rules are the whole reason a merchandiser would trust a drill-down
(spec sections 9.3 and 10.1). They are also the two easiest things for an LLM
to get subtly wrong — a paraphrase that reads like a quote, a `chest` on a
comment that never mentions a body part — and neither failure throws an
exception, so without these assertions both would ship silently.

Two layers here:

1. Unit tests of `evidence_is_grounded`, including the cases it must NOT flag
   (ordinary case/whitespace normalisation) and the one it must (paraphrase).
2. Invariants re-checked against the committed run artifact, so the claim
   "evidence is verbatim" is measured on all 440 real rows rather than asserted
   about a handcrafted string.
"""

import json

import pytest

from src.config import RUN_ARTIFACT
from src.config import DEFAULT_DATASET
from src.data_io import load_dataset
from src.models import evidence_is_grounded
from src.schemas import BodyArea, PrimaryReason, attribute_schema_for

COMMENT = "Kurta ka chest area bahut tight tha, waapas kar rahi hoon"


# --- unit: what counts as grounded ---------------------------------------


def test_verbatim_span_is_grounded():
    assert evidence_is_grounded("chest area bahut tight", COMMENT)


def test_case_and_whitespace_differences_are_not_fabrication():
    """Normalisation is not invention.

    Providers routinely return `Chest  Area Bahut Tight`. Treating that as
    ungrounded would fill the review queue with false positives and train the
    reviewer to ignore the flag.
    """
    assert evidence_is_grounded("Chest  Area   Bahut Tight", COMMENT)


def test_empty_evidence_is_grounded():
    """An empty string is the model correctly declining to claim support."""
    assert evidence_is_grounded("", COMMENT)
    assert evidence_is_grounded("   ", COMMENT)


def test_paraphrase_is_not_grounded():
    """The failure this exists to catch: plausible, fluent, not in the input."""
    assert not evidence_is_grounded("the chest was too tight for her", COMMENT)


def test_invented_detail_is_not_grounded():
    assert not evidence_is_grounded("sleeves were loose", COMMENT)


# --- unit: body area is bounded by the schema -----------------------------


def test_body_area_enum_includes_an_explicit_none():
    """`none` has to be a legal value, or the model is forced to pick a part.

    An enum of eight body parts with no escape hatch makes inventing one the
    only schema-valid answer — the constraint itself would cause the failure.
    """
    assert BodyArea.NONE.value == "none"
    assert "none" in {b.value for b in BodyArea}


@pytest.mark.parametrize("primary", list(PrimaryReason))
def test_every_primary_schema_carries_the_body_area_contract(primary):
    """Including the none-by-default instruction, on every generated schema."""
    field = attribute_schema_for(primary).model_fields["body_area"]
    assert field.annotation is BodyArea
    assert "Never infer" in (field.description or "")


# --- measured: the same rules over the real run ---------------------------


def _artifact_records():
    if not RUN_ARTIFACT.exists():
        pytest.skip(f"no run artifact at {RUN_ARTIFACT}; run run_pipeline.py")
    return json.loads(RUN_ARTIFACT.read_text(encoding="utf-8"))["records"]


def test_all_artifact_evidence_is_verbatim():
    """Re-derived, not trusted: the stored flag is recomputed from the text."""
    records = _artifact_records()
    assert records, "artifact has no records"
    ungrounded = [
        r["return_id"]
        for r in records
        if not evidence_is_grounded(r["evidence_text"] or "", r["original_text"] or "")
    ]
    assert not ungrounded, f"fabricated evidence in {len(ungrounded)} rows: {ungrounded[:5]}"


def test_stored_grounding_flag_agrees_with_recomputation():
    """A stale or mis-set flag would make the UI's grounding badge a lie."""
    for r in _artifact_records():
        recomputed = evidence_is_grounded(
            r["evidence_text"] or "", r["original_text"] or ""
        )
        assert bool(r["evidence_grounded"]) is recomputed, r["return_id"]


def test_body_area_is_none_outside_fit():
    """Body area only means something under `fit` in this taxonomy.

    A `waist` on a `late_delivery` row is not a near-miss, it is noise in the
    fit charts, which are the charts the sizing decision is made from.
    """
    offenders = [
        (r["return_id"], r["primary_reason"], r["body_area"])
        for r in _artifact_records()
        if r["primary_reason"] != PrimaryReason.FIT.value
        and r["body_area"] not in (None, "", BodyArea.NONE.value)
    ]
    assert not offenders, offenders[:5]


def test_body_area_is_never_invented_where_truth_says_none():
    """The spec section 9.3 rule, scored against ground truth.

    Only the over-claim direction is asserted. Missing a stated body area is an
    accuracy miss and belongs in the evaluation screen; inventing one is a
    trust failure and belongs in a test.
    """
    truth = load_dataset(DEFAULT_DATASET).ground_truth.set_index("return_id")
    invented = []
    for r in _artifact_records():
        if r["return_id"] not in truth.index:
            continue
        expected = str(truth.loc[r["return_id"], "expected_body_area"]).strip().lower()
        if expected in {"", "none", "nan"} and r["body_area"] not in (
            None,
            "",
            BodyArea.NONE.value,
        ):
            invented.append((r["return_id"], r["body_area"], r["original_text"]))
    assert not invented, f"{len(invented)} invented body areas: {invented[:3]}"
