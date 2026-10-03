"""Taxonomy and validated model-output schemas.

DETERMINISTIC MODULE — no model calls happen here. This module is the single
source of truth for the classification taxonomy (spec section 9). Both the
pydantic validators AND the prompt text in `prompts.py` are derived from the
`TAXONOMY` dict below; nothing restates the taxonomy in prose. If the two were
maintained separately they would drift, and the model would be told about
primary/sub-reason pairs the validator rejects.

Spec section 31 requires that "no unsupported label can silently enter the
dataset". That is enforced two ways:
  1. `attribute_schema_for(primary)` builds a per-primary schema whose
     sub_reason field is a Literal of only the sub-reasons legal for that
     primary, so the provider's own schema enforcement rejects bad pairs.
  2. pydantic validates again on the way in, because JSON Schema cannot
     express numeric bounds (see ConfidenceFloat below).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field, create_model


class PrimaryReason(str, Enum):
    """Spec section 9.1. Exhaustive and closed."""

    FIT = "fit"
    QUALITY_ISSUE = "quality_issue"
    APPEARANCE_ISSUE = "appearance_issue"
    WRONG_ITEM = "wrong_item"
    DAMAGED_DEFECTIVE = "damaged_defective"
    CHANGED_MIND = "changed_mind"
    DELIVERY_ISSUE = "delivery_issue"
    PRODUCT_NOT_AS_EXPECTED = "product_not_as_expected"
    UNCLEAR = "unclear"


class BodyArea(str, Enum):
    """Spec section 9.3. `NONE` is used when no body area is stated."""

    CHEST = "chest"
    WAIST = "waist"
    HIPS = "hips"
    SHOULDER = "shoulder"
    SLEEVES = "sleeves"
    LENGTH = "length"
    THIGH = "thigh"
    NECK = "neck"
    NONE = "none"


#: primary reason -> the sub-reasons legal beneath it (spec section 9.2).
#: `comfort` deliberately appears under two primaries; that is per the spec.
TAXONOMY: dict[PrimaryReason, tuple[str, ...]] = {
    PrimaryReason.FIT: (
        "too_small",
        "too_large",
        "too_long",
        "too_short",
        "unspecified_fit",
    ),
    PrimaryReason.QUALITY_ISSUE: (
        "fabric_quality",
        "material_mismatch",
        "comfort",
        "stitching_quality",
        "other_quality",
    ),
    PrimaryReason.APPEARANCE_ISSUE: (
        "colour_mismatch",
        "shade_mismatch",
        "print_mismatch",
        "other_appearance",
    ),
    PrimaryReason.WRONG_ITEM: ("wrong_product_or_variant",),
    PrimaryReason.DAMAGED_DEFECTIVE: (
        "stitching",
        "zip",
        "hole",
        "button",
        "general_defect",
    ),
    PrimaryReason.CHANGED_MIND: ("preference_or_no_longer_needed",),
    PrimaryReason.DELIVERY_ISSUE: (
        "late_delivery",
        "package_damage",
        "other_delivery",
    ),
    PrimaryReason.PRODUCT_NOT_AS_EXPECTED: (
        "style_mismatch",
        "transparency",
        "occasion_unsuitable",
        "comfort",
        "shrinkage_concern",
        "other_expectation_mismatch",
    ),
    PrimaryReason.UNCLEAR: ("insufficient_information",),
}

#: Sub-reason spellings that differ between the spec taxonomy and the hidden
#: ground-truth column, normalised at the evaluation boundary only (premise P3).
#:
#: The dataset's `expected_sub_reason` uses `defect`; spec section 9.2 defines
#: `general_defect` for the same concept. Without this mapping every
#: damaged_defective row scores as wrong and sub-reason accuracy is silently
#: depressed, with no test failing. `normalise_sub_reason` is applied to BOTH
#: sides of every comparison — applying it to one side only reintroduces the
#: bug in a harder-to-see form.
SUB_REASON_ALIASES: dict[str, str] = {
    "defect": "general_defect",
}

#: Body areas that appear in the spec taxonomy but never in the ground truth.
#: Legal for a model to emit; simply unattested in this dataset.
UNATTESTED_BODY_AREAS: frozenset[str] = frozenset({"hips", "thigh", "neck"})

REVIEW_REASONS: tuple[str, ...] = (
    "LOW_CONFIDENCE",
    "AMBIGUOUS_TEXT",
    "MULTIPLE_REASONS",
    "SCHEMA_FAILURE",
    "NO_USEFUL_TEXT",
    "MODEL_DISAGREEMENT",
    "UNSUPPORTED_OUTPUT",
)

#: Terminal row states (spec section 23). Every eligible input row ends in
#: exactly one of these; `metrics.py` asserts the counts sum to the input size.
ROW_STATES: tuple[str, ...] = (
    "CLASSIFIED_MODEL_A",
    "CLASSIFIED_MODEL_B",
    "NEEDS_HUMAN_REVIEW",
    "PROCESSING_ERROR",
    "INVALID_INPUT",
)


def normalise_sub_reason(value: str | None) -> str | None:
    """Map a sub-reason onto its canonical spelling.

    Applied to predictions and ground truth alike at the single comparison
    site in `metrics.py`. Returns the input unchanged when no alias applies.
    """
    if value is None:
        return None
    return SUB_REASON_ALIASES.get(value, value)


def all_sub_reasons() -> tuple[str, ...]:
    """Every sub-reason legal under any primary, de-duplicated, sorted."""
    seen: set[str] = set()
    for subs in TAXONOMY.values():
        seen.update(subs)
    return tuple(sorted(seen))


def is_legal_pair(primary: str, sub_reason: str) -> bool:
    """True when `sub_reason` is legal beneath `primary`."""
    try:
        key = PrimaryReason(primary)
    except ValueError:
        return False
    return sub_reason in TAXONOMY[key]


# --- Model output schemas -------------------------------------------------
#
# NOTE on `confidence`: JSON Schema numeric constraints (minimum/maximum) are
# NOT supported by provider structured-output implementations, so the 0.0-1.0
# bound cannot be enforced server-side. pydantic enforces it on the way in.
# This is why the design keeps a validating layer behind provider enforcement
# rather than trusting the provider alone (decision D4).
ConfidenceFloat = Annotated[
    float,
    Field(
        ge=0.0,
        le=1.0,
        description="Model's own confidence, 0.0 to 1.0. A routing signal, "
        "not a calibrated probability.",
    ),
]


class PrimaryClassification(BaseModel):
    """Chain step A output (spec section 12, Step A)."""

    primary_reason: PrimaryReason = Field(
        description="The single best-fitting primary return reason."
    )
    confidence: ConfidenceFloat
    needs_review: bool = Field(
        description="True when the comment is ambiguous, contradictory, "
        "carries multiple incompatible reasons, or contains no usable "
        "information. Prefer true over forcing a weak label."
    )
    explanation: str = Field(
        max_length=400,
        description="Short internal rationale. Not customer-facing.",
    )


class AttributeExtraction(BaseModel):
    """Chain step B output (spec section 12, Step B).

    Prefer `attribute_schema_for(primary)` over this permissive base: it
    narrows `sub_reason` to the primary's legal values so the provider rejects
    an illegal pair before pydantic ever sees it.
    """

    sub_reason: str = Field(description="Sub-reason legal for the primary reason.")
    body_area: BodyArea = Field(
        description="Body area, only where stated or strongly implied by the "
        "customer. Use 'none' otherwise. Never infer an unstated body area."
    )
    evidence_text: str = Field(
        max_length=300,
        description="A short verbatim excerpt from the customer's own comment "
        "supporting this classification. Must be copied from the input. Use an "
        "empty string when nothing in the comment supports it. Never invent.",
    )
    confidence: ConfidenceFloat
    needs_review: bool = Field(
        description="True when the sub-reason or body area cannot be "
        "determined from the comment."
    )
    explanation: str = Field(max_length=400, description="Short internal rationale.")


_ATTRIBUTE_SCHEMA_CACHE: dict[PrimaryReason, type[BaseModel]] = {}


def attribute_schema_for(primary: PrimaryReason | str) -> type[BaseModel]:
    """Build (and cache) a step-B schema constrained to one primary's sub-reasons.

    The returned model's `sub_reason` is a `Literal` of exactly the sub-reasons
    legal beneath `primary`, so an illegal pair is structurally unrepresentable
    rather than caught after the fact.
    """
    key = PrimaryReason(primary) if isinstance(primary, str) else primary
    cached = _ATTRIBUTE_SCHEMA_CACHE.get(key)
    if cached is not None:
        return cached

    allowed = TAXONOMY[key]
    sub_field = (
        Literal[allowed],  # type: ignore[valid-type]
        Field(
            description=f"Sub-reason for {key.value}. One of: {', '.join(allowed)}."
        ),
    )
    model = create_model(
        f"AttributeExtraction_{key.name}",
        __base__=AttributeExtraction,
        sub_reason=sub_field,
    )
    _ATTRIBUTE_SCHEMA_CACHE[key] = model
    return model


def sub_reasons_in_schema(model: type[BaseModel]) -> tuple[str, ...]:
    """Read back the sub-reasons a generated step-B schema actually permits.

    Used by the test suite to assert the generated schema matches TAXONOMY,
    which is what keeps prompt text and validator from drifting apart.
    """
    annotation = model.model_fields["sub_reason"].annotation
    args = get_args(annotation)
    return tuple(args) if args else ()
