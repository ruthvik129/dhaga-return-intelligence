"""Prompt construction. All taxonomy text is GENERATED from `schemas.TAXONOMY`.

DETERMINISTIC MODULE — builds strings, makes no model calls.

Nothing here restates the taxonomy by hand. `_primary_block()` and
`_sub_reason_block()` walk the same dict the pydantic validators are built
from, so it is structurally impossible to tell the model about a
primary/sub-reason pair the validator would reject (finding C1). Add a
sub-reason to `TAXONOMY` and both the schema and the prompt pick it up.

Prompt version is stamped into the run artifact so a reviewer can tell which
wording produced which numbers.
"""

from __future__ import annotations

from .schemas import TAXONOMY, BodyArea, PrimaryReason

PROMPT_VERSION = "2026-10-02.1"

#: One-line definitions, used only as prompt prose. The authoritative list of
#: primaries is `TAXONOMY`; a key missing here degrades to the bare value
#: rather than silently dropping the category.
_PRIMARY_GLOSS: dict[PrimaryReason, str] = {
    PrimaryReason.FIT: "the garment's size or proportions were wrong for the customer",
    PrimaryReason.QUALITY_ISSUE: "the fabric, material, stitching or comfort was poor",
    PrimaryReason.APPEARANCE_ISSUE: "the colour, shade or print differed from expectation",
    PrimaryReason.WRONG_ITEM: "the wrong product or variant was delivered",
    PrimaryReason.DAMAGED_DEFECTIVE: "the item arrived damaged or with a defect",
    PrimaryReason.CHANGED_MIND: "the customer no longer wanted it; no fault with the item",
    PrimaryReason.DELIVERY_ISSUE: "something about delivery, not the product itself",
    PrimaryReason.PRODUCT_NOT_AS_EXPECTED: (
        "the item differed from expectation in a way that is not fit, quality, "
        "or colour — style, transparency, occasion suitability, shrinkage concern"
    ),
    PrimaryReason.UNCLEAR: (
        "the comment carries no usable signal about why it was returned"
    ),
}

#: Comments of this shape must NOT be forced into a confident category. These
#: examples come from spec section 17, not from the hidden ground truth.
_AMBIGUOUS_EXAMPLES = (
    "not good",
    "fit nahi",
    "different",
    "acha nahi",
    "problem hai",
    "return karna hai",
)


def _primary_block() -> str:
    lines = []
    for primary in TAXONOMY:
        gloss = _PRIMARY_GLOSS.get(primary, "")
        lines.append(f"- {primary.value}: {gloss}" if gloss else f"- {primary.value}")
    return "\n".join(lines)


def _sub_reason_block(primary: PrimaryReason) -> str:
    return "\n".join(f"- {sub}" for sub in TAXONOMY[primary])


def _body_area_list() -> str:
    return ", ".join(area.value for area in BodyArea)


STEP_A_SYSTEM = f"""\
You classify why a customer returned a clothing item, from their own free-text \
comment.

The comments come from Indian customers and mix English, Hindi written in Latin \
script (Hinglish), abbreviations, and spelling mistakes. Read them the way a \
native Hinglish speaker would. "chest se tight hai" means it was tight across \
the chest. "pasand nahi aya" means they simply did not like it. "chota hai" \
means it is small.

Choose exactly one primary reason from this list:

{_primary_block()}

Rules that matter more than picking a label:

1. If the comment does not actually say why the item was returned, set \
needs_review to true and choose `unclear`. Do not guess. Comments like \
{", ".join(repr(x) for x in _AMBIGUOUS_EXAMPLES)} carry no usable signal — they \
tell you the customer was unhappy, not what was wrong.
2. If the comment describes two or more incompatible reasons and a single label \
would misrepresent it, set needs_review to true and pick the dominant one.
3. Distinguish `product_not_as_expected` from the specific categories. Use it \
only when the mismatch is not about fit, quality, or colour. A comment about \
sizing is `fit`, not `product_not_as_expected`.
4. `changed_mind` means no fault with the item. If the customer gives any \
product fault, it is not `changed_mind`.
5. Your confidence should reflect how clearly the comment supports your label, \
not how plausible the label is in general. Short, vague comments deserve low \
confidence even when your guess is reasonable.

A returned-to-human row costs a few minutes of a category manager's time. A \
confidently wrong label costs far more, because it silently distorts the \
numbers she uses to make decisions. When in doubt, flag it.\
"""


def step_a_user(
    text: str,
    dropdown_reason: str | None = None,
    category: str | None = None,
) -> str:
    """Build the step-A user turn.

    `dropdown_reason` and `category` are optional context the spec permits
    (section 12, Step A). No ground-truth field is ever passed here — the only
    type this receives data from is `ReturnRecord`, which cannot hold one.
    """
    parts = [f"Customer comment:\n{text!r}"]
    if category:
        parts.append(f"Product category: {category}")
    if dropdown_reason:
        parts.append(
            f"Dropdown reason the customer also selected: {dropdown_reason} "
            "(treat as weak context only; it is why this row needs classifying)"
        )
    parts.append("Classify the primary return reason.")
    return "\n\n".join(parts)


def step_b_system(primary: PrimaryReason) -> str:
    """Build the step-B system prompt, scoped to one primary reason.

    Generated per primary so the model is shown only the sub-reasons legal
    beneath it — the same set the generated schema will accept.
    """
    return f"""\
A customer's return comment has already been classified as `{primary.value}`. \
Extract the supporting detail.

Choose exactly one sub-reason from this list and no other:

{_sub_reason_block(primary)}

Then determine the body area, if and only if the customer states or strongly \
implies one. Valid values: {_body_area_list()}.

Rules:

1. Use `none` for body_area unless the comment points at a specific part of the \
body. "tight hai" alone is `none`. "chest pe tight" is `chest`. Never infer a \
body area from the garment type — a kurta complaint is not automatically about \
the chest.
2. `evidence_text` must be a short span copied from the customer's comment, \
character for character. Do not paraphrase, translate, or tidy it. If nothing \
in the comment supports your choice, return an empty string rather than \
inventing support.
3. If the comment does not contain enough detail to choose a sub-reason \
confidently, set needs_review to true and pick the least specific option \
available.

The comment is in Hinglish or informal English. Read it as a native speaker \
would.\
"""


def step_b_user(text: str, primary: PrimaryReason) -> str:
    return (
        f"Customer comment:\n{text!r}\n\n"
        f"Already-determined primary reason: {primary.value}\n\n"
        "Extract the sub-reason, body area, and supporting evidence."
    )


def repair_suffix(error: str) -> str:
    """Appended on the single schema-repair retry (spec section 23)."""
    return (
        "\n\nYour previous response did not match the required schema. "
        f"The validation error was: {error}\n"
        "Return a response that matches the schema exactly. Use only the "
        "permitted values."
    )
