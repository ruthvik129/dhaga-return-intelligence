"""Deterministic keyword baseline — the control arm.

DETERMINISTIC MODULE — no model calls, no network, no API key. Runs over all
440 rows in milliseconds.

This exists to answer the first question a cost-conscious sponsor asks: *why do
you need an LLM at all?* The only honest answer is a measured one, so this is
scored against the same ground truth as the cascade and shown beside it.

It is also the no-credential fallback: with no API key configured the app still
renders real classifications from this module rather than an error page.

Two deliberate design choices:

1. **It abstains.** Returning `None` on input it cannot read is the whole point.
   A keyword matcher that guesses on `okay nahi laga` would score better on
   coverage and worse on truth, which is exactly the trade the trust spine
   argues against.
2. **Longest-match-first within a category.** `chhota` must beat `chota`, and
   `not needed` must beat `not`, or the shorter pattern shadows the specific
   one and the result depends on dict ordering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import BodyArea, PrimaryReason

#: Body-area cues, including Hinglish. Matched only to set `body_area`; the
#: spec forbids inferring a body area that is not stated (section 9.3), so
#: there is no fallback guess here.
_BODY_AREA_CUES: dict[str, tuple[str, ...]] = {
    BodyArea.CHEST.value: ("chest", "seena", "seene"),
    BodyArea.WAIST.value: ("waist", "kamar"),
    BodyArea.HIPS.value: ("hips", "hip"),
    BodyArea.SHOULDER.value: ("shoulder", "kandha", "kandhe"),
    BodyArea.SLEEVES.value: ("sleeve", "sleeves", "baju", "bazu"),
    BodyArea.LENGTH.value: ("length", "lambai"),
    BodyArea.THIGH.value: ("thigh", "thighs", "jangh"),
    BodyArea.NECK.value: ("neck", "gala", "collar"),
}

#: (primary, sub_reason, patterns). Order matters only within a category; the
#: scorer picks the longest matching pattern across all rules.
_RULES: tuple[tuple[PrimaryReason, str, tuple[str, ...]], ...] = (
    # --- fit ---
    (PrimaryReason.FIT, "too_small", (
        "bahut tight", "too tight", "very tight", "tight hai", "tight tha",
        "tight ho", "tight", "chhota", "chota", "choti", "small", "snug",
        "nahi aaya size", "size small",
    )),
    (PrimaryReason.FIT, "too_large", (
        "bahut loose", "too loose", "very loose", "loose hai", "loose",
        "dhila", "dheela", "bada hai", "bada", "badi", "oversized", "large",
        "size big", "big size",
    )),
    (PrimaryReason.FIT, "too_long", ("bahut lamba", "too long", "lamba", "lambi", "long hai")),
    (PrimaryReason.FIT, "too_short", ("too short", "short hai", "chhoti length", "short")),
    (PrimaryReason.FIT, "unspecified_fit", ("fit nahi aaya", "fit issue", "size issue", "fitting")),
    # --- quality ---
    (PrimaryReason.QUALITY_ISSUE, "fabric_quality", (
        "fabric quality", "cloth quality", "kapda kharab", "quality kharab",
        "poor quality", "bad quality", "cheap quality", "fabric", "cloth",
        "kapda", "itchy", "rough", "thin", "patla",
    )),
    (PrimaryReason.QUALITY_ISSUE, "material_mismatch", (
        "material alag", "material different", "material not", "material",
    )),
    (PrimaryReason.QUALITY_ISSUE, "comfort", ("uncomfortable", "comfortable nahi", "itch")),
    (PrimaryReason.QUALITY_ISSUE, "stitching_quality", ("stitching kharab", "silai", "stitch loose")),
    # --- appearance ---
    (PrimaryReason.APPEARANCE_ISSUE, "colour_mismatch", (
        "colour alag", "color alag", "colour different", "color different",
        "colour mismatch", "rang alag", "colour", "color", "rang",
    )),
    (PrimaryReason.APPEARANCE_ISSUE, "shade_mismatch", ("shade alag", "shade different", "shade")),
    (PrimaryReason.APPEARANCE_ISSUE, "print_mismatch", (
        "photo mein", "photo me", "picture mein", "print alag", "print different", "print",
    )),
    # --- damaged ---
    (PrimaryReason.DAMAGED_DEFECTIVE, "hole", ("hole", "chhed", "tear", "torn", "phata", "phat")),
    (PrimaryReason.DAMAGED_DEFECTIVE, "zip", ("zip kharab", "zip broken", "zip toot", "zip")),
    (PrimaryReason.DAMAGED_DEFECTIVE, "button", ("button toot", "button missing", "button")),
    (PrimaryReason.DAMAGED_DEFECTIVE, "stitching", ("stitching khul", "seam", "stitching")),
    (PrimaryReason.DAMAGED_DEFECTIVE, "general_defect", ("damaged", "damage", "defective", "defect", "broken")),
    # --- wrong item ---
    (PrimaryReason.WRONG_ITEM, "wrong_product_or_variant", (
        "wrong item", "wrong product", "galat product", "galat item",
        "alag product", "different item", "wrong size sent", "wrong",
    )),
    # --- changed mind ---
    (PrimaryReason.CHANGED_MIND, "preference_or_no_longer_needed", (
        "pasand nahi aya", "pasand nahi aaya", "pasand nahi", "not needed now",
        "not needed", "no longer needed", "changed my mind", "changed mind",
        "didn't like", "did not like", "dont want", "don't want",
    )),
    # --- delivery ---
    (PrimaryReason.DELIVERY_ISSUE, "late_delivery", (
        "late delivery", "delivery late", "der se aaya", "bahut late", "late",
    )),
    (PrimaryReason.DELIVERY_ISSUE, "package_damage", (
        "package damaged", "packet damaged", "packaging kharab", "box damaged",
    )),
    # --- not as expected ---
    (PrimaryReason.PRODUCT_NOT_AS_EXPECTED, "transparency", ("transparent", "see through", "patla dikh")),
    (PrimaryReason.PRODUCT_NOT_AS_EXPECTED, "style_mismatch", ("style alag", "style different", "design alag")),
    (PrimaryReason.PRODUCT_NOT_AS_EXPECTED, "shrinkage_concern", ("shrink", "sikud")),
    (PrimaryReason.PRODUCT_NOT_AS_EXPECTED, "occasion_unsuitable", ("not for wedding", "occasion")),
)


@dataclass(frozen=True)
class BaselineResult:
    """One rules-baseline classification, or an abstention."""

    primary_reason: str | None
    sub_reason: str | None
    body_area: str
    matched: str | None

    @property
    def abstained(self) -> bool:
        return self.primary_reason is None


ABSTAIN = BaselineResult(None, None, BodyArea.NONE.value, None)


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace/punctuation for matching."""
    return re.sub(r"[^a-z0-9\s]+", " ", text.lower()).strip()


def _find_body_area(norm: str) -> str:
    """Return the stated body area, or `none`. Never guesses."""
    best, best_len = BodyArea.NONE.value, 0
    for area, cues in _BODY_AREA_CUES.items():
        for cue in cues:
            if cue in norm and len(cue) > best_len:
                best, best_len = area, len(cue)
    return best


def classify(text: str) -> BaselineResult:
    """Classify one comment by keyword, or abstain.

    Picks the longest matching pattern across all rules, so a specific cue
    (`not needed`) wins over a substring of itself and the outcome does not
    depend on rule ordering.
    """
    if not text or not text.strip():
        return ABSTAIN

    norm = _normalise(text)
    if len(norm) < 3:
        return ABSTAIN

    winner: tuple[PrimaryReason, str, str] | None = None
    winner_len = 0
    for primary, sub_reason, patterns in _RULES:
        for pattern in patterns:
            if pattern in norm and len(pattern) > winner_len:
                winner, winner_len = (primary, sub_reason, pattern), len(pattern)

    if winner is None:
        # Nothing matched. This is the honest outcome for comments like
        # "okay nahi laga" or "quality?" — the baseline cannot read them, and
        # pretending otherwise is what the control arm exists to disprove.
        return ABSTAIN

    primary, sub_reason, matched = winner
    body_area = (
        _find_body_area(norm)
        if primary is PrimaryReason.FIT
        else BodyArea.NONE.value
    )
    return BaselineResult(primary.value, sub_reason, body_area, matched)


def classify_many(texts: list[str]) -> list[BaselineResult]:
    return [classify(t) for t in texts]


def coverage(results: list[BaselineResult]) -> float:
    """Share of rows the baseline was willing to label at all."""
    if not results:
        return 0.0
    return sum(1 for r in results if not r.abstained) / len(results)
