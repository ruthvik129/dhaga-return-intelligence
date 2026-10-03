"""Deterministic analytics, evaluation, and cost. No model calls anywhere.

DETERMINISTIC MODULE — this file is the whole of spec section 16's "use
deterministic code for" list: counting, percentages, grouping, aggregation,
sorting, threshold comparison, cost arithmetic, evaluation metrics. No LLM is
consulted to compute a number. The module is split into three clearly separated
sections so the merge of analytics + evaluator + costing (decision D2) does not
turn into a grab bag:

    1. AGGREGATION  — what the dashboard shows Neha
    2. EVALUATION   — scoring against hidden ground truth, after inference
    3. COST         — token arithmetic against configured prices

Nothing in section 1 reads ground truth. Ground truth enters only in section 2,
joined by return_id at read time, and is never written back into the run
artifact.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd

from . import config
from .config import ModelSpec, price_for
from .schemas import ROW_STATES, normalise_sub_reason

# =========================================================================
# 0. Invariants
# =========================================================================


class InvariantViolation(AssertionError):
    """Raised when a deterministic guarantee is broken.

    Loud by design. Spec section 23 says never silently drop a record, so a
    short dashboard is a hard failure rather than a quietly wrong chart.
    """


def assert_no_rows_lost(results: Sequence[dict], eligible_count: int) -> None:
    """Every eligible row must end in exactly one terminal state."""
    if len(results) != eligible_count:
        raise InvariantViolation(
            f"Row count mismatch: {len(results)} results for "
            f"{eligible_count} eligible rows. {eligible_count - len(results)} "
            "row(s) were dropped somewhere in the pipeline."
        )
    missing_state = [r.get("return_id") for r in results if not r.get("status")]
    if missing_state:
        raise InvariantViolation(
            f"{len(missing_state)} row(s) have no terminal status, "
            f"e.g. {missing_state[:5]}."
        )
    unknown = {r["status"] for r in results} - set(ROW_STATES)
    if unknown:
        raise InvariantViolation(f"Unknown row status values: {sorted(unknown)}")

    ids = [r.get("return_id") for r in results]
    if len(set(ids)) != len(ids):
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        raise InvariantViolation(f"Duplicate return_id in results: {dupes[:5]}")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Reported alongside every accuracy figure that comes from a small bucket.
    Five lines of stdlib maths; adding scipy for this would fail the reuse
    ladder. Verified against the published worked example in the test suite.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# =========================================================================
# 1. AGGREGATION — what the dashboard shows. Never reads ground truth.
# =========================================================================


def results_frame(results: Sequence[dict]) -> pd.DataFrame:
    """Build the working frame, applying reviewer corrections if present."""
    frame = pd.DataFrame(list(results))
    if frame.empty:
        return frame
    for col in ("primary_reason", "sub_reason", "body_area", "status"):
        if col not in frame.columns:
            frame[col] = None
    return frame


def apply_corrections(
    frame: pd.DataFrame, corrections: dict[str, dict]
) -> pd.DataFrame:
    """Overlay reviewer corrections onto the immutable run.

    Spec section 7.4 requires corrections to update the classification the
    dashboard uses. The run artifact is never mutated — corrections live in
    their own session-keyed store and are layered at read time, so two
    reviewers cannot alter each other's numbers (decision D5).
    """
    if frame.empty or not corrections:
        return frame
    out = frame.copy()
    out["corrected"] = False
    for return_id, fix in corrections.items():
        mask = out["return_id"] == return_id
        if not mask.any():
            continue
        action = (fix.get("action") or "").lower()
        if action == "mark_unclear":
            out.loc[mask, "primary_reason"] = "unclear"
            out.loc[mask, "sub_reason"] = "insufficient_information"
            out.loc[mask, "body_area"] = "none"
        else:
            for src, col in (
                ("primary_reason", "primary_reason"),
                ("sub_reason", "sub_reason"),
                ("body_area", "body_area"),
            ):
                if fix.get(src):
                    out.loc[mask, col] = fix[src]
        if action in {"confirm", "change", "mark_unclear"}:
            out.loc[mask, "status"] = "CLASSIFIED_HUMAN"
        out.loc[mask, "corrected"] = True
    return out


def _report_field(report: Any, name: str) -> Any:
    """Read a validation field from either a ValidationReport or its dict form.

    The app caches the report as a dict (dataclasses are not hashable cache
    values), while the pipeline passes the object. Accepting both keeps one
    implementation of the counters instead of two that can drift.
    """
    if isinstance(report, dict):
        return report.get(name)
    return getattr(report, name)


def overview(frame: pd.DataFrame, report: Any) -> dict:
    """Spec section 18.1 counters."""
    status = frame["status"].value_counts() if not frame.empty else pd.Series(dtype=int)
    reviewed = int(status.get("CLASSIFIED_HUMAN", 0))
    return {
        "total_returns_loaded": _report_field(report, "total_rows"),
        "total_other": _report_field(report, "other_rows"),
        "pct_other": _report_field(report, "other_pct"),
        "classified_model_a": int(status.get("CLASSIFIED_MODEL_A", 0)),
        "classified_model_b": int(status.get("CLASSIFIED_MODEL_B", 0)),
        "needs_human_review": int(status.get("NEEDS_HUMAN_REVIEW", 0)),
        "resolved_by_human": reviewed,
        "processing_error": int(status.get("PROCESSING_ERROR", 0)),
        "invalid_input": int(status.get("INVALID_INPUT", 0)),
    }


def reason_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Decomposed `Other` reasons — the headline 44%-becomes-meaningful chart."""
    if frame.empty:
        return pd.DataFrame(columns=["primary_reason", "rows", "pct"])
    counts = frame["primary_reason"].value_counts(dropna=True)
    total = int(counts.sum()) or 1
    return (
        counts.rename_axis("primary_reason")
        .reset_index(name="rows")
        .assign(pct=lambda d: (100.0 * d["rows"] / total).round(1))
    )


def sub_reason_distribution(
    frame: pd.DataFrame, primary: str | None = None
) -> pd.DataFrame:
    subset = frame if primary is None else frame[frame["primary_reason"] == primary]
    if subset.empty:
        return pd.DataFrame(columns=["sub_reason", "rows"])
    return (
        subset["sub_reason"]
        .value_counts(dropna=True)
        .rename_axis("sub_reason")
        .reset_index(name="rows")
    )


def body_area_distribution(
    frame: pd.DataFrame, primary: str | None = "fit", include_none: bool = False
) -> pd.DataFrame:
    subset = frame if primary is None else frame[frame["primary_reason"] == primary]
    if not include_none:
        subset = subset[subset["body_area"].fillna("none") != "none"]
    if subset.empty:
        return pd.DataFrame(columns=["body_area", "rows"])
    return (
        subset["body_area"]
        .value_counts(dropna=True)
        .rename_axis("body_area")
        .reset_index(name="rows")
    )


def group_intelligence(
    frame: pd.DataFrame,
    group_col: str,
    target_primary: str,
    min_rows: int = 5,
) -> pd.DataFrame:
    """Share-within-group against the overall baseline, with lift.

    This is premise P4, and it is the difference between a defensible finding
    and a misleading one. Vendor_12 holds 137 of the 440 `Other` rows, so it
    tops ANY raw-count chart on volume alone. What identifies a real problem is
    the share of *that group's own* returns carrying the reason, compared with
    the overall baseline.

    Deliberately NOT called a return rate: the dataset contains returned orders
    only, so there is no sales denominator and no true rate can be computed
    (spec section 18.4). Column names say `share_pct`, never `rate`.
    """
    if frame.empty or group_col not in frame.columns:
        return pd.DataFrame()

    overall = frame["primary_reason"].eq(target_primary).mean()
    grouped = frame.groupby(group_col, dropna=True)
    rows = []
    for name, sub in grouped:
        n = len(sub)
        if n < min_rows:
            continue
        hits = int(sub["primary_reason"].eq(target_primary).sum())
        share = hits / n
        lo, hi = wilson_interval(hits, n)
        rows.append(
            {
                group_col: name,
                "classified_rows": n,
                "matching_rows": hits,
                "share_pct": round(100 * share, 1),
                "baseline_pct": round(100 * overall, 1),
                "lift": round(share / overall, 2) if overall > 0 else None,
                "share_ci_low_pct": round(100 * lo, 1),
                "share_ci_high_pct": round(100 * hi, 1),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("lift", ascending=False) if not out.empty else out


def drill_down(
    frame: pd.DataFrame,
    *,
    group_col: str | None = None,
    group_value: str | None = None,
    primary: str | None = None,
    sub_reason: str | None = None,
    body_area: str | None = None,
) -> pd.DataFrame:
    """The source comments behind any aggregate (spec section 18.6).

    Every number on the dashboard must be traceable to the customer comments
    that produced it. This is the mechanism.
    """
    subset = frame
    if group_col and group_value is not None:
        subset = subset[subset[group_col] == group_value]
    if primary:
        subset = subset[subset["primary_reason"] == primary]
    if sub_reason:
        subset = subset[subset["sub_reason"] == sub_reason]
    if body_area:
        subset = subset[subset["body_area"] == body_area]
    cols = [
        c
        for c in (
            "return_id", "sku", "product_name", "vendor", "category",
            "size_ordered", "original_text", "primary_reason", "sub_reason",
            "body_area", "confidence", "evidence_text", "status",
            "review_reason",
        )
        if c in subset.columns
    ]
    return subset[cols]


def headline_findings(
    frame: pd.DataFrame,
    group_cols: Sequence[str] = ("sku", "vendor"),
    *,
    min_rows: int = 10,
    min_matching: int = 5,
    max_findings: int = 3,
    max_per_dimension: int = 2,
) -> list[dict]:
    """The "what should I fix this week" list, ranked by excess returns.

    The dashboard used to hand a merchandiser a table and leave the judgment
    to them. This does the judgment: it keeps only groups whose share of a
    reason is above the overall baseline *even at the low end of its
    confidence interval*, so a handful of returns cannot manufacture a
    finding, and ranks what survives by `excess_rows` — how many more returns
    carry that reason than the baseline would predict.

    `excess_rows` is the ranking key rather than `lift` on purpose. A tiny
    SKU at 4x lift is three returns; a large one at 1.6x is twenty. The second
    is the one worth a meeting, and it is also the one a non-technical reader
    understands without being taught what a multiplier means.

    `max_per_dimension` keeps the list from filling up with three views of the
    same SKU problem when a vendor finding is more useful alongside it.
    """
    if frame.empty or "primary_reason" not in frame.columns:
        return []

    primaries = [p for p in frame["primary_reason"].dropna().unique() if p != "unclear"]
    candidates: list[dict] = []

    for col in group_cols:
        if col not in frame.columns:
            continue
        for primary in primaries:
            groups = group_intelligence(frame, col, primary, min_rows=min_rows)
            if groups.empty:
                continue
            for _, row in groups.iterrows():
                matching = int(row["matching_rows"])
                baseline = float(row["baseline_pct"])
                # Not a finding unless the whole interval clears the baseline.
                if matching < min_matching or row["share_ci_low_pct"] <= baseline:
                    continue
                expected = row["classified_rows"] * baseline / 100.0
                candidates.append(
                    {
                        "dimension": col,
                        "group": str(row[col]),
                        "reason": primary,
                        "matching_rows": matching,
                        "classified_rows": int(row["classified_rows"]),
                        "share_pct": float(row["share_pct"]),
                        "baseline_pct": baseline,
                        "lift": None if row["lift"] is None else float(row["lift"]),
                        "share_ci_low_pct": float(row["share_ci_low_pct"]),
                        "share_ci_high_pct": float(row["share_ci_high_pct"]),
                        "excess_rows": int(round(matching - expected)),
                    }
                )

    candidates.sort(key=lambda f: (-f["excess_rows"], -(f["lift"] or 0)))

    picked: list[dict] = []
    per_dimension: Counter = Counter()
    seen: set[tuple[str, str]] = set()
    covered: list[set[str]] = []
    for cand in candidates:
        key = (cand["dimension"], cand["group"])
        if key in seen or per_dimension[cand["dimension"]] >= max_per_dimension:
            continue
        # A single-SKU vendor produces the same finding twice under two labels
        # ("KRT-184 has a fit problem" / "Vendor_07 has a fit problem") over
        # the identical returns. Saying it twice teaches nothing, so a
        # candidate that mostly re-covers an accepted finding is dropped.
        ids = _finding_ids(frame, cand)
        if any(_overlap(ids, prev) > 0.7 for prev in covered):
            continue
        seen.add(key)
        covered.append(ids)
        per_dimension[cand["dimension"]] += 1
        picked.append(finding_detail(frame, cand))
        if len(picked) >= max_findings:
            break
    return picked


def _finding_ids(frame: pd.DataFrame, finding: dict) -> set[str]:
    """The return_ids one finding is about."""
    subset = frame[
        (frame[finding["dimension"]] == finding["group"])
        & (frame["primary_reason"] == finding["reason"])
    ]
    return set(subset["return_id"].astype(str)) if "return_id" in subset.columns else set()


def _overlap(a: set[str], b: set[str]) -> float:
    """Share of the smaller set that the two have in common."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def finding_detail(frame: pd.DataFrame, finding: dict) -> dict:
    """Attach the texture a sentence needs: the dominant detail and real quotes.

    Evidence is the customer's own words, deduplicated, because three rows all
    reading "chest bahut tight tha" is one quote and two repeats — and a
    reader who sees the same sentence three times learns less, not more.
    """
    rows = drill_down(
        frame,
        group_col=finding["dimension"],
        group_value=finding["group"],
        primary=finding["reason"],
    )
    out = dict(finding)
    out["product_name"] = None
    if "product_name" in rows.columns and not rows.empty:
        names = rows["product_name"].dropna()
        out["product_name"] = str(names.iloc[0]) if len(names) else None

    subs = sub_reason_distribution(rows, finding["reason"])
    out["top_detail"] = str(subs.iloc[0]["sub_reason"]) if not subs.empty else None
    out["top_detail_rows"] = int(subs.iloc[0]["rows"]) if not subs.empty else 0

    areas = body_area_distribution(rows, finding["reason"])
    out["top_body_area"] = str(areas.iloc[0]["body_area"]) if not areas.empty else None
    out["top_body_area_rows"] = int(areas.iloc[0]["rows"]) if not areas.empty else 0

    quotes: list[dict] = []
    if not rows.empty and "original_text" in rows.columns:
        # Quote the returns the headline is actually about. A finding that says
        # "tight at the chest" supported by a comment about shoulders reads as
        # sloppy even when both numbers are right, so rows matching the
        # dominant body area and detail are offered first.
        def relevance(r: pd.Series) -> tuple[int, int]:
            area_hit = (
                out["top_body_area"] is not None
                and r.get("body_area") == out["top_body_area"]
            )
            detail_hit = (
                out["top_detail"] is not None and r.get("sub_reason") == out["top_detail"]
            )
            return (0 if area_hit else 1, 0 if detail_hit else 1)

        ordered = sorted((r for _, r in rows.iterrows()), key=relevance)
        seen_text: set[str] = set()
        for r in ordered:
            text = str(r.get("original_text") or "").strip()
            norm = " ".join(text.lower().split())
            if not text or norm in seen_text:
                continue
            seen_text.add(norm)
            quotes.append(
                {
                    "return_id": str(r.get("return_id") or ""),
                    "text": text,
                    "size_ordered": (
                        None if pd.isna(r.get("size_ordered")) else r.get("size_ordered")
                    )
                    if "size_ordered" in rows.columns
                    else None,
                }
            )
            if len(quotes) == 2:
                break
    out["quotes"] = quotes
    return out


def queue_summary(frame: pd.DataFrame) -> dict:
    """What is waiting for a person, in their terms rather than row counts."""
    if frame.empty or "status" not in frame.columns:
        return {"rows": 0, "minutes": 0, "by_reason": [], "example": None}
    queue = frame[frame["status"] == "NEEDS_HUMAN_REVIEW"]
    by_reason = (
        queue["review_reason"].dropna().value_counts().rename_axis("reason").reset_index(name="rows")
        if not queue.empty and "review_reason" in queue.columns
        else pd.DataFrame(columns=["reason", "rows"])
    )
    example = None
    if not queue.empty and "original_text" in queue.columns:
        texts = [str(t).strip() for t in queue["original_text"].fillna("").tolist() if str(t).strip()]
        if texts:
            # The shortest comment makes the point fastest: the ones a human
            # has to read are short because they say nothing. "no" is too short
            # to read as a sentence though, so prefer the shortest real phrase.
            phrases = [t for t in texts if len(t) >= 8]
            example = min(phrases or texts, key=len)
    return {
        "rows": int(len(queue)),
        "minutes": round(len(queue) * config.REVIEW_MINUTES_PER_ROW),
        "by_reason": by_reason.to_dict("records"),
        "example": example,
    }


# =========================================================================
# 2. EVALUATION — reads ground truth, only after inference.
# =========================================================================


@dataclass
class AccuracyBucket:
    """An accuracy figure that always carries its denominator."""

    label: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float | None:
        return (self.correct / self.total) if self.total else None

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.correct, self.total)

    def to_dict(self) -> dict:
        lo, hi = self.interval
        return {
            "label": self.label,
            "correct": self.correct,
            "n": self.total,
            "accuracy_pct": None
            if self.accuracy is None
            else round(100 * self.accuracy, 1),
            "ci_low_pct": round(100 * lo, 1),
            "ci_high_pct": round(100 * hi, 1),
        }


def _join_truth(frame: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Join predictions to ground truth by return_id, at read time only."""
    if frame.empty or ground_truth.empty:
        return pd.DataFrame()
    merged = frame.merge(ground_truth, on="return_id", how="inner")
    # Normalisation is applied to BOTH sides of every comparison. Applying it
    # to one side only is the subtle version of this bug: sub-reason accuracy
    # would be wrong on every damaged row and no test would fail (premise P3).
    merged["pred_sub"] = merged["sub_reason"].map(normalise_sub_reason)
    merged["true_sub"] = merged["expected_sub_reason"].map(normalise_sub_reason)
    return merged


def evaluate(frame: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    """Spec section 20 metrics, each with its denominator and interval."""
    merged = _join_truth(frame, ground_truth)
    if merged.empty:
        return {"available": False, "reason": "no ground truth joined"}

    auto_a = merged[merged["status"] == "CLASSIFIED_MODEL_A"]
    auto_b = merged[merged["status"] == "CLASSIFIED_MODEL_B"]
    automated = merged[
        merged["status"].isin(["CLASSIFIED_MODEL_A", "CLASSIFIED_MODEL_B"])
    ]
    review = merged[merged["status"] == "NEEDS_HUMAN_REVIEW"]
    n = len(merged)

    def primary_bucket(subset: pd.DataFrame, label: str) -> AccuracyBucket:
        if subset.empty:
            return AccuracyBucket(label, 0, 0)
        hits = int(
            (subset["primary_reason"] == subset["expected_primary_reason"]).sum()
        )
        return AccuracyBucket(label, hits, len(subset))

    # Body area is scored only where the truth states one, per spec section 20
    # ("body-area accuracy where applicable"). Scoring `none == none` as a win
    # would inflate the figure with rows that required no judgement.
    body_scope = automated[automated["expected_body_area"].fillna("none") != "none"]
    body_hits = int(
        (body_scope["body_area"] == body_scope["expected_body_area"]).sum()
    )

    sub_scope = automated[automated["pred_sub"].notna()]
    sub_hits = int((sub_scope["pred_sub"] == sub_scope["true_sub"]).sum())

    return {
        "available": True,
        "rows_evaluated": n,
        "primary_accuracy_overall": primary_bucket(merged, "All rows").to_dict(),
        "primary_accuracy_automated": primary_bucket(
            automated, "Automated only (excludes human review)"
        ).to_dict(),
        "auto_accept_precision_a": primary_bucket(
            auto_a, "Accuracy of rows Model A auto-accepted"
        ).to_dict(),
        "accuracy_after_b": primary_bucket(
            auto_b, "Accuracy of rows accepted after Model B"
        ).to_dict(),
        "sub_reason_accuracy": AccuracyBucket(
            "Sub-reason, automated rows", sub_hits, len(sub_scope)
        ).to_dict(),
        "body_area_accuracy": AccuracyBucket(
            "Body area, where ground truth states one", body_hits, len(body_scope)
        ).to_dict(),
        "rates": {
            "auto_accept_pct": round(100 * len(auto_a) / n, 1) if n else 0.0,
            "escalated_pct": round(
                100 * int((merged["escalated"] == True).sum()) / n, 1  # noqa: E712
            )
            if "escalated" in merged.columns and n
            else 0.0,
            "human_review_pct": round(100 * len(review) / n, 1) if n else 0.0,
        },
        "confusion_matrix": confusion_matrix(merged).to_dict(orient="index"),
        "notable_confusions": notable_confusions(merged),
    }


def confusion_matrix(merged: pd.DataFrame) -> pd.DataFrame:
    """Primary-reason confusion matrix (spec section 20)."""
    if merged.empty:
        return pd.DataFrame()
    return pd.crosstab(
        merged["expected_primary_reason"].rename("truth"),
        merged["primary_reason"].rename("predicted"),
        dropna=False,
    )


#: The pairs spec section 20 asks us to inspect specifically.
_CONFUSION_PAIRS = (
    ("fit", "product_not_as_expected"),
    ("quality_issue", "product_not_as_expected"),
    ("appearance_issue", "product_not_as_expected"),
    ("changed_mind", "unclear"),
)


def notable_confusions(merged: pd.DataFrame) -> list[dict]:
    """Counts for the specific confusion pairs the spec calls out."""
    if merged.empty:
        return []
    out = []
    for truth, predicted in _CONFUSION_PAIRS:
        both_ways = int(
            (
                (merged["expected_primary_reason"] == truth)
                & (merged["primary_reason"] == predicted)
            ).sum()
        ) + int(
            (
                (merged["expected_primary_reason"] == predicted)
                & (merged["primary_reason"] == truth)
            ).sum()
        )
        out.append({"pair": f"{truth} vs {predicted}", "confused_rows": both_ways})
    return out


def evaluate_baseline(
    texts: Sequence[str],
    return_ids: Sequence[str],
    ground_truth: pd.DataFrame,
) -> dict:
    """Score the rules control arm — reporting BOTH accuracy figures.

    A keyword matcher that abstains looks far better than it is if you score
    only the rows it answered. Both numbers are reported because quoting the
    flattering one would be exactly the kind of selectively-true metric this
    project argues against:

      * `accuracy_on_answered` — of the rows it was willing to label
      * `accuracy_over_all`    — of every row, counting abstentions as misses

    The second is the one comparable to the cascade.
    """
    from . import baseline_rules

    truth = ground_truth.set_index("return_id")["expected_primary_reason"].to_dict()
    results = [baseline_rules.classify(t) for t in texts]

    answered = 0
    correct = 0
    for rid, res in zip(return_ids, results, strict=True):
        if res.abstained:
            continue
        answered += 1
        if res.primary_reason == truth.get(rid):
            correct += 1

    total = len(results)
    return {
        "rows": total,
        "answered": answered,
        "abstained": total - answered,
        "coverage_pct": round(100 * answered / total, 1) if total else 0.0,
        "accuracy_on_answered": AccuracyBucket(
            "Rules baseline, rows it answered", correct, answered
        ).to_dict(),
        "accuracy_over_all": AccuracyBucket(
            "Rules baseline, all rows (abstentions count as misses)",
            correct,
            total,
        ).to_dict(),
    }


# =========================================================================
# 3. COST — token arithmetic against configured prices.
# =========================================================================


@dataclass
class ModelCost:
    model_key: str
    calls: int
    input_tokens: int
    output_tokens: int
    input_cost: float | None
    output_cost: float | None
    priced: bool
    price_as_of: str
    measured_tokens: bool

    @property
    def total(self) -> float | None:
        if not self.priced:
            return None
        return (self.input_cost or 0.0) + (self.output_cost or 0.0)

    def to_dict(self) -> dict:
        return {
            "model": self.model_key,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost_usd": self.input_cost,
            "output_cost_usd": self.output_cost,
            "total_cost_usd": self.total,
            "priced": self.priced,
            "price_as_of": self.price_as_of,
            "token_counts": "measured" if self.measured_tokens else "estimated",
        }


def cost_report(usage: dict[str, dict], rows_classified: int) -> dict:
    """Deterministic cost arithmetic (spec section 21).

    An unpriced model reports `None`, never 0.0. A silent zero would make an
    expensive run look free, which is the class of quiet wrongness spec
    section 23 exists to prevent. The UI must render `None` as "not priced"
    and say which model is missing a rate.
    """
    per_model: list[ModelCost] = []
    for key, entry in usage.items():
        provider, _, name = key.partition(":")
        price = price_for(ModelSpec(provider=provider, name=name))
        in_tok = int(entry.get("input_tokens", 0))
        out_tok = int(entry.get("output_tokens", 0))
        if price.is_configured:
            in_cost = in_tok / 1_000_000 * (price.input_per_mtok or 0.0)
            out_cost = out_tok / 1_000_000 * (price.output_per_mtok or 0.0)
        else:
            in_cost = out_cost = None
        per_model.append(
            ModelCost(
                model_key=key,
                calls=int(entry.get("calls", 0)),
                input_tokens=in_tok,
                output_tokens=out_tok,
                input_cost=in_cost,
                output_cost=out_cost,
                priced=price.is_configured,
                price_as_of=price.as_of,
                measured_tokens=bool(entry.get("token_counts_measured", True)),
            )
        )

    priced = [m for m in per_model if m.priced]
    unpriced = [m.model_key for m in per_model if not m.priced]
    run_total = sum((m.total or 0.0) for m in priced) if priced else None

    return {
        "per_model": [m.to_dict() for m in per_model],
        "run_total_usd": run_total,
        "fully_priced": not unpriced,
        "unpriced_models": unpriced,
        "cost_per_1000_comments_usd": (
            round(run_total / rows_classified * 1000, 4)
            if run_total is not None and rows_classified
            else None
        ),
        "rows_classified": rows_classified,
        "all_tokens_measured": all(m.measured_tokens for m in per_model),
    }


def escalation_economics(usage: dict[str, dict], model_b_key: str) -> dict:
    """What the cascade bought, in calls and share (spec section 21)."""
    total_calls = sum(int(e.get("calls", 0)) for e in usage.values())
    b_calls = int(usage.get(model_b_key, {}).get("calls", 0))
    return {
        "total_model_calls": total_calls,
        "model_b_calls": b_calls,
        "pct_calls_on_model_b": round(100 * b_calls / total_calls, 1)
        if total_calls
        else 0.0,
    }


def projected_cost(cost: dict, comments: int) -> float | None:
    """Extrapolate the measured per-1000 cost to a larger volume."""
    per_1000 = cost.get("cost_per_1000_comments_usd")
    if per_1000 is None:
        return None
    return round(per_1000 * comments / 1000, 2)
