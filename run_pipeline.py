#!/usr/bin/env python
"""CLI runner — the canonical way to produce a run artifact.

This is the entry point the committed artifact comes from (premise P7). The
web app runs the same `run()` per browser session when a user uploads a file
and presses Classify; this script is the reproducible path:

    python run_pipeline.py                 # full 440-row run
    python run_pipeline.py --limit 20      # quick smoke run
    python run_pipeline.py --refresh-prices
    python run_pipeline.py --inject-faults # exercise the schema-repair path

Concurrency and retry policy live in config (decision D6): a bounded worker
pool with results re-keyed by return_id, so output order is deterministic
regardless of completion order.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable

from src import config, metrics
from src.data_io import ReturnRecord, load_dataset, save_run
from src.models import (
    Classifier,
    ModelUnavailable,
    UsageLedger,
    evidence_is_grounded,
    probe_model,
    reconcile_attribute_fallback,
)
from src.router import (
    Route,
    StepOutcome,
    route_after_a,
    route_after_b,
    route_no_text,
    terminal_state,
)
from src.schemas import PrimaryReason


def _base_record(rec: ReturnRecord) -> dict:
    """The fields every result row carries, whatever path it took."""
    return {
        "return_id": rec.return_id,
        "sku": rec.sku,
        "product_name": rec.product_name,
        "category": rec.category,
        "vendor": rec.vendor,
        "size_ordered": rec.size_ordered,
        "original_dropdown": rec.dropdown_reason,
        "original_text": rec.text,
        "primary_reason": None,
        "sub_reason": None,
        "body_area": "none",
        "confidence": None,
        "classifier_model": None,
        "escalated": False,
        "needs_review": False,
        "review_reason": None,
        "routing_detail": None,
        "evidence_text": "",
        "evidence_grounded": True,
        "schema_repaired": False,
        "status": None,
        "trace": [],
    }


def run(
    dataset: str,
    limit: int | None = None,
    progress: Callable[[str, float], None] | None = None,
    model_a: config.ModelSpec | None = None,
    model_b: config.ModelSpec | None = None,
) -> dict:
    """Execute the full pipeline and return the run payload.

    `model_a` / `model_b` override the first-pass and second-check models from
    `.env` for this run only. `compare_models.py` uses them to run several
    candidates over the same rows; the app and the CLI leave them unset.
    """
    spec_a = model_a or config.MODEL_A
    spec_b = model_b or config.MODEL_B
    started = time.time()

    def tick(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)
        else:
            print(f"  [{frac:5.0%}] {msg}", flush=True)

    tick("Loading dataset", 0.0)
    loaded = load_dataset(dataset)
    if not loaded.report.is_valid:
        raise SystemExit(
            "Dataset failed validation:\n  - "
            + "\n  - ".join(loaded.report.errors)
        )

    records = loaded.eligible_records()
    if limit:
        records = records[:limit]
    eligible_count = len(records)
    tick(f"{eligible_count} eligible rows ('Other' dropdown)", 0.05)

    ledger = UsageLedger()
    results: dict[str, dict] = {r.return_id: _base_record(r) for r in records}

    # Rows with no usable text never reach a model: there is nothing to
    # classify, so spending two calls to discover that would be waste.
    no_text = [r for r in records if not r.has_usable_text]
    with_text = [r for r in records if r.has_usable_text]
    for rec in no_text:
        decision = route_no_text()
        row = results[rec.return_id]
        row.update(
            primary_reason="unclear",
            sub_reason="insufficient_information",
            needs_review=True,
            review_reason=decision.reason,
            routing_detail=decision.detail,
            status="NEEDS_HUMAN_REVIEW",
            trace=[{"step": "pre-flight", "detail": decision.detail}],
        )
    if no_text:
        tick(f"{len(no_text)} row(s) had no usable text -> review", 0.08)

    clf_a = Classifier(spec_a, ledger)

    # --- Model A, chain step 1 -------------------------------------------
    tick(f"Model A step 1/2 on {len(with_text)} rows", 0.10)
    a_primary = clf_a.classify_primary_batch(with_text)

    accepted_a: list[tuple[ReturnRecord, PrimaryReason]] = []
    escalate: list[ReturnRecord] = []
    a_outcomes: dict[str, StepOutcome] = {}

    for rec, res in zip(with_text, a_primary, strict=True):
        row = results[rec.return_id]
        a_outcomes[rec.return_id] = res.outcome
        row["schema_repaired"] = res.repaired
        row["trace"].append({
            "step": "model_a_primary",
            "model": spec_a.key,
            "primary": res.outcome.primary,
            "confidence": res.outcome.confidence,
            "needs_review": res.outcome.needs_review,
            "schema_ok": res.outcome.schema_ok,
            "repaired": res.repaired,
            "error": res.outcome.error,
        })
        decision = route_after_a(res.outcome)
        row["routing_detail"] = decision.detail
        row["trace"].append({"step": "router_a", "route": decision.route.value,
                             "reason": decision.reason, "detail": decision.detail})
        if decision.route is Route.ACCEPT:
            row.update(
                primary_reason=res.outcome.primary,
                confidence=res.outcome.confidence,
                classifier_model=spec_a.key,
            )
            accepted_a.append((rec, PrimaryReason(res.outcome.primary)))
        else:
            row["review_reason"] = decision.reason
            escalate.append(rec)

    tick(f"Model A accepted {len(accepted_a)}, escalating {len(escalate)}", 0.35)

    # --- Model A, chain step 2 (attributes for accepted rows) ------------
    if accepted_a:
        tick(f"Model A step 2/2 on {len(accepted_a)} accepted rows", 0.40)
        for (rec, primary), res in zip(
            accepted_a, clf_a.extract_attributes_batch(accepted_a), strict=True
        ):
            _apply_step_b(results[rec.return_id], rec, primary, res,
                          spec_a.key)
            results[rec.return_id]["status"] = terminal_state(
                Route.ACCEPT, escalated=False
            )

    # --- Model B on escalated rows ---------------------------------------
    if escalate:
        tick(f"Model B step 1/2 on {len(escalate)} escalated rows", 0.60)
        try:
            clf_b = Classifier(spec_b, ledger)
        except ModelUnavailable as exc:
            for rec in escalate:
                row = results[rec.return_id]
                row.update(
                    escalated=True,
                    needs_review=True,
                    review_reason=row["review_reason"] or "LOW_CONFIDENCE",
                    routing_detail=f"Model B unavailable: {exc}",
                    status="NEEDS_HUMAN_REVIEW",
                )
            clf_b = None
        if clf_b is not None:
            b_primary = clf_b.classify_primary_batch(escalate)
            accepted_b: list[tuple[ReturnRecord, PrimaryReason]] = []
            for rec, res in zip(escalate, b_primary, strict=True):
                row = results[rec.return_id]
                row["escalated"] = True
                row["trace"].append({
                    "step": "model_b_primary",
                    "model": spec_b.key,
                    "primary": res.outcome.primary,
                    "confidence": res.outcome.confidence,
                    "needs_review": res.outcome.needs_review,
                    "schema_ok": res.outcome.schema_ok,
                    "repaired": res.repaired,
                    "error": res.outcome.error,
                })
                decision = route_after_b(res.outcome, a_outcomes[rec.return_id])
                row["routing_detail"] = decision.detail
                row["trace"].append({"step": "router_b", "route": decision.route.value,
                                     "reason": decision.reason,
                                     "detail": decision.detail})
                if decision.route is Route.ACCEPT:
                    row.update(
                        primary_reason=res.outcome.primary,
                        confidence=res.outcome.confidence,
                        classifier_model=spec_b.key,
                        review_reason=None,
                    )
                    accepted_b.append((rec, PrimaryReason(res.outcome.primary)))
                else:
                    row.update(
                        primary_reason=res.outcome.primary or "unclear",
                        sub_reason=None,
                        confidence=res.outcome.confidence,
                        classifier_model=spec_b.key,
                        needs_review=True,
                        review_reason=decision.reason,
                        status="NEEDS_HUMAN_REVIEW",
                    )

            if accepted_b:
                tick(f"Model B step 2/2 on {len(accepted_b)} rows", 0.80)
                for (rec, primary), res in zip(
                    accepted_b, clf_b.extract_attributes_batch(accepted_b),
                    strict=True,
                ):
                    _apply_step_b(results[rec.return_id], rec, primary, res,
                                  spec_b.key)
                    results[rec.return_id]["status"] = terminal_state(
                        Route.ACCEPT, escalated=True
                    )

    # --- finalise ---------------------------------------------------------
    rows = list(results.values())
    for row in rows:
        if row["status"] is None:  # defensive: nothing may leave unlabelled
            row["status"] = "PROCESSING_ERROR"
            row["routing_detail"] = (
                row["routing_detail"] or "Row reached the end with no terminal state."
            )
    rows.sort(key=lambda r: r["return_id"])

    metrics.assert_no_rows_lost(rows, eligible_count)

    elapsed = time.time() - started
    usage = ledger.to_dict()
    payload = {
        "run": {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": round(elapsed, 2),
            "eligible_rows": eligible_count,
            "prompt_version": __import__("src.prompts", fromlist=["x"]).PROMPT_VERSION,
            "config": config.RunConfig(model_a=spec_a, model_b=spec_b).to_dict(),
        },
        "validation": loaded.report.to_dict(),
        "dropdown_distribution": loaded.dropdown_distribution.to_dict(),
        "usage": usage,
        "cost": metrics.cost_report(usage, eligible_count),
        "escalation": metrics.escalation_economics(usage, spec_b.key),
        "records": rows,
    }
    tick(f"Done in {elapsed:.1f}s", 1.0)
    return payload


def _apply_step_b(row: dict, rec: ReturnRecord, primary: PrimaryReason,
                  res, model_key: str) -> None:
    """Write step-B output onto a result row, checking evidence grounding."""
    row["trace"].append({
        "step": "attributes",
        "model": model_key,
        "sub_reason": res.outcome.sub_reason,
        "schema_ok": res.outcome.schema_ok,
        "repaired": res.repaired,
        "error": res.outcome.error,
    })
    if res.outcome.schema_ok and res.parsed is not None:
        evidence = str(getattr(res.parsed, "evidence_text", "") or "")
        grounded = evidence_is_grounded(evidence, rec.text)
        row.update(
            sub_reason=res.outcome.sub_reason,
            body_area=str(getattr(res.parsed, "body_area").value),
            evidence_text=evidence if grounded else "",
            evidence_grounded=grounded,
        )
        if not grounded:
            # Ungrounded evidence is a quiet failure mode: the label may be
            # right while the quote is invented. Keep the label, drop the
            # quote, and flag it rather than showing the user a fabricated
            # citation (spec section 10.1).
            row["trace"].append({
                "step": "grounding_check",
                "detail": f"Evidence {evidence!r} is not a span of the comment; "
                          "discarded.",
            })
    else:
        row.update(
            sub_reason=reconcile_attribute_fallback(primary),
            body_area="none",
            schema_repaired=row["schema_repaired"] or res.repaired,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="classify only the first N eligible rows")
    parser.add_argument("--dataset", help="path to .xlsx or .csv (required to classify)")
    parser.add_argument("--refresh-prices", action="store_true",
                        help="fetch live OpenRouter pricing and exit")
    parser.add_argument("--inject-faults", action="store_true",
                        help="force schema failures to exercise the repair path")
    parser.add_argument("--no-cache", action="store_true",
                        help="disable the SQLite response cache")
    args = parser.parse_args()

    if args.refresh_prices:
        n = config.refresh_openrouter_prices()
        print(f"Wrote pricing for {n} OpenRouter models to "
              f"{config.OPENROUTER_PRICE_SNAPSHOT}")
        return 0

    if not args.dataset:
        parser.error("--dataset is required, e.g. --dataset data/"
                     f"{config.DEFAULT_DATASET.name}")

    if args.inject_faults:
        config.INJECT_SCHEMA_FAULTS = True
    if args.no_cache:
        config.ENABLE_RESPONSE_CACHE = False

    print(f"Model A: {config.MODEL_A.key}")
    print(f"Model B: {config.MODEL_B.key}")
    for spec in (config.MODEL_A, config.MODEL_B):
        ok, msg = probe_model(spec)
        print(f"  {'OK ' if ok else 'XX '} {msg}")
        if not ok:
            print("\nCannot run live classification. "
                  "The app still works from the committed run artifact, and "
                  "the rules baseline needs no credential.")
            return 2

    payload = run(dataset=args.dataset, limit=args.limit)
    path = save_run(payload)

    cost = payload["cost"]
    print(f"\nArtifact: {path}")
    print(f"Rows: {payload['run']['eligible_rows']}")
    counts: dict[str, int] = {}
    for r in payload["records"]:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for state, n in sorted(counts.items()):
        print(f"  {state:22s} {n}")
    total = cost.get("run_total_usd")
    print(f"Cost: {'$%.4f' % total if total is not None else 'not priced'}"
          f"  |  per 1k comments: {cost.get('cost_per_1000_comments_usd')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
