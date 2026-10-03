#!/usr/bin/env python
"""Compare candidate models on the same rows, and print the result as a table.

DETERMINISTIC MODULE apart from the runs it starts — every model call goes
through `run_pipeline.run`, which goes through `src/models.py`.

Why this exists: the configured pair (`.env`) was chosen to make the pipeline
measurable cheaply, not because it was shown to be the right pair (TODOS.md,
"Eval and cost matrix"). This runs each candidate over exactly the same rows,
scores it with the same `metrics.evaluate`, and prices it with the same
`metrics.cost_report` the app uses, so the numbers are comparable.

The table names models on purpose: comparing them is the point. That is why
this is a terminal tool and not a screen in the web app, which never shows
model names.

    python compare_models.py --dataset data/dhaga_returns_intelligence_synthetic_1000.xlsx \\
        --rows 200 \\
        --first-pass mistralai/mistral-nemo openai/gpt-oss-20b \\
        --second-check openai/gpt-oss-120b

Every first-pass model is paired with every second-check model. Responses are
cached in SQLite, so re-running a comparison, or adding one more candidate,
only pays for the runs that have not happened before.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import config, metrics
from src.data_io import load_dataset
from src.models import probe_model

COMPARISONS_DIR = config.OUTPUT_DIR / "comparisons"

#: The first-pass model is judged on the accuracy of the rows it accepted
#: with nobody checking. Below this, a candidate is not recommended however
#: cheap it is. Override with --target.
DEFAULT_TARGET_PCT = 95.0


# --- candidates -----------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One first-pass / second-check pairing to run."""

    first: config.ModelSpec
    second: config.ModelSpec

    @property
    def label(self) -> str:
        return f"{self.first.name} → {self.second.name}"


def spec_for(name: str, *, second: bool) -> config.ModelSpec:
    """An OpenRouter model id as a ModelSpec, with the same token limits the
    configured models use, so candidates differ only in the model."""
    base = config.MODEL_B if second else config.MODEL_A
    return config.ModelSpec(
        provider=config.PROVIDER_OPENROUTER,
        name=name,
        temperature=None,
        structured_output_method=None,
        max_tokens=base.max_tokens,
    )


def snapshot_models() -> dict[str, dict]:
    if not config.OPENROUTER_PRICE_SNAPSHOT.exists():
        return {}
    blob = json.loads(config.OPENROUTER_PRICE_SNAPSHOT.read_text(encoding="utf-8"))
    return blob.get("models") or {}


def check_candidates(names: list[str]) -> list[str]:
    """Problems that make a model unusable or its numbers unpriceable.

    Uses the OpenRouter price snapshot, so it needs no network. Refresh it with
    `python run_pipeline.py --refresh-prices` if a model is reported unknown.
    """
    known = snapshot_models()
    problems: list[str] = []
    for name in names:
        info = known.get(name)
        if info is None:
            problems.append(
                f"{name}: not in the OpenRouter price snapshot "
                f"({config.OPENROUTER_PRICE_SNAPSHOT.name}). Check the id, or "
                "refresh with `python run_pipeline.py --refresh-prices`."
            )
            continue
        if not info.get("supports_tools") or not info.get("supports_structured_outputs"):
            problems.append(
                f"{name}: does not support tool calling and structured output, "
                "which both pipeline steps need."
            )
    return problems


def warnings_for(names: list[str]) -> list[str]:
    """Usable, but the reader should know before trusting the table."""
    known = snapshot_models()
    out: list[str] = []
    for name in names:
        info = known.get(name) or {}
        if float(info.get("input_per_mtok") or 0) < 0:
            out.append(
                f"{name}: variable pricing (a router). Cost will show as "
                "'not priced', and results can change between runs."
            )
    return out


# --- scoring --------------------------------------------------------------


@dataclass
class Result:
    candidate: Candidate
    error: str | None = None
    row: dict = field(default_factory=dict)


def _pct(bucket: dict | None) -> float | None:
    return None if not bucket else bucket.get("accuracy_pct")


def hard_row_reads(records: list[dict], ground_truth: pd.DataFrame) -> dict:
    """How each stage did on the escalated rows, whatever routing did next.

    The router accepts a second-check answer only when it agrees with the
    first pass, so "accuracy of rows accepted after the second check" is
    usually n/a and says nothing about the second-check model. This reads each
    stage's own answer from the trace instead, on exactly the same hard rows,
    which is what choosing a second-check model needs.
    """
    if ground_truth.empty or "expected_primary_reason" not in ground_truth:
        return {}
    truth = ground_truth.set_index("return_id")["expected_primary_reason"].to_dict()
    first = second = n = 0
    for rec in records:
        if not rec.get("escalated") or rec["return_id"] not in truth:
            continue
        reads = {s.get("step"): s.get("primary") for s in rec.get("trace") or [] if isinstance(s, dict)}
        if "model_b_primary" not in reads:
            continue  # escalated but the second check never answered
        n += 1
        expected = truth[rec["return_id"]]
        first += reads.get("model_a_primary") == expected
        second += reads["model_b_primary"] == expected
    return {"hard_rows": n, "hard_first_right": first, "hard_second_right": second}


def summarise(payload: dict, ground_truth: pd.DataFrame) -> dict:
    """The comparison numbers for one finished run.

    Pure: the same payload and answer key always give the same row, which is
    what makes it testable without a network (tests/test_compare_models.py).
    """
    frame = metrics.results_frame(payload["records"])
    n = len(frame)
    status = frame["status"].value_counts() if n else pd.Series(dtype=int)
    escalated = int(frame["escalated"].sum()) if n and "escalated" in frame else 0
    cost = payload["cost"]
    cfg = payload["run"]["config"]
    first_usage = payload["usage"].get(cfg["model_a"], {})

    ev = metrics.evaluate(frame, ground_truth) if not ground_truth.empty else {}
    scored = bool(ev.get("available"))
    return {
        "rows": n,
        "auto_accept_accuracy_pct": _pct(ev.get("auto_accept_precision_a")) if scored else None,
        "auto_accept_n": ev["auto_accept_precision_a"]["n"] if scored else None,
        "overall_accuracy_pct": _pct(ev.get("primary_accuracy_overall")) if scored else None,
        "sub_reason_accuracy_pct": _pct(ev.get("sub_reason_accuracy")) if scored else None,
        "second_check_pct": round(100 * escalated / n, 1) if n else None,
        "review_pct": round(100 * int(status.get("NEEDS_HUMAN_REVIEW", 0)) / n, 1) if n else None,
        "unreadable_responses": int(first_usage.get("schema_failures", 0)),
        "cost_per_1000_usd": cost.get("cost_per_1000_comments_usd"),
        "run_total_usd": cost.get("run_total_usd"),
        "fully_priced": bool(cost.get("fully_priced")),
        "seconds": payload["run"].get("elapsed_seconds"),
        **hard_row_reads(payload["records"], ground_truth),
    }


def recommend(results: list[Result], target_pct: float) -> str:
    """The decision rule from the README: the cheapest first pass that clears
    the auto-accept accuracy bar. Said plainly when nothing does."""
    ok = [
        r for r in results
        if not r.error
        and r.row.get("auto_accept_accuracy_pct") is not None
        and r.row["auto_accept_accuracy_pct"] >= target_pct
    ]
    if not any(r.row.get("auto_accept_accuracy_pct") is not None for r in results if not r.error):
        return "No answer key in this file, so nothing could be scored. Use a labelled file to compare accuracy."
    if not ok:
        return (
            f"No candidate reached {target_pct:.0f}% auto-accept accuracy. "
            "Try stronger first-pass models, or lower --target if that bar is too strict."
        )
    priced = [r for r in ok if r.row.get("cost_per_1000_usd") is not None]
    pool = priced or ok
    best = min(pool, key=lambda r: r.row.get("cost_per_1000_usd") or 0.0)
    cost = best.row.get("cost_per_1000_usd")
    cost_txt = f"${cost:.4f} per 1,000 comments" if cost is not None else "price unknown"
    return (
        f"Cheapest pairing with auto-accept accuracy ≥ {target_pct:.0f}%: "
        f"{best.candidate.label} ({best.row['auto_accept_accuracy_pct']:.1f}%, {cost_txt})."
    )


# --- output ---------------------------------------------------------------


COLUMNS = [
    ("first pass → second check", lambda r: r.candidate.label),
    ("auto-accept\naccuracy", lambda r: _fmt_pct(r.row.get("auto_accept_accuracy_pct"))),
    ("overall\naccuracy", lambda r: _fmt_pct(r.row.get("overall_accuracy_pct"))),
    ("sub-reason\naccuracy", lambda r: _fmt_pct(r.row.get("sub_reason_accuracy_pct"))),
    ("sent to\n2nd check", lambda r: _fmt_pct(r.row.get("second_check_pct"))),
    ("sent to\nreview", lambda r: _fmt_pct(r.row.get("review_pct"))),
    ("2nd check right\non hard rows", lambda r: _fmt_share(r.row.get("hard_second_right"), r.row.get("hard_rows"))),
    ("1st pass right\non same rows", lambda r: _fmt_share(r.row.get("hard_first_right"), r.row.get("hard_rows"))),
    ("unreadable\nresponses", lambda r: str(r.row.get("unreadable_responses", "—"))),
    ("$ per 1,000\ncomments", lambda r: _fmt_usd(r.row.get("cost_per_1000_usd"))),
    ("time", lambda r: _fmt_secs(r.row.get("seconds"))),
]


def _fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}%"


def _fmt_share(right: int | None, n: int | None) -> str:
    """`75% (18/24)`: the count stays visible because n is always small."""
    if right is None or not n:
        return "n/a"
    return f"{round(100 * right / n)}% ({right}/{n})"


def _fmt_usd(v: float | None) -> str:
    return "not priced" if v is None else f"${v:.4f}"


def _fmt_secs(v: float | None) -> str:
    return "—" if v is None else f"{v:.0f}s"


def render_table(results: list[Result]) -> str:
    """Plain-text table: two header lines, right-aligned numbers."""
    heads = [h.split("\n") + [""] * (2 - len(h.split("\n"))) for h, _ in COLUMNS]
    cells = []
    for r in results:
        if r.error:
            cells.append([r.candidate.label, f"FAILED: {r.error}"] + [""] * (len(COLUMNS) - 2))
        else:
            cells.append([fn(r) for _, fn in COLUMNS])
    widths = [
        max(len(heads[i][0]), len(heads[i][1]), *(len(c[i]) for c in cells if not c[1].startswith("FAILED") or i == 0))
        for i in range(len(COLUMNS))
    ]

    def line(values: list[str]) -> str:
        first = values[0].ljust(widths[0])
        rest = [v.rjust(w) for v, w in zip(values[1:], widths[1:])]
        return "  ".join([first, *rest]).rstrip()

    out = [line([h[0] for h in heads]), line([h[1] for h in heads])]
    out.append("─" * len(out[0]))
    for c in cells:
        out.append(f"{c[0].ljust(widths[0])}  {c[1]}" if c[1].startswith("FAILED") else line(c))
    return "\n".join(out)


def write_csv(results: list[Result], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "rows", "auto_accept_accuracy_pct", "auto_accept_n", "overall_accuracy_pct",
        "sub_reason_accuracy_pct", "second_check_pct", "review_pct",
        "hard_rows", "hard_second_right", "hard_first_right",
        "unreadable_responses", "cost_per_1000_usd", "run_total_usd", "fully_priced",
        "seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["first_pass", "second_check", "error", *keys])
        for r in results:
            w.writerow([
                r.candidate.first.name,
                r.candidate.second.name,
                r.error or "",
                *(r.row.get(k, "") for k in keys),
            ])


# --- main -----------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run candidate models over the same rows and compare them.",
    )
    p.add_argument("--dataset", required=True, help="path to .xlsx or .csv")
    p.add_argument("--rows", type=int, default=200,
                   help="how many 'Other' rows to classify per candidate, in file order (default 200)")
    p.add_argument("--first-pass", nargs="+", required=True, metavar="MODEL",
                   help="OpenRouter ids to try as the first pass")
    p.add_argument("--second-check", nargs="+", metavar="MODEL",
                   default=[config.MODEL_B.name],
                   help=f"OpenRouter ids to try as the second check (default: {config.MODEL_B.name})")
    p.add_argument("--target", type=float, default=DEFAULT_TARGET_PCT,
                   help=f"auto-accept accuracy bar for the recommendation (default {DEFAULT_TARGET_PCT:.0f})")
    p.add_argument("--yes", action="store_true", help="skip the confirmation before spending")
    args = p.parse_args(argv)
    if args.rows < 1:
        p.error("--rows must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import run_pipeline  # imported late: it installs the response cache

    names = list(dict.fromkeys([*args.first_pass, *args.second_check]))
    problems = check_candidates(names)
    if problems:
        print("Cannot run these candidates:\n  - " + "\n  - ".join(problems))
        return 2

    loaded = load_dataset(args.dataset)
    if not loaded.report.is_valid:
        print("Dataset failed validation:\n  - " + "\n  - ".join(loaded.report.errors))
        return 2
    eligible = loaded.report.other_rows
    rows = min(args.rows, eligible)
    has_key = not loaded.ground_truth.empty and all(
        c in loaded.ground_truth.columns for c in config.GROUND_TRUTH_COLUMNS
    )

    candidates = [
        Candidate(spec_for(a, second=False), spec_for(b, second=True))
        for a in args.first_pass
        for b in args.second_check
    ]

    ok, msg = probe_model(candidates[0].first)
    if not ok:
        print(msg)
        return 2

    print(f"Dataset: {args.dataset}  ({eligible} 'Other' rows; using the first {rows})")
    print(f"Answer key: {'yes' if has_key else 'NO, accuracy columns will be n/a'}")
    if Path(args.dataset).resolve() == config.DEFAULT_DATASET.resolve():
        print("Note: the sample's answer key is synthetic and contradicts itself on "
              "sub-reasons. Use it to rank candidates, not to quote accuracy.")
    for w in warnings_for(names):
        print(f"Warning: {w}")
    print(f"\n{len(candidates)} run(s), {rows} rows each, roughly {2 * rows} calls per run:")
    for c in candidates:
        print(f"  - {c.label}")
    if not args.yes:
        answer = input("\nThis calls paid models (cached rows are free). Continue? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled.")
            return 1

    results: list[Result] = []
    for i, cand in enumerate(candidates, 1):
        print(f"\n[{i}/{len(candidates)}] {cand.label}")

        def tick(msg: str, frac: float) -> None:
            print(f"    {frac:5.0%}  {msg}", flush=True)

        try:
            payload = run_pipeline.run(
                dataset=args.dataset,
                limit=rows,
                progress=tick,
                model_a=cand.first,
                model_b=cand.second,
            )
            results.append(Result(cand, row=summarise(payload, loaded.ground_truth)))
        except KeyboardInterrupt:
            print("\nStopped. Results so far are below.")
            break
        except BaseException as exc:  # noqa: BLE001 - one bad candidate must not end the comparison
            results.append(Result(cand, error=f"{type(exc).__name__}: {str(exc)[:160]}"))

    if not results:
        return 1

    print("\n" + render_table(results))
    print(
        "\nHard rows = the rows sent for a second check. Each stage's own answer is "
        "scored, whether or not it was accepted. There are few of them, so treat "
        "small differences as noise."
    )
    stamp = time.strftime("%Y-%m-%d-%H%M%S")
    out = COMPARISONS_DIR / f"{stamp}.csv"
    write_csv(results, out)
    print(f"\n{recommend(results, args.target)}")
    print(f"Saved: {out.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
