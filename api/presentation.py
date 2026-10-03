"""Presentation layer: what the browser is allowed to see.

DETERMINISTIC MODULE — no model calls, no I/O.

The UI deliberately does not name the models or refer to them as "Model A" /
"Model B". The cascade is described as a **first pass** and a **second
check**. That rule is enforced here, on the server, rather than in the React
code, so a model name can never leak through the network tab, a CSV export or
an error message.

Everything in this module is a pure function of the run artifact and the
configured model keys, which makes the guarantee unit-testable
(`tests/test_api.py`).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from src import config

# --- vocabulary -----------------------------------------------------------

FIRST = "first pass"
SECOND = "second check"

#: Internal status code -> public status code.
STATUS = {
    "CLASSIFIED_MODEL_A": "AUTO_ACCEPTED",
    "CLASSIFIED_MODEL_B": "ACCEPTED_AFTER_CHECK",
    "CLASSIFIED_HUMAN": "RESOLVED_BY_REVIEWER",
    "NEEDS_HUMAN_REVIEW": "NEEDS_REVIEW",
    "PROCESSING_ERROR": "PROCESSING_ERROR",
    "INVALID_INPUT": "INVALID_INPUT",
}

#: Internal review reason -> public review reason.
REVIEW_REASON = {
    "MODEL_DISAGREEMENT": "DISAGREEMENT",
}

#: Internal trace step -> public trace step.
TRACE_STEP = {
    "model_a_primary": "first_pass",
    "router_a": "routing_after_first_pass",
    "model_b_primary": "second_check",
    "router_b": "routing_after_second_check",
    "attributes": "attributes",
}


def public_status(status: Any) -> Any:
    return STATUS.get(status, status) if isinstance(status, str) else status


def public_review_reason(reason: Any) -> Any:
    return REVIEW_REASON.get(reason, reason) if isinstance(reason, str) else reason


# --- free-text scrubbing --------------------------------------------------


def model_identifiers(extra: Iterable[str] = ()) -> list[str]:
    """Every string that would identify a model: `provider:name` and `name`.

    Longest first so `openrouter:openai/gpt-oss-120b` is removed before
    `openai/gpt-oss-120b` would leave a dangling `openrouter:` behind.
    """
    ids: set[str] = set()
    for key in [config.MODEL_A.key, config.MODEL_B.key, *extra]:
        if not key:
            continue
        ids.add(key)
        _, _, name = key.partition(":")
        if name:
            ids.add(name)
    return sorted(ids, key=len, reverse=True)


_MODEL_REF = re.compile(r"\bModel ([AB])\b")
_SENTENCE_START = re.compile(r"(^|[.!?]\s+)$")


def scrub(text: Any, extra_ids: Iterable[str] = ()) -> Any:
    """Rewrite one piece of free text so it names no model.

    `Model A` becomes "the first pass" (or "The first pass" at the start of a
    sentence), `Model B` becomes "the second check", and any configured model
    identifier is replaced with a neutral word.
    """
    if not isinstance(text, str) or not text:
        return text

    for ident in model_identifiers(extra_ids):
        text = text.replace(ident, "the classifier")

    def repl(m: re.Match) -> str:
        phrase = FIRST if m.group(1) == "A" else SECOND
        at_start = _SENTENCE_START.search(text[: m.start()]) is not None
        return ("The " if at_start else "the ") + phrase

    text = _MODEL_REF.sub(repl, text)
    # Router wording that only makes sense when models are named.
    text = text.replace("Two models disagreeing", "Two readings disagreeing")
    text = text.replace("two models", "two passes")
    text = text.replace("Accepted from the ", "Accepted on the ")
    return text


# --- records --------------------------------------------------------------

#: Record fields that are never sent to the browser.
_HIDDEN_RECORD_FIELDS = {"classifier_model"}


def public_trace(trace: Any, extra_ids: Iterable[str] = ()) -> list[dict]:
    out: list[dict] = []
    for step in trace or []:
        if not isinstance(step, dict):
            continue
        clean = {
            k: scrub(v, extra_ids) if isinstance(v, str) else v
            for k, v in step.items()
            if k != "model"
        }
        clean["step"] = TRACE_STEP.get(step.get("step"), scrub(step.get("step")))
        # Provider error text can name a model in forms no list can predict
        # (display names, versioned ids), so the browser gets a fixed line.
        if clean.get("error"):
            clean["error"] = "the classifier returned an error"
        if "reason" in clean:
            clean["reason"] = public_review_reason(clean["reason"])
        out.append(clean)
    return out


def public_record(row: dict, extra_ids: Iterable[str] = ()) -> dict:
    """One classification row, safe to send to the browser."""
    out: dict = {}
    for key, value in row.items():
        if key in _HIDDEN_RECORD_FIELDS:
            continue
        if key == "trace":
            out[key] = public_trace(value, extra_ids)
        elif key == "status":
            out[key] = public_status(value)
        elif key == "review_reason":
            out[key] = public_review_reason(value)
        elif key in {"routing_detail"}:
            out[key] = scrub(value, extra_ids)
        else:
            out[key] = value
    return out


# --- run-level blocks -----------------------------------------------------


def stage_label(model_key: str, run_config: dict) -> str:
    """Which cascade stage a usage/cost row belongs to, without naming it."""
    if model_key == run_config.get("model_a"):
        return "First pass"
    if model_key == run_config.get("model_b"):
        return "Second check"
    return "Other"


def run_ids(payload: dict) -> list[str]:
    """Model identifiers recorded in a run, so they can be scrubbed too.

    The run on screen may have been produced by a different configuration than
    the one currently in `.env`, so its own keys are scrubbed as well.
    """
    cfg = (payload.get("run") or {}).get("config") or {}
    ids = [cfg.get("model_a"), cfg.get("model_b")]
    ids += list((payload.get("usage") or {}).keys())
    return [i for i in ids if i]


def public_run(payload: dict) -> dict:
    """The run banner: thresholds, prompt version, timing. No model keys."""
    run = payload["run"]
    cfg = run.get("config") or {}
    return {
        "finished_at": run.get("finished_at"),
        "elapsed_seconds": run.get("elapsed_seconds"),
        "eligible_rows": run.get("eligible_rows"),
        "prompt_version": run.get("prompt_version"),
        "accept_threshold_first_pass": cfg.get("accept_threshold_a"),
        "accept_threshold_second_check": cfg.get("accept_threshold_b"),
    }


def public_cost(cost: dict, run_config: dict) -> dict:
    per_stage = []
    for row in cost.get("per_model") or []:
        clean = {k: v for k, v in row.items() if k != "model"}
        clean["stage"] = stage_label(row.get("model", ""), run_config)
        per_stage.append(clean)
    unpriced = [
        stage_label(m, run_config) for m in cost.get("unpriced_models") or []
    ]
    return {
        **{k: v for k, v in cost.items() if k not in {"per_model", "unpriced_models"}},
        "per_stage": per_stage,
        "unpriced_stages": unpriced,
    }


def public_escalation(esc: dict) -> dict:
    return {
        "total_calls": esc.get("total_model_calls", 0),
        "second_check_calls": esc.get("model_b_calls", 0),
        "pct_calls_on_second_check": esc.get("pct_calls_on_model_b", 0.0),
    }


def public_usage(usage: dict, run_config: dict) -> list[dict]:
    rows = []
    for key, entry in (usage or {}).items():
        clean = {k: v for k, v in entry.items() if k != "model"}
        rows.append({"stage": stage_label(key, run_config), **clean})
    return rows


def public_overview(o: dict) -> dict:
    return {
        "total_returns_loaded": o["total_returns_loaded"],
        "total_other": o["total_other"],
        "pct_other": o["pct_other"],
        "auto_accepted": o["classified_model_a"],
        "accepted_after_check": o["classified_model_b"],
        "auto_classified": o["classified_model_a"] + o["classified_model_b"],
        "needs_review": o["needs_human_review"],
        "resolved_by_reviewer": o["resolved_by_human"],
        "processing_error": o["processing_error"],
        "invalid_input": o["invalid_input"],
    }


def public_evaluation(ev: dict, extra_ids: Iterable[str] = ()) -> dict:
    """Rename the evaluation buckets; bucket labels are scrubbed too."""
    if not ev.get("available"):
        return ev

    def bucket(b: dict) -> dict:
        return {**b, "label": scrub(b.get("label"), extra_ids)}

    auto_accept = {
        **ev["auto_accept_precision_a"],
        "label": "Accuracy of rows auto-accepted on the first pass",
    }
    overall = bucket(ev["primary_accuracy_overall"])
    return {
        "available": True,
        "rows_evaluated": ev["rows_evaluated"],
        "buckets": [
            overall,
            bucket(ev["primary_accuracy_automated"]),
            auto_accept,
            {**ev["accuracy_after_b"],
             "label": "Accuracy of rows accepted after the second check"},
            bucket(ev["sub_reason_accuracy"]),
            bucket(ev["body_area_accuracy"]),
        ],
        "auto_accept_precision": auto_accept,
        "primary_accuracy_overall": overall,
        "rates": {
            "auto_accept_pct": ev["rates"]["auto_accept_pct"],
            "second_check_pct": ev["rates"]["escalated_pct"],
            "review_pct": ev["rates"]["human_review_pct"],
        },
        "confusion_matrix": ev["confusion_matrix"],
        "notable_confusions": ev["notable_confusions"],
    }


_PROGRESS = re.compile(r"^Model ([AB]) (.*)$")


def progress_message(text: str) -> str:
    """Pipeline progress lines, e.g. `Model A step 1/2 on 40 rows` becomes
    `First pass: step 1/2 on 40 rows`."""
    m = _PROGRESS.match(text or "")
    if m:
        stage = "First pass" if m.group(1) == "A" else "Second check"
        return f"{stage}: {scrub(m.group(2))}"
    return scrub(text)


def probe_message(ok: bool, message: str, extra_ids: Iterable[str] = ()) -> str:
    """Pre-flight credential check, without naming the model it probed."""
    if ok:
        return "Classifier ready: credential present."
    return scrub(message, extra_ids)
