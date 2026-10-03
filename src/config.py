"""Configuration: model selection, routing thresholds, price assumptions.

DETERMINISTIC MODULE — no model calls happen here.

Everything a reviewer might want to change lives here or in `.env`. Nothing in
this file hardcodes a provider: `init_chat_model` resolves `provider:name`
pairs at runtime, so swapping Anthropic for OpenAI or Google is an env-var
edit, not a code change (spec section 14).

No secrets are stored here. Keys are read from the environment only, and are
never logged or echoed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

#: Canonical run artifact. Committed so the deployed app renders cold with no
#: API key and no live calls (premise P7). Contains predictions only — never
#: the hidden `expected_*` columns (premise P6, enforced in data_io).
RUN_ARTIFACT = OUTPUT_DIR / "run_latest.json"

#: Session-keyed reviewer corrections + the LLM response cache (decision D5).
STATE_DB = OUTPUT_DIR / "state.sqlite3"

#: Source dataset. The .xlsx is read directly — there is no manual CSV
#: conversion step to get wrong.
DEFAULT_DATASET = DATA_DIR / "dhaga_returns_intelligence_synthetic_1000.xlsx"

#: Preferred worksheet in an uploaded workbook. Used when present; otherwise
#: the first sheet is read, because uploaded files name their sheets anything.
DATASET_SHEET = "Returns_Data"


# --- Model configuration --------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One configured model, provider-agnostic.

    `temperature` is deliberately optional. Spec section 15 asks for explicit
    temperatures, but several current reasoning-class models across providers
    REJECT the parameter outright (an Anthropic 5-family model returns a 400 if
    sent one). So the honest configuration is: set it where the provider
    accepts it, leave it None where the provider removed it, and say which in
    the README. Sending it blindly would break the run; omitting it silently
    would fail spec section 31's "temperatures are explicitly configured".

    `structured_output_method` is likewise optional. The accepted values differ
    per provider (Anthropic: function_calling | json_schema; OpenAI's set
    differs), so leaving it None uses each provider's own default and keeps the
    config portable. Set it only when tuning a specific provider.
    """

    provider: str
    name: str
    temperature: float | None = None
    structured_output_method: str | None = None
    max_tokens: int | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.name}"

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.key


def _env_float(key: str) -> float | None:
    """Read an optional float env var. Empty or 'none' means 'do not send'."""
    raw = os.getenv(key, "").strip()
    if raw == "" or raw.lower() == "none":
        return None
    return float(raw)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    return int(raw) if raw else default


def _env_str(key: str) -> str | None:
    raw = os.getenv(key, "").strip()
    return raw or None


# --- OpenRouter -----------------------------------------------------------
#
# OpenRouter speaks the OpenAI wire protocol, so it is reached through the
# `openai` integration with a different base_url. Treating it as its own
# provider name keeps the config readable and keeps the base_url in one place.
#
# Why it is the default here: it fronts 460+ models behind one key, which makes
# the planned eval-and-cost sweep a config change rather than a credential
# hunt. Swapping back to a first-party provider is still just an env edit.

PROVIDER_OPENROUTER = "openrouter"
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).strip()

#: The key env var name is read as the user actually set it. Several spellings
#: are accepted so a reasonable guess works.
OPENROUTER_KEY_ENV: tuple[str, ...] = (
    "OPEN_ROUTER_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
)


def openrouter_api_key() -> str | None:
    """Read the OpenRouter key from whichever supported env name is set."""
    for name in OPENROUTER_KEY_ENV:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


#: Model A — the high-volume first pass (spec section 14).
#:
#: `mistralai/mistral-nemo`. Every row goes through this model twice, so its cost
#: and its sub-reason accuracy dominate the run. It replaced a cheaper model
#: that was unstable on sub-reasons: two runs over the same 440 rows disagreed
#: on 5.9% of them, including calling "kapda bahut patla hai" (thin fabric)
#: a stitching problem. Still about 4 cents per 1,000 comments.
MODEL_A = ModelSpec(
    provider=os.getenv("MODEL_A_PROVIDER", PROVIDER_OPENROUTER).strip(),
    name=os.getenv("MODEL_A_NAME", "mistralai/mistral-nemo").strip(),
    temperature=_env_float("MODEL_A_TEMPERATURE"),
    structured_output_method=_env_str("MODEL_A_STRUCTURED_OUTPUT_METHOD"),
    # Reasoning models spend output tokens thinking before they emit the
    # structured answer, and that thinking counts against max_tokens. At 1024
    # this model truncated mid-JSON on ~8% of rows, which surfaced as a parse
    # failure and a fallback sub-reason rather than an error anyone would see.
    max_tokens=_env_int("MODEL_A_MAX_TOKENS", 4096),
)

#: Model B — the escalation path (spec section 14).
#:
#: `openai/gpt-oss-120b`. Roughly 50x Model A's input price, which is affordable
#: only because it sees the few per cent of rows the first pass could not
#: settle. That ratio is the whole economic argument for the cascade, and the
#: Run Cost screen reports it so the day it stops holding is visible.
MODEL_B = ModelSpec(
    provider=os.getenv("MODEL_B_PROVIDER", PROVIDER_OPENROUTER).strip(),
    name=os.getenv("MODEL_B_NAME", "openai/gpt-oss-120b").strip(),
    temperature=_env_float("MODEL_B_TEMPERATURE"),
    structured_output_method=_env_str("MODEL_B_STRUCTURED_OUTPUT_METHOD"),
    max_tokens=_env_int("MODEL_B_MAX_TOKENS", 2048),
)


# --- Routing thresholds (spec section 13) ---------------------------------
#
# These are starting assumptions, not tuned values, and the spec requires them
# to be configurable. Treat `confidence` as a ranking signal rather than a
# calibrated probability (premise P2 as revised).

#: Accept Model A when confidence >= this AND needs_review is False.
ACCEPT_THRESHOLD_A: float = float(os.getenv("ACCEPT_THRESHOLD_A", "0.85"))

#: Accept Model B when confidence >= this AND needs_review is False AND it
#: agrees with Model A on the primary reason.
ACCEPT_THRESHOLD_B: float = float(os.getenv("ACCEPT_THRESHOLD_B", "0.80"))

#: Human-review budget, expressed in Neha's time rather than as a bare count.
#: This per-row figure is an ASSUMPTION and is flagged as such in the UI; the
#: real number comes from timing a human labelling the 22 ambiguous rows (see
#: "The Assignment" in docs/designs/return-reason-intelligence.md).
REVIEW_MINUTES_PER_ROW: float = float(os.getenv("REVIEW_MINUTES_PER_ROW", "2.5"))


# --- Execution policy (decision D6) ---------------------------------------

#: Bounded concurrency for the batch run. Results are re-keyed by return_id so
#: output order is deterministic regardless of completion order.
MAX_CONCURRENCY: int = _env_int("MAX_CONCURRENCY", 8)

#: Transport retries for rate limits and 5xx, counted and reported separately
#: from schema failures so the two never get conflated in the metrics.
MAX_TRANSPORT_RETRIES: int = _env_int("MAX_TRANSPORT_RETRIES", 3)

#: Proactive client-side rate limit, requests/second. 0 disables.
REQUESTS_PER_SECOND: float = float(os.getenv("REQUESTS_PER_SECOND", "0") or 0)

#: Cache model responses in SQLite so re-running the same 440 rows during
#: development costs nothing. Provider prefix-caching is deliberately NOT used:
#: it is not portable across providers, and its minimum-prefix rules differ by
#: model in ways that fail silently.
ENABLE_RESPONSE_CACHE: bool = os.getenv("ENABLE_RESPONSE_CACHE", "1") not in {
    "0",
    "false",
    "False",
}

#: Fault injection for the schema-repair path (decision D4). Off by default and
#: settable ONLY here or via env — never reachable from the UI. Without it the
#: repair retry and the SCHEMA_FAILURE review reason are unreachable code,
#: because provider-side schema enforcement makes organic failures near-zero.
INJECT_SCHEMA_FAULTS: bool = os.getenv("INJECT_SCHEMA_FAULTS", "0") in {
    "1",
    "true",
    "True",
}


# --- Price assumptions (spec section 21) ----------------------------------
#
# LangChain normalises token USAGE across providers but deliberately does not
# price it, so the price table is ours to own and keep current.
#
# Prices come only from the OpenRouter snapshot, and cost is only ever looked up
# for the two models a run actually used. There is deliberately no hand-typed
# table: one goes stale silently and looks authoritative while it does.
#
# A model with no snapshot price (for example one pointed at a provider
# directly rather than through OpenRouter) is reported as UNAVAILABLE rather
# than as $0.00 — a silent zero would make an expensive run look free, which is
# exactly the kind of quiet wrongness spec section 23 is about.


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input_per_mtok: float | None
    output_per_mtok: float | None
    as_of: str = "unverified"

    @property
    def is_configured(self) -> bool:
        return self.input_per_mtok is not None and self.output_per_mtok is not None


#: Live-fetched OpenRouter rates are snapshotted here so a run prices itself
#: from real data without needing the network, and so the numbers on screen are
#: reproducible. Refresh with `python run_pipeline.py --refresh-prices`.
OPENROUTER_PRICE_SNAPSHOT = DATA_DIR / "openrouter_prices.json"

#: `provider:model` -> price, filled from the snapshot at import time.
PRICES: dict[str, Price] = {}


def _load_openrouter_snapshot() -> None:
    """Merge the OpenRouter price snapshot into PRICES, if one exists."""
    if not OPENROUTER_PRICE_SNAPSHOT.exists():
        return
    try:
        import json

        blob = json.loads(OPENROUTER_PRICE_SNAPSHOT.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a bad snapshot must not break startup
        return
    as_of = str(blob.get("fetched_at", "unknown"))
    for model_id, rates in (blob.get("models") or {}).items():
        try:
            pin = float(rates["input_per_mtok"])
            pout = float(rates["output_per_mtok"])
        except (KeyError, TypeError, ValueError):
            continue
        # OpenRouter lists routers (e.g. `typesafe/jev-router`) at -1 per
        # token: the price depends on which model the router picks. Treating
        # that as a rate would report a large negative cost, so such models
        # are left unpriced and show as "not priced".
        if pin < 0 or pout < 0:
            continue
        PRICES[f"{PROVIDER_OPENROUTER}:{model_id}"] = Price(pin, pout, as_of)


def refresh_openrouter_prices(timeout: int = 30) -> int:
    """Fetch OpenRouter's live model pricing and write the snapshot.

    OpenRouter publishes per-token prices on its public models endpoint, so the
    cost table can be grounded in real rates instead of hand-maintained. Prices
    are stored per million tokens to match how the UI reports them.

    Returns the number of models written.
    """
    import json
    import urllib.request

    req = urllib.request.Request(
        f"{OPENROUTER_BASE_URL}/models",
        headers={"User-Agent": "dhaga-return-intelligence/0.1"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)["data"]

    models: dict[str, dict] = {}
    for entry in data:
        pricing = entry.get("pricing") or {}
        try:
            per_token_in = float(pricing.get("prompt") or 0)
            per_token_out = float(pricing.get("completion") or 0)
        except (TypeError, ValueError):
            continue
        params = set(entry.get("supported_parameters") or [])
        models[entry["id"]] = {
            "input_per_mtok": per_token_in * 1_000_000,
            "output_per_mtok": per_token_out * 1_000_000,
            "context_length": entry.get("context_length"),
            "supports_tools": "tools" in params,
            "supports_structured_outputs": "structured_outputs" in params,
        }

    OPENROUTER_PRICE_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    OPENROUTER_PRICE_SNAPSHOT.write_text(
        json.dumps(
            {
                "fetched_at": __import__("datetime")
                .datetime.now(__import__("datetime").timezone.utc)
                .isoformat(timespec="seconds"),
                "source": f"{OPENROUTER_BASE_URL}/models",
                "models": models,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _load_openrouter_snapshot()
    return len(models)


def openrouter_candidates(
    require_structured_output: bool = True,
) -> list[tuple[str, float, float]]:
    """Cheapest-first OpenRouter models usable by this pipeline.

    Feeds the planned cost matrix: returns `(model_id, in_per_mtok,
    out_per_mtok)` for paid models that support the capabilities the two-step
    chain actually needs. Reads the snapshot; call `refresh_openrouter_prices`
    first if it is missing or stale.
    """
    import json

    if not OPENROUTER_PRICE_SNAPSHOT.exists():
        return []
    blob = json.loads(OPENROUTER_PRICE_SNAPSHOT.read_text(encoding="utf-8"))
    out: list[tuple[str, float, float]] = []
    for model_id, rates in (blob.get("models") or {}).items():
        if not rates.get("supports_tools"):
            continue
        if require_structured_output and not rates.get(
            "supports_structured_outputs"
        ):
            continue
        pin = float(rates.get("input_per_mtok") or 0)
        pout = float(rates.get("output_per_mtok") or 0)
        if pin <= 0:
            continue  # free tiers are heavily rate-limited; exclude by default
        out.append((model_id, pin, pout))
    out.sort(key=lambda r: r[1] + r[2])
    return out


_load_openrouter_snapshot()


def price_for(spec: ModelSpec) -> Price:
    """Look up a model's configured price, or an unconfigured placeholder."""
    return PRICES.get(spec.key, Price(None, None))


def unpriced_models() -> list[str]:
    """Configured models whose price is unknown, for an explicit UI warning."""
    return [s.key for s in (MODEL_A, MODEL_B) if not price_for(s).is_configured]


# --- Required input columns (spec section 8) ------------------------------

REQUIRED_COLUMNS: tuple[str, ...] = (
    "return_id",
    "order_id",
    "sku",
    "product_name",
    "category",
    "vendor",
    "return_reason_dropdown",
    "return_date",
)

OPTIONAL_COLUMNS: tuple[str, ...] = ("size_ordered", "return_reason_text")

#: Evaluation-only columns. The classifier never receives them because
#: `ReturnRecord` has no field that could hold one, not because they are
#: filtered out. `has_answer_key` in the API checks for them before scoring.
GROUND_TRUTH_COLUMNS: tuple[str, ...] = (
    "expected_primary_reason",
    "expected_sub_reason",
    "expected_body_area",
)

#: The dropdown value whose free text we classify (premise P1).
OTHER_DROPDOWN_VALUE: str = "Other"


@dataclass
class RunConfig:
    """A snapshot of everything that shaped one run, persisted with it.

    Written into the run artifact so a reviewer can tell which models,
    thresholds, and price assumptions produced the numbers on screen.
    """

    model_a: ModelSpec = field(default_factory=lambda: MODEL_A)
    model_b: ModelSpec = field(default_factory=lambda: MODEL_B)
    accept_threshold_a: float = ACCEPT_THRESHOLD_A
    accept_threshold_b: float = ACCEPT_THRESHOLD_B
    max_concurrency: int = MAX_CONCURRENCY
    max_transport_retries: int = MAX_TRANSPORT_RETRIES
    inject_schema_faults: bool = INJECT_SCHEMA_FAULTS

    def to_dict(self) -> dict:
        return {
            "model_a": self.model_a.key,
            "model_a_temperature": self.model_a.temperature,
            "model_b": self.model_b.key,
            "model_b_temperature": self.model_b.temperature,
            "accept_threshold_a": self.accept_threshold_a,
            "accept_threshold_b": self.accept_threshold_b,
            "max_concurrency": self.max_concurrency,
            "max_transport_retries": self.max_transport_retries,
            "inject_schema_faults": self.inject_schema_faults,
        }
