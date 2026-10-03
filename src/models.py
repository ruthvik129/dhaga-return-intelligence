"""The model layer: the only module in this project that calls an LLM.

Everything else is deterministic. The boundary is deliberately this sharp so
the README can answer spec section 16 ("which code is deterministic, which is
the model") by pointing at one file.

Provider-agnostic by construction. `init_chat_model("provider:name")` resolves
the pair at runtime, so switching from Anthropic to OpenAI or Google is an
env-var edit (spec section 14). Nothing below imports a provider package.

Two things are worth reading before changing this file.

`include_raw=True` is load-bearing, not stylistic. With the default
`include_raw=False`, a structured-output parse failure RAISES, which would make
the spec's required schema-failure path an exception handler and throw away the
response. With `include_raw=True` we get `{"raw", "parsed", "parsing_error"}`:
the error becomes data we can route on, and `raw` carries the token usage that
spec section 21 needs. Both requirements point at the same flag.

pydantic still validates behind the provider's own enforcement, because JSON
Schema cannot express numeric bounds — the `confidence` 0.0-1.0 range is not
server-enforceable, so the second layer is doing real work rather than
belt-and-braces (see `schemas.ConfidenceFloat`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.globals import set_llm_cache
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig

from . import config, prompts
from .config import ModelSpec
from .data_io import ReturnRecord, SQLiteResponseCache
from .router import StepOutcome
from .schemas import (
    AttributeExtraction,
    PrimaryClassification,
    PrimaryReason,
    attribute_schema_for,
)

_CACHE_INSTALLED = False


def install_response_cache() -> None:
    """Install the SQLite response cache once per process.

    Exact-match response caching, not provider prefix-caching: prefix caching
    is not portable across providers and its minimum-prefix rules differ per
    model in ways that fail silently. This makes re-running the same 440 rows
    during development free.
    """
    global _CACHE_INSTALLED
    if _CACHE_INSTALLED or not config.ENABLE_RESPONSE_CACHE:
        return
    set_llm_cache(SQLiteResponseCache())
    _CACHE_INSTALLED = True


# --- Usage and cost ledger (spec section 21) ------------------------------


@dataclass
class ModelUsage:
    """Token and call totals for one configured model."""

    model_key: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0
    transport_retries: int = 0
    schema_repairs: int = 0
    schema_failures: int = 0
    errors: int = 0
    measured: bool = True

    def add(self, usage_metadata: dict | None) -> None:
        """Accumulate one call's usage.

        LangChain normalises usage across providers: `input_tokens` is the TOTAL
        input including cached reads, with the cached portion broken out under
        `input_token_details`. That differs from some providers' raw shape,
        where the top-level figure excludes cache reads — summing the wrong
        field there under-reports cost. Using the normalised field is one of
        the concrete wins from going through LangChain.
        """
        self.calls += 1
        if not usage_metadata:
            self.measured = False
            return
        self.input_tokens += int(usage_metadata.get("input_tokens") or 0)
        self.output_tokens += int(usage_metadata.get("output_tokens") or 0)
        in_details = usage_metadata.get("input_token_details") or {}
        out_details = usage_metadata.get("output_token_details") or {}
        self.cache_read_tokens += int(in_details.get("cache_read") or 0)
        self.cache_creation_tokens += int(in_details.get("cache_creation") or 0)
        self.reasoning_tokens += int(out_details.get("reasoning") or 0)

    def to_dict(self) -> dict:
        return {
            "model": self.model_key,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "transport_retries": self.transport_retries,
            "schema_repairs": self.schema_repairs,
            "schema_failures": self.schema_failures,
            "errors": self.errors,
            "token_counts_measured": self.measured,
        }


@dataclass
class UsageLedger:
    """Per-model usage for a whole run."""

    by_model: dict[str, ModelUsage] = field(default_factory=dict)

    def for_model(self, spec: ModelSpec) -> ModelUsage:
        return self.by_model.setdefault(spec.key, ModelUsage(model_key=spec.key))

    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self.by_model.items()}


# --- Fault injection (decision D4) ----------------------------------------


def _injected_fault(return_id: str, attempt: int) -> bool:
    """Deterministically force a schema failure, when explicitly enabled.

    Provider-side schema enforcement makes organic schema failures near-zero,
    which would leave the repair-retry path and the SCHEMA_FAILURE review
    reason as unreachable, untested code while spec section 20 still asks us to
    report a schema-failure rate. This makes both reachable on demand.

    Off unless `INJECT_SCHEMA_FAULTS=1`, settable only via env or config and
    never from the UI. Deterministic in `return_id` so a run is reproducible:
    roughly 1 row in 20 fails its first attempt (exercising repair), and
    roughly 1 in 50 fails both (exercising SCHEMA_FAILURE).
    """
    if not config.INJECT_SCHEMA_FAULTS:
        return False
    digest = hashlib.sha256(return_id.encode("utf-8")).digest()
    bucket = digest[0] % 50
    if attempt == 1:
        return bucket % 20 == 0 or bucket == 7
    return bucket == 7


# --- Chat model construction ----------------------------------------------


class ModelUnavailable(RuntimeError):
    """Raised when a configured model cannot be constructed.

    Carries the provider's own message so the UI can show the real cause — a
    missing credential reads very differently from an unknown model id, and
    collapsing both into "model unavailable" wastes the user's time.
    """


def build_chat_model(spec: ModelSpec) -> Any:
    """Construct a chat model from a provider-agnostic spec.

    `temperature` and `max_tokens` are passed only when configured. Sending
    `temperature` to a model that removed sampling parameters returns a 400, so
    "unset" has to mean "do not send the field" rather than "send None"
    (spec section 15; see `ModelSpec.temperature`).
    """
    kwargs: dict[str, Any] = {}
    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature
    if spec.max_tokens is not None:
        kwargs["max_tokens"] = spec.max_tokens
    if config.MAX_TRANSPORT_RETRIES:
        kwargs["max_retries"] = config.MAX_TRANSPORT_RETRIES
    if config.REQUESTS_PER_SECOND > 0:
        kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=config.REQUESTS_PER_SECOND,
            check_every_n_seconds=0.05,
            max_bucket_size=max(1.0, config.REQUESTS_PER_SECOND),
        )

    provider = spec.provider
    if provider == config.PROVIDER_OPENROUTER:
        # OpenRouter speaks the OpenAI wire protocol, so it is reached through
        # the `openai` integration pointed at a different base_url. This is the
        # only provider-specific branch in the codebase.
        api_key = config.openrouter_api_key()
        if not api_key:
            raise ModelUnavailable(
                f"Could not initialise {spec.key}: no OpenRouter key found. "
                f"Set one of {', '.join(config.OPENROUTER_KEY_ENV)} in .env."
            )
        provider = "openai"
        kwargs["base_url"] = config.OPENROUTER_BASE_URL
        kwargs["api_key"] = api_key

    try:
        return init_chat_model(spec.name, model_provider=provider, **kwargs)
    except Exception as exc:  # noqa: BLE001 - message is shown to the user
        raise ModelUnavailable(
            f"Could not initialise {spec.key}: {exc}"
        ) from exc


def credential_present(spec: ModelSpec) -> bool | None:
    """Is a credential available for this model's provider?

    Only OpenRouter is checked, because OpenRouter is the only gateway this
    project configures: one key fronts every model, so the per-provider
    credential table that used to live here (Anthropic / OpenAI / Google env
    names) was dead weight the moment the gateway became the default.

    Returns None for any other provider — we do not know its convention, and
    guessing an env var name would turn "unknown" into a false "missing".
    """
    if spec.provider == config.PROVIDER_OPENROUTER:
        return config.openrouter_api_key() is not None
    return None


def probe_model(spec: ModelSpec) -> tuple[bool, str]:
    """Check a model is usable, without making a network call.

    Construction alone is not a sufficient check: `init_chat_model` builds the
    client lazily and does NOT validate credentials, so a model with no API key
    constructs happily and fails on first use — mid-run, after partial work.
    We therefore also check the gateway credential, so the app can show an
    honest empty state up front.

    What this still cannot catch is a key that is present but invalid or
    revoked; only a real call reveals that.
    """
    try:
        build_chat_model(spec)
    except ModelUnavailable as exc:
        return False, str(exc)

    has_cred = credential_present(spec)
    if has_cred is False:
        names = ", ".join(config.OPENROUTER_KEY_ENV)
        return False, (
            f"{spec.key} constructed, but no OpenRouter key found. Set one of "
            f"{names} in your .env to run live classification."
        )
    if has_cred is None:
        return True, (
            f"{spec.key} initialised. Provider '{spec.provider}' is not routed "
            f"through OpenRouter, so its credential was not verified here."
        )
    return True, f"{spec.key} initialised, credential present."


def _structured(model: Any, schema: type, spec: ModelSpec) -> Any:
    """Wrap a model in structured output, honouring an optional method override.

    `method` values differ per provider (Anthropic accepts
    function_calling | json_schema; other providers differ), so it is left
    unset by default and each provider uses its own default. This is the one
    place provider differences leak through the abstraction.
    """
    kwargs: dict[str, Any] = {"include_raw": True}
    if spec.structured_output_method:
        kwargs["method"] = spec.structured_output_method
    return model.with_structured_output(schema, **kwargs)


# --- The two chain steps (spec section 12) --------------------------------


@dataclass
class StepResult:
    """One model step's outcome plus the evidence needed to explain it."""

    outcome: StepOutcome
    parsed: Any | None
    repaired: bool = False
    raw_text: str | None = None


class Classifier:
    """Runs the two-step chain for one configured model.

    Deliberately two separate calls rather than one combined prompt
    (spec section 12): step B is given the already-decided primary reason and
    only the sub-reasons legal beneath it, which keeps each schema small and
    makes a wrong sub-reason attributable to the step that produced it.
    """

    def __init__(self, spec: ModelSpec, ledger: UsageLedger) -> None:
        self.spec = spec
        self.ledger = ledger
        install_response_cache()
        self.model = build_chat_model(spec)
        self._step_a = _structured(self.model, PrimaryClassification, spec)

    # -- step A ----------------------------------------------------------
    def classify_primary_batch(
        self, records: list[ReturnRecord]
    ) -> list[StepResult]:
        """Classify the primary reason for many records, bounded-concurrently.

        `.batch()` preserves input order and `return_exceptions=True` turns a
        failing row into a value rather than killing the whole batch — which is
        what spec section 23's "never silently drop a record" needs.
        """
        if not records:
            return []
        messages = [
            [
                SystemMessage(prompts.STEP_A_SYSTEM),
                HumanMessage(
                    prompts.step_a_user(
                        r.text,
                        dropdown_reason=r.dropdown_reason,
                        category=r.category,
                    )
                ),
            ]
            for r in records
        ]
        raw_results = self._step_a.batch(
            messages,
            config=RunnableConfig(max_concurrency=config.MAX_CONCURRENCY),
            return_exceptions=True,
        )
        return [
            self._finish_step_a(rec, res, msgs)
            for rec, res, msgs in zip(records, raw_results, messages, strict=True)
        ]

    def _finish_step_a(
        self, record: ReturnRecord, result: Any, messages: list
    ) -> StepResult:
        usage = self.ledger.for_model(self.spec)

        if isinstance(result, BaseException):
            usage.errors += 1
            return StepResult(
                outcome=StepOutcome(
                    schema_ok=False, error=f"{type(result).__name__}: {result}"
                ),
                parsed=None,
            )

        parsed, err, raw = _unpack(result)
        usage.add(_usage_of(raw))

        if _injected_fault(record.return_id, attempt=1):
            parsed, err = None, "injected schema fault (attempt 1)"

        repaired = False
        if parsed is None:
            usage.schema_repairs += 1
            repaired = True
            retry = list(messages) + [
                HumanMessage(prompts.repair_suffix(str(err)))
            ]
            try:
                retry_result = self._step_a.invoke(retry)
                parsed, err, raw2 = _unpack(retry_result)
                usage.add(_usage_of(raw2))
            except Exception as exc:  # noqa: BLE001
                parsed, err = None, f"{type(exc).__name__}: {exc}"
            if _injected_fault(record.return_id, attempt=2):
                parsed, err = None, "injected schema fault (attempt 2)"

        if parsed is None:
            usage.schema_failures += 1
            return StepResult(
                outcome=StepOutcome(schema_ok=False, error=str(err)),
                parsed=None,
                repaired=repaired,
            )

        return StepResult(
            outcome=StepOutcome(
                schema_ok=True,
                primary=parsed.primary_reason.value,
                confidence=float(parsed.confidence),
                needs_review=bool(parsed.needs_review),
            ),
            parsed=parsed,
            repaired=repaired,
        )

    # -- step B ----------------------------------------------------------
    def extract_attributes_batch(
        self, items: list[tuple[ReturnRecord, PrimaryReason]]
    ) -> list[StepResult]:
        """Extract sub-reason, body area, and evidence for already-classified rows.

        Grouped by primary reason because each primary gets its own schema (one
        narrowed to exactly its legal sub-reasons), so one `.batch()` call per
        primary rather than per row.
        """
        if not items:
            return []
        results: list[StepResult | None] = [None] * len(items)
        by_primary: dict[PrimaryReason, list[int]] = {}
        for i, (_, primary) in enumerate(items):
            by_primary.setdefault(primary, []).append(i)

        for primary, indices in by_primary.items():
            schema = attribute_schema_for(primary)
            runnable = _structured(self.model, schema, self.spec)
            messages = [
                [
                    SystemMessage(prompts.step_b_system(primary)),
                    HumanMessage(prompts.step_b_user(items[i][0].text, primary)),
                ]
                for i in indices
            ]
            raw_results = runnable.batch(
                messages,
                config=RunnableConfig(max_concurrency=config.MAX_CONCURRENCY),
                return_exceptions=True,
            )
            for i, res, msgs in zip(indices, raw_results, messages, strict=True):
                results[i] = self._finish_step_b(
                    items[i][0], primary, res, msgs, runnable
                )

        return [r for r in results if r is not None]

    def _finish_step_b(
        self,
        record: ReturnRecord,
        primary: PrimaryReason,
        result: Any,
        messages: list,
        runnable: Any,
    ) -> StepResult:
        usage = self.ledger.for_model(self.spec)

        if isinstance(result, BaseException):
            usage.errors += 1
            return StepResult(
                outcome=StepOutcome(
                    schema_ok=False,
                    primary=primary.value,
                    error=f"{type(result).__name__}: {result}",
                ),
                parsed=None,
            )

        parsed, err, raw = _unpack(result)
        usage.add(_usage_of(raw))

        repaired = False
        if parsed is None:
            usage.schema_repairs += 1
            repaired = True
            try:
                retry_result = runnable.invoke(
                    list(messages) + [HumanMessage(prompts.repair_suffix(str(err)))]
                )
                parsed, err, raw2 = _unpack(retry_result)
                usage.add(_usage_of(raw2))
            except Exception as exc:  # noqa: BLE001
                parsed, err = None, f"{type(exc).__name__}: {exc}"

        if parsed is None:
            usage.schema_failures += 1
            return StepResult(
                outcome=StepOutcome(
                    schema_ok=False, primary=primary.value, error=str(err)
                ),
                parsed=None,
                repaired=repaired,
            )

        return StepResult(
            outcome=StepOutcome(
                schema_ok=True,
                primary=primary.value,
                sub_reason=str(parsed.sub_reason),
                confidence=float(parsed.confidence),
                needs_review=bool(parsed.needs_review),
            ),
            parsed=parsed,
            repaired=repaired,
        )


# --- helpers --------------------------------------------------------------


def _unpack(result: Any) -> tuple[Any | None, Any | None, Any | None]:
    """Normalise an `include_raw=True` result into (parsed, error, raw)."""
    if isinstance(result, dict):
        return (
            result.get("parsed"),
            result.get("parsing_error"),
            result.get("raw"),
        )
    # A provider or version that returned the parsed object directly.
    return result, None, None


def _usage_of(raw: Any) -> dict | None:
    """Pull normalised usage metadata off a raw AIMessage, if present."""
    if raw is None:
        return None
    meta = getattr(raw, "usage_metadata", None)
    if isinstance(meta, dict):
        return meta
    return None


def evidence_is_grounded(evidence: str, source_text: str) -> bool:
    """True when `evidence` is genuinely a span of `source_text`.

    Spec section 10.1 requires evidence be grounded in the input and not
    invented. Compared case- and whitespace-insensitively so ordinary
    normalisation is not treated as fabrication. An empty evidence string is
    grounded by definition: it is the model correctly declining to claim
    support.
    """
    if not evidence or not evidence.strip():
        return True

    def norm(s: str) -> str:
        return " ".join(s.lower().split())

    return norm(evidence) in norm(source_text)


def reconcile_attribute_fallback(primary: PrimaryReason) -> str:
    """The least-specific legal sub-reason for a primary.

    Used when step B could not produce one but the row is still being recorded,
    so the stored sub-reason is always a legal value for its primary rather
    than a null that downstream grouping would have to special-case.
    """
    from .schemas import TAXONOMY

    allowed = TAXONOMY[primary]
    for preferred in (
        "unspecified_fit",
        "insufficient_information",
        "general_defect",
        "other_quality",
        "other_appearance",
        "other_delivery",
        "other_expectation_mismatch",
    ):
        if preferred in allowed:
            return preferred
    return allowed[-1]


__all__ = [
    "AttributeExtraction",
    "Classifier",
    "ModelUnavailable",
    "ModelUsage",
    "StepResult",
    "UsageLedger",
    "build_chat_model",
    "evidence_is_grounded",
    "install_response_cache",
    "probe_model",
    "reconcile_attribute_fallback",
]
