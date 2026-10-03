"""Routing decisions. Pure, deterministic, fully unit-testable.

DETERMINISTIC MODULE — no model calls, no I/O, no randomness. Every threshold
comparison and escalation rule lives here so the policy can be read in one
place and tested without a network (spec section 16).

Routing shape (spec section 13):

    Model A result
      ├── schema invalid after repair ──────────> escalate (SCHEMA_FAILURE)
      ├── unsupported taxonomy pair ────────────> escalate (UNSUPPORTED_OUTPUT)
      ├── confidence < threshold_A ─────────────> escalate (LOW_CONFIDENCE)
      ├── needs_review ────────────────────────> escalate (AMBIGUOUS_TEXT)
      └── otherwise ───────────────────────────> accept

    Model B result
      ├── schema invalid after repair ──────────> human (SCHEMA_FAILURE)
      ├── needs_review ────────────────────────> human (AMBIGUOUS_TEXT)
      ├── confidence < threshold_B ─────────────> human (LOW_CONFIDENCE)
      ├── disagrees with A on primary ──────────> human (MODEL_DISAGREEMENT)
      └── otherwise ───────────────────────────> accept

Two things worth knowing about the disagreement rule:

1. It can only ever fire on ESCALATED rows, because Model B is never invoked on
   a row Model A accepted. Disagreement is therefore not a quality signal for
   the ~80% of rows A accepts — those have no label-free quality evidence in
   this build. That limit was accepted deliberately; a sampled Model B audit of
   auto-accepted rows was the 10/10 option and was declined.
   gstack-shortcut(dec-bc063cc1): disagreement routing covers escalated rows
   only; upgrade when running on real unlabelled returns, or when anyone asks
   for evidence about the auto-accepted population.

2. A row with no usable text never reaches a model at all. There is nothing to
   classify, so spending two calls to discover that would be waste. It goes
   straight to human review with NO_USEFUL_TEXT, which satisfies the spec's
   intent (never force a label onto meaningless input) more strongly than
   escalating would. Documented as a deliberate deviation in the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from . import config
from .schemas import PrimaryReason, is_legal_pair


class Route(str, Enum):
    ACCEPT = "accept"
    ESCALATE = "escalate"
    HUMAN = "human"


@dataclass(frozen=True)
class RoutingDecision:
    """Where a row goes next, and the human-readable reason it went there."""

    route: Route
    reason: str | None
    detail: str

    @property
    def needs_human(self) -> bool:
        return self.route is Route.HUMAN


@dataclass(frozen=True)
class StepOutcome:
    """A model step's result, flattened for routing.

    `schema_ok` is False when the output failed validation even after the single
    repair retry. `primary`/`sub_reason` are None in that case.
    """

    schema_ok: bool
    primary: str | None = None
    sub_reason: str | None = None
    confidence: float = 0.0
    needs_review: bool = False
    error: str | None = None


def route_no_text() -> RoutingDecision:
    """A row whose free text is empty or whitespace. No model call is made."""
    return RoutingDecision(
        route=Route.HUMAN,
        reason="NO_USEFUL_TEXT",
        detail=(
            "The free-text field is empty, so there is nothing to classify. "
            "Sent straight to review without spending a model call."
        ),
    )


def route_after_a(
    outcome: StepOutcome,
    threshold: float | None = None,
) -> RoutingDecision:
    """Decide what happens to a row after Model A has run."""
    limit = config.ACCEPT_THRESHOLD_A if threshold is None else threshold

    if not outcome.schema_ok:
        return RoutingDecision(
            Route.ESCALATE,
            "SCHEMA_FAILURE",
            f"Model A output failed schema validation after one repair retry: "
            f"{outcome.error or 'unknown error'}. Escalating to Model B.",
        )

    if outcome.primary is None:
        return RoutingDecision(
            Route.ESCALATE,
            "UNSUPPORTED_OUTPUT",
            "Model A returned no primary reason. Escalating to Model B.",
        )

    if outcome.sub_reason is not None and not is_legal_pair(
        outcome.primary, outcome.sub_reason
    ):
        return RoutingDecision(
            Route.ESCALATE,
            "UNSUPPORTED_OUTPUT",
            f"Model A produced the unsupported pair "
            f"{outcome.primary}/{outcome.sub_reason}. Escalating to Model B.",
        )

    if outcome.needs_review:
        return RoutingDecision(
            Route.ESCALATE,
            "AMBIGUOUS_TEXT",
            "Model A flagged the comment as ambiguous, contradictory, or "
            "unsupported. Escalating to Model B.",
        )

    if outcome.confidence < limit:
        return RoutingDecision(
            Route.ESCALATE,
            "LOW_CONFIDENCE",
            f"Model A confidence {outcome.confidence:.2f} is below the "
            f"{limit:.2f} accept threshold. Escalating to Model B.",
        )

    return RoutingDecision(
        Route.ACCEPT,
        None,
        f"Accepted from Model A: confidence {outcome.confidence:.2f} at or above "
        f"the {limit:.2f} threshold, and not flagged for review.",
    )


def route_after_b(
    outcome_b: StepOutcome,
    outcome_a: StepOutcome,
    threshold: float | None = None,
) -> RoutingDecision:
    """Decide what happens to an escalated row after Model B has run."""
    limit = config.ACCEPT_THRESHOLD_B if threshold is None else threshold

    if not outcome_b.schema_ok:
        return RoutingDecision(
            Route.HUMAN,
            "SCHEMA_FAILURE",
            f"Model B output also failed schema validation after a repair "
            f"retry: {outcome_b.error or 'unknown error'}. Sent to review.",
        )

    if outcome_b.primary is None:
        return RoutingDecision(
            Route.HUMAN,
            "UNSUPPORTED_OUTPUT",
            "Model B returned no primary reason. Sent to review.",
        )

    if outcome_b.sub_reason is not None and not is_legal_pair(
        outcome_b.primary, outcome_b.sub_reason
    ):
        return RoutingDecision(
            Route.HUMAN,
            "UNSUPPORTED_OUTPUT",
            f"Model B produced the unsupported pair "
            f"{outcome_b.primary}/{outcome_b.sub_reason}. Sent to review.",
        )

    if outcome_b.needs_review:
        return RoutingDecision(
            Route.HUMAN,
            "AMBIGUOUS_TEXT",
            "Model B still considers the comment ambiguous or unsupported. "
            "This is the intended outcome for genuinely unusable text.",
        )

    if outcome_b.confidence < limit:
        return RoutingDecision(
            Route.HUMAN,
            "LOW_CONFIDENCE",
            f"Model B confidence {outcome_b.confidence:.2f} is below the "
            f"{limit:.2f} accept threshold. Sent to review.",
        )

    if (
        outcome_a.schema_ok
        and outcome_a.primary is not None
        and outcome_a.primary != outcome_b.primary
    ):
        return RoutingDecision(
            Route.HUMAN,
            "MODEL_DISAGREEMENT",
            f"Model A read this as {outcome_a.primary} and Model B as "
            f"{outcome_b.primary}. Two models disagreeing on an escalated row "
            "is a signal to let a human decide, not to pick one.",
        )

    return RoutingDecision(
        Route.ACCEPT,
        None,
        f"Accepted from Model B: confidence {outcome_b.confidence:.2f} at or "
        f"above the {limit:.2f} threshold, agrees with Model A, not flagged.",
    )


def terminal_state(route: Route, *, escalated: bool) -> str:
    """Map a final route onto one of the spec section 23 row states."""
    if route is Route.HUMAN:
        return "NEEDS_HUMAN_REVIEW"
    if route is Route.ACCEPT:
        return "CLASSIFIED_MODEL_B" if escalated else "CLASSIFIED_MODEL_A"
    raise ValueError(f"{route} is not a terminal route")  # pragma: no cover


def review_budget_minutes(review_row_count: int) -> float:
    """Express a review queue as a claim on the reviewer's time, not a count.

    "25 rows" means nothing to a category head; "about an hour of your week"
    does. The per-row constant is an assumption until someone times a human
    labelling the ambiguous rows — the UI labels it as such.
    """
    return review_row_count * config.REVIEW_MINUTES_PER_ROW


def primary_values() -> tuple[str, ...]:
    return tuple(p.value for p in PrimaryReason)
