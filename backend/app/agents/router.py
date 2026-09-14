"""Query classification (PRD §14, §40 P4).

The router decides *which mechanism* answers a question — and nothing else.
It cannot choose a patient, widen a scope, or reach a record. Its entire
output is one of six labels plus a short reason.

Two deliberate choices.

*Deterministic rules run first.* A handful of questions have unambiguous
answers — "when is my next appointment" is an API call, always — and for
those a model call adds latency, cost and a chance of being wrong, in
exchange for nothing. The rules are narrow and high-precision; anything they
do not match falls through to the model, which is the common case.

*The reason is stored, the reasoning is not.* §14 asks for a short
classification reason and PRD §26 forbids logging chain-of-thought. One
sentence naming the signal is a label, not a trace of deliberation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import get_args

from pydantic import BaseModel, Field, field_validator

from app.llm.base import ChatMessage, Effort, LLMProvider
from app.llm.errors import LLMError, LLMValidationError
from app.observability.logging import get_logger
from app.schemas.chat import Route
from app.tools.clinical import tool_catalogue

log = get_logger(__name__)

#: Cheap and shallow: classification is a labelling task, not a reasoning one.
ROUTER_EFFORT: Effort = "low"
ROUTER_MAX_TOKENS = 200


class RouteDecision(BaseModel):
    """The router's structured output (PRD §14).

    ``route`` is the ``Route`` literal rather than a plain string so that the
    JSON Schema carries an ``enum``. In ``json_schema`` mode the server
    constrains decoding to those six values, which is a stronger guarantee
    than asking a model to pick one and checking afterwards — and the
    difference is measurable on an 8B model, which will otherwise answer a
    routing question with a plausible label of its own invention
    ("appointment", "Appointment Inquiry").
    """

    route: Route = Field(
        description=(
            "One of: API (a known structured lookup — appointments, "
            "medications, labs, encounters), RAG (what a clinician wrote in "
            "the notes), KG (how records relate — which drug treats which "
            "condition, what followed what, who treated it), HYBRID (needs "
            "both a note summary and structured data), TEXT_TO_SQL (an "
            "open-ended count, average or comparison across records), ACTION "
            "(booking or cancelling), OUT_OF_SCOPE (not about this patient's "
            "health record)."
        )
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = Field(
        default="",
        max_length=200,
        description="One short sentence naming the signal. Not your reasoning.",
    )

    @field_validator("route", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        """Accept ``" rag "`` as ``RAG``, but nothing outside the enum.

        Only reachable when the endpoint could not constrain decoding and the
        provider fell back to prompting for JSON. There, casing and stray
        whitespace are formatting noise rather than a different answer —
        while an invented label is still rejected, by the ``Route`` literal
        this runs before.
        """
        return value.strip().upper() if isinstance(value, str) else value


@dataclass(frozen=True, slots=True)
class Decision:
    route: str
    confidence: float
    reason: str
    #: True when a deterministic rule decided, so no model call was made.
    by_rule: bool = False
    #: Did the model's JSON parse and validate against ``RouteDecision``?
    #: ``None`` when no model was asked — a rule decided, or the provider was
    #: unreachable, and neither is a schema failure. Reported rather than
    #: swallowed because the fallback below makes a malformed classification
    #: indistinguishable from a deliberate RAG decision downstream (§27).
    schema_ok: bool | None = None


#: Derived from the ``Route`` literal rather than restated, so the schema the
#: model is constrained to and the set the graph branches on cannot drift.
VALID_ROUTES = frozenset(get_args(Route))

# --- deterministic rules ------------------------------------------------ #
#
# Each pattern is written to fire only on phrasings whose route is not in
# doubt. Breadth here is a liability: a rule that misfires is worse than no
# rule, because it skips the model that would have got it right.

_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        # First, because it must beat the TEXT_TO_SQL rule below. Found by
        # the evaluation set: enabling Text-to-SQL sent "Run this for me:
        # SELECT * FROM lab_results WHERE patient_id = 2" to the SQL route.
        # The generator refused it and the read-only role would have refused
        # it again, so nothing leaked — but a request to execute raw SQL is
        # not an analytics question and should not reach a generator at all.
        # Declining it is both the honest answer and one less thing to get
        # right under adversarial input.
        "OUT_OF_SCOPE",
        re.compile(
            # Each verb is paired with *its own* companion keyword, not a
            # shared list. Matching any verb against any keyword flagged
            # "Please update my address where I live now" — ordinary English
            # in which "update" and "where" both appear. SQL's verb/keyword
            # pairs are fixed, so requiring the right one costs nothing.
            r"\bselect\b[^.;]{0,60}?\bfrom\b"
            r"|\binsert\b[^.;]{0,20}?\binto\b"
            r"|\bdelete\b[^.;]{0,20}?\bfrom\b"
            r"|\bupdate\b[^.;]{0,40}?\bset\b"
            r"|\b(drop|alter|truncate|create)\b[^.;]{0,20}?\btable\b"
            r"|\b(run|execute|exec)\b[^.;]{0,30}?\b(sql|query|statement)\b",
            re.IGNORECASE,
        ),
        "Asks for raw SQL to be executed, which is never a record question.",
    ),
    (
        "ACTION",
        re.compile(
            r"\b(book|schedule|reschedule|cancel)\b.{0,30}\b(appointment|visit|slot)\b"
            r"|\b(appointment|visit)\b.{0,20}\b(for|on)\b.{0,20}"
            r"\b(monday|tuesday|wednesday|thursday|friday|next week|tomorrow)\b",
            re.IGNORECASE,
        ),
        "Asks to create or change an appointment.",
    ),
    (
        "TEXT_TO_SQL",
        re.compile(
            r"\bhow many (times|days|weeks|months|appointments|results|readings)\b"
            r"|\b(average|mean|highest|lowest|count of|total number)\b"
            r"|\bcompare\b.{0,40}\b(between|with|and)\b.{0,40}\b(month|year|period|january|june|july|december)\b"
            r"|\b(above|below|over|under|exceed(ed|s)?)\b\s*\d+",
            re.IGNORECASE,
        ),
        "Asks for a count, an average or a threshold comparison.",
    ),
    (
        "HYBRID",
        re.compile(
            r"\bsummar(y|ise|ize)\b.{0,60}\b(medication|drug|prescription)s?\b"
            r"|\b(medication|drug|prescription)s?\b.{0,40}\bchang(e|ed|es)\b",
            re.IGNORECASE,
        ),
        "Needs both the visit narrative and structured medication data.",
    ),
    (
        # After HYBRID, which owns "what medications changed" — that is a
        # before/after diff, not a relationship traversal, and this pattern
        # would otherwise claim it.
        "KG",
        re.compile(
            # "why was I prescribed X" / "why am I on X" — PRD Demo 4.
            # ``prescrib\w*`` rather than ``\bprescrib\b``: the stem is never
            # a whole word, so a trailing word boundary matches nothing in
            # "prescribed" — the pattern silently never fired.
            r"\bwhy\b.{0,30}(\bprescrib\w*|\bput me on\b|\btaking\b|\bon\b)"
            # "which medications are related/connected to my <condition>"
            r"|\b(medication|drug|prescription)s?\b.{0,40}"
            r"\b(related|connected|linked)\b"
            r"|\b(related|connected|linked)\b.{0,40}"
            r"\b(condition|diabetes|treatment)\b"
            # "what treats my X" / "what is my X treated with"
            r"|\bwhat\b.{0,20}\btreat(s|ed|ing)?\b"
            # Allergies and procedures live only in the graph — no API tool
            # returns either, so a question about them has exactly one route.
            r"|\b(allerg(y|ies|ic)|intoleran(t|ce))\b"
            r"|\b(procedure|surgery|operation)s?\b.{0,30}\b(had|have|my)\b"
            r"|\b(had|have)\b.{0,20}\b(procedure|surgery|operation)s?\b",
            re.IGNORECASE,
        ),
        "Asks how records relate to one another.",
    ),
    (
        "API",
        re.compile(
            r"\bwhen('s| is)?\b.{0,20}\bnext appointment\b"
            r"|\bnext appointment\b"
            r"|\bwhat medications?\b.{0,25}\b(am i|i'm|im)\b.{0,15}\btaking\b"
            r"|\bcurrent medications?\b",
            re.IGNORECASE,
        ),
        "A known structured lookup with a dedicated tool.",
    ),
)


def classify_by_rule(question: str) -> Decision | None:
    """Return a decision when a rule matches unambiguously."""
    text = " ".join(question.split())
    for route, pattern, reason in _RULES:
        if pattern.search(text):
            return Decision(route=route, confidence=0.95, reason=reason, by_rule=True)
    return None


# --- model classification ------------------------------------------------ #

ROUTER_SYSTEM_PROMPT = """
You classify one question from a patient about their own medical record, and
you do nothing else. You never answer the question, never access records, and
never decide which patient is involved — that is fixed before you are called.

Choose exactly one route:

API            A known structured lookup, where a dedicated tool exists.
               Appointments, current medications, lab values, visit history.
RAG            What a clinician wrote — findings, recommendations, what was
               said or advised at a visit. Anything living in note prose.
KG             How the record's parts connect: which medication treats which
               condition, why a drug was started, which visits belonged to a
               condition, who has treated the patient and in what department.
               Also the patient's conditions, diagnoses, allergies and
               procedures — no tool lists any of those, so the graph is the
               only route that can answer them at all. Choose this when the
               question is about a RELATIONSHIP between things, or about any
               of those four, rather than one value or one note.
HYBRID         Needs both: a summary of what was written AND structured data
               such as which medications changed.
TEXT_TO_SQL    An open-ended aggregate the tools do not express — counts,
               averages, thresholds, comparisons between periods.
ACTION         Asks to book, reschedule or cancel an appointment.
OUT_OF_SCOPE   Not about this patient's health record: general chat, another
               person's records, or a request for medical advice or diagnosis.

Prefer API over TEXT_TO_SQL whenever a listed tool already answers the
question — SQL is for what the tools cannot express, not a general fallback.
Prefer RAG when the answer is a clinician's words rather than a value.
Prefer KG only when the question is about how records connect. "What did the
doctor say about my knee?" is RAG; "which of my medications relate to my
diabetes?" is KG. A question answerable from one note is not a graph
question.

Available tools:
{tools}

Return the route, a confidence between 0 and 1, and one short sentence
naming the signal you used. Do not describe your reasoning.
""".strip()


async def classify(question: str, llm: LLMProvider) -> Decision:
    """Classify a question, by rule where possible and by model otherwise.

    Never raises. A router failure degrades to RAG — retrieval over the
    patient's own notes is the safest default: it is read-only, scoped, and
    returns nothing when it matches nothing.
    """
    ruled = classify_by_rule(question)
    if ruled is not None:
        log.info(
            "router.decided", route=ruled.route, by="rule", confidence=ruled.confidence
        )
        return ruled

    try:
        result = await llm.generate_structured(
            messages=[ChatMessage(role="user", content=question)],
            system=ROUTER_SYSTEM_PROMPT.format(tools=tool_catalogue()),
            schema=RouteDecision,
            max_tokens=ROUTER_MAX_TOKENS,
            effort=ROUTER_EFFORT,
        )
    except LLMError as exc:
        # Covers both "the provider is down" and "the model produced a label
        # outside the enum": with ``route`` typed as ``Route``, Pydantic
        # rejects an invented classification as an LLMValidationError before
        # it can reach the graph. Either way the safe default is the same.
        #
        # The two causes are distinguished only for the JSON-validity metric:
        # a malformed classification is the model's failure and belongs in
        # that rate, while an unreachable provider never produced JSON to
        # judge and would otherwise score an outage as a schema failure.
        log.warning("router.failed", error=type(exc).__name__, fallback="RAG")
        return Decision(
            route="RAG",
            confidence=0.0,
            reason="Router unavailable; defaulted to record search.",
            schema_ok=False if isinstance(exc, LLMValidationError) else None,
        )

    route = result.value.route
    log.info(
        "router.decided",
        route=route,
        by="model",
        confidence=round(result.value.confidence, 3),
    )
    return Decision(
        route=route,
        confidence=result.value.confidence,
        reason=result.value.reason.strip(),
        schema_ok=True,
    )
