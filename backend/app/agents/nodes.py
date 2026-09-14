"""The graph's nodes (PRD §11).

Each node takes the state and returns a partial update. They share three
rules:

* No node reads or writes ``patient_id``. Scope arrives in the state and is
  handed to services as a whole ``AuthContext``; a node that constructed one
  would be deciding authorization, which §40 P5 puts outside the graph.
* Every node that can fail records the failure in ``validation_errors`` and
  returns rather than raising. A half-answered turn with a recorded reason
  is more useful than a stack trace, and ``generate_answer`` knows what to
  say when evidence is missing.
* Tool invocations are counted, and the count is checked against
  ``MAX_TOOL_CALLS`` before each batch (§11).

Nodes are closures over the request's session, context, provider and trace —
LangGraph state carries data, not connections.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.actions.appointments import ActionError, AppointmentRequest, propose
from app.agents.router import classify
from app.agents.state import MAX_TOOL_CALLS, AgentState
from app.auth.context import AuthContext, AuthorizationError
from app.config import settings
from app.guardrails import validate_answer
from app.knowledge_graph import (
    INTENT_DESCRIPTIONS,
    GraphIntent,
    GraphQueryError,
    GraphResult,
    GraphUnavailable,
    query_patient_graph,
)
from app.llm.base import ChatMessage, LLMProvider
from app.llm.errors import LLMError, LLMRefusalError, LLMValidationError
from app.observability.logging import get_logger
from app.observability.trace import Trace
from app.prompts import build_system_prompt
from app.rag import pipeline as rag_pipeline
from app.rag.embeddings import EmbeddingProvider
from app.rag.reranking import NoopReranker, Reranker
from app.rag.rewrite import rewrite_query
from app.sql.analytics import (
    AnalyticsResult,
    AnalyticsUnavailable,
    run_patient_analytics,
)
from app.tools.base import ToolError, ToolResult, run_tool
from app.tools.clinical import TOOLS, TOOLS_BY_NAME, tool_catalogue

log = get_logger(__name__)

Node = Callable[[AgentState], Awaitable[dict]]

FAILURE_MESSAGE = (
    "I could not reach the assistant service just now. Please try again in a "
    "moment, and contact your care team directly if this is urgent."
)
REFUSAL_MESSAGE = (
    "I am not able to answer that one. If it concerns your care, please "
    "raise it with your care team."
)
OUT_OF_SCOPE_MESSAGE = (
    "I can only help with what is in your own medical record — appointments, "
    "medications, lab results, visits and your clinical notes. For anything "
    "else, including medical advice, please speak to your care team."
)

#: The API route's fallback plan, used when the model's selection is
#: unusable. Broad on purpose: if we are guessing, guess wide enough that
#: the answer is probably in there.
API_TOOL_PLAN: tuple[str, ...] = (
    "get_my_next_appointment",
    "get_my_medications",
    "get_my_lab_results",
)

#: HYBRID's plan stays fixed, and this is not an oversight about §5.
#:
#: These two tools are not a *selection* for the question — they are a data
#: dependency of the route. ``get_my_last_encounter`` produces the
#: ``encounter_id`` that anchors the retrieval below it, so a model that
#: dropped it would not run one fewer tool, it would silently un-anchor the
#: notes and summarise whichever visit matched the wording best.
#: ``get_my_medication_changes`` computes the diff the route exists to show.
#: The route *is* the plan here; there is no choice to delegate.
HYBRID_TOOL_PLAN: tuple[str, ...] = (
    "get_my_last_encounter",
    "get_my_medication_changes",
)

#: Enough for a list of a few tool names. Built from the catalogue so it
#: scales if tools are added.
TOOL_SELECT_MAX_TOKENS = 200

#: The selectable set, derived from the catalogue rather than restated.
#:
#: A ``StrEnum`` rather than a list of strings, for the same reason
#: ``RouteDecision.route`` is a ``Literal``: it puts an enum in the JSON
#: Schema, so the server constrains decoding to real tool names and an 8B
#: model cannot invent ``get_my_bloodwork``. Deriving it from ``TOOLS``
#: means a tool added to the catalogue is selectable the same day, and one
#: removed stops being offered — no second list to keep in step.
ToolName = StrEnum(  # type: ignore[misc]
    "ToolName", {spec.name.upper(): spec.name for spec in TOOLS}
)


class ToolPlan(BaseModel):
    """Which record lookups answer this question (PRD §5).

    §5 makes tool selection the model's responsibility, and until this
    existed the API route ran a fixed three-tool plan for every question —
    which meant §27's tool-selection accuracy scored a constant, not a
    decision. A metric that cannot fail is not measuring anything.

    Selection is narrow by construction: the model chooses *which* approved,
    read-only, self-scoped lookups to run. It cannot choose whose record to
    read — no tool takes a patient identifier — so a wrong selection costs a
    worse answer and never a wider one. That is what makes delegating this
    safe where delegating the patient scope would not be.
    """

    tools: list[ToolName] = Field(  # type: ignore[valid-type]
        default_factory=list,
        description=(
            "The lookups needed to answer, most important first. Choose only "
            "what the question actually needs — one is common and correct. "
            "Return an empty list if no record lookup helps."
        ),
    )


TOOL_SELECT_PROMPT = """
You choose which record lookups answer a patient's question. You do not
answer it, and you do not decide whose record is read — that is fixed
before you are called.

Available lookups:
{tools}

Choose the fewest that cover the question. "When is my next appointment?"
needs only the next appointment. "What am I taking and what were my last
results?" needs medications and lab results. Do not add a lookup because it
might be interesting; an unused result is noise in the answer.
""".strip()

#: Output cap for the graph-planning call. The answer is an enum label and a
#: word; this is sized so a verbose model still closes the JSON object.
KG_PLAN_MAX_TOKENS = 200


@dataclass(frozen=True, slots=True)
class NodeDeps:
    """Everything a node needs that is not state."""

    session: AsyncSession
    ctx: AuthContext
    llm: LLMProvider
    embedder: EmbeddingProvider
    trace: Trace
    #: Built from settings, with the provider injected so an LLM reranker
    #: shares the request's client rather than opening its own.
    reranker: Reranker = field(default_factory=NoopReranker)
    #: When true, ``generate_answer`` emits deltas through LangGraph's custom
    #: stream writer instead of returning one block. Routing and retrieval
    #: are identical either way — only the shape of generation changes.
    stream: bool = False


# ---------------------------------------------------------------------- #
# classify
# ---------------------------------------------------------------------- #


def make_classify(deps: NodeDeps) -> Node:
    async def classify_query(state: AgentState) -> dict:
        with deps.trace.stage("router"):
            decision = await classify(state["question"], deps.llm)

        deps.trace.route = decision.route
        deps.trace.route_confidence = decision.confidence
        if decision.schema_ok is not None:
            deps.trace.record_structured(ok=decision.schema_ok)
        return {
            "route": decision.route,
            "route_confidence": decision.confidence,
            "route_reason": decision.reason,
            "visited": ["classify_query"],
        }

    return classify_query


def route_query(state: AgentState) -> str:
    """The conditional edge. Pure — it reads the label and nothing else."""
    return state.get("route") or "RAG"


# ---------------------------------------------------------------------- #
# API tools
# ---------------------------------------------------------------------- #


async def _run_plan(
    deps: NodeDeps, plan: tuple[str, ...], already_used: int
) -> tuple[list[ToolResult], list[str]]:
    """Run a fixed tool plan, respecting the per-request ceiling."""
    results: list[ToolResult] = []
    errors: list[str] = []

    for name in plan:
        if already_used + len(results) >= MAX_TOOL_CALLS:
            errors.append(f"Tool budget of {MAX_TOOL_CALLS} reached before {name}.")
            log.warning("agent.tool_budget_exhausted", next_tool=name)
            break

        spec = TOOLS_BY_NAME.get(name)
        if spec is None:  # pragma: no cover - plans are literals
            errors.append(f"Unknown tool {name}.")
            continue

        try:
            result = await run_tool(spec, deps.session, deps.ctx)
        except ToolError as exc:
            errors.append(str(exc))
            deps.trace.record_tool(name, ms=0, ok=False)
            continue

        deps.trace.record_tool(name, ms=result.latency_ms, ok=True)
        results.append(result)

    return results, errors


def make_api_node(deps: NodeDeps) -> Node:
    async def select_tools(question: str) -> tuple[tuple[str, ...], list[str]]:
        """Ask the model which lookups to run. Returns (plan, guardrails).

        Falls back to :data:`API_TOOL_PLAN` whenever the selection is
        unusable, and says so in a guardrail code rather than silently. The
        distinction matters to §27: a run where the model chose well and one
        where it failed and got the fallback produce the same answer, and
        only the code tells them apart.
        """
        with deps.trace.stage("tool_select"):
            try:
                planned = await deps.llm.generate_structured(
                    messages=[ChatMessage(role="user", content=question)],
                    system=TOOL_SELECT_PROMPT.format(tools=tool_catalogue()),
                    schema=ToolPlan,
                    max_tokens=TOOL_SELECT_MAX_TOKENS,
                    effort="low",
                    model=settings.router_model,
                )
            except LLMError as exc:
                if isinstance(exc, LLMValidationError):
                    deps.trace.record_structured(ok=False)
                log.warning("agent.tool_select_failed", error=type(exc).__name__)
                return API_TOOL_PLAN, ["tool_select_fallback"]
            deps.trace.record_structured(ok=True)

        # Deduplicated, order preserved. The enum already guarantees every
        # name is real, so this only removes a model that listed one twice.
        chosen: list[str] = []
        for name in planned.value.tools:
            value = str(name)
            if value not in chosen:
                chosen.append(value)

        if not chosen:
            # An empty selection on a route the router sent here means the
            # model disagreed with the router, not that no lookup helps.
            # The fallback answers; the code records the disagreement.
            return API_TOOL_PLAN, ["tool_select_empty"]
        return tuple(chosen[:MAX_TOOL_CALLS]), []

    async def execute_api_tool(state: AgentState) -> dict:
        plan, guardrails = await select_tools(state["question"])
        with deps.trace.stage("tools"):
            results, errors = await _run_plan(
                deps, plan, state.get("tool_calls", 0)
            )
        return {
            "tool_results": results,
            "tool_calls": len(results),
            "validation_errors": errors,
            "guardrails": guardrails,
            "visited": ["execute_api_tool"],
        }

    return execute_api_tool


# ---------------------------------------------------------------------- #
# RAG
# ---------------------------------------------------------------------- #


def make_rag_node(deps: NodeDeps) -> Node:
    async def retrieve(state: AgentState) -> dict:
        # PRD §19's first step. Retrieval runs on the rewritten question;
        # generation still sees the original, because the patient asked the
        # original and an answer echoing a rewrite reads as a misquote.
        query = await rewrite_query(
            question=state["question"],
            history=state.get("history", []),
            llm=deps.llm,
            trace=deps.trace,
        )
        result = await rag_pipeline.retrieve(
            deps.session,
            deps.ctx,
            question=query,
            embedder=deps.embedder,
            trace=deps.trace,
            reranker=deps.reranker,
        )
        errors: list[str] = []
        if result.index_empty:
            errors.append("No clinical notes have been indexed for this patient.")
        return {
            "retrieved": result.chunks,
            "context_text": result.context.text,
            "sources": result.sources,
            "validation_errors": errors,
            "visited": ["retrieve"],
        }

    return retrieve


# ---------------------------------------------------------------------- #
# Hybrid
# ---------------------------------------------------------------------- #


def make_hybrid_node(deps: NodeDeps) -> Node:
    async def hybrid(state: AgentState) -> dict:
        """Structured facts and note prose, combined deterministically.

        The medication diff comes from ``get_my_medication_changes``, which
        computes it in Python (§40 P12). The model is handed the finished list to
        narrate — it is never asked to work out what changed.
        """
        with deps.trace.stage("tools"):
            results, errors = await _run_plan(
                deps, HYBRID_TOOL_PLAN, state.get("tool_calls", 0)
            )

        # Anchor retrieval on the encounter the tools just identified, so the
        # notes summarized are the ones from that visit rather than whichever
        # visit happens to match the wording best.
        encounter_id = None
        for result in results:
            if result.name == "get_my_last_encounter" and result.data is not None:
                encounter_id = result.data.id
                break

        query = await rewrite_query(
            question=state["question"],
            history=state.get("history", []),
            llm=deps.llm,
            trace=deps.trace,
        )
        retrieval = await rag_pipeline.retrieve(
            deps.session,
            deps.ctx,
            question=query,
            embedder=deps.embedder,
            trace=deps.trace,
            encounter_id=encounter_id,
        )
        if encounter_id is not None and not retrieval.chunks:
            # The visit exists but nothing from it matched. Fall back to an
            # unanchored search rather than answering with no notes at all.
            retrieval = await rag_pipeline.retrieve(
                deps.session,
                deps.ctx,
                question=query,
                embedder=deps.embedder,
            )

        return {
            "tool_results": results,
            "tool_calls": len(results),
            "retrieved": retrieval.chunks,
            "context_text": retrieval.context.text,
            "sources": retrieval.sources,
            "validation_errors": errors,
            "visited": ["hybrid"],
        }

    return hybrid


# ---------------------------------------------------------------------- #
# Text-to-SQL
# ---------------------------------------------------------------------- #

NO_SQL_ANSWER = (
    "I can't work that one out from your records. I can look up your "
    "appointments, medications, lab results and visits, and tell you what "
    "your clinicians wrote in your notes."
)
SQL_FAILED = (
    "I wasn't able to work that out reliably, so I'd rather not guess. "
    "Please ask your care team, or try asking in a simpler way."
)


# ---------------------------------------------------------------------- #
# Knowledge graph
# ---------------------------------------------------------------------- #


class GraphPlan(BaseModel):
    """Which approved traversal to run, and what to look for.

    The model's entire influence over the graph. It picks an intent from a
    closed enum and supplies a search term; it cannot name a patient, and it
    cannot write Cypher. ``intent`` is a ``Literal`` so the JSON Schema
    carries the enum and the server constrains decoding to it — the same
    reason ``RouteDecision.route`` is one (PRD §14).
    """

    # The StrEnum itself, so Pydantic emits a JSON Schema ``enum`` and the
    # server constrains decoding to the approved traversals.
    intent: GraphIntent = Field(
        description="The traversal that answers the question."
    )
    term: str = Field(
        default="",
        max_length=80,
        description=(
            "The condition or medication named in the question, e.g. "
            '"diabetes" or "metformin". Empty when the question names neither.'
        ),
    )


GRAPH_PLAN_PROMPT = """
You choose which relationship lookup answers a patient's question about their
own medical record. You do not answer the question and you do not decide
whose record is involved — that is fixed before you are called.

Available lookups:
{intents}

Return the lookup and, when the question names a condition or a medication,
that word on its own — "diabetes", not "my diabetes treatment history".
""".strip()

GRAPH_UNAVAILABLE = (
    "I could not reach the relationship data for your record just now. Please "
    "try again in a moment, and contact your care team directly if this is "
    "urgent."
)


def make_kg_node(deps: NodeDeps) -> Node:
    def _intent_catalogue() -> str:
        return "\n".join(
            f"  {intent.value}: {description}"
            for intent, description in INTENT_DESCRIPTIONS.items()
        )

    async def query_graph(state: AgentState) -> dict:
        """Pick an approved traversal, run it, hand the rows back as facts.

        The graph is optional infrastructure (PRD §33 — it is derived, and
        PostgreSQL remains the source of truth), so an unreachable Neo4j
        degrades to a stated refusal rather than a 500. A *malformed* plan is
        different: it means the model chose badly, and retrying the whole turn
        would cost another traversal to reach the same place, so it is
        reported as missing evidence and generation says so.
        """
        with deps.trace.stage("kg_plan"):
            try:
                planned = await deps.llm.generate_structured(
                    messages=[ChatMessage(role="user", content=state["question"])],
                    system=GRAPH_PLAN_PROMPT.format(intents=_intent_catalogue()),
                    schema=GraphPlan,
                    max_tokens=KG_PLAN_MAX_TOKENS,
                    effort="low",
                )
            except LLMError as exc:
                # A schema failure is the model's; an unreachable provider is
                # not. Only the former counts against JSON validity (§27).
                if isinstance(exc, LLMValidationError):
                    deps.trace.record_structured(ok=False)
                deps.trace.error = type(exc).__name__
                log.warning("agent.kg_plan_failed", error=type(exc).__name__)
                return {
                    "final_answer": FAILURE_MESSAGE,
                    "guardrails": ["llm_unavailable"],
                    "visited": ["query_graph"],
                }
            deps.trace.record_structured(ok=True)

        plan = planned.value
        with deps.trace.stage("kg"):
            try:
                result = await query_patient_graph(
                    deps.ctx, intent=plan.intent, term=plan.term
                )
            except GraphQueryError as exc:
                # A term-requiring traversal chosen without a term. Stating it
                # beats a second model call that would probably repeat itself.
                log.info("agent.kg_plan_incomplete", intent=plan.intent)
                return {
                    "validation_errors": [str(exc)],
                    "guardrails": ["kg_incomplete"],
                    "visited": ["query_graph"],
                }
            except GraphUnavailable as exc:
                deps.trace.error = type(exc).__name__
                log.warning("agent.kg_unavailable", error=type(exc).__name__)
                return {
                    "final_answer": GRAPH_UNAVAILABLE,
                    "validation_errors": [str(exc)],
                    "guardrails": ["kg_unavailable"],
                    "visited": ["query_graph"],
                }
            except AuthorizationError as exc:
                # Not linked to a patient record. The graph never ran.
                deps.trace.error = type(exc).__name__
                deps.trace.record_authorization_failure()
                return {
                    "final_answer": GRAPH_UNAVAILABLE,
                    "validation_errors": [str(exc)],
                    "guardrails": ["kg_unauthorized"],
                    "visited": ["query_graph"],
                }

        deps.trace.record_tool(
            f"kg:{result.intent.value}", ms=result.latency_ms, ok=not result.is_empty
        )
        # An empty traversal is not an error — the graph genuinely holds
        # nothing matching — but it is the failure mode of entity resolution,
        # and it is otherwise indistinguishable downstream from a traversal
        # that found things: both produce an answer, and both record the same
        # tool name. The code is what lets the developer panel and the
        # evaluation tell "matched nothing" from "matched".
        errors = (
            ["The relationship graph holds nothing matching that."]
            if result.is_empty
            else []
        )
        return {
            "system_prompt_extra": _graph_facts(result),
            "graph_results": list(result.rows),
            "graph_intent": result.intent.value,
            "validation_errors": errors,
            "guardrails": ["kg_no_match"] if result.is_empty else [],
            "visited": ["query_graph"],
        }

    return query_graph


#: Traversals that locate an explanation without containing one.
#:
#: The graph can say that metformin was prescribed at the encounter that
#: recorded type 2 diabetes. It cannot say *why* the clinician chose it —
#: that sentence was written in the note, and PRD §18 and §37 Demo 4 both
#: describe this question as KG *and* RAG for exactly that reason: the
#: traversal finds the link, retrieval supplies the words.
#:
#: Narrow on purpose. "Which medications treat my diabetes" is answered by
#: the rows themselves, and sending it through retrieval would spend a
#: retrieval and a reranker pass to decorate an answer that was already
#: complete.
EXPLANATORY_INTENTS: frozenset[str] = frozenset({GraphIntent.WHY_MEDICATION.value})


def after_graph(state: AgentState) -> str:
    """Whether a traversal still needs the notes behind it (PRD §37 Demo 4).

    Three conditions, and each one is a case where chaining would be wrong
    rather than merely unnecessary. An answer already set means the graph
    was unreachable or the plan was incomplete, and retrieval cannot mend
    either. No rows means entity resolution found nothing, so there is no
    link for a note to explain, and retrieving anyway would hand the model
    prose with nothing to anchor it — which is how an ungrounded answer gets
    written. Any other intent is already complete in its rows.
    """
    if state.get("final_answer"):
        return "generate_answer"
    if state.get("graph_intent") not in EXPLANATORY_INTENTS:
        return "generate_answer"
    if not state.get("graph_results"):
        return "generate_answer"
    return "retrieve"


def _graph_facts(result: GraphResult) -> str:
    """Traversal rows as the model sees them.

    The summary line first, because it is the one statement the backend can
    make reliably (PRD Principle 12) and the model should be narrating it
    rather than recomputing it from the rows.
    """
    lines = [
        "RELATIONSHIP DATA from this patient's record "
        f"({result.intent.value}):",
        f"  {result.summary}",
    ]
    for row in result.rows:
        rendered = ", ".join(f"{key}={value}" for key, value in row.items())
        lines.append(f"  - {rendered}")
    return "\n".join(lines)


def make_text_to_sql_node(deps: NodeDeps) -> Node:
    async def text_to_sql(state: AgentState) -> dict:
        """Run the analytics capability and hand the rows back as facts.

        Goes through :func:`app.sql.analytics.run_patient_analytics` rather
        than calling the generator and executor directly, so this path and
        the MCP tool are the *same* capability — including the audit row
        §16 requires. Two call sites composing the pipeline themselves is how
        one of them ends up without the audit.

        Two failure modes, answered differently on purpose. The schema not
        covering the question is a fact about the data and is stated as one.
        Generation failing twice, or execution erroring, is our problem and
        gets a refusal rather than a guess — an aggregate is exactly the
        kind of question where a plausible wrong number is indistinguishable
        from a right one.
        """
        with deps.trace.stage("analytics"):
            try:
                result = await run_patient_analytics(
                    deps.ctx, deps.llm, question=state["question"]
                )
            except AnalyticsUnavailable as exc:
                kind = "invalid" if exc.answerable else "declined"
                deps.trace.error = f"analytics:{kind}"
                # Both codes, not one. ``sql_unavailable`` is the coarse fact
                # that no figure was produced; the second names which of the
                # two causes it was. They deserve opposite reactions — a
                # refused question is the schema working, a rejected
                # statement is the generator failing — and a single code
                # averages them into one uninterpretable rate (§27).
                return {
                    "final_answer": SQL_FAILED if exc.answerable else NO_SQL_ANSWER,
                    "validation_errors": [str(exc)],
                    "guardrails": [
                        "sql_unavailable",
                        "sql_invalid" if exc.answerable else "sql_declined",
                    ],
                    "visited": ["text_to_sql"],
                }
            except AuthorizationError as exc:
                deps.trace.error = type(exc).__name__
                deps.trace.record_authorization_failure()
                log.warning("agent.sql_failed", error=type(exc).__name__)
                return {
                    "final_answer": SQL_FAILED,
                    "validation_errors": [str(exc)],
                    "guardrails": ["sql_unavailable", "sql_unauthorized"],
                    "visited": ["text_to_sql"],
                }

        deps.trace.generated_sql = result.sql
        deps.trace.sql_row_count = result.row_count

        notes: list[str] = []
        if result.truncated:
            notes.append(f"Result capped at {result.row_count} rows.")

        # The rows go in as established facts, the same channel tool output
        # uses. The model narrates the number; it never recomputes it.
        return {
            "generated_sql": result.sql,
            "system_prompt_extra": _sql_facts(result),
            "validation_errors": notes,
            "visited": ["text_to_sql"],
        }

    return text_to_sql


def _sql_facts(result: AnalyticsResult) -> str:
    """Rows as the model sees them.

    The SQL itself is deliberately not included. A model shown the query
    tends to explain it; the patient asked for a number.
    """
    if not result.row_count:
        return (
            "QUERY RESULT (computed by the backend, treat as authoritative):\n"
            "The query returned no rows. Say plainly that there is nothing "
            "on record matching the question — do not speculate about why."
        )
    return (
        "QUERY RESULT (computed by the backend, treat as authoritative):\n"
        f"{result.table}\n\n"
        "State this figure directly. Do not recompute it, do not mention SQL "
        "or databases, and do not describe how it was obtained."
    )


# ---------------------------------------------------------------------- #
# Actions
# ---------------------------------------------------------------------- #

ACTION_PARSE_PROMPT = """
You extract the appointment action a patient is asking for. You do not
perform it, you do not confirm it, and you do not tell the patient it is
done — something else decides that, after they agree.

Today is {today}. Resolve relative dates ("next Tuesday", "tomorrow")
against it and return an ISO 8601 timestamp. If the patient gave no time of
day, use 09:00. If they gave no date at all, leave `when` empty rather than
inventing one.
""".strip()

ACTION_FAILED = (
    "I wasn't able to set that up. Please try again, or contact the clinic "
    "directly to arrange it."
)


def make_action_node(deps: NodeDeps) -> Node:
    async def action(state: AgentState) -> dict:
        """Parse and propose. Never execute.

        The node's whole output is a question and a signed token. Execution
        happens in ``POST /api/actions/confirm``, which the patient's next
        click reaches directly — no model call is on that path, so no
        sentence anyone types can cause a write here.
        """
        today = date.today()
        with deps.trace.stage("action_parse"):
            try:
                parsed = await deps.llm.generate_structured(
                    messages=[
                        ChatMessage(role="user", content=state["question"]),
                    ],
                    system=ACTION_PARSE_PROMPT.format(today=today.isoformat()),
                    schema=AppointmentRequest,
                    max_tokens=settings.sql_generation_max_tokens,
                    effort="low",
                    model=settings.router_model,
                )
            except LLMError as exc:
                if isinstance(exc, LLMValidationError):
                    deps.trace.record_structured(ok=False)
                deps.trace.error = type(exc).__name__
                log.warning("agent.action_parse_failed", error=type(exc).__name__)
                return {
                    "final_answer": FAILURE_MESSAGE,
                    "guardrails": ["llm_unavailable"],
                    "visited": ["action"],
                }
            deps.trace.record_structured(ok=True)

        try:
            proposal = await propose(deps.session, deps.ctx, parsed.value)
        except ActionError as exc:
            # A refusal with a reason the patient can act on — not a failure.
            return {
                "final_answer": str(exc),
                "validation_errors": [str(exc)],
                "guardrails": ["action_declined"],
                "visited": ["action"],
            }
        except AuthorizationError as exc:
            deps.trace.record_authorization_failure()
            return {
                "final_answer": str(exc),
                "validation_errors": [str(exc)],
                "guardrails": ["action_declined"],
                "visited": ["action"],
            }
        except Exception as exc:  # pragma: no cover - defensive
            deps.trace.error = type(exc).__name__
            log.warning("agent.action_failed", error=type(exc).__name__)
            return {
                "final_answer": ACTION_FAILED,
                "guardrails": ["action_failed"],
                "visited": ["action"],
            }

        deps.trace.action = proposal.action
        return {
            # Written here rather than generated, because the sentence the
            # patient confirms has to match the parameters in the token
            # exactly. A model asked to phrase it could paraphrase the time.
            "final_answer": (
                f"I can set up {proposal.summary}. "
                "Shall I go ahead? Nothing is booked until you confirm."
            ),
            "pending_action": {
                "action": proposal.action,
                "summary": proposal.summary,
                "token": proposal.token,
                "expires_at": proposal.expires_at.isoformat(),
            },
            "visited": ["action"],
        }

    return action


# ---------------------------------------------------------------------- #
# Out of scope
# ---------------------------------------------------------------------- #


async def out_of_scope(state: AgentState) -> dict:
    return {"final_answer": OUT_OF_SCOPE_MESSAGE, "visited": ["out_of_scope"]}


# ---------------------------------------------------------------------- #
# Generate and validate
# ---------------------------------------------------------------------- #


def _tool_facts(results: list[ToolResult]) -> str:
    """Tool output as the model sees it: finished sentences, not rows."""
    if not results:
        return ""
    lines = "\n".join(f"- {result.summary}" for result in results)
    return (
        f"ESTABLISHED FACTS (computed by the backend, treat as authoritative):\n{lines}"
    )


def make_generate_node(deps: NodeDeps) -> Node:
    async def generate_answer(state: AgentState) -> dict:
        # A node upstream already produced the answer (out of scope, or a
        # route that is not available). Nothing to generate.
        if state.get("final_answer"):
            return {"visited": ["generate_answer"]}

        facts = _tool_facts(state.get("tool_results", []))
        sql_facts = state.get("system_prompt_extra") or ""
        context = state.get("context_text") or ""
        extra_parts = [part for part in (facts, sql_facts) if part]

        system = build_system_prompt(
            context=context or None,
            patient_label=None,
            extra_instructions="\n\n".join(extra_parts) or None,
            # Tool output and SQL results are record facts. Without this the
            # prompt would tell the model it has nothing about the patient
            # in the same breath as handing it the patient's figures.
            record_facts_supplied=bool(extra_parts),
        )
        messages = [
            *state.get("history", []),
            ChatMessage(role="user", content=state["question"]),
        ]

        if deps.stream:
            return await _generate_streaming(deps, messages=messages, system=system)

        try:
            with deps.trace.stage("llm"):
                result = await deps.llm.generate(messages=messages, system=system)
        except LLMRefusalError as exc:
            deps.trace.error = f"refusal:{exc.category or 'unspecified'}"
            return {
                "final_answer": REFUSAL_MESSAGE,
                "guardrails": ["provider_refusal"],
                "visited": ["generate_answer"],
            }
        except LLMError as exc:
            deps.trace.error = type(exc).__name__
            log.warning("agent.llm_failed", error=type(exc).__name__)
            return {
                "final_answer": FAILURE_MESSAGE,
                "guardrails": ["llm_unavailable"],
                "visited": ["generate_answer"],
            }

        # The *resolved* id, which is not necessarily the configured one.
        # `trace.model` keeps what was asked for (PRD §26 wants both).
        deps.trace.model_version = result.model
        deps.trace.provider = result.provider
        deps.trace.record_usage(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            cost_usd=result.estimated_cost_usd,
        )
        return {
            "final_answer": result.text,
            "guardrails": ["truncated_answer"] if result.truncated else [],
            "visited": ["generate_answer"],
        }

    return generate_answer


async def _generate_streaming(
    deps: NodeDeps, *, messages: list[ChatMessage], system: str
) -> dict:
    """Generate, emitting each delta through LangGraph's custom stream.

    Token usage is unavailable on this path — the provider's streaming
    interface yields text and nothing else — so a streamed turn records
    latency and route on its trace but no token counts. Buffered turns are
    the ones to read for cost.

    Model and provider *are* recorded, from the provider itself rather than
    from a response that never arrives as an object. PRD §26 asks for both on
    every request, and streaming is the path the UI actually uses, so taking
    them from the response alone would leave the field empty on almost every
    real turn. The one thing lost is a server-side alias resolution: this
    reports the model asked for, where a buffered turn reports the version
    that answered.
    """
    writer = get_stream_writer()
    chunks: list[str] = []
    deps.trace.model = deps.llm.model
    deps.trace.provider = deps.llm.name

    try:
        with deps.trace.stage("llm"):
            async for delta in deps.llm.stream(messages=messages, system=system):
                chunks.append(delta)
                writer({"type": "delta", "text": delta})
    except LLMRefusalError as exc:
        deps.trace.error = f"refusal:{exc.category or 'unspecified'}"
        return {
            "final_answer": REFUSAL_MESSAGE,
            "guardrails": ["provider_refusal"],
            "visited": ["generate_answer"],
        }
    except LLMError as exc:
        deps.trace.error = type(exc).__name__
        log.warning("agent.stream_failed", error=type(exc).__name__)
        return {
            "final_answer": FAILURE_MESSAGE,
            "guardrails": ["llm_unavailable"],
            "visited": ["generate_answer"],
        }

    return {"final_answer": "".join(chunks), "visited": ["generate_answer"]}


def make_validate_node(deps: NodeDeps) -> Node:
    async def validate_result(state: AgentState) -> dict:
        """Output validation, after generation (PRD §25)."""
        sources = state.get("sources", [])
        checked = validate_answer(
            state.get("final_answer", ""),
            truncated="truncated_answer" in state.get("guardrails", []),
            available_source_ids={s.citation_key for s in sources},
            # A RAG answer built on retrieved notes should cite them; a
            # tool-backed or out-of-scope answer has nothing to cite.
            expect_sources=bool(state.get("retrieved")) and not sources,
        )
        if checked.violations:
            log.info(
                "agent.guardrails",
                codes=checked.codes,
                blocked=checked.blocked,
                route=state.get("route"),
            )
        return {
            "final_answer": checked.answer,
            "guardrails": checked.codes,
            "visited": ["validate_result"],
        }

    return validate_result
