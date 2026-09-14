"""The LangGraph workflow (PRD §11, §14).

Runs the compiled graph directly, with a router pinned to each route in
turn, so every branch is exercised — including the ones that are not yet
implemented, which must answer honestly rather than fall through.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, text

from app.actions.appointments import AppointmentRequest
from app.agents.graph import ROUTE_TO_NODE, build_graph, graph_shape
from app.agents.nodes import (
    API_TOOL_PLAN,
    NO_SQL_ANSWER,
    OUT_OF_SCOPE_MESSAGE,
    SQL_FAILED,
    GraphPlan,
    NodeDeps,
    ToolName,
    ToolPlan,
)
from app.agents.router import RouteDecision
from app.agents.state import MAX_TOOL_CALLS, initial_state
from app.auth.context import AuthContext
from app.knowledge_graph import GraphIntent
from app.llm.base import StructuredResponse, TokenUsage
from app.llm.errors import LLMError
from app.llm.stub import StubProvider
from app.models import Patient, User, UserPatientMapping
from app.observability.trace import Trace
from app.rag.embeddings import build_embedder
from app.sql.generator import GeneratedSQL

pytestmark = pytest.mark.integration


class PinnedRouter(StubProvider):
    """A provider whose classification is fixed by the test.

    Dispatches on the requested schema rather than answering every
    structured call with a ``RouteDecision``. More than one node now asks
    for structured output — the SQL generator wants a ``GeneratedSQL`` —
    and a double that ignores ``schema`` returns the wrong type to whichever
    caller it did not have in mind.
    """

    def __init__(
        self,
        route: str,
        *,
        sql: GeneratedSQL | None = None,
        action: AppointmentRequest | None = None,
        graph_plan: GraphPlan | None = None,
        tool_plan: ToolPlan | None = None,
    ) -> None:
        super().__init__()
        self._route = route
        self._sql = sql or GeneratedSQL(
            answerable=False, reason="no SQL pinned for this test"
        )
        self._action = action or AppointmentRequest(action="book_appointment")
        # A traversal that needs no search term, so the KG branch reaches
        # Neo4j rather than short-circuiting on an incomplete plan.
        self._graph_plan = graph_plan or GraphPlan(intent=GraphIntent.CONDITIONS)
        # The API route selects its own tools now (PRD §5). The default
        # mirrors the old fixed plan so tests written before selection
        # existed still exercise the same lookups.
        self._tool_plan = tool_plan or ToolPlan(
            tools=[
                "get_my_next_appointment",  # type: ignore[list-item]
                "get_my_medications",  # type: ignore[list-item]
                "get_my_lab_results",  # type: ignore[list-item]
            ]
        )

    async def generate_structured(self, *, messages, system, schema, **kwargs):  # type: ignore[override]
        if schema is GeneratedSQL:
            value: object = self._sql
        elif schema is AppointmentRequest:
            value = self._action
        elif schema is GraphPlan:
            value = self._graph_plan
        elif schema is ToolPlan:
            value = self._tool_plan
        elif schema is RouteDecision:
            value = RouteDecision(route=self._route, confidence=0.9, reason="pinned")
        else:  # pragma: no cover - a new structured caller needs a case here
            raise AssertionError(f"PinnedRouter has no answer for {schema.__name__}")
        return StructuredResponse(
            value=value,
            model="fake-router",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            latency_ms=1,
            provider="fake",
        )


@pytest.fixture(scope="module")
def embedder():
    return build_embedder(provider="local")


@pytest.fixture
async def ctx(session) -> AuthContext:
    patient = await session.scalar(
        select(Patient).where(Patient.external_id == "P001")
    )
    if patient is None:
        pytest.skip("P001 not seeded; run scripts/generate_data.py --reset")
    user = await session.scalar(
        select(User)
        .join(UserPatientMapping, UserPatientMapping.user_id == User.id)
        .where(UserPatientMapping.patient_id == patient.id)
    )
    return AuthContext(
        user_id=user.id, role="patient", patient_id=patient.id, request_id="graph-test"
    )


async def _run(
    session,
    ctx,
    embedder,
    *,
    route: str,
    question: str,
    sql: GeneratedSQL | None = None,
    action: AppointmentRequest | None = None,
    tool_plan: ToolPlan | None = None,
    llm=None,
) -> dict:
    llm = llm or PinnedRouter(route, sql=sql, action=action, tool_plan=tool_plan)
    trace = Trace(request_id="graph-test", user_id=ctx.user_id, patient_id=ctx.patient_id)
    deps = NodeDeps(
        session=session, ctx=ctx, llm=llm, embedder=embedder, trace=trace
    )
    compiled = build_graph(deps)
    state = initial_state(
        question=question,
        user_id=ctx.user_id,
        role=ctx.role,
        patient_id=ctx.patient_id,
    )
    return await compiled.ainvoke(state)


# --- shape --------------------------------------------------------------- #


def test_every_route_has_a_node() -> None:
    """A route the graph cannot branch on would fall through silently."""
    from app.models.enums import ROUTES

    assert set(ROUTE_TO_NODE) == set(ROUTES)


def test_every_branch_converges_on_validation() -> None:
    shape = graph_shape()
    for node in shape["classify_query"]:
        assert shape[node] == ["generate_answer"], node
    assert shape["generate_answer"] == ["validate_result"]
    assert shape["validate_result"] == ["END"]


# --- branches ------------------------------------------------------------ #


async def test_api_route_runs_tools_and_records_them(session, ctx, embedder) -> None:
    final = await _run(
        session, ctx, embedder, route="API", question="What are my medications?"
    )
    assert final["route"] == "API"
    assert "execute_api_tool" in final["visited"]
    assert final["tool_results"], "expected tool output"
    assert 0 < final["tool_calls"] <= MAX_TOOL_CALLS
    assert final["final_answer"]


async def test_rag_route_retrieves_and_cites(session, ctx, embedder) -> None:
    final = await _run(
        session,
        ctx,
        embedder,
        route="RAG",
        question="What did my doctor say about my knee pain?",
    )
    assert final["route"] == "RAG"
    assert "retrieve" in final["visited"]
    assert final["retrieved"]
    assert final["sources"]
    assert "SOURCE 1" in final["context_text"]


async def test_hybrid_route_combines_tools_and_retrieval(
    session, ctx, embedder
) -> None:
    """Demo 3: the medication diff is computed, the notes are retrieved."""
    final = await _run(
        session,
        ctx,
        embedder,
        route="HYBRID",
        question="Summarize my last visit and tell me which medications changed.",
    )
    assert final["route"] == "HYBRID"
    assert "hybrid" in final["visited"]
    assert final["tool_results"]
    assert final["retrieved"], "hybrid needs the note prose too"

    names = {result.name for result in final["tool_results"]}
    assert "get_my_last_encounter" in names
    assert "get_my_medication_changes" in names

    changes = next(
        r for r in final["tool_results"] if r.name == "get_my_medication_changes"
    )
    assert "Metformin" in changes.summary


# --- actions ------------------------------------------------------------- #


async def test_the_action_route_proposes_and_does_not_write(
    session, ctx, embedder
) -> None:
    """The central claim of the action design: the graph never writes.

    A turn that reaches the ACTION route must leave the appointment count
    unchanged and hand back a token instead.
    """
    before = await session.scalar(
        text("SELECT COUNT(*) FROM appointments WHERE patient_id = :p"),
        {"p": ctx.patient_id},
    )
    when = (datetime.now(UTC) + timedelta(days=9)).replace(
        hour=10, minute=0, second=0, microsecond=0
    )
    final = await _run(
        session,
        ctx,
        embedder,
        route="ACTION",
        question="Book me a follow-up next Tuesday at 10am.",
        action=AppointmentRequest(
            action="book_appointment",
            when=when.isoformat(),
            appointment_type="follow_up",
        ),
    )

    assert "action" in final["visited"]
    pending = final["pending_action"]
    assert pending and pending["token"]
    assert "confirm" in final["final_answer"].lower()

    after = await session.scalar(
        text("SELECT COUNT(*) FROM appointments WHERE patient_id = :p"),
        {"p": ctx.patient_id},
    )
    assert after == before, "proposing must not create an appointment"


async def test_a_proposal_is_audited_before_anything_is_confirmed(
    session, ctx, embedder
) -> None:
    """"What did the assistant do" must be answerable even for abandoned turns."""
    when = (datetime.now(UTC) + timedelta(days=11)).replace(
        hour=14, minute=0, second=0, microsecond=0
    )
    await _run(
        session,
        ctx,
        embedder,
        route="ACTION",
        question="book a follow-up",
        action=AppointmentRequest(action="book_appointment", when=when.isoformat()),
    )
    proposed = await session.scalar(
        text(
            "SELECT COUNT(*) FROM audit_logs WHERE patient_id = :p "
            "AND outcome = 'proposed'"
        ),
        {"p": ctx.patient_id},
    )
    assert proposed >= 1


async def test_a_past_date_is_refused_with_a_reason(session, ctx, embedder) -> None:
    past = (datetime.now(UTC) - timedelta(days=3)).replace(microsecond=0)
    final = await _run(
        session,
        ctx,
        embedder,
        route="ACTION",
        question="book me something last week",
        action=AppointmentRequest(action="book_appointment", when=past.isoformat()),
    )
    assert final.get("pending_action") is None
    assert "past" in final["final_answer"].lower()
    assert "action_declined" in final["guardrails"]


async def test_a_request_with_no_date_asks_for_one(session, ctx, embedder) -> None:
    """Better to ask than to invent a time the patient never said."""
    final = await _run(
        session,
        ctx,
        embedder,
        route="ACTION",
        question="book me an appointment",
        action=AppointmentRequest(action="book_appointment", when=""),
    )
    assert final.get("pending_action") is None
    assert "date and time" in final["final_answer"].lower()


# --- Text-to-SQL --------------------------------------------------------- #


async def test_text_to_sql_runs_the_query_and_reports_the_figure(
    session, ctx, embedder
) -> None:
    """End to end on the real read-only connection, against real rows."""
    final = await _run(
        session,
        ctx,
        embedder,
        route="TEXT_TO_SQL",
        question="How many lab results do I have on record?",
        sql=GeneratedSQL(
            answerable=True,
            sql="SELECT COUNT(*) AS result_count FROM lab_results",
            reason="counts the patient's lab results",
        ),
    )
    assert "text_to_sql" in final["visited"]
    assert final["generated_sql"], "the validated statement should reach the state"
    assert "QUERY RESULT" in (final["system_prompt_extra"] or "")
    assert "sql_unavailable" not in final.get("guardrails", [])


async def test_text_to_sql_sees_only_this_patient(session, ctx, embedder) -> None:
    """The count must match this patient's rows, not the whole table.

    Nothing in the generated SQL filters by patient — that is the point.
    Row-level security on the analytics role is what makes the unfiltered
    COUNT(*) return one patient's total.
    """
    final = await _run(
        session,
        ctx,
        embedder,
        route="TEXT_TO_SQL",
        question="How many lab results do I have?",
        sql=GeneratedSQL(
            answerable=True, sql="SELECT COUNT(*) AS n FROM lab_results"
        ),
    )
    facts = final["system_prompt_extra"] or ""
    reported = int(facts.split("n\n")[1].split()[0])

    total = await session.scalar(text("SELECT COUNT(*) FROM lab_results"))
    mine = await session.scalar(
        text("SELECT COUNT(*) FROM lab_results WHERE patient_id = :pid"),
        {"pid": ctx.patient_id},
    )
    assert reported == mine
    assert reported < total, "the fixture needs more than one patient to prove scoping"


async def test_text_to_sql_declines_when_the_schema_cannot_answer(
    session, ctx, embedder
) -> None:
    """"Not in the data" is an answer, and a different one from a failure."""
    final = await _run(
        session,
        ctx,
        embedder,
        route="TEXT_TO_SQL",
        question="How many steps did I walk last week?",
        sql=GeneratedSQL(answerable=False, reason="no step-count data in the schema"),
    )
    assert final["final_answer"] == NO_SQL_ANSWER
    assert "sql_unavailable" in final["guardrails"]


async def test_rejected_sql_never_reaches_the_database(
    session, ctx, embedder
) -> None:
    """A DELETE is refused by the validator, before the connection is opened.

    The read-only role would refuse it too. Both layers matter: this asserts
    the one that produces a clear answer rather than a driver error.
    """
    final = await _run(
        session,
        ctx,
        embedder,
        route="TEXT_TO_SQL",
        question="remove my records",
        sql=GeneratedSQL(answerable=True, sql="DELETE FROM lab_results"),
    )
    assert final["final_answer"] == SQL_FAILED
    assert "sql_unavailable" in final["guardrails"]
    assert not final.get("generated_sql")


async def test_out_of_scope_declines_without_retrieving(
    session, ctx, embedder
) -> None:
    final = await _run(
        session, ctx, embedder, route="OUT_OF_SCOPE", question="What is the weather?"
    )
    assert final["final_answer"] == OUT_OF_SCOPE_MESSAGE
    assert final["retrieved"] == []
    assert final["tool_calls"] == 0


# --- invariants ---------------------------------------------------------- #


async def test_validation_runs_on_every_branch(session, ctx, embedder) -> None:
    for route in ROUTE_TO_NODE:
        final = await _run(
            session, ctx, embedder, route=route, question="a question"
        )
        assert "validate_result" in final["visited"], route
        assert final["visited"][-1] == "validate_result", route


async def test_the_graph_never_rewrites_patient_scope(
    session, ctx, embedder
) -> None:
    """§40 P4: the graph transports scope, it does not decide it."""
    for route in ROUTE_TO_NODE:
        final = await _run(
            session, ctx, embedder, route=route, question="a question"
        )
        assert final["patient_id"] == ctx.patient_id, route
        assert final["user_id"] == ctx.user_id, route


async def test_tool_calls_stay_within_the_ceiling(session, ctx, embedder) -> None:
    for route in ("API", "HYBRID"):
        final = await _run(
            session, ctx, embedder, route=route, question="a question"
        )
        assert final["tool_calls"] <= MAX_TOOL_CALLS, route


# --- model-driven tool selection (PRD §5) -------------------------------- #


async def test_the_api_route_runs_the_tools_the_model_selected(
    session, demo_ctx, embedder
) -> None:
    """§5's actual requirement: the model decides which lookups run.

    Before this the route ran three tools for every question, so "when is my
    next appointment?" also fetched every medication and every lab result.
    """
    final = await _run(
        session,
        demo_ctx,
        embedder,
        route="API",
        question="When is my next appointment?",
        tool_plan=ToolPlan(tools=["get_my_next_appointment"]),  # type: ignore[list-item]
    )
    names = [r.name for r in final["tool_results"]]
    assert names == ["get_my_next_appointment"]
    assert "tool_select_fallback" not in final.get("guardrails", [])


async def test_an_empty_selection_falls_back_and_says_so(
    session, demo_ctx, embedder
) -> None:
    """The router sent it here, so "no lookup helps" is a disagreement.

    Answering anyway is right; hiding that the model declined to choose is
    not — the fallback and a good selection are indistinguishable from the
    answer alone.
    """
    final = await _run(
        session,
        demo_ctx,
        embedder,
        route="API",
        question="When is my next appointment?",
        tool_plan=ToolPlan(tools=[]),
    )
    assert "tool_select_empty" in final["guardrails"]
    assert {r.name for r in final["tool_results"]} == set(API_TOOL_PLAN)


async def test_a_failed_selection_call_falls_back_and_says_so(
    session, demo_ctx, embedder
) -> None:
    """A provider fault must not leave the route with no tools at all."""

    class BrokenSelector(PinnedRouter):
        async def generate_structured(self, *, messages, system, schema, **kwargs):  # type: ignore[override]
            if schema is ToolPlan:
                raise LLMError("selector unavailable")
            return await super().generate_structured(
                messages=messages, system=system, schema=schema, **kwargs
            )

    final = await _run(
        session,
        demo_ctx,
        embedder,
        route="API",
        question="When is my next appointment?",
        llm=BrokenSelector("API"),
    )
    assert "tool_select_fallback" in final["guardrails"]
    assert {r.name for r in final["tool_results"]} == set(API_TOOL_PLAN)


async def test_a_selection_longer_than_the_budget_is_capped(
    session, demo_ctx, embedder
) -> None:
    """The per-request ceiling still applies to a model-chosen plan.

    §11 puts the tool-call limit on the graph, not on the plan's author — a
    model asking for every tool must not be able to raise it.
    """
    every = ToolPlan(tools=[m.value for m in ToolName])  # type: ignore[list-item]
    final = await _run(
        session,
        demo_ctx,
        embedder,
        route="API",
        question="Tell me everything.",
        tool_plan=every,
    )
    assert len(final["tool_results"]) <= MAX_TOOL_CALLS
