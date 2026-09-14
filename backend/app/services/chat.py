"""The chat turn: one question in, one validated answer out.

    load memory -> run the graph -> persist -> trace

Everything between classification and validation now lives in the LangGraph
workflow (:mod:`app.agents.graph`). This module owns the parts that are not
the agent's business: conversation memory, persistence, and turning the
final state into an HTTP response.

Memory sits outside the graph deliberately. It is the transcript the turn
happens inside rather than a step in answering it, and §26 is explicit that
memory must never act as an authorization mechanism — keeping it here means
no node can write to it.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.graph import build_graph
from app.agents.nodes import FAILURE_MESSAGE, NodeDeps
from app.agents.state import AgentState, initial_state
from app.auth.context import AuthContext
from app.auth.demo import DEMO_DISCLAIMER
from app.db.session import AppSession
from app.llm.base import ChatMessage, LLMProvider
from app.models import Conversation
from app.observability.logging import get_logger
from app.observability.trace import Trace, persist_trace
from app.prompts.summary import render_summary_for_prompt
from app.rag.embeddings import EmbeddingProvider, get_embedder
from app.rag.reranking import build_reranker
from app.schemas.chat import ChatMetadata, ChatResponse, PendingAction, Source
from app.services import clinical
from app.services import conversation as memory

log = get_logger(__name__)


@dataclass(slots=True)
class ChatTurn:
    """Everything the endpoint needs, already validated."""

    response: ChatResponse
    conversation_id: uuid.UUID


async def answer_question(
    session: AsyncSession,
    ctx: AuthContext,
    *,
    question: str,
    conversation_id: uuid.UUID | None,
    llm: LLMProvider,
    embedder: EmbeddingProvider | None = None,
) -> ChatTurn:
    trace = Trace(
        request_id=ctx.request_id or uuid.uuid4().hex,
        user_id=ctx.user_id,
        patient_id=ctx.patient_id,
        model=llm.model,
    )

    conversation, history, summary = await _prepare_turn(
        session,
        ctx,
        question=question,
        conversation_id=conversation_id,
        llm=llm,
        trace=trace,
    )

    final = await _run_graph(
        session,
        ctx,
        question=question,
        history=history,
        summary=summary,
        llm=llm,
        embedder=embedder or get_embedder(),
        trace=trace,
    )

    answer: str = final.get("final_answer") or FAILURE_MESSAGE
    sources: list[Source] = final.get("sources", [])
    guardrail_codes: list[str] = final.get("guardrails", [])
    _record_outcome(trace, final)

    response = await _finalize_turn(
        session,
        conversation,
        answer=answer,
        sources=sources,
        guardrail_codes=guardrail_codes,
        pending_action=final.get("pending_action"),
        trace=trace,
        llm=llm,
    )
    return ChatTurn(response=response, conversation_id=conversation.id)


def _record_outcome(trace: Trace, final: dict[str, Any]) -> None:
    """Put the finished turn's §26 counters on the trace.

    Both failure counts come from the final state rather than from a running
    tally, so a turn is counted by what it produced rather than by how many
    nodes happened to touch the trace on the way.

    A guardrail code and a validation error are summed, not merged: the first
    is "the answer was unacceptable", the second "a step could not complete",
    and a turn can legitimately have one of each for the same underlying
    fault. Over-counting a failed turn is the safe direction — the number is
    read to find turns worth looking at, not to bill anyone.
    """
    trace.record_outcome(
        visited=len(final.get("visited", [])),
        validation_failures=(
            len(final.get("validation_errors", [])) + len(final.get("guardrails", []))
        ),
    )


async def _finalize_turn(
    session: AsyncSession,
    conversation: Conversation,
    *,
    answer: str,
    sources: list[Source],
    guardrail_codes: list[str],
    pending_action: dict[str, Any] | None = None,
    trace: Trace,
    llm: LLMProvider,
) -> ChatResponse:
    """Persist the reply and the trace, and assemble the response body."""
    await memory.append_message(
        session,
        conversation,
        role="assistant",
        content=answer,
        route=trace.route,
        sources=[s.model_dump(mode="json") for s in sources] or None,
        meta={"guardrails": guardrail_codes} if guardrail_codes else None,
    )

    metadata = ChatMetadata(
        **_public_metadata(trace),
        guardrails=guardrail_codes,
        # The router now genuinely classifies, so the developer panel may
        # present the route as a decision rather than a default.
        router_enabled=True,
        llm_provider=llm.name,
    )
    await persist_trace(session, trace)

    return ChatResponse(
        answer=answer,
        route=trace.route or "OUT_OF_SCOPE",
        conversation_id=conversation.id,
        sources=sources,
        metadata=metadata,
        pending_action=(
            PendingAction.model_validate(pending_action) if pending_action else None
        ),
        disclaimer=DEMO_DISCLAIMER,
    )


@dataclass(slots=True)
class StreamEvent:
    """One Server-Sent Event. ``name`` becomes the SSE ``event:`` field."""

    name: str
    data: dict[str, Any]


async def stream_answer(
    ctx: AuthContext,
    *,
    question: str,
    conversation_id: uuid.UUID | None,
    llm: LLMProvider,
    embedder: EmbeddingProvider | None = None,
    session_factory: async_sessionmaker[AsyncSession] = AppSession,
) -> AsyncIterator[StreamEvent]:
    """Stream an answer as it is produced (PRD §6).

    Owns its database session rather than taking the request-scoped one.
    A dependency-provided session's lifetime is tied to the response object,
    not to the body generator, so work done while streaming can outlive the
    session that was injected — which surfaces much later as a closed-session
    error under load rather than a failure in development.

    The honest trade-off in streaming: guardrails can only run once the
    answer is complete, but by then the text has already reached the screen.
    So a blocking violation cannot withhold the text — it arrives in the
    ``done`` event as a replacement, with ``blocked`` set, and the client is
    expected to swap what it rendered. The buffered ``POST /api/chat`` has no
    such window, which is why it remains the default.

    A client that disconnects mid-stream rolls the whole turn back. A
    half-turn — a question with no answer — is worse to resume from than no
    turn at all.
    """
    trace = Trace(
        request_id=ctx.request_id or uuid.uuid4().hex,
        user_id=ctx.user_id,
        patient_id=ctx.patient_id,
        model=llm.model,
    )

    async with session_factory() as session:
        conversation, history, summary = await _prepare_turn(
            session,
            ctx,
            question=question,
            conversation_id=conversation_id,
            llm=llm,
            trace=trace,
        )

        yield StreamEvent(
            "meta",
            {
                "conversation_id": str(conversation.id),
                "request_id": trace.request_id,
                "disclaimer": DEMO_DISCLAIMER,
            },
        )

        chunks: list[str] = []
        final: dict = {}

        # The graph emits generation deltas on the "custom" channel and a
        # full state snapshot on "values" after each node; the last snapshot
        # is the finished turn.
        async for mode, payload in _stream_graph(
            session,
            ctx,
            question=question,
            history=history,
            summary=summary,
            llm=llm,
            embedder=embedder or get_embedder(),
            trace=trace,
        ):
            if mode == "custom" and payload.get("type") == "delta":
                chunks.append(payload["text"])
                yield StreamEvent("delta", {"text": payload["text"]})
            elif mode == "values":
                final = payload

        answer = final.get("final_answer") or FAILURE_MESSAGE
        sources: list[Source] = final.get("sources", [])
        guardrail_codes: list[str] = final.get("guardrails", [])
        _record_outcome(trace, final)

        # Citations are known only once retrieval has run, which is inside
        # the graph — so they arrive here rather than in the meta event.
        if sources:
            yield StreamEvent(
                "sources",
                {"sources": [s.model_dump(mode="json") for s in sources]},
            )

        response = await _finalize_turn(
            session,
            conversation,
            answer=answer,
            sources=sources,
            guardrail_codes=guardrail_codes,
            # Without this a booking asked for over SSE would render the
            # confirmation question and no way to answer it — the token
            # only reaches the client through the response body.
            pending_action=final.get("pending_action"),
            trace=trace,
            llm=llm,
        )
        await session.commit()

        payload = response.model_dump(mode="json")
        # True when the streamed text must be discarded and replaced.
        # Compared stripped: validation normalizes surrounding whitespace,
        # and a trailing newline is not a reason to make the client repaint.
        payload["replaces_streamed_text"] = answer.strip() != "".join(chunks).strip()
        yield StreamEvent("done", payload)


async def _prepare_turn(
    session: AsyncSession,
    ctx: AuthContext,
    *,
    question: str,
    conversation_id: uuid.UUID | None,
    llm: LLMProvider,
    trace: Trace,
) -> tuple[Conversation, list[ChatMessage], str | None]:
    """Memory handling, before the graph runs.

    Conversation memory is deliberately outside the graph. It is not a step
    in answering; it is the transcript the turn happens inside, and §26 is
    explicit that memory is never an authorization mechanism. Keeping it
    here means the graph receives history as plain input and has no way to
    write to it.
    """
    conversation = await memory.get_or_create_conversation(
        session, ctx, conversation_id, title=question
    )
    trace.conversation_id = conversation.id

    with trace.stage("memory"):
        prior = await memory.unsummarized_messages(session, conversation)
        selection = memory.select_history(prior)

    summary: str | None = conversation.summary
    if selection.dropped:
        with trace.stage("summarize"):
            summary = await memory.ensure_summary(
                session, conversation, selection.dropped, llm
            )

    await memory.append_message(session, conversation, role="user", content=question)
    return conversation, selection.kept, summary


async def _graph_inputs(
    session: AsyncSession,
    ctx: AuthContext,
    *,
    question: str,
    history: list[ChatMessage],
    summary: str | None,
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    trace: Trace,
    stream: bool = False,
) -> tuple[NodeDeps, AgentState]:
    """Build the graph's dependencies and starting state.

    Shared by the buffered and streaming paths so the two cannot diverge on
    scope, history or the prompt preamble — the only intended difference
    between them is how generation emits.

    ``patient_id`` is read from the authenticated context here and nowhere
    else. This is the single point at which scope enters the graph.
    """
    patient_label = await _patient_label(session, ctx)
    preamble = [
        part
        for part in (
            f"You are speaking with {patient_label}." if patient_label else None,
            render_summary_for_prompt(summary) if summary else None,
        )
        if part
    ]

    deps = NodeDeps(
        session=session,
        ctx=ctx,
        llm=llm,
        embedder=embedder,
        trace=trace,
        reranker=build_reranker(llm),
        stream=stream,
    )
    state = initial_state(
        question=question,
        user_id=ctx.user_id,
        role=ctx.role,
        patient_id=ctx.patient_id,
        history=list(history),
        system_prompt_extra="\n\n".join(preamble) or None,
    )
    return deps, state


async def _run_graph(
    session: AsyncSession,
    ctx: AuthContext,
    *,
    question: str,
    history: list[ChatMessage],
    summary: str | None,
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    trace: Trace,
) -> dict:
    """Run the workflow to completion and return the final state."""
    deps, state = await _graph_inputs(
        session,
        ctx,
        question=question,
        history=history,
        summary=summary,
        llm=llm,
        embedder=embedder,
        trace=trace,
    )
    compiled = build_graph(deps)

    with trace.stage("graph"):
        final = await compiled.ainvoke(state)

    _record_graph_outcome(trace, final)
    return final


async def _stream_graph(
    session: AsyncSession,
    ctx: AuthContext,
    *,
    question: str,
    history: list[ChatMessage],
    summary: str | None,
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    trace: Trace,
) -> AsyncIterator[tuple[str, dict]]:
    """Run the graph, yielding ``(stream_mode, payload)`` as it goes."""
    deps, state = await _graph_inputs(
        session,
        ctx,
        question=question,
        history=history,
        summary=summary,
        llm=llm,
        embedder=embedder,
        trace=trace,
        stream=True,
    )
    compiled = build_graph(deps)

    final: dict = {}
    async for mode, payload in compiled.astream(
        state, stream_mode=["custom", "values"]
    ):
        if mode == "values":
            final = payload
        yield mode, payload

    _record_graph_outcome(trace, final)


def _record_graph_outcome(trace: Trace, final: dict) -> None:
    trace.route = final.get("route") or "OUT_OF_SCOPE"
    trace.route_confidence = final.get("route_confidence")
    # The pipeline already recorded how many candidates were considered.
    # Overwriting it with the final count would make the developer panel
    # read "5 retrieved, 5 reranked", which says nothing about whether the
    # reranker had anything to choose between.
    if trace.retrieved_count is None:
        trace.retrieved_count = len(final.get("retrieved") or []) or None
    if final.get("validation_errors"):
        log.info(
            "chat.validation_notes",
            route=trace.route,
            notes=final["validation_errors"],
        )
    return final


def _public_metadata(trace: Trace) -> dict[str, object]:
    data = trace.as_public_metadata()
    # ChatMetadata owns these two; everything else maps straight across.
    data.pop("route", None)
    return {"route": trace.route or "UNKNOWN", **data}


async def _patient_label(session: AsyncSession, ctx: AuthContext) -> str | None:
    """A name for the prompt, so the assistant addresses the right person.

    Read from the database under the session's own scope — never taken from
    the request — so it cannot be used to imply a different patient.
    """
    if ctx.patient_id is None:
        return None
    patient = await clinical.get_patient_profile(session, ctx)
    if patient is None:
        return None
    return f"{patient.full_name} (record {patient.external_id})"
