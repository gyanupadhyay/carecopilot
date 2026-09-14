"""The chat endpoint.

Thin by design. It resolves the authorization context and the provider, then
hands off to :func:`app.services.chat.answer_question`. Every later phase —
routing, retrieval, tools, Text-to-SQL — changes that function and leaves
this file alone.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.deps import (
    DbSession,
    Embedder,
    Llm,
    PatientScoped,
    embedding_provider,
    llm_provider,
)
from app.auth.context import AuthorizationError
from app.models import Conversation
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationSummary,
    MessageOut,
)
from app.services import chat as chat_service
from app.services import conversation as memory

router = APIRouter(tags=["chat"])


#: Re-exported so existing dependency overrides keep addressing the same
#: callable after the providers moved to app.api.deps.
provider = llm_provider
embedder = embedding_provider


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    ctx: PatientScoped,
    session: DbSession,
    llm: Llm,
    embed: Embedder,
) -> ChatResponse:
    question = payload.message.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Message cannot be empty.",
        )

    turn = await chat_service.answer_question(
        session,
        ctx,
        question=question,
        conversation_id=payload.conversation_id,
        llm=llm,
        embedder=embed,
    )
    return turn.response


def _sse(event: chat_service.StreamEvent) -> str:
    """Render one Server-Sent Event frame.

    ``json.dumps`` with no newlines matters: a literal newline inside the
    data payload would terminate the frame early and split one event into
    two malformed ones.
    """
    payload = json.dumps(event.data, separators=(",", ":"), default=str)
    return f"event: {event.name}\ndata: {payload}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    ctx: PatientScoped,
    llm: Llm,
    embed: Embedder,
) -> StreamingResponse:
    """Stream the answer token by token (PRD §6).

    Only final answer text is streamed — the provider interface has no way
    to emit reasoning, so "do not stream internal reasoning" holds by
    construction rather than by filtering.

    Takes no request-scoped database session: the service opens its own for
    the life of the stream. See :func:`app.services.chat.stream_answer`.
    """
    question = payload.message.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Message cannot be empty.",
        )

    async def frames() -> AsyncIterator[str]:
        try:
            async for event in chat_service.stream_answer(
                ctx,
                question=question,
                conversation_id=payload.conversation_id,
                llm=llm,
                embedder=embed,
            ):
                yield _sse(event)
        except AuthorizationError as exc:
            # Raised before any text is streamed, but the response status is
            # already committed to 200 by then, so the refusal has to travel
            # as an event rather than as an HTTP status.
            yield _sse(chat_service.StreamEvent("error", {"detail": str(exc)}))

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Stops nginx buffering the stream into one delivery.
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    ctx: PatientScoped,
    session: DbSession,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[ConversationSummary]:
    rows = (
        await session.scalars(
            select(Conversation)
            .where(Conversation.user_id == ctx.user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        )
    ).all()
    return [ConversationSummary.model_validate(row) for row in rows]


@router.get("/conversations/{conversation_id}", response_model=list[MessageOut])
async def read_conversation(
    conversation_id: uuid.UUID, ctx: PatientScoped, session: DbSession
) -> list[MessageOut]:
    """Replay a conversation. Ownership is checked by the memory service."""
    conversation = await memory.load_conversation(session, ctx, conversation_id)
    messages = await memory.recent_messages(session, conversation, limit=100)
    return [MessageOut.model_validate(message) for message in messages]
