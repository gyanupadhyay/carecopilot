"""Short-term conversation memory (PRD §32).

Two things this module is responsible for.

*Ownership.* A conversation id travels to the browser and comes back on the
next turn. Loading one therefore checks that it belongs to the requesting
user, and a mismatch is a refusal — not a new empty conversation, which
would hide the probe.

*Bounded history.* The API is stateless, so every turn resends the
transcript. Left alone that grows without limit, and cost, latency and the
chance of the model losing the thread all grow with it. Recent turns are
replayed verbatim up to a character budget; everything older is folded into
a running summary (PRD §32).

The summary is stored on the conversation and extended incrementally, so a
long conversation costs one summarization per overflow rather than one per
turn. When summarization fails the turn still proceeds on the verbatim
window alone — losing older context degrades an answer, while failing the
request denies one.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth.context import AuthContext, AuthorizationError
from app.config import settings
from app.llm.base import ChatMessage, LLMProvider
from app.llm.errors import LLMError
from app.models import Conversation, Message
from app.observability.logging import get_logger
from app.prompts.summary import SUMMARY_SYSTEM_PROMPT, build_summary_request

log = get_logger(__name__)


async def load_conversation(
    session: AsyncSession, ctx: AuthContext, conversation_id: uuid.UUID
) -> Conversation:
    """Load a conversation the caller owns, or raise."""
    conversation = await session.scalar(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
    )
    if conversation is None or conversation.user_id != ctx.user_id:
        # One message for "no such conversation" and "not yours": telling
        # them apart would confirm that an id exists.
        raise AuthorizationError("Conversation not found.")
    return conversation


async def create_conversation(
    session: AsyncSession, ctx: AuthContext, *, title: str | None = None
) -> Conversation:
    conversation = Conversation(
        user_id=ctx.user_id,
        patient_id=ctx.patient_id,
        title=_title_from(title),
    )
    session.add(conversation)
    await session.flush()
    return conversation


async def get_or_create_conversation(
    session: AsyncSession,
    ctx: AuthContext,
    conversation_id: uuid.UUID | None,
    *,
    title: str | None = None,
) -> Conversation:
    if conversation_id is None:
        return await create_conversation(session, ctx, title=title)
    return await load_conversation(session, ctx, conversation_id)


def _title_from(text: str | None) -> str | None:
    """A short label for the conversation list.

    Derived from the opening question, truncated on a word boundary. Stored
    because the alternative — generating one with a model call — spends a
    request on something a substring answers.
    """
    if not text:
        return None
    cleaned = " ".join(text.split())
    if len(cleaned) <= 60:
        return cleaned
    return cleaned[:57].rsplit(" ", 1)[0] + "…"


async def append_message(
    session: AsyncSession,
    conversation: Conversation,
    *,
    role: str,
    content: str,
    route: str | None = None,
    sources: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
) -> Message:
    message = Message(
        conversation_id=conversation.id,
        role=role,
        content=content,
        route=route,
        sources=sources or None,
        meta=meta or None,
    )
    session.add(message)
    await session.flush()
    return message


async def recent_messages(
    session: AsyncSession, conversation: Conversation, *, limit: int | None = None
) -> list[Message]:
    """The most recent messages, oldest first."""
    turns = limit or settings.chat_history_turns
    rows = (
        await session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(turns * 2)  # a turn is a user message plus a reply
        )
    ).all()
    return list(reversed(rows))


async def unsummarized_messages(
    session: AsyncSession, conversation: Conversation, *, cap: int = 200
) -> list[Message]:
    """Every message not yet folded into the summary, oldest first.

    Bounded by ``cap`` so that a pathologically long conversation cannot
    load without limit. Anything older than the cap is neither replayed nor
    summarized — an acceptable loss at 200 messages, and far better than an
    unbounded query on the request path.
    """
    after_id = conversation.summary_through_message_id or 0
    rows = (
        await session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation.id, Message.id > after_id)
            .order_by(Message.id.desc())
            .limit(cap)
        )
    ).all()
    return list(reversed(rows))


@dataclass(slots=True)
class HistorySelection:
    """What gets replayed verbatim, and what has to be summarized instead."""

    kept: list[ChatMessage]
    dropped: list[Message]


def select_history(
    messages: Sequence[Message], *, char_budget: int | None = None
) -> HistorySelection:
    """Split stored messages into a verbatim window and an overflow.

    The window is a *contiguous suffix* of the transcript. Skipping an
    oversized message and keeping older ones either side of it would hand
    the model a conversation with a hole in it, which reads as a non
    sequitur rather than as missing context — so the walk backwards stops at
    the first message that does not fit, and everything from there back goes
    to the summary instead.
    """
    budget = char_budget or settings.chat_history_char_budget
    replayable = [m for m in messages if m.role in ("user", "assistant")]

    start = len(replayable)  # index of the oldest message still replayed
    used = 0
    for index in range(len(replayable) - 1, -1, -1):
        cost = len(replayable[index].content)
        if cost > budget or used + cost > budget:
            break
        start = index
        used += cost

    # The Messages API requires the first replayed message to be a user
    # turn; trimming can leave an assistant reply stranded at the front.
    while start < len(replayable) and replayable[start].role != "user":
        start += 1

    kept_messages = replayable[start:]
    kept_identities = {id(message) for message in kept_messages}

    return HistorySelection(
        kept=[
            ChatMessage(role=m.role, content=m.content) for m in kept_messages
        ],
        dropped=[m for m in replayable if id(m) not in kept_identities],
    )


async def ensure_summary(
    session: AsyncSession,
    conversation: Conversation,
    dropped: Sequence[Message],
    llm: LLMProvider,
) -> str | None:
    """Fold newly-overflowed turns into the conversation's running summary.

    Returns the summary to prepend to this turn's prompt, which may be the
    existing one if there is nothing new to add — or ``None`` if the
    conversation has never overflowed.

    Summarization failure is not turn failure. If the model call fails the
    previous summary is returned unchanged and ``summary_through_message_id``
    is left alone, so the same turns are retried on the next overflow rather
    than being silently lost.
    """
    already_covered = conversation.summary_through_message_id or 0
    fresh = [m for m in dropped if (m.id or 0) > already_covered]
    if not fresh:
        return conversation.summary

    transcript = "\n\n".join(f"{m.role.upper()}: {m.content}" for m in fresh)
    request = build_summary_request(
        previous_summary=conversation.summary, transcript=transcript
    )

    try:
        result = await llm.generate(
            messages=[ChatMessage(role="user", content=request)],
            system=SUMMARY_SYSTEM_PROMPT,
            max_tokens=settings.chat_summary_max_tokens,
            effort="low",
            # The cheaper router model, when one is configured: condensing a
            # transcript does not need the model that answers questions.
            model=settings.router_model,
        )
    except LLMError as exc:
        log.warning(
            "conversation.summary_failed",
            conversation_id=str(conversation.id),
            error=type(exc).__name__,
            turns=len(fresh),
        )
        return conversation.summary

    text = result.text.strip()
    if not text:
        return conversation.summary

    conversation.summary = text
    conversation.summary_through_message_id = max((m.id or 0) for m in fresh)
    await session.flush()
    log.info(
        "conversation.summarized",
        conversation_id=str(conversation.id),
        turns=len(fresh),
        through_message_id=conversation.summary_through_message_id,
    )
    return conversation.summary
