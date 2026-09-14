"""Running summarization of older turns (PRD §32)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.auth.context import AuthContext
from app.llm.base import ChatMessage, LLMResponse, TokenUsage
from app.llm.errors import LLMServiceError
from app.llm.stub import StubProvider
from app.models import Patient
from app.services import conversation as memory

pytestmark = pytest.mark.integration


class FixedProvider(StubProvider):
    """Returns a known string, and records what it was asked to summarize."""

    def __init__(
        self, text: str = "Patient asked about knee pain and metformin."
    ) -> None:
        super().__init__()
        self._text = text
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    async def generate(self, *, messages, system, **kwargs) -> LLMResponse:  # type: ignore[override]
        self.calls.append((system, list(messages)))
        return LLMResponse(
            text=self._text,
            model="fake-summarizer",
            usage=TokenUsage(input_tokens=50, output_tokens=20),
            latency_ms=1,
            stop_reason="end_turn",
            provider="fake",
        )


class FailingProvider(StubProvider):
    async def generate(self, *, messages, system, **kwargs) -> LLMResponse:  # type: ignore[override]
        raise LLMServiceError("provider down", provider="fake")


@pytest.fixture
async def ctx(demo_patient: Patient, session) -> AuthContext:
    from sqlalchemy import select

    from app.models import User, UserPatientMapping

    user = await session.scalar(
        select(User)
        .join(UserPatientMapping, UserPatientMapping.user_id == User.id)
        .where(
            UserPatientMapping.patient_id == demo_patient.id,
            UserPatientMapping.is_active.is_(True),
        )
    )
    if user is None:
        pytest.skip("No seeded user for the demo patient")
    return AuthContext(
        user_id=user.id, role="patient", patient_id=demo_patient.id, request_id="itest"
    )


async def _conversation_with(session, ctx: AuthContext, contents: Sequence[str]):
    conversation = await memory.create_conversation(session, ctx, title="summary test")
    for index, content in enumerate(contents):
        await memory.append_message(
            session,
            conversation,
            role="user" if index % 2 == 0 else "assistant",
            content=content,
        )
    return conversation


async def test_no_summary_until_history_overflows(session, ctx: AuthContext) -> None:
    conversation = await _conversation_with(session, ctx, ["hello", "hi there"])
    prior = await memory.unsummarized_messages(session, conversation)
    selection = memory.select_history(prior)

    assert selection.dropped == []
    assert conversation.summary is None


async def test_overflow_is_summarized_and_stored(session, ctx: AuthContext) -> None:
    llm = FixedProvider()
    conversation = await _conversation_with(
        session, ctx, ["x" * 400, "y" * 400, "z" * 400, "recent question"]
    )

    prior = await memory.unsummarized_messages(session, conversation)
    selection = memory.select_history(prior, char_budget=500)
    assert selection.dropped, "expected overflow with a small budget"

    summary = await memory.ensure_summary(session, conversation, selection.dropped, llm)

    assert summary == "Patient asked about knee pain and metformin."
    assert conversation.summary == summary
    assert conversation.summary_through_message_id == max(
        m.id for m in selection.dropped
    )
    assert len(llm.calls) == 1

    system, messages = llm.calls[0]
    assert "compress" in system.lower()
    assert "x" * 400 in messages[0].content, "dropped turns must reach the summarizer"


async def test_summary_is_not_recomputed_when_nothing_new_overflowed(
    session, ctx: AuthContext
) -> None:
    """One summarization per overflow, not one per turn."""
    llm = FixedProvider()
    conversation = await _conversation_with(
        session, ctx, ["x" * 400, "y" * 400, "z" * 400, "recent"]
    )
    prior = await memory.unsummarized_messages(session, conversation)
    dropped = memory.select_history(prior, char_budget=500).dropped

    await memory.ensure_summary(session, conversation, dropped, llm)
    await memory.ensure_summary(session, conversation, dropped, llm)

    assert len(llm.calls) == 1


async def test_new_overflow_is_folded_into_the_existing_summary(
    session, ctx: AuthContext
) -> None:
    llm = FixedProvider()
    conversation = await _conversation_with(session, ctx, ["a" * 400, "b" * 400])
    first_batch = memory.select_history(
        await memory.unsummarized_messages(session, conversation), char_budget=100
    ).dropped
    await memory.ensure_summary(session, conversation, first_batch, llm)

    await memory.append_message(
        session, conversation, role="user", content="c" * 400
    )
    second_batch = memory.select_history(
        await memory.unsummarized_messages(session, conversation), char_budget=100
    ).dropped
    llm._text = "Merged digest."
    await memory.ensure_summary(session, conversation, second_batch, llm)

    assert len(llm.calls) == 2
    _, messages = llm.calls[1]
    assert "EARLIER SUMMARY" in messages[0].content
    assert conversation.summary == "Merged digest."


async def test_already_summarized_turns_are_not_reloaded(
    session, ctx: AuthContext
) -> None:
    """``unsummarized_messages`` starts after the summary watermark."""
    llm = FixedProvider()
    conversation = await _conversation_with(session, ctx, ["a" * 400, "b" * 400, "c"])
    dropped = memory.select_history(
        await memory.unsummarized_messages(session, conversation), char_budget=50
    ).dropped
    await memory.ensure_summary(session, conversation, dropped, llm)

    remaining = await memory.unsummarized_messages(session, conversation)
    watermark = conversation.summary_through_message_id or 0
    assert all(m.id > watermark for m in remaining)


async def test_summarization_failure_leaves_the_watermark_alone(
    session, ctx: AuthContext
) -> None:
    """A failed summary is retried next time, not silently lost."""
    conversation = await _conversation_with(session, ctx, ["a" * 400, "b" * 400])
    dropped = memory.select_history(
        await memory.unsummarized_messages(session, conversation), char_budget=100
    ).dropped

    summary = await memory.ensure_summary(
        session, conversation, dropped, FailingProvider()
    )

    assert summary is None
    assert conversation.summary is None
    assert conversation.summary_through_message_id is None
