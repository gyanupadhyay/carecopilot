"""The §19 rewrite step: the gate, the guards, and the fallback.

The gate is what most of this file tests, because it is the part that runs
on every RAG turn and the part that is deterministic enough to pin down. The
rewrite itself is a model call, and what matters about it here is that every
way it can go wrong returns the original question.
"""

from __future__ import annotations

import pytest

from app.llm.base import ChatMessage
from app.llm.errors import LLMServiceError, LLMValidationError
from app.llm.stub import StubProvider
from app.observability.trace import Trace
from app.rag.rewrite import (
    MAX_REWRITE_CHARS,
    RewrittenQuery,
    needs_rewrite,
    rewrite_query,
)

HISTORY = [
    ChatMessage(
        role="user", content="What did the cardiologist say about my chest pain?"
    ),
    ChatMessage(role="assistant", content="Dr. Okafor noted atypical chest pain."),
]


# --- the gate ---------------------------------------------------------- #


def test_a_first_question_is_never_rewritten() -> None:
    """Nothing to resolve against, whatever pronouns it contains."""
    assert needs_rewrite("What did he say about it?", []) is False


@pytest.mark.parametrize(
    "question",
    [
        "And what did he suggest?",
        "What about that?",
        "Any side effects?",
        "Why?",
        "Did they change it?",
        "What was her advice?",
        "So what now?",
    ],
)
def test_context_dependent_questions_are_caught(question: str) -> None:
    assert needs_rewrite(question, HISTORY) is True


@pytest.mark.parametrize(
    "question",
    [
        "What were my most recent cholesterol results?",
        "When is my next appointment with Dr. Smith?",
        "Which medications was I prescribed at my last visit?",
    ],
)
def test_self_contained_questions_are_left_alone(question: str) -> None:
    """The gate exists to avoid paying for a model call on these."""
    assert needs_rewrite(question, HISTORY) is False


def test_an_empty_question_is_not_sent_to_the_model() -> None:
    assert needs_rewrite("   ", HISTORY) is False


# --- the fallback ------------------------------------------------------ #


@pytest.mark.anyio
async def test_a_question_that_needs_nothing_skips_the_model() -> None:
    """No call at all, not a call whose result is discarded."""

    class Exploding(StubProvider):
        async def generate_structured(self, **kwargs: object) -> object:
            raise AssertionError("the gate should have prevented this call")

    question = "What were my most recent cholesterol results?"
    assert await rewrite_query(
        question=question, history=HISTORY, llm=Exploding()
    ) == question


@pytest.mark.anyio
@pytest.mark.parametrize("failure", [LLMServiceError, LLMValidationError])
async def test_an_unreachable_model_returns_the_original(failure: type) -> None:
    """The worst case is the behaviour from before this module existed."""

    class Failing(StubProvider):
        async def generate_structured(self, **kwargs: object) -> object:
            raise failure("nope")

    assert await rewrite_query(
        question="And what did he suggest?", history=HISTORY, llm=Failing()
    ) == "And what did he suggest?"


@pytest.mark.anyio
async def test_a_runaway_rewrite_is_rejected() -> None:
    """A model that pasted the conversation in retrieves worse than a pronoun."""
    stub = StubProvider()
    stub.register(RewrittenQuery, RewrittenQuery(question="x" * (MAX_REWRITE_CHARS + 1)))
    assert await rewrite_query(
        question="And what did he suggest?", history=HISTORY, llm=stub
    ) == "And what did he suggest?"


@pytest.mark.anyio
async def test_an_empty_rewrite_is_rejected() -> None:
    stub = StubProvider()
    stub.register(RewrittenQuery, RewrittenQuery(question="   "))
    assert await rewrite_query(
        question="Any side effects?", history=HISTORY, llm=stub
    ) == "Any side effects?"


# --- the happy path ----------------------------------------------------- #


@pytest.mark.anyio
async def test_a_resolved_question_replaces_the_original() -> None:
    stub = StubProvider()
    stub.register(RewrittenQuery, RewrittenQuery(question="What did Dr. Okafor suggest?"))
    result = await rewrite_query(
        question="And what did he suggest?", history=HISTORY, llm=stub
    )
    assert result == "What did Dr. Okafor suggest?"


@pytest.mark.anyio
async def test_the_trace_records_that_it_happened_but_not_the_text() -> None:
    """PRD §26: shape, never content."""
    trace = Trace(request_id="req-1")
    stub = StubProvider()
    stub.register(RewrittenQuery, RewrittenQuery(question="What did Dr. Okafor suggest?"))

    await rewrite_query(
        question="And what did he suggest?", history=HISTORY, llm=stub, trace=trace
    )

    assert trace.query_rewritten is True
    published = repr(trace.as_public_metadata())
    assert "Okafor" not in published
    assert "he suggest" not in published


@pytest.mark.anyio
async def test_an_unchanged_rewrite_is_not_reported_as_one() -> None:
    """The model returning the question verbatim is not a rewrite."""
    trace = Trace(request_id="req-1")
    stub = StubProvider()
    stub.register(RewrittenQuery, RewrittenQuery(question="Any side effects?"))

    await rewrite_query(
        question="Any side effects?", history=HISTORY, llm=stub, trace=trace
    )
    assert trace.query_rewritten is False
