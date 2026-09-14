"""A deterministic provider used when no API key is configured.

Two jobs. It lets the whole request path — auth, memory, validation,
guardrails, tracing — be developed and tested without a key or a network
call. And it makes tests of *that* path deterministic, which tests that
call a real model never are.

Its output is deliberately unmistakable. A stub that produced plausible
clinical prose would eventually be screenshotted and believed; every
response here says what it is in its first line.

It is refused outside development by :func:`app.llm.factory.build_provider`,
so there is no configuration in which a deployed CareCopilot answers a
patient from this class.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from app.llm.base import (
    ChatMessage,
    Effort,
    LLMProvider,
    LLMResponse,
    StructuredResponse,
    TokenUsage,
    ToolCall,
    ToolCallResponse,
    ToolDefinition,
)
from app.llm.errors import LLMValidationError

STUB_PREFIX = "[stub LLM — no LLM_API_KEY configured]"

STUB_BODY = (
    "This deployment has no language model configured, so this text is "
    "generated locally and contains no information from any patient record. "
    "Set LLM_API_KEY in .env to enable real answers."
)


def _approx_tokens(text: str) -> int:
    """Rough token count for trace realism. Four characters per token."""
    return max(1, len(text) // 4)


class StubProvider(LLMProvider):
    name = "stub"

    def __init__(
        self,
        *,
        model: str = "stub-echo-1",
        canned: dict[type, Any] | None = None,
    ) -> None:
        self._model = model
        # Lets a test register the object a given schema should produce,
        # so structured call sites can be exercised without a real model.
        self._canned: dict[type, Any] = canned or {}
        self._tool_calls: tuple[ToolCall, ...] = ()

    @property
    def model(self) -> str:
        return self._model

    def register(self, schema: type[BaseModel], value: BaseModel) -> None:
        self._canned[schema] = value

    def _answer(self, messages: Sequence[ChatMessage]) -> str:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        # A short digest of the question makes responses distinguishable in
        # a transcript while staying byte-for-byte reproducible.
        digest = hashlib.sha256(last_user.encode("utf-8")).hexdigest()[:8]
        return f"{STUB_PREFIX} ({digest})\n\n{STUB_BODY}"

    async def generate(
        self,
        *,
        messages: Sequence[ChatMessage],
        system: str,
        max_tokens: int | None = None,
        effort: Effort | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        text = self._answer(messages)
        return LLMResponse(
            text=text,
            model=model or self._model,
            usage=TokenUsage(
                input_tokens=_approx_tokens(system)
                + sum(_approx_tokens(m.content) for m in messages),
                output_tokens=_approx_tokens(text),
            ),
            latency_ms=0,
            stop_reason="end_turn",
            provider=self.name,
        )

    async def generate_structured[T: BaseModel](
        self,
        *,
        messages: Sequence[ChatMessage],
        system: str,
        schema: type[T],
        max_tokens: int | None = None,
        effort: Effort | None = None,
        model: str | None = None,
    ) -> StructuredResponse[T]:
        registered = self._canned.get(schema)
        if registered is not None:
            value: T = registered
        else:
            # Fall back to a default-constructed instance. Schemas whose
            # fields are all required cannot be invented, and guessing
            # values would make a test pass against fiction.
            try:
                value = schema()
            except ValidationError as exc:
                raise LLMValidationError(
                    f"StubProvider cannot construct {schema.__name__}: it has "
                    "required fields. Register a canned value for it, or run "
                    "with a real LLM_API_KEY.",
                    provider=self.name,
                    model=self._model,
                ) from exc

        return StructuredResponse(
            value=value,
            model=model or self._model,
            usage=TokenUsage(
                input_tokens=sum(_approx_tokens(m.content) for m in messages),
                output_tokens=8,
            ),
            latency_ms=0,
            provider=self.name,
            raw_text=value.model_dump_json(),
        )

    def register_tool_calls(self, calls: Sequence[ToolCall]) -> None:
        """Set what the next tool-calling turn should ask for."""
        self._tool_calls = tuple(calls)

    async def generate_with_tools(
        self,
        *,
        messages: Sequence[ChatMessage],
        system: str,
        tools: Sequence[ToolDefinition],
        max_tokens: int | None = None,
        effort: Effort | None = None,
        model: str | None = None,
    ) -> ToolCallResponse:
        # No calls unless a test registered some. Inventing a plausible one
        # would make a dispatch test pass against fiction, which is the same
        # reason generate_structured refuses to guess required fields.
        offered = {tool.name for tool in tools}
        calls = tuple(call for call in self._tool_calls if call.name in offered)
        text = "" if calls else self._answer(messages)
        return ToolCallResponse(
            text=text,
            tool_calls=calls,
            model=model or self._model,
            usage=TokenUsage(
                input_tokens=sum(_approx_tokens(m.content) for m in messages),
                output_tokens=_approx_tokens(text) if text else 8,
            ),
            latency_ms=0,
            stop_reason="tool_use" if calls else "end_turn",
            provider=self.name,
        )

    async def stream(
        self,
        *,
        messages: Sequence[ChatMessage],
        system: str,
        max_tokens: int | None = None,
        effort: Effort | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        # Word by word, so the frontend's streaming path is genuinely
        # exercised rather than handed one large chunk.
        for word in self._answer(messages).split(" "):
            yield word + " "
