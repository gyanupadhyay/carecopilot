"""Anthropic implementation of :class:`~app.llm.base.LLMProvider`.

Retries are delegated to the SDK's own bounded backoff (``max_retries``)
rather than wrapped in a second retry layer. Two independent retry loops
multiply: three application attempts over two SDK attempts is six calls and
six times the bill for one user request. PRD §4 asks for bounded retry —
this is the bound.

Refusals are not retried. ``stop_reason == "refusal"`` is a decision, not a
fault, and asking again produces the same decision at twice the cost.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError

from app.config import settings
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
from app.llm.errors import (
    LLMNotConfiguredError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMRequestError,
    LLMServiceError,
    LLMTimeoutError,
    LLMValidationError,
)
from app.observability.logging import get_logger

log = get_logger(__name__)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        key = api_key or settings.llm_api_key
        if not key:
            raise LLMNotConfiguredError(
                "LLM_API_KEY is not set.", provider=self.name
            )

        self._model = model or settings.llm_model
        self._client = anthropic.AsyncAnthropic(
            api_key=key,
            timeout=timeout if timeout is not None else settings.llm_timeout_seconds,
            max_retries=(
                max_retries if max_retries is not None else settings.llm_max_retries
            ),
        )

    @property
    def model(self) -> str:
        return self._model

    # ------------------------------------------------------------------ #
    # Request construction
    # ------------------------------------------------------------------ #

    def _request_kwargs(
        self,
        *,
        messages: Sequence[ChatMessage],
        system: str,
        max_tokens: int | None,
        effort: Effort | None,
        model: str | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "max_tokens": max_tokens or settings.llm_max_output_tokens,
            "system": system,
            "messages": [m.as_dict() for m in messages],
        }
        if effort is not None:
            # Nested under output_config, not top-level.
            kwargs["output_config"] = {"effort": effort}
        return kwargs

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def generate(
        self,
        *,
        messages: Sequence[ChatMessage],
        system: str,
        max_tokens: int | None = None,
        effort: Effort | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        kwargs = self._request_kwargs(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            effort=effort,
            model=model,
        )
        started = time.perf_counter()
        with _translated_errors(self.name, kwargs["model"]):
            message = await self._client.messages.create(**kwargs)
        latency_ms = _elapsed_ms(started)

        _raise_on_refusal(message, provider=self.name)

        return LLMResponse(
            text=_text_of(message),
            model=message.model,
            usage=_usage_of(message),
            latency_ms=latency_ms,
            stop_reason=message.stop_reason,
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
        kwargs = self._request_kwargs(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            effort=effort,
            model=model,
        )
        started = time.perf_counter()
        with _translated_errors(self.name, kwargs["model"]):
            # messages.parse constrains the response to the schema and
            # validates it server-side; parsed_output is already a model
            # instance, so nothing here parses JSON by hand.
            message = await self._client.messages.parse(output_format=schema, **kwargs)
        latency_ms = _elapsed_ms(started)

        _raise_on_refusal(message, provider=self.name)

        value = message.parsed_output
        if value is None:
            raise LLMValidationError(
                f"Model returned no {schema.__name__} payload.",
                provider=self.name,
                model=kwargs["model"],
            )
        if not isinstance(value, schema):  # pragma: no cover - SDK guarantees this
            try:
                value = schema.model_validate(value)
            except ValidationError as exc:
                raise LLMValidationError(
                    f"Model output failed {schema.__name__} validation: {exc}",
                    provider=self.name,
                    model=kwargs["model"],
                ) from exc

        return StructuredResponse(
            value=value,
            model=message.model,
            usage=_usage_of(message),
            latency_ms=latency_ms,
            provider=self.name,
            raw_text=_text_of(message),
        )

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
        kwargs = self._request_kwargs(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            effort=effort,
            model=model,
        )
        if tools:
            kwargs["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in tools
            ]
            kwargs["tool_choice"] = {"type": "auto"}

        started = time.perf_counter()
        with _translated_errors(self.name, kwargs["model"]):
            message = await self._client.messages.create(**kwargs)
        latency_ms = _elapsed_ms(started)

        _raise_on_refusal(message, provider=self.name)

        return ToolCallResponse(
            text=_text_of(message),
            tool_calls=_tool_calls_of(message),
            model=message.model,
            usage=_usage_of(message),
            latency_ms=latency_ms,
            stop_reason=message.stop_reason,
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
        kwargs = self._request_kwargs(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            effort=effort,
            model=model,
        )
        with _translated_errors(self.name, kwargs["model"]):
            async with self._client.messages.stream(**kwargs) as stream:
                # text_stream yields only text deltas. Thinking blocks are
                # never surfaced here, which is what keeps PRD §26's "do not
                # stream internal reasoning" true by construction.
                async for chunk in stream.text_stream:
                    yield chunk

                final = await stream.get_final_message()
                _raise_on_refusal(final, provider=self.name)

    async def aclose(self) -> None:
        await self._client.close()


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _text_of(message: Any) -> str:
    """Concatenate text blocks, ignoring every other block type."""
    return "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    ).strip()


def _tool_calls_of(message: Any) -> tuple[ToolCall, ...]:
    """Read ``tool_use`` blocks, ignoring every other block type.

    Arguments arrive already parsed here — the SDK decodes ``input`` — so
    unlike the Chat Completions path there is no JSON to fail on.
    """
    return tuple(
        ToolCall(
            name=getattr(block, "name", "") or "",
            arguments=dict(getattr(block, "input", {}) or {}),
            call_id=getattr(block, "id", "") or "",
        )
        for block in message.content
        if getattr(block, "type", "") == "tool_use" and getattr(block, "name", "")
    )


def _usage_of(message: Any) -> TokenUsage:
    usage = getattr(message, "usage", None)
    if usage is None:  # pragma: no cover - always present in practice
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


def _raise_on_refusal(message: Any, *, provider: str) -> None:
    """Turn a safety refusal into a typed error.

    A refusal arrives as HTTP 200 with empty-ish content, so a caller that
    only checks for exceptions would render it as a blank answer.
    """
    if getattr(message, "stop_reason", None) != "refusal":
        return
    details = getattr(message, "stop_details", None)
    category = getattr(details, "category", None)
    log.warning("llm.refusal", provider=provider, category=category)
    raise LLMRefusalError(
        "The model declined to answer this request.",
        category=category,
        provider=provider,
        model=getattr(message, "model", ""),
    )


class _translated_errors:
    """Map SDK exceptions onto :mod:`app.llm.errors`.

    A context manager rather than a decorator so that it wraps the streaming
    generator's ``async with`` body as readably as it wraps a single call.
    """

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> bool:
        if exc is None:
            return False

        context = {"provider": self.provider, "model": self.model}

        if isinstance(exc, anthropic.APITimeoutError):
            raise LLMTimeoutError(
                "The model did not respond in time.", **context
            ) from exc
        if isinstance(exc, anthropic.RateLimitError):
            retry_after = _retry_after(exc)
            raise LLMRateLimitError(
                "Rate limited by the model provider.",
                retry_after=retry_after,
                **context,
            ) from exc
        if isinstance(exc, anthropic.APIConnectionError):
            raise LLMServiceError(
                "Could not reach the model provider.", **context
            ) from exc
        if isinstance(exc, anthropic.APIStatusError):
            if exc.status_code >= 500:
                raise LLMServiceError(
                    f"Model provider error ({exc.status_code}).", **context
                ) from exc
            # 4xx: our request is wrong. The message is logged, not returned,
            # because it can echo prompt content back to the caller.
            log.error(
                "llm.request_rejected",
                status=exc.status_code,
                provider=self.provider,
            )
            raise LLMRequestError(
                f"The model provider rejected the request ({exc.status_code}).",
                **context,
            ) from exc
        return False


def _retry_after(exc: anthropic.RateLimitError) -> float | None:
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}) or {}
    try:
        return float(header.get("retry-after", ""))
    except (TypeError, ValueError):
        return None
