"""One provider for every OpenAI-compatible endpoint (PRD §4).

Ollama, vLLM, Groq and Google AI Studio all speak the OpenAI Chat
Completions wire format, so they share an implementation and differ only in
a base URL, a default model, and whether they want a key. Writing a class
per provider would have duplicated the structured-output handling below,
which is the only genuinely hard part — and it is what PRD §4 and Principle
11 are asking for: swapping Ollama for vLLM, or Qwen3-8B for 14B, must not
reach into the agent, RAG or business logic.

Four things here differ from the Anthropic provider. The first is specific
to self-hosted Qwen3; the rest are consequences of what this wire format
does and does not offer.

*Reasoning is stripped, not forwarded.* Qwen3 is a hybrid-thinking model: in
thinking mode it opens its reply with a ``<think>`` block and only then
answers. Passing that through would break structured calls (the JSON no
longer starts at character zero) and would put chain-of-thought into
answers, traces and logs, which PRD §26 forbids outright. So it is removed
at the provider boundary — the one place that knows the wire format — and
nothing downstream has to know the model reasons at all.

*Structured output is negotiated, not assumed.* Anthropic validates against
the schema server-side. Here, support for ``response_format`` varies by
provider and by model: some accept a full JSON Schema, some accept only
``json_object``, some accept neither. So the provider tries strict schema
first and steps down on a 4xx, remembering what worked so the fallback is
paid once per process rather than once per call. Pydantic validates the
result either way — the schema is a hint to the model, never the check.

*Effort is not forwarded.* ``reasoning_effort`` exists on some models on
both providers and is a 400 on the rest, and there is no reliable way to
know which from the model id alone. Sending it would trade a cost
optimisation for an outage. The parameter stays in the interface, is
honoured by Anthropic, and is dropped here — documented rather than
silently ignored.

*Cost is reported as unknown, not zero.* The hosted providers are being used
on free tiers, where the marginal cost really is nothing, but ``pricing.py``
returns ``None`` for an unpriced model and that is the honest answer: the
tier is a billing state that can change, not a property of the model. The
same applies, for a different reason, to a model you host yourself.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal

import openai
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
from app.llm.endpoints import ENDPOINTS, endpoint_for
from app.llm.errors import (
    LLMNotConfiguredError,
    LLMRateLimitError,
    LLMRequestError,
    LLMServiceError,
    LLMTimeoutError,
    LLMValidationError,
)
from app.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["ENDPOINTS", "OpenAICompatibleProvider", "StructuredMode"]

#: Sent as the bearer token to a local model server, which ignores it. The
#: OpenAI SDK refuses to construct a client with an empty key, so something
#: has to go here; a word saying what is going on beats a fake-looking
#: ``sk-...`` that invites someone to hunt for where it was issued.
LOCAL_PLACEHOLDER_KEY = "no-key-required"

#: How structured output is being requested. Negotiated downward on first
#: rejection and cached for the life of the provider.
StructuredMode = Literal["json_schema", "json_object", "prompt"]

#: How to ask each local server to skip Qwen3's thinking phase.
#:
#: This is not a cost optimisation, it is what makes the model usable. Qwen3
#: reasons before answering, and every structured call in this application
#: is capped — the router at 200 tokens. Measured against qwen3:8b: the
#: model spends the entire budget reasoning, stops at ``length``, and returns
#: an empty answer, so the router raises and falls back to RAG on roughly a
#: third of questions. The reasoning would then be discarded anyway, because
#: PRD §26 forbids surfacing it.
#:
#: The two servers take different flags, and each ignores the other's, so
#: they are sent per provider rather than as one union.
THINKING_OFF: dict[str, dict[str, Any]] = {
    # Ollama's OpenAI layer keeps reasoning out of `content` entirely — an
    # over-budget call returns empty content rather than a <think> block —
    # so raising the cap alone would not have found this.
    "ollama": {"reasoning": {"effort": "none"}},
    # Ollama's hosted API, so Ollama's spelling. Not inherited from the entry
    # above — each provider is looked up by its own name, and a cloud call
    # sent vLLM's `chat_template_kwargs` would be ignored, putting the
    # 200-token router back on the failure path this table exists to close.
    "ollama_cloud": {"reasoning": {"effort": "none"}},
    # vLLM renders Qwen3's chat template itself, and the template reads this.
    "vllm": {"chat_template_kwargs": {"enable_thinking": False}},
    # The HF router forwards to whichever backend serves the model, and the
    # ones that serve Qwen3 are vLLM or TGI — both render the chat template
    # themselves, so it is vLLM's spelling that reaches the template. A
    # backend that does not read it ignores an unknown extra field rather
    # than erroring, which is why sending it unconditionally is safe.
    "huggingface": {"chat_template_kwargs": {"enable_thinking": False}},
}


class OpenAICompatibleProvider(LLMProvider):
    """Talks Chat Completions to Ollama, vLLM, Groq, Google AI Studio or OpenAI."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.name = (provider or settings.llm_provider).lower()
        endpoint = endpoint_for(self.name)

        key = api_key or settings.llm_api_key
        if not key:
            if endpoint.requires_key:
                raise LLMNotConfiguredError("LLM_API_KEY is not set.", provider=self.name)
            key = LOCAL_PLACEHOLDER_KEY

        url = base_url or settings.llm_base_url or endpoint.base_url
        if not url:
            raise LLMNotConfiguredError(
                f"No base URL for provider {self.name!r}. Set LLM_BASE_URL.",
                provider=self.name,
            )

        self._model = model or settings.llm_model or endpoint.default_model
        self._structured_mode: StructuredMode = "json_schema"
        self._client = openai.AsyncOpenAI(
            api_key=key,
            base_url=url,
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
        model: str | None,
    ) -> dict[str, Any]:
        # The system prompt is a message with role "system" here, not a
        # top-level field as in the Anthropic format.
        wire: list[dict[str, str]] = []
        if system:
            wire.append({"role": "system", "content": system})
        wire.extend(m.as_dict() for m in messages)
        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "max_tokens": max_tokens or settings.llm_max_output_tokens,
            "messages": wire,
        }
        thinking_off = THINKING_OFF.get(self.name)
        if thinking_off is not None and not settings.llm_thinking:
            kwargs["extra_body"] = dict(thinking_off)
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
            messages=messages, system=system, max_tokens=max_tokens, model=model
        )
        started = time.perf_counter()
        with _translated_errors(self.name, kwargs["model"]):
            completion = await self._client.chat.completions.create(**kwargs)
        latency_ms = _elapsed_ms(started)

        choice = completion.choices[0] if completion.choices else None
        return LLMResponse(
            text=_strip_reasoning(getattr(choice.message, "content", "") or "").strip()
            if choice
            else "",
            model=completion.model or kwargs["model"],
            usage=_usage_of(completion),
            latency_ms=latency_ms,
            stop_reason=_stop_reason(choice),
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
        started = time.perf_counter()
        completion, text, value = await self._structured_call(
            messages=messages,
            system=system,
            schema=schema,
            max_tokens=max_tokens,
            model=model,
        )
        latency_ms = _elapsed_ms(started)
        resolved_model = completion.model or (model or self._model)

        return StructuredResponse(
            value=value,
            model=resolved_model,
            usage=_usage_of(completion),
            latency_ms=latency_ms,
            provider=self.name,
            raw_text=text,
        )

    async def _structured_call[T: BaseModel](
        self,
        *,
        messages: Sequence[ChatMessage],
        system: str,
        schema: type[T],
        max_tokens: int | None,
        model: str | None,
    ) -> tuple[Any, str, T]:
        """Request JSON, stepping down through the modes the endpoint takes.

        Each step-down is logged and remembered. A provider that rejects
        strict schemas rejects them for every call, so re-discovering that on
        every request would double the latency of every structured call for
        the life of the process.

        **A step-down is triggered by unusable output, not only by an error**,
        and that distinction is the whole reason the parse happens in here
        rather than in the caller. Ollama's hosted API accepts a
        ``json_schema`` request with HTTP 200 and then ignores it, answering
        in prose — so a step-down keyed on the status code never fires, the
        parse fails in the caller, and every structured call in the process
        keeps requesting a mode the endpoint will never honour. It looks
        exactly like a model too weak to follow a schema. The same endpoint
        honours ``json_object`` perfectly.
        """
        json_schema = _strict_schema(schema)

        while True:
            mode = self._structured_mode
            kwargs = self._request_kwargs(
                messages=messages,
                system=_augment_system(system, schema, json_schema, mode),
                max_tokens=max_tokens,
                model=model,
            )
            if mode == "json_schema":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": json_schema,
                        "strict": True,
                    },
                }
            elif mode == "json_object":
                kwargs["response_format"] = {"type": "json_object"}

            try:
                with _translated_errors(self.name, kwargs["model"]):
                    completion = await self._client.chat.completions.create(**kwargs)
            except LLMRequestError:
                nxt = _next_mode(mode)
                if nxt is None:
                    raise
                log.warning(
                    "llm.structured_downgrade",
                    provider=self.name,
                    model=kwargs["model"],
                    was=mode,
                    now=nxt,
                )
                self._structured_mode = nxt
                continue

            choice = completion.choices[0] if completion.choices else None
            raw = (getattr(choice.message, "content", "") or "") if choice else ""
            # Stripped before the parse, so the truncation check and the JSON
            # parse both operate on the answer rather than on a reasoning
            # preamble that would fail both for the wrong reason.
            text = _strip_reasoning(raw)
            resolved_model = completion.model or kwargs["model"]

            # Checked before parsing, because truncated JSON fails to parse
            # for a reason the parse error describes badly: "EOF while
            # parsing" points at the model's grammar when the real fault is
            # the caller's max_tokens. Saying so turns a confusing report
            # into a fix — and stepping down would not help, since a looser
            # mode produces *more* text against the same cap.
            if _stop_reason(choice) == "max_tokens":
                raise LLMValidationError(
                    f"{schema.__name__} output was cut off by max_tokens "
                    f"({len(text)} chars produced). Raise the output cap.",
                    provider=self.name,
                    model=resolved_model,
                )

            try:
                return completion, text, schema.model_validate_json(
                    _strip_json_fence(text)
                )
            except (ValidationError, ValueError) as exc:
                nxt = _next_mode(mode)
                if nxt is None:
                    # The last mode also failed, so this is the model's
                    # limitation rather than the endpoint's. Report it as
                    # such: the two are indistinguishable from the caller,
                    # and blaming the wrong one sends the fix to the wrong
                    # place.
                    raise LLMValidationError(
                        f"Model output failed {schema.__name__} validation "
                        f"in every structured mode: {exc}",
                        provider=self.name,
                        model=resolved_model,
                    ) from exc
                log.warning(
                    "llm.structured_downgrade",
                    provider=self.name,
                    model=resolved_model,
                    was=mode,
                    now=nxt,
                    # The endpoint said 200 and ignored the request. Recorded
                    # distinctly from the rejection path above, because
                    # "refused the mode" and "accepted and disregarded it"
                    # need different fixes from whoever reads this.
                    reason="unparseable",
                )
                self._structured_mode = nxt

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
            messages=messages, system=system, max_tokens=max_tokens, model=model
        )
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
            # "auto", not "required". A question the tools cannot answer
            # should produce a sentence saying so, and forcing a call turns
            # that into whichever tool looked closest.
            kwargs["tool_choice"] = "auto"

        started = time.perf_counter()
        with _translated_errors(self.name, kwargs["model"]):
            completion = await self._client.chat.completions.create(**kwargs)
        latency_ms = _elapsed_ms(started)

        choice = completion.choices[0] if completion.choices else None
        message = getattr(choice, "message", None)
        return ToolCallResponse(
            text=_strip_reasoning(getattr(message, "content", "") or "").strip(),
            tool_calls=_tool_calls_of(message),
            model=completion.model or kwargs["model"],
            usage=_usage_of(completion),
            latency_ms=latency_ms,
            stop_reason=_stop_reason(choice),
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
            messages=messages, system=system, max_tokens=max_tokens, model=model
        )
        filter_ = _ReasoningFilter()
        with _translated_errors(self.name, kwargs["model"]):
            stream = await self._client.chat.completions.create(stream=True, **kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # Only `content` is yielded. Some models on these endpoints
                # also emit `reasoning_content`; PRD §26 forbids surfacing
                # it, and not reading the field is the simplest way to keep
                # that true. Qwen3 does not use that field — it puts its
                # reasoning in `content` inside a <think> block — so the
                # filter handles what ignoring a field cannot.
                text = getattr(delta, "content", None)
                if text:
                    visible = filter_.feed(text)
                    if visible:
                        yield visible
            tail = filter_.flush()
            if tail:
                yield tail

    async def aclose(self) -> None:
        await self._client.close()


# ---------------------------------------------------------------------- #
# Reasoning removal
# ---------------------------------------------------------------------- #
#
# Qwen3 in thinking mode replies with ``<think>…</think>`` followed by the
# answer. Three separate rules make removing it non-optional: PRD §26 forbids
# logging chain-of-thought, §25 requires answers be validated against the
# retrieved context (reasoning is neither), and structured calls need JSON at
# character zero. Doing it here means one implementation rather than one per
# caller, and the rest of the application never learns the model reasons.

_OPEN_TAG = "<think"
_CLOSE_TAG = "</think>"
#: Longest tag we must not split across a chunk boundary.
_TAG_GUARD = len(_CLOSE_TAG) - 1

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
#: A block the model opened and never closed — the reply hit its token cap
#: mid-reasoning. There is no answer after it to preserve.
_UNCLOSED_THINK = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Remove any ``<think>`` block from a complete response."""
    if _OPEN_TAG not in text.lower():
        return text
    return _UNCLOSED_THINK.sub("", _THINK_BLOCK.sub("", text)).strip()


class _ReasoningFilter:
    """Removes ``<think>`` blocks from a token stream.

    A tag can arrive split across chunks — ``<thi`` then ``nk>`` — so the
    tail of the buffer that could still become one is held back instead of
    emitted. That costs at most seven characters of latency and is the
    difference between suppressing reasoning and leaking it one fragment at
    a time.

    :meth:`flush` must be called when the stream ends, or those held-back
    characters are dropped from the end of the answer.
    """

    __slots__ = ("_buffer", "_inside")

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, chunk: str) -> str:
        """Return the visible text in ``chunk``, which may be empty."""
        self._buffer += chunk
        out: list[str] = []

        while True:
            if self._inside:
                end = self._buffer.lower().find(_CLOSE_TAG)
                if end == -1:
                    # Discard reasoning, but keep what could be a partial
                    # closing tag.
                    self._buffer = self._buffer[-_TAG_GUARD:]
                    break
                self._buffer = self._buffer[end + len(_CLOSE_TAG) :]
                self._inside = False
                continue

            start = self._buffer.lower().find(_OPEN_TAG)
            if start == -1:
                if len(self._buffer) > _TAG_GUARD:
                    out.append(self._buffer[:-_TAG_GUARD])
                    self._buffer = self._buffer[-_TAG_GUARD:]
                break

            out.append(self._buffer[:start])
            # The tag may carry attributes, so skip to its '>'. If that has
            # not arrived yet, wait rather than guess where it ends.
            gt = self._buffer.find(">", start)
            if gt == -1:
                self._buffer = self._buffer[start:]
                break
            self._buffer = self._buffer[gt + 1 :]
            self._inside = True

        return "".join(out)

    def flush(self) -> str:
        """Return whatever was held back, now that no more chunks are coming."""
        if self._inside:
            # An unclosed block: the stream ended mid-reasoning.
            self._buffer = ""
            return ""
        tail, self._buffer = self._buffer, ""
        return tail


# ---------------------------------------------------------------------- #
# Structured-output helpers
# ---------------------------------------------------------------------- #


def _next_mode(mode: StructuredMode) -> StructuredMode | None:
    return {"json_schema": "json_object", "json_object": "prompt"}.get(mode)  # type: ignore[return-value]


def _augment_system(
    system: str,
    schema: type[BaseModel],
    json_schema: dict[str, Any],
    mode: StructuredMode,
) -> str:
    """Add the schema to the prompt when the endpoint cannot enforce it.

    In ``json_schema`` mode the endpoint constrains decoding and repeating
    the schema in the prompt only spends tokens.
    """
    if mode == "json_schema":
        return system
    instruction = (
        f"Respond with a single JSON object matching this schema for "
        f"{schema.__name__}. No prose, no code fence.\n"
        f"{json.dumps(json_schema)}"
    )
    return f"{system}\n\n{instruction}" if system else instruction


def _strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Pydantic's JSON Schema, tightened to what strict mode requires.

    Strict mode demands ``additionalProperties: false`` on every object and
    every property listed in ``required`` — including optional ones, which
    express optionality as a nullable type instead. Pydantic emits neither
    by default.
    """
    return _tighten(schema.model_json_schema())


def _tighten(node: Any) -> Any:
    if isinstance(node, list):
        return [_tighten(item) for item in node]
    if not isinstance(node, dict):
        return node

    out = {key: _tighten(value) for key, value in node.items()}
    if out.get("type") == "object" or "properties" in out:
        properties = out.get("properties")
        if isinstance(properties, dict):
            out["additionalProperties"] = False
            out["required"] = list(properties)
    return out


def _strip_json_fence(text: str) -> str:
    """Remove a ``` fence if the model wrapped its JSON in one.

    Common in ``prompt`` mode, where nothing constrains the output format
    and markdown habits win.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped[3:]
    if body.lower().startswith("json"):
        body = body[4:]
    return body.rsplit("```", 1)[0].strip()


# ---------------------------------------------------------------------- #
# Response helpers
# ---------------------------------------------------------------------- #


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _stop_reason(choice: Any) -> str | None:
    """Normalise finish_reason onto the vocabulary the rest of the app uses.

    ``LLMResponse.truncated`` checks for ``"max_tokens"``; this format calls
    the same condition ``"length"``. Translating here means one definition
    of "truncated" instead of one per provider.
    """
    if choice is None:
        return None
    reason = getattr(choice, "finish_reason", None)
    return "max_tokens" if reason == "length" else reason


def _tool_calls_of(message: Any) -> tuple[ToolCall, ...]:
    """Read the requested calls off a completion message.

    Malformed arguments are reported as an empty mapping rather than raised.
    A model that emits invalid JSON for one call has made a tool-argument
    error, which §27 measures — and the caller, which knows the tool's real
    signature, is better placed to reject it than this parser is.
    """
    raw = getattr(message, "tool_calls", None) or []
    calls: list[ToolCall] = []
    for item in raw:
        fn = getattr(item, "function", None)
        name = getattr(fn, "name", "") or ""
        if not name:
            continue
        try:
            arguments = json.loads(getattr(fn, "arguments", "") or "{}")
        except (TypeError, ValueError):
            log.warning("llm.tool_arguments_unparseable", tool=name)
            arguments = {}
        calls.append(
            ToolCall(
                name=name,
                arguments=arguments if isinstance(arguments, dict) else {},
                call_id=getattr(item, "id", "") or "",
            )
        )
    return tuple(calls)


def _usage_of(completion: Any) -> TokenUsage:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return TokenUsage()
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    return TokenUsage(
        # Cached tokens are reported inside prompt_tokens here, unlike the
        # Anthropic format where the two are disjoint. Subtracting keeps
        # TokenUsage.total_tokens from double-counting them.
        input_tokens=max(0, prompt_tokens - cached),
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cache_read_tokens=cached,
    )


class _translated_errors:
    """Map SDK exceptions onto :mod:`app.llm.errors`."""

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

        if isinstance(exc, openai.APITimeoutError):
            raise LLMTimeoutError(
                "The model did not respond in time.", **context
            ) from exc
        if isinstance(exc, openai.RateLimitError):
            raise LLMRateLimitError(
                "Rate limited by the model provider.",
                retry_after=_retry_after(exc),
                **context,
            ) from exc
        if isinstance(exc, openai.APIConnectionError):
            raise LLMServiceError(
                "Could not reach the model provider.", **context
            ) from exc
        if isinstance(exc, openai.APIStatusError):
            if exc.status_code >= 500:
                raise LLMServiceError(
                    f"Model provider error ({exc.status_code}).", **context
                ) from exc
            # 4xx: our request is wrong. Logged, not returned — the body can
            # echo prompt content back to the caller.
            log.error(
                "llm.request_rejected",
                status=exc.status_code,
                provider=self.provider,
                detail=str(exc)[:200],
            )
            raise LLMRequestError(
                f"The model provider rejected the request ({exc.status_code}).",
                **context,
            ) from exc
        return False


def _retry_after(exc: openai.RateLimitError) -> float | None:
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}) or {}
    try:
        return float(header.get("retry-after", ""))
    except (TypeError, ValueError):
        return None
