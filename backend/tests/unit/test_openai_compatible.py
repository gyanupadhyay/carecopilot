"""The OpenAI-compatible provider: Ollama, vLLM, Groq, Gemini, and anything
else that speaks Chat Completions (PRD §4).

The network is never touched here. What is worth testing is the translation
layer — schema tightening, the structured-output negotiation, the removal of
Qwen3's ``<think>`` blocks, and the places where this wire format disagrees
with the Anthropic one (``length`` vs ``max_tokens``, cached tokens counted
inside ``prompt_tokens`` rather than beside them). Those are the parts that
would silently report the wrong thing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from app.llm.base import ChatMessage
from app.llm.endpoints import endpoint_for
from app.llm.errors import LLMNotConfiguredError, LLMRequestError, LLMValidationError
from app.llm.openai_compatible import (
    ENDPOINTS,
    OpenAICompatibleProvider,
    _next_mode,
    _ReasoningFilter,
    _stop_reason,
    _strict_schema,
    _strip_json_fence,
    _strip_reasoning,
    _usage_of,
)


class RouteDecision(BaseModel):
    route: str
    confidence: float
    reason: str | None = None


# --- schema tightening -------------------------------------------------- #


def test_strict_schema_closes_every_object() -> None:
    """Strict mode rejects a schema that allows unlisted properties."""
    schema = _strict_schema(RouteDecision)
    assert schema["additionalProperties"] is False


def test_strict_schema_requires_optional_fields_too() -> None:
    """Optional-ness is expressed as a nullable type, not an absent key.

    Pydantic leaves ``reason`` out of ``required`` because it has a default.
    Strict mode reads that as a malformed schema, so every property has to
    be listed.
    """
    schema = _strict_schema(RouteDecision)
    assert set(schema["required"]) == {"route", "confidence", "reason"}


def test_strict_schema_recurses_into_nested_definitions() -> None:
    class Inner(BaseModel):
        value: int

    class Outer(BaseModel):
        inner: Inner

    schema = _strict_schema(Outer)
    inner = schema["$defs"]["Inner"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["value"]


# --- response translation ------------------------------------------------ #


def test_length_finish_reason_maps_to_max_tokens() -> None:
    """``LLMResponse.truncated`` checks one word; providers use two."""
    assert _stop_reason(SimpleNamespace(finish_reason="length")) == "max_tokens"
    assert _stop_reason(SimpleNamespace(finish_reason="stop")) == "stop"
    assert _stop_reason(None) is None


def test_cached_tokens_are_not_double_counted() -> None:
    """Here ``prompt_tokens`` already includes the cached ones.

    In the Anthropic format the two are disjoint. Adding them the same way
    would inflate every cached request's token count.
    """
    usage = _usage_of(
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=50,
                prompt_tokens_details=SimpleNamespace(cached_tokens=800),
            )
        )
    )
    assert usage.input_tokens == 200
    assert usage.cache_read_tokens == 800
    assert usage.total_tokens == 1050


def test_usage_survives_a_provider_that_omits_it() -> None:
    assert _usage_of(SimpleNamespace(usage=None)).total_tokens == 0


@pytest.mark.parametrize(
    "raw",
    ['```json\n{"a": 1}\n```', '```\n{"a": 1}\n```', '{"a": 1}', '  {"a": 1}  '],
)
def test_json_fences_are_stripped(raw: str) -> None:
    assert _strip_json_fence(raw) == '{"a": 1}'


def test_mode_downgrade_order_terminates() -> None:
    assert _next_mode("json_schema") == "json_object"
    assert _next_mode("json_object") == "prompt"
    assert _next_mode("prompt") is None


# --- provider behaviour -------------------------------------------------- #


class _FakeCompletions:
    """Records calls and replays scripted outcomes."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _completion(text: str, *, model: str = "test-model") -> Any:
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text), finish_reason="stop"
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None
        ),
    )


def _provider(outcomes: list[Any]) -> tuple[OpenAICompatibleProvider, _FakeCompletions]:
    provider = OpenAICompatibleProvider(
        provider="groq", api_key="test-key", model="test-model"
    )
    fake = _FakeCompletions(outcomes)
    provider._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=fake)
    )
    return provider, fake


def test_known_providers_have_an_endpoint_and_default_model() -> None:
    for name in ENDPOINTS:
        resolved = endpoint_for(name)
        assert resolved.base_url.startswith("http"), name
        assert resolved.default_model, name


def test_hosted_endpoints_are_https_and_want_a_key() -> None:
    """A hosted endpoint reached over plaintext would put the key on the wire."""
    for name, endpoint in ENDPOINTS.items():
        if not endpoint.requires_key:
            continue
        assert endpoint.base_url.startswith("https://"), name


def test_local_endpoints_resolve_their_host_from_settings() -> None:
    """PRD §3: the Ollama host differs between a shell and a container.

    Resolving it from settings rather than a constant is what lets compose
    point the backend at the ``ollama`` service without LLM_BASE_URL, which
    would redirect every other provider too.
    """
    from app.config import settings

    assert endpoint_for("ollama").base_url == settings.ollama_base_url
    assert endpoint_for("vllm").base_url == settings.vllm_base_url
    assert endpoint_for("ollama").requires_key is False
    assert endpoint_for("vllm").requires_key is False


def test_an_unknown_provider_resolves_to_an_empty_endpoint() -> None:
    """The provider, not the lookup, reports a missing base URL."""
    assert endpoint_for("not-a-provider").base_url == ""


def test_a_local_provider_builds_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama has no key to set, so demanding one would make it unusable."""
    from app.config import settings

    monkeypatch.setattr(settings, "llm_api_key", None)
    # Cleared so the endpoint default is what gets asserted. An explicit
    # LLM_MODEL outranks it by design, and the developer running this suite
    # may well have one set for a different provider.
    monkeypatch.setattr(settings, "llm_model", "")
    provider = OpenAICompatibleProvider(provider="ollama")
    assert provider.name == "ollama"
    assert provider.model == "qwen3:8b"


def test_a_hosted_provider_still_requires_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "llm_api_key", None)
    with pytest.raises(LLMNotConfiguredError):
        OpenAICompatibleProvider(provider="groq")


@pytest.mark.asyncio
async def test_system_prompt_becomes_a_message() -> None:
    """This format has no top-level system field."""
    provider, fake = _provider([_completion("hello")])
    await provider.generate(
        messages=[ChatMessage(role="user", content="hi")], system="be brief"
    )
    wire = fake.calls[0]["messages"]
    assert wire[0] == {"role": "system", "content": "be brief"}
    assert wire[1] == {"role": "user", "content": "hi"}


@pytest.mark.asyncio
async def test_effort_is_never_forwarded() -> None:
    """``reasoning_effort`` is a 400 on most models on these endpoints."""
    provider, fake = _provider([_completion("hello")])
    await provider.generate(
        messages=[ChatMessage(role="user", content="hi")], system="", effort="low"
    )
    assert "reasoning_effort" not in fake.calls[0]
    assert "output_config" not in fake.calls[0]


@pytest.mark.asyncio
async def test_structured_output_asks_for_a_strict_schema_first() -> None:
    provider, fake = _provider([_completion('{"route": "RAG", "confidence": 0.9}')])
    result = await provider.generate_structured(
        messages=[ChatMessage(role="user", content="q")],
        system="classify",
        schema=RouteDecision,
    )
    assert result.value.route == "RAG"
    assert fake.calls[0]["response_format"]["type"] == "json_schema"
    # The schema is enforced by the endpoint, so it is not also pasted into
    # the prompt.
    assert fake.calls[0]["messages"][0]["content"] == "classify"


@pytest.mark.asyncio
async def test_structured_output_steps_down_when_the_schema_is_rejected() -> None:
    """Groq returns 400 for json_schema on models that do not support it."""
    provider, fake = _provider(
        [
            LLMRequestError("rejected (400)"),
            _completion('{"route": "API", "confidence": 0.8}'),
        ]
    )
    result = await provider.generate_structured(
        messages=[ChatMessage(role="user", content="q")],
        system="classify",
        schema=RouteDecision,
    )
    assert result.value.route == "API"
    assert fake.calls[0]["response_format"]["type"] == "json_schema"
    assert fake.calls[1]["response_format"] == {"type": "json_object"}
    # Having lost the schema constraint, the prompt has to carry it.
    assert "RouteDecision" in fake.calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_a_downgrade_is_remembered_for_later_calls() -> None:
    """Re-discovering the limit per request would double every latency."""
    provider, fake = _provider(
        [
            LLMRequestError("rejected (400)"),
            _completion('{"route": "API", "confidence": 0.8}'),
            _completion('{"route": "RAG", "confidence": 0.7}'),
        ]
    )
    kwargs = {
        "messages": [ChatMessage(role="user", content="q")],
        "system": "classify",
        "schema": RouteDecision,
    }
    await provider.generate_structured(**kwargs)  # type: ignore[arg-type]
    await provider.generate_structured(**kwargs)  # type: ignore[arg-type]

    assert len(fake.calls) == 3
    assert fake.calls[2]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_prose_in_every_mode_raises_the_guardrail() -> None:
    """Exhausting the step-downs is the model's limit, not the endpoint's."""
    provider, fake = _provider([_completion("I'm afraid I can't do that.")] * 3)
    with pytest.raises(LLMValidationError, match="every structured mode"):
        await provider.generate_structured(
            messages=[ChatMessage(role="user", content="q")],
            system="",
            schema=RouteDecision,
        )
    assert len(fake.calls) == 3


@pytest.mark.asyncio
async def test_json_that_violates_the_schema_raises_the_guardrail() -> None:
    """Valid JSON is not the same as a valid RouteDecision."""
    provider, _ = _provider([_completion('{"route": "RAG"}')] * 3)
    with pytest.raises(LLMValidationError):
        await provider.generate_structured(
            messages=[ChatMessage(role="user", content="q")],
            system="",
            schema=RouteDecision,
        )


@pytest.mark.asyncio
async def test_an_endpoint_that_accepts_json_schema_and_ignores_it_steps_down() -> None:
    """The Ollama-hosted failure, and why a status code is not enough.

    ``https://ollama.com/v1`` returns 200 for a ``json_schema`` request and
    then answers in prose. A step-down keyed on the response status never
    fires, so every structured call in the process keeps asking for a mode
    the endpoint will never honour — and it presents as a model too weak to
    follow a schema, which is the wrong fix entirely. The same endpoint
    honours ``json_object``.
    """
    provider, fake = _provider(
        [
            _completion("**Intent:** Scheduling\n**Category:** Appointment"),
            _completion('{"route": "API", "confidence": 0.9, "reason": "ok"}'),
        ]
    )
    result = await provider.generate_structured(
        messages=[ChatMessage(role="user", content="q")],
        system="",
        schema=RouteDecision,
    )

    assert result.value.route == "API"
    assert fake.calls[0]["response_format"]["type"] == "json_schema"
    assert fake.calls[1]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_a_working_mode_is_remembered_across_calls() -> None:
    """Re-discovering the step-down would cost an extra call every time."""
    provider, fake = _provider(
        [
            _completion("prose, not JSON"),
            _completion('{"route": "API", "confidence": 0.9, "reason": "ok"}'),
            _completion('{"route": "RAG", "confidence": 0.8, "reason": "ok"}'),
        ]
    )
    for _ in range(2):
        await provider.generate_structured(
            messages=[ChatMessage(role="user", content="q")],
            system="",
            schema=RouteDecision,
        )

    # Three calls for two requests: one wasted discovering the downgrade,
    # and none wasted afterwards.
    assert len(fake.calls) == 3
    assert fake.calls[2]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_truncated_output_is_not_treated_as_a_wrong_mode() -> None:
    """Stepping down would make it worse — a looser mode emits more text.

    The cap is the caller's to raise, and saying so beats a step-down that
    produces the same failure twice more before reporting a parse error.
    """
    truncated = _completion('{"route": "AP')
    truncated.choices[0].finish_reason = "length"
    provider, fake = _provider([truncated])

    with pytest.raises(LLMValidationError, match="max_tokens"):
        await provider.generate_structured(
            messages=[ChatMessage(role="user", content="q")],
            system="",
            schema=RouteDecision,
        )
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_a_fenced_response_is_still_parsed() -> None:
    """Models in prompt mode wrap JSON in markdown out of habit."""
    provider, _ = _provider(
        [_completion('```json\n{"route": "RAG", "confidence": 0.5}\n```')]
    )
    result = await provider.generate_structured(
        messages=[ChatMessage(role="user", content="q")],
        system="",
        schema=RouteDecision,
    )
    assert result.value.route == "RAG"


@pytest.mark.asyncio
async def test_truncated_json_blames_max_tokens_not_the_model() -> None:
    """The real cause of a silent reranker fallback, found by the eval run.

    Truncated JSON fails to parse with "EOF while parsing", which points at
    the model's grammar when the fault is the caller's output cap.
    """
    truncated = SimpleNamespace(
        model="test-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"route": "RAG", "conf'),
                finish_reason="length",
            )
        ],
        usage=None,
    )
    provider, _ = _provider([truncated])
    with pytest.raises(LLMValidationError, match="max_tokens"):
        await provider.generate_structured(
            messages=[ChatMessage(role="user", content="q")],
            system="",
            schema=RouteDecision,
        )


@pytest.mark.asyncio
async def test_exhausting_every_mode_surfaces_the_request_error() -> None:
    """Three rejections is the end of the road, not an infinite loop."""
    provider, fake = _provider([LLMRequestError("rejected (400)")] * 3)
    with pytest.raises(LLMRequestError):
        await provider.generate_structured(
            messages=[ChatMessage(role="user", content="q")],
            system="",
            schema=RouteDecision,
        )
    assert len(fake.calls) == 3


# --- Qwen3 reasoning removal -------------------------------------------- #
#
# Qwen3 is a hybrid-thinking model: in thinking mode it emits
# <think>…</think> before the answer. PRD §26 forbids surfacing
# chain-of-thought, and a structured call needs JSON at character zero, so
# this has to be right in both the whole-response and the streaming path.


def test_a_think_block_is_removed_from_a_complete_response() -> None:
    assert (
        _strip_reasoning("<think>the patient asked about labs</think>Your A1c was 6.1.")
        == "Your A1c was 6.1."
    )


def test_text_without_a_think_block_is_untouched() -> None:
    assert _strip_reasoning("Your A1c was 6.1.") == "Your A1c was 6.1."


def test_an_unclosed_think_block_leaves_nothing() -> None:
    """The reply hit its token cap mid-reasoning; there is no answer to keep."""
    assert _strip_reasoning("<think>still working it out") == ""


def test_a_think_tag_with_attributes_is_still_removed() -> None:
    assert _strip_reasoning('<think mode="on">hmm</think>Answer.') == "Answer."


def test_structured_output_survives_a_reasoning_preamble() -> None:
    """Without stripping, the JSON parse fails on the '<' and blames the model."""
    text = '<think>API or RAG?</think>{"route": "API", "confidence": 0.9}'
    assert _strip_json_fence(_strip_reasoning(text)).startswith("{")


# --- streaming filter ---------------------------------------------------- #


def _stream(chunks: list[str]) -> str:
    f = _ReasoningFilter()
    return "".join(f.feed(c) for c in chunks) + f.flush()


def test_streamed_reasoning_is_suppressed() -> None:
    assert _stream(["<think>", "why", "</think>", "Hello."]) == "Hello."


def test_a_tag_split_across_chunks_is_still_caught() -> None:
    """The leak this prevents: '<thi' emitted, then 'nk>' swallowed."""
    assert _stream(["<thi", "nk>secret", "</thi", "nk>", "Visible."]) == "Visible."


def test_flush_returns_the_held_back_tail() -> None:
    """Without flush the last few characters are held forever and lost."""
    assert _stream(["Hi."]) == "Hi."
    assert _stream(["Your next visit is on the 4th."]) == "Your next visit is on the 4th."


def test_a_stream_with_no_reasoning_passes_through_unchanged() -> None:
    chunks = ["Your ", "next ", "appointment ", "is ", "Tuesday."]
    assert _stream(chunks) == "".join(chunks)


def test_text_before_a_think_block_is_kept() -> None:
    assert _stream(["Sure. ", "<think>x</think>", "Done."]) == "Sure. Done."


def test_a_stream_ending_inside_reasoning_yields_nothing() -> None:
    assert _stream(["<think>", "unfinished"]) == ""
