"""The provider abstraction, pricing, and the stub (PRD §4)."""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import BaseModel

from app.llm.base import (
    ChatMessage,
    LLMResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from app.llm.errors import LLMNotConfiguredError, LLMValidationError
from app.llm.pricing import estimate_cost_usd
from app.llm.stub import STUB_PREFIX, StubProvider
from app.tools.base import FORBIDDEN_TOOL_FIELDS, ToolResult, ToolSpec


async def _unused(*args: object, **kwargs: object) -> ToolResult:  # pragma: no cover
    raise AssertionError("not called")


class RouteDecision(BaseModel):
    route: str = "UNKNOWN"
    confidence: float = 0.0


class RequiresFields(BaseModel):
    must_provide: str


# --- pricing ---------------------------------------------------------- #


def test_cost_is_estimated_for_a_known_model() -> None:
    cost = estimate_cost_usd(
        "claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert cost == pytest.approx(30.0)  # $5 in + $25 out


def test_cache_reads_are_cheaper_than_fresh_input() -> None:
    cached = estimate_cost_usd(
        "claude-opus-5",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=1_000_000,
    )
    fresh = estimate_cost_usd("claude-opus-5", input_tokens=1_000_000, output_tokens=0)
    assert cached is not None and fresh is not None
    assert cached < fresh


def test_unknown_model_costs_none_not_zero() -> None:
    """Zero would read as 'this route is free' on a dashboard."""
    cost = estimate_cost_usd("some-future-model", input_tokens=100, output_tokens=100)
    assert cost is None


def test_small_requests_do_not_round_to_zero() -> None:
    cost = estimate_cost_usd("claude-opus-5", input_tokens=500, output_tokens=200)
    assert cost is not None and cost > 0


# --- response helpers -------------------------------------------------- #


def test_truncation_is_detected_from_stop_reason() -> None:
    response = LLMResponse(
        text="cut off here",
        model="claude-opus-5",
        usage=TokenUsage(input_tokens=10, output_tokens=4096),
        latency_ms=10,
        stop_reason="max_tokens",
    )
    assert response.truncated
    assert response.estimated_cost_usd is not None


def test_usage_totals() -> None:
    usage = TokenUsage(input_tokens=10, output_tokens=5, cache_read_tokens=2)
    assert usage.total_tokens == 17


# --- stub provider ----------------------------------------------------- #


async def test_stub_generate_is_clearly_labelled() -> None:
    stub = StubProvider()
    result = await stub.generate(
        messages=[ChatMessage(role="user", content="What are my medications?")],
        system="irrelevant",
    )
    assert result.text.startswith(STUB_PREFIX)
    assert result.provider == "stub"


async def test_stub_is_deterministic() -> None:
    stub = StubProvider()
    messages = [ChatMessage(role="user", content="same question")]
    first = await stub.generate(messages=messages, system="s")
    second = await stub.generate(messages=messages, system="s")
    assert first.text == second.text


async def test_stub_distinguishes_different_questions() -> None:
    stub = StubProvider()
    a = await stub.generate(messages=[ChatMessage(role="user", content="a")], system="s")
    b = await stub.generate(messages=[ChatMessage(role="user", content="b")], system="s")
    assert a.text != b.text


async def test_stub_structured_uses_schema_defaults() -> None:
    stub = StubProvider()
    result = await stub.generate_structured(
        messages=[ChatMessage(role="user", content="classify")],
        system="s",
        schema=RouteDecision,
    )
    assert isinstance(result.value, RouteDecision)


async def test_stub_structured_refuses_to_invent_required_fields() -> None:
    """Guessing a value would make a test pass against fiction."""
    stub = StubProvider()
    with pytest.raises(LLMValidationError):
        await stub.generate_structured(
            messages=[ChatMessage(role="user", content="x")],
            system="s",
            schema=RequiresFields,
        )


async def test_stub_structured_returns_registered_value() -> None:
    stub = StubProvider()
    stub.register(RequiresFields, RequiresFields(must_provide="canned"))
    result = await stub.generate_structured(
        messages=[ChatMessage(role="user", content="x")],
        system="s",
        schema=RequiresFields,
    )
    assert result.value.must_provide == "canned"


async def test_stub_streams_in_pieces() -> None:
    stub = StubProvider()
    chunks = [
        chunk
        async for chunk in stub.stream(
            messages=[ChatMessage(role="user", content="hello there")], system="s"
        )
    ]
    assert len(chunks) > 1
    assert "".join(chunks).strip().startswith(STUB_PREFIX)


# --- factory ----------------------------------------------------------- #


def test_unknown_provider_is_rejected() -> None:
    from app.llm.factory import build_provider

    with pytest.raises(LLMNotConfiguredError):
        build_provider(provider="not-a-provider")


@pytest.fixture
def no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the key from settings for the duration of one test.

    These assertions are about what the factory does when no key is
    configured, so the absence has to be created here. Relying on the
    ambient environment made the result depend on whether the developer
    running the suite happened to have a key in ``.env``.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "llm_api_key", None)


@pytest.mark.parametrize("provider", ["anthropic", "gemini", "groq"])
def test_missing_key_falls_back_to_stub_in_development(
    provider: str, no_api_key: None
) -> None:
    from app.llm.factory import build_provider

    assert build_provider(provider=provider).name == "stub"


def test_stub_is_refused_outside_development(
    monkeypatch: pytest.MonkeyPatch, no_api_key: None
) -> None:
    from app.config import settings
    from app.llm.factory import build_provider

    monkeypatch.setattr(settings, "environment", "production")
    with pytest.raises(LLMNotConfiguredError):
        build_provider(provider="stub")
    with pytest.raises(LLMNotConfiguredError):
        build_provider(provider="anthropic")  # no key configured


@pytest.mark.parametrize("provider", ["ollama", "vllm"])
def test_a_local_provider_does_not_fall_back_to_the_stub(
    provider: str, no_api_key: None
) -> None:
    """A self-hosted server has no key, so a missing one means nothing.

    Falling back here would swap "Ollama is not running" — a message that
    names its own fix — for an assistant quietly answering patients in
    placeholder text.
    """
    from app.llm.factory import build_provider

    assert build_provider(provider=provider).name == provider


def _production_settings(**overrides: object) -> object:
    """Production settings with every *other* required secret supplied.

    So that what a test asserts about LLM_API_KEY is not confounded by a
    missing JWT_SECRET.
    """
    from app.config import Settings

    return Settings(
        environment="production",
        jwt_secret="x" * 48,
        action_token_secret="y" * 48,
        analytics_database_url="postgresql+psycopg://ro:ro@localhost:5432/db",
        llm_api_key=None,
        **overrides,  # type: ignore[arg-type]
    )


def test_a_local_provider_needs_no_key_in_production() -> None:
    """PRD §30's production path is vLLM, which has no key to demand.

    Requiring one would make the deployment the PRD actually asks for
    impossible to start.
    """
    assert _production_settings(llm_provider="vllm") is not None


def test_a_hosted_provider_still_needs_a_key_in_production() -> None:
    """The counterpart: exempting local providers must not exempt the rest."""
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        _production_settings(llm_provider="gemini")


def test_a_configured_key_builds_the_real_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The counterpart to the fallback: a key must not be ignored.

    Constructing a provider opens no connection, so this makes no network
    call — it asserts the factory's dispatch, which the stub-fallback tests
    above cannot see.
    """
    from app.config import settings
    from app.llm.factory import build_provider

    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    assert build_provider(provider="gemini").name == "gemini"
    assert build_provider(provider="groq").name == "groq"


# --- tool calling (PRD §4) --------------------------------------------- #


def _defs() -> tuple[ToolDefinition, ...]:
    from app.tools import tool_definitions

    return tool_definitions()


def test_every_registered_tool_has_a_definition() -> None:
    """The native surface and the enum surface must not drift apart."""
    from app.tools import TOOLS

    assert {d.name for d in _defs()} == {spec.name for spec in TOOLS}


def test_no_tool_definition_lets_the_model_name_a_patient() -> None:
    """§40 P4 made structural: the argument simply does not exist.

    This is the assertion that matters in this file. A tool schema carrying
    ``patient_id`` would let a model choose whose record to read, and no
    amount of prompt wording would take it back.
    """
    for definition in _defs():
        properties = definition.parameters.get("properties", {})
        named = FORBIDDEN_TOOL_FIELDS & {k.lower() for k in properties}
        assert not named, f"{definition.name} exposes {sorted(named)}"


def test_a_params_model_naming_a_patient_is_refused() -> None:
    """The check fires on construction, not on review."""

    class Leaky(BaseModel):
        patient_id: int = 0

    spec = ToolSpec(name="x", description="", fn=_unused, params=Leaky)
    with pytest.raises(ValueError, match="cannot name a patient"):
        spec.as_definition()


def test_argument_free_tools_get_an_empty_schema() -> None:
    by_name = {d.name: d for d in _defs()}
    assert by_name["get_my_medications"].parameters["properties"] == {}
    assert by_name["get_my_medications"].parameters["additionalProperties"] is False


def test_a_parameterised_tool_exposes_only_its_own_fields() -> None:
    by_name = {d.name: d for d in _defs()}
    properties = by_name["get_my_lab_results"].parameters["properties"]
    assert properties  # it does take arguments
    assert "patient_id" not in properties


@pytest.mark.anyio
async def test_the_stub_invents_no_tool_call() -> None:
    """A dispatch test must fail loudly rather than pass against fiction."""
    response = await StubProvider().generate_with_tools(
        messages=[ChatMessage(role="user", content="what are my medications?")],
        system="",
        tools=_defs(),
    )
    assert response.tool_calls == ()
    assert response.wants_tools is False
    assert STUB_PREFIX in response.text


@pytest.mark.anyio
async def test_a_registered_call_is_reported_not_executed() -> None:
    stub = StubProvider()
    stub.register_tool_calls([ToolCall(name="get_my_medications")])
    response = await stub.generate_with_tools(
        messages=[ChatMessage(role="user", content="meds?")],
        system="",
        tools=_defs(),
    )
    assert [c.name for c in response.tool_calls] == ["get_my_medications"]
    assert response.stop_reason == "tool_use"


@pytest.mark.anyio
async def test_a_call_for_a_tool_that_was_not_offered_is_dropped() -> None:
    """The provider reports calls; it does not smuggle in unoffered ones."""
    stub = StubProvider()
    stub.register_tool_calls([ToolCall(name="get_every_patient")])
    response = await stub.generate_with_tools(
        messages=[ChatMessage(role="user", content="everything")],
        system="",
        tools=_defs(),
    )
    assert response.tool_calls == ()


def test_unparseable_tool_arguments_do_not_raise() -> None:
    """A bad-JSON call is an argument error to measure, not a crash."""
    from app.llm.openai_compatible import _tool_calls_of

    class _Fn:
        name = "get_my_lab_results"
        arguments = "{not json"

    class _Call:
        id = "call_1"
        function = _Fn()

    class _Message:
        tool_calls: ClassVar[list[object]] = [_Call()]

    calls = _tool_calls_of(_Message())
    assert calls[0].name == "get_my_lab_results"
    assert calls[0].arguments == {}
