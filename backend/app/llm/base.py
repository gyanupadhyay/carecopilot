"""The provider-agnostic LLM interface (PRD §4).

Four capabilities, being the three §4 names plus the one the chat UI needs:

``generate``            free text — the final answer a user reads.
``generate_structured`` a validated Pydantic object — routing decisions,
                        generated SQL, anything the application will branch
                        on. Validation happens at the boundary so no caller
                        ever parses model output by hand.
``generate_with_tools`` native tool-calling: the model is given tool
                        definitions and may answer with calls instead of
                        prose. The provider *reports* the calls; it never
                        executes one. Dispatch belongs to the caller, which
                        is the layer holding the ``AuthContext`` (§40 P4).
``stream``              incremental text for the chat UI (PRD §4).

Deliberately *not* on this interface: ``temperature`` and other sampling
knobs. They are rejected outright by current Anthropic models, and exposing
a parameter that one provider silently ignores and another rejects with a
400 is worse than not having it. Determinism, where it matters, comes from
schema constraints and from doing the work in code instead of in the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from app.llm.pricing import estimate_cost_usd

Role = Literal["user", "assistant"]

#: How hard the model should work. Lower effort means fewer thinking tokens
#: and lower latency; the router and reranker use "low", user-facing answers
#: use the default.
Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    usage: TokenUsage
    latency_ms: int
    stop_reason: str | None = None
    provider: str = ""

    @property
    def estimated_cost_usd(self) -> float | None:
        return estimate_cost_usd(
            self.model,
            input_tokens=self.usage.input_tokens,
            output_tokens=self.usage.output_tokens,
            cache_read_tokens=self.usage.cache_read_tokens,
            cache_write_tokens=self.usage.cache_write_tokens,
        )

    @property
    def truncated(self) -> bool:
        """True when the answer was cut off by the output cap.

        Worth checking before showing an answer: a truncated clinical
        summary can end mid-sentence in a way that changes its meaning.
        """
        return self.stop_reason == "max_tokens"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A tool as described *to the model* (PRD §4, §15).

    ``parameters`` is a JSON Schema object. It never contains a patient
    identifier — not by convention but because it is derived from a
    :class:`~app.tools.base.ToolSpec`, whose params model has no such field.
    The model chooses *which* lookup runs; the patient it runs for comes
    from the ``AuthContext`` the caller holds (§40 P4).
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: _EMPTY_SCHEMA.copy())


#: A tool taking no arguments still needs a schema; this is the empty one.
_EMPTY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A call the model asked for. Requested, never executed here."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    #: The provider's correlation id, needed to reply to a specific call.
    call_id: str = ""


@dataclass(frozen=True, slots=True)
class ToolCallResponse:
    """What a tool-calling turn produced: prose, calls, or both."""

    text: str
    tool_calls: tuple[ToolCall, ...]
    model: str
    usage: TokenUsage
    latency_ms: int
    stop_reason: str | None = None
    provider: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    @property
    def estimated_cost_usd(self) -> float | None:
        return estimate_cost_usd(
            self.model,
            input_tokens=self.usage.input_tokens,
            output_tokens=self.usage.output_tokens,
            cache_read_tokens=self.usage.cache_read_tokens,
            cache_write_tokens=self.usage.cache_write_tokens,
        )


@dataclass(frozen=True, slots=True)
class StructuredResponse[T: BaseModel]:
    value: T
    model: str
    usage: TokenUsage
    latency_ms: int
    provider: str = ""
    raw_text: str = field(default="", repr=False)

    @property
    def estimated_cost_usd(self) -> float | None:
        return estimate_cost_usd(
            self.model,
            input_tokens=self.usage.input_tokens,
            output_tokens=self.usage.output_tokens,
            cache_read_tokens=self.usage.cache_read_tokens,
            cache_write_tokens=self.usage.cache_write_tokens,
        )


class LLMProvider(ABC):
    """What every provider must implement."""

    #: Short identifier used in traces and logs, e.g. "anthropic".
    name: str = "unknown"

    @property
    @abstractmethod
    def model(self) -> str:
        """The default model id this provider calls."""

    @abstractmethod
    async def generate(
        self,
        *,
        messages: Sequence[ChatMessage],
        system: str,
        max_tokens: int | None = None,
        effort: Effort | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Produce free text."""

    @abstractmethod
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
        """Produce an instance of ``schema``, or raise ``LLMValidationError``."""

    @abstractmethod
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
        """Offer ``tools`` to the model and report back what it asked for.

        The contract is deliberately one turn: the provider returns the
        requested calls and stops. It does not dispatch them, and it does not
        loop. Both are the caller's to own — dispatch because only the caller
        holds the ``AuthContext`` that decides whose record a tool reads, and
        the loop because an unbounded one is exactly what §8 rules out.
        """

    @abstractmethod
    def stream(
        self,
        *,
        messages: Sequence[ChatMessage],
        system: str,
        max_tokens: int | None = None,
        effort: Effort | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield answer text as it is produced.

        Only final answer text is ever yielded — never reasoning. PRD §26 and
        §25 both forbid exposing chain-of-thought, and the easiest way to
        honour that is for the interface to have no way to express it.
        """

    async def aclose(self) -> None:
        """Release connections. Safe to call more than once.

        Concrete by design: most providers hold no resources, and forcing
        every one to implement an empty method adds noise, not safety.
        """
        return None
