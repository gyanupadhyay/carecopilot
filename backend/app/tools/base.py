"""The tool protocol (PRD §15).

Every tool is named ``..._my_...`` and takes no patient identifier. That is
not a naming convention — it is the enforcement. A tool whose signature
cannot express "which patient" cannot be called with the wrong one, however
the model phrases its request, and there is no argument for a validator to
check. The patient comes from :class:`~app.auth.context.AuthContext`, which
the graph carries and the model never sees.

Each invocation is recorded on the trace with its name, duration and outcome
(PRD §26) so a developer panel can show what ran without exposing what came
back.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.context import AuthContext
from app.llm.base import ToolDefinition
from app.observability.logging import get_logger

log = get_logger(__name__)

#: Any field name that would let the model name a patient. No tool declares
#: one — :func:`_tool_schema` asserts that rather than trusting it, because
#: the day someone adds ``patient_id`` to a params model is the day the
#: "no tool takes a patient identifier" guarantee quietly stops being true.
FORBIDDEN_TOOL_FIELDS = frozenset(
    {"patient_id", "patientid", "patient", "pid", "user_id", "subject_id", "mrn"}
)


def empty_schema() -> dict[str, Any]:
    """A fresh empty-object JSON Schema, for tools taking no arguments."""
    return {"type": "object", "properties": {}, "additionalProperties": False}


def _tool_schema(params: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for a params model, refusing any patient-naming field."""
    schema = params.model_json_schema()
    offending = sorted(
        name
        for name in (schema.get("properties") or {})
        if name.lower() in FORBIDDEN_TOOL_FIELDS
    )
    if offending:
        raise ValueError(
            f"{params.__name__} declares {offending}: a tool argument cannot "
            "name a patient. Scope comes from AuthContext (PRD §40 P4)."
        )
    schema.setdefault("additionalProperties", False)
    return schema


class ToolError(Exception):
    """A tool could not complete. Carries a message safe to show a user."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool returns, ready to be put in front of the model.

    ``data`` is already validated — every tool returns Pydantic models, so
    the context builder never formats a raw row and the model never sees a
    shape the application has not vouched for.
    """

    name: str
    data: Any
    summary: str
    latency_ms: int = 0
    count: int | None = None

    def as_trace_entry(self, *, ok: bool = True) -> dict[str, Any]:
        return {"name": self.name, "ms": self.latency_ms, "ok": ok}


#: A tool is any coroutine taking (session, ctx, **kwargs) and returning a
#: ToolResult. Deliberately a plain callable rather than a class hierarchy:
#: there is nothing to inherit, and a function is easier to read and test.
ToolFn = Callable[..., Awaitable[ToolResult]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the router may select, and what to tell the model about it."""

    name: str
    description: str
    fn: ToolFn
    #: Parameters the model may supply. Never includes a patient identifier.
    params: type[BaseModel] | None = None
    tags: tuple[str, ...] = field(default=())

    def as_definition(self) -> ToolDefinition:
        """Describe this tool to a model for native tool-calling (PRD §4).

        The schema comes from ``params``, so the set of arguments the model
        can express is exactly the set the application declared. A tool with
        no ``params`` gets the empty object schema and can therefore be
        *chosen* but never *parameterised* — which is the whole of the
        guarantee for the six lookups that take no arguments at all.
        """
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=(
                _tool_schema(self.params) if self.params is not None else empty_schema()
            ),
        )


async def run_tool(
    spec: ToolSpec,
    session: AsyncSession,
    ctx: AuthContext,
    /,
    **kwargs: Any,
) -> ToolResult:
    """Invoke a tool, timing it and logging the invocation (PRD §15).

    Arguments are logged by *name* only. A tool argument can carry a search
    phrase the patient typed, and a log line is not the place for it.
    """
    started = time.perf_counter()
    try:
        result = await spec.fn(session, ctx, **kwargs)
    except ToolError:
        raise
    except Exception as exc:
        log.exception("tool.failed", tool=spec.name, args=sorted(kwargs))
        raise ToolError(f"{spec.name} could not complete.") from exc

    elapsed = int((time.perf_counter() - started) * 1000)
    log.info(
        "tool.invoked",
        tool=spec.name,
        ms=elapsed,
        args=sorted(kwargs),
        count=result.count,
        patient_id=ctx.patient_id,
    )
    return ToolResult(
        name=result.name,
        data=result.data,
        summary=result.summary,
        latency_ms=elapsed,
        count=result.count,
    )
