"""Per-request tracing (PRD §26).

Collects the stage timings, token counts and tool calls that make up one AI
request, then writes a single :class:`~app.models.audit.RequestTrace` row.

Two properties are deliberate. It records *shape*, not content — counts,
durations and identifiers, never the question, the retrieved text or the
answer. And persisting it never fails a request: a trace is diagnostic, and
losing one is much cheaper than turning a good answer into a 500.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import RequestTrace
from app.observability.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class Trace:
    request_id: str
    conversation_id: uuid.UUID | None = None
    user_id: int | None = None
    patient_id: int | None = None

    route: str | None = None
    route_confidence: float | None = None
    #: The model *asked for* — the configured id.
    model: str | None = None
    #: The model that *answered*, as the provider reported it (PRD §26 lists
    #: these as two fields, and they genuinely differ: an alias resolves
    #: server-side, so a request for `gemini-2.5-flash` comes back as a dated
    #: build and `qwen3:8b` as a specific quantisation. Comparing two
    #: benchmark runs needs the second; reproducing a deployment needs the
    #: first). ``None`` on a streamed turn, where no response object arrives
    #: to read it from.
    model_version: str | None = None
    #: The inference backend that served the answer (PRD §26). Recorded
    #: beside the model because the pair is what identifies a measurement:
    #: qwen3:8b on Ollama and on vLLM are the same weights and not the same
    #: latency.
    provider: str | None = None

    stage_ms: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None

    retrieved_count: int | None = None
    reranked_count: int | None = None
    #: Whether the question was rewritten before retrieval (PRD §19). The
    #: flag, never either question — an unrewritten follow-up retrieves
    #: badly in a way the chunk counts alone do not explain, and this is
    #: what tells the two apart in the developer panel.
    query_rewritten: bool = False
    #: Which reranker ran, and how many duplicate passages were collapsed.
    #: Both belong in the developer panel: "32 retrieved, 6 reranked" is
    #: uninterpretable without knowing what did the reranking.
    reranker: str | None = None
    deduplicated: int | None = None
    #: The validated statement Text-to-SQL ran, and how many rows it
    #: returned. Developer-panel only — see ``as_public_metadata``
    #: for why the SQL itself does not go to the patient.
    generated_sql: str | None = None
    sql_row_count: int | None = None
    #: The action proposed on this turn, if any. The *proposal* — execution
    #: happens on a different request and is recorded in audit_logs.
    action: str | None = None
    #: How many times this request asked the model for JSON against a schema,
    #: and how many of those failed to parse or validate (PRD §27, "JSON
    #: validity"). Counted rather than sampled because the failure is
    #: invisible downstream: the router falls back to RAG, the graph planner
    #: reports missing evidence, and the turn still produces an answer — so
    #: nothing else in the system distinguishes "the model returned malformed
    #: JSON" from "the model chose RAG". Shape, not content: a count, never
    #: the text that failed.
    structured_calls: int = 0
    structured_failures: int = 0
    #: Graph nodes executed on this turn (PRD §26, "agent iterations").
    #: Bounded by construction — the graph has no cycle and tool calls are
    #: capped at ``MAX_TOOL_CALLS`` — so this is a number to watch for
    #: regressions rather than a runaway to catch.
    agent_iterations: int = 0
    #: Times a data access was refused inside this turn (PRD §26). A refusal
    #: is a *handled* outcome here: the node returns a message and the turn
    #: completes, which is why counting it needs its own field — nothing
    #: else on a successful trace records that it happened.
    authorization_failures: int = 0
    #: Guardrail violations plus node-level validation errors (PRD §26).
    validation_failures: int = 0
    error: str | None = None

    _started: float = field(default_factory=time.perf_counter, repr=False)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a named stage — ``router``, ``retrieval``, ``llm``, ...

        Records the elapsed time even when the body raises, because the
        duration of a step that failed is usually the interesting one.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = int((time.perf_counter() - started) * 1000)
            self.stage_ms[name] = self.stage_ms.get(name, 0) + elapsed

    def record_tool(self, name: str, *, ms: int, ok: bool = True) -> None:
        self.tool_calls.append({"name": name, "ms": ms, "ok": ok})

    def record_structured(self, *, ok: bool) -> None:
        """Count one schema-constrained model call and whether it parsed.

        Called at every ``generate_structured`` site, on both paths. A site
        that records only its successes would make the failure rate look
        perfect precisely when it is not.
        """
        self.structured_calls += 1
        if not ok:
            self.structured_failures += 1

    def record_authorization_failure(self) -> None:
        """Count one refused data access (PRD §26)."""
        self.authorization_failures += 1

    def record_outcome(self, *, visited: int, validation_failures: int) -> None:
        """Record what the completed turn did, once the graph has finished.

        Set from the final state rather than accumulated per node: a node
        that returns a partial state and a node that raises would otherwise
        be counted differently, and the count would depend on where the turn
        stopped rather than on what it did.
        """
        self.agent_iterations = visited
        self.validation_failures = validation_failures

    def record_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None = None,
    ) -> None:
        """Accumulate token usage across every model call in this request."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if cost_usd is not None:
            self.estimated_cost_usd = (self.estimated_cost_usd or 0.0) + cost_usd

    @property
    def total_ms(self) -> int:
        return int((time.perf_counter() - self._started) * 1000)

    def as_row(self) -> RequestTrace:
        return RequestTrace(
            request_id=self.request_id,
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            patient_id=self.patient_id,
            route=self.route,
            route_confidence=self.route_confidence,
            model=self.model,
            model_version=self.model_version,
            provider=self.provider,
            total_ms=self.total_ms,
            stage_ms=dict(self.stage_ms) or None,
            tool_calls=list(self.tool_calls) or None,
            input_tokens=self.input_tokens or None,
            output_tokens=self.output_tokens or None,
            estimated_cost_usd=self.estimated_cost_usd,
            retrieved_count=self.retrieved_count,
            reranked_count=self.reranked_count,
            agent_iterations=self.agent_iterations or None,
            authorization_failures=self.authorization_failures or None,
            validation_failures=self.validation_failures or None,
            error=self.error,
        )

    def as_public_metadata(self) -> dict[str, Any]:
        """The subset safe to show in the developer panel (PRD §26).

        Stage timings and counts only. Nothing here reveals reasoning, and
        nothing here is clinical content.
        """
        return {
            "request_id": self.request_id,
            "route": self.route,
            "model": self.model,
            "model_version": self.model_version,
            # `provider` is deliberately absent: ChatMetadata already carries
            # it as `llm_provider`, and publishing it twice under two names
            # would let the panel show two answers to one question.
            "latency_ms": self.total_ms,
            "stage_ms": dict(self.stage_ms),
            "tools_used": [call["name"] for call in self.tool_calls],
            "tool_calls": [dict(call) for call in self.tool_calls],
            "input_tokens": self.input_tokens or None,
            "output_tokens": self.output_tokens or None,
            "estimated_cost_usd": self.estimated_cost_usd,
            "retrieved_chunks": self.retrieved_count,
            "reranked_chunks": self.reranked_count,
            "reranker": self.reranker,
            "deduplicated_chunks": self.deduplicated,
            "query_rewritten": self.query_rewritten,
            # The statement, not its results. Showing the SQL explains how
            # the number was reached; the rows would be a second copy of
            # clinical data on a channel that does not need one.
            "generated_sql": self.generated_sql,
            "sql_row_count": self.sql_row_count,
            "action": self.action,
            "structured_calls": self.structured_calls or None,
            "structured_failures": self.structured_failures or None,
            "agent_iterations": self.agent_iterations or None,
            # Both are counts of things that happened, not descriptions of
            # what was refused — a panel showing "1 authorization failure"
            # reveals that a boundary held, which is the opposite of leaking.
            "authorization_failures": self.authorization_failures or None,
            "validation_failures": self.validation_failures or None,
        }


async def persist_trace(session: AsyncSession, trace: Trace) -> None:
    """Write the trace row. Never raises."""
    try:
        session.add(trace.as_row())
        await session.flush()
    except Exception:
        # A duplicate request_id or a closed session must not take down a
        # request whose real work already succeeded.
        log.warning("trace.persist_failed", request_id=trace.request_id)
        await session.rollback()
