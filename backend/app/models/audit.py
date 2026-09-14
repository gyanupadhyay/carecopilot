"""Audit and per-request tracing.

Two separate concerns, two tables:

``audit_logs``     the immutable record of a security-sensitive event (PRD
                   §26) — who asked, what was proposed, what was confirmed,
                   what actually happened. Written on every branch,
                   including refusals, because "the system declined" is the
                   interesting entry during an incident review.

``request_traces`` one row per AI request with the stage timings and token
                   counts the developer panel renders (PRD §26).

Neither table stores prompts, retrieved chunk text, or answer bodies: the
trace records *shape* (counts, latencies, ids), not clinical content.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin
from app.models.enums import ROUTES, check_in

ACTION_OUTCOMES = ("proposed", "confirmed", "executed", "rejected", "failed")


class AuditLog(Base, TimestampMixin):
    __tablename__ = "audit_logs"
    __table_args__ = (
        CheckConstraint(check_in("outcome", ACTION_OUTCOMES), name="action_outcome"),
        Index("ix_action_audit_patient_id_created_at", "patient_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64))
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    patient_id: Mapped[int | None] = mapped_column(
        ForeignKey("patients.id", ondelete="SET NULL")
    )
    action: Mapped[str] = mapped_column(String(64))
    target_type: Mapped[str | None] = mapped_column(String(64))
    target_id: Mapped[int | None] = mapped_column(Integer)
    outcome: Mapped[str] = mapped_column(String(16))
    #: Validated action parameters, never the raw user utterance.
    params: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    detail: Mapped[str | None] = mapped_column(Text)


class RequestTrace(Base, TimestampMixin):
    __tablename__ = "request_traces"
    __table_args__ = (
        CheckConstraint(
            f"route IS NULL OR {check_in('route', ROUTES)}", name="trace_route"
        ),
        Index("ix_request_traces_created_at_route", "created_at", "route"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL")
    )
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    patient_id: Mapped[int | None] = mapped_column(
        ForeignKey("patients.id", ondelete="SET NULL")
    )
    route: Mapped[str | None] = mapped_column(String(16))
    route_confidence: Mapped[float | None] = mapped_column()
    #: The model id that was *asked for* — what the deployment is configured
    #: with, and therefore what reproduces it.
    model: Mapped[str | None] = mapped_column(String(120))
    #: The model id the provider *reported*, which is not always the one that
    #: was asked for — an alias such as ``gemini-flash-lite-latest`` resolves
    #: to a dated version. PRD §26 asks for model and model version as two
    #: fields, and this is the second: the version that actually ran. Null on
    #: a streamed turn, which never receives a response object to read.
    model_version: Mapped[str | None] = mapped_column(String(120))
    #: Which inference backend served it: "ollama", "vllm", "groq", ...
    #: PRD §26 tracks it, and §28's 8B-vs-14B comparison is uninterpretable
    #: without it — the same model id served by Ollama on CPU and by vLLM on
    #: a GPU produces latencies an order of magnitude apart.
    provider: Mapped[str | None] = mapped_column(String(32))

    total_ms: Mapped[int | None] = mapped_column(Integer)
    #: {"router": 250, "retrieval": 180, "rerank": 100, "llm": 1500, ...}
    stage_ms: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    #: [{"name": "get_appointments", "ms": 12, "ok": true}, ...]
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)

    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    estimated_cost_usd: Mapped[float | None] = mapped_column()

    retrieved_count: Mapped[int | None] = mapped_column(Integer)
    reranked_count: Mapped[int | None] = mapped_column(Integer)

    #: PRD §26's last three counters. Each is null rather than zero when
    #: nothing happened, so "no failures" and "column added after this row
    #: was written" stay distinguishable.
    agent_iterations: Mapped[int | None] = mapped_column(Integer)
    authorization_failures: Mapped[int | None] = mapped_column(Integer)
    validation_failures: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
