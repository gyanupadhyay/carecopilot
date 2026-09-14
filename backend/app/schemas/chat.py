"""Chat request and response payloads (PRD §6).

``ChatResponse`` is also the output-validation boundary from PRD §25: the
answer the user sees is whatever survives constructing this model. A field
that cannot be populated honestly is left absent rather than guessed at.
"""

from __future__ import annotations

import uuid
from datetime import date as date_type
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Route = Literal[
    "API", "RAG", "KG", "HYBRID", "TEXT_TO_SQL", "ACTION", "OUT_OF_SCOPE"
]


class Source(BaseModel):
    """A citation the frontend renders under the answer.

    Ids are the application's own; nothing here is a free-text label the
    model produced, so a citation cannot point at a document that was never
    retrieved.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: int | None = None
    encounter_id: int | None = None
    chunk_id: int | None = None
    document_type: str | None = None
    title: str | None = None
    section: str | None = None
    #: The field is named ``date`` on the wire (PRD §19). The type is
    #: imported under an alias because a field named ``date`` shadows the
    #: ``date`` type inside the class body, leaving the annotation
    #: unresolvable.
    date: date_type | None = None
    #: Retrieval score, when the source came from the RAG path.
    score: float | None = None

    @property
    def citation_key(self) -> str:
        """Stable identity used by the guardrail's citation check."""
        if self.chunk_id is not None:
            return f"chunk:{self.chunk_id}"
        if self.document_id is not None:
            return f"doc:{self.document_id}"
        return f"encounter:{self.encounter_id}"


class ChatMetadata(BaseModel):
    """What the developer panel shows (PRD §26).

    Timings, counts and model identity — never reasoning, never prompt text.
    """

    request_id: str
    route: Route
    #: The model asked for, and the one the provider reported answering with.
    #: PRD §26 names both; they differ whenever an alias resolves server-side.
    model: str | None = None
    model_version: str | None = None
    latency_ms: int
    stage_ms: dict[str, int] = Field(default_factory=dict)
    tools_used: list[str] = Field(default_factory=list)
    #: The same invocations with their outcome — ``{name, ms, ok}``. Kept
    #: beside ``tools_used`` rather than replacing it: the frontend renders
    #: the names, and a tool that ran and returned nothing is invisible in
    #: that list while being exactly what a developer panel exists to show.
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    retrieved_chunks: int | None = None
    reranked_chunks: int | None = None
    reranker: str | None = None
    deduplicated_chunks: int | None = None
    #: Whether the question was rewritten to stand alone before retrieval
    #: (PRD §19). The flag only — never either form of the question.
    query_rewritten: bool = False
    #: Text-to-SQL: the statement that ran and how many rows it returned.
    #: The statement, never the rows — the answer already carries the
    #: figures, and a second copy of clinical data on the metadata channel
    #: would be one more place it has to be protected.
    generated_sql: str | None = None
    sql_row_count: int | None = None
    #: The action *proposed* on this turn. Execution happens on a separate
    #: request and is recorded in audit_logs, not here.
    action: str | None = None
    #: Schema-constrained model calls on this turn, and how many produced
    #: JSON that failed to parse or validate. Both ``None`` when the turn
    #: made none — an absent count and a count of zero say different things,
    #: and a panel rendering "0/0 valid" for a rule-routed turn would be
    #: reporting a measurement that never happened.
    structured_calls: int | None = None
    structured_failures: int | None = None
    #: Graph nodes executed, refused data accesses, and failed checks
    #: (PRD §26). ``None`` rather than 0 for the same reason as above.
    agent_iterations: int | None = None
    authorization_failures: int | None = None
    validation_failures: int | None = None
    #: Guardrail codes that fired, e.g. ["truncated_answer"]. Empty is the
    #: normal case and is surfaced so the panel can show a clean check.
    guardrails: list[str] = Field(default_factory=list)
    #: False until the router lands in Phase 5; the UI labels the route
    #: honestly rather than implying a classification that did not happen.
    router_enabled: bool = False
    llm_provider: str | None = None


class PendingAction(BaseModel):
    """A proposed write, awaiting the patient's explicit confirmation (§11).

    The token is opaque to the frontend: it holds the validated parameters
    and is signed, so the UI renders ``summary`` and passes ``token`` back
    untouched. A UI that constructed its own confirmation payload would be
    re-introducing the free-text path this design removes.
    """

    action: str
    #: What the patient is agreeing to, in words. Composed by the backend
    #: from validated values, never phrased by the model.
    summary: str
    token: str
    expires_at: str


class ActionProposeRequest(BaseModel):
    """A typed action request, for the propose endpoint and its MCP tools.

    Mirrors ``actions.appointments.AppointmentRequest`` rather than reusing
    it: that model is the schema a *language model* fills in, with
    descriptions written to steer generation, and its field constraints are
    tuned for that. This one is the public API contract, where a caller is
    entitled to a 422 naming the field it got wrong.

    Note what is absent, in both: no patient id, no appointment id, no
    status, no free text beyond a short reason. Validation against real data
    happens in ``propose``; these bounds only keep malformed input out.
    """

    action: Literal["book_appointment", "cancel_appointment"]
    #: ISO 8601, "YYYY-MM-DDTHH:MM". Empty is allowed and refused downstream
    #: with a message the caller can act on, rather than a schema error that
    #: says nothing about which appointment was meant.
    when: str = Field(default="", max_length=64)
    appointment_type: str = Field(default="follow_up", max_length=32)
    reason: str = Field(default="", max_length=200)


class ActionConfirmRequest(BaseModel):
    token: str = Field(min_length=1, max_length=4000)


class ActionResult(BaseModel):
    status: Literal["executed", "declined"]
    message: str
    appointment_id: int | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    #: Omit to start a new conversation. Must belong to the caller.
    conversation_id: uuid.UUID | None = None


class ChatResponse(BaseModel):
    answer: str
    route: Route
    conversation_id: uuid.UUID
    sources: list[Source] = Field(default_factory=list)
    metadata: ChatMetadata
    #: Present only when the turn proposed a write. Its presence is what
    #: tells the UI to render a confirm/cancel control instead of plain text.
    pending_action: PendingAction | None = None
    #: Always present, always rendered by the UI (PRD §34).
    disclaimer: str


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    created_at: Any
    updated_at: Any


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    route: str | None = None
    sources: list[dict[str, Any]] | None = None
    created_at: Any
