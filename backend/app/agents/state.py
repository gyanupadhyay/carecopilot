"""The workflow state LangGraph threads between nodes (PRD §13).

The important line in this file is the comment on ``patient_id``: it is
populated from the authenticated context before the graph starts and is
never written by a node. LangGraph is orchestration, not a security
boundary (§40 P5), so the state carries the scope as a read-only fact that
the graph transports rather than decides.

``TypedDict`` with reducers rather than a Pydantic model because that is
what LangGraph merges natively — each node returns a partial dict and the
framework folds it in.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from app.rag.retrieval import RetrievedChunk
from app.schemas.chat import Source
from app.tools.base import ToolResult

#: PRD §11. A hard ceiling on tool invocations per request, so a
#: misclassification or a confused plan cannot turn into an unbounded loop.
MAX_TOOL_CALLS = 5


class AgentState(TypedDict, total=False):
    """State for one chat turn."""

    # --- input, set before the graph runs ---------------------------- #
    question: str
    #: Prior turns, already trimmed and summarized by the memory service.
    history: list[Any]
    system_prompt_extra: str | None

    # --- trusted context: written once, read everywhere -------------- #
    #: Resolved from the JWT through the identity mapping. A node that wrote
    #: to this would be making an authorization decision, which is exactly
    #: what §40 P4 forbids — nothing in app/agents/nodes assigns it.
    user_id: int
    role: str
    patient_id: int | None

    # --- routing ------------------------------------------------------ #
    route: str
    route_confidence: float
    route_reason: str

    # --- work products ------------------------------------------------ #
    tool_results: Annotated[list[ToolResult], operator.add]
    tool_calls: Annotated[int, operator.add]
    retrieved: list[RetrievedChunk]
    #: Traversal rows from the KG route (PRD §13's ``graph_results``).
    #:
    #: What the *model* reads is the rendered form in ``system_prompt_extra``;
    #: this keeps the rows themselves, because that rendering is lossy. Once
    #: "2 medication(s) linked to Type 2 diabetes mellitus: Glipizide,
    #: Metformin" is a sentence, nothing downstream can count it, filter it
    #: or assert against it without parsing prose back into data.
    #:
    #: Nothing consumes it yet — it is state the KG route produces and the
    #: graph carries. Said plainly because a field with no reader is a fair
    #: thing to challenge: §13 names it, the rows exist either way, and
    #: discarding them here is what would have to be undone later.
    graph_results: list[dict[str, Any]]
    #: Which approved traversal ran, as its ``GraphIntent`` value.
    #:
    #: Read by the edge out of ``query_graph``: a traversal that answers
    #: *what* is connected is finished when it returns rows, while one that
    #: answers *why* has only found the link and still owes the explanation,
    #: which lives in note prose rather than in the graph (PRD §18, §37
    #: Demo 4). The intent is the only thing that separates the two, and
    #: ``visited`` cannot carry it — it records that the node ran, not what
    #: it asked for.
    graph_intent: str
    context_text: str
    sources: list[Source]
    #: The validated statement that produced the answer, for the developer
    #: panel (PRD §26). Shown to developers, never to the patient: the SQL
    #: is mechanism, and a patient reading it learns nothing they asked for.
    generated_sql: str
    #: Set by the action node: a validated proposal plus the signed token
    #: that authorises it. The graph never executes it — the patient's
    #: confirmation reaches a separate endpoint (PRD §11).
    pending_action: dict[str, Any] | None

    # --- validation and output ---------------------------------------- #
    validation_errors: Annotated[list[str], operator.add]
    guardrails: Annotated[list[str], operator.add]
    final_answer: str

    #: Node names in execution order, for the developer panel (PRD §26).
    #: Not chain-of-thought: a list of steps the graph took, which is
    #: mechanism rather than reasoning.
    visited: Annotated[list[str], operator.add]


def initial_state(
    *,
    question: str,
    user_id: int,
    role: str,
    patient_id: int | None,
    history: list[Any] | None = None,
    system_prompt_extra: str | None = None,
) -> AgentState:
    """Build the starting state. The only place scope enters the graph."""
    return AgentState(
        question=question,
        history=history or [],
        system_prompt_extra=system_prompt_extra,
        user_id=user_id,
        role=role,
        patient_id=patient_id,
        route="",
        route_confidence=0.0,
        route_reason="",
        tool_results=[],
        tool_calls=0,
        retrieved=[],
        graph_results=[],
        graph_intent="",
        context_text="",
        sources=[],
        generated_sql="",
        pending_action=None,
        validation_errors=[],
        guardrails=[],
        final_answer="",
        visited=[],
    )
