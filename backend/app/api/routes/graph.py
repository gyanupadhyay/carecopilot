"""Knowledge-graph endpoint (PRD §17, §21).

``GET /api/graph`` runs one approved traversal over the authenticated
patient's own subgraph. Like every other record endpoint, no path or query
parameter names a patient — "your" is resolved from the JWT through the
identity mapping, so there is no request shape in which asking for someone
else's relationships is expressible.

``intent`` is a closed enum and ``term`` is a search word, never a fragment
of Cypher. FastAPI rejects an unknown intent with a 422 before any handler
code runs, which is the same closed set the MCP tool and the agent node pick
from — one definition, three callers.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import PatientScoped
from app.knowledge_graph import (
    INTENT_DESCRIPTIONS,
    GraphIntent,
    GraphQueryError,
    GraphUnavailable,
    query_patient_graph,
)
from app.schemas.clinical import GraphQueryOut

router = APIRouter(tags=["graph"])


@router.get("/graph", response_model=GraphQueryOut)
async def query_graph(
    ctx: PatientScoped,
    intent: Annotated[
        GraphIntent,
        Query(description="Which approved traversal to run over your own record."),
    ],
    term: Annotated[
        str,
        Query(
            max_length=80,
            description='A condition or medication name, e.g. "diabetes".',
        ),
    ] = "",
    limit: Annotated[int | None, Query(ge=1, le=500)] = None,
) -> GraphQueryOut:
    """One traversal of the patient's own relationship graph."""
    try:
        result = await query_patient_graph(
            ctx, intent=intent, term=term, limit=limit
        )
    except GraphQueryError as exc:
        # A malformed request — a term-requiring traversal without one.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except GraphUnavailable as exc:
        # The graph is a derived projection (§33), so its absence degrades
        # this endpoint rather than the application. 503 says "try again",
        # which is the truth: PostgreSQL still holds everything.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The relationship graph is unavailable.",
        ) from exc

    return GraphQueryOut(
        intent=result.intent.value,
        term=result.term,
        summary=result.summary,
        rows=result.rows,
        count=len(result.rows),
    )


@router.get("/graph/intents")
async def list_intents(ctx: PatientScoped) -> dict[str, str]:
    """What the graph can be asked, as intent → description.

    Authenticated like everything else: the catalogue is not secret, but an
    unauthenticated endpoint here would be one more surface to reason about
    for no gain.
    """
    return {intent.value: text for intent, text in INTENT_DESCRIPTIONS.items()}


__all__ = ["router"]
