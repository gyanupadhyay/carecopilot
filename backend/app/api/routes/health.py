"""Liveness and readiness.

``/api/health`` actually touches the database. A health check that only
proves the process is running is the kind that reports green while every
request fails.
"""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import DbSession
from app.config import settings
from app.db.vector import using_pgvector
from app.schemas.common import HealthResponse

router = APIRouter(tags=["system"])

API_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health(session: DbSession) -> HealthResponse:
    try:
        await session.execute(text("SELECT 1"))
        database = "ok"
    except Exception:
        # Reported, not raised: a 200 with "database: down" is more useful
        # to a load balancer's log than a 500 with no detail.
        database = "unavailable"

    return HealthResponse(
        status="ok" if database == "ok" else "degraded",
        environment=settings.environment,
        database=database,
        vector_backend="pgvector" if using_pgvector() else "array",
        version=API_VERSION,
        # Reported rather than assumed: a deployment that swapped the model
        # should say so, and the process serving the answers is the only
        # thing that cannot be wrong about which one it is.
        model=settings.llm_model,
        llm_provider=settings.llm_provider,
    )
