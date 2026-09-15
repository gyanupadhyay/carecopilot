"""The analytics endpoint (PRD §16).

``POST /api/analytics`` answers a counting or averaging question about the
caller's own record by generating SQL, validating it, and running it as the
read-only role. It is what ``run_my_patient_analytics`` proxies, and it is
the only way SQL reaches the database on behalf of a caller.

What it does **not** accept is a SQL statement. The request body is a
question in English; the SQL is produced inside, parsed with sqlglot,
checked against the allowlisted schema, capped, and executed on a connection
whose role holds SELECT on four tables and cannot bypass the row-level
security keyed to the caller's patient id. A caller who sends SQL is sending
a string that will be read as a question and almost certainly refused —
which is the point: there is no parameter through which a statement can
arrive.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import Llm, PatientScoped
from app.api.rate_limit import enforce_rate_limit
from app.auth.context import AuthorizationError
from app.observability.logging import get_logger
from app.schemas.clinical import AnalyticsOut, AnalyticsRequest
from app.sql.analytics import AnalyticsUnavailable, run_patient_analytics

log = get_logger(__name__)

router = APIRouter(tags=["analytics"])


@router.post(
    "/analytics",
    response_model=AnalyticsOut,
    # Text-to-SQL is a model call like any other, and reachable without
    # going through /api/chat — limiting only the chat endpoints would leave
    # the cheaper door open.
    dependencies=[Depends(enforce_rate_limit)],
)
async def run_analytics(
    payload: AnalyticsRequest,
    ctx: PatientScoped,
    llm: Llm,
) -> AnalyticsOut:
    """Answer one aggregate question about the caller's own records."""
    try:
        result = await run_patient_analytics(ctx, llm, question=payload.question)
    except AuthorizationError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
        ) from exc
    except AnalyticsUnavailable as exc:
        # 422 for a question the schema cannot express — the request was
        # well-formed and the answer does not exist. 502 when our own
        # generation or execution failed, which is not the caller's fault
        # and is worth retrying.
        code = (
            status.HTTP_502_BAD_GATEWAY
            if exc.answerable
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        raise HTTPException(status_code=code, detail=str(exc)) from exc

    return AnalyticsOut(
        question=result.question,
        sql=result.sql,
        columns=result.columns,
        rows=result.rows,
        count=result.row_count,
        truncated=result.truncated,
        table=result.table,
    )


__all__ = ["router"]
