"""Run validated SQL on the least-privilege connection (PRD §16).

Everything that makes this safe lives outside this module: the analytics
role's SELECT-only grants, ``default_transaction_read_only``, the short
``statement_timeout``, and the row-level security policies that
``analytics_session`` activates with ``SET LOCAL app.patient_id``. This
module's only job is to run the statement on *that* session and never on
another one — which is why it takes an ``AuthContext`` and opens the session
itself rather than accepting a session from its caller. A caller that could
pass in a session could pass in the application's read/write one.

Results come back as plain Python values with column names attached, ready
to be rendered as a small table for the model. Nothing here interprets the
numbers; the node hands them to generation as established facts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.auth.context import AuthContext
from app.config import settings
from app.db.session import analytics_session
from app.observability.logging import get_logger
from app.sql.validator import ValidatedSQL

log = get_logger(__name__)


class SQLExecutionError(Exception):
    """The database refused or failed to run the query."""


@dataclass(slots=True)
class SQLResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    sql: str
    latency_ms: int = 0
    truncated: bool = False
    #: Set when the validator tightened or added the row cap.
    limit_applied: bool = field(default=False)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def is_empty(self) -> bool:
        return not self.rows

    @property
    def scalar(self) -> Any | None:
        """The single value, when the result is one row of one column."""
        if len(self.rows) == 1 and len(self.rows[0]) == 1:
            return self.rows[0][0]
        return None

    def as_table(self, *, max_rows: int = 25) -> str:
        """Render for the model: a header, then rows, pipe-separated.

        Capped independently of the SQL LIMIT. The database cap bounds what
        is fetched; this one bounds what is put in the prompt, and 25 rows
        of numbers is already more than any answer to these questions needs.
        """
        if self.is_empty:
            return "(no rows)"
        lines = [" | ".join(self.columns)]
        for row in self.rows[:max_rows]:
            lines.append(" | ".join(_render(value) for value in row))
        if self.row_count > max_rows:
            lines.append(f"... {self.row_count - max_rows} more row(s)")
        return "\n".join(lines)


def _render(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, Decimal):
        # Trailing zeros from NUMERIC(10,3) make a table hard to scan, and
        # "7.100" reads as more precision than the measurement carries.
        return format(value.normalize(), "f")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


async def execute_sql(ctx: AuthContext, validated: ValidatedSQL) -> SQLResult:
    """Run ``validated`` scoped to ``ctx``'s patient.

    The patient scope comes from the context, never from the SQL and never
    from a caller-supplied id — ``AuthContext.patient_scope`` is what
    refuses a context that has no patient scope at all.
    """
    patient_id = ctx.patient_scope

    started = time.perf_counter()
    try:
        async with analytics_session(patient_id) as session:
            result = await session.execute(text(validated.sql))
            columns = list(result.keys())
            rows = [tuple(row) for row in result.fetchall()]
    except SQLAlchemyError as exc:
        # The driver's message can quote the statement, which may echo the
        # question back. Logged for the trace, not returned to the caller.
        log.warning(
            "sql.execution_failed",
            error=type(exc).__name__,
            detail=str(exc)[:300],
        )
        raise SQLExecutionError(
            f"The query could not be executed ({type(exc).__name__})."
        ) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    truncated = len(rows) >= settings.sql_max_rows

    log.info(
        "sql.executed",
        rows=len(rows),
        ms=latency_ms,
        truncated=truncated,
        patient_id=patient_id,
    )
    return SQLResult(
        columns=columns,
        rows=rows,
        sql=validated.sql,
        latency_ms=latency_ms,
        truncated=truncated,
        limit_applied=validated.limit_applied,
    )
