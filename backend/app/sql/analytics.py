"""The analytics capability, as one audited call (PRD §16).

§16 describes a pipeline and a list of things the tool must enforce:

    run_my_patient_analytics()
      → Qwen3 generates SQL → parser → validator
      → authorization enforcement → read-only PostgreSQL → result

    patient scope · allowed schema · read-only access · SQL validation
    · timeouts · row limits · audit logging

Every one of those existed already, spread across
:mod:`app.sql.generator`, :mod:`app.sql.validator` and
:mod:`app.sql.executor` — except the last. Analytics ran unaudited, which is
the one item on that list that is not enforced by some other layer refusing:
a query that was scoped, validated and capped still leaves no record that it
happened.

So this module is the seam §16 names. It composes the pipeline once, writes
an audit row for every outcome including the refusals, and is what both the
HTTP endpoint and the agent's node call — so neither can acquire analytics
without the audit trail, and there is one definition of "running analytics"
rather than two that drift.

Refusals are audited too, and deliberately. "The schema does not cover that"
and "generation failed twice" are the interesting rows in an analytics audit:
a burst of them is what an attempt to probe the schema looks like.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.auth.context import AuthContext, AuthorizationError
from app.db.session import AppSession
from app.llm.base import LLMProvider
from app.models import AuditLog
from app.observability.logging import get_logger
from app.sql.executor import SQLExecutionError, SQLResult, execute_sql
from app.sql.generator import SQLGenerationError, generate_sql

log = get_logger(__name__)

#: Recorded as the action on every analytics audit row.
ANALYTICS_ACTION = "run_patient_analytics"


class AnalyticsUnavailable(Exception):
    """The question could not be answered as a query.

    ``answerable`` distinguishes the two cases the caller must report
    differently: False means the schema does not cover the question, which is
    a fact about the data and is stated as one; True means our side failed,
    which gets a refusal rather than a guess. An aggregate is exactly the
    kind of answer where a plausible wrong number is indistinguishable from a
    right one.
    """

    def __init__(self, message: str, *, answerable: bool = True) -> None:
        super().__init__(message)
        self.answerable = answerable


@dataclass(frozen=True, slots=True)
class AnalyticsResult:
    """A completed analytics run, with the statement that produced it."""

    question: str
    sql: str
    columns: list[str]
    rows: list[list[object]]
    row_count: int
    truncated: bool
    latency_ms: int
    #: Rendered for a model to read — a header and pipe-separated rows.
    table: str


async def run_patient_analytics(
    ctx: AuthContext,
    llm: LLMProvider,
    *,
    question: str,
) -> AnalyticsResult:
    """Generate, validate, execute and audit one analytics query.

    The patient scope is never a parameter: :func:`app.sql.executor.execute_sql`
    takes it from ``ctx`` and the read-only role's row-level security enforces
    it again inside PostgreSQL. Generated SQL must not filter on
    ``patient_id`` at all — RLS already scopes the connection, so such a
    predicate can only wrongly exclude the patient's own rows.
    """
    asked = (question or "").strip()
    if not asked:
        raise AnalyticsUnavailable("No question was supplied.", answerable=False)

    # Read here, and read again by the executor. `patient_scope` raises when
    # the session is not linked to a patient record, and that has to happen
    # before the model call rather than after it — an unscoped session should
    # not be able to spend a generation to find out it was never allowed.
    patient_id = ctx.patient_scope

    try:
        validated = await generate_sql(asked, llm)
    except SQLGenerationError as exc:
        await _audit(
            ctx,
            outcome="rejected",
            detail=str(exc)[:500],
            params={"stage": "generation", "answerable": exc.answerable},
        )
        raise AnalyticsUnavailable(str(exc), answerable=exc.answerable) from exc

    try:
        result = await execute_sql(ctx, validated)
    except AuthorizationError:
        # Re-raised unchanged: the API layer turns this into a 403, and
        # flattening it into AnalyticsUnavailable would report a refusal as
        # an unanswerable question.
        await _audit(
            ctx,
            outcome="rejected",
            detail="authorization",
            params={"stage": "execution", "sql": validated.sql},
        )
        raise
    except SQLExecutionError as exc:
        await _audit(
            ctx,
            outcome="failed",
            detail=str(exc)[:500],
            params={"stage": "execution", "sql": validated.sql},
        )
        raise AnalyticsUnavailable(str(exc)) from exc

    await _audit(
        ctx,
        outcome="executed",
        params={
            "stage": "execution",
            # The statement, not its rows. The SQL is mechanism and belongs
            # in an audit trail; the rows are clinical data and would make
            # the audit log a second, less protected copy of the record.
            "sql": result.sql,
            "row_count": result.row_count,
            "truncated": result.truncated,
        },
    )

    log.info(
        "analytics.ran",
        patient_id=patient_id,
        rows=result.row_count,
        ms=result.latency_ms,
        truncated=result.truncated,
        # Length, not text: the question is the patient's own words.
        question_len=len(asked),
    )
    return _as_result(asked, result)


def _as_result(question: str, result: SQLResult) -> AnalyticsResult:
    return AnalyticsResult(
        question=question,
        sql=result.sql,
        columns=list(result.columns),
        # Tuples are not JSON, and every caller here serialises.
        rows=[list(row) for row in result.rows],
        row_count=result.row_count,
        truncated=result.truncated,
        latency_ms=result.latency_ms,
        table=result.as_table(),
    )


async def _audit(
    ctx: AuthContext,
    *,
    outcome: str,
    params: dict[str, object],
    detail: str | None = None,
) -> None:
    """Record one analytics attempt, in its own transaction. Never raises.

    **Its own session, deliberately.** The request's session is rolled back
    when the handler raises, and the handler raises on exactly the outcomes
    worth auditing — a question the schema refused, a statement the validator
    rejected. Writing the audit row through the request's session meant those
    rows vanished with the request that produced them, leaving an audit trail
    that records only the successes. A refusal that leaves no trace is the
    one an attacker can repeat.

    Never raises, either: an audit write that failed the request it was
    recording would turn the audit trail into a source of outages. A missing
    row is logged loudly instead.
    """
    try:
        async with AppSession() as audit_session:
            audit_session.add(
                AuditLog(
                    request_id=ctx.request_id,
                    user_id=ctx.user_id,
                    patient_id=ctx.patient_id,
                    action=ANALYTICS_ACTION,
                    target_type="analytics",
                    outcome=outcome,
                    params=params,
                    detail=detail,
                )
            )
            await audit_session.commit()
    except Exception:
        log.error("analytics.audit_failed", outcome=outcome)
