"""Turn a patient's question into one validated SELECT (PRD §16).

Generation and validation are one loop, not two steps. The model gets at
most ``MAX_ATTEMPTS`` tries, and a rejected attempt is fed back with the
validator's reasons attached — which fixes the common failures (a
hallucinated column, a forgotten alias, a stray ``patient_id`` filter)
without a human in the loop.

The loop is bounded at two for a reason that is about honesty rather than
cost: a model that cannot produce valid SQL for a question in two tries is
usually being asked something the schema cannot answer, and the right
response is to say so. Retrying until something parses produces a query that
runs and answers a different question.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, Field

from app.config import settings
from app.llm.base import ChatMessage, Effort, LLMProvider
from app.llm.errors import LLMError
from app.observability.logging import get_logger
from app.sql.schema import GENERATION_RULES, SCHEMA_PROMPT
from app.sql.validator import SQLValidationError, ValidatedSQL, validate_sql

log = get_logger(__name__)

#: Two. See the module docstring — this is a correctness bound, not a budget.
MAX_ATTEMPTS: Final = 2

SQL_EFFORT: Effort = "low"

SYSTEM_PROMPT: Final = f"""
You write one PostgreSQL SELECT that answers a patient's question about
their own medical record. You do not answer the question in prose and you
do not explain the query.

SCHEMA
{SCHEMA_PROMPT}

{GENERATION_RULES}

If the question cannot be answered from these four tables, set
`answerable` to false and leave `sql` empty. That is a correct outcome, not
a failure — inventing a query against columns that do not exist produces a
confident wrong answer, which is worse than saying the data is not there.
""".strip()


class GeneratedSQL(BaseModel):
    """What the model returns: a query, or an honest refusal to write one."""

    answerable: bool = Field(
        description=(
            "True if the question can be answered from the four tables. "
            "False if it needs data the schema does not contain."
        )
    )
    sql: str = Field(
        default="",
        description="The single SELECT statement. Empty when answerable is false.",
    )
    reason: str = Field(
        default="",
        description=(
            "One short sentence: what the query computes, or why the "
            "question cannot be answered from this schema."
        ),
    )


class SQLGenerationError(Exception):
    """No valid SQL was produced. ``answerable`` distinguishes why.

    A model that declined to write SQL (``answerable=False``) and a model
    that wrote invalid SQL twice are different outcomes: the first is the
    schema not covering the question, the second is a generation failure.
    The node phrases the answer differently for each.
    """

    def __init__(
        self, message: str, *, answerable: bool = True, attempts: int = 0
    ) -> None:
        super().__init__(message)
        self.answerable = answerable
        self.attempts = attempts


async def generate_sql(question: str, llm: LLMProvider) -> ValidatedSQL:
    """Generate and validate SQL for ``question``.

    Raises :class:`SQLGenerationError` when no valid query was produced.
    """
    messages: list[ChatMessage] = [ChatMessage(role="user", content=question)]
    last_reasons: list[str] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = await llm.generate_structured(
                messages=messages,
                system=SYSTEM_PROMPT,
                schema=GeneratedSQL,
                max_tokens=settings.sql_generation_max_tokens,
                effort=SQL_EFFORT,
                model=settings.router_model,
            )
        except LLMError as exc:
            log.warning("sql.generation_failed", error=type(exc).__name__)
            raise SQLGenerationError(
                f"The model could not be reached: {type(exc).__name__}.",
                attempts=attempt,
            ) from exc

        generated = result.value

        if not generated.answerable:
            log.info("sql.declined", reason=generated.reason[:120])
            raise SQLGenerationError(
                generated.reason or "The schema does not contain this data.",
                answerable=False,
                attempts=attempt,
            )

        try:
            validated = validate_sql(generated.sql)
        except SQLValidationError as exc:
            last_reasons = exc.reasons
            log.info(
                "sql.rejected",
                attempt=attempt,
                reasons=exc.reasons,
                sql=generated.sql[:300],
            )
            if attempt == MAX_ATTEMPTS:
                break
            # The rejected query and every reason go back in, so the retry
            # is a correction rather than a second guess at the same task.
            messages = [
                ChatMessage(role="user", content=question),
                ChatMessage(role="assistant", content=generated.sql),
                ChatMessage(
                    role="user",
                    content=(
                        "That query was rejected:\n"
                        + "\n".join(f"- {reason}" for reason in exc.reasons)
                        + "\n\nWrite it again, fixing every point above. If "
                        "the question cannot be answered within the rules, "
                        "set answerable to false instead."
                    ),
                ),
            ]
            continue

        log.info(
            "sql.generated",
            attempt=attempt,
            tables=sorted(validated.tables),
            limit_applied=validated.limit_applied,
        )
        return validated

    raise SQLGenerationError(
        "Generated SQL failed validation: " + "; ".join(last_reasons),
        attempts=MAX_ATTEMPTS,
    )
