"""Validate generated SQL before it reaches the database (PRD §16).

**This is not the security boundary.** The boundary is the analytics role:
SELECT-only grants on four tables, ``default_transaction_read_only``, and
row-level security keyed to ``app.patient_id``. Those hold even if every
line below is wrong, which is the property that makes them the boundary.

This module exists for two other reasons. It turns a bad query into a clear
message instead of a database error the user cannot act on, and it refuses
categories of query that the role would permit but the product should not —
a full table dump is legal SQL for an authorized patient and still a bad
answer to "how many times was my blood pressure high".

It parses rather than pattern-matches. Regex validation of SQL reads as
security and is not: comments, string literals, unicode escapes and nested
subqueries all defeat it, and the resulting false confidence is worse than
having no validator. ``sqlglot`` builds a real AST and every check below
walks it.

The rejection message is written for the *developer* reading a trace. What
the patient sees is a generic apology chosen by the node — an error that
quotes the offending SQL back to a user is a small information leak and a
large confusion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

import sqlglot
from sqlglot import exp

from app.config import settings
from app.sql.schema import ALLOWED_COLUMNS, ALLOWED_TABLES, ENUMERATED_COLUMNS

#: Only these node types may appear at the top of a statement. Anything else
#: — INSERT, UPDATE, DELETE, DDL, a transaction command — is rejected on its
#: type alone, before any other check runs.
_ALLOWED_ROOTS: Final = (exp.Select, exp.Union, exp.Except, exp.Intersect)

#: Functions that read server state, touch the filesystem, or reach out of
#: the database. None has a legitimate use in an analytics query.
_FORBIDDEN_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {
        "current_setting",
        "set_config",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_sleep",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "dblink",
        "dblink_connect",
        "lo_import",
        "lo_export",
        "query_to_xml",
        "pg_stat_file",
        "txid_current",
        "version",
    }
)

#: Schemas whose contents describe the database rather than the patient.
_FORBIDDEN_SCHEMAS: Final[frozenset[str]] = frozenset(
    {"pg_catalog", "information_schema", "pg_toast", "pg_temp"}
)

_PG_PREFIXED = re.compile(r"^pg_", re.IGNORECASE)


class SQLValidationError(Exception):
    """The generated SQL was rejected. Carries every reason, not the first.

    All of them, because a model given one correction at a time will often
    fix it and introduce the next — and each round trip costs a call.
    """

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@dataclass(slots=True)
class ValidatedSQL:
    """A query that passed every check, in the form that should be run."""

    sql: str
    tables: set[str] = field(default_factory=set)
    #: True when this module added the LIMIT rather than the model writing it.
    limit_applied: bool = False


def validate_sql(raw: str, *, max_rows: int | None = None) -> ValidatedSQL:
    """Parse and check one generated query.

    Raises :class:`SQLValidationError` listing every problem found.
    """
    max_rows = max_rows if max_rows is not None else settings.sql_max_rows

    text = (raw or "").strip().rstrip(";").strip()
    if not text:
        raise SQLValidationError(["The model returned no SQL."])

    # Parsed before anything else: statement counting has to be done by the
    # parser, since a semicolon inside a string literal is not a separator.
    try:
        statements = sqlglot.parse(text, read="postgres")
    except Exception as exc:
        raise SQLValidationError([f"Could not parse as PostgreSQL: {exc}"]) from exc

    statements = [s for s in statements if s is not None]
    if not statements:
        raise SQLValidationError(["The model returned no SQL."])
    if len(statements) > 1:
        raise SQLValidationError(
            [f"Expected exactly one statement, found {len(statements)}."]
        )

    statement = statements[0]
    reasons: list[str] = []

    if not isinstance(statement, _ALLOWED_ROOTS):
        # Named rather than described: "found Insert" tells a developer
        # reading a trace what the model actually did.
        raise SQLValidationError(
            [f"Only SELECT is permitted, found {type(statement).__name__.upper()}."]
        )

    tables = _check_tables(statement, reasons)
    _check_columns(statement, tables, reasons)
    _check_functions(statement, reasons)
    _check_joins(statement, reasons)
    _check_patient_filter(statement, reasons)
    _check_enumerated_literals(statement, reasons)

    if reasons:
        raise SQLValidationError(reasons)

    statement, limit_applied = _apply_limit(statement, max_rows)
    return ValidatedSQL(
        sql=statement.sql(dialect="postgres"),
        tables=tables,
        limit_applied=limit_applied,
    )


# --- individual checks --------------------------------------------------- #


def _check_tables(statement: exp.Expression, reasons: list[str]) -> set[str]:
    """Every table named must be on the allowlist.

    CTE names are collected first and exempted: ``WITH recent AS (...)
    SELECT * FROM recent`` references ``recent``, which is a label for a
    subquery that was itself checked, not a table.
    """
    cte_names = {
        cte.alias_or_name.lower()
        for cte in statement.find_all(exp.CTE)
        if cte.alias_or_name
    }

    seen: set[str] = set()
    for table in statement.find_all(exp.Table):
        name = (table.name or "").lower()
        schema = (table.db or "").lower()

        if schema and schema in _FORBIDDEN_SCHEMAS:
            reasons.append(f"Schema {schema!r} is not readable.")
            continue
        if schema and schema != "public":
            reasons.append(f"Schema {schema!r} is not permitted.")
            continue
        if name in cte_names:
            continue
        if _PG_PREFIXED.match(name) or name in _FORBIDDEN_SCHEMAS:
            reasons.append(f"System table {name!r} is not readable.")
            continue
        if name not in ALLOWED_TABLES:
            reasons.append(
                f"Table {name!r} is not in the analytics allowlist "
                f"({', '.join(sorted(ALLOWED_TABLES))})."
            )
            continue
        seen.add(name)

    if not seen and not cte_names:
        reasons.append("The query reads no allowlisted table.")
    return seen


def _check_columns(
    statement: exp.Expression, tables: set[str], reasons: list[str]
) -> None:
    """Reject columns that exist in no allowlisted table.

    Deliberately loose about *which* table a column belongs to. Resolving
    that correctly needs alias tracking through subqueries and CTEs, and a
    validator that is subtly wrong about scope would reject working queries
    — a worse failure here than letting a hallucinated column reach a
    database that will reject it anyway with a precise message.
    """
    if not tables:
        return

    known: set[str] = set()
    for table in tables:
        known |= ALLOWED_COLUMNS.get(table, frozenset())
    # Aliases the query itself introduces are legitimate references.
    known |= {
        alias.alias_or_name.lower()
        for alias in statement.find_all(exp.Alias)
        if alias.alias_or_name
    }
    known |= {
        cte.alias_or_name.lower()
        for cte in statement.find_all(exp.CTE)
        if cte.alias_or_name
    }
    known |= {
        table_alias.alias_or_name.lower()
        for table_alias in statement.find_all(exp.TableAlias)
        if table_alias.alias_or_name
    }

    unknown: set[str] = set()
    for column in statement.find_all(exp.Column):
        name = (column.name or "").lower()
        if not name or name == "*":
            continue
        if name not in known:
            unknown.add(name)

    if unknown:
        reasons.append(
            f"Unknown column(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(known & _all_schema_columns()))}."
        )


def _all_schema_columns() -> set[str]:
    columns: set[str] = set()
    for names in ALLOWED_COLUMNS.values():
        columns |= names
    return columns


def _check_functions(statement: exp.Expression, reasons: list[str]) -> None:
    for node in statement.find_all(exp.Anonymous, exp.Func):
        name = (
            node.name
            if isinstance(node, exp.Anonymous)
            else node.sql_name() if hasattr(node, "sql_name") else ""
        )
        lowered = (name or "").lower()
        if not lowered:
            continue
        if lowered in _FORBIDDEN_FUNCTIONS or _PG_PREFIXED.match(lowered):
            reasons.append(f"Function {lowered!r} is not permitted.")


def _check_joins(statement: exp.Expression, reasons: list[str]) -> None:
    joins = len(list(statement.find_all(exp.Join)))
    if joins > settings.sql_max_joins:
        reasons.append(
            f"{joins} joins exceeds the limit of {settings.sql_max_joins}."
        )


def _check_patient_filter(statement: exp.Expression, reasons: list[str]) -> None:
    """Reject a query that filters on ``patient_id``.

    Not a security control — RLS already confines the connection to one
    patient, so this predicate cannot widen access. It is rejected because
    it can only *narrow*: a model that writes ``WHERE patient_id = 2`` on a
    session scoped to patient 1 gets an empty result and reports "you have
    no records", which is a confident wrong answer. Better to reject the
    query and say why.
    """
    for column in statement.find_all(exp.Column):
        if (column.name or "").lower() != "patient_id":
            continue
        if column.find_ancestor(exp.Where, exp.Join, exp.Having):
            reasons.append(
                "Do not filter on patient_id: the connection is already "
                "restricted to one patient, so the predicate can only "
                "exclude that patient's own rows."
            )
            return


def _check_enumerated_literals(
    statement: exp.Expression, reasons: list[str]
) -> None:
    """Reject an equality test against a value the column never holds.

    This catches the worst failure mode in the whole Text-to-SQL path, and
    the only one that produces no error anywhere. ``WHERE test_name =
    'Systolic BP'`` is valid SQL against a real column; it simply matches
    nothing, and "0" is then reported to the patient as a fact about their
    health. A count that is wrong because the filter was misspelled is
    indistinguishable, downstream, from a count that is right.

    Only ``=``, ``!=`` and ``IN`` are checked. ``ILIKE '%systolic%'`` is a
    deliberate loose match and stays legal — the generation rules point the
    model at it for exactly this reason.
    """
    for node in statement.find_all(exp.EQ, exp.NEQ, exp.In):
        column = node.this
        if not isinstance(column, exp.Column):
            continue
        vocabulary = ENUMERATED_COLUMNS.get((column.name or "").lower())
        if vocabulary is None:
            continue

        candidates = (
            node.expressions if isinstance(node, exp.In) else [node.expression]
        )

        for candidate in candidates:
            if not isinstance(candidate, exp.Literal) or not candidate.is_string:
                continue
            if candidate.this in vocabulary:
                continue
            reasons.append(
                f"{column.name} has no value {candidate.this!r}. "
                f"It is exactly one of: {', '.join(vocabulary)}."
            )


def _apply_limit(
    statement: exp.Expression, max_rows: int
) -> tuple[exp.Expression, bool]:
    """Cap the result size, tightening a limit the model set too high.

    A row cap is not a security control either — it bounds the answer, the
    payload and the tokens spent summarizing it.
    """
    existing = statement.args.get("limit")
    if existing is not None:
        try:
            current = int(existing.expression.name)
        except (AttributeError, TypeError, ValueError):
            # An expression rather than a literal. Replace it: an unbounded
            # or computed limit defeats the cap.
            return statement.limit(max_rows), True
        if current <= max_rows:
            return statement, False
        return statement.limit(max_rows), True

    if isinstance(statement, exp.Select) and _is_scalar_aggregate(statement):
        # A bare aggregate returns exactly one row. A LIMIT would be noise
        # in the trace and in the SQL shown on the developer panel.
        return statement, False

    return statement.limit(max_rows), True


def _is_scalar_aggregate(statement: exp.Select) -> bool:
    """True for a SELECT that returns one row by construction."""
    if statement.args.get("group"):
        return False
    projections = statement.selects
    if not projections:
        return False
    return all(
        isinstance(projection.unalias(), exp.AggFunc)
        if isinstance(projection, exp.Alias)
        else isinstance(projection, exp.AggFunc)
        for projection in projections
    )
