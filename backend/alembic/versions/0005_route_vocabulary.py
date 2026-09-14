"""Adopt the router's route vocabulary from §14.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-13

``STRUCTURED`` becomes ``API`` and ``UNKNOWN`` becomes ``OUT_OF_SCOPE``
(PRD §14). Both are stored as plain strings under CHECK constraints, so the
change is: drop the constraint, rewrite the rows, put the constraint back.

``OUT_OF_SCOPE`` is not a rename of convenience. ``UNKNOWN`` described the
system's state — "we did not classify this" — while ``OUT_OF_SCOPE``
describes the question. The router can now say a question is outside what
the assistant covers, which is an answer rather than a failure, and the
stored value says which of the two happened.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD = ("STRUCTURED", "RAG", "HYBRID", "TEXT_TO_SQL", "ACTION", "UNKNOWN")
NEW = ("API", "RAG", "HYBRID", "TEXT_TO_SQL", "ACTION", "OUT_OF_SCOPE")

#: (table, constraint name) for every column carrying a route.
ROUTE_COLUMNS = (
    ("messages", "ck_messages_message_route"),
    ("request_traces", "ck_request_traces_trace_route"),
)


def _rendered(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def _recheck(table: str, constraint: str, values: tuple[str, ...]) -> None:
    op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")
    op.execute(
        f"ALTER TABLE {table} ADD CONSTRAINT {constraint} "
        f"CHECK (route IS NULL OR route IN ({_rendered(values)}))"
    )


def upgrade() -> None:
    for table, constraint in ROUTE_COLUMNS:
        # The constraint has to come off first: the UPDATE below transits
        # through values the old constraint forbids.
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")
        op.execute(f"UPDATE {table} SET route = 'API' WHERE route = 'STRUCTURED'")
        op.execute(f"UPDATE {table} SET route = 'OUT_OF_SCOPE' WHERE route = 'UNKNOWN'")
        _recheck(table, constraint, NEW)


def downgrade() -> None:
    for table, constraint in ROUTE_COLUMNS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")
        op.execute(f"UPDATE {table} SET route = 'STRUCTURED' WHERE route = 'API'")
        op.execute(f"UPDATE {table} SET route = 'UNKNOWN' WHERE route = 'OUT_OF_SCOPE'")
        _recheck(table, constraint, OLD)
