"""Admit ``KG`` to the route vocabulary.

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-14

PRD §14 lists seven routes; the CHECK constraints written in 0005 allow six.
Until this runs, a turn the router sends to the knowledge graph fails on
INSERT — after the traversal has already happened, so the work is done and
the answer is lost at the last step.

Widening only. Unlike 0005 this rewrites no rows: ``KG`` is a new value, not
a rename, so every existing row is still legal under the new constraint and
the downgrade is safe as long as no row has taken the new value. The
downgrade therefore rewrites ``KG`` rows to ``RAG`` first — retrieval over
the patient's own notes is where a KG question degrades to anyway, so the
route recorded stays the one that would have run.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD = ("API", "RAG", "HYBRID", "TEXT_TO_SQL", "ACTION", "OUT_OF_SCOPE")
NEW = ("API", "RAG", "KG", "HYBRID", "TEXT_TO_SQL", "ACTION", "OUT_OF_SCOPE")

#: (table, constraint name) for every column carrying a route. The names are
#: the ones migration 0006 repaired; passing a different spelling here would
#: drop nothing and add a second, conflicting constraint.
ROUTE_COLUMNS = (
    ("messages", "ck_messages_message_route"),
    ("request_traces", "ck_request_traces_trace_route"),
)


def _rendered(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def _recheck(table: str, constraint: str, values: tuple[str, ...]) -> None:
    op.execute(
        f"ALTER TABLE {table} ADD CONSTRAINT {constraint} "
        f"CHECK (route IS NULL OR route IN ({_rendered(values)}))"
    )


def upgrade() -> None:
    for table, constraint in ROUTE_COLUMNS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")
        _recheck(table, constraint, NEW)


def downgrade() -> None:
    for table, constraint in ROUTE_COLUMNS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")
        # Before narrowing, or the constraint is rejected by the rows it is
        # meant to govern.
        op.execute(f"UPDATE {table} SET route = 'RAG' WHERE route = 'KG'")
        _recheck(table, constraint, OLD)
