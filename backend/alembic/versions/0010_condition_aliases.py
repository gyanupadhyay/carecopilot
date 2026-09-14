"""Store what patients call each condition.

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-14

Found by the knowledge-graph evaluation. The case "which of my visits were
about my blood pressure, and who did I see?" reached the right route and the
right traversal and returned nothing, because the traversal matches a term
against ``display`` and ``key`` — "Essential hypertension" and
"hypertension" — and no patient says either.

The synthetic catalogue already carried aliases and used them nowhere past
generation. Putting them in the system of record is what lets the graph match
them, since PRD §33 forbids the projection holding anything PostgreSQL does
not.

``text[]`` with a ``'{}'`` default, so existing rows are empty rather than
NULL: a NULL array would make every alias comparison NULL, and the traversal
would silently match nothing instead of matching fewer things.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "conditions",
        sa.Column(
            "aliases",
            postgresql.ARRAY(sa.String(length=64)),
            nullable=False,
            server_default="{}",
        ),
    )


def downgrade() -> None:
    op.drop_column("conditions", "aliases")
