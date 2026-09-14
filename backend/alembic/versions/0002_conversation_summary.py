"""Running summary of older conversation turns.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-13

Adds the two columns behind PRD §32's "summarize older history". Both are
nullable and have no default: a conversation that has never overflowed the
replay budget carries no summary, and that absence is meaningful rather than
a value waiting to be backfilled.

A separate migration rather than an edit to 0001, because 0001 has been
applied to a populated database. Amending an applied migration leaves every
environment that already ran it silently out of step with the file.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("conversations", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column(
        "conversations",
        sa.Column("summary_through_message_id", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("conversations", "summary_through_message_id")
    op.drop_column("conversations", "summary")
