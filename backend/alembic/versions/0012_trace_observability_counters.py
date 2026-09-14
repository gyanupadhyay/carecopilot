"""Record the four §26 fields the trace was missing.

Revision ID: 0012
Revises: 0011
Create Date: 2026-09-14

PRD §26 lists model, model version, agent iterations, authorization failures
and validation failures among the things to track. ``request_traces`` carried
the first and none of the rest.

``model_version`` is the interesting one, because the column that existed was
doing two jobs. A trace recorded whichever id the provider handed back, which
is the *resolved* version — so a deployment configured with an alias had no
row anywhere saying what it was configured with. Splitting them costs one
column and makes both questions answerable: ``model`` reproduces a
deployment, ``model_version`` identifies a measurement.

The three counters are nullable with no default, so a row written before this
migration reads as "not recorded" rather than as a confident zero. A
dashboard that cannot tell those apart reports a clean security record for
every request that predates the column.

No backfill, for the same reason: the values are unknown, and inventing zeros
would make them look measured.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COUNTERS = ("agent_iterations", "authorization_failures", "validation_failures")


def upgrade() -> None:
    op.add_column(
        "request_traces",
        sa.Column("model_version", sa.String(length=120), nullable=True),
    )
    for name in _COUNTERS:
        op.add_column(
            "request_traces", sa.Column(name, sa.Integer(), nullable=True)
        )


def downgrade() -> None:
    for name in _COUNTERS:
        op.drop_column("request_traces", name)
    op.drop_column("request_traces", "model_version")
