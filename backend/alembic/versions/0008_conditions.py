"""Make the condition behind a visit a fact rather than an inference.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-14

Groundwork for the knowledge graph (PRD §17, §33). The graph puts two
relationships through ``Condition`` — ``Patient -HAS_CONDITION->`` and
``Encounter -FOR_CONDITION->`` — and §33 requires it be rebuildable from
PostgreSQL, with no business truth held only in Neo4j. Neither is possible
while the condition a visit was about exists solely as free-text prose in
``encounters.reason`` ("Reports occasional fatigue in the afternoons").

``encounters.condition_id`` is nullable and ``ON DELETE SET NULL``: not every
visit is about a catalogued problem, and retiring a condition from the
catalogue must not delete the visits that referenced it. ``patient_conditions``
cascades instead — an edge to a deleted patient or condition is not a fact
about anything.

No backfill. Existing rows keep ``condition_id IS NULL`` until the data is
regenerated, because the mapping this adds is exactly the one that cannot be
recovered from the prose. ``scripts/generate_data.py --reset`` produces a
corpus with it populated.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _created_at() -> sa.Column:
    """Matches ``TimestampMixin``: created_at only, timezone-aware, indexed.

    The index is not decoration — the mixin declares ``index=True``, and a
    migration that omits it leaves the ORM and the database disagreeing in a
    way only ``--autogenerate`` notices, months later.
    """
    return sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        server_default=sa.text("now()"),
        nullable=False,
    )


def upgrade() -> None:
    op.create_table(
        "conditions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("display", sa.String(length=120), nullable=False),
        _created_at(),
        sa.UniqueConstraint("key", name="uq_conditions_key"),
    )
    op.create_index("ix_conditions_created_at", "conditions", ["created_at"])

    op.create_table(
        "patient_conditions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "patient_id",
            sa.Integer(),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "condition_id",
            sa.Integer(),
            sa.ForeignKey("conditions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("onset_date", sa.Date(), nullable=True),
        _created_at(),
        sa.UniqueConstraint("patient_id", "condition_id", name="uq_patient_condition"),
    )
    op.create_index(
        "ix_patient_conditions_created_at", "patient_conditions", ["created_at"]
    )

    op.add_column(
        "encounters",
        sa.Column("condition_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_encounters_condition_id_conditions",
        "encounters",
        "conditions",
        ["condition_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # And on medications, which is not redundant with the encounter's. One
    # visit routinely starts therapy for several conditions, so joining a
    # drug to its condition *through* the encounter pairs it with whichever
    # problem the visit was filed under. Measured on the demo patient before
    # this column existed: "Essential hypertension → Metformin".
    op.add_column(
        "medications",
        sa.Column("condition_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_medications_condition_id_conditions",
        "medications",
        "conditions",
        ["condition_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_medications_condition_id_conditions", "medications", type_="foreignkey"
    )
    op.drop_column("medications", "condition_id")
    op.drop_constraint(
        "fk_encounters_condition_id_conditions", "encounters", type_="foreignkey"
    )
    op.drop_column("encounters", "condition_id")
    op.drop_table("patient_conditions")
    op.drop_table("conditions")
