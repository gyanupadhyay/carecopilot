"""Close the gap between §17's entity list and §32's schema.

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-14

PRD §17 names Procedure, Allergy and Diagnosis among the knowledge graph's
entities; §32's PostgreSQL schema had no table for any of them. The graph
could not project them without inventing them, which §33 forbids, so they
were left out — and left out, the graph could not answer "am I allergic to
anything?" or "when was I diagnosed?" at all.

``diagnoses`` is not a duplicate of ``patient_conditions``. That table says a
problem is on the patient's list; this one says a clinician recorded it at a
particular visit, on a date, with a code. A problem list entry has no author;
a diagnosis does.

Codes are denormalised onto ``diagnoses`` rather than read through the
condition. A catalogue correction must not silently rewrite what was recorded
years ago, which is why coded records keep their own copy in the first place.

No backfill: nothing in the existing corpus records any of this, and deriving
it would be fabrication. ``scripts/generate_data.py --reset`` produces a
dataset with all three populated.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SEVERITIES = ("mild", "moderate", "severe")


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        server_default=sa.text("now()"),
        nullable=False,
    )


def upgrade() -> None:
    op.create_table(
        "procedures",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "patient_id",
            sa.Integer(),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "encounter_id",
            sa.Integer(),
            sa.ForeignKey("encounters.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "provider_id",
            sa.Integer(),
            sa.ForeignKey("providers.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "condition_id",
            sa.Integer(),
            sa.ForeignKey("conditions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("code", sa.String(length=16), nullable=True),
        sa.Column("performed_date", sa.Date(), nullable=False),
        _created_at(),
    )
    op.create_index("ix_procedures_created_at", "procedures", ["created_at"])
    op.create_index(
        "ix_procedures_patient_id_performed_date",
        "procedures",
        ["patient_id", "performed_date"],
    )

    op.create_table(
        "allergies",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "patient_id",
            sa.Integer(),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("substance", sa.String(length=120), nullable=False),
        sa.Column("reaction", sa.String(length=200), nullable=True),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("recorded_date", sa.Date(), nullable=True),
        _created_at(),
        sa.UniqueConstraint("patient_id", "substance", name="uq_patient_allergy"),
        sa.CheckConstraint(
            "severity IN (" + ", ".join(f"'{s}'" for s in SEVERITIES) + ")",
            # The BARE name. `Base.metadata`'s convention is
            # ck_%(table_name)s_%(constraint_name)s and op.create_table
            # applies it, so passing the prefixed spelling here produces
            # `ck_allergies_ck_allergies_allergy_severity` — the exact fault
            # migration 0006 was written to repair, and which
            # test_schema_integrity caught again here.
            name="allergy_severity",
        ),
    )
    op.create_index("ix_allergies_created_at", "allergies", ["created_at"])

    op.create_table(
        "diagnoses",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "patient_id",
            sa.Integer(),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "encounter_id",
            sa.Integer(),
            sa.ForeignKey("encounters.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "condition_id",
            sa.Integer(),
            sa.ForeignKey("conditions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("code", sa.String(length=16), nullable=True),
        sa.Column("diagnosed_date", sa.Date(), nullable=False),
        sa.Column(
            "rank", sa.String(length=16), nullable=False, server_default="primary"
        ),
        _created_at(),
        sa.UniqueConstraint(
            "encounter_id", "condition_id", name="uq_encounter_diagnosis"
        ),
    )
    op.create_index("ix_diagnoses_created_at", "diagnoses", ["created_at"])
    op.create_index(
        "ix_diagnoses_patient_id_diagnosed_date",
        "diagnoses",
        ["patient_id", "diagnosed_date"],
    )


def downgrade() -> None:
    op.drop_table("diagnoses")
    op.drop_table("allergies")
    op.drop_table("procedures")
