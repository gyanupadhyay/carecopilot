"""Model user-to-patient authorization as a table.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-13

Replaces ``users.patient_id`` with ``user_patient_mapping`` (PRD §9, §10) so
that a user's access to a patient is a recorded, revocable fact rather than
a nullable column.

The upgrade copies every existing link before dropping the column, so no
account loses its scope. The downgrade can only restore one patient per
user — it takes the oldest active link — which is lossless today because
nothing issues more than one, and is the reason that limitation is
acceptable rather than hidden.

Renames ``action_audit`` to ``audit_logs`` in the same revision: both are
schema-shape corrections against the same PRD section, and splitting them
would leave a commit where the model layer and the database disagree.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "user_patient_mapping",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column(
            "relationship_type", sa.String(32), nullable=False, server_default="self"
        ),
        sa.Column(
            "is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_user_patient_mapping"),
        sa.UniqueConstraint("user_id", "patient_id", name="uq_user_patient"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_user_patient_mapping_user_id_users",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_user_patient_mapping_patient_id_patients",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_user_patient_mapping_created_at", "user_patient_mapping", ["created_at"]
    )
    op.create_index(
        "ix_user_patient_mapping_user_id_is_active",
        "user_patient_mapping",
        ["user_id", "is_active"],
    )

    # Carry every existing grant across before the column disappears.
    op.execute(
        """
        INSERT INTO user_patient_mapping
            (user_id, patient_id, relationship_type, is_active)
        SELECT id, patient_id, 'self', true
          FROM users
         WHERE patient_id IS NOT NULL
        """
    )

    op.drop_constraint("fk_users_patient_id_patients", "users", type_="foreignkey")
    op.drop_column("users", "patient_id")

    op.rename_table("action_audit", "audit_logs")


def downgrade() -> None:
    op.rename_table("audit_logs", "action_audit")

    op.add_column("users", sa.Column("patient_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_users_patient_id_patients",
        "users",
        "patients",
        ["patient_id"],
        ["id"],
        ondelete="CASCADE",
    )
    # One patient per user is all the old shape can hold; take the oldest
    # active grant so the choice is deterministic rather than arbitrary.
    op.execute(
        """
        UPDATE users u
           SET patient_id = m.patient_id
          FROM (
                SELECT DISTINCT ON (user_id) user_id, patient_id
                  FROM user_patient_mapping
                 WHERE is_active
                 ORDER BY user_id, id
               ) m
         WHERE m.user_id = u.id
        """
    )

    op.drop_index(
        "ix_user_patient_mapping_user_id_is_active", table_name="user_patient_mapping"
    )
    op.drop_index(
        "ix_user_patient_mapping_created_at", table_name="user_patient_mapping"
    )
    op.drop_table("user_patient_mapping")
