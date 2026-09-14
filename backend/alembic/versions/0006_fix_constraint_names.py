"""Repair double-prefixed CHECK constraint names.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-13

Migration 0001 passed already-prefixed names to ``op.create_table`` — for
example ``name="ck_messages_message_route"`` — while ``Base.metadata``
carries the convention ``ck_%(table_name)s_%(constraint_name)s``. Alembic
applied the convention on top, so the database ended up with
``ck_messages_ck_messages_message_route`` while the models believe the name
is ``ck_messages_message_route``.

That was invisible until 0005 tried to replace the route constraint by
name: the drop matched nothing, the add created a *second* constraint, and
both were enforced — so writing ``route = 'API'`` satisfied the new rule and
violated the stale one. The tests missed it because every test wrote
``RAG``, which is legal under both vocabularies. A live request routed to
``API`` and failed.

This renames each constraint to what the models expect and removes the
duplicates 0005 introduced, so metadata and database agree again and a
future autogenerate does not propose spurious drops.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: (table, stale name, intended name). The stale names are what 0001 left
#: behind; the intended names are what the model metadata renders.
RENAMES: tuple[tuple[str, str, str], ...] = (
    (
        "appointments",
        "ck_appointments_ck_appointments_appointment_status",
        "ck_appointments_appointment_status",
    ),
    (
        "appointments",
        "ck_appointments_ck_appointments_appointment_type",
        "ck_appointments_appointment_type",
    ),
    (
        "audit_logs",
        "ck_action_audit_ck_action_audit_action_outcome",
        "ck_audit_logs_action_outcome",
    ),
    (
        "clinical_documents",
        "ck_clinical_documents_ck_clinical_documents_document_status",
        "ck_clinical_documents_document_status",
    ),
    (
        "clinical_documents",
        "ck_clinical_documents_ck_clinical_documents_document_type",
        "ck_clinical_documents_document_type",
    ),
    (
        "encounters",
        "ck_encounters_ck_encounters_encounter_type",
        "ck_encounters_encounter_type",
    ),
    (
        "medications",
        "ck_medications_ck_medications_medication_date_order",
        "ck_medications_medication_date_order",
    ),
    (
        "medications",
        "ck_medications_ck_medications_medication_status",
        "ck_medications_medication_status",
    ),
    (
        "messages",
        "ck_messages_ck_messages_message_role",
        "ck_messages_message_role",
    ),
    (
        "users",
        "ck_users_ck_users_user_role",
        "ck_users_user_role",
    ),
)

#: The two route constraints 0005 could not replace. The stale copy still
#: enforces the old vocabulary and has to go; 0005's correct copy stays.
STALE_ROUTE_CONSTRAINTS: tuple[tuple[str, str], ...] = (
    ("messages", "ck_messages_ck_messages_message_route"),
    ("request_traces", "ck_request_traces_ck_request_traces_trace_route"),
)


def upgrade() -> None:
    for table, stale in STALE_ROUTE_CONSTRAINTS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {stale}")

    for table, stale, intended in RENAMES:
        # IF EXISTS on the drop half of a rename is not available, so guard
        # the whole statement: a database created after this migration will
        # not carry the stale name at all.
        op.execute(
            f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_constraint
                     WHERE conname = '{stale}'
                       AND conrelid = '{table}'::regclass
                ) THEN
                    ALTER TABLE {table}
                      RENAME CONSTRAINT {stale} TO {intended};
                END IF;
            END
            $$;
            """
        )


def downgrade() -> None:
    for table, stale, intended in RENAMES:
        op.execute(
            f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_constraint
                     WHERE conname = '{intended}'
                       AND conrelid = '{table}'::regclass
                ) THEN
                    ALTER TABLE {table}
                      RENAME CONSTRAINT {intended} TO {stale};
                END IF;
            END
            $$;
            """
        )
    # The stale route constraints are not restored: they encode the old
    # vocabulary, and re-adding them would reject rows this schema writes.
