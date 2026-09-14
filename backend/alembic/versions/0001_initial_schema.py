"""Initial schema: clinical data, retrieval chunks, conversations, audit.

Revision ID: 0001
Revises:
Create Date: 2026-09-13

Beyond the tables, this migration installs the three things the application
depends on but cannot create for itself at runtime:

1. the ``vector`` extension (pgvector backend only);
2. ``cc_cosine_distance``, the in-database ranking function used by the
   ``array`` fallback backend;
3. row-level security policies plus SELECT grants for the read-only
   analytics role.

The RLS policies are the authorization boundary for Text-to-SQL. They hold
even if the SQL validator has a bug and even if the LLM writes a query with
no patient predicate at all, because the analytics role is not the table
owner and therefore cannot escape its own policies.

If ``CREATE EXTENSION vector`` fails here, the server does not have pgvector
installed. Set ``VECTOR_BACKEND=array`` and re-run: the schema is identical
apart from the embedding column type and the absence of the HNSW index.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from app.db.vector import embedding_type, using_pgvector
from app.models.enums import (
    APPOINTMENT_STATUSES,
    APPOINTMENT_TYPES,
    DOCUMENT_STATUSES,
    DOCUMENT_TYPES,
    ENCOUNTER_TYPES,
    MEDICATION_STATUSES,
    MESSAGE_ROLES,
    ROUTES,
    USER_ROLES,
    check_in,
)

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Tables the read-only analytics role may see at all. Anything absent here
#: is invisible to Text-to-SQL no matter what the generated query says.
ANALYTICS_PATIENT_SCOPED = (
    "appointments",
    "encounters",
    "medications",
    "lab_results",
    "clinical_documents",
    "document_chunks",
)

#: Reference data with no patient column. Provider names appear on a
#: patient's own appointments, so hiding the table would break ordinary
#: joins without protecting anything patient-specific.
ANALYTICS_REFERENCE = ("providers",)

ANALYTICS_ROLE = "carecopilot_ro"

#: The patient scope every policy compares against.
#:
#: ``NULLIF(..., '')`` is load-bearing. ``current_setting(name, true)``
#: returns NULL only while a custom setting has *never* been assigned in the
#: session; once ``SET LOCAL`` has run and been rolled back, the setting
#: reverts to an empty string instead. Connections are pooled, so that is
#: the normal state of a reused connection — and ``''::int`` raises
#: ``invalid input syntax for type integer`` rather than evaluating to
#: false. Mapping empty to NULL makes an unscoped session return zero rows,
#: which is the intended failure mode: closed, and quiet about it.
SCOPE_EXPRESSION = "NULLIF(current_setting('app.patient_id', true), '')::int"


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        server_default=sa.text("now()"),
        nullable=False,
    )


def _updated_at() -> sa.Column:
    return sa.Column(
        "updated_at",
        sa.DateTime(timezone=True),
        server_default=sa.text("now()"),
        nullable=False,
    )


COSINE_DISTANCE_FN = """
CREATE OR REPLACE FUNCTION cc_cosine_distance(
    a double precision[], b double precision[]
) RETURNS double precision
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE STRICT
AS $fn$
DECLARE
    dot double precision;
    norm_a double precision;
    norm_b double precision;
BEGIN
    -- Mismatched dimensions mean the row was embedded by a different
    -- model. Rank it last rather than silently comparing a prefix.
    IF array_length(a, 1) IS DISTINCT FROM array_length(b, 1) THEN
        RETURN 1.0;
    END IF;

    SELECT sum(x * y), sqrt(sum(x * x)), sqrt(sum(y * y))
      INTO dot, norm_a, norm_b
      FROM unnest(a, b) AS t(x, y);

    IF norm_a = 0 OR norm_b = 0 THEN
        RETURN 1.0;
    END IF;

    RETURN 1.0 - (dot / (norm_a * norm_b));
END;
$fn$;
"""


def _apply_analytics_security() -> None:
    """Enable RLS and grant the analytics role its narrow view.

    Wrapped in a role-existence check so that a developer database created
    without ``docker/initdb/20-roles.sql`` still migrates cleanly; in that
    case ANALYTICS_DATABASE_URL falls back to the app URL, which
    ``Settings._check_secrets`` already refuses outside development.
    """
    for table in ANALYTICS_PATIENT_SCOPED:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY {table}_patient_isolation ON {table}
            FOR SELECT
            USING (patient_id = {SCOPE_EXPRESSION})
            """
        )

    # patients itself keys on id rather than patient_id.
    op.execute("ALTER TABLE patients ENABLE ROW LEVEL SECURITY")
    op.execute(
        f"""
        CREATE POLICY patients_patient_isolation ON patients
        FOR SELECT
        USING (id = {SCOPE_EXPRESSION})
        """
    )

    granted = ", ".join(ANALYTICS_PATIENT_SCOPED + ANALYTICS_REFERENCE + ("patients",))
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{ANALYTICS_ROLE}') THEN
                EXECUTE 'GRANT USAGE ON SCHEMA public TO {ANALYTICS_ROLE}';
                EXECUTE 'GRANT SELECT ON {granted} TO {ANALYTICS_ROLE}';
            END IF;
        END
        $$;
        """
    )


def upgrade() -> None:
    if using_pgvector():
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(COSINE_DISTANCE_FN)

    # ---------------------------------------------------------------- #
    # Reference and demographic data
    # ---------------------------------------------------------------- #
    op.create_table(
        "patients",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("external_id", sa.String(32), nullable=False),
        sa.Column("first_name", sa.String(100), nullable=False),
        sa.Column("last_name", sa.String(100), nullable=False),
        sa.Column("date_of_birth", sa.Date(), nullable=False),
        sa.Column("gender", sa.String(32), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_patients"),
        sa.UniqueConstraint("external_id", name="uq_patients_external_id"),
    )
    op.create_index("ix_patients_created_at", "patients", ["created_at"])

    op.create_table(
        "providers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("specialty", sa.String(120), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_providers"),
    )
    op.create_index("ix_providers_created_at", "providers", ["created_at"])

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(200), nullable=False),
        sa.Column("display_name", sa.String(120), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=True),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
        sa.CheckConstraint(check_in("role", USER_ROLES), name="ck_users_user_role"),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_users_patient_id_patients",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_users_created_at", "users", ["created_at"])

    # ---------------------------------------------------------------- #
    # Clinical timeline
    # ---------------------------------------------------------------- #
    op.create_table(
        "encounters",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column("provider_id", sa.Integer(), nullable=True),
        sa.Column("encounter_date", sa.Date(), nullable=False),
        sa.Column("encounter_type", sa.String(32), nullable=False),
        sa.Column("reason", sa.String(200), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_encounters"),
        sa.CheckConstraint(
            check_in("encounter_type", ENCOUNTER_TYPES),
            name="ck_encounters_encounter_type",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_encounters_patient_id_patients",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["provider_id"],
            ["providers.id"],
            name="fk_encounters_provider_id_providers",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_encounters_created_at", "encounters", ["created_at"])
    op.create_index(
        "ix_encounters_patient_id_encounter_date",
        "encounters",
        ["patient_id", "encounter_date"],
    )

    op.create_table(
        "appointments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column("provider_id", sa.Integer(), nullable=True),
        sa.Column("appointment_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("appointment_type", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_appointments"),
        sa.CheckConstraint(
            check_in("status", APPOINTMENT_STATUSES),
            name="ck_appointments_appointment_status",
        ),
        sa.CheckConstraint(
            check_in("appointment_type", APPOINTMENT_TYPES),
            name="ck_appointments_appointment_type",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_appointments_patient_id_patients",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["provider_id"],
            ["providers.id"],
            name="fk_appointments_provider_id_providers",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_appointments_created_at", "appointments", ["created_at"])
    op.create_index(
        "ix_appointments_patient_id_appointment_date",
        "appointments",
        ["patient_id", "appointment_date"],
    )

    op.create_table(
        "medications",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column("encounter_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("dosage", sa.String(64), nullable=False),
        sa.Column("frequency", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_medications"),
        sa.CheckConstraint(
            check_in("status", MEDICATION_STATUSES),
            name="ck_medications_medication_status",
        ),
        sa.CheckConstraint(
            "end_date IS NULL OR end_date >= start_date",
            name="ck_medications_medication_date_order",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_medications_patient_id_patients",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["encounter_id"],
            ["encounters.id"],
            name="fk_medications_encounter_id_encounters",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_medications_created_at", "medications", ["created_at"])
    op.create_index(
        "ix_medications_patient_id_status", "medications", ["patient_id", "status"]
    )
    op.create_index(
        "ix_medications_patient_id_start_date",
        "medications",
        ["patient_id", "start_date"],
    )

    op.create_table(
        "lab_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column("encounter_id", sa.Integer(), nullable=True),
        sa.Column("test_name", sa.String(120), nullable=False),
        sa.Column("value", sa.Numeric(10, 3), nullable=False),
        sa.Column("unit", sa.String(32), nullable=False),
        sa.Column("reference_range", sa.String(64), nullable=True),
        sa.Column("result_date", sa.Date(), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_lab_results"),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_lab_results_patient_id_patients",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["encounter_id"],
            ["encounters.id"],
            name="fk_lab_results_encounter_id_encounters",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_lab_results_created_at", "lab_results", ["created_at"])
    op.create_index(
        "ix_lab_results_patient_id_test_name_result_date",
        "lab_results",
        ["patient_id", "test_name", "result_date"],
    )

    # ---------------------------------------------------------------- #
    # Unstructured documentation and retrieval chunks
    # ---------------------------------------------------------------- #
    op.create_table(
        "clinical_documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column("encounter_id", sa.Integer(), nullable=True),
        sa.Column("document_type", sa.String(32), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        _created_at(),
        _updated_at(),
        sa.PrimaryKeyConstraint("id", name="pk_clinical_documents"),
        sa.CheckConstraint(
            check_in("document_type", DOCUMENT_TYPES),
            name="ck_clinical_documents_document_type",
        ),
        sa.CheckConstraint(
            check_in("status", DOCUMENT_STATUSES),
            name="ck_clinical_documents_document_status",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_clinical_documents_patient_id_patients",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["encounter_id"],
            ["encounters.id"],
            name="fk_clinical_documents_encounter_id_encounters",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_clinical_documents_created_at", "clinical_documents", ["created_at"]
    )
    op.create_index(
        "ix_clinical_documents_patient_id_encounter_id",
        "clinical_documents",
        ["patient_id", "encounter_id"],
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column("encounter_id", sa.Integer(), nullable=True),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("section", sa.String(64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_date", sa.Date(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("embedding", embedding_type(), nullable=True),
        sa.Column(
            "search_tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', chunk_text)", persisted=True),
            nullable=True,
        ),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_document_chunks"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunk_position"),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["clinical_documents.id"],
            name="fk_document_chunks_document_id_clinical_documents",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_document_chunks_patient_id_patients",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["encounter_id"],
            ["encounters.id"],
            name="fk_document_chunks_encounter_id_encounters",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_document_chunks_created_at", "document_chunks", ["created_at"])
    op.create_index("ix_document_chunks_patient_id", "document_chunks", ["patient_id"])
    op.create_index(
        "ix_document_chunks_patient_id_encounter_id",
        "document_chunks",
        ["patient_id", "encounter_id"],
    )
    op.create_index(
        "ix_document_chunks_search_tsv",
        "document_chunks",
        ["search_tsv"],
        postgresql_using="gin",
    )
    if using_pgvector():
        op.create_index(
            "ix_document_chunks_embedding_hnsw",
            "document_chunks",
            ["embedding"],
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )

    # ---------------------------------------------------------------- #
    # Conversation memory
    # ---------------------------------------------------------------- #
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=True),
        sa.Column("title", sa.String(200), nullable=True),
        _created_at(),
        _updated_at(),
        sa.PrimaryKeyConstraint("id", name="pk_conversations"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_conversations_user_id_users",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_conversations_patient_id_patients",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_conversations_created_at", "conversations", ["created_at"])
    op.create_index(
        "ix_conversations_user_id_created_at",
        "conversations",
        ["user_id", "created_at"],
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("route", sa.String(16), nullable=True),
        sa.Column("sources", postgresql.JSONB(), nullable=True),
        sa.Column("meta", postgresql.JSONB(), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_messages"),
        sa.CheckConstraint(
            check_in("role", MESSAGE_ROLES), name="ck_messages_message_role"
        ),
        sa.CheckConstraint(
            f"route IS NULL OR {check_in('route', ROUTES)}",
            name="ck_messages_message_route",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_messages_conversation_id_conversations",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_messages_created_at", "messages", ["created_at"])
    op.create_index(
        "ix_messages_conversation_id_created_at",
        "messages",
        ["conversation_id", "created_at"],
    )

    # ---------------------------------------------------------------- #
    # Audit and observability
    # ---------------------------------------------------------------- #
    op.create_table(
        "action_audit",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("patient_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("target_type", sa.String(64), nullable=True),
        sa.Column("target_id", sa.Integer(), nullable=True),
        sa.Column("outcome", sa.String(16), nullable=False),
        sa.Column("params", postgresql.JSONB(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_action_audit"),
        sa.CheckConstraint(
            "outcome IN ('proposed', 'confirmed', 'executed', 'rejected', 'failed')",
            name="ck_action_audit_action_outcome",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_action_audit_user_id_users",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_action_audit_patient_id_patients",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_action_audit_created_at", "action_audit", ["created_at"])
    op.create_index(
        "ix_action_audit_patient_id_created_at",
        "action_audit",
        ["patient_id", "created_at"],
    )

    op.create_table(
        "request_traces",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("patient_id", sa.Integer(), nullable=True),
        sa.Column("route", sa.String(16), nullable=True),
        sa.Column("route_confidence", sa.Float(), nullable=True),
        sa.Column("model", sa.String(120), nullable=True),
        sa.Column("total_ms", sa.Integer(), nullable=True),
        sa.Column("stage_ms", postgresql.JSONB(), nullable=True),
        sa.Column("tool_calls", postgresql.JSONB(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
        sa.Column("retrieved_count", sa.Integer(), nullable=True),
        sa.Column("reranked_count", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_request_traces"),
        sa.UniqueConstraint("request_id", name="uq_request_traces_request_id"),
        sa.CheckConstraint(
            f"route IS NULL OR {check_in('route', ROUTES)}",
            name="ck_request_traces_trace_route",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_request_traces_conversation_id_conversations",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_request_traces_user_id_users",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_request_traces_patient_id_patients",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_request_traces_created_at", "request_traces", ["created_at"])
    op.create_index(
        "ix_request_traces_created_at_route", "request_traces", ["created_at", "route"]
    )

    _apply_analytics_security()


def downgrade() -> None:
    for table in (
        "request_traces",
        "action_audit",
        "messages",
        "conversations",
        "document_chunks",
        "clinical_documents",
        "lab_results",
        "medications",
        "appointments",
        "encounters",
        "users",
        "providers",
        "patients",
    ):
        # Policies and indexes are dropped with the table they belong to.
        op.drop_table(table)

    op.execute("DROP FUNCTION IF EXISTS cc_cosine_distance(double precision[], double precision[])")
