"""Chunk provenance, and narrow the Text-to-SQL allowlist to §16.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-13

Two changes, both closing acceptance-criteria gaps.

``document_version`` and ``status`` on every chunk (PRD §19). A retrieved
fragment currently cannot say which revision of a note it came from, or
whether that note was a draft. Both matter for a citation: quoting a
superseded draft as though it were the final record is a correctness
failure, not a formatting one. Backfilled from the parent document.

Narrowing the analytics grants to the four approved tables (PRD §16).
Migration 0001 granted the read-only role eight tables — it also covered
``patients``, ``providers``, ``clinical_documents`` and ``document_chunks``,
which is broader than the PRD approves for generated SQL. The RLS policies
on those tables stay: a revoked grant and a policy fail independently, and
keeping both means re-granting a table by mistake does not also re-expose
every patient in it.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ANALYTICS_ROLE = "carecopilot_ro"

#: The only tables generated SQL may name (PRD §16).
APPROVED_TABLES = ("lab_results", "appointments", "medications", "encounters")

#: Granted by 0001, outside §16's list, revoked here.
REVOKED_TABLES = ("patients", "providers", "clinical_documents", "document_chunks")


def _if_role_exists(statements: str) -> str:
    return f"""
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{ANALYTICS_ROLE}') THEN
            {statements}
        END IF;
    END
    $$;
    """


def upgrade() -> None:
    op.add_column(
        "document_chunks", sa.Column("document_version", sa.Integer(), nullable=True)
    )
    op.add_column("document_chunks", sa.Column("status", sa.String(16), nullable=True))
    op.execute(
        """
        UPDATE document_chunks c
           SET document_version = d.version,
               status           = d.status
          FROM clinical_documents d
         WHERE d.id = c.document_id
        """
    )

    op.execute(
        _if_role_exists(
            f"EXECUTE 'REVOKE ALL ON {', '.join(REVOKED_TABLES)} "
            f"FROM {ANALYTICS_ROLE}';"
        )
    )


def downgrade() -> None:
    op.execute(
        _if_role_exists(
            f"EXECUTE 'GRANT SELECT ON {', '.join(REVOKED_TABLES)} "
            f"TO {ANALYTICS_ROLE}';"
        )
    )
    op.drop_column("document_chunks", "status")
    op.drop_column("document_chunks", "document_version")
