"""Unstructured clinical documentation and its retrievable chunks.

Two tables, deliberately:

``clinical_documents``  the authored artifact — full text, version, status.
``document_chunks``     the retrieval unit — section-scoped text plus the
                        metadata every chunk must carry (PRD §19) so that a
                        retrieved fragment can be filtered, cited, and traced
                        back to an encounter without a second query.

``patient_id`` is denormalized onto the chunk on purpose. Retrieval filters
on it *inside* the vector/keyword query rather than joining up to the
document afterwards — filtering after retrieval would mean the ANN search
ranked other patients' chunks, which is exactly the failure mode the PRD
forbids in §33.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    Computed,
    Date,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UpdatedAtMixin
from app.db.vector import embedding_type, using_pgvector
from app.models.enums import DOCUMENT_STATUSES, DOCUMENT_TYPES, check_in

if TYPE_CHECKING:
    from app.models.encounter import Encounter
    from app.models.patient import Patient


class ClinicalDocument(Base, TimestampMixin, UpdatedAtMixin):
    __tablename__ = "clinical_documents"
    __table_args__ = (
        CheckConstraint(check_in("document_type", DOCUMENT_TYPES), name="document_type"),
        CheckConstraint(check_in("status", DOCUMENT_STATUSES), name="document_status"),
        Index(
            "ix_clinical_documents_patient_id_encounter_id",
            "patient_id",
            "encounter_id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE")
    )
    encounter_id: Mapped[int | None] = mapped_column(
        ForeignKey("encounters.id", ondelete="CASCADE")
    )
    document_type: Mapped[str] = mapped_column(String(32), default="clinical_note")
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="final")

    patient: Mapped[Patient] = relationship(back_populates="documents")
    encounter: Mapped[Encounter | None] = relationship(back_populates="documents")
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
    )


def _chunk_indexes() -> tuple[Index, ...]:
    """Indexes for ``document_chunks``.

    The ANN index is backend-specific: HNSW with ``vector_cosine_ops`` only
    exists when pgvector is in play. Under the array fallback there is no
    index to create — ranking degrades to a sequential scan over one
    patient's chunks, which the ``patient_id`` filter keeps small.
    """
    common = (
        Index("ix_document_chunks_patient_id", "patient_id"),
        Index(
            "ix_document_chunks_patient_id_encounter_id",
            "patient_id",
            "encounter_id",
        ),
        Index("ix_document_chunks_search_tsv", "search_tsv", postgresql_using="gin"),
    )
    if not using_pgvector():
        return common
    return (
        *common,
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class DocumentChunk(Base, TimestampMixin):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_position"),
        *_chunk_indexes(),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("clinical_documents.id", ondelete="CASCADE")
    )
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE")
    )
    encounter_id: Mapped[int | None] = mapped_column(
        ForeignKey("encounters.id", ondelete="CASCADE")
    )
    chunk_text: Mapped[str] = mapped_column(Text)
    section: Mapped[str] = mapped_column(String(64))
    chunk_index: Mapped[int] = mapped_column(Integer)
    #: Denormalized encounter date. Metadata filters ("during the last year")
    #: run against the chunk row itself, never a post-retrieval join.
    chunk_date: Mapped[date | None] = mapped_column(Date)
    token_count: Mapped[int | None] = mapped_column(Integer)

    #: Copied from the parent document at ingestion (PRD §19). A citation
    #: has to be able to say *which revision* it quotes, and whether that
    #: revision was a draft — quoting a superseded draft as the record is a
    #: correctness failure, not a formatting one.
    document_version: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str | None] = mapped_column(String(16))

    #: Dimensionality and column type follow ``VECTOR_BACKEND``; see
    #: :mod:`app.db.vector`. NULL until the ingestion pipeline embeds it.
    embedding: Mapped[Any] = mapped_column(embedding_type(), nullable=True)

    #: Maintained by PostgreSQL, so a chunk can never drift out of sync with
    #: its own full-text index the way an application-maintained column can.
    search_tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', chunk_text)", persisted=True),
        nullable=True,
    )

    document: Mapped[ClinicalDocument] = relationship(back_populates="chunks")
