"""Document ingestion: parse, chunk, embed, store (PRD §19).

    clinical_document -> split into sections -> chunk -> embed -> document_chunks

Two properties this is built around.

*Idempotence.* Re-ingesting a document deletes its existing chunks and
writes fresh ones in one transaction. Ingestion is re-run whenever the
chunker or the embedding model changes, and a pipeline that accumulated
duplicate chunks on every run would degrade retrieval a little more each
time — the same passage appearing three times in a top-5 crowds out three
other passages.

*Model identity matters.* Vectors from different models are not comparable;
mixing them produces rankings that look plausible and are meaningless. The
embedder's dimension is checked on every batch, and switching models means
re-ingesting everything rather than topping up.

This runs as a batch job, not on the request path. It is called from
``scripts/ingest_documents.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ClinicalDocument, DocumentChunk, Encounter
from app.observability.logging import get_logger
from app.rag.chunking import chunk_document, contextualize
from app.rag.embeddings import EmbeddingProvider

log = get_logger(__name__)


@dataclass(slots=True)
class IngestionReport:
    documents: int = 0
    chunks: int = 0
    skipped: int = 0

    def __str__(self) -> str:
        return (
            f"{self.documents} documents, {self.chunks} chunks "
            f"({self.skipped} unchanged and skipped)"
        )


async def ingest_document(
    session: AsyncSession,
    document: ClinicalDocument,
    *,
    embedder: EmbeddingProvider,
    encounter_date=None,  # type: ignore[no-untyped-def]
) -> int:
    """Re-chunk and re-embed one document. Returns the chunk count.

    The document's own ``patient_id`` is copied onto every chunk. Retrieval
    filters on the chunk's copy, so this is the single point where that
    denormalization is established — get it right here and no query can
    cross patients later.
    """
    chunks = chunk_document(document.content)
    if not chunks:
        log.warning("ingestion.empty_document", document_id=document.id)
        return 0

    date_label = encounter_date.isoformat() if encounter_date else None
    passages = [
        contextualize(chunk, title=document.title, date=date_label)
        for chunk in chunks
    ]
    vectors = await embedder.embed_documents(passages)

    await session.execute(
        delete(DocumentChunk).where(DocumentChunk.document_id == document.id)
    )
    session.add_all(
        DocumentChunk(
            document_id=document.id,
            patient_id=document.patient_id,
            encounter_id=document.encounter_id,
            chunk_text=chunk.text,
            section=chunk.section,
            chunk_index=chunk.index,
            chunk_date=encounter_date,
            token_count=chunk.token_estimate,
            # Provenance travels with the chunk so a citation can name the
            # revision it quotes without re-reading the document.
            document_version=document.version,
            status=document.status,
            embedding=vector,
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    )
    await session.flush()
    return len(chunks)


async def ingest_documents(
    session: AsyncSession,
    *,
    embedder: EmbeddingProvider,
    document_ids: Sequence[int] | None = None,
    only_missing: bool = False,
    batch_size: int = 25,
) -> IngestionReport:
    """Ingest every clinical document, or a named subset.

    ``only_missing`` restricts the run to documents with no chunks yet,
    which makes re-running after a partial failure cheap. It is off by
    default because the common reason to re-run is that the chunker or the
    model changed, and in that case every document needs redoing.
    """
    report = IngestionReport()

    stmt = select(ClinicalDocument, Encounter.encounter_date).outerjoin(
        Encounter, Encounter.id == ClinicalDocument.encounter_id
    )
    if document_ids:
        stmt = stmt.where(ClinicalDocument.id.in_(document_ids))
    if only_missing:
        already = select(DocumentChunk.document_id).where(
            DocumentChunk.embedding.is_not(None)
        )
        stmt = stmt.where(ClinicalDocument.id.not_in(already))
    stmt = stmt.order_by(ClinicalDocument.id)

    rows = (await session.execute(stmt)).all()
    log.info("ingestion.started", documents=len(rows), model=embedder.model_name)

    for position, (document, encounter_date) in enumerate(rows, start=1):
        count = await ingest_document(
            session, document, embedder=embedder, encounter_date=encounter_date
        )
        if count == 0:
            report.skipped += 1
            continue
        report.documents += 1
        report.chunks += count

        # Commit in batches: one transaction over hundreds of documents
        # holds locks for the whole run and loses everything on a failure
        # at the end.
        if position % batch_size == 0:
            await session.commit()
            log.info("ingestion.progress", done=position, total=len(rows))

    await session.commit()
    log.info(
        "ingestion.finished",
        documents=report.documents,
        chunks=report.chunks,
        model=embedder.model_name,
    )
    return report


async def chunk_statistics(session: AsyncSession) -> dict[str, int]:
    """Counts for the ingestion script's summary."""
    total = await session.scalar(select(func.count()).select_from(DocumentChunk)) or 0
    embedded = (
        await session.scalar(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.embedding.is_not(None))
        )
        or 0
    )
    documents = (
        await session.scalar(
            select(func.count(func.distinct(DocumentChunk.document_id)))
        )
        or 0
    )
    return {"chunks": total, "embedded": embedded, "documents": documents}
