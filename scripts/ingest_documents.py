"""Chunk and embed every clinical document.

    python scripts/ingest_documents.py

Run after ``generate_data.py``, and again after any change to the chunker or
to ``EMBEDDING_MODEL`` — vectors from different models are not comparable,
so a partial re-embed produces rankings that look fine and mean nothing.

    --only-missing   only documents that have no chunks yet (resume a run)
    --limit N        first N documents, for a quick check
    --provider X     override EMBEDDING_PROVIDER (e.g. "hashing", offline)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from sqlalchemy import select

from app.db.session import AppSession, dispose_engines
from app.db.vector import using_pgvector
from app.models import ClinicalDocument
from app.observability.logging import configure_logging
from app.rag.embeddings import build_embedder
from app.rag.ingestion import chunk_statistics, ingest_documents
from app.runtime import selector_loop_factory


async def run(args: argparse.Namespace) -> int:
    embedder = build_embedder(provider=args.provider)

    async with AppSession() as session:
        document_ids = None
        if args.limit:
            document_ids = list(
                (
                    await session.scalars(
                        select(ClinicalDocument.id)
                        .order_by(ClinicalDocument.id)
                        .limit(args.limit)
                    )
                ).all()
            )
            if not document_ids:
                print("No clinical documents found. Run generate_data.py first.")
                return 1

        started = time.perf_counter()
        report = await ingest_documents(
            session,
            embedder=embedder,
            document_ids=document_ids,
            only_missing=args.only_missing,
        )
        elapsed = time.perf_counter() - started

        stats = await chunk_statistics(session)

    await dispose_engines()

    backend = "pgvector" if using_pgvector() else "array (no ANN index)"
    print(f"Ingested {report} in {elapsed:.1f}s")
    print(f"  embedding model : {embedder.model_name} ({embedder.dimension}d)")
    print(f"  vector backend  : {backend}")
    print(f"  chunks in db    : {stats['embedded']} embedded / {stats['chunks']} total")
    print(f"  documents       : {stats['documents']}")
    if report.chunks:
        per_doc = stats["chunks"] / max(1, stats["documents"])
        print(f"  avg chunks/doc  : {per_doc:.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--provider", default=None)
    args = parser.parse_args(argv)

    configure_logging()
    return asyncio.run(run(args), loop_factory=selector_loop_factory)


if __name__ == "__main__":
    raise SystemExit(main())
