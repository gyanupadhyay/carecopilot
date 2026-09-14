"""The RAG path: question in, grounded answer and citations out (PRD §19).

    normalize → authorize → ┌ vector search  ┐ → fuse → dedupe → rerank
                            └ keyword search ┘        → context → generate

Authorization is not a step the query passes through; it is the scope both
retrievals are performed *in* (see :mod:`app.rag.retrieval`). It appears in
the diagram because §20 names it, not because there is a filter here that
could be omitted.

Three behaviours worth stating, because each is a case where returning less
is the correct answer:

*Nothing ingested.* If the patient has no embedded chunks the pipeline says
so rather than answering from the model's general knowledge dressed up as a
record lookup.

*Nothing relevant.* If every candidate is below the similarity floor, the
context is left empty and the grounding prompt does its job — the model is
told it has no record context and answers accordingly.

*The same thing three times.* Identical passages recorded at several visits
are collapsed to one, carrying a note of how often they recur.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.context import AuthContext
from app.config import settings
from app.observability.logging import get_logger
from app.observability.trace import Trace
from app.rag.context import BuiltContext, build_context
from app.rag.embeddings import EmbeddingProvider
from app.rag.fusion import fuse
from app.rag.reranking import NoopReranker, Reranker
from app.rag.retrieval import (
    RetrievedChunk,
    count_indexed_chunks,
    keyword_search,
    vector_search,
)
from app.schemas.chat import Source

log = get_logger(__name__)

#: Chunks below this cosine similarity are treated as noise. With a
#: normalized BGE model, unrelated clinical sentences sit around 0.5-0.6, so
#: a floor here mostly removes results that would pad the context without
#: informing the answer. It is intentionally permissive: dropping a relevant
#: chunk is worse than including a marginal one the model can ignore.
MIN_SIMILARITY = 0.55


@dataclass(frozen=True, slots=True)
class RagResult:
    context: BuiltContext
    chunks: list[RetrievedChunk] = field(default_factory=list)
    #: True when the patient has no ingested chunks at all — a different
    #: problem from a question that simply did not match anything.
    index_empty: bool = False
    #: How the candidate set was assembled, for the developer panel.
    vector_candidates: int = 0
    keyword_candidates: int = 0
    overlap: int = 0
    deduplicated: int = 0
    reranker: str = "none"

    @property
    def sources(self) -> list[Source]:
        return self.context.sources

    @property
    def has_context(self) -> bool:
        return not self.context.is_empty

    @property
    def candidates(self) -> int:
        """Total candidates considered, counting an overlap once."""
        return self.vector_candidates + self.keyword_candidates - self.overlap


def normalize_query(question: str) -> str:
    """Light normalization before embedding.

    Collapses whitespace and strips the conversational scaffolding that
    carries no retrieval signal. Deliberately shallow — aggressive rewriting
    (stemming, stopword removal, synonym expansion) fights the embedding
    model, which was trained on natural sentences.
    """
    text = " ".join(question.split())
    text = re.sub(
        r"^(hi|hello|hey)[,!.\s]+", "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"^(can you|could you|please|i want to know|tell me)\s+", "", text,
        flags=re.IGNORECASE,
    )
    return text.strip() or question.strip()


@contextmanager
def _stage(trace: Trace | None, name: str) -> Iterator[None]:
    """Time a stage when a trace is present, and do nothing when it is not."""
    if trace is None:
        yield
    else:
        with trace.stage(name):
            yield


async def retrieve(
    session: AsyncSession,
    ctx: AuthContext,
    *,
    question: str,
    embedder: EmbeddingProvider,
    trace: Trace | None = None,
    top_k: int | None = None,
    min_similarity: float = MIN_SIMILARITY,
    reranker: Reranker | None = None,
    **filters: object,
) -> RagResult:
    """Run hybrid retrieval for one question and build the prompt context."""
    query = normalize_query(question)
    final_k = top_k or settings.retrieval_top_k
    ranker = reranker or NoopReranker()

    with _stage(trace, "embed"):
        query_vector = await embedder.embed_query(query)

    with _stage(trace, "retrieval"):
        # Both retrievers run against the same scope and the same filters.
        # They are sequential rather than concurrent on purpose: one session
        # cannot serve two statements at once, and a second connection per
        # question would double the pool for a few milliseconds' saving.
        vector_hits = await vector_search(
            session,
            ctx,
            query_vector=query_vector,
            limit=settings.retrieval_vector_candidates,
            **filters,  # type: ignore[arg-type]
        )
        keyword_hits = await keyword_search(
            session,
            ctx,
            query=query,
            limit=settings.retrieval_keyword_candidates,
            **filters,  # type: ignore[arg-type]
        )

    fused = fuse(vector_hits, keyword_hits)

    if not fused.chunks:
        # Distinguish "no match" from "nothing indexed" with one cheap count,
        # and only when neither retriever found anything.
        empty = await count_indexed_chunks(session, ctx) == 0
        if empty:
            log.info("rag.index_empty", patient_id=ctx.patient_id)
        return RagResult(context=build_context([]), chunks=[], index_empty=empty)

    # The floor applies to the vector score, which keyword-only hits do not
    # have. An exact term match is evidence in its own right — dropping it
    # for lacking a cosine score would discard precisely what keyword search
    # was added to catch.
    relevant = [
        chunk
        for chunk in fused.chunks
        if chunk.retriever == "keyword" or chunk.score >= min_similarity
    ]
    if not relevant:
        log.info("rag.below_floor", candidates=len(fused.chunks))
        return RagResult(
            context=build_context([]),
            chunks=[],
            vector_candidates=fused.vector_count,
            keyword_candidates=fused.keyword_count,
            overlap=fused.overlap_count,
            deduplicated=fused.deduplicated,
            reranker=ranker.name,
        )

    with _stage(trace, "rerank"):
        kept = await ranker.rerank(query, relevant, top_k=final_k)

    context = build_context(kept)

    if trace is not None:
        trace.retrieved_count = len(fused.chunks)
        trace.reranked_count = len(kept)
        trace.reranker = ranker.name
        trace.deduplicated = fused.deduplicated

    log.info(
        "rag.retrieved",
        vector=fused.vector_count,
        keyword=fused.keyword_count,
        overlap=fused.overlap_count,
        deduplicated=fused.deduplicated,
        above_floor=len(relevant),
        reranker=ranker.name,
        used=context.used_chunks,
        top_score=round(kept[0].score, 4) if kept else None,
    )
    return RagResult(
        context=context,
        chunks=kept,
        index_empty=False,
        vector_candidates=fused.vector_count,
        keyword_candidates=fused.keyword_count,
        overlap=fused.overlap_count,
        deduplicated=fused.deduplicated,
        reranker=ranker.name,
    )
