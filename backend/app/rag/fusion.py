"""Merging the two retrievers, and collapsing duplicates (PRD §19).

Vector similarity and ``ts_rank_cd`` are not comparable numbers. Cosine
similarity lives in [0, 1] and clusters tightly — unrelated clinical
sentences sit around 0.55 — while ``ts_rank_cd`` is unbounded and depends on
document length and term proximity. Normalizing them onto a shared scale
means inventing a conversion that has no meaning, and the weighting it
implies would be arbitrary.

Reciprocal Rank Fusion sidesteps that entirely: it uses each retriever's
*ordering* and ignores its scores. A chunk ranked first by either retriever
scores ``1/(k+1)``; one found by both accumulates from both lists, which is
exactly the signal worth rewarding — agreement between two methods that fail
in different ways.

``k`` (default 60) flattens the curve. A small ``k`` makes rank 1 dominate
and effectively picks one retriever's winner; a large one makes the first
twenty results nearly interchangeable. 60 is the value from the original RRF
paper and is a reasonable default rather than a tuned one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from app.config import settings
from app.rag.retrieval import RetrievedChunk


@dataclass(frozen=True, slots=True)
class FusionResult:
    chunks: list[RetrievedChunk]
    vector_count: int
    keyword_count: int
    #: Chunks that both retrievers returned — the strongest candidates.
    overlap_count: int
    #: Identical passages collapsed into their highest-ranked occurrence.
    deduplicated: int


def reciprocal_rank_fusion(
    *ranked_lists: Sequence[RetrievedChunk], k: int | None = None
) -> list[RetrievedChunk]:
    """Merge ranked lists by position, best first.

    Ties are broken by vector score so the ordering is deterministic; two
    chunks with the same fused score would otherwise depend on dict
    insertion order, which makes evaluation runs irreproducible.
    """
    constant = k if k is not None else settings.rrf_k
    scores: dict[int, float] = {}
    best: dict[int, RetrievedChunk] = {}
    seen_in: dict[int, set[str]] = {}

    for ranked in ranked_lists:
        for position, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                constant + position
            )
            seen_in.setdefault(chunk.chunk_id, set()).add(chunk.retriever)
            # Keep whichever copy carries a real vector distance, so the
            # surviving chunk reports a meaningful similarity score.
            existing = best.get(chunk.chunk_id)
            if existing is None or (
                existing.retriever == "keyword" and chunk.retriever == "vector"
            ):
                best[chunk.chunk_id] = chunk

    def _label(chunk_id: int) -> str:
        found = seen_in[chunk_id]
        return "hybrid" if len(found) > 1 else next(iter(found))

    merged = [
        replace(best[chunk_id], retriever=_label(chunk_id))
        for chunk_id in sorted(
            scores, key=lambda cid: (-scores[cid], -best[cid].score, cid)
        )
    ]
    return merged


def deduplicate(chunks: Iterable[RetrievedChunk]) -> tuple[list[RetrievedChunk], int]:
    """Collapse identical passages, keeping the highest-ranked one.

    Clinical notes repeat themselves — the same sentence recorded at three
    visits is three chunks with identical text. Sending all three spends the
    context budget three times on one fact and pushes out passages that
    would have added something.

    The fact that it recurred is not discarded: the survivor records how
    many occurrences it stands for and the other dates, so a citation can
    still say the finding appears across several visits.
    """
    kept: list[RetrievedChunk] = []
    index: dict[str, int] = {}
    removed = 0

    for chunk in chunks:
        key = chunk.dedup_key
        position = index.get(key)
        if position is None:
            index[key] = len(kept)
            kept.append(chunk)
            continue

        removed += 1
        survivor = kept[position]
        extra_dates = tuple(
            date
            for date in (*survivor.other_dates, chunk.chunk_date)
            if date is not None and date != survivor.chunk_date
        )
        kept[position] = replace(
            survivor,
            occurrences=survivor.occurrences + 1,
            other_dates=tuple(dict.fromkeys(extra_dates)),
        )

    return kept, removed


def fuse(
    vector_hits: Sequence[RetrievedChunk],
    keyword_hits: Sequence[RetrievedChunk],
    *,
    k: int | None = None,
) -> FusionResult:
    """Merge both retrievers and deduplicate, in that order.

    Order matters. Fusing first lets a passage that both retrievers found
    earn its rank before deduplication picks which copy survives, so the
    survivor is the best-ranked one rather than whichever happened to be
    encountered first.
    """
    vector_ids = {chunk.chunk_id for chunk in vector_hits}
    keyword_ids = {chunk.chunk_id for chunk in keyword_hits}

    merged = reciprocal_rank_fusion(vector_hits, keyword_hits, k=k)
    deduped, removed = deduplicate(merged)

    return FusionResult(
        chunks=deduped,
        vector_count=len(vector_hits),
        keyword_count=len(keyword_hits),
        overlap_count=len(vector_ids & keyword_ids),
        deduplicated=removed,
    )
