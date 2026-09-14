"""Vector retrieval over one patient's chunks (PRD §19).

The rule that shapes this module: the patient filter is part of the query,
not a step after it. ``WHERE patient_id = :scope`` sits inside the same
statement as the ``ORDER BY distance LIMIT k``, so the nearest-neighbour
search ranks only that patient's chunks and there is no moment at which
another patient's row exists in a result set. Retrieving broadly and
filtering in Python would mean the ANN index had already ranked other
patients' content, and one missing filter later would leak it.

The scope comes from :class:`~app.auth.context.AuthContext`. As everywhere
else in the services layer, no function here accepts a ``patient_id``.

Two retrievers, one scope. Vector search finds passages that *mean* the same
thing; keyword search finds the ones that contain the exact term. Clinical
questions need both — "HbA1c", "metformin", "LDL" and "MRI" are precisely
the tokens an embedding blurs into their neighbours, and precisely the ones
a patient types when they want that specific thing (PRD §19).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.context import AuthContext
from app.config import settings
from app.db.vector import distance_expression, similarity_from_distance
from app.models import ClinicalDocument, DocumentChunk


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk plus everything needed to cite it, in one row.

    Carrying the document title and date here means building a citation
    never needs a second query per result, and a chunk can never be cited
    with metadata belonging to a different document.
    """

    chunk_id: int
    document_id: int
    encounter_id: int | None
    section: str
    text: str
    chunk_date: date | None
    title: str | None
    document_type: str | None
    distance: float
    retriever: str = "vector"
    #: Set by the keyword retriever; ``ts_rank_cd`` is unbounded, so it is
    #: kept separate from the cosine-derived score rather than pretending
    #: the two are on one scale.
    keyword_rank: float = 0.0
    #: How many identical chunks this one stands for after deduplication,
    #: and when those others were recorded.
    occurrences: int = 1
    other_dates: tuple[date, ...] = ()

    @property
    def score(self) -> float:
        """Similarity in [0, 1] — higher is better."""
        return similarity_from_distance(self.distance)

    @property
    def dedup_key(self) -> str:
        """Identity for deduplication: the text itself, normalized.

        Not the chunk id — the point is to collapse the *same sentence*
        recorded at several visits, which have different ids by definition.
        """
        return " ".join(self.text.lower().split())


def _scoped_select(ctx: AuthContext) -> Select:
    """The base statement: one patient, joined to its document metadata."""
    return (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.encounter_id,
            DocumentChunk.section,
            DocumentChunk.chunk_text,
            DocumentChunk.chunk_date,
            ClinicalDocument.title,
            ClinicalDocument.document_type,
        )
        .join(ClinicalDocument, ClinicalDocument.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.patient_id == ctx.patient_scope,
            DocumentChunk.embedding.is_not(None),
        )
    )


def _apply_filters(
    stmt: Select,
    *,
    encounter_id: int | None,
    sections: tuple[str, ...] | None,
    from_date: date | None,
    to_date: date | None,
) -> Select:
    """Metadata filters, applied before ranking rather than after.

    Filtering after the fact would silently shrink a top-k of 20 into a
    top-3, because the discarded rows are the ones the ANN search already
    spent its budget on.
    """
    if encounter_id is not None:
        stmt = stmt.where(DocumentChunk.encounter_id == encounter_id)
    if sections:
        stmt = stmt.where(DocumentChunk.section.in_(sections))
    if from_date is not None:
        stmt = stmt.where(DocumentChunk.chunk_date >= from_date)
    if to_date is not None:
        stmt = stmt.where(DocumentChunk.chunk_date <= to_date)
    return stmt


async def vector_search(
    session: AsyncSession,
    ctx: AuthContext,
    *,
    query_vector: list[float],
    limit: int | None = None,
    encounter_id: int | None = None,
    sections: tuple[str, ...] | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
) -> list[RetrievedChunk]:
    """Nearest chunks to ``query_vector``, within this patient's records."""
    top_k = limit or settings.retrieval_vector_candidates
    distance = distance_expression(DocumentChunk.embedding, query_vector)

    stmt = _apply_filters(
        _scoped_select(ctx).add_columns(distance.label("distance")),
        encounter_id=encounter_id,
        sections=sections,
        from_date=from_date,
        to_date=to_date,
    ).order_by(distance.asc()).limit(top_k)

    rows = (await session.execute(stmt)).all()
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            encounter_id=row.encounter_id,
            section=row.section,
            text=row.chunk_text,
            chunk_date=row.chunk_date,
            title=row.title,
            document_type=row.document_type,
            distance=float(row.distance),
        )
        for row in rows
    ]


#: Question scaffolding that carries no retrieval signal. Kept short on
#: purpose: a long stop list strips "no" and "not", which change clinical
#: meaning, and PostgreSQL already drops English stop words when it builds
#: the tsquery.
_QUESTION_WORDS = frozenset(
    {
        "what", "when", "where", "which", "who", "why", "how", "did", "does",
        "do", "is", "are", "was", "were", "am", "my", "me", "i", "you", "your",
        "the", "a", "an", "about", "tell", "say", "said", "show", "give",
        "please", "can", "could", "would", "any", "have", "has", "had",
    }
)
_TOKEN = re.compile(r"[a-z0-9]+")


def build_tsquery_terms(query: str) -> list[str]:
    """Reduce a question to the terms worth matching on.

    Only ``[a-z0-9]+`` tokens survive, which is also what makes it safe to
    interpolate them into a ``to_tsquery`` expression: nothing that reaches
    PostgreSQL can contain tsquery operators.
    """
    return [
        token
        for token in _TOKEN.findall(query.lower())
        if len(token) > 1 and token not in _QUESTION_WORDS
    ]


async def keyword_search(
    session: AsyncSession,
    ctx: AuthContext,
    *,
    query: str,
    limit: int | None = None,
    encounter_id: int | None = None,
    sections: tuple[str, ...] | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
) -> list[RetrievedChunk]:
    """Full-text search over the same patient's chunks.

    The terms are OR-ed rather than AND-ed, and that choice decides whether
    this retriever contributes anything at all. ``websearch_to_tsquery``
    requires *every* unquoted term, so "what did my doctor say about my knee
    pain" demands a passage containing all of doctor, say, knee and pain —
    which no real note does, and keyword search silently returns nothing for
    precisely the natural-language questions it was added to help with.

    OR-ing means any term can match and ``ts_rank_cd`` decides how well.
    That ranking accounts for how close the matched terms sit to each other,
    so a passage about knee pain outranks one that mentions a knee in one
    sentence and pain in another.
    """
    terms = build_tsquery_terms(query)
    if not terms:
        # Nothing but question scaffolding. Returning early avoids a query
        # that would match every chunk with rank zero.
        return []

    top_k = limit or settings.retrieval_keyword_candidates
    tsquery = func.to_tsquery("english", " | ".join(terms))
    rank = func.ts_rank_cd(DocumentChunk.search_tsv, tsquery)

    stmt = (
        _apply_filters(
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                DocumentChunk.encounter_id,
                DocumentChunk.section,
                DocumentChunk.chunk_text,
                DocumentChunk.chunk_date,
                ClinicalDocument.title,
                ClinicalDocument.document_type,
                rank.label("rank"),
            )
            .join(ClinicalDocument, ClinicalDocument.id == DocumentChunk.document_id)
            .where(
                # The same scope predicate as the vector path, inside the
                # same statement. Keyword search is not a side door.
                DocumentChunk.patient_id == ctx.patient_scope,
                DocumentChunk.search_tsv.op("@@")(tsquery),
            ),
            encounter_id=encounter_id,
            sections=sections,
            from_date=from_date,
            to_date=to_date,
        )
        .order_by(rank.desc())
        .limit(top_k)
    )

    rows = (await session.execute(stmt)).all()
    return [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            encounter_id=row.encounter_id,
            section=row.section,
            text=row.chunk_text,
            chunk_date=row.chunk_date,
            title=row.title,
            document_type=row.document_type,
            # No vector distance on this path. Fusion ranks by position, so
            # the two retrievers never need a shared numeric scale.
            distance=1.0,
            retriever="keyword",
            keyword_rank=float(row.rank),
        )
        for row in rows
    ]


async def count_indexed_chunks(session: AsyncSession, ctx: AuthContext) -> int:
    """How many embedded chunks this patient has.

    Used to distinguish "nothing matched your question" from "nothing has
    been ingested yet" — two very different answers to give a user.
    """
    return (
        await session.scalar(
            select(func.count())
            .select_from(DocumentChunk)
            .where(
                DocumentChunk.patient_id == ctx.patient_scope,
                DocumentChunk.embedding.is_not(None),
            )
        )
    ) or 0
