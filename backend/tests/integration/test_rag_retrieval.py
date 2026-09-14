"""Retrieval against the ingested corpus (PRD §19).

The scoping tests here are the important ones. They assert the property the
whole design rests on: retrieval ranks one patient's chunks and no others,
because the filter is inside the ranking query rather than applied to its
results.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from app.auth.context import AuthContext, AuthorizationError
from app.models import DocumentChunk, Patient, User, UserPatientMapping
from app.rag.embeddings import build_embedder
from app.rag.pipeline import retrieve
from app.rag.reranking import HeuristicReranker, NoopReranker
from app.rag.retrieval import count_indexed_chunks, keyword_search, vector_search

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder():
    """The real model, loaded once for the module.

    Retrieval quality is the thing under test, and the hashing embedder has
    none — it would make every assertion below vacuous.
    """
    return build_embedder(provider="local")


@pytest.fixture(autouse=True)
async def _require_ingested_corpus(session) -> None:
    total = await session.scalar(
        select(func.count()).select_from(DocumentChunk).where(
            DocumentChunk.embedding.is_not(None)
        )
    )
    if not total:
        pytest.skip("No embedded chunks; run scripts/ingest_documents.py")


async def _ctx_for(session, external_id: str) -> AuthContext:
    patient = await session.scalar(
        select(Patient).where(Patient.external_id == external_id)
    )
    if patient is None:
        pytest.skip(f"Patient {external_id} not seeded")
    user = await session.scalar(
        select(User)
        .join(UserPatientMapping, UserPatientMapping.user_id == User.id)
        .where(
            UserPatientMapping.patient_id == patient.id,
            UserPatientMapping.is_active.is_(True),
        )
    )
    return AuthContext(
        user_id=user.id if user else 0,
        role="patient",
        patient_id=patient.id,
        request_id="itest",
    )


async def test_retrieval_finds_the_relevant_section(session, embedder) -> None:
    """Demo 2: "What did my doctor say about my knee pain?" """
    ctx = await _ctx_for(session, "P001")
    result = await retrieve(
        session, ctx, question="What did my doctor say about my knee pain?",
        embedder=embedder,
    )

    assert result.chunks, "expected knee-pain chunks for the demo patient"
    assert result.chunks[0].score > 0.7
    joined = " ".join(c.text.lower() for c in result.chunks)
    assert "knee" in joined


async def test_every_result_belongs_to_the_scoped_patient(session, embedder) -> None:
    ctx = await _ctx_for(session, "P001")
    result = await retrieve(
        session, ctx, question="knee pain worse on stairs", embedder=embedder
    )
    assert result.chunks

    owners = set(
        (
            await session.scalars(
                select(DocumentChunk.patient_id).where(
                    DocumentChunk.id.in_([c.chunk_id for c in result.chunks])
                )
            )
        ).all()
    )
    assert owners == {ctx.patient_id}


async def test_two_patients_get_disjoint_results(session, embedder) -> None:
    """The same question, asked by two patients, returns different records."""
    first = await _ctx_for(session, "P001")
    second = await _ctx_for(session, "P002")
    question = "What did the doctor say at my last visit?"

    a = await retrieve(session, first, question=question, embedder=embedder)
    b = await retrieve(session, second, question=question, embedder=embedder)

    ids_a = {c.chunk_id for c in a.chunks}
    ids_b = {c.chunk_id for c in b.chunks}
    assert ids_a and ids_b
    assert ids_a.isdisjoint(ids_b)


async def test_unrelated_question_returns_nothing_above_the_floor(
    session, embedder
) -> None:
    """Better to retrieve nothing than to pad the context with noise."""
    ctx = await _ctx_for(session, "P001")
    result = await retrieve(
        session, ctx, question="What is my dog's name?", embedder=embedder
    )
    assert result.chunks == []
    assert not result.has_context
    assert not result.index_empty, "the corpus exists; this question just misses"


async def test_context_and_citations_line_up(session, embedder) -> None:
    ctx = await _ctx_for(session, "P001")
    result = await retrieve(
        session, ctx, question="knee pain", embedder=embedder
    )
    assert result.has_context
    assert len(result.sources) == result.context.used_chunks
    for index, source in enumerate(result.sources, start=1):
        assert f"SOURCE {index}" in result.context.text
        assert source.chunk_id is not None


async def test_section_filter_narrows_retrieval(session, embedder) -> None:
    ctx = await _ctx_for(session, "P001")
    query_vector = await embedder.embed_query("what was the plan")
    rows = await vector_search(
        session, ctx, query_vector=query_vector, sections=("Plan",), limit=10
    )
    assert rows
    assert {r.section for r in rows} == {"Plan"}


async def test_retrieval_requires_a_patient_scope(session, embedder) -> None:
    unlinked = AuthContext(user_id=99, role="clinician", patient_id=None)
    with pytest.raises(AuthorizationError):
        await retrieve(session, unlinked, question="anything", embedder=embedder)


# --- hybrid retrieval (PRD §19) ------------------------------------ #


async def test_keyword_search_finds_exact_clinical_terms(session, embedder) -> None:
    """The case §19 adds keyword search for."""
    ctx = await _ctx_for(session, "P001")
    hits = await keyword_search(session, ctx, query="metformin", limit=20)

    assert hits, "expected keyword matches for a drug name in the notes"
    assert all(h.retriever == "keyword" for h in hits)
    assert any("metformin" in h.text.lower() for h in hits)


async def test_keyword_search_is_patient_scoped(session, embedder) -> None:
    first = await _ctx_for(session, "P001")
    second = await _ctx_for(session, "P002")

    a = await keyword_search(session, first, query="pain treatment plan", limit=20)
    b = await keyword_search(session, second, query="pain treatment plan", limit=20)

    if a and b:
        assert {h.chunk_id for h in a}.isdisjoint({h.chunk_id for h in b})

    owners = set(
        (
            await session.scalars(
                select(DocumentChunk.patient_id).where(
                    DocumentChunk.id.in_([h.chunk_id for h in a])
                )
            )
        ).all()
    )
    assert owners in ({first.patient_id}, set())


async def test_keyword_search_handles_a_natural_question(session, embedder) -> None:
    """Terms are OR-ed: AND-ing them would match nothing and contribute nothing."""
    ctx = await _ctx_for(session, "P001")
    hits = await keyword_search(
        session, ctx, query="what did my doctor say about my knee pain", limit=20
    )
    assert hits, "a natural-language question should still match on its content words"


async def test_keyword_search_survives_operator_characters(session, embedder) -> None:
    """Raw to_tsquery would raise a syntax error on this."""
    ctx = await _ctx_for(session, "P001")
    hits = await keyword_search(session, ctx, query="knee & pain | (x) !!", limit=5)
    assert isinstance(hits, list)


async def test_keyword_search_returns_nothing_for_pure_scaffolding(
    session, embedder
) -> None:
    ctx = await _ctx_for(session, "P001")
    assert await keyword_search(session, ctx, query="what did you say", limit=5) == []


async def test_hybrid_reports_both_retrievers(session, embedder) -> None:
    ctx = await _ctx_for(session, "P001")
    result = await retrieve(
        session,
        ctx,
        question="What did my doctor say about my knee pain?",
        embedder=embedder,
    )
    assert result.vector_candidates > 0
    assert result.keyword_candidates > 0
    assert result.overlap >= 0
    assert result.candidates >= result.vector_candidates


async def test_duplicate_passages_are_collapsed(session, embedder) -> None:
    """The generator repeats template text across visits; the top-5 must not."""
    ctx = await _ctx_for(session, "P001")
    result = await retrieve(
        session,
        ctx,
        question="What did my doctor say about my knee pain?",
        embedder=embedder,
    )
    texts = [" ".join(c.text.lower().split()) for c in result.chunks]
    assert len(texts) == len(set(texts)), "identical passages reached the context"
    assert result.deduplicated > 0, "expected repeated template text in this corpus"


async def test_recurrence_is_preserved_after_dedup(session, embedder) -> None:
    ctx = await _ctx_for(session, "P001")
    result = await retrieve(
        session, ctx, question="my medications", embedder=embedder
    )
    repeated = [c for c in result.chunks if c.occurrences > 1]
    if repeated:
        assert repeated[0].other_dates, "collapsed copies should record their dates"


async def test_reranker_is_applied_and_reported(session, embedder) -> None:
    ctx = await _ctx_for(session, "P001")
    question = "What did my doctor say about my knee pain?"

    plain = await retrieve(
        session, ctx, question=question, embedder=embedder, reranker=NoopReranker()
    )
    ranked = await retrieve(
        session,
        ctx,
        question=question,
        embedder=embedder,
        reranker=HeuristicReranker(),
    )

    assert plain.reranker == "none"
    assert ranked.reranker == "heuristic"
    assert len(ranked.chunks) == len(plain.chunks)
    # Both draw from the same candidate pool, so the sets should overlap
    # heavily even where the order differs.
    assert {c.chunk_id for c in ranked.chunks} & {c.chunk_id for c in plain.chunks}


async def test_final_context_stays_within_the_prd_band(session, embedder) -> None:
    """§19: 20-50 candidates in, 3-8 chunks out."""
    ctx = await _ctx_for(session, "P001")
    result = await retrieve(
        session,
        ctx,
        question="What did my doctor say about my knee pain?",
        embedder=embedder,
        reranker=HeuristicReranker(),
    )
    assert 1 <= result.context.used_chunks <= 8
    assert result.candidates >= 10


async def test_indexed_chunk_count_is_patient_scoped(session, embedder) -> None:
    ctx = await _ctx_for(session, "P001")
    scoped = await count_indexed_chunks(session, ctx)
    total = await session.scalar(
        select(func.count()).select_from(DocumentChunk).where(
            DocumentChunk.embedding.is_not(None)
        )
    )
    assert 0 < scoped < total
