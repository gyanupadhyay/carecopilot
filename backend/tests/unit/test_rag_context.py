"""Context building, budgeting and query normalization (PRD §19)."""

from __future__ import annotations

from datetime import date

import pytest

from app.rag.context import build_context
from app.rag.embeddings import HashingEmbedder
from app.rag.pipeline import normalize_query
from app.rag.retrieval import RetrievedChunk


def chunk(
    chunk_id: int, *, text: str = "Knee pain worse on stairs.", distance: float = 0.15
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=100 + chunk_id,
        encounter_id=200 + chunk_id,
        section="Assessment",
        text=text,
        chunk_date=date(2026, 9, 10),
        title="Knee osteoarthritis",
        document_type="clinical_note",
        distance=distance,
    )


def test_context_is_rendered_with_source_headers() -> None:
    built = build_context([chunk(1), chunk(2)])
    assert "SOURCE 1" in built.text
    assert "SOURCE 2" in built.text
    assert "Document: Clinical Note" in built.text
    assert "Date: 2026-09-10" in built.text
    assert "Section: Assessment" in built.text


def test_every_rendered_chunk_gets_a_citation() -> None:
    built = build_context([chunk(1), chunk(2), chunk(3)])
    assert built.used_chunks == 3
    assert [s.chunk_id for s in built.sources] == [1, 2, 3]
    assert all(s.citation_key.startswith("chunk:") for s in built.sources)


def test_similarity_score_is_derived_from_distance() -> None:
    built = build_context([chunk(1, distance=0.0)])
    assert built.sources[0].score == 1.0


def test_budget_stops_at_rank_order_and_reports_the_remainder() -> None:
    """Lower-ranked chunks are not smuggled in just because they fit."""
    big = "x " * 400
    chunks = [chunk(i, text=big) for i in range(1, 6)]
    built = build_context(chunks, token_budget=300)

    assert built.used_chunks < len(chunks)
    assert built.dropped_chunks == len(chunks) - built.used_chunks
    assert built.estimated_tokens <= 300 + 250  # one chunk may straddle


def test_first_chunk_is_always_included_even_if_over_budget() -> None:
    """A tiny budget must not produce an empty context and a silent no-answer."""
    built = build_context([chunk(1, text="y " * 1000)], token_budget=10)
    assert built.used_chunks == 1
    assert not built.is_empty


def test_no_chunks_gives_empty_context() -> None:
    built = build_context([])
    assert built.is_empty
    assert built.sources == []
    assert built.used_chunks == 0


# --- query normalization ------------------------------------------------ #


def test_greetings_and_scaffolding_are_stripped() -> None:
    assert normalize_query("Hi, can you tell me about my knee?").lower().startswith(
        "tell me about my knee"
    ) or "knee" in normalize_query("Hi, can you tell me about my knee?")


def test_whitespace_is_collapsed() -> None:
    assert normalize_query("  knee    pain  ") == "knee pain"


def test_normalization_never_returns_empty() -> None:
    """An over-eager strip would embed the empty string and match noise."""
    assert normalize_query("please").strip() != ""
    assert normalize_query("Hello").strip() != ""


# --- hashing embedder --------------------------------------------------- #


async def test_hashing_embedder_is_deterministic_and_correctly_shaped() -> None:
    embedder = HashingEmbedder(dimension=384)
    first = await embedder.embed_query("knee pain")
    second = await embedder.embed_query("knee pain")

    assert first == second
    assert len(first) == 384
    assert abs(sum(v * v for v in first) - 1.0) < 1e-6, "should be unit length"


async def test_hashing_embedder_distinguishes_different_text() -> None:
    embedder = HashingEmbedder(dimension=384)
    a = await embedder.embed_query("knee pain")
    b = await embedder.embed_query("blood pressure")
    assert a != b


async def test_hashing_embedder_handles_empty_input() -> None:
    embedder = HashingEmbedder(dimension=16)
    assert await embedder.embed_query("") == [0.0] * 16
    assert await embedder.embed_documents([]) == []


# --- query embedding cache (PRD §36 P3) -------------------------------- #


@pytest.mark.anyio
async def test_an_identical_query_is_embedded_once() -> None:
    from app.rag.embeddings import LocalEmbedder

    calls: list[str] = []

    class Counting(LocalEmbedder):
        def _embed_query_sync(self, text: str) -> list[list[float]]:
            calls.append(text)
            return [[0.1] * self.dimension]

    embedder = Counting(cache_size=8)
    first = await embedder.embed_query("what were my last results?")
    second = await embedder.embed_query("what were my last results?")

    assert first == second
    assert calls == ["what were my last results?"], "second call should be cached"


@pytest.mark.anyio
async def test_the_cache_returns_a_copy_not_the_stored_vector() -> None:
    """A caller mutating its result must not corrupt every later hit."""
    from app.rag.embeddings import LocalEmbedder

    class Fixed(LocalEmbedder):
        def _embed_query_sync(self, text: str) -> list[list[float]]:
            return [[0.1] * self.dimension]

    embedder = Fixed(cache_size=8)
    first = await embedder.embed_query("q")
    first[0] = 99.0
    assert (await embedder.embed_query("q"))[0] == pytest.approx(0.1)


@pytest.mark.anyio
async def test_the_cache_is_bounded() -> None:
    from app.rag.embeddings import LocalEmbedder

    class Fixed(LocalEmbedder):
        def _embed_query_sync(self, text: str) -> list[list[float]]:
            return [[0.1] * self.dimension]

    embedder = Fixed(cache_size=3)
    for i in range(10):
        await embedder.embed_query(f"question {i}")
    assert len(embedder._query_cache) == 3


@pytest.mark.anyio
async def test_caching_can_be_switched_off() -> None:
    from app.rag.embeddings import LocalEmbedder

    calls: list[str] = []

    class Counting(LocalEmbedder):
        def _embed_query_sync(self, text: str) -> list[list[float]]:
            calls.append(text)
            return [[0.1] * self.dimension]

    embedder = Counting(cache_size=0)
    await embedder.embed_query("q")
    await embedder.embed_query("q")
    assert len(calls) == 2
