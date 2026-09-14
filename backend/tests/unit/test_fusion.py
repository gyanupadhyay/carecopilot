"""Rank fusion and deduplication (PRD §19)."""

from __future__ import annotations

from datetime import date

from app.rag.fusion import deduplicate, fuse, reciprocal_rank_fusion
from app.rag.retrieval import RetrievedChunk, build_tsquery_terms


def chunk(
    chunk_id: int,
    *,
    text: str | None = None,
    distance: float = 0.2,
    retriever: str = "vector",
    when: date | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=100 + chunk_id,
        encounter_id=200 + chunk_id,
        section="Assessment",
        text=text if text is not None else f"passage {chunk_id}",
        chunk_date=when or date(2026, 9, 10),
        title="Clinical note",
        document_type="clinical_note",
        distance=distance,
        retriever=retriever,
    )


# --- reciprocal rank fusion --------------------------------------------- #


def test_a_single_list_keeps_its_order() -> None:
    ranked = [chunk(1), chunk(2), chunk(3)]
    assert [c.chunk_id for c in reciprocal_rank_fusion(ranked)] == [1, 2, 3]


def test_agreement_between_retrievers_outranks_either_alone() -> None:
    """The whole point of fusing: two methods agreeing is strong evidence."""
    vector = [chunk(1), chunk(2, retriever="vector")]
    keyword = [chunk(3, retriever="keyword"), chunk(2, retriever="keyword")]

    fused = reciprocal_rank_fusion(vector, keyword)
    assert fused[0].chunk_id == 2, "found by both, so it should lead"


def test_chunks_found_by_both_are_labelled_hybrid() -> None:
    fused = reciprocal_rank_fusion(
        [chunk(1)], [chunk(1, retriever="keyword")]
    )
    assert fused[0].retriever == "hybrid"


def test_single_retriever_hits_keep_their_label() -> None:
    fused = reciprocal_rank_fusion(
        [chunk(1)], [chunk(2, retriever="keyword")]
    )
    labels = {c.chunk_id: c.retriever for c in fused}
    assert labels == {1: "vector", 2: "keyword"}


def test_the_vector_copy_survives_so_the_score_is_meaningful() -> None:
    """A keyword hit has no cosine distance; the fused chunk should."""
    fused = reciprocal_rank_fusion(
        [chunk(1, distance=0.1)],
        [chunk(1, distance=1.0, retriever="keyword")],
    )
    assert fused[0].score > 0.5


def test_fusion_is_deterministic() -> None:
    """Evaluation runs must be reproducible."""
    vector = [chunk(1), chunk(2), chunk(3)]
    keyword = [chunk(3, retriever="keyword"), chunk(4, retriever="keyword")]
    first = [c.chunk_id for c in reciprocal_rank_fusion(vector, keyword)]
    second = [c.chunk_id for c in reciprocal_rank_fusion(vector, keyword)]
    assert first == second


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion([], []) == []


# --- deduplication ------------------------------------------------------- #


def test_identical_passages_collapse_to_one() -> None:
    repeated = "Patient describes knee pain worse when climbing stairs."
    kept, removed = deduplicate(
        [
            chunk(1, text=repeated, when=date(2026, 9, 10)),
            chunk(2, text=repeated, when=date(2026, 4, 6)),
            chunk(3, text="Something else entirely."),
        ]
    )
    assert [c.chunk_id for c in kept] == [1, 3]
    assert removed == 1


def test_the_highest_ranked_copy_survives() -> None:
    repeated = "same text"
    kept, _ = deduplicate([chunk(7, text=repeated), chunk(2, text=repeated)])
    assert kept[0].chunk_id == 7, "order in equals order out"


def test_recurrence_is_recorded_not_discarded() -> None:
    """Collapsing saves context; the fact it recurred is still information."""
    repeated = "Physical therapy referral placed."
    kept, _ = deduplicate(
        [
            chunk(1, text=repeated, when=date(2026, 9, 10)),
            chunk(2, text=repeated, when=date(2026, 4, 6)),
            chunk(3, text=repeated, when=date(2025, 11, 17)),
        ]
    )
    assert kept[0].occurrences == 3
    assert set(kept[0].other_dates) == {date(2026, 4, 6), date(2025, 11, 17)}


def test_deduplication_ignores_whitespace_and_case() -> None:
    kept, removed = deduplicate(
        [chunk(1, text="Knee pain."), chunk(2, text="  knee   PAIN.  ")]
    )
    assert len(kept) == 1
    assert removed == 1


def test_distinct_passages_are_untouched() -> None:
    kept, removed = deduplicate([chunk(1), chunk(2), chunk(3)])
    assert len(kept) == 3
    assert removed == 0


# --- fuse() -------------------------------------------------------------- #


def test_fuse_reports_how_the_candidates_were_assembled() -> None:
    vector = [chunk(1), chunk(2)]
    keyword = [chunk(2, retriever="keyword"), chunk(3, retriever="keyword")]

    result = fuse(vector, keyword)
    assert result.vector_count == 2
    assert result.keyword_count == 2
    assert result.overlap_count == 1
    assert {c.chunk_id for c in result.chunks} == {1, 2, 3}


def test_fuse_runs_before_dedup_so_the_best_copy_wins() -> None:
    """A passage both retrievers found should survive deduplication."""
    repeated = "identical text"
    vector = [chunk(1, text="unrelated"), chunk(2, text=repeated)]
    keyword = [chunk(2, text=repeated, retriever="keyword"), chunk(3, text=repeated)]

    result = fuse(vector, keyword)
    survivors = {c.chunk_id for c in result.chunks}
    assert 2 in survivors, "the doubly-found copy should be the survivor"
    assert result.deduplicated == 1


# --- tsquery term extraction --------------------------------------------- #


def test_question_scaffolding_is_stripped() -> None:
    terms = build_tsquery_terms("What did my doctor say about my knee pain?")
    assert "knee" in terms and "pain" in terms
    assert "what" not in terms and "my" not in terms


def test_clinical_terms_survive_intact() -> None:
    terms = build_tsquery_terms("Was my HbA1c or LDL above target on metformin?")
    assert {"hba1c", "ldl", "metformin"}.issubset(set(terms))


def test_only_alphanumeric_tokens_are_produced() -> None:
    """Nothing reaching to_tsquery can contain an operator."""
    terms = build_tsquery_terms("knee & pain | (injection) !! 'quoted'")
    assert all(term.isalnum() for term in terms)


def test_a_query_of_pure_scaffolding_yields_nothing() -> None:
    assert build_tsquery_terms("what did you say about me") == []
