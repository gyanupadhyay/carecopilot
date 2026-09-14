"""Reranking (PRD §19)."""

from __future__ import annotations

from datetime import date

import pytest

from app.llm.base import StructuredResponse, TokenUsage
from app.llm.errors import LLMServiceError
from app.llm.stub import StubProvider
from app.rag.reranking import (
    HeuristicReranker,
    LLMReranker,
    NoopReranker,
    _Ranking,
    _RankingList,
    _rerank_token_budget,
    build_reranker,
)
from app.rag.retrieval import RetrievedChunk

TODAY = date(2026, 9, 13)


def chunk(
    chunk_id: int,
    *,
    text: str = "passage",
    section: str = "History of Present Illness",
    distance: float = 0.25,
    when: date | None = None,
    retriever: str = "vector",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=100 + chunk_id,
        encounter_id=200 + chunk_id,
        section=section,
        text=text,
        chunk_date=when or date(2026, 9, 10),
        title="Clinical note",
        document_type="clinical_note",
        distance=distance,
        retriever=retriever,
    )


class CannedReranker(StubProvider):
    def __init__(self, rankings: list[_Ranking]) -> None:
        super().__init__()
        self._rankings = rankings
        self.calls = 0

    async def generate_structured(self, *, messages, system, schema, **kwargs):  # type: ignore[override]
        self.calls += 1
        return StructuredResponse(
            value=_RankingList(rankings=self._rankings),
            model="fake-reranker",
            usage=TokenUsage(input_tokens=100, output_tokens=20),
            latency_ms=1,
            provider="fake",
        )


class BrokenReranker(StubProvider):
    async def generate_structured(self, **kwargs):  # type: ignore[override]
        raise LLMServiceError("reranker down", provider="fake")


class BudgetSpy(CannedReranker):
    """Records the output cap the reranker asked for."""

    def __init__(self) -> None:
        super().__init__([_Ranking(index=1, relevance=1.0)])
        self.max_tokens: int | None = None

    async def generate_structured(self, *, messages, system, schema, **kwargs):  # type: ignore[override]
        self.max_tokens = kwargs.get("max_tokens")
        return await super().generate_structured(
            messages=messages, system=system, schema=schema, **kwargs
        )


# --- output budget --------------------------------------------------------- #


def test_rerank_budget_covers_measured_output_cost() -> None:
    """Regression: the eval run found the LLM reranker never actually ran.

    20 documents genuinely cost 505 output tokens on a model that
    pretty-prints its JSON. The previous 16-per-document budget allowed
    448, so the response truncated mid-object, failed to parse, and fell
    back to the heuristic — visible only as a log line.
    """
    assert _rerank_token_budget(20) >= 505


def test_rerank_budget_scales_with_candidate_count() -> None:
    assert _rerank_token_budget(40) > _rerank_token_budget(20)
    assert _rerank_token_budget(0) > 0


async def test_reranker_requests_the_computed_budget() -> None:
    spy = BudgetSpy()
    docs = [chunk(i) for i in range(1, 9)]
    await LLMReranker(spy).rerank("knee pain", docs, top_k=3)
    assert spy.max_tokens == _rerank_token_budget(len(docs))


# --- noop ---------------------------------------------------------------- #


async def test_noop_preserves_order_and_truncates() -> None:
    docs = [chunk(i) for i in range(1, 6)]
    out = await NoopReranker().rerank("anything", docs, top_k=3)
    assert [c.chunk_id for c in out] == [1, 2, 3]


# --- heuristic ------------------------------------------------------------ #


async def test_term_overlap_lifts_the_passage_that_names_the_thing() -> None:
    """The case embeddings are worst at: a specific named drug."""
    docs = [
        chunk(1, text="General advice about diet and exercise.", distance=0.20),
        chunk(2, text="Metformin increased from 500mg to 1000mg.", distance=0.24),
    ]
    out = await HeuristicReranker(today=TODAY).rerank("metformin dose", docs, top_k=2)
    assert out[0].chunk_id == 2


async def test_answer_bearing_sections_are_preferred_on_a_tie() -> None:
    docs = [
        chunk(1, text="knee pain", section="History of Present Illness"),
        chunk(2, text="knee pain", section="Assessment"),
    ]
    out = await HeuristicReranker(today=TODAY).rerank("knee pain", docs, top_k=2)
    assert out[0].section == "Assessment"


async def test_recent_notes_win_when_everything_else_is_equal() -> None:
    docs = [
        chunk(1, text="knee pain", when=date(2021, 1, 1)),
        chunk(2, text="knee pain", when=date(2026, 9, 1)),
    ]
    out = await HeuristicReranker(today=TODAY).rerank("knee pain", docs, top_k=2)
    assert out[0].chunk_id == 2


async def test_agreement_between_retrievers_is_rewarded() -> None:
    docs = [
        chunk(1, text="knee pain", retriever="vector"),
        chunk(2, text="knee pain", retriever="hybrid"),
    ]
    out = await HeuristicReranker(today=TODAY).rerank("knee pain", docs, top_k=2)
    assert out[0].chunk_id == 2


async def test_retrieval_score_still_dominates() -> None:
    """The heuristics adjust within a band; they do not override retrieval."""
    docs = [
        chunk(1, text="unrelated text", distance=0.05, section="Plan"),
        chunk(2, text="knee pain", distance=0.60, section="History"),
    ]
    out = await HeuristicReranker(today=TODAY).rerank("knee pain", docs, top_k=2)
    assert out[0].chunk_id == 1


async def test_heuristic_is_deterministic() -> None:
    docs = [chunk(i, text="knee pain") for i in range(1, 6)]
    ranker = HeuristicReranker(today=TODAY)
    first = [c.chunk_id for c in await ranker.rerank("knee", docs, top_k=3)]
    shuffled = list(reversed(docs))
    second = [c.chunk_id for c in await ranker.rerank("knee", shuffled, top_k=3)]
    assert first == second


# --- llm ------------------------------------------------------------------ #


async def test_llm_ordering_is_applied() -> None:
    docs = [chunk(1), chunk(2), chunk(3), chunk(4)]
    llm = CannedReranker(
        [
            _Ranking(index=1, relevance=0.1),
            _Ranking(index=2, relevance=0.9),
            _Ranking(index=3, relevance=0.5),
            _Ranking(index=4, relevance=0.2),
        ]
    )
    out = await LLMReranker(llm).rerank("q", docs, top_k=2)
    assert [c.chunk_id for c in out] == [2, 3]


async def test_llm_is_not_called_when_nothing_needs_choosing() -> None:
    """Ranking a list already short enough spends a call on the inevitable."""
    docs = [chunk(1), chunk(2)]
    llm = CannedReranker([])
    out = await LLMReranker(llm).rerank("q", docs, top_k=5)
    assert llm.calls == 0
    assert len(out) == 2


async def test_llm_failure_falls_back_to_the_heuristic() -> None:
    docs = [chunk(i, text="knee pain") for i in range(1, 6)]
    out = await LLMReranker(BrokenReranker()).rerank("knee pain", docs, top_k=3)
    assert len(out) == 3, "a failed reranker degrades order, never the answer"


async def test_an_empty_ranking_falls_back() -> None:
    docs = [chunk(i) for i in range(1, 6)]
    out = await LLMReranker(CannedReranker([])).rerank("q", docs, top_k=3)
    assert len(out) == 3


async def test_unscored_candidates_keep_their_retrieval_score() -> None:
    """A partial response must not silently drop what it did not look at.

    Four candidates for a top-3 so the reranker actually runs — with as many
    documents as slots it returns them untouched, by design.
    """
    docs = [
        chunk(1, distance=0.5),
        chunk(2, distance=0.1),
        chunk(3, distance=0.4),
        chunk(4, distance=0.45),
    ]
    llm = CannedReranker([_Ranking(index=1, relevance=0.05)])
    out = await LLMReranker(llm).rerank("q", docs, top_k=3)

    assert len(out) == 3
    assert out[0].chunk_id == 2, "unscored, so ranked by retrieval score"
    assert 1 not in [c.chunk_id for c in out], "explicitly scored lowest, so cut"


async def test_out_of_range_indexes_are_ignored() -> None:
    docs = [chunk(1), chunk(2), chunk(3)]
    llm = CannedReranker(
        [_Ranking(index=99, relevance=1.0), _Ranking(index=2, relevance=0.9)]
    )
    out = await LLMReranker(llm).rerank("q", docs, top_k=2)
    assert out[0].chunk_id == 2


# --- factory --------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("none", "none"), ("heuristic", "heuristic"), ("llm", "llm")],
)
def test_factory_builds_the_configured_reranker(kind: str, expected: str) -> None:
    assert build_reranker(StubProvider(), kind=kind).name == expected


def test_llm_reranker_without_a_provider_degrades() -> None:
    assert build_reranker(None, kind="llm").name == "heuristic"


def test_unknown_kind_degrades_rather_than_raising() -> None:
    assert build_reranker(StubProvider(), kind="nonsense").name == "heuristic"
