"""Reranking the fused candidates down to a final 3-8 (PRD §19).

Retrieval optimizes for recall — cast wide, 20 candidates from each side.
Reranking optimizes for precision: of those, which few actually answer *this*
question. The two jobs want different models, which is why they are separate
stages rather than one better retriever.

Three implementations behind one interface, chosen by ``RERANKER``:

``none``       keep fusion order. Honest baseline, and what the evaluation
               set should be compared against before claiming a reranker
               earned its latency.
``heuristic``  term overlap, section prior and recency. No model call, so
               no latency and no cost, and entirely deterministic.
``llm``        ask the model to score each candidate. Best quality, one
               extra round trip, and the only one that can be wrong in
               interesting ways.

The abstraction exists because §19 asks for it, but also because the right
choice here is an empirical question this project has not yet answered — the
evaluation set is what decides it, and swapping implementations must not
mean touching the pipeline.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import date

from pydantic import BaseModel, Field

from app.config import settings
from app.llm.base import ChatMessage, Effort, LLMProvider
from app.llm.errors import LLMError
from app.observability.logging import get_logger
from app.rag.retrieval import RetrievedChunk

log = get_logger(__name__)

RERANK_EFFORT: Effort = "low"

#: Sections that tend to carry the answer rather than the background. A
#: mild prior, not a filter: a question about symptoms is answered by the
#: history, and this must not bury it.
SECTION_PRIOR: dict[str, float] = {
    "Assessment": 0.10,
    "Plan": 0.10,
    "Impression": 0.10,
    "Medications": 0.05,
    "Chief Complaint": 0.03,
}

_WORD = re.compile(r"[a-z0-9]+")
#: Words carrying no retrieval signal. Deliberately short — an aggressive
#: stop list removes terms like "no" and "not" that change clinical meaning.
_STOP = frozenset(
    {
        "a", "about", "am", "and", "any", "are", "as", "at", "be", "did",
        "do", "does", "for", "from", "had", "has", "have", "i", "in", "is",
        "it", "me", "my", "of", "on", "or", "that", "the", "to", "was",
        "were", "what", "when", "which", "who", "why", "with", "you", "your",
    }
)


def _terms(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _STOP}


class Reranker(ABC):
    """Reorder candidates by relevance to the query, best first."""

    name: str = "unknown"

    @abstractmethod
    async def rerank(
        self, query: str, documents: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        """Return at most ``top_k`` documents, most relevant first."""


class NoopReranker(Reranker):
    """Keep fusion order. The baseline everything else must beat."""

    name = "none"

    async def rerank(
        self, query: str, documents: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        return documents[:top_k]


class HeuristicReranker(Reranker):
    """Cheap, deterministic signals — no model call.

    Three of them, each with a reason to exist:

    *Term overlap* catches the case embeddings are worst at, where the
    question names a specific thing ("metformin") and the passage either
    contains that word or does not.

    *Section prior* encodes that an Assessment or Plan usually states a
    conclusion, while History states background.

    *Recency* breaks ties toward the present, because "what did my doctor
    say" almost always means the most recent time they said it.

    The retrieval score stays dominant; these adjust within a band rather
    than overriding it.
    """

    name = "heuristic"

    def __init__(self, *, today: date | None = None) -> None:
        self._today = today or date.today()

    def _score(self, query_terms: set[str], chunk: RetrievedChunk) -> float:
        base = chunk.score

        chunk_terms = _terms(chunk.text)
        overlap = (
            len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0.0
        )

        prior = SECTION_PRIOR.get(chunk.section, 0.0)

        recency = 0.0
        if chunk.chunk_date is not None:
            age_days = (self._today - chunk.chunk_date).days
            # Full credit inside a year, decaying to nothing at three.
            recency = max(0.0, 1.0 - max(0, age_days - 365) / 730) * 0.08

        # A passage both retrievers surfaced is a stronger candidate than
        # one either found alone.
        agreement = 0.05 if chunk.retriever == "hybrid" else 0.0

        return base + overlap * 0.20 + prior + recency + agreement

    async def rerank(
        self, query: str, documents: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        query_terms = _terms(query)
        ordered = sorted(
            documents,
            key=lambda chunk: (-self._score(query_terms, chunk), chunk.chunk_id),
        )
        return ordered[:top_k]


class _Ranking(BaseModel):
    """The model's verdict on one candidate."""

    index: int = Field(description="The SOURCE number being scored, 1-based.")
    relevance: float = Field(
        ge=0.0, le=1.0, description="0 = irrelevant, 1 = directly answers the question."
    )


class _RankingList(BaseModel):
    rankings: list[_Ranking] = Field(default_factory=list)


RERANK_SYSTEM_PROMPT = """
You score how well each numbered passage answers a patient's question about
their own medical record. You do not answer the question and you do not
summarize the passages.

Score each passage from 0 to 1:
  1.0  directly answers the question
  0.5  related and useful context, but not the answer
  0.0  unrelated to the question

Judge only relevance to the question as asked. A passage that is clinically
interesting but does not address the question scores low. Return one entry
per passage, using the SOURCE number as the index.
""".strip()


def _rerank_token_budget(document_count: int) -> int:
    """Output cap for one reranking call.

    Sized from a measurement, not a guess: 20 documents cost 505 output
    tokens on Gemini flash-lite, which pretty-prints its JSON with newlines
    and two-space indent. An earlier 16-per-document budget was tuned
    against compact output and truncated that response mid-object — and
    because a failed rerank falls back to the heuristic rather than
    erroring, the only symptom was a `rerank.failed` line in the log and an
    LLM reranker that silently never ran.

    32 per document plus a fixed 256 leaves roughly 75% headroom at the
    candidate counts this system uses. The cost of being generous is
    nothing: max_tokens is a cap, and billing follows tokens produced.
    """
    return 32 * document_count + 256


class LLMReranker(Reranker):
    """Ask the model to score each candidate.

    One extra round trip, at low effort and a small output cap — the task is
    scoring, not reasoning. Any failure falls back to the heuristic reranker
    rather than propagating: a reranker that cannot rank should degrade the
    ordering, never the answer.
    """

    name = "llm"

    def __init__(self, llm: LLMProvider, *, fallback: Reranker | None = None) -> None:
        self._llm = llm
        self._fallback = fallback or HeuristicReranker()

    async def rerank(
        self, query: str, documents: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        if len(documents) <= top_k:
            # Nothing to choose between — ranking a list that is already
            # short enough spends a model call to reorder the inevitable.
            return documents

        listing = "\n\n".join(
            f"SOURCE {position}\nSection: {chunk.section}\n{chunk.text}"
            for position, chunk in enumerate(documents, start=1)
        )
        payload = f"QUESTION\n{query}\n\nPASSAGES\n{listing}"

        try:
            result = await self._llm.generate_structured(
                messages=[ChatMessage(role="user", content=payload)],
                system=RERANK_SYSTEM_PROMPT,
                schema=_RankingList,
                max_tokens=_rerank_token_budget(len(documents)),
                effort=RERANK_EFFORT,
                model=settings.router_model,
            )
        except LLMError as exc:
            log.warning(
                "rerank.failed", error=type(exc).__name__, fallback=self._fallback.name
            )
            return await self._fallback.rerank(query, documents, top_k=top_k)

        scored: dict[int, float] = {}
        for ranking in result.value.rankings:
            position = ranking.index - 1
            if 0 <= position < len(documents):
                scored[position] = ranking.relevance

        if not scored:
            log.warning("rerank.empty", fallback=self._fallback.name)
            return await self._fallback.rerank(query, documents, top_k=top_k)

        # Candidates the model skipped keep their retrieval score, so an
        # incomplete response degrades the ordering rather than dropping
        # passages it never looked at.
        ordered = sorted(
            range(len(documents)),
            key=lambda i: (-scored.get(i, documents[i].score), i),
        )
        return [documents[i] for i in ordered[:top_k]]


def build_reranker(
    llm: LLMProvider | None = None, *, kind: str | None = None
) -> Reranker:
    name = (kind or settings.reranker).lower()
    if name == "none":
        return NoopReranker()
    if name == "heuristic":
        return HeuristicReranker()
    if name == "llm":
        if llm is None:
            log.warning("rerank.no_provider", fallback="heuristic")
            return HeuristicReranker()
        return LLMReranker(llm)
    log.warning("rerank.unknown_kind", kind=name, fallback="heuristic")
    return HeuristicReranker()
