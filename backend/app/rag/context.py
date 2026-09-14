"""Turning retrieved chunks into prompt context (PRD §19).

Two responsibilities, and the second is the one that matters.

*Format.* Numbered ``SOURCE`` blocks carrying document type, date and
section, so the model can refer to "SOURCE 2" and the answer can be traced
back to a row.

*Budget.* The context is capped in tokens. PRD §19 is explicit that the
patient's entire history must not be sent on every query, and the reason is
not only cost: a model given twenty marginally relevant passages answers
worse than one given the three that matter. Chunks are added in rank order
until the budget is reached, and the ones that do not fit are reported
rather than silently dropped, so the caller can put the count on the trace.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.rag.retrieval import RetrievedChunk
from app.schemas.chat import Source

#: Four characters per token. An estimate, deliberately: calling the token
#: counting endpoint per chunk would add a network round trip to every
#: retrieval to save a few percent of a budget that is already a heuristic.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class BuiltContext:
    text: str
    sources: list[Source]
    used_chunks: int
    dropped_chunks: int
    estimated_tokens: int

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def render_source(index: int, chunk: RetrievedChunk) -> str:
    """One passage as the model reads it: header, blank line, text.

    Public because the faithfulness judge has to score an answer against the
    *same bytes the model saw*. Given the bare chunk text instead, it marks
    every date the answer correctly quoted as unsupported — the header is
    where the date lives. A private copy of this format in the evaluation
    package would drift the first time the header changed, and the symptom
    would be a faithfulness score that dropped for no reason anyone could
    find in the model.
    """
    fields = [f"SOURCE {index}"]
    if chunk.document_type:
        fields.append(f"Document: {chunk.document_type.replace('_', ' ').title()}")
    if chunk.chunk_date:
        fields.append(f"Date: {chunk.chunk_date.isoformat()}")
    if chunk.section:
        fields.append(f"Section: {chunk.section}")
    header = "\n".join(fields)
    return f"{header}\n\n{chunk.text}"


def build_context(
    chunks: list[RetrievedChunk], *, token_budget: int | None = None
) -> BuiltContext:
    """Render ranked chunks into a bounded context block with citations."""
    budget = token_budget or settings.rag_context_token_budget
    blocks: list[str] = []
    sources: list[Source] = []
    used_tokens = 0
    dropped = 0

    for chunk in chunks:
        block = render_source(len(blocks) + 1, chunk)
        cost = max(1, len(block) // CHARS_PER_TOKEN)
        if used_tokens + cost > budget and blocks:
            # Rank order is meaningful, so stop rather than skipping ahead to
            # a smaller, lower-ranked chunk that happens to fit.
            dropped = len(chunks) - len(blocks)
            break
        blocks.append(block)
        used_tokens += cost
        sources.append(
            Source(
                document_id=chunk.document_id,
                encounter_id=chunk.encounter_id,
                chunk_id=chunk.chunk_id,
                document_type=chunk.document_type,
                title=chunk.title,
                section=chunk.section,
                date=chunk.chunk_date,
                score=round(chunk.score, 4),
            )
        )

    return BuiltContext(
        text="\n\n---\n\n".join(blocks),
        sources=sources,
        used_chunks=len(blocks),
        dropped_chunks=dropped,
        estimated_tokens=used_tokens,
    )
