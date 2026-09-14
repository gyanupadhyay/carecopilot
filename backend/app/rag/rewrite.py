"""Making a follow-up question retrievable on its own (PRD §19).

§19's online flow opens ``Query → Rewrite``. The rewrite that existed was
:func:`app.rag.pipeline.normalize_query` — whitespace and conversational
scaffolding — which is the right treatment for a *first* question and does
nothing for the second one:

    "What did the cardiologist say about my chest pain?"   retrieves well
    "And what did he suggest?"                             retrieves nothing

The second embeds as ``what did he suggest``. Every content word that would
match a note is in the previous turn, so the vector is close to nothing in
the corpus and the keyword search has no term to match. The answer that comes
back is grounded in whatever happened to score least badly.

**The gate is deterministic; only the rewrite is not.** Resolving "he" to
"the cardiologist" needs the model — no rule does coreference reliably — but
deciding *whether* a question depends on its predecessor does not, and
spending a model call on every RAG turn to discover that most of them are
self-contained is latency paid for nothing. So :func:`needs_rewrite` filters
first, in Python, and the model is asked only about the questions that look
context-dependent (§40 P12).

**A failed rewrite is not an error.** Anything that goes wrong — the model is
unreachable, the output fails validation, the rewrite comes back empty or
absurdly long — returns the original question. Retrieval on the raw question
is exactly what happened before this module existed, so the worst case is the
old behaviour rather than a failed turn.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.llm.base import ChatMessage, LLMProvider
from app.llm.errors import LLMError
from app.observability.logging import get_logger
from app.observability.trace import Trace

log = get_logger(__name__)

#: A rewrite is one sentence. The cap is what stops a model that ignores the
#: instruction from pasting the whole conversation into the query, which
#: retrieves worse than the pronoun did.
REWRITE_MAX_TOKENS = 120

#: Turns of history the rewriter sees. Coreference resolves against the
#: recent turn, not the whole conversation, and a longer window costs tokens
#: while adding older topics the pronoun does not refer to.
HISTORY_WINDOW = 4

#: Longest rewrite worth trusting, in characters. Past this the model has
#: summarized the conversation instead of rewriting one question.
MAX_REWRITE_CHARS = 300

#: Referring expressions that point outside the sentence containing them.
#: "it", "that", "those", "he", "she", "they", "them", "this" — plus the
#: possessives, which carry the same dependency ("his dose", "their advice").
_REFERRING = re.compile(
    r"\b(it|its|that|those|these|this|he|him|his|she|her|hers|they|them|"
    r"their|theirs|the same|the other one)\b",
    re.IGNORECASE,
)

#: Openers that continue a previous question rather than starting a new one.
_CONTINUATION = re.compile(
    r"^\s*(and|but|so|also|what about|how about|why|why not|then|ok|okay|"
    r"and what|and why|what else|anything else)\b",
    re.IGNORECASE,
)

#: Below this many words a question is almost certainly leaning on context.
#: "Any side effects?" is four words and unanswerable alone; "what did my
#: cardiologist say about my chest pain" is nine and stands by itself.
SHORT_QUESTION_WORDS = 6


def needs_rewrite(question: str, history: Sequence[ChatMessage]) -> bool:
    """Whether this question can be retrieved on without its predecessor.

    Deliberately over-inclusive rather than precise. A false positive costs
    one short model call on a question that did not need it; a false negative
    costs a wrong answer, silently. Those are not symmetric, so the gate errs
    toward rewriting.
    """
    if not history:
        # Nothing to resolve against. A first question is self-contained by
        # definition, whatever pronouns it happens to contain.
        return False
    text = question.strip()
    if not text:
        return False
    return bool(
        _REFERRING.search(text)
        or _CONTINUATION.match(text)
        or len(text.split()) < SHORT_QUESTION_WORDS
    )


class RewrittenQuery(BaseModel):
    """One self-contained question.

    A single field, so the model has one job. Asking it to also report
    confidence or explain the change invites it to spend its token budget on
    the explanation and truncate the question.
    """

    question: str = Field(
        description=(
            "The user's latest question, rewritten to stand alone. Keep the "
            "original wording wherever it already stands alone."
        )
    )


REWRITE_PROMPT = """
You rewrite a patient's follow-up question so it can be understood without
the conversation before it.

Replace pronouns and references with what they refer to, taken from the
conversation. Keep everything else exactly as the patient wrote it.

- "And what did he suggest?" after a question about Dr. Okafor
  -> "What did Dr. Okafor suggest?"
- "Any side effects?" after a question about metformin
  -> "What are the side effects of metformin?"
- "What were my last results?" (nothing to resolve)
  -> "What were my last results?"

Rules:
- Output one question, in the patient's own voice, using "my" and "I".
- Never answer it, and never add a fact the conversation does not contain.
- If nothing needs resolving, return the question unchanged.
"""


async def rewrite_query(
    *,
    question: str,
    history: Sequence[ChatMessage],
    llm: LLMProvider,
    trace: Trace | None = None,
) -> str:
    """Return a self-contained form of ``question``, or ``question`` itself.

    Never raises. See the module docstring for why every failure path falls
    back to the original rather than surfacing.
    """
    if not needs_rewrite(question, history):
        return question

    recent = list(history)[-HISTORY_WINDOW:]
    try:
        if trace is not None:
            with trace.stage("rewrite"):
                result = await _ask(question, recent, llm)
        else:
            result = await _ask(question, recent, llm)
    except LLMError as exc:
        log.warning("rag.rewrite_failed", error=type(exc).__name__)
        if trace is not None:
            trace.record_structured(ok=False)
        return question

    if trace is not None:
        trace.record_structured(ok=True)

    rewritten = result.strip()
    if not rewritten or len(rewritten) > MAX_REWRITE_CHARS:
        # Empty or runaway. Both mean the model did something other than the
        # one thing asked, and the original is known to be no worse.
        log.warning("rag.rewrite_rejected", length=len(rewritten))
        return question

    if trace is not None:
        # Shape, not content (PRD §26): that a rewrite happened, never the
        # text of either question.
        trace.query_rewritten = rewritten.lower() != question.strip().lower()
    return rewritten


async def _ask(
    question: str, recent: Sequence[ChatMessage], llm: LLMProvider
) -> str:
    response = await llm.generate_structured(
        messages=[*recent, ChatMessage(role="user", content=question)],
        system=REWRITE_PROMPT,
        schema=RewrittenQuery,
        max_tokens=REWRITE_MAX_TOKENS,
        effort="low",
    )
    return response.value.question
