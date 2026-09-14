"""An LLM judge for faithfulness (PRD §27).

Faithfulness asks a narrower question than correctness: *is every claim in
this answer supported by the passages the system put in front of the model?*
An answer can be faithful and wrong (the retrieved note was out of date) or
unfaithful and right (the model knew the fact without being shown it). The
second is the dangerous one in a medical assistant, and it is the one no
other metric in this suite can see — keyword containment scores a fabricated
sentence containing the right word as correct.

Three things about this judge are deliberate, and the first is a weakness
that has to be stated rather than designed around.

*The judge shares weights with the generator.* Both are the configured local
model. A judge that shares a model's blind spots will not flag the
hallucinations that model finds plausible, so this number is a **floor**, not
an estimate — real faithfulness is at most what this reports, and a score
here of 1.000 means "the model does not object to itself", which is weaker
than it looks. Passing ``--judge-model`` runs a different model and is worth
doing whenever one is available.

*The judge scores claims, not the whole answer.* A verdict per extracted
claim gives a graded score and, more usefully, names the unsupported claim in
the report. One yes/no per answer throws away which sentence was the problem,
which is the only part anyone acts on.

*A judge that fails returns ``None``.* Not 0.0, and not 1.0. The provider
being unreachable is not evidence about the answer, and both defaults would
state something the run did not measure — see the honesty rules in
``metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.llm.base import ChatMessage, LLMProvider
from app.llm.errors import LLMError
from app.observability.logging import get_logger

log = get_logger(__name__)

#: Enough for a handful of claims and their verdicts. Structured calls that
#: hit the cap raise rather than return partial JSON, so this is sized for
#: the worst case rather than the typical one.
JUDGE_MAX_TOKENS = 900

#: Beyond this the judge is reading more context than the generator did, and
#: the cost per case stops being worth the signal.
MAX_CONTEXT_CHARS = 6000


class ClaimVerdict(BaseModel):
    """One factual claim and whether the passages support it."""

    claim: str = Field(description="The claim, quoted or closely paraphrased.")
    supported: bool = Field(
        description="True only if a passage states this. Not 'plausible', "
        "not 'consistent with' — stated."
    )


class FaithfulnessVerdict(BaseModel):
    claims: list[ClaimVerdict] = Field(
        default_factory=list,
        description="Every checkable factual claim in the answer. Omit "
        "hedges, disclaimers, and instructions to contact a clinician.",
    )


JUDGE_PROMPT = """
You check whether an answer stays within its evidence. You are not judging
whether the answer is medically correct, helpful, or well written.

You will be given PASSAGES and an ANSWER. Extract every checkable factual
claim the ANSWER makes — a value, a date, a name, a dose, a count, a stated
relationship between two things — and for each one decide whether a PASSAGE
states it.

Rules:
- Supported means a passage states it. A claim that is merely plausible, or
  consistent with the passages, or true of medicine generally, is NOT
  supported.
- A number that appears nowhere in the passages is not supported, however
  reasonable it looks.
- Ignore safety disclaimers, hedging, and "contact your care team" advice.
  They make no factual claim.
- If the answer is a refusal or states that nothing is on record, return no
  claims at all.

Return only the JSON object. Do not explain your reasoning.
""".strip()


@dataclass(slots=True)
class FaithfulnessResult:
    """What the judge decided, or why it could not decide."""

    #: Share of extracted claims the passages support. ``None`` when the
    #: judge did not run, failed, or found no claim to check.
    score: float | None = None
    claims_checked: int = 0
    unsupported: list[str] = field(default_factory=list)
    #: Set when ``score`` is None, so the report can say which it was.
    skipped: str = ""


async def judge_faithfulness(
    llm: LLMProvider,
    *,
    answer: str,
    passages: list[str],
    model: str | None = None,
) -> FaithfulnessResult:
    """Score one answer against the passages that produced it.

    Never raises. Every failure path returns a ``FaithfulnessResult`` whose
    ``score`` is ``None`` and whose ``skipped`` says why, because a judge
    that throws would abort a run that has already spent an hour of local
    inference on the cases before it.
    """
    if not passages:
        return FaithfulnessResult(skipped="no grounding passages to judge against")
    if not answer.strip():
        return FaithfulnessResult(skipped="empty answer")

    context = _render(passages)
    try:
        verdict = await llm.generate_structured(
            messages=[
                ChatMessage(
                    role="user",
                    content=f"PASSAGES:\n{context}\n\nANSWER:\n{answer.strip()}",
                )
            ],
            system=JUDGE_PROMPT,
            schema=FaithfulnessVerdict,
            max_tokens=JUDGE_MAX_TOKENS,
            effort="low",
            model=model,
        )
    except LLMError as exc:
        log.warning("judge.failed", error=type(exc).__name__)
        return FaithfulnessResult(
            skipped=f"judge unavailable ({type(exc).__name__})"
        )

    claims = verdict.value.claims
    if not claims:
        # A refusal, or an answer that asserts nothing. Not a failure and not
        # a perfect score: there was nothing to be faithful about.
        return FaithfulnessResult(skipped="answer makes no checkable claim")

    unsupported = [c.claim for c in claims if not c.supported]
    return FaithfulnessResult(
        score=(len(claims) - len(unsupported)) / len(claims),
        claims_checked=len(claims),
        unsupported=unsupported,
    )


def _render(passages: list[str]) -> str:
    """Number the passages and cap the total the judge reads."""
    rendered: list[str] = []
    budget = MAX_CONTEXT_CHARS
    for index, text in enumerate(passages, start=1):
        body = text.strip()
        if not body:
            continue
        if len(body) > budget:
            body = body[:budget]
        rendered.append(f"[{index}] {body}")
        budget -= len(body)
        if budget <= 0:
            break
    return "\n\n".join(rendered)


__all__ = [
    "ClaimVerdict",
    "FaithfulnessResult",
    "FaithfulnessVerdict",
    "judge_faithfulness",
]
