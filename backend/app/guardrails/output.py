"""Output validation and guardrails.

What this layer can and cannot do is worth being honest about, because the
difference decides how much weight the rest of the system may put on it.

It *can* catch structural failures deterministically: an empty answer, an
answer truncated by the token cap, a citation pointing at a source that was
never retrieved, a leaked prompt fence. These are exact checks and they are
where the value is.

It *cannot* reliably detect "unsupported medical claim" by pattern matching.
Phrasing is unbounded, and a regex that flags "you have" will flag "you have
an appointment". So the diagnosis check here is narrow and advisory: it
raises a warning recorded on the trace, and it never silently edits the
answer. The real controls against ungrounded claims are the grounding
prompt, giving the model only one patient's data, and the faithfulness
metric in the evaluation set — not a word list.

Blocking is reserved for cases where returning the text would be worse than
returning nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from app.prompts.system import CONTEXT_CLOSE, CONTEXT_OPEN


class GuardrailViolation(StrEnum):
    EMPTY = "empty_answer"
    TRUNCATED = "truncated_answer"
    LEAKED_PROMPT = "leaked_prompt_scaffolding"
    UNCITED_SOURCE = "cited_unavailable_source"
    MISSING_SOURCES = "missing_sources"
    POSSIBLE_DIAGNOSIS = "possible_diagnosis"


#: Violations that mean the answer must not be shown at all.
BLOCKING = frozenset(
    {
        GuardrailViolation.EMPTY,
        GuardrailViolation.LEAKED_PROMPT,
    }
)

BLOCKED_MESSAGE = (
    "I could not produce a reliable answer to that. Please try rephrasing, "
    "or contact your care team if it is urgent."
)

TRUNCATION_NOTE = (
    "\n\n_This answer was cut short. Ask a narrower question to see the rest._"
)

#: Narrow, high-precision phrasings of a clinical judgement. Each one asserts
#: a conclusion about the patient rather than reporting a recorded fact.
#: Deliberately does not include bare "you have", which matches ordinary
#: sentences like "you have an appointment on Tuesday".
_DIAGNOSIS_PATTERNS = (
    r"\byou (?:likely |probably |may |might )?have (?:a |an )?"
    r"(?:condition|disease|disorder|syndrome|infection)\b",
    r"\byou (?:are|appear to be) suffering from\b",
    r"\bthis (?:means|indicates|suggests) (?:that )?you have\b",
    r"\bI (?:would )?(?:diagnose|recommend that you (?:start|stop|change))\b",
    r"\byou should (?:start|stop|increase|decrease|change) (?:taking |your )?\w+",
)
_DIAGNOSIS_RE = re.compile("|".join(_DIAGNOSIS_PATTERNS), re.IGNORECASE)

#: Prompt scaffolding that must never reach a user.
_LEAK_MARKERS = (CONTEXT_OPEN, CONTEXT_CLOSE, "SYSTEM INSTRUCTIONS", "<thinking>")


@dataclass(slots=True)
class GuardrailResult:
    answer: str
    violations: list[GuardrailViolation] = field(default_factory=list)
    blocked: bool = False

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def codes(self) -> list[str]:
        return [v.value for v in self.violations]


def validate_answer(
    answer: str,
    *,
    truncated: bool = False,
    available_source_ids: set[str] | None = None,
    cited_source_ids: set[str] | None = None,
    expect_sources: bool = False,
) -> GuardrailResult:
    """Check an answer and return it, possibly annotated or replaced.

    ``expect_sources`` is set by routes that retrieved record content: an
    answer built from retrieved documents that cites nothing is a grounding
    failure worth recording, whereas a general conversational reply with no
    sources is entirely normal.
    """
    violations: list[GuardrailViolation] = []
    text = (answer or "").strip()

    if not text:
        violations.append(GuardrailViolation.EMPTY)

    if any(marker in text for marker in _LEAK_MARKERS):
        violations.append(GuardrailViolation.LEAKED_PROMPT)

    if truncated:
        violations.append(GuardrailViolation.TRUNCATED)

    if cited_source_ids:
        unavailable = cited_source_ids - (available_source_ids or set())
        if unavailable:
            # The model referred to a document that was not retrieved. The
            # citation is fabricated even if the sentence happens to be true.
            violations.append(GuardrailViolation.UNCITED_SOURCE)
    elif expect_sources:
        violations.append(GuardrailViolation.MISSING_SOURCES)

    if text and _DIAGNOSIS_RE.search(text):
        violations.append(GuardrailViolation.POSSIBLE_DIAGNOSIS)

    blocked = any(v in BLOCKING for v in violations)
    if blocked:
        return GuardrailResult(
            answer=BLOCKED_MESSAGE, violations=violations, blocked=True
        )

    if GuardrailViolation.TRUNCATED in violations:
        text += TRUNCATION_NOTE

    return GuardrailResult(answer=text, violations=violations, blocked=False)
