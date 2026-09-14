"""Output validation (PRD §25)."""

from __future__ import annotations

from app.guardrails import validate_answer
from app.guardrails.output import BLOCKED_MESSAGE, GuardrailViolation
from app.prompts.system import CONTEXT_OPEN


def test_clean_answer_passes_unchanged() -> None:
    result = validate_answer("Your next appointment is on 20 September.")
    assert result.ok
    assert not result.blocked
    assert result.answer == "Your next appointment is on 20 September."


def test_empty_answer_is_blocked() -> None:
    result = validate_answer("   ")
    assert result.blocked
    assert GuardrailViolation.EMPTY in result.violations
    assert result.answer == BLOCKED_MESSAGE


def test_leaked_prompt_scaffolding_is_blocked() -> None:
    """Prompt fences in the output mean the model is echoing its context."""
    result = validate_answer(f"Sure. {CONTEXT_OPEN} your note says...")
    assert result.blocked
    assert GuardrailViolation.LEAKED_PROMPT in result.violations
    assert CONTEXT_OPEN not in result.answer


def test_truncated_answer_is_flagged_and_annotated() -> None:
    result = validate_answer("Your medications are Metformin, Lisinopr", truncated=True)
    assert not result.blocked
    assert GuardrailViolation.TRUNCATED in result.violations
    assert "cut short" in result.answer


def test_citation_of_a_source_that_was_never_retrieved() -> None:
    result = validate_answer(
        "According to your note...",
        available_source_ids={"doc:1"},
        cited_source_ids={"doc:99"},
    )
    assert GuardrailViolation.UNCITED_SOURCE in result.violations


def test_citation_within_retrieved_sources_is_accepted() -> None:
    result = validate_answer(
        "According to your note...",
        available_source_ids={"doc:1", "doc:2"},
        cited_source_ids={"doc:1"},
    )
    assert result.ok


def test_missing_sources_flagged_only_when_expected() -> None:
    """A conversational reply with no sources is normal; a RAG answer is not."""
    assert validate_answer("Hello.", expect_sources=False).ok
    assert GuardrailViolation.MISSING_SOURCES in validate_answer(
        "Hello.", expect_sources=True
    ).violations


def test_diagnosis_phrasing_is_flagged_but_not_blocked() -> None:
    result = validate_answer("Based on this, you have a condition called gout.")
    assert GuardrailViolation.POSSIBLE_DIAGNOSIS in result.violations
    assert not result.blocked, "advisory only — the answer is still shown"


def test_treatment_change_advice_is_flagged() -> None:
    result = validate_answer("You should stop taking metformin.")
    assert GuardrailViolation.POSSIBLE_DIAGNOSIS in result.violations


def test_ordinary_sentences_are_not_flagged_as_diagnosis() -> None:
    """The check must not fire on normal record reporting.

    A word list keyed on "you have" would flag every one of these, which is
    why the patterns are narrow.
    """
    benign = [
        "You have an appointment on Tuesday at 10:00.",
        "You have three medications recorded.",
        "Your record shows a diagnosis of type 2 diabetes made in 2024.",
        "Your doctor recommended physical therapy and a follow-up in six weeks.",
        "Your HbA1c was 7.2% on 10 September.",
    ]
    for answer in benign:
        assert validate_answer(answer).ok, answer
