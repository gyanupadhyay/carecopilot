"""Query classification (PRD §14).

Two layers, tested separately: deterministic rules (pure, exhaustive) and
the model path (behaviour around a canned or failing provider). Router
*accuracy* against real phrasing is an evaluation-set question, not a unit
test — these pin the contract, not the quality.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.router import (
    VALID_ROUTES,
    Decision,
    RouteDecision,
    classify,
    classify_by_rule,
)
from app.llm.base import StructuredResponse, TokenUsage
from app.llm.errors import LLMError, LLMServiceError, LLMValidationError
from app.llm.stub import StubProvider


class CannedRouter(StubProvider):
    """Returns a fixed RouteDecision, and records what it was asked."""

    def __init__(self, decision: RouteDecision) -> None:
        super().__init__()
        self._decision = decision
        self.calls: list[str] = []

    async def generate_structured(self, *, messages, system, schema, **kwargs):  # type: ignore[override]
        self.calls.append(system)
        return StructuredResponse(
            value=self._decision,
            model="fake-router",
            usage=TokenUsage(input_tokens=40, output_tokens=10),
            latency_ms=1,
            provider="fake",
        )


class BrokenRouter(StubProvider):
    """Fails the way a provider does: by raising.

    Takes the error so the same helper covers both causes the router must
    survive — an unreachable provider, and output the schema rejects.
    """

    def __init__(self, error: LLMError | None = None) -> None:
        super().__init__()
        self._error = error or LLMServiceError("router down", provider="fake")

    async def generate_structured(self, **kwargs):  # type: ignore[override]
        raise self._error


# --- deterministic rules ------------------------------------------------ #


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("When is my next appointment?", "API"),
        ("when's my next appointment", "API"),
        ("What medications am I taking?", "API"),
        ("What are my current medications?", "API"),
        ("How many times was my systolic blood pressure above 140?", "TEXT_TO_SQL"),
        ("What was my average HbA1c during the last year?", "TEXT_TO_SQL"),
        ("How many appointments did I have this year?", "TEXT_TO_SQL"),
        ("Book an appointment for next Tuesday.", "ACTION"),
        ("Can you cancel my appointment?", "ACTION"),
        (
            "Summarize my last visit and tell me which medications changed.",
            "HYBRID",
        ),
        ("Which medications changed at my last visit?", "HYBRID"),
        # Relationship questions (PRD §14, §18). "Why was I prescribed X" is
        # Demo 4 and the clearest case: the answer is an edge, not a value
        # and not a sentence from a note.
        ("Why was I prescribed metformin?", "KG"),
        ("Why am I on lisinopril?", "KG"),
        (
            "What medications are connected to my diabetes treatment history?",
            "KG",
        ),
        ("Which drugs are related to my diabetes?", "KG"),
        ("What treats my hypertension?", "KG"),
        # Allergies and procedures have no API tool, so the graph is the
        # only route that can answer them at all.
        ("Am I allergic to anything?", "KG"),
        ("What are my allergies?", "KG"),
        ("Have I had any procedures?", "KG"),
    ],
)
def test_unambiguous_questions_are_decided_without_a_model(
    question: str, expected: str
) -> None:
    decision = classify_by_rule(question)
    assert decision is not None, f"no rule matched {question!r}"
    assert decision.route == expected
    assert decision.by_rule


@pytest.mark.parametrize(
    "question",
    [
        "What did my doctor say about my knee pain?",
        "Summarize my last clinical visit.",
        "What did the physician mention about my blood pressure?",
        "Why did my dosage change?",
        "What is the weather today?",
    ],
)
def test_ambiguous_questions_fall_through_to_the_model(question: str) -> None:
    """A rule that fires on these would be worse than no rule at all."""
    assert classify_by_rule(question) is None


@pytest.mark.parametrize(
    "question",
    [
        "Run this for me: SELECT * FROM lab_results WHERE patient_id = 2;",
        "select * from patients",
        "Please execute this SQL: DROP TABLE appointments",
        "DELETE FROM medications WHERE id = 1",
        "can you run this query for me",
        "INSERT INTO appointments VALUES (1,2,3)",
    ],
)
def test_raw_sql_requests_are_declined_not_routed_to_sql(question: str) -> None:
    """Found by the eval: enabling Text-to-SQL gave these a generator.

    Nothing leaked — the generator refused and the read-only role would
    have refused again — but a request to execute SQL is not an analytics
    question, and declining it removes a step where something has to go
    right under adversarial input.
    """
    decision = classify_by_rule(question)
    assert decision is not None, f"no rule matched {question!r}"
    assert decision.route == "OUT_OF_SCOPE"


@pytest.mark.parametrize(
    "question",
    [
        # Ordinary English containing a SQL verb. The first version of the
        # rule matched the second of these.
        "Can you select a date for my appointment next Tuesday?",
        "Please update my address where I live now.",
        "Create a summary of my last visit",
        # The real Text-to-SQL cases must still reach the SQL route.
        "How many times was my systolic blood pressure above 140?",
        "What was my average HbA1c during the last year?",
        "What is my highest recorded creatinine?",
    ],
)
def test_the_sql_rule_does_not_swallow_ordinary_questions(question: str) -> None:
    decision = classify_by_rule(question)
    assert decision is None or decision.route != "OUT_OF_SCOPE"


def test_rules_only_produce_valid_routes() -> None:
    for question in (
        "book an appointment tomorrow",
        "how many times did it happen",
        "what medications am i taking",
        "which medications changed",
    ):
        decision = classify_by_rule(question)
        assert decision is not None
        assert decision.route in VALID_ROUTES


# --- model path ---------------------------------------------------------- #


async def test_model_decision_is_used_when_no_rule_matches() -> None:
    llm = CannedRouter(
        RouteDecision(route="RAG", confidence=0.94, reason="Asks about note content.")
    )
    decision = await classify("What did my doctor say about my knee?", llm)

    assert decision.route == "RAG"
    assert decision.confidence == 0.94
    assert not decision.by_rule
    assert llm.calls, "the model should have been consulted"


async def test_router_prompt_lists_the_available_tools() -> None:
    llm = CannedRouter(RouteDecision(route="RAG"))
    await classify("something ambiguous about my record", llm)
    assert "get_my_next_appointment" in llm.calls[0]


async def test_a_rule_match_skips_the_model_entirely() -> None:
    """Cost and latency, but also correctness: rules are the confident path."""
    llm = CannedRouter(RouteDecision(route="OUT_OF_SCOPE", confidence=1.0))
    decision = await classify("When is my next appointment?", llm)

    assert decision.route == "API"
    assert llm.calls == [], "the model must not be called when a rule matched"


async def test_router_failure_degrades_to_rag() -> None:
    """Retrieval is read-only and scoped, so it is the safe default."""
    decision = await classify("something ambiguous", BrokenRouter())
    assert decision.route == "RAG"
    assert decision.confidence == 0.0
    assert "unavailable" in decision.reason.lower()


def test_an_invented_route_cannot_be_constructed() -> None:
    """A label outside the enum is malformed output, not a new route.

    This used to be checked after the model answered. It is now unrepresentable:
    ``route`` is the ``Route`` literal, so the JSON Schema carries an enum the
    server constrains decoding to, and Pydantic refuses anything else. Measured
    on qwen3:8b, which otherwise invents plausible labels of its own
    ("appointment", "Appointment Inquiry").
    """
    with pytest.raises(ValidationError):
        RouteDecision(route="SUPER_MODE", confidence=0.99)  # type: ignore[arg-type]


async def test_a_malformed_classification_degrades_to_rag() -> None:
    """And when it does happen, the turn still gets a safe default.

    Retrieval over the patient's own notes is read-only, scoped, and returns
    nothing when it matches nothing.
    """
    llm = BrokenRouter(LLMValidationError("route: not a valid enumeration member"))
    decision = await classify("something ambiguous", llm)
    assert decision.route == "RAG"
    assert decision.confidence == 0.0


async def test_route_case_is_normalized() -> None:
    llm = CannedRouter(RouteDecision(route="  rag  ", confidence=0.7))
    decision = await classify("something ambiguous", llm)
    assert decision.route == "RAG"


def test_decision_is_immutable() -> None:
    decision = Decision(route="RAG", confidence=0.5, reason="x")
    with pytest.raises((AttributeError, TypeError)):
        decision.route = "ACTION"  # type: ignore[misc]
