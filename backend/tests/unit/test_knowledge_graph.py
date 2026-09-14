"""The knowledge graph's approved traversals (PRD §17, §21).

No Neo4j here. What these check is the property that decides whether the
graph is safe to expose to a model at all: that the *set of statements the
agent can cause to run* is closed, read-only, patient-anchored, and bounded.
Those are facts about the templates, so they are checkable without a
database — and worth checking that way, because a test needing a live graph
is a test that gets skipped on the machine where the mistake is made.

The cross-patient isolation assertion below is the important one. Isolation
is *not* a property of the projection's shape: Condition, Lab, Department and
Provider nodes are shared between patients, so paths from one patient to
another exist in the data. It holds only because no approved traversal has a
second ``:Patient`` pattern to arrive at.
"""

from __future__ import annotations

import re

import pytest

from app.auth.context import AuthContext, AuthorizationError
from app.knowledge_graph import queries
from app.knowledge_graph.queries import CYPHER, GraphIntent
from app.knowledge_graph.service import (
    GraphQueryError,
    _clean_row,
    _summarize,
    parse_intent,
    query_patient_graph,
)

ALL_INTENTS = list(GraphIntent)


def _normalized(cypher: str) -> str:
    return " ".join(cypher.split())


# --- the closed set ------------------------------------------------------ #


def test_every_intent_has_cypher_and_a_description() -> None:
    """A route the model can pick but the backend cannot run is a 500."""
    for intent in ALL_INTENTS:
        assert intent in CYPHER, intent
        assert queries.INTENT_DESCRIPTIONS.get(intent), intent


def test_no_cypher_exists_outside_the_intent_enum() -> None:
    """The reverse: a template with no intent is unreachable code."""
    assert set(CYPHER) == set(ALL_INTENTS)


# --- the four invariants ------------------------------------------------- #


@pytest.mark.parametrize("intent", ALL_INTENTS, ids=lambda i: i.value)
def test_every_traversal_anchors_on_the_authenticated_patient(
    intent: GraphIntent,
) -> None:
    assert queries.PATIENT_ANCHOR in _normalized(CYPHER[intent])


@pytest.mark.parametrize("intent", ALL_INTENTS, ids=lambda i: i.value)
def test_no_traversal_can_reach_a_second_patient(intent: GraphIntent) -> None:
    """The isolation guarantee, stated as the thing that actually enforces it.

    Shared catalogue nodes mean the *data* contains cross-patient paths. What
    keeps one patient's traversal inside their own subgraph is that no
    approved statement mentions ``:Patient`` twice, so there is nowhere for a
    traversal to arrive.
    """
    occurrences = len(re.findall(r":Patient\b", CYPHER[intent]))
    assert occurrences == 1, (
        f"{intent.value} references :Patient {occurrences} times; a second "
        "reference is a path out of the authorized subgraph."
    )


@pytest.mark.parametrize("intent", ALL_INTENTS, ids=lambda i: i.value)
def test_every_traversal_is_read_only(intent: GraphIntent) -> None:
    """§33: the application never writes to a derived projection.

    Matched on word boundaries, not as substrings. A plain ``"SET" in cypher``
    fires on ``r.onset_date`` — which is how this test first failed, against a
    traversal that only reads.
    """
    cypher = CYPHER[intent].upper()
    for keyword in queries.FORBIDDEN_KEYWORDS:
        pattern = r"\b" + keyword.replace(" ", r"\s+") + r"\b"
        assert not re.search(pattern, cypher), f"{intent.value} contains {keyword}"


@pytest.mark.parametrize("intent", ALL_INTENTS, ids=lambda i: i.value)
def test_every_traversal_is_bounded(intent: GraphIntent) -> None:
    """A dense subgraph must not become an unbounded context window."""
    assert "LIMIT $limit" in _normalized(CYPHER[intent])


@pytest.mark.parametrize("intent", ALL_INTENTS, ids=lambda i: i.value)
def test_no_traversal_interpolates_a_patient_id(intent: GraphIntent) -> None:
    """Scope arrives as a bound parameter, never as query text.

    A template that formatted an id into the statement would be one
    ``str.format`` away from accepting one from the model.
    """
    cypher = CYPHER[intent]
    assert "$patient_id" in cypher
    # No f-string placeholders survived into the final template.
    assert "{patient_id}" not in cypher


def test_the_search_term_is_always_a_parameter() -> None:
    """Cypher parameters are not substituted into the query text.

    So a term like ``"' OR 1=1 //"`` is matched as that literal string. Any
    template that referenced a term by name instead would be interpolating.
    """
    for intent in queries.REQUIRES_TERM:
        assert "$term" in CYPHER[intent], intent.value


# --- intent parsing ------------------------------------------------------ #


def test_a_known_intent_parses() -> None:
    assert parse_intent("why_medication") is GraphIntent.WHY_MEDICATION
    assert parse_intent("  WHY_MEDICATION  ") is GraphIntent.WHY_MEDICATION


def test_an_invented_intent_is_refused() -> None:
    """Not a new capability — a malformed tool call."""
    with pytest.raises(GraphQueryError, match="Unknown graph intent"):
        parse_intent("read_everything")


async def test_a_term_requiring_intent_without_one_is_refused() -> None:
    """Silently matching everything would answer a different question."""
    ctx = AuthContext(user_id=1, role="patient", patient_id=1)
    with pytest.raises(GraphQueryError, match="needs a search term"):
        await query_patient_graph(
            ctx, intent=GraphIntent.WHY_MEDICATION, term="   "
        )


async def test_a_session_without_a_patient_cannot_traverse() -> None:
    """Raised before any statement is built, let alone sent."""
    ctx = AuthContext(user_id=9, role="patient", patient_id=None)
    with pytest.raises(AuthorizationError):
        await query_patient_graph(ctx, intent=GraphIntent.CONDITIONS)


# --- result shaping ------------------------------------------------------ #


def test_null_columns_are_dropped() -> None:
    """OPTIONAL MATCH nulls carry no information and cost context tokens.

    Left in, they also invite the model to report absent data as a finding.
    """
    assert _clean_row({"a": 1, "b": None, "c": [], "d": [None]}) == {"a": 1}


def test_dates_are_stringified() -> None:
    from datetime import date

    assert _clean_row({"when": date(2026, 1, 2)}) == {"when": "2026-01-02"}


def test_an_empty_result_says_so_rather_than_guessing() -> None:
    summary = _summarize(GraphIntent.WHY_MEDICATION, [], "metformin")
    assert "no" in summary.lower()
    assert "metformin" in summary


def test_the_summary_is_composed_from_returned_values() -> None:
    """PRD Principle 12: the backend states what it can state reliably."""
    rows = [{"medication": "Metformin", "treats": "Type 2 diabetes mellitus"}]
    summary = _summarize(GraphIntent.WHY_MEDICATION, rows, "metformin")
    assert "Metformin" in summary
    assert "Type 2 diabetes mellitus" in summary


def test_state_carries_graph_rows_as_data_not_only_as_prose() -> None:
    """PRD §13 names `graph_results`; the rendered facts are not a substitute.

    The rendering is what the model reads and it is lossy — a row count is
    not recoverable from the sentence describing it without parsing prose.
    """
    from app.agents.state import initial_state

    state = initial_state(question="q", user_id=1, role="patient", patient_id=1)
    assert state["graph_results"] == []
