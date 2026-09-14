"""The KG and Text-to-SQL metric families (PRD §27).

These score the *query the system chose*, not the sentence it produced, and
that separation is the whole point of them. An answer can name the right
drug after walking the wrong edges — on this dataset one demonstrably does,
because joining medication to condition through the encounter returns the
right row for any patient whose visit treated a single condition. Answer
correctness calls that a pass. Relationship correctness does not.

The Cypher assertions run against the real templates in
``app.knowledge_graph.queries`` rather than against fixtures, so a traversal
rewritten in a way that changes its shape fails here instead of quietly
changing what the metric reports.
"""

from __future__ import annotations

import pytest

from app.knowledge_graph.queries import CYPHER, GraphIntent
from app.sql.schema import ALLOWED_TABLES
from evaluation import metrics as M
from evaluation.runner import CaseResult

# --- Cypher shape ------------------------------------------------------- #


def test_relationship_types_reads_both_directions() -> None:
    cypher = "MATCH (p:Patient)-[:TAKES]->(m)<-[:PRESCRIBED]-(e) RETURN m"
    assert M.relationship_types(cypher) == {"TAKES", "PRESCRIBED"}


def test_hop_depth_follows_variables_across_match_clauses() -> None:
    """The measurement that per-clause counting gets wrong.

    ``care_team`` walks patient → encounter → provider with one relationship
    pattern per clause. Counting patterns within a clause calls that one hop.
    """
    assert (
        M.hop_depth(
            """
            MATCH (p:Patient {id: $patient_id})-[:HAD_ENCOUNTER]->(e:Encounter)
            MATCH (e)-[:WITH_PROVIDER]->(pr:Provider)
            RETURN pr
            """
        )
        == 2
    )


def test_hop_depth_counts_chained_patterns_in_one_clause() -> None:
    """``(a)-[]->(b)-[]->(c)`` is two hops, not one.

    A non-overlapping scan finds only the first step, because it resumes
    past ``(b)`` instead of at it.
    """
    assert (
        M.hop_depth(
            "MATCH (p:Patient {id: 1})-[:TAKES]->(m:Medication)-[:TREATS]->(c) RETURN c"
        )
        == 2
    )


def test_hop_depth_takes_the_longest_path_not_the_shortest() -> None:
    """The triangle case, and why BFS depth is the wrong measure.

    ``medications_for_condition`` reaches both the condition and the
    medication directly from the patient, so every node sits at distance 1 —
    and the hop that matters is the third edge, ``(m)-[:TREATS]->(c)``, which
    is precisely what stops the answer being assembled through the encounter.
    """
    assert M.hop_depth(CYPHER[GraphIntent.MEDICATIONS_FOR_CONDITION]) == 2


def test_hop_depth_ignores_patterns_in_the_return_clause() -> None:
    """``why_medication`` tests an edge in a CASE without travelling it."""
    cypher = """
        MATCH (p:Patient {id: 1})-[:TAKES]->(m:Medication)
        RETURN CASE WHEN (m)-[:FOR_CONDITION]->(c) THEN 1 ELSE NULL END AS x
    """
    assert M.hop_depth(cypher) == 1


def test_hop_depth_is_zero_without_a_patient_anchor() -> None:
    """No anchor means nothing to measure from — not a depth of one."""
    assert M.hop_depth("MATCH (x)-[:A]->(y) RETURN y") == 0


@pytest.mark.parametrize("intent", list(GraphIntent))
def test_every_template_is_measurable(intent: GraphIntent) -> None:
    """Each approved traversal yields a hop count and at least one edge.

    A template the metric cannot read would score as zero hops and no
    relationships, which reads as "the traversal walks nothing" rather than
    as "the metric could not parse it".
    """
    cypher = CYPHER[intent]
    assert M.hop_depth(cypher) >= 1
    assert M.relationship_types(cypher)


# --- relationship and multi-hop correctness ------------------------------ #


def test_relationship_correctness_is_a_subset_check() -> None:
    """Extra optional edges are context, not error."""
    cypher = CYPHER[GraphIntent.CARE_TEAM]
    assert M.relationship_correctness(cypher, ["HAD_ENCOUNTER", "WITH_PROVIDER"])


def test_relationship_correctness_fails_on_a_missing_edge() -> None:
    """The failure it exists for: right answer, wrong path.

    ``medication_history`` lists every drug and would mention metformin for
    a diabetes question — while never touching HAS_CONDITION, so it cannot
    have restricted the list to that condition.
    """
    assert not M.relationship_correctness(
        CYPHER[GraphIntent.MEDICATION_HISTORY],
        ["HAS_CONDITION", "TAKES", "TREATS"],
    )


def test_relationship_correctness_skips_when_nothing_is_expected() -> None:
    assert M.relationship_correctness(CYPHER[GraphIntent.CONDITIONS], []) is None


def test_multi_hop_skips_single_hop_cases() -> None:
    """Every traversal clears a bar of one, so scoring them dilutes."""
    assert M.multi_hop_correctness(CYPHER[GraphIntent.CONDITIONS], 1) is None
    assert M.multi_hop_correctness(CYPHER[GraphIntent.CONDITIONS], None) is None


def test_multi_hop_fails_a_shallow_traversal() -> None:
    assert M.multi_hop_correctness(CYPHER[GraphIntent.ALLERGIES], 3) is False


def test_multi_hop_passes_a_deep_enough_traversal() -> None:
    assert M.multi_hop_correctness(CYPHER[GraphIntent.CARE_TEAM], 3) is True


# --- entity resolution ---------------------------------------------------- #


def _kg_case(**kwargs: object) -> CaseResult:
    fields: dict[str, object] = {
        "case_id": "kg-x",
        "category": "kg",
        "patient": "P001",
        "question": "q",
        "route": "KG",
        "expected_route": "KG",
    }
    fields.update(kwargs)
    return CaseResult(**fields)  # type: ignore[arg-type]


def test_a_traversal_that_matched_nothing_is_a_resolution_failure() -> None:
    """The aliases bug, as a metric.

    The patient says "blood pressure", the catalogue says "Essential
    hypertension", neither contains the other. Routing and tool selection
    both score 1.000 on that case; only this registers it.
    """
    case = _kg_case(
        tool_calls=[{"name": "kg:condition_timeline", "ms": 3, "ok": False}],
        guardrails=["kg_no_match"],
    )
    assert case.kg_intent == "condition_timeline"
    assert M.entity_resolution(case.kg_matched) is False


def test_a_traversal_that_matched_is_a_resolution_success() -> None:
    case = _kg_case(tool_calls=[{"name": "kg:conditions", "ms": 3, "ok": True}])
    assert M.entity_resolution(case.kg_matched) is True


def test_a_case_that_never_reached_a_traversal_is_not_scored() -> None:
    """A KG question misrouted to RAG is one mistake, not two.

    It is already counted as a routing failure; counting it again here would
    report the same error twice under two names.
    """
    case = _kg_case(route="RAG", tool_calls=[])
    assert case.kg_intent is None
    assert case.kg_matched is None


# --- Text-to-SQL ---------------------------------------------------------- #


def test_sql_tables_and_functions_are_parsed_not_matched() -> None:
    sql = "SELECT COUNT(*) FROM lab_results WHERE test_name = 'HbA1c'"
    assert M.sql_tables(sql) == {"lab_results"}
    assert "COUNT" in M.sql_functions(sql)


def test_boolean_connectors_are_not_reported_as_functions() -> None:
    """sqlglot registers ``And`` as a Func subclass.

    Harmless for the subset check this feeds, and actively confusing in a
    report that tells a reader the query "called AND".
    """
    sql = "SELECT COUNT(*) FROM lab_results WHERE a = 1 AND b > 2"
    assert M.sql_functions(sql) == {"COUNT"}


def test_unparseable_sql_is_not_valid() -> None:
    assert not M.sql_parses("SELECT COUNT( FROM WHERE")


def test_a_patient_predicate_is_a_defect_not_a_safeguard() -> None:
    """RLS already scopes the connection.

    A ``patient_id`` predicate can only narrow further, using a value the
    generator guessed — so it excludes the patient's own rows. It looks like
    defence in depth, which is why the metric checks for it rather than
    trusting review.
    """
    assert M.sql_filters_patient(
        "SELECT COUNT(*) FROM appointments WHERE patient_id = 2"
    )


def test_selecting_the_patient_column_is_not_filtering_on_it() -> None:
    """Detected as a predicate, not as a substring."""
    assert not M.sql_filters_patient("SELECT patient_id FROM appointments")


def test_authorization_fails_on_a_table_outside_the_allowlist() -> None:
    assert (
        M.sql_authorization_correctness(
            "SELECT COUNT(*) FROM users", ALLOWED_TABLES
        )
        is False
    )


def test_authorization_fails_on_a_patient_predicate() -> None:
    assert (
        M.sql_authorization_correctness(
            "SELECT COUNT(*) FROM appointments WHERE patient_id = 2", ALLOWED_TABLES
        )
        is False
    )


def test_authorization_passes_an_unscoped_allowlisted_query() -> None:
    """What correct generated SQL looks like here: no patient predicate."""
    assert (
        M.sql_authorization_correctness(
            "SELECT COUNT(*) FROM lab_results WHERE result_date > '2026-01-01'",
            ALLOWED_TABLES,
        )
        is True
    )


def test_query_correctness_catches_how_many_answered_with_select_star() -> None:
    assert (
        M.sql_query_correctness(
            "SELECT * FROM appointments", ["appointments"], ["COUNT"]
        )
        is False
    )


def test_query_correctness_catches_the_wrong_table() -> None:
    assert (
        M.sql_query_correctness(
            "SELECT COUNT(*) FROM encounters", ["appointments"], ["COUNT"]
        )
        is False
    )


def test_query_correctness_skips_a_case_that_declares_nothing() -> None:
    assert M.sql_query_correctness("SELECT 1", [], []) is None


# --- agent ---------------------------------------------------------------- #


def test_tool_call_success_is_separate_from_selection() -> None:
    """A run where selection is 1.000 and success is 0.500 is broken.

    The selection number alone calls it healthy, which is why both exist.
    """
    calls = [{"name": "a", "ok": True}, {"name": "b", "ok": False}]
    assert M.tool_call_success_rate(calls) == 0.5


def test_tool_call_success_skips_a_turn_that_called_nothing() -> None:
    assert M.tool_call_success_rate([]) is None


def test_json_validity_skips_a_turn_that_asked_for_no_json() -> None:
    """A rule-routed question makes no structured call.

    Scoring it 1.0 reports a passed check that never ran — the same mistake
    as scoring stub text for correctness.
    """
    assert M.json_validity(0, 0) is None


def test_json_validity_counts_failures_against_calls() -> None:
    assert M.json_validity(4, 1) == 0.75


# --- the None-not-zero rule, for the new families ------------------------- #


def test_rate_ignores_undecided_checks() -> None:
    """A skipped check must not be averaged in as a failure."""
    assert M.rate([True, None, True]) == 1.0


def test_rate_is_none_when_nothing_was_decided() -> None:
    assert M.rate([None, None]) is None
