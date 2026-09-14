"""The traversals the agent may run, as fixed templates.

This module is the whole of PRD §21. The rule it exists to enforce:

    Correct:   patient_id -> query_my_patient_graph() -> authorized subgraph
    Incorrect: Qwen3 -> arbitrary Cypher with arbitrary patient ID -> Neo4j

So the model never writes Cypher. It picks an :class:`GraphIntent` — one of a
closed set of labels — and may supply a search term. Everything else is
written here, reviewed, and parameterised. A model that asks for
``patient_id`` 999 cannot get it, because no template takes a patient id from
its caller's arguments: ``$patient_id`` is bound by
:mod:`app.knowledge_graph.service` from the ``AuthContext``, and the intent
the model chose cannot change which value goes in.

Four properties every template must hold, and which
``test_knowledge_graph.py`` asserts mechanically rather than trusting review:

1. It anchors on ``(p:Patient {id: $patient_id})``. A traversal that started
   anywhere else could leave the patient's subgraph.
2. It mentions ``:Patient`` exactly once. This is the one that carries the
   weight. The projection shares ``Condition``, ``Lab``, ``Department`` and
   ``Provider`` between patients — deliberately, since a graph in which every
   patient owns a private "Cardiology" is not a graph — so cross-patient
   paths do exist, and there are six figures of them at length ≤ 4 on a
   hundred synthetic patients (156,555 as this is written, up from 138,372
   before ``Diagnosis`` was added; the count grows with the graph, which is
   the point). Nearly all are
   ``(p1)-[:HAS_CONDITION]->(c)<-[:HAS_CONDITION]-(p2)``. Isolation is
   therefore not a property of the data's shape; it holds because no approved
   traversal has a second ``:Patient`` pattern to arrive at.
3. It is read-only — no CREATE, MERGE, SET, DELETE, REMOVE, DROP or CALL.
   The application never writes to a derived projection (§33).
4. It ends in ``LIMIT $limit``. A dense subgraph must not become an
   unbounded context window.

The search term is a parameter, never interpolated. Cypher parameters are
not substituted into the query text, so a term like ``"' OR 1=1 //"`` is
matched as that literal string and finds nothing.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class GraphIntent(StrEnum):
    """What the agent wants from the graph. The model chooses one of these."""

    CONDITIONS = "conditions"
    MEDICATIONS_FOR_CONDITION = "medications_for_condition"
    WHY_MEDICATION = "why_medication"
    CONDITION_TIMELINE = "condition_timeline"
    LABS_FOR_CONDITION = "labs_for_condition"
    CARE_TEAM = "care_team"
    MEDICATION_HISTORY = "medication_history"
    ALLERGIES = "allergies"
    PROCEDURES = "procedures"
    DIAGNOSIS_HISTORY = "diagnosis_history"


#: What each intent answers, in the words a router or tool description needs.
#: Kept beside the Cypher so the description and the traversal cannot drift.
INTENT_DESCRIPTIONS: Final[dict[GraphIntent, str]] = {
    GraphIntent.CONDITIONS: (
        "Every condition on the patient's record, with when it was first seen."
    ),
    GraphIntent.MEDICATIONS_FOR_CONDITION: (
        "Only the medications that treat ONE named condition. Use whenever the "
        'question names a condition — "what am I taking for my diabetes", '
        '"which drugs relate to my blood pressure". Needs that condition as '
        'the term, e.g. "diabetes".'
    ),
    GraphIntent.WHY_MEDICATION: (
        "Why a named medication was prescribed: what it treats, and the visit "
        'and clinician that started it. Needs a medication term, e.g. "metformin".'
    ),
    GraphIntent.CONDITION_TIMELINE: (
        "The visits for a named condition, in order, with the clinician seen. "
        "Needs a condition term."
    ),
    GraphIntent.LABS_FOR_CONDITION: (
        "Lab tests ordered at visits for a named condition, with their results. "
        "Needs a condition term."
    ),
    GraphIntent.CARE_TEAM: (
        "Clinicians the patient has seen, their department, and what for."
    ),
    GraphIntent.MEDICATION_HISTORY: (
        "EVERY medication on record, across all conditions, with what each "
        "treats and its status. Use only when no particular condition is "
        "named — otherwise use medications_for_condition, which answers the "
        "narrower question that was actually asked."
    ),
    GraphIntent.ALLERGIES: (
        "Substances the patient reacts to, with the reaction and how severe "
        "it is. Use for any question about allergies or intolerances."
    ),
    GraphIntent.PROCEDURES: (
        "Procedures the patient has had, with the date, the clinician and "
        "what it was for."
    ),
    GraphIntent.DIAGNOSIS_HISTORY: (
        "When each condition was diagnosed, by whom, and with what code. Use "
        'for "when was I diagnosed with X" — conditions answers the simpler '
        '"what do I have".'
    ),
}

#: Intents whose Cypher references ``$term``. Asking for one without a term
#: is a malformed request, caught in the service rather than silently
#: matching everything.
REQUIRES_TERM: Final[frozenset[GraphIntent]] = frozenset(
    {
        GraphIntent.MEDICATIONS_FOR_CONDITION,
        GraphIntent.WHY_MEDICATION,
        GraphIntent.CONDITION_TIMELINE,
        GraphIntent.LABS_FOR_CONDITION,
    }
)


# Matching is case-insensitive and substring-based, because the term comes
# from a patient's phrasing: "diabetes" has to find "Type 2 diabetes
# mellitus". `toLower(...) CONTAINS toLower($term)` rather than a regex —
# a regex built from user text is a denial-of-service waiting to happen,
# and CONTAINS cannot backtrack.
#
# The alias clause is not a refinement. Patients name conditions the way they
# were explained to them — "blood pressure", "blood sugar", "thyroid" — and
# none of those is a substring of the stored display name. Without it the
# evaluation case "which of my visits were about my blood pressure?" chose
# the right route and the right traversal and returned nothing at all.
# `c.aliases` is stored already lower-cased by the projection.
_CONDITION_MATCHES: Final = (
    "(toLower(c.display) CONTAINS toLower($term) "
    "OR toLower(c.key) CONTAINS toLower($term) "
    "OR any(alias IN coalesce(c.aliases, []) "
    "WHERE alias CONTAINS toLower($term) OR toLower($term) CONTAINS alias))"
)


CYPHER: Final[dict[GraphIntent, str]] = {
    GraphIntent.CONDITIONS: """
        MATCH (p:Patient {id: $patient_id})-[r:HAS_CONDITION]->(c:Condition)
        RETURN c.display AS condition,
               r.onset_date AS first_seen
        ORDER BY coalesce(r.onset_date, date('9999-12-31')), c.display
        LIMIT $limit
    """,
    # Via TREATS, not by joining through the encounter. The encounter path
    # would return every drug started at a visit filed under this condition,
    # including the ones started for the patient's other problems that day.
    GraphIntent.MEDICATIONS_FOR_CONDITION: f"""
        MATCH (p:Patient {{id: $patient_id}})-[:HAS_CONDITION]->(c:Condition)
        WHERE {_CONDITION_MATCHES}
        MATCH (p)-[:TAKES]->(m:Medication)-[:TREATS]->(c)
        RETURN c.display AS condition,
               m.name AS medication,
               m.dosage AS dosage,
               m.frequency AS frequency,
               m.status AS status,
               m.start_date AS started,
               m.end_date AS ended
        ORDER BY m.name, m.start_date
        LIMIT $limit
    """,
    # Demo 4. The condition is the answer; the encounter and provider are
    # what make it an explanation rather than an assertion.
    GraphIntent.WHY_MEDICATION: """
        MATCH (p:Patient {id: $patient_id})-[:TAKES]->(m:Medication)
        WHERE toLower(m.name) CONTAINS toLower($term)
        OPTIONAL MATCH (m)-[:TREATS]->(c:Condition)
        OPTIONAL MATCH (e:Encounter)-[:PRESCRIBED]->(m)
        OPTIONAL MATCH (e)-[:WITH_PROVIDER]->(pr:Provider)
        RETURN m.name AS medication,
               m.dosage AS dosage,
               m.status AS status,
               m.start_date AS started,
               c.display AS treats,
               e.encounter_date AS prescribed_at,
               // The visit's own reason, but only when the visit was about
               // the same condition the drug treats. A real appointment
               // covers several problems at once, so the chief complaint is
               // often about something else entirely — returning it here
               // invited the answer "you were prescribed metformin because
               // of knee pain", which the data does not say. Absent is
               // better than misleading.
               CASE WHEN (e)-[:FOR_CONDITION]->(c) THEN e.reason ELSE NULL END
                   AS visit_reason,
               pr.name AS prescriber,
               pr.specialty AS prescriber_specialty
        ORDER BY m.start_date
        LIMIT $limit
    """,
    GraphIntent.CONDITION_TIMELINE: f"""
        MATCH (p:Patient {{id: $patient_id}})-[:HAS_CONDITION]->(c:Condition)
        WHERE {_CONDITION_MATCHES}
        MATCH (p)-[:HAD_ENCOUNTER]->(e:Encounter)-[:FOR_CONDITION]->(c)
        OPTIONAL MATCH (e)-[:WITH_PROVIDER]->(pr:Provider)
        OPTIONAL MATCH (e)-[:PRESCRIBED]->(m:Medication)
        RETURN c.display AS condition,
               e.encounter_date AS visit_date,
               e.encounter_type AS visit_type,
               e.reason AS reason,
               pr.name AS clinician,
               collect(DISTINCT m.name) AS medications_started
        ORDER BY e.encounter_date
        LIMIT $limit
    """,
    GraphIntent.LABS_FOR_CONDITION: f"""
        MATCH (p:Patient {{id: $patient_id}})-[:HAS_CONDITION]->(c:Condition)
        WHERE {_CONDITION_MATCHES}
        MATCH (p)-[:HAD_ENCOUNTER]->(e:Encounter)-[:FOR_CONDITION]->(c)
        MATCH (e)-[:ORDERED]->(l:Lab)
        MATCH (p)-[:HAS_LAB_RESULT]->(lr:LabResult)-[:OF_LAB]->(l)
        WHERE lr.encounter_id = e.id
        RETURN c.display AS condition,
               l.name AS test,
               lr.value AS value,
               lr.unit AS unit,
               lr.reference_range AS reference_range,
               lr.result_date AS result_date
        ORDER BY lr.result_date DESC, l.name
        LIMIT $limit
    """,
    GraphIntent.CARE_TEAM: """
        MATCH (p:Patient {id: $patient_id})-[:HAD_ENCOUNTER]->(e:Encounter)
        MATCH (e)-[:WITH_PROVIDER]->(pr:Provider)
        OPTIONAL MATCH (pr)-[:IN_DEPARTMENT]->(d:Department)
        OPTIONAL MATCH (e)-[:FOR_CONDITION]->(c:Condition)
        RETURN pr.name AS clinician,
               d.name AS department,
               count(DISTINCT e) AS visits,
               max(e.encounter_date) AS last_seen,
               collect(DISTINCT c.display) AS conditions
        ORDER BY visits DESC, clinician
        LIMIT $limit
    """,
    GraphIntent.MEDICATION_HISTORY: """
        MATCH (p:Patient {id: $patient_id})-[:TAKES]->(m:Medication)
        OPTIONAL MATCH (m)-[:TREATS]->(c:Condition)
        RETURN m.name AS medication,
               m.dosage AS dosage,
               m.status AS status,
               m.start_date AS started,
               m.end_date AS ended,
               c.display AS treats
        ORDER BY m.start_date DESC, m.name
        LIMIT $limit
    """,
    # Ordered most serious first. An allergy list read top-down should lead
    # with the one that matters, not with whichever substance sorts first.
    GraphIntent.ALLERGIES: """
        MATCH (p:Patient {id: $patient_id})-[:HAS_ALLERGY]->(a:Allergy)
        RETURN a.substance AS substance,
               a.reaction AS reaction,
               a.severity AS severity,
               a.recorded_date AS recorded
        ORDER BY CASE a.severity
                     WHEN 'severe' THEN 0
                     WHEN 'moderate' THEN 1
                     ELSE 2
                 END,
                 a.substance
        LIMIT $limit
    """,
    GraphIntent.PROCEDURES: """
        MATCH (p:Patient {id: $patient_id})-[:HAD_PROCEDURE]->(proc:Procedure)
        OPTIONAL MATCH (proc)-[:FOR_CONDITION]->(c:Condition)
        OPTIONAL MATCH (e:Encounter)-[:PERFORMED]->(proc)
        OPTIONAL MATCH (e)-[:WITH_PROVIDER]->(pr:Provider)
        RETURN proc.name AS procedure,
               proc.code AS code,
               proc.performed_date AS performed,
               c.display AS for_condition,
               pr.name AS clinician
        ORDER BY proc.performed_date DESC
        LIMIT $limit
    """,
    # Earliest diagnosis per condition: a chronic problem is restated at
    # every visit for it, and "when was I diagnosed?" means the first time,
    # not the most recent mention.
    GraphIntent.DIAGNOSIS_HISTORY: """
        MATCH (p:Patient {id: $patient_id})-[:DIAGNOSED]->(d:Diagnosis)
        MATCH (d)-[:OF_CONDITION]->(c:Condition)
        OPTIONAL MATCH (e:Encounter)-[:MADE_DIAGNOSIS]->(d)
        OPTIONAL MATCH (e)-[:WITH_PROVIDER]->(pr:Provider)
        WITH c,
             min(d.diagnosed_date) AS first_recorded,
             count(d) AS times_recorded,
             collect(DISTINCT pr.name) AS clinicians,
             collect(DISTINCT d.code) AS codes
        RETURN c.display AS condition,
               first_recorded AS diagnosed,
               times_recorded AS recorded_at_visits,
               clinicians AS clinicians,
               codes AS codes
        ORDER BY first_recorded
        LIMIT $limit
    """,
}

#: Cypher keywords that would make a template something other than a
#: read-only traversal. Checked by test, so a future edit that adds one
#: fails the suite rather than shipping.
FORBIDDEN_KEYWORDS: Final = (
    "CREATE",
    "MERGE",
    "SET",
    "DELETE",
    "REMOVE",
    "DROP",
    "CALL",
    "LOAD CSV",
)

#: The anchor every template must contain (whitespace-normalised).
PATIENT_ANCHOR: Final = "(p:Patient {id: $patient_id})"
