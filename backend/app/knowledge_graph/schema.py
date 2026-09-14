"""The shape of the projection: labels, relationships, and constraints.

PRD §17 names the entities and edges; this is that list made executable, in
one place, so the builder and the queries cannot drift apart.

Two things about the entity list.

*Procedure, Allergy and Diagnosis were added to PostgreSQL first.* §17 lists
them among the graph's entities and §32's schema had no table for any of
them. Projecting them anyway would have meant inventing clinical facts in
Neo4j, which is precisely what §33 forbids ("do not maintain independent
business truth in Neo4j"), so migration 0011 made them real in the system of
record and the projection follows.

*Department is derived, not stored.* ``providers.specialty`` is the only
departmental fact in the record, so ``Department`` is that value promoted to
a node. It is a projection of an existing column, not a new fact.

Every node carries ``patient_id`` — including the ones that look global, like
``Medication``. That is deliberate and is the whole of the authorization
design: see :mod:`app.knowledge_graph.queries`.
"""

from __future__ import annotations

from typing import Final

# --- node labels -------------------------------------------------------- #

PATIENT: Final = "Patient"
PROVIDER: Final = "Provider"
ENCOUNTER: Final = "Encounter"
CONDITION: Final = "Condition"
MEDICATION: Final = "Medication"
LAB: Final = "Lab"
LAB_RESULT: Final = "LabResult"
DEPARTMENT: Final = "Department"
PROCEDURE: Final = "Procedure"
ALLERGY: Final = "Allergy"
#: The coded assertion made at a visit, distinct from ``Condition``, which is
#: the entry on the problem list. A problem list entry has no author; a
#: diagnosis has a date, a code and an encounter.
DIAGNOSIS: Final = "Diagnosis"

NODE_LABELS: Final = (
    PATIENT,
    PROVIDER,
    ENCOUNTER,
    CONDITION,
    MEDICATION,
    LAB,
    LAB_RESULT,
    DEPARTMENT,
    PROCEDURE,
    ALLERGY,
    DIAGNOSIS,
)

# --- relationship types -------------------------------------------------- #

HAD_ENCOUNTER: Final = "HAD_ENCOUNTER"
HAS_CONDITION: Final = "HAS_CONDITION"
TAKES: Final = "TAKES"
HAS_LAB_RESULT: Final = "HAS_LAB_RESULT"
WITH_PROVIDER: Final = "WITH_PROVIDER"
FOR_CONDITION: Final = "FOR_CONDITION"
PRESCRIBED: Final = "PRESCRIBED"
ORDERED: Final = "ORDERED"
#: Not in §17's list, and load-bearing. §17 reaches a condition's drugs by
#: joining through the encounter, but one visit routinely starts therapy for
#: several conditions while carrying a single condition of its own — so that
#: path pairs every drug started that day with whichever problem the visit
#: was filed under ("Essential hypertension → Metformin", observed on the
#: demo patient). TREATS is the direct, true edge; the indirect one is kept
#: for the questions it does answer correctly, such as what happened at a
#: particular visit.
TREATS: Final = "TREATS"
#: Connects a result to the test it is an instance of, so "my HbA1c results"
#: is one traversal rather than a string match over every result.
OF_LAB: Final = "OF_LAB"
IN_DEPARTMENT: Final = "IN_DEPARTMENT"

HAD_PROCEDURE: Final = "HAD_PROCEDURE"
#: Encounter -> Procedure. The patient edge above answers "what have I had
#: done"; this one answers "what happened at that visit".
PERFORMED: Final = "PERFORMED"
HAS_ALLERGY: Final = "HAS_ALLERGY"
DIAGNOSED: Final = "DIAGNOSED"
#: Encounter -> Diagnosis, and Diagnosis -> Condition. Two edges rather than
#: one so the coded assertion is a node with its own date and code, which is
#: what lets "when was I diagnosed, and by whom?" be a traversal instead of a
#: property lookup on an edge.
MADE_DIAGNOSIS: Final = "MADE_DIAGNOSIS"
OF_CONDITION: Final = "OF_CONDITION"

RELATIONSHIP_TYPES: Final = (
    HAD_ENCOUNTER,
    HAS_CONDITION,
    TAKES,
    HAS_LAB_RESULT,
    WITH_PROVIDER,
    FOR_CONDITION,
    PRESCRIBED,
    ORDERED,
    TREATS,
    OF_LAB,
    IN_DEPARTMENT,
    HAD_PROCEDURE,
    PERFORMED,
    HAS_ALLERGY,
    DIAGNOSED,
    MADE_DIAGNOSIS,
    OF_CONDITION,
)

# --- constraints --------------------------------------------------------- #

#: Uniqueness on every node's key, so ``MERGE`` updates rather than
#: duplicates and a rebuild is idempotent (PRD §33). Without these a second
#: ``build_kg.py`` run silently doubles every node, and a count drawn from
#: the graph doubles with it.
#:
#: Keys are scoped the way the data is: a Patient is unique by its
#: PostgreSQL id, and so are Encounter, Medication and LabResult. Condition
#: is unique by catalogue key, Provider by id, and Lab and Department by
#: name — those three are genuinely shared, which is what lets two patients'
#: subgraphs meet at "Cardiology" without either learning of the other.
CONSTRAINTS: Final = (
    f"CREATE CONSTRAINT patient_id IF NOT EXISTS "
    f"FOR (n:{PATIENT}) REQUIRE n.id IS UNIQUE",
    f"CREATE CONSTRAINT provider_id IF NOT EXISTS "
    f"FOR (n:{PROVIDER}) REQUIRE n.id IS UNIQUE",
    f"CREATE CONSTRAINT encounter_id IF NOT EXISTS "
    f"FOR (n:{ENCOUNTER}) REQUIRE n.id IS UNIQUE",
    f"CREATE CONSTRAINT condition_key IF NOT EXISTS "
    f"FOR (n:{CONDITION}) REQUIRE n.key IS UNIQUE",
    f"CREATE CONSTRAINT medication_id IF NOT EXISTS "
    f"FOR (n:{MEDICATION}) REQUIRE n.id IS UNIQUE",
    f"CREATE CONSTRAINT lab_result_id IF NOT EXISTS "
    f"FOR (n:{LAB_RESULT}) REQUIRE n.id IS UNIQUE",
    f"CREATE CONSTRAINT lab_name IF NOT EXISTS "
    f"FOR (n:{LAB}) REQUIRE n.name IS UNIQUE",
    f"CREATE CONSTRAINT department_name IF NOT EXISTS "
    f"FOR (n:{DEPARTMENT}) REQUIRE n.name IS UNIQUE",
    f"CREATE CONSTRAINT procedure_id IF NOT EXISTS "
    f"FOR (n:{PROCEDURE}) REQUIRE n.id IS UNIQUE",
    f"CREATE CONSTRAINT allergy_id IF NOT EXISTS "
    f"FOR (n:{ALLERGY}) REQUIRE n.id IS UNIQUE",
    f"CREATE CONSTRAINT diagnosis_id IF NOT EXISTS "
    f"FOR (n:{DIAGNOSIS}) REQUIRE n.id IS UNIQUE",
)

#: Every traversal begins by selecting one patient's own nodes, so
#: ``patient_id`` is the property that decides whether a query scans the
#: patient or the graph. Indexed on the three labels that grow per patient;
#: Patient itself is covered by its uniqueness constraint.
INDEXES: Final = tuple(
    f"CREATE INDEX {label.lower()}_patient IF NOT EXISTS "
    f"FOR (n:{label}) ON (n.patient_id)"
    for label in (ENCOUNTER, MEDICATION, LAB_RESULT, PROCEDURE, ALLERGY, DIAGNOSIS)
)
