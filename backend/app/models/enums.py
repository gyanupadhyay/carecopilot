"""Controlled vocabularies.

These are plain string constants rather than PostgreSQL ``ENUM`` types:
adding a value to a native enum requires a migration and an exclusive lock,
while a ``CHECK`` constraint is cheap to alter and just as effective at
keeping the synthetic data honest. The tuples are the single source of
truth shared by the models, the Pydantic schemas, and the data generator.
"""

from __future__ import annotations

from typing import Final

APPOINTMENT_STATUSES: Final = ("scheduled", "completed", "cancelled", "no_show")
APPOINTMENT_TYPES: Final = (
    "follow_up",
    "annual_physical",
    "urgent_care",
    "specialist_consult",
    "lab_review",
    "telehealth",
)

MEDICATION_STATUSES: Final = ("active", "discontinued", "completed")

ENCOUNTER_TYPES: Final = (
    "office_visit",
    "telehealth",
    "urgent_care",
    "annual_physical",
    "specialist_consult",
)

DOCUMENT_TYPES: Final = (
    "clinical_note",
    "discharge_summary",
    "consult_note",
    "progress_note",
)
DOCUMENT_STATUSES: Final = ("draft", "final", "amended")

#: Section headings used by the section-aware chunker (PRD §19). Order is
#: the order they appear in a generated note.
NOTE_SECTIONS: Final = (
    "Chief Complaint",
    "History of Present Illness",
    "Examination",
    "Assessment",
    "Medications",
    "Plan",
)

MESSAGE_ROLES: Final = ("user", "assistant", "system")

#: The router's decision space (PRD §14). ``OUT_OF_SCOPE`` is a real answer,
#: not an error: "that is not something this assistant covers" is the right
#: response to a question about the weather, and naming it lets the graph
#: end cleanly instead of forcing a retrieval that was never going to match.
#:
#: ``KG`` is for relationship and multi-hop questions — what connects a
#: condition to a drug, who has treated it, what followed what. §18 is
#: explicit that it is a *specialised* route rather than a better RAG: a
#: question answerable from one note should not become a graph traversal.
ROUTES: Final = (
    "API",
    "RAG",
    "KG",
    "HYBRID",
    "TEXT_TO_SQL",
    "ACTION",
    "OUT_OF_SCOPE",
)

USER_ROLES: Final = ("patient", "clinician", "admin")

#: Ordered, least to most serious. A display that sorts by this says
#: something; one that sorts alphabetically puts "mild" above "severe".
ALLERGY_SEVERITIES: Final = ("mild", "moderate", "severe")

#: Whether a diagnosis was the reason for the visit or noted alongside it.
DIAGNOSIS_RANKS: Final = ("primary", "secondary")

#: Lab/vital tests the generator produces. Blood pressure is stored here as
#: two scalar rows rather than in a separate vitals table, which keeps the
#: Text-to-SQL schema (PRD §16) to one numeric-result table.
LAB_TESTS: Final = (
    "HbA1c",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "LDL Cholesterol",
    "HDL Cholesterol",
    "Triglycerides",
    "Fasting Glucose",
    "Creatinine",
    "eGFR",
    "TSH",
    "Hemoglobin",
    "Vitamin D",
)


def check_in(column: str, values: tuple[str, ...]) -> str:
    """Render a ``CHECK (col IN (...))`` expression for a constraint."""
    rendered = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({rendered})"
