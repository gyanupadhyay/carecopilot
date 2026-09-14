"""Procedures, allergies and diagnoses.

PRD §17 lists all three among the knowledge graph's entities, and §32's
PostgreSQL schema had no table that could source any of them. Projecting them
from nothing would mean inventing clinical facts inside Neo4j, which §33
forbids — so the gap closes here, in the system of record, and the graph
projects what it finds.

The three are genuinely different shapes, which is why they are three tables
and not one:

*Procedure* is an event at a visit — something done, on a date, by whoever
the encounter was with. It hangs off an encounter.

*Allergy* belongs to the patient and to no visit. It is the one clinical fact
here that is not anchored in time by an encounter, which is exactly why it
cannot be inferred from the encounter timeline.

*Diagnosis* is the coded assertion that a patient has a condition, made at a
visit. It is **not** a duplicate of ``PatientCondition``: that row says the
problem is on the patient's list, this one says a clinician recorded it on a
particular day with a particular code. A problem list entry has no author; a
diagnosis does. Keeping them apart is what lets the graph answer "when was I
diagnosed with this, and by whom?" separately from "what do I have?".
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Date, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.models.enums import ALLERGY_SEVERITIES, check_in

if TYPE_CHECKING:
    from app.models.condition import Condition
    from app.models.encounter import Encounter
    from app.models.patient import Patient
    from app.models.provider import Provider


class Procedure(Base, TimestampMixin):
    """Something performed at a visit."""

    __tablename__ = "procedures"
    __table_args__ = (
        Index("ix_procedures_patient_id_performed_date", "patient_id", "performed_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE")
    )
    #: The visit it happened at. Nullable so a procedure recorded outside the
    #: encounter history is still representable rather than unstorable.
    encounter_id: Mapped[int | None] = mapped_column(
        ForeignKey("encounters.id", ondelete="SET NULL")
    )
    provider_id: Mapped[int | None] = mapped_column(
        ForeignKey("providers.id", ondelete="SET NULL")
    )
    #: What it was for, when it was prompted by a catalogued problem.
    condition_id: Mapped[int | None] = mapped_column(
        ForeignKey("conditions.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(160))
    #: CPT-shaped. Short because a code that needs 64 characters is prose.
    code: Mapped[str | None] = mapped_column(String(16))
    performed_date: Mapped[date] = mapped_column(Date)

    patient: Mapped[Patient] = relationship(back_populates="procedures")
    encounter: Mapped[Encounter | None] = relationship()
    provider: Mapped[Provider | None] = relationship()
    condition: Mapped[Condition | None] = relationship()


class Allergy(Base, TimestampMixin):
    """A substance the patient reacts to."""

    __tablename__ = "allergies"
    __table_args__ = (
        CheckConstraint(
            check_in("severity", ALLERGY_SEVERITIES), name="allergy_severity"
        ),
        # One row per substance per patient. Two rows for penicillin is not a
        # second allergy, it is the same allergy counted twice — and an
        # allergy list that over-reports is the kind of wrong that changes
        # what a clinician prescribes.
        UniqueConstraint("patient_id", "substance", name="uq_patient_allergy"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE")
    )
    substance: Mapped[str] = mapped_column(String(120))
    reaction: Mapped[str | None] = mapped_column(String(200))
    severity: Mapped[str] = mapped_column(String(16))
    recorded_date: Mapped[date | None] = mapped_column(Date)

    patient: Mapped[Patient] = relationship(back_populates="allergies")


class Diagnosis(Base, TimestampMixin):
    """A coded condition recorded at a visit."""

    __tablename__ = "diagnoses"
    __table_args__ = (
        # A visit may restate a diagnosis it has already made, but not twice
        # in the same row set — the graph counts these.
        UniqueConstraint("encounter_id", "condition_id", name="uq_encounter_diagnosis"),
        Index("ix_diagnoses_patient_id_diagnosed_date", "patient_id", "diagnosed_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE")
    )
    encounter_id: Mapped[int] = mapped_column(
        ForeignKey("encounters.id", ondelete="CASCADE")
    )
    condition_id: Mapped[int] = mapped_column(
        ForeignKey("conditions.id", ondelete="CASCADE")
    )
    #: ICD-10-shaped, copied from the condition catalogue at the time of
    #: recording. Denormalised on purpose: a code that is corrected in the
    #: catalogue must not silently rewrite what a clinician recorded years
    #: ago, which is the whole reason coded records keep their own copy.
    code: Mapped[str | None] = mapped_column(String(16))
    diagnosed_date: Mapped[date] = mapped_column(Date)
    #: "primary" when it was the reason for the visit, "secondary" otherwise.
    rank: Mapped[str] = mapped_column(String(16), default="primary")

    patient: Mapped[Patient] = relationship(back_populates="diagnoses")
    encounter: Mapped[Encounter] = relationship()
    condition: Mapped[Condition] = relationship()
