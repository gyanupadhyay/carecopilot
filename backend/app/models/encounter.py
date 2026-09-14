"""A clinical encounter: the anchor that ties notes, meds and labs together.

``encounter_date`` is a plain date. Encounters are the unit the HYBRID route
pivots on ("summarize my last visit"), so the (patient_id, encounter_date)
index exists to make "latest encounter" a single index scan.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Date, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.models.enums import ENCOUNTER_TYPES, check_in

if TYPE_CHECKING:
    from app.models.clinical_document import ClinicalDocument
    from app.models.condition import Condition
    from app.models.medication import Medication
    from app.models.patient import Patient
    from app.models.provider import Provider


class Encounter(Base, TimestampMixin):
    __tablename__ = "encounters"
    __table_args__ = (
        CheckConstraint(
            check_in("encounter_type", ENCOUNTER_TYPES), name="encounter_type"
        ),
        Index("ix_encounters_patient_id_encounter_date", "patient_id", "encounter_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE")
    )
    provider_id: Mapped[int | None] = mapped_column(
        ForeignKey("providers.id", ondelete="SET NULL")
    )
    #: What the visit was about, as a catalogue reference rather than prose.
    #: ``reason`` below is the chief complaint in the patient's words; this is
    #: the clinical problem behind it, and it is what ``FOR_CONDITION``
    #: projects into the graph (PRD §17). Nullable because not every visit has
    #: one — an acute infection or a routine check need not.
    condition_id: Mapped[int | None] = mapped_column(
        ForeignKey("conditions.id", ondelete="SET NULL")
    )
    encounter_date: Mapped[date] = mapped_column(Date)
    encounter_type: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str | None] = mapped_column(String(200))

    patient: Mapped[Patient] = relationship(back_populates="encounters")
    provider: Mapped[Provider | None] = relationship()
    condition: Mapped[Condition | None] = relationship()
    documents: Mapped[list[ClinicalDocument]] = relationship(
        back_populates="encounter", cascade="all, delete-orphan"
    )
    medications: Mapped[list[Medication]] = relationship(back_populates="encounter")
