"""Medication orders.

A medication row is a *state over an interval*, not an event: ``start_date``
and ``end_date`` (NULL while active) are what make the deterministic
before/after comparison in the HYBRID route possible without asking the LLM
to reason about dates. ``encounter_id`` records which visit changed it.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Date, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.models.enums import MEDICATION_STATUSES, check_in

if TYPE_CHECKING:
    from app.models.condition import Condition
    from app.models.encounter import Encounter
    from app.models.patient import Patient


class Medication(Base, TimestampMixin):
    __tablename__ = "medications"
    __table_args__ = (
        CheckConstraint(
            check_in("status", MEDICATION_STATUSES), name="medication_status"
        ),
        CheckConstraint(
            "end_date IS NULL OR end_date >= start_date", name="medication_date_order"
        ),
        Index("ix_medications_patient_id_status", "patient_id", "status"),
        Index("ix_medications_patient_id_start_date", "patient_id", "start_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE")
    )
    encounter_id: Mapped[int | None] = mapped_column(
        ForeignKey("encounters.id", ondelete="SET NULL")
    )
    #: What this drug treats — the ``TREATS`` edge in the graph (PRD §17).
    #:
    #: Recorded directly rather than inferred by joining through
    #: ``encounter.condition_id``, because that inference is wrong. One visit
    #: routinely starts therapy for several conditions at once, and the visit
    #: carries a single condition, so the join pairs every drug started that
    #: day with whichever problem the visit was filed under — it produced
    #: "Essential hypertension → Metformin" on the demo patient. A graph that
    #: answers "what treats my diabetes?" from that join is confidently wrong,
    #: which is worse than having no graph.
    condition_id: Mapped[int | None] = mapped_column(
        ForeignKey("conditions.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(120))
    dosage: Mapped[str] = mapped_column(String(64))
    frequency: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="active")
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)

    patient: Mapped[Patient] = relationship(back_populates="medications")
    encounter: Mapped[Encounter | None] = relationship(back_populates="medications")
    condition: Mapped[Condition | None] = relationship()

    def active_on(self, when: date) -> bool:
        """True if this order was in force on ``when``."""
        if self.start_date > when:
            return False
        return self.end_date is None or self.end_date >= when
