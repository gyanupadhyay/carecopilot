"""Numeric results: laboratory panels and vitals.

Blood pressure lives here as two scalar tests ("Systolic Blood Pressure",
"Diastolic Blood Pressure") instead of a separate vitals table. That keeps
every analytical question in the PRD ("average HbA1c", "systolic above 140")
answerable from a single table, which in turn keeps the semantic schema
handed to the Text-to-SQL generator small and the join budget low.

``value`` is ``Numeric`` rather than float so that averages returned to the
user are exact decimals and not float artifacts.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Date, ForeignKey, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.encounter import Encounter
    from app.models.patient import Patient


class LabResult(Base, TimestampMixin):
    __tablename__ = "lab_results"
    __table_args__ = (
        Index(
            "ix_lab_results_patient_id_test_name_result_date",
            "patient_id",
            "test_name",
            "result_date",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE")
    )
    encounter_id: Mapped[int | None] = mapped_column(
        ForeignKey("encounters.id", ondelete="SET NULL")
    )
    test_name: Mapped[str] = mapped_column(String(120))
    value: Mapped[Decimal] = mapped_column(Numeric(10, 3))
    unit: Mapped[str] = mapped_column(String(32))
    reference_range: Mapped[str | None] = mapped_column(String(64))
    result_date: Mapped[date] = mapped_column(Date)

    patient: Mapped[Patient] = relationship(back_populates="lab_results")
    encounter: Mapped[Encounter | None] = relationship()
