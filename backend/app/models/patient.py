"""Patient demographics.

All values are synthetic. ``external_id`` is the human-facing identifier
("P001") used in demo logins and eval fixtures; ``id`` is the surrogate key
every other table references and the value the authorization layer pins.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Date, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.appointment import Appointment
    from app.models.clinical_document import ClinicalDocument
    from app.models.clinical_events import Allergy, Diagnosis, Procedure
    from app.models.condition import PatientCondition
    from app.models.encounter import Encounter
    from app.models.lab_result import LabResult
    from app.models.medication import Medication


class Patient(Base, TimestampMixin):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(32), unique=True)
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    date_of_birth: Mapped[date] = mapped_column(Date)
    gender: Mapped[str] = mapped_column(String(32))

    appointments: Mapped[list[Appointment]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    encounters: Mapped[list[Encounter]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    medications: Mapped[list[Medication]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    lab_results: Mapped[list[LabResult]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    documents: Mapped[list[ClinicalDocument]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    conditions: Mapped[list[PatientCondition]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    procedures: Mapped[list[Procedure]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    allergies: Mapped[list[Allergy]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    diagnoses: Mapped[list[Diagnosis]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
