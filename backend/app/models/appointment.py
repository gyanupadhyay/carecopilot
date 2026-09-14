"""Scheduled visits.

``appointment_date`` is timezone-aware: "your next appointment is at 10:00"
is only meaningful with an instant, and the booking tool needs to detect
conflicts across a real timeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.models.enums import APPOINTMENT_STATUSES, APPOINTMENT_TYPES, check_in

if TYPE_CHECKING:
    from app.models.patient import Patient
    from app.models.provider import Provider


class Appointment(Base, TimestampMixin):
    __tablename__ = "appointments"
    __table_args__ = (
        CheckConstraint(
            check_in("status", APPOINTMENT_STATUSES), name="appointment_status"
        ),
        CheckConstraint(
            check_in("appointment_type", APPOINTMENT_TYPES), name="appointment_type"
        ),
        Index(
            "ix_appointments_patient_id_appointment_date",
            "patient_id",
            "appointment_date",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE")
    )
    provider_id: Mapped[int | None] = mapped_column(
        ForeignKey("providers.id", ondelete="SET NULL")
    )
    appointment_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    appointment_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="scheduled")
    notes: Mapped[str | None] = mapped_column(Text)

    patient: Mapped[Patient] = relationship(back_populates="appointments")
    provider: Mapped[Provider | None] = relationship()
