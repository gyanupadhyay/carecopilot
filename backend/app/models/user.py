"""Application users and the mapping from a user to the patient they may read.

The mapping is a table rather than a column on ``users`` (PRD §9, §10), and
the reason is design intent rather than normalization. A foreign key on the
user row says "a user has a patient". A mapping table says "a user's access
to a patient is a fact the backend records, with a grant reason and an
issue date, and it can be revoked" — which is what authorization actually
is. It also means the day a clinician needs access to several patients, the
model already expresses that; a column would have to be migrated under
pressure.

Either shape enforces the same rule today. The difference is that this one
makes the rule visible in the schema instead of implied by a nullable
column, so a reader can see where patient scope comes from without reading
the service layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.models.enums import USER_ROLES, check_in

if TYPE_CHECKING:
    from app.models.patient import Patient


class User(Base, TimestampMixin):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(check_in("role", USER_ROLES), name="user_role"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(200), unique=True)
    display_name: Mapped[str] = mapped_column(String(120))
    #: Argon2id hash. Never a plaintext or reversible value.
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="patient")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    patient_links: Mapped[list[UserPatientMapping]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    @property
    def patient_id(self) -> int | None:
        """The single patient this account may read, if any.

        A patient account has exactly one active link; the property exists so
        that call sites reading a scope do not have to know the mapping is a
        collection. Inactive links are ignored, which is what makes revoking
        access a data change rather than a deployment.
        """
        for link in self.patient_links:
            if link.is_active:
                return link.patient_id
        return None

    @property
    def patient(self) -> Patient | None:
        for link in self.patient_links:
            if link.is_active:
                return link.patient
        return None


class UserPatientMapping(Base, TimestampMixin):
    """One user's authorized access to one patient's records.

    This row is the authorization fact. ``AuthContext.patient_id`` is derived
    from it and from nothing else — not from a request parameter, not from a
    tool argument, not from conversation history (PRD §10, §26, §40 P3).
    """

    __tablename__ = "user_patient_mapping"
    __table_args__ = (
        # A user may not hold the same patient twice; revocation flips the
        # flag rather than inserting a competing row.
        UniqueConstraint("user_id", "patient_id", name="uq_user_patient"),
        Index("ix_user_patient_mapping_user_id_is_active", "user_id", "is_active"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE")
    )
    #: Why the grant exists — "self" for a patient reading their own record.
    #: Recorded so an audit can answer "on what basis?" without inference.
    relationship_type: Mapped[str] = mapped_column(String(32), default="self")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    user: Mapped[User] = relationship(back_populates="patient_links")
    patient: Mapped[Patient] = relationship(lazy="joined")
