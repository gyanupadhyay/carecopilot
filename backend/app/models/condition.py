"""Conditions, and which patients carry them.

These exist because of PRD §33: "the KG must be rebuildable from PostgreSQL"
and "do not maintain independent business truth in Neo4j". §17 lists
``Condition`` among the graph's entities and puts two relationships through
it — ``Patient -HAS_CONDITION->`` and ``Encounter -FOR_CONDITION->`` — so the
condition a visit was about has to be a fact in the system of record, not
something the projection infers.

It was already known and then thrown away. The generator picks a condition
from a structured catalogue, and uses it to choose the chief complaint, the
drugs, the labs and the note prose — but persisted only the prose, leaving
``encounters.reason`` as free text like "Reports occasional fatigue in the
afternoons". Recovering "this was a diabetes visit" from that sentence means
guessing, and a graph built on guesses answers relationship questions
confidently and wrongly.

``key`` rather than the display name as the natural identifier: display names
get reworded ("Type 2 diabetes mellitus" → "Type 2 diabetes"), and a rename
must not fork a patient's history into two conditions.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Date, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.patient import Patient


class Condition(Base, TimestampMixin):
    """A clinical condition in the catalogue, independent of any patient."""

    __tablename__ = "conditions"

    id: Mapped[int] = mapped_column(primary_key=True)
    #: Stable slug, e.g. "type_2_diabetes". Matches the synthetic catalogue's
    #: ``Condition.key`` so a reseed updates rows rather than duplicating them.
    key: Mapped[str] = mapped_column(String(64), unique=True)
    display: Mapped[str] = mapped_column(String(120))
    #: What patients call it: "blood pressure" for Essential hypertension,
    #: "blood sugar" for Type 2 diabetes.
    #:
    #: In the system of record rather than only in the generator, because the
    #: knowledge graph matches against them and §33 requires the projection be
    #: rebuildable from PostgreSQL. Without this column the graph answered
    #: "which visits were about my blood pressure?" with nothing: the stored
    #: display name is "Essential hypertension", and no patient says that.
    aliases: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)), default=list, server_default="{}"
    )

    patients: Mapped[list[PatientCondition]] = relationship(
        back_populates="condition", cascade="all, delete-orphan"
    )


class PatientCondition(Base, TimestampMixin):
    """That a patient has a condition — the ``HAS_CONDITION`` edge.

    A join table rather than a column on ``patients`` because the
    relationship is many-to-many: PRD §34's synthetic patients carry one or
    two conditions, and multi-hop questions ("what connects my diabetes to my
    blood pressure medication") need both present at once.
    """

    __tablename__ = "patient_conditions"
    __table_args__ = (
        # One row per patient per condition. Without this a reseed against a
        # non-empty database, or a second visit for the same problem, silently
        # doubles the edge and every count drawn from the graph.
        # Spelled with its full ``uq_`` prefix, as the other models do. The
        # naming convention has no ``%(constraint_name)s`` token for unique
        # constraints, so an explicit name is used verbatim rather than
        # prefixed — the mismatch migration 0006 had to repair for CHECKs.
        UniqueConstraint("patient_id", "condition_id", name="uq_patient_condition"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE")
    )
    condition_id: Mapped[int] = mapped_column(
        ForeignKey("conditions.id", ondelete="CASCADE")
    )
    #: The first encounter for it, which is as close to an onset date as a
    #: record of visits can honestly get. Nullable: a condition can be
    #: recorded without the visit that first raised it being in range.
    onset_date: Mapped[date | None] = mapped_column(Date)

    patient: Mapped[Patient] = relationship(back_populates="conditions")
    condition: Mapped[Condition] = relationship(back_populates="patients")
