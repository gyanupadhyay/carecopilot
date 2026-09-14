"""Read access to structured clinical data.

Two rules hold throughout this module:

* the patient filter comes from ``ctx.patient_scope`` and nothing else — no
  function takes a ``patient_id`` argument, so no caller (and no tool schema
  the model can see) is able to supply one;
* every query is bounded. A result set with no ``LIMIT`` eventually becomes
  a context-window problem, a latency problem, or both.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth.context import AuthContext
from app.models import Appointment, Encounter, LabResult, Medication, Patient
from app.schemas.clinical import MedicationChange

#: Hard ceiling applied to every list endpoint regardless of what the caller
#: asks for. Callers may request less; nothing may request more.
MAX_ROWS = 200


def _bounded(limit: int | None, default: int) -> int:
    if limit is None:
        return default
    return max(1, min(limit, MAX_ROWS))


def _scoped(stmt: Select, column, ctx: AuthContext) -> Select:
    """Attach the session's patient filter to a statement."""
    return stmt.where(column == ctx.patient_scope)


async def get_patient_profile(session: AsyncSession, ctx: AuthContext) -> Patient | None:
    stmt = select(Patient).where(Patient.id == ctx.patient_scope)
    return await session.scalar(stmt)


async def get_appointments(
    session: AsyncSession,
    ctx: AuthContext,
    *,
    status: str | None = None,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    limit: int | None = None,
    newest_first: bool = True,
) -> list[Appointment]:
    stmt = _scoped(
        select(Appointment).options(selectinload(Appointment.provider)),
        Appointment.patient_id,
        ctx,
    )
    if status:
        stmt = stmt.where(Appointment.status == status)
    if from_date:
        stmt = stmt.where(Appointment.appointment_date >= from_date)
    if to_date:
        stmt = stmt.where(Appointment.appointment_date <= to_date)

    order = (
        Appointment.appointment_date.desc()
        if newest_first
        else Appointment.appointment_date.asc()
    )
    stmt = stmt.order_by(order).limit(_bounded(limit, 50))
    return list((await session.scalars(stmt)).all())


async def get_next_appointment(
    session: AsyncSession, ctx: AuthContext, *, now: datetime | None = None
) -> Appointment | None:
    """The soonest scheduled appointment at or after ``now``.

    Cancelled and completed rows are excluded here rather than left for the
    model to filter: "your next appointment" must never resolve to a visit
    the patient already cancelled.
    """
    reference = now or datetime.now(UTC)
    stmt = (
        _scoped(
            select(Appointment).options(selectinload(Appointment.provider)),
            Appointment.patient_id,
            ctx,
        )
        .where(
            Appointment.appointment_date >= reference,
            Appointment.status == "scheduled",
        )
        .order_by(Appointment.appointment_date.asc())
        .limit(1)
    )
    return await session.scalar(stmt)


async def get_current_medications(
    session: AsyncSession, ctx: AuthContext, *, as_of: date | None = None
) -> list[Medication]:
    """Medications in force on ``as_of`` (default: today).

    "Current" is evaluated against the date interval, not only the status
    column, so a row left as ``active`` with a past ``end_date`` cannot be
    reported as something the patient is still taking.
    """
    when = as_of or date.today()
    stmt = (
        _scoped(select(Medication), Medication.patient_id, ctx)
        .where(
            Medication.start_date <= when,
            Medication.status == "active",
            (Medication.end_date.is_(None)) | (Medication.end_date >= when),
        )
        .order_by(Medication.name.asc())
        .limit(MAX_ROWS)
    )
    return list((await session.scalars(stmt)).all())


async def get_medications_in_effect(
    session: AsyncSession, ctx: AuthContext, *, on: date
) -> list[Medication]:
    """Every order in force on a given date, whatever its current status.

    Unlike :func:`get_current_medications` this ignores ``status``: a
    medication discontinued last week was still in effect during a visit two
    months ago, and the before/after comparison depends on that.
    """
    stmt = (
        _scoped(select(Medication), Medication.patient_id, ctx)
        .where(
            Medication.start_date <= on,
            (Medication.end_date.is_(None)) | (Medication.end_date >= on),
        )
        .order_by(Medication.name.asc())
        .limit(MAX_ROWS)
    )
    return list((await session.scalars(stmt)).all())


async def get_lab_results(
    session: AsyncSession,
    ctx: AuthContext,
    *,
    test_name: str | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
    limit: int | None = None,
) -> list[LabResult]:
    stmt = _scoped(select(LabResult), LabResult.patient_id, ctx)
    if test_name:
        # Case-insensitive exact match. Substring matching would make
        # "Blood Pressure" silently mix systolic and diastolic values.
        stmt = stmt.where(LabResult.test_name.ilike(test_name))
    if from_date:
        stmt = stmt.where(LabResult.result_date >= from_date)
    if to_date:
        stmt = stmt.where(LabResult.result_date <= to_date)

    stmt = stmt.order_by(LabResult.result_date.desc(), LabResult.id.desc()).limit(
        _bounded(limit, 100)
    )
    return list((await session.scalars(stmt)).all())


async def get_encounters(
    session: AsyncSession,
    ctx: AuthContext,
    *,
    from_date: date | None = None,
    to_date: date | None = None,
    limit: int | None = None,
) -> list[Encounter]:
    stmt = _scoped(
        select(Encounter).options(selectinload(Encounter.provider)),
        Encounter.patient_id,
        ctx,
    )
    if from_date:
        stmt = stmt.where(Encounter.encounter_date >= from_date)
    if to_date:
        stmt = stmt.where(Encounter.encounter_date <= to_date)

    stmt = stmt.order_by(Encounter.encounter_date.desc(), Encounter.id.desc()).limit(
        _bounded(limit, 20)
    )
    return list((await session.scalars(stmt)).all())


async def get_last_encounter(
    session: AsyncSession, ctx: AuthContext
) -> Encounter | None:
    """The most recent encounter — the anchor for the HYBRID route."""
    encounters = await get_encounters(session, ctx, limit=1)
    return encounters[0] if encounters else None


def compare_medications(
    before: list[Medication], after: list[Medication]
) -> list[MedicationChange]:
    """Diff two medication lists deterministically.

    This is the function that answers "which medications changed", and it is
    plain Python on purpose (PRD §40 P12). The LLM receives the finished diff as
    a fact to narrate, not two lists to reason over.

    Medications are matched by name, case-insensitively. A name present only
    in ``after`` was started; only in ``before``, stopped; in both with a
    different dosage or frequency, changed.
    """
    by_name_before = {m.name.strip().lower(): m for m in before}
    by_name_after = {m.name.strip().lower(): m for m in after}

    changes: list[MedicationChange] = []
    for key in sorted(set(by_name_before) | set(by_name_after)):
        prior = by_name_before.get(key)
        current = by_name_after.get(key)
        name = (current or prior).name  # type: ignore[union-attr]

        if prior is None and current is not None:
            changes.append(
                MedicationChange(
                    change="started", name=name, after=_describe(current)
                )
            )
        elif current is None and prior is not None:
            changes.append(
                MedicationChange(change="stopped", name=name, before=_describe(prior))
            )
        elif _describe(prior) != _describe(current):
            changes.append(
                MedicationChange(
                    change="dose_changed",
                    name=name,
                    before=_describe(prior),
                    after=_describe(current),
                )
            )
    return changes


def _describe(medication: Medication) -> str:
    return f"{medication.dosage} {medication.frequency}".strip()
