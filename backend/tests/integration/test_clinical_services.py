"""Service-layer behaviour against the seeded dataset.

Each test here corresponds to a question the assistant must answer
correctly, and checks the *data path* that answers it rather than any model
output. If these pass, a wrong answer in the product is a prompting or
routing bug, not a retrieval-of-facts bug.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.context import AuthContext, AuthorizationError
from app.models import LabResult, Patient
from app.services import clinical

pytestmark = pytest.mark.integration


async def test_profile_is_scoped_to_the_session(
    session: AsyncSession, demo_ctx: AuthContext, demo_patient: Patient
) -> None:
    profile = await clinical.get_patient_profile(session, demo_ctx)
    assert profile is not None
    assert profile.id == demo_patient.id
    assert profile.external_id == "P001"


async def test_next_appointment_skips_cancelled_earlier_slot(
    session: AsyncSession, demo_ctx: AuthContext
) -> None:
    """Demo 1.

    The seeded data puts a *cancelled* appointment four days before the real
    one precisely so that a naive "soonest future appointment" query returns
    the wrong answer. This asserts the status filter does its job.
    """
    appointment = await clinical.get_next_appointment(session, demo_ctx)
    assert appointment is not None
    assert appointment.status == "scheduled"
    assert appointment.appointment_date > datetime.now(UTC)
    assert appointment.provider is not None
    assert appointment.provider.name == "Dr. Sarah Smith"

    earlier = await clinical.get_appointments(
        session, demo_ctx, from_date=datetime.now(UTC), newest_first=False
    )
    cancelled = [a for a in earlier if a.status == "cancelled"]
    assert cancelled, "expected a cancelled decoy appointment in the dataset"
    assert cancelled[0].appointment_date < appointment.appointment_date


async def test_current_medications_excludes_finished_courses(
    session: AsyncSession, demo_ctx: AuthContext
) -> None:
    """Demo 2 support: 'what am I currently taking' is not 'every row'."""
    current = await clinical.get_current_medications(session, demo_ctx)
    assert current, "demo patient should be on active therapy"

    today = date.today()
    for medication in current:
        assert medication.status == "active"
        assert medication.active_on(today)

    everything = await clinical.get_medications_in_effect(
        session, demo_ctx, on=today - timedelta(days=365)
    )
    assert len(current) != len(everything) or current != everything


async def test_last_encounter_is_the_most_recent(
    session: AsyncSession, demo_ctx: AuthContext
) -> None:
    last = await clinical.get_last_encounter(session, demo_ctx)
    assert last is not None

    everything = await clinical.get_encounters(session, demo_ctx, limit=50)
    assert last.encounter_date == max(e.encounter_date for e in everything)
    assert last.encounter_date <= date.today()


async def test_medication_diff_around_last_encounter(
    session: AsyncSession, demo_ctx: AuthContext
) -> None:
    """Demo 3: the deterministic before/after the HYBRID route reports.

    The generator scripts exactly one of each change type at this visit, so
    a drift in either the generator or the diff shows up here rather than as
    a plausible-sounding but wrong sentence in a summary.
    """
    last = await clinical.get_last_encounter(session, demo_ctx)
    assert last is not None

    before = await clinical.get_medications_in_effect(
        session, demo_ctx, on=last.encounter_date - timedelta(days=1)
    )
    after = await clinical.get_medications_in_effect(
        session, demo_ctx, on=last.encounter_date
    )
    changes = {c.name: c for c in clinical.compare_medications(before, after)}

    assert changes["Metformin"].change == "dose_changed"
    assert changes["Metformin"].before.startswith("500mg")
    assert changes["Metformin"].after.startswith("1000mg")
    assert changes["Glipizide"].change == "stopped"
    assert changes["Ibuprofen"].change == "started"

    # Lisinopril is unchanged across the visit and must not be reported.
    assert "Lisinopril" not in changes


async def test_lab_history_supports_the_analytical_question(
    session: AsyncSession, demo_ctx: AuthContext
) -> None:
    """Demo 4 needs a non-trivial answer, not zero and not everything."""
    six_months_ago = date.today() - timedelta(days=182)
    readings = await clinical.get_lab_results(
        session,
        demo_ctx,
        test_name="Systolic Blood Pressure",
        from_date=six_months_ago,
        limit=200,
    )
    assert len(readings) >= 5, "not enough systolic readings to analyse"

    above_140 = [r for r in readings if r.value > 140]
    assert 0 < len(above_140) < len(readings), (
        "expected some but not all systolic readings above 140"
    )


async def test_hba1c_series_spans_a_year(
    session: AsyncSession, demo_ctx: AuthContext
) -> None:
    results = await clinical.get_lab_results(
        session, demo_ctx, test_name="HbA1c", from_date=date.today() - timedelta(days=365)
    )
    assert len(results) >= 3
    assert all(r.test_name == "HbA1c" for r in results)


async def test_lab_filter_does_not_conflate_systolic_and_diastolic(
    session: AsyncSession, demo_ctx: AuthContext
) -> None:
    """Substring matching here would silently average two different tests."""
    results = await clinical.get_lab_results(
        session, demo_ctx, test_name="Systolic Blood Pressure", limit=200
    )
    assert results
    assert {r.test_name for r in results} == {"Systolic Blood Pressure"}


async def test_every_reader_is_scoped_to_one_patient(
    session: AsyncSession, demo_ctx: AuthContext, demo_patient: Patient
) -> None:
    """No service function may return a row belonging to anyone else."""
    appointments = await clinical.get_appointments(session, demo_ctx, limit=200)
    medications = await clinical.get_current_medications(session, demo_ctx)
    labs = await clinical.get_lab_results(session, demo_ctx, limit=200)
    encounters = await clinical.get_encounters(session, demo_ctx, limit=200)

    for rows in (appointments, medications, labs, encounters):
        assert rows, "fixture patient should have data in every table"
        assert {r.patient_id for r in rows} == {demo_patient.id}


async def test_service_refuses_a_context_with_no_patient(
    session: AsyncSession,
) -> None:
    unlinked = AuthContext(user_id=99, role="clinician", patient_id=None)
    with pytest.raises(AuthorizationError):
        await clinical.get_appointments(session, unlinked)


async def test_one_patients_context_cannot_reach_another(
    session: AsyncSession, demo_ctx: AuthContext, other_patient: Patient
) -> None:
    """P001 cannot access P002 (PRD §38, Demo 5)."""
    with pytest.raises(AuthorizationError):
        demo_ctx.assert_patient(other_patient.id)

    labs = await clinical.get_lab_results(session, demo_ctx, limit=200)
    assert other_patient.id not in {r.patient_id for r in labs}


async def test_results_are_bounded(
    session: AsyncSession, demo_ctx: AuthContext
) -> None:
    """An unbounded result set is a context-window bug waiting to happen."""
    total = await session.scalar(
        select(func.count())
        .select_from(LabResult)
        .where(LabResult.patient_id == demo_ctx.patient_scope)
    )
    assert total > clinical.MAX_ROWS or total > 0

    rows = await clinical.get_lab_results(session, demo_ctx, limit=10_000)
    assert len(rows) <= clinical.MAX_ROWS
