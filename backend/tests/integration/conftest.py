"""Fixtures for tests that need a live, seeded PostgreSQL.

These skip rather than fail when the database is unreachable or empty, so
``pytest`` is still useful on a machine that has not run the bootstrap. They
are marked ``integration`` so CI can select or exclude them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.context import AuthContext
from app.db.session import AppSession
from app.models import Patient

pytestmark = pytest.mark.integration

DEMO_EXTERNAL_ID = "P001"


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    try:
        async with AppSession() as db:
            await db.execute(text("SELECT 1"))
            yield db
    except Exception as exc:  # any connection failure means "skip", not "fail"
        pytest.skip(f"PostgreSQL not available: {type(exc).__name__}")


async def _patient_by_external_id(db: AsyncSession, external_id: str) -> Patient:
    patient = await db.scalar(
        select(Patient).where(Patient.external_id == external_id)
    )
    if patient is None:
        pytest.skip(
            f"No seeded patient {external_id}; run scripts/generate_data.py --reset"
        )
    return patient


@pytest.fixture
async def demo_patient(session: AsyncSession) -> Patient:
    """``P001`` — the patient the PRD §37 demo sequence is scripted around."""
    return await _patient_by_external_id(session, DEMO_EXTERNAL_ID)


@pytest.fixture
async def other_patient(session: AsyncSession) -> Patient:
    return await _patient_by_external_id(session, "P002")


@pytest.fixture
async def demo_ctx(demo_patient: Patient) -> AuthContext:
    return AuthContext(
        user_id=1, role="patient", patient_id=demo_patient.id, request_id="itest"
    )
