"""User-to-patient identity mapping (PRD §9, §10).

The mapping table is the authorization fact. These tests pin the two
properties that make it worth being a table rather than a column: scope is
resolved *through* it on every request, and revoking a row takes access away
without touching code or redeploying.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.auth.demo import DEMO_PASSWORD
from app.main import create_app
from app.models import Patient, User, UserPatientMapping

pytestmark = pytest.mark.integration

BASE = "http://test/api"


@pytest.fixture
async def client(session):  # session fixture forces the DB-availability skip
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        yield http


async def _user(session, external_id: str) -> User:
    patient = await session.scalar(
        select(Patient).where(Patient.external_id == external_id)
    )
    if patient is None:
        pytest.skip(f"{external_id} not seeded")
    user = await session.scalar(
        select(User)
        .join(UserPatientMapping, UserPatientMapping.user_id == User.id)
        .where(UserPatientMapping.patient_id == patient.id)
    )
    if user is None:
        pytest.skip(f"No user mapped to {external_id}")
    return user


async def test_every_seeded_patient_account_has_exactly_one_grant(session) -> None:
    user = await _user(session, "P001")
    links = list(user.patient_links)
    assert len(links) == 1
    assert links[0].is_active
    assert links[0].relationship_type == "self"


async def test_scope_is_resolved_through_the_mapping(session) -> None:
    """``User.patient_id`` is derived, not stored."""
    user = await _user(session, "P001")
    assert user.patient_id == user.patient_links[0].patient_id
    assert user.patient is not None
    assert user.patient.external_id == "P001"


async def test_an_account_with_no_grant_has_no_scope(session) -> None:
    """A user row alone confers nothing — access is the mapping, not the row."""
    orphan = User(
        email="unmapped@carecopilot.demo",
        display_name="Unmapped Account",
        password_hash="x",
        role="clinician",
        is_active=True,
        # Set explicitly: an unpopulated collection would lazy-load on first
        # access, which is synchronous IO inside an async session.
        patient_links=[],
    )
    session.add(orphan)
    await session.flush()
    try:
        assert orphan.patient_id is None
        assert orphan.patient is None
    finally:
        await session.rollback()


async def test_revoking_the_grant_removes_scope(session) -> None:
    """Deactivating the row is enough; no code change, no redeploy."""
    user = await _user(session, "P001")
    link = user.patient_links[0]
    assert user.patient_id is not None

    link.is_active = False
    await session.flush()
    await session.refresh(user, ["patient_links"])

    try:
        assert user.patient_id is None, "an inactive grant confers no scope"
    finally:
        await session.rollback()


async def test_a_revoked_account_cannot_read_records(
    client: AsyncClient, session
) -> None:
    """End to end: the revocation reaches the HTTP surface as a 403.

    The account still authenticates — it is a valid user — but it can no
    longer resolve a patient, so every record endpoint refuses it.
    """
    user = await _user(session, "P001")
    login = {"email": user.email, "password": DEMO_PASSWORD}

    before = await client.post(f"{BASE}/auth/login", json=login)
    assert before.status_code == 200
    token = before.json()["access_token"]
    auth = {"Authorization": f"Bearer {token}"}
    assert (await client.get(f"{BASE}/labs", headers=auth)).status_code == 200

    link = user.patient_links[0]
    link.is_active = False
    await session.commit()

    try:
        # The existing token is still validly signed and unexpired — scope is
        # re-resolved per request, so it stops working immediately.
        assert (await client.get(f"{BASE}/labs", headers=auth)).status_code == 403
        assert (await client.get(f"{BASE}/me", headers=auth)).status_code == 403

        after = await client.post(f"{BASE}/auth/login", json=login)
        assert after.status_code == 200, "the account itself is still valid"
        assert after.json()["user"]["patient_id"] is None
    finally:
        link.is_active = True
        await session.commit()
