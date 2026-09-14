"""The confirm endpoint, end to end (PRD §15, §26).

The propose half is exercised through the service; the confirm half through
a real HTTP request with a real login, because the binding between token and
session is the property under test and a direct service call would not
exercise it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.actions.appointments import BOOK, AppointmentRequest, propose
from app.agents.router import RouteDecision
from app.api.deps import embedding_provider
from app.api.routes.chat import provider as chat_provider
from app.auth.context import AuthContext
from app.auth.demo import DEMO_PASSWORD
from app.llm.base import StructuredResponse, TokenUsage
from app.llm.stub import StubProvider
from app.main import create_app
from app.rag.embeddings import build_embedder

pytestmark = pytest.mark.integration

BASE = "http://test/api"


@pytest.fixture(scope="module")
def embedder():
    return build_embedder(provider="local")


#: Marks every row these tests create, so teardown can remove exactly them.
TEST_REQUEST_ID = "actions-itest"


@pytest.fixture
async def client(session, embedder):
    app = create_app()
    app.dependency_overrides[embedding_provider] = lambda: embedder
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        yield http
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
async def clean_up_bookings(session):
    """Remove the appointments and audit rows each test creates.

    These tests really write, against the shared development database. Left
    behind, the rows make the suite fail on its second run — the clash check
    correctly refuses to book a slot the previous run already took. That is
    the feature working; the test was the thing at fault.
    """
    baseline = await session.scalar(
        text("SELECT COALESCE(MAX(id), 0) FROM appointments")
    )
    yield
    await session.rollback()
    # Two kinds of audit row, with two different request ids. The proposal
    # is made by this module under TEST_REQUEST_ID; the execution is audited
    # by the confirm endpoint under the HTTP request's own generated id, so
    # it has to be found through the appointment it points at.
    await session.execute(
        text(
            "DELETE FROM audit_logs WHERE request_id = :r "
            "OR (target_type = 'appointment' AND target_id > :b)"
        ),
        {"r": TEST_REQUEST_ID, "b": baseline},
    )
    await session.execute(
        text("DELETE FROM appointments WHERE id > :b"), {"b": baseline}
    )
    await session.commit()


async def _login(client: AsyncClient, external_id: str) -> dict[str, str]:
    response = await client.post(
        f"{BASE}/auth/login",
        json={
            "email": f"{external_id.lower()}@carecopilot.demo",
            "password": DEMO_PASSWORD,
        },
    )
    if response.status_code != 200:
        pytest.skip(f"{external_id} not seeded; run scripts/generate_data.py --reset")
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


async def _identity(client: AsyncClient, headers: dict[str, str]) -> tuple[int, int]:
    """The user and patient ids behind a logged-in session."""
    me = (await client.get(f"{BASE}/me", headers=headers)).json()
    import jwt as pyjwt

    claims = pyjwt.decode(
        headers["Authorization"].removeprefix("Bearer "),
        options={"verify_signature": False},
    )
    return int(claims["sub"]), int(me["id"])


async def _propose_booking(
    session, user_id: int, patient_id: int, *, days_ahead: int = 21
) -> str:
    """Make a real proposal and return its wire token."""
    ctx = AuthContext(
        user_id=user_id,
        role="patient",
        patient_id=patient_id,
        request_id=TEST_REQUEST_ID,
    )
    when = (datetime.now(UTC) + timedelta(days=days_ahead)).replace(
        hour=11, minute=0, second=0, microsecond=0
    )
    proposal = await propose(
        session,
        ctx,
        AppointmentRequest(
            action=BOOK, when=when.isoformat(), appointment_type="follow_up"
        ),
    )
    await session.commit()
    return proposal.token


# --- authentication ------------------------------------------------------ #


async def test_confirm_requires_a_session(client: AsyncClient) -> None:
    response = await client.post(f"{BASE}/actions/confirm", json={"token": "anything"})
    assert response.status_code == 401


async def test_a_garbage_token_is_refused(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(
        f"{BASE}/actions/confirm", json={"token": "not-a-token"}, headers=auth
    )
    assert response.status_code == 400


@pytest.fixture
async def auth(client: AsyncClient) -> dict[str, str]:
    return await _login(client, "P001")


# --- the happy path ------------------------------------------------------ #


async def test_confirming_a_proposal_books_the_appointment(
    client: AsyncClient, auth: dict[str, str], session
) -> None:
    user_id, patient_id = await _identity(client, auth)
    token = await _propose_booking(session, user_id, patient_id, days_ahead=23)

    before = await session.scalar(
        text("SELECT COUNT(*) FROM appointments WHERE patient_id = :p"),
        {"p": patient_id},
    )
    response = await client.post(
        f"{BASE}/actions/confirm", json={"token": token}, headers=auth
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "executed"
    assert body["appointment_id"]

    after = await session.scalar(
        text("SELECT COUNT(*) FROM appointments WHERE patient_id = :p"),
        {"p": patient_id},
    )
    assert after == before + 1


async def test_the_execution_is_audited(
    client: AsyncClient, auth: dict[str, str], session
) -> None:
    user_id, patient_id = await _identity(client, auth)
    token = await _propose_booking(session, user_id, patient_id, days_ahead=25)

    await client.post(f"{BASE}/actions/confirm", json={"token": token}, headers=auth)

    executed = await session.scalar(
        text(
            "SELECT COUNT(*) FROM audit_logs WHERE patient_id = :p "
            "AND outcome = 'executed' AND action = 'book_appointment'"
        ),
        {"p": patient_id},
    )
    assert executed >= 1


async def test_a_token_executes_at_most_once(
    client: AsyncClient, auth: dict[str, str], session
) -> None:
    """Replaying a confirmation must not book twice.

    The token is stateless, so the audit log is what remembers.
    """
    user_id, patient_id = await _identity(client, auth)
    token = await _propose_booking(session, user_id, patient_id, days_ahead=27)

    first = await client.post(
        f"{BASE}/actions/confirm", json={"token": token}, headers=auth
    )
    assert first.json()["status"] == "executed"

    before = await session.scalar(
        text("SELECT COUNT(*) FROM appointments WHERE patient_id = :p"),
        {"p": patient_id},
    )
    second = await client.post(
        f"{BASE}/actions/confirm", json={"token": token}, headers=auth
    )
    assert second.json()["status"] == "declined"
    assert "already" in second.json()["message"].lower()

    after = await session.scalar(
        text("SELECT COUNT(*) FROM appointments WHERE patient_id = :p"),
        {"p": patient_id},
    )
    assert after == before, "a replayed confirmation must not book again"


# --- cross-session ------------------------------------------------------- #


# --- the whole loop, over HTTP ------------------------------------------- #


class ActionPinnedProvider(StubProvider):
    """Routes to ACTION and parses a fixed booking, so the flow is testable.

    The real model is not used here: the property under test is that a chat
    turn surfaces a token the confirm endpoint then accepts, and a live
    model would make that assertion probabilistic.
    """

    def __init__(self, when: datetime) -> None:
        super().__init__()
        self._when = when

    async def generate_structured(self, *, messages, system, schema, **kwargs):  # type: ignore[override]
        if schema is AppointmentRequest:
            value: object = AppointmentRequest(
                action=BOOK,
                when=self._when.isoformat(),
                appointment_type="follow_up",
            )
        else:
            value = RouteDecision(route="ACTION", confidence=0.95, reason="pinned")
        return StructuredResponse(
            value=value,
            model="fake",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            latency_ms=1,
            provider="fake",
        )


async def test_chat_proposes_and_the_token_it_returns_confirms(session) -> None:
    """The full loop: ask in chat, get a token, confirm it, see the row.

    This is the path a patient actually takes, and the one where a mismatch
    between what the UI is handed and what the endpoint accepts would show.
    """
    when = (datetime.now(UTC) + timedelta(days=31)).replace(
        hour=9, minute=0, second=0, microsecond=0
    )
    app = create_app()
    app.dependency_overrides[chat_provider] = lambda: ActionPinnedProvider(when)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        auth = await _login(http, "P001")

        chat = await http.post(
            f"{BASE}/chat",
            json={"message": "Book me a follow-up a month from now."},
            headers=auth,
        )
        assert chat.status_code == 200, chat.text
        body = chat.json()
        assert body["route"] == "ACTION"

        pending = body["pending_action"]
        assert pending, "a booking turn must hand back a confirmation"
        assert pending["action"] == BOOK
        assert "confirm" in body["answer"].lower()

        before = await session.scalar(
            text("SELECT COUNT(*) FROM appointments WHERE patient_id = 1")
        )
        confirmed = await http.post(
            f"{BASE}/actions/confirm",
            json={"token": pending["token"]},
            headers=auth,
        )
        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["status"] == "executed"

        after = await session.scalar(
            text("SELECT COUNT(*) FROM appointments WHERE patient_id = 1")
        )
        assert after == before + 1
    app.dependency_overrides.clear()


async def test_another_patients_session_cannot_use_the_token(
    client: AsyncClient, auth: dict[str, str], session
) -> None:
    """The attack the binding exists for.

    P002 presenting P001's confirmation must be refused, and must not create
    an appointment on either record.
    """
    user_id, patient_id = await _identity(client, auth)
    token = await _propose_booking(session, user_id, patient_id, days_ahead=29)

    other = await _login(client, "P002")
    before = await session.scalar(text("SELECT COUNT(*) FROM appointments"))

    response = await client.post(
        f"{BASE}/actions/confirm", json={"token": token}, headers=other
    )
    assert response.status_code == 400
    assert "different" in response.json()["detail"].lower()

    after = await session.scalar(text("SELECT COUNT(*) FROM appointments"))
    assert after == before
