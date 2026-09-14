"""The chat endpoint end to end (PRD §38 "user can chat").

Runs the real application — auth, memory, guardrails, persistence, tracing —
against a deterministic stub provider. Substituting the provider is what
makes these assertions exact: a test that called a real model could only
check that *something* came back.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.api.routes.chat import provider
from app.auth.demo import DEMO_PASSWORD
from app.llm.stub import STUB_PREFIX, StubProvider
from app.main import create_app
from app.models import Message, RequestTrace

pytestmark = pytest.mark.integration

BASE = "http://test/api"


@pytest.fixture
async def client(session):  # session fixture forces the DB-availability skip
    """An app wired to the stub provider, sharing the test's database."""
    app = create_app()
    stub = StubProvider()
    app.dependency_overrides[provider] = lambda: stub
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
    app.dependency_overrides.clear()


@pytest.fixture
async def token(client: AsyncClient) -> str:
    response = await client.post(
        f"{BASE}/auth/login",
        json={"email": "p001@carecopilot.demo", "password": DEMO_PASSWORD},
    )
    if response.status_code != 200:
        pytest.skip("Demo user not seeded; run scripts/generate_data.py --reset")
    return response.json()["access_token"]


@pytest.fixture
def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_chat_requires_authentication(client: AsyncClient) -> None:
    response = await client.post(f"{BASE}/chat", json={"message": "hello"})
    assert response.status_code == 401


async def test_chat_returns_the_documented_shape(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(
        f"{BASE}/chat", json={"message": "What are my medications?"}, headers=auth
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["answer"].startswith(STUB_PREFIX)
    # The stub cannot produce a RouteDecision, so the router degrades to RAG
    # — the safe default, and the behaviour under test here.
    assert body["route"] == "RAG"
    assert uuid.UUID(body["conversation_id"])
    assert "synthetic patient data" in body["disclaimer"].lower()

    # The seeded patient has ingested notes, so a medication question must
    # retrieve something and cite it.
    assert body["sources"], "expected citations from the ingested notes"
    first = body["sources"][0]
    assert first["chunk_id"] and first["document_id"]
    assert first["section"]
    assert 0.0 <= first["score"] <= 1.0

    meta = body["metadata"]
    assert meta["llm_provider"] == "stub"
    assert meta["router_enabled"] is True, "a router now classifies every question"
    assert meta["latency_ms"] >= 0
    assert "llm" in meta["stage_ms"]
    assert "graph" in meta["stage_ms"]
    assert "retrieval" in meta["stage_ms"]
    assert meta["retrieved_chunks"] and meta["retrieved_chunks"] > 0
    assert meta["guardrails"] == []


async def test_empty_message_is_rejected(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(f"{BASE}/chat", json={"message": "   "}, headers=auth)
    assert response.status_code == 422


async def test_oversized_message_is_rejected(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(
        f"{BASE}/chat", json={"message": "x" * 5000}, headers=auth
    )
    assert response.status_code == 422


async def test_conversation_is_remembered_across_turns(
    client: AsyncClient, auth: dict[str, str], session
) -> None:
    first = await client.post(
        f"{BASE}/chat", json={"message": "First question"}, headers=auth
    )
    conversation_id = first.json()["conversation_id"]

    second = await client.post(
        f"{BASE}/chat",
        json={"message": "Second question", "conversation_id": conversation_id},
        headers=auth,
    )
    assert second.status_code == 200
    assert second.json()["conversation_id"] == conversation_id

    stored = (
        await session.scalars(
            select(Message)
            .where(Message.conversation_id == uuid.UUID(conversation_id))
            .order_by(Message.id)
        )
    ).all()
    assert [m.role for m in stored] == ["user", "assistant", "user", "assistant"]
    assert stored[0].content == "First question"
    assert stored[2].content == "Second question"


async def test_conversation_can_be_replayed(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    created = await client.post(
        f"{BASE}/chat", json={"message": "Replay me"}, headers=auth
    )
    conversation_id = created.json()["conversation_id"]

    replay = await client.get(f"{BASE}/conversations/{conversation_id}", headers=auth)
    assert replay.status_code == 200
    roles = [m["role"] for m in replay.json()]
    assert roles == ["user", "assistant"]


async def test_another_users_conversation_is_refused(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    """P002's conversation id must not open for P001 (PRD §38)."""
    other = await client.post(
        f"{BASE}/auth/login",
        json={"email": "p002@carecopilot.demo", "password": DEMO_PASSWORD},
    )
    if other.status_code != 200:
        pytest.skip("P002 not seeded")
    other_auth = {"Authorization": f"Bearer {other.json()['access_token']}"}

    theirs = await client.post(
        f"{BASE}/chat", json={"message": "private"}, headers=other_auth
    )
    their_conversation = theirs.json()["conversation_id"]

    # Reading it as P001
    read = await client.get(f"{BASE}/conversations/{their_conversation}", headers=auth)
    assert read.status_code == 403

    # Posting into it as P001
    post = await client.post(
        f"{BASE}/chat",
        json={"message": "hello", "conversation_id": their_conversation},
        headers=auth,
    )
    assert post.status_code == 403


async def test_unknown_conversation_id_is_refused_not_created(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(
        f"{BASE}/chat",
        json={"message": "hi", "conversation_id": str(uuid.uuid4())},
        headers=auth,
    )
    assert response.status_code == 403


async def test_conversation_list_shows_only_own_conversations(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    await client.post(f"{BASE}/chat", json={"message": "Mine alone"}, headers=auth)
    listing = await client.get(f"{BASE}/conversations", headers=auth)
    assert listing.status_code == 200
    assert any(c["title"] == "Mine alone" for c in listing.json())


async def test_a_trace_row_is_written_for_every_turn(
    client: AsyncClient, auth: dict[str, str], session
) -> None:
    """PRD §26: every AI request is observable after the fact."""
    response = await client.post(
        f"{BASE}/chat", json={"message": "Trace this"}, headers=auth
    )
    request_id = response.json()["metadata"]["request_id"]

    trace = await session.scalar(
        select(RequestTrace).where(RequestTrace.request_id == request_id)
    )
    assert trace is not None
    assert trace.route == "RAG"
    assert trace.total_ms is not None
    assert trace.stage_ms and "llm" in trace.stage_ms
    assert trace.retrieved_count is not None
    assert trace.error is None


async def test_trace_records_no_clinical_content(
    client: AsyncClient, auth: dict[str, str], session
) -> None:
    """A trace stores shape, never the question or the answer (PRD §26)."""
    secret = "my knee has been hurting since August"
    response = await client.post(
        f"{BASE}/chat", json={"message": secret}, headers=auth
    )
    request_id = response.json()["metadata"]["request_id"]

    trace = await session.scalar(
        select(RequestTrace).where(RequestTrace.request_id == request_id)
    )
    assert trace is not None
    serialized = str(trace.__dict__)
    assert secret not in serialized
