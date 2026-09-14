"""Server-Sent Events streaming (PRD §6)."""

from __future__ import annotations

import json
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.routes.chat import provider
from app.auth.demo import DEMO_PASSWORD
from app.llm.stub import STUB_PREFIX, StubProvider
from app.main import create_app
from app.models import Message
from app.services.chat import FAILURE_MESSAGE

pytestmark = pytest.mark.integration

BASE = "http://test/api"


@pytest.fixture
async def client(session):  # session fixture forces the DB-availability skip
    app = create_app()
    app.dependency_overrides[provider] = lambda: StubProvider()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        yield http
    app.dependency_overrides.clear()


@pytest.fixture
async def auth(client: AsyncClient) -> dict[str, str]:
    response = await client.post(
        f"{BASE}/auth/login",
        json={"email": "p001@carecopilot.demo", "password": DEMO_PASSWORD},
    )
    if response.status_code != 200:
        pytest.skip("Demo user not seeded; run scripts/generate_data.py --reset")
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event name, payload) pairs."""
    events: list[tuple[str, dict]] = []
    for frame in body.split("\n\n"):
        name = data = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        if name and data:
            events.append((name, json.loads(data)))
    return events


async def _stream(client: AsyncClient, auth: dict[str, str], **payload) -> list:
    async with client.stream(
        "POST", f"{BASE}/chat/stream", json=payload, headers=auth, timeout=60
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join([chunk async for chunk in response.aiter_text()])
    return parse_sse(body)


async def test_stream_requires_authentication(client: AsyncClient) -> None:
    response = await client.post(f"{BASE}/chat/stream", json={"message": "hi"})
    assert response.status_code == 401


async def test_stream_emits_meta_then_deltas_then_done(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    events = await _stream(client, auth, message="What are my medications?")
    names = [name for name, _ in events]

    assert names[0] == "meta"
    assert names[-1] == "done"
    assert names.count("delta") > 1, "answer should arrive in pieces"

    meta = events[0][1]
    assert uuid.UUID(meta["conversation_id"])
    assert meta["request_id"]
    assert "synthetic patient data" in meta["disclaimer"].lower()


async def test_stream_reports_the_route_it_took(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    events = await _stream(client, auth, message="What are my medications?")
    done = next(payload for name, payload in events if name == "done")
    assert done["route"] in {
        "API",
        "RAG",
        "HYBRID",
        "TEXT_TO_SQL",
        "ACTION",
        "OUT_OF_SCOPE",
    }
    assert done["metadata"]["router_enabled"] is True


async def test_streamed_deltas_reconstruct_the_final_answer(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    events = await _stream(client, auth, message="Reconstruct me")
    streamed = "".join(
        payload["text"] for name, payload in events if name == "delta"
    )
    done = next(payload for name, payload in events if name == "done")

    assert streamed.strip().startswith(STUB_PREFIX)
    assert done["answer"].strip() == streamed.strip()
    assert done["replaces_streamed_text"] is False


async def test_done_event_carries_the_full_response_body(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    events = await _stream(client, auth, message="Metadata please")
    done = next(payload for name, payload in events if name == "done")

    assert done["route"] == "RAG"
    assert done["metadata"]["llm_provider"] == "stub"
    assert "llm" in done["metadata"]["stage_ms"]
    assert "retrieval" in done["metadata"]["stage_ms"]


async def test_sources_arrive_before_the_done_event(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    """Citations stream as their own event, once retrieval has run.

    They cannot ride in ``meta`` any more: retrieval happens inside the
    graph, after classification, so at ``meta`` time there is nothing to
    cite yet.
    """
    events = await _stream(
        client, auth, message="What did my doctor say about my knee?"
    )
    names = [name for name, _ in events]
    assert "sources" in names
    assert names.index("sources") < names.index("done")

    sources = next(payload for name, payload in events if name == "sources")
    done = next(payload for name, payload in events if name == "done")
    assert sources["sources"], "expected citations"
    assert [s["chunk_id"] for s in sources["sources"]] == [
        s["chunk_id"] for s in done["sources"]
    ]


async def test_streamed_turn_is_persisted(
    client: AsyncClient, auth: dict[str, str], session
) -> None:
    from sqlalchemy import select

    events = await _stream(client, auth, message="Persist this stream")
    conversation_id = uuid.UUID(events[0][1]["conversation_id"])

    stored = (
        await session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
    ).all()
    assert [m.role for m in stored] == ["user", "assistant"]
    assert stored[0].content == "Persist this stream"


async def test_stream_continues_an_existing_conversation(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    first = await _stream(client, auth, message="Turn one")
    conversation_id = first[0][1]["conversation_id"]

    second = await _stream(
        client, auth, message="Turn two", conversation_id=conversation_id
    )
    assert second[0][1]["conversation_id"] == conversation_id


async def test_stream_refuses_another_users_conversation(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    """The refusal arrives as an error event: status is already 200."""
    other = await client.post(
        f"{BASE}/auth/login",
        json={"email": "p002@carecopilot.demo", "password": DEMO_PASSWORD},
    )
    if other.status_code != 200:
        pytest.skip("P002 not seeded")
    other_auth = {"Authorization": f"Bearer {other.json()['access_token']}"}

    theirs = await _stream(client, other_auth, message="private")
    their_conversation = theirs[0][1]["conversation_id"]

    events = await _stream(
        client, auth, message="intrude", conversation_id=their_conversation
    )
    names = [name for name, _ in events]
    assert names == ["error"]
    assert "not found" in events[0][1]["detail"].lower()


async def test_empty_message_is_rejected_before_streaming(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(
        f"{BASE}/chat/stream", json={"message": "   "}, headers=auth
    )
    assert response.status_code == 422


async def test_provider_failure_mid_stream_ends_with_a_safe_message(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    from app.llm.errors import LLMServiceError

    class BrokenProvider(StubProvider):
        async def stream(self, **kwargs):  # type: ignore[override]
            yield "partial "
            raise LLMServiceError("upstream died", provider="fake")

    app = create_app()
    app.dependency_overrides[provider] = lambda: BrokenProvider()
    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
        http.stream(
            "POST", f"{BASE}/chat/stream", json={"message": "break"}, headers=auth
        ) as response,
    ):
        body = "".join([chunk async for chunk in response.aiter_text()])
    app.dependency_overrides.clear()

    events = parse_sse(body)
    done = next(payload for name, payload in events if name == "done")
    assert done["answer"] == FAILURE_MESSAGE
    assert "llm_unavailable" in done["metadata"]["guardrails"]
    assert done["replaces_streamed_text"] is True
