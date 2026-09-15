"""The rate limiter as the API actually applies it.

tests/unit/test_rate_limit.py covers the counting. This covers the wiring,
which is the half that breaks silently: a limit attached to the wrong
endpoints protects nothing, and one attached to /api/health would let a
health check cause the outage the limiter exists to prevent.

The suite disables the limiter globally (see tests/conftest.py), so these
tests build their own limiter and enable it for themselves.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import rate_limit
from app.api.routes.chat import provider
from app.auth.demo import DEMO_PASSWORD
from app.config import settings
from app.llm.stub import StubProvider
from app.main import create_app

pytestmark = pytest.mark.integration

BASE = "http://test/api"

#: Small enough that a test trips it in a few requests, and the stub
#: provider means each one is cheap.
ALLOWANCE = 3


@pytest.fixture
async def client(session, monkeypatch: pytest.MonkeyPatch):
    """An app with the limiter on and a fresh, tight allowance.

    The limiter is cached per process, so a test that used the real one
    would inherit whatever earlier tests had spent and would pass or fail
    depending on execution order.
    """
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(
        rate_limit,
        "get_limiter",
        lambda: limiter,
    )
    limiter = rate_limit.RateLimiter(
        per_minute=ALLOWANCE, per_day=0, daily_budget=0
    )

    app = create_app()
    app.dependency_overrides[provider] = lambda: StubProvider()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
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


async def test_chat_is_limited(client: AsyncClient, auth: dict[str, str]) -> None:
    for _ in range(ALLOWANCE):
        ok = await client.post(f"{BASE}/chat", json={"message": "hi"}, headers=auth)
        assert ok.status_code == 200

    limited = await client.post(f"{BASE}/chat", json={"message": "hi"}, headers=auth)
    assert limited.status_code == 429


async def test_the_refusal_carries_retry_after_and_the_standard_error_shape(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    for _ in range(ALLOWANCE):
        await client.post(f"{BASE}/chat", json={"message": "hi"}, headers=auth)

    response = await client.post(
        f"{BASE}/chat", json={"message": "hi"}, headers=auth
    )

    assert response.status_code == 429
    # Without Retry-After a client retries at once, and the limiter
    # amplifies the load it is shedding.
    assert int(response.headers["retry-after"]) > 0

    body = response.json()
    assert body["code"] == "rate_limited"
    assert body["request_id"]  # same envelope as every other error


async def test_the_stream_endpoint_shares_the_budget(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    """Both chat endpoints spend a model call, so limiting only one of them
    would leave the other as a way around it."""
    for _ in range(ALLOWANCE):
        await client.post(f"{BASE}/chat", json={"message": "hi"}, headers=auth)

    response = await client.post(
        f"{BASE}/chat/stream", json={"message": "hi"}, headers=auth
    )
    assert response.status_code == 429


async def test_analytics_shares_the_budget(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    """Text-to-SQL is reachable without going through /api/chat."""
    for _ in range(ALLOWANCE):
        await client.post(f"{BASE}/chat", json={"message": "hi"}, headers=auth)

    response = await client.post(
        f"{BASE}/analytics", json={"question": "how many labs?"}, headers=auth
    )
    assert response.status_code == 429


async def test_health_is_never_limited(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    """The container health check polls this on a schedule. A limiter that
    can throttle it would cause the outage it exists to prevent."""
    for _ in range(ALLOWANCE):
        await client.post(f"{BASE}/chat", json={"message": "hi"}, headers=auth)

    for _ in range(ALLOWANCE * 3):
        assert (await client.get(f"{BASE}/health")).status_code == 200


async def test_reading_a_conversation_is_never_limited(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    """Replaying history is a database read. Limiting it would make an
    exhausted allowance also hide what the user already asked."""
    for _ in range(ALLOWANCE):
        await client.post(f"{BASE}/chat", json={"message": "hi"}, headers=auth)

    for _ in range(ALLOWANCE * 3):
        response = await client.get(f"{BASE}/conversations", headers=auth)
        assert response.status_code == 200


async def test_an_unauthenticated_request_is_rejected_before_it_costs_anything(
    client: AsyncClient,
) -> None:
    """401 rather than 429: no token means no model call, so these must not
    consume an allowance that a legitimate visitor would then be denied."""
    for _ in range(ALLOWANCE * 3):
        response = await client.post(f"{BASE}/chat", json={"message": "hi"})
        assert response.status_code == 401
