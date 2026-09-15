"""Demo rate limiting.

The limiter exists because a public demo runs on a free inference quota
behind published credentials. So these are written as the things that would
actually go wrong: a client that keeps retrying, a client that waits out its
window, many clients that are each individually polite, a clock that rolls
past midnight, and a table of per-client state that has to stay bounded.
"""

from __future__ import annotations

import pytest
from starlette.requests import Request

from app.api import rate_limit
from app.api.rate_limit import RateLimited, RateLimiter, client_key
from app.config import settings


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    """Controllable replacement for ``time.time`` inside the module."""

    class Clock:
        def __init__(self) -> None:
            self.now = 1_000_000.0

        def advance(self, seconds: float) -> None:
            self.now += seconds

    c = Clock()
    monkeypatch.setattr(rate_limit, "time", lambda: c.now)
    return c


def _limiter(*, per_minute: int = 3, per_day: int = 10, budget: int = 100):
    return RateLimiter(per_minute=per_minute, per_day=per_day, daily_budget=budget)


# --- the per-minute window ----------------------------------------------- #


def test_requests_under_the_limit_pass(clock) -> None:
    limiter = _limiter(per_minute=3)
    for _ in range(3):
        limiter.check("1.2.3.4")


def test_the_limit_trips_on_the_request_after_the_allowance(clock) -> None:
    limiter = _limiter(per_minute=3)
    for _ in range(3):
        limiter.check("1.2.3.4")

    with pytest.raises(RateLimited) as caught:
        limiter.check("1.2.3.4")

    assert caught.value.code == "rate_limited"
    assert 0 < caught.value.retry_after <= 60


def test_the_window_slides_rather_than_resetting_on_the_minute(clock) -> None:
    """A fixed window would let a client spend twice its allowance across
    the boundary. Here the oldest hit has to actually age out."""
    limiter = _limiter(per_minute=2)
    limiter.check("1.2.3.4")
    clock.advance(30)
    limiter.check("1.2.3.4")

    with pytest.raises(RateLimited):
        limiter.check("1.2.3.4")

    # 31s later the first hit has aged out, the second has not.
    clock.advance(31)
    limiter.check("1.2.3.4")
    with pytest.raises(RateLimited):
        limiter.check("1.2.3.4")


def test_a_rejected_request_is_not_counted(clock) -> None:
    """Otherwise a client that retries in a loop pushes its own reset
    further away on every attempt — which punishes a broken frontend far
    harder than an abusive caller."""
    limiter = _limiter(per_minute=2)
    limiter.check("1.2.3.4")
    limiter.check("1.2.3.4")

    for _ in range(5):
        with pytest.raises(RateLimited):
            limiter.check("1.2.3.4")

    # Only the two accepted requests aged; the window opens on schedule.
    clock.advance(61)
    limiter.check("1.2.3.4")


def test_clients_are_limited_independently(clock) -> None:
    limiter = _limiter(per_minute=1)
    limiter.check("1.1.1.1")
    limiter.check("2.2.2.2")

    with pytest.raises(RateLimited):
        limiter.check("1.1.1.1")


# --- the daily limits ----------------------------------------------------- #


def test_the_per_client_daily_limit_trips_and_points_at_midnight(clock) -> None:
    limiter = _limiter(per_minute=0, per_day=3)
    for _ in range(3):
        limiter.check("1.2.3.4")

    with pytest.raises(RateLimited) as caught:
        limiter.check("1.2.3.4")

    assert caught.value.code == "rate_limited"
    # Anything up to a full day away, never zero — a Retry-After of 0 is an
    # invitation to retry immediately.
    assert 0 < caught.value.retry_after <= 86_400


def test_the_global_budget_stops_a_client_who_is_within_its_own_limits(
    clock,
) -> None:
    """The limit that actually protects the API key. Each of these clients
    is individually polite; together they exhaust the quota."""
    limiter = _limiter(per_minute=0, per_day=0, budget=5)
    for i in range(5):
        limiter.check(f"10.0.0.{i}")

    with pytest.raises(RateLimited) as caught:
        limiter.check("10.0.0.99")

    assert caught.value.code == "demo_quota_exhausted"


def test_the_budget_refusal_is_distinguishable_from_a_personal_limit(
    clock,
) -> None:
    """Different codes because they need different advice: one says wait,
    the other says run it locally."""
    limiter = _limiter(per_minute=1, budget=1)
    limiter.check("1.1.1.1")

    with pytest.raises(RateLimited) as caught:
        limiter.check("2.2.2.2")
    assert caught.value.code == "demo_quota_exhausted"
    assert "locally" in caught.value.detail


def test_the_budget_is_checked_before_the_personal_limit(clock) -> None:
    """A breached global budget affects every visitor, so it must not be
    reachable only by clients who happen to be under their own limit."""
    limiter = _limiter(per_minute=1, budget=1)
    limiter.check("1.1.1.1")

    with pytest.raises(RateLimited) as caught:
        limiter.check("1.1.1.1")  # also over its own per-minute limit
    assert caught.value.code == "demo_quota_exhausted"


def test_counters_reset_at_the_day_boundary(
    clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import date

    day = date(2026, 9, 15)
    monkeypatch.setattr(rate_limit, "_utc_today", lambda: day)

    limiter = _limiter(per_minute=0, per_day=2, budget=2)
    limiter.check("1.2.3.4")
    limiter.check("1.2.3.4")
    with pytest.raises(RateLimited):
        limiter.check("1.2.3.4")

    day = date(2026, 9, 16)
    monkeypatch.setattr(rate_limit, "_utc_today", lambda: day)

    limiter.check("1.2.3.4")
    assert limiter.spent_today == 1


def test_a_zero_disables_one_limit_without_disabling_the_others(clock) -> None:
    limiter = _limiter(per_minute=0, per_day=0, budget=2)
    for _ in range(2):
        limiter.check("1.2.3.4")  # no per-client ceiling applies

    with pytest.raises(RateLimited) as caught:
        limiter.check("1.2.3.4")
    assert caught.value.code == "demo_quota_exhausted"


def test_budget_remaining_is_none_when_no_budget_is_set(clock) -> None:
    assert _limiter(budget=0).budget_remaining is None


# --- memory ---------------------------------------------------------------- #


def test_stale_clients_are_swept_so_the_table_stays_bounded(clock) -> None:
    """Without this, the client table grows once per distinct address and
    never shrinks — a memory leak reachable by anyone who can vary a source
    address."""
    limiter = _limiter(per_minute=5, per_day=0, budget=0)
    for i in range(50):
        limiter.check(f"10.0.0.{i}")
    assert len(limiter._hits) == 50

    # Past the sweep interval, with every recorded hit now outside the
    # per-minute window.
    clock.advance(rate_limit._SWEEP_INTERVAL_SECONDS + 1)
    limiter.check("192.168.0.1")

    assert len(limiter._hits) == 1


# --- client identity ------------------------------------------------------- #


def _request(ip: str = "1.2.3.4", forwarded: str | None = None) -> Request:
    headers = []
    if forwarded is not None:
        headers.append((b"x-forwarded-for", forwarded.encode()))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": headers,
            "client": (ip, 51234),
        }
    )


def test_forwarded_headers_are_ignored_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The header is client-supplied. Trusting it with no proxy in front
    lets anyone forge a fresh identity per request, which would leave the
    limiter configured and enforcing nothing."""
    monkeypatch.setattr(settings, "trust_proxy_headers", False)
    assert client_key(_request("1.2.3.4", forwarded="9.9.9.9")) == "1.2.3.4"


def test_the_original_client_is_used_when_a_proxy_is_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "trust_proxy_headers", True)
    key = client_key(_request("10.0.0.1", forwarded="9.9.9.9, 10.0.0.1"))
    assert key == "9.9.9.9"


def test_a_trusted_proxy_with_no_header_falls_back_to_the_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "trust_proxy_headers", True)
    assert client_key(_request("10.0.0.1")) == "10.0.0.1"
