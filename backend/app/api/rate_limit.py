"""Inbound rate limiting for the model-backed endpoints.

A public demo runs on someone's free inference quota with credentials the
README publishes, so "anyone can try it" and "anyone can drain it" are the
same sentence. This module is what separates them.

Two limits, and they are not redundant:

*Per client, per minute and per day.* Stops one person hammering the chat
endpoint. Keyed by IP, which is the only stable thing about an anonymous
visitor — the demo accounts are shared, so ``user_id`` would put every
visitor in the same bucket.

*Globally, per day.* The one that actually protects the API key. A per-IP
limit does nothing against a hundred IPs, and the failure it prevents is
the expensive one: a drained quota takes the demo down for everyone until
the provider's window rolls over. Past the budget the API refuses in a way
that says so, rather than letting the provider fail and surfacing as a
broken assistant.

Three things about the design are worth knowing before changing it.

*State is in-process, deliberately.* One container, one uvicorn worker, no
replicas — see ``run_server.py``, which runs a single server rather than a
worker pool. A Redis dependency to share counters between workers that do
not exist would be infrastructure for its own sake. **If this ever grows a
second worker or replica, every limit here silently becomes per-worker**,
and that is the moment to move the counters out of the process.

*The dependency is ``async``, which is load-bearing.* FastAPI runs a plain
``def`` dependency in a threadpool; these counters are mutated without a
lock because an ``async def`` dependency runs on the event loop and the
critical section contains no ``await``. Making it synchronous would
introduce real concurrency and with it lost updates.

*It is a dependency, not middleware.* Middleware would need a path
allowlist to avoid limiting ``/api/health``, which the container health
check and any uptime monitor poll on a schedule and which must never be
throttled — a limiter that can cause the outage it exists to prevent is
worse than none.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from math import ceil
from time import time

from fastapi import Request

from app.api.deps import PatientScoped
from app.config import settings
from app.observability.logging import get_logger

log = get_logger(__name__)

_MINUTE_SECONDS = 60.0

#: How often stale per-client entries are dropped. Without a sweep the
#: client table grows once per distinct IP and never shrinks, which turns a
#: rate limiter into a slow memory leak reachable by anyone who can vary a
#: source address.
_SWEEP_INTERVAL_SECONDS = 300.0


class RateLimited(Exception):
    """Raised when a request exceeds a limit.

    Carries the ``Retry-After`` value rather than leaving the caller to
    guess: a client told only "too many requests" retries immediately, and
    a limiter that provokes retries is amplifying the traffic it is meant
    to shed.
    """

    def __init__(self, detail: str, *, code: str, retry_after: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.code = code
        self.retry_after = retry_after


def _utc_today() -> date:
    return datetime.now(UTC).date()


def _seconds_until_utc_midnight(now: datetime | None = None) -> int:
    moment = now or datetime.now(UTC)
    tomorrow = moment.date() + timedelta(days=1)
    midnight = datetime.combine(tomorrow, datetime.min.time(), tzinfo=UTC)
    return max(1, ceil((midnight - moment).total_seconds()))


class RateLimiter:
    """Fixed daily counters over a sliding per-minute window.

    The per-minute window slides because a fixed one lets a client spend
    its whole allowance at :59 and again at :00. The daily counters do not,
    because a sliding 24-hour window needs every timestamp of the day held
    in memory per client to evict correctly — and "resets at midnight UTC"
    is something a demo can state plainly in its refusal message.
    """

    def __init__(self, *, per_minute: int, per_day: int, daily_budget: int) -> None:
        self._per_minute = per_minute
        self._per_day = per_day
        self._daily_budget = daily_budget

        self._hits: dict[str, deque[float]] = {}
        self._day_counts: dict[str, int] = {}
        self._spent_today = 0
        self._day = _utc_today()
        self._budget_tripped = False
        self._last_sweep = time()

    # -- introspection ---------------------------------------------- #

    @property
    def spent_today(self) -> int:
        return self._spent_today

    @property
    def budget_remaining(self) -> int | None:
        """``None`` when no global budget is configured."""
        if self._daily_budget <= 0:
            return None
        return max(0, self._daily_budget - self._spent_today)

    # -- enforcement ------------------------------------------------ #

    def check(self, client: str) -> None:
        """Record one request, or raise :class:`RateLimited`.

        Nothing is recorded when a limit is hit. A rejected request costs
        no model call, so counting it would make a client that keeps
        retrying push its own reset further away — punishing a
        misconfigured frontend far more than an abusive one.
        """
        now = time()
        self._roll_day()
        self._sweep(now)

        # Global budget first: it is the limit whose breach affects every
        # other visitor, so it should not be reachable only by clients who
        # happen to be under their own.
        if self._daily_budget > 0 and self._spent_today >= self._daily_budget:
            if not self._budget_tripped:
                # Once per day, not once per rejected request — a tripped
                # budget under load would otherwise write the log line
                # thousands of times.
                self._budget_tripped = True
                log.warning(
                    "rate_limit.daily_budget_exhausted",
                    budget=self._daily_budget,
                    spent=self._spent_today,
                )
            raise RateLimited(
                "This demo's daily request budget is exhausted. It resets at "
                "00:00 UTC. To use it without limits, run the project "
                "locally — see the README.",
                code="demo_quota_exhausted",
                retry_after=_seconds_until_utc_midnight(),
            )

        if self._per_day > 0 and self._day_counts.get(client, 0) >= self._per_day:
            raise RateLimited(
                "You have reached this demo's daily question limit. It resets "
                "at 00:00 UTC.",
                code="rate_limited",
                retry_after=_seconds_until_utc_midnight(),
            )

        hits = self._hits.setdefault(client, deque())
        cutoff = now - _MINUTE_SECONDS
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if self._per_minute > 0 and len(hits) >= self._per_minute:
            raise RateLimited(
                "Too many questions in a short time. Please wait a moment.",
                code="rate_limited",
                retry_after=max(1, ceil(hits[0] + _MINUTE_SECONDS - now)),
            )

        hits.append(now)
        self._day_counts[client] = self._day_counts.get(client, 0) + 1
        self._spent_today += 1

    # -- housekeeping ----------------------------------------------- #

    def _roll_day(self) -> None:
        today = _utc_today()
        if today != self._day:
            self._day = today
            self._day_counts.clear()
            self._spent_today = 0
            self._budget_tripped = False

    def _sweep(self, now: float) -> None:
        if now - self._last_sweep < _SWEEP_INTERVAL_SECONDS:
            return
        self._last_sweep = now

        cutoff = now - _MINUTE_SECONDS
        stale = [
            client
            for client, hits in self._hits.items()
            if not hits or hits[-1] <= cutoff
        ]
        for client in stale:
            del self._hits[client]

        # _day_counts is intentionally left alone: it is one small integer
        # per client and it is cleared wholesale at the day boundary, so
        # sweeping it would discard exactly the state the daily limit
        # exists to keep.


@lru_cache(maxsize=1)
def get_limiter() -> RateLimiter:
    return RateLimiter(
        per_minute=settings.rate_limit_per_minute,
        per_day=settings.rate_limit_per_day,
        daily_budget=settings.rate_limit_daily_budget,
    )


def client_key(request: Request) -> str:
    """Identify the caller for limiting purposes.

    ``X-Forwarded-For`` is read only when ``TRUST_PROXY_HEADERS`` is set,
    and this is not a formality. The header is client-supplied: with no
    proxy in front, anyone can put a fresh value on every request and the
    per-IP limits stop existing. Trusting it by default would produce a
    limiter that looks configured and enforces nothing.

    Behind a proxy that appends, the leftmost entry is the original client.
    The deployed stack instead puts Caddy in front, configured to *replace*
    the header rather than append to it — see deploy/Caddyfile, which
    explains why appending would let a visitor forge this value.
    """
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()

    client = request.client
    return client.host if client else "unknown"


async def enforce_rate_limit(request: Request, _: PatientScoped) -> None:
    """Dependency for endpoints that spend a model call.

    Async on purpose — see this module's docstring.

    ``_`` is unused and load-bearing. A path-operation dependency runs
    before the endpoint's own parameters, so without it the limiter would
    see unauthenticated requests too — and those never reach a model. An
    attacker could then exhaust the global daily budget with a flood of
    requests carrying no token at all, denying the demo to everyone
    without spending a single inference call. Naming ``PatientScoped``
    here makes authentication a *sub*-dependency, so it resolves first and
    a request without a valid session is a 401 that costs no allowance.

    FastAPI caches dependencies within a request, so the auth context is
    still built exactly once.
    """
    if not settings.rate_limit_enabled:
        return
    get_limiter().check(client_key(request))


__all__ = [
    "RateLimited",
    "RateLimiter",
    "client_key",
    "enforce_rate_limit",
    "get_limiter",
]
