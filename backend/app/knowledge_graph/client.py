"""The Neo4j driver, and the one way to run a statement against it.

The driver owns a connection pool, so there is one per process, built lazily
and closed on shutdown — the same shape as :mod:`app.llm.factory`.

``run_read`` is deliberately the only entry point, and it opens a **read**
transaction. Neo4j enforces that: a write inside one is refused by the
server, not by a convention this module hopes callers follow. The graph is a
derived projection (PRD §33) and PostgreSQL is the system of record, so the
application has no business writing here at all — only
``scripts/build_kg.py`` does, through its own session.

Failures are translated into :class:`GraphUnavailable` rather than escaping
as driver exceptions. The graph is optional infrastructure: a KG question
asked while Neo4j is down should answer "I could not reach the relationship
data", not 500.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j import exceptions as neo4j_exceptions

from app.config import settings
from app.observability.logging import get_logger

log = get_logger(__name__)

_driver: AsyncDriver | None = None
#: The loop the cached driver was built on. An async pool holds futures and
#: transports belonging to one loop; used from another, it raises "got Future
#: attached to a different loop" from somewhere deep inside asyncio, which
#: says nothing about the cause. The server process has a single loop, so
#: this only ever differs under a test runner that makes one per test — but
#: the failure is confusing enough, and the check cheap enough, to be worth
#: making impossible rather than documenting.
_driver_loop: asyncio.AbstractEventLoop | None = None


class GraphUnavailable(RuntimeError):
    """Neo4j could not be reached, or refused the statement."""


class GraphDisabled(GraphUnavailable):
    """KG_ENABLED is false. A configuration state, not a fault."""


def get_driver() -> AsyncDriver:
    """The driver for the running loop, built on first use.

    Must be called from async context: the loop identity is what the cache
    is keyed on.
    """
    global _driver, _driver_loop
    if not settings.kg_enabled:
        raise GraphDisabled("KG_ENABLED is false.")

    loop = asyncio.get_running_loop()
    if _driver is not None and _driver_loop is not loop:
        # Dropped, not closed: closing it would mean awaiting I/O on a loop
        # that is no longer running. Its sockets are released when the old
        # driver is collected.
        log.debug("kg.driver_rebuilt", reason="event loop changed")
        _driver = None

    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            connection_acquisition_timeout=settings.neo4j_timeout_seconds,
        )
        _driver_loop = loop
    return _driver


async def dispose_driver() -> None:
    """Close the pool, for application shutdown. Safe to call twice."""
    global _driver, _driver_loop
    if _driver is not None:
        # A driver belonging to a loop that has already finished cannot be
        # closed from here, and shutdown must not fail over a cache detail.
        with contextlib.suppress(RuntimeError):
            await _driver.close()
        _driver = None
        _driver_loop = None


async def run_read(
    cypher: str,
    parameters: dict[str, Any] | None = None,
    *,
    # ASYNC109 wants an asyncio timeout around the call instead of a
    # parameter. Not here: this value goes to the *server*, which stops
    # executing the traversal. Cancelling the coroutine locally would leave
    # Neo4j running the query it was told to run, which is the thing worth
    # bounding.
    timeout: float | None = None,  # noqa: ASYNC109
) -> list[dict[str, Any]]:
    """Execute one read-only statement and return its rows as plain dicts.

    Returning dicts rather than driver ``Record`` objects keeps the Neo4j
    types inside this module: everything above works with data it can
    serialise, and swapping the driver does not reach the agent.
    """
    driver = get_driver()
    try:
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(  # type: ignore[arg-type]
                cypher,
                parameters or {},
                timeout=timeout or settings.neo4j_timeout_seconds,
            )
            records = await result.data()
    except neo4j_exceptions.AuthError as exc:
        # Logged without the statement: a Cypher template is not secret, but
        # the parameters carry a patient id.
        log.error("kg.auth_failed", uri=settings.neo4j_uri)
        raise GraphUnavailable("Neo4j rejected the credentials.") from exc
    except neo4j_exceptions.ServiceUnavailable as exc:
        log.warning("kg.unavailable", uri=settings.neo4j_uri)
        raise GraphUnavailable("Could not reach the graph database.") from exc
    except neo4j_exceptions.Neo4jError as exc:
        log.error("kg.query_failed", code=getattr(exc, "code", None))
        raise GraphUnavailable("The graph database refused the query.") from exc

    return list(records)


async def healthy() -> bool:
    """Whether the graph is reachable, for the health endpoint."""
    try:
        await run_read("RETURN 1 AS ok")
    except GraphUnavailable:
        return False
    return True
