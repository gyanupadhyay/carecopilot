"""Event-loop setup required before any async database work.

psycopg's async mode cannot run on Windows' default ``ProactorEventLoop``;
every connection attempt fails with ``InterfaceError``. A selector-based
loop is required instead. This is a platform quirk, not a configuration
choice, so it is applied automatically rather than left to each entry point
to remember — the backend has several (uvicorn, pytest, the MCP server, the
ingestion and evaluation scripts) and a missed one fails at the first query
rather than at startup.

On every other platform this is a no-op: the default loop is already
compatible, and overriding it would cost the performance of uvloop where it
is installed.

Event-loop *policies* are deprecated in Python 3.14 in favour of passing a
``loop_factory`` to ``asyncio.Runner``. The policy is still what uvicorn and
pytest-asyncio consult when they build their own loops, so it remains the
only lever that reaches them; the deprecation warning is suppressed at the
point of use rather than project-wide, so a genuine warning elsewhere is
still visible.
"""

from __future__ import annotations

import asyncio
import sys
import warnings


def configure_async_runtime() -> None:
    """Ensure new event loops are selector-based on Windows."""
    if sys.platform != "win32":
        return

    # Every access below is deprecated in 3.14, including the attribute
    # lookup itself — asyncio raises the warning from a module __getattr__,
    # so the suppression has to enclose the lookup and not just the call.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if selector_policy is None:  # pragma: no cover - non-Windows CPython
            return
        if isinstance(asyncio.get_event_loop_policy(), selector_policy):
            return
        asyncio.set_event_loop_policy(selector_policy())


def selector_loop_factory():  # type: ignore[no-untyped-def]
    """A loop factory for callers that own their runner.

    Preferred over the policy where the caller controls loop creation —
    ``asyncio.run(main(), loop_factory=selector_loop_factory)`` — because it
    changes nothing globally and survives the policy deprecation.
    """
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()
