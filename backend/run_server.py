"""Development server entrypoint.

    python run_server.py

Exists because ``uvicorn app.main:app`` cannot be used directly on Windows.
uvicorn selects its event loop with an explicit ``loop_factory`` that
hardcodes ``ProactorEventLoop`` on win32 (uvicorn/loops/asyncio.py), which
bypasses the event-loop policy entirely — and psycopg's async mode cannot
run on that loop. The result is a server that starts cleanly and then fails
every single database query at runtime.

Owning the loop here fixes it for every platform at once, and is also how
the port and reload flag stay tied to ``Settings`` rather than to a command
line that each developer has to remember.

On Linux and macOS this is equivalent to the normal uvicorn invocation.
"""

from __future__ import annotations

import argparse
import asyncio

import uvicorn

from app.config import settings
from app.runtime import selector_loop_factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload on source changes. Delegates to the uvicorn "
        "supervisor, which runs the server in a subprocess and therefore "
        "already selects a selector loop.",
    )
    args = parser.parse_args(argv)

    if args.reload:
        # The reload supervisor sets use_subprocess=True, under which
        # uvicorn's own factory already returns SelectorEventLoop.
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level=settings.log_level.lower(),
        )
        return 0

    config = uvicorn.Config(
        "app.main:app",
        host=args.host,
        port=args.port,
        log_level=settings.log_level.lower(),
    )
    server = uvicorn.Server(config)

    # loop_factory requires Python 3.12, which is this project's floor.
    asyncio.run(server.serve(), loop_factory=selector_loop_factory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
