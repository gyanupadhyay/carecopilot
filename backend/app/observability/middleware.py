"""Request correlation.

Assigns every inbound request an id, binds it to the structlog context for
the duration, and echoes it in ``X-Request-ID``. The same id is what the
:class:`~app.models.audit.RequestTrace` row and any audit entries are keyed
by, so a user who reports "the assistant said something odd at 14:32" can be
traced through one identifier instead of a timestamp search.

An inbound ``X-Request-ID`` is accepted only when it looks like an id we
would have generated. Echoing arbitrary client input into log records is how
log-injection starts.
"""

from __future__ import annotations

import re
import time
import uuid
from contextvars import ContextVar

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.observability.logging import get_logger

_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

log = get_logger(__name__)


def current_request_id() -> str:
    return _REQUEST_ID.get()


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if _SAFE_ID.match(supplied) else uuid.uuid4().hex

        token = _REQUEST_ID.set(request_id)
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            # Log and re-raise: the exception handlers own the response
            # body, this only guarantees the failure is correlated.
            log.exception(
                "request.failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round((time.perf_counter() - started) * 1000),
            )
            raise
        else:
            response.headers["X-Request-ID"] = request_id
            log.info(
                "request.completed",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round((time.perf_counter() - started) * 1000),
            )
            return response
        finally:
            # Runs after the else clause, so both log calls above still see
            # the bound request_id.
            structlog.contextvars.unbind_contextvars("request_id")
            _REQUEST_ID.reset(token)
