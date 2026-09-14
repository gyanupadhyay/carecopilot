"""Exception handlers.

Every error leaves the API in one shape (:class:`ErrorResponse`) carrying
the request id, so a user can quote an identifier and the log stream can be
searched for it.

Unhandled exceptions return a fixed sentence. Exception text routinely
contains a SQL fragment, a file path, or a row of clinical data, none of
which belongs in an HTTP response — the detail goes to the log instead,
keyed by the same request id.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.auth.context import AuthorizationError
from app.auth.security import TokenError
from app.observability.logging import get_logger
from app.observability.middleware import current_request_id
from app.schemas.common import ErrorResponse

log = get_logger(__name__)


def _error(status_code: int, detail: str, code: str) -> JSONResponse:
    body = ErrorResponse(
        detail=detail, code=code, request_id=current_request_id() or None
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AuthorizationError)
    async def _authorization(_: Request, exc: AuthorizationError) -> JSONResponse:
        # The message is written to be user-safe at the raise site; it names
        # no patient and no record.
        log.warning("authorization.denied", reason=str(exc))
        return _error(status.HTTP_403_FORBIDDEN, str(exc), "forbidden")

    @app.exception_handler(TokenError)
    async def _token(_: Request, __: TokenError) -> JSONResponse:
        return _error(
            status.HTTP_401_UNAUTHORIZED, "Invalid or expired session.", "unauthorized"
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, str(exc.detail), "http_error")

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Field names and positions are safe to return; submitted values are
        # not, so the pydantic error list is summarized rather than echoed.
        fields = ", ".join(
            ".".join(str(p) for p in err.get("loc", ()) if p != "body")
            for err in exc.errors()
        )
        return _error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Invalid request: {fields or 'malformed body'}.",
            "invalid_request",
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("request.unhandled_error", path=request.url.path)
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "An internal error occurred.",
            "internal_error",
        )
