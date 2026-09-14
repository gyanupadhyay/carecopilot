"""Structured logging.

JSON in deployed environments, human-readable in development. Every event
carries ``request_id`` so a single chat turn can be reassembled from the log
stream without correlating timestamps.

What is *not* logged is as deliberate as what is: no prompts, no retrieved
chunk text, no answer bodies, no names or dates of birth. Events record
identifiers and measurements — ``patient_id``, counts, latencies — which are
enough to debug a route and not enough to reconstruct a record from logs
(PRD §26).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.config import settings

#: Keys that must never appear in a log event, whatever a caller passes.
_REDACTED_KEYS = frozenset(
    {
        "answer",
        "chunk_text",
        "content",
        "context",
        "date_of_birth",
        "first_name",
        "last_name",
        "messages",
        "password",
        "prompt",
        "question",
        "sql",
        "token",
    }
)


def _redact(_logger: Any, _method: str, event: dict[str, Any]) -> dict[str, Any]:
    """Drop clinical and secret payloads before they reach a handler.

    A blocklist is the wrong shape for a security boundary in general, but
    here it is a backstop for an ordinary mistake — someone logging the
    thing they were debugging — not the primary control.
    """
    for key in list(event):
        if key.lower() in _REDACTED_KEYS:
            event[key] = "[redacted]"
    return event


def configure_logging() -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redact,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if settings.environment == "development":
        processors.append(structlog.dev.ConsoleRenderer(colors=False))
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
