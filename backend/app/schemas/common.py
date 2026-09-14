"""Shared response envelopes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    """Base for models read directly off SQLAlchemy instances."""

    model_config = ConfigDict(from_attributes=True)


class Page[T](BaseModel):
    """A slice of a result set.

    ``total`` is deliberately optional: counting is a second query, and most
    tool responses are already capped at a small limit where the count adds
    latency without informing the answer.
    """

    items: list[T]
    count: int = Field(description="Number of items in this page.")
    total: int | None = None


class ErrorResponse(BaseModel):
    """The single error shape every endpoint returns.

    ``detail`` is safe to show a user. Nothing here carries a stack trace,
    a generated SQL string, or any clinical content — a failure message is
    one of the easier places to leak both.
    """

    detail: str
    code: str = "error"
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    environment: str
    database: str
    vector_backend: str
    version: str
