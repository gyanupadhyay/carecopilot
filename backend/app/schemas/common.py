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
    """What this server is actually running.

    ``model`` and ``llm_provider`` are here so a deployment can say which
    model answered, without the sign-in page hard-coding a name that drifts
    the moment LLM_MODEL changes. The project is built around a self-hosted
    Qwen3-8B and a public demo will usually be serving something else; a
    visitor who assumes otherwise has been misled by omission, and the
    honest fix is for the answer to come from the process that knows.

    Neither is a secret — both are published in the README — and this
    endpoint stays unauthenticated so the login screen can read it.
    """

    status: str
    environment: str
    database: str
    vector_backend: str
    version: str
    model: str
    llm_provider: str
