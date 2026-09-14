"""Login and session payloads."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class SessionUser(BaseModel):
    """What the frontend needs to render a session.

    ``patient_id`` is echoed so the UI can label the demo, not so it can be
    sent back: the server reads the scope from the token on every request
    and ignores any patient id in a request body.
    """

    user_id: int
    email: str
    display_name: str
    role: str
    patient_id: int | None
    patient_external_id: str | None = None
    patient_name: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: SessionUser


class DemoAccount(BaseModel):
    """A seeded login, listed by ``GET /api/auth/demo-accounts``.

    This endpoint exists because the whole dataset is synthetic and the
    demo is meant to be self-service. It is refused outside development so
    that the same code cannot hand out credentials in a deployed setting.
    """

    email: str
    password: str
    display_name: str
    patient_external_id: str | None
