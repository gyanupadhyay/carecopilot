"""Password hashing and bearer tokens.

Argon2id is used for passwords rather than a fast hash: the demo seeds
accounts with a shared weak password, and a fast hash would make that a
genuinely bad example to copy.

Tokens are short-lived HS256 JWTs. The claim that matters is ``pid`` — the
patient scope — because it is minted from the database row at login and
never read back from user input afterwards.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.config import settings

_hasher = PasswordHasher()


class TokenError(Exception):
    """Token missing, malformed, expired, or signed with the wrong key."""


@dataclass(frozen=True, slots=True)
class TokenClaims:
    user_id: int
    role: str
    patient_id: int | None
    expires_at: datetime


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Constant-time-ish verification that never raises on a bad password.

    Every failure mode — wrong password, corrupt hash, unknown algorithm —
    collapses to ``False`` so that callers cannot accidentally distinguish
    "no such user" from "wrong password" by the shape of the exception.
    """
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def create_access_token(
    *, user_id: int, role: str, patient_id: int | None
) -> tuple[str, datetime]:
    """Mint an access token. Returns the token and its expiry."""
    now = datetime.now(UTC)
    expires_at = now + timedelta(minutes=settings.jwt_ttl_minutes)
    payload = {
        "sub": str(user_id),
        "role": role,
        "pid": patient_id,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "jti": uuid.uuid4().hex,
    }
    token = jwt.encode(
        payload,
        _require_secret(),
        algorithm=settings.jwt_algorithm,
    )
    return token, expires_at


def decode_access_token(token: str) -> TokenClaims:
    try:
        payload = jwt.decode(
            token,
            _require_secret(),
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            # Requiring the claims is as important as checking them: PyJWT
            # verifies `iss` and `aud` only when they are present, so a token
            # minted without them would otherwise sail through unexamined.
            options={"require": ["exp", "sub", "iss", "aud"]},
        )
    except jwt.PyJWTError as exc:  # expired, bad signature, wrong iss/aud
        raise TokenError(str(exc)) from exc

    patient_id = payload.get("pid")
    return TokenClaims(
        user_id=int(payload["sub"]),
        role=str(payload.get("role", "patient")),
        patient_id=int(patient_id) if patient_id is not None else None,
        expires_at=datetime.fromtimestamp(payload["exp"], tz=UTC),
    )


def _require_secret() -> str:
    """Fail loudly rather than sign with ``None``.

    ``Settings`` already generates an ephemeral key outside production and
    refuses to start without one in production, so reaching this error means
    the settings object was constructed in a way that bypassed both.
    """
    if not settings.jwt_secret:
        raise TokenError("JWT_SECRET is not configured.")
    return settings.jwt_secret
