"""Signed confirmation tokens for write actions (PRD §15).

The token is what makes "the model cannot write to the record" structurally
true rather than a claim about prompt discipline. Its properties:

*It carries the validated parameters, not the user's sentence.* Whatever the
model parsed out of "book me something next Tuesday" was already checked
against real providers and real slots before a token existed. Execution
reads the token's fields and never re-parses anything, so the confirmed
action is exactly the proposed one — there is no second interpretation step
in which it could drift.

*It is bound to the user and the patient.* A token minted for one session is
refused in another, so a token that leaks — copied from a log, a screenshot,
a shared terminal — is useless to anyone else.

*It expires in minutes.* A confirmation is a response to something the user
was just shown; a token still valid tomorrow is a standing authorisation
nobody granted.

*It is signed with its own secret.* ``ACTION_TOKEN_SECRET`` is separate from
``JWT_SECRET`` so that a leaked session-signing key cannot be used to mint
write authorisations, and so the two can be rotated independently.

Deliberately **not** stateful. A server-side pending-actions table would add
single-use semantics, which this does not have: a token can be presented
twice within its TTL. That is acceptable because execution is idempotent per
token — ``execute`` records the token's id on the audit row and refuses an
id it has already executed (see ``appointments.py``). Statelessness keeps
the confirm endpoint free of a cleanup job and a second source of truth.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import jwt

from app.config import settings

#: Distinct from the session token's audience so a session JWT can never be
#: presented as a confirmation token, or the reverse, even by accident.
ACTION_AUDIENCE: Final = "carecopilot-action"


class ActionTokenError(Exception):
    """The token is missing, malformed, expired, or not the caller's."""


@dataclass(frozen=True, slots=True)
class ActionToken:
    """The verified contents of a confirmation token."""

    token_id: str
    action: str
    user_id: int
    patient_id: int
    params: dict[str, Any]
    expires_at: datetime


def _secret() -> str:
    secret = settings.action_token_secret
    if not secret:  # pragma: no cover - config refuses this in production
        raise ActionTokenError("ACTION_TOKEN_SECRET is not configured.")
    return secret


def mint_action_token(
    *,
    action: str,
    user_id: int,
    patient_id: int,
    params: dict[str, Any],
    ttl_seconds: int | None = None,
) -> tuple[str, ActionToken]:
    """Sign a proposal. Returns the wire token and its decoded contents.

    Only ever called with parameters the service layer has already
    validated — minting is the last step of proposing, not the first step of
    executing.
    """
    ttl = ttl_seconds if ttl_seconds is not None else settings.action_token_ttl_seconds
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=ttl)
    token_id = uuid.uuid4().hex

    payload = {
        "jti": token_id,
        "act": action,
        "sub": str(user_id),
        "pid": patient_id,
        "params": params,
        "iss": settings.jwt_issuer,
        "aud": ACTION_AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    wire = jwt.encode(payload, _secret(), algorithm=settings.jwt_algorithm)
    return wire, ActionToken(
        token_id=token_id,
        action=action,
        user_id=user_id,
        patient_id=patient_id,
        params=params,
        expires_at=expires_at,
    )


def verify_action_token(wire: str, *, user_id: int, patient_id: int) -> ActionToken:
    """Decode and check a confirmation token against the calling session.

    ``user_id`` and ``patient_id`` come from the request's ``AuthContext``,
    never from the token itself — the token is the thing being checked, so
    trusting its own claims about who may use it would check nothing.
    """
    if not wire or not wire.strip():
        raise ActionTokenError("No confirmation token was supplied.")

    try:
        payload = jwt.decode(
            wire,
            _secret(),
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer,
            audience=ACTION_AUDIENCE,
            options={"require": ["exp", "sub", "aud", "iss", "jti"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise ActionTokenError(
            "This confirmation has expired. Please ask again to get a fresh one."
        ) from exc
    except jwt.InvalidTokenError as exc:
        # The specific reason is deliberately not echoed: distinguishing
        # "bad signature" from "wrong audience" tells a prober which half of
        # a forgery attempt worked.
        raise ActionTokenError("This confirmation token is not valid.") from exc

    if str(payload.get("sub")) != str(user_id):
        raise ActionTokenError("This confirmation belongs to a different session.")
    if payload.get("pid") != patient_id:
        raise ActionTokenError("This confirmation belongs to a different record.")

    params = payload.get("params")
    if not isinstance(params, dict):
        raise ActionTokenError("This confirmation token is not valid.")

    return ActionToken(
        token_id=str(payload["jti"]),
        action=str(payload.get("act") or ""),
        user_id=user_id,
        patient_id=patient_id,
        params=params,
        expires_at=datetime.fromtimestamp(payload["exp"], tz=UTC),
    )
