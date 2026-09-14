"""Confirmation tokens (PRD §15).

The token is what makes "no sentence can cause a write" structural rather
than aspirational, so these tests are written as the attacks they defend
against: replaying someone else's confirmation, editing the parameters,
signing your own, and using one after it should have died.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import jwt
import pytest

from app.actions.tokens import (
    ACTION_AUDIENCE,
    ActionTokenError,
    mint_action_token,
    verify_action_token,
)
from app.config import settings

PARAMS = {"when": "2026-10-01T10:00:00+00:00", "appointment_type": "follow_up"}


def _mint(user_id: int = 1, patient_id: int = 1, **kw) -> str:
    wire, _ = mint_action_token(
        action="book_appointment",
        user_id=user_id,
        patient_id=patient_id,
        params=PARAMS,
        **kw,
    )
    return wire


# --- the happy path ------------------------------------------------------ #


def test_a_minted_token_verifies_for_its_own_session() -> None:
    token = verify_action_token(_mint(), user_id=1, patient_id=1)
    assert token.action == "book_appointment"
    assert token.params == PARAMS
    assert token.expires_at > datetime.now(UTC)


def test_the_parameters_survive_the_round_trip_exactly() -> None:
    """Execution reads these, so a lossy round trip books the wrong thing."""
    token = verify_action_token(_mint(), user_id=1, patient_id=1)
    assert token.params["when"] == PARAMS["when"]
    assert token.params["appointment_type"] == PARAMS["appointment_type"]


def test_each_token_has_a_distinct_id() -> None:
    """The id is what the audit log uses to enforce execute-once."""
    first = verify_action_token(_mint(), user_id=1, patient_id=1)
    second = verify_action_token(_mint(), user_id=1, patient_id=1)
    assert first.token_id != second.token_id


# --- binding to the session ---------------------------------------------- #


def test_another_users_token_is_refused() -> None:
    """A token copied from a log or a screenshot is useless elsewhere."""
    wire = _mint(user_id=1, patient_id=1)
    with pytest.raises(ActionTokenError, match="different session"):
        verify_action_token(wire, user_id=2, patient_id=1)


def test_a_token_for_another_patient_is_refused() -> None:
    """Belt and braces for a clinician account that can reach two records."""
    wire = _mint(user_id=1, patient_id=1)
    with pytest.raises(ActionTokenError, match="different record"):
        verify_action_token(wire, user_id=1, patient_id=2)


# --- forgery -------------------------------------------------------------- #


def test_a_tampered_payload_is_refused() -> None:
    """Editing the parameters invalidates the signature over them."""
    wire = _mint()
    head, _original_body, signature = wire.split(".")
    # Re-sign nothing; just swap in a different body.
    forged_body = jwt.encode(
        {
            "jti": "x",
            "act": "book_appointment",
            "sub": "1",
            "pid": 1,
            "params": {"when": "2099-01-01T00:00:00+00:00"},
            "iss": settings.jwt_issuer,
            "aud": ACTION_AUDIENCE,
            "exp": int(time.time()) + 300,
        },
        "not-the-real-secret",
        algorithm=settings.jwt_algorithm,
    ).split(".")[1]
    with pytest.raises(ActionTokenError):
        verify_action_token(f"{head}.{forged_body}.{signature}", user_id=1, patient_id=1)


def test_a_token_signed_with_another_secret_is_refused() -> None:
    wire = jwt.encode(
        {
            "jti": "x",
            "act": "book_appointment",
            "sub": "1",
            "pid": 1,
            "params": PARAMS,
            "iss": settings.jwt_issuer,
            "aud": ACTION_AUDIENCE,
            "exp": int(time.time()) + 300,
        },
        "attacker-chosen-secret",
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(ActionTokenError):
        verify_action_token(wire, user_id=1, patient_id=1)


def test_a_session_jwt_cannot_be_used_as_a_confirmation() -> None:
    """The audiences are distinct precisely so this fails.

    Otherwise any valid login token would authorise a write.
    """
    wire = jwt.encode(
        {
            "jti": "x",
            "sub": "1",
            "pid": 1,
            "params": PARAMS,
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,  # the session audience, not the action one
            "exp": int(time.time()) + 300,
        },
        settings.action_token_secret,
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(ActionTokenError):
        verify_action_token(wire, user_id=1, patient_id=1)


def test_an_unsigned_token_is_refused() -> None:
    """The 'alg: none' classic."""
    wire = jwt.encode(
        {"jti": "x", "sub": "1", "pid": 1, "params": PARAMS, "aud": ACTION_AUDIENCE},
        key="",
        algorithm="none",
    )
    with pytest.raises(ActionTokenError):
        verify_action_token(wire, user_id=1, patient_id=1)


# --- expiry and absence --------------------------------------------------- #


def test_an_expired_token_is_refused() -> None:
    wire = _mint(ttl_seconds=-1)
    with pytest.raises(ActionTokenError, match="expired"):
        verify_action_token(wire, user_id=1, patient_id=1)


@pytest.mark.parametrize("wire", ["", "   ", "not-a-token", "a.b.c"])
def test_missing_or_malformed_tokens_are_refused(wire: str) -> None:
    with pytest.raises(ActionTokenError):
        verify_action_token(wire, user_id=1, patient_id=1)


def test_the_refusal_does_not_say_which_check_failed() -> None:
    """Distinguishing 'bad signature' from 'wrong audience' helps a prober."""
    wire = jwt.encode(
        {"jti": "x", "sub": "1", "pid": 1, "params": PARAMS, "aud": "wrong"},
        "wrong-secret",
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(ActionTokenError) as excinfo:
        verify_action_token(wire, user_id=1, patient_id=1)
    assert "not valid" in str(excinfo.value)
    assert "signature" not in str(excinfo.value).lower()
    assert "audience" not in str(excinfo.value).lower()


def test_the_action_secret_is_not_the_session_secret() -> None:
    """Rotating one must not silently authorise tokens signed by the other."""
    assert settings.action_token_secret != settings.jwt_secret
