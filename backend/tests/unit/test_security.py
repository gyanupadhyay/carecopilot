"""Password hashing and access tokens."""

from __future__ import annotations

import time

import jwt
import pytest

from app.auth.security import (
    TokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.config import settings


def test_hash_is_not_reversible_and_is_salted() -> None:
    first = hash_password("correct horse battery staple")
    second = hash_password("correct horse battery staple")
    assert "correct horse" not in first
    assert first != second, "identical passwords must not produce identical hashes"


def test_verify_accepts_correct_password() -> None:
    stored = hash_password("s3cret")
    assert verify_password(stored, "s3cret")


def test_verify_rejects_wrong_password() -> None:
    stored = hash_password("s3cret")
    assert not verify_password(stored, "s3cre7")


def test_verify_returns_false_for_a_corrupt_hash() -> None:
    """A malformed stored value must fail closed, not raise."""
    assert not verify_password("not-a-hash", "anything")


def test_token_round_trip_preserves_scope() -> None:
    token, expires_at = create_access_token(user_id=7, role="patient", patient_id=42)
    claims = decode_access_token(token)
    assert claims.user_id == 7
    assert claims.patient_id == 42
    assert claims.role == "patient"
    assert claims.expires_at == expires_at.replace(microsecond=0)


def test_token_without_patient_scope_decodes_as_none() -> None:
    token, _ = create_access_token(user_id=8, role="admin", patient_id=None)
    assert decode_access_token(token).patient_id is None


def _claims(**overrides: object) -> dict[str, object]:
    """A well-formed claim set, so each test varies exactly one thing."""
    base: dict[str, object] = {
        "sub": "1",
        "pid": 42,
        "role": "patient",
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "exp": int(time.time()) + 600,
    }
    base.update(overrides)
    return base


def _mint(claims: dict[str, object], *, key: str | None = None) -> str:
    return jwt.encode(
        claims, key or settings.jwt_secret, algorithm=settings.jwt_algorithm
    )


def test_token_signed_with_another_key_is_rejected() -> None:
    with pytest.raises(TokenError):
        decode_access_token(_mint(_claims(), key="a-different-secret"))


def test_expired_token_is_rejected() -> None:
    with pytest.raises(TokenError):
        decode_access_token(_mint(_claims(exp=int(time.time()) - 1)))


def test_token_without_expiry_is_rejected() -> None:
    """An unbounded session is not a session we are willing to accept."""
    claims = _claims()
    del claims["exp"]
    with pytest.raises(TokenError):
        decode_access_token(_mint(claims))


# --- issuer and audience (PRD §9) -------------------------------- #


def test_token_from_another_issuer_is_rejected() -> None:
    """A correctly signed token minted elsewhere is still not ours."""
    with pytest.raises(TokenError):
        decode_access_token(_mint(_claims(iss="some-other-service")))


def test_token_for_another_audience_is_rejected() -> None:
    """A token meant for a sibling service must not open this one."""
    with pytest.raises(TokenError):
        decode_access_token(_mint(_claims(aud="carecopilot-mcp")))


def test_token_missing_issuer_is_rejected() -> None:
    """Absent claims must fail, not skip the check."""
    claims = _claims()
    del claims["iss"]
    with pytest.raises(TokenError):
        decode_access_token(_mint(claims))


def test_token_missing_audience_is_rejected() -> None:
    claims = _claims()
    del claims["aud"]
    with pytest.raises(TokenError):
        decode_access_token(_mint(claims))


def test_minted_tokens_carry_issuer_and_audience() -> None:
    token, _ = create_access_token(user_id=7, role="patient", patient_id=42)
    payload = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
        audience=settings.jwt_audience,
        issuer=settings.jwt_issuer,
    )
    assert payload["iss"] == settings.jwt_issuer
    assert payload["aud"] == settings.jwt_audience


def test_garbage_token_is_rejected() -> None:
    with pytest.raises(TokenError):
        decode_access_token("not.a.token")
