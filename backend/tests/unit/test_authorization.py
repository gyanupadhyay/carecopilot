"""Authorization-context behaviour.

These tests encode the rule the rest of the system leans on: a session's
patient scope comes from the context and a mismatch is refused rather than
silently rewritten. PRD §38 lists cross-patient access as an acceptance
criterion; this is the unit-level half of it.
"""

from __future__ import annotations

import pytest

from app.auth.context import AuthContext, AuthorizationError


def test_patient_scope_returns_bound_patient(patient_ctx: AuthContext) -> None:
    assert patient_ctx.patient_scope == 42


def test_patient_scope_raises_when_unlinked(unlinked_ctx: AuthContext) -> None:
    with pytest.raises(AuthorizationError):
        _ = unlinked_ctx.patient_scope


def test_assert_patient_accepts_own_id(patient_ctx: AuthContext) -> None:
    assert patient_ctx.assert_patient(42) == 42


def test_assert_patient_refuses_other_patient(patient_ctx: AuthContext) -> None:
    """P001 must not reach P002 — the refusal is an error, not a redirect."""
    with pytest.raises(AuthorizationError):
        patient_ctx.assert_patient(43)


def test_assert_patient_refuses_when_unlinked(unlinked_ctx: AuthContext) -> None:
    with pytest.raises(AuthorizationError):
        unlinked_ctx.assert_patient(42)


def test_context_is_immutable(patient_ctx: AuthContext) -> None:
    """No code path may widen a scope after the context is built."""
    with pytest.raises((AttributeError, TypeError)):
        patient_ctx.patient_id = 43  # type: ignore[misc]


def test_error_message_names_no_patient(patient_ctx: AuthContext) -> None:
    """Refusals must not confirm that another patient exists."""
    with pytest.raises(AuthorizationError) as excinfo:
        patient_ctx.assert_patient(43)
    assert "43" not in str(excinfo.value)
