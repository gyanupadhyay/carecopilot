"""Shared test configuration.

Settings are pinned to a development environment *before* ``app.config`` is
imported anywhere: the settings object is constructed at import time and
cached, so a test that imports the app first would inherit whatever happens
to be in the developer's ``.env``.
"""

from __future__ import annotations

import os

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("JWT_SECRET", "test-secret-not-used-outside-tests")
os.environ.setdefault("ACTION_TOKEN_SECRET", "test-action-secret")

import pytest

from app.auth.context import AuthContext


@pytest.fixture
def patient_ctx() -> AuthContext:
    return AuthContext(user_id=1, role="patient", patient_id=42, request_id="test")


@pytest.fixture
def unlinked_ctx() -> AuthContext:
    """A session whose account has no patient record."""
    return AuthContext(user_id=2, role="clinician", patient_id=None, request_id="test")
