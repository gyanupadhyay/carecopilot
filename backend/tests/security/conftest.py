"""Database fixtures for the security suite.

Re-exported from ``tests/integration/conftest.py`` rather than copied. The
security tests need the same live, seeded PostgreSQL and the same
skip-don't-fail behaviour, and two definitions of "a session scoped to
P001" is exactly the kind of drift that makes a security test pass against
a fixture nobody else uses.

Importing a fixture function and re-binding it is how pytest shares
fixtures across sibling directories: ``conftest.py`` is only consulted along
a test's own path, so ``tests/integration/conftest.py`` is invisible here.
"""

from __future__ import annotations

from tests.integration.conftest import (  # noqa: F401
    demo_ctx,
    demo_patient,
    other_patient,
    session,
)
