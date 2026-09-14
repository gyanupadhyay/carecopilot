"""Demo-account constants.

The seeded password is a shared, publicly documented value because the
entire dataset is fabricated and the point of the project is that anyone can
open the demo and try it. It is defined once, here, so the seeder and the
``/api/auth/demo-accounts`` endpoint cannot drift apart — and so there is
exactly one line to delete if this ever meets data that matters.
"""

from __future__ import annotations

from typing import Final

DEMO_PASSWORD: Final = "carecopilot-demo"

#: How many seeded logins the demo-accounts endpoint will list.
DEMO_ACCOUNT_LIMIT: Final = 5

#: Shown in the UI banner (PRD §34). Kept next to the password so the
#: disclaimer and the thing it disclaims live in the same file.
DEMO_DISCLAIMER: Final = (
    "DEMO — Uses synthetic patient data. Not for medical diagnosis or treatment."
)
