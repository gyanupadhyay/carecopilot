"""Prompt text, kept out of the code that calls the model.

Prompts are configuration, not logic: they get reviewed by people who are
not reading Python, they get tuned against the evaluation set, and a change
to one is a behavioural change worth seeing on its own in a diff. Keeping
them in this package means ``git log app/prompts/`` is a readable history of
how the assistant's instructions evolved.
"""

from app.prompts.system import (
    ASSISTANT_SYSTEM_PROMPT,
    DATA_BOUNDARY_NOTE,
    build_system_prompt,
)

__all__ = [
    "ASSISTANT_SYSTEM_PROMPT",
    "DATA_BOUNDARY_NOTE",
    "build_system_prompt",
]
