"""Backend services: the only place clinical data is read or written.

Every function here takes an :class:`~app.auth.context.AuthContext` first
and derives its patient filter from it. The API routes, the tool layer, the
MCP server and the evaluation runner all call these same functions, so the
authorization rule is written once instead of once per entry point.
"""

from app.services.clinical import (
    compare_medications,
    get_appointments,
    get_current_medications,
    get_encounters,
    get_lab_results,
    get_last_encounter,
    get_medications_in_effect,
    get_next_appointment,
    get_patient_profile,
)

__all__ = [
    "compare_medications",
    "get_appointments",
    "get_current_medications",
    "get_encounters",
    "get_lab_results",
    "get_last_encounter",
    "get_medications_in_effect",
    "get_next_appointment",
    "get_patient_profile",
]
