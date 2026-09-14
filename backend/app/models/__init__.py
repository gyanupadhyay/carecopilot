"""ORM models.

Every model is imported here so that ``Base.metadata`` is complete by the
time Alembic's ``env.py`` reads it. A missing import means autogenerate
silently proposes dropping the table.
"""

from __future__ import annotations

from app.models.appointment import Appointment
from app.models.audit import AuditLog, RequestTrace
from app.models.clinical_document import ClinicalDocument, DocumentChunk
from app.models.clinical_events import Allergy, Diagnosis, Procedure
from app.models.condition import Condition, PatientCondition
from app.models.conversation import Conversation, Message
from app.models.encounter import Encounter
from app.models.lab_result import LabResult
from app.models.medication import Medication
from app.models.patient import Patient
from app.models.provider import Provider
from app.models.user import User, UserPatientMapping

__all__ = [
    "Allergy",
    "Appointment",
    "AuditLog",
    "ClinicalDocument",
    "Condition",
    "Conversation",
    "Diagnosis",
    "DocumentChunk",
    "Encounter",
    "LabResult",
    "Medication",
    "Message",
    "Patient",
    "PatientCondition",
    "Procedure",
    "Provider",
    "RequestTrace",
    "User",
    "UserPatientMapping",
]
