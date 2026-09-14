"""Clinical record payloads.

These double as the typed return values of the tool layer (PRD §15), which
is why they carry display-ready derived fields such as ``provider_name``
and ``is_active``. Computing them here, once, keeps the LLM from having to
join or infer anything — and an LLM that never has to infer a fact cannot
get it wrong.
"""

from __future__ import annotations

from datetime import date, datetime
from datetime import date as date_type
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, computed_field

from app.schemas.common import ORMModel


class ProviderOut(ORMModel):
    id: int
    name: str
    specialty: str


class PatientOut(ORMModel):
    id: int
    external_id: str
    first_name: str
    last_name: str
    date_of_birth: date
    gender: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def age(self) -> int:
        """Whole years, computed server-side.

        Age is the kind of small arithmetic a model will happily get wrong
        near a birthday, and it appears in almost every clinical summary.
        """
        today = date.today()
        had_birthday = (today.month, today.day) >= (
            self.date_of_birth.month,
            self.date_of_birth.day,
        )
        return today.year - self.date_of_birth.year - (0 if had_birthday else 1)


class AppointmentOut(ORMModel):
    id: int
    appointment_date: datetime
    appointment_type: str
    status: str
    notes: str | None = None
    provider: ProviderOut | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def provider_name(self) -> str | None:
        return self.provider.name if self.provider else None


class MedicationOut(ORMModel):
    id: int
    name: str
    dosage: str
    frequency: str
    status: str
    start_date: date
    end_date: date | None = None
    encounter_id: int | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_active(self) -> bool:
        return self.status == "active" and self.end_date is None


class LabResultOut(ORMModel):
    id: int
    test_name: str
    value: Decimal
    unit: str
    reference_range: str | None = None
    result_date: date
    encounter_id: int | None = None


class EncounterOut(ORMModel):
    id: int
    encounter_date: date
    encounter_type: str
    reason: str | None = None
    provider: ProviderOut | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def provider_name(self) -> str | None:
        return self.provider.name if self.provider else None


class ClinicalNoteHit(ORMModel):
    """One retrieved note fragment, with everything needed to cite it.

    Returned by ``GET /api/clinical-notes/search``. The ids are the
    application's own, so a caller can follow a hit back to its document and
    encounter without trusting anything the model wrote.
    """

    chunk_id: int
    document_id: int
    encounter_id: int | None = None
    title: str | None = None
    document_type: str | None = None
    section: str
    #: Aliased type: a field named ``date`` shadows the ``date`` type inside
    #: the class body, leaving the annotation unresolvable.
    date: date_type | None = None
    score: float
    text: str


class MedicationChange(ORMModel):
    """One deterministic before/after difference around an encounter.

    Produced by backend code, never by the model (PRD §40 P12): comparing two
    medication lists is exact arithmetic over dates and dosages, and an LLM
    asked to do it will occasionally invent a change that did not happen.
    """

    change: str = Field(description="started | stopped | dose_changed | unchanged")
    name: str
    before: str | None = None
    after: str | None = None


class GraphQueryOut(BaseModel):
    """One traversal of the patient's own relationship graph (PRD §17).

    ``rows`` is deliberately untyped beyond "JSON values": each intent
    returns a different shape, and seven near-identical row models would
    describe the Cypher rather than help anyone reading this. The
    ``summary`` is the part composed by the backend from returned values,
    which is what both the model and the developer panel should lead with.
    """

    intent: str
    term: str | None = None
    summary: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    count: int


class AnalyticsRequest(BaseModel):
    """A question, in English. Never a SQL statement (PRD §16).

    There is deliberately no field through which SQL could arrive. A caller
    who sends a statement is sending a string that gets read as a question,
    and the generator declines it — which is what the router's OUT_OF_SCOPE
    rule for "run this SELECT for me" already does one layer earlier.
    """

    question: str = Field(min_length=3, max_length=500)


class AnalyticsOut(BaseModel):
    """One analytics answer, with the statement that produced it.

    The SQL is returned because an aggregate the caller cannot check is an
    assertion. It is mechanism, not clinical content: the figures are in
    ``rows``, and showing the query is what lets a developer see why the
    number is what it is.
    """

    question: str
    sql: str
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    count: int
    truncated: bool = False
    #: The same rows rendered as a small table, for a model to read.
    table: str
