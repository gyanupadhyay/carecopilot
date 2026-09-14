"""Patient record endpoints (PRD §9).

No path or query parameter names a patient. ``GET /api/labs`` returns *your*
labs, where "your" is resolved from the JWT through the identity mapping —
there is no id in the URL to tamper with, and therefore no request shape in
which tampering is even expressible.

That is the whole point of §9. An endpoint like
``/api/labs?patient_id=456`` can be made safe by checking the parameter
against the session, and the earlier revision of this API did exactly that.
But it leaves a URL that *looks* like it addresses any patient, invites a
client to construct one, and makes every future handler a place where
someone can forget the check. Removing the parameter removes the class.

Booking and cancellation (``POST /api/appointments/book``, ``/cancel``) are
deliberately absent: they are actions, and §25 requires validation,
confirmation and audit logging around them. They arrive with the action
workflow rather than as bare mutating endpoints.
"""

from __future__ import annotations

from datetime import date, datetime

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import DbSession, Embedder, PatientScoped
from app.rag import pipeline as rag_pipeline
from app.schemas.clinical import (
    AppointmentOut,
    ClinicalNoteHit,
    EncounterOut,
    LabResultOut,
    MedicationOut,
    PatientOut,
)
from app.schemas.common import Page
from app.services import clinical

router = APIRouter(tags=["records"])


def _page[T](items: list[T]) -> Page[T]:
    return Page(items=items, count=len(items))


@router.get("/me", response_model=PatientOut)
async def read_me(ctx: PatientScoped, session: DbSession) -> PatientOut:
    """The authenticated patient's own demographics."""
    patient = await clinical.get_patient_profile(session, ctx)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found."
        )
    return PatientOut.model_validate(patient)


# ---------------------------------------------------------------------- #
# Appointments
# ---------------------------------------------------------------------- #


@router.get("/appointments", response_model=Page[AppointmentOut])
async def read_appointments(
    ctx: PatientScoped,
    session: DbSession,
    status_filter: str | None = Query(default=None, alias="status"),
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> Page[AppointmentOut]:
    rows = await clinical.get_appointments(
        session,
        ctx,
        status=status_filter,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
    )
    return _page([AppointmentOut.model_validate(r) for r in rows])


@router.get("/appointments/next", response_model=AppointmentOut | None)
async def read_next_appointment(
    ctx: PatientScoped, session: DbSession
) -> AppointmentOut | None:
    """The soonest scheduled appointment, or null.

    Null rather than 404: "you have nothing booked" is a successful answer
    to the question, not a missing resource.
    """
    appointment = await clinical.get_next_appointment(session, ctx)
    return AppointmentOut.model_validate(appointment) if appointment else None


# ---------------------------------------------------------------------- #
# Medications, labs, encounters
# ---------------------------------------------------------------------- #


@router.get("/medications", response_model=Page[MedicationOut])
async def read_medications(
    ctx: PatientScoped,
    session: DbSession,
    as_of: date | None = None,
) -> Page[MedicationOut]:
    rows = await clinical.get_current_medications(session, ctx, as_of=as_of)
    return _page([MedicationOut.model_validate(r) for r in rows])


@router.get("/labs", response_model=Page[LabResultOut])
async def read_labs(
    ctx: PatientScoped,
    session: DbSession,
    test_name: str | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
    limit: int = Query(default=100, ge=1, le=200),
) -> Page[LabResultOut]:
    rows = await clinical.get_lab_results(
        session,
        ctx,
        test_name=test_name,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
    )
    return _page([LabResultOut.model_validate(r) for r in rows])


@router.get("/encounters", response_model=Page[EncounterOut])
async def read_encounters(
    ctx: PatientScoped,
    session: DbSession,
    limit: int = Query(default=20, ge=1, le=200),
) -> Page[EncounterOut]:
    rows = await clinical.get_encounters(session, ctx, limit=limit)
    return _page([EncounterOut.model_validate(r) for r in rows])


@router.get("/encounters/latest", response_model=EncounterOut | None)
async def read_latest_encounter(
    ctx: PatientScoped, session: DbSession
) -> EncounterOut | None:
    """The most recent encounter — the anchor for hybrid questions."""
    encounter = await clinical.get_last_encounter(session, ctx)
    return EncounterOut.model_validate(encounter) if encounter else None


# ---------------------------------------------------------------------- #
# Clinical note search
# ---------------------------------------------------------------------- #


@router.get("/clinical-notes/search", response_model=Page[ClinicalNoteHit])
async def search_clinical_notes(
    ctx: PatientScoped,
    session: DbSession,
    embed: Embedder,
    q: str = Query(min_length=2, max_length=500, description="What to search for"),
    limit: int = Query(default=8, ge=1, le=20),
    section: str | None = Query(
        default=None, description="Restrict to one note section, e.g. Assessment"
    ),
) -> Page[ClinicalNoteHit]:
    """Semantic search across the caller's own clinical notes.

    Runs the same retrieval path the assistant uses, so what this returns is
    exactly what the model would have been given — which makes it the honest
    way to inspect why an answer said what it did.
    """
    result = await rag_pipeline.retrieve(
        session,
        ctx,
        question=q,
        embedder=embed,
        top_k=limit,
        sections=(section,) if section else None,
    )
    return _page(
        [
            ClinicalNoteHit(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                encounter_id=chunk.encounter_id,
                title=chunk.title,
                document_type=chunk.document_type,
                section=chunk.section,
                date=chunk.chunk_date,
                score=round(chunk.score, 4),
                text=chunk.text,
            )
            for chunk in result.chunks
        ]
    )


__all__ = ["router"]
