"""Booking and cancelling, as a two-phase action (PRD §15, §26).

The shape of this module is the point. A write is split into **propose** and
**confirm**, and the model participates only in the first half:

    propose  parse the request → validate against real data → audit
             "proposed" → mint a signed token describing the validated
             action → answer the patient with what will happen
    confirm  verify the token against the calling session → execute → audit
             "executed"

The model never touches ``confirm``. It cannot: the confirm path takes a
token and reads its fields, and nothing in it accepts free text. So "the
assistant booked something I did not agree to" is not a prompt-injection
question — there is no sentence that reaches the INSERT.

Every outcome is audited, including the ones that never wrote anything.
A proposal the patient abandoned and a proposal that was rejected as invalid
are both facts worth having when someone asks what the assistant did.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from pydantic import BaseModel, Field
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.actions.tokens import ActionToken, mint_action_token
from app.auth.context import AuthContext
from app.models import Appointment, AuditLog, Provider
from app.models.enums import APPOINTMENT_TYPES
from app.observability.logging import get_logger

log = get_logger(__name__)

BOOK: Final = "book_appointment"
CANCEL: Final = "cancel_appointment"

#: How far ahead a booking may be made. Not a clinical rule — a sanity
#: bound, so a misparsed year ("2206") is refused rather than stored.
MAX_LEAD_DAYS: Final = 365
#: Appointments are on the hour or half hour; a proposal is snapped to the
#: nearest slot so two patients cannot hold 09:07 and 09:08.
SLOT_MINUTES: Final = 30


class ActionError(Exception):
    """The action cannot proceed. The message is shown to the patient.

    Distinct from a crash: every instance is a *valid* refusal with a reason
    the patient can act on ("that time is already booked"), so the wording
    is part of the interface rather than a developer diagnostic.
    """


class AppointmentRequest(BaseModel):
    """What the model is allowed to extract from the patient's sentence.

    Note what is absent: no patient id, no appointment id, no status, no
    free-text SQL. The model's entire influence over a write is these four
    fields, each of which is validated against real data before anything is
    proposed.
    """

    action: str = Field(
        description=f"Either {BOOK!r} or {CANCEL!r}.",
    )
    when: str = Field(
        default="",
        description=(
            "The date and time, ISO 8601 (YYYY-MM-DDTHH:MM). Resolve "
            "relative expressions like 'next Tuesday' against today's date, "
            "which is given in the prompt. Empty if the patient did not say."
        ),
    )
    appointment_type: str = Field(
        default="follow_up",
        description=f"One of: {', '.join(APPOINTMENT_TYPES)}.",
    )
    reason: str = Field(
        default="",
        description="The patient's stated reason, if any. One short phrase.",
    )


@dataclass(frozen=True, slots=True)
class Proposal:
    """A validated, not-yet-executed action awaiting the patient's word."""

    action: str
    summary: str
    token: str
    expires_at: datetime
    params: dict[str, Any]


# --- propose ------------------------------------------------------------- #


async def propose(
    session: AsyncSession,
    ctx: AuthContext,
    request: AppointmentRequest,
    *,
    now: datetime | None = None,
) -> Proposal:
    """Validate a parsed request and mint a confirmation token.

    Nothing is written to the appointment tables here. The only row this
    creates is the audit entry recording that a proposal was made.
    """
    now = now or datetime.now(UTC)

    if request.action == BOOK:
        params, summary = await _validate_booking(session, ctx, request, now=now)
    elif request.action == CANCEL:
        params, summary = await _validate_cancellation(session, ctx, request, now=now)
    else:
        raise ActionError(
            "I can only book or cancel appointments, and I could not tell "
            "which you meant. Please say 'book' or 'cancel'."
        )

    wire, token = mint_action_token(
        action=request.action,
        user_id=ctx.user_id,
        patient_id=ctx.patient_scope,
        params=params,
    )

    await _audit(
        session,
        ctx,
        action=request.action,
        outcome="proposed",
        params=params,
        target_id=params.get("appointment_id"),
        detail=token.token_id,
    )
    log.info(
        "action.proposed",
        action=request.action,
        token_id=token.token_id,
        patient_id=ctx.patient_scope,
    )
    return Proposal(
        action=request.action,
        summary=summary,
        token=wire,
        expires_at=token.expires_at,
        params=params,
    )


async def _validate_booking(
    session: AsyncSession,
    ctx: AuthContext,
    request: AppointmentRequest,
    *,
    now: datetime,
) -> tuple[dict[str, Any], str]:
    when = _parse_when(request.when)
    if when is None:
        raise ActionError(
            "I need a date and time to book an appointment. "
            "For example: 'book a follow-up next Tuesday at 10am'."
        )

    when = _snap_to_slot(when)
    if when <= now:
        raise ActionError(
            f"{_human(when)} is in the past. Please choose a future date."
        )
    if when > now + timedelta(days=MAX_LEAD_DAYS):
        raise ActionError(
            f"I can only book up to {MAX_LEAD_DAYS} days ahead. "
            f"{_human(when)} is further out than that."
        )

    appointment_type = request.appointment_type
    if appointment_type not in APPOINTMENT_TYPES:
        # Not an error the patient caused — the model chose a label outside
        # the vocabulary. Fall back rather than refuse a valid request.
        log.info("action.type_coerced", given=appointment_type)
        appointment_type = "follow_up"

    clash = await session.scalar(
        select(Appointment).where(
            and_(
                Appointment.patient_id == ctx.patient_scope,
                Appointment.appointment_date == when,
                Appointment.status == "scheduled",
            )
        )
    )
    if clash is not None:
        raise ActionError(
            f"You already have an appointment at {_human(when)}. "
            "Please pick another time."
        )

    provider = await _choose_provider(session, ctx)
    if provider is None:
        raise ActionError(
            "I could not find a provider to book with. Please contact the "
            "clinic directly."
        )

    params = {
        "when": when.isoformat(),
        "appointment_type": appointment_type,
        "provider_id": provider.id,
        "reason": request.reason[:200],
    }
    summary = (
        f"a {appointment_type.replace('_', ' ')} with {provider.name} "
        f"on {_human(when)}"
    )
    return params, summary


async def _validate_cancellation(
    session: AsyncSession,
    ctx: AuthContext,
    request: AppointmentRequest,
    *,
    now: datetime,
) -> tuple[dict[str, Any], str]:
    """Resolve which appointment to cancel — by time if given, else the next.

    The lookup is scoped to the caller's patient, so "cancel appointment
    1234" cannot reach another patient's row: an id that is not theirs
    simply does not match.
    """
    when = _parse_when(request.when)
    stmt = select(Appointment).where(
        and_(
            Appointment.patient_id == ctx.patient_scope,
            Appointment.status == "scheduled",
            Appointment.appointment_date >= now,
        )
    )
    if when is not None:
        window_start = _snap_to_slot(when) - timedelta(hours=12)
        window_end = _snap_to_slot(when) + timedelta(hours=12)
        stmt = stmt.where(
            Appointment.appointment_date.between(window_start, window_end)
        )

    target = await session.scalar(stmt.order_by(Appointment.appointment_date.asc()))
    if target is None:
        raise ActionError(
            "I could not find a scheduled appointment matching that. "
            "You can ask me to list your upcoming appointments."
        )

    params = {
        "appointment_id": target.id,
        "when": target.appointment_date.isoformat(),
    }
    return params, f"cancelling your appointment on {_human(target.appointment_date)}"


async def _choose_provider(
    session: AsyncSession, ctx: AuthContext
) -> Provider | None:
    """Prefer a provider the patient has already seen.

    Continuity of care, and it also keeps the model out of the choice: the
    provider is selected from the patient's own history rather than named in
    a sentence the model wrote.
    """
    seen = await session.scalar(
        select(Provider)
        .join(Appointment, Appointment.provider_id == Provider.id)
        .where(Appointment.patient_id == ctx.patient_scope)
        .order_by(Appointment.appointment_date.desc())
        .limit(1)
    )
    if seen is not None:
        return seen
    return await session.scalar(select(Provider).order_by(Provider.id).limit(1))


# --- confirm ------------------------------------------------------------- #


async def confirm(
    session: AsyncSession, ctx: AuthContext, token: ActionToken
) -> Appointment:
    """Execute a proposal. The only path in this module that writes.

    ``token`` has already been verified against the calling session by
    ``verify_action_token``; this function re-reads its parameters and
    nothing else.
    """
    if await _already_executed(session, token.token_id):
        raise ActionError("That confirmation has already been carried out.")

    try:
        if token.action == BOOK:
            appointment = await _execute_booking(session, ctx, token)
        elif token.action == CANCEL:
            appointment = await _execute_cancellation(session, ctx, token)
        else:
            raise ActionError("That confirmation is for an action I cannot perform.")
    except ActionError:
        await _audit(
            session,
            ctx,
            action=token.action,
            outcome="rejected",
            params=token.params,
            detail=token.token_id,
        )
        raise

    await _audit(
        session,
        ctx,
        action=token.action,
        outcome="executed",
        params=token.params,
        target_id=appointment.id,
        detail=token.token_id,
    )
    log.info(
        "action.executed",
        action=token.action,
        token_id=token.token_id,
        appointment_id=appointment.id,
        patient_id=ctx.patient_scope,
    )
    return appointment


async def _execute_booking(
    session: AsyncSession, ctx: AuthContext, token: ActionToken
) -> Appointment:
    when = datetime.fromisoformat(str(token.params["when"]))

    # Re-checked at execution, not just at proposal: minutes passed between
    # the two, and the slot may have been taken in between.
    clash = await session.scalar(
        select(Appointment).where(
            and_(
                Appointment.patient_id == ctx.patient_scope,
                Appointment.appointment_date == when,
                Appointment.status == "scheduled",
            )
        )
    )
    if clash is not None:
        raise ActionError(
            "That slot was taken while we were confirming. "
            "Please pick another time."
        )

    appointment = Appointment(
        patient_id=ctx.patient_scope,
        provider_id=int(token.params["provider_id"]),
        appointment_date=when,
        appointment_type=str(token.params["appointment_type"]),
        status="scheduled",
        notes=str(token.params.get("reason") or "") or None,
    )
    session.add(appointment)
    await session.flush()
    return appointment


async def _execute_cancellation(
    session: AsyncSession, ctx: AuthContext, token: ActionToken
) -> Appointment:
    appointment_id = int(token.params["appointment_id"])
    # Scoped by patient as well as id. The token is already bound to the
    # patient, so this is redundant — and it is the redundancy that means a
    # forged or mis-signed token still cannot reach another record.
    appointment = await session.scalar(
        select(Appointment).where(
            and_(
                Appointment.id == appointment_id,
                Appointment.patient_id == ctx.patient_scope,
            )
        )
    )
    if appointment is None:
        raise ActionError("That appointment is no longer in your record.")
    if appointment.status == "cancelled":
        raise ActionError("That appointment was already cancelled.")

    appointment.status = "cancelled"
    await session.flush()
    return appointment


async def _already_executed(session: AsyncSession, token_id: str) -> bool:
    """Idempotency: a token executes at most once.

    The token itself is stateless, so the audit log is what remembers. This
    is the reason ``detail`` carries the token id on every row.
    """
    existing = await session.scalar(
        select(AuditLog).where(
            and_(AuditLog.detail == token_id, AuditLog.outcome == "executed")
        )
    )
    return existing is not None


# --- audit --------------------------------------------------------------- #


async def _audit(
    session: AsyncSession,
    ctx: AuthContext,
    *,
    action: str,
    outcome: str,
    params: dict[str, Any],
    target_id: int | None = None,
    detail: str | None = None,
) -> None:
    """Record one step of an action.

    ``params`` are the validated fields, never the patient's raw sentence —
    an audit log that quotes free text becomes a second copy of the
    conversation, with none of its access controls.
    """
    session.add(
        AuditLog(
            request_id=ctx.request_id,
            user_id=ctx.user_id,
            patient_id=ctx.patient_id,
            action=action,
            target_type="appointment",
            target_id=target_id,
            outcome=outcome,
            params=params,
            detail=detail,
        )
    )
    await session.flush()


# --- parsing helpers ----------------------------------------------------- #


def _parse_when(raw: str) -> datetime | None:
    """Parse the model's ISO timestamp. Returns None rather than guessing."""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        log.info("action.unparsed_datetime", given=text[:40])
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _snap_to_slot(when: datetime) -> datetime:
    minute = (when.minute // SLOT_MINUTES) * SLOT_MINUTES
    return when.replace(minute=minute, second=0, microsecond=0)


def _human(when: datetime) -> str:
    return when.strftime("%A %d %B %Y at %H:%M")
