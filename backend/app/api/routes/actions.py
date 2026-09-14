"""Confirming a proposed write (PRD §15, §26).

One endpoint, and its shape is the security argument. It accepts a signed
token and nothing else — no date, no appointment id, no free text. Every
parameter of the action was fixed when the proposal was made and validated;
this request only says *yes*.

So the question "can a prompt injection make the assistant book something?"
has a structural answer rather than a probabilistic one. Reaching this
endpoint requires a token, minting a token happens only in
``actions.appointments.propose`` after validation against real data, and the
token is bound to the session that was shown the proposal. There is no
sentence — in a note, in a document, in the patient's own message — that
produces a write, because no sentence reaches this handler.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.actions import appointments
from app.actions.appointments import AppointmentRequest
from app.actions.tokens import ActionTokenError, verify_action_token
from app.api.deps import DbSession, PatientScoped
from app.observability.logging import get_logger
from app.schemas.chat import (
    ActionConfirmRequest,
    ActionProposeRequest,
    ActionResult,
    PendingAction,
)

log = get_logger(__name__)

router = APIRouter(prefix="/actions", tags=["actions"])


@router.post("/propose", response_model=PendingAction)
async def propose_action(
    payload: ActionProposeRequest,
    ctx: PatientScoped,
    session: DbSession,
) -> PendingAction:
    """Validate a requested action and return it for confirmation.

    Writes nothing to the appointment tables. It validates the parameters
    against real data — is that slot free, does that appointment exist and
    belong to the caller — records that a proposal was made, and mints a
    short-lived token bound to this user and patient.

    This exists so ``book_my_appointment`` and ``cancel_my_appointment`` can
    be MCP tools (PRD §15) without the tool surface being able to complete a
    write. The confirm endpoint below is deliberately **not** exposed through
    MCP: if one caller could both propose and confirm, the two-step design
    would collapse into one step and "can a prompt injection make the
    assistant book something?" would go back to being a question about how
    good the model is at resisting persuasion.

    The parameters here are typed and validated, never free text. The
    patient's sentence is parsed into an ``AppointmentRequest`` somewhere
    else, by a model, and that model's entire influence is these fields.
    """
    request = AppointmentRequest(
        action=payload.action,
        when=payload.when,
        appointment_type=payload.appointment_type,
        reason=payload.reason,
    )
    try:
        proposal = await appointments.propose(session, ctx, request)
    except appointments.ActionError as exc:
        # 422: the request was understood and refused for a reason the
        # caller can act on ("that time is already booked").
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return PendingAction(
        action=proposal.action,
        summary=proposal.summary,
        token=proposal.token,
        expires_at=proposal.expires_at.isoformat(),
    )


@router.post("/confirm", response_model=ActionResult)
async def confirm_action(
    payload: ActionConfirmRequest,
    ctx: PatientScoped,
    session: DbSession,
) -> ActionResult:
    """Execute a previously proposed action.

    ``ctx`` supplies the identity the token is checked against — the token's
    own claims about who owns it are never taken at face value.
    """
    try:
        token = verify_action_token(
            payload.token, user_id=ctx.user_id, patient_id=ctx.patient_scope
        )
    except ActionTokenError as exc:
        # 400, not 403: an expired or mismatched confirmation is a request
        # that can be retried by asking again, not an authorization failure
        # against a resource the caller might otherwise reach.
        log.info("action.token_rejected", reason=str(exc)[:80])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    try:
        appointment = await appointments.confirm(session, ctx, token)
    except appointments.ActionError as exc:
        # A refusal with a reason the patient can act on — the slot was
        # taken, the appointment is already cancelled. Reported as a result,
        # not an error, because the request itself was well-formed.
        return ActionResult(status="declined", message=str(exc))

    verb = "booked" if token.action == appointments.BOOK else "cancelled"
    return ActionResult(
        status="executed",
        message=f"Your appointment has been {verb}.",
        appointment_id=appointment.id,
    )
