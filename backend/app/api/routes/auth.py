"""Login, session introspection, and the demo-account listing."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.auth.demo import DEMO_ACCOUNT_LIMIT, DEMO_PASSWORD
from app.auth.security import create_access_token, hash_password, verify_password
from app.config import settings
from app.models import User
from app.observability.logging import get_logger
from app.schemas.auth import DemoAccount, LoginRequest, SessionUser, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])
log = get_logger(__name__)

#: Verified against when the email is unknown, purely so that both branches
#: of a failed login do the same amount of work. Computed once at import.
_DUMMY_HASH = hash_password("this-password-matches-nothing")


def _session_user(user: User) -> SessionUser:
    return SessionUser(
        user_id=user.id,
        email=user.email,
        display_name=user.display_name,
        role=user.role,
        patient_id=user.patient_id,
        patient_external_id=user.patient.external_id if user.patient else None,
        patient_name=user.patient.full_name if user.patient else None,
    )


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, session: DbSession) -> TokenResponse:
    # The patient link and its patient load eagerly with the user (see
    # User.patient_links), so the scope is resolved in the same round trip
    # that authenticates.
    user = await session.scalar(
        select(User).where(User.email == payload.email.lower())
    )

    # Hash verification runs even when there is no such user. Skipping it
    # would return in microseconds for unknown addresses and in ~50ms for
    # known ones, which turns login latency into a user-enumeration oracle.
    password_ok = verify_password(
        user.password_hash if user else _DUMMY_HASH, payload.password
    )
    if user is None or not password_ok or not user.is_active:
        log.info("auth.login_failed", email_domain=payload.email.split("@")[-1])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    token, expires_at = create_access_token(
        user_id=user.id, role=user.role, patient_id=user.patient_id
    )
    log.info("auth.login", user_id=user.id, patient_id=user.patient_id)
    return TokenResponse(
        access_token=token, expires_at=expires_at, user=_session_user(user)
    )


@router.get("/me", response_model=SessionUser)
async def me(ctx: CurrentUser, session: DbSession) -> SessionUser:
    user = await session.scalar(
        select(User).where(User.id == ctx.user_id)
    )
    if user is None:  # pragma: no cover - get_auth_context already checked
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return _session_user(user)


@router.get("/demo-accounts", response_model=list[DemoAccount])
async def demo_accounts(session: DbSession) -> list[DemoAccount]:
    """List seeded logins so the demo can be opened without a handoff.

    Refused outside development unless DEMO_ACCOUNTS_PUBLIC is set. The
    dataset is synthetic either way, but an endpoint that hands out working
    credentials should not be reachable in a deployed environment just
    because the data behind it is fake.

    The exception exists because a public demo inverts the trade: a visitor
    who cannot get past the sign-in screen is the whole thing failing, and
    the credentials are published in the README regardless. See
    ``demo_accounts_public`` in app/config.py.
    """
    if settings.environment != "development" and not settings.demo_accounts_public:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Not found."
        )

    users = (
        await session.scalars(
            select(User)
                .where(User.role == "patient", User.is_active.is_(True))
            .order_by(User.id)
            .limit(DEMO_ACCOUNT_LIMIT)
        )
    ).all()

    return [
        DemoAccount(
            email=user.email,
            password=DEMO_PASSWORD,
            display_name=user.display_name,
            patient_external_id=user.patient.external_id if user.patient else None,
        )
        for user in users
    ]
