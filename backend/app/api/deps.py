"""Request-scoped dependencies.

The important decision here is that the *database* is authoritative for a
session's patient scope, not the token. The JWT carries ``pid`` so the
frontend can render without a round trip, but
:func:`get_auth_context` re-reads the user row on every request and builds
the :class:`~app.auth.context.AuthContext` from that. A token minted before
an account was deactivated or re-pointed therefore stops working at once,
without a revocation list.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.context import AuthContext, Role
from app.auth.security import TokenError, decode_access_token
from app.db.session import get_session
from app.llm.base import LLMProvider
from app.llm.factory import get_llm
from app.models import User
from app.observability.middleware import current_request_id
from app.rag.embeddings import EmbeddingProvider, get_embedder

_bearer = HTTPBearer(auto_error=False, description="Bearer access token")

DbSession = Annotated[AsyncSession, Depends(get_session)]


async def get_auth_context(
    session: DbSession,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> AuthContext:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        claims = decode_access_token(credentials.credentials)
    except TokenError:
        # One message for every token failure: distinguishing "expired" from
        # "bad signature" tells an attacker which half to work on.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    user = await session.scalar(select(User).where(User.id == claims.user_id))
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return AuthContext(
        user_id=user.id,
        role=user.role,  # type: ignore[arg-type]
        patient_id=user.patient_id,
        request_id=current_request_id(),
    )


CurrentUser = Annotated[AuthContext, Depends(get_auth_context)]


async def require_patient_scope(ctx: CurrentUser) -> AuthContext:
    """Reject sessions with no patient record before any query runs."""
    if ctx.patient_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account is not linked to a patient record.",
        )
    return ctx


PatientScoped = Annotated[AuthContext, Depends(require_patient_scope)]


# --- AI dependencies ---------------------------------------------------- #
#
# Injectable rather than imported directly at the call site, so a test can
# substitute a deterministic provider or the fast hashing embedder without
# monkeypatching module globals.


def llm_provider() -> LLMProvider:
    return get_llm()


def embedding_provider() -> EmbeddingProvider:
    return get_embedder()


Llm = Annotated[LLMProvider, Depends(llm_provider)]
Embedder = Annotated[EmbeddingProvider, Depends(embedding_provider)]

__all__ = [
    "CurrentUser",
    "DbSession",
    "Embedder",
    "Llm",
    "PatientScoped",
    "Role",
    "embedding_provider",
    "get_auth_context",
    "llm_provider",
    "require_patient_scope",
]
