"""Short-term conversation memory (PRD §32).

Conversations are addressed by UUID rather than a serial id: the identifier
travels to the browser, and a guessable sequence invites the exact
cross-patient probing the authorization tests are meant to rule out.

Only the final, user-visible assistant text is stored. Routing rationale is
kept as the short ``reason`` string on the trace, never as chain-of-thought.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UpdatedAtMixin
from app.models.enums import MESSAGE_ROLES, ROUTES, check_in


class Conversation(Base, TimestampMixin, UpdatedAtMixin):
    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_user_id_created_at", "user_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    #: Denormalized from the user at creation time so that a conversation
    #: keeps the scope it was created under even if the account changes.
    patient_id: Mapped[int | None] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE")
    )
    title: Mapped[str | None] = mapped_column(String(200))

    #: A running summary of turns too old to replay verbatim (PRD §32).
    #: Stored rather than recomputed because summarizing is itself a model
    #: call: recomputing it every turn would add a call and its latency to
    #: every message in a long conversation, forever.
    summary: Mapped[str | None] = mapped_column(Text)
    #: The newest message included in ``summary``. Lets the next overflow be
    #: folded into the existing summary instead of re-reading the whole
    #: transcript, and makes it obvious which turns are already covered.
    summary_through_message_id: Mapped[int | None] = mapped_column(Integer)

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base, TimestampMixin):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(check_in("role", MESSAGE_ROLES), name="message_role"),
        CheckConstraint(
            f"route IS NULL OR {check_in('route', ROUTES)}", name="message_route"
        ),
        Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    route: Mapped[str | None] = mapped_column(String(16))
    #: Citations exactly as returned to the client, so a stored conversation
    #: can be re-rendered without re-running retrieval.
    sources: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
