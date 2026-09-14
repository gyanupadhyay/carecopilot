"""Declarative base with a deterministic constraint-naming convention.

Explicit names matter for Alembic: without them, autogenerate produces
unnamed constraints that cannot be dropped cleanly in a downgrade.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


#: Audit trails and clinical timelines are compared across sessions and,
#: eventually, across deployments; a naive local timestamp makes those
#: comparisons quietly wrong. Every bookkeeping column is timezone-aware.
_TZ_TIMESTAMP = DateTime(timezone=True)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        _TZ_TIMESTAMP, server_default=func.now(), nullable=False, index=True
    )


class UpdatedAtMixin:
    updated_at: Mapped[datetime] = mapped_column(
        _TZ_TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False
    )
