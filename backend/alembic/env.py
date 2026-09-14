"""Alembic environment.

Migrations run over a *synchronous* psycopg connection even though the
application is async. DDL is a one-shot administrative operation; an event
loop buys nothing here and an async engine would only add failure modes to
the one code path that must work before anything else does.

The URL comes from ``app.config.settings``, never from alembic.ini, so a
migration cannot be aimed at a different database than the app by editing a
tracked file.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.models  # noqa: F401  (populates Base.metadata)
from app.config import settings
from app.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _sync_url() -> str:
    """psycopg3 speaks both protocols; the same URL works sync and async."""
    return settings.database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_sync_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_sync_url(), poolclass=pool.NullPool, future=True)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # Generated columns and vector opclasses are not round-tripped
            # faithfully by reflection; comparing them produces phantom
            # diffs on every autogenerate run.
            compare_server_default=False,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
