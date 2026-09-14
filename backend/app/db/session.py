"""Engine and session management.

Two distinct engines exist by design:

``app_engine``       read/write, used by every normal request path.
``analytics_engine`` the Text-to-SQL connection. It points at a role with
                     SELECT-only grants and row-level security, opens every
                     transaction ``READ ONLY``, and sets a short
                     ``statement_timeout``. Keeping it a separate engine means
                     a coding mistake in the SQL path cannot accidentally
                     borrow a privileged connection from the app pool.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.runtime import configure_async_runtime

# Must happen before any loop is created, and importing this module is the
# one thing every async database caller necessarily does.
configure_async_runtime()


def _make_engine(url: str, *, readonly: bool, pool_size: int) -> AsyncEngine:
    options = [f"-c statement_timeout={settings.db_statement_timeout_ms}"]
    if readonly:
        options = [
            f"-c statement_timeout={settings.sql_statement_timeout_ms}",
            "-c default_transaction_read_only=on",
            "-c idle_in_transaction_session_timeout=5000",
        ]
    return create_async_engine(
        url,
        pool_size=pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
        pool_recycle=1800,
        echo=False,
        connect_args={"options": " ".join(options)},
    )


app_engine: AsyncEngine = _make_engine(
    settings.database_url, readonly=False, pool_size=settings.db_pool_size
)

analytics_engine: AsyncEngine = _make_engine(
    settings.analytics_url, readonly=True, pool_size=4
)

AppSession = async_sessionmaker(app_engine, expire_on_commit=False, autoflush=False)
AnalyticsSession = async_sessionmaker(
    analytics_engine, expire_on_commit=False, autoflush=False
)


@event.listens_for(app_engine.sync_engine, "connect")
def _register_vector(dbapi_connection, _record) -> None:  # pragma: no cover
    """Teach psycopg how to adapt pgvector types on each new connection."""
    try:
        from pgvector.psycopg import register_vector

        register_vector(dbapi_connection)
    except Exception:
        # pgvector extension or package absent: the "array" backend handles
        # embeddings and needs no adapter.
        pass


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a read/write session."""
    async with AppSession() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def analytics_session(patient_id: int) -> AsyncIterator[AsyncSession]:
    """Open a read-only, patient-scoped session for Text-to-SQL.

    ``app.patient_id`` is set with ``SET LOCAL`` inside the transaction so
    the row-level security policies on the analytics role resolve to exactly
    one patient. This is the authorization boundary that does not depend on
    the generated SQL being correct — or on the SQL validator being
    bug-free.
    """
    async with AnalyticsSession() as session:
        await session.begin()
        await session.execute(text("SET LOCAL transaction_read_only = on"))
        await session.execute(
            text("SELECT set_config('app.patient_id', :pid, true)"),
            {"pid": str(int(patient_id))},
        )
        try:
            yield session
        finally:
            # Read-only work never commits; rollback releases the snapshot
            # and clears every SET LOCAL in one step.
            await session.rollback()


async def dispose_engines() -> None:
    await app_engine.dispose()
    await analytics_engine.dispose()
