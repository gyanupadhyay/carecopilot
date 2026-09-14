"""Row-level security on the analytics connection.

This is the test that matters most in the whole suite. Everything else
checks that correct code produces correct results; this checks what happens
when the code is *wrong* — when a generated query carries no patient
predicate at all, or names a table it should never see, or tries to write.

The queries below are deliberately the queries a broken or hostile
Text-to-SQL path would emit. None of them are filtered by application code:
they are executed raw against the analytics role, and the database is what
refuses them.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError

from app.db.session import AnalyticsSession, analytics_session
from app.models import Patient

pytestmark = pytest.mark.integration


async def _analytics_available() -> bool:
    try:
        async with AnalyticsSession() as db:
            await db.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
async def _require_analytics_role() -> None:
    if not await _analytics_available():
        pytest.skip("Analytics role unavailable; run scripts/bootstrap_db.sql")


@pytest.fixture
async def patient_ids(session) -> tuple[int, int]:
    rows = (
        await session.scalars(select(Patient).order_by(Patient.id).limit(2))
    ).all()
    if len(rows) < 2:
        pytest.skip("Need at least two seeded patients")
    return rows[0].id, rows[1].id


async def test_unfiltered_select_returns_only_the_scoped_patient(
    patient_ids: tuple[int, int],
) -> None:
    """The failure mode this exists to stop.

    ``SELECT ... FROM lab_results`` has no WHERE clause. Under the analytics
    role it still returns one patient's rows, because the policy — not the
    query — decides what is visible.
    """
    scoped, _ = patient_ids
    async with analytics_session(scoped) as db:
        owners = (
            await db.execute(text("SELECT DISTINCT patient_id FROM lab_results"))
        ).scalars().all()
    assert owners == [scoped]


async def test_explicitly_requesting_another_patient_returns_nothing(
    patient_ids: tuple[int, int],
) -> None:
    """A query that names another patient outright yields an empty result."""
    scoped, other = patient_ids
    async with analytics_session(scoped) as db:
        result = await db.execute(
            text("SELECT count(*) FROM lab_results WHERE patient_id = :pid"),
            {"pid": other},
        )
    assert result.scalar_one() == 0


async def test_every_scoped_table_is_isolated(patient_ids: tuple[int, int]) -> None:
    scoped, _ = patient_ids
    tables = ("appointments", "encounters", "medications", "lab_results")
    async with analytics_session(scoped) as db:
        for table in tables:
            leaked = await db.execute(
                text(f"SELECT count(*) FROM {table} WHERE patient_id <> :pid"),
                {"pid": scoped},
            )
            assert leaked.scalar_one() == 0, f"{table} leaked rows across patients"


async def test_switching_scope_switches_the_visible_rows(
    patient_ids: tuple[int, int],
) -> None:
    """Each session sees its own patient — the scope is per-transaction."""
    first, second = patient_ids
    statement = text("SELECT DISTINCT patient_id FROM lab_results")

    async with analytics_session(first) as db:
        first_view = (await db.execute(statement)).scalars().all()
    async with analytics_session(second) as db:
        second_view = (await db.execute(statement)).scalars().all()

    assert first_view == [first]
    assert second_view == [second]


async def test_tables_outside_the_allowlist_are_invisible(
    patient_ids: tuple[int, int],
) -> None:
    """No grant means the table does not exist as far as this role knows."""
    scoped, _ = patient_ids
    # §16 approves exactly four tables for generated SQL. Everything else —
    # including patient-scoped clinical tables — is invisible to this role.
    for table in (
        "users",
        "user_patient_mapping",
        "conversations",
        "audit_logs",
        "patients",
        "providers",
        "clinical_documents",
        "document_chunks",
    ):
        async with analytics_session(scoped) as db:
            with pytest.raises(DBAPIError) as excinfo:
                await db.execute(text(f"SELECT count(*) FROM {table}"))
        assert "permission denied" in str(excinfo.value).lower()


async def test_writes_are_refused(patient_ids: tuple[int, int]) -> None:
    """Three independent layers refuse this; any one of them is enough."""
    scoped, _ = patient_ids
    statements = (
        "INSERT INTO lab_results (patient_id, test_name, value, unit, result_date) "
        "VALUES (:pid, 'Injected', 1, 'x', CURRENT_DATE)",
        "UPDATE medications SET dosage = '9999mg' WHERE patient_id = :pid",
        "DELETE FROM appointments WHERE patient_id = :pid",
    )
    for statement in statements:
        async with analytics_session(scoped) as db:
            # Read-only transaction, missing grant, or RLS — whichever fires
            # first, psycopg surfaces it as a DBAPIError.
            with pytest.raises(DBAPIError):
                await db.execute(text(statement), {"pid": scoped})


async def test_ddl_is_refused(patient_ids: tuple[int, int]) -> None:
    scoped, _ = patient_ids
    for statement in (
        "CREATE TABLE cc_should_not_exist (id int)",
        "DROP TABLE lab_results",
        "ALTER TABLE lab_results ADD COLUMN injected text",
    ):
        async with analytics_session(scoped) as db:
            with pytest.raises(DBAPIError):
                await db.execute(text(statement))


async def test_session_without_a_scope_sees_nothing() -> None:
    """An unset ``app.patient_id`` fails closed, not open.

    ``current_setting(..., true)`` returns NULL when the variable was never
    set, and ``patient_id = NULL`` is never true — so a bug that forgets to
    establish the scope returns an empty result rather than the whole table.
    """
    async with AnalyticsSession() as db:
        await db.begin()
        try:
            rows = (
                await db.execute(text("SELECT id FROM lab_results"))
            ).scalars().all()
        finally:
            await db.rollback()
    assert rows == []
