"""The database and the models must agree (PRD §32).

This file exists because of a specific bug. Migration 0001 passed
already-prefixed constraint names to ``op.create_table`` while the metadata
carried a naming convention, so every explicit CHECK ended up double
prefixed — ``ck_messages_ck_messages_message_route``. Nothing noticed until
a later migration tried to replace one *by name*: the drop matched nothing,
the add created a second constraint, and both were enforced. Requests that
satisfied the new rule violated the stale one.

The unit and integration suites all missed it, because every test wrote a
route value that was legal under both vocabularies. Only a live request took
the new branch. These tests compare the schema against the models directly,
so the next such drift fails here instead of in production.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.db.base import Base
from app.models.enums import ROUTES

pytestmark = pytest.mark.integration


async def _constraints(session, pattern: str = "%") -> list[tuple[str, str]]:
    rows = await session.execute(
        text(
            """
            SELECT conrelid::regclass::text AS tbl, conname
              FROM pg_constraint
             WHERE contype = 'c' AND conname LIKE :pattern
             ORDER BY 1, 2
            """
        ),
        {"pattern": pattern},
    )
    return [(row.tbl, row.conname) for row in rows]


async def test_no_constraint_name_is_double_prefixed(session) -> None:
    """``ck_x_ck_x_y`` is the signature of a convention applied twice."""
    doubled = await _constraints(session, "ck_%_ck_%")
    assert doubled == [], f"double-prefixed constraints: {doubled}"


async def test_model_check_constraints_exist_in_the_database(session) -> None:
    """Every CHECK the models declare is present under the expected name.

    A mismatch means a migration renamed or replaced something the models
    still address by the old name — exactly the failure this file records.
    """
    expected: set[tuple[str, str]] = set()
    for table in Base.metadata.tables.values():
        for constraint in table.constraints:
            name = getattr(constraint, "name", None)
            if name and str(name).startswith("ck_"):
                expected.add((table.name, str(name)))

    actual = set(await _constraints(session, "ck_%"))
    missing = expected - actual
    assert not missing, f"declared by the models but absent from the database: {missing}"


async def test_exactly_one_route_constraint_per_table(session) -> None:
    """Two constraints on one column can disagree, and both are enforced."""
    rows = await session.execute(
        text(
            """
            SELECT conrelid::regclass::text AS tbl, count(*) AS n
              FROM pg_constraint
             WHERE contype = 'c' AND pg_get_constraintdef(oid) ILIKE '%route%'
             GROUP BY 1
            """
        )
    )
    counts = {row.tbl: row.n for row in rows}
    assert counts, "expected route constraints on messages and request_traces"
    for table, count in counts.items():
        assert count == 1, f"{table} has {count} route constraints"


@pytest.mark.parametrize("route", ROUTES)
async def test_every_route_value_is_accepted_by_the_database(
    session, route: str
) -> None:
    """The vocabulary the code emits must be the one the schema permits.

    Parametrized over the enum so a route the router can produce but the
    database rejects fails here rather than on the first live request that
    takes that branch.
    """
    for constraint in ("ck_messages_message_route", "ck_request_traces_trace_route"):
        definition = await session.scalar(
            text(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = :name"
            ),
            {"name": constraint},
        )
        assert definition is not None, f"{constraint} is missing"
        assert f"'{route}'" in definition, (
            f"the router can emit {route}, but {constraint} rejects it"
        )
