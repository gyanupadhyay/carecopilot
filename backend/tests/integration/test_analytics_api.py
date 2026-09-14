"""The analytics endpoint and the propose endpoint (PRD §15, §16).

These are the two capabilities PRD §15 names as MCP tools, and the two whose
*absence of power* is the thing worth testing. Analytics must answer a
question without ever accepting a statement; proposing an action must
validate and record without writing anything.

The model is stubbed. What is under test is the boundary — what the endpoint
accepts, what it writes, what it refuses — not whether Qwen3 writes good SQL,
which is the evaluation set's job and takes a minute per case.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from app.api.deps import llm_provider
from app.auth.demo import DEMO_PASSWORD
from app.llm.base import StructuredResponse, TokenUsage
from app.llm.stub import StubProvider
from app.main import create_app
from app.models import Appointment, AuditLog
from app.sql.analytics import ANALYTICS_ACTION
from app.sql.generator import GeneratedSQL

pytestmark = pytest.mark.integration

BASE = "http://test/api"

#: Deliberately without a patient_id predicate. Row-level security on the
#: read-only role scopes the connection, so a predicate here could only
#: wrongly exclude the caller's own rows.
COUNT_SQL = "SELECT COUNT(*) AS n FROM lab_results"


class PinnedSQL(StubProvider):
    """Returns a fixed ``GeneratedSQL``, recording what it was asked."""

    def __init__(self, generated: GeneratedSQL) -> None:
        super().__init__()
        self._generated = generated
        self.questions: list[str] = []

    async def generate_structured(self, *, messages, system, schema, **kwargs):  # type: ignore[override]
        self.questions.append(messages[-1].content if messages else "")
        return StructuredResponse(
            value=self._generated,
            model="fake-sql",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            latency_ms=1,
            provider="fake",
        )


def _client_with(session, generated: GeneratedSQL) -> tuple[AsyncClient, PinnedSQL]:
    llm = PinnedSQL(generated)
    app = create_app()
    app.dependency_overrides[llm_provider] = lambda: llm
    return (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test"),
        llm,
    )


@pytest.fixture
async def answerable(session):
    http, llm = _client_with(
        session, GeneratedSQL(answerable=True, sql=COUNT_SQL, reason="counts labs")
    )
    async with http:
        yield http, llm


async def _login(client: AsyncClient, external_id: str) -> dict[str, str]:
    response = await client.post(
        f"{BASE}/auth/login",
        json={
            "email": f"{external_id.lower()}@carecopilot.demo",
            "password": DEMO_PASSWORD,
        },
    )
    if response.status_code != 200:
        pytest.skip(f"{external_id} not seeded; run scripts/generate_data.py --reset")
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


async def _post(client, path: str, headers, payload: dict[str, Any]):
    return await client.post(f"{BASE}{path}", json=payload, headers=headers)


# --- analytics ----------------------------------------------------------- #


async def test_analytics_answers_an_aggregate_question(answerable) -> None:
    client, _ = answerable
    headers = await _login(client, "P001")
    response = await _post(
        client, "/analytics", headers, {"question": "How many lab results do I have?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["columns"] == ["n"]
    assert body["sql"].lower().startswith("select")
    assert body["table"]


async def test_analytics_is_scoped_to_the_caller(answerable) -> None:
    """The same statement, two patients, two answers.

    Nothing in the request distinguishes them. The row-level security keyed
    to the connection's patient id does, which is the §16 guarantee: the
    scope is not in the SQL and cannot be removed from it.
    """
    client, _ = answerable
    first = await _post(
        client, "/analytics", await _login(client, "P001"), {"question": "how many?"}
    )
    second = await _post(
        client, "/analytics", await _login(client, "P002"), {"question": "how many?"}
    )
    assert first.status_code == second.status_code == 200
    assert first.json()["rows"][0][0] != second.json()["rows"][0][0], (
        "two patients returned the same lab count; RLS may not be scoping "
        "the analytics connection"
    )


async def test_every_analytics_run_is_audited(answerable, session) -> None:
    """§16 lists audit logging beside scope, validation and row limits.

    It is the one item on that list no other layer enforces: a query that
    was scoped, validated and capped still leaves no trace that it ran.
    """
    client, _ = answerable
    before = await session.scalar(
        select(func.count())
        .select_from(AuditLog)
        .where(AuditLog.action == ANALYTICS_ACTION)
    )
    await _post(
        client, "/analytics", await _login(client, "P001"), {"question": "how many?"}
    )
    await session.commit()
    after = await session.scalar(
        select(func.count())
        .select_from(AuditLog)
        .where(AuditLog.action == ANALYTICS_ACTION)
    )
    assert after == before + 1


async def test_the_audit_row_records_the_statement_and_not_the_rows(
    answerable, session
) -> None:
    """The SQL is mechanism and belongs in an audit trail.

    The rows are clinical data; putting them here would make the audit log a
    second, less protected copy of the record.
    """
    client, _ = answerable
    await _post(
        client, "/analytics", await _login(client, "P001"), {"question": "how many?"}
    )
    await session.commit()
    row = await session.scalar(
        select(AuditLog)
        .where(AuditLog.action == ANALYTICS_ACTION)
        .order_by(AuditLog.id.desc())
        .limit(1)
    )
    assert row is not None
    assert row.outcome == "executed"
    assert "select" in str(row.params.get("sql", "")).lower()
    assert "rows" not in row.params


async def test_a_question_the_schema_cannot_answer_is_refused(session) -> None:
    """Stated as a fact about the data, not dressed up as an answer."""
    http, _ = _client_with(
        session,
        GeneratedSQL(answerable=False, reason="The schema holds no billing data."),
    )
    async with http as client:
        response = await _post(
            client,
            "/analytics",
            await _login(client, "P001"),
            {"question": "How much have I been billed this year?"},
        )
    assert response.status_code == 422
    assert "billing" in response.json()["detail"].lower()


async def test_a_refusal_is_audited_too(session) -> None:
    """A burst of refusals is what probing the schema looks like."""
    http, _ = _client_with(
        session, GeneratedSQL(answerable=False, reason="not in this schema")
    )
    before = await session.scalar(
        select(func.count())
        .select_from(AuditLog)
        .where(AuditLog.action == ANALYTICS_ACTION)
    )
    async with http as client:
        await _post(
            client,
            "/analytics",
            await _login(client, "P001"),
            {"question": "anything at all?"},
        )
    await session.commit()
    after = await session.scalar(
        select(func.count())
        .select_from(AuditLog)
        .where(AuditLog.action == ANALYTICS_ACTION)
    )
    assert after == before + 1


async def test_the_endpoint_has_no_parameter_carrying_sql(answerable) -> None:
    """A statement sent as the question is read as a question, not run."""
    client, llm = answerable
    headers = await _login(client, "P001")
    response = await client.post(
        f"{BASE}/analytics",
        json={
            "question": "x",
            "sql": "DROP TABLE appointments",
            "statement": "DELETE FROM medications",
        },
        headers=headers,
    )
    # Extra keys are ignored by the schema rather than reaching anything.
    assert response.status_code in (200, 422)
    assert not any("drop table" in q.lower() for q in llm.questions)


async def test_analytics_requires_authentication(answerable) -> None:
    client, _ = answerable
    response = await client.post(f"{BASE}/analytics", json={"question": "how many?"})
    assert response.status_code == 401


# --- propose ------------------------------------------------------------- #


async def test_proposing_a_booking_writes_no_appointment(answerable, session) -> None:
    """The central claim of the action design, at the HTTP boundary.

    ``book_my_appointment`` is an MCP tool; if proposing wrote a row, the
    tool surface would be able to change the record on its own say-so.
    """
    from datetime import UTC, datetime, timedelta

    client, _ = answerable
    headers = await _login(client, "P001")
    when = (datetime.now(UTC) + timedelta(days=11)).replace(
        hour=10, minute=0, second=0, microsecond=0
    )

    before = await session.scalar(select(func.count()).select_from(Appointment))
    response = await _post(
        client,
        "/actions/propose",
        headers,
        {
            "action": "book_appointment",
            "when": when.strftime("%Y-%m-%dT%H:%M"),
            "appointment_type": "follow_up",
            "reason": "knee review",
        },
    )
    await session.commit()
    after = await session.scalar(select(func.count()).select_from(Appointment))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["token"]
    assert body["summary"]
    assert body["action"] == "book_appointment"
    assert after == before, "proposing an appointment created one"


async def test_propose_refuses_an_unknown_action(answerable) -> None:
    """The action is a closed set, rejected by the schema before the handler."""
    client, _ = answerable
    response = await _post(
        client,
        "/actions/propose",
        await _login(client, "P001"),
        {"action": "delete_my_records", "when": "2026-10-01T10:00"},
    )
    assert response.status_code == 422


async def test_propose_takes_no_patient_or_appointment_id(answerable, session) -> None:
    """Extra identifiers are ignored, not honoured.

    An appointment is identified by its time and the caller's own scope, so
    there is no id parameter whose validation could be the only thing
    standing between a caller and someone else's row.
    """
    client, _ = answerable
    response = await _post(
        client,
        "/actions/propose",
        await _login(client, "P001"),
        {
            "action": "cancel_appointment",
            "when": "2020-01-01T09:00",
            "patient_id": 2,
            "appointment_id": 999999,
        },
    )
    # Refused because no such appointment exists for *this* patient at that
    # time — never because it found appointment 999999.
    assert response.status_code == 422


async def test_propose_requires_authentication(answerable) -> None:
    client, _ = answerable
    response = await client.post(
        f"{BASE}/actions/propose",
        json={"action": "book_appointment", "when": "2026-10-01T10:00"},
    )
    assert response.status_code == 401
