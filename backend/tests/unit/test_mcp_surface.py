"""The MCP tool surface (PRD §15).

Schema-level tests, which is where the guarantee actually lives. §15 asks
that "MCP exposes only approved tools" and that it "cannot bypass backend
authorization" — the first is a list comparison, and the second is provable
from the tool signatures: if no tool accepts a patient identifier, no client
can ask for another patient's records, whatever it sends.

These run without a server. The live path is covered by the end-to-end
smoke, which needs both processes up.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "mcp-server"))

server = pytest.importorskip("server", reason="mcp-server/server.py not importable")


class _FakeContext:
    """Just enough of the MCP Context to exercise token extraction."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


def _ctx(headers: dict[str, str]) -> _FakeContext:
    return _FakeContext(headers)


@pytest.fixture(scope="module")
def tools() -> dict:
    listed = asyncio.run(server.mcp.list_tools())
    return {tool.name: tool for tool in listed}


def _properties(tool) -> dict:
    # MCP 2.x renamed the field from inputSchema; accept either so the test
    # does not silently pass against a version that exposes neither.
    schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
    assert schema is not None, f"{tool.name} exposes no input schema"
    return schema.get("properties", {}) or {}


def test_only_approved_tools_are_exposed(tools: dict) -> None:
    assert set(tools) == set(server.APPROVED_TOOLS)


def test_no_tool_accepts_a_patient_identifier(tools: dict) -> None:
    """The authorization guarantee, stated as a schema property.

    A tool that cannot express "which patient" cannot be called with the
    wrong one — there is no argument for a validator to miss.
    """
    forbidden = {"patient_id", "patientid", "patient", "user_id", "subject"}
    for name, tool in tools.items():
        supplied = {key.lower() for key in _properties(tool)}
        assert not (supplied & forbidden), f"{name} accepts {supplied & forbidden}"


def test_no_tool_accepts_raw_sql(tools: dict) -> None:
    """§16: ``execute_sql`` must never be exposed as a general tool."""
    for name, tool in tools.items():
        assert "sql" not in name.lower()
        assert not {"sql", "query_sql", "statement"} & set(_properties(tool))


def test_the_graph_tool_accepts_no_cypher(tools: dict) -> None:
    """§21's "incorrect" path is arbitrary Cypher from the model.

    The tool takes an ``intent`` from a closed set and a search word. A
    parameter named for a query language, or a free-text one called
    ``cypher``/``query``, would reopen exactly what the fixed templates in
    ``app/knowledge_graph/queries.py`` exist to close.
    """
    graph = tools.get("query_my_patient_graph")
    assert graph is not None, "the KG tool is missing from the MCP surface"

    supplied = {key.lower() for key in _properties(graph)}
    assert not (supplied & {"cypher", "query", "statement", "match", "where"})
    assert "intent" in supplied


def test_only_the_two_appointment_tools_can_change_anything(tools: dict) -> None:
    """The surface was entirely read-only until §15's action tools landed.

    The old rule — no tool name contains a mutating verb — is gone, because
    PRD §15 names ``book_my_appointment`` and ``cancel_my_appointment``. What
    replaces it is narrower and checkable: exactly those two, and nothing
    else, may imply a write.
    """
    mutating = ("book", "cancel", "create", "update", "delete", "schedule", "confirm")
    implied = {
        name for name in tools if any(verb in name.lower() for verb in mutating)
    }
    assert implied == server.PROPOSE_ONLY_TOOLS, (
        "a tool outside the propose-only set implies a write; if it is "
        "genuinely an action, it must go through propose/confirm too"
    )


def test_no_tool_can_confirm_an_action(tools: dict) -> None:
    """The load-bearing assertion of the whole action design.

    ``book_my_appointment`` proposes and returns a signed token; only
    ``POST /api/actions/confirm`` executes, and it is absent from this
    surface. A caller that could both propose and confirm would turn "can a
    prompt injection make the assistant book something?" back into a question
    about how persuadable the model is.

    Checked two ways, because a confirm capability could arrive either as its
    own tool or as a token parameter bolted onto an existing one.
    """
    assert not any("confirm" in name.lower() for name in tools)
    for name, tool in tools.items():
        supplied = {key.lower() for key in _properties(tool)}
        assert "token" not in supplied, f"{name} accepts a confirmation token"


def test_the_action_tools_take_no_identifier(tools: dict) -> None:
    """An appointment is named by its time, not by an id the caller invents.

    ``cancel_my_appointment(appointment_id=...)`` would be a parameter whose
    validation is the only thing standing between a caller and someone else's
    row. Identifying by time means the lookup is scoped to the patient before
    anything is matched.
    """
    for name in server.PROPOSE_ONLY_TOOLS:
        supplied = {key.lower() for key in _properties(tools[name])}
        assert not (supplied & {"appointment_id", "id", "target_id"}), name


def test_the_analytics_tool_accepts_no_sql(tools: dict) -> None:
    """§16: Text-to-SQL is a question interface, not a statement interface."""
    analytics = tools.get("run_my_patient_analytics")
    assert analytics is not None, "the analytics tool is missing"

    supplied = {key.lower() for key in _properties(analytics)}
    assert "question" in supplied
    assert not (supplied & {"sql", "statement", "query", "select"})


def test_tools_are_named_for_the_caller(tools: dict) -> None:
    """§16 prefers ``get_my_appointments`` over ``get_patient_appointments``."""
    for name in tools:
        assert "_my_" in name, name


def test_every_tool_has_a_description(tools: dict) -> None:
    """The description is what the model routes on; a blank one is a bug."""
    for name, tool in tools.items():
        assert tool.description and len(tool.description) > 20, name


def test_search_takes_a_query(tools: dict) -> None:
    properties = _properties(tools["search_my_clinical_notes"])
    assert "query" in properties


def test_the_server_holds_no_credentials_of_its_own() -> None:
    """A service token would be a second way into the data (§15).

    The module must not read a credential from the environment at import
    time; the only token it ever uses is the caller's, per request.
    """
    source = Path(server.__file__).read_text(encoding="utf-8")
    assert "MCP_SERVICE_TOKEN" not in source
    assert "DATABASE_URL" not in source, "the MCP server must not reach the database"


def test_the_server_has_no_database_access() -> None:
    """It proxies the API; it does not query. Nothing to bypass."""
    source = Path(server.__file__).read_text(encoding="utf-8")
    for forbidden in ("sqlalchemy", "psycopg", "AppSession", "app.models"):
        assert forbidden not in source, f"{forbidden} imported by the MCP server"


def test_a_call_without_a_token_is_refused() -> None:
    """Refused here, before the backend is ever contacted."""

    with pytest.raises(server.BackendError) as excinfo:
        server._bearer(_ctx({}))
    assert "bearer token" in str(excinfo.value).lower()


def test_a_non_bearer_authorization_is_refused() -> None:
    with pytest.raises(server.BackendError):
        server._bearer(_ctx({"authorization": "Basic dXNlcjpwYXNz"}))


def test_the_bearer_token_is_passed_through_unchanged() -> None:
    assert (
        server._bearer(_ctx({"Authorization": "Bearer abc.def.ghi"}))
        == "Bearer abc.def.ghi"
    )
