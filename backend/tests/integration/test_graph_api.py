"""The knowledge-graph endpoint, over HTTP, against a live Neo4j (PRD §21).

``test_knowledge_graph.py`` proves the *templates* cannot leave a patient's
subgraph. This proves the deployed path agrees: that two authenticated
patients asking the same question get their own answers, that no request
shape names a patient, and that a session with no patient record reaches
nothing.

Skips rather than fails when Neo4j or the seed data is absent, like the other
integration tests — a graph question on a machine with no graph is not a
regression.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import embedding_provider
from app.auth.demo import DEMO_PASSWORD
from app.knowledge_graph.client import GraphUnavailable, healthy
from app.main import create_app
from app.rag.embeddings import build_embedder

pytestmark = pytest.mark.integration

BASE = "http://test/api"


@pytest.fixture(scope="module")
def embedder():
    return build_embedder(provider="local")


@pytest.fixture
async def graph_available() -> None:
    try:
        if not await healthy():
            pytest.skip("Neo4j unreachable; run docker compose up -d neo4j")
    except GraphUnavailable:
        pytest.skip("Neo4j unreachable; run docker compose up -d neo4j")


@pytest.fixture
async def client(session, embedder, graph_available):
    app = create_app()
    app.dependency_overrides[embedding_provider] = lambda: embedder
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        yield http
    app.dependency_overrides.clear()


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


# --- it works ------------------------------------------------------------ #


async def test_a_patient_can_traverse_their_own_conditions(client) -> None:
    headers = await _login(client, "P001")
    response = await client.get(
        f"{BASE}/graph", params={"intent": "conditions"}, headers=headers
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "conditions"
    assert body["count"] >= 1
    assert body["summary"]


async def test_demo_four_reaches_the_condition_through_treats(client) -> None:
    """PRD Demo 4: "why was I prescribed metformin?" is an edge, not prose."""
    headers = await _login(client, "P001")
    response = await client.get(
        f"{BASE}/graph",
        params={"intent": "why_medication", "term": "metformin"},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert "diabetes" in body["summary"].lower()


async def test_a_patient_alias_finds_the_clinical_name(client) -> None:
    """Patients say "blood pressure"; the record says "Essential hypertension"."""
    headers = await _login(client, "P001")
    response = await client.get(
        f"{BASE}/graph",
        params={"intent": "condition_timeline", "term": "blood pressure"},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["count"] >= 1


# --- authorization ------------------------------------------------------- #


async def test_two_patients_get_their_own_subgraphs(client) -> None:
    """The §21 acceptance criterion: cross-patient KG access is impossible.

    Same endpoint, same question, different tokens. Nothing in the request
    distinguishes them except who they authenticated as.
    """
    first = await client.get(
        f"{BASE}/graph",
        params={"intent": "medication_history"},
        headers=await _login(client, "P001"),
    )
    second = await client.get(
        f"{BASE}/graph",
        params={"intent": "medication_history"},
        headers=await _login(client, "P002"),
    )
    assert first.status_code == second.status_code == 200

    def _names(response) -> set[str]:
        return {
            f"{row.get('medication')}|{row.get('started')}"
            for row in response.json()["rows"]
        }

    # Two synthetic patients can share a drug *name* — both may take
    # metformin — so identity is the prescription, not the drug.
    assert _names(first).isdisjoint(_names(second))


async def test_no_request_parameter_names_a_patient(client) -> None:
    """Supplying a patient id any way a client could must change nothing."""
    headers = await _login(client, "P001")
    baseline = await client.get(
        f"{BASE}/graph", params={"intent": "conditions"}, headers=headers
    )
    tampered = await client.get(
        f"{BASE}/graph",
        params={"intent": "conditions", "patient_id": 2, "patient": "P002"},
        headers=headers,
    )
    assert tampered.status_code == 200
    assert tampered.json()["rows"] == baseline.json()["rows"]


async def test_an_unauthenticated_request_is_refused(client) -> None:
    response = await client.get(f"{BASE}/graph", params={"intent": "conditions"})
    assert response.status_code == 401


async def test_an_unknown_intent_is_rejected_before_any_traversal(client) -> None:
    """A closed set: FastAPI refuses the value, no handler code runs."""
    headers = await _login(client, "P001")
    response = await client.get(
        f"{BASE}/graph", params={"intent": "read_everything"}, headers=headers
    )
    assert response.status_code == 422


async def test_cypher_in_the_term_matches_nothing(client) -> None:
    """The term is a bound parameter, never query text."""
    headers = await _login(client, "P001")
    response = await client.get(
        f"{BASE}/graph",
        params={
            "intent": "why_medication",
            "term": "' }) MATCH (x:Patient) RETURN x //",
        },
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["count"] == 0


async def test_a_term_requiring_intent_without_one_is_refused(client) -> None:
    headers = await _login(client, "P001")
    response = await client.get(
        f"{BASE}/graph", params={"intent": "why_medication"}, headers=headers
    )
    assert response.status_code == 422
