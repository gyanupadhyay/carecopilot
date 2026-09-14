"""Record endpoints (PRD §9, §10).

The security assertion here is structural rather than behavioural: there is
no request these endpoints accept that names a patient. The older shape
(``/api/patients/{id}/labs``) was safe because every handler checked the id
against the session; this shape is safe because the id was never accepted.
The tests below pin that — a patient id supplied any way a client could
supply one must not change what comes back.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import embedding_provider
from app.auth.demo import DEMO_PASSWORD
from app.main import create_app
from app.rag.embeddings import build_embedder

pytestmark = pytest.mark.integration

BASE = "http://test/api"


@pytest.fixture(scope="module")
def embedder():
    return build_embedder(provider="local")


@pytest.fixture
async def client(session, embedder):  # session fixture forces the DB skip
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


@pytest.fixture
async def auth(client: AsyncClient) -> dict[str, str]:
    return await _login(client, "P001")


RECORD_PATHS = (
    "/me",
    "/appointments",
    "/appointments/next",
    "/medications",
    "/labs",
    "/encounters",
    "/encounters/latest",
)


@pytest.mark.parametrize("path", RECORD_PATHS)
async def test_every_record_endpoint_requires_a_token(
    client: AsyncClient, path: str
) -> None:
    assert (await client.get(f"{BASE}{path}")).status_code == 401


@pytest.mark.parametrize("path", RECORD_PATHS)
async def test_every_record_endpoint_rejects_a_forged_token(
    client: AsyncClient, path: str
) -> None:
    bad = {"Authorization": "Bearer not.a.real.token"}
    assert (await client.get(f"{BASE}{path}", headers=bad)).status_code == 401


async def test_me_returns_the_authenticated_patient(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.get(f"{BASE}/me", headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert body["external_id"] == "P001"
    assert body["full_name"]
    assert isinstance(body["age"], int)


async def test_record_collections_return_data(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    for path in ("/appointments", "/medications", "/labs", "/encounters"):
        response = await client.get(f"{BASE}{path}", headers=auth)
        assert response.status_code == 200, path
        assert response.json()["count"] > 0, path


async def test_next_appointment_is_scheduled_and_future(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    """Demo 1. A cancelled decoy sits earlier in the seeded data."""
    response = await client.get(f"{BASE}/appointments/next", headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert body is not None
    assert body["status"] == "scheduled"
    assert body["provider_name"] == "Dr. Sarah Smith"


async def test_latest_encounter_is_returned(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.get(f"{BASE}/encounters/latest", headers=auth)
    assert response.status_code == 200
    assert response.json()["encounter_date"]


# --- the shape itself is the control (PRD §9) ----------------------- #


async def test_a_supplied_patient_id_changes_nothing(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    """The classic IDOR attempt has nowhere to land.

    A query parameter the endpoint does not declare is ignored by FastAPI,
    so the response is byte-identical to the request without it.
    """
    plain = await client.get(f"{BASE}/labs?limit=5", headers=auth)
    tampered = await client.get(f"{BASE}/labs?limit=5&patient_id=2", headers=auth)

    assert plain.status_code == tampered.status_code == 200
    assert plain.json() == tampered.json()


async def test_the_old_patient_scoped_urls_are_gone(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    """§9 rejects the shape, so it should not still be routable."""
    for path in ("/patients/1", "/patients/1/labs", "/patients/2/labs"):
        assert (await client.get(f"{BASE}{path}", headers=auth)).status_code == 404


async def test_two_patients_see_their_own_records(client: AsyncClient) -> None:
    """The same URL, two sessions, disjoint data."""
    first = await _login(client, "P001")
    second = await _login(client, "P002")

    a = await client.get(f"{BASE}/me", headers=first)
    b = await client.get(f"{BASE}/me", headers=second)

    assert a.json()["external_id"] == "P001"
    assert b.json()["external_id"] == "P002"
    assert a.json()["id"] != b.json()["id"]

    labs_a = await client.get(f"{BASE}/labs?limit=200", headers=first)
    labs_b = await client.get(f"{BASE}/labs?limit=200", headers=second)
    ids_a = {row["id"] for row in labs_a.json()["items"]}
    ids_b = {row["id"] for row in labs_b.json()["items"]}
    assert ids_a and ids_b
    assert ids_a.isdisjoint(ids_b)


# --- clinical note search ----------------------------------------------- #


async def test_note_search_returns_cited_fragments(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.get(
        f"{BASE}/clinical-notes/search", params={"q": "knee pain"}, headers=auth
    )
    assert response.status_code == 200
    hits = response.json()["items"]
    assert hits, "expected note hits for the demo patient"

    top = hits[0]
    assert top["chunk_id"] and top["document_id"]
    assert top["section"]
    assert 0.0 <= top["score"] <= 1.0
    assert "knee" in " ".join(h["text"].lower() for h in hits)


async def test_note_search_can_be_restricted_to_a_section(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.get(
        f"{BASE}/clinical-notes/search",
        params={"q": "what was the plan", "section": "Plan"},
        headers=auth,
    )
    assert response.status_code == 200
    hits = response.json()["items"]
    assert hits
    assert {h["section"] for h in hits} == {"Plan"}


async def test_note_search_finds_nothing_for_an_unrelated_question(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.get(
        f"{BASE}/clinical-notes/search",
        params={"q": "what is my dog's name"},
        headers=auth,
    )
    assert response.status_code == 200
    assert response.json()["items"] == []


async def test_note_search_never_returns_another_patients_notes(
    client: AsyncClient,
) -> None:
    first = await _login(client, "P001")
    second = await _login(client, "P002")
    query = {"q": "what did the doctor say at my last visit"}

    a = await client.get(f"{BASE}/clinical-notes/search", params=query, headers=first)
    b = await client.get(f"{BASE}/clinical-notes/search", params=query, headers=second)

    ids_a = {h["chunk_id"] for h in a.json()["items"]}
    ids_b = {h["chunk_id"] for h in b.json()["items"]}
    assert ids_a and ids_b
    assert ids_a.isdisjoint(ids_b)


async def test_note_search_requires_a_query(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.get(f"{BASE}/clinical-notes/search", headers=auth)
    assert response.status_code == 422
