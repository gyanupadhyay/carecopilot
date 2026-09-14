"""The projection must not accumulate ownership (PRD §21, §33).

``test_knowledge_graph.py`` proves no approved traversal can leave a
patient's subgraph, and ``test_graph_api.py`` proves the deployed path
agrees. Both were passing on the day the graph returned one patient's
prescription to another.

The traversals were not the problem. ``build_kg.py`` projects with ``MERGE``,
which adds an edge and never removes one, so a row that moved between
patients between seeds left the graph holding *both* ownership edges — and a
correct traversal anchored on the wrong patient then walked the stale one.
Isolation held; the data underneath it had drifted. 4,739 stale edges had
accumulated before anything noticed.

So this file asserts the property the query layer cannot: that every
per-patient clinical node has exactly one owner, and that it is the owner
PostgreSQL names. The first test is the invariant; the second proves the
repair actually repairs, by constructing the drift deliberately rather than
waiting for a reseed to produce it again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.config import settings
from app.knowledge_graph.client import GraphUnavailable, get_driver, healthy

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.build_kg import OWNED, reconcile

pytestmark = pytest.mark.integration

#: Ids well outside anything the generator produces, so the fixtures below
#: cannot collide with a real row and a failed cleanup cannot corrupt one.
FAKE_NODE_ID = -9001


@pytest.fixture
async def graph_available() -> None:
    try:
        if not await healthy():
            pytest.skip("Neo4j unreachable; run docker compose up -d neo4j")
    except GraphUnavailable:
        pytest.skip("Neo4j unreachable; run docker compose up -d neo4j")


async def _run(cypher: str, params: dict | None = None) -> list[dict]:
    driver = get_driver()
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(cypher, params or {})  # type: ignore[arg-type]
        return [record.data() async for record in result]


# --- the invariant -------------------------------------------------------- #


@pytest.mark.parametrize(("label", "relationship"), OWNED)
async def test_no_clinical_node_has_two_owners(
    graph_available, label: str, relationship: str
) -> None:
    """One prescription, one patient. Asserted over the live projection.

    This is the check that would have caught the leak on the day it appeared,
    and it needs no knowledge of which row moved: a second owning edge is
    provably wrong whatever produced it.
    """
    rows = await _run(
        f"""
        MATCH (p:Patient)-[:{relationship}]->(n:{label})
        WITH n, count(DISTINCT p) AS owners
        WHERE owners > 1
        RETURN n.id AS id, owners
        ORDER BY owners DESC
        LIMIT 5
        """
    )
    assert rows == [], (
        f"{label} nodes with more than one {relationship} owner: {rows}. "
        "Rebuild the projection: python scripts/build_kg.py"
    )


@pytest.mark.parametrize(("label", "relationship"), OWNED)
async def test_every_owner_is_the_one_the_source_names(
    graph_available, label: str, relationship: str
) -> None:
    """The owning edge must agree with the node's own ``patient_id``.

    Stronger than the count above and cheaper than re-reading PostgreSQL:
    ``patient_id`` is copied onto every clinical node straight from its
    source row, so it *is* what the database says, and an edge from any other
    patient contradicts it.
    """
    rows = await _run(
        f"""
        MATCH (p:Patient)-[:{relationship}]->(n:{label})
        WHERE n.patient_id IS NOT NULL AND p.id <> n.patient_id
        RETURN n.id AS id, p.id AS edge_owner, n.patient_id AS source_owner
        LIMIT 5
        """
    )
    assert rows == [], f"{label} edges contradicting the source row: {rows}"


# --- the repair ----------------------------------------------------------- #


async def test_reconcile_removes_an_edge_the_source_contradicts(
    graph_available,
) -> None:
    """Construct the drift, then prove the repair removes exactly it.

    Waiting for a reseed to reproduce this would make the test a coin flip;
    building the bad edge by hand makes it deterministic. The node uses a
    negative id so nothing here can touch a real prescription, and the
    ``finally`` removes it whether or not the assertions hold.
    """
    patients = await _run("MATCH (p:Patient) RETURN p.id AS id ORDER BY p.id LIMIT 2")
    if len(patients) < 2:
        pytest.skip("needs two seeded patients; run scripts/generate_data.py --reset")
    owner, interloper = patients[0]["id"], patients[1]["id"]

    try:
        # A medication that belongs to `owner`, wired to both — exactly the
        # shape a reseed produces when a row changes hands.
        await _run(
            """
            MATCH (a:Patient {id: $owner}), (b:Patient {id: $interloper})
            MERGE (m:Medication {id: $node})
            SET m.patient_id = $owner, m.name = 'drift-fixture'
            MERGE (a)-[:TAKES]->(m)
            MERGE (b)-[:TAKES]->(m)
            """,
            {"owner": owner, "interloper": interloper, "node": FAKE_NODE_ID},
        )
        before = await _run(
            "MATCH (p:Patient)-[:TAKES]->(m:Medication {id: $node}) "
            "RETURN count(p) AS owners",
            {"node": FAKE_NODE_ID},
        )
        assert before[0]["owners"] == 2, "fixture did not create the drift"

        await reconcile()

        after = await _run(
            "MATCH (p:Patient)-[:TAKES]->(m:Medication {id: $node}) "
            "RETURN p.id AS owner",
            {"node": FAKE_NODE_ID},
        )
        # The stale edge is gone and the legitimate one survives — a repair
        # that deleted both would pass a "no two owners" check while losing
        # the patient's own prescription.
        assert [row["owner"] for row in after] == [owner]
    finally:
        await _run(
            "MATCH (m:Medication {id: $node}) DETACH DELETE m",
            {"node": FAKE_NODE_ID},
        )
