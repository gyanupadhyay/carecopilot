"""The knowledge graph: a derived, patient-scoped view of relationships.

Neo4j holds a *projection* of PostgreSQL, never business truth of its own
(PRD §33), and every traversal is a fixed template anchored on the
authenticated patient (§21). The layering:

    schema.py    what the projection may contain — labels, edges, constraints
    client.py    the driver, and read-only execution
    queries.py   the approved traversals, as reviewed Cypher templates
    service.py   binds $patient_id from the AuthContext and runs one

``scripts/build_kg.py`` is the only writer, and it rebuilds from PostgreSQL.
"""

from __future__ import annotations

from app.knowledge_graph.client import (
    GraphDisabled,
    GraphUnavailable,
    dispose_driver,
    healthy,
)
from app.knowledge_graph.queries import INTENT_DESCRIPTIONS, GraphIntent
from app.knowledge_graph.service import (
    GraphQueryError,
    GraphResult,
    parse_intent,
    query_patient_graph,
)

__all__ = [
    "INTENT_DESCRIPTIONS",
    "GraphDisabled",
    "GraphIntent",
    "GraphQueryError",
    "GraphResult",
    "GraphUnavailable",
    "dispose_driver",
    "healthy",
    "parse_intent",
    "query_patient_graph",
]
