"""The agent workflow (PRD §11).

    START
      → classify_query
      → route_query ──┬─ API          → execute_api_tool
                      ├─ RAG          → retrieve
                      ├─ KG           → query_graph
                      ├─ HYBRID       → hybrid
                      ├─ TEXT_TO_SQL  → text_to_sql
                      ├─ ACTION       → action
                      └─ OUT_OF_SCOPE → out_of_scope
      → generate_answer
      → validate_result
      → END

What LangGraph is doing here, and what it is not. It owns the state merge,
the conditional branch and the execution order — the things §11 lists. It
does not authenticate, authorize, resolve identity or touch the database
directly; every data path goes through a service that takes an
``AuthContext`` the graph merely transports (§40 P5). Deleting the graph
would change how the turn is orchestrated and not one thing about who can
read what.

The graph is compiled per request rather than once at import. Nodes are
closures over the request's session, provider and trace, and a compiled
graph shared across requests would close over the first request's session.
Compilation is in-process graph construction, not I/O, so the cost is
negligible against a model call.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import (
    NodeDeps,
    make_action_node,
    make_api_node,
    make_classify,
    make_generate_node,
    make_hybrid_node,
    make_kg_node,
    make_rag_node,
    make_text_to_sql_node,
    make_validate_node,
    out_of_scope,
    route_query,
)
from app.agents.state import AgentState

#: Route label → node name. The keys are exactly the router's enum, so a
#: route with no branch is a construction-time error rather than a request
#: that silently falls through.
ROUTE_TO_NODE: dict[str, str] = {
    "API": "execute_api_tool",
    "RAG": "retrieve",
    "KG": "query_graph",
    "HYBRID": "hybrid",
    "TEXT_TO_SQL": "text_to_sql",
    "ACTION": "action",
    "OUT_OF_SCOPE": "out_of_scope",
}


def build_graph(deps: NodeDeps):  # type: ignore[no-untyped-def]
    """Compile the workflow for one request."""
    graph = StateGraph(AgentState)

    graph.add_node("classify_query", make_classify(deps))
    graph.add_node("execute_api_tool", make_api_node(deps))
    graph.add_node("retrieve", make_rag_node(deps))
    graph.add_node("query_graph", make_kg_node(deps))
    graph.add_node("hybrid", make_hybrid_node(deps))
    graph.add_node("text_to_sql", make_text_to_sql_node(deps))
    graph.add_node("action", make_action_node(deps))
    graph.add_node("out_of_scope", out_of_scope)
    graph.add_node("generate_answer", make_generate_node(deps))
    graph.add_node("validate_result", make_validate_node(deps))

    graph.add_edge(START, "classify_query")
    graph.add_conditional_edges("classify_query", route_query, ROUTE_TO_NODE)

    # Every branch converges on generation, then validation. The out-of-scope
    # and action nodes set an answer first, and generate_answer passes it
    # through untouched — so validation runs over every answer the system
    # emits, whichever branch produced it.
    for node in (
        "execute_api_tool",
        "retrieve",
        "query_graph",
        "hybrid",
        "text_to_sql",
        "action",
        "out_of_scope",
    ):
        graph.add_edge(node, "generate_answer")

    graph.add_edge("generate_answer", "validate_result")
    graph.add_edge("validate_result", END)

    return graph.compile()


def graph_shape() -> dict[str, list[str]]:
    """The static edge map, for documentation and tests.

    Kept beside the builder so a branch added to one and not the other is a
    visible inconsistency rather than an untested path.
    """
    return {
        "START": ["classify_query"],
        "classify_query": sorted(set(ROUTE_TO_NODE.values())),
        "execute_api_tool": ["generate_answer"],
        "retrieve": ["generate_answer"],
        "query_graph": ["generate_answer"],
        "hybrid": ["generate_answer"],
        "text_to_sql": ["generate_answer"],
        "action": ["generate_answer"],
        "out_of_scope": ["generate_answer"],
        "generate_answer": ["validate_result"],
        "validate_result": ["END"],
    }
