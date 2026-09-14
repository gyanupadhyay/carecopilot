"""CareCopilot MCP server (PRD §15).

    python mcp-server/server.py

Exposes the approved patient-record tools over the Model Context Protocol so
that an external MCP client — Claude Desktop, an IDE, another agent — can
read the same records the chat UI reads, under the same rules.

The architectural point, and the reason this file is short:

    MCP client → MCP server → backend HTTP API → authorization → database

**This server has no database credentials and no database code.** It cannot
reach a record except by calling the backend API with the caller's own JWT,
which means it inherits authorization rather than re-implementing it. There
is no code path here that could bypass a check, because there is no path
here that reaches data at all. §15 says "the MCP server must not bypass
backend authorization"; the way to guarantee that is to give it nothing to
bypass with.

Two consequences follow:

*No tool takes a patient identifier.* Every tool is ``..._my_...`` and the
patient comes from the token, exactly as in §16. A client that wants another
patient's records has nowhere to say so.

*The caller's token is required per request.* The server holds no
credentials of its own — no service account, no shared secret that would
let it read on someone's behalf. An unauthenticated call fails here rather
than reaching the backend.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import httpx
from mcp.server.mcpserver import Context, MCPServer

BACKEND_URL = os.environ.get("BACKEND_INTERNAL_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("MCP_BACKEND_TIMEOUT", "30"))
#: Longer, because analytics and action proposals both make a model call on
#: the backend, and a locally served Qwen3 on CPU is measured in tens of
#: seconds. A timeout shorter than the work turns a slow answer into a
#: retried one, which is how a proposal gets made twice.
ACTION_TIMEOUT = float(os.environ.get("MCP_ACTION_TIMEOUT", "180"))

mcp = MCPServer(
    name="carecopilot",
    version="0.1.0",
    instructions=(
        "Read-only access to the authenticated patient's own synthetic medical "
        "record: appointments, medications, lab results, encounters and "
        "clinical notes. DEMO — synthetic patient data. Not for medical "
        "diagnosis or treatment. No tool accepts a patient identifier; the "
        "patient is resolved from the caller's bearer token."
    ),
)


class BackendError(RuntimeError):
    """The backend refused or could not serve the request."""


def _bearer(ctx: Context) -> str:
    """The caller's token, or a refusal.

    Taken from the live request rather than from configuration. A server
    that held its own long-lived credential would be a second way into the
    data, which is the thing §15 rules out.
    """
    headers = ctx.headers or {}
    # Header names are case-insensitive over the wire; the mapping may not be.
    raw = headers.get("authorization") or headers.get("Authorization") or ""
    if not raw.lower().startswith("bearer "):
        raise BackendError(
            "No bearer token was supplied. Connect with an Authorization "
            "header carrying the patient's CareCopilot JWT."
        )
    return raw


async def _get(ctx: Context, path: str, **params: Any) -> Any:
    """Call the backend as the caller, and translate failures honestly."""
    token = _bearer(ctx)
    query = {k: v for k, v in params.items() if v is not None}

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(
                f"{BACKEND_URL}/api{path}",
                headers={"Authorization": token},
                params=query,
            )
    except httpx.TransportError as exc:
        raise BackendError("The CareCopilot backend is unreachable.") from exc

    return _translated(response)


async def _post(ctx: Context, path: str, payload: dict[str, Any]) -> Any:
    """POST as the caller. Same rules, and the same absence of a patient id.

    Separate from ``_get`` only because the body goes in the body. Every
    guarantee is identical: the caller's own token, no credential of this
    server's own, and no parameter naming a patient.
    """
    token = _bearer(ctx)

    try:
        async with httpx.AsyncClient(timeout=ACTION_TIMEOUT) as client:
            response = await client.post(
                f"{BACKEND_URL}/api{path}",
                headers={"Authorization": token},
                json=payload,
            )
    except httpx.TransportError as exc:
        raise BackendError("The CareCopilot backend is unreachable.") from exc

    return _translated(response)


def _translated(response: httpx.Response) -> Any:
    """Map a backend status onto an answer or a refusal."""
    if response.status_code == 401:
        raise BackendError("The token is invalid or has expired.")
    if response.status_code == 403:
        # Surfaced verbatim rather than softened: a refusal is the answer.
        raise BackendError("Not authorized to access these records.")
    if response.status_code == 422:
        # The backend understood and declined for a reason the caller can
        # act on — "that time is already booked", "the schema does not cover
        # that question". Passing the reason through is the whole value of
        # the status; replacing it with "422" makes the tool useless.
        raise BackendError(_detail(response) or "The request was refused.")
    if response.status_code >= 400:
        raise BackendError(f"The backend refused the request ({response.status_code}).")

    return response.json()


def _detail(response: httpx.Response) -> str:
    """FastAPI's ``detail``, when there is one and it is a string."""
    try:
        body = response.json()
    except ValueError:
        return ""
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail if isinstance(detail, str) else ""


# ---------------------------------------------------------------------- #
# Tools — every one scoped to the caller, none taking a patient id
# ---------------------------------------------------------------------- #


@mcp.tool(
    name="get_my_profile",
    description="The authenticated patient's own name, age and record number.",
)
async def get_my_profile(ctx: Context) -> dict[str, Any]:
    return await _get(ctx, "/me")


@mcp.tool(
    name="get_my_next_appointment",
    description=(
        "The soonest upcoming scheduled appointment, with date and clinician. "
        "Returns null when nothing is booked."
    ),
)
async def get_my_next_appointment(ctx: Context) -> Any:
    return await _get(ctx, "/appointments/next")


@mcp.tool(
    name="get_my_appointments",
    description="Appointment history, most recent first, including past visits.",
)
async def get_my_appointments(ctx: Context, limit: int = 20) -> dict[str, Any]:
    return await _get(ctx, "/appointments", limit=min(max(limit, 1), 200))


@mcp.tool(
    name="get_my_medications",
    description="Medications currently in effect, with dose and frequency.",
)
async def get_my_medications(ctx: Context) -> dict[str, Any]:
    return await _get(ctx, "/medications")


@mcp.tool(
    name="get_my_lab_results",
    description=(
        "Laboratory and vital-sign results. Optionally filter to one exact "
        "test name, such as 'HbA1c' or 'Systolic Blood Pressure'."
    ),
)
async def get_my_lab_results(
    ctx: Context, test_name: str | None = None, limit: int = 50
) -> dict[str, Any]:
    return await _get(
        ctx, "/labs", test_name=test_name, limit=min(max(limit, 1), 200)
    )


@mcp.tool(
    name="get_my_encounters",
    description="History of clinical visits, most recent first.",
)
async def get_my_encounters(ctx: Context, limit: int = 20) -> dict[str, Any]:
    return await _get(ctx, "/encounters", limit=min(max(limit, 1), 200))


@mcp.tool(
    name="get_my_last_encounter",
    description="The most recent clinical visit: date, clinician and reason.",
)
async def get_my_last_encounter(ctx: Context) -> Any:
    return await _get(ctx, "/encounters/latest")


@mcp.tool(
    name="search_my_clinical_notes",
    description=(
        "Search the patient's own clinical notes and return the passages that "
        "match, each with its document, section, date and relevance score. "
        "Use this for what a clinician wrote or recommended."
    ),
)
async def search_my_clinical_notes(
    ctx: Context, query: str, limit: int = 8, section: str | None = None
) -> dict[str, Any]:
    return await _get(
        ctx,
        "/clinical-notes/search",
        q=query,
        limit=min(max(limit, 1), 20),
        section=section,
    )


@mcp.tool(
    name="query_my_patient_graph",
    description=(
        "Traverse the relationships in the patient's own record: which "
        "medication treats which condition, why a drug was started, which "
        "visits belonged to a condition, who has treated it.\n\n"
        "`intent` selects one of a fixed set of approved traversals:\n"
        "  conditions                 every condition on record, with onset\n"
        "  medications_for_condition  what treats a named condition (needs term)\n"
        "  why_medication             what a named drug treats, and the visit "
        "and clinician that started it (needs term)\n"
        "  condition_timeline         the visits for a named condition (needs term)\n"
        "  labs_for_condition         labs ordered for a named condition (needs term)\n"
        "  care_team                  clinicians seen, their department, and for what\n"
        "  medication_history         every medication with the condition it treats\n"
        "  allergies                  substances the patient reacts to, and how badly\n"
        "  procedures                 procedures had, with date, clinician and reason\n"
        "  diagnosis_history          when each condition was diagnosed, and "
        "by whom\n\n"
        "`term` is a single condition or medication word, e.g. \"diabetes\". "
        "It is matched as a literal string, never interpreted as a query. "
        "Use this for how records RELATE; use search_my_clinical_notes for "
        "what a clinician wrote."
    ),
)
async def query_my_patient_graph(
    ctx: Context, intent: str, term: str = "", limit: int = 50
) -> dict[str, Any]:
    return await _get(
        ctx,
        "/graph",
        intent=intent,
        term=term,
        limit=min(max(limit, 1), 100),
    )


@mcp.tool(
    name="run_my_patient_analytics",
    description=(
        "Answer a counting, averaging or threshold question about the "
        "patient's own records — \"how many times was my blood pressure over "
        '140?", "what was my average HbA1c last year?", "how many '
        'appointments did I have in 2025?".\n\n'
        "Takes a QUESTION IN ENGLISH, never SQL. The statement is generated, "
        "parsed, validated against an allowlisted schema, capped and run on a "
        "read-only connection whose role is scoped to this patient by "
        "row-level security. Sending SQL here does not execute it — it is "
        "read as a question and declined.\n\n"
        "Prefer the get_my_* tools when one of them already answers the "
        "question: they are exact lookups, and this is for the aggregates "
        "they cannot express."
    ),
)
async def run_my_patient_analytics(ctx: Context, question: str) -> dict[str, Any]:
    return await _post(ctx, "/analytics", {"question": question})


@mcp.tool(
    name="book_my_appointment",
    description=(
        "PROPOSE booking an appointment for the patient. This does NOT book "
        "it.\n\n"
        "The parameters are validated against real availability and a signed, "
        "short-lived confirmation token is returned with a plain-English "
        "summary of what was proposed. Nothing is written to the record. The "
        "patient must then confirm it themselves in CareCopilot — there is "
        "deliberately no tool here that completes the booking, because a "
        "caller able to both propose and confirm could be talked into "
        "booking by text it read somewhere.\n\n"
        "Show the patient the returned summary and tell them to confirm it."
    ),
)
async def book_my_appointment(
    ctx: Context,
    when: str,
    appointment_type: str = "follow_up",
    reason: str = "",
) -> dict[str, Any]:
    return await _post(
        ctx,
        "/actions/propose",
        {
            "action": "book_appointment",
            "when": when,
            "appointment_type": appointment_type,
            "reason": reason,
        },
    )


@mcp.tool(
    name="cancel_my_appointment",
    description=(
        "PROPOSE cancelling one of the patient's upcoming appointments. This "
        "does NOT cancel it.\n\n"
        "`when` identifies which appointment, as an ISO 8601 date-time. The "
        "appointment is checked to exist, to belong to the patient and to be "
        "cancellable, and a signed confirmation token is returned with a "
        "summary. Nothing is written. As with booking, the patient confirms "
        "in CareCopilot; no tool here completes it."
    ),
)
async def cancel_my_appointment(
    ctx: Context, when: str, reason: str = ""
) -> dict[str, Any]:
    return await _post(
        ctx,
        "/actions/propose",
        {"action": "cancel_appointment", "when": when, "reason": reason},
    )


#: The approved set (PRD §15: "MCP exposes only approved tools").
#:
#: Three of these need their boundary stated, because their names suggest
#: more power than they have.
#:
#: ``query_my_patient_graph`` takes no patient id and no Cypher — only an
#: intent from a closed set and a search word. The traversal it names is
#: written in ``app/knowledge_graph/queries.py``, and ``$patient_id`` is bound
#: by the backend from the caller's token (§21).
#:
#: ``run_my_patient_analytics`` takes no SQL. It takes a question; the
#: backend generates, parses, validates and caps the statement and runs it as
#: a role holding SELECT on four tables under row-level security. There is
#: still no ``execute_sql`` here and there never will be.
#:
#: ``book_my_appointment`` and ``cancel_my_appointment`` **propose**. They
#: validate against real data and return a signed token; they write nothing.
#: The endpoint that executes a proposal, ``POST /api/actions/confirm``, is
#: deliberately absent from this surface — a caller that could both propose
#: and confirm would collapse the two-step design into one, and "can a prompt
#: injection make the assistant book something?" would stop having a
#: structural answer. Confirmation is the patient's own action in
#: CareCopilot.
APPROVED_TOOLS: tuple[str, ...] = (
    "get_my_profile",
    "get_my_next_appointment",
    "get_my_appointments",
    "get_my_medications",
    "get_my_lab_results",
    "get_my_encounters",
    "get_my_last_encounter",
    "search_my_clinical_notes",
    "query_my_patient_graph",
    "run_my_patient_analytics",
    "book_my_appointment",
    "cancel_my_appointment",
)

#: Tools that can change the record — and which therefore may only propose.
#: Named so the surface test can assert the property rather than infer it
#: from a verb list that a new tool would quietly fall outside of.
PROPOSE_ONLY_TOOLS: frozenset[str] = frozenset(
    {"book_my_appointment", "cancel_my_appointment"}
)


def main() -> int:
    # streamable-http rather than stdio: the tools need the caller's
    # Authorization header, and stdio has no request to carry one.
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("MCP_PORT", "8100"))

    print(
        f"CareCopilot MCP server → backend {BACKEND_URL}\n"
        f"  transport : {transport}\n"
        f"  tools     : {len(APPROVED_TOOLS)} (read-only, patient-scoped)\n"
        f"  auth      : caller's bearer token, per request",
        file=sys.stderr,
    )

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # Host and port are run() keyword arguments in the 2.x SDK; the 1.x
        # `mcp.settings.host` attribute no longer exists.
        mcp.run(transport=transport, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
