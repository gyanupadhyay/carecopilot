"""Running an approved traversal for the authenticated patient.

The single place ``$patient_id`` is bound, and it is bound from the
``AuthContext`` — never from an argument, a tool call, or anything the model
produced. :func:`query_patient_graph` takes an intent and a term; there is
deliberately no parameter through which a patient id could arrive, which is
what makes "the LLM cannot choose the patient scope" (PRD Principle 4) a
property of the signature rather than a rule someone has to remember.

Results come back as rows of plain values plus a short summary. The summary
is composed here, from values the database returned, rather than by the model
— PRD Principle 12: if the backend can state something reliably, it should.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.auth.context import AuthContext
from app.config import settings
from app.knowledge_graph import queries
from app.knowledge_graph.client import GraphUnavailable, run_read
from app.knowledge_graph.queries import GraphIntent
from app.observability.logging import get_logger

log = get_logger(__name__)


class GraphQueryError(Exception):
    """The request was malformed — an unknown intent, or a missing term."""


@dataclass(frozen=True, slots=True)
class GraphResult:
    intent: GraphIntent
    rows: list[dict[str, Any]] = field(default_factory=list)
    #: One sentence stating what was found, for the context the model reads.
    summary: str = ""
    term: str | None = None
    latency_ms: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.rows


def parse_intent(value: str) -> GraphIntent:
    """Resolve a model-supplied label, or raise.

    An unrecognised label is a malformed tool call, not a new capability.
    """
    try:
        return GraphIntent(value.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(sorted(i.value for i in GraphIntent))
        raise GraphQueryError(
            f"Unknown graph intent {value!r}. Allowed: {allowed}."
        ) from exc


async def query_patient_graph(
    ctx: AuthContext,
    *,
    intent: GraphIntent | str,
    term: str | None = None,
    limit: int | None = None,
) -> GraphResult:
    """Run one approved traversal, scoped to the caller's own patient.

    Raises :class:`GraphQueryError` for a malformed request and
    :class:`~app.knowledge_graph.client.GraphUnavailable` when the graph
    cannot be reached — the caller decides how to degrade.
    """
    import time

    resolved = intent if isinstance(intent, GraphIntent) else parse_intent(intent)

    cleaned = (term or "").strip()
    if resolved in queries.REQUIRES_TERM and not cleaned:
        raise GraphQueryError(
            f"The {resolved.value!r} traversal needs a search term "
            f"({queries.INTENT_DESCRIPTIONS[resolved]})"
        )

    # The one binding of patient scope. `patient_scope` raises when the
    # session is not linked to a patient, so an unscoped traversal cannot be
    # issued even by a bug here.
    patient_id = ctx.patient_scope
    # Clamped, not trusted: `limit` reaches this from a tool argument, and an
    # enormous one would turn a traversal into a context-window flood.
    rows_wanted = min(max(limit or settings.kg_max_rows, 1), settings.kg_max_rows)

    started = time.perf_counter()
    rows = await run_read(
        queries.CYPHER[resolved],
        {"patient_id": patient_id, "term": cleaned, "limit": rows_wanted},
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    log.info(
        "kg.queried",
        intent=resolved.value,
        rows=len(rows),
        ms=latency_ms,
        # The term is the patient's own phrasing and can be clinical, so its
        # length is recorded rather than its text (PRD §26).
        term_len=len(cleaned),
    )

    return GraphResult(
        intent=resolved,
        rows=[_clean_row(row) for row in rows],
        summary=_summarize(resolved, rows, cleaned),
        term=cleaned or None,
        latency_ms=latency_ms,
    )


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    """Drop null columns and stringify dates.

    Nulls come from the ``OPTIONAL MATCH`` clauses and carry no information;
    leaving them in spends context tokens on ``"prescriber": null`` and
    invites the model to remark on absent data as though it were a finding.
    """
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if value is None or value == [] or value == [None]:
            continue
        if isinstance(value, list):
            value = [item for item in value if item is not None]
            if not value:
                continue
        cleaned[key] = str(value) if hasattr(value, "isoformat") else value
    return cleaned


def _summarize(intent: GraphIntent, rows: list[dict[str, Any]], term: str) -> str:
    """State the finding in one sentence, from the returned values."""
    if not rows:
        subject = f" for {term!r}" if term else ""
        what = intent.value.replace("_", " ")
        return f"The relationship graph holds no {what}{subject}."

    count = len(rows)
    match intent:
        case GraphIntent.CONDITIONS:
            names = _distinct(rows, "condition")
            return f"{count} condition(s) on record: {', '.join(names)}."
        case GraphIntent.MEDICATIONS_FOR_CONDITION:
            names = _distinct(rows, "medication")
            conditions = _distinct(rows, "condition")
            return (
                f"{len(names)} medication(s) linked to "
                f"{', '.join(conditions) or term}: {', '.join(names)}."
            )
        case GraphIntent.WHY_MEDICATION:
            treats = _distinct(rows, "treats")
            drug = _distinct(rows, "medication")
            if treats:
                return f"{', '.join(drug)} is recorded as treating {', '.join(treats)}."
            return f"{', '.join(drug)} is on record, with no condition linked to it."
        case GraphIntent.CONDITION_TIMELINE:
            subject = ", ".join(_distinct(rows, "condition")) or term
            return f"{count} visit(s) recorded for {subject}."
        case GraphIntent.LABS_FOR_CONDITION:
            tests = _distinct(rows, "test")
            return f"{count} result(s) across {len(tests)} test(s): {', '.join(tests)}."
        case GraphIntent.CARE_TEAM:
            clinicians = ", ".join(_distinct(rows, "clinician"))
            return f"{count} clinician(s) seen: {clinicians}."
        case GraphIntent.MEDICATION_HISTORY:
            return f"{count} medication(s) on record."
        case GraphIntent.ALLERGIES:
            # Severity is named, not just counted: "3 allergies" and "3
            # allergies, one severe" are different facts, and the second is
            # the one a reader needs.
            substances = _distinct(rows, "substance")
            severe = [r for r in rows if str(r.get("severity")) == "severe"]
            tail = f", {len(severe)} severe" if severe else ""
            return f"{count} recorded allergy/allergies{tail}: {', '.join(substances)}."
        case GraphIntent.PROCEDURES:
            names = _distinct(rows, "procedure")
            return f"{count} procedure(s) on record: {', '.join(names)}."
        case GraphIntent.DIAGNOSIS_HISTORY:
            conditions = _distinct(rows, "condition")
            return f"{count} diagnosis/diagnoses on record: {', '.join(conditions)}."
    return f"{count} row(s) returned."


def _distinct(rows: list[dict[str, Any]], key: str) -> list[str]:
    """Distinct non-null values for a column, in first-seen order."""
    seen: dict[str, None] = {}
    for row in rows:
        value = row.get(key)
        if value:
            seen.setdefault(str(value), None)
    return list(seen)


__all__ = [
    "GraphIntent",
    "GraphQueryError",
    "GraphResult",
    "GraphUnavailable",
    "parse_intent",
    "query_patient_graph",
]
