"""The approved tool set (PRD §15).

Each tool wraps a service function, returns validated Pydantic models, and
carries a one-line summary written for the model to read. The summary is
what makes a tool result usable without the model re-deriving it: given
``"Next appointment: 20 September 2026 at 10:00 with Dr. Sarah Smith"`` it
has nothing left to compute, and nothing left to get wrong.

Note the absence of a ``get_appointments(patient_id)``. There is no overload,
no optional argument, no admin variant — §15 asks for exactly one shape and
this module has exactly one shape.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.context import AuthContext
from app.llm.base import ToolDefinition
from app.schemas.clinical import (
    AppointmentOut,
    EncounterOut,
    LabResultOut,
    MedicationChange,
    MedicationOut,
    PatientOut,
)
from app.services import clinical
from app.tools.base import ToolResult, ToolSpec


def _day(value: date) -> str:
    """"09 September 2026" → "9 September 2026", portably.

    ``%-d`` is a glibc extension and raises on Windows, so the leading zero
    is stripped after formatting rather than avoided during it.
    """
    return value.strftime("%d %B %Y").lstrip("0")


def _humanize(moment: datetime) -> str:
    return f"{_day(moment.date())} at {moment.strftime('%H:%M')}"


#: How many rows a list summary names before it stops.
#:
#: The summary goes into the prompt, so this is a token budget as much as a
#: readability one. Ten covers every seeded patient's full history; beyond
#: that the tail is truncated and *said* to be truncated, because a list
#: that silently stops is one the model will describe as complete.
_SUMMARY_LIMIT = 10


def _listed(parts: Iterable[str]) -> str:
    """Join row descriptions, naming the overflow rather than hiding it."""
    rendered = list(parts)
    shown = rendered[:_SUMMARY_LIMIT]
    tail = len(rendered) - len(shown)
    return "; ".join(shown) + (f"; and {tail} older" if tail > 0 else "")


def _encounter_line(item: EncounterOut) -> str:
    who = f" with {item.provider_name}" if item.provider_name else ""
    why = f" for {item.reason}" if item.reason else ""
    return f"{_day(item.encounter_date)} {item.encounter_type}{why}{who}"


def _appointment_line(item: AppointmentOut) -> str:
    who = f" with {item.provider_name}" if item.provider_name else ""
    return (
        f"{_humanize(item.appointment_date)} {item.appointment_type}"
        f"{who} ({item.status})"
    )


# ---------------------------------------------------------------------- #
# Parameter schemas — none of them carry a patient identifier
# ---------------------------------------------------------------------- #


class LabQuery(BaseModel):
    test_name: str | None = Field(
        default=None, description="Exact test name, e.g. 'HbA1c'. Omit for all tests."
    )
    months_back: int | None = Field(
        default=None, ge=1, le=120, description="Only results from the last N months."
    )


class HistoryWindow(BaseModel):
    limit: int = Field(default=10, ge=1, le=50)


# ---------------------------------------------------------------------- #
# Tools
# ---------------------------------------------------------------------- #


async def get_my_profile(session: AsyncSession, ctx: AuthContext) -> ToolResult:
    patient = await clinical.get_patient_profile(session, ctx)
    if patient is None:
        return ToolResult("get_my_profile", None, "No patient record found.", count=0)
    out = PatientOut.model_validate(patient)
    return ToolResult(
        "get_my_profile",
        out,
        f"{out.full_name}, age {out.age}, record {out.external_id}.",
        count=1,
    )


async def get_my_next_appointment(
    session: AsyncSession, ctx: AuthContext
) -> ToolResult:
    appointment = await clinical.get_next_appointment(session, ctx)
    if appointment is None:
        return ToolResult(
            "get_my_next_appointment",
            None,
            "No upcoming appointment is scheduled.",
            count=0,
        )
    out = AppointmentOut.model_validate(appointment)
    with_whom = f" with {out.provider_name}" if out.provider_name else ""
    return ToolResult(
        "get_my_next_appointment",
        out,
        f"Next appointment: {_humanize(out.appointment_date)}{with_whom}.",
        count=1,
    )


async def get_my_appointments(
    session: AsyncSession, ctx: AuthContext, *, limit: int = 10
) -> ToolResult:
    rows = await clinical.get_appointments(session, ctx, limit=limit)
    items = [AppointmentOut.model_validate(r) for r in rows]
    if not items:
        return ToolResult(
            "get_my_appointments", [], "No appointments are on record.", count=0
        )
    upcoming = sum(1 for i in items if i.appointment_date > datetime.now(UTC))
    # See get_my_encounters below for why the slots are named rather than
    # counted. Status is included because a cancelled slot and an attended
    # one are the same row here, and an answer that conflates them is wrong
    # in the way a patient would notice.
    return ToolResult(
        "get_my_appointments",
        items,
        f"{len(items)} appointments on record, {upcoming} of them upcoming: "
        f"{_listed(_appointment_line(i) for i in items)}.",
        count=len(items),
    )


async def get_my_medications(session: AsyncSession, ctx: AuthContext) -> ToolResult:
    rows = await clinical.get_current_medications(session, ctx)
    items = [MedicationOut.model_validate(r) for r in rows]
    if not items:
        return ToolResult(
            "get_my_medications", [], "No active medications are recorded.", count=0
        )
    listed = "; ".join(f"{m.name} {m.dosage} {m.frequency}" for m in items)
    return ToolResult(
        "get_my_medications",
        items,
        f"{len(items)} active medications: {listed}.",
        count=len(items),
    )


async def get_my_lab_results(
    session: AsyncSession,
    ctx: AuthContext,
    *,
    test_name: str | None = None,
    months_back: int | None = None,
) -> ToolResult:
    from_date = (
        date.today() - timedelta(days=30 * months_back) if months_back else None
    )
    rows = await clinical.get_lab_results(
        session, ctx, test_name=test_name, from_date=from_date, limit=100
    )
    items = [LabResultOut.model_validate(r) for r in rows]
    if not items:
        which = f" for {test_name}" if test_name else ""
        return ToolResult(
            "get_my_lab_results", [], f"No lab results found{which}.", count=0
        )

    newest = items[0]
    summary = (
        f"{len(items)} results"
        + (f" for {test_name}" if test_name else "")
        + f"; most recent {newest.test_name} {newest.value} {newest.unit} "
        f"on {_day(newest.result_date)}."
    )
    return ToolResult("get_my_lab_results", items, summary, count=len(items))


async def get_my_encounters(
    session: AsyncSession, ctx: AuthContext, *, limit: int = 10
) -> ToolResult:
    rows = await clinical.get_encounters(session, ctx, limit=limit)
    items = [EncounterOut.model_validate(r) for r in rows]
    if not items:
        return ToolResult("get_my_encounters", [], "No encounters are recorded.", count=0)
    # The summary carries the visits themselves, not just how many there
    # were. `_tool_facts` forwards only this string to the model — `items`
    # never reaches it — so a count-only summary meant that "What visits
    # have I had?" could be answered with nothing but "you have had 5",
    # while the reason for each sat unread in the rows. Every other tool
    # here already names its content; these two were the exceptions.
    return ToolResult(
        "get_my_encounters",
        items,
        f"{len(items)} encounters on record: "
        f"{_listed(_encounter_line(i) for i in items)}.",
        count=len(items),
    )


async def get_my_last_encounter(session: AsyncSession, ctx: AuthContext) -> ToolResult:
    encounter = await clinical.get_last_encounter(session, ctx)
    if encounter is None:
        return ToolResult(
            "get_my_last_encounter", None, "No encounters are recorded.", count=0
        )
    out = EncounterOut.model_validate(encounter)
    with_whom = f" with {out.provider_name}" if out.provider_name else ""
    reason = f" Reason: {out.reason}" if out.reason else ""
    return ToolResult(
        "get_my_last_encounter",
        out,
        f"Last visit: {_day(out.encounter_date)}{with_whom}.{reason}",
        count=1,
    )


async def get_my_medication_changes(
    session: AsyncSession, ctx: AuthContext
) -> ToolResult:
    """What changed at the most recent visit — computed, not inferred.

    This is PRD §17 and §40 P8 in one function: comparing two medication
    lists is exact arithmetic over dates and dosages, so the backend does it
    and hands the model a finished answer. Asking the model to diff the
    lists is how a medication change that never happened gets reported.
    """
    encounter = await clinical.get_last_encounter(session, ctx)
    if encounter is None:
        return ToolResult(
            "get_my_medication_changes", [], "No encounters are recorded.", count=0
        )

    when = encounter.encounter_date
    before = await clinical.get_medications_in_effect(
        session, ctx, on=when - timedelta(days=1)
    )
    after = await clinical.get_medications_in_effect(session, ctx, on=when)
    changes: list[MedicationChange] = clinical.compare_medications(before, after)

    if not changes:
        return ToolResult(
            "get_my_medication_changes",
            [],
            f"No medication changes were recorded at the {_day(when)} visit.",
            count=0,
        )

    described = "; ".join(_describe_change(c) for c in changes)
    return ToolResult(
        "get_my_medication_changes",
        changes,
        f"At the {_day(when)} visit: {described}.",
        count=len(changes),
    )


def _describe_change(change: MedicationChange) -> str:
    if change.change == "started":
        return f"{change.name} started ({change.after})"
    if change.change == "stopped":
        return f"{change.name} discontinued (was {change.before})"
    return f"{change.name} changed from {change.before} to {change.after}"


# ---------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------- #

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_my_profile",
        description="The patient's own name, age, date of birth and record number.",
        fn=get_my_profile,
        tags=("profile",),
    ),
    ToolSpec(
        name="get_my_next_appointment",
        description=(
            "The soonest upcoming scheduled appointment, with date and clinician."
        ),
        fn=get_my_next_appointment,
        tags=("appointments",),
    ),
    ToolSpec(
        name="get_my_appointments",
        # "including past visits" used to end this line, and it was the whole
        # of the confusion with get_my_encounters below: the two tools were
        # separated by the word "visit" while this description claimed it.
        # Asked "What visits have I had at the clinic?", the model picked
        # this tool — correctly, by the catalogue as written — and answered
        # with a count of booked slots instead of what happened at them.
        # The axis is scheduling versus clinical, so both descriptions now
        # say which side they are on.
        description=(
            "Scheduled appointment slots and their status — booked, "
            "cancelled, completed. A scheduling record: when the patient was "
            "due in, not what happened once they arrived."
        ),
        fn=get_my_appointments,
        params=HistoryWindow,
        tags=("appointments",),
    ),
    ToolSpec(
        name="get_my_medications",
        description=(
            "Medications the patient is currently taking, with dose and frequency."
        ),
        fn=get_my_medications,
        tags=("medications",),
    ),
    ToolSpec(
        name="get_my_lab_results",
        description=(
            "Laboratory and vital-sign results. Optionally filtered to one test "
            "name such as HbA1c or Systolic Blood Pressure, and to a recent window."
        ),
        fn=get_my_lab_results,
        params=LabQuery,
        tags=("labs",),
    ),
    ToolSpec(
        name="get_my_encounters",
        # "most recent first" is kept, and "all" is the word that separates
        # this from get_my_last_encounter directly below — which answers the
        # same question about one visit.
        description=(
            "All clinical visits that took place, most recent first: date, "
            "clinician and the reason for each. The record of attended "
            "care, as opposed to the appointment slots that scheduled it."
        ),
        fn=get_my_encounters,
        params=HistoryWindow,
        tags=("encounters",),
    ),
    ToolSpec(
        name="get_my_last_encounter",
        description="The most recent clinical visit: date, clinician and reason.",
        fn=get_my_last_encounter,
        tags=("encounters",),
    ),
    ToolSpec(
        name="get_my_medication_changes",
        description=(
            "Which medications were started, stopped or changed at the most recent "
            "visit. Computed deterministically by the backend."
        ),
        fn=get_my_medication_changes,
        tags=("medications", "hybrid"),
    ),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in TOOLS}


def tool_catalogue() -> str:
    """The tool list as the router sees it."""
    return "\n".join(f"- {spec.name}: {spec.description}" for spec in TOOLS)


def tool_definitions() -> tuple[ToolDefinition, ...]:
    """The same tools, described for native tool-calling (PRD §4, §15).

    The agent's API route does not use this: it selects tools through an
    enum-constrained ``ToolPlan`` instead, because a small model asked to
    emit free-form function calls invents names, and an enum makes that
    unrepresentable rather than merely detectable. This exists so the
    capability PRD §4 names is real and exercised — an MCP client or a
    larger model can drive the same eight lookups natively — and so the
    choice above stays a choice rather than a limitation.
    """
    return tuple(spec.as_definition() for spec in TOOLS)
