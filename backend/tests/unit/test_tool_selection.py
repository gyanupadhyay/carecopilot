"""Model-driven tool selection (PRD §5, §27).

§5 makes tool selection Qwen3's responsibility. Until this landed the API
route ran a fixed three-tool plan for every question, which meant §27's
tool-selection accuracy reported 1.000 on every run by construction — a
number that cannot fall is not a measurement.

Two things this file is careful about.

*Selection is narrow, and that is what makes delegating it safe.* The model
picks which approved, read-only, self-scoped lookups to run. It cannot pick
whose record to read, because no tool takes a patient identifier. A wrong
selection costs a worse answer, never a wider one — which is exactly why
tool *selection* can be the model's job while patient *scope* cannot.

*The fallback must be visible.* A turn where the model chose well and one
where it failed and got the fallback produce the same answer from the same
tools. Only the guardrail code distinguishes them, so the codes are tested
as carefully as the happy path — an invisible fallback would let selection
quietly stop working while the metric kept reporting a plan.
"""

from __future__ import annotations

from app.agents.nodes import (
    API_TOOL_PLAN,
    HYBRID_TOOL_PLAN,
    ToolName,
    ToolPlan,
)
from app.agents.state import MAX_TOOL_CALLS
from app.tools.clinical import TOOLS, TOOLS_BY_NAME

# --- the selectable set ---------------------------------------------------- #


def test_the_enum_is_exactly_the_catalogue() -> None:
    """Derived, not restated.

    A second hand-written list of tool names is how a tool gets added to the
    catalogue and stays unselectable for a release — or worse, stays
    selectable after being removed.
    """
    assert {member.value for member in ToolName} == {spec.name for spec in TOOLS}


def test_every_selectable_tool_actually_exists() -> None:
    for member in ToolName:
        assert member.value in TOOLS_BY_NAME


def test_the_schema_constrains_decoding_to_real_tool_names() -> None:
    """The lesson ``RouteDecision.route`` already paid for.

    With an enum in the JSON Schema the server constrains decoding, so an 8B
    model cannot emit ``get_my_bloodwork``. Without it the model invents
    plausible names and the validator drops the whole plan.
    """
    schema = ToolPlan.model_json_schema()
    enum_values = schema["$defs"]["ToolName"]["enum"]
    assert set(enum_values) == {spec.name for spec in TOOLS}


def test_a_tool_outside_the_catalogue_is_rejected() -> None:
    """Validation, not filtering — the plan fails rather than being trimmed.

    A silently trimmed plan would run a subset nobody chose and report it as
    the model's selection.
    """
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ToolPlan.model_validate({"tools": ["get_my_bloodwork"]})


def test_an_empty_selection_is_representable() -> None:
    """"No lookup helps" has to be sayable, or the model will invent one."""
    assert ToolPlan(tools=[]).tools == []


# --- the plans ------------------------------------------------------------- #


def test_the_fallback_plan_is_all_real_tools() -> None:
    for name in API_TOOL_PLAN:
        assert name in TOOLS_BY_NAME


def test_the_fallback_plan_fits_the_call_budget() -> None:
    """A fallback that trips the ceiling would fail where guessing should not."""
    assert len(API_TOOL_PLAN) <= MAX_TOOL_CALLS


def test_the_fallback_is_broad_rather_than_narrow() -> None:
    """If we are guessing, guess wide.

    The fallback runs when the model's selection was unusable, which means
    nothing is known about what the question needs. A single-tool fallback
    would be a confident guess made with no information.
    """
    assert len(API_TOOL_PLAN) > 1


def test_hybrid_keeps_a_fixed_plan() -> None:
    """Not an oversight about §5 — see the constant's comment.

    ``get_my_last_encounter`` produces the ``encounter_id`` that anchors
    HYBRID's retrieval. A model dropping it would not run one fewer tool; it
    would silently un-anchor the notes and summarise whichever visit matched
    the wording best. The route is the plan, so there is no choice to
    delegate.
    """
    assert "get_my_last_encounter" in HYBRID_TOOL_PLAN
    for name in HYBRID_TOOL_PLAN:
        assert name in TOOLS_BY_NAME


def test_no_selectable_tool_takes_a_patient_identifier() -> None:
    """The property that makes delegating selection safe at all.

    If any tool accepted a patient id, letting the model choose tools would
    also let it choose whose record to read. It does not, so a wrong
    selection is a worse answer and never a broader one.
    """
    for member in ToolName:
        spec = TOOLS_BY_NAME[member.value]
        parameters = getattr(spec, "parameters", None) or {}
        fields = set(parameters) if isinstance(parameters, dict) else set()
        assert not {"patient_id", "patient", "external_id"} & fields
