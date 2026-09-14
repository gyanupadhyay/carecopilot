"""Deterministic medication diffing.

The HYBRID route reports medication changes from this function rather than
from the model (PRD §40 P12), so its edge cases are the ones that would
otherwise surface as a confidently wrong sentence in a clinical summary.
"""

from __future__ import annotations

from datetime import date

from app.models import Medication
from app.services.clinical import compare_medications


def med(
    name: str,
    dosage: str,
    frequency: str = "once daily",
    *,
    start: date = date(2026, 1, 1),
    end: date | None = None,
    status: str = "active",
) -> Medication:
    return Medication(
        patient_id=1,
        name=name,
        dosage=dosage,
        frequency=frequency,
        status=status,
        start_date=start,
        end_date=end,
    )


def test_no_changes_returns_empty() -> None:
    before = [med("Metformin", "500mg"), med("Lisinopril", "10mg")]
    after = [med("Metformin", "500mg"), med("Lisinopril", "10mg")]
    assert compare_medications(before, after) == []


def test_detects_started_medication() -> None:
    changes = compare_medications([], [med("Ibuprofen", "400mg")])
    assert [(c.change, c.name) for c in changes] == [("started", "Ibuprofen")]
    assert changes[0].before is None
    assert changes[0].after == "400mg once daily"


def test_detects_stopped_medication() -> None:
    changes = compare_medications([med("Glipizide", "5mg")], [])
    assert [(c.change, c.name) for c in changes] == [("stopped", "Glipizide")]
    assert changes[0].after is None


def test_detects_dose_increase() -> None:
    changes = compare_medications(
        [med("Metformin", "500mg", "twice daily")],
        [med("Metformin", "1000mg", "twice daily")],
    )
    assert len(changes) == 1
    assert changes[0].change == "dose_changed"
    assert changes[0].before == "500mg twice daily"
    assert changes[0].after == "1000mg twice daily"


def test_detects_frequency_change_at_same_dose() -> None:
    """A frequency change is a real change even when the dose is identical."""
    changes = compare_medications(
        [med("Omeprazole", "20mg", "once daily")],
        [med("Omeprazole", "20mg", "twice daily")],
    )
    assert [c.change for c in changes] == ["dose_changed"]


def test_matches_names_case_and_whitespace_insensitively() -> None:
    changes = compare_medications(
        [med(" metformin ", "500mg")], [med("Metformin", "500mg")]
    )
    assert changes == []


def test_full_demo_change_set() -> None:
    """The three-way change behind Demo 3: increase, stop, start."""
    before = [
        med("Metformin", "500mg", "twice daily"),
        med("Glipizide", "5mg"),
        med("Lisinopril", "10mg"),
    ]
    after = [
        med("Metformin", "1000mg", "twice daily"),
        med("Lisinopril", "10mg"),
        med("Ibuprofen", "400mg", "three times daily as needed"),
    ]
    by_name = {c.name: c.change for c in compare_medications(before, after)}
    assert by_name == {
        "Metformin": "dose_changed",
        "Glipizide": "stopped",
        "Ibuprofen": "started",
    }


def test_changes_are_ordered_deterministically() -> None:
    """Same inputs, same order — so snapshots and evals are stable."""
    before = [med("Zolpidem", "5mg"), med("Atorvastatin", "10mg")]
    after = [med("Atorvastatin", "20mg")]
    first = compare_medications(before, after)
    second = compare_medications(list(reversed(before)), after)
    assert [c.name for c in first] == [c.name for c in second]


def test_active_on_respects_interval() -> None:
    order = med(
        "Metformin", "500mg", start=date(2026, 1, 1), end=date(2026, 6, 30)
    )
    assert not order.active_on(date(2025, 12, 31))
    assert order.active_on(date(2026, 1, 1))
    assert order.active_on(date(2026, 6, 30))
    assert not order.active_on(date(2026, 7, 1))


def test_active_on_treats_null_end_date_as_open_ended() -> None:
    order = med("Lisinopril", "10mg", start=date(2026, 1, 1), end=None)
    assert order.active_on(date(2030, 1, 1))
