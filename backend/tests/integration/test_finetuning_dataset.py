"""The training corpus must contain no patient data (PRD §22, Principle 10).

This is the one property of the fine-tuning pipeline that cannot be checked
by reading the generator, and the reason is worth stating rather than
assuming. Every other guarantee in this project sits in front of the
*database*: row-level security, the identity mapping, the scoped tools, the
read-only analytics role. Model weights are behind none of them. A value
that reaches the training set leaves through any prompt that asks for it,
and no amount of authorization downstream can take it back.

So the assertion runs against the live database rather than against the
generator's vocabulary lists. It pulls the real names, external ids and
clinical values out of PostgreSQL and looks for them in the corpus. Checking
the generator instead would prove only that the lists someone wrote do not
contain patient data — which is true by construction and answers a different
question. The failure mode this catches is a future edit that reads from the
database "just to make the examples more realistic", which is a completely
reasonable-sounding thing to do and is the exact prohibition.

Skips when the corpus has not been generated: a machine that has not run
``build_finetuning_dataset.py`` has nothing to check, and failing there
would be a build-order complaint rather than a finding.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from sqlalchemy import select

from app.models import LabResult, Medication, Patient, Provider

pytestmark = pytest.mark.integration

CORPUS = Path(__file__).resolve().parents[3] / "data" / "fine_tuning"
SPLITS = ("train.jsonl", "val.jsonl", "test.jsonl")

#: Tokens too short or too common to be evidence of anything. Matching
#: "Lee" or "An" as a leaked surname would fail on ordinary English and
#: train everyone to ignore this test, which is worse than not having it.
MIN_NAME_LENGTH = 4


@pytest.fixture(scope="module")
def corpus() -> str:
    if not CORPUS.exists() or not any((CORPUS / name).exists() for name in SPLITS):
        pytest.skip(
            "no training corpus; run scripts/build_finetuning_dataset.py first"
        )
    return "\n".join(
        (CORPUS / name).read_text(encoding="utf-8")
        for name in SPLITS
        if (CORPUS / name).exists()
    )


@pytest.fixture(scope="module")
def corpus_lower(corpus: str) -> str:
    return corpus.lower()


# --- identity ------------------------------------------------------------- #


async def test_no_patient_name_appears(session, corpus_lower: str) -> None:
    """Not one of the hundred seeded patients.

    Matched as a whole name, or as a surname carrying a title or first name.
    A bare surname token is deliberately *not* enough, and the first version
    of this test proved why: it failed on patient ``P015``, surname "White",
    because the corpus says "white cell count". Real surnames are ordinary
    English words — White, Long, Young, Reed, Rose — so a token match reports
    a leak roughly whenever the vocabulary mentions a colour.

    A test that cries wolf gets muted, and a muted leak test is worse than
    none. What a genuine leak looks like is the name as a name: the full
    thing, or a surname with something in front of it identifying a person.
    """
    patients = (await session.scalars(select(Patient))).all()
    if not patients:
        pytest.skip("no patients seeded; run scripts/generate_data.py --reset")

    leaked: list[str] = []
    for patient in patients:
        full = str(patient.full_name).strip()
        if not full:
            continue
        if full.lower() in corpus_lower:
            leaked.append(f"{patient.external_id}:{full}")
            continue
        parts = full.split()
        if len(parts) < 2:
            continue
        first, surname = parts[0].lower(), parts[-1].strip(",.").lower()
        if len(surname) < MIN_NAME_LENGTH:
            continue
        # "Dr White", "Mr White", "Anna White" — a surname doing a surname's
        # job. "white cell count" is not.
        titles = f"dr|doctor|mr|mrs|ms|miss|patient|{re.escape(first)}"
        qualified = rf"\b(?:{titles})\.?\s+{re.escape(surname)}\b"
        if re.search(qualified, corpus_lower):
            leaked.append(f"{patient.external_id}:{full}")
    assert not leaked, (
        f"patient names in the training corpus: {sorted(set(leaked))[:10]}. "
        "PRD §22 Principle 10 — fine-tune behaviour, never patient data."
    )


async def test_no_external_id_appears(session, corpus: str) -> None:
    """``P001`` and friends. The most direct form of leakage there is."""
    external_ids = (await session.scalars(select(Patient.external_id))).all()
    if not external_ids:
        pytest.skip("no patients seeded")
    leaked = [eid for eid in external_ids if eid and str(eid) in corpus]
    assert not leaked, f"patient external ids in the corpus: {leaked[:10]}"


async def test_no_provider_name_appears(session, corpus_lower: str) -> None:
    """Clinicians are people too, and the graph traversals return them."""
    providers = (await session.scalars(select(Provider))).all()
    if not providers:
        pytest.skip("no providers seeded")
    leaked: list[str] = []
    for provider in providers:
        for part in str(provider.name).split():
            token = part.strip(",.").lower()
            if len(token) < MIN_NAME_LENGTH or token in {"nurse", "practitioner"}:
                continue
            if re.search(rf"\b{re.escape(token)}\b", corpus_lower):
                leaked.append(part)
    assert not leaked, f"provider names in the corpus: {sorted(set(leaked))[:10]}"


# --- clinical content ----------------------------------------------------- #


async def test_no_prescription_belongs_to_anyone(session, corpus_lower: str) -> None:
    """Drug names are generic; a drug *with a dosage* is a prescription.

    ``metformin`` is a word in a medical vocabulary and appears in the
    corpus deliberately — a patient asking "why am I on metformin" is the
    behaviour being trained. ``metformin 500mg`` is what somebody takes.
    """
    rows = (await session.execute(select(Medication.name, Medication.dosage))).all()
    if not rows:
        pytest.skip("no medications seeded")
    leaked = [
        f"{name} {dosage}"
        for name, dosage in rows
        if name and dosage and f"{name} {dosage}".lower() in corpus_lower
    ]
    assert not leaked, f"prescriptions in the corpus: {sorted(set(leaked))[:10]}"


#: How close a value must sit to its test name to count as that test's
#: result. A pair further apart than this is two independent strings.
PAIR_WINDOW = 40


async def test_no_lab_value_appears_beside_its_test(
    session, corpus: str
) -> None:
    """A number alone is not a record; a number *beside* its test is.

    Both halves of that sentence are load-bearing, and the first version of
    this test ignored the second. It asked whether the test name appeared
    anywhere in the corpus and whether the value appeared anywhere in the
    corpus — and reported eight HbA1c results, because "HbA1c" occurs in the
    production tool catalogue that every routing example carries as its
    system prompt, while "5.1" and "6.9" are invented placeholders in the
    grounding examples. Two unrelated strings, several messages apart, in
    text the generator never wrote.

    Values collide by arithmetic: with two significant figures, some patient
    somewhere has a result matching any small number that appears for any
    reason. Only adjacency distinguishes a record from a coincidence, so
    that is what this looks for — and within one message, since a system
    prompt and an assistant reply are not the same document.
    """
    rows = (
        await session.execute(
            select(LabResult.test_name, LabResult.value).limit(2000)
        )
    ).all()
    if not rows:
        pytest.skip("no lab results seeded")

    messages = [
        message["content"]
        for line in corpus.splitlines()
        if line.strip()
        for message in json.loads(line)["messages"]
    ]

    leaked: list[str] = []
    for test_name, value in rows:
        if not test_name:
            continue
        name = str(test_name).lower()
        rendered = f"{float(value):g}"
        pattern = re.compile(
            rf"{re.escape(name)}.{{0,{PAIR_WINDOW}}}?\b{re.escape(rendered)}\b"
            rf"|\b{re.escape(rendered)}\b.{{0,{PAIR_WINDOW}}}?{re.escape(name)}",
            re.IGNORECASE | re.DOTALL,
        )
        if any(pattern.search(message) for message in messages):
            leaked.append(f"{test_name}={rendered}")
    assert not leaked, (
        f"lab results in the corpus: {sorted(set(leaked))[:10]}. A value beside "
        "its test name is a record, however generic the test name is."
    )


# --- shape ---------------------------------------------------------------- #


def test_every_row_is_a_well_formed_chat_example(corpus: str) -> None:
    """Malformed rows fail silently in most trainers — as skipped examples.

    A corpus that quietly trains on half its rows produces a weak adapter
    and no error, which is indistinguishable from a behaviour that did not
    tune well.
    """
    for number, line in enumerate(corpus.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages")
        assert isinstance(messages, list) and len(messages) == 3, f"line {number}"
        assert [m["role"] for m in messages] == [
            "system",
            "user",
            "assistant",
        ], f"line {number}"
        for message in messages:
            assert message.get("content", "").strip(), f"empty content, line {number}"


def test_no_template_spans_two_splits() -> None:
    """The split's whole purpose, asserted where it can be seen.

    A template in both train and test turns the test set into a memorisation
    check and reports a tuned-model gain that will not survive a new
    phrasing. The generator writes this into the manifest; this fails the
    build if it ever stops being true.
    """
    manifest_path = CORPUS / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("no manifest; run scripts/build_finetuning_dataset.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["template_overlap"] == [], (
        f"templates in more than one split: {manifest['template_overlap']}"
    )


def test_the_assistant_never_narrates_reasoning(corpus: str) -> None:
    """§26 forbids exposing chain-of-thought.

    Training targets are where it would enter: a model tuned on "first I
    considered, then I decided" produces exactly what the output guardrail
    then has to strip, on every single turn.
    """
    tells = (
        "let me think",
        "first, i",
        "step 1",
        "my reasoning",
        "i'll start by",
        "<think>",
    )
    offenders: list[str] = []
    for line in corpus.splitlines():
        if not line.strip():
            continue
        assistant = json.loads(line)["messages"][2]["content"].lower()
        offenders.extend(tell for tell in tells if tell in assistant)
    assert not offenders, f"reasoning narration in training targets: {set(offenders)}"
