"""Generate the synthetic patient dataset.

    python scripts/generate_data.py --reset

Everything produced here is fabricated. No record, name, date or value is
derived from a real person (PRD §34).

Two properties matter more than volume:

*Internal consistency.* A patient's notes, medications, labs and
appointments describe one coherent history. A note that says therapy was
increased is accompanied by a medication row that actually ends and another
that actually starts, so the deterministic before/after comparison and the
RAG summary agree with each other. Data that contradicts itself makes
retrieval evaluation meaningless, because there is no correct answer to
score against.

*A scripted demo patient.* ``P001`` is generated deliberately rather than at
random so that the demo sequence in PRD §37 works on a fresh database: a
recent knee-pain encounter, a dose increase, a discontinuation, a new
prescription, an upcoming appointment, and enough blood-pressure readings
for the analytical question to have a non-trivial answer.

Chunks and embeddings are *not* created here. Ingestion is a separate
pipeline over ``clinical_documents`` (``scripts/ingest_documents.py``), so
that re-chunking or re-embedding never requires regenerating the dataset.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

# Run from the repository root without installing the backend as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from faker import Faker
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session
from synthetic_catalog import (
    ALLERGIES,
    APPOINTMENT_NOTE_TEMPLATES,
    CONDITION_KEY_BY_MEDICATION,
    CONDITIONS,
    CONDITIONS_BY_KEY,
    DIASTOLIC,
    HBA1C,
    PROVIDERS,
    ROUTINE_LABS,
    SHORT_COURSES,
    SYSTOLIC,
    Condition,
    LabSpec,
    MedSpec,
)

from app.auth.demo import DEMO_PASSWORD
from app.auth.security import hash_password
from app.config import settings
from app.models import (
    Allergy,
    Appointment,
    ClinicalDocument,
    Diagnosis,
    Encounter,
    LabResult,
    Medication,
    Patient,
    PatientCondition,
    Procedure,
    Provider,
    User,
    UserPatientMapping,
)
from app.models import Condition as ConditionRow

DEFAULT_SEED = 20260913
DEMO_EXTERNAL_ID = "P001"

#: Tables emptied by --reset, in dependency order. Chunks and documents are
#: included because a regenerated dataset invalidates any existing index.
RESET_ORDER = (
    "request_traces",
    "audit_logs",
    "messages",
    "conversations",
    "user_patient_mapping",
    "document_chunks",
    "clinical_documents",
    "lab_results",
    "medications",
    "appointments",
    # Before `encounters`, which references conditions, and before
    # `patients`, which patient_conditions references. CASCADE would handle
    # it either way; the order is kept explicit so the dependency is readable.
    "patient_conditions",
    "diagnoses",
    "procedures",
    "allergies",
    "encounters",
    "conditions",
    "users",
    "patients",
    "providers",
)


# --------------------------------------------------------------------- #
# Note assembly
# --------------------------------------------------------------------- #

def render_note(
    *,
    chief_complaint: str,
    history: str,
    examination: str,
    assessment: str,
    medications: list[str],
    plan: str,
) -> str:
    """Render a clinical note with predictable section headings.

    The headings are the contract between this generator and the
    section-aware chunker: an upper-case line on its own, followed by the
    section body. Keeping that shape stable is what lets a retrieved chunk
    report which section it came from without a parser that guesses.
    """
    med_lines = "\n".join(f"- {line}" for line in medications) or "- None recorded."
    sections = (
        ("CHIEF COMPLAINT", chief_complaint),
        ("HISTORY OF PRESENT ILLNESS", history),
        ("EXAMINATION", examination),
        ("ASSESSMENT", assessment),
        ("MEDICATIONS", med_lines),
        ("PLAN", plan),
    )
    return "\n\n".join(f"{heading}\n{body}" for heading, body in sections)


def _describe(med: Medication) -> str:
    return f"{med.name} {med.dosage} {med.frequency}".strip()


# --------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------- #

class Generator:
    def __init__(self, session: Session, *, seed: int, today: date) -> None:
        self.session = session
        self.rng = random.Random(seed)
        self.faker = Faker()
        Faker.seed(seed)
        self.today = today
        # Argon2 is intentionally slow. The demo password is identical for
        # every seeded account, so it is hashed once rather than a hundred
        # times; the stored value is still a real salted Argon2id hash.
        self.demo_hash = hash_password(DEMO_PASSWORD)
        self.providers: list[Provider] = []
        #: Catalogue key -> persisted row, populated by ``create_conditions``.
        self.conditions_by_key: dict[str, ConditionRow] = {}
        self.counts: dict[str, int] = {}

    # -- reference data ------------------------------------------------ #

    def create_providers(self) -> None:
        self.providers = [
            Provider(name=name, specialty=specialty) for name, specialty in PROVIDERS
        ]
        self.session.add_all(self.providers)
        self.session.flush()

    def create_conditions(self) -> None:
        """Persist the catalogue, so the graph has a Condition to point at.

        Reconciled by ``key`` rather than inserted blindly: ``--if-empty``
        runs against a database that may already hold the catalogue, and a
        second insert would violate the unique constraint and abort the seed.
        """
        existing = {
            row.key: row for row in self.session.scalars(select(ConditionRow)).all()
        }
        self.conditions_by_key = dict(existing)

        for condition in CONDITIONS:
            row = existing.get(condition.key)
            if row is None:
                row = ConditionRow(
                    key=condition.key,
                    display=condition.display,
                    aliases=list(condition.aliases),
                )
                self.session.add(row)
                self.conditions_by_key[condition.key] = row
            else:
                # A reworded display name updates in place. Inserting a second
                # row would fork every patient's history across the two.
                row.display = condition.display
                row.aliases = list(condition.aliases)
        self.session.flush()

    def _provider_for(self, condition: Condition | None) -> Provider:
        """Prefer a plausible specialty, but never fail if none matches."""
        preferred = {
            "type_2_diabetes": "Endocrinology",
            "hypertension": "Cardiology",
            "knee_osteoarthritis": "Orthopaedics",
            "asthma": "Respiratory Medicine",
            "gerd": "Gastroenterology",
            "hyperlipidemia": "Internal Medicine",
            "hypothyroidism": "Endocrinology",
        }.get(condition.key if condition else "", "")
        candidates = [p for p in self.providers if p.specialty == preferred]
        if candidates and self.rng.random() < 0.6:
            return self.rng.choice(candidates)
        return self.rng.choice(self.providers)

    # -- patients ------------------------------------------------------ #

    def create_patients(self, count: int) -> list[Patient]:
        patients: list[Patient] = []
        for index in range(1, count + 1):
            gender = self.rng.choice(["female", "male", "non-binary"])
            if gender == "female":
                first = self.faker.first_name_female()
            elif gender == "male":
                first = self.faker.first_name_male()
            else:
                first = self.faker.first_name_nonbinary()

            patient = Patient(
                external_id=f"P{index:03d}",
                first_name=first,
                last_name=self.faker.last_name(),
                date_of_birth=self.faker.date_of_birth(minimum_age=24, maximum_age=86),
                gender=gender,
            )
            patients.append(patient)

        self.session.add_all(patients)
        self.session.flush()

        # One login per patient. The email is derived from the external id
        # so that credentials are predictable in the demo and in tests.
        users = [
            User(
                email=f"{p.external_id.lower()}@carecopilot.demo",
                display_name=p.full_name,
                password_hash=self.demo_hash,
                role="patient",
                is_active=True,
            )
            for p in patients
        ]
        self.session.add_all(users)
        self.session.flush()

        # Authorization is a recorded grant, not a column on the account.
        self.session.add_all(
            UserPatientMapping(
                user_id=user.id,
                patient_id=patient.id,
                relationship_type="self",
                is_active=True,
            )
            for user, patient in zip(users, patients, strict=True)
        )
        self.session.flush()
        return patients

    def _conditions_for(self, patient: Patient, *, demo: bool) -> list[Condition]:
        if demo:
            return [
                CONDITIONS_BY_KEY["knee_osteoarthritis"],
                CONDITIONS_BY_KEY["type_2_diabetes"],
                CONDITIONS_BY_KEY["hypertension"],
            ]
        how_many = self.rng.choices([1, 2, 3], weights=[3, 5, 2])[0]
        return self.rng.sample(list(CONDITIONS), how_many)

    # -- per-patient history ------------------------------------------- #

    def build_history(self, patient: Patient, *, demo: bool) -> None:
        conditions = self._conditions_for(patient, demo=demo)
        encounter_dates = self._encounter_dates(demo=demo)

        encounters: list[Encounter] = []
        # A medication ledger keyed by name, holding the row currently in
        # force. Changes close the old row and open a new one, which is what
        # makes the history diffable rather than merely descriptive.
        ledger: dict[str, Medication] = {}
        #: condition key -> first encounter date, for the HAS_CONDITION edge.
        first_seen: dict[str, date] = {}

        for position, encounter_date in enumerate(encounter_dates):
            is_last = position == len(encounter_dates) - 1
            if demo and is_last:
                condition = conditions[0]
            elif position < len(conditions):
                # Cover every condition once before repeating any. Pure
                # random choice left roughly a third of patients with a
                # condition that never came up at a visit, so
                # HAS_CONDITION.onset_date was null and the graph's
                # condition timeline for it was empty — the question
                # "what happened with my diabetes?" answered "nothing on
                # record" for a patient whose record says they have it.
                condition = conditions[position]
            else:
                condition = self.rng.choice(conditions)
            provider = self._provider_for(condition)

            encounter = Encounter(
                patient_id=patient.id,
                provider_id=provider.id,
                # The clinical problem behind the visit, kept alongside the
                # chief complaint rather than only inside it. `reason` is
                # prose the graph cannot traverse (PRD §17).
                condition_id=self.conditions_by_key[condition.key].id,
                encounter_date=encounter_date,
                encounter_type=self.rng.choice(
                    ["office_visit", "office_visit", "telehealth", "specialist_consult"]
                ),
                reason=self.rng.choice(condition.chief_complaints),
            )
            self.session.add(encounter)
            self.session.flush()
            encounters.append(encounter)
            # Earliest visit for this problem, which is the closest thing to
            # an onset date a record of visits can honestly supply.
            first_seen.setdefault(condition.key, encounter_date)

            if position == 0:
                # Every diagnosis a patient carries is already on therapy by
                # the time the record starts. Starting only the condition
                # discussed at the first visit would leave most patients on a
                # single drug, which makes "what am I currently taking?" a
                # trivial question and the before/after diff nearly always
                # empty.
                changed = self._start_baseline_therapy(
                    patient, encounter, conditions, ledger, demo=demo
                )
            elif demo:
                # The demo patient's therapy is held constant until the final
                # visit. Random escalation in between would consume the dose
                # ladder, and "increased from 500mg to 1000mg" would no longer
                # be the change Demo 3 shows.
                changed = (
                    self._apply_demo_changes(patient, encounter, ledger)
                    if is_last
                    else []
                )
            else:
                changed = self._apply_medication_changes(
                    patient, encounter, condition, ledger
                )
            self._create_labs(patient, encounter, condition, demo=demo)
            self._record_diagnosis(patient, encounter, condition)
            self._create_procedures(patient, encounter, condition, provider)
            self._create_note(patient, encounter, condition, ledger, changed)

        self._record_conditions(patient, conditions, first_seen)
        self._create_allergies(patient)
        self._create_past_courses(patient)
        self._create_interim_vitals(patient, conditions, demo=demo)
        self._create_appointments(patient, encounters, demo=demo)

    def _record_conditions(
        self,
        patient: Patient,
        conditions: list[Condition],
        first_seen: dict[str, date],
    ) -> None:
        """Write the ``HAS_CONDITION`` edges for this patient.

        Every condition assigned to the patient is recorded, including any
        that never came up at a visit in range — the patient still has it,
        and a graph that showed only what was discussed would answer "what
        conditions do I have?" with a subset.
        """
        self.session.add_all(
            PatientCondition(
                patient_id=patient.id,
                condition_id=self.conditions_by_key[condition.key].id,
                onset_date=first_seen.get(condition.key),
            )
            for condition in conditions
        )

    def _create_past_courses(self, patient: Patient) -> None:
        """Finished courses of treatment from earlier in the record.

        Every course ends at least sixty days before the run date, so it can
        never appear in the before/after window around the most recent
        encounter and confuse the medication diff.
        """
        for spec in self.rng.sample(
            list(SHORT_COURSES), self.rng.choices([2, 3, 4], weights=[4, 4, 2])[0]
        ):
            duration = self.rng.randint(5, 14)
            days_ago = self.rng.randint(60 + duration, 700)
            start = self.today - timedelta(days=days_ago)
            self.session.add(
                Medication(
                    patient_id=patient.id,
                    encounter_id=None,
                    name=spec.name,
                    dosage=spec.doses[0],
                    frequency=spec.frequency,
                    status="completed",
                    start_date=start,
                    end_date=start + timedelta(days=duration),
                )
            )

    def _encounter_dates(self, *, demo: bool) -> list[date]:
        """Encounter dates, oldest first, spread over the past ~2 years.

        Dates are walked backwards with a minimum gap rather than sampled
        independently. Independent sampling produces collisions, and two
        encounters on one day are not merely unrealistic: the second one
        would close a medication the first one started, giving it an
        ``end_date`` before its ``start_date``.
        """
        if demo:
            # Anchored to the run date so the demo stays current: a visit a
            # few days ago, then earlier visits at roughly quarterly spacing.
            offsets = [640, 470, 300, 160, 3]
            return [self.today - timedelta(days=d) for d in offsets]

        how_many = self.rng.choices([3, 4, 5, 6], weights=[3, 4, 3, 1])[0]
        dates: list[date] = []
        days_ago = self.rng.randint(20, 90)
        for _ in range(how_many):
            dates.append(self.today - timedelta(days=days_ago))
            days_ago += self.rng.randint(55, 150)
        return sorted(dates)

    # -- medications ---------------------------------------------------- #

    def _new_medication(
        self,
        patient: Patient,
        encounter: Encounter,
        spec: MedSpec,
        dose_index: int,
        start: date,
    ) -> Medication:
        medication = Medication(
            patient_id=patient.id,
            encounter_id=encounter.id,
            condition_id=self._condition_id_for_medication(spec.name),
            name=spec.name,
            dosage=spec.doses[min(dose_index, len(spec.doses) - 1)],
            frequency=spec.frequency,
            status="active",
            start_date=start,
            end_date=None,
        )
        self.session.add(medication)
        return medication

    # -- procedures, allergies, diagnoses -------------------------------- #

    def _record_diagnosis(
        self, patient: Patient, encounter: Encounter, condition: Condition
    ) -> None:
        """The coded assertion made at this visit (PRD §17).

        Every visit filed under a condition records one. ``rank`` is always
        primary here because the generator gives each encounter a single
        condition; the column exists because real records distinguish the
        reason for the visit from what was noted alongside it, and a graph
        that cannot express that difference misreports it.
        """
        self.session.add(
            Diagnosis(
                patient_id=patient.id,
                encounter_id=encounter.id,
                condition_id=self.conditions_by_key[condition.key].id,
                # Copied, not referenced: correcting the catalogue must not
                # rewrite what was recorded at an earlier visit.
                code=condition.icd10 or None,
                diagnosed_date=encounter.encounter_date,
                rank="primary",
            )
        )

    def _create_procedures(
        self,
        patient: Patient,
        encounter: Encounter,
        condition: Condition,
        provider: Provider,
    ) -> None:
        """Occasionally, something was done as well as prescribed.

        Deliberately uncommon. Most visits for these conditions are talk,
        examination and a prescription; a dataset where every visit carries a
        procedure would make "have I had any procedures?" a question with the
        same boring answer for all hundred patients.
        """
        if not condition.procedures or self.rng.random() > 0.25:
            return
        spec = self.rng.choice(condition.procedures)
        self.session.add(
            Procedure(
                patient_id=patient.id,
                encounter_id=encounter.id,
                provider_id=provider.id,
                condition_id=self.conditions_by_key[condition.key].id,
                name=spec.name,
                code=spec.code,
                performed_date=encounter.encounter_date,
            )
        )

    def _create_allergies(self, patient: Patient) -> None:
        """Allergies belong to the patient, not to a visit.

        Sampled without replacement so a patient cannot be given the same
        substance twice — which the unique constraint would reject, aborting
        the whole seed rather than the one row.
        """
        how_many = self.rng.choices([0, 1, 2], weights=[3, 5, 2])[0]
        if not how_many:
            return
        for spec in self.rng.sample(list(ALLERGIES), how_many):
            self.session.add(
                Allergy(
                    patient_id=patient.id,
                    substance=spec.substance,
                    reaction=spec.reaction,
                    severity=spec.severity,
                    # Recorded at some point before the record's first visit:
                    # allergies are asked about at registration, not
                    # discovered on a schedule.
                    recorded_date=self.today
                    - timedelta(days=self.rng.randint(400, 2200)),
                )
            )

    def _condition_id_for_medication(self, name: str) -> int | None:
        """The condition a drug treats, or None for a short course."""
        key = CONDITION_KEY_BY_MEDICATION.get(name)
        return self.conditions_by_key[key].id if key else None

    def _close(self, medication: Medication, *, on: date) -> None:
        """End an order the day *before* the visit that replaced it.

        Closing it on the visit date instead would leave the old and new
        rows both in force on that date, and the before/after comparison
        would see one medication twice rather than a change.

        The clamp keeps the ``end_date >= start_date`` constraint true even
        if an order is somehow started and stopped on the same day. With
        distinct encounter dates that case cannot arise, so this is a
        guard rather than a code path the generator relies on.
        """
        medication.end_date = max(on - timedelta(days=1), medication.start_date)
        medication.status = "discontinued"

    def _start_baseline_therapy(
        self,
        patient: Patient,
        encounter: Encounter,
        conditions: list[Condition],
        ledger: dict[str, Medication],
        *,
        demo: bool,
    ) -> list[str]:
        """Put the patient on therapy for each condition at the first visit."""
        when = encounter.encounter_date
        narrative: list[str] = []

        if demo:
            return self._start_demo_baseline(patient, encounter, ledger)

        for condition in conditions:
            if not condition.medications:
                continue
            primary = condition.medications[0]
            if primary.name in ledger:
                continue
            # Start part-way up the ladder sometimes: a record that begins
            # mid-treatment is more realistic than one where every patient
            # starts at the lowest dose on the same day.
            dose_index = self.rng.choices([0, 1], weights=[7, 3])[0]
            started = self._new_medication(
                patient, encounter, primary, dose_index, when
            )
            ledger[primary.name] = started
            narrative.append(f"{primary.name} {started.dosage} continued.")

            # A minority are already on a second agent for the same problem.
            if len(condition.medications) > 1 and self.rng.random() < 0.3:
                second = condition.medications[1]
                if second.name not in ledger:
                    extra = self._new_medication(patient, encounter, second, 0, when)
                    ledger[second.name] = extra
                    narrative.append(f"{second.name} {extra.dosage} continued.")

        self.session.flush()
        return narrative

    def _apply_medication_changes(
        self,
        patient: Patient,
        encounter: Encounter,
        condition: Condition,
        ledger: dict[str, Medication],
    ) -> list[str]:
        when = encounter.encounter_date
        narrative: list[str] = []

        primary = condition.medications[0] if condition.medications else None
        starting_new = primary is not None and primary.name not in ledger
        if starting_new and self.rng.random() < 0.7:
            started = self._new_medication(patient, encounter, primary, 0, when)
            ledger[primary.name] = started
            narrative.append(f"{primary.name} started at {started.dosage}.")
            self.session.flush()
            return narrative

        if not primary or self.rng.random() > condition.change_rate:
            return narrative

        current = ledger.get(primary.name)
        if current is None:
            return narrative

        dose_index = primary.doses.index(current.dosage)
        if dose_index < len(primary.doses) - 1:
            self._close(current, on=when)
            replacement = self._new_medication(
                patient, encounter, primary, dose_index + 1, when
            )
            ledger[primary.name] = replacement
            narrative.append(
                f"{primary.name} increased from {current.dosage} to "
                f"{replacement.dosage}."
            )
        else:
            # Already at the top of the ladder: add a second agent instead.
            extras = [m for m in condition.medications[1:] if m.name not in ledger]
            if extras:
                added = self.rng.choice(extras)
                ledger[added.name] = self._new_medication(
                    patient, encounter, added, 0, when
                )
                narrative.append(f"{added.name} added at {ledger[added.name].dosage}.")

        self.session.flush()
        return narrative

    def _start_demo_baseline(
        self, patient: Patient, encounter: Encounter, ledger: dict[str, Medication]
    ) -> list[str]:
        """The fixed starting regimen for ``P001``.

        Chosen so that the final visit can demonstrate all three kinds of
        change: Metformin sits at the bottom of its ladder so it can be
        increased, Glipizide is present so it can be stopped, and no knee
        medication is started here so that one can genuinely begin at the
        last visit.
        """
        when = encounter.encounter_date
        diabetes = CONDITIONS_BY_KEY["type_2_diabetes"]
        hypertension = CONDITIONS_BY_KEY["hypertension"]

        baseline = (
            (diabetes.medications[0], 0),  # Metformin 500mg
            (diabetes.medications[1], 0),  # Glipizide 5mg
            (hypertension.medications[0], 0),  # Lisinopril 10mg
        )
        narrative: list[str] = []
        for spec, dose_index in baseline:
            started = self._new_medication(patient, encounter, spec, dose_index, when)
            ledger[spec.name] = started
            narrative.append(f"{spec.name} {started.dosage} continued.")

        self.session.flush()
        return narrative

    def _apply_demo_changes(
        self, patient: Patient, encounter: Encounter, ledger: dict[str, Medication]
    ) -> list[str]:
        """The scripted change set behind Demo 3 in PRD §37.

        One increase, one discontinuation, one new start — the three cases
        :func:`app.services.clinical.compare_medications` distinguishes.
        """
        when = encounter.encounter_date
        narrative: list[str] = []
        diabetes = CONDITIONS_BY_KEY["type_2_diabetes"]
        knee = CONDITIONS_BY_KEY["knee_osteoarthritis"]

        metformin_spec = diabetes.medications[0]
        metformin = ledger.get(metformin_spec.name)
        if metformin is not None:
            self._close(metformin, on=when)
        replacement = self._new_medication(patient, encounter, metformin_spec, 1, when)
        ledger[metformin_spec.name] = replacement
        narrative.append(
            f"Metformin increased from {metformin.dosage if metformin else '500mg'} "
            f"to {replacement.dosage}."
        )

        glipizide_spec = diabetes.medications[1]
        glipizide = ledger.get(glipizide_spec.name)
        if glipizide is not None:
            self._close(glipizide, on=when)
            ledger.pop(glipizide_spec.name)
            narrative.append(f"{glipizide_spec.name} discontinued.")

        ibuprofen_spec = knee.medications[0]
        if ibuprofen_spec.name not in ledger:
            started = self._new_medication(patient, encounter, ibuprofen_spec, 1, when)
            ledger[ibuprofen_spec.name] = started
            narrative.append(f"{ibuprofen_spec.name} {started.dosage} started.")

        self.session.flush()
        return narrative

    # -- labs ----------------------------------------------------------- #

    def _lab_value(self, spec: LabSpec, *, abnormal: bool) -> Decimal:
        low, high = spec.abnormal if abnormal else spec.normal
        value = self.rng.uniform(low, high)
        return Decimal(f"{value:.{spec.decimals}f}")

    def _add_lab(
        self,
        patient: Patient,
        spec: LabSpec,
        when: date,
        *,
        abnormal: bool,
        encounter: Encounter | None = None,
    ) -> None:
        self.session.add(
            LabResult(
                patient_id=patient.id,
                encounter_id=encounter.id if encounter else None,
                test_name=spec.test_name,
                value=self._lab_value(spec, abnormal=abnormal),
                unit=spec.unit,
                reference_range=spec.reference_range,
                result_date=when,
            )
        )

    def _create_labs(
        self, patient: Patient, encounter: Encounter, condition: Condition, *, demo: bool
    ) -> None:
        when = encounter.encounter_date
        for spec in ROUTINE_LABS:
            abnormal = self.rng.random() < (0.45 if demo else 0.25)
            self._add_lab(patient, spec, when, abnormal=abnormal, encounter=encounter)

        for spec in condition.labs:
            if spec in ROUTINE_LABS:
                continue
            abnormal = self.rng.random() < (0.7 if demo else 0.45)
            self._add_lab(patient, spec, when, abnormal=abnormal, encounter=encounter)

    def _create_interim_vitals(
        self, patient: Patient, conditions: list[Condition], *, demo: bool
    ) -> None:
        """Home monitoring between visits.

        Without these there are only three or four blood-pressure readings
        per patient, and "how many times was my systolic above 140 in the
        last six months?" has a trivial answer that proves nothing about the
        Text-to-SQL path.
        """
        keys = {c.key for c in conditions}
        monitored = demo or bool(keys & {"hypertension", "type_2_diabetes"})
        interval_days = 30 if monitored else 90
        months = 18

        when = self.today - timedelta(days=months * 30)
        while when < self.today:
            abnormal = self.rng.random() < (0.55 if monitored else 0.2)
            self._add_lab(patient, SYSTOLIC, when, abnormal=abnormal)
            self._add_lab(patient, DIASTOLIC, when, abnormal=abnormal)
            when += timedelta(days=interval_days)

        if demo or "type_2_diabetes" in keys:
            # Quarterly HbA1c, which is the cadence the notes describe.
            for quarter in range(6):
                when = self.today - timedelta(days=90 * quarter + 5)
                self._add_lab(
                    patient, HBA1C, when, abnormal=self.rng.random() < 0.7
                )

    # -- documents ------------------------------------------------------ #

    def _create_note(
        self,
        patient: Patient,
        encounter: Encounter,
        condition: Condition,
        ledger: dict[str, Medication],
        changes: list[str],
    ) -> None:
        plan = self.rng.choice(condition.plan)
        if changes:
            plan = f"{' '.join(changes)} {plan}"

        current_medications = sorted(ledger.values(), key=lambda m: m.name)
        content = render_note(
            chief_complaint=(
                encounter.reason or self.rng.choice(condition.chief_complaints)
            ),
            history=self.rng.choice(condition.history),
            examination=self.rng.choice(condition.examination),
            assessment=self.rng.choice(condition.assessment),
            medications=[_describe(m) for m in current_medications],
            plan=plan,
        )

        self.session.add(
            ClinicalDocument(
                patient_id=patient.id,
                encounter_id=encounter.id,
                document_type="clinical_note",
                title=(
                    f"{condition.display} — "
                    f"{encounter.encounter_date.isoformat()}"
                ),
                content=content,
                version=1,
                status="final",
            )
        )

    # -- appointments --------------------------------------------------- #

    def _create_appointments(
        self, patient: Patient, encounters: list[Encounter], *, demo: bool
    ) -> None:
        # A completed appointment for each encounter that happened.
        for encounter in encounters:
            slot = self.rng.choice([9, 10, 11, 13, 14, 15, 16])
            self.session.add(
                Appointment(
                    patient_id=patient.id,
                    provider_id=encounter.provider_id,
                    appointment_date=datetime.combine(
                        encounter.encounter_date, time(hour=slot), tzinfo=UTC
                    ),
                    appointment_type=self.rng.choice(
                        [
                            "follow_up",
                            "annual_physical",
                            "specialist_consult",
                            "lab_review",
                        ]
                    ),
                    status="completed",
                    notes=self.rng.choice(APPOINTMENT_NOTE_TEMPLATES) or None,
                )
            )

        # A cancelled or missed visit for roughly a third of patients: the
        # "next appointment" tool has to exclude these, so the dataset must
        # contain some.
        if self.rng.random() < 0.35:
            days_ago = self.rng.randint(30, 300)
            self.session.add(
                Appointment(
                    patient_id=patient.id,
                    provider_id=self.rng.choice(self.providers).id,
                    appointment_date=datetime.combine(
                        self.today - timedelta(days=days_ago), time(hour=11), tzinfo=UTC
                    ),
                    appointment_type="follow_up",
                    status=self.rng.choice(["cancelled", "no_show"]),
                    notes=None,
                )
            )

        if demo:
            # Fixed: an upcoming visit with a named provider, one week out.
            provider = next(
                (p for p in self.providers if p.name == "Dr. Sarah Smith"),
                self.providers[0],
            )
            self.session.add(
                Appointment(
                    patient_id=patient.id,
                    provider_id=provider.id,
                    appointment_date=datetime.combine(
                        self.today + timedelta(days=7), time(hour=10), tzinfo=UTC
                    ),
                    appointment_type="follow_up",
                    status="scheduled",
                    notes="Physical therapy progress review.",
                )
            )
            # A cancelled slot *earlier* than the real one, so that a naive
            # "soonest appointment" query gives the wrong answer and the
            # tool's status filter is actually exercised.
            self.session.add(
                Appointment(
                    patient_id=patient.id,
                    provider_id=provider.id,
                    appointment_date=datetime.combine(
                        self.today + timedelta(days=3), time(hour=9), tzinfo=UTC
                    ),
                    appointment_type="follow_up",
                    status="cancelled",
                    notes="Cancelled by the patient.",
                )
            )
            return

        for _ in range(self.rng.choices([1, 2], weights=[7, 3])[0]):
            days_ahead = self.rng.randint(4, 120)
            self.session.add(
                Appointment(
                    patient_id=patient.id,
                    provider_id=self.rng.choice(self.providers).id,
                    appointment_date=datetime.combine(
                        self.today + timedelta(days=days_ahead),
                        time(hour=self.rng.choice([9, 10, 11, 14, 15])),
                        tzinfo=UTC,
                    ),
                    appointment_type=self.rng.choice(
                        ["follow_up", "annual_physical", "lab_review", "telehealth"]
                    ),
                    status="scheduled",
                    notes=self.rng.choice(APPOINTMENT_NOTE_TEMPLATES) or None,
                )
            )

    # -- orchestration -------------------------------------------------- #

    def run(self, patient_count: int) -> None:
        self.create_providers()
        # Before any patient: encounters carry a condition_id, so the
        # catalogue has to exist before the first history is built.
        self.create_conditions()
        patients = self.create_patients(patient_count)
        for patient in patients:
            self.build_history(patient, demo=patient.external_id == DEMO_EXTERNAL_ID)
            self.session.flush()


# --------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------- #

def reset(session: Session) -> None:
    """Empty the dataset, restarting identity sequences.

    TRUNCATE rather than DELETE so that a regenerated database has the same
    ids as a freshly created one — which keeps the evaluation fixtures and
    the demo URLs stable across runs.
    """
    session.execute(
        text(f"TRUNCATE TABLE {', '.join(RESET_ORDER)} RESTART IDENTITY CASCADE")
    )


def report(session: Session) -> dict[str, int]:
    models = {
        "patients": Patient,
        "users": User,
        "providers": Provider,
        "encounters": Encounter,
        "appointments": Appointment,
        "medications": Medication,
        "lab_results": LabResult,
        "clinical_documents": ClinicalDocument,
        "conditions": ConditionRow,
        "patient_conditions": PatientCondition,
        "diagnoses": Diagnosis,
        "procedures": Procedure,
        "allergies": Allergy,
    }
    return {
        label: session.scalar(select(func.count()).select_from(model)) or 0
        for label, model in models.items()
    }


#: Tables that must hold rows for a database to count as seeded, beyond the
#: patients themselves. Each arrived in a migration after the original
#: schema, which is exactly why a patient count cannot stand in for the
#: whole: a database seeded before 0008-0011 satisfies "has patients" and
#: projects an empty knowledge graph.
REQUIRED_TABLES: dict[str, type] = {
    "conditions": ConditionRow,
    "patient_conditions": PatientCondition,
    "diagnoses": Diagnosis,
    "procedures": Procedure,
    "allergies": Allergy,
}


def _unpopulated_tables(session: Session) -> list[str]:
    """Which required tables are empty. Empty list means fully seeded."""
    return [
        label
        for label, model in REQUIRED_TABLES.items()
        if not session.scalar(select(func.count()).select_from(model))
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--patients", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        default=date.today(),
        help="Anchor date for the generated timeline (default: today).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Truncate existing data first. Required to re-run on a "
        "populated database.",
    )
    parser.add_argument(
        "--if-empty",
        action="store_true",
        help="Seed only when the database has no patients, and succeed "
        "quietly when it does. For automated startup, where 'already "
        "seeded' is the expected steady state rather than an error.",
    )
    args = parser.parse_args(argv)

    if args.reset and args.if_empty:
        # --reset destroys data, --if-empty exists to avoid touching it.
        # Guessing which one was meant is not this script's call.
        print("--reset and --if-empty are mutually exclusive.", file=sys.stderr)
        return 2

    engine = create_engine(settings.database_url, future=True)
    with Session(engine) as session:
        existing = session.scalar(select(func.count()).select_from(Patient)) or 0
        if existing and args.if_empty:
            # "Has patients" is not the same as "is seeded", and treating
            # them as equivalent produced a container stack whose knowledge
            # graph answered nothing for weeks. A database seeded before
            # migrations 0008-0011 has its hundred patients and zero rows in
            # every table those migrations added — so this check passed, the
            # seed was skipped, the projection had no conditions to project,
            # and the KG route reported "the relationship graph holds
            # nothing matching that" on every question. Which reads as a data
            # problem, not as a startup step that never ran.
            empty = _unpopulated_tables(session)
            if empty:
                # Exit non-zero. The stack is genuinely misconfigured, and
                # coming up while pretending otherwise is the whole bug: the
                # failure then surfaces as wrong answers rather than as a
                # container that refused to start and said why.
                print(
                    f"Database has {existing} patients but no rows in: "
                    f"{', '.join(empty)}.\n"
                    "It was seeded before a migration that added these "
                    "tables, so it is partially populated — the knowledge "
                    "graph will project nothing and the KG route will answer "
                    "nothing.\n"
                    "Fix with:  python scripts/generate_data.py --reset",
                    file=sys.stderr,
                )
                return 1
            # Exit 0: the container that runs this on every `up` must not
            # fail the whole stack because the data it wanted is present.
            print(f"Database already has {existing} patients; nothing to do.")
            return 0
        if existing and not args.reset:
            print(
                f"Refusing to run: {existing} patients already exist. "
                "Re-run with --reset to replace them.",
                file=sys.stderr,
            )
            return 1
        if args.reset:
            reset(session)

        Generator(session, seed=args.seed, today=args.today).run(args.patients)
        session.commit()

        counts = report(session)

    engine.dispose()

    width = max(len(k) for k in counts)
    print("Synthetic dataset generated (all values fictional):")
    for label, count in counts.items():
        print(f"  {label:<{width}}  {count:>6}")
    print(
        f"\nDemo login: {DEMO_EXTERNAL_ID.lower()}@carecopilot.demo / {DEMO_PASSWORD}"
        "\nNext: python scripts/ingest_documents.py  (chunks + embeddings)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
