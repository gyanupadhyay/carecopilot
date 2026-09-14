"""The schema generated SQL is allowed to touch (PRD §16).

One list, used three ways: to describe the tables to the model, to validate
what comes back, and to document what the analytics role is granted. Keeping
those in one file means widening the allowlist is a single reviewable change
rather than three edits that can drift apart — and drift here is a privilege
escalation, not a formatting bug.

``patient_id`` is deliberately described to the model as a column it must
never filter on. Not because the filter would be wrong, but because the
model's SQL is not what enforces scope: row-level security on the analytics
role does (see ``db/session.py``). A generated ``WHERE patient_id = 2``
returns nothing rather than another patient's rows, and telling the model to
leave it alone keeps the generated SQL honest about where the boundary
lives.
"""

from __future__ import annotations

from typing import Final

from app.models.enums import (
    APPOINTMENT_STATUSES,
    APPOINTMENT_TYPES,
    ENCOUNTER_TYPES,
    LAB_TESTS,
    MEDICATION_STATUSES,
)

#: The only tables generated SQL may name. Mirrored by the SELECT grants in
#: migration 0004 and by the RLS policies in 0001.
ALLOWED_TABLES: Final[frozenset[str]] = frozenset(
    {"lab_results", "appointments", "medications", "encounters"}
)

#: Column allowlist per table. A generated query naming a column that is not
#: here is rejected — which catches a hallucinated column with a clear error
#: instead of a database-level "column does not exist".
ALLOWED_COLUMNS: Final[dict[str, frozenset[str]]] = {
    "lab_results": frozenset(
        {
            "id",
            "patient_id",
            "encounter_id",
            "test_name",
            "value",
            "unit",
            "reference_range",
            "result_date",
        }
    ),
    "appointments": frozenset(
        {
            "id",
            "patient_id",
            "provider_id",
            "appointment_date",
            "appointment_type",
            "status",
            "notes",
        }
    ),
    "medications": frozenset(
        {
            "id",
            "patient_id",
            "encounter_id",
            "name",
            "dosage",
            "frequency",
            "status",
            "start_date",
            "end_date",
        }
    ),
    "encounters": frozenset(
        {
            "id",
            "patient_id",
            "provider_id",
            "encounter_date",
            "encounter_type",
            "reason",
        }
    ),
}

#: Columns whose values come from a controlled vocabulary. Two uses: the
#: model is shown the exact permitted strings, and the validator rejects an
#: equality test against a value outside the list.
#:
#: This mapping is the fix for a real failure. The schema prompt originally
#: carried hand-written example values ("Systolic BP"), the stored value is
#: "Systolic Blood Pressure", and the model faithfully filtered on the
#: example. The SQL was valid, ran without error, returned zero rows, and
#: the assistant reported "that is not in your records" — a wrong answer
#: with no error anywhere to notice. Deriving from the same constants the
#: generator and the CHECK constraints use makes that drift impossible.
ENUMERATED_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "test_name": LAB_TESTS,
    "appointment_type": APPOINTMENT_TYPES,
    "encounter_type": ENCOUNTER_TYPES,
}

#: ``status`` exists on two tables with different vocabularies, so it is
#: kept out of the validator's equality check — rejecting 'active' because
#: it is not an appointment status would be worse than allowing both.
STATUS_VOCABULARIES: Final[dict[str, tuple[str, ...]]] = {
    "appointments": APPOINTMENT_STATUSES,
    "medications": MEDICATION_STATUSES,
}


def _values(names: tuple[str, ...]) -> str:
    return ", ".join(f"'{name}'" for name in names)


#: Handed to the model verbatim. Written as DDL rather than prose because a
#: model generating SQL reads DDL more reliably than a description of it,
#: and every vocabulary is interpolated from the constants rather than
#: transcribed — see ENUMERATED_COLUMNS above for what transcribing cost.
SCHEMA_PROMPT: Final[str] = f"""
lab_results       -- one numeric result per row; blood pressure is two rows
                  -- ('Systolic Blood Pressure', 'Diastolic Blood Pressure'),
                  -- never one combined reading
  id              integer
  patient_id      integer      -- DO NOT filter on this; see the rules below
  encounter_id    integer      -- nullable; a result may not belong to a visit
  value           numeric
  unit            text
  reference_range text
  result_date     date
  test_name       text         -- EXACTLY one of:
                  --   {_values(LAB_TESTS)}

appointments
  id               integer
  patient_id       integer     -- DO NOT filter on this
  provider_id      integer
  appointment_date timestamp
  notes            text
  status           text        -- {_values(APPOINTMENT_STATUSES)}
  appointment_type text        -- {_values(APPOINTMENT_TYPES)}

medications
  id           integer
  patient_id   integer         -- DO NOT filter on this
  encounter_id integer         -- nullable
  name         text            -- free text drug name, e.g. 'Metformin'
  dosage       text            -- free text, e.g. '500 mg'; NOT numeric
  frequency    text
  start_date   date
  end_date     date            -- NULL while the medication is ongoing
  status       text            -- {_values(MEDICATION_STATUSES)}

encounters
  id             integer
  patient_id     integer       -- DO NOT filter on this
  provider_id    integer
  encounter_date date
  reason         text
  encounter_type text          -- {_values(ENCOUNTER_TYPES)}
""".strip()

GENERATION_RULES: Final[str] = """
Rules, all enforced — a query that breaks one is rejected, not repaired:

1. Exactly one statement. A SELECT or a WITH ... SELECT. Never INSERT,
   UPDATE, DELETE, DROP, ALTER, CREATE, GRANT, COPY or a transaction command.
2. Only these four tables. No other table, no system catalogue, no
   information_schema, no subquery onto anything outside the list.
3. Never write a patient_id filter, and never reference another patient.
   The connection is already restricted to exactly one patient by the
   database itself, so a patient_id predicate is at best redundant. Write
   the query as though the tables contained only this patient's rows.
4. No semicolons inside the statement, no comments, no set-returning or
   system functions (pg_*, current_setting, set_config, dblink, lo_*).
5. Aggregate when the question asks for a count, average, maximum or a
   comparison. Return the smallest result that answers it — a scalar or a
   handful of rows, not a dump.
6. Give every computed column an explicit, readable alias.
7. Dates: result_date, encounter_date and start_date are DATE;
   appointment_date is TIMESTAMP. Use CURRENT_DATE for "now" and interval
   arithmetic for "in the last N months".
8. dosage is free text ('500 mg'), not a number. Do not do arithmetic on it.
9. For a column with a listed vocabulary, use one of the listed strings
   exactly. A value that is not on the list matches no rows, which produces
   a confident wrong answer rather than an error. If you want to match
   loosely, use ILIKE with a wildcard instead of an equality test.
""".strip()
