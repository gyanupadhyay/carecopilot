"""SQL validation (PRD §16).

The validator is the *fourth* layer, behind SELECT-only grants,
``default_transaction_read_only`` and row-level security. These tests assert
what it adds on top of those: a clear reason instead of a driver error, and
refusal of queries the role would happily run but the product should not.

Each rejection case is written as an attack or a mistake a model actually
makes, not as a synthetic string, so a passing test says something about
the system rather than about the regex.
"""

from __future__ import annotations

import pytest

from app.models.enums import LAB_TESTS
from app.sql.schema import (
    ALLOWED_COLUMNS,
    ALLOWED_TABLES,
    ENUMERATED_COLUMNS,
    SCHEMA_PROMPT,
)
from app.sql.validator import SQLValidationError, validate_sql


def reasons(sql: str) -> list[str]:
    with pytest.raises(SQLValidationError) as excinfo:
        validate_sql(sql)
    return excinfo.value.reasons


# --- statement type ------------------------------------------------------ #


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM lab_results",
        "UPDATE medications SET status = 'active'",
        "INSERT INTO appointments (patient_id) VALUES (2)",
        "DROP TABLE lab_results",
        "ALTER TABLE lab_results ADD COLUMN x int",
        "CREATE TABLE evil (id int)",
        "GRANT SELECT ON lab_results TO PUBLIC",
        "TRUNCATE lab_results",
    ],
)
def test_only_select_is_permitted(sql: str) -> None:
    assert "Only SELECT" in reasons(sql)[0]


def test_a_select_is_permitted() -> None:
    result = validate_sql("SELECT COUNT(*) AS n FROM lab_results")
    assert result.tables == {"lab_results"}


def test_a_cte_is_permitted() -> None:
    """WITH ... SELECT is a SELECT, and models reach for it on comparisons."""
    result = validate_sql(
        """
        WITH monthly AS (
            SELECT date_trunc('month', result_date) AS month, AVG(value) AS avg_value
              FROM lab_results
             WHERE test_name = 'Systolic Blood Pressure'
             GROUP BY 1
        )
        SELECT month, avg_value FROM monthly ORDER BY month
        """
    )
    assert "lab_results" in result.tables


# --- statement stacking -------------------------------------------------- #


def test_a_second_statement_is_rejected() -> None:
    found = reasons("SELECT 1 FROM lab_results; DROP TABLE lab_results")
    assert "exactly one statement" in found[0]


def test_a_semicolon_inside_a_string_is_not_a_statement_break() -> None:
    """Why this parses rather than splits on ';'.

    A regex validator either rejects this valid query or, worse, splits it
    and validates the fragments — which is the bypass that makes regex SQL
    validation unsafe.
    """
    result = validate_sql(
        "SELECT COUNT(*) AS n FROM medications WHERE dosage = '500 mg; oral'"
    )
    assert result.tables == {"medications"}


def test_a_comment_cannot_hide_a_second_statement() -> None:
    found = reasons("SELECT 1 FROM lab_results --\n; DELETE FROM lab_results")
    assert found


# --- table allowlist ----------------------------------------------------- #


@pytest.mark.parametrize(
    "table", ["patients", "users", "clinical_documents", "document_chunks", "audit_log"]
)
def test_tables_outside_the_allowlist_are_rejected(table: str) -> None:
    found = reasons(f"SELECT * FROM {table}")
    assert any("allowlist" in reason for reason in found)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM pg_catalog.pg_tables",
        "SELECT * FROM information_schema.columns",
        "SELECT * FROM pg_shadow",
        "SELECT * FROM pg_user",
    ],
)
def test_system_catalogues_are_rejected(sql: str) -> None:
    assert reasons(sql)


def test_a_join_onto_a_forbidden_table_is_rejected() -> None:
    """The allowlist applies to every table named, not just the first."""
    found = reasons(
        "SELECT l.value FROM lab_results l JOIN patients p ON p.id = l.patient_id"
    )
    assert any("patients" in reason for reason in found)


def test_a_subquery_onto_a_forbidden_table_is_rejected() -> None:
    found = reasons(
        "SELECT COUNT(*) AS n FROM lab_results "
        "WHERE patient_id IN (SELECT id FROM patients)"
    )
    assert any("patients" in reason for reason in found)


def test_the_allowlist_matches_the_column_map() -> None:
    """Two lists that must not drift: the tables and their column sets."""
    assert set(ALLOWED_COLUMNS) == set(ALLOWED_TABLES)


# --- columns ------------------------------------------------------------- #


def test_a_hallucinated_column_is_rejected_with_the_real_ones() -> None:
    found = reasons("SELECT bmi FROM lab_results")
    assert "bmi" in found[0]
    assert "test_name" in found[0], "the message should list what is available"


def test_aliases_the_query_introduces_are_accepted() -> None:
    result = validate_sql(
        "SELECT AVG(value) AS average_value FROM lab_results "
        "WHERE test_name = 'HbA1c'"
    )
    assert result.sql


def test_a_table_alias_is_accepted() -> None:
    result = validate_sql(
        "SELECT l.value FROM lab_results l WHERE l.test_name = 'HbA1c' LIMIT 10"
    )
    assert result.tables == {"lab_results"}


# --- forbidden functions ------------------------------------------------- #


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT current_setting('app.patient_id') FROM lab_results",
        "SELECT set_config('app.patient_id', '2', false) FROM lab_results",
        "SELECT pg_read_file('/etc/passwd') FROM lab_results",
        "SELECT pg_sleep(10) FROM lab_results",
    ],
)
def test_server_state_functions_are_rejected(sql: str) -> None:
    found = reasons(sql)
    assert any("not permitted" in reason for reason in found)


def test_set_config_cannot_be_used_to_rescope_the_session() -> None:
    """The attack this check exists for.

    RLS reads ``app.patient_id``. A query that could set it would move the
    boundary. It cannot — the role is read-only and ``set_config`` with
    is_local=false needs no write privilege but the statement is refused
    here first, and the transaction is read-only besides.
    """
    assert reasons("SELECT set_config('app.patient_id', '999', false)")


# --- patient scope ------------------------------------------------------- #


def test_filtering_on_patient_id_is_rejected() -> None:
    """Not a widening risk — a narrowing one. See the validator docstring."""
    found = reasons("SELECT COUNT(*) AS n FROM lab_results WHERE patient_id = 2")
    assert any("patient_id" in reason for reason in found)


def test_filtering_on_your_own_patient_id_is_also_rejected() -> None:
    """Even the 'correct' id is refused: the SQL is not what scopes the read."""
    assert reasons("SELECT COUNT(*) AS n FROM lab_results WHERE patient_id = 1")


def test_selecting_patient_id_is_allowed() -> None:
    """Only filtering is refused. Projecting it is harmless under RLS."""
    result = validate_sql("SELECT patient_id, value FROM lab_results LIMIT 5")
    assert result.sql


# --- enumerated values --------------------------------------------------- #
#
# The regression these guard is the nastiest one in the SQL path: valid SQL
# against a real column, matching a value that does not exist, returning 0
# rows, reported to the patient as a fact. Nothing errors.


def test_a_test_name_that_does_not_exist_is_rejected() -> None:
    """The exact failure seen live: 'Systolic BP' vs 'Systolic Blood Pressure'."""
    found = reasons(
        "SELECT COUNT(*) AS n FROM lab_results WHERE test_name = 'Systolic BP'"
    )
    assert any("Systolic BP" in reason for reason in found)
    assert any("Systolic Blood Pressure" in reason for reason in found), (
        "the message must name the real value, since the generator retries on it"
    )


def test_the_real_test_name_is_accepted() -> None:
    result = validate_sql(
        "SELECT COUNT(*) AS n FROM lab_results "
        "WHERE test_name = 'Systolic Blood Pressure' AND value > 140"
    )
    assert result.sql


def test_a_bad_value_inside_IN_is_rejected() -> None:
    assert reasons(
        "SELECT value FROM lab_results WHERE test_name IN ('HbA1c', 'Systolic BP')"
    )


def test_an_all_valid_IN_list_is_accepted() -> None:
    result = validate_sql(
        "SELECT value FROM lab_results "
        "WHERE test_name IN ('HbA1c', 'Creatinine') LIMIT 10"
    )
    assert result.sql


def test_ilike_stays_legal_for_loose_matching() -> None:
    """The escape hatch the generation rules point the model at."""
    result = validate_sql(
        "SELECT COUNT(*) AS n FROM lab_results WHERE test_name ILIKE '%systolic%'"
    )
    assert result.sql


@pytest.mark.parametrize("bad", ["walkin", "checkup", "Follow Up"])
def test_a_bad_appointment_type_is_rejected(bad: str) -> None:
    assert reasons(
        f"SELECT COUNT(*) AS n FROM appointments WHERE appointment_type = '{bad}'"
    )


def test_status_is_not_checked_because_two_tables_disagree() -> None:
    """medications.status and appointments.status have different vocabularies.

    Checking one would reject the other's legitimate values, which is worse
    than not checking at all.
    """
    assert "status" not in ENUMERATED_COLUMNS
    assert validate_sql(
        "SELECT COUNT(*) AS n FROM medications WHERE status = 'active'"
    ).sql
    assert validate_sql(
        "SELECT COUNT(*) AS n FROM appointments WHERE status = 'cancelled'"
    ).sql


def test_the_prompt_lists_every_real_test_name() -> None:
    """The prompt is generated from the constants, never transcribed.

    A hand-written example value is exactly what caused the failure above.
    """
    for name in LAB_TESTS:
        assert name in SCHEMA_PROMPT, name


def test_every_enumerated_column_is_a_real_column() -> None:
    all_columns: set[str] = set()
    for names in ALLOWED_COLUMNS.values():
        all_columns |= names
    assert set(ENUMERATED_COLUMNS) <= all_columns


# --- limits -------------------------------------------------------------- #


def test_a_limit_is_added_when_the_model_omits_one() -> None:
    result = validate_sql("SELECT value FROM lab_results", max_rows=200)
    assert result.limit_applied
    assert "LIMIT 200" in result.sql.upper()


def test_a_limit_that_is_too_high_is_tightened() -> None:
    result = validate_sql("SELECT value FROM lab_results LIMIT 100000", max_rows=200)
    assert result.limit_applied
    assert "100000" not in result.sql


def test_a_reasonable_limit_is_left_alone() -> None:
    result = validate_sql("SELECT value FROM lab_results LIMIT 10", max_rows=200)
    assert not result.limit_applied
    assert "LIMIT 10" in result.sql.upper()


def test_a_scalar_aggregate_needs_no_limit() -> None:
    """One row by construction; a LIMIT would be noise on the trace."""
    result = validate_sql("SELECT COUNT(*) AS n FROM lab_results")
    assert not result.limit_applied
    assert "LIMIT" not in result.sql.upper()


def test_a_grouped_aggregate_still_gets_a_limit() -> None:
    result = validate_sql(
        "SELECT test_name, COUNT(*) AS n FROM lab_results GROUP BY test_name"
    )
    assert result.limit_applied


def test_too_many_joins_are_rejected() -> None:
    sql = (
        "SELECT 1 FROM lab_results a "
        "JOIN encounters b ON b.id = a.encounter_id "
        "JOIN medications c ON c.encounter_id = b.id "
        "JOIN appointments d ON d.provider_id = b.provider_id "
        "JOIN encounters e ON e.id = a.encounter_id "
        "JOIN medications f ON f.encounter_id = e.id"
    )
    assert any("joins" in reason for reason in reasons(sql))


# --- malformed input ----------------------------------------------------- #


@pytest.mark.parametrize("sql", ["", "   ", ";", None])
def test_empty_input_is_rejected(sql: str | None) -> None:
    found = reasons(sql)  # type: ignore[arg-type]
    assert "no SQL" in found[0]


def test_unparseable_input_is_rejected() -> None:
    assert reasons("this is not sql at all, it is an apology")


def test_every_reason_is_reported_not_just_the_first() -> None:
    """The generator feeds these back for a retry; one at a time wastes calls."""
    found = reasons(
        "SELECT bmi FROM patients WHERE patient_id = 2"
    )
    assert len(found) >= 2
