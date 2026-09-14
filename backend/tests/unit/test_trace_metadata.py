"""The trace → ChatMetadata boundary (PRD §26).

This file exists because of a silent failure. Text-to-SQL fields were added
to ``Trace.as_public_metadata`` but not to ``ChatMetadata``, and Pydantic's
default ``extra="ignore"`` dropped them without a word: no error, no log,
just a developer panel that never showed the generated SQL. The two shapes
have to be checked against each other, because nothing else will.
"""

from __future__ import annotations

from app.observability.trace import Trace
from app.schemas.chat import ChatMetadata
from app.services.chat import _public_metadata


def _trace() -> Trace:
    trace = Trace(request_id="req-1", user_id=1, patient_id=1)
    trace.route = "TEXT_TO_SQL"
    trace.model = "qwen3:8b"
    trace.provider = "ollama"
    trace.generated_sql = "SELECT COUNT(*) AS n FROM lab_results"
    trace.sql_row_count = 1
    trace.action = "book_appointment"
    trace.reranker = "llm"
    trace.retrieved_count = 20
    trace.reranked_count = 5
    trace.deduplicated = 6
    trace.record_usage(input_tokens=100, output_tokens=20, cost_usd=0.001)
    return trace


def test_every_public_trace_field_survives_into_chat_metadata() -> None:
    """The regression: a field added to one side and not the other vanishes."""
    published = _public_metadata(_trace())
    metadata = ChatMetadata(**published, guardrails=[])  # type: ignore[arg-type]

    missing = [
        key
        for key in published
        if key not in ChatMetadata.model_fields and key != "route"
    ]
    assert not missing, (
        f"Trace publishes {missing} but ChatMetadata has no field for them; "
        "Pydantic drops them silently."
    )
    assert metadata.generated_sql == "SELECT COUNT(*) AS n FROM lab_results"
    assert metadata.sql_row_count == 1
    assert metadata.action == "book_appointment"


def test_the_sql_statement_is_published_but_never_its_rows() -> None:
    """The panel explains how the figure was reached, not the figure again.

    Row *contents* on this channel would be a second copy of clinical data
    in a place that does not need one.
    """
    published = _public_metadata(_trace())
    assert "generated_sql" in published
    assert not any("row" in key and key != "sql_row_count" for key in published)


def test_no_reasoning_or_prompt_text_is_published() -> None:
    """PRD §26: mechanism may be shown, reasoning may not."""
    published = _public_metadata(_trace())
    forbidden = {"prompt", "system", "context", "reasoning", "thinking", "history"}
    assert not (forbidden & set(published))


# --- the §26 fields added in 0012 -------------------------------------- #


def test_the_model_asked_for_and_the_one_that_answered_are_both_kept() -> None:
    """§26 lists model and model version separately, and they differ.

    A trace that overwrote `model` with the resolved id would leave no row
    anywhere saying what the deployment was configured with.
    """
    trace = _trace()
    trace.model_version = "qwen3:8b-q4_K_M"

    published = _public_metadata(trace)
    metadata = ChatMetadata(**published, guardrails=[])  # type: ignore[arg-type]
    assert metadata.model == "qwen3:8b"
    assert metadata.model_version == "qwen3:8b-q4_K_M"

    row = trace.as_row()
    assert row.model == "qwen3:8b"
    assert row.model_version == "qwen3:8b-q4_K_M"


def test_a_refused_access_is_counted_even_though_the_turn_succeeded() -> None:
    """The whole reason the counter exists.

    A turn where the graph was refused still returns an answer and still
    writes a trace whose `error` may be unset — so without this field the
    row says nothing happened.
    """
    trace = _trace()
    trace.record_authorization_failure()
    trace.record_outcome(visited=3, validation_failures=1)

    assert trace.as_row().authorization_failures == 1
    assert _public_metadata(trace)["authorization_failures"] == 1


def test_counters_are_null_not_zero_when_nothing_happened() -> None:
    """Zero would read as 'measured, and clean' on a row that measured nothing."""
    row = _trace().as_row()
    assert row.authorization_failures is None
    assert row.validation_failures is None
    assert row.agent_iterations is None


def test_agent_iterations_counts_the_nodes_the_turn_executed() -> None:
    from app.services.chat import _record_outcome

    trace = _trace()
    _record_outcome(
        trace,
        {
            "visited": ["classify_query", "query_graph", "generate_answer"],
            "validation_errors": ["kg unavailable"],
            "guardrails": ["kg_unauthorized"],
        },
    )
    assert trace.agent_iterations == 3
    # One of each, summed: the step failed and the answer was unacceptable.
    assert trace.validation_failures == 2
