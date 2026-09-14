"""Run the evaluation set against the real system (PRD §27).

Executes each case end to end — identity mapping, router, graph, retrieval,
generation, guardrails — and scores the result. Nothing is mocked: a number
produced here reflects what a request would actually do.

Three decisions worth stating.

*Ground truth is resolved, not stored.* Cases describe relevance as a
matcher over document metadata (``title_prefix``, ``sections``), and this
module turns that into chunk ids at run time. Storing ids would mean the
dataset silently decays the next time ``generate_data.py --reset`` renumbers
the corpus — and a decayed eval reports failures that are really staleness.

*Model-dependent cases are skipped, not guessed.* Without an API key the
stub answers, so answer correctness and the model's routing decisions cannot
be judged. Those cases are marked ``skipped`` with a reason and excluded
from their metrics, rather than scored against placeholder text.

*Every case runs under its own patient's authenticated context.* Security
cases are not simulated — case ``safety-001`` really does ask P001's session
for P002's records, and the result is whatever the system really returns.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.context import AuthContext
from app.knowledge_graph.queries import CYPHER, GraphIntent
from app.llm.base import LLMProvider
from app.models import ClinicalDocument, DocumentChunk, Patient, User, UserPatientMapping
from app.observability.logging import get_logger
from app.rag.context import render_source
from app.rag.embeddings import EmbeddingProvider
from app.rag.retrieval import RetrievedChunk
from app.services import chat as chat_service
from app.sql.schema import ALLOWED_TABLES
from evaluation import metrics as M
from evaluation.judge import judge_faithfulness

log = get_logger(__name__)

DATASET_PATH = Path(__file__).with_name("dataset.json")
REPORTS_DIR = Path(__file__).with_name("reports")


@dataclass(slots=True)
class CaseResult:
    case_id: str
    category: str
    patient: str
    question: str

    route: str | None = None
    expected_route: str | None = None
    tools_used: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)

    retrieved: list[int] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    relevant: set[int] = field(default_factory=set)
    expects_sources: bool = False

    answer: str = ""
    latency_ms: float = 0.0
    guardrails: list[str] = field(default_factory=list)

    #: ``{name, ms, ok}`` per invocation. Selection accuracy reads the names;
    #: the success rate reads ``ok``, and the two fail independently.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    #: Schema-constrained model calls on this turn and how many failed to
    #: parse. Both zero for a turn that asked the model for no JSON.
    structured_calls: int = 0
    structured_failures: int = 0

    #: Text-to-SQL: the validated statement and what it returned.
    generated_sql: str = ""
    sql_row_count: int | None = None

    #: Faithfulness, filled in by the judge pass after the run.
    faithfulness: float | None = None
    faithfulness_skipped: str = ""
    unsupported_claims: list[str] = field(default_factory=list)

    skipped: str = ""
    error: str = ""

    # --- derived ---------------------------------------------------- #

    @property
    def ran(self) -> bool:
        return not self.skipped and not self.error

    @property
    def degraded(self) -> bool:
        """The provider was unreachable, so no model decided anything here.

        The request still succeeded — the system degraded honestly and told
        the user so — which is why this is not an ``error``. But the answer
        is a fixed apology string and the route is a fallback, so scoring
        either one measures the degradation path, not the system. A free
        tier returning 429 would otherwise show up as a *safe* refusal and a
        *correct* apology, inflating exactly the metrics that matter most.
        """
        return "llm_unavailable" in self.guardrails

    @property
    def judgeable(self) -> bool:
        """Ran, and a model actually participated."""
        return self.ran and not self.degraded

    @property
    def route_correct(self) -> bool | None:
        if not self.expected_route or not self.route or self.degraded:
            return None
        return self.route == self.expected_route

    @property
    def kg_intent(self) -> str | None:
        """The traversal the model chose, from the recorded tool name.

        The name is ``kg:<intent>`` and the intent is the whole of what the
        model decided — everything else about the traversal is a reviewed
        template — so this is the handle the KG metrics score against.
        """
        for call in self.tool_calls:
            name = str(call.get("name", ""))
            if name.startswith("kg:"):
                return name.removeprefix("kg:")
        return None

    @property
    def kg_matched(self) -> bool | None:
        """Did the traversal resolve the question's term to real entities?

        ``None`` when no traversal ran at all — a KG question misrouted to
        RAG has no resolution to score, and calling that a resolution failure
        would double-count the routing failure that caused it.
        """
        if self.kg_intent is None:
            return None
        return "kg_no_match" not in self.guardrails

    @property
    def tools_correct(self) -> bool | None:
        """Every expected tool ran — recall over the selection."""
        if not self.expected_tools:
            return None
        return set(self.expected_tools).issubset(set(self.tools_used))

    @property
    def tool_precision(self) -> float | None:
        """Share of the tools run that the case actually needed.

        Recall alone cannot fail a model that runs everything, and until the
        API route selected its own tools that is exactly what it did — a
        fixed three-tool plan scored 1.000 recall on every question by
        construction. Now that §5's selection is the model's, precision is
        the half that can fall, and the failure it catches is the cheap one:
        answering "when is my next appointment?" by also fetching every
        medication and every lab result.

        Requires ``expected_tools`` to be the complete set for the question,
        which the dataset's field note states.
        """
        if not self.expected_tools or not self.tools_used:
            return None
        used = set(self.tools_used)
        return len(used & set(self.expected_tools)) / len(used)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "category": self.category,
            "patient": self.patient,
            "question": self.question,
            "route": self.route,
            "expected_route": self.expected_route,
            "route_correct": self.route_correct,
            "tools_used": self.tools_used,
            "tools_correct": self.tools_correct,
            "retrieved": len(self.retrieved),
            "cited": len(self.cited),
            "relevant": len(self.relevant),
            "latency_ms": round(self.latency_ms, 1),
            "guardrails": self.guardrails,
            "degraded": self.degraded,
            "kg_intent": self.kg_intent,
            "kg_matched": self.kg_matched,
            "generated_sql": self.generated_sql or None,
            "sql_row_count": self.sql_row_count,
            "structured_calls": self.structured_calls,
            "structured_failures": self.structured_failures,
            "faithfulness": self.faithfulness,
            "faithfulness_skipped": self.faithfulness_skipped,
            # Named, not counted. "0.75 faithful" tells a reader to worry;
            # the sentence that was not supported tells them what to fix.
            "unsupported_claims": self.unsupported_claims,
            "skipped": self.skipped,
            "error": self.error,
            "answer_preview": self.answer[:160],
        }


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    data = json.loads((path or DATASET_PATH).read_text(encoding="utf-8"))
    return data["cases"]


async def _auth_context(session: AsyncSession, external_id: str) -> AuthContext | None:
    """Build the real AuthContext for a patient, via the identity mapping."""
    patient = await session.scalar(
        select(Patient).where(Patient.external_id == external_id)
    )
    if patient is None:
        return None
    user = await session.scalar(
        select(User)
        .join(UserPatientMapping, UserPatientMapping.user_id == User.id)
        .where(
            UserPatientMapping.patient_id == patient.id,
            UserPatientMapping.is_active.is_(True),
        )
    )
    if user is None:
        return None
    return AuthContext(
        user_id=user.id,
        role=user.role,  # type: ignore[arg-type]
        patient_id=patient.id,
        # Unique per case: request_id is the trace primary key, and reusing
        # one per patient made every case after the first fail to persist.
        request_id=f"eval-{external_id}-{uuid4().hex[:8]}",
    )


async def resolve_relevant(
    session: AsyncSession, ctx: AuthContext, matcher: dict[str, Any] | None
) -> set[int]:
    """Turn a relevance matcher into the chunk ids it describes.

    Scoped to the case's own patient, so a matcher can never accidentally
    define another patient's chunks as the right answer.
    """
    if not matcher:
        return set()

    stmt = (
        select(DocumentChunk.id)
        .join(ClinicalDocument, ClinicalDocument.id == DocumentChunk.document_id)
        .where(DocumentChunk.patient_id == ctx.patient_scope)
    )
    title_prefix = matcher.get("title_prefix")
    if title_prefix:
        stmt = stmt.where(ClinicalDocument.title.like(f"{title_prefix}%"))
    sections = matcher.get("sections")
    if sections:
        stmt = stmt.where(DocumentChunk.section.in_(sections))

    return set((await session.scalars(stmt)).all())


async def run_case(
    session: AsyncSession,
    case: dict[str, Any],
    *,
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    has_real_llm: bool,
) -> CaseResult:
    result = CaseResult(
        case_id=case["id"],
        category=case["category"],
        patient=case["patient"],
        question=case["question"],
        expected_route=case.get("expected_route"),
        expected_tools=list(case.get("expected_tools") or []),
    )

    if case.get("requires_llm") and not has_real_llm:
        result.skipped = "requires a real LLM (LLM_API_KEY is not set)"
        return result

    ctx = await _auth_context(session, case["patient"])
    if ctx is None:
        result.skipped = f"patient {case['patient']} is not seeded"
        return result

    try:
        result.relevant = await resolve_relevant(session, ctx, case.get("relevant"))
    except Exception as exc:  # pragma: no cover - matcher is dataset-controlled
        result.error = f"ground truth: {type(exc).__name__}"
        return result

    started = time.perf_counter()
    try:
        turn = await chat_service.answer_question(
            session,
            ctx,
            question=case["question"],
            conversation_id=None,
            llm=llm,
            embedder=embedder,
        )
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"[:200]
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    result.latency_ms = (time.perf_counter() - started) * 1000
    response = turn.response
    result.route = response.route
    result.answer = response.answer
    result.guardrails = list(response.metadata.guardrails)
    result.tools_used = list(response.metadata.tools_used)
    result.tool_calls = [dict(call) for call in response.metadata.tool_calls]
    result.structured_calls = response.metadata.structured_calls or 0
    result.structured_failures = response.metadata.structured_failures or 0
    result.generated_sql = response.metadata.generated_sql or ""
    result.sql_row_count = response.metadata.sql_row_count
    result.cited = [s.chunk_id for s in response.sources if s.chunk_id is not None]
    result.expects_sources = bool(result.relevant)

    # The trace reports how many candidates retrieval considered; the cited
    # ids are what actually reached the answer. Recall is scored over what
    # was put in front of the model, which is the citation list.
    result.retrieved = result.cited
    return result


# --- faithfulness ----------------------------------------------------------- #


async def judge_run(
    session: AsyncSession,
    results: Sequence[CaseResult],
    *,
    llm: LLMProvider,
    judge_model: str | None = None,
) -> None:
    """Score faithfulness for every case that cited something. In place.

    A second pass rather than part of ``run_case``, for two reasons. The
    judge is a model call per case and would double an already hour-long run
    when nobody asked for it, so it is opt-in (``--judge``). And judging
    inside the case would put the judge's latency into the case's
    ``latency_ms``, making the system look half as fast as it is.

    Only cases with citations are judged: faithfulness is "supported by the
    passages", and a case with no passages has no grounding to be faithful
    to. An API answer built from structured rows is covered by answer
    correctness instead.
    """
    for result in results:
        if not result.judgeable:
            continue
        if not result.cited:
            result.faithfulness_skipped = "no cited passages to judge against"
            continue
        passages = await _chunk_texts(session, result.cited)
        if not passages:
            result.faithfulness_skipped = "cited chunks no longer resolve"
            continue
        verdict = await judge_faithfulness(
            llm, answer=result.answer, passages=passages, model=judge_model
        )
        result.faithfulness = verdict.score
        result.faithfulness_skipped = verdict.skipped
        result.unsupported_claims = verdict.unsupported
        log.info(
            "eval.judged",
            case=result.case_id,
            score=verdict.score,
            claims=verdict.claims_checked,
        )


async def _chunk_texts(session: AsyncSession, chunk_ids: Sequence[int]) -> list[str]:
    """The passages an answer cited, rendered exactly as the model saw them.

    The header is not decoration, and leaving it off is not a small
    inaccuracy — it changes what the judge is measuring. The model reads
    ``Date:``, ``Document:`` and ``Section:`` above each passage and cites
    from them, so a judge given the bare ``chunk_text`` marks every correct
    date attribution as unsupported. Measured: faithfulness read 0.449 that
    way, and all eleven "fabrications" were dates the note genuinely carried
    in its metadata.

    That is the failure this suite exists to prevent, arriving from the
    scoring side — a number that is arithmetically right and says something
    false about the system. Rendering goes through ``context._render`` rather
    than a local copy of the format, so the judge cannot drift out of step
    with the prompt again.
    """
    if not chunk_ids:
        return []
    rows = (
        await session.execute(
            select(
                DocumentChunk.id,
                DocumentChunk.chunk_text,
                DocumentChunk.section,
                DocumentChunk.chunk_date,
                DocumentChunk.document_id,
                ClinicalDocument.title,
                ClinicalDocument.document_type,
            )
            .join(ClinicalDocument, ClinicalDocument.id == DocumentChunk.document_id)
            .where(DocumentChunk.id.in_(list(chunk_ids)))
        )
    ).all()

    by_id = {
        row.id: RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            encounter_id=None,
            section=row.section,
            text=row.chunk_text,
            chunk_date=row.chunk_date,
            title=row.title,
            document_type=row.document_type,
            distance=0.0,
        )
        for row in rows
    }
    return [
        render_source(index, by_id[cid])
        for index, cid in enumerate(chunk_ids, start=1)
        if cid in by_id
    ]


# --- scoring --------------------------------------------------------------- #


def score(
    results: Sequence[CaseResult],
    *,
    top_k: int = 5,
    has_real_llm: bool = True,
) -> list[M.MetricGroup]:
    ran = [r for r in results if r.ran]
    skipped = [r for r in results if r.skipped]
    errored = [r for r in results if r.error]
    degraded = [r for r in ran if r.degraded]
    # Everything downstream of the model is scored over this, not over
    # ``ran``. See CaseResult.degraded for why.
    judgeable = [r for r in ran if r.judgeable]

    groups: list[M.MetricGroup] = []

    # Routing ------------------------------------------------------------- #
    routing = M.MetricGroup("routing")
    judged = [r for r in ran if r.route_correct is not None]
    if judged:
        routing.add(
            "router_accuracy",
            M.accuracy(
                [r.route or "" for r in judged],
                [r.expected_route or "" for r in judged],
            ),
        )
        routing.values["cases_judged"] = len(judged)
    else:
        routing.add(
            "router_accuracy",
            None,
            skip_reason="no case had both an expected route and a live model",
        )
    routing.values["confusion"] = M.confusion(  # type: ignore[assignment]
        [r.route or "" for r in judged], [r.expected_route or "" for r in judged]
    )
    groups.append(routing)

    # Retrieval ----------------------------------------------------------- #
    retrieval = M.MetricGroup("retrieval")
    with_truth = [r for r in ran if r.relevant]
    if with_truth:
        retrieval.add(
            f"recall_at_{top_k}",
            M.mean(
                [M.recall_at_k(r.retrieved, r.relevant, top_k) for r in with_truth]
            ),
        )
        retrieval.add(
            f"precision_at_{top_k}",
            M.mean(
                [M.precision_at_k(r.retrieved, r.relevant, top_k) for r in with_truth]
            ),
        )
        retrieval.add(
            "mrr",
            M.mean([M.reciprocal_rank(r.retrieved, r.relevant) for r in with_truth]),
        )
        retrieval.values["cases_judged"] = len(with_truth)
    else:
        reason = "no case with retrieval ground truth ran"
        for key in (f"recall_at_{top_k}", f"precision_at_{top_k}", "mrr"):
            retrieval.add(key, None, skip_reason=reason)
    groups.append(retrieval)

    # Generation ---------------------------------------------------------- #
    generation = M.MetricGroup("generation")
    # Scoring stub output against required facts produces a real number from
    # text no model wrote. 0.0 there would read as "the model omitted every
    # fact" when the truth is "no model ran" — the exact confusion this
    # module exists to prevent. Skip, don't score.
    if not has_real_llm:
        generation.add(
            "answer_correctness",
            None,
            skip_reason="stub provider — no generated text to score",
        )
    else:
        scored_answers = [
            M.answer_correctness(r.answer, _must_mention(results, r))
            for r in judgeable
        ]
        correctness = M.mean(scored_answers)
        if correctness is None:
            generation.add(
                "answer_correctness",
                None,
                skip_reason="no case ran that declares required facts",
            )
        else:
            generation.add("answer_correctness", correctness)

    generation.add(
        "citation_correctness",
        M.mean(
            [M.citation_correctness(r.cited, r.retrieved, r.relevant) for r in with_truth]
        ),
        skip_reason="no cited answers with ground truth",
    )
    generation.add(
        "grounding_rate",
        M.grounding_rate(judgeable),
        skip_reason="no record-backed answers",
    )
    # Judged by an LLM, and only when --judge asked for it. The three states
    # are kept distinct on purpose: not run, run but nothing judgeable, and a
    # real score. Collapsing the first two would report a pass for a check
    # that never happened.
    judged_faith = [r for r in judgeable if r.faithfulness is not None]
    if judged_faith:
        generation.add(
            "faithfulness", M.mean([r.faithfulness for r in judged_faith])
        )
        generation.values["faithfulness_cases_judged"] = len(judged_faith)
        # A floor, not an estimate: the judge is the same local model that
        # wrote the answers unless --judge-model said otherwise, and a model
        # does not flag the hallucinations it finds plausible.
        generation.values["faithfulness_is_self_judged"] = True  # type: ignore[assignment]
    else:
        attempted = [r for r in judgeable if r.faithfulness_skipped]
        generation.add(
            "faithfulness",
            None,
            skip_reason=(
                f"judge ran, nothing judgeable ({attempted[0].faithfulness_skipped})"
                if attempted
                else "not run; pass --judge to score faithfulness"
            ),
        )
    groups.append(generation)

    # Knowledge graph ------------------------------------------------------ #
    # Scored over cases that actually reached a traversal, not over every
    # case labelled KG. A KG question misrouted to RAG is a routing failure
    # and is already counted as one above; counting it again here would make
    # one mistake look like two.
    kg = M.MetricGroup("knowledge_graph")
    kg_cases = [r for r in judgeable if r.kg_intent is not None]
    if kg_cases:
        kg.add(
            "entity_resolution_rate",
            M.rate([M.entity_resolution(r.kg_matched) for r in kg_cases]),
            skip_reason="no traversal ran",
        )
        kg.add(
            "relationship_correctness",
            M.rate(
                [
                    M.relationship_correctness(
                        _cypher_for(r.kg_intent), _case_list(r, "expected_relationships")
                    )
                    for r in kg_cases
                ]
            ),
            skip_reason="no KG case declares expected relationships",
        )
        kg.add(
            "multi_hop_correctness",
            M.rate(
                [
                    M.multi_hop_correctness(
                        _cypher_for(r.kg_intent), _case_int(r, "expected_hops")
                    )
                    for r in kg_cases
                ]
            ),
            skip_reason="no KG case declares a multi-hop expectation",
        )
        kg.values["cases_judged"] = len(kg_cases)
    else:
        reason = "no case reached a graph traversal"
        for key in (
            "entity_resolution_rate",
            "relationship_correctness",
            "multi_hop_correctness",
        ):
            kg.add(key, None, skip_reason=reason)
    groups.append(kg)

    # Text-to-SQL ---------------------------------------------------------- #
    sql = M.MetricGroup("text_to_sql")
    sql_cases = [r for r in judgeable if r.route == "TEXT_TO_SQL"]
    if sql_cases:
        # Validity is scored over every case that reached the route, not over
        # the ones that produced SQL — otherwise the denominator excludes
        # exactly the failures being measured and the rate is always 1.000.
        sql.add(
            "sql_validity",
            M.rate(
                [
                    bool(r.generated_sql) and M.sql_parses(r.generated_sql)
                    for r in sql_cases
                    # A question the schema genuinely cannot answer is the
                    # generator refusing, which is correct behaviour and not
                    # an invalid statement.
                    if "sql_declined" not in r.guardrails
                ]
            ),
            skip_reason="every SQL case was declined as unanswerable",
        )
        sql.add(
            "execution_success",
            M.rate(
                [
                    r.sql_row_count is not None
                    for r in sql_cases
                    if "sql_declined" not in r.guardrails
                ]
            ),
            skip_reason="no statement reached execution",
        )
        sql.add(
            "query_correctness",
            M.rate(
                [
                    M.sql_query_correctness(
                        r.generated_sql,
                        _case_list(r, "expected_sql_tables"),
                        _case_list(r, "expected_sql_functions"),
                    )
                    for r in sql_cases
                ]
            ),
            skip_reason="no SQL case declares expected tables or functions",
        )
        sql.add(
            "authorization_correctness",
            M.rate(
                [
                    M.sql_authorization_correctness(r.generated_sql, ALLOWED_TABLES)
                    for r in sql_cases
                ]
            ),
            skip_reason="no statement to check for scope violations",
        )
        sql.values["cases_judged"] = len(sql_cases)
        sql.values["statements_generated"] = sum(
            1 for r in sql_cases if r.generated_sql
        )
    else:
        reason = "no case reached the Text-to-SQL route"
        for key in (
            "sql_validity",
            "execution_success",
            "query_correctness",
            "authorization_correctness",
        ):
            sql.add(key, None, skip_reason=reason)
    groups.append(sql)

    # Agent ---------------------------------------------------------------- #
    agent = M.MetricGroup("agent")
    tool_judged = [r for r in judgeable if r.tools_correct is not None]
    if tool_judged:
        agent.add(
            "tool_selection_accuracy",
            sum(1 for r in tool_judged if r.tools_correct) / len(tool_judged),
        )
        agent.values["cases_judged"] = len(tool_judged)
        agent.values["mean_tool_calls"] = round(
            sum(len(r.tools_used) for r in judgeable) / max(1, len(judgeable)), 2
        )
        agent.add(
            "tool_selection_precision",
            M.mean([r.tool_precision for r in tool_judged]),
            skip_reason="no case ran a tool it could be judged on",
        )
    else:
        agent.add("tool_selection_accuracy", None, skip_reason="no case expected tools")

    # Selecting the right tool and having it return something are separate
    # failures. A run where selection is 1.000 and success is 0.400 is badly
    # broken, and the selection number alone calls it healthy.
    agent.add(
        "tool_call_success_rate",
        M.tool_call_success_rate(
            [call for r in judgeable for call in r.tool_calls]
        ),
        skip_reason="no tool was invoked",
    )
    # Only the routes where the *model* supplies arguments. The API route
    # runs a fixed plan of no-argument lookups, so including it would score
    # a decision the model never made — and would drown the two routes where
    # the argument is the whole of what the model decided.
    arg_cases = [r for r in judgeable if r.route in ("KG", "ACTION")]
    agent.add(
        "tool_argument_validity",
        M.rate(
            [
                "kg_incomplete" not in r.guardrails
                and "llm_unavailable" not in r.guardrails
                for r in arg_cases
            ]
        ),
        skip_reason="no case reached a route where the model supplies arguments",
    )
    agent.add(
        "json_validity",
        M.mean(
            [
                M.json_validity(r.structured_calls, r.structured_failures)
                for r in judgeable
            ]
        ),
        skip_reason="no turn asked the model for schema-constrained JSON",
    )
    agent.values["structured_calls_total"] = sum(r.structured_calls for r in judgeable)
    groups.append(agent)

    # Security -------------------------------------------------------------- #
    security = M.MetricGroup("security")
    # Degraded cases are excluded here for a sharper reason than elsewhere:
    # "I could not reach the assistant service" contains no forbidden string,
    # so an outage would score as a perfect safety record. A safety metric
    # that improves when the system stops working is worse than no metric.
    safety = [r for r in judgeable if r.category == "safety"]
    if safety:
        held = [
            M.contains_none_of(r.answer, _must_not_mention(results, r)) for r in safety
        ]
        checked = [h for h in held if h is not None]
        if checked:
            security.add("leak_free_rate", sum(1 for h in checked if h) / len(checked))
            security.values["cases_judged"] = len(checked)
        else:
            security.add(
                "leak_free_rate",
                None,
                skip_reason="no safety case declares forbidden content",
            )
    else:
        security.add(
            "leak_free_rate",
            None,
            skip_reason="no safety case reached a live model",
        )
    groups.append(security)

    # System ---------------------------------------------------------------- #
    system = M.MetricGroup("system")
    summary = M.latency([r.latency_ms for r in ran])
    system.values.update(
        {
            "cases_total": len(results),
            "cases_ran": len(ran),
            "cases_skipped": len(skipped),
            "cases_errored": len(errored),
            # Not an error — the request succeeded and degraded honestly —
            # but the model did not participate, so these are excluded from
            # every model-dependent metric above. A high number here means
            # the run is thin, not that the system is broken.
            "cases_degraded": len(degraded),
            "latency_mean_ms": round(summary.mean_ms, 1) if summary.mean_ms else None,
            "latency_p50_ms": summary.p50_ms,
            "latency_p95_ms": summary.p95_ms,
        }
    )
    system.add("error_rate", M.error_rate(len(errored), len(results)))
    groups.append(system)

    return groups


def _must_mention(results: Sequence[CaseResult], result: CaseResult) -> list[str]:
    return _case_field(results, result, "answer_should_mention")


def _must_not_mention(results: Sequence[CaseResult], result: CaseResult) -> list[str]:
    return _case_field(results, result, "answer_should_not_mention")


#: Populated by ``run_dataset`` so scoring can reach the original case.
_CASE_INDEX: dict[str, dict[str, Any]] = {}


def _case_field(
    _results: Sequence[CaseResult], result: CaseResult, key: str
) -> list[str]:
    case = _CASE_INDEX.get(result.case_id, {})
    return list(case.get(key) or [])


def _case_list(result: CaseResult, key: str) -> list[str]:
    return list(_CASE_INDEX.get(result.case_id, {}).get(key) or [])


def _case_int(result: CaseResult, key: str) -> int | None:
    value = _CASE_INDEX.get(result.case_id, {}).get(key)
    return int(value) if value is not None else None


def _cypher_for(intent: str | None) -> str:
    """The reviewed template behind a chosen intent, or empty for an unknown.

    Empty rather than raising: an intent the enum does not hold means the
    model invented a label, which is a tool-argument failure already counted
    as one — and a KeyError here would abort a run over it.
    """
    if not intent:
        return ""
    try:
        return CYPHER[GraphIntent(intent)]
    except (ValueError, KeyError):
        return ""


async def run_dataset(
    session: AsyncSession,
    *,
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    has_real_llm: bool,
    cases: list[dict[str, Any]] | None = None,
    only: str | None = None,
    delay_seconds: float = 0.0,
    judge: bool = False,
    judge_model: str | None = None,
) -> tuple[list[CaseResult], list[M.MetricGroup]]:
    selected = cases if cases is not None else load_cases()
    if only:
        selected = [c for c in selected if c["category"] == only or c["id"] == only]

    _CASE_INDEX.clear()
    _CASE_INDEX.update({c["id"]: c for c in selected})

    results: list[CaseResult] = []
    for index, case in enumerate(selected):
        # Paced, not parallel. Free tiers rate-limit per minute, and a 429
        # degrades the case into a non-measurement — so going faster than
        # the quota buys nothing but a thinner report.
        if delay_seconds and index:
            await asyncio.sleep(delay_seconds)
        result = await run_case(
            session, case, llm=llm, embedder=embedder, has_real_llm=has_real_llm
        )
        results.append(result)
        log.info(
            "eval.case",
            case=result.case_id,
            route=result.route,
            skipped=bool(result.skipped),
            error=bool(result.error),
        )

    if judge and has_real_llm:
        # After every case, not interleaved: the judge's latency would
        # otherwise land in the case's own latency_ms and halve the reported
        # speed of a system that did not slow down.
        #
        # And never fatally. The judge is an optional extra pass that runs
        # once every case has already been executed and scored, so a fault
        # in it must cost the faithfulness column and nothing else — the
        # first version of this raised an AttributeError on a misnamed
        # column and discarded a complete 55-case run at the last step,
        # which is the same way a UnicodeEncodeError in the summary once
        # destroyed an hour of local inference.
        try:
            await judge_run(session, results, llm=llm, judge_model=judge_model)
        except Exception as exc:
            log.warning("eval.judge_pass_failed", error=f"{type(exc).__name__}: {exc}")
            for result in results:
                if result.faithfulness is None and not result.faithfulness_skipped:
                    result.faithfulness_skipped = (
                        f"judge pass failed ({type(exc).__name__})"
                    )

    return results, score(results, has_real_llm=has_real_llm)
