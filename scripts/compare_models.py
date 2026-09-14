"""Run the evaluation set against several models and tabulate the difference.

    python scripts/compare_models.py --models qwen3:8b,qwen3:14b
    python scripts/compare_models.py --models Qwen/Qwen3-8B,Qwen/Qwen3-14B \
        --provider huggingface
    python scripts/compare_models.py --models gemma4:31b,gpt-oss:120b --judge

This is PRD §28: a model comparison that is *measured* rather than asserted.
The whole value of it is that both columns come from the same cases, the
same corpus, the same retrieval and the same scoring code — so a difference
in a number is a difference between the models and not between two runs that
happened to be configured differently.

Four things follow from that, and each is enforced here rather than left to
whoever reads the table.

*One process, one dataset, back to back.* Re-seeding or re-indexing between
columns would change retrieval under the second model, and recall would move
for a reason that has nothing to do with it.

*Only the model changes.* The provider, embedder, reranker and top-k come
from the same settings for every column. ``--provider`` applies to all
models, not per model, because a column served by different hardware cannot
be compared on latency — and latency is one of the things §28 asks for.

*A metric that one model skipped is not compared.* If faithfulness ran for
one column and not the other, the row prints the scores it has and marks the
gap; it never fills the hole with a zero and calls one model better. This is
the same rule the runner applies within a single run, and the reason for it
is identical — see ``evaluation/metrics``.

*The winner is not computed.* The table reports; the reader decides. A
composite "score" would need weights, the weights would be invented here,
and the number would then look objective while encoding one opinion about
whether safety matters more than latency. It does, but that belongs in
prose, not in a column.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.db.session import AppSession, dispose_engines
from app.llm.base import LLMProvider
from app.llm.factory import build_provider
from app.observability.logging import configure_logging
from app.rag.embeddings import build_embedder
from app.runtime import selector_loop_factory
from evaluation.metrics import MetricGroup
from evaluation.runner import REPORTS_DIR, CaseResult, load_cases, run_dataset

#: Rows in the printed table, in the order a reader should meet them:
#: did it route correctly, did it find the right passages, was the answer
#: right, did the agent machinery hold, was it safe, what did it cost.
#: ``confusion`` and the ``cases_judged`` counters are deliberately absent —
#: they are per-run diagnostics, not comparable quantities.
COMPARED: tuple[tuple[str, str], ...] = (
    ("routing", "router_accuracy"),
    ("retrieval", "recall_at_5"),
    ("retrieval", "precision_at_5"),
    ("retrieval", "mrr"),
    ("generation", "answer_correctness"),
    ("generation", "citation_correctness"),
    ("generation", "grounding_rate"),
    ("generation", "faithfulness"),
    ("knowledge_graph", "entity_resolution_rate"),
    ("knowledge_graph", "relationship_correctness"),
    ("knowledge_graph", "multi_hop_correctness"),
    ("text_to_sql", "sql_validity"),
    ("text_to_sql", "execution_success"),
    ("text_to_sql", "query_correctness"),
    ("text_to_sql", "authorization_correctness"),
    ("agent", "tool_selection_accuracy"),
    ("agent", "tool_call_success_rate"),
    ("agent", "tool_argument_validity"),
    ("agent", "json_validity"),
    ("security", "leak_free_rate"),
    ("system", "latency_mean_ms"),
    ("system", "latency_p95_ms"),
    ("system", "error_rate"),
    ("system", "cases_degraded"),
)

#: Metrics where lower is better, so the table's delta column can say which
#: way a difference points without the reader having to remember.
LOWER_IS_BETTER = frozenset(
    {"latency_mean_ms", "latency_p95_ms", "error_rate", "cases_degraded"}
)


class ModelRun:
    """One column: the model, its metrics, and what it spent."""

    def __init__(self, model: str, provider: str) -> None:
        self.model = model
        self.provider = provider
        self.groups: list[MetricGroup] = []
        self.results: list[CaseResult] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd: float | None = None
        self.failed = ""

    def value(self, group: str, key: str) -> float | None:
        for candidate in self.groups:
            if candidate.name == group:
                found = candidate.values.get(key)
                return found if isinstance(found, (int, float)) else None
        return None

    def skipped(self, group: str, key: str) -> str:
        for candidate in self.groups:
            if candidate.name == group:
                return candidate.skipped.get(key, "")
        return ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "failed": self.failed,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.cost_usd,
            "metrics": [g.as_dict() for g in self.groups],
            "cases": [r.as_dict() for r in self.results],
        }


def _fmt(value: float | None, key: str) -> str:
    if value is None:
        return "—"
    if key.endswith("_ms"):
        return f"{value:,.0f}"
    if key in ("cases_degraded",):
        return f"{int(value)}"
    return f"{value:.3f}"


def _delta(first: float | None, second: float | None, key: str) -> str:
    """The second column minus the first, signed so + always means better.

    Printed only when both columns have a number. A delta against a skipped
    metric would be a comparison with nothing, and rendering it as a large
    improvement is exactly the misreading this whole module is arranged to
    prevent.
    """
    if first is None or second is None:
        return "—"
    change = second - first
    if key in LOWER_IS_BETTER:
        change = -change
    # Collapse negative zero. Two identical columns produce -0.0 whenever the
    # sign was flipped above, and "-0.000" reads as a regression that did not
    # happen — in a table whose entire job is telling a reader which way a
    # difference points.
    if change == 0:
        change = 0.0
    if key.endswith("_ms"):
        return f"{change:+,.0f}"
    return f"{change:+.3f}"


def stratified_sample(cases: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Up to ``size`` cases, spread across categories in round-robin order.

    Exists because the honest alternative to a full run is a *representative*
    subset, not the first N. The dataset is grouped by category, so a head
    slice is all API cases — and a comparison drawn from it would report
    nothing about KG, Text-to-SQL or safety while looking like a comparison
    of the system.

    Deterministic: same dataset, same sample, so two runs of this script are
    comparable to each other and not only within themselves.
    """
    by_category: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_category.setdefault(case["category"], []).append(case)

    picked: list[dict[str, Any]] = []
    index = 0
    while len(picked) < size:
        added = False
        for category in sorted(by_category):
            bucket = by_category[category]
            if index < len(bucket) and len(picked) < size:
                picked.append(bucket[index])
                added = True
        if not added:
            break
        index += 1
    # Back into dataset order, so a report reads like a short dataset rather
    # than like a shuffled one.
    order = {case["id"]: n for n, case in enumerate(cases)}
    return sorted(picked, key=lambda c: order[c["id"]])


async def run_one(
    model: str,
    *,
    provider_name: str,
    judge: bool,
    delay: float,
    only: str | None,
    cases: list[dict[str, Any]] | None = None,
) -> ModelRun:
    """Score one model over the whole dataset."""
    run = ModelRun(model, provider_name)
    llm: LLMProvider | None = None
    try:
        llm = build_provider(provider=provider_name)
        # The factory reads the model from settings; this is the one thing
        # that differs between columns, so it is set here rather than by
        # mutating settings — which would leak into the next column.
        llm = _with_model(llm, model)
        embedder = build_embedder()

        async with AppSession() as session:
            results, groups = await run_dataset(
                session,
                llm=llm,
                embedder=embedder,
                has_real_llm=llm.name != "stub",
                cases=cases,
                only=only,
                delay_seconds=delay,
                judge=judge,
            )
        run.results = results
        run.groups = groups
    except Exception as exc:  # pragma: no cover - a column may legitimately fail
        # One model being unavailable must not discard the column that
        # already ran. The report says so per model rather than aborting.
        run.failed = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        if llm is not None:
            await llm.aclose()
    return run


def _with_model(llm: LLMProvider, model: str) -> LLMProvider:
    """Point a built provider at a specific model.

    Providers expose ``model`` as a read-only property over a private
    attribute, so this sets the attribute the property reads. Rebuilding the
    provider per model would be cleaner, but the factory takes its model from
    settings and nothing else — and mutating settings between columns is how
    the second column silently inherits the first one's configuration.
    """
    llm._model = model  # type: ignore[attr-defined]
    return llm


async def run(args: argparse.Namespace) -> int:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) < 2:
        print("Need at least two models to compare.", file=sys.stderr)
        return 2

    provider_name = args.provider or settings.llm_provider

    cases = None
    if args.sample:
        cases = stratified_sample(load_cases(), args.sample)
        counts = Counter(c["category"] for c in cases)
        print(
            f"Sampling {len(cases)} of {len(load_cases())} cases: "
            + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        )
        print(
            "A sample, not a full run — every figure below is over these "
            "cases only.\n"
        )

    runs: list[ModelRun] = []
    for index, model in enumerate(models, start=1):
        print(f"\n[{index}/{len(models)}] {provider_name}:{model} …", flush=True)
        result = await run_one(
            model,
            provider_name=provider_name,
            judge=args.judge,
            delay=args.delay,
            only=args.only,
            cases=cases,
        )
        if result.failed:
            print(f"    failed: {result.failed}", file=sys.stderr)
        else:
            system = next((g for g in result.groups if g.name == "system"), None)
            ran = system.values.get("cases_ran") if system else "?"
            print(f"    {ran} cases scored", flush=True)
        runs.append(result)

    await dispose_engines()
    _print_table(runs)
    return _write_report(
        runs,
        provider_name,
        models,
        compared=len(cases) if cases else len(load_cases()),
        sampled=bool(cases),
    )


def _print_table(runs: list[ModelRun]) -> None:
    usable = [r for r in runs if not r.failed]
    if not usable:
        print("\nNo model produced a scorable run.", file=sys.stderr)
        return

    width = max(14, max(len(r.model) for r in usable) + 2)
    header = "  ".join(f"{r.model:>{width}}" for r in usable)
    delta_column = len(usable) == 2
    print(f"\n{'metric':<30}{header}" + ("       delta" if delta_column else ""))
    print("-" * (30 + (width + 2) * len(usable) + (12 if delta_column else 0)))

    last_group = ""
    for group, key in COMPARED:
        if group != last_group:
            print(f"\n{group.upper()}")
            last_group = group
        cells = "  ".join(
            f"{_fmt(r.value(group, key), key):>{width}}" for r in usable
        )
        line = f"  {key:<28}{cells}"
        if delta_column:
            change = _delta(
                usable[0].value(group, key), usable[1].value(group, key), key
            )
            line += f"  {change:>10}"
        print(line)

    # Why a cell is a dash, stated once rather than guessed at per row.
    print("\nSKIPPED")
    any_skipped = False
    for group, key in COMPARED:
        for candidate in usable:
            reason = candidate.skipped(group, key)
            if reason:
                any_skipped = True
                print(f"  {candidate.model:<20} {key:<28} {reason}")
    if not any_skipped:
        print("  (nothing skipped — every metric was computable for every model)")

    for failure in (r for r in runs if r.failed):
        print(f"\nFAILED  {failure.model}: {failure.failed}")


def _write_report(
    runs: list[ModelRun],
    provider: str,
    models: list[str],
    *,
    compared: int,
    sampled: bool,
) -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS_DIR / f"comparison-{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": stamp,
                "provider": provider,
                "models": models,
                # A reader cannot tell a 6-case comparison from a full one
                # by looking at the metrics, so the report says which it is.
                "cases_compared": compared,
                "sampled": sampled,
                # Recorded because they are held constant across columns and
                # a later reader cannot otherwise tell whether two reports
                # are comparable to each other.
                "embedding_model": settings.embedding_model,
                "reranker": settings.reranker,
                "vector_backend": settings.vector_backend,
                "retrieval_top_k": settings.retrieval_top_k,
                "runs": [r.as_dict() for r in runs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nReport written to {path.relative_to(ROOT)}")
    return 1 if any(r.failed for r in runs) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model ids, in the order the columns should "
        "appear. Two gives a delta column.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider for every column. Defaults to LLM_PROVIDER. Applies to "
        "all models on purpose: columns served by different hardware cannot "
        "be compared on latency, which is one of the things §28 asks for.",
    )
    parser.add_argument(
        "--only", default=None, help="Restrict to one category or case id."
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Also score faithfulness. Doubles the run, and the judge is the "
        "column's own model unless the runner is told otherwise — so a "
        "faithfulness row compares two self-assessments, not one standard.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Compare over N cases spread across categories instead of the "
        "whole set. "
        "For CPU inference, where a full run of two models is hours. The "
        "sample is stratified and deterministic, and the report records its "
        "size — a subset is an honest comparison only while it says so.",
    )
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args(argv)

    # Same reason as run_evaluation.py: on Windows a redirected stdout gets
    # the ANSI codepage, and the em dash used for a skipped metric would
    # raise UnicodeEncodeError after every case had already run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    configure_logging()
    return asyncio.run(run(args), loop_factory=selector_loop_factory)


if __name__ == "__main__":
    raise SystemExit(main())
