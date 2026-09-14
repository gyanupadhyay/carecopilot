"""Run the evaluation set and write a report.

    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --only rag
    python scripts/run_evaluation.py --only hybrid-001 --verbose

Without ``LLM_API_KEY`` the stub provider answers, so cases marked
``requires_llm`` are skipped and the metrics that depend on generated text
are reported as skipped with a reason. Retrieval and routing metrics that do
not need a model are measured either way — which is most of the retrieval
half of PRD §27.

The report is written to ``evaluation/reports/`` as JSON, and a summary is
printed. Reports are gitignored: they are run artifacts, and committing them
turns every eval run into a diff.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.db.session import AppSession, dispose_engines
from app.llm.factory import build_provider
from app.observability.logging import configure_logging
from app.rag.embeddings import build_embedder
from app.runtime import selector_loop_factory
from evaluation.runner import REPORTS_DIR, run_dataset

GREEN, YELLOW, RESET = "", "", ""


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


async def run(args: argparse.Namespace) -> int:
    llm = build_provider()
    has_real_llm = llm.name != "stub"
    embedder = build_embedder()

    if not has_real_llm:
        print(
            "No LLM_API_KEY configured — running with the stub provider.\n"
            "Cases needing generated text are skipped; retrieval and "
            "rule-based routing are still measured.\n"
        )

    async with AppSession() as session:
        results, groups = await run_dataset(
            session,
            llm=llm,
            embedder=embedder,
            has_real_llm=has_real_llm,
            only=args.only,
            delay_seconds=args.delay,
            judge=args.judge,
            judge_model=args.judge_model,
        )

    await dispose_engines()

    # --- report ------------------------------------------------------- #
    for group in groups:
        print(f"\n{group.name.upper()}")
        for key, value in group.values.items():
            if key == "confusion":
                continue
            print(f"  {key:<28} {_fmt(value)}")
        for key, reason in group.skipped.items():
            print(f"  {key:<28} skipped — {reason}")

    routing = next((g for g in groups if g.name == "routing"), None)
    matrix = routing.values.get("confusion") if routing else None
    if matrix:
        print("\nROUTING CONFUSION (expected → predicted)")
        for expected, predictions in sorted(matrix.items()):
            rendered = ", ".join(f"{p} x{n}" for p, n in sorted(predictions.items()))
            print(f"  {expected:<14} {rendered}")

    failures = [r for r in results if r.error]
    misroutes = [r for r in results if r.route_correct is False]
    if misroutes:
        print("\nMISROUTED")
        for result in misroutes:
            print(
                f"  {result.case_id:<12} expected {result.expected_route:<12} "
                f"got {result.route:<12} {result.question[:48]!r}"
            )
    if failures:
        print("\nERRORED")
        for result in failures:
            print(f"  {result.case_id:<12} {result.error}")

    # The number says how much to worry; the claim says what to fix.
    unfaithful = [r for r in results if r.unsupported_claims]
    if unfaithful:
        print("\nUNSUPPORTED CLAIMS (judge)")
        for result in unfaithful:
            for claim in result.unsupported_claims:
                print(f"  {result.case_id:<12} {claim[:88]}")

    if args.verbose:
        print("\nPER CASE")
        for result in results:
            state = (
                "skip" if result.skipped else "ERR" if result.error else "ok"
            )
            print(
                f"  [{state:>4}] {result.case_id:<12} {result.route or '-':<12} "
                f"{round(result.latency_ms):>5}ms  {result.question[:44]}"
            )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = REPORTS_DIR / f"eval-{stamp}.json"
    report_path.write_text(
        json.dumps(
            {
                "generated_at": stamp,
                "provider": llm.name,
                "model": llm.model,
                "embedding_model": embedder.model_name,
                "reranker": settings.reranker,
                "vector_backend": settings.vector_backend,
                "metrics": [g.as_dict() for g in groups],
                "cases": [r.as_dict() for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nReport written to {report_path.relative_to(ROOT)}")

    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        default=None,
        help="Run one category (api, rag, hybrid, ...) or one case id.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print every case.")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help=(
            "Seconds to wait between cases. Use on a free tier: a 429 turns "
            "a case into a non-measurement, so pacing produces a fuller "
            "report than running flat out. 5 is about right for Gemini free."
        ),
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "Score faithfulness with an LLM judge. Off by default because it "
            "adds a model call per cited case, which roughly doubles the wall "
            "clock of a local CPU run."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Model for the faithfulness judge. Defaults to the model under "
            "test, which makes the score a floor rather than an estimate — a "
            "model does not flag the hallucinations it finds plausible. Point "
            "this at a different model whenever one is available."
        ),
    )
    args = parser.parse_args(argv)

    # Windows gives a redirected stdout the legacy ANSI codepage, not UTF-8.
    # The summary contains an em dash and an arrow, so `> report.txt` raised
    # UnicodeEncodeError *after* every case had run — losing a report that
    # cost an hour of local inference to produce. Reconfiguring beats
    # rewriting the output as ASCII: the next non-ASCII character someone
    # adds should not be able to do this again.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    configure_logging()
    return asyncio.run(run(args), loop_factory=selector_loop_factory)


if __name__ == "__main__":
    raise SystemExit(main())
