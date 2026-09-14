"""Evaluation metrics (PRD §27).

Pure functions over already-collected results. Nothing here calls the
system under test, which keeps the metrics themselves testable and keeps
"what we measured" separate from "how we ran it".

Two honesty rules run through this module.

*A metric that cannot be computed returns ``None``, never zero.* A skipped
faithfulness check and a faithfulness of 0.0 mean opposite things, and a
dashboard that renders them the same way is worse than one that renders
nothing.

*Proxies are labelled as proxies.* ``answer_correctness`` here is keyword
containment, which is a weak stand-in for whether an answer is right. It
catches the failure that matters most in this system — an answer that omits
the fact it was asked for — and it cannot detect an answer that contains
the right words inside a wrong sentence. Where a metric is a proxy, its
docstring says so rather than letting a number imply more than it knows.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

# --- routing -------------------------------------------------------------- #


def accuracy(predicted: Sequence[str], expected: Sequence[str]) -> float | None:
    """Exact-match accuracy. ``None`` when there is nothing to score."""
    pairs = [(p, e) for p, e in zip(predicted, expected, strict=True) if e]
    if not pairs:
        return None
    return sum(1 for p, e in pairs if p == e) / len(pairs)


def confusion(
    predicted: Sequence[str], expected: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Expected route → predicted route → count.

    More useful than the accuracy number on its own: "RAG questions are
    being routed to API" is an actionable finding, while "78% accurate" is
    not.
    """
    matrix: dict[str, dict[str, int]] = {}
    for pred, exp in zip(predicted, expected, strict=True):
        if not exp:
            continue
        matrix.setdefault(exp, {})
        matrix[exp][pred] = matrix[exp].get(pred, 0) + 1
    return matrix


# --- retrieval ------------------------------------------------------------- #


def recall_at_k(retrieved: Sequence[int], relevant: set[int], k: int) -> float | None:
    """Share of relevant documents appearing in the top ``k``.

    ``None`` when nothing is relevant — a question with no ground truth
    cannot be scored, and counting it as a miss would drag the mean down
    for a reason unrelated to retrieval quality.
    """
    if not relevant:
        return None
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def precision_at_k(
    retrieved: Sequence[int], relevant: set[int], k: int
) -> float | None:
    """Share of the top ``k`` that is relevant."""
    if not relevant:
        return None
    window = retrieved[:k]
    if not window:
        return 0.0
    return len(set(window) & relevant) / len(window)


def reciprocal_rank(retrieved: Sequence[int], relevant: set[int]) -> float | None:
    """1/rank of the first relevant hit, or 0.0 if none was retrieved.

    Distinguishes "the right passage was second" from "the right passage
    was tenth", which Recall@5 cannot.
    """
    if not relevant:
        return None
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def mean(values: Sequence[float | None]) -> float | None:
    """Average the scores that exist, ignoring the ones that do not."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


# --- generation ------------------------------------------------------------ #


def answer_correctness(answer: str, must_mention: Sequence[str]) -> float | None:
    """Proxy: share of required facts that appear in the answer.

    A weak signal, and deliberately so. It reliably catches the failure this
    system most needs to catch — the answer that never states the fact it
    was asked for — and it cannot tell a right fact in a right sentence from
    a right fact in a wrong one. Treat a high score as "not obviously
    broken", not as "correct".
    """
    if not must_mention:
        return None
    lowered = answer.lower()
    return sum(1 for term in must_mention if term.lower() in lowered) / len(must_mention)


def contains_none_of(answer: str, forbidden: Sequence[str]) -> bool | None:
    """True when the answer avoids every forbidden string.

    Used for safety expectations — a refusal that still leaks the thing it
    refused is not a refusal.
    """
    if not forbidden:
        return None
    lowered = answer.lower()
    return not any(term.lower() in lowered for term in forbidden)


def citation_correctness(
    cited: Sequence[int], retrieved: Sequence[int], relevant: set[int]
) -> float | None:
    """Share of citations that point at a genuinely relevant passage.

    Distinct from precision: precision asks what retrieval surfaced, this
    asks what the answer actually pointed the patient at. An answer may
    retrieve well and cite badly.
    """
    if not cited:
        return None
    if not relevant:
        # Without ground truth the strongest available check is that every
        # citation was at least something retrieval returned — a citation
        # to anything else is fabricated.
        surfaced = set(retrieved)
        return sum(1 for c in cited if c in surfaced) / len(cited)
    return sum(1 for c in cited if c in relevant) / len(cited)


def grounding_rate(results: Sequence[Any]) -> float | None:
    """Share of record-backed answers that cited at least one source.

    An answer built from retrieved notes that cites nothing is ungrounded
    whatever it says.
    """
    applicable = [r for r in results if getattr(r, "expects_sources", False)]
    if not applicable:
        return None
    return sum(1 for r in applicable if r.cited) / len(applicable)


def rate(flags: Sequence[bool | None]) -> float | None:
    """Share of the decided flags that are True. ``None`` when none decided.

    The counterpart of :func:`mean` for boolean checks, and it exists so a
    caller never has to write ``sum(...)/len(...)`` over a list that might
    contain ``None`` — which is how a skipped check becomes a zero.
    """
    decided = [f for f in flags if f is not None]
    if not decided:
        return None
    return sum(1 for f in decided if f) / len(decided)


# --- knowledge graph -------------------------------------------------------- #
#
# These score the *traversal the system chose*, not the sentence it produced.
# That is the distinction PRD §27 is drawing when it asks for relationship and
# multi-hop correctness separately from answer correctness: an answer can name
# the right drug after walking the wrong edges, and on this dataset one
# demonstrably does — joining medication to condition through the encounter
# returns the right row for the wrong reason on any patient whose visit
# treated a single condition. Scoring only the prose would call that correct.

#: ``-[r:TYPE]->``, ``<-[:TYPE]-``, ``-[:TYPE*1..2]-``. The label is optional
#: in Cypher; a pattern without one contributes a hop but no type.
_RELATIONSHIP = re.compile(
    r"-\[\s*\w*\s*:\s*(\w+)[^\]]*\]\s*->"
    r"|<-\s*\[\s*\w*\s*:\s*(\w+)[^\]]*\]\s*-"
)
_CLAUSE = re.compile(
    r"\b(OPTIONAL\s+MATCH|MATCH|WHERE|RETURN|WITH|ORDER\s+BY|LIMIT)\b", re.I
)
#: A node pattern's variable: ``(p:Patient {...})`` → ``p``. An anonymous
#: node ``(:Condition)`` yields no variable and cannot be chained to.
_NODE = re.compile(r"\(\s*(\w*)\s*(?::[^)]*)?(?:\{[^}]*\})?\s*\)")
#: One node-relationship-node step. Group 2 is the right node's opening
#: paren rather than its variable, so a scan can resume *at* that node and
#: pick up the next step of a chained pattern — see the loop in `hop_depth`.
_STEP = re.compile(
    r"\(\s*(\w+)[^)]*\)\s*"
    r"(?:-\[[^\]]*\]\s*->|<-\s*\[[^\]]*\]\s*-|-\[[^\]]*\]\s*-)\s*"
    r"(\()\s*(\w+)"
)


def relationship_types(cypher: str) -> set[str]:
    """Every relationship type a traversal template walks."""
    found: set[str] = set()
    for forward, backward in _RELATIONSHIP.findall(cypher):
        found.add(forward or backward)
    return found


def hop_depth(cypher: str) -> int:
    """The longest chain of relationships a traversal walks from the patient.

    Counting relationship patterns per clause does not work here, and the
    reason is worth stating: these templates chain *across* clauses through
    shared variables. ``care_team`` is

        MATCH (p:Patient ...)-[:HAD_ENCOUNTER]->(e:Encounter)
        MATCH (e)-[:WITH_PROVIDER]->(pr:Provider)

    — two hops from the patient, one relationship pattern per clause. A
    per-clause count calls that one hop and a whole-statement count calls it
    two only by accident, since the next ``OPTIONAL MATCH`` would make it
    three whether or not it extended the path.

    So the variables are walked as a graph: edges are the node pairs each
    pattern binds, and the answer is the distance from the ``:Patient``
    anchor to the furthest node reachable through them. Patterns in
    ``RETURN`` are excluded — the ``CASE WHEN (e)-[:FOR_CONDITION]->(c)``
    in ``why_medication`` tests an edge rather than travelling along one.
    """
    edges: dict[str, set[str]] = {}
    anchor = ""
    parts = _CLAUSE.split(cypher)
    for keyword, pattern in zip(parts[1::2], parts[2::2], strict=False):
        if not keyword.upper().endswith("MATCH"):
            continue
        if not anchor:
            for match in _NODE.finditer(pattern):
                variable = match.group(1)
                if variable and ":Patient" in match.group(0):
                    anchor = variable
                    break
        # Overlapping steps: ``(a)-[]->(b)-[]->(c)`` must yield a→b *and*
        # b→c. Resuming after the whole match would restart past ``(b)`` and
        # see only the first, so the scan resumes at ``(b)`` itself.
        position = 0
        while (step := _STEP.search(pattern, position)) is not None:
            left, right = step.group(1), step.group(3)
            edges.setdefault(left, set()).add(right)
            edges.setdefault(right, set()).add(left)
            position = step.start(2)

    if not anchor or anchor not in edges:
        return 0

    # Longest simple path from the anchor, not shortest-path depth. The
    # difference decides ``medications_for_condition``, whose pattern is a
    # triangle: the patient reaches both the condition and the medication
    # directly, so every node sits at BFS distance 1 and the traversal scores
    # as single-hop — when the hop that matters is precisely the third edge,
    # ``(m)-[:TREATS]->(c)``, which is what stops the answer being assembled
    # through the encounter instead. Exhaustive search is fine here: these
    # patterns bind at most a handful of variables.
    def walk(node: str, seen: frozenset[str]) -> int:
        return max(
            (
                1 + walk(neighbour, seen | {neighbour})
                for neighbour in edges.get(node, ())
                if neighbour not in seen
            ),
            default=0,
        )

    return walk(anchor, frozenset({anchor}))


def relationship_correctness(cypher: str, expected: Sequence[str]) -> bool | None:
    """Does the chosen traversal walk every relationship the case requires?

    Subset, not equality: a template may also walk optional edges for
    context, and penalising that would measure the template's verbosity.
    """
    if not expected:
        return None
    return {r.upper() for r in expected}.issubset(relationship_types(cypher))


def multi_hop_correctness(cypher: str, expected_hops: int | None) -> bool | None:
    """Did a multi-hop question get a traversal of at least that depth?

    ``None`` for single-hop cases: every traversal clears a bar of one, so
    including them would dilute the metric with cases it cannot fail.
    """
    if not expected_hops or expected_hops < 2:
        return None
    return hop_depth(cypher) >= expected_hops


def entity_resolution(matched: bool | None) -> bool | None:
    """Did the patient's phrasing resolve to something in the graph?

    The failure this catches is specific and was real: a patient says "blood
    pressure", the catalogue says "Essential hypertension", neither is a
    substring of the other, and the traversal returns zero rows having
    chosen the right intent. Routing and tool selection both score 1.0 on
    that case; only this metric registers it.
    """
    return matched


# --- text to sql ------------------------------------------------------------ #


def sql_tables(sql: str) -> set[str]:
    """Table names a statement reads, lowercased. Empty if it will not parse."""
    try:
        import sqlglot
        from sqlglot import expressions as exp
    except ImportError:  # pragma: no cover - sqlglot is a backend dependency
        return set()
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return set()
    if tree is None:
        return set()
    return {t.name.lower() for t in tree.find_all(exp.Table) if t.name}


def sql_functions(sql: str) -> set[str]:
    """Function names a statement calls, uppercased — ``{"COUNT", "AVG"}``.

    Boolean connectors are excluded. sqlglot registers ``And`` and ``Or`` as
    ``Func`` subclasses, so a plain ``WHERE a = 1 AND b > 2`` would otherwise
    report calling a function named ``AND`` — harmless for the subset check
    this feeds, and actively confusing in a report someone reads.
    """
    try:
        import sqlglot
        from sqlglot import expressions as exp
    except ImportError:  # pragma: no cover
        return set()
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return set()
    if tree is None:
        return set()
    return {
        type(node).__name__.upper()
        for node in tree.find_all(exp.Func)
        if not isinstance(node, exp.Connector)
    }


def sql_parses(sql: str) -> bool:
    try:
        import sqlglot
    except ImportError:  # pragma: no cover
        return False
    try:
        return sqlglot.parse_one(sql, read="postgres") is not None
    except Exception:
        return False


def sql_filters_patient(sql: str) -> bool:
    """Does the statement predicate on ``patient_id``?

    In this system that is a defect, not a safeguard. Row-level security
    already scopes the connection to one patient, so a ``patient_id``
    predicate can only narrow the result *further* — and the only value the
    generator could put there is one it guessed, which means the patient's
    own rows get excluded. Detected as a predicate rather than as a
    substring, so ``SELECT patient_id`` in a projection does not count.
    """
    try:
        import sqlglot
        from sqlglot import expressions as exp
    except ImportError:  # pragma: no cover
        return "patient_id" in sql.lower()
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return "patient_id" in sql.lower()
    if tree is None:
        return False
    for where in tree.find_all(exp.Where):
        for column in where.find_all(exp.Column):
            if column.name.lower() == "patient_id":
                return True
    return False


def sql_query_correctness(
    sql: str, expected_tables: Sequence[str], expected_functions: Sequence[str]
) -> bool | None:
    """Does the statement read the right tables and compute the right thing?

    A proxy, and labelled as one: it cannot tell a correct ``COUNT`` from one
    with a wrong ``WHERE``. It does catch the two failures that matter for an
    8B generator — counting the wrong table, and answering "how many" with a
    ``SELECT *`` — which are the ones a plausible-looking number hides.
    """
    if not expected_tables and not expected_functions:
        return None
    if not sql:
        return False
    tables_ok = {t.lower() for t in expected_tables}.issubset(sql_tables(sql))
    functions_ok = {f.upper() for f in expected_functions}.issubset(sql_functions(sql))
    return tables_ok and functions_ok


def sql_authorization_correctness(
    sql: str, allowed_tables: Iterable[str]
) -> bool | None:
    """Two conditions, both required.

    The statement may name only allowlisted tables, and it may not predicate
    on ``patient_id``. The first is the obvious one; the second is the one a
    reviewer talks themselves out of, because a ``patient_id`` filter *looks*
    like defence in depth. It is not — see :func:`sql_filters_patient`.
    """
    if not sql:
        return None
    allowed = {t.lower() for t in allowed_tables}
    referenced = sql_tables(sql)
    if referenced and not referenced.issubset(allowed):
        return False
    return not sql_filters_patient(sql)


# --- agent ------------------------------------------------------------------ #


def tool_call_success_rate(calls: Sequence[dict[str, Any]]) -> float | None:
    """Share of recorded tool invocations that succeeded.

    Distinct from tool *selection*: choosing the right tool and having it
    return something are different failures, and a run where selection is
    1.000 and success is 0.400 is a broken system that the selection number
    alone would call healthy.
    """
    if not calls:
        return None
    return sum(1 for c in calls if c.get("ok")) / len(calls)


def json_validity(calls: int, failures: int) -> float | None:
    """Share of schema-constrained model calls whose output parsed.

    ``None`` when the turn made no structured call — a rule-routed API
    question asks the model for no JSON at all, and scoring it 1.0 would
    report a passed test that never ran.
    """
    if calls <= 0:
        return None
    return (calls - failures) / calls


# --- system ---------------------------------------------------------------- #


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile. ``None`` for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


@dataclass(frozen=True, slots=True)
class LatencySummary:
    count: int
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    max_ms: float | None


def latency(values: Sequence[float]) -> LatencySummary:
    if not values:
        return LatencySummary(0, None, None, None, None)
    return LatencySummary(
        count=len(values),
        mean_ms=statistics.fmean(values),
        p50_ms=percentile(values, 0.50),
        p95_ms=percentile(values, 0.95),
        max_ms=max(values),
    )


def error_rate(errors: int, total: int) -> float | None:
    return errors / total if total else None


# --- aggregation ------------------------------------------------------------ #


@dataclass(slots=True)
class MetricGroup:
    """A named block of metrics, with the ones that could not run recorded."""

    name: str
    values: dict[str, float | None] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def add(self, key: str, value: float | None, *, skip_reason: str = "") -> None:
        if value is None and skip_reason:
            self.skipped[key] = skip_reason
        else:
            self.values[key] = value

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values": self.values,
            "skipped": self.skipped,
        }
