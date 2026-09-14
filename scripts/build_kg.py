"""Project PostgreSQL into Neo4j (PRD §33).

    python scripts/build_kg.py                # rebuild from the current database
    python scripts/build_kg.py --incremental  # MERGE into the existing graph
    python scripts/build_kg.py --verify       # report counts and exit

PostgreSQL is the system of record; this writes a derived relationship view
and nothing else. Every statement is a ``MERGE`` keyed on the PostgreSQL id,
so running it twice produces the same graph as running it once — which is
what makes "rebuildable" true rather than aspirational.

``MERGE`` alone does not make it true, though, and assuming it did cost a
real cross-patient leak. MERGE adds and never removes, so a row that moved
between patients left the graph holding *both* ownership edges, and the
traversal anchored on the wrong patient returned a prescription belonging to
someone else — through correct queries, over an intact isolation invariant,
on drifted data. ``test_two_patients_get_their_own_subgraphs`` caught it
after a reseed; 4,739 stale edges had accumulated by then.

**So a full rebuild is the default, and MERGE is the opt-in.** That inverts
the obvious choice, and deliberately: the drift is not one bug with six
instances, it is a property of projecting with MERGE, and every edge type
added later inherits it. Dropping the graph first is a few seconds against a
derived view whose source is right there, and it makes the guarantee
structural instead of a list of reconciliations someone has to remember to
extend. :func:`reconcile` and :func:`sweep` still run, so ``--incremental``
is safe for the edges that carry patient scope — but only the default is
safe for all of them.

The one design decision worth stating: **clinical nodes are per patient.** A
``Medication`` node here is one patient's prescription, not the drug
Metformin — two patients on metformin get two nodes, and the same holds for
``Encounter`` and ``LabResult``. That costs duplication and buys a small
blast radius: a traversal that wandered one hop too far still lands on
something belonging to the patient it started from.

What it does **not** buy is isolation by construction. ``Condition``,
``Lab``, ``Department`` and ``Provider`` are deliberately shared — they are
catalogue entries, and a graph where every patient has a private copy of
"Cardiology" is a hundred disjoint trees rather than a graph. Shared nodes
mean cross-patient paths exist: measured on this dataset, 156,555 of them at
length 4 or less, mostly ``(p1)-[:HAS_CONDITION]->(c)<-[:HAS_CONDITION]-(p2)``.
That number rises every time an entity is added — it was 138,372 before
``Diagnosis`` — which is worth knowing before treating it as an alarm.

So PRD §21 is enforced where the queries are, not here. Every approved
traversal anchors on one ``(p:Patient {id: $patient_id})`` and moves outward,
and none contains a second ``:Patient`` pattern to arrive at — which is an
invariant ``test_knowledge_graph.py`` checks mechanically, because it is the
kind of property a well-meaning edit silently breaks.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import AppSession
from app.knowledge_graph import schema
from app.knowledge_graph.client import (
    GraphUnavailable,
    dispose_driver,
    get_driver,
)
from app.models import (
    Allergy,
    Condition,
    Diagnosis,
    Encounter,
    LabResult,
    Medication,
    Patient,
    PatientCondition,
    Procedure,
    Provider,
)

#: Rows per write transaction. Large enough that a hundred patients is a
#: handful of round trips, small enough that one failure does not roll back
#: the whole projection.
BATCH = 500


async def _write(cypher: str, rows: list[dict[str, Any]]) -> None:
    """Run one UNWIND-ed write over a batch of rows."""
    if not rows:
        return
    driver = get_driver()
    async with driver.session(database=settings.neo4j_database) as session:
        for start in range(0, len(rows), BATCH):
            await session.run(cypher, {"rows": rows[start : start + BATCH]})  # type: ignore[arg-type]


async def apply_schema() -> None:
    """Constraints first: MERGE without them duplicates instead of updating."""
    driver = get_driver()
    async with driver.session(database=settings.neo4j_database) as session:
        for statement in schema.CONSTRAINTS + schema.INDEXES:
            await session.run(statement)  # type: ignore[arg-type]


async def reset() -> None:
    """Drop every node and relationship. The projection, not the source."""
    driver = get_driver()
    async with driver.session(database=settings.neo4j_database) as session:
        await session.run("MATCH (n) DETACH DELETE n")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------- #


async def project_patients(db: AsyncSession) -> int:
    rows = [
        {
            "id": p.id,
            "external_id": p.external_id,
            "name": p.full_name,
            "date_of_birth": p.date_of_birth.isoformat(),
            "gender": p.gender,
        }
        for p in (await db.scalars(select(Patient))).all()
    ]
    await _write(
        f"""
        UNWIND $rows AS row
        MERGE (p:{schema.PATIENT} {{id: row.id}})
        SET p.external_id = row.external_id,
            p.name = row.name,
            p.date_of_birth = date(row.date_of_birth),
            p.gender = row.gender
        """,
        rows,
    )
    return len(rows)


async def project_providers(db: AsyncSession) -> int:
    rows = [
        {"id": pr.id, "name": pr.name, "specialty": pr.specialty}
        for pr in (await db.scalars(select(Provider))).all()
    ]
    await _write(
        f"""
        UNWIND $rows AS row
        MERGE (pr:{schema.PROVIDER} {{id: row.id}})
        SET pr.name = row.name, pr.specialty = row.specialty
        MERGE (d:{schema.DEPARTMENT} {{name: row.specialty}})
        MERGE (pr)-[:{schema.IN_DEPARTMENT}]->(d)
        """,
        rows,
    )
    return len(rows)


async def project_conditions(db: AsyncSession) -> int:
    rows = [
        {
            "key": c.key,
            "display": c.display,
            # Lower-cased once here rather than on every traversal: the
            # traversal compares a lower-cased term against them, and doing
            # it at match time would prevent an index from ever being useful.
            "aliases": [alias.lower() for alias in (c.aliases or [])],
        }
        for c in (await db.scalars(select(Condition))).all()
    ]
    await _write(
        f"""
        UNWIND $rows AS row
        MERGE (c:{schema.CONDITION} {{key: row.key}})
        SET c.display = row.display,
            c.aliases = row.aliases
        """,
        rows,
    )
    return len(rows)


async def project_patient_conditions(db: AsyncSession) -> int:
    stmt = select(PatientCondition, Condition.key).join(
        Condition, Condition.id == PatientCondition.condition_id
    )
    rows = [
        {
            "patient_id": pc.patient_id,
            "key": key,
            "onset_date": pc.onset_date.isoformat() if pc.onset_date else None,
        }
        for pc, key in (await db.execute(stmt)).all()
    ]
    await _write(
        f"""
        UNWIND $rows AS row
        MATCH (p:{schema.PATIENT} {{id: row.patient_id}})
        MATCH (c:{schema.CONDITION} {{key: row.key}})
        MERGE (p)-[r:{schema.HAS_CONDITION}]->(c)
        SET r.onset_date =
            CASE WHEN row.onset_date IS NULL THEN NULL ELSE date(row.onset_date) END
        """,
        rows,
    )
    return len(rows)


async def project_encounters(db: AsyncSession) -> int:
    stmt = select(Encounter, Condition.key).outerjoin(
        Condition, Condition.id == Encounter.condition_id
    )
    rows = [
        {
            "id": e.id,
            "patient_id": e.patient_id,
            "provider_id": e.provider_id,
            "condition_key": key,
            "encounter_date": e.encounter_date.isoformat(),
            "encounter_type": e.encounter_type,
            "reason": e.reason,
        }
        for e, key in (await db.execute(stmt)).all()
    ]
    await _write(
        f"""
        UNWIND $rows AS row
        MATCH (p:{schema.PATIENT} {{id: row.patient_id}})
        MERGE (e:{schema.ENCOUNTER} {{id: row.id}})
        SET e.patient_id = row.patient_id,
            e.encounter_date = date(row.encounter_date),
            e.encounter_type = row.encounter_type,
            e.reason = row.reason
        MERGE (p)-[:{schema.HAD_ENCOUNTER}]->(e)
        WITH e, row
        // FOREACH over a 0- or 1-element list is Cypher's conditional write:
        // the clause runs once when the id is present and not at all when it
        // is null, without splitting this into two statements.
        FOREACH (_ IN CASE WHEN row.provider_id IS NULL THEN [] ELSE [1] END |
            MERGE (pr:{schema.PROVIDER} {{id: row.provider_id}})
            MERGE (e)-[:{schema.WITH_PROVIDER}]->(pr))
        FOREACH (_ IN CASE WHEN row.condition_key IS NULL THEN [] ELSE [1] END |
            MERGE (c:{schema.CONDITION} {{key: row.condition_key}})
            MERGE (e)-[:{schema.FOR_CONDITION}]->(c))
        """,
        rows,
    )
    return len(rows)


async def project_medications(db: AsyncSession) -> int:
    stmt = select(Medication, Condition.key).outerjoin(
        Condition, Condition.id == Medication.condition_id
    )
    rows = [
        {
            "id": m.id,
            "patient_id": m.patient_id,
            "encounter_id": m.encounter_id,
            "condition_key": key,
            "name": m.name,
            "dosage": m.dosage,
            "frequency": m.frequency,
            "status": m.status,
            "start_date": m.start_date.isoformat(),
            "end_date": m.end_date.isoformat() if m.end_date else None,
        }
        for m, key in (await db.execute(stmt)).all()
    ]
    await _write(
        f"""
        UNWIND $rows AS row
        MATCH (p:{schema.PATIENT} {{id: row.patient_id}})
        MERGE (m:{schema.MEDICATION} {{id: row.id}})
        SET m.patient_id = row.patient_id,
            m.name = row.name,
            m.dosage = row.dosage,
            m.frequency = row.frequency,
            m.status = row.status,
            m.start_date = date(row.start_date),
            m.end_date =
                CASE WHEN row.end_date IS NULL THEN NULL ELSE date(row.end_date) END
        MERGE (p)-[:{schema.TAKES}]->(m)
        WITH m, row
        FOREACH (_ IN CASE WHEN row.encounter_id IS NULL THEN [] ELSE [1] END |
            MERGE (e:{schema.ENCOUNTER} {{id: row.encounter_id}})
            MERGE (e)-[:{schema.PRESCRIBED}]->(m))
        FOREACH (_ IN CASE WHEN row.condition_key IS NULL THEN [] ELSE [1] END |
            MERGE (c:{schema.CONDITION} {{key: row.condition_key}})
            MERGE (m)-[:{schema.TREATS}]->(c))
        """,
        rows,
    )
    return len(rows)


async def project_lab_results(db: AsyncSession) -> int:
    rows = [
        {
            "id": lr.id,
            "patient_id": lr.patient_id,
            "encounter_id": lr.encounter_id,
            "test_name": lr.test_name,
            "value": float(lr.value),
            "unit": lr.unit,
            "reference_range": lr.reference_range,
            "result_date": lr.result_date.isoformat(),
        }
        for lr in (await db.scalars(select(LabResult))).all()
    ]
    await _write(
        f"""
        UNWIND $rows AS row
        MATCH (p:{schema.PATIENT} {{id: row.patient_id}})
        MERGE (l:{schema.LAB} {{name: row.test_name}})
        MERGE (lr:{schema.LAB_RESULT} {{id: row.id}})
        SET lr.patient_id = row.patient_id,
            lr.encounter_id = row.encounter_id,
            lr.value = row.value,
            lr.unit = row.unit,
            lr.reference_range = row.reference_range,
            lr.result_date = date(row.result_date)
        MERGE (p)-[:{schema.HAS_LAB_RESULT}]->(lr)
        MERGE (lr)-[:{schema.OF_LAB}]->(l)
        WITH lr, l, row
        FOREACH (_ IN CASE WHEN row.encounter_id IS NULL THEN [] ELSE [1] END |
            MERGE (e:{schema.ENCOUNTER} {{id: row.encounter_id}})
            MERGE (e)-[:{schema.ORDERED}]->(l))
        """,
        rows,
    )
    return len(rows)


async def project_procedures(db: AsyncSession) -> int:
    stmt = select(Procedure, Condition.key).outerjoin(
        Condition, Condition.id == Procedure.condition_id
    )
    rows = [
        {
            "id": pr.id,
            "patient_id": pr.patient_id,
            "encounter_id": pr.encounter_id,
            "provider_id": pr.provider_id,
            "condition_key": key,
            "name": pr.name,
            "code": pr.code,
            "performed_date": pr.performed_date.isoformat(),
        }
        for pr, key in (await db.execute(stmt)).all()
    ]
    await _write(
        f"""
        UNWIND $rows AS row
        MATCH (p:{schema.PATIENT} {{id: row.patient_id}})
        MERGE (proc:{schema.PROCEDURE} {{id: row.id}})
        SET proc.patient_id = row.patient_id,
            proc.name = row.name,
            proc.code = row.code,
            proc.performed_date = date(row.performed_date)
        MERGE (p)-[:{schema.HAD_PROCEDURE}]->(proc)
        WITH proc, row
        FOREACH (_ IN CASE WHEN row.encounter_id IS NULL THEN [] ELSE [1] END |
            MERGE (e:{schema.ENCOUNTER} {{id: row.encounter_id}})
            MERGE (e)-[:{schema.PERFORMED}]->(proc))
        FOREACH (_ IN CASE WHEN row.condition_key IS NULL THEN [] ELSE [1] END |
            MERGE (c:{schema.CONDITION} {{key: row.condition_key}})
            MERGE (proc)-[:{schema.FOR_CONDITION}]->(c))
        """,
        rows,
    )
    return len(rows)


async def project_allergies(db: AsyncSession) -> int:
    rows = [
        {
            "id": a.id,
            "patient_id": a.patient_id,
            "substance": a.substance,
            "reaction": a.reaction,
            "severity": a.severity,
            "recorded_date": a.recorded_date.isoformat() if a.recorded_date else None,
        }
        for a in (await db.scalars(select(Allergy))).all()
    ]
    await _write(
        f"""
        UNWIND $rows AS row
        MATCH (p:{schema.PATIENT} {{id: row.patient_id}})
        MERGE (a:{schema.ALLERGY} {{id: row.id}})
        SET a.patient_id = row.patient_id,
            a.substance = row.substance,
            a.reaction = row.reaction,
            a.severity = row.severity,
            a.recorded_date =
                CASE WHEN row.recorded_date IS NULL THEN NULL
                     ELSE date(row.recorded_date) END
        MERGE (p)-[:{schema.HAS_ALLERGY}]->(a)
        """,
        rows,
    )
    return len(rows)


async def project_diagnoses(db: AsyncSession) -> int:
    stmt = select(Diagnosis, Condition.key).join(
        Condition, Condition.id == Diagnosis.condition_id
    )
    rows = [
        {
            "id": d.id,
            "patient_id": d.patient_id,
            "encounter_id": d.encounter_id,
            "condition_key": key,
            "code": d.code,
            "diagnosed_date": d.diagnosed_date.isoformat(),
            "rank": d.rank,
        }
        for d, key in (await db.execute(stmt)).all()
    ]
    await _write(
        f"""
        UNWIND $rows AS row
        MATCH (p:{schema.PATIENT} {{id: row.patient_id}})
        MATCH (c:{schema.CONDITION} {{key: row.condition_key}})
        MERGE (d:{schema.DIAGNOSIS} {{id: row.id}})
        SET d.patient_id = row.patient_id,
            d.code = row.code,
            d.diagnosed_date = date(row.diagnosed_date),
            d.rank = row.rank
        MERGE (p)-[:{schema.DIAGNOSED}]->(d)
        MERGE (d)-[:{schema.OF_CONDITION}]->(c)
        WITH d, row
        MERGE (e:{schema.ENCOUNTER} {{id: row.encounter_id}})
        MERGE (e)-[:{schema.MADE_DIAGNOSIS}]->(d)
        """,
        rows,
    )
    return len(rows)


#: Per-patient clinical nodes and the edge that says whose they are. Each of
#: these belongs to exactly one patient, which is what makes reconciliation
#: possible: a second such edge into the same node is provably stale.
#: ``Condition`` is absent on purpose — it is a shared catalogue entry and
#: many patients legitimately point at one.
OWNED: tuple[tuple[str, str], ...] = (
    (schema.ENCOUNTER, schema.HAD_ENCOUNTER),
    (schema.MEDICATION, schema.TAKES),
    (schema.LAB_RESULT, schema.HAS_LAB_RESULT),
    (schema.PROCEDURE, schema.HAD_PROCEDURE),
    (schema.ALLERGY, schema.HAS_ALLERGY),
    (schema.DIAGNOSIS, schema.DIAGNOSED),
)


async def reconcile() -> dict[str, int]:
    """Delete ownership edges PostgreSQL no longer supports.

    ``MERGE`` adds and never removes, so a projection run on a database whose
    rows have moved leaves the old edges in place. That is not a tidiness
    problem. Every clinical node carries ``patient_id`` from its source row,
    and the owning edge is drawn from the patient that row names — so when
    medication 8 moves from one patient to another between seeds, the graph
    ends up with *both* edges, and the traversal anchored on the wrong
    patient returns a prescription that is not theirs.

    That is exactly the cross-patient read §21 forbids, arriving through the
    back door: the traversals are still correct, the isolation invariant
    still holds, and the data underneath them is wrong.
    ``test_two_patients_get_their_own_subgraphs`` caught it on real data
    after a reseed, which is the only way it could have been caught — nothing
    about the query layer is broken.

    Cheap enough to run unconditionally: it touches only nodes whose owning
    edge disagrees with the ``patient_id`` already stored on them.
    """
    driver = get_driver()
    removed: dict[str, int] = {}
    async with driver.session(database=settings.neo4j_database) as session:
        for label, relationship in OWNED:
            result = await session.run(  # type: ignore[arg-type]
                f"""
                MATCH (p:{schema.PATIENT})-[stale:{relationship}]->(n:{label})
                WHERE n.patient_id IS NOT NULL AND p.id <> n.patient_id
                DELETE stale
                RETURN count(*) AS n
                """
            )
            record = await result.single()
            count = record["n"] if record else 0
            if count:
                removed[f"-[:{relationship}]->"] = count
    return removed


async def sweep(kept: dict[str, list[int]]) -> dict[str, int]:
    """Delete per-patient nodes whose source row is gone.

    The other half of making "rebuildable from PostgreSQL" true. Reconcile
    fixes a node that moved; this removes one that was deleted, which no
    ``MERGE`` can do because the projection never sees the absent row.

    Scoped to the per-patient labels. Catalogue nodes — Condition, Lab,
    Provider, Department — are left alone: they are shared, and an orphaned
    catalogue entry is harmless where an orphaned prescription is not.
    """
    driver = get_driver()
    removed: dict[str, int] = {}
    async with driver.session(database=settings.neo4j_database) as session:
        for label, _ in OWNED:
            result = await session.run(  # type: ignore[arg-type]
                f"MATCH (n:{label}) WHERE NOT n.id IN $kept "
                "DETACH DELETE n RETURN count(*) AS n",
                {"kept": kept.get(label, [])},
            )
            record = await result.single()
            count = record["n"] if record else 0
            if count:
                removed[label] = count
    return removed


async def verify() -> dict[str, int]:
    """Count what is in the graph, per label and relationship type."""
    driver = get_driver()
    counts: dict[str, int] = {}
    async with driver.session(database=settings.neo4j_database) as session:
        for label in schema.NODE_LABELS:
            result = await session.run(f"MATCH (n:{label}) RETURN count(n) AS n")  # type: ignore[arg-type]
            record = await result.single()
            counts[label] = record["n"] if record else 0
        for rel in schema.RELATIONSHIP_TYPES:
            result = await session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n")  # type: ignore[arg-type]
            record = await result.single()
            counts[f"-[:{rel}]->"] = record["n"] if record else 0
    return counts


#: Per-patient label → the table whose surviving ids define it.
_SOURCE_OF: tuple[tuple[str, Any], ...] = (
    (schema.ENCOUNTER, Encounter),
    (schema.MEDICATION, Medication),
    (schema.LAB_RESULT, LabResult),
    (schema.PROCEDURE, Procedure),
    (schema.ALLERGY, Allergy),
    (schema.DIAGNOSIS, Diagnosis),
)


async def build(*, do_reset: bool) -> dict[str, int]:
    if do_reset:
        await reset()
    await apply_schema()

    async with AppSession() as db:
        # Order matters: edges MATCH nodes that must already exist. Patients,
        # providers and conditions first, then everything that hangs off them.
        counts = {
            "patients": await project_patients(db),
            "providers": await project_providers(db),
            "conditions": await project_conditions(db),
            "patient_conditions": await project_patient_conditions(db),
            "encounters": await project_encounters(db),
            "medications": await project_medications(db),
            "lab_results": await project_lab_results(db),
            "procedures": await project_procedures(db),
            "allergies": await project_allergies(db),
            "diagnoses": await project_diagnoses(db),
        }
        kept = {
            label: list((await db.scalars(select(model.id))).all())
            for label, model in _SOURCE_OF
        }

    # After projecting, not before: reconcile compares each node's owning
    # edge against the ``patient_id`` this run just wrote, so it has to see
    # the current values. Running it first would reconcile against the
    # previous projection and leave today's drift in place.
    for name, count in (await reconcile()).items():
        counts[f"reconciled {name}"] = count
    for name, count in (await sweep(kept)).items():
        counts[f"swept {name}"] = count
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="MERGE into the existing graph instead of rebuilding it. Faster, "
        "and reconciles the ownership edges that carry patient scope — but a "
        "stale catalogue edge (OF_LAB, ORDERED) can survive it, because the "
        "node does not store the key its edge was drawn from. Prefer the "
        "default unless the rebuild is too slow to sit through.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=argparse.SUPPRESS,  # the default since rebuilding became default
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="report node and relationship counts without rebuilding",
    )
    args = parser.parse_args(argv)

    async def run() -> int:
        try:
            if args.verify:
                for name, count in (await verify()).items():
                    print(f"  {name:<24} {count:>7}")
                return 0

            counts = await build(do_reset=not args.incremental)
            print("Projected PostgreSQL into Neo4j:")
            for name, count in counts.items():
                print(f"  {name:<24} {count:>7}")
            print("\nGraph contents:")
            for name, count in (await verify()).items():
                print(f"  {name:<24} {count:>7}")
            return 0
        except GraphUnavailable as exc:
            print(f"Graph unavailable: {exc}", file=sys.stderr)
            print(
                "Is Neo4j running?  docker compose up -d neo4j",
                file=sys.stderr,
            )
            return 1
        finally:
            await dispose_driver()

    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
