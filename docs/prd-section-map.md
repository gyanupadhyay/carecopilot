# PRD section map

Every `PRD §N` citation in this repository refers to **CareCopilot PRD (2)**,
which has 41 sections. This file records how the codebase got there, because
the remap was a judgement exercise rather than a lookup and someone should be
able to disagree with it.

## Why it was needed

The codebase was written against PRD (1). That document was deleted, PRD (2)
is not a renumbering of it, and no mapping table ever existed. Citations were
corrected piecemeal as files were touched, which left the codebase *mixed* —
arguably worse than consistently stale, because a reader could no longer tell
which numbering any given comment used.

Remapped on 2026-09-14: **145 citations across 128 files**. The tell that
made it tractable: PRD (2) has 41 sections, so every citation above §41
(§42–§56, 14 of them) was provably stale, and the rest could be checked by
reading the comment's intent against PRD (2)'s actual section content.

Verified afterwards: no citation anywhere in the repository is now outside
§1–§41.

## PRD (2) sections

| § | Title | § | Title |
|---|---|---|---|
| 1 | Product Overview | 22 | Fine-Tuning |
| 2 | Primary LLM Strategy | 23 | Fine-Tuning Pipeline |
| 3 | Model Serving Strategy | 24 | Prompt Injection |
| 4 | LLM Abstraction | 25 | Output Validation |
| 5 | Model Responsibilities | 26 | Observability |
| 6 | Overall Architecture | 27 | Evaluation |
| 7 | Goals | 28 | Model Evaluation |
| 8 | Scope and Non-Goals | 29 | Local Development |
| 9 | Authentication and Authorization | 30 | Production-Style Inference |
| 10 | AuthContext | 31 | Backend Stack |
| 11 | LangGraph | 32 | Data Model |
| 12 | LangChain | 33 | KG Synchronization |
| 13 | Agent State | 34 | Synthetic Data |
| 14 | Agent Routing | 35 | Project Structure |
| 15 | MCP Architecture | 36 | Development Priorities |
| 16 | MCP Text-to-SQL | 37 | Demo Scenarios |
| 17 | Knowledge Graph | 38 | Acceptance Criteria |
| 18 | KG vs RAG vs SQL | 39 | Production-Style Definition |
| 19 | RAG | 40 | Final Architecture Principles |
| 20 | RAG Authorization | 41 | Target Interview Architecture |
| 21 | KG Authorization | | |

## Where each topic now cites

| Topic | § | Note |
|---|---|---|
| Chunking, embedding, retrieval, fusion, reranking, context budget | 19 | §19 carries both the offline and online RAG flows |
| The authorization filter on retrieval | 20 | |
| Generated SQL: schema allowlist, generator, validator, executor, audit | 16 | §16 lists scope, schema, read-only, validation, timeouts, row limits, audit |
| Tool protocol, tool surface, `get_my_*`, the MCP server | 15 | |
| Graph workflow, node ceiling, human confirmation | 11 | §11 lists tool selection, retries and human confirmation as LangGraph's job |
| `AgentState` and its fields | 13 | |
| Route vocabulary, the router, `RouteDecision` | 14 | |
| JWT, `iss`/`aud`, no patient id in a URL, identity mapping | 9, 10 | |
| Traces, token counts, cost, developer panel, no chain-of-thought | 26 | The single largest destination, 40 citations |
| Guardrails, grounding, citation checks | 25 | |
| Provider abstraction, one interface for five endpoints, bounded retry | 4 | |
| Evaluation dataset, metrics, scoring | 27 | |
| 8B-vs-14B comparison | 28 | |
| Synthetic-data banner, disclaimer, generator provenance | 34 | |
| Demo sequences | 37 | |
| Acceptance criteria ("user can chat", cross-patient access) | 38 | |
| `§44 P<n>` principles | 40 P*n* | PRD (2)'s principles section; numbering preserved |

## Judgement calls worth challenging

Four places where the mapping is defensible rather than obvious. Each is
recorded here rather than buried in a diff.

**Actions → §15 + §11, not a section of their own.** PRD (2) has no
"Actions" section. §15 names `book_my_appointment` and
`cancel_my_appointment` among the MCP tools, and §11 lists "Human
confirmation" among LangGraph's responsibilities — so the propose/confirm
split cites both: the tools at §15, the confirmation step at §11.

**Streaming → §6.** PRD (2) has no streaming section. SSE is part of how the
chat path is wired, so it cites Overall Architecture. The *chain-of-thought*
half of the old §47 citations went to §26 instead, which is where the
prohibition actually lives — those were two different claims sharing one
stale number.

**Conversation memory → §32.** Also has no section of its own. The summary
columns are data model, so the tables, the service and the prompt all cite
§32. That is a weaker fit than the others here.

**"The backend computes it, not the model" → §40 P12.** These cited PRD (1)
§31, which in PRD (2) is Backend Stack — clearly wrong. The claim is a
principle, not a stack choice, so it moved to Final Architecture Principles.

## Keeping it true

There is no automated check that a citation points at the *right* section —
that needs a reader. There is one cheap check worth running after any edit,
which catches the whole class of error this remap existed to fix:

```bash
grep -rn "§[0-9]\+" --include=*.py --include=*.ts --include=*.tsx \
  --include=*.md --include=*.yml --include=*.txt --include=*.toml \
  --include=Makefile --include=Dockerfile . | grep -v node_modules |
  grep -o "§[0-9]\+" | sed 's/§//' | sort -n | uniq | awk '$1 > 41'
```

Anything it prints is a citation to a section PRD (2) does not have — except
the ones inside this file, which quote the old numbering on purpose. Check
the file names, not just the numbers:

```bash
grep -rn "§4[2-9]\|§5[0-9]" --include=*.py --include=*.ts --include=*.tsx \
  --include=*.md --include=*.yml --include=*.txt . | grep -v node_modules
```

**The extension list is the whole check.** The original version scanned five
extensions, and a stale `§44 P5` sat in `backend/requirements.txt` for as
long as the check existed, reported clean every time it ran. A citation goes
wherever a comment goes, which is more file types than one remembers while
writing the glob — `.txt`, `.toml`, `Makefile` and the Dockerfiles all carry
comments here. Widen it again rather than trusting that this list is final.
