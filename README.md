# CareCopilot

A patient-facing medical assistant that answers questions about *your own*
record — appointments, medications, labs, visit notes, and the relationships
between them — using a self-hosted Qwen3-8B rather than a cloud API.

Every patient in this repository is synthetic. There is no real clinical data
here and no path by which any could enter.

```
┌──────────┐   JWT    ┌───────────────────────────────┐   ┌──────────────┐
│ Next.js  │ ───────► │ FastAPI  ·  LangGraph agent   │──►│ Ollama/vLLM  │
│ frontend │ ◄─────── │  classify → route → generate  │   │  qwen3:8b    │
└──────────┘   SSE    └───┬────────┬────────┬─────────┘   └──────────────┘
                          │        │        │
                  ┌───────▼──┐ ┌───▼────┐ ┌─▼──────────┐
                  │ Postgres │ │ Neo4j  │ │ carecopilot│
                  │ +pgvector│ │  (KG)  │ │ _ro (RLS)  │
                  └──────────┘ └────────┘ └────────────┘
                          ▲
                  ┌───────┴──────┐
                  │  MCP server  │  proxies the API with the caller's JWT
                  └──────────────┘  (holds no database credentials)
```

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

No API key is required. The first `up` downloads ~5GB of Qwen3-8B weights and
~130MB of embedding model, migrates the schema, generates 100 synthetic
patients, embeds their notes, and projects the whole thing into Neo4j. After
that:

| | |
|---|---|
| Frontend | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| MCP endpoint | http://localhost:8100/mcp |
| Neo4j browser | http://localhost:7474 |

Log in with any account from `GET /api/auth/demo-accounts` — e.g.
`p001@carecopilot.demo` / `carecopilot-demo`.

**On CPU, expect tens of seconds to a few minutes per answer.** Qwen3-8B
without a GPU is genuinely that slow — a measured 215s for a knowledge-graph
turn in a container on this machine. The architecture is the point, not the
throughput. For a fast demo, either uncomment the `deploy` block on the
`ollama` service for an NVIDIA GPU, or point `LLM_PROVIDER` at a hosted
endpoint that serves the same model:

```bash
LLM_PROVIDER=ollama_cloud   LLM_MODEL=qwen3:8b       # ollama.com/settings/keys
LLM_PROVIDER=huggingface    LLM_MODEL=Qwen/Qwen3-8B  # hf.co/settings/tokens
```

Same weights, someone else's hardware — about 3s per turn rather than 215s.
The trace records the provider beside the model, so a measurement never
loses the context of where it ran. Note that Ollama Cloud's catalogue is
hosted models only and **does not include `qwen3:8b`**; Hugging Face's router
serves both Qwen3-8B and Qwen3-14B, which is what §28's comparison needs.

## Architecture

### The turn

Every question goes through one LangGraph workflow (`app/agents/graph.py`):

```
START → classify_query → route ──┬─ API          → execute_api_tool
                                 ├─ RAG          → retrieve
                                 ├─ KG           → query_graph
                                 ├─ HYBRID       → hybrid
                                 ├─ TEXT_TO_SQL  → text_to_sql
                                 ├─ ACTION       → action
                                 └─ OUT_OF_SCOPE → out_of_scope
      → generate_answer → validate_result → END
```

Seven routes, each a different way of being right about a different kind of
question:

- **API** — "when is my next appointment?" Structured lookups through eight
  `get_my_*` tools, with the model choosing which ones the question needs
  (§5). No retrieval, no generation of facts; the model narrates rows the
  backend fetched.
- **RAG** — "what did the cardiologist say about my chest pain?" Hybrid
  retrieval over chunked clinical notes: pgvector similarity plus keyword
  search, fused with RRF, deduplicated, then reranked (heuristic or LLM).
- **KG** — "why was I prescribed metformin?" One of ten approved Cypher
  traversals over a Neo4j projection of 11 entity types and 17 relationship
  types.
- **HYBRID** — questions needing both structured rows and note text.
- **TEXT_TO_SQL** — "how many lab tests have I had this year?" Aggregates the
  other routes cannot express.
- **ACTION** — "book me a follow-up next Tuesday." Proposes a write; never
  performs one.
- **OUT_OF_SCOPE** — refuses, with a reason.

`generate_answer` is the only node that produces prose, and it narrates facts
the backend established rather than recalling them. `validate_result` runs
over every answer whichever branch produced it.

### LangChain is deliberately absent

§8 and §31 list "selective LangChain"; §12 says it is optional and adds *do
not introduce LangChain abstractions where normal Python is clearer*. Those
pull in opposite directions, and this repository resolves it toward §12: it
has no LangChain dependency at all.

The five uses §12 offers were each considered against what is already here.
LLM integrations and structured outputs are `app/llm/` — one interface over
five endpoints, with schema negotiation a wrapper would have hidden exactly
when it broke (see the Ollama Cloud note below). Embeddings are `fastembed`
called directly. Retrievers and tool abstractions are `app/rag/` and
`app/tools/`, both of which take an `AuthContext` as a required argument —
and a retriever interface that has no place to put one is a poor trade for a
few lines saved.

So the honest claim is "LangGraph for orchestration, no LangChain", not
"selective LangChain usage". Adding the dependency to match the PRD's word
list would be the tail wagging the dog.

### What LangGraph does and does not do

It owns state merge, the conditional branch, and execution order. It is never
on the authorization path: every data access goes through a service taking an
`AuthContext` the graph merely transports. Deleting the graph would change how
a turn is orchestrated and nothing about who can read what.

### Authorization

Four properties, each structural rather than a rule someone has to remember:

1. **Identity is a mapping table.** `user_patient_mapping`, not
   `users.patient_id` — a user may be linked to a record, or not, and the link
   can be deactivated without deleting the user.
2. **No patient id ever appears in a URL.** `/api/labs`, never
   `/api/patients/{id}/labs`. The id in the path is the anti-pattern; the
   patient is resolved from the token.
3. **No tool takes a patient identifier.** Not the eight clinical tools, not
   the graph traversals, not the analytics capability. `ctx.patient_scope` is
   the single binding site, so "the LLM cannot choose whose record to read" is
   a property of the function signatures.
4. **JWT validation checks `iss` and `aud`**, not just signature and expiry. A
   correctly-signed token minted for another audience is rejected.

Row-level security in PostgreSQL sits underneath all four.

### Text-to-SQL

Four layers, in order: schema (an allowlist of four tables and their columns)
→ generator → validator → executor.

The validator parses with `sqlglot` rather than pattern-matching for
dangerous strings, because a blocklist of `DROP`/`DELETE`/`;` is a list of the
attacks someone thought of. But the validator is the *fourth* layer, not the
boundary. The boundary is the `carecopilot_ro` role: SELECT-only grants on
four tables, `default_transaction_read_only`, and row-level security keyed to
`app.patient_id`, which the analytics session sets from the `AuthContext`. It
owns nothing, so it cannot bypass its own policies.

Generated SQL must **not** filter on `patient_id`. RLS already scopes the
connection; such a predicate can only wrongly exclude the patient's own rows.

### Actions: propose, then confirm

Two requests, deliberately.

`POST /api/actions/propose` parses the sentence into a four-field
`AppointmentRequest` — and nothing else reaches the backend from the model.
The backend validates against real availability, writes an audit row, and
mints a token signed with a separate secret under its own JWT audience, bound
to user and patient, expiring in five minutes.

`POST /api/actions/confirm` accepts a token and no free text. No sentence
anywhere reaches the INSERT. Idempotency is enforced through the audit log,
since the token itself is stateless.

**`confirm` is deliberately absent from the MCP surface.** A caller that could
both propose and confirm collapses the two-step design into one, and "can a
prompt injection make the assistant book something?" stops having a structural
answer. `test_no_tool_can_confirm_an_action` asserts both halves: no
confirm-shaped tool name, and no tool taking a `token`.

### Knowledge graph

Neo4j holds a **derived** projection of PostgreSQL — never business truth of
its own. Rebuild it any time with `python scripts/build_kg.py --reset`.

Isolation is *not* a property of the projection's shape. Condition, Lab,
Department and Provider nodes are deliberately shared between patients, so
cross-patient paths exist by construction. What enforces isolation is that no
approved traversal mentions `:Patient` twice — asserted mechanically in
`test_knowledge_graph.py`, because that is exactly what a well-meaning edit
breaks.

Two things the graph exposed that were invisible before it:
`medications.condition_id` is not redundant with the encounter's (one visit
routinely starts therapy for several conditions), and `conditions.aliases` is
load-bearing (patients say "blood pressure", the catalogue says "Essential
hypertension", and neither is a substring of the other).

### MCP

`mcp-server/server.py` exposes twelve tools and **holds no database
credentials**. It proxies the backend API with the caller's own JWT, which is
what makes "an MCP client cannot bypass authorization" structural rather than
enforced by the tool implementations.

### Inference

`app/llm/openai_compatible.py` serves ollama, vllm, gemini, groq and openai
from one implementation, because all five speak Chat Completions. Switching
providers changes no application logic.

Two things about Qwen3 worth knowing before changing anything here:

- **Thinking must be disabled or capped structured calls return nothing.** The
  model spends its entire budget reasoning, stops at `length`, and yields
  empty content — the 200-token router failed on roughly one question in three
  this way. Ollama takes `reasoning: {"effort": "none"}`; vLLM takes
  `chat_template_kwargs: {"enable_thinking": false}`; each ignores the
  other's, so they are sent per provider. Also a 21× latency win on CPU.
- **`RouteDecision.route` is a `Literal`, not `str`**, so the JSON Schema
  carries an enum the server constrains decoding to. An 8B model otherwise
  invents plausible labels ("appointment", "Appointment Inquiry") that a
  post-hoc check then drops to a RAG fallback.

## Running it

### Docker (recommended)

```bash
docker compose up --build       # or: make up
docker compose down             # make down; ARGS=-v also drops the database
docker compose logs -f          # make logs
```

Four services run to completion and exit, in order, before the API starts:
`init` (migrate → seed → embed), `init-kg` (project into Neo4j),
`ollama-pull` (fetch weights). Each is idempotent and runs on every `up`, so a
container that is up is a container whose schema, index and graph match its
code. Folding these into the API entrypoint would make every replica race to
migrate the same database.

### Locally

Needs Python 3.14, Node 20+, PostgreSQL 18 with pgvector, and Neo4j 5.

```bash
make install          # venv + backend dependencies
make bootstrap        # roles, database, extensions (needs a superuser)
make reset            # migrate + seed
python scripts/ingest_documents.py
python scripts/build_kg.py
make run              # API on :8000
make web-install && make web   # frontend on :3000
```

## Evaluation

```bash
python scripts/run_evaluation.py                 # all 56 cases
python scripts/run_evaluation.py --only kg       # one category
python scripts/run_evaluation.py --delay 5       # paced, for a rate-limited tier
```

56 cases across six categories (20 API, 15 RAG, 6 KG, 5 hybrid, 5 Text-to-SQL,
5 safety), scored on routing, retrieval, generation, agent, KG, SQL, security
and system metrics. Every §27 category minimum is met. Reports land in
`evaluation/reports/`.

**These numbers are not Qwen3's.** The run below used `gemma4:31b` on Ollama
Cloud, because a 56-case run against Qwen3-8B on this machine's CPU takes
about an hour and was not something to sit through repeatedly while the
scoring code was still changing. The architecture, the corpus, the retrieval
and the scoring are identical either way — but a routing accuracy measured on
a 31B model says nothing about what an 8B model does, and §28 exists
precisely to stop that substitution passing unnoticed. Treat the table as
"the system works end to end", not as a Qwen3 benchmark; the Qwen3 column is
the missing work §28 describes.

Latest full run — `gemma4:31b` on Ollama Cloud, 55/55 cases, 0 errors,
0 degraded. It predates `api-016`, so it covers 55 of the current 56:

| Group | Metric | |
|---|---|---|
| Routing | router accuracy | 1.000 |
| Retrieval | recall@5 / precision@5 / MRR | 0.569 / 0.633 / 0.917 |
| Generation | answer correctness | 1.000 |
| | citation correctness | 0.633 |
| | grounding rate | 1.000 |
| | faithfulness *(self-judged)* | 0.967 |
| Knowledge graph | entity resolution / relationship / multi-hop | 1.000 |
| Text-to-SQL | validity / execution / correctness / authorization | 1.000 |
| Agent | tool selection accuracy | 1.000 |
| | tool selection precision | 0.981 |
| | mean tool calls per turn | 0.62 |
| | call success / argument validity / JSON validity | 1.000 |
| Security | leak-free rate | 1.000 |
| System | latency mean / p95 | 7.35s / 10.81s |

**Read the agent rows with the denominators.** Tool-selection accuracy is
scored over 26 cases, not 55, and two API cases are deliberately excluded
because their questions admit more than one correct tool — scoring those
would measure which reading the dataset author picked. Retrieval's 0.569
recall is the weakest number here and is not hidden: the corpus is
synthetic and several notes for one condition are near-duplicates, so the
"right" chunk is frequently one of several equally good ones.

Nothing is mocked. Security cases are not simulated — `safety-001` really does
ask P001's authenticated session for P002's records, and the result is
whatever the system really returns.

**Two honesty rules run through the scoring**, and they exist because both
were violated by real bugs:

- *A metric that cannot be computed returns `None`, never zero.*
  `answer_correctness` once reported 0.000 computed from stub text, which
  reads as "the model omitted every required fact" when the truth was "no
  model ran".
- *Degraded cases are excluded, not counted.* `leak_free_rate` once reported
  1.000 because every safety case had degraded to "I could not reach the
  assistant service" — a string containing no forbidden term. A safety metric
  that improves when the system stops working is worse than no metric.

`CaseResult.degraded` marks a case where no model participated; every
model-dependent metric scores over `judgeable` cases only, and
`cases_degraded` is reported so a thin run announces itself.

### Model comparison

```bash
python scripts/compare_models.py --models qwen3:8b,qwen3:14b
python scripts/compare_models.py --models Qwen/Qwen3-8B,Qwen/Qwen3-14B \
    --provider huggingface --sample 12
```

Runs the same cases against each model in turn and writes a side-by-side
table with a delta column. Only the model changes between columns — provider,
embedder, reranker and top-k are held constant, because a column served by
different hardware cannot be compared on latency.

`--sample N` compares over N cases stratified across all six categories
instead of all 55, for CPU-bound runs. The report records `cases_compared`
and `sampled`, so a subset cannot be mistaken for a full run.

**No 8B-vs-14B table is published here.** `qwen3:14b` is 9.3 GB and is
OOM-killed on this machine (7.6 GiB container limit, ~6.1 GB host RAM free).
The harness is built and tested; it needs a machine that can hold the model,
or a hosted provider that serves both. See
[docs/fine-tuning.md](docs/fine-tuning.md).

## Fine-tuning

A LoRA pipeline over a synthetic **behavioural** instruction set — 490
examples across routing, graph-intent selection, refusals, grounding and
action parsing. It contains no patient name, no external id and no record
value, and `test_finetuning_dataset.py` asserts that against the live
database rather than by inspection: it pulls the real names and values out
of PostgreSQL and fails if any appears in the corpus.

```bash
python scripts/build_finetuning_dataset.py     # generate + split
python scripts/analyze_base_failures.py        # where the base model fails
python scripts/finetune_lora.py --check        # what this machine can do
python scripts/finetune_lora.py                # needs a CUDA GPU
```

The split is **by template, not by row** — a random row split puts the same
template in train and test, so the test set measures template memorisation
and the tuned model posts a gain that evaporates on any new phrasing.

Baseline on the 147 held-out examples: **0.959 overall** (routing 0.983,
graph planning 0.789). Roughly half of what the failure analysis surfaced was
wrong with the *corpus* rather than the model, and tuning would have baked it
in — see [docs/fine-tuning.md](docs/fine-tuning.md).

**No tuned-versus-base numbers are published**: training needs a GPU and
`torch`/`peft`/`transformers`, which this machine does not have.
`finetune_lora.py --check` says so and the script refuses to start rather
than beginning a run it cannot finish.

## Testing

```bash
make test     # pytest
make lint     # ruff
make check    # both
```

Integration tests run against a real PostgreSQL with real RLS policies; the
LLM is stubbed there, and the cases that need a real model are the evaluation
set's job, not pytest's.

## Repository layout

```
backend/
  app/
    agents/          LangGraph workflow, router, nodes
    api/routes/      FastAPI endpoints
    auth/            JWT, AuthContext, demo accounts
    knowledge_graph/ Neo4j client, approved traversals, projection schema
    llm/             providers; openai_compatible serves five of them
    rag/             chunking, embeddings, retrieval, fusion, reranking
    sql/             Text-to-SQL: schema → generator → validator → executor
    actions/         propose/confirm appointment writes
    guardrails/      output validation
  alembic/versions/  migrations 0001–0011
  tests/             unit, integration, security
data/
  synthetic/         generated patient corpus exports
  fine_tuning/       instruction set, splits, adapters
docker/              Dockerfiles and database init scripts
evaluation/          dataset, metrics, runner, reports
frontend/            Next.js App Router, TypeScript strict, CSS modules
mcp-server/          MCP server (no database credentials)
ollama/              Modelfile for the served model
scripts/             data generation, ingestion, KG build, evaluation
```

### Where this differs from PRD §35

§35 sketches a tree; this is that tree with six names resolved differently.
Each one is a placement decision, so each is worth being able to defend:

| §35 names | Lives at | Why |
|---|---|---|
| `app/text_to_sql/` | `app/sql/` | Shorter, and the package is the four-layer pipeline rather than the prompt technique |
| `app/mcp/` | `mcp-server/` | §35 lists *both*; the server holds no database credentials, and a package inside the backend would sit next to the ones it must not import |
| `app/evaluation/` | `evaluation/` | It imports the app, not the reverse. Shipping the eval harness inside the deployed image would ship the answer key with the product |
| `app/fine_tuning/` | `scripts/` + `data/fine_tuning/` + `docs/fine-tuning.md` | A pipeline run by hand on a GPU box, not a module the API imports |
| `app/repositories/` | `app/services/` | One layer, not two. A repository layer that every service passes straight through is indirection without a caller |
| `app/authorization/` | `app/auth/context.py` | Authentication and authorization are 200 lines together; splitting them into two packages would hide how short the authorization rule actually is |

## Design decisions worth knowing

**PostgreSQL 18 mounts at `/var/lib/postgresql`, not `/var/lib/postgresql/data`.**
The image now stores data in a major-version subdirectory so `pg_upgrade
--link` can run without crossing a mount boundary. Mounting at the old path
makes the entrypoint refuse to start.

**Next 16, not 15.5.4**, which carries CVE-2025-66478.

**The JWT lives in `sessionStorage` and travels as a bearer header**, so there
is no cookie and therefore no CSRF surface. SSE is read with `fetch` plus a
manual frame parser, because `EventSource` can neither POST nor send an
`Authorization` header.

**Route vocabulary lives in three places that must agree**: `models/enums.ROUTES`,
`schemas/chat.Route`, and a CHECK constraint on `messages.route` /
`request_traces.route`. Widening it needs a migration, or the turn fails on
INSERT *after* the work has been done.

**Traces record shape, never content** — counts, durations and identifiers,
never the question, the retrieved text or the answer. The developer panel
shows timings and token counts; it never shows reasoning.

## Limitations

- **The HYBRID route's tool plan is fixed**, while the API route's is chosen
  by the model (§5). That is a distinction rather than an inconsistency:
  HYBRID's `get_my_last_encounter` produces the encounter id that anchors its
  retrieval, so those two tools are a data dependency of the route rather
  than a selection for the question. The API route genuinely chooses, and a
  bad choice there costs a worse answer — never a wider one, since no tool
  accepts a patient identifier.
- **Faithfulness** is scored by an LLM judge running the same local model that
  produced the answer. A judge and a generator sharing weights share blind
  spots; treat the number as a floor.
- **`answer_correctness` is keyword containment**, a deliberate proxy. It
  catches the failure that matters most here — an answer that never states the
  fact it was asked for — and cannot tell a right fact in a right sentence
  from a right fact in a wrong one.
- **No tuned-vs-base numbers are published** from this machine: it has no GPU.
  The pipeline and dataset exist and the baseline is measured; the comparison
  is not fabricated.
</content>
