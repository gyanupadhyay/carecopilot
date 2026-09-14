# Fine-tuning

CareCopilot trains a LoRA adapter on **behaviour**, never on patient data.
This document covers what that means, how to run the pipeline, and what this
repository does and does not claim about the result.

## The constraint

> Fine-tune behaviour, never patient data. — PRD §22, Principle 10

This is not a stylistic preference. Every authorization guarantee in this
project sits in front of the *database*: row-level security, the
`user_patient_mapping` table, tools that take no patient identifier, the
SELECT-only `carecopilot_ro` role. **Model weights are behind none of them.**

A value that reaches the training set leaves through any prompt that asks
for it, and nothing downstream can take it back. You cannot revoke a
memorised lab result, you cannot scope it to one user, and you cannot audit
who read it.

So the corpus teaches decisions:

| Task | What it teaches | Train / Val / Test |
|---|---|---|
| `routing` | Question → which of seven routes | 208 / 38 / 118 |
| `graph_planning` | Question → graph intent + search term | 24 / 2 / 19 |
| `refusal` | When to decline, and when *not* to | 22 / 8 / 4 |
| `grounding` | Answer from supplied context or say it is absent | 11 / 2 / 3 |
| `action_parsing` | Scheduling sentence → four typed fields | 24 / 4 / 3 |

490 examples from 45 templates. No patient name, no external id, no lab
value, no prescription.

`backend/tests/integration/test_finetuning_dataset.py` asserts that against
the **live database** — it pulls the real names, ids and clinical values out
of PostgreSQL and fails if any appears in the corpus. Checking the generator
instead would only prove that the word lists someone wrote contain no
patient data, which is true by construction and answers a different
question. The failure it is built to catch is a future edit that reads from
the database "just to make the examples more realistic" — a reasonable
sounding change, and the exact prohibition.

## The split is by template, not by row

The decision most worth understanding here, because the obvious alternative
is wrong in a way that flatters the result.

Every example comes from a template with slots. A random row split puts the
**same template** in train and test with different fillers — so the test set
measures whether the model memorised a sentence pattern it was trained on.
The tuned model then posts a large gain that evaporates on any phrasing the
templates did not cover.

Splitting on `template_id` means the test set contains phrasings the model
has never seen. It reports a smaller improvement and a true one.
`test_no_template_spans_two_splits` fails the build if that ever stops
holding.

The consequence is visible in the table above: the test split has more
routing rows than val, because template sizes differ wildly. That is the
correct trade and not a bug — an even row split would require breaking
templates across splits, which is the thing being avoided.

## Running it

```bash
# 1. Generate the corpus and the split
python scripts/build_finetuning_dataset.py

# 2. Measure the base model on the held-out split
python scripts/analyze_base_failures.py --verbose

# 3. Check the training environment (reports what is missing)
python scripts/finetune_lora.py --check

# 4. Train (needs a CUDA GPU)
python scripts/finetune_lora.py             # bf16, ~24GB
python scripts/finetune_lora.py --qlora     # 4-bit base, ~16GB

# 5. Score the adapter on the same held-out split, then the full eval set
python scripts/analyze_base_failures.py --model data/fine_tuning/adapter
python scripts/compare_models.py --models Qwen/Qwen3-8B,data/fine_tuning/adapter
```

## Design decisions

**LoRA, not a full fine-tune.** The behaviours are narrow — pick a label
from an enum, extract a term, decline, narrate supplied facts. None needs
new knowledge; the model already has the capability and lacks consistency.
A rank-16 adapter over the attention and MLP projections moves that, trains
on one consumer GPU, and can be removed. A full fine-tune costs an order of
magnitude more and risks catastrophic forgetting of the general competence
that grounding and refusal depend on.

**Loss is computed on the assistant turn only.** The single most
consequential detail in `finetune_lora.py`. Each example is a system prompt,
a question and a target — and the system prompt is the longest part by far,
identical across every routing row. Training on all tokens means most of the
gradient teaches the model to reproduce a prompt it will always be *given*,
while the behaviour that matters is a rounding error in the loss. Prompt
tokens are masked to `-100`.

**Training prompts are the production prompts.** The system prompt on each
routing example is imported from `app.agents.router`, not retyped. A model
tuned against a paraphrase is trained for a system that does not exist, and
the mismatch appears as an adapter that scores well in evaluation and worse
in the application — the most expensive kind of wrong, because the dataset
looks fine.

**Tasks are oversampled to their inverse frequency.** Routing generates from
the most templates and the widest vocabulary, so it is ~72% of the corpus by
raw count. Untouched, the adapter gets better at routing and quietly worse
at refusing — and the refusal regression is the one that matters and the one
least visible in a loss curve. `manifest.json` carries per-task weights;
`finetune_lora.py` applies them by oversampling, which survives gradient
accumulation where a hand-scaled loss does not.

**A refusal corpus includes things it must not refuse.** The
`refuse.not_actually` family is legitimate record questions phrased oddly. A
model trained only on things to decline learns that declining is safe, and
starts refusing "what medications am I on" when it is worded unusually.

## Baseline

Measured on the 147 held-out examples, via the same structured-decoding path
production uses:

| Task | n | Accuracy | Failure modes |
|---|---|---|---|
| `routing` | 118 | 0.983 | 2 × wrong label |
| `graph_planning` | 19 | 0.789 | 4 × wrong intent or term |
| `action_parsing` | 3 | 1.000 | — |
| `grounding` | 3 | 1.000 | — |
| `refusal` | 4 | 1.000 | — |
| **Overall** | **147** | **0.959** | |

Model: `gemma4:31b` on Ollama Cloud. Re-run against Qwen3-8B before treating
this as the baseline for a Qwen3 adapter — it is a different model, and the
number is not transferable.

Two findings from the first run were **dataset bugs, not model failures**,
and are fixed:

- A condition vocabulary carrying its own article produced `"my the
  anaemia"`, with a target term of `"the anaemia"`. The model answered
  `"anaemia"` and was marked wrong. Training that would have taught it to
  include the article, and the graph would then match nothing.
- `"What is lisinopril for?"` was labelled `KG`; the model chose
  `OUT_OF_SCOPE` with the reason "a request for general drug information
  rather than personal record data" — a fair reading, arguably the right
  one. Training a disputed label teaches the model to resolve an ambiguity
  the prompt does not justify. The phrasing now carries a possessive.

This is what the failure analysis is for. Roughly half of what it surfaced
was wrong with the *corpus*, and tuning on it would have baked both in.

## Qwen3-14B does not fit on this machine

§38 asks that "Qwen3-14B can be evaluated where hardware permits". It does
not permit, and the failure is worth recording because it is not obvious
from the model's size alone.

`qwen3:14b` pulls to 9.3 GB. The Ollama container runs under WSL2 with a
7.6 GiB memory limit, and loading the model is killed outright:

```
level=INFO source=sched.go:641 msg="Load failed"
  error="llama-server process has terminated: signal: killed"
```

Raising the container limit does not rescue it: the host has 15.8 GB total
with ~6.1 GB free while the stack is up, against a 9.3 GB resident model.
The request surfaces as a 500 from Ollama and, on a cold load, as a timeout
first — neither of which says "out of memory", which is why this is written
down.

`scripts/compare_models.py` is built, tested and ready; `--sample N` exists
precisely for CPU-bound comparisons. What is missing is a machine that can
hold both models. Hugging Face's router serves `Qwen/Qwen3-8B` and
`Qwen/Qwen3-14B` and closes this with a token and no new hardware.

## What this repository does not claim

**No tuned-versus-base numbers are published**, because the development
machine has no GPU and no `torch`/`peft`/`transformers`.

The pipeline is real, the dataset is real, the split is real, and the
baseline above was measured. The comparison is the one deliverable that
needs hardware this project does not have. `finetune_lora.py --check`
reports exactly that, and the script refuses to start rather than beginning
a run it cannot finish.

Publishing a plausible-looking comparison would be the precise failure the
evaluation module is built to prevent — a number that is arithmetically
presentable and says something false about the system — arriving through the
back door.
