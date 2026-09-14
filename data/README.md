# data/

PRD §35's layout. Both directories hold **generated** artefacts, and neither
is committed — see `.gitignore`.

## `synthetic/`

Exports of the synthetic patient corpus: 100 patients, their encounters,
medications, lab results, procedures, allergies, diagnoses and clinical
notes. PostgreSQL is the system of record for all of it; this directory is
for exports, fixtures and anything that wants the corpus as files.

Regenerate with:

    python scripts/generate_data.py --reset

Every patient here is invented. Nothing in this repository has ever held
real clinical data, and the generator is the only thing that writes it.

## `fine_tuning/`

The behavioural instruction set, its splits, and any trained adapter.

    python scripts/build_finetuning_dataset.py

- `train.jsonl`, `val.jsonl`, `test.jsonl` — chat-format examples
- `manifest.json` — counts, per-task weights, and the split rationale
- `adapter/` — LoRA weights, once something has trained them

**This directory must never contain patient data.** Not a name, not an
external id, not a lab value. Model weights sit behind none of the
authorization this project relies on, so anything trained into them leaves
through any prompt that asks. `backend/tests/integration/test_finetuning_dataset.py`
asserts it against the live database on every test run.

See [docs/fine-tuning.md](../docs/fine-tuning.md).
