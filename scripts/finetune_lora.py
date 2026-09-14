"""LoRA fine-tuning for the behavioural adapter (PRD §22, §23).

    python scripts/finetune_lora.py --check          # verify the environment
    python scripts/finetune_lora.py                  # train (needs a GPU)
    python scripts/finetune_lora.py --qlora          # 4-bit base, for 16GB
    python scripts/finetune_lora.py --push my/repo   # publish the adapter

Trains a low-rank adapter on ``data/fine_tuning/train.jsonl`` and evaluates
it on ``val.jsonl``. The base model is Qwen3-8B by default, which is what
PRD §2 names.

--------------------------------------------------------------------------
This machine cannot run it, and that is stated rather than worked around
--------------------------------------------------------------------------

Training needs a CUDA GPU and ``torch``, ``transformers``, ``peft`` and
``datasets``, none of which are in ``backend/requirements.txt`` — they are a
training-time dependency of a script, not a runtime dependency of the API,
and installing ~3GB of CUDA wheels into the application image to support a
script nobody runs in production would be the wrong trade.

``--check`` reports exactly what is missing and exits. Run it first.

**No tuned-versus-base numbers are published from this repository**, because
the development machine has no GPU. The pipeline, the dataset, the split and
the baseline all exist and are real; the comparison is the one deliverable
that needs hardware this project does not have. Reporting an untrained
comparison — or worse, plausible numbers — would be the exact failure the
evaluation module is arranged to prevent, arriving through the back door.

--------------------------------------------------------------------------
Why LoRA and not a full fine-tune
--------------------------------------------------------------------------

The behaviours being taught are narrow: pick a label from an enum, extract a
term, decline a request, narrate supplied facts. None of them requires new
knowledge, and the model already has the capabilities — what it lacks is
consistency. A rank-16 adapter over the attention and MLP projections is
enough to move that, trains on one consumer GPU, and can be removed. A full
fine-tune would cost an order of magnitude more and carry a risk LoRA does
not: catastrophic forgetting of the general competence the grounding and
refusal behaviours depend on.

QLoRA (``--qlora``) quantises the frozen base to 4-bit so an 8B model fits
in ~16GB. The adapter still trains in bf16; only the base is quantised, so
the thing being learned is not what gets rounded.

--------------------------------------------------------------------------
Loss is computed on the assistant turn only
--------------------------------------------------------------------------

The single most consequential detail in the file. Each example is a system
prompt, a user question and a target — and the system prompt is the longest
part by far, identical across every routing row. Training on all tokens
means most of the gradient teaches the model to reproduce a prompt it will
always be *given*, and the behaviour that actually matters is a rounding
error in the loss. Masking the prompt is the difference between an adapter
that learns to route and one that learns to recite the router's
instructions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fine_tuning"
ADAPTER_OUT = DATA / "adapter"

#: PRD §2's development model. The Hugging Face repo id rather than the
#: Ollama tag: training reads safetensors, and `qwen3:8b` is a GGUF that
#: transformers cannot load.
DEFAULT_BASE = "Qwen/Qwen3-8B"

#: Rank 16 over attention and MLP projections. Rank is the knob that trades
#: capacity for overfitting, and on a corpus of a few hundred behavioural
#: examples 16 is already generous — the failure to expect here is
#: memorising templates, not underfitting.
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

REQUIRED = ("torch", "transformers", "peft", "datasets", "accelerate")


@dataclass(slots=True)
class Environment:
    missing: list[str]
    torch_version: str = ""
    cuda: bool = False
    gpu: str = ""
    vram_gb: float = 0.0

    @property
    def ready(self) -> bool:
        return not self.missing and self.cuda


def inspect_environment() -> Environment:
    """What is installed, and whether a GPU is visible. Never raises."""
    missing: list[str] = []
    for package in REQUIRED:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    env = Environment(missing=missing)
    if "torch" in missing:
        return env
    import torch

    env.torch_version = torch.__version__
    env.cuda = bool(torch.cuda.is_available())
    if env.cuda:
        env.gpu = torch.cuda.get_device_name(0)
        env.vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return env


def report_environment(env: Environment) -> int:
    print("Fine-tuning environment\n")
    for package in REQUIRED:
        mark = "missing" if package in env.missing else "ok"
        print(f"  {package:<16} {mark}")
    print(f"  {'torch version':<16} {env.torch_version or '—'}")
    print(f"  {'CUDA':<16} {'yes' if env.cuda else 'no'}")
    if env.cuda:
        print(f"  {'GPU':<16} {env.gpu} ({env.vram_gb:.1f} GB)")

    if env.missing:
        print(
            "\nInstall the training extras (not in backend/requirements.txt — "
            "they are a training dependency of this script, not a runtime "
            "dependency of the API):\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install transformers peft datasets accelerate bitsandbytes"
        )
    if not env.cuda:
        print(
            "\nNo CUDA device. LoRA on an 8B model is not practical on CPU — "
            "not slow, but days-per-epoch impractical. Options: a cloud GPU, "
            "or --qlora on a 16GB card."
        )
    return 0 if env.ready else 1


def load_rows(name: str) -> list[dict[str, Any]]:
    path = DATA / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run scripts/build_finetuning_dataset.py first."
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def oversample(rows: list[dict[str, Any]], weights: dict[str, float]) -> list[dict]:
    """Repeat under-represented tasks to their manifest weight.

    Oversampling rather than a per-example loss scale, because it needs no
    cooperation from the trainer and survives gradient accumulation, which a
    hand-scaled loss quietly does not. The cost is a longer epoch; the
    alternative is an adapter that is 72% routing by gradient and forgets how
    to refuse.
    """
    if not weights:
        return list(rows)
    expanded: list[dict[str, Any]] = []
    for row in rows:
        weight = weights.get(row.get("task", ""), 1.0)
        whole, fraction = int(weight), weight - int(weight)
        expanded.extend([row] * max(1, whole))
        if fraction >= 0.5:
            expanded.append(row)
    return expanded


def build_masked_example(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict:
    """Tokenise one row, masking every token before the assistant turn.

    See the module docstring: the system prompt is most of each example and
    is identical across rows, so training on it spends the run teaching the
    model to reproduce text it is always handed. ``-100`` is the index
    PyTorch's cross-entropy ignores.
    """
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full = prompt + messages[-1]["content"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        full, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]

    labels = list(full_ids)
    for index in range(min(len(prompt_ids), len(labels))):
        labels[index] = -100
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def train(args: argparse.Namespace, env: Environment) -> int:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    manifest_path = DATA / "manifest.json"
    weights: dict[str, float] = {}
    if manifest_path.exists():
        weights = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "task_weights", {}
        )

    train_rows = oversample(load_rows("train"), weights if not args.no_weights else {})
    val_rows = load_rows("val")
    print(f"train {len(train_rows)} rows (after weighting), val {len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Qwen ships no pad token. Reusing EOS is standard and safe here
        # because the collator masks padding out of the attention mask, so
        # the model never attends to it and the loss never sees it.
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.qlora:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            # Double quantisation saves a further ~0.4 bits/param. The
            # compute dtype stays bf16: only storage is quantised, so the
            # adapter is not learning against rounded activations.
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.base, **load_kwargs)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    if args.qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(TARGET_MODULES),
        ),
    )
    model.print_trainable_parameters()

    def prepare(rows: list[dict[str, Any]]) -> Dataset:
        return Dataset.from_list(
            [build_masked_example(tokenizer, row, args.max_length) for row in rows]
        )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            # Cosine with warmup: the corpus is small, so a constant rate
            # spends the last epochs bouncing around a minimum it already
            # found, which shows up as val loss rising while train loss falls.
            lr_scheduler_type="cosine",
            warmup_ratio=0.06,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            # The checkpoint that generalises, not the last one. On a few
            # hundred examples the final epoch is usually overfit.
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],
            seed=args.seed,
        ),
        train_dataset=prepare(train_rows),
        eval_dataset=prepare(val_rows),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100
        ),
    )

    trainer.train()
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"\nAdapter written to {args.out}")

    if args.push:
        model.push_to_hub(args.push)
        tokenizer.push_to_hub(args.push)
        print(f"Pushed to https://huggingface.co/{args.push}")

    print(
        "\nNext: score it against the same held-out split the base model was "
        "scored on —\n"
        f"  python scripts/analyze_base_failures.py --model {args.out}\n"
        "and then the full evaluation set, so the comparison is over the "
        "metrics the system is actually judged by."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=ADAPTER_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what is installed and whether a GPU is visible, then exit.",
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Quantise the frozen base to 4-bit so an 8B model fits in ~16GB. "
        "The adapter still trains in bf16.",
    )
    parser.add_argument("--rank", type=int, default=LORA_RANK)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Ignore the manifest's per-task weights. The corpus is ~72% "
        "routing by raw count, so this trains routing and erodes the rest — "
        "useful only to measure that effect deliberately.",
    )
    parser.add_argument("--push", default=None, metavar="REPO")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    env = inspect_environment()
    if args.check:
        return report_environment(env)
    if not env.ready:
        report_environment(env)
        print("\nRefusing to start: the environment cannot train this model.")
        return 1
    return train(args, env)


if __name__ == "__main__":
    raise SystemExit(main())
