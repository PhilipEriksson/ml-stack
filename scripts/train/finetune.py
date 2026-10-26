#!/usr/bin/env python3
"""Fine-tune a model with Unsloth SFTTrainer.

Expects a config.json from create-training-run with:
  - model_path: path to the model directory
  - dataset_path: path to the processed dataset directory (with train/ and eval/ subdirs)
  - model_type: "base" or "gguf" (optional)

The processed dataset should be in alpaca format with columns:
  - instruction: the user's instruction/prompt
  - input: additional context (optional, can be empty string)
  - output: the expected assistant response
"""

import sys
import os
import argparse

# Check required dependencies before importing
MISSING = []
for pkg, hint in [
    ("unsloth", "pip install unsloth"),
    ("trl", "pip install trl"),
    ("bitsandbytes", "pip install bitsandbytes"),
]:
    try:
        __import__(pkg)
    except ImportError:
        MISSING.append(f"  - {pkg} (install: {hint})")

if MISSING:
    print("❌ Missing required package(s):")
    print("\n".join(MISSING))
    print("")
    print("Try: conda env create -f envs/training.yml (or envs/training-cuda13.yml)")
    sys.exit(1)

from unsloth import FastLanguageModel
from datasets import load_from_disk
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
import json


class AimMetricsCallback(TrainerCallback):
    """Forwards every logged metric (loss, eval_loss, lr, ...) to an Aim run."""

    def __init__(self, aim_run):
        self.aim_run = aim_run

    def _track(self, metrics, step):
        if not (self.aim_run and metrics):
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.aim_run.track(value, name=key, step=step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._track(logs, state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._track(metrics, state.global_step)

# -------------------------
# INPUT
# -------------------------
parser = argparse.ArgumentParser(description="Fine-tune a model with Unsloth SFTTrainer")
parser.add_argument("config", help="Path to run config.json")
parser.add_argument("--sample", action="store_true",
                    help="Fast test mode: 1 epoch, batch=8, sparse logging")
parser.add_argument("--max-samples", type=int, default=None,
                    help="Limit training to N samples")
args = parser.parse_args()

if not os.path.isfile(args.config):
    print("❌ Config file not found: {}".format(args.config))
    print("")
    print("  Run 'create-training-run <name> <model> <dataset>' first,")
    print("  then: python finetune.py outputs/runs/<name>/config.json")
    sys.exit(1)

with open(args.config, "r") as f:
    config = json.load(f)

model_path = config["model_path"]
dataset_path = config.get("dataset_path")
run_dir = os.path.dirname(args.config)
model_type = config.get("model_type", "base")

if not dataset_path:
    print("❌ No dataset_path in config.")
    print("   Run 'create-training-run <name> <family/model> <family/dataset>' first.")
    sys.exit(1)

# -------------------------
# WARN IF USING QUANTIZED MODEL
# -------------------------
if model_type == "gguf":
    print("❌ Cannot fine-tune GGUF models with Unsloth.")
    print("   Download a base model instead: add-model auto <hf-repo> --type base")
    sys.exit(1)

# -------------------------
# METHOD SELECTION
# -------------------------
# - qlora:  4-bit NF4 base + LoRA adapters   (default; lowest VRAM)
# - lora:   bf16 base + LoRA adapters        (fastest per step)
# - fullft: all weights in bf16, no adapters (knowledge injection / reference point)
METHOD = config.get("method", "qlora")
if METHOD not in ("lora", "qlora", "fullft"):
    print("❌ Invalid method in config: {} (expected lora | qlora | fullft)".format(METHOD))
    sys.exit(1)
print(f"🧠 Method: {METHOD}")

# -------------------------
# LORA ADAPTER CONFIG
# -------------------------
# Every knob is optional in the run config. The defaults reproduce the original
# hardcoded adapter exactly (r=16, alpha=16, dropout=0, 7 projection modules,
# plain LoRA), so existing configs keep behaving identically.
DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]
LORA_R = int(config.get("lora_r", 16))
LORA_ALPHA = int(config.get("lora_alpha", LORA_R))
LORA_DROPOUT = float(config.get("lora_dropout", 0.0))
LORA_TARGETS = config.get("lora_target_modules", DEFAULT_LORA_TARGETS)
LORA_RSLORA = bool(config.get("lora_rslora", False))
LORA_BIAS = config.get("lora_bias", "none")

if METHOD != "fullft":
    print(f"🔧 LoRA: r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT} "
          f"rslora={LORA_RSLORA} targets={len(LORA_TARGETS)} modules")
    # Record what actually ran in the run's own config.json
    config.update({
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGETS,
        "lora_rslora": LORA_RSLORA,
        "lora_bias": LORA_BIAS,
    })
    try:
        with open(args.config, "w") as f:
            json.dump(config, f, indent=2)
    except OSError as e:
        print(f"⚠️  Could not write resolved LoRA params back to config: {e}")

# -------------------------
# EXPERIMENT TRACKING (Aim)
# -------------------------
# All runs log to a single local Aim repo at outputs/aim — browse with: ml aim-ui
# TensorBoard logging is enabled per-run in TrainingArguments (ml tensorboard).
AIM_REPO = os.path.join(os.path.dirname(os.path.dirname(run_dir)), "aim")
aim_run = None
try:
    from aim import Run as AimRun

    aim_run = AimRun(
        repo=AIM_REPO,
        experiment=METHOD,
        log_system_params=True,
    )
    aim_run.name = f"{config.get('run', os.path.basename(run_dir))} ({METHOD})"
    aim_run["method"] = METHOD
    aim_run["hparams"] = {
        "method": METHOD,
        "model": config.get("model"),
        "dataset": config.get("dataset"),
        "model_path": model_path,
        "lora_r": LORA_R if METHOD != "fullft" else None,
        "lora_alpha": LORA_ALPHA if METHOD != "fullft" else None,
        "lora_dropout": LORA_DROPOUT if METHOD != "fullft" else None,
        "lora_rslora": LORA_RSLORA if METHOD != "fullft" else None,
        "lora_target_modules": ",".join(LORA_TARGETS) if METHOD != "fullft" else None,
    }
    print(f"📈 Aim tracking → {AIM_REPO} (experiment: {METHOD})")
except Exception as e:  # tracking is optional — never block training on it
    aim_run = None
    print(f"⚠️  Aim tracking unavailable ({type(e).__name__}: {e}) — skipping")

# -------------------------
# LOAD MODEL
# -------------------------
print(f"🤖 Loading model: {model_path}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    load_in_4bit=(METHOD == "qlora"),
)

# Add LoRA adapters unless doing full fine-tuning
if METHOD != "fullft":
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGETS,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=LORA_RSLORA,
    )

# -------------------------
# LOAD DATASET
# -------------------------
train_path = os.path.join(dataset_path, "train")
eval_path = os.path.join(dataset_path, "eval")

if os.path.isdir(train_path):
    train_dataset = load_from_disk(train_path)
else:
    print("❌ Processed dataset not found at:", train_path)
    print("   Run 'process-dataset <family> <dataset-name>' first.")
    sys.exit(1)

eval_dataset = None
if os.path.isdir(eval_path):
    eval_dataset = load_from_disk(eval_path)
    print(f"📊 Eval dataset: {len(eval_dataset)} examples")

# -------------------------
# FORMAT DATASET (alpaca → messages)
# -------------------------
# Unsloth/trl SFTTrainer expects a 'messages' column where each entry
# is a list of {"role": "user"/"assistant", "content": "..."} dicts.
# The tokenizer's apply_chat_template is then used automatically.

def formatting_prompts_func(examples):
    messages = []
    for instruction, inp, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"],
    ):
        if inp:
            content = f"{instruction}\n\n{inp}"
        else:
            content = instruction
        messages.append([
            {"role": "user", "content": content},
            {"role": "assistant", "content": output},
        ])
    return {"messages": messages}

column_names = train_dataset.column_names
if set(["instruction", "input", "output"]).issubset(set(column_names)):
    print("📋 Alpaca format detected, converting to messages...")
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=column_names,
    )
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            formatting_prompts_func,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )
else:
    # If already in messages format, use as-is
    if "messages" in column_names:
        print("📋 Messages format detected, using directly.")
    else:
        print("⚠️  Dataset columns:", column_names)
        print("   Expected: instruction, input, output (or: messages)")
        sys.exit(1)

if args.max_samples:
    import random
    n = min(args.max_samples, len(train_dataset))
    indices = random.sample(range(len(train_dataset)), n)
    train_dataset = train_dataset.select(indices)
    print(f"📊 Limited to {n} training examples (--max-samples)")

print(f"📊 Training dataset: {len(train_dataset)} examples")

# Use fast settings for sample runs
if args.sample:
    print("⚡ Sample mode: 1 epoch, batch=8, logging every 50 steps")

# Pre-format: render chat template into a 'text' column so Unsloth
# doesn't need to call formatting_func during multiprocessing tokenization
def format_to_text(examples):
    texts = []
    for msgs in examples["messages"]:
        texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
    return {"text": texts}

train_dataset = train_dataset.map(format_to_text, batched=True, remove_columns="messages")
if eval_dataset:
    eval_dataset = eval_dataset.map(format_to_text, batched=True, remove_columns="messages")


def formatting_func(example):
    if isinstance(example["text"], list):
        # Batch mode: example["text"] is a list of strings
        return example["text"]
    else:
        # Single example mode (validation)
        return [example["text"]]

# -------------------------
# TRAINING
# -------------------------
output_dir = os.path.join(run_dir, "artifacts")

# transformers 5.x: TensorBoard event dir comes from this env var (not a TrainingArguments arg)
os.environ["TENSORBOARD_LOGGING_DIR"] = os.path.join(run_dir, "logs")

# Calculate training steps for warmup_steps
if args.sample:
    num_epochs = 1
    batch_size = 8
    accum_steps = 1
    logging_every = 50
else:
    num_epochs = 4
    batch_size = 2
    accum_steps = 4
    logging_every = 1

total_steps = (len(train_dataset) // (batch_size * accum_steps)) * num_epochs
warmup_steps = max(1, int(total_steps * 0.05))

# Adapters want a high LR; full FT wants ~10x lower
learning_rate = 2e-4 if METHOD != "fullft" else 2e-5

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_func,
    max_seq_length=2048,
    packing=False,
    args=TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=logging_every,
        optim="adamw_8bit",
        max_steps=-1,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        report_to="tensorboard",
    ),
)

if aim_run:
    trainer.add_callback(AimMetricsCallback(aim_run))

trainer.train()
if aim_run:
    aim_run.close()

# -------------------------
# SAVE OUTPUT
# -------------------------
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("")
print("✅ Training complete →", output_dir)
