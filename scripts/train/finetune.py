#!/usr/bin/env python3
"""Fine-tune a model with Unsloth SFTTrainer.

Expects a config.json from train-run with:
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

# Check required dependencies before importing
MISSING = []
for pkg, hint in [
    ("unsloth", "conda activate training && pip install unsloth"),
    ("trl", "conda activate training && pip install trl"),
    ("bitsandbytes", "conda activate training && pip install bitsandbytes"),
]:
    try:
        __import__(pkg)
    except ImportError:
        MISSING.append(f"  - {pkg} (install: {hint})")

if MISSING:
    print("❌ Missing required package(s):")
    print("\n".join(MISSING))
    print("")
    print("Try: conda env create -f envs/training.yml")
    sys.exit(1)

from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import json

# -------------------------
# INPUT
# -------------------------
config_path = sys.argv[1] if len(sys.argv) > 1 else None
if not config_path or not os.path.isfile(config_path):
    print("Usage: python finetune.py <path/to/config.json>")
    print("")
    print("  Run 'train-run <name> <family/model> <family/dataset>' first,")
    print("  then: python finetune.py outputs/runs/<name>/config.json")
    sys.exit(1)

with open(config_path, "r") as f:
    config = json.load(f)

model_path = config["model_path"]
dataset_path = config.get("dataset_path")
run_dir = os.path.dirname(config_path)
model_type = config.get("model_type", "base")

if not dataset_path:
    print("❌ No dataset_path in config.")
    print("   Run 'train-run <name> <family/model> <family/dataset>' first.")
    sys.exit(1)

# -------------------------
# WARN IF USING QUANTIZED MODEL
# -------------------------
if model_type == "gguf":
    print("❌ Cannot fine-tune GGUF models with Unsloth.")
    print("   Download a base model instead: add-model auto <hf-repo> --type base")
    sys.exit(1)

# -------------------------
# LOAD MODEL
# -------------------------
print(f"🤖 Loading model: {model_path}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters for fine-tuning
model = FastLanguageModel.for_finetuning(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# -------------------------
# LOAD DATASET
# -------------------------
train_path = os.path.join(dataset_path, "train")
eval_path = os.path.join(dataset_path, "eval")

if os.path.isdir(train_path):
    train_dataset = load_dataset(train_path, split="train")
else:
    print("❌ Processed dataset not found at:", train_path)
    print("   Run 'process-dataset <family> <dataset-name>' first.")
    sys.exit(1)

eval_dataset = None
if os.path.isdir(eval_path):
    eval_dataset = load_dataset(eval_path, split="train")
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

print(f"📊 Training dataset: {len(train_dataset)} examples")

# -------------------------
# TRAINING
# -------------------------
output_dir = os.path.join(run_dir, "artifacts")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="messages",
    max_seq_length=2048,
    packing=False,
    args=TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=1,
        optim="adamw_8bit",
        max_steps=-1,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
    ),
)

trainer.train()

# -------------------------
# SAVE OUTPUT
# -------------------------
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("")
print("✅ Training complete →", output_dir)
