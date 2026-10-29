#!/usr/bin/env python3
"""Stage 1: SFT cold-start for a verifier-reward (RLVR) pipeline.

Purpose: lift the base model's pass rate on the verifier task into the
10-30% band so GRPO has gradient signal. Keep it short — 1-2 epochs on a
few thousand examples. This is a cold-start, not a showcase.

Expected dataset shape (JSONL, {"prompt": ..., "answer": ..., "solution": ...}).

Usage:
  ~/miniconda/envs/grpo-trl/bin/python sft.py \
      --train datasets/processed/<family>/<dataset>/train.jsonl \
      --eval  datasets/processed/<family>/<dataset>/eval.jsonl \
      --model <hf-repo-id-or-local-path> \
      --out   outputs/runs/<run-name>-sft
"""

import argparse
import json

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


SYSTEM = (
    "Solve the math problem step by step, then give the final numeric "
    "answer inside <answer></answer> tags."
)


def to_chat(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": example["prompt"]},
        ],
        "completion": (
            "Let's think step by step.\n" + example["solution"]
            + f"\n<answer>{example['answer']}</answer>"
            if "solution" in example
            else f"\n<answer>{example['answer']}</answer>"
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--eval", default=None)
    p.add_argument("--model", required=True, help="HF repo id or local path")
    p.add_argument("--out", default="outputs/runs/sft")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--sample", action="store_true", help="fast smoke test")
    args = p.parse_args()

    ds = load_dataset("json", data_files={"train": args.train})
    ds["train"] = ds["train"].map(to_chat)
    eval_ds = None
    if args.eval:
        eval_ds = load_dataset("json", data_files={"eval": args.eval})["eval"].map(to_chat)
    if args.sample:
        ds["train"] = ds["train"].select(range(64))

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=50,
        max_length=1024,
        report_to="trackio",
        seed=13,
    )
    trainer = SFTTrainer(
        model=args.model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=eval_ds,
    )
    trainer.train()
    trainer.save_model(args.out + "/final")
    print(json.dumps({"status": "done", "output": args.out + "/final"}))


if __name__ == "__main__":
    main()
