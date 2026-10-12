#!/usr/bin/env python3
"""Stage 2: GRPO/RLVR run for grpo-0.6b (TRL GRPOTrainer, vLLM colocate).

Single-GPU shape (RTX 5090): trainer + vLLM rollout engine share the card
via vllm_mode="colocate". The verifier (verifier.py) IS the reward — no
reward model, no human labels. Stop the :30000 server first; colocate
needs the VRAM.

Prereqs, in order (see scripts/train/README.md):
  1. baseline_probe.py shows 10-30% pass rate (else SFT cold-start first)
  2. dataset JSONL with "prompt" and "answer" columns

Usage:
  ~/miniconda/envs/grpo-trl/bin/python grpo.py \
      --train datasets/processed/gsm8k-coldstart/train.jsonl \
      --model outputs/runs/grpo-0.6b-sft/final \
      --out ../../outputs/runs/grpo-0.6b-grpo
"""

import argparse
import json

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from verifier import accuracy_reward, format_reward


SYSTEM = (
    "Solve the math problem step by step, then give the final numeric "
    "answer inside <answer></answer> tags."
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--model", required=True, help="SFT checkpoint or HF repo id")
    p.add_argument("--out", default="../../outputs/runs/grpo-0.6b-grpo")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--sample", action="store_true", help="fast smoke test")
    args = p.parse_args()

    ds = load_dataset("json", data_files={"train": args.train})["train"]
    ds = ds.rename_column("prompt", "prompt")  # GRPOTrainer expects "prompt"
    if args.sample:
        ds = ds.select(range(8))

    cfg = GRPOConfig(
        output_dir=args.out,
        max_steps=args.steps,
        learning_rate=args.lr,
        # Rollouts: G generations per prompt, advantage normalized within group
        num_generations=args.num_generations,
        max_prompt_length=512,
        max_completion_length=512,
        temperature=0.9,
        # Single-GPU colocate: vLLM lives inside the trainer process
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,
        bf16=True,
        per_device_train_batch_size=args.num_generations * 2,
        gradient_accumulation_steps=2,
        beta=0.0,  # no KL penalty by default; add reference-KL if entropy collapses
        logging_steps=1,
        save_steps=25,
        report_to="trackio",
        seed=13,
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=cfg,
        reward_funcs=[format_reward, accuracy_reward],
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(args.out + "/final")
    print(json.dumps({"status": "done", "output": args.out + "/final"}))


if __name__ == "__main__":
    main()
