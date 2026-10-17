#!/usr/bin/env python3
"""Build the gsm8k-coldstart dataset (train/eval JSONL) for grpo-0.6b.

Step 1 of the pipeline — the artifact sft.py and grpo.py both consume.
Source: openai/gsm8k ("main", train split). Held-out eval is carved out
FIRST with a fixed seed so baseline_probe.py and every later eval see the
same slice (probe uses seed 13 shuffle on the TEST split; this builds the
TRAIN-split train/eval pair — disjoint from the probe's test data).

Output schema (matches sft.py to_chat + grpo.py reward kwargs):
  {"prompt": <question>, "answer": <final numeric>, "solution": <CoT>}

Usage:
  ~/miniconda/envs/grpo-trl/bin/python build_coldstart.py
  ~/miniconda/envs/grpo-trl/bin/python build_coldstart.py --n 500 --sample
"""

import argparse
import json
import pathlib
import random

from datasets import load_dataset

HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE / "../../datasets/processed/gsm8k-coldstart"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-size", type=int, default=500)
    p.add_argument("--n", type=int, default=None, help="cap total examples (smoke test)")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=args.seed)
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))

    rows = []
    for ex in ds:
        solution, _, answer = ex["answer"].partition("####")
        rows.append(
            {
                "prompt": ex["question"],
                "answer": answer.strip(),
                "solution": solution.strip(),
            }
        )

    eval_rows = rows[: args.eval_size]
    train_rows = rows[args.eval_size :]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train_rows), ("eval", eval_rows)):
        path = (OUT_DIR / f"{name}.jsonl").resolve()
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
        print(json.dumps({"file": str(path), "examples": len(data)}))

    print(json.dumps({"status": "done", "train": len(train_rows), "eval": len(eval_rows)}))


if __name__ == "__main__":
    main()
