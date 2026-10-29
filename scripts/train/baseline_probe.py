#!/usr/bin/env python3
"""Measure the BASE model's pass rate on a held-out verifier slice.

Run this BEFORE any training (skill rule #1: verifier before anything):
  - pass rate ~0%      -> reward is all-zero, GRPO has no gradient -> SFT cold-start first
  - pass rate 10-30%   -> GO for GRPO
  - pass rate >50%     -> task too easy, pick a harder slice

Usage:
  ~/miniconda/envs/grpo-trl/bin/python baseline_probe.py \
      --model <hf-repo-id-or-local-path> --n 200
"""

import argparse
import json

from datasets import load_dataset

from verifier import ANSWER_RE, accuracy_reward


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF repo id or local path")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    ds = ds.shuffle(seed=args.seed)  # HF Dataset: shuffle() returns a new Dataset
    subset = ds.select(range(min(args.n, len(ds))))

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=2048)
    sp = SamplingParams(temperature=0.6, max_tokens=512)

    prompts = []
    tok = llm.get_tokenizer()
    for ex in subset:
        prompts.append(
            tok.apply_chat_template(
                [
                    {"role": "user", "content": f"{ex['question']}\n\nEnd your reply with the final numeric answer inside <answer></answer> tags."},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    outs = llm.generate(prompts, sp)

    golds = [ex["answer"].split("####")[-1].strip() for ex in subset]
    # Reuse the GRPO reward function verbatim so the baseline measures the
    # exact same verifier the training run optimizes.
    rewards = accuracy_reward(
        [o.outputs[0].text for o in outs], answer=golds
    )
    fmt = [1.0 if ANSWER_RE.search(o.outputs[0].text) else 0.0 for o in outs]

    result = {
        "model": args.model,
        "n": len(rewards),
        "accuracy_pass_rate": sum(rewards) / len(rewards),
        "format_pass_rate": sum(fmt) / len(fmt),
        "verdict": (
            "GO for GRPO" if 0.10 <= sum(rewards) / len(rewards) <= 0.30
            else "SFT cold-start first" if sum(rewards) / len(rewards) < 0.10
            else "task too easy — harder slice"
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
