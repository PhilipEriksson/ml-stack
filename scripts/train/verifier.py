#!/usr/bin/env python3
"""Verifier-as-reward for a verifier-reward (RLVR) pipeline.

The verifier is the reward model. Two components:
  1. Format reward: did the model follow the required output contract?
  2. Accuracy reward: is the final answer mathematically correct (math_verify)?

Keep verifiers STRICT and DETERMINISTIC. A verifier the model can trick
("reward hacking") is the single most common RLVR failure — when the
pass-rate curve spikes while quality doesn't move, audit the verifier first.

Usage (inside GRPOTrainer):
    from verifier import accuracy_reward, format_reward
    trainer = GRPOTrainer(..., reward_funcs=[format_reward, accuracy_reward])
"""

import re

from math_verify import parse, verify


# Output contract: <answer> ... </answer>. Single strict tag pair.
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(completion: str) -> str | None:
    """Pull the final answer out of the model's completion."""
    m = ANSWER_RE.findall(completion)
    return m[-1].strip() if m else None


def format_reward(completions, **kwargs) -> list[float]:
    """+0.2 for a well-formed <answer> tag. Shaping only — never let this
    dominate accuracy (see the vukrosic guide: 0.5 vs 2.0 there)."""
    return [0.2 if extract_answer(c) is not None else 0.0 for c in completions]


def accuracy_reward(completions, answer, **kwargs) -> list[float]:
    """+1.0 if the extracted answer matches the gold answer.

    `answer` comes from the dataset column of the same name — TRL passes
    extra prompt columns through as kwargs to reward functions.
    """
    rewards = []
    for completion, gold in zip(completions, answer):
        pred = extract_answer(completion)
        if pred is None:
            rewards.append(0.0)
            continue
        try:
            gold_parsed = parse(f"${gold}$")
            pred_parsed = parse(f"${pred}$")
            rewards.append(1.0 if verify(gold_parsed, pred_parsed) else 0.0)
        except Exception:
            # Verifier must never crash the run — score conservatively.
            rewards.append(0.0)
    return rewards
