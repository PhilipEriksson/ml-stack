#!/usr/bin/env python3
"""Sample N examples from a processed dataset for faster training."""

import sys
import os
import random
from datasets import load_from_disk, Dataset

train_path = sys.argv[1]
output_path = sys.argv[2]
n_samples = int(sys.argv[3])

# Load both splits
train_ds = load_from_disk(os.path.join(train_path, "train"))
eval_path = os.path.join(train_path, "eval")
eval_ds = load_from_disk(eval_path) if os.path.isdir(eval_path) else None

# Sample from train
if n_samples > len(train_ds):
    print(f"⚠️  Requested {n_samples} but only {len(train_ds)} train examples available")
    print("   Using all examples")
    n_samples = len(train_ds)

indices = random.sample(range(len(train_ds)), n_samples)
sampled_train = train_ds.select(indices)

# Take a proportional eval sample (keep it small)
n_eval = max(1, n_samples // 20)  # 5% of train size
if eval_ds and len(eval_ds) > n_eval:
    eval_indices = random.sample(range(len(eval_ds)), n_eval)
    sampled_eval = eval_ds.select(eval_indices)
elif eval_ds:
    sampled_eval = eval_ds
else:
    sampled_eval = None

# Save
os.makedirs(output_path, exist_ok=True)
sampled_train.save_to_disk(os.path.join(output_path, "train"))
print(f"  Saved train: {len(sampled_train)} examples")
if sampled_eval is not None:
    sampled_eval.save_to_disk(os.path.join(output_path, "eval"))
    print(f"  Saved eval:  {len(sampled_eval)} examples")
else:
    print("  No eval split")

print(f"📁 Saved to: {output_path}")
