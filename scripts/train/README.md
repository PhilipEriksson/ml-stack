# grpo-0.6b — RLVR/GRPO post-training on the RTX 5090

Portfolio project: take Qwen3-0.6B through a complete SFT → GRPO/RLVR
pipeline against a verifiable reward (GSM8K with a programmatic checker),
on a single 32 GB Blackwell card. **The deliverable is this writeup, not
the model.**

## Pipeline

```
envs/grpo-trl.txt                     pinned stack (trl 0.23.1 / vllm 0.11.0 / torch 2.8.0+cu128)
scripts/train/verifier.py             the reward: format + math_verify accuracy — BUILD FIRST
scripts/train/baseline_probe.py       measure base pass rate; gates everything (10-30% band)
scripts/train/sft.py                  stage 1: SFT cold-start (TRL SFTTrainer)
scripts/train/grpo.py                 stage 2: GRPO (TRL GRPOTrainer, vLLM colocate)
outputs/runs/                         artifacts: checkpoints, curves, logs (gitignored)
```

Learn the mechanics visually in **[notebooks/grpo-learn.ipynb](../notebooks/grpo-learn.ipynb)**
(JupyterLab: `~/.local/bin/jupyter-lab --no-browser --ip 0.0.0.0 --port=8890 --notebook-dir=$HOME/Workspace/ml-stack/notebooks`):
data formatting → verifier → group-advantage math → live rollouts, plus a section
that plots the real run's reward curve from `outputs/runs/*/checkpoint-*/trainer_state.json`.

## Operating rules (learned the hard way, keep them)

1. **Verifier before anything.** Measure base pass rate on a held-out slice
   before training. ~0% → no gradient, SFT cold-start first. >50% → task
   too easy. 10-30% → go.
2. **Stop the :30000 server before stage 2.** Colocate vLLM needs the VRAM;
   the Qwen3.8-27B server holds ~30/32 GiB.
3. **BF16, no quantization.** Quant is a serving concern; don't contaminate
   the RL run with it.
4. **Log the failures, not just the curve.** Reward hacking, dead rewards,
   entropy collapse — each failure with its diagnosis is worth more than a
   clean run.

## Results

(filled at the end of the run)

| Stage | Accuracy (held-out) | Format pass rate | Notes |
|---|---|---|---|
| Base Qwen3-0.6B | — | — | baseline_probe output |
| After SFT cold-start | — | — | |
| After GRPO | — | — | |

## Failures log

(numbered entries: symptom → diagnosis → fix. This section is the point.)

## Env

```bash
conda create -n grpo-trl python=3.11 -y
~/miniconda/envs/grpo-trl/bin/pip install -r envs/grpo-trl.txt
```
