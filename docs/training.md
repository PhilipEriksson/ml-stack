# Training

Two paths, both driven by the same registries:

| Path | Script | Use when |
|---|---|---|
| **Supervised fine-tuning** | `scripts/train/finetune.py` (Unsloth) | behaviour shaping, format/style, instruction following |
| **Post-training / RLVR** | `scripts/train/sft.py` → `grpo.py` (TRL) | a programmatic verifier can score the output |

The first is the everyday path and is fully wrapped by `ml create-training-run` / `ml execute-training-run`. The second is a pipeline you drive directly, because the reward function is task-specific.

---

## LoRA vs QLoRA vs full fine-tuning

Two independent choices that get confused: **what parameters you train** (all weights vs adapters) and **what stage you run** (SFT, DPO/GRPO, continued pretraining). SFT is a *stage*; full FT is a *parameterization*. Any stage can run either way.

| Method | What trains | Base weights | Use when |
|---|---|---|---|
| **LoRA** | adapter A/B matrices only | bf16 | behaviour shaping when memory allows — fastest per step |
| **QLoRA** | adapter A/B matrices only | 4-bit NF4 | same as LoRA, but the model does not fit in 16-bit; slower steps (on-the-fly dequant) |
| **Full FT** | all weights | bf16 | knowledge injection (continued pretraining), or as the reference point |

**Concepts**

- **PEFT** — *Parameter-Efficient Fine-Tuning*: the umbrella term for adapter methods (LoRA, DoRA, IA³, prefix tuning), and the name of Hugging Face's library implementing them. LoRA is the dominant one.
- **LoRA** — for each target weight matrix `W`, train two small matrices `A` (d×r) and `B` (r×k); the effective update is `W + (α/r)·A·B` with rank ≤ r. Only A/B are saved (megabytes) and they can be merged into the base weights afterwards.
- **QLoRA** — same adapters, but the frozen base is quantized to 4-bit **NF4** and dequantized on the fly. NF4's 16 levels are quantiles of a normal distribution, so each bucket gets equal expected mass for Gaussian-like weights — unlike INT4's equally spaced grid, or FP4 (NVFP4/MXFP4) which uses a float-style ladder (dense near zero, coarse at high magnitude). That distribution matching is why NF4 wins at 4-bit.
- **When LoRA matches full FT** — post-training on capability the model already has (instruction following, style, format), provided adapters target **all** linear layers (`q/k/v/o/gate/up/down`) and the learning rate is ~10× the full-FT value. Full FT still wins for continued pretraining on large corpora (low-rank updates are a capacity bottleneck for new *knowledge*), very large batch sizes, and repeated sequential fine-tuning (LoRA accumulates "intruder dimensions").
- **Continued pretraining** — teaching new knowledge at scale: full FT, plain next-token loss on packed raw text (no chat template), LR ~1e-5–2e-5, mixing ~5–10% general data to limit catastrophic forgetting.

**In this repo**

`finetune.py` supports all three methods via the run config's `method` field:

```bash
# QLoRA (default): 4-bit NF4 base + adapters on all 7 linear modules
ml create-training-run <run> <family>/<model> <family>/<dataset>

# LoRA: bf16 base + adapters (faster steps, needs the model in 16-bit)
ml create-training-run <run> <family>/<model> <family>/<dataset> --method lora

# Full FT: all weights in bf16, LR auto-drops
ml create-training-run <run> <family>/<model> <family>/<dataset> --method fullft
```

Switching is one change: pass a `peft_config` (LoRA/QLoRA) or omit it (full FT). Dataset, chat template, trainer, and eval stay identical.

### Adapter knobs

Optional keys in `outputs/runs/<run>/config.json` (or flags on `create-training-run`). Omitted keys use the defaults shown.

| Key | Default | Meaning |
|---|---|---|
| `lora_r` | `16` | adapter rank — capacity of the update |
| `lora_alpha` | `lora_r` | scaling; effective strength is `alpha/r` |
| `lora_dropout` | `0.0` | dropout on adapter inputs |
| `lora_target_modules` | `q,k,v,o,gate,up,down` | which linear layers get adapters |
| `lora_rslora` | `false` | rank-stabilized LoRA (scales by `alpha/sqrt(r)`) |
| `lora_bias` | `none` | `none` / `all` / `lora_only` |

```json
{ "method": "lora", "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05 }
```

```bash
ml create-training-run <run> <family>/<model> <family>/<dataset> \
  --method lora --lora-r 32 --lora-alpha 64 --lora-dropout 0.05
```

Resolved values are written back into the run's `config.json` and logged as hparams. Rank is never auto-selected — it is whatever the config says.

`--lora-targets` accepts short names (`q,k,v,o,gate,up,down`) or full module names (`q_proj,...`); short names are expanded to `<name>_proj` before PEFT sees them.

### Run knobs

Optional keys in the run config, with the defaults shown. `--sample` changes the defaults for `epochs`, `batch_size`, and `grad_accum` (1 / 8 / 1) but an explicit value in the config always wins.

| Key | Default | Meaning |
|---|---|---|
| `max_seq_length` | `2048` | sequence length, set at model load |
| `epochs` | `4` (`1` in sample mode) | training epochs |
| `batch_size` | `2` (`8` in sample mode) | per-device batch size |
| `grad_accum` | `4` (`1` in sample mode) | gradient accumulation steps |
| `learning_rate` | `2e-4` adapters, `2e-5` full FT | peak LR |

**Target modules matter more than rank.** Architectures with hybrid attention (some layers full attention, some linear attention) use different projection names per layer type. Targeting only the conventional names leaves most of the attention stack frozen. Check the module names in the checkpoint before assuming the default list applies.

---

## Workflow

```bash
# 1. Dataset
ml add-dataset <family> <org>/<dataset>
ml process-dataset <family> <dataset>

# 2. Base model
ml add-model auto <org>/<model> --type base

# 3. Run config
ml create-training-run <run> <family>/<model> <family>/<dataset>

# 4. Environment
ml use-training-env training

# 5. Train
ml execute-training-run <run>

# 6. Stop if needed
ml kill-training
```

Each run gets `outputs/runs/<run>/` with `config.json`, `model/`, `dataset/`, `logs/`, and `artifacts/`.

### Fast test runs

Datasets under 10K examples automatically switch to fast settings (1 epoch, batch 8, sparse logging). For a deliberate smoke test, sample the dataset first:

```bash
ml sample-dataset <family> <dataset> 1000
ml create-training-run <test-run> <family>/<model> <family>/<dataset>-1000
ml execute-training-run <test-run>
```

`ml execute-training-run <run> --sample` forces the same fast settings on a full dataset. Wall-clock time scales with model size, sequence length, and sample count — measure it on your own hardware rather than trusting a table.

### Dashboards

Every `finetune.py` run logs to two local dashboards, no accounts, no upload:

- **TensorBoard** — train/eval loss, LR, grad norm per step. `ml tensorboard` → `http://localhost:6006`. Event files live in `outputs/runs/<run>/logs/`.
- **Aim** — cross-run comparison: eval loss across runs coloured by method/LR, param tables, run diffing. `ml aim-ui` → `http://localhost:43801`. Runs land in `outputs/aim/`, grouped by experiment, with method/model/dataset as params.

Both bind `0.0.0.0`, so they are reachable over a VPN using the host name rather than `localhost`.

### Run tracking

```bash
ml runs                      # all runs: model, dataset, status, timestamp
ml compare-runs <a> <b>      # model version, dataset, hash; warns on identical weights
```

---

## Post-training / RLVR

A verifier-reward pipeline: SFT cold-start → GRPO with a programmatic checker as the reward, TRL + vLLM in colocate mode on a single GPU.

```
scripts/train/build_coldstart.py   build the train/eval JSONL from a source dataset
scripts/train/verifier.py          the reward: format + accuracy — BUILD FIRST
scripts/train/baseline_probe.py    measure base pass rate; gates everything
scripts/train/sft.py               stage 1: SFT cold-start (TRL SFTTrainer)
scripts/train/grpo.py              stage 2: GRPO (TRL GRPOTrainer, vLLM colocate)
notebooks/grpo-learn.ipynb         the mechanics, visually
```

```bash
# 0. Gate check: base pass rate must land in 10-30%
~/miniconda/envs/grpo-trl/bin/python scripts/train/baseline_probe.py --model <model> --n 200

# 1. SFT cold-start (only if the probe says so)
~/miniconda/envs/grpo-trl/bin/python scripts/train/sft.py \
    --train datasets/processed/<family>/<dataset>/train.jsonl \
    --eval  datasets/processed/<family>/<dataset>/eval.jsonl \
    --model <model> --out outputs/runs/<run>-sft

# 2. GRPO/RLVR (stop any other GPU server first — colocate vLLM needs the VRAM)
~/miniconda/envs/grpo-trl/bin/python scripts/train/grpo.py \
    --train datasets/processed/<family>/<dataset>/train.jsonl \
    --model outputs/runs/<run>-sft/final \
    --out   outputs/runs/<run>-grpo
```

### Operating rules

1. **Verifier before anything.** Measure the base pass rate on a held-out slice before training. ~0% → no gradient, SFT cold-start first. >50% → task too easy. 10–30% → go.
2. **Stop other GPU servers before stage 2.** Colocate vLLM shares the card with the trainer; a resident inference server will OOM it.
3. **BF16, no quantization.** Quantization is a serving concern; do not contaminate the RL run with it.
4. **Log the failures, not just the curve.** Reward hacking, dead rewards, entropy collapse — each failure with its diagnosis is worth more than a clean run.

### Reading the curves

| Symptom | Diagnosis |
|---|---|
| reward flat from step 0 | verifier never fires → re-run the probe, check the format pass rate |
| reward jumps then flatlines at a hack | the model found a verifier loophole → tighten the verifier, log completions |
| entropy → 0 early | no KL (`beta=0`) plus low temperature → raise temperature or add a small beta |
| loss NaN | LR too high for the model size, or advantage std explosion |

`grpo.py` sets `log_completions=True`, so the tracker shows sample completions per step — the fastest way to spot reward hacking.

### Notebook

`notebooks/grpo-learn.ipynb` walks the mechanics: data formatting → verifier → group-advantage math → live rollouts → plotting a real run's reward curve from `outputs/runs/*/checkpoint-*/trainer_state.json`.

```bash
~/.local/bin/jupyter-lab --no-browser --ip 0.0.0.0 --port 8890 --notebook-dir=$HOME/ml-stack/notebooks
```

Set `MODEL` and the run names in the first cells before executing.
