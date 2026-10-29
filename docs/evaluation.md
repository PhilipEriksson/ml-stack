# Evaluation

Benchmarks run through [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) against whatever backend is currently serving — no separate model load, no duplicated weights.

## Install

lm-eval pins its own `datasets`/`transformers`/`scipy` stack, which conflicts with the training env. It lives in its own environment:

```bash
conda env create -f envs/eval.yml
```

`ml eval` resolves `lm-eval` from `~/miniconda/envs/eval/bin` automatically and falls back to `PATH`.

## Suites

Task lists live in `configs/eval-suites.json`:

```bash
ml eval --list-suites
```

| Suite | Contents |
|---|---|
| `core` | `mmlu`, `gsm8k`, `hellaswag`, `arc_challenge`, `arc_easy`, `winogrande`, `truthfulqa_mc2`, `ifeval`, `piqa`, `boolq`, `openbookqa`, `lambada_openai` |
| `reasoning` | `mmlu_pro`, `bbh`, `minerva_math`, `aime` |
| `reasoning-gated` | `gpqa_diamond_zeroshot` — gated dataset, see below |
| `all` | core + reasoning (open datasets only) |

`core` is the standard set papers report — cheap, fast, comparable across models. `reasoning` is where models actually separate, and where scores get noisy: `aime` in particular is brutal for small models, and a single-digit score is normal rather than a bug.

**GPQA is gated.** `Idavidrein/gpqa` requires accepting the terms on the Hub and an authenticated token:

```bash
# accept the terms at https://huggingface.co/datasets/Idavidrein/gpqa
hf auth login
ml eval --suite reasoning-gated
```

It is kept out of `reasoning` so that suite runs without any Hub access.

Edit the JSON to add a suite; `ml eval --suite <name>` picks it up with no code change.

## Prefetch

Datasets download on first use. Warm the cache before a long run:

```bash
ml eval --suite all --prefetch
```

This runs each task against lm-eval's `dummy` model with `--limit 1`, which downloads the data and answers instantly — no server needed. `core` + `reasoning` is roughly 500 MB.

## Run

```bash
ml eval                          # core suite, batch 4
ml eval --suite reasoning
ml eval --suite all
ml eval mmlu,gsm8k               # explicit tasks
ml eval mmlu,gsm8k 8             # explicit tasks, batch 8
ml eval --suite core --limit 5   # smoke test: 5 examples per task
```

| Flag | Default | Meaning |
|---|---|---|
| `--suite` | `core` | task suite from `configs/eval-suites.json` |
| `--tasks` | — | explicit comma-separated task names |
| `--batch` | `4` | batch size |
| `--limit` | — | examples per task (smoke tests) |
| `--chat-template` | off | apply the model's chat template to prompts |
| `--prefetch` | off | download datasets only |
| `--list-suites` | — | show suites and their tasks |

**Tokenizer.** lm-eval talks to the completions API (`/v1/completions`), not the chat endpoint — multiple-choice tasks need loglikelihood, which chat completions does not support. That means it needs a tokenizer to compute token boundaries. It defaults to the served model's name; override it when the server reports a container path rather than an HF id:

```bash
ML_EVAL_TOKENIZER=/path/to/model ml eval --suite core
```

**Chat template.** Prompts go in raw by default, which is the standard for multiple-choice benchmarks and keeps runs comparable. For instruct models on generative tasks, `--chat-template` applies the model's own template instead.

Backend detection, in order:

1. a running `vllm-server` container → `http://localhost:8080/v1`
2. `configs/vllm/active.env` → `http://localhost:8080/v1`
3. a recorded llama.cpp model in `configs/llama/.active-llama` → `http://localhost:8000/v1`

Override it to evaluate a remote endpoint:

```bash
ML_EVAL_BACKEND_URL=http://other-host:8080/v1 ML_EVAL_MODEL=my-model ml eval --suite core
```

Results are written to `outputs/evals/<model>/results_*.json` and recorded in `configs/evals/registry.json`.

## Registry

Each run records the model, backend, suite, tasks, batch size, limit, lm-eval version, sample counts, and per-task scores with the metric name:

```json
{
  "my-model": {
    "evals": [{
      "timestamp": "2026-10-29_22-18",
      "model": "org/model",
      "backend": "vllm",
      "suite": "core",
      "tasks": "mmlu,gsm8k,...",
      "batch_size": 4,
      "limit": null,
      "lm_eval_version": "0.4.13",
      "scores": {
        "mmlu":  {"metric": "acc",         "value": 0.71},
        "gsm8k": {"metric": "exact_match", "value": 0.55}
      }
    }]
  }
}
```

```bash
ml evals                    # all models, latest scores
ml evals <model-slug>       # one model's history
```

The metric name matters: `acc` and `acc_norm` are different numbers for the same task, and `exact_match` has `strict-match` and `flexible-extract` variants. The registry records which one was used so two runs are never compared on different metrics by accident.

## Reading results honestly

- **Same backend, same batch size, same tasks, same metric** — otherwise the numbers are not comparable.
- **Quantization shows up in the scores.** A 4-bit build that matches BF16 on MMLU can still lose a code or long-reasoning task.
- **Small deltas are noise.** Benchmark variance at these sample sizes is often larger than the difference you are trying to measure. `--limit` runs are smoke tests, not measurements.
- **`mmlu` is 57 subtasks.** It dominates the runtime of the `core` suite; use `--limit` while iterating and a full run for the number you keep.
