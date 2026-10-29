# Evaluation

Benchmarks run through [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) against whatever backend is currently serving — no separate model load, no duplicated weights.

## Install

```bash
pip install 'lm-eval[multitask]'
# or inside the training env:
conda activate training && pip install 'lm-eval[multitask]'
```

## Run

```bash
ml eval                       # mmlu,gsm8k,hellaswag,arc_challenge @ batch 4
ml eval mmlu,gsm8k            # specific tasks
ml eval mmlu,ifeval 8         # batch size 8 (faster, more VRAM)
```

`ml eval` detects the active backend in order:

1. a running `vllm-server` container → `http://localhost:8080/v1`
2. `configs/vllm/active.env` → `http://localhost:8080/v1`
3. a recorded llama.cpp model in `configs/llama/.active-llama` → `http://localhost:8000/v1`

Results are written to `outputs/evals/` and recorded in `configs/evals/registry.json`.

## Tasks

| Task | Description | Questions |
|---|---|---|
| `mmlu` | Massive Multitask Language Understanding (57 subjects) | ~5,700 |
| `gsm8k` | grade-school math | 1,319 |
| `hellaswag` | commonsense NLI | 10,042 |
| `arc_challenge` | AI2 Reasoning Challenge (hard) | 1,102 |
| `arc_easy` | AI2 Reasoning Challenge (easy) | 2,258 |
| `truthfulqa_gen` | TruthfulQA (generation) | 817 |
| `ifeval` | instruction following | 3,132 |
| `winogrande` | Winograd schema | 1,267 |
| `bbh` | Beyond Benchmark-Hard (17 tasks) | ~3,900 |

Wall-clock time depends on model size, quantization, batch size, and context length — measure it on your hardware. Benchmark datasets (~200 MB total) download on first run and cache in `~/.cache/huggingface/datasets/`.

## Registry

Each run is recorded with model, backend, timestamp, tasks, scores, and the result file path, so you can track a model across quantizations and training runs without digging through raw JSON.

```bash
ml evals                                  # all recorded evals
ml evals <model-slug>                     # one model's history
cat outputs/evals/<file>.json | jq .results
```

To compare two runs, diff the task keys and scores:

```bash
diff <(jq '.results | keys' a.json) <(jq '.results | keys' b.json)
```

## Reading results honestly

- **Same backend, same batch size, same tasks** — otherwise the numbers are not comparable.
- **Quantization shows up in the scores.** A 4-bit build that matches BF16 on MMLU can still lose a code or long-reasoning task; a smoke-test prompt set catches gross breakage, a benchmark suite catches the rest.
- **Small deltas are noise.** Benchmark variance at these sample sizes is often larger than the difference you are trying to measure.
