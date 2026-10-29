# Architecture

## Layers

```
                    ┌──────────────────────────────┐
                    │            cli/ml            │  help + dispatch
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │       engine/runner.sh       │  command → script
                    └──────────────┬───────────────┘
                                   │
        ┌──────────────┬───────────┼───────────┬──────────────┐
        ▼              ▼           ▼           ▼              ▼
   scripts/utils  scripts/train  scripts/quantize  scripts/eval  services/
   (registry ops) (Unsloth/TRL)  (GGUF/EXL3/NVFP4) (lm-eval)     (Docker)
        │              │           │           │              │
        └──────────────┴───────────┴───────────┴──────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  configs/*/registry.json     │  source of truth
                    │  models/  datasets/  outputs/│  artifacts
                    └──────────────────────────────┘
```

`cli/ml` holds no logic beyond help text. `engine/runner.sh` is a `case` statement mapping each command to one script. Every script is independently runnable, which is what makes debugging a single command possible without going through the CLI.

## Data flow

**Model lifecycle**

```
HF repo ──add-model──▶ models/<category>/<family>/<variant>/ ──▶ registry entry (path, hash, type)
                                                                    │
                          ┌─────────────────────────────────────────┤
                          ▼                                         ▼
              create-training-run                            serve-model / vLLM env
              (records hash in run config)                   (resolves path from registry)
```

**Training lifecycle**

```
raw dataset ──process-dataset──▶ processed/<family>/<dataset>/{train,eval}
                                        │
                                        ▼
                          create-training-run ──▶ outputs/runs/<run>/config.json
                                        │
                                        ▼
                          execute-training-run ──▶ outputs/runs/<run>/{logs,artifacts}
                                        │
                                        ▼
                          adapters / merged weights ──▶ quantize ──▶ models/quantized/...
```

**Evaluation** reads the live backend, not the filesystem: it asks the running server which model it is serving, sends the benchmark prompts through the API, and records the scores.

## Why the registries exist

Paths are long and easy to get wrong; names are short and stable. Every command takes `<family>/<variant>` and resolves it through `configs/models/registry.json` or `configs/datasets/registry.json`. That gives three properties:

- **One place to look.** `ml runs`, `ml evals`, and `ml compare-runs` all read the same records.
- **Traceability.** Model hashes are recorded at download time and copied into every run config, so a run can be tied to exact weights.
- **Failure visibility.** Downloads are registered as `downloading` first and `error` on failure, so a half-finished transfer never looks installed.

Registries are local state and gitignored. The tracked files under `configs/` are templates (`vllm/example.env`, `training/*.example.json`).

## Why one environment per toolchain

The dependency sets genuinely conflict:

| Env | Pins | Conflicts with |
|---|---|---|
| `training` | torch cu130, transformers 5.x, unsloth | EXL3 wheels are built against torch 2.10+cu128 |
| `quant-exl3` | torch 2.10.0+cu128, exllamav3 | training env's torch/CUDA |
| `quant-nvfp4` | its own transformers + compressed-tensors | training env's transformers |
| `grpo-trl` | torch 2.8.0+cu128, vllm 0.11.0, trl 0.23.1 | training env's torch/TRL |

Rather than fight that, `scripts/quantize/quantize.py` runs in any Python (stdlib + `huggingface_hub`) and shells out to the right interpreter per format. The format modules never import each other's dependencies, and adding a format means adding one module plus one entry in `ENV_PYTHON`.

`ml use-training-env` writes `configs/.active-training-env`; `get-training-env` resolves it to a python path. Training scripts and GGUF quantization both read it, so switching environments is one command rather than an edit in several places.

## Extension points

| Task | Where |
|---|---|
| Add a CLI command | `cli/ml` (help) + `engine/runner.sh` (dispatch) + `scripts/utils/<name>` |
| Add a quantization format | `scripts/quantize/formats/<fmt>.py` + `FORMATS` and `ENV_PYTHON` in `quantize.py` |
| Add a model family | keyword table in `scripts/utils/add-model` (`detect_family`), or pass `--family` |
| Add an NVFP4 ignore list | `scripts/quantize/configs/nvfp4-ignore.yaml`, keyed on `architectures[0]` |
| Add a benchmark task | pass it to `ml eval`; lm-eval resolves task names itself |
| Add a Docker service | `services/<name>/` + a service block in `services/docker/docker-compose.yml` |

## Tests

```bash
bash tests/run.sh              # all
bash tests/run.sh add-model    # one file
```

Shell-level smoke tests. Each one builds a throwaway stack (registries, `models/`, `outputs/`) and points `ML_STACK_HOME` at it, stubbing `hf`, `curl`, `lm-eval`, `docker`, and `llama-server` on PATH — so the suite needs no network, no GPU, and no Docker, and never touches real state. They cover the paths that break silently: model type detection, GGUF file selection, registry writes, eval result parsing, and CLI help/dispatch consistency.

## Conventions

- **Bash for orchestration, Python for work.** Scripts under `scripts/utils/` are bash; anything that touches tensors or datasets is Python under `scripts/train/` or `scripts/quantize/`.
- **`ML_STACK_HOME` over hardcoded paths.** Every script resolves the repo root from `${ML_STACK_HOME:-$HOME/ml-stack}`.
- **Fail loudly.** Unsupported architectures, missing environments, and unregistered models exit with the reason and the fix, rather than producing a broken artifact.
- **Artifacts are gitignored, structure is tracked.** `.gitkeep` files preserve the directory tree; contents stay local.
