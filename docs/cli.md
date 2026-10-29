# CLI reference

`ml` is a thin dispatcher: `cli/ml` prints help and hands the command to `engine/runner.sh`, which calls one script under `scripts/`. Every command is also runnable directly (`scripts/utils/<name>`) if you prefer.

```bash
ml                  # global help
ml help <command>   # per-command help
```

Model and dataset references are always `<family>/<variant>` — the keys in the registries, not filesystem paths.

---

## Models

### `ml add-model auto <hf-repo>`

Downloads a Hugging Face repo and registers it. Detects:

- **type** — `base` (safetensors), `gguf` (GGUF repos), `quantized` (GPTQ/AWQ/AutoRound/NVFP4/MXFP4), `finetuned` (adapter files)
- **quant format and bit-width** — from the repo name and file list
- **variant** — `pretrained`, `instruct`, or `reasoning` from the repo name

```bash
ml add-model auto <org>/<model>
ml add-model auto <org>/<model>-GGUF --file <model>-Q4_K_M.gguf
ml add-model auto <org>/<model> --type base
ml add-model auto <org>/<model> --family <name>   # when the repo name is ambiguous
```

Family detection matches a keyword table (`qwen`, `llama`, `mistral`, `mixtral`, `gemma`, `phi`, `deepseek`, `glm`, `granite`, `nemotron`, `olmo`, `falcon`, `smollm`, `exaone`, `kimi`, `minimax`, `gpt-oss`). Anything else needs `--family`.

**GGUF repos ship many quantizations.** If the repo has more than one `.gguf` file, `add-model` lists them and exits rather than downloading tens of GB and leaving `serve-model` to guess. Pick one with `--file`, or pass `--all-quants` to take everything.

| Flag | Meaning |
|---|---|
| `--type` | `base`, `quantized`, `finetuned`, or `gguf` |
| `--family` | override family detection |
| `--file` | download a single file from the repo |
| `--mmproj` | vision projection file (relative to the model dir, or absolute) |
| `--all-quants` | download every file in a multi-quant GGUF repo |

### `ml add-model <family> <variant> <source>`

Explicit naming, same download path.

```bash
ml add-model <family> <variant> <org>/<model> --type base
ml add-model <family> <variant> <org>/<model>-GGUF --file <model>-Q4_K_M.gguf
ml add-model <family> <variant> --url https://huggingface.co/<org>/<repo>/resolve/main/<file>.gguf
```

`--file` downloads a single file — useful for GGUF repos that ship many quantizations. `--url` parses the repo and filename out of a direct HF file URL.

Every download is hashed (sha256 over the sorted file list) and recorded in `configs/models/registry.json` with `status: ready` or `status: error`.

---

## Datasets

### `ml add-dataset <family> <hf-dataset-repo>`

```bash
ml add-dataset <family> <org>/<dataset>
# → datasets/raw/<family>/<dataset>/
```

### `ml process-dataset <family> <dataset>`

Converts a raw dataset to alpaca format (`instruction`, `input`, `output`) with `train/` and `eval/` splits under `datasets/processed/<family>/<dataset>/`. Column mapping is auto-detected from common patterns; override it when the guess is wrong.

```bash
ml process-dataset <family> <dataset>
ml process-dataset <family> <dataset> --instruction prompt --output response
```

### `ml sample-dataset <family> <dataset> <n>`

Randomly samples `n` training examples (keeps the full eval split) and registers the result as `<dataset>-<n>`.

```bash
ml sample-dataset <family> <dataset> 1000
```

---

## Training

### `ml create-training-run <run> <family/model> <family/dataset>`

Writes `outputs/runs/<run>/config.json`, records the model hash, and registers the run.

```bash
ml create-training-run <run> <family>/<model> <family>/<dataset>
ml create-training-run <run> <family>/<model> <family>/<dataset> --method lora
ml create-training-run <run> <family>/<model> <family>/<dataset> --method fullft
```

| Flag | Default | Meaning |
|---|---|---|
| `--method` | `qlora` | `lora` (bf16 base + adapters), `qlora` (4-bit NF4 base + adapters), `fullft` (all weights) |
| `--lora-r` | `16` | adapter rank |
| `--lora-alpha` | same as rank | scaling; effective strength is `alpha/r` |
| `--lora-dropout` | `0.0` | dropout on adapter inputs |
| `--lora-targets` | `q,k,v,o,gate,up,down` | which linear layers get adapters |
| `--rslora` | off | rank-stabilized LoRA (`alpha/sqrt(r)`) |

Resolved values are written back into the run's `config.json` and logged as hparams, so every run records the rank it actually used.

### `ml execute-training-run <run> [--sample]`

Runs `scripts/train/finetune.py` in the active training environment. Datasets under 10K examples automatically switch to fast settings (1 epoch, batch 8, sparse logging); `--sample` forces them.

### `ml use-training-env [env]`

Without arguments: prints the current env and the envs defined in `envs/`. With an argument: verifies the conda env exists and makes it active for training and GGUF quantization.

### `ml kill-training`

Finds and kills running `finetune.py` processes (SIGTERM, then SIGKILL).

### `ml tensorboard` / `ml aim-ui`

TensorBoard on `:6006` over `outputs/runs/`; Aim on `:43801` over `outputs/aim/`. Both bind `0.0.0.0`, so they are reachable over a VPN/Tailscale using the host name rather than `localhost`.

### `ml runs` / `ml compare-runs <a> <b>`

List registered runs, or diff two of them (model, dataset, hash, status). `compare-runs` warns when both runs used identical weights.

### `ml status`

Prints the active training env, the selected vLLM env, the recorded llama.cpp model, every registered model/dataset/run/eval, whether each backend is up, and a "needs attention" list. Use it before debugging anything else — it answers "what is actually configured" in one screen.

---

## Serving

### `ml serve-model <family> <variant>`

Serves a registered GGUF model with llama.cpp.

```bash
ml serve-model <family> <variant>
ml serve-model <family> <variant> --ctx 32768 --port 8000 --fa
```

| Flag | Default | Meaning |
|---|---|---|
| `--ctx` | auto (16384, or 32768 at 48 GB+ VRAM) | context length |
| `--ngl` | `999` | GPU layers (999 = all) |
| `--port` | `8000` | listen port |
| `--host` | `0.0.0.0` | bind address |
| `--file` | registry `file` | pick one `.gguf` when the directory holds several |
| `--threads` | `nproc` | CPU threads |
| `--stop-vllm` | off | stop the running vLLM container first |
| `--fa` | off | flash attention |

If the registry entry has an `mmproj` field, the vision projection is loaded automatically. If the model directory holds more than one `.gguf` and no file is recorded, the command lists them and exits instead of picking one arbitrarily.

When a vLLM container is running, `serve-model` warns and asks before stopping it. In a non-interactive shell it leaves the container alone — pass `--stop-vllm` to stop it explicitly. The active model is recorded in `configs/llama/.active-llama` so `claude-local` and `ml eval` can find it.

### `ml set-vllm-env [file]`

Without arguments: lists the env files in `configs/vllm/` and shows the active one. With an argument: copies that file to `configs/vllm/active.env`, which `docker-compose.yml` and `claude-local` both read.

```bash
ml set-vllm-env
ml set-vllm-env my-model.env
docker compose -f services/docker/docker-compose.yml up -d vllm
```

---

## Quantization

### `ml quantize <model> --format gguf|exl3|nvfp4|all`

`<model>` is an HF repo id or a local directory. The dispatcher resolves the architecture, gates it against the target tool's support table, and shells out to the right interpreter per format.

```bash
ml quantize <model> --format gguf  --quants Q4_K_M,Q8_0
ml quantize <model> --format exl3  --bpw 4.0,3.0
ml quantize <model> --format nvfp4 --calib <dataset-or-jsonl> --samples 256
ml quantize <model> --format all   --dry-run
```

| Flag | Default | Meaning |
|---|---|---|
| `--quants` | `Q4_K_M,Q8_0` | GGUF quant types |
| `--bpw` | `4.0` | EXL3 bits per weight |
| `--calib` | required for NVFP4 | HF dataset id or `.jsonl` path |
| `--samples` | `256` | NVFP4 calibration samples |
| `--slug` | model directory name | output name |
| `--family` | `config.json` → `model_type` | output family directory |
| `--out` / `--work` | derived | override output / working directory |
| `--overwrite` | off | replace an existing output |
| `--dry-run` | off | print the commands without running them |

Details, environments, and verification: [quantization.md](quantization.md).

---

## Evaluation

### `ml eval [tasks] [batch_size]`

Runs lm-eval against the active backend (vLLM container first, then the recorded llama.cpp model).

```bash
ml eval                       # mmlu,gsm8k,hellaswag,arc_challenge @ batch 4
ml eval mmlu,gsm8k
ml eval mmlu,ifeval 8
```

Results land in `outputs/evals/` and are recorded in `configs/evals/registry.json`.

### `ml evals [model-slug]`

Lists recorded benchmarks, or the detail for one model.

See [evaluation.md](evaluation.md) for the task table and timings.
