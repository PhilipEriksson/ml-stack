# ML Stack

A local machine learning stack for model inference, fine-tuning, quantization, and evaluation — one GPU, one repo, no cloud.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE) [![Platform](https://img.shields.io/badge/Platform-Linux_%26_WSL2-success?style=flat-square)]() [![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)]() [![Docker](https://img.shields.io/badge/Docker-Required-blue?style=flat-square&logo=docker)]() [![CUDA](https://img.shields.io/badge/CUDA-%E2%89%A5%2013-76B900?style=flat-square)](https://developer.nvidia.com/cuda-toolkit)

**Inference • Fine-tuning • Quantization • Evaluation — all local, all modular**

---

## What this is

Four independent layers, each with one job, wired together by a single `ml` command:

| Layer | Tool | Where |
|---|---|---|
| **Inference** | vLLM (Docker) for HF checkpoints, llama.cpp for GGUF | `services/`, `scripts/utils/serve-model` |
| **Training** | Unsloth (LoRA/QLoRA/full FT) and TRL (SFT, GRPO/RLVR) | `scripts/train/`, `envs/` |
| **Quantization** | GGUF, EXL3, NVFP4 | `scripts/quantize/` |
| **Evaluation** | lm-eval-harness against the live backend | `scripts/eval/` |

Everything is **model-agnostic**. Commands take a model reference (`<family>/<variant>` from the registry, or an HF repo id) and resolve paths, formats, and interpreters themselves. Nothing in the code is tied to a particular checkpoint.

Three conventions make that work:

- **Registries are the source of truth.** `configs/models/registry.json`, `configs/datasets/registry.json`, `configs/runs/registry.json` map names to paths, hashes, and status. They are local state and gitignored.
- **One environment per toolchain.** Training, EXL3 quantization, and NVFP4 quantization pin conflicting dependency sets, so each runs in its own conda env and the dispatcher shells out instead of importing across them.
- **Artifacts live under `models/` and `outputs/`.** Downloads, quantized builds, adapters, and run logs all land in a predictable layout, so serving and evaluation find them without configuration.

## Requirements

| Component | Requirement | Notes |
|---|---|---|
| GPU | 1× NVIDIA GPU, 24 GB+ VRAM | Blackwell (sm_120) recommended; larger models need more VRAM or quantization |
| CPU | 8+ cores | Dataset processing, llama.cpp threads |
| RAM | 32 GB+ | Shared between system, inference, and training |
| Storage | 50 GB+ free | Weights, datasets, conda envs, Docker images |
| OS | Linux or WSL2 | Ubuntu 22.04+ recommended |
| NVIDIA driver | ≥ 580.x | Required for the CUDA 13 runtime in the vLLM image |
| CUDA toolkit | ≥ 13 | Backward-compatible with older PyTorch CUDA wheels |
| Docker | Latest stable | With the NVIDIA Container Toolkit for GPU passthrough |
| Conda | Miniconda | Training and quantization environments |
| Python | 3.12 | Training env; the CLI itself is bash + `jq` |

Optional: `huggingface_hub` (`hf` CLI) for downloads, `lm-eval[multitask]` for benchmarks.

Full setup, driver notes, and Docker configuration: **[docs/install.md](docs/install.md)**.

## Quick start

```bash
# 1. Clone
git clone https://github.com/PhilipEriksson/ml-stack.git
cd ml-stack

# 2. Check the system (add --install to auto-fix what it can)
bash scripts/init.sh

# 3. Put the CLI on PATH
source ~/.bashrc
ml

# 4. Register a model
ml add-model auto <org>/<model>            # HF repo id; auto-detects type/format
ml add-model auto <org>/<model> --type base

# 5. Serve it
cp configs/vllm/example.env configs/vllm/my-model.env   # set MODEL_NAME
ml set-vllm-env my-model.env
docker compose -f services/docker/docker-compose.yml up -d vllm

# 6. Verify
curl -s http://localhost:8080/v1/models | jq
```

Optional web interface (Open WebUI + search proxy):

```bash
docker compose -f services/docker/docker-compose.yml up -d api-webui
# http://localhost:3000
```

## CLI

`ml` is the only entry point. `ml help <command>` prints per-command usage.

| Command | Purpose |
|---|---|
| `ml add-model auto <hf-repo>` | Download + register a model (auto-detects type, quant format, bit-width) |
| `ml add-model <family> <variant> <source>` | Download + register with explicit names |
| `ml add-dataset <family> <hf-dataset>` | Download + register a dataset |
| `ml process-dataset <family> <dataset>` | Convert to alpaca format (`instruction`, `input`, `output`) |
| `ml sample-dataset <family> <dataset> <n>` | Build a smaller processed dataset for smoke tests |
| `ml create-training-run <run> <family/model> <family/dataset>` | Write a run config (`--method lora\|qlora\|fullft`) |
| `ml execute-training-run <run>` | Run it in the active training env |
| `ml use-training-env [env]` | Switch the active training conda env |
| `ml kill-training` | Stop running training processes |
| `ml tensorboard` / `ml aim-ui` | Loss curves (6006) / cross-run comparison (43801) |
| `ml runs` / `ml compare-runs <a> <b>` | List / diff training runs |
| `ml serve-model <family> <variant>` | Serve a registered GGUF via llama.cpp |
| `ml set-vllm-env [file]` | Select the vLLM env file (copied to `configs/vllm/active.env`) |
| `ml quantize <model> --format gguf\|exl3\|nvfp4\|all` | Quantize an HF repo id or local directory |
| `ml eval [tasks] [batch]` / `ml evals` | Run / list lm-eval benchmarks |

Full reference with flags and examples: **[docs/cli.md](docs/cli.md)**.

## Repository layout

```
.
├── cli/ml                  ← the `ml` command (help + dispatch)
├── engine/runner.sh        ← maps commands to scripts/
├── configs/                ← tracked templates + gitignored local state
│   ├── models/             ← model registry (local)
│   ├── datasets/           ← dataset registry (local)
│   ├── runs/               ← training run registry (local)
│   ├── evals/              ← benchmark registry (local, created on first eval)
│   ├── llama/              ← active llama.cpp model (local)
│   ├── vllm/               ← example.env + active.env (local)
│   ├── training/           ← run config templates (*.example.json)
│   └── searxng/            ← search settings
├── datasets/               ← raw/ + processed/ (gitignored contents)
├── docs/                   ← documentation
├── envs/                   ← conda environment definitions
├── models/                 ← base/ + quantized/ + finetuned/ (gitignored contents)
├── notebooks/              ← teaching notebooks
├── outputs/                ← runs/, evals/, quantize/, aim/ (gitignored contents)
├── runtimes/               ← third-party runtimes (llama.cpp, exllamav3)
├── scripts/
│   ├── init.sh             ← one-time setup
│   ├── build.sh            ← docker compose build wrapper
│   ├── train/              ← finetune.py, sft.py, grpo.py, verifier.py, process.py
│   ├── quantize/           ← dispatcher + per-format modules
│   ├── eval/               ← run-benchmark, list-evals
│   └── utils/              ← one script per `ml` command
└── services/
    ├── docker/             ← docker-compose.yml
    ├── vllm/               ← vLLM image + env-driven entrypoint
    ├── api-webui/          ← FastAPI proxy + Open WebUI
    └── searxng/            ← SearXNG metasearch
```

## Documentation

| Doc | Contents |
|---|---|
| [docs/install.md](docs/install.md) | Drivers, Docker, conda envs, llama.cpp, `init.sh` |
| [docs/cli.md](docs/cli.md) | Every command, flag, and example |
| [docs/models.md](docs/models.md) | Storage layout, registry schema, variant types |
| [docs/serving.md](docs/serving.md) | vLLM service, llama.cpp, proxy/WebUI, `claude-local` |
| [docs/training.md](docs/training.md) | LoRA/QLoRA/full FT, adapter knobs, runs, dashboards, GRPO/RLVR |
| [docs/quantization.md](docs/quantization.md) | GGUF/EXL3/NVFP4, environments, verification, publishing |
| [docs/evaluation.md](docs/evaluation.md) | lm-eval tasks, registry, comparing results |
| [docs/gpu-tuning.md](docs/gpu-tuning.md) | Power limits, clocks, undervolting, vLLM power settings |
| [docs/architecture.md](docs/architecture.md) | How the pieces fit, data flow, extension points |

## License

MIT — see [LICENSE](LICENSE).
