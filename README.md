# ML Stack

A local machine learning stack for model inference, fine-tuning, and evaluation.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE) [![Platform](https://img.shields.io/badge/Platform-Linux_%26_WSL2-success?style=flat-square)]() [![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)]() [![Docker](https://img.shields.io/badge/Docker-Required-blue?style=flat-square&logo=docker)]() [![GPU](https://img.shields.io/badge/GPU-RTX_5090-orange?style=flat-square)](https://www.nvidia.com/en-us/hardware/geforce/rtx-5090/)

[![vLLM](https://img.shields.io/badge/vLLM-v0.20.1-8B18A4?style=flat-square)](https://docs.vllm.ai) [![Open%20WebUI](https://img.shields.io/badge/Open_WebUI-main-green?style=flat-square)](https://openwebui.com) [![FastAPI](https://img.shields.io/badge/FastAPI-0.112.0-009688?style=flat-square)](https://fastapi.tiangolo.com) [![CUDA](https://img.shields.io/badge/CUDA-≥%2013-76B900?style=flat-square)](https://developer.nvidia.com/cuda-toolkit)

**Model inference • Fine-tuning • Evaluation — all local, all modular**

> **Hardware target:** Optimized for **NVIDIA RTX 5090** (32 GB VRAM) with **CUDA ≥ 13**.

[⚙️ Requirements](#requirements) • [🚀 Quick Start](#quick-start) • [🛠️ CLI Commands](#cli-commands) • [📚 Model Storage](#model-storage-structure) • [🐳 Docker Services](#docker-services) • [📖 Training](#training) • [🐍 Conda Environments](#conda-environments) • [🤖 Claude Code](#using-with-claude-code) • [📊 Evaluation](#evaluation) • [⚡ GPU Optimization](#gpu-optimization)

---

## ⚙️ Requirements

**Hardware**

| Component | Requirement | Notes |
|---|---|---|
| GPU | 1× NVIDIA RTX 5090 (32 GB VRAM) | Blackwell GB202; lower-end GPUs may not fit 27B int4 models |
| CPU | 8+ cores | For dataset processing and llama.cpp multi-threading |
| RAM | 32 GB | Shared between system, inference and training; running both simultaneously will be tight |
| Storage | ~50 GB free | Model weights, datasets, conda envs, Docker images |

**Software**

| Component | Version | Notes |
|---|---|---|
| OS | Linux / WSL2 | Ubuntu 22.04+ recommended |
| NVIDIA Driver | ≥ 580.x | Required for CUDA 13 runtime in the vLLM Docker image |
| CUDA Toolkit | ≥ 13 | System CUDA ≥ 13 is backward-compatible with PyTorch CUDA 12.1 wheels |
| Docker | Latest stable | With NVIDIA Container Toolkit installed for GPU passthrough |
| Git | ≥ 2.x | For cloning and model management |
| Conda (Miniconda) | Latest | For `training` and `inference-vllm` environments |
| Python | 3.10 | Pinned in conda environments; system Python not required |

**Optional**

| Component | Purpose |
|---|---|
| `huggingface_hub` (hf CLI) | Model/dataset downloads (`pip install huggingface_hub`) |
| `lm-eval` | Benchmarking (`pip install 'lm-eval[multitask]'`) |

---

## 🚀 Quick Start

**1. Clone the repository:**
```bash
git clone https://github.com/PhilipEriksson/ml-stack.git
cd ml-stack
```

**2. Run the setup check:**
```bash
bash scripts/init.sh
```
This checks your system (Python, jq, curl, GPU, Docker, conda, llama.cpp, Docker images) and reports what's missing.

**3. Auto-install what it can:**
```bash
bash scripts/init.sh --install
```
This will:
- Add `ml` to your PATH (in `~/.bashrc`)
- Install missing system tools (`jq`)
- Install Python packages (`huggingface_hub`)
- Create Docker wrapper on WSL2 (if Docker Desktop is installed but not in PATH)
- Create missing conda environments
- Build missing Docker images (`vllm`, `api-webui`)
- Create missing project directories

> **Note:** Docker Desktop itself must be installed manually ([download](https://docs.docker.com/desktop/)).

**4. Source your shell:**
```bash
source ~/.bashrc
ml          # shows all available commands
ml help add-model
```

**5. Launch vLLM inference:**
```bash
docker compose -f services/docker/docker-compose.yml up -d vllm
```
The first launch will download the model weights (~20 GB for a 27B int4 model). Wait for "Application startup complete" in the logs.

> **Performance:** Qwen3.6-27B int4 AutoRound with MTP speculative decoding delivers 100+ tokens/sec on a single RTX 5090 (32 GB) with 256K context.

**6. Verify it's working:**
```bash
curl http://localhost:8080/v1/models | jq
```

**7. Optional — start the web interface:**
```bash
docker compose -f services/docker/docker-compose.yml up -d api-webui
```
Open `http://localhost:3000` for the Open WebUI.

## 📁 Project Structure

```
.
├── cli/                ← CLI entry point (the `ml` command)
│   └── ml
├── configs/            ← JSON registries and configuration files
│   ├── datasets/       ← dataset registry
│   ├── models/         ← model registry (used by `serve-model`)
│   ├── runs/           ← training run registry
│   └── vllm/           ← vLLM environment config files (.env)
├── datasets/           ← downloaded datasets (raw/ + processed/)
├── engine/             ← script orchestration
│   └── runner.sh       ← dispatches commands to scripts/
├── envs/               ← conda environment YAML files
│   ├── training.yml    ← PyTorch + Unsloth + PEFT + TRL + huggingface_hub
│   └── inference-vllm.yml    ← vLLM inference (conda env: inference-vllm)
├── models/             ← local model storage
│   ├── base/           ← full-precision, unmodified weights
│   ├── quantized/      ← compressed model variants (GGUF, GPTQ, AWQ, etc)
│   └── finetuned/      ← LoRA adapters and merged fine-tuned models
├── outputs/            ← experiment outputs and training artifacts
├── runtimes/           ← third-party runtime dependencies (e.g. llama.cpp)
├── scripts/            ← entrypoints
│   ├── init.sh         ← one-time setup (adds `ml` to PATH)
│   ├── train/          ← training scripts (finetune.py)
│   ├── eval/           ← benchmark scripts (run-benchmark)
│   └── utils/          ← utility scripts
│       ├── add-model       ← download and register models from HF
│       ├── add-dataset     ← download and register datasets from HF
│       ├── compare-runs    ← compare two training runs
│       ├── ml-runs         ← list all recorded training runs
│       ├── process-dataset ← convert raw datasets to alpaca format (instruction, input, output)
│       ├── serve-model     ← serve a registered GGUF model via llama.cpp
│       └── train-run       ← create and register a new training run
└── services/           ← Docker services
    ├── api-webui/      ← FastAPI proxy + Open WebUI
    ├── docker/         ← Docker Compose orchestration
    │   └── docker-compose.yml
    └── vllm/           ← vLLM Docker service (env-driven config)
```

## 🛠️ CLI Commands

The `ml` command (in `cli/ml`) is the unified entry point for all operations. Add it to your PATH:

```bash
export PATH="$HOME/ml-stack/cli:$PATH"
```

### Model Management

**Download and register a model from Hugging Face (auto-detect):**
```bash
ml add-model auto <hf-repo>
```
Auto-detects the family (`qwen`, `llama3`), quantization format, and bit-width. Places the model in the correct folder under `models/`.

```bash
# Examples:
ml add-model auto Lorbus/Qwen3.6-27B-int4-AutoRound
# → models/quantized/qwen/auto-round/4bit/...

ml add-model auto bartowski/Llama-3.1-8B-Instruct-GGUF
# → models/quantized/llama3/gguf/...
```

**Download with manual family + variant names:**
```bash
ml add-model <family> <variant> <hf-repo>
```

```bash
# Examples:
ml add-model qwen qwen3.6-27b Lorbus/Qwen3.6-27B-int4-AutoRound
ml add-model qwen qwen3.5-32b HuggingFaceTB/SmolLM2-1.7B-Instruct --type base
```

**Override auto-detection type:**
```bash
ml add-model auto <hf-repo> --type <base|quantized|finetuned>
```

### Serving Models

**Serve a GGUF model via llama.cpp (local):**
```bash
ml serve-model <family> <variant>
```

```bash
# Example:
ml serve-model qwen qwen3.6-27b-gguf
```

The script reads the registry entry, resolves the `.gguf` file path, and launches `llama-server` from `runtimes/llama.cpp/`.

**VRAM safety:** Before launching, the script checks if the vLLM Docker container is running. If so, it warns you and offers to stop it — running both vLLM and llama.cpp simultaneously on a consumer GPU will likely exceed your VRAM.

**For HF models (serve via Docker vLLM):**
```bash
docker compose -f services/docker/docker-compose.yml up -d vllm
```
This launches vLLM in Docker with the model from `configs/vllm/qwen3.6-27b-int4.env`. Swap models by changing the `.env` file and rebuilding.

**Optional flags for `serve-model`:**
```bash
--ctx <n>     Context length (auto-detected from GPU VRAM, default 16384)
--ngl <n>     Number of GPU layers (default 999 = all layers on GPU)
--port <n>    Port to serve on (default 8000)
--host <addr> Bind address (default 0.0.0.0)
--fa          Enable flash attention
```

If the registry entry has an `mmproj` field (vision projection), it is auto-loaded.

### Dataset Management

**Download a dataset from Hugging Face:**
```bash
ml add-dataset <family> <hf-dataset-repo>
```
```bash
# Example:
ml add-dataset qwen HuggingFaceTB/cnn_dailymail
# → datasets/raw/qwen/cnn_dailymail/
```

**Convert to alpaca format for training:**
```bash
ml process-dataset <family> <dataset-name>
```
Auto-detects column mapping from common dataset patterns. Override with `--instruction`, `--input`, `--output` flags. Outputs alpaca-format dataset (`instruction`, `input`, `output`) to `datasets/processed/` with `train/` and `eval/` splits.

```bash
# Examples:
ml process-dataset qwen cnn_dailymail
#   auto-detects: instruction=(none), input=article, output=highlights

ml process-dataset qwen my_alpaca --instruction prompt --output response
```

### Training

Full workflow — download, process, train:

```bash
# 1. Download raw dataset
ml add-dataset qwen HuggingFaceTB/cnn_dailymail

# 2. Convert to alpaca format (instruction, input, output)
ml process-dataset qwen cnn_dailymail

# 3. Download a base model for fine-tuning
ml add-model auto <hf-repo> --type base

# 4. Create a training run
ml train my-run qwen/qwen3.6-27b qwen/cnn_dailymail

# 5. Run fine-tuning (activate training env first)
conda activate training
python scripts/train/finetune.py outputs/runs/my-run/config.json
```

Uses Unsloth's `FastLanguageModel` for efficient 4-bit LoRA fine-tuning with `SFTTrainer`. The dataset is converted from alpaca format into chat messages and formatted with the model's native chat template.

### Run Tracking

**List all training runs:**
```bash
ml runs
```

**Compare two runs:**
```bash
ml compare-runs <run1> <run2>
```
Shows model version, dataset, and model hash for each run. Warns if both use identical weights.

### Help
```bash
ml                # Shows global help with all commands
ml help <cmd>     # Shows detailed help for a specific command
```

## 📚 Model Storage Structure

Models are organized by **category** (base/quantized/finetuned), then by **family** (qwen, llama3), then by **format/bit-width**.

### `models/base/` — Full-Precision Weights

Original, unmodified model weights. Used as the starting point for fine-tuning.

```
base/
└── <family>/
    └── <variant>/
        ├── config.json
        ├── *.safetensors
        ├── tokenizer.json
        └── ...
```

### `models/quantized/` — Compressed Variants

Organized by quantization format and bit-width:

```
quantized/
└── <family>/
    ├── gguf/                ← GGUF format (bit in filename: Q3_K_XL, Q4_K_M)
    │   └── <variant>/
    │       └── *.gguf
    │
    ├── auto-round/          ← AutoRound quantization
    │   ├── 4bit/
    │   ├── 8bit/
    │   └── other/
    │
    ├── gptq/                ← GPTQ quantization
    │   ├── 4bit/
    │   ├── 8bit/
    │   └── other/
    │
    ├── awq/                 ← AWQ quantization
    │   ├── 4bit/
    │   └── other/
    │
    ├── nvfp/                ← NVIDIA FP4/FP8 formats
    │   ├── fp4/
    │   └── fp8/
    │
    ├── mx/                  ← MX FP4/FP8 formats
    │   ├── fp4/
    │   └── fp8/
    │
    └── other/               ← exotic/custom quantizations (PARO, etc)
```

The `add-model` script auto-detects format and bit-width from the HF repo name and file contents, placing files in the correct folder.

### `models/finetuned/` — LoRA & Merged Models

```
finetuned/
└── <family>/
    ├── lora/                ← LoRA adapter weights
    ├── checkpoints/         ← training checkpoints (saved per epoch)
    └── merged/              ← adapter merged back into base weights
```

### Model Registry

`configs/models/registry.json` tracks all local models:

```json
{
  "qwen": {
    "models": {
      "qwen3.6-27b-gguf": {
        "type": "gguf",
        "variant": "instruct",
        "path": "/path/to/model/dir",
        "file": "model.gguf",
        "mmproj": "/path/to/vision-projection.gguf",
        "status": "ready",
        "hash": "sha256..."
      }
    }
  }
}
```

**Variant types:**
| Variant | Description | Auto-Detect Hint |
|---|---|---|
| `pretrained` | Raw token predictor (no fine-tuning) | No "Instruct"/"Reasoning" in name |
| `instruct` | SFT-finetuned for instruction following | Repo name contains `Instruct` |
| `reasoning` | CoT-trained + RLHF/RLVR for chain-of-thought | Repo name contains `Thinking`, `Reasoning`, `R1` |

The `serve-model` script reads this registry to resolve the model path, type, and optional vision projection.

## 🐳 Docker Services

### Auto-Start with Windows Task Scheduler

On WSL2, Docker containers don't survive a reboot. To auto-start vLLM and API+WebUI on login, create two scheduled tasks in Windows:

**Start vLLM on login:**
```powershell
schtasks /create /tn "ML-Stack-vLLM" /tr "wsl bash -c 'cd /home/YOUR_USER/ml-stack && docker compose -f services/docker/docker-compose.yml up -d vllm'" /sc onlogon /ru YOUR_USER
```

**Start API + WebUI on login:**
```powershell
schtasks /create /tn "ML-Stack-API-WebUI" /tr "wsl bash -c 'cd /home/YOUR_USER/ml-stack && docker compose -f services/docker/docker-compose.yml up -d api-webui'" /sc onlogon /ru YOUR_USER
```

> **Note:** Replace `YOUR_USER` with your actual WSL username. Also replace `docker` with `"/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"` if you use the WSL2 wrapper script.

### Architecture

Two modular Docker services:

| Service | Dockerfile | Description |
|---|---|---|
| **vllm** | `services/vllm/Dockerfile` | vLLM model serving (GPU) |
| **api-webui** | `services/api-webui/Dockerfile` | FastAPI proxy + Open WebUI |

### vLLM Service

Based on `vllm/vllm-openai:v0.20.1`. All model settings are driven by environment variables — no rebuild needed to swap models.

**Config file:** `configs/vllm/qwen3.6-27b-int4.env`

Contains all vLLM flags (`MODEL_NAME`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, etc). To swap models, create a new `.env` file with different settings.

**How it works:**
1. The `start.sh` entrypoint reads env vars and builds the `vllm serve` command
2. `HF_HOME=/hf-cache` is set; the HF cache is volume-mounted from the host
3. vLLM resolves `MODEL_NAME` through its native HF model resolution

**Env file variables:**
| Variable | Description |
|---|---|
| `MODEL_NAME` | HF model name or local path |
| `MAX_MODEL_LEN` | Maximum context length |
| `GPU_MEMORY_UTILIZATION` | GPU memory fraction (0.0–1.0) |
| `ATTENTION_BACKEND` | e.g. `flashinfer` |
| `PERFORMANCE_MODE` | e.g. `interactivity` |
| `LANGUAGE_MODEL_ONLY` | `true` / omitted |
| `KV_CACHE_DTYPE` | e.g. `fp8_e4m3` |
| `MAX_NUM_SEQS` | Max concurrent sequences |
| `SKIP_MM_PROFILING` | `true` / omitted |
| `QUANTIZATION` | e.g. `auto_round` |
| `REASONING_PARSER` | e.g. `qwen3` |
| `ENABLE_AUTO_TOOL_CHOICE` | `true` / omitted |
| `TOOL_CALL_PARSER` | e.g. `qwen3_coder` |
| `ENABLE_PREFIX_CACHING` | `true` / omitted |
| `ENABLE_CHUNKED_PREFILL` | `true` / omitted |
| `SPECULATIVE_CONFIG` | JSON config string |
| `HOST` | Bind address |
| `PORT` | Container port (default 8000) |

**Swap models at runtime:**

**Option A — from HF cache (model already downloaded):**
```bash
# Create configs/vllm/other-model.env with new MODEL_NAME + settings
# MODEL_NAME can be an HF repo ID (uses ~/.cache/huggingface/hub)
docker compose -f services/docker/docker-compose.yml --env-file configs/vllm/other-model.env up -d vllm
```

**Option B — from local model storage (`~/ml-stack/models/`):**
```bash
# Point MODEL_NAME to the path inside the container:
MODEL_NAME=/opt/models/quantized/qwen/auto-round/4bit/your-model-folder
```

Both the HF cache (`~/.cache/huggingface/hub`) and local model directory (`~/ml-stack/models`) are mounted read-only into the container.

### API + WebUI Service

Based on `ghcr.io/open-webui/open-webui:main`. Adds a FastAPI proxy that:
- Translates between OpenAI API format and Responses API format
- Proxies all requests to the vLLM backend
- Runs on port 8000 alongside Open WebUI on port 8080

**Ports:**
| Host Port | Container Port | Service |
|---|---|---|
| 8080 | 8000 | vLLM (direct API access) |
| 8000 | 8000 | FastAPI proxy |
| 3000 | 8080 | Open WebUI (web interface) |

### Running with Docker Compose

```bash
# Build images
docker compose -f services/docker/docker-compose.yml build

# Start everything
docker compose -f services/docker/docker-compose.yml up -d

# Start only vLLM
docker compose -f services/docker/docker-compose.yml up -d vllm

# Start only the API+WebUI
docker compose -f services/docker/docker-compose.yml up -d api-webui

# Stop everything
docker compose -f services/docker/docker-compose.yml down

# View logs
docker compose -f services/docker/docker-compose.yml logs -f vllm
docker compose -f services/docker/docker-compose.yml logs -f api-webui
```

## 🤖 Using with Claude Code

`init.sh --install` automatically sets up `claude-local` in your `~/.bashrc`. It dynamically detects your active inference backend — no manual config needed.

**How it works:**
1. **vLLM Docker** — `claude-local` checks if the `vllm-server` container is running and reads `MODEL_NAME` from its environment
2. **llama.cpp (GGUF)** — When you run `ml serve-model`, it records the active model. `claude-local` reads this on next launch

**Switch models for vLLM:**
```bash
# Point to a different vLLM env file
ml set-vllm-env qwen3.6-27b-int4.env
docker compose -f services/docker/docker-compose.yml --env-file configs/vllm/qwen3.6-27b-int4.env up -d vllm
claude-local   # picks up the new model automatically
```

**Serve a GGUF model and connect:**
```bash
ml serve-model qwen qwen3.6-27b-gguf
# In another terminal:
claude-local   # detects the running llama.cpp server
```

> **Note:** The `.claude_env` file in `~/ml-stack/` is created by `init.sh` and sources `scripts/utils/get-claude-env` to detect the active model. Don't edit it manually.

## 🐍 Conda Environments

| File | Conda Env Name | Purpose |
|---|---|---|
| `envs/inference-vllm.yml` | `inference-vllm` | vLLM inference |
| `envs/training.yml` | `training` | Unsloth + PEFT + TRL + datasets + huggingface_hub |

```bash
# Create environments
conda env create -f envs/inference-vllm.yml
conda env create -f envs/training.yml
```

## 📊 Evaluation

Evaluate your local model against standard benchmarks using [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness).

**Install once:**
```bash
pip install 'lm-eval[multitask]'
# or inside your conda env:
conda activate training
pip install 'lm-eval[multitask]'
```

**Run against your active backend:**
```bash
# Default suite (MMLU, GSM8K, HellaSwag, ARC-Challenge)
ml eval

# Specific tasks
ml eval mmlu,gsm8k

# With custom batch size (higher = faster but more VRAM)
ml eval mmlu,ifeval 8
```

The `ml eval` script automatically detects your running vLLM container or llama.cpp server, sends all benchmark prompts through your API, and saves result JSONs to `outputs/evals/`.

### Available Tasks

| Task | Description | Approx. Questions | Time on RTX 5090 |
|---|---|---|---|
| `mmlu` | Massive Multitask Language Understanding (57 subjects) | ~5,700 | ~10 min |
| `gsm8k` | Grade School Math | 1,319 | ~3 min |
| `hellaswag` | Commonsense NLI | 10,042 | ~15 min |
| `arc_challenge` | AI2 Reasoning Challenge (hard) | 1,102 | ~3 min |
| `arc_easy` | AI2 Reasoning Challenge (easy) | 2,258 | ~5 min |
| `truthfulqa_gen` | TruthfulQA (generation) | 817 | ~2 min |
| `ifeval` | Instruction Following Eval | 3,132 | ~8 min |
| `winogrande` | Winograd Schema | 1,267 | ~3 min |
| `bbh` | Beyond Benchmark-Hard (17 tasks) | ~3,900 | ~10 min |

> **Note:** Times are approximate for a 27B int4 model at batch_size=4. Your results may vary.
>
> Benchmark datasets (~200 MB total) are downloaded on first run and cached in `~/.cache/huggingface/datasets/`.

### Eval Registry

Each benchmark run is recorded in `configs/evals/registry.json` with the model, backend, timestamp, tasks, scores, and result file path. This lets you track how models improve over time without digging through raw JSON files.

```bash
# View all recorded evals
ml evals

# View a single eval result in detail
cat outputs/evals/qwen3.6-27b-int4-2026-05-08_22-00.json | jq .results

# Compare two results (manually or with a diff tool)
diff <(jq '.results | keys' file1.json) <(jq '.results | keys' file2.json)
```

## ⚡ GPU Optimization

### Power Management

The RTX 5090 has a 575W TDP by default. For quieter, cooler operation (especially during long eval runs or idle serving), you can lower the power limit.

**Set a power limit:**
```bash
# Check current power limit
nvidia-smi -q | grep "Power"

# Set a custom power limit (watts)
sudo nvidia-smi -pl 350

# Reset to default
sudo nvidia-smi -pl 575
```

**Common power profiles:**

| Target Power | Use Case | Performance Impact |
|---|---|---|
| 575W (default) | Max throughput (training, large batch evals) | 100% baseline |
| 450W | Balanced — good for serving with lower noise | ~5-10% slower token gen |
| 350W | Silent/quiet — good for interactivity, idle serving | ~10-15% slower |
| 250W | Battery/ultra-quiet — sacrifices throughput for power | ~20-25% slower |

> **Note:** Lower power limits reduce the boost clock frequency. For interactive use (single requests, chat), 350W-450W feels nearly as fast as 575W while running much cooler and quieter.

**Make power limit persistent (WSL2):**
```bash
# Add to ~/.bashrc or a systemd service
sudo nvidia-smi -pl 450
```

### Clock Speed Management

For fine-grained control, you can also limit the maximum GPU clock:
```bash
# List supported clock speeds
nvidia-smi -q -d CLOCK | grep "Graphics"

# Set max graphics clock (MHz)
sudo nvidia-smi -lgc 2000

# Remove clock limit
sudo nvidia-smi -lgc 0
```

### vLLM Settings for RTX 5090

In your vLLM `.env` file, these settings affect power usage:

| Variable | Power-Conscious | Performance | Description |
|---|---|---|---|
| `GPU_MEMORY_UTILIZATION` | `0.85` | `0.95` | Lower = less VRAM pressure, less compute work |
| `MAX_NUM_SEQS` | `1` | `4` | Fewer concurrent sequences = less power |
| `PERFORMANCE_MODE` | `interactivity` | `throughput` | Interactivity prioritizes low latency over batch size |

For a quiet 350W setup serving a single model interactively:
```bash
GPU_MEMORY_UTILIZATION=0.85
MAX_NUM_SEQS=1
PERFORMANCE_MODE=interactivity
```

## 📦 Dependencies

See [Requirements](#requirements) above for the full list.
