# Install

Everything here is one-time setup. When you are done, `ml` works and the Docker services can start.

## 1. Requirements

| Component | Requirement |
|---|---|
| GPU | 1× NVIDIA GPU, 24 GB+ VRAM (Blackwell / sm_120 recommended) |
| CPU | 8+ cores |
| RAM | 32 GB+ |
| Storage | 50 GB+ free |
| OS | Linux or WSL2 (Ubuntu 22.04+ recommended) |
| NVIDIA driver | ≥ 580.x for the CUDA 13 runtime in the vLLM image |
| CUDA toolkit | ≥ 13 (backward-compatible with older PyTorch CUDA wheels) |
| Docker | Latest stable + NVIDIA Container Toolkit |
| Conda | Miniconda |
| Python | 3.12 for the training env; the CLI itself needs only bash + `jq` |

## 2. NVIDIA driver

**Linux (Ubuntu 24.04+, RTX 50-series / Blackwell)**

The open kernel module is required for driver stability on Blackwell; the proprietary drivers may fail to detect the GPU.

```bash
# Secure Boot must be disabled for the open driver on a fresh install
ubuntu-drivers devices
sudo apt install nvidia-driver-open
# reboot, then:
nvidia-smi
```

Install the CUDA toolkit from your distro's repos and put it on PATH:

```bash
sudo apt install cuda-toolkit-13
# ~/.bashrc
export PATH="/usr/local/cuda-13/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13/lib64:${LD_LIBRARY_PATH}"
```

**Windows / WSL2**

Install the [NVIDIA App](https://www.nvidia.com/en-us/drivers/nvidia-app/) (or the GeForce driver installer) on the Windows host. WSL2 shares the host driver — no CUDA install inside WSL2.

**Troubleshooting:** `nvidia-smi` reporting "No devices found" almost always means the wrong driver flavour. Switch to `nvidia-driver-open`. On a headless machine you cannot enroll MOK keys at boot, so either disable Secure Boot or use the open driver, which does not require key enrollment.

## 3. Docker

**Windows / WSL2 — Docker Desktop**

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/), enable the WSL2 backend, and turn on GPU support in Settings → General. GPU passthrough is automatic.

**Linux — Docker Engine + NVIDIA Container Toolkit**

```bash
sudo apt install docker.io nvidia-container-toolkit
sudo systemctl enable --now docker

sudo mkdir -p /etc/docker
echo '{"runtimes":{"nvidia":{"path":"nvidia-container-runtime","args":[]}}}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
```

Optional terminal UI for containers:

```bash
LATEST=$(curl -s https://api.github.com/repos/jesseduffield/lazydocker/releases/latest | grep -oP '"tag_name":\K[^"]+')
curl -L https://github.com/jesseduffield/lazydocker/releases/download/${LATEST}/lazydocker_${LATEST#v}_Linux_x86_64.tar.gz | tar xz
sudo mv lazydocker /usr/local/bin/
```

## 4. Run the setup check

```bash
bash scripts/init.sh            # report only
bash scripts/init.sh --install  # fix what it can
```

`--install` adds `ml` to PATH, installs `jq` and `huggingface_hub`, creates missing conda envs, builds missing Docker images, creates the project directories, and seeds `configs/vllm/active.env` from `example.env`.

Then:

```bash
source ~/.bashrc
ml
```

## 5. Conda environments

Each toolchain pins conflicting dependencies, so they live in separate environments. `ml use-training-env` selects which one training and GGUF quantization use.

| File | Env name | Purpose |
|---|---|---|
| `envs/training.yml` | `training` | PyTorch cu130 + Unsloth + TRL + TensorBoard + Aim |
| `envs/eval.yml` | `eval` | lm-eval-harness (benchmarks), isolated from training |
| `envs/grpo-trl.yml` + `envs/grpo-trl.txt` | `grpo-trl` | TRL 0.23.1 + vLLM 0.11.0 + torch 2.8.0+cu128, for GRPO/RLVR |
| (manual) | `quant-exl3` | exllamav3, pinned to torch 2.10.0+cu128 |
| (manual) | `quant-nvfp4` | llm-compressor with its own transformers/compressed-tensors |

```bash
conda env create -f envs/training.yml
ml use-training-env training

# Benchmarks
conda env create -f envs/eval.yml

# GRPO/RLVR
conda env create -f envs/grpo-trl.yml
~/miniconda/envs/grpo-trl/bin/pip install -r envs/grpo-trl.txt
```

**Do not run two `conda create` commands at the same time.** They share `~/miniconda/pkgs` and collide on `.partial` extraction, failing with `No such file or directory` on package renames.

The two quantization environments are only needed if you use `ml quantize`; setup commands are in [quantization.md](quantization.md#setup).

## 6. llama.cpp (GGUF serving)

```bash
git clone --recursive https://github.com/ggml-org/llama.cpp.git ~/ml-stack/runtimes/llama.cpp
cd ~/ml-stack/runtimes/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --target llama-server llama-cli -j$(nproc)
```

`ml serve-model` expects `runtimes/llama.cpp/build/bin/llama-server`; `scripts/quantize/verify.py` expects `llama-cli` next to it.

## 7. Verify the install

```bash
ml                                  # command list
ml help add-model                   # per-command help
nvidia-smi --query-gpu=name,memory.total --format=csv
docker compose -f services/docker/docker-compose.yml config >/dev/null && echo "compose ok"
ls runtimes/llama.cpp/build/bin/llama-server
```

If `docker compose config` fails on `configs/vllm/active.env`, create it:

```bash
cp configs/vllm/example.env configs/vllm/active.env
# edit MODEL_NAME, then:
ml set-vllm-env active.env
```
