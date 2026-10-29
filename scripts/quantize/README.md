# Quantization toolkit

Quantize any supported model into **GGUF**, **EXL3**, or **NVFP4** with one command.
Built to be reused across models, not tied to any single checkpoint.

```bash
ml quantize <model> --format gguf  --quants Q4_K_M,Q8_0
ml quantize <model> --format exl3  --bpw 4.0,3.0
ml quantize <model> --format nvfp4 --calib <dataset-or-jsonl> --samples 256
ml quantize <model> --format all   --dry-run
```

`<model>` is an HF repo id **or** a local directory. Quantizing a model you just trained
is the same code path as quantizing something off the Hub.

## Why three environments

Each format needs a different toolchain, and they conflict:

| Format | Tool | Interpreter | Why separate |
|---|---|---|---|
| GGUF | Unsloth's bundled llama.cpp converter | `training-cuda13` | Unsloth is installed there |
| EXL3 | exllamav3 | `quant-exl3` | Wheels pinned to `torch 2.10.0+cu128`; training env is `2.14.1+cu130` |
| NVFP4 | llm-compressor | `quant-nvfp4` | Pins its own `transformers`/`compressed-tensors` |

`quantize.py` is stdlib + `huggingface_hub` only. It resolves the model and architecture,
then shells out to the right interpreter per format. That's why the dispatcher can run
anywhere and the format modules never see each other's dependencies.

**Creating a quant and serving it are different toolchains.** NVFP4 *creation* is
llm-compressor and runs on any CUDA GPU — FP4 tensor cores only matter for *running* W4A4
inference. Serving is vLLM with SM120 kernels compiled in, which is a separate problem.

## Setup

```bash
# EXL3
~/miniconda/bin/conda create -n quant-exl3 python=3.12 -y
~/miniconda/envs/quant-exl3/bin/pip install torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
~/miniconda/envs/quant-exl3/bin/pip install \
  https://github.com/turboderp-org/exllamav3/releases/download/v1.6.5/exllamav3-1.6.5+cu128.torch2.10.0-cp312-cp312-linux_x86_64.whl

# convert.py is a repo script, not part of the wheel
git clone --depth 1 https://github.com/turboderp-org/exllamav3 \
  ~/ml-stack/runtimes/exllamav3

# NVFP4
~/miniconda/bin/conda create -n quant-nvfp4 python=3.12 -y
~/miniconda/envs/quant-nvfp4/bin/pip install torch \
  --index-url https://download.pytorch.org/whl/cu130
~/miniconda/envs/quant-nvfp4/bin/pip install llmcompressor
```

**Do not run two `conda create` commands concurrently.** They share `~/miniconda/pkgs` and
collide on `.partial` extraction, failing with `No such file or directory` on package
renames. Serialize them.

Verify EXL3 with **torch imported first** — a bare `import exllamav3_ext` fails with a
misleading `libc10.so` error:

```bash
~/miniconda/envs/quant-exl3/bin/python -c "import torch, exllamav3_ext; print('ok')"
```

## Output layout

```
models/quantized/<family>/<format>/<bits>/<slug>/
```

Matches the convention already documented in the main README, so `ml add-model` and the
serving commands find results without changes.

- GGUF: `models/quantized/qwen3_5/gguf/Q4_K_M/qwopus-0.8b/qwopus-0.8b-Q4_K_M.gguf`
- EXL3: `models/quantized/qwen3_5/exl3/4.0bpw/qwopus-0.8b/`
- NVFP4: `models/quantized/qwen3_5/nvfp4/4bit/qwopus-0.8b/`

Working directories (intermediate f16 GGUF, exllamav3 state) go to
`outputs/quantize/<slug>/`.

## Format notes

### GGUF

Uses Unsloth's co-versioned converter, staged via
`unsloth_zoo.llama_cpp._download_convert_hf_to_gguf()`. That returns
`(converter_path, text_archs, vision_archs)` — the arch sets double as a support gate, so
an unsupported architecture fails loudly instead of producing a broken file.

**Why not stock llama.cpp:** its `convert_hf_to_gguf.py` is broken for `qwen3_5`
(issue #27019, closed as not planned). `_reorder_v_heads` hardcodes `head_dim=1` for the
linear-attention V-head reorder and doesn't expand `in_proj_a`/`in_proj_b`:

```
RuntimeError: shape '[16, 2, 1, 1]' is invalid for input of size 65536
```

The **runtime** is fine — `src/models/qwen35.cpp` exists and runs Qwen3.5 GGUF correctly.
It's purely the converter. Unsloth's staged converter handles it.

### EXL3

EXL3 is a streamlined QTIP variant: trellis-coded quantization against a learned codebook
lattice, Hessians computed on the fly with a fused Viterbi kernel, single pass. No
calibration dataset needed.

`convert.py` flags worth knowing: `-b` bits per weight, `-hb` head bits, `-vb` vision bits
(relevant for multimodal checkpoints), `-r` resume an interrupted job, `-d` device
selection.

The work directory holds state checkpoints and needs room for roughly one full copy of the
output.

### NVFP4

The only format needing **calibration data** — activation global scales come from
representative samples. `--calib` is required and has no default, because the choice
materially affects quality. Point it at text the model actually sees:

```bash
ml quantize <model> --format nvfp4 --calib ~/p5090-scratch/qwopus-0.8b/data/train.jsonl
```

Emits `compressed-tensors`, auto-detected by vLLM with no `--quantization` flag and no
patches. The older NVIDIA ModelOpt `modelopt_fp4` format caused `k_scale`/`v_scale`
KeyErrors and garbled MoE output — don't use it.

## Adding an architecture

Two places:

1. **GGUF** — nothing to do. Unsloth's arch set is the gate; if the arch is missing, the
   converter genuinely doesn't support it.
2. **NVFP4** — add an entry to `configs/nvfp4-ignore.yaml`. Without one the toolkit falls
   back to `lm_head` only and prints a warning, because the small-layer FP4 threshold
   hasn't been measured for that architecture.

EXL3 support comes from exllamav3's own architecture table; nothing to configure here.

## Verification

```bash
python scripts/quantize/verify.py \
  --artifact hf:bf16:/path/to/merged_16bit \
  --artifact gguf:Q4_K_M:/path/to/model-Q4_K_M.gguf \
  --artifact exl3:4.0bpw:/path/to/exl3-dir \
  --out report.json
```

Runs a fixed prompt set through each artifact and records: output length, whether the
reasoning block survived, and whether an expected substring appeared. Writes JSON so
results diff across runs instead of being eyeballed.

Prompts live in `configs/verify-prompts.yaml`, keyed by model slug with a `default`
fallback.

## Pitfalls

- **Two concurrent `conda create` calls corrupt the package cache.** Serialize them.
- **`import exllamav3_ext` alone fails** with a misleading `libc10.so` error. Import torch
  first.
- **`convert.py` is not in the exllamav3 wheel.** Clone the repo.
- **GGUF export needs 16-bit weights.** A checkpoint with a `quantization_config` (bnb 4/8
  bit) can't be converted — merge the LoRA to `merged_16bit` first.
- **Unsloth's merge output is an incomplete model directory.** For multimodal
  architectures it omits `preprocessor_config.json` and `video_preprocessor_config.json`,
  and it omits `vocab.json`/`merges.txt`. EXL3's config reader requires the preprocessor
  files and fails with `FileNotFoundError` before quantizing anything. Copy them from the
  base model before quantizing:
  ```bash
  for f in preprocessor_config.json video_preprocessor_config.json vocab.json merges.txt; do
    cp -n <base_model_dir>/$f <merged_dir>/$f
  done
  ```
- **NVFP4 ignore lists are per-architecture and untuned by default.** The Qwen3.5 entries
  came from a 27B build; at 604M the layer sizes differ, so verify with and without.
