# Quantization

Quantize any supported model into **GGUF**, **EXL3**, or **NVFP4** with one command. Built to be reused across models, not tied to a single checkpoint.

```bash
ml quantize <model> --format gguf  --quants Q4_K_M,Q8_0
ml quantize <model> --format exl3  --bpw 4.0,3.0
ml quantize <model> --format nvfp4 --calib <dataset-or-jsonl> --samples 256
ml quantize <model> --format all   --dry-run
```

`<model>` is an HF repo id **or** a local directory. Quantizing a model you just trained is the same code path as quantizing something off the Hub.

## The three formats

| Format | What it is | Runtime | Needs calibration |
|---|---|---|---|
| **GGUF** | block-wise integer quantization (Q4_K_M, Q8_0, …) | llama.cpp, Ollama, LM Studio | no |
| **EXL3** | QTIP trellis-coded quantization, 2–8 bpw | exllamav3, TabbyAPI | no (Hessians on the fly) |
| **NVFP4** | NVIDIA 4-bit float, W4A4, two-level micro-block scaling | vLLM (Blackwell) | **yes** |

## Why three environments

Each format needs a different toolchain, and they conflict:

| Format | Tool | Interpreter | Why separate |
|---|---|---|---|
| GGUF | Unsloth's bundled llama.cpp converter | active training env | Unsloth lives there |
| EXL3 | exllamav3 | `quant-exl3` | wheels pinned to `torch 2.10.0+cu128` |
| NVFP4 | llm-compressor | `quant-nvfp4` | pins its own `transformers` / `compressed-tensors` |

`scripts/quantize/quantize.py` is stdlib + `huggingface_hub` only. It resolves the model and architecture, then shells out to the right interpreter per format — so the dispatcher runs anywhere and the format modules never see each other's dependencies.

**Creating a quant and serving it are different toolchains.** NVFP4 *creation* is llm-compressor and runs on any CUDA GPU; FP4 tensor cores only matter for *running* W4A4 inference. Serving is vLLM with SM120 kernels compiled in — a separate problem.

## Setup

```bash
# EXL3
~/miniconda/bin/conda create -n quant-exl3 python=3.12 -y
~/miniconda/envs/quant-exl3/bin/pip install torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
~/miniconda/envs/quant-exl3/bin/pip install \
  https://github.com/turboderp-org/exllamav3/releases/download/v1.6.5/exllamav3-1.6.5+cu128.torch2.10.0-cp312-cp312-linux_x86_64.whl

# convert.py is a repo script, not part of the wheel
git clone --depth 1 https://github.com/turboderp-org/exllamav3 ~/ml-stack/runtimes/exllamav3

# NVFP4
~/miniconda/bin/conda create -n quant-nvfp4 python=3.12 -y
~/miniconda/envs/quant-nvfp4/bin/pip install torch \
  --index-url https://download.pytorch.org/whl/cu130
~/miniconda/envs/quant-nvfp4/bin/pip install llmcompressor
```

GGUF uses the active training environment (`ml use-training-env`), which already has Unsloth.

**Do not run two `conda create` commands concurrently.** They share `~/miniconda/pkgs` and collide on `.partial` extraction, failing with `No such file or directory` on package renames.

Verify EXL3 with **torch imported first** — a bare `import exllamav3_ext` fails with a misleading `libc10.so` error:

```bash
~/miniconda/envs/quant-exl3/bin/python -c "import torch, exllamav3_ext; print('ok')"
```

## Output layout

```
models/quantized/<family>/<format>/<bits>/<slug>/
```

- GGUF: `models/quantized/<family>/gguf/Q4_K_M/<slug>/<slug>-Q4_K_M.gguf`
- EXL3: `models/quantized/<family>/exl3/4.0bpw/<slug>/`
- NVFP4: `models/quantized/<family>/nvfp4/4bit/<slug>/`

`<family>` defaults to the checkpoint's `model_type` from `config.json`; pass `--family` to match the key you use in `ml add-model`. `<slug>` defaults to the model directory name (`--slug` overrides).

Working directories (intermediate f16 GGUF, exllamav3 state) go to `outputs/quantize/<slug>/`.

## Architecture gating

Every run reads `config.json` → `architectures[0]` and checks it against the target tool's support table. An unsupported architecture fails loudly with the supported list rather than producing a silently broken artifact. NVFP4 ignore lists are per-architecture in `scripts/quantize/configs/nvfp4-ignore.yaml`; an unknown architecture falls back to `lm_head` only, with a printed warning.

## Format notes

### GGUF

Uses Unsloth's co-versioned converter, staged via `unsloth_zoo.llama_cpp._download_convert_hf_to_gguf()`. That returns `(converter_path, text_archs, vision_archs)` — the arch sets double as a support gate, so an unsupported architecture fails loudly instead of producing a broken file.

Stock llama.cpp's `convert_hf_to_gguf.py` lags behind new architectures, particularly ones with linear-attention or hybrid attention layouts. The **runtime** is usually fine; it is the converter that breaks. Unsloth's staged converter is the gate — if an architecture is missing there, the converter genuinely does not support it yet.

### EXL3

EXL3 is a streamlined QTIP variant: trellis-coded quantization against a learned codebook lattice, Hessians computed on the fly with a fused Viterbi kernel, single pass. No calibration dataset needed.

`convert.py` flags worth knowing: `-b` bits per weight, `-hb` head bits, `-vb` vision bits (relevant for multimodal checkpoints), `-r` resume an interrupted job, `-d` device selection.

The work directory holds state checkpoints and needs room for roughly one full copy of the output.

### NVFP4

The only format needing **calibration data** — activation global scales come from representative samples. `--calib` is required and has no default, because the choice materially affects quality. Point it at text the model actually sees:

```bash
ml quantize <model> --format nvfp4 --calib datasets/processed/<family>/<dataset>/train.jsonl
ml quantize <model> --format nvfp4 --calib <hf-dataset-id> --samples 256
```

Emits `compressed-tensors`, auto-detected by vLLM with no `--quantization` flag and no patches. The older NVIDIA ModelOpt `modelopt_fp4` format caused `k_scale`/`v_scale` KeyErrors and garbled MoE output — do not use it.

## Adding an architecture

Two places:

1. **GGUF** — nothing to do. Unsloth's arch set is the gate; if the arch is missing, the converter genuinely does not support it.
2. **NVFP4** — add an entry to `scripts/quantize/configs/nvfp4-ignore.yaml`. Without one the toolkit falls back to `lm_head` only and prints a warning, because the small-layer FP4 threshold has not been measured for that architecture.

EXL3 support comes from exllamav3's own architecture table; nothing to configure here.

## Verification

```bash
python scripts/quantize/verify.py \
  --artifact hf:bf16:/path/to/merged_16bit \
  --artifact gguf:Q4_K_M:/path/to/model-Q4_K_M.gguf \
  --artifact exl3:4.0bpw:/path/to/exl3-dir \
  --out report.json
```

Runs a fixed prompt set through each artifact and records output length, whether the reasoning block survived, and whether an expected substring appeared. JSON output so results diff across runs instead of being eyeballed.

Prompts live in `scripts/quantize/configs/verify-prompts.yaml`, keyed by model slug with a `default` fallback. The file also sets `template` (chat scaffold applied to every prompt) and `close_tag` (the token that ends a reasoning block) — set `close_tag: ""` for non-reasoning models.

**Verification needs the same interpreter split as creation.** `verify.py` imports the loader for each artifact kind in-process, so one run cannot cover all three formats:

| Artifacts | Interpreter | Why |
|---|---|---|
| `hf:bf16`, `gguf:*` | active training env | transformers + llama-cli |
| `hf:nvfp4` | `quant-nvfp4` | needs `compressed-tensors>=0.15.0` |
| `exl3:*` | `quant-exl3` | exllamav3 |

Loading an NVFP4 artifact under the training env fails with `ImportError: compressed-tensors>=0.15.0 is required` — the artifact is fine, the interpreter is wrong.

`summarize.py` merges every `verify-*.json` in a directory, newest report per artifact label winning, so a stale run cannot overwrite a fresh one:

```bash
python scripts/quantize/summarize.py outputs/quantize/<slug>
```

Three prompts is a smoke test, not a benchmark. It catches gross breakage and obvious cliffs; it cannot rank formats by quality. For that, run a real eval suite ([evaluation.md](evaluation.md)).

## Publishing to the Hub

One repo per format, suffixed from the base model's repo id:

```
<base>-GGUF     Q4_K_M + Q8_0 as single files
<base>-EXL3     4.0bpw/ and 3.0bpw/ subdirectories
<base>-NVFP4    backbone + any extra heads at the repo root
```

Group the family in a Hub collection so the base, adapter, and every quant are reachable from any one of them.

**`hf upload` commits at the end.** It uploads every file and creates one commit when the whole folder lands, so a multi-GB repo shows only `.gitattributes` while it runs. That is not a failed upload — judge the repo after the commit URL prints. If the card must be visible immediately, upload `README.md` as its own commit first.

**Quantized repos need `base_model_relation: quantized`** in the card frontmatter. Without it the Hub tags the repo `base_model:finetune:<base>` instead of `base_model:quantized:<base>`, so it does not show as a quantization of the base model. `library_name: gguf` infers it; EXL3 and NVFP4 do not.

## Pitfalls

- **Two concurrent `conda create` calls corrupt the package cache.** Serialize them.
- **`import exllamav3_ext` alone fails** with a misleading `libc10.so` error. Import torch first.
- **`convert.py` is not in the exllamav3 wheel.** Clone the repo.
- **GGUF export needs 16-bit weights.** A checkpoint with a `quantization_config` (bnb 4/8-bit) cannot be converted — merge the LoRA to `merged_16bit` first.
- **Unsloth's merge output can be an incomplete model directory.** For multimodal architectures it may omit `preprocessor_config.json` and `video_preprocessor_config.json`, and it may omit `vocab.json`/`merges.txt`. EXL3's config reader requires the preprocessor files and fails with `FileNotFoundError` before quantizing anything. Copy them from the base model first:
  ```bash
  for f in preprocessor_config.json video_preprocessor_config.json vocab.json merges.txt; do
    cp -n <base_model_dir>/$f <merged_dir>/$f
  done
  ```
- **NVFP4 ignore lists are per-architecture and untuned by default.** Entries tuned at one model scale may not fit another; verify with and without.
- **Size comparisons across formats are not apples-to-apples.** EXL3 may quantize the vision tower at a different bitrate and keep extra heads, so a lower decoder bitrate does not imply a smaller directory.
