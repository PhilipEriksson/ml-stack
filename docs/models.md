# Models and datasets

## Storage layout

Models are grouped by **category**, then **family**, then **format/bit-width**. The layout is a convention, not a constraint — the registry is what commands read — but keeping it consistent means `ml add-model`, `ml quantize`, and `ml serve-model` all agree on where things live.

```
models/
├── base/                       ← full-precision weights (fine-tuning starts here)
│   └── <family>/<variant>/
│       ├── config.json
│       ├── *.safetensors
│       └── tokenizer.json
├── quantized/                  ← compressed variants
│   └── <family>/
│       ├── gguf/<variant>/*.gguf
│       ├── auto-round/<bits>bit/<variant>/
│       ├── gptq/<bits>bit/<variant>/
│       ├── awq/<bits>bit/<variant>/
│       ├── nvfp/fp4/<variant>/  nvfp/fp8/<variant>/
│       ├── mx/fp4/<variant>     mx/fp8/<variant>/
│       └── other/<variant>/     ← exotic formats
└── finetuned/                  ← adapters and merged models
    └── <family>/
        ├── lora/
        ├── checkpoints/
        └── merged/
```

Datasets mirror the same idea:

```
datasets/
├── raw/<family>/<dataset>/          ← as downloaded
└── processed/<family>/<dataset>/    ← alpaca format, train/ + eval/
```

Everything under `models/`, `datasets/`, and `outputs/` is gitignored except the `.gitkeep` files that preserve the directory structure.

## Registry

`configs/models/registry.json` is the source of truth for what is installed:

```json
{
  "<family>": {
    "models": {
      "<variant>": {
        "type": "gguf",
        "variant": "instruct",
        "path": "/abs/path/to/model/dir",
        "file": "model.gguf",
        "mmproj": "/abs/path/to/vision-projection.gguf",
        "hash": "sha256...",
        "status": "ready"
      }
    }
  }
}
```

| Field | Meaning |
|---|---|
| `type` | `base`, `gguf`, `quantized`, or `finetuned` — `serve-model` dispatches on it |
| `variant` | `pretrained`, `instruct`, or `reasoning` |
| `path` | absolute path to the model directory |
| `file` | specific file inside `path`, when the directory holds several |
| `mmproj` | vision projection for multimodal GGUF models |
| `hash` | sha256 over the sorted file list — identifies the exact weights |
| `status` | `downloading`, `ready`, or `error` |

The registry is written in two steps: `downloading` before the transfer, `ready` (with hash) after. A failed download leaves `error` behind rather than a half-registered model.

`configs/datasets/registry.json` follows the same shape with `local_path`, `processed_path`, and `status`.

## Variant types

| Variant | Description | Detection hint |
|---|---|---|
| `pretrained` | raw next-token predictor, no post-training | no `Instruct`/`Reasoning` in the name |
| `instruct` | SFT-tuned for instruction following | name contains `Instruct`, `Chat`, `Assistant` |
| `reasoning` | chain-of-thought trained | name contains `Thinking`, `Reasoning`, `R1`, `CoT` |

The variant is metadata, not behaviour: it tells you what the checkpoint is, and it is recorded in training runs so a run can be traced back to the exact weights it started from.

## Families

A family is just a directory key. `ml add-model auto` derives it from a keyword table; `--family` overrides it. `ml quantize` defaults to the checkpoint's `model_type` from `config.json` and accepts `--family` to match whatever key you use elsewhere.

If you want one key per model line rather than per architecture, pass `--family` consistently in both commands.

## Hashes

`add-model` computes `sha256` over the sorted list of file hashes in the model directory, excluding `hf`'s download metadata (`.cache/`) and dotfiles. Two consequences worth knowing:

- Re-downloading the same weights produces the same hash, so `ml compare-runs` can tell you whether two runs actually started from different weights.
- Any *model* file added to the directory changes the hash. Keep notes and scratch files out of `models/`.
