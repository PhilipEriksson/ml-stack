#!/usr/bin/env python3
"""NVFP4 (W4A4) quantization via llm-compressor.

Runs in the quant-nvfp4 environment. Produces a compressed-tensors checkpoint
that vLLM auto-detects with no --quantization flag and no patches.

NVFP4 is the only one of the three formats that needs calibration data: the
activation global scales are derived from representative samples. Calibrate on
text the model actually sees, not generic web text.

Note: quantization runs on any CUDA GPU. The FP4 tensor cores only matter for
*running* W4A4 inference — creating the quant does not require Blackwell.

Called by quantize.py — not meant to be run directly, though it can be.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--ml-stack", required=True)
    ap.add_argument("--calib", default=None,
                    help="HF dataset id or .jsonl of calibration text")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def load_ignore(arch: str) -> list[str]:
    path = CONFIGS / "nvfp4-ignore.yaml"
    table = yaml.safe_load(path.read_text()) or {}
    if arch in table:
        print(f"ignore list: {arch} ({len(table[arch])} entries)")
        return table[arch]
    print(
        f"WARNING: no tuned ignore list for {arch}; falling back to 'default' "
        f"(lm_head only).\n"
        f"         The small-layer FP4 threshold has not been measured for this "
        f"architecture.\n"
        f"         Add an entry to {path} once you have compared quality.",
        file=sys.stderr,
    )
    return table.get("default", ["lm_head"])


def load_calibration(calib: str, samples: int, tokenizer) -> list[dict]:
    """Return a list of {'text': ...} rows for llm-compressor."""
    p = Path(calib).expanduser()
    rows: list[str] = []

    if p.is_file():
        print(f"calibration: {p}")
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if "text" in obj:
                rows.append(obj["text"])
            elif "messages" in obj:
                rows.append(tokenizer.apply_chat_template(
                    obj["messages"], tokenize=False))
            elif "prompt" in obj and "response" in obj:
                rows.append(f"{obj['prompt']}\n{obj['response']}")
            if len(rows) >= samples:
                break
    else:
        print(f"calibration: HF dataset {calib}")
        from datasets import load_dataset
        ds = load_dataset(calib, split="train")
        col = "text" if "text" in ds.column_names else ds.column_names[0]
        for i, row in enumerate(ds):
            if i >= samples:
                break
            rows.append(str(row[col]))

    if not rows:
        sys.exit(f"ERROR: no calibration rows extracted from {calib}")
    print(f"calibration rows: {len(rows)}")
    # llm-compressor expects a datasets.Dataset (it reads .column_names), not a
    # plain list.
    from datasets import Dataset
    return Dataset.from_list([{"text": r} for r in rows])


def main() -> int:
    a = parse()
    model_dir = Path(a.model_dir)
    ml_stack = Path(a.ml_stack)

    out_dir = (
        Path(a.out) if a.out
        else ml_stack / "models" / "quantized" / a.family / "nvfp4" / "4bit" / a.slug
    )

    print(f"arch:    {a.arch}")
    print(f"samples: {a.samples}")
    print(f"out:     {out_dir}")

    if not a.calib:
        print(
            "ERROR: --calib is required for NVFP4.\n"
            "       Activation global scales come from calibration data, and the\n"
            "       choice materially affects quality. Point it at text the model\n"
            "       actually sees — e.g. the training corpus you fine-tuned on:\n"
            "         --calib datasets/processed/<family>/<dataset>/train.jsonl\n"
            "       or an HF dataset id:\n"
            "         --calib HuggingFaceH4/ultrachat_200k",
            file=sys.stderr,
        )
        return 1

    if a.dry_run:
        print(f"[dry-run] oneshot NVFP4 on {model_dir} -> {out_dir}")
        return 0

    if out_dir.exists() and any(out_dir.iterdir()) and not a.overwrite:
        print(f"skip: {out_dir} already populated")
        return 0

    import torch
    from transformers import AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    ignore = load_ignore(a.arch)

    # Multimodal architectures need the image-text class; plain LMs use the
    # causal class. Try the richer one first.
    model = None
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            import transformers
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(str(model_dir), dtype="bfloat16")
            print(f"loaded with {cls_name}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"{cls_name} failed: {type(e).__name__}: {e}")
    if model is None:
        print("ERROR: could not load the model with any known auto class",
              file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    ds = load_calibration(a.calib, a.samples, tokenizer)

    # Multimodal models need a processor, not just a tokenizer, when a dataset
    # is supplied — llm-compressor raises "model processor ... is required"
    # otherwise.
    processor = None
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(str(model_dir))
        print("processor: loaded")
    except Exception as e:  # noqa: BLE001
        print(f"processor: unavailable ({type(e).__name__}); using tokenizer only")

    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=ignore)

    print("\nrunning oneshot ...")
    kwargs = dict(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=a.seq_len,
        num_calibration_samples=a.samples,
    )
    if processor is not None:
        kwargs["processor"] = processor
    else:
        kwargs["tokenizer"] = tokenizer
    oneshot(**kwargs)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    total = sum(p.stat().st_size for p in out_dir.rglob("*.safetensors"))
    print(f"\nNVFP4: {total/1e9:.2f} GB -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
