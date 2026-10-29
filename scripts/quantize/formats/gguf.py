#!/usr/bin/env python3
"""GGUF conversion + quantization via Unsloth's bundled llama.cpp converter.

Runs in the training-cuda13 environment (Unsloth is installed there). Unsloth
stages its own co-versioned convert_hf_to_gguf + gguf-py, which is why this path
works for architectures where stock llama.cpp's converter does not.

Called by quantize.py — not meant to be run directly, though it can be.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Unsloth must be imported before transformers.
import unsloth  # noqa: F401
from unsloth.save import convert_to_gguf, quantize_gguf

# llama.cpp issue #27019: the stock converter's _reorder_v_heads hardcodes
# head_dim=1 for the linear-attention V-head reorder and does not expand
# in_proj_a/in_proj_b. Closed as not planned; the runtime is fine, only the
# converter is affected.
QWEN35_CONVERTER_BUG = "shape '[16, 2, 1, 1]' is invalid for input of size"

# llama-quantize lives in the ml-stack runtimes tree. Unsloth's default
# (~/.unsloth/llama.cpp/llama-quantize) is only populated if Unsloth itself
# built llama.cpp, which it does not do for a bare convert_to_gguf call.
QUANTIZER_CANDIDATES = [
    Path.home() / "ml-stack/runtimes/llama.cpp/build/bin/llama-quantize",
    Path.home() / ".unsloth/llama.cpp/llama-quantize",
    Path.home() / ".unsloth/llama.cpp/build/bin/llama-quantize",
]


def find_quantizer() -> Path | None:
    for p in QUANTIZER_CANDIDATES:
        if p.exists():
            return p
    return None


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--ml-stack", required=True)
    ap.add_argument("--quants", default="Q4_K_M,Q8_0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    a = parse()
    model_dir = Path(a.model_dir)
    ml_stack = Path(a.ml_stack)
    quants = [q.strip() for q in a.quants.split(",") if q.strip()]

    work = Path(a.work) if a.work else ml_stack / "outputs" / "quantize" / a.slug / "gguf"
    work.mkdir(parents=True, exist_ok=True)

    print(f"arch:   {a.arch}")
    print(f"quants: {', '.join(quants)}")
    print(f"work:   {work}")

    if a.dry_run:
        print(f"[dry-run] convert {model_dir} -> f16 GGUF in {work}")
        for q in quants:
            print(f"[dry-run] quantize f16 -> {q}")
        return 0

    # ---- 1. convert to f16 ----
    f16 = work / f"{a.slug}-f16.gguf"
    if f16.exists() and not a.overwrite:
        print(f"reusing existing {f16}")
    else:
        # Stage Unsloth's co-versioned converter + gguf-py. This is what makes
        # the export work for architectures where stock llama.cpp's converter
        # does not (e.g. qwen3_5's linear-attention tensors).
        from unsloth_zoo.llama_cpp import _download_convert_hf_to_gguf

        converter, text_archs, vision_archs = _download_convert_hf_to_gguf()
        print(f"converter: {converter}")

        if a.arch not in text_archs:
            print(
                f"ERROR: {a.arch} is not in Unsloth's supported text architectures.\n"
                f"       The converter would fail or produce a broken GGUF.\n"
                f"       Supported: {len(text_archs)} architectures "
                f"(see unsloth_zoo.llama_cpp._download_convert_hf_to_gguf).",
                file=sys.stderr,
            )
            return 3

        print(f"\nconverting {model_dir} -> f16 ...")
        try:
            files, is_vlm = convert_to_gguf(
                model_name=a.slug,
                input_folder=str(model_dir),
                model_dtype="bf16",
                quantization_type="f16",
                converter_location=converter,
                supported_text_archs=text_archs,
                supported_vision_archs=vision_archs,
                print_output=True,
            )
        except RuntimeError as e:
            msg = str(e)
            if QWEN35_CONVERTER_BUG in msg:
                print(
                    "\nERROR: hit the known llama.cpp qwen3_5 converter bug "
                    "(issue #27019).\n"
                    "  _reorder_v_heads hardcodes head_dim=1 for the "
                    "linear-attention V-head reorder\n"
                    "  and does not expand in_proj_a/in_proj_b.\n"
                    "  https://github.com/ggml-org/llama.cpp/issues/27019\n"
                    "\n"
                    "  Workaround from the issue: --no-mtp plus an a/b transpose "
                    "plus the conv kernel fix,\n"
                    "  or a post-process mmap patch. The runtime is unaffected — "
                    "only the converter.\n",
                    file=sys.stderr,
                )
                return 2
            raise

        if not files:
            print("ERROR: converter produced no output files", file=sys.stderr)
            return 1
        produced = Path(files[0])
        if produced.resolve() != f16.resolve():
            shutil.move(str(produced), str(f16))
        print(f"f16 GGUF: {f16} ({f16.stat().st_size/1e9:.2f} GB)")

    # ---- 2. quantize ----
    quantizer = find_quantizer()
    if quantizer is None:
        print(
            "ERROR: llama-quantize not found. Build it:\n"
            "  cd ~/ml-stack/runtimes && git clone --depth 1 "
            "https://github.com/ggml-org/llama.cpp\n"
            "  cd llama.cpp && cmake -B build -DGGML_CUDA=OFF && "
            "cmake --build build -j --target llama-quantize llama-cli\n"
            f"  looked in: {[str(p) for p in QUANTIZER_CANDIDATES]}",
            file=sys.stderr,
        )
        return 1
    print(f"quantizer: {quantizer}")

    results = []
    for q in quants:
        out_dir = (
            Path(a.out) / q if a.out
            else ml_stack / "models" / "quantized" / a.family / "gguf" / q / a.slug
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{a.slug}-{q}.gguf"

        if out.exists() and not a.overwrite:
            print(f"skip {q}: {out} exists")
            results.append(out)
            continue

        print(f"\nquantizing -> {q} ...")
        quantize_gguf(str(f16), str(out), q,
                      quantizer_location=str(quantizer),
                      print_output=True)
        if not out.exists():
            print(f"ERROR: {q} produced no file at {out}", file=sys.stderr)
            return 1
        print(f"{q}: {out} ({out.stat().st_size/1e9:.2f} GB)")
        results.append(out)

    print("\n=== GGUF outputs ===")
    for r in results:
        print(f"  {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
