#!/usr/bin/env python3
"""Quantization dispatcher for the ML Stack.

Quantizes any supported model into GGUF, EXL3, or NVFP4. Runs in any Python
(stdlib only) and shells out to the per-format interpreter, because each format
needs a different environment:

    gguf  -> training-cuda13  (Unsloth's bundled converter)
    exl3  -> quant-exl3       (exllamav3, torch 2.10.0+cu128)
    nvfp4 -> quant-nvfp4      (llm-compressor)

Usage:
    ml quantize <model> --format gguf  --quants Q4_K_M,Q8_0
    ml quantize <model> --format exl3  --bpw 4.0,3.0
    ml quantize <model> --format nvfp4 --calib <dataset-or-jsonl> --samples 256
    ml quantize <model> --format all   --dry-run

<model> is an HF repo id or a local directory.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ML_STACK = Path(os.environ.get("ML_STACK_HOME", Path.home() / "ml-stack"))
SCRIPTS = ML_STACK / "scripts" / "quantize"
CONFIGS = SCRIPTS / "configs"

# Each format runs in its own interpreter. This is the whole reason the
# dispatcher shells out instead of importing the format modules.
ENV_PYTHON = {
    "gguf": Path.home() / "miniconda/envs/training-cuda13/bin/python",
    "exl3": Path.home() / "miniconda/envs/quant-exl3/bin/python",
    "nvfp4": Path.home() / "miniconda/envs/quant-nvfp4/bin/python",
}

FORMATS = ("gguf", "exl3", "nvfp4")


# --------------------------------------------------------------------------
# model resolution
# --------------------------------------------------------------------------
def resolve_model(model: str) -> Path:
    """Accept an HF repo id or a local directory; return a local directory."""
    p = Path(model).expanduser()
    if p.is_dir():
        return p.resolve()

    # HF repo id -> snapshot_download
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "ERROR: model is not a local directory and huggingface_hub is not "
            "installed in this interpreter.\n"
            f"       Pass a local path instead, or run: pip install huggingface_hub"
        )
    print(f"downloading {model} ...")
    return Path(snapshot_download(model))


def read_config(model_dir: Path) -> dict:
    cfg = model_dir / "config.json"
    if not cfg.exists():
        sys.exit(f"ERROR: no config.json in {model_dir} — not a HF model directory")
    return json.loads(cfg.read_text())


def arch_of(cfg: dict) -> str:
    archs = cfg.get("architectures") or []
    if not archs:
        sys.exit("ERROR: config.json has no 'architectures' entry")
    return archs[0]


def family_of(cfg: dict, model_dir: Path) -> str:
    """Family for the output path. Prefers model_type, falls back to dir name."""
    mt = cfg.get("model_type")
    if mt:
        return mt
    return model_dir.name.split("-")[0].lower()


def slug_of(model_dir: Path) -> str:
    return model_dir.name


def out_root(family: str, fmt: str, bits: str, slug: str) -> Path:
    """models/quantized/<family>/<format>/<bits>/<slug>/ — matches the README."""
    return ML_STACK / "models" / "quantized" / family / fmt / bits / slug


# --------------------------------------------------------------------------
# format dispatch
# --------------------------------------------------------------------------
def run_format(fmt: str, model_dir: Path, cfg: dict, args) -> int:
    py = ENV_PYTHON[fmt]
    if not py.exists():
        print(f"ERROR: {fmt} environment missing: {py}")
        print(f"       see {SCRIPTS}/README.md for setup")
        return 1

    script = SCRIPTS / "formats" / f"{fmt}.py"
    cmd = [
        str(py), str(script),
        "--model-dir", str(model_dir),
        "--arch", arch_of(cfg),
        "--family", family_of(cfg, model_dir),
        "--slug", args.slug or slug_of(model_dir),
        "--ml-stack", str(ML_STACK),
    ]

    if fmt == "gguf":
        cmd += ["--quants", args.quants]
    elif fmt == "exl3":
        cmd += ["--bpw", args.bpw]
    elif fmt == "nvfp4":
        cmd += ["--samples", str(args.samples)]
        if args.calib:
            cmd += ["--calib", args.calib]

    if args.out:
        cmd += ["--out", args.out]
    if args.work:
        cmd += ["--work", args.work]
    if args.overwrite:
        cmd += ["--overwrite"]
    if args.dry_run:
        cmd += ["--dry-run"]

    print(f"\n=== {fmt} ===")
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="ml quantize",
        description="Quantize a model into GGUF, EXL3, or NVFP4.",
    )
    ap.add_argument("model", help="HF repo id or local model directory")
    ap.add_argument("--format", required=True, choices=FORMATS + ("all",))
    ap.add_argument("--quants", default="Q4_K_M,Q8_0",
                    help="GGUF quant types, comma-separated (default: Q4_K_M,Q8_0)")
    ap.add_argument("--bpw", default="4.0",
                    help="EXL3 bits per weight, comma-separated (default: 4.0)")
    ap.add_argument("--calib", default=None,
                    help="NVFP4 calibration source: HF dataset id or .jsonl path")
    ap.add_argument("--samples", type=int, default=256,
                    help="NVFP4 calibration samples (default: 256)")
    ap.add_argument("--out", default=None, help="override output directory")
    ap.add_argument("--work", default=None, help="override working directory")
    ap.add_argument("--slug", default=None,
                    help="output name (default: the model directory name)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    args = ap.parse_args()

    model_dir = resolve_model(args.model)
    cfg = read_config(model_dir)
    arch = arch_of(cfg)

    print(f"model:  {model_dir}")
    print(f"arch:   {arch}")
    print(f"family: {family_of(cfg, model_dir)}")

    targets = FORMATS if args.format == "all" else (args.format,)
    rc = 0
    for fmt in targets:
        rc |= run_format(fmt, model_dir, cfg, args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
