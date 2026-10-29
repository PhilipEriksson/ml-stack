#!/usr/bin/env python3
"""EXL3 (QTIP trellis) quantization via exllamav3's convert.py.

Runs in the quant-exl3 environment (torch 2.10.0+cu128 + exllamav3 1.6.5).
convert.py is a repo script, not part of the wheel, so it is cloned into
ml-stack/runtimes/exllamav3.

EXL3 is a streamlined QTIP variant: trellis-coded quantization against a learned
codebook lattice, Hessians computed on the fly with a fused Viterbi kernel,
single pass. No calibration dataset needed.

Called by quantize.py — not meant to be run directly, though it can be.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CONVERT = Path.home() / "ml-stack/runtimes/exllamav3/convert.py"


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--ml-stack", required=True)
    ap.add_argument("--bpw", default="4.0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    a = parse()
    model_dir = Path(a.model_dir)
    ml_stack = Path(a.ml_stack)
    bpws = [b.strip() for b in a.bpw.split(",") if b.strip()]

    if not CONVERT.exists():
        print(f"ERROR: convert.py not found at {CONVERT}", file=sys.stderr)
        print("       git clone https://github.com/turboderp-org/exllamav3 "
              f"{CONVERT.parent}", file=sys.stderr)
        return 1

    print(f"arch: {a.arch}")
    print(f"bpw:  {', '.join(bpws)}")

    results = []
    for bpw in bpws:
        out_dir = (
            Path(a.out) / bpw if a.out
            else ml_stack / "models" / "quantized" / a.family / "exl3" / f"{bpw}bpw" / a.slug
        )
        work_dir = (
            Path(a.work) / f"work-{bpw}" if a.work
            else ml_stack / "outputs" / "quantize" / a.slug / f"exl3-{bpw}bpw"
        )

        cmd = [
            sys.executable, str(CONVERT),
            "-i", str(model_dir),
            "-o", str(out_dir),
            "-w", str(work_dir),
            "-b", bpw,
        ]

        print(f"\n=== EXL3 {bpw} bpw ===")
        print(" ".join(cmd))

        if a.dry_run:
            continue

        if out_dir.exists() and any(out_dir.iterdir()) and not a.overwrite:
            print(f"skip {bpw}: {out_dir} already populated")
            results.append(out_dir)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"ERROR: convert.py exited {rc} for {bpw} bpw", file=sys.stderr)
            return rc

        # convert.py writes safetensors + config into out_dir
        produced = sorted(out_dir.glob("*.safetensors"))
        if not produced:
            print(f"ERROR: no safetensors in {out_dir}", file=sys.stderr)
            return 1
        total = sum(p.stat().st_size for p in produced)
        print(f"{bpw} bpw: {len(produced)} shard(s), {total/1e9:.2f} GB -> {out_dir}")
        results.append(out_dir)

    if not a.dry_run:
        print("\n=== EXL3 outputs ===")
        for r in results:
            print(f"  {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
