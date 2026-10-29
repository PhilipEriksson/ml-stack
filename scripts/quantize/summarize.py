#!/usr/bin/env python3
"""Aggregate verify-*.json reports into one comparison table.

When several reports cover the same artifact label, the NEWEST report wins —
otherwise a stale run silently overwrites a fresh one (alphabetical glob order
is not chronological).

Usage:
    python summarize.py [report-dir]
"""
import json
import pathlib
import sys

D = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (
    pathlib.Path.home() / "ml-stack/outputs/quantize/qwopus-0.8b")
ORDER = ["bf16", "Q8_0", "Q4_K_M", "4.0bpw", "3.0bpw", "nvfp4"]

rows = {}
stamp = {}
for f in sorted(D.glob("verify-*.json")):
    rep = json.loads(f.read_text())
    mtime = f.stat().st_mtime
    for label, art in rep["artifacts"].items():
        if label in stamp and stamp[label] > mtime:
            continue  # an older report must not overwrite a newer one
        rows[label] = art
        stamp[label] = mtime

print(f"{'artifact':10s} {'kind':6s} {'close':6s} {'expect':7s}  per-prompt")
print("-" * 78)
for label in ORDER:
    if label not in rows:
        continue
    art = rows[label]
    res = art["results"]
    close = sum(1 for r in res if r["has_close_tag"])
    exp = sum(1 for r in res if r["expect_found"])
    detail = " ".join(
        f"{r['id']}={'ok' if r['expect_found'] else 'MISS'}" for r in res)
    print(f"{label:10s} {art['kind']:6s} {close}/{len(res):<4d} {exp}/{len(res):<5d}  {detail}")

print()
print("sizes:")
for p in sorted((pathlib.Path.home() / "ml-stack/models/quantized/qwen3_5").rglob("*.gguf")):
    print(f"  {p.parent.parent.name:8s} {p.stat().st_size/1e6:7.0f} MB")
for p in sorted((pathlib.Path.home() / "ml-stack/models/quantized/qwen3_5").glob("exl3/*/qwopus-0.8b")):
    tot = sum(x.stat().st_size for x in p.rglob("*.safetensors"))
    print(f"  {p.parent.name:8s} {tot/1e6:7.0f} MB")
for p in sorted((pathlib.Path.home() / "ml-stack/models/quantized/qwen3_5").glob("nvfp4/*/qwopus-0.8b")):
    tot = sum(x.stat().st_size for x in p.rglob("*.safetensors"))
    print(f"  {'nvfp4':8s} {tot/1e6:7.0f} MB")
