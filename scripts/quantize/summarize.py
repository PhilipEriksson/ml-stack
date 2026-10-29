#!/usr/bin/env python3
"""Aggregate verify-*.json reports into one comparison table.

When several reports cover the same artifact label, the NEWEST report wins —
otherwise a stale run silently overwrites a fresh one (alphabetical glob order
is not chronological).

Usage:
    python summarize.py [report-dir]

report-dir defaults to the current directory. Labels are whatever you passed to
verify.py's --artifact kind:label:path, so the table works for any model.
"""
import json
import pathlib
import sys

D = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")

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

if not rows:
    sys.exit(f"no verify-*.json reports in {D.resolve()}")

print(f"{'artifact':14s} {'kind':6s} {'close':7s} {'expect':8s}  per-prompt")
print("-" * 78)
for label in sorted(rows):
    art = rows[label]
    res = art["results"]
    close = sum(1 for r in res if r["has_close_tag"])
    exp = sum(1 for r in res if r["expect_found"])
    detail = " ".join(
        f"{r['id']}={'ok' if r['expect_found'] else 'MISS'}" for r in res)
    print(f"{label:14s} {art['kind']:6s} {close}/{len(res):<5d} {exp}/{len(res):<6d}  {detail}")
