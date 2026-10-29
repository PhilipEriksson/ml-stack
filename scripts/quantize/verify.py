#!/usr/bin/env python3
"""Compare quantized artifacts against a baseline on a fixed prompt set.

Supports three artifact kinds:
    gguf  -> llama-cli
    hf    -> transformers (BF16, NVFP4/compressed-tensors)
    exl3  -> exllamav3

Writes a JSON report so results can be diffed across runs rather than eyeballed.

Usage:
    python verify.py --prompts configs/verify-prompts.yaml \
        --artifact hf:bf16:/path/to/merged_16bit \
        --artifact gguf:Q4_K_M:/path/to/model-Q4_K_M.gguf \
        --artifact exl3:4.0bpw:/path/to/exl3-dir \
        --out report.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parent / "configs"
LLAMA_CLI = Path.home() / "ml-stack/runtimes/llama.cpp/build/bin/llama-cli"

OPEN_TAG = "<" + "think" + ">"
CLOSE_TAG = "<" + "/" + "think" + ">"


def load_prompts(path: Path, slug: str | None) -> list[dict]:
    table = yaml.safe_load(path.read_text()) or {}
    if slug and slug in table:
        return table[slug]
    return table.get("default", [])


def run_gguf(model_path: Path, prompt: str, max_new: int) -> str:
    if not LLAMA_CLI.exists():
        return f"ERROR: llama-cli not found at {LLAMA_CLI}"
    # Apply the Qwen chat scaffold manually so the comparison is apples-to-apples.
    full = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{OPEN_TAG}\n"
    )
    cmd = [
        str(LLAMA_CLI), "-m", str(model_path),
        "-p", full, "-n", str(max_new), "--temp", "0.6",
        "--no-display-prompt",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return out.stdout.strip() or out.stderr.strip()[-500:]
    except subprocess.TimeoutExpired:
        return "ERROR: llama-cli timed out"


def run_hf(model_dir: Path, prompt: str, max_new: int) -> str:
    import torch
    import transformers

    model = None
    errors = []
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            cls = getattr(transformers, cls_name)
            # device_map="cuda" is required — from_pretrained defaults to CPU,
            # which makes generation ~50x slower and looks like a hang.
            model = cls.from_pretrained(
                str(model_dir), dtype="bfloat16", device_map="cuda")
            break
        except Exception as e:  # noqa: BLE001
            # Never swallow the cause: a bare "could not load model" hides a
            # missing dependency (e.g. compressed-tensors for NVFP4) and costs
            # a debugging cycle. Keep the last real error and report it.
            errors.append(f"{cls_name}: {type(e).__name__}: {e}")
            continue
    if model is None:
        return "ERROR: could not load model\n" + "\n".join(errors)

    tok = transformers.AutoTokenizer.from_pretrained(str(model_dir))
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    except TypeError:
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, temperature=0.6,
                             top_p=0.95, do_sample=True)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


def run_exl3(model_dir: Path, prompt: str, max_new: int) -> str:
    """EXL3 inference via exllamav3's Generator.generate()."""
    try:
        from exllamav3 import Cache, Config, Generator, Model, Tokenizer
        from exllamav3.generator.sampler import DefaultSampler
    except ImportError as e:
        return f"ERROR: exllamav3 not importable in this interpreter: {e}"

    config = Config.from_directory(str(model_dir))
    model = Model.from_config(config)
    # Cache must be constructed BEFORE model.load() — exllamav3 allocates the
    # cache tensors during load and asserts on an uninitialized cache otherwise.
    cache = Cache(model, max_num_tokens=4096)
    model.load()
    tok = Tokenizer.from_config(config)
    gen = Generator(model=model, cache=cache, tokenizer=tok)

    text = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{OPEN_TAG}\n"
    )
    out = gen.generate(prompt=text, max_new_tokens=max_new,
                       sampler=DefaultSampler())
    if isinstance(out, list):
        return out[0] if out else ""
    return str(out)


RUNNERS = {"gguf": run_gguf, "hf": run_hf, "exl3": run_exl3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(CONFIGS / "verify-prompts.yaml"))
    ap.add_argument("--artifact", action="append", required=True,
                    help="kind:label:path  (kind = gguf|hf|exl3)")
    ap.add_argument("--slug", default=None, help="prompt-set key override")
    ap.add_argument("--max-new", type=int, default=700)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompts = load_prompts(Path(args.prompts), args.slug)
    if not prompts:
        sys.exit("ERROR: no prompts loaded")

    report = {"prompts": [p["id"] for p in prompts], "artifacts": {}}

    for spec in args.artifact:
        kind, label, path = spec.split(":", 2)
        if kind not in RUNNERS:
            sys.exit(f"ERROR: unknown artifact kind {kind!r}")
        p = Path(path).expanduser()
        print(f"\n=== {label} ({kind}) ===")
        results = []
        for pr in prompts:
            print(f"  {pr['id']} ...", end=" ", flush=True)
            text = RUNNERS[kind](p, pr["prompt"], args.max_new)
            expect = pr.get("expect")
            # llama-cli renders the closing think token as "[End thinking]"
            # rather than emitting the raw token, so accept either form.
            has_close = (CLOSE_TAG in text) or ("[End thinking]" in text)
            results.append({
                "id": pr["id"],
                "output": text,
                "chars": len(text),
                "has_close_tag": has_close,
                "expect": expect,
                "expect_found": (expect in text) if expect else None,
            })
            print(f"{len(text)} chars, close_tag={has_close}, "
                  f"expect={expect in text if expect else 'n/a'}")
        report["artifacts"][label] = {"kind": kind, "path": str(p), "results": results}

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
