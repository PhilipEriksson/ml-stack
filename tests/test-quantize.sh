#!/usr/bin/env bash
# quantize: dispatcher resolves the model, architecture, and per-format interpreter.
set -u
source "$(dirname "$0")/lib.sh"

S=$(make_stack)
export ML_STACK_HOME="$S"
export PATH="$S/shim:$PATH"

MODEL="$S/fakemodel"
mkdir -p "$MODEL"
printf '{"architectures":["LlamaForCausalLM"],"model_type":"llama"}' > "$MODEL/config.json"

# A dry run must work without the per-format environments installed.
out=$("$S/cli/ml" quantize "$MODEL" --format all --dry-run 2>&1); rc=$?
assert_eq "dry-run exits zero" "$rc" "0"
assert_contains "arch resolved" "$out" "LlamaForCausalLM"
assert_contains "family resolved from model_type" "$out" "--family llama"
assert_contains "gguf module dispatched" "$out" "formats/gguf.py"
assert_contains "exl3 module dispatched" "$out" "formats/exl3.py"
assert_contains "nvfp4 module dispatched" "$out" "formats/nvfp4.py"
assert_contains "exl3 uses quant-exl3" "$out" "envs/quant-exl3/bin/python"
assert_contains "nvfp4 uses quant-nvfp4" "$out" "envs/quant-nvfp4/bin/python"

# --family overrides the model_type-derived directory
out=$("$S/cli/ml" quantize "$MODEL" --format gguf --family custom --dry-run 2>&1)
assert_contains "--family override applied" "$out" "--family custom"

# a directory without config.json is rejected
mkdir -p "$S/notamodel"
out=$("$S/cli/ml" quantize "$S/notamodel" --format gguf --dry-run 2>&1); rc=$?
assert_eq "missing config.json exits non-zero" "$rc" "1"
assert_contains "missing config.json explains why" "$out" "config.json"

rm -rf "$S"
finish
