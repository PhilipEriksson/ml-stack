#!/usr/bin/env bash
# add-model: type detection, GGUF quant selection, registry fields, hash stability.
set -u
source "$(dirname "$0")/lib.sh"

S=$(make_stack)
export ML_STACK_HOME="$S"
export PATH="$S/shim:$PATH"
stub_hf "$S"

cat > "$S/shim/curl" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do case "$a" in *api/models/*) repo="${a##*api/models/}";; esac; done
case "$repo" in
  *-GGUF) printf '{"siblings":[{"rfilename":"m-Q4_K_M.gguf"},{"rfilename":"m-Q8_0.gguf"},{"rfilename":"m-Q2_K.gguf"}]}' ;;
  *-LoRA) printf '{"siblings":[{"rfilename":"adapter_config.json"},{"rfilename":"adapter_model.safetensors"}]}' ;;
  *)      printf '{"siblings":[{"rfilename":"model.safetensors"},{"rfilename":"config.json"}]}' ;;
esac
EOF
chmod +x "$S/shim/curl"

REG="$S/configs/models/registry.json"

# A. multi-quant GGUF repo without --file must refuse and list the files
out=$("$S/cli/ml" add-model auto org/Llama-3.1-8B-GGUF 2>&1); rc=$?
assert_eq "multi-quant repo exits non-zero" "$rc" "1"
assert_contains "multi-quant repo lists candidates" "$out" "m-Q8_0.gguf"
assert_contains "multi-quant repo suggests --file" "$out" "--file"

# B. explicit --file registers type=gguf with the file recorded
out=$("$S/cli/ml" add-model auto org/Llama-3.1-8B-GGUF --file m-Q4_K_M.gguf 2>&1); rc=$?
assert_eq "gguf download exits zero" "$rc" "0"
assert_eq "gguf type" "$(jq -r '.llama.models["llama-3.1-8b-gguf"].type' "$REG")" "gguf"
assert_eq "gguf file recorded" "$(jq -r '.llama.models["llama-3.1-8b-gguf"].file' "$REG")" "m-Q4_K_M.gguf"
assert_eq "gguf status ready" "$(jq -r '.llama.models["llama-3.1-8b-gguf"].status' "$REG")" "ready"

# C. adapter repo is finetuned, not base
"$S/cli/ml" add-model auto org/Llama-3.1-8B-LoRA >/dev/null 2>&1
assert_eq "adapter repo type" "$(jq -r '.llama.models["llama-3.1-8b-lora"].type' "$REG")" "finetuned"

# D. plain safetensors repo is base
"$S/cli/ml" add-model auto org/Llama-3.1-8B >/dev/null 2>&1
assert_eq "base repo type" "$(jq -r '.llama.models["llama-3.1-8b"].type' "$REG")" "base"

# E. hash is stable across re-downloads (hf writes .cache/ metadata each time)
h1=$(jq -r '.llama.models["llama-3.1-8b-gguf"].hash' "$REG")
"$S/cli/ml" add-model auto org/Llama-3.1-8B-GGUF --file m-Q4_K_M.gguf >/dev/null 2>&1
h2=$(jq -r '.llama.models["llama-3.1-8b-gguf"].hash' "$REG")
assert_eq "hash stable across re-download" "$h2" "$h1"

# F. unknown family needs --family
out=$("$S/cli/ml" add-model auto org/Unknown-Model 2>&1); rc=$?
assert_eq "unknown family exits non-zero" "$rc" "1"
assert_contains "unknown family suggests --family" "$out" "--family"

out=$("$S/cli/ml" add-model auto org/Unknown-Model --family custom 2>&1); rc=$?
assert_eq "family override exits zero" "$rc" "0"
assert_eq "family override applied" "$(jq -r '.custom.models["unknown-model"].type' "$REG")" "base"

# G. --type gguf on a repo with no .gguf files is rejected
out=$("$S/cli/ml" add-model auto org/Llama-3.1-8B --type gguf 2>&1); rc=$?
assert_eq "type gguf without gguf files exits non-zero" "$rc" "1"

rm -rf "$S"
finish
