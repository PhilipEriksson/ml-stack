#!/usr/bin/env bash
# serve-model: type gate, .gguf resolution, --file override, --threads, --stop-vllm.
set -u
source "$(dirname "$0")/lib.sh"

S=$(make_stack)
export ML_STACK_HOME="$S"
export PATH="$S/shim:$PATH"
stub_llama_server "$S"

GGUF_DIR="$S/models/quantized/llama/gguf/test"
mkdir -p "$GGUF_DIR"
touch "$GGUF_DIR/a-Q4_K_M.gguf" "$GGUF_DIR/b-Q8_0.gguf"

cat > "$S/configs/models/registry.json" <<EOF
{"llama":{"models":{
  "with-file":{"type":"gguf","variant":"instruct","path":"$GGUF_DIR","file":"a-Q4_K_M.gguf","status":"ready"},
  "no-file":{"type":"gguf","variant":"instruct","path":"$GGUF_DIR","status":"ready"},
  "legacy":{"type":"quantized","variant":"instruct","path":"$GGUF_DIR","file":"b-Q8_0.gguf","status":"ready"},
  "hfmodel":{"type":"base","variant":"pretrained","path":"/tmp","status":"ready"}
}}}
EOF

ARGS="$S/llama-args"

# A. recorded file is used
"$S/cli/ml" serve-model llama with-file >/dev/null 2>&1
assert_contains "uses recorded file" "$(cat "$ARGS")" "a-Q4_K_M.gguf"

# B. several candidates and no recorded file is an error, not a guess
out=$("$S/cli/ml" serve-model llama no-file 2>&1); rc=$?
assert_eq "ambiguous gguf dir exits non-zero" "$rc" "1"
assert_contains "ambiguous gguf dir lists candidates" "$out" "b-Q8_0.gguf"

# C. --file overrides the registry
"$S/cli/ml" serve-model llama no-file --file b-Q8_0.gguf >/dev/null 2>&1
assert_contains "--file override wins" "$(cat "$ARGS")" "b-Q8_0.gguf"

# D. legacy type=quantized still serves
"$S/cli/ml" serve-model llama legacy >/dev/null 2>&1
assert_contains "legacy quantized type serves" "$(cat "$ARGS")" "b-Q8_0.gguf"

# E. non-GGUF types are rejected
out=$("$S/cli/ml" serve-model llama hfmodel 2>&1); rc=$?
assert_eq "base type rejected" "$rc" "1"
assert_contains "base type points at vLLM" "$out" "vllm"

# F. --threads is passed through
"$S/cli/ml" serve-model llama with-file --threads 4 >/dev/null 2>&1
assert_contains "--threads passed" "$(cat "$ARGS")" "--threads 4"

# G. a missing --file target is an error
out=$("$S/cli/ml" serve-model llama no-file --file nope.gguf 2>&1); rc=$?
assert_eq "missing file exits non-zero" "$rc" "1"

# H. non-interactive without --stop-vllm must NOT stop the container
cat > "$S/shim/docker" <<EOF
#!/usr/bin/env bash
case "\$1" in
  ps) echo "vllm-server" ;;
  stop) echo "stopped \$2" >> "$S/docker-stops" ;;
esac
EOF
chmod +x "$S/shim/docker"

"$S/cli/ml" serve-model llama with-file </dev/null >/dev/null 2>&1
if [ -f "$S/docker-stops" ]; then
  fail "non-interactive run leaves vLLM alone" "docker stop was called"
else
  ok "non-interactive run leaves vLLM alone"
fi

# I. --stop-vllm stops it explicitly
"$S/cli/ml" serve-model llama with-file --stop-vllm </dev/null >/dev/null 2>&1
assert_contains "--stop-vllm stops the container" "$(cat "$S/docker-stops" 2>/dev/null)" "stopped vllm-server"

rm -rf "$S"
finish
