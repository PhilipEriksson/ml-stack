#!/usr/bin/env bash
# status: reports registries, environments, and what is missing.
set -u
source "$(dirname "$0")/lib.sh"

S=$(make_stack)
export ML_STACK_HOME="$S"
export PATH="$S/shim:$PATH"

cat > "$S/shim/docker" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$S/shim/docker"

cat > "$S/configs/models/registry.json" <<'EOF'
{"llama":{"models":{"test-model":{"type":"gguf","variant":"instruct","path":"/tmp","status":"ready"}}}}
EOF
cat > "$S/configs/datasets/registry.json" <<'EOF'
{"llama":{"datasets":{"test-data":{"status":"processed"}}}}
EOF
cat > "$S/configs/runs/registry.json" <<'EOF'
{"run-a":{"model":"llama/test-model","dataset":"llama/test-data","method":"lora","status":"created","timestamp":"2026-01-01_00-00-00"}}
EOF

out=$("$S/cli/ml" status 2>&1); rc=$?
assert_eq "status exits zero" "$rc" "0"
assert_contains "status shows environments" "$out" "Environments"
assert_contains "status shows models" "$out" "llama/test-model"
assert_contains "status shows datasets" "$out" "llama/test-data"
assert_contains "status shows runs" "$out" "run-a"
assert_contains "status flags the missing vLLM env" "$out" "active.env"
assert_contains "status has a warnings section" "$out" "Needs attention"

# with an active vLLM env and a built llama-server, that warning disappears
printf 'MODEL_NAME=test/model\n' > "$S/configs/vllm/active.env"
printf '#!/usr/bin/env bash\n' > "$S/runtimes/llama.cpp/build/bin/llama-server"
chmod +x "$S/runtimes/llama.cpp/build/bin/llama-server"

out=$("$S/cli/ml" status 2>&1)
assert_contains "status shows the active vLLM model" "$out" "test/model"
assert_not_contains "no active.env warning once set" "$out" "active.env missing"

rm -rf "$S"
finish
