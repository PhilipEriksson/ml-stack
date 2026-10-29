#!/usr/bin/env bash
# eval: result-file detection and score extraction into the registry.
set -u
source "$(dirname "$0")/lib.sh"

S=$(make_stack)
export ML_STACK_HOME="$S"
export PATH="$S/shim:$PATH"

printf 'MODEL_NAME=test/model\n' > "$S/configs/vllm/active.env"

# lm-eval writes into a per-model subdirectory under --output_base_dir
cat > "$S/shim/lm-eval" <<'EOF'
#!/usr/bin/env bash
dir=""
while [ $# -gt 0 ]; do
  case "$1" in --output_base_dir) dir="$2"; shift 2 ;; *) shift ;; esac
done
mkdir -p "$dir/test-model"
printf '{"results":{"mmlu":{"accuracy":0.71},"gsm8k":{"exact_match":0.55}}}' \
  > "$dir/test-model/results_$(date +%s%N).json"
EOF
chmod +x "$S/shim/lm-eval"

# docker is only probed for a running container; keep it quiet
cat > "$S/shim/docker" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$S/shim/docker"

REG="$S/configs/evals/registry.json"

out=$("$S/cli/ml" eval mmlu,gsm8k 4 2>&1); rc=$?
assert_eq "eval exits zero" "$rc" "0"
assert_contains "eval reports the result file" "$out" "results_"
assert_eq "registry entry created" "$(jq -r '.["test-model"].evals | length' "$REG")" "1"
assert_eq "mmlu score recorded" "$(jq -r '.["test-model"].evals[0].scores.mmlu' "$REG")" "0.71"
assert_eq "gsm8k score recorded" "$(jq -r '.["test-model"].evals[0].scores.gsm8k' "$REG")" "0.55"
assert_eq "tasks recorded" "$(jq -r '.["test-model"].evals[0].tasks' "$REG")" "mmlu,gsm8k"

# a second run appends rather than replacing
"$S/cli/ml" eval mmlu 4 >/dev/null 2>&1
assert_eq "second eval appended" "$(jq -r '.["test-model"].evals | length' "$REG")" "2"

out=$("$S/cli/ml" evals test-model 2>&1)
assert_contains "ml evals lists the model" "$out" "test-model"
assert_contains "ml evals shows latest scores" "$out" "Latest scores"

rm -rf "$S"
finish
