#!/usr/bin/env bash
# eval: suites, prefetch, result-file detection, score extraction.
set -u
source "$(dirname "$0")/lib.sh"

S=$(make_stack)
export ML_STACK_HOME="$S"
export PATH="$S/shim:$PATH"

# Suites file is read from the stack root
cp "$REPO/configs/eval-suites.json" "$S/configs/eval-suites.json"

# lm-eval 0.4.x: `run --model ... --tasks a b --output_path DIR`
cat > "$S/shim/lm-eval" <<'EOF'
#!/usr/bin/env bash
model=""; dir=""; tasks=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) model="$2"; shift 2 ;;
    --output_path) dir="$2"; shift 2 ;;
    --tasks) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do tasks="$tasks $1"; shift; done ;;
    *) shift ;;
  esac
done
if [ "$model" = "dummy" ]; then
  echo "prefetched:$tasks"
  exit 0
fi
mkdir -p "$dir/stub"
cat > "$dir/stub/results_$(date +%s%N).json" <<JSON
{"results":{"mmlu":{"acc,none":0.71,"acc_stderr,none":0.01},
            "gsm8k":{"exact_match,strict-match":0.55,"exact_match_stderr,strict-match":0.02}},
 "lm_eval_version":"0.4.13",
 "n-samples":{"mmlu":{"original":2},"gsm8k":{"original":2}}}
JSON
EOF
chmod +x "$S/shim/lm-eval"
export ML_EVAL_BIN="$S/shim/lm-eval"
export ML_EVAL_BACKEND_URL="http://127.0.0.1:9/v1"
export ML_EVAL_MODEL="stub"

# docker is only probed for a running container
cat > "$S/shim/docker" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$S/shim/docker"

REG="$S/configs/evals/registry.json"

# A. suite listing
out=$("$S/cli/ml" eval --list-suites 2>&1); rc=$?
assert_eq "list-suites exits zero" "$rc" "0"
assert_contains "list-suites shows core" "$out" "core"
assert_contains "list-suites shows reasoning" "$out" "reasoning"
assert_contains "list-suites shows tasks" "$out" "gpqa_diamond_zeroshot"

# B. unknown suite is rejected
out=$("$S/cli/ml" eval --suite nope 2>&1); rc=$?
assert_eq "unknown suite exits non-zero" "$rc" "1"
assert_contains "unknown suite lists the valid ones" "$out" "core"

# C. prefetch runs the dummy model and exits without a registry entry
out=$("$S/cli/ml" eval --suite core --prefetch 2>&1); rc=$?
assert_eq "prefetch exits zero" "$rc" "0"
assert_contains "prefetch reports cached" "$out" "Datasets cached"
if [ -f "$REG" ]; then
  fail "prefetch writes no registry entry" "registry exists"
else
  ok "prefetch writes no registry entry"
fi

# D. a real run records scores with metric names
out=$("$S/cli/ml" eval --suite core --limit 2 2>&1); rc=$?
assert_eq "eval exits zero" "$rc" "0"
assert_contains "eval reports the result file" "$out" "results_"
assert_eq "registry entry created" "$(jq -r '.["stub"].evals | length' "$REG")" "1"
assert_eq "mmlu score recorded" "$(jq -r '.["stub"].evals[0].scores.mmlu.value' "$REG")" "0.71"
assert_eq "mmlu metric name recorded" "$(jq -r '.["stub"].evals[0].scores.mmlu.metric' "$REG")" "acc"
assert_eq "gsm8k score recorded" "$(jq -r '.["stub"].evals[0].scores.gsm8k.value' "$REG")" "0.55"
assert_eq "gsm8k metric name recorded" "$(jq -r '.["stub"].evals[0].scores.gsm8k.metric' "$REG")" "exact_match"
assert_eq "suite recorded" "$(jq -r '.["stub"].evals[0].suite' "$REG")" "core"
assert_eq "limit recorded" "$(jq -r '.["stub"].evals[0].limit' "$REG")" "2"
assert_eq "lm-eval version recorded" "$(jq -r '.["stub"].evals[0].lm_eval_version' "$REG")" "0.4.13"

# E. explicit tasks still work (positional form)
"$S/cli/ml" eval mmlu,gsm8k 8 >/dev/null 2>&1
assert_eq "second eval appended" "$(jq -r '.["stub"].evals | length' "$REG")" "2"
assert_eq "explicit tasks recorded" "$(jq -r '.["stub"].evals[1].tasks' "$REG")" "mmlu,gsm8k"
assert_eq "positional batch recorded" "$(jq -r '.["stub"].evals[1].batch_size' "$REG")" "8"

out=$("$S/cli/ml" evals stub 2>&1)
assert_contains "ml evals lists the model" "$out" "stub"
assert_contains "ml evals shows latest scores" "$out" "Latest scores"

rm -rf "$S"
finish
