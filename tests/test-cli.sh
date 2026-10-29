#!/usr/bin/env bash
# CLI surface: help lists every command, per-command help exists, unknown input is handled.
set -u
source "$(dirname "$0")/lib.sh"

S=$(make_stack)
export ML_STACK_HOME="$S"

out=$("$S/cli/ml" 2>&1)
for cmd in add-model add-dataset process-dataset sample-dataset create-training-run \
           execute-training-run tensorboard aim-ui use-training-env kill-training \
           serve-model set-vllm-env eval evals compare-runs runs status quantize; do
  assert_contains "global help lists $cmd" "$out" "ml $cmd"
done

# every command in the global help must have a per-command help entry
for cmd in add-model add-dataset process-dataset sample-dataset create-training-run \
           execute-training-run tensorboard aim-ui use-training-env kill-training \
           serve-model set-vllm-env eval evals compare-runs runs status quantize; do
  h=$("$S/cli/ml" help "$cmd" 2>&1)
  assert_not_contains "help $cmd is not 'Unknown command'" "$h" "Unknown command"
done

out=$("$S/cli/ml" help status 2>&1)
assert_contains "status help describes the command" "$out" "One-screen"

out=$("$S/cli/ml" help nope 2>&1)
assert_contains "unknown help topic falls back to global help" "$out" "Unknown command"

out=$("$S/cli/ml" bogus 2>&1); rc=$?
assert_eq "unknown command exits non-zero" "$rc" "1"
assert_contains "unknown command names the action" "$out" "Unknown action: bogus"

# the dispatcher must not reference scripts that no longer exist
for script in $(grep -oE '"\$ROOT/[a-z-]+"' "$REPO/engine/runner.sh" | tr -d '"' | sed 's|\$ROOT/||' | sort -u); do
  if [ -f "$REPO/scripts/utils/$script" ]; then
    ok "runner target exists: $script"
  else
    fail "runner target exists: $script" "scripts/utils/$script is missing"
  fi
done

rm -rf "$S"
finish
