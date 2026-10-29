#!/usr/bin/env bash
# sample-dataset: argument validation.
set -u
source "$(dirname "$0")/lib.sh"

S=$(make_stack)
export ML_STACK_HOME="$S"

out=$("$S/cli/ml" sample-dataset 2>&1); rc=$?
assert_eq "no args exits non-zero" "$rc" "1"
assert_contains "no args prints usage" "$out" "Usage"

out=$("$S/cli/ml" sample-dataset qwen ds 2>&1); rc=$?
assert_eq "missing count exits non-zero" "$rc" "1"

out=$("$S/cli/ml" sample-dataset qwen ds abc 2>&1); rc=$?
assert_eq "non-numeric count exits non-zero" "$rc" "1"
assert_contains "non-numeric count explains why" "$out" "positive integer"

out=$("$S/cli/ml" sample-dataset qwen ds 0 2>&1); rc=$?
assert_eq "zero count exits non-zero" "$rc" "1"

# a valid count with no processed dataset fails on the missing dataset, not on parsing
out=$("$S/cli/ml" sample-dataset qwen ds 100 2>&1); rc=$?
assert_eq "valid count reaches the dataset check" "$rc" "1"
assert_contains "missing dataset is reported" "$out" "Processed dataset not found"

rm -rf "$S"
finish
