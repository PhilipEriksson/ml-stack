#!/usr/bin/env bash
# Run the shell test suite.
#
#   bash tests/run.sh              # everything
#   bash tests/run.sh add-model    # one file (substring match)
#
# No network, no GPU, no Docker: external tools are stubbed per test.

set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export REPO

FILTER="${1:-}"
FAILED=0
RAN=0

for t in "$REPO"/tests/test-*.sh; do
  [ -f "$t" ] || continue
  name=$(basename "$t" .sh)
  if [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]]; then
    continue
  fi
  RAN=$((RAN + 1))
  echo "▶ $name"
  if ! bash "$t"; then
    FAILED=$((FAILED + 1))
  fi
  echo ""
done

if [ "$RAN" -eq 0 ]; then
  echo "No tests matched '$FILTER'"
  exit 1
fi

if [ "$FAILED" -eq 0 ]; then
  echo "✅ $RAN test file(s) passed"
  exit 0
fi

echo "❌ $FAILED of $RAN test file(s) failed"
exit 1
