#!/usr/bin/env bash
# Shared helpers for the shell test suite.
#
# Each test builds a throwaway stack (registries, models/, outputs/) and points
# ML_STACK_HOME at it, so nothing touches the real repo state. External tools
# (hf, curl, lm-eval, docker, llama-server) are stubbed on PATH.

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PASSES=0
FAILS=0

ok()   { PASSES=$((PASSES + 1)); printf '  ok   %s\n' "$1"; }
fail() {
  FAILS=$((FAILS + 1))
  printf '  FAIL %s\n' "$1"
  [ -n "${2:-}" ] && printf '       %s\n' "$2"
}

assert_eq() {
  if [ "$2" = "$3" ]; then ok "$1"; else fail "$1" "expected [$3], got [$2]"; fi
}

assert_contains() {
  case "$2" in
    *"$3"*) ok "$1" ;;
    *) fail "$1" "output does not contain [$3]" ;;
  esac
}

assert_not_contains() {
  case "$2" in
    *"$3"*) fail "$1" "output unexpectedly contains [$3]" ;;
    *) ok "$1" ;;
  esac
}

# make_stack — create an isolated ML_STACK_HOME and echo its path.
make_stack() {
  local stack
  stack=$(mktemp -d "${TMPDIR:-/tmp}/mlstack-test.XXXXXX")
  mkdir -p "$stack"/configs/{models,datasets,runs,evals,vllm,llama} \
           "$stack"/models/{base,quantized,finetuned} \
           "$stack"/outputs/{runs,evals} \
           "$stack"/runtimes/llama.cpp/build/bin \
           "$stack"/shim
  # Reuse the real code, keep the state local.
  ln -s "$REPO/cli" "$stack/cli"
  ln -s "$REPO/engine" "$stack/engine"
  ln -s "$REPO/scripts" "$stack/scripts"
  ln -s "$REPO/envs" "$stack/envs"
  echo "$stack"
}

# stub <stack> <name> <<'EOF' ... EOF
stub() {
  local stack="$1" name="$2"
  cat > "$stack/shim/$name"
  chmod +x "$stack/shim/$name"
}

# stub_llama_server <stack> — records argv to $STACK/llama-args
stub_llama_server() {
  local stack="$1"
  cat > "$stack/runtimes/llama.cpp/build/bin/llama-server" <<'EOF'
#!/usr/bin/env bash
echo "$*" > "$(dirname "$0")/../../../../llama-args"
EOF
  chmod +x "$stack/runtimes/llama.cpp/build/bin/llama-server"
}

# stub_hf <stack> — fake `hf download`, writes the requested files
stub_hf() {
  local stack="$1"
  cat > "$stack/shim/hf" <<'EOF'
#!/usr/bin/env bash
repo="$2"; file=""; dir=""
shift 2
while [ $# -gt 0 ]; do
  case "$1" in
    --local-dir) dir="$2"; shift 2 ;;
    *) file="$1"; shift ;;
  esac
done
mkdir -p "$dir/.cache/huggingface/download"
if [ -n "$file" ]; then
  touch "$dir/$file"
else
  touch "$dir/model.safetensors"
fi
EOF
  chmod +x "$stack/shim/hf"
}

# stub_hf_api <stack> <repo-pattern> <json> — fake the HF file listing
stub_hf_api() {
  local stack="$1" pattern="$2" json="$3"
  cat > "$stack/shim/curl" <<EOF
#!/usr/bin/env bash
for a in "\$@"; do case "\$a" in *api/models/*) repo="\${a##*api/models/}";; esac; done
case "\$repo" in
  $pattern) printf '%s' '$json' ;;
  *) printf '{"siblings":[]}' ;;
esac
EOF
  chmod +x "$stack/shim/curl"
}

finish() {
  echo ""
  if [ "$FAILS" -eq 0 ]; then
    echo "  $PASSES passed"
    exit 0
  fi
  echo "  $FAILS failed, $PASSES passed"
  exit 1
}
