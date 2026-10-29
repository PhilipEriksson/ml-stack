#!/usr/bin/env bash

# ============================================================
# ML Stack — One-Time Setup Script
# ============================================================
# Usage:
#   bash scripts/init.sh          # check dependencies only
#   bash scripts/init.sh --install # check + auto-install what it can
# ============================================================

set -e

INSTALL_MODE=false
if [ "$1" == "--install" ]; then
  INSTALL_MODE=true
fi

ROOT="${ML_STACK_HOME:-$HOME/ml-stack}"
BASHRC="$HOME/.bashrc"

echo "🧠 Setting up ML Stack..."
echo "📍 Project: $ROOT"
if [ "$INSTALL_MODE" = true ]; then
  echo "⚙️  Install mode — will auto-install where possible"
else
  echo "🔍 Check mode — add --install to auto-fix"
fi
echo ""

# =========================
# 1. Add 'ml' to PATH
# =========================
if grep -qF "ml-stack/cli" "$BASHRC" 2>/dev/null; then
  echo "✅ 'ml' already in PATH ($BASHRC)"
elif [ "$INSTALL_MODE" = true ]; then
  echo "🔧 Adding 'ml' command to PATH in $BASHRC..."
  echo "" >> "$BASHRC"
  echo "# ML Stack CLI" >> "$BASHRC"
  echo 'export PATH="${ML_STACK_HOME:-$HOME/ml-stack}/cli:$PATH"' >> "$BASHRC"
  echo "" >> "$BASHRC"

  # Add claude-local function if not already present
  if ! grep -qF "claude-local" "$BASHRC" 2>/dev/null; then
    echo "# Claude Code with local model" >> "$BASHRC"
    echo 'claude-local() {' >> "$BASHRC"
    echo '  set -a' >> "$BASHRC"
    echo '  source "${ML_STACK_HOME:-$HOME/ml-stack}/.claude_env"' >> "$BASHRC"
    echo '  set +a' >> "$BASHRC"
    echo '  claude "$@"' >> "$BASHRC"
    echo '}' >> "$BASHRC"
    echo "✅ Added claude-local function to $BASHRC"
  fi

  # Create .claude_env if missing
  if [ ! -f "$ROOT/.claude_env" ]; then
    cat > "$ROOT/.claude_env" <<'ENVEOF'
#!/usr/bin/env bash
# Dynamic Claude Code environment — detects active vLLM or llama.cpp backend
ML_STACK="${ML_STACK_HOME:-$HOME/ml-stack}"
source "$ML_STACK/scripts/utils/get-claude-env"
ENVEOF
    echo "✅ Created .claude_env (dynamic model detection)"
  fi

  echo "✅ Added. Run: source $BASHRC"
else
  echo "⚠️  'ml' not in PATH ($BASHRC)"
  echo "     → Add with: bash scripts/init.sh --install"
  echo "     → Or for this shell: export PATH=\"$ROOT/cli:\$PATH\""
fi

# =========================
# 2. System dependencies
# =========================
echo ""
echo "🔍 Checking system dependencies..."

MISSING=0

# Python
if command -v python3 &>/dev/null; then
  echo "  ✅ Python $(python3 --version 2>&1 | awk '{print $2}')"
else
  echo "  ❌ Python 3 not found"
  if [ "$INSTALL_MODE" = true ]; then
    echo "     → Install with: sudo apt install python3 python3-pip"
  fi
  MISSING=1
fi

# jq
if command -v jq &>/dev/null; then
  echo "  ✅ jq"
else
  echo "  ❌ jq"
  if [ "$INSTALL_MODE" = true ]; then
    echo "     → Installing..."
    if command -v apt-get &>/dev/null; then
      sudo apt-get update && sudo apt-get install -y jq
    elif command -v brew &>/dev/null; then
      brew install jq
    else
      echo "     → Cannot auto-install. Install manually: apt install jq / brew install jq"
      MISSING=1
    fi
  else
    echo "     → Install with: apt install jq / brew install jq"
    MISSING=1
  fi
fi

# curl
if command -v curl &>/dev/null; then
  echo "  ✅ curl"
else
  echo "  ❌ curl"
  MISSING=1
fi

# pip
if command -v pip3 &>/dev/null; then
  echo "  ✅ pip3"
elif command -v pip &>/dev/null; then
  echo "  ✅ pip"
else
  echo "  ⚠️  pip not found (needed for Python packages)"
  if [ "$INSTALL_MODE" = true ]; then
    echo "     → Install with: sudo apt install python3-pip"
  fi
fi

# =========================
# 3. Python packages
# =========================
echo ""
echo "📦 Checking Python packages..."

PIP_CMD=""
if command -v pip3 &>/dev/null; then
  PIP_CMD="pip3"
elif command -v pip &>/dev/null; then
  PIP_CMD="pip"
fi

# huggingface_hub (hf CLI)
if command -v hf &>/dev/null; then
  echo "  ✅ huggingface_hub (hf CLI)"
else
  echo "  ⚠️  huggingface_hub (hf CLI) not found"
  if [ "$INSTALL_MODE" = true ] && [ -n "$PIP_CMD" ]; then
    echo "     → Installing..."
    $PIP_CMD install --upgrade huggingface_hub
    echo "     ✅ Installed. Run: hash -r"
  else
    echo "     → Install with: pip install huggingface_hub"
  fi
fi

# =========================
# 4. GPU
# =========================
echo ""
echo "🖥️  GPU:"

if command -v nvidia-smi &>/dev/null; then
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
  echo "  ✅ NVIDIA GPU ($VRAM GB VRAM)"
else
  echo "  ⚠️  NVIDIA GPU not detected"
  echo "     → Install NVIDIA drivers + CUDA toolkit"
  echo "     → On WSL2: use Windows NVIDIA drivers (no separate CUDA install needed)"
fi

# =========================
# 5. Docker
# =========================
echo ""
echo "🐳 Docker:"

DOCKER_CMD=""

if command -v docker &>/dev/null && docker version &>/dev/null 2>&1; then
  DOCKER_CMD="docker"
  echo "  ✅ Docker $(docker --version | awk '{print $3}')"
elif [ -f "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" ] && uname -r 2>/dev/null | grep -q "Microsoft" && "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" version &>/dev/null 2>&1; then
  DOCKER_CMD="/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
  echo "  ✅ Docker Desktop (via Windows)"
  if [ "$INSTALL_MODE" = true ]; then
    echo "     → Creating ~/bin/docker wrapper..."
    mkdir -p ~/bin
    cat > ~/bin/docker <<'WRAPPER'
#!/bin/bash
exec "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" "$@"
WRAPPER
    chmod +x ~/bin/docker
    if ! grep -qF '$HOME/bin' "$BASHRC" 2>/dev/null; then
      echo 'export PATH="$HOME/bin:$HOME/.local/bin:$PATH"' >> "$BASHRC"
      echo "     → Added ~/bin to PATH in .bashrc"
    fi
    DOCKER_CMD="$HOME/bin/docker"
    echo "     ✅ Created. Run: source $BASHRC"
  fi
else
  echo "  ❌ Docker not found"
  if [ "$INSTALL_MODE" = true ]; then
    echo "     → Download Docker Desktop: https://docs.docker.com/desktop/"
  else
    echo "     → Download from: https://docs.docker.com/desktop/"
  fi
  MISSING=1
fi

# Check Docker Compose
if [ -n "$DOCKER_CMD" ]; then
  if "$DOCKER_CMD" compose version &>/dev/null 2>&1; then
    echo "  ✅ Docker Compose (v2 plugin)"
  else
    echo "  ⚠️  Docker Compose v2 not found"
  fi
fi

# =========================
# 6. Conda environments
# =========================
echo ""
echo "🐍 Conda environments:"

if command -v conda &>/dev/null; then
  for env_file in "$ROOT"/envs/*.yml; do
    [ -f "$env_file" ] || continue
    # Read the actual name from the yml file (e.g. "vllm-env" not "vllm")
    ENV_NAME=$(grep '^name:' "$env_file" | awk '{print $2}')
    if [ -z "$ENV_NAME" ]; then
      ENV_NAME=$(basename "$env_file" .yml)
    fi

    # Some envs keep their exact pins in a companion .txt (pip freeze), because
    # the yml only declares python + pip. Creating the env is not enough.
    PINS="$ROOT/envs/${ENV_NAME}.txt"

    if ! conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
      echo "  ⚠️  $ENV_NAME (missing)"
      if [ "$INSTALL_MODE" = true ]; then
        echo "     → Creating..."
        conda env create -f "$env_file" -q
        echo "     ✅ Created"
      else
        echo "     → Create with: conda env create -f $env_file"
        MISSING=1
        continue
      fi
    fi

    if [ -f "$PINS" ]; then
      ENV_PIP=""
      for CONDA_ROOT in "$HOME/miniconda" "$HOME/anaconda3" "$HOME/conda"; do
        if [ -x "$CONDA_ROOT/envs/$ENV_NAME/bin/pip" ]; then
          ENV_PIP="$CONDA_ROOT/envs/$ENV_NAME/bin/pip"
          break
        fi
      done
      FIRST_PKG=$(grep -vE '^[[:space:]]*(#|$)' "$PINS" | head -1 | cut -d= -f1 | tr '[:upper:]' '[:lower:]')
      if [ -n "$ENV_PIP" ] && "$ENV_PIP" list 2>/dev/null | awk '{print tolower($1)}' | grep -qx "$FIRST_PKG"; then
        echo "  ✅ $ENV_NAME (pins applied)"
      else
        echo "  ⚠️  $ENV_NAME (created, pins not applied)"
        if [ "$INSTALL_MODE" = true ] && [ -n "$ENV_PIP" ]; then
          echo "     → Installing pins from $(basename "$PINS")..."
          "$ENV_PIP" install -q -r "$PINS"
          echo "     ✅ Pins installed"
        else
          echo "     → Install with: ~/miniconda/envs/$ENV_NAME/bin/pip install -r $PINS"
          MISSING=1
        fi
      fi
    else
      echo "  ✅ $ENV_NAME"
    fi
  done
else
  echo "  ❌ conda not found"
  echo "     → Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
  echo "     → Or on WSL2: sudo apt install wget && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh"
fi

echo ""
echo "💡 Training environments are defined in envs/. Create one with:"
echo "   conda env create -f envs/training.yml"
echo "   ml use-training-env training"

# =========================
# 7. vLLM env selection
# =========================
echo ""
echo "⚙️  vLLM config:"

if [ -f "$ROOT/configs/vllm/active.env" ]; then
  echo "  ✅ configs/vllm/active.env"
else
  echo "  ⚠️  configs/vllm/active.env (missing)"
  if [ "$INSTALL_MODE" = true ]; then
    cp "$ROOT/configs/vllm/example.env" "$ROOT/configs/vllm/active.env"
    echo "     → Created from example.env — set MODEL_NAME before starting vLLM"
  else
    echo "     → Create with: cp configs/vllm/example.env configs/vllm/active.env"
    echo "       then edit MODEL_NAME, or run: ml set-vllm-env <file>"
    MISSING=1
  fi
fi

# =========================
# 8. llama.cpp
# =========================
echo ""
echo "📦 llama.cpp:"

LLAMA_BUILD="$ROOT/runtimes/llama.cpp"
if [ -d "$LLAMA_BUILD/.git" ] || [ -f "$LLAMA_BUILD/CMakeLists.txt" ]; then
  if [ -f "$LLAMA_BUILD/build/bin/llama-server" ]; then
    echo "  ✅ llama-server built"
  else
    echo "  ⚠️  llama.cpp source present but not built"
    if [ "$INSTALL_MODE" = true ]; then
      echo "     → Building (this may take 5-10 minutes)..."
      (cd "$LLAMA_BUILD" && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target llama-server -j$(nproc))
      echo "     ✅ Built"
    else
      echo "     → Build with: cd $LLAMA_BUILD && cmake -B build && cmake --build build"
    fi
  fi
else
  echo "  ⚠️  llama.cpp source not found"
  echo "     → Clone with: git clone --recursive https://github.com/ggml-org/llama.cpp.git $LLAMA_BUILD"
fi

# =========================
# 9. Docker images
# =========================
echo ""
echo "🐳 Docker images:"

if [ -n "$DOCKER_CMD" ]; then
  # Ask compose for the image names it will actually use instead of guessing
  # the project prefix from the directory name.
  COMPOSE_FILE="$ROOT/services/docker/docker-compose.yml"
  COMPOSE_IMAGES=$("$DOCKER_CMD" compose -f "$COMPOSE_FILE" config --images 2>/dev/null || true)

  if [ -z "$COMPOSE_IMAGES" ]; then
    echo "  ⚠️  Could not resolve compose images (is configs/vllm/active.env present?)"
    MISSING=1
  else
    for IMAGE in $COMPOSE_IMAGES; do
      # compose reports untagged names; docker images lists repo:tag
      case "$IMAGE" in
        *:*) IMAGE_REF="$IMAGE" ;;
        *)   IMAGE_REF="$IMAGE:latest" ;;
      esac
      if "$DOCKER_CMD" images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -qx "$IMAGE_REF"; then
        echo "  ✅ $IMAGE_REF"
      else
        echo "  ⚠️  $IMAGE_REF (not built)"
        if [ "$INSTALL_MODE" = true ]; then
          echo "     → Building..."
          (cd "$ROOT" && "$DOCKER_CMD" compose -f "$COMPOSE_FILE" build)
          echo "     ✅ Built"
        else
          echo "     → Build with: docker compose -f services/docker/docker-compose.yml build"
          MISSING=1
        fi
      fi
    done
  fi
fi

# =========================
# 10. Project structure
# =========================
echo ""
echo "📂 Project structure:"

for dir in models/base models/quantized models/finetuned datasets/raw datasets/processed outputs/evals outputs/runs scripts/utils scripts/eval scripts/train scripts/quantize services/vllm services/api-webui services/searxng services/docker configs/vllm configs/searxng configs/llama configs/evals configs/runs configs/training; do
  if [ -d "$ROOT/$dir" ]; then
    echo "  ✅ $dir/"
  else
    echo "  ❌ $dir/ (missing)"
    if [ "$INSTALL_MODE" = true ]; then
      mkdir -p "$ROOT/$dir"
      echo "     → Created"
    fi
  fi
  done

# =========================
# 11. Summary
# =========================
echo ""
if [ "$MISSING" -eq 0 ] && [ "$INSTALL_MODE" = false ]; then
  echo "✅ All checks passed! Run: source $BASHRC"
  echo ""
  echo "Then try: ml"
elif [ "$INSTALL_MODE" = true ]; then
  echo "✅ Setup complete! Run: source $BASHRC"
  echo ""
  echo "Then try: ml"
else
  echo "⚠️  Some dependencies are missing."
  echo "   Re-run with --install to auto-fix what we can:"
  echo "   bash scripts/init.sh --install"
fi
