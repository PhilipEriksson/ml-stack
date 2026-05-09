#!/usr/bin/env bash
set -e

# ML Stack — Build
# Usage: bash scripts/build.sh [service...]

ROOT="${ML_STACK_HOME:-$HOME/ml-stack}"
COMPOSE_FILE="$ROOT/services/docker/docker-compose.yml"

docker compose -f "$COMPOSE_FILE" build "$@"
