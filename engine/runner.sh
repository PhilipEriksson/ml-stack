#!/usr/bin/env bash

set -e

ACTION=$1
shift

ROOT="${ML_STACK_HOME:-$HOME/ml-stack}/scripts/utils"

case "$ACTION" in

  train)
    RUN=$1
    MODEL=$2
    DATASET=$3
    bash "$ROOT/train-run" "$RUN" "$MODEL" "$DATASET"
    ;;

  add-model)
    bash "$ROOT/add-model" "$@"
    ;;

  add-dataset)
    bash "$ROOT/add-dataset" "$@"
    ;;

  process-dataset)
    bash "$ROOT/process-dataset" "$@"
    ;;

  serve-model)
    bash "$ROOT/serve-model" "$@"
    ;;

  compare-runs)
    bash "$ROOT/compare-runs" "$@"
    ;;

  runs)
    bash "$ROOT/ml-runs" "$@"
    ;;

  eval)
    bash "$ROOT/../eval/run-benchmark" "$@"
    ;;

  evals)
    bash "$ROOT/../eval/list-evals" "$@"
    ;;

  set-vllm-env)
    bash "$ROOT/set-vllm-env" "$@"
    ;;

  *)
    echo "Unknown action: $ACTION"
    exit 1
    ;;
esac