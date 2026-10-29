#!/usr/bin/env bash

set -e

ACTION=$1
shift

ROOT="${ML_STACK_HOME:-$HOME/ml-stack}/scripts/utils"

case "$ACTION" in

  create-training-run)
    RUN=$1
    MODEL=$2
    DATASET=$3
    shift 3
    bash "$ROOT/create-training-run" "$RUN" "$MODEL" "$DATASET" "$@"
    ;;

  execute-training-run)
    RUN=$1
    bash "$ROOT/execute-training-run" "$RUN" "${@:2}"
    ;;

  tensorboard)
    CONDA_PYTHON=$("$ROOT/get-training-env" --python 2>/dev/null || echo "$HOME/miniconda/envs/training/bin/python")
    exec "$(dirname "$CONDA_PYTHON")/tensorboard" --logdir "${ML_STACK_HOME:-$HOME/ml-stack}/outputs/runs" --host 0.0.0.0 --port 6006
    ;;

  aim-ui)
    CONDA_PYTHON=$("$ROOT/get-training-env" --python 2>/dev/null || echo "$HOME/miniconda/envs/training/bin/python")
    exec "$(dirname "$CONDA_PYTHON")/aim" up --repo "${ML_STACK_HOME:-$HOME/ml-stack}/outputs/aim" --host 0.0.0.0 --port 43801
    ;;

  quantize)
    # Dispatcher is stdlib + huggingface_hub only; it shells out to the
    # per-format interpreter (training / quant-exl3 / quant-nvfp4).
    CONDA_PYTHON=$("$ROOT/get-training-env" --python 2>/dev/null || echo "$HOME/miniconda/envs/training/bin/python")
    exec "$CONDA_PYTHON" "${ML_STACK_HOME:-$HOME/ml-stack}/scripts/quantize/quantize.py" "$@"
    ;;

  kill-training)
    bash "$ROOT/kill-training"
    ;;

  sample-dataset)
    bash "$ROOT/sample-dataset" "$@"
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

  use-training-env)
    bash "$ROOT/use-training-env" "$@"
    ;;

  *)
    echo "Unknown action: $ACTION"
    exit 1
    ;;
esac