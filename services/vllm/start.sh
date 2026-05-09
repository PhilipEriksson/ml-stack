#!/bin/bash
set -e

# Build vLLM serve command from environment variables
CMD=(vllm serve "$MODEL_NAME" --trust-remote-code)

[ -n "$MAX_MODEL_LEN" ] && CMD+=(--max-model-len "$MAX_MODEL_LEN")
[ -n "$GPU_MEMORY_UTILIZATION" ] && CMD+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
[ -n "$ATTENTION_BACKEND" ] && CMD+=(--attention-backend "$ATTENTION_BACKEND")
[ -n "$PERFORMANCE_MODE" ] && CMD+=(--performance-mode "$PERFORMANCE_MODE")
[ "$LANGUAGE_MODEL_ONLY" = "true" ] && CMD+=(--language-model-only)
[ -n "$KV_CACHE_DTYPE" ] && CMD+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[ -n "$MAX_NUM_SEQS" ] && CMD+=(--max-num-seqs "$MAX_NUM_SEQS")
[ "$SKIP_MM_PROFILING" = "true" ] && CMD+=(--skip-mm-profiling)
[ -n "$QUANTIZATION" ] && CMD+=(--quantization "$QUANTIZATION")
[ -n "$REASONING_PARSER" ] && CMD+=(--reasoning-parser "$REASONING_PARSER")
[ "$ENABLE_AUTO_TOOL_CHOICE" = "true" ] && CMD+=(--enable-auto-tool-choice)
[ -n "$TOOL_CALL_PARSER" ] && CMD+=(--tool-call-parser "$TOOL_CALL_PARSER")
[ "$ENABLE_PREFIX_CACHING" = "true" ] && CMD+=(--enable-prefix-caching)
[ "$ENABLE_CHUNKED_PREFILL" = "true" ] && CMD+=(--enable-chunked-prefill)
[ -n "$SPECULATIVE_CONFIG" ] && CMD+=(--speculative-config "$SPECULATIVE_CONFIG")
[ -n "$HOST" ] && CMD+=(--host "$HOST")
[ -n "$PORT" ] && CMD+=(--port "$PORT")
[ -n "$DOWNLOAD_DIR" ] && CMD+=(--download-dir "$DOWNLOAD_DIR")
[ -n "$DTYPE" ] && CMD+=(--dtype "$DTYPE")
[ -n "$BLOCK_SIZE" ] && CMD+=(--block-size "$BLOCK_SIZE")

echo "Starting vLLM: ${CMD[*]}"
exec "${CMD[@]}"
