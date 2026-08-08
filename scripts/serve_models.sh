#!/usr/bin/env bash
# Serve the model pool with vLLM (one OpenAI-compatible server per model).
# Run on the GPU VM inside tmux:   bash scripts/serve_models.sh
#
# Profile: single 24GB GPU (RTX 4090), both models co-resident.
#   Qwen3-4B FP8 (~4.5GB) at frac 0.32  ·  Qwen3-14B-AWQ 4-bit (~10GB) at frac 0.62
# FP8 KV cache on both — bf16 KV doesn't leave room for 28k context on 24GB.
# Servers start SEQUENTIALLY (they share the GPU; concurrent startup can OOM
# during profiling). On a bigger GPU, raise fractions / swap in larger models.
set -euo pipefail

mkdir -p logs

start_and_wait() { # name, port, gpu_frac, extra args...
  local name=$1 port=$2 frac=$3; shift 3
  echo ">> serving $name on :$port (gpu frac $frac)"
  nohup vllm serve "$name" \
    --port "$port" \
    --gpu-memory-utilization "$frac" \
    --max-model-len 28672 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 8 \
    --enable-auto-tool-choice \
    "$@" \
    > "logs/vllm_${port}.log" 2>&1 &
  echo "$!" > "logs/vllm_${port}.pid"
  echo "   waiting for :$port (first run downloads weights; watch logs/vllm_${port}.log)"
  until curl -sf "http://localhost:${port}/v1/models" > /dev/null; do
    if ! kill -0 "$(cat "logs/vllm_${port}.pid")" 2>/dev/null; then
      echo "SERVER ON :$port DIED — check logs/vllm_${port}.log"; exit 1
    fi
    sleep 10; printf '.'
  done
  echo " :$port ready"
}

# Ports must match configs/pilot.yaml api_base entries.
start_and_wait Qwen/Qwen3-4B-Instruct-2507-FP8 8001 0.36 --tool-call-parser hermes --enforce-eager

# Qwen3-14B thinking mode is disabled per-request via chat_template_kwargs in
# configs/pilot.yaml (this vLLM version has no server-side flag for it).
# --enforce-eager frees the ~2GB cudagraph pool; without it the KV cache
# doesn't fit 28k context on a 24GB card.
# 0.54, not higher: each vLLM process carries ~0.5GB of CUDA-context overhead
# that its --gpu-memory-utilization fraction doesn't account for.
start_and_wait Qwen/Qwen3-14B-AWQ 8002 0.54 --tool-call-parser hermes --enforce-eager

echo "All models up. Logs in ./logs/, PIDs in logs/vllm_*.pid (kill with: kill \$(cat logs/vllm_*.pid))"
