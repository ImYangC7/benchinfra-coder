#!/bin/bash
# ============================================================
# run_kernelbench.sh — KernelBench (CUDA/Triton kernel generation, level 1..4).
# Two stages in the KB venv:
#   1. generate: query the model → runs/<key>_level<N>/...kernel.py
#   2. eval:     compile each kernel + run on GPU vs the PyTorch ref (correctness
#                + speedup). Eval uses the GPUs, so if the engine is still resident
#                it CONTENDS for VRAM — stop the engine first, or pass eval_gpus<=4.
#
# Metric: compiled count + fast_0 = correctness (functional pass@1).
# Backend: triton (default) / cuda / cute / tilelang.
#
# For a clean full-GPU eval, prefer the sharded flow (see kb_sharded_eval.sh):
# 8 processes each pinned to one card avoids the mp.Pool crash on shared FS+CUDA.
#
# Usage:  bash run_kernelbench.sh <served_name> [levels] [eval_gpus] [backend]
# ============================================================
set -e
source "$(dirname "$0")/../config.sh"
source "$BENCHINFRA_ROOT/lib/common.sh"

KB="${KERNELBENCH_ROOT:-$CODERBENCH_ROOT/KernelBench}"
KB_PY="${KERNELBENCH_PY:-$KB/.venv/bin/python}"
KEY=$1
LEVELS=${2:-1}; EVAL_GPUS=${3:-8}; BACKEND=${4:-triton}
LEVELS=${LEVELS//,/ }
MAX_TOKENS="${MAXTOK:-32768}"; WORKERS="${WORKERS:-128}"; TEMPERATURE="${TEMP:-0}"
GPU_ARCH="${GPU_ARCH:-Hopper}"; GPU_LABEL="${GPU_LABEL:-H100}"
[ -z "$KEY" ] && { echo "usage: run_kernelbench.sh <served_name> [levels] [eval_gpus] [backend]"; exit 1; }

require_engine
source "$KB/activate_kb.sh" >/dev/null 2>&1   # nvcc on PATH + reused-torch LD paths
export OPENAI_API_KEY SGLANG_API_KEY="$OPENAI_API_KEY" OPENAI_BASE_URL
cd "$KB"
for LV in $LEVELS; do
  RUN="${KEY}_level${LV}"
  log "level $LV generate (backend=$BACKEND)"
  "$KB_PY" scripts/generate_samples.py dataset_src=local level=$LV run_name="$RUN" \
    server_type=local model_name="$KEY" server_address=localhost server_port=$PROXY_PORT \
    temperature=$TEMPERATURE max_tokens=$MAX_TOKENS num_workers=$WORKERS backend=$BACKEND check_kernel=False \
    > /tmp/kb_gen_${RUN}.log 2>&1 || log "generate nonzero"
  log "level $LV eval ($EVAL_GPUS GPU, backend=$BACKEND)"
  "$KB_PY" scripts/eval_from_generations.py dataset_src=local level=$LV run_name="$RUN" \
    gpu="$GPU_LABEL" gpu_arch="['$GPU_ARCH']" num_gpu_devices=$EVAL_GPUS backend=$BACKEND \
    > /tmp/kb_eval_${RUN}.log 2>&1 || log "eval nonzero"
  DEST="$RESULTS_DIR/$KEY/kernelbench/level${LV}"; mkdir -p "$DEST"
  cp -f "$KB/runs/$RUN/eval_results.json" "$DEST/" 2>/dev/null && log "saved $DEST" || log "no eval_results.json"
done
log "DONE $KEY (levels: $LEVELS)"
