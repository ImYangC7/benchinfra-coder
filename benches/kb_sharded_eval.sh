#!/usr/bin/env bash
# ============================================================
# kb_sharded_eval.sh — KernelBench eval split across N GPUs, one process per card
# (num_gpu_devices=1 each). Avoids the BrokenPipe crash that an 8-worker mp.Pool
# hits under shared-FS + CUDA init contention. Each process evals a problem-id
# range and writes its own eval_results_gpuN.json; merge with kb_merge_shards.py.
#
# Run this ONLY after the engine is stopped (eval needs the full VRAM).
# Usage:  bash kb_sharded_eval.sh <run_name> [backend=triton] [total=100] [timeout=300]
# ============================================================
source "$(dirname "$0")/../config.sh"
set -u
KB="${KERNELBENCH_ROOT:-$CODERBENCH_ROOT/KernelBench}"
KB_PY="${KERNELBENCH_PY:-$KB/.venv/bin/python}"
RUN=$1; BACKEND=${2:-triton}; TOTAL=${3:-100}; TIMEOUT=${4:-300}
GPU_ARCH="${GPU_ARCH:-Hopper}"
source "$KB/activate_kb.sh" >/dev/null 2>&1
cd "$KB"
CHUNK=$(( (TOTAL + N_GPU - 1) / N_GPU ))
rm -f runs/$RUN/eval_results_gpu*.json 2>/dev/null
mkdir -p /tmp/logs/kb_shard
echo "=== KB sharded eval [$RUN] backend=$BACKEND $N_GPU shards start $(date +%H:%M:%S) ==="
for g in $(seq 0 $((N_GPU-1))); do
  start=$(( g * CHUNK + 1 )); end=$(( start + CHUNK - 1 ))
  [ $start -gt $TOTAL ] && continue
  [ $end -gt $TOTAL ] && end=$TOTAL
  CUDA_VISIBLE_DEVICES=$g setsid "$KB_PY" scripts/eval_from_generations.py \
    run_name="$RUN" dataset_src=local level=1 \
    num_gpu_devices=1 timeout=$TIMEOUT gpu_arch="['$GPU_ARCH']" backend="$BACKEND" \
    subset="($start,$end)" eval_file_suffix="_gpu${g}" \
    > "/tmp/logs/kb_shard/${RUN}_gpu${g}.log" 2>&1 < /dev/null &
  disown
  echo "  GPU$g -> [$start,$end]"
done
echo "=== $N_GPU shards launched $(date +%H:%M:%S) — merge with kb_merge_shards.py when done ==="
