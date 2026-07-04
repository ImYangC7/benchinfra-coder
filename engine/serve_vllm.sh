#!/bin/bash
# ============================================================
# serve_vllm.sh — N single-GPU vLLM OpenAI servers + a round-robin proxy on
# :$PROXY_PORT. All benches hit the one proxy; the engine starts once.
#
# Why vLLM (not a hand-rolled HF serve): continuous batching does per-seq early
# stop + dynamic scheduling, so short requests don't idle-wait behind long ones.
#
# Verified config (Qwen3.x GDN/Mamba-hybrid 27B; tune per model):
#   - gdn_prefill_backend=flashinfer  (triton/fla numerics are broken for GDN)
#   - VLLM_USE_DEEP_GEMM=0            (else KV-cache init forces FP8 deep_gemm -> crash)
#   - max_num_seqs default 256        (GDN: 1 Mamba cache block per decode seq)
#   - TP=1 per card                   (27B bf16 ~54GB fits; GDN risky at TP>1)
#   - dense (e.g. IQuestCoder) models: works too; drop --additional-config if unused
#
# Usage:  bash serve_vllm.sh <model_path> [served_name] [max_num_seqs] [max_model_len]
#         bash serve_vllm.sh stop
# ============================================================
source "$(dirname "$0")/../config.sh"

log(){ echo "[serve_vllm $(date +%H:%M:%S)] $*"; }

# Robust stop: vLLM subprocs show as VLLM::EngineCore / VLLM::Worker_TPn, NOT
# lowercase 'vllm' — plain `pkill -f vllm` misses them and leaks ~90GB/GPU.
if [ "$1" = "stop" ]; then
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  pkill -f "proxy_rr.py" 2>/dev/null
  pkill -9 -f "VLLM::" 2>/dev/null
  pkill -9 -f "EngineCore" 2>/dev/null
  echo "stopped vllm servers / proxy"
  exit 0
fi

[ -f "$SETUP_ENV" ] && source "$SETUP_ENV" >/dev/null 2>&1  # CUDA compat libs on LD path

MODEL_PATH=$1
SERVED_NAME=${2:-model}
NSEQ=${3:-$MAX_NUM_SEQS}
MLEN=${4:-$MAX_MODEL_LEN}
[ -z "$MODEL_PATH" ] && { echo "usage: serve_vllm.sh <model_path> [name] [max_num_seqs] [max_model_len] | stop"; exit 1; }

export VLLM_USE_V1=1 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_WORKER_MULTIPROC_METHOD=spawn

pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; pkill -f "proxy_rr.py" 2>/dev/null; sleep 3
BACKENDS=""
for g in $(seq 0 $((N_GPU-1))); do
  port=$((BASE_PORT+g))
  CUDA_VISIBLE_DEVICES=$g nohup "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" \
    --port $port --host 0.0.0.0 \
    --tensor-parallel-size 1 --gpu-memory-utilization $GPU_MEM_UTIL \
    --max-model-len $MLEN --max-num-seqs $NSEQ \
    --trust-remote-code --dtype bfloat16 \
    --additional-config '{"gdn_prefill_backend":"flashinfer"}' \
    > /tmp/vllm_${SERVED_NAME}_g${g}.log 2>&1 &
  BACKENDS="$BACKENDS http://localhost:$port"
done
log "launched $N_GPU vLLM servers for $SERVED_NAME ($MODEL_PATH) mlen=$MLEN nseq=$NSEQ"

# startup is slow (weight load + torch.compile + cuda graph + flashinfer JIT), ~20min cap
for g in $(seq 0 $((N_GPU-1))); do
  port=$((BASE_PORT+g)); ok=0
  for _ in $(seq 1 240); do
    curl -s --max-time 2 http://localhost:$port/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ "$ok" = 1 ] && log "vLLM g$g ready" || { log "vLLM g$g FAILED"; tail -8 /tmp/vllm_${SERVED_NAME}_g${g}.log; exit 1; }
done

THINK=$THINK nohup "$SYS_PY" "$BENCHINFRA_ROOT/engine/proxy_rr.py" $PROXY_PORT $BACKENDS \
  > /tmp/proxy_${SERVED_NAME}.log 2>&1 &
sleep 3
curl -s --max-time 2 http://localhost:$PROXY_PORT/ | grep -q ok && log "proxy up on :$PROXY_PORT — ready"
