# ============================================================
# config.sh — central configuration for benchinfra-coder.
#
# Every path and tunable lives here so the scripts stay host-agnostic. Override
# any value by exporting it before sourcing (or edit this file for a fixed host).
#   export CODERBENCH_ROOT=/data/coderbench
#   source config.sh
# ============================================================

# --- Root of the benchmark datasets + venvs (NOT this infra repo) ---
# This is where the actual benchmark repos live: VerilogEval/, RTLLM/,
# ArchXBench/, RealBench/, KernelBench/. The infra scripts drive them in place.
: "${CODERBENCH_ROOT:=/apdcephfs_szcf/share_303740000/zeuscyang/coderbench}"

# --- This infra repo (auto-detected; the dir containing this config.sh) ---
BENCHINFRA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Where per-model results are written ---
: "${RESULTS_DIR:=$CODERBENCH_ROOT/results}"

# --- Engine / proxy ---
: "${PROXY_PORT:=8000}"          # OpenAI-compatible endpoint all benches hit
: "${N_GPU:=8}"                  # single-GPU vLLM backends (ports BASE_PORT+i)
: "${BASE_PORT:=8101}"
: "${MAX_MODEL_LEN:=32768}"      # engine context window
: "${MAX_NUM_SEQS:=256}"
: "${GPU_MEM_UTIL:=0.9}"
: "${THINK:=1}"                  # 1 = keep model reasoning, strip <think> from output

# --- Toolchain (Verilog benches need iverilog v12 on PATH front) ---
# oss-cad-suite's iverilog v14-devel has a $dumpvars forward-ref bug -> pass_rate=0.
: "${IVERILOG12_BIN:=/apdcephfs_szcf/share_303740000/zeuscyang/env/iverilog12/bin}"
# Sourced for verilator/yosys (RealBench) + CUDA compat libs. Optional.
: "${SETUP_ENV:=/apdcephfs_szcf/share_303740000/zeuscyang/env/setup_env.sh}"

# --- Python interpreters ---
: "${VLLM_PY:=/apdcephfs_szcf/share_303740000/zeuscyang/web_gspo/.venv/bin/python}"
: "${SYS_PY:=python3}"           # stdlib-only runners (RTLLM/ArchX/RealBench gen)

# --- OpenAI-compatible client env (local vLLM ignores the key) ---
: "${OPENAI_BASE_URL:=http://localhost:$PROXY_PORT/v1}"
: "${OPENAI_API_KEY:=dummy-local-key}"

export CODERBENCH_ROOT BENCHINFRA_ROOT RESULTS_DIR PROXY_PORT N_GPU BASE_PORT \
       MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEM_UTIL THINK IVERILOG12_BIN SETUP_ENV \
       VLLM_PY SYS_PY OPENAI_BASE_URL OPENAI_API_KEY
