# ============================================================
# lib/common.sh — shared helpers sourced by every bench runner.
# Assumes config.sh has already been sourced.
# ============================================================

log() { echo "[$(basename "${0%.sh}") $(date +%H:%M:%S)] $*"; }

# Fail early if the engine isn't serving on :$PROXY_PORT.
require_engine() {
  curl -s --max-time 3 "http://localhost:$PROXY_PORT/" | grep -q ok || {
    echo "ERROR: no engine on :$PROXY_PORT. Start it first:"
    echo "  bash $BENCHINFRA_ROOT/engine/serve_vllm.sh <model_path> <served_name>"
    exit 1
  }
}

# Block until all N_GPU backends report healthy (or time out).
wait_for_engine() {
  local rounds="${1:-240}" i ok p
  for ((i=0; i<rounds; i++)); do
    ok=0
    for ((p=0; p<N_GPU; p++)); do
      curl -s --max-time 2 "http://localhost:$((BASE_PORT+p))/health" >/dev/null 2>&1 && ok=$((ok+1))
    done
    [ "$ok" -eq "$N_GPU" ] && { log "engine ready ($ok/$N_GPU backends)"; return 0; }
    sleep 15
  done
  log "engine NOT ready after $rounds rounds"; return 1
}

# Put iverilog v12 on PATH front + source setup_env for verilator/yosys/CUDA compat.
load_verilog_toolchain() {
  [ -f "$SETUP_ENV" ] && source "$SETUP_ENV" >/dev/null 2>&1
  [ -d "$IVERILOG12_BIN" ] && export PATH="$IVERILOG12_BIN:$PATH"
  # verilog-eval's configure pipes through `column -t` to build samples.mk. If
  # util-linux's column is missing the pipe silently yields an EMPTY samples.mk,
  # so --with-samples=K is dropped and average@K degrades to single-sample.
  command -v column >/dev/null 2>&1 || \
    echo "[warn] 'column' not found (util-linux); VerilogEval average@K will silently fall back to 1 sample. Install column or add a shim to PATH." >&2
}
