#!/bin/bash
# ============================================================
# run_all.sh — one-click: start the engine ONCE, run every registered bench
# against the shared :$PROXY_PORT proxy, then stop the engine.
#
# The bench list lives in benches/registry.sh — add a line there and it runs
# here automatically. KernelBench is excluded by default (it needs the engine
# stopped for VRAM); run it separately after the sweep.
#
# Usage:
#   bash run_all.sh <model_path> <served_name> [max_model_len]
#   bash run_all.sh --no-engine <served_name>      # engine already up; just sweep
#
# NOTE: <served_name> must be registered in verilog-eval/scripts/sv-generate.
# ============================================================
set -u
source "$(dirname "$0")/config.sh"
source "$BENCHINFRA_ROOT/lib/common.sh"
source "$BENCHINFRA_ROOT/benches/registry.sh"

if [ "${1:-}" = "--no-engine" ]; then
  KEY=$2; START_ENGINE=0
else
  MODEL_PATH=$1; KEY=$2; MLEN=${3:-$MAX_MODEL_LEN}; START_ENGINE=1
fi
[ -z "${KEY:-}" ] && { echo "usage: run_all.sh <model_path> <served_name> [max_model_len] | --no-engine <served_name>"; exit 1; }

if [ "$START_ENGINE" = 1 ]; then
  log "starting engine for $KEY ($MODEL_PATH) mlen=$MLEN"
  bash "$BENCHINFRA_ROOT/engine/serve_vllm.sh" "$MODEL_PATH" "$KEY" "$MAX_NUM_SEQS" "$MLEN"
else
  require_engine
fi

for entry in "${BENCHES[@]}"; do
  name=$(echo "$entry"   | cut -d'|' -f1 | xargs)
  runner=$(echo "$entry" | cut -d'|' -f2 | xargs)
  denv=$(echo "$entry"   | cut -d'|' -f3 | xargs)
  log "############ $name ############"
  # Run benches SERIALLY (no `&`). Each bench already saturates the engine with
  # its own worker pool; launching several in parallel oversubscribes vLLM's
  # request queue and triggers transient HTTP 500s. The runners now retry 5xx,
  # but keeping benches serial is the primary safeguard against overload.
  env $denv bash "$BENCHINFRA_ROOT/benches/$runner" "$KEY" || log "$name nonzero (continuing)"
done

if [ "$START_ENGINE" = 1 ]; then
  log "stopping engine"
  bash "$BENCHINFRA_ROOT/engine/serve_vllm.sh" stop
fi
log "ALL DONE $KEY — results in $RESULTS_DIR/$KEY/"
echo
echo "Summarize with:  bash $BENCHINFRA_ROOT/summarize.sh $KEY"
