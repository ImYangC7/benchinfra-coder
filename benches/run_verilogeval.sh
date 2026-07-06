#!/bin/bash
# ============================================================
# run_verilogeval.sh — VerilogEval v2 (spec-to-rtl + code-complete-iccad2023,
# 156 problems each). Drives the official verilog-eval Makefile → sv-generate →
# proxy, then compiles+runs each testbench with iverilog v12.
#
# Metric: sv-iv-analyze reports pass_rate = mean over problems of npass/nsamples,
#   i.e. average@SAMPLES. SAMPLES=1 TEMP=0 => pass@1; SAMPLES=4 TEMP=0.8 => average@4.
#
# Assumes the engine is already up on :$PROXY_PORT.
# NOTE: <served_name> must be registered in verilog-eval/scripts/sv-generate.
# NOTE: the official samples.mk fans out multi-sample generation via `column`.
#   If `column` is not on PATH (util-linux), samples.mk silently degrades to a
#   SINGLE sample and SAMPLES=4 quietly becomes average@1. load_verilog_toolchain
#   warns on this; ensure util-linux's `column` is installed before average@4.
#
# Usage:  SAMPLES=4 TEMP=0.8 bash run_verilogeval.sh <served_name> [max_tokens]
# ============================================================
set -e
source "$(dirname "$0")/../config.sh"
source "$BENCHINFRA_ROOT/lib/common.sh"
load_verilog_toolchain
export OPENAI_BASE_URL OPENAI_API_KEY

VE_ROOT="${VERILOGEVAL_ROOT:-$CODERBENCH_ROOT/verilog-eval}"
KEY=$1
MAX_TOKENS=${2:-32768}
JOBS="${JOBS:-128}"
SAMPLES="${SAMPLES:-1}"
GEN_TEMP="${TEMP:-0}"
TOPP="${TOPP:-0.01}"
[ -z "$KEY" ] && { echo "usage: [SAMPLES=n TEMP=t] run_verilogeval.sh <served_name> [max_tokens]"; exit 1; }

# iverilog/vvp read the TEMP/TMP env vars for their scratch dir. If the sampling
# temperature is left in $TEMP, iverilog tries to write to a dir literally named
# "0.8" and EVERY compile fails -> pass_rate=0. Scrub them; pin TMPDIR.
unset TEMP TMP
export TMPDIR="${TMPDIR:-/tmp}"

require_engine
for TASK in spec-to-rtl code-complete-iccad2023; do
  BUILD="$RESULTS_DIR/$KEY/$TASK"
  mkdir -p "$BUILD"; cd "$BUILD"
  "$VE_ROOT/configure" --with-model="$KEY" --with-task="$TASK" \
    --with-examples=0 --with-samples=$SAMPLES --with-temperature=$GEN_TEMP --with-top-p=$TOPP > configure.log 2>&1
  make -j$JOBS MAX_TOKENS=$MAX_TOKENS > make.log 2>&1 || true
  echo "[$KEY/$TASK] $(grep pass_rate summary.txt 2>/dev/null | tail -1)"
done
log "DONE $KEY (SAMPLES=$SAMPLES GEN_TEMP=$GEN_TEMP)"
