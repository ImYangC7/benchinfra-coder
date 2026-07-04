# ============================================================
# benches/registry.sh — the single place that lists every benchmark.
#
# To add a new bench: append one line to BENCHES below and drop a runner in
# benches/run_<name>.sh that takes <served_name> and honours config.sh env.
# run_all.sh iterates this registry, so nothing else needs to change.
#
# Format (pipe-separated):  name | runner_script | default_env
#   name         short id, also the results/<key>/<name> subdir hint
#   runner       script under benches/ (receives $KEY as $1)
#   default_env  space-separated VAR=VAL applied when this bench runs
# ============================================================

BENCHES=(
  "verilogeval | run_verilogeval.sh | SAMPLES=4 TEMP=0.8 TOPP=0.95"
  "rtllm       | run_rtllm.sh       | SAMPLES=4 TEMP=0.8 WORKERS=64 MAXTOK=32768"
  "archx       | run_archx.sh       | SAMPLES=5 TEMP=0.8 WORKERS=64 MAXTOK=49152"
  "realbench   | run_realbench.sh   | SAMPLES=5 TEMP=0.8 WORKERS=60 MAXTOK=49152"
  # kernelbench needs the engine STOPPED (VRAM) — excluded from the shared-engine
  # sweep by default. Run it separately: benches/run_kernelbench.sh <key> "1" 8 triton
  # "kernelbench | run_kernelbench.sh | MAXTOK=28000"
)
