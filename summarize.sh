#!/bin/bash
# ============================================================
# summarize.sh — print one table of all bench metrics for a served model,
# reading results/<key>/. Robust to partial runs (missing files -> "-").
#   bash summarize.sh <served_name>
# ============================================================
source "$(dirname "$0")/config.sh"
KEY=$1
[ -z "$KEY" ] && { echo "usage: summarize.sh <served_name>"; exit 1; }
R="$RESULTS_DIR/$KEY"

"$SYS_PY" - "$R" "$KEY" <<'PY'
import json, os, sys, re
R, KEY = sys.argv[1], sys.argv[2]
print(f"\n=== {KEY} ===  ({R})\n")

# VerilogEval: pass_rate line in results/<key>/<task>/summary.txt
def ve(task):
    f = os.path.join(R, task, "summary.txt")
    if not os.path.exists(f): return "-"
    for line in open(f):
        m = re.search(r"pass_rate\s*=\s*([\d.]+)", line)
        if m: return f"{float(m.group(1)):.2f}"
    return "-"
print(f"VerilogEval  spec-to-rtl     : {ve('spec-to-rtl')}")
print(f"VerilogEval  code-complete   : {ve('code-complete-iccad2023')}")

# RTLLM: summary.json
def js(path):
    return json.load(open(path))["summary"] if os.path.exists(path) else None
s = js(os.path.join(R, "rtllm", "summary.json"))
if s:
    k = s.get("num_samples", 1)
    print(f"RTLLM  syntax_avg@{k}         : {s.get('syntax_avg@%d'%k, s.get('syntax_pass@1'))}")
    print(f"RTLLM  func_avg@{k}           : {s.get('func_avg@%d'%k, s.get('func_pass@1'))}")
else:
    print("RTLLM                        : -")

# ArchX: summary.json (n/t)
s = js(os.path.join(R, "archxbench", "summary.json"))
if s:
    print(f"ArchXBench  n                : {s.get('n')}/{s.get('num_samples')}")
    print(f"ArchXBench  t                : {s.get('t')}%")
else:
    print("ArchXBench                   : -")

# RealBench: verify.log last "task_level:module s1 s5 f1 f5 ..." line
f = os.path.join(R, "realbench", "verify.log")
rb = "-"
if os.path.exists(f):
    for line in open(f):
        if line.startswith("task_level:module"):
            rb = line.split("task_level:module", 1)[1].strip()
if rb != "-":
    p = rb.split()
    # order: syntax_1 syntax_5 function_1 function_5 formal_1 formal_5
    def pct(x):
        try: return f"{float(x)*100:.1f}"
        except: return x
    if len(p) >= 4:
        print(f"RealBench  Syn@1 / Syn@5     : {pct(p[0])} / {pct(p[1])}")
        print(f"RealBench  Func@1 / Func@5   : {pct(p[2])} / {pct(p[3])}")
    else:
        print(f"RealBench                    : {rb}")
else:
    print("RealBench                    : -")

# KernelBench: level dirs
kb = os.path.join(R, "kernelbench")
if os.path.isdir(kb):
    for lv in sorted(os.listdir(kb)):
        f = os.path.join(kb, lv, "eval_results.json")
        if not os.path.exists(f): continue
        rows = []
        def w(o):
            if isinstance(o, dict):
                if "compiled" in o: rows.append(o); return
                [w(v) for v in o.values()]
            elif isinstance(o, list):
                [w(v) for v in o]
        w(json.load(open(f)))
        n = len(rows); corr = sum(1 for r in rows if r.get("correctness"))
        print(f"KernelBench {lv}  fast_0     : {100*corr/n if n else 0:.1f}%  (compiled {sum(1 for r in rows if r.get('compiled'))}/{n})")
print()
PY
