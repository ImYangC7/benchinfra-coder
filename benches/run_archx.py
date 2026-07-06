#!/usr/bin/env python3
"""
ArchXBench evaluation with iverilog. 71 RTL designs across level-0..level-6.
Default (--num-samples 1): single greedy sample -> syntax/func pass@1.
Optional n/t mode (--num-samples K --temperature T, K>1): draw K candidates
(k=0 greedy, 1..K-1 sampled), report per-design averaged:
  n = #candidates that compile (0..K); t = best compiling candidate's assertion-pass %.
Flattened (design,k) tasks + as_completed so a slow long-think sample never blocks.

t = 100*passed/(passed+failed) for "Passed:N,Failed:M"/JSON self-check tb;
    binary 0/100 for golden-compare and plain PASS/FAIL tb.

  python run_archx.py --base-url http://localhost:8000/v1 --model base-9b \
      --out results/base-9b/archxbench --workers 64 --num-samples 5 --temperature 0.8
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

ARCHX_ROOT = os.environ.get("ARCHX_ROOT",
    os.path.join(os.environ.get("CODERBENCH_ROOT", "."), "ArchXBench"))
SYS_MSG = "You are a Verilog RTL designer that only writes code using correct Verilog syntax."


def find_designs():
    designs = []
    for lvl in sorted(os.listdir(ARCHX_ROOT)):
        lvl_dir = os.path.join(ARCHX_ROOT, lvl)
        if not (lvl.startswith("level-") and os.path.isdir(lvl_dir)):
            continue
        for name in sorted(os.listdir(lvl_dir)):
            d = os.path.join(lvl_dir, name)
            if not os.path.isdir(d):
                continue
            if os.path.exists(os.path.join(d, "problem-description.txt")):
                designs.append(d)
    return designs


def tb_file(design_dir):
    cands = [f for f in os.listdir(design_dir)
             if f.endswith(".v") and (f == "tb.v" or f.startswith("tb") or "testbench" in f)]
    for pref in ("tb.v", "testbench.v"):
        if pref in cands:
            return pref
    return cands[0] if cands else None


def build_prompt(design_dir):
    desc = open(os.path.join(design_dir, "problem-description.txt")).read()
    spec_path = os.path.join(design_dir, "design-specs.txt")
    spec = open(spec_path).read() if os.path.exists(spec_path) else ""
    return (f"{desc}\n\n## Design Specification\n{spec}\n\n"
            "Write the complete, synthesizable Verilog module(s) implementing the "
            "design above. Match the module name and port list exactly as specified. "
            "Enclose your code with ```verilog and ```.")


def query(base_url, model, prompt, max_tokens=32768, temperature=0.0):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYS_MSG},
                     {"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    body = json.dumps(payload).encode()
    # Retry transient 5xx / connection errors (vLLM queue overflow under load).
    # timeout=3600 (not 1800): reasoning/"thinking" models can run away on hard
    # designs and generate for >30min. A shorter client timeout cuts them off and
    # the retry re-runs into the same runaway -> a wasted loop that never finishes.
    last = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                         data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                resp = json.load(r)
            return resp["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as e:
            last = e
            if e.code < 500:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(min(2 ** attempt, 30))
    raise last


def extract_verilog(text):
    if not text:
        return ""
    m = re.search(r"```(?:verilog|systemverilog)?\s*(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    mods = re.search(r"(module\b.*endmodule)", code, re.DOTALL)
    return mods.group(1) if mods else code


def assertion_pct(out):
    m = re.search(r'"passed"\s*:\s*(\d+).*?"failed"\s*:\s*(\d+)', out, re.DOTALL)
    if not m:
        m = re.search(r'Passed:\s*(\d+).*?Failed:\s*(\d+)', out, re.DOTALL | re.IGNORECASE)
    if m:
        passed, failed = int(m.group(1)), int(m.group(2))
        tot = passed + failed
        return (100.0 * passed / tot) if tot > 0 else 0.0
    up = out.upper()
    if "PASS" in up and "FAIL" not in up and "MISMATCH" not in up and not re.search(r"\bERROR\b", up):
        return 100.0
    return 0.0


def verify_candidate(code, design_dir, tb, has_golden):
    with tempfile.TemporaryDirectory() as wd:
        for item in os.listdir(design_dir):
            if item.endswith(".v") or item.startswith("."):
                continue
            os.symlink(os.path.join(design_dir, item), os.path.join(wd, item))
        shutil.copy(os.path.join(design_dir, tb), os.path.join(wd, tb))
        gen = os.path.join(wd, "generated.v")
        with open(gen, "w") as f:
            f.write(code)
        os.makedirs(os.path.join(wd, "outputs"), exist_ok=True)
        simv = os.path.join(wd, "simv")
        comp = subprocess.run(
            ["iverilog", "-g2012", "-Wno-timescale", "-o", simv, gen, os.path.join(wd, tb)],
            capture_output=True, text=True, errors="replace", timeout=120, cwd=wd)
        if comp.returncode != 0:
            return 0, 0.0, "compile: " + comp.stderr[-300:]
        try:
            run = subprocess.run(["vvp", simv], capture_output=True, text=True,
                                 errors="replace", timeout=180, cwd=wd)
            out = run.stdout + run.stderr
        except subprocess.TimeoutExpired:
            return 1, 0.0, "sim timeout"
        if has_golden:
            cmp = subprocess.run(["python3", "scripts/compare_outputs.py"],
                                 capture_output=True, text=True, errors="replace", timeout=60, cwd=wd)
            if cmp.returncode == 0 and "PASS" in (cmp.stdout + cmp.stderr).upper():
                return 1, 100.0, ""
            return 1, 0.0, "golden mismatch: " + (cmp.stdout + cmp.stderr)[-200:]
        pct = assertion_pct(out)
        return 1, pct, "" if pct >= 100.0 else f"func partial/fail: {pct:.0f}%"


def gen_verify_one(design_dir, k, base_url, model, out_dir, max_tokens, temperature):
    name = os.path.basename(design_dir)
    tb = tb_file(design_dir)
    r = {"design": name, "k": k, "syntax": 0, "t": 0.0, "error": ""}
    if tb is None:
        r["error"] = "no testbench"
        return r
    has_golden = os.path.exists(os.path.join(design_dir, "scripts", "compare_outputs.py"))
    temp = 0.0 if k == 0 else temperature
    try:
        resp = query(base_url, model, build_prompt(design_dir), max_tokens=max_tokens, temperature=temp)
    except Exception as e:
        r["error"] = f"query: {e}"
        return r
    code = extract_verilog(resp)
    with open(os.path.join(out_dir, f"{name}_s{k}.v"), "w") as f:
        f.write(code)
    with open(os.path.join(out_dir, f"{name}_s{k}_response.txt"), "w") as f:
        f.write(resp)
    syntax, pct, err = verify_candidate(code, design_dir, tb, has_golden)
    r["syntax"], r["t"], r["error"] = syntax, pct, err[:120]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    designs = find_designs()
    if args.limit:
        designs = designs[:args.limit]
    K = args.num_samples
    print(f"[archx] {len(designs)} designs x {K} = {len(designs)*K} tasks, "
          f"model={args.model}, workers={args.workers}, temp={args.temperature}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    per = {os.path.basename(d): {"n": 0, "bt": 0.0, "s0s": 0, "s0f": 0, "s0e": ""} for d in designs}
    done = 0; total = len(designs) * K
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(gen_verify_one, d, k, args.base_url, args.model, args.out,
                          args.max_tokens, args.temperature)
                for d in designs for k in range(K)]
        for fu in as_completed(futs):
            r = fu.result(); done += 1
            p = per[r["design"]]; p["n"] += r["syntax"]
            if r["syntax"] and r["t"] > p["bt"]:
                p["bt"] = r["t"]
            if r["k"] == 0:
                p["s0s"] = r["syntax"]; p["s0f"] = 1 if r["t"] >= 100.0 else 0; p["s0e"] = r["error"]
            print(f"  [{done}/{total}] {r['design']:30s} k={r['k']} syn={r['syntax']} t={r['t']:5.0f} {r['error'][:38]}", flush=True)

    ndes = len(designs)
    results = [{"design": os.path.basename(d), "num_samples": K, "n": per[os.path.basename(d)]["n"],
                "t": round(per[os.path.basename(d)]["bt"], 2), "syntax": per[os.path.basename(d)]["s0s"],
                "func": per[os.path.basename(d)]["s0f"], "error": per[os.path.basename(d)]["s0e"]} for d in designs]
    syn = sum(r["syntax"] for r in results); fun = sum(r["func"] for r in results)
    avg_n = sum(r["n"] for r in results)/ndes if ndes else 0
    avg_t = sum(r["t"] for r in results)/ndes if ndes else 0
    summary = {"model": args.model, "num_designs": ndes, "num_samples": K,
               "n": round(avg_n, 2), "t": round(avg_t, 2),
               "syntax_pass@1": round(100*syn/ndes, 2) if ndes else 0,
               "func_pass@1": round(100*fun/ndes, 2) if ndes else 0,
               "syntax_count": syn, "func_count": fun}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"[archx] {args.model}: n={summary['n']}/{K} t={summary['t']}%  "
          f"(legacy syntax@1={summary['syntax_pass@1']}% func@1={summary['func_pass@1']}%)", flush=True)


if __name__ == "__main__":
    main()
