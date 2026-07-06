#!/usr/bin/env python3
"""
RTLLM v2 evaluation with iverilog (RTLLM ships Synopsys-VCS makefiles which we
don't have). Default (--num-samples 1): single greedy sample -> pass@1.
Optional average@K (--num-samples K --temperature T, K>1): draw K samples
(k=0 greedy, 1..K-1 sampled), report syntax_avg@K / func_avg@K = mean over
designs of (#passing candidates / K). Flattened (design,k) tasks + as_completed
so a slow long-think sample never blocks others.

  python run_rtllm.py --base-url http://localhost:8000/v1 --model base-9b \
      --out results/base-9b/rtllm --workers 64 --num-samples 4 --temperature 0.8
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

RTLLM_ROOT = os.environ.get("RTLLM_ROOT",
    os.path.join(os.environ.get("CODERBENCH_ROOT", "."), "RTLLM"))
SYS_MSG = "You are a Verilog RTL designer that only writes code using correct Verilog syntax."


def find_designs():
    designs = []
    for root, _, files in os.walk(RTLLM_ROOT):
        if "design_description.txt" in files and "testbench.v" in files:
            if "_chatgpt" in root:
                continue
            designs.append(root)
    return sorted(designs)


def query(base_url, model, prompt, max_tokens=32768, temperature=0.0):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS_MSG},
            {"role": "user", "content": prompt +
             "\n\nGive the complete Verilog code. Enclose your code with ```verilog and ```."},
        ],
        "temperature": temperature,
    }
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    body = json.dumps(payload).encode()
    # Retry transient 5xx / connection errors (vLLM queue overflow under load).
    # timeout=3600 (not 1800): reasoning/"thinking" models can run away on hard
    # problems and generate for >30min. A shorter client timeout cuts them off and
    # the retry re-runs from scratch into the same runaway -> a wasted retry loop
    # that never completes. 3600 lets a long think finish or hit max_tokens first.
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
    m = re.search(r"```(?:verilog|systemverilog)?\s*(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    mods = re.search(r"(module\b.*endmodule)", code, re.DOTALL)
    return mods.group(1) if mods else code


def verify_candidate(gen_path, tb_path):
    with tempfile.TemporaryDirectory() as wd:
        simv = os.path.join(wd, "simv")
        comp = subprocess.run(
            ["iverilog", "-g2012", "-Wno-timescale", "-o", simv, gen_path, tb_path],
            capture_output=True, text=True, timeout=60)
        if comp.returncode != 0:
            return 0, 0, "compile: " + comp.stderr[-300:]
        try:
            run = subprocess.run(["vvp", simv], capture_output=True, text=True, timeout=60)
            out = run.stdout + run.stderr
        except subprocess.TimeoutExpired:
            return 1, 0, "sim timeout"
        if re.search(r"\b(pass|passed)\b", out, re.IGNORECASE) and \
           not re.search(r"failure", out, re.IGNORECASE):
            return 1, 1, ""
        return 1, 0, "func fail"


def gen_verify_one(design_dir, k, base_url, model, out_dir, max_tokens, temperature):
    name = os.path.basename(design_dir)
    desc = open(os.path.join(design_dir, "design_description.txt")).read()
    tb = os.path.join(design_dir, "testbench.v")
    temp = 0.0 if k == 0 else temperature
    r = {"design": name, "k": k, "syntax": 0, "func": 0, "error": ""}
    try:
        resp = query(base_url, model, desc, max_tokens=max_tokens, temperature=temp)
    except Exception as e:
        r["error"] = f"query: {e}"
        return r
    code = extract_verilog(resp)
    gen_path = os.path.join(out_dir, f"{name}_s{k}.v")
    with open(gen_path, "w") as f:
        f.write(code)
    with open(os.path.join(out_dir, f"{name}_s{k}_response.txt"), "w") as f:
        f.write(resp)
    syntax, func, err = verify_candidate(gen_path, tb)
    r["syntax"], r["func"], r["error"] = syntax, func, err[:120]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    designs = find_designs()
    K = args.num_samples
    print(f"[rtllm] {len(designs)} designs x {K} = {len(designs)*K} tasks, "
          f"model={args.model}, workers={args.workers}, temp={args.temperature}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    per = {os.path.basename(d): {"syn": 0, "fun": 0, "s0s": 0, "s0f": 0, "s0e": ""} for d in designs}
    done = 0; total = len(designs) * K
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(gen_verify_one, d, k, args.base_url, args.model, args.out,
                          args.max_tokens, args.temperature)
                for d in designs for k in range(K)]
        for fu in as_completed(futs):
            r = fu.result(); done += 1
            p = per[r["design"]]; p["syn"] += r["syntax"]; p["fun"] += r["func"]
            if r["k"] == 0:
                p["s0s"] = r["syntax"]; p["s0f"] = r["func"]; p["s0e"] = r["error"]
            print(f"  [{done}/{total}] {r['design']:28s} k={r['k']} syn={r['syntax']} func={r['func']} {r['error'][:40]}", flush=True)

    n = len(designs)
    results = [{"design": os.path.basename(d), "num_samples": K,
                "syntax_avg": round(per[os.path.basename(d)]["syn"]/K, 4),
                "func_avg": round(per[os.path.basename(d)]["fun"]/K, 4),
                "syntax": per[os.path.basename(d)]["s0s"], "func": per[os.path.basename(d)]["s0f"],
                "error": per[os.path.basename(d)]["s0e"]} for d in designs]
    syn = sum(r["syntax"] for r in results); fun = sum(r["func"] for r in results)
    syn_avg = sum(r["syntax_avg"] for r in results)/n if n else 0
    fun_avg = sum(r["func_avg"] for r in results)/n if n else 0
    summary = {"model": args.model, "num_designs": n, "num_samples": K,
               "syntax_avg@%d" % K: round(100*syn_avg, 2), "func_avg@%d" % K: round(100*fun_avg, 2),
               "syntax_pass@1": round(100*syn/n, 2) if n else 0,
               "func_pass@1": round(100*fun/n, 2) if n else 0,
               "syntax_count": syn, "func_count": fun}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"[rtllm] {args.model}: syntax_avg@{K}={round(100*syn_avg,2)}% func_avg@{K}={round(100*fun_avg,2)}%  "
          f"(legacy syntax@1={summary['syntax_pass@1']}% func@1={summary['func_pass@1']}%)", flush=True)


if __name__ == "__main__":
    main()
