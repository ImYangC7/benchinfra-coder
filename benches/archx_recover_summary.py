#!/usr/bin/env python3
"""
ArchX summary recovery — for when a run_archx.py run is killed or a few candidates
never finished (reasoning models can run away on hard designs: fft/matmul/aes/fir).

run_archx writes summary.json only after ALL (design, k) candidates finish, so a
handful of runaway candidates block the whole summary even though the other ~99%
of *_sN.v files are already on disk and fully usable.

This script:
  1. re-verifies every saved *_sN.v candidate from disk (fast, no engine needed),
  2. OPTIONALLY regenerates missing candidates via the engine (--regen), or treats
     them as syntax=0 fail (default — matches "a runaway that never produced code"),
  3. writes summary.json in the identical schema run_archx would have.

verify_candidate can raise subprocess.TimeoutExpired (iverilog 120s cap on a huge
generated.v); we catch it so one pathological file can't crash the whole recovery.

Usage (finalize from disk, missing -> fail; no engine needed):
  python archx_recover_summary.py --model M --out results/M/archxbench --num-samples 5
Usage (also regenerate missing candidates against a live engine):
  python archx_recover_summary.py --model M --out results/M/archxbench --num-samples 5 \
      --regen --base-url http://localhost:8000/v1 --max-tokens 32768 --temperature 0.8
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import run_archx as R  # reuse find_designs, tb_file, build_prompt, query, extract_verilog, verify_candidate


def recover_one(design_dir, k, out_dir, regen, base_url, model, max_tokens, temperature):
    name = os.path.basename(design_dir)
    tb = R.tb_file(design_dir)
    r = {"design": name, "k": k, "syntax": 0, "t": 0.0, "error": ""}
    if tb is None:
        r["error"] = "no testbench"
        return r
    has_golden = os.path.exists(os.path.join(design_dir, "scripts", "compare_outputs.py"))
    vpath = os.path.join(out_dir, f"{name}_s{k}.v")
    if os.path.exists(vpath) and os.path.getsize(vpath) > 0:
        code = open(vpath).read()
    elif regen:
        # missing candidate -> regenerate with the fixed query (retry + 3600s timeout)
        temp = 0.0 if k == 0 else temperature
        try:
            resp = R.query(base_url, model, R.build_prompt(design_dir), max_tokens=max_tokens, temperature=temp)
        except Exception as e:
            r["error"] = f"query: {e}"
            return r
        code = R.extract_verilog(resp)
        with open(vpath, "w") as f:
            f.write(code)
        with open(os.path.join(out_dir, f"{name}_s{k}_response.txt"), "w") as f:
            f.write(resp)
    else:
        # missing and not regenerating -> a runaway that never produced code = fail
        r["error"] = "missing (runaway -> syn=0)"
        return r
    try:
        syntax, pct, err = R.verify_candidate(code, design_dir, tb, has_golden)
        r["syntax"], r["t"], r["error"] = syntax, pct, err[:120]
    except Exception as e:  # e.g. iverilog subprocess.TimeoutExpired on a huge file
        r["error"] = f"verify-timeout/err: {type(e).__name__}"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--regen", action="store_true",
                    help="regenerate missing candidates against a live engine (needs --base-url)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    designs = R.find_designs()
    K = args.num_samples
    per = {os.path.basename(d): {"n": 0, "bt": 0.0, "s0s": 0, "s0f": 0, "s0e": ""} for d in designs}
    done = 0
    total = len(designs) * K
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(recover_one, d, k, args.out, args.regen, args.base_url,
                          args.model, args.max_tokens, args.temperature)
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
    print(f"[archx-recover] {args.model}: n={summary['n']}/{K} t={summary['t']}%  "
          f"(syntax@1={summary['syntax_pass@1']}% func@1={summary['func_pass@1']}%)", flush=True)


if __name__ == "__main__":
    main()
