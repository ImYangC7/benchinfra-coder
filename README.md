# benchinfra-coder

Infrastructure + one-click scripts for evaluating LLM **code generation** on
Verilog/RTL and CUDA-kernel benchmarks. One vLLM engine is started once and
**every benchmark reuses the same OpenAI-compatible proxy** — no per-bench
re-serving.

Currently wired up:

| Benchmark      | Task                              | Size            | Default metric |
|----------------|-----------------------------------|-----------------|----------------|
| **VerilogEval**| spec-to-rtl + code-complete       | 156 + 156       | average@4 (`pass_rate`) |
| **RTLLM** v2   | RTL design from description       | 50 designs      | average@4 (syntax / func) |
| **ArchXBench** | arithmetic RTL, level 0–6         | 71 designs      | `n` / `t` (5 samples) |
| **RealBench**  | module-level RTL from real IP     | 60 modules      | Syn@1/@5 + Func@1/@5 |
| **KernelBench**| CUDA / Triton kernels             | level 1 (100)   | compiled + fast_0 (pass@1) |

> This repo holds **only the infra** (engine, runners, one-click orchestration).
> The benchmark datasets + their venvs live under `$CODERBENCH_ROOT` and are
> driven in place.

---

## Architecture — engine/bench decoupling

The engine starts once; all benches hit the one proxy on `:8000`.

```
  engine/serve_vllm.sh  → N single-GPU vLLM (:8101..) + round-robin proxy (:8000)
                │
                ▼   OPENAI_BASE_URL=http://localhost:8000/v1
  ┌─────────────┼──────────────┬──────────────┬───────────────┐
  ▼             ▼              ▼              ▼               ▼
 verilogeval   rtllm         archx        realbench      kernelbench
        all hit :8000, no re-serve; sweep ends with serve_vllm.sh stop
```

`engine/proxy_rr.py` also handles **thinking**: with `THINK=1` (default) it keeps
the model reasoning and strips `<think>…</think>` from the response so the code
extractors see clean output; `THINK=0` forces `enable_thinking=false`.

---

## Quickstart

```bash
# 0. point config at your dataset root (edit config.sh or export)
export CODERBENCH_ROOT=/path/to/coderbench

# 1. one-click: start engine, run all registered benches, stop engine
bash run_all.sh /path/to/model qwen36-base 65536

# 2. summarize
bash summarize.sh qwen36-base
```

Run a single bench (engine already up):

```bash
bash engine/serve_vllm.sh /path/to/model qwen36-base 256 65536
SAMPLES=4 TEMP=0.8 bash benches/run_rtllm.sh qwen36-base
bash engine/serve_vllm.sh stop
```

KernelBench needs the engine **stopped** (eval uses the full VRAM):

```bash
# generate while engine is up, then stop it, then sharded eval
bash engine/serve_vllm.sh stop
bash benches/kb_sharded_eval.sh qwen36-base_level1 triton
python benches/kb_merge_shards.py qwen36-base_level1
```

---

## Metrics

- **average@K** (VerilogEval, RTLLM): mean over problems of `#passing / K` — the
  expected single-shot success rate, lower variance than pass@1.
- **pass@K** (RealBench): `1 - C(n-c, k)/C(n, k)` — probability ≥1 of K passes.
- **n / t** (ArchXBench): `n` = avg #syntactically-valid candidates (0..K);
  `t` = avg best candidate's testbench assertion-pass % (fractional for
  `Passed:N,Failed:M` self-checks, binary 0/100 for golden/plain-PASS).
- **fast_0** (KernelBench): functional correctness rate (pass@1).

Multi-sample modes use `temperature>0` (default 0.8) — the k=0 sample stays
greedy so the legacy pass@1 is reproducible.

---

## Adding a new benchmark

1. Write `benches/run_<name>.sh` taking `<served_name>` as `$1`, sourcing
   `config.sh` + `lib/common.sh`, calling `require_engine`, and writing results
   under `$RESULTS_DIR/<key>/`.
2. Add one line to `benches/registry.sh`:
   ```
   "<name> | run_<name>.sh | SAMPLES=4 TEMP=0.8"
   ```
3. `run_all.sh` picks it up automatically. Optionally extend `summarize.sh`.

---

## Config

All paths/tunables are in `config.sh` (override by exporting before sourcing):

| Var | Meaning | Default |
|-----|---------|---------|
| `CODERBENCH_ROOT` | dataset + venv root | (edit for your host) |
| `PROXY_PORT` | shared endpoint | 8000 |
| `N_GPU` / `BASE_PORT` | single-GPU backends | 8 / 8101 |
| `MAX_MODEL_LEN` | engine context window | 32768 |
| `THINK` | keep reasoning, strip `<think>` | 1 |
| `IVERILOG12_BIN` | iverilog v12 on PATH front | env/iverilog12/bin |

**Gotcha:** a bench's `max_tokens` must be **< engine `MAX_MODEL_LEN` minus the
longest prompt**, or vLLM returns HTTP 500 on context overflow. With a 65536
window, ArchX/RealBench use `MAXTOK=49152`.

---

## Known pitfalls

Hard-won lessons from running reasoning ("thinking") models at scale. Read
before evaluating a new checkpoint.

### 1. Thinking runaway → timeout/retry deadlock
Reasoning models can generate for **>30 min** on hard designs (ArchX
fft/matmul/aes, RTLLM greedy, RealBench aes/e203) — a few runaway candidates
never hit `max_tokens` and never stop. Symptoms:
- completion counter frozen for a long time while the engine is still 100% busy
  producing tokens (48 concurrent slots all full);
- a whole run blocked on the last 1–2 candidates.

Mitigations already baked in:
- `run_rtllm.py` / `run_archx.py` use a **3600 s** client `urlopen` timeout (not
  1800). A shorter timeout cuts a long think off mid-stream and the retry
  re-runs into the *same* runaway → a wasted loop that never finishes. 3600 lets
  the think finish or hit `max_tokens` first.
- If a run still won't finish, **finalize from disk** (see recovery tools below)
  instead of waiting: every already-saved `*_sN.v` is scored, missing runaway
  candidates are counted as `syntax=0` fail (they never produced code anyway).
  Impact is <1% — the missing ones are the hardest designs, which fail regardless.

### 2. RealBench `max_tokens` quasi-deadlock
On RealBench (only ~30–60 samples), a large `max_tokens` (e.g. 49152) lets a
couple of runaway samples generate for **hours** without hitting the cap, hanging
the whole bench. If a thinking model wedges here, **drop `max_tokens` to 16384**
and rerun — note in your report that the token budget differs from other models.

### 3. VerilogEval FUSE VCD-write D-state hang
On a FUSE-backed working dir (e.g. dop-fuse), `vvp` processes that dump a VCD can
wedge in uninterruptible **D-state** while writing, hanging the `make` verify.
Use `benches/retest_ve_tmp.py` to re-run just the stuck sims in a real `/tmp`
overlay; the official `sv-iv-analyze` still computes `pass_rate` for identical
scoring.

### 4. `column` missing → average@4 silently becomes average@1
VerilogEval's `samples.mk` fans out multi-sample generation via `column`
(util-linux). If `column` is not on PATH it **silently degrades to a single
sample** — `SAMPLES=4` quietly scores average@1. `load_verilog_toolchain` warns
on this; install util-linux's `column` before running average@4.

### 5. RealBench verify needs `verilator` on PATH
If `verilator` (oss-cad-suite/bin) isn't on PATH, RealBench syntax scores come
back **all zero** — a false regression, not a model problem. Confirm the toolchain
is loaded before trusting a 0.

## Recovery tools

For finalizing a run that a thinking runaway won't let finish cleanly:

- `benches/archx_recover_summary.py` — re-verify every saved ArchX `*_sN.v` from
  disk and write `summary.json` in the exact schema `run_archx.py` would. Missing
  candidates count as `syntax=0` (default), or pass `--regen` to regenerate them
  against a live engine. Guards `verify_candidate` with try/except so one
  pathological file (iverilog 120 s cap → `TimeoutExpired`) can't crash recovery.
  ```bash
  python benches/archx_recover_summary.py --model M --out results/M/archxbench --num-samples 5
  ```
- `benches/retest_ve_tmp.py` — regenerate VerilogEval iv-test logs in `/tmp` to
  dodge the FUSE VCD hang (pitfall 3). Only redoes missing / TIMEOUT logs.
  ```bash
  K=4 python benches/retest_ve_tmp.py <task_dir> <dataset_dir> <problems_file> [jobs]
  ```

---

## Layout

```
benchinfra-coder/
├── config.sh              # central paths + tunables
├── run_all.sh             # one-click sweep (engine → benches → stop)
├── summarize.sh           # one metrics table per model
├── engine/
│   ├── serve_vllm.sh      #   N single-GPU vLLM + proxy (start/stop)
│   └── proxy_rr.py        #   round-robin proxy + thinking handling
├── benches/
│   ├── registry.sh        #   the list run_all.sh iterates
│   ├── run_verilogeval.sh
│   ├── run_rtllm.sh / .py
│   ├── run_archx.sh / .py
│   ├── run_realbench.sh / gen_realbench.py
│   ├── run_kernelbench.sh
│   ├── kb_sharded_eval.sh #   8-shard KB eval (avoids mp.Pool crash)
│   ├── kb_merge_shards.py
│   ├── archx_recover_summary.py  # finalize ArchX from disk (runaway recovery)
│   └── retest_ve_tmp.py   #   re-run stuck VerilogEval sims in /tmp (FUSE hang)
└── lib/
    └── common.sh          # log / require_engine / wait_for_engine / toolchain
```
