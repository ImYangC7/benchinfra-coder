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
│   └── kb_merge_shards.py
└── lib/
    └── common.sh          # log / require_engine / wait_for_engine / toolchain
```
