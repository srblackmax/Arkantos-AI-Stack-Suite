# Sutra Benchmark Methodology (`BENCHMARKS_METHODOLOGY.md`)

**Library:** Sutra (`1.0.0`) — Native C++23 Agentic Reasoning, Tool Orchestration & Multimodal Memory Framework  
**Executable:** `./build-release-cuda/bench_sutra`  
**Regression CI Target:** `cmake --build build-release-cuda --target bench-ci`  

---

## Hardware Baseline

- **CPU:** `12th Gen Intel(R) Core(TM) i7-12700K` (12 physical cores: 8 P-cores + 4 E-cores, 20 logical threads, 800–5,100 MHz, AVX2 + FMA)
- **GPU:** `NVIDIA GeForce RTX 3070` (`8,192 MiB` GDDR6 VRAM, Driver `550.163.01`, Compute Capability `sm_86`, CUDA `12.4.131`)
- **RAM:** `48 GiB` Dual-Channel DDR4
- **Disk:** `Samsung SSD 970 EVO Plus 1TB` NVMe (`nvme0n1`)
- **Kernel & Compilers:** Debian GNU/Linux 13 (trixie), Kernel `6.12.113+deb13-amd64` (`x86_64`), `GCC 14.2.0` (`-O3 -std=c++23 -march=x86-64-v3`), `NVCC 12.4.131` (`-O3 -arch=sm_86`)

---

## 1. Cold-Start Initialization (`cold_start_init`)

**What is measured:** Wall-clock time (`p50` and `p99` over 20 timed runs after 5 warmup runs) to construct `sutra::ToolRegistry`, register 3 reflected C++ tools via `SUTRA_REGISTER_TOOL`, instantiate `sutra::Agent::create(cfg, reg, llm)`, and spawn a 10-worker `sutra::AgentRuntime(10U)` lock-free thread pool.  
**What is excluded:** Dynamic linker (`ld.so`) process startup and GGUF model weight loading (see `SUTRA_LLM_BACKENDS.md` for live GGUF init).  
**Instrumentation:** `sutra::monotonic_time_ns()` (`CLOCK_MONOTONIC_RAW`).  
**Hardware:** Intel Core i7-12700K (12C/20T), 48 GiB DDR4, Linux `6.12.113+deb13-amd64`.  
**Command to reproduce:**
```bash
./build-release-cuda/bench_sutra --json-out=results.json
```
**Sanity check:** Verifies `ag.has_value() == true` on every iteration; proves the `Agent` and `AgentRuntime` worker pool are live and valid.

---

## 2. Parallel vs. Sequential Tool Execution (`tool_10x20ms_sequential`, `tool_10x20ms_parallel`)

**What is measured:** Wall-clock time (`p50` and `p99` over 5 timed runs after 1 warmup run) to execute 10 independent `simulated_io_tool` calls (`20 ms` sleep each) via `sutra::ToolScheduler::execute_batch` in `ToolExecutionMode::Sequential` (`~201.4 ms`) vs `ToolExecutionMode::Parallel` (`~20.35 ms`, `9.89x` speedup).  
**What is excluded:** `ToolRegistry` registration and `InferenceArena` allocation.  
**Instrumentation:** `sutra::monotonic_time_ns()` around `scheduler.execute_batch(...)`.  
**Hardware:** Intel Core i7-12700K (12C/20T), 10-worker `WorkStealingPool`, Linux `6.12.113+deb13-amd64`.  
**Command to reproduce:**
```bash
./build-release-cuda/bench_sutra --json-out=results.json
```
**Sanity check:** Verifies `sr->size() == 10`, `pr->size() == 10`, and `*(*pr)[0].output == "io_ok"`; proves all 10 tool calls completed and returned valid payloads.

---

## 3. Overlapped Inference and Tool CPU Utilization (`overlapped_cpu_utilization`)

**What is measured:** Aggregate multi-core CPU utilization (`(process_cpu_time_ns / wall_time_ns) * 100%`, `p50` and `p99` over 7 timed batches of 8 agent runs after 2 warmup batches) while `OverlappedInferenceLlm` streams 8 `cpu_compute` tools (`250,000` trig iterations each) via `on_early_tool` and concurrently executes simulated token decoding (`400,000` iterations).  
**What is excluded:** Agent construction.  
**Instrumentation:** `clock_gettime(CLOCK_PROCESS_CPUTIME_ID)` divided by `sutra::monotonic_time_ns()`.  
**Hardware:** Intel Core i7-12700K (12C/20T), Linux `6.12.113+deb13-amd64`.  
**Command to reproduce:**
```bash
./build-release-cuda/bench_sutra --json-out=results.json
```
**Sanity check:** Verifies `r.has_value() && r->final_answer == "overlap_complete"` on every run; proves both early-streamed tools and final step completed.

---

## 4. Incremental Binary Checkpointing (`checkpoint_500turn_early_10_60`, `checkpoint_500turn_late_450_500`)

**What is measured:** Per-turn agent execution + incremental delta binary checkpointing (`enable_checkpointing = true` to `/tmp/sutra_bench_cpk.bin`) across 500 sequential turns after 25 warmup turns, comparing `p50`/`p99` latency across Early Turns (`10..60`) vs Late Turns (`450..500`).  
**What is excluded:** Final `flush_checkpoints()` synchronization after turn 500.  
**Instrumentation:** `sutra::monotonic_time_ns()` per turn.  
**Hardware:** Intel Core i7-12700K, Samsung SSD 970 EVO Plus 1TB NVMe, Linux `6.12.113+deb13-amd64`.  
**Command to reproduce:**
```bash
./build-release-cuda/bench_sutra --json-out=results.json
```
**Sanity check:** Calls `flush_checkpoints()` and verifies `std::filesystem::exists(cpk_file)` and `std::filesystem::file_size(cpk_file) > 16000` bytes before cleanup; proves all 525 turns were written to disk with headers and CRC32 checksums.

---

## 5. Mock-LLM Zero-Wait Orchestration & Theoretical Minimum (`theoretical_min_direct_dispatch`, `mock_llm_zero_wait_orchestration_n*`)

**What is measured:**
- **`theoretical_min_direct_dispatch`:** Theoretical floor for a thread-safe tool call in C++ (`std::lock_guard<std::mutex>` + indirect function pointer call `raw_fn(41)` returning `Result<std::string>("42")`, `p50 = 0.02 us/turn`).
- **`mock_llm_zero_wait_orchestration_n{1,2,4,8,16}`:** Full Sutra 2-step ReAct turn (`FastReActLlm` step 0 tool call $\to$ JSON schema argument extraction $\to$ `fast_noop_tool(41)` $\to$ `FastReActLlm` step 1 final answer `"42"`) across $N \in \{1, 2, 4, 8, 16\}$ concurrent agents on a 16-worker `AgentRuntime`, reported as `us/turn` (`p50`/`p99`) and `turns/s (mock-LLM)`.  
**What is excluded:** Neural network forward-pass inference (explicitly labeled `[mock-LLM]`; live GGUF `Qwen2.5-7B-Instruct-Q4_K_M` GPU decode is reported separately in `BENCHMARKS.md` §3.2.B).  
**Instrumentation:** `sutra::monotonic_time_ns()` over 15 timed batches of `80 * N` turns (after 4 warmup batches).  
**Hardware:** Intel Core i7-12700K (12C/20T), 48 GiB DDR4, Linux `6.12.113+deb13-amd64`.  
**Command to reproduce:**
```bash
./build-release-cuda/bench_sutra --json-out=results.json
```
**Sanity check:** Verifies `res.has_value() && res->final_answer == "42"` for every `AsyncRunHandle::wait()` across all $N$ agents; proves every concurrent ReAct loop executed both steps and parsed the reflected tool result.

---

## 6. Semantic Memory Vector Search (`semantic_search_cpu_1000x1536`, `semantic_search_gpu_1000x1536`)

**What is measured:** Top-5 dot-product similarity search over $1{,}000 \times 1{,}536$ `fp32` document embeddings (`6.14 MB` index) on:
- **CPU SIMD (`prefer_gpu = false`):** Runtime-dispatched AVX2 widening `fp32 -> fp64` dot-product scan (`~150.2 us/query` `p50`).
- **GPU CUDA (`prefer_gpu = true`):** Full host-to-device transfer (`Tensor::allocate` + `cudaMemcpyAsync` H2D), `batch_dot_f32_kernel` warp-shuffle reduction on RTX 3070, device-to-host score copy, and top-5 selection (`~1,382.7 us/query` `p50`).  
**What is excluded:** Document insertion (`add_document`).  
**Instrumentation:** `sutra::monotonic_time_ns()` over 50 timed queries after 5 warmup queries.  
**Hardware:** Intel Core i7-12700K + NVIDIA GeForce RTX 3070 (`8,192 MiB` VRAM, CUDA `12.4.131`).  
**Command to reproduce:**
```bash
./build-release-cuda/bench_sutra --json-out=results.json
```
**Sanity check:** Computes exact double-precision self dot product `gt_self_dot` for `doc_999` and asserts `(*res)[0].id == "doc_999"` and `|(*res)[0].score - gt_self_dot| <= 1e-4` on both CPU and GPU.

---

## 7. Three-Stage Process RSS (`rss_bare_process`, `rss_post_library_load`, `rss_post_gpu_init`)

**What is measured:** Resident Set Size (`/proc/self/statm` `rss_pages * 4096`) at three process lifecycle stages:
1. `rss_bare_process`: First line of `main()` before any Sutra object is constructed (`109.25 MB` when `libllama.so.0` + `libggml-cuda.so.0` + `libcublasLt.so.12` are linked via `DT_NEEDED`; `3.84 MB` in `build-release-headless`).
2. `rss_post_library_load`: After `ToolRegistry`, `Agent`, and `AgentRuntime(10)` initialization (`111.40 MB`, a `+2.15 MB` Sutra framework heap delta).
3. `rss_post_gpu_init`: After `SemanticMemory::search(..., true)` initializes the CUDA primary context on RTX 3070 + `1.0 s` idle (`217.95 MB`).  
**What is excluded:** N/A (exact OS kernel page accounting).  
**Instrumentation:** `/proc/self/statm`.  
**Hardware:** Intel Core i7-12700K + NVIDIA GeForce RTX 3070, Linux `6.12.113+deb13-amd64`.  
**Command to reproduce:**
```bash
./build-release-cuda/bench_sutra --json-out=results.json
```
**Sanity check:** Verifies `bare_rss_mb > 0.5`, `(post_lib_rss_mb - bare_rss_mb) <= 20.0`, and `post_gpu_rss_mb >= post_lib_rss_mb`.
