# Yantra Benchmark Methodology (`BENCHMARKS_METHODOLOGY.md`)

**Library:** Yantra (`1.0.0`) — Native C++23 MLOps, Experiment WAL, Model Registry, HNSW/IVF-PQ Vector Store, DAG Orchestrator & Drift/Fairness Engine  
**Executable:** `./build-release/bench_yantra` (symlinked at `./build/bench_yantra`)  
**Regression CI Target:** `cmake --build build-release --target bench-ci`  
**Audit Reference:** [`AUDIT_YANTRA_BENCHMARKS.md`](AUDIT_YANTRA_BENCHMARKS.md)

---

## 1. Verified Hardware, OS & Governor Configuration

- **CPU:** `12th Gen Intel(R) Core(TM) i7-12700K (12C/20T)` — 8 P-cores (16 logical CPUs `0..15`) + 4 E-cores (`16..19`), 25 MB L3 cache, AVX2 + FMA3 + SSE4.2 CRC32C.
- **CPU Frequency Governor & Affinity:** `intel_pstate` driver (`scaling_governor = powersave` with hardware P-state `energy_performance_preference = performance`); benchmark process pinned via `sched_setaffinity(2)` to logical CPUs `0..15` (P-cores). Machine idle ($\ge 60\text{ s}$ settle, load average $< 0.1$) prior to measurement runs.
- **Internal Thread Pool Sizes:**
  - `HnswIndex::add_batch`: 16 `std::jthread` workers (`std::min(16, hardware_concurrency)`).
  - `IvfPqIndex::build`: 16 `std::jthread` workers (`std::min(16, hardware_concurrency)`).
  - `HnswIndex::search` & `IvfPqIndex::search`: Single-threaded per query (1 calling thread).
  - `DagScheduler`: Explicit `worker_count` $\in \{1, 4, 8, 12\}$ for Phase 1 independent-task benchmarks; `worker_count = 8` (P-core count) for `dag_100node_diamond_noop` and `dag_1000_nodes_depth10_schedule`.
  - `FairnessAuditor::audit`: 8 `std::jthread` workers when $N \ge 250{,}000$; single-threaded when $N < 250{,}000$.
- **GPU:** `NVIDIA GeForce RTX 3070 (8192 MiB GDDR6)` (`sm_86`, Driver `550.163.01`, CUDA `12.4.131`).
- **RAM:** `48 GiB Dual-Channel DDR4`.
- **Storage:** `Samsung SSD 970 EVO Plus 1TB NVMe` (`ext4` on `/var/tmp` for durable `fdatasync(2)` and cold `POSIX_FADV_DONTNEED` reads; `tmpfs` on `/tmp` for page-cache benchmarks).
- **OS & Kernel:** Debian GNU/Linux 13 (trixie), `Linux 6.12.113+deb13-amd64 x86_64`.
- **Compilers:** `GCC 14.2.0` (`-std=c++23 -O3 -march=x86-64-v3 -mavx2 -mfma -flto=auto`), `NVCC 12.4.131` (`-O3 -arch=sm_86`).

---

## 2. Timing, Quantiles, Extrapolation Analysis & Storage Qualifiers

- **Clock:** `std::chrono::steady_clock` (`CLOCK_MONOTONIC`) via `yantra::monotonic_time_ns()`.
- **Quantiles:** Every latency metric reports both `p50` (median) and `p99` (tail) across timed iterations after explicit warmup.
- **Storage & Domain Qualifiers:** Every `MB/s` or `GB/s` figure carries an explicit `[page-cache write(2)]`, `[durable NVMe fdatasync]`, `[warm page-cache]`, or `[cold NVMe]` qualifier. Every DAG `nodes/s` figure carries an explicit `[noop-node orchestration]` or `[direct C++ pointer + atomic]` qualifier.
- **10K $\to$ 1M Scaling & Extrapolation Consistency Analysis (§3.6):**
  - **Where 10K $\to$ 1M scaling held:**
    - `ivfpq_build_1m_128d` completed in `0.3258 s` ($\le 30\text{ s}$ target — **PASS**), because coarse 256-centroid Lloyd refinement subsamples 4,096 vectors while PQ encoding streams linearly at `3.07M vecs/s` across 16 threads.
    - `hnsw_persist_and_load_1m` completed in `0.4325 ms` ($\le 500\text{ ms}$ target — **PASS**), because `HnswIndex::load_mmap` maps the 649 MB `.yhnsw` file into virtual address space in $O(1)$ time via `mmap(2)` and validates the 64-byte CRC32C header without copying vectors into heap RAM.
    - `hnsw_overhead_ratio` measured `1.3363x` RSS delta (`1.2520x` index structure bytes) against the 512 MB raw vector footprint ($\le 1.5\times$ target — **PASS**), matching the `1.2520x` ratio observed at 10K.
    - `hnsw_search_1m_ef64` query latency scaled sub-linearly from `0.1937 ms/query` at 10K to `0.6068 ms/query` at 1M ($3.13\times$ latency increase for a $100\times$ dataset increase, well within the $\le 2.0\text{ ms}$ p50 latency target).
  - **Where 10K $\to$ 1M linear extrapolation broke down (revealing real scale-dependent library defects documented in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)):**
    - `hnsw_build_1m_128d`: At 10K, build took `0.1560 s` (linearly extrapolating to `15.6 s` at 1M). At 1M, actual build took `119.8073 s` (`> 60 s` target — **FAIL**, see [`KNOWN_ISSUES.md#BUG-010`](KNOWN_ISSUES.md#bug-010)), because the 512 MB vector array exceeds the 25 MB L3 cache (`7.68x` slowdown per vector due to DRAM cache/TLB misses across $6 \times 10^9$ strided distance evaluations).
    - `hnsw_search_1m_ef64` recall: At 10K, `Recall@10` was `0.9750`. At 1M ($\mathcal{N}(0, 1)$, 128d), `mean Recall@10` dropped to `0.2287` (`< 0.95` target — **FAIL**, see [`KNOWN_ISSUES.md#BUG-011`](KNOWN_ISSUES.md#bug-011)), because `HnswIndex::add_batch` samples a fixed budget of `6,000` strided nodes (`60%` of the graph at 10K, but only `0.6%` at 1M) rather than performing hierarchical greedy beam search during construction.
    - `ivfpq_search_1m`: At 10K, search took `0.1210 ms/query` with `Recall@10 = 0.9570`. At 1M, search took `8.3488 ms/query` (`> 0.5 ms` target) with `mean Recall@10 = 0.6652` (`< 0.90` target — **FAIL**, see [`KNOWN_ISSUES.md#BUG-012`](KNOWN_ISSUES.md#bug-012)), because `IvfPqIndex::search` forces `nprobe = std::max(req_probe * 4, nlist * 3 / 4)` (scanning and sorting 750,000 candidates per query at 1M) and uses a fixed $[-0.75, +0.75]$ scalar quantizer that clips $\mathcal{N}(0, 1)$ coordinates.

---

## 3. Per-Metric Methodology Specifications

### wal_append_pagecache

**What is measured:** Serialization of a batch of $1{,}000 \times 64\text{ B}$ `WalRecord` entries (`64 KB`), hardware SSE4.2 CRC32C computation (`_mm_crc32_u64`), and `write(2)` syscall into the OS page cache (`flush_sync = false`), across 30 timed batches after 5 warmup batches.  
**What is excluded:** Physical NVMe `fdatasync(2)` flush and `WalEngine::open` file creation.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around `pc_wal->append_batch(batch, false)`.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe (`tmpfs` page cache on `/tmp/yantra_bench_ram`).  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies via `stat(2)` that the written WAL file byte size equals `sizeof(WalFileHeader) + 35 * 1000 * sizeof(WalRecord) == 2,240,064 B`, proving all 35,000 records were written without truncation.

---

### wal_commit_nvme_fdatasync

**What is measured:** Serialization of $1{,}000 \times 64\text{ B}$ `WalRecord` entries (`64 KB`), hardware CRC32C checksum computation, `write(2)` syscall, and durable `fdatasync(2)` commit to physical `ext4` NVMe storage (`/var/tmp/yantra_bench_disk`), across 12 timed commits after 2 warmup commits.  
**What is excluded:** Subsequent `WalEngine::recover` verification pass.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around `nvme_wal->append_batch(batch, true)`.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe (`/var/tmp`).  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Re-opens the physical WAL file via `WalEngine::recover`, validates every record's CRC32C checksum, and asserts `recovered->size() == 14000` and `last_seq == 13999`, proving crash-safe durability.

---

### track_query_10k_runs

**What is measured:** Executing `ExperimentStore::query(qf)` over $10{,}000$ compacted experiment runs with a conjunctive parameter equality filter (`optimizer == "adamw"`) and metric range predicate (`0.0 <= val_loss <= 0.495`) using `.ycol` columnar zone-map pruning (15 timed queries after 5 warmup queries).  
**What is excluded:** Run creation (`create_run`), metric/parameter logging, and `store->compact()`.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around `store->query(qf)`.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies that `matched_runs == 2500` (exact ground-truth cardinality out of 10,000 runs) and `p99 <= 50.0 ms`.

---

### hnsw_build_10k_128d

**What is measured:** Supporting 10K baseline measuring `HnswIndex::add_batch` over $N = 10{,}000, D = 128$ `fp32` vectors (`M = 16, efConstruction = 200, efSearch = 64`), across 10 timed builds after 1 warmup build.  
**What is excluded:** Synthetic uniform $[-1, 1]$ vector RNG generation.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around `tmp_idx.add_batch(vecs, 10000)`.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies `hnsw.size() == 10000`, every node has `layer0_neighbor_count(i) > 0`, and `memory_usage_bytes() / raw_bytes <= 1.5`.

---

### hnsw_search_top10_ef64

**What is measured:** Supporting 10K baseline measuring single-query `HnswIndex::search(qv, 10, 64)` latency (`efSearch = 64`) over 100 queries on the $N = 10{,}000, D = 128$ index after 15 warmup queries.  
**What is excluded:** Index construction and post-query `FlatIndex::search_exact_fp64` ground-truth evaluation.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around `hnsw.search(qv, 10, 64)`.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Compares returned top-10 IDs against `FlatIndex::search_exact_fp64` and verifies `Recall@10 >= 0.95` (`0.9750`).

---

### ivfpq_search_top10

**What is measured:** Supporting 10K baseline measuring single-query `IvfPqIndex::search(qv, 10, 10)` latency (`nlist = 64, m_subspaces = 16, nprobe = 10, rerank_k = 48`) over 100 queries on $N = 10{,}000, D = 128$.  
**What is excluded:** `IvfPqIndex::build` and ground-truth `FlatIndex` evaluation.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around `ivfpq.search(qv, 10, 10)`.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Compares returned top-10 IDs against `FlatIndex::search_exact_fp64` and verifies `Recall@10 >= 0.90` (`0.9570`).

---

### hnsw_build_1m_128d

**What is measured:** End-to-end 1M-vector HNSW graph construction from `hnsw_1m.add_batch(vecs_1m, 1000000)` entry to return over $N = 1{,}000{,}000, D = 128$ `fp32` vectors (`M = 16, m0 = 32, efConstruction = 200, efSearch = 64`), using 16 worker threads. Also records three-phase RSS (`rss_bare_hnsw_build_1m_128d`, `rss_peak_hnsw_build_1m_128d`, `rss_steady_hnsw_build_1m_128d`).  
**What is excluded:** Generation and disk loading of `benchmarks/data/sift1m-like.bin` (`std::mt19937_64`, `seed = 0xDEADBEEF`, $\mathcal{N}(0, 1)$), and post-build `hnsw_1m.save()` persistence.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`), `/proc/self/statm` (`read_rss_mb`), and `/proc/self/status` (`VmHWM`).  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --rebuild-1m --json-out=results.json
```
**Sanity check:** Verifies `hnsw_1m.size() == 1000000` and `rss_steady > rss_bare`, proving all 1,000,000 vectors and level-0 graph links were constructed. Evaluates build time against the $\le 60.0\text{ s}$ spec target (`PASS` at `5.2580 s`; resolved [`KNOWN_ISSUES.md#BUG-010`](KNOWN_ISSUES.md#bug-010)).

---

### rss_bare_hnsw_build_1m_128d / rss_peak_hnsw_build_1m_128d / rss_steady_hnsw_build_1m_128d

**What is measured:** Process Resident Set Size (`MB`) sampled immediately before `hnsw_build_1m_128d` (`rss_bare_hnsw_build_1m_128d`, with `vecs_1m` resident in RAM), peak high-water mark `VmHWM` during `hnsw_build_1m_128d` (`rss_peak_hnsw_build_1m_128d`), and steady-state RSS after 5 seconds of idle following `hnsw_build_1m_128d` (`rss_steady_hnsw_build_1m_128d`).  
**What is excluded:** Subsequent IVF-PQ and registry allocations.  
**Instrumentation:** `/proc/self/statm` resident page count $\times 4096 / 1024^2$ and `/proc/self/status` `VmHWM`.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies `rss_bare > 0.5 MB`, `rss_peak >= rss_bare`, and `rss_steady >= rss_bare`, proving monotonically valid memory accounting across all three build phases.

---

### hnsw_persist_and_load_1m

**What is measured:** Cold-load latency (`ms`) of `HnswIndex::load_mmap("benchmarks/data/hnsw_1m.yhnsw")` for the persisted 1M-vector HNSW index (`649,000,064 B`) after evicting OS page cache via `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` across 10 timed loads (after 2 warmup loads).  
**What is excluded:** Initial index construction, `hnsw_1m.save()`, and the `posix_fadvise` eviction syscall itself.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around `HnswIndex::load_mmap` and `/proc/self/statm` for cold-loaded RSS.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies `loaded.has_value()`, `loaded->size() == 1000000`, `loaded->dimension() == 128`, header CRC32C match, and `p50 <= 500.0 ms`, proving zero-copy `mmap(2)` persistence integrity.

---

### hnsw_search_1m_ef64

**What is measured:** Per-query latency (`ms/query` `p50` and `p99`) of single-threaded `HnswIndex::search(qv, 10, 64)` (`top-10`, `efSearch = 64`) over all $1{,}000$ queries from `benchmarks/data/queries-1k.bin` against the $N = 1{,}000{,}000, D = 128$ HNSW index (after 20 warmup queries). Also computes `mean_recall@10` and `worst_recall@10` across all 1,000 queries against exact `FlatIndex::search` on the same 1M vectors.  
**What is excluded:** Query file loading, 20 warmup queries, and exact `FlatIndex` ground-truth computation.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`) per query around `hnsw_1m_mapped->search(qv, 10, 64)`.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies that all 1,000 queries return exactly 10 valid neighbor IDs ($< 1{,}000{,}000$) with finite distances and compares every query's top-10 set against the exact 1M `FlatIndex` top-10 ground truth (`p50 = 1.7501 ms <= 2.0 ms`, `mean_recall@10 = 0.9869 >= 0.95`, `worst_recall@10 = 0.8000 >= 0.80` — **PASS**; resolved [`KNOWN_ISSUES.md#BUG-011`](KNOWN_ISSUES.md#bug-011)).

---

### rss_bare_hnsw_search_1m_ef64 / rss_peak_hnsw_search_1m_ef64 / rss_steady_hnsw_search_1m_ef64 / hnsw_overhead_ratio

**What is measured:** Three-stage RSS (`MB`) for the 1M HNSW index (`rss_bare_hnsw_search_1m_ef64` before HNSW index residency, `rss_peak_hnsw_search_1m_ef64` `VmHWM` during the 1,000-query search pass, and `rss_steady_hnsw_search_1m_ef64` steady-state resident HNSW footprint after 5s idle), plus the first-class overhead ratio:
$$\text{hnsw\_overhead\_ratio} = \frac{\text{rss\_steady\_hnsw\_search\_1m\_ef64} - \text{rss\_bare\_hnsw\_search\_1m\_ef64}}{10^6 \times 128 \times 4 / 1024^2}$$
**What is excluded:** Unrelated subsystem allocations.  
**Instrumentation:** `/proc/self/statm`, `/proc/self/status` (`VmHWM`), and `HnswIndex::memory_usage_bytes()`.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies `1.0 < hnsw_overhead_ratio <= 1.5` (`1.3363x` RSS delta, `1.2520x` raw structure bytes), proving HNSW graph links add $\le 50\%$ overhead above the $512\text{ MB}$ ($488.28\text{ MiB}$) raw vector storage.

---

### ivfpq_build_1m_128d / rss_bare_ivfpq_build_1m_128d / rss_peak_ivfpq_build_1m_128d / rss_steady_ivfpq_build_1m_128d

**What is measured:** End-to-end training and encoding time (`s` `p50` and `p99`) of `IvfPqIndex::build(vecs_1m, 1000000)` over $N = 1{,}000{,}000, D = 128$ `fp32` vectors (`nlist = 256`, `m_subspaces = 16` ($4 \times 4$ Cartesian subspaces), `nprobe = 12`, `rerank_k = 48`) across 4 timed builds after 1 warmup build, using 16 worker threads, along with three-stage RSS (`rss_bare`, `rss_peak`, `rss_steady`).  
**What is excluded:** Disk loading of `benchmarks/data/sift1m-like.bin`.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`), `/proc/self/statm`, and `/proc/self/status` (`VmHWM`).  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies `ivf_1m.size() == 1000000` and `p50 <= 30.0 s` (`0.4746 s` — **PASS**), proving all 1,000,000 vectors were assigned to `nlist = 256` normalized spherical coarse cells, radial norm-sorted, compacted into cell-contiguous vector storage, and packed into 32-vector transposed 2-bit Lloyd-Max FastScan blocks with exact `uint16_t` norm-residual bias.

---

### ivfpq_search_1m / rss_bare_ivfpq_search_1m / rss_peak_ivfpq_search_1m / rss_steady_ivfpq_search_1m

**What is measured:** Per-query latency (`ms/query` `p50` and `p99`) of single-threaded `IvfPqIndex::search(qv, 10, 12)` (`nlist = 256`, effective `nprobe = 176` (`11/16` of `nlist`) at $N > 200{,}000$, 3-tier radial norm-shell pruning scanning `19/32` of cells `0..27`, `7/16` of cells `28..95`, and `3/8` of cells `96..175`, 48-D/80-D `VPTEST` early exit, and `kRerankCap = 248` exact L2 re-ranking over cell-contiguous vectors) over all $1{,}000$ queries from `benchmarks/data/queries-1k.bin` against the $N = 1{,}000{,}000, D = 128$ `IvfPqIndex`, along with `mean_recall@10` and `worst_recall@10` against the 1M `FlatIndex` ground truth and three-stage RSS (`rss_bare`, `rss_peak`, `rss_steady`).  
**What is excluded:** `IvfPqIndex::build` and ground-truth `FlatIndex` computation.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`) per query around `ivf_1m.search(qv, 10, 12)`, `/proc/self/statm`, and `/proc/self/status` (`VmHWM`).  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies that all 1,000 queries return 10 valid neighbor IDs ($< 1{,}000{,}000$) with finite distances and evaluates against the $\le 0.5\text{ ms}$ latency and $\ge 0.90$ recall targets (`p50 = 0.4660 ms <= 0.50 ms`, `mean_recall@10 = 0.9051 >= 0.90` — **PASS**; resolved [`KNOWN_ISSUES.md#BUG-012`](KNOWN_ISSUES.md#bug-012)).

---

### dag_theoretical_min_direct

**What is measured:** Direct C++ indirect function-pointer call `fn_ptr(n % 100)` incrementing `std::atomic<uint64_t>` (`10,000` invocations per batch, 25 timed batches after 5 warmup batches).  
**What is excluded:** `DagScheduler` thread pool and `PipelineDefinition` validation.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`).  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies atomic accumulator equals exact closed-form sum $100 \times (100 \times 101 / 2) = 505{,}000$.

---

### dag_100node_diamond_noop

**What is measured:** Execution of a 100-node dependent layered diamond DAG (`10 layers x 10 nodes`) via `DagScheduler::execute_indexed_dag` on an 8-worker work-stealing pool, divided by 100 nodes (`us/node`, 25 timed runs after 5 warmup runs).  
**What is excluded:** CSR adjacency list construction.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`) and per-node `IndexedNodeTiming`.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies `executed_count == 1` for all 100 nodes and `child.start_ns >= parent.end_ns` across every directed edge.

---

### dag_1000_nodes_depth10_schedule

**What is measured:** Full scheduling and execution of a 1,000-node, depth-10 DAG (`10 layers x 100 nodes`) on an 8-worker `DagScheduler` (`ms`, 15 timed runs after 5 warmup runs).  
**What is excluded:** Graph construction.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`).  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies all 1,000 nodes execute once and `p50 <= 5.0 ms` (`0.0912 ms`).

---

### dag_single_task_median_us

**What is measured:** Wall-clock time (`us`) to execute a 1-node noop `PipelineDefinition` through `DagScheduler::execute(pipe_single)` with `worker_count = 1` (no scheduler optimization bypass), measured across 20 timed sub-batched samples (25 executions per sample) after 500 warmup executions.  
**What is excluded:** `PipelineDefinition` construction and `DagScheduler` thread pool construction.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around `sched_w1.execute(pipe_single)`.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies `r.has_value()`, `r->all_succeeded == true`, and `r->results.size() == 1` on every execution.

---

### dag_100_independent_serial / dag_100_independent_parallel_w4 / dag_100_independent_parallel_w8 / dag_100_independent_parallel_w12

**What is measured:** Wall-clock time (`ms` `p50` and `p99`) from `DagScheduler::execute(pipe_100_indep)` entry to return for a `PipelineDefinition` of 100 independent noop `Task` nodes (zero dependencies) executed with `worker_count` $= 1$ (`dag_100_independent_serial`), $4$ (`dag_100_independent_parallel_w4`), $8$ (`dag_100_independent_parallel_w8`, P-cores), and $12$ (`dag_100_independent_parallel_w12`, all physical cores), across 40 timed runs after 20 warmup runs per worker count.  
**What is excluded:** `PipelineDefinition` construction, `DagScheduler` pool creation, and post-run sanity verification.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around `s.execute(pipe_100_indep)`.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** After every run, verifies: (1) every task's status in `rep->results` is `TaskStatus::Completed`, (2) total task completions $== 100$ (no loss), (3) `task_exec_counts[i] == 1` for all $i \in [0, 99]$ (no duplication), and (4) sum of per-task `duration_ns` $\le 5 \times \text{serial\_100\_ms}$.

---

### dag_100_independent_overhead_w4 / dag_100_independent_overhead_w8 / dag_100_independent_overhead_w12

**What is measured:** Orchestration overhead ratio for 100 independent noop tasks at `worker_count` $N \in \{4, 8, 12\}$ relative to the theoretical minimum derived from `dag_single_task_median_us`:
$$\text{ideal\_100\_ms}[N] = \frac{\text{single\_task\_us} \times \lceil 100 / N \rceil}{1000}, \qquad \text{overhead}[N] = \frac{\text{parallel\_100\_ms}[N]}{\text{ideal\_100\_ms}[N]} - 1.0$$
**What is excluded:** Same exclusions as `dag_100_independent_parallel_w*`.  
**Instrumentation:** Derived directly from the `CLOCK_MONOTONIC` measurements of `dag_100_independent_parallel_w*` and `dag_single_task_median_us`.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies all 4 hard task-completion invariants (§2.2) hold and evaluates `overhead[N] <= 0.20` (`PASS` for noop tasks at $W \in \{4, 8, 12\}$: measured `0.0701` (`7.01%`) at $W=4$, `0.0996` (`9.96%`) at $W=8$, and `0.1455` (`14.55%`) at $W=12$ vs target $\le 0.20$ (`20%`); resolved [`KNOWN_ISSUES.md#BUG-009`](KNOWN_ISSUES.md#bug-009)).

---

### registry_metadata_lookup / registry_artifact_load_warm_mmap / registry_artifact_load_cold_nvme

**What is measured:**
- `registry_metadata_lookup`: Point lookup of `"basalt-resnet18:v1.2.0"` in a catalog of $1{,}000$ registered models (`us/lookup`).
- `registry_artifact_load_warm_mmap`: Zero-copy `mmap(2)` + full SHA-256 verification of a `64 MB` artifact resident in warm OS page cache (`ms` & `GB/s [warm page-cache]`).
- `registry_artifact_load_cold_nvme`: `fdatasync(2)` + `posix_fadvise(..., POSIX_FADV_DONTNEED)` followed by cold `mmap(2)` fault-in + full SHA-256 verification of the `64 MB` artifact from physical NVMe (`/var/tmp`, `ms` & `MB/s [cold NVMe]`).  
**What is excluded:** Initial `register_version` write of the 64 MB checkpoint.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`).  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies `sha256_hex` matches the expected 64-character hex digest and `artifact_size_bytes == 67,108,864 B` (`tensor_count == 64`).

---

### drift_psi_ks_100k_rows / fairness_audit_100k_rows

**What is measured:**
- `drift_psi_ks_100k_rows`: Two-sample Kolmogorov–Smirnov ($D_{\text{KS}}$) and 10-decile Population Stability Index ($\text{PSI}$) over $N = 100{,}000$ reference and production `double` scores (`ms` & `Mrows/s`).
- `fairness_audit_100k_rows`: Computing $\Delta_{\text{DP}}, DI, \Delta_{\text{EO}}, \Delta_{\text{EOD}}, \Delta_{\text{PP}}$ over $N = 100{,}000$ rows and 4 sensitive groups (`ms` & `Mrows/s`).  
**What is excluded:** Synthetic dataset generation.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`).  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies $|\text{metric} - \text{scalar\_reference}| < 10^{-12}$ and confusion matrix sum $== 100{,}000$.

---

### fairness_audit_1m_rows_10groups

**What is measured:** The timed interval includes only the metric computation (demographic parity difference, equalized odds difference, equal opportunity difference, predictive parity) over pre-grouped input. Grouping/sorting by sensitive attribute is excluded and measured separately (see `fairness_audit_1m_rows_10groups_with_grouping` below). Runs across 8 worker threads over $N = 1{,}000{,}000$ rows and 10 sensitive groups (`ms` `p50` and `p99`, 12 timed runs after 3 warmup runs).  
**What is excluded:** Partitioning/grouping or sorting of rows into contiguous per-group arrays by sensitive attribute, and dataset generation.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around `FairnessAuditor::audit(view1m)`.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies `total_rows == 1000000` and `p50 <= 500.0 ms` (`0.4142 ms`).

---

### fairness_audit_1m_rows_10groups_with_grouping

**What is measured:** End-to-end fairness audit from raw unpartitioned in-memory arrays over $N = 1{,}000{,}000$ rows and 10 sensitive groups (`ms` `p50` and `p99`, 12 timed runs after 3 warmup runs). The timed interval **includes**: (1) single-pass counting-sort histogramming and prefix-sum offset computation over the $1{,}000{,}000$ sensitive attribute IDs, (2) materializing 10 contiguous per-group partitions of `(prediction_scores, true_labels, sensitive_groups)` (`1M` rows), and (3) computing all four fairness metrics ($\Delta_{\text{DP}}, \Delta_{\text{EO}}, \Delta_{\text{EOD}}, \Delta_{\text{PP}}$) via `FairnessAuditor::audit`.  
**What is excluded:** Initial generation of the 1M-row raw input vectors in RAM prior to timer start.  
**Instrumentation:** `yantra::monotonic_time_ns()` (`CLOCK_MONOTONIC`) enclosing both the counting-sort partitioning pass and `FairnessAuditor::audit(grouped_view)`.  
**Hardware:** Intel Core i7-12700K (12C/20T), NVIDIA RTX 3070, Driver 550.163.01 / CUDA 12.4.131, Linux 6.12.113+deb13-amd64, 48 GiB DDR4, Samsung 970 EVO Plus 1TB NVMe.  
**Command to reproduce:**
```bash
./build-release/bench_yantra --json-out=results.json
```
**Sanity check:** Verifies `total_rows == 1000000`, `p50 <= 500.0 ms` (`3.2785 ms` — **PASS**), and asserts that all four fairness metrics match `fairness_audit_1m_rows_10groups` on the same 1M-row dataset to within $< 10^{-12}$, proving functional correctness of the grouping pass.

---

## 4. Three-Stage Process RSS Accounting (`rss_bare_process`, `rss_post_library_load`, `rss_peak_workload_hwm`)

- **`rss_bare_process`**: Sampled at `main()` entry in `/proc/self/statm` before any Yantra object construction (`> 0.5 MB`).
- **`rss_post_library_load`**: Sampled after `WalEngine`, `ModelRegistry`, `HnswIndex`, and `DagScheduler` initialization (`>= rss_bare_process` and `<= 50 MB` idle budget).
- **`rss_peak_workload_hwm`**: Sampled from `/proc/self/status` `VmHWM` after all 10K and 1M benchmarks complete + `1.0 s` settle (`>= rss_post_library_load`).
