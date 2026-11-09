# Yantra — Known Issues & Autonomous Bug-Hunting Log

This document records all concurrency, memory-safety, algorithmic, and build
issues discovered and resolved during the development and verification of
**Yantra** (`libyantra.so.1.0.0`), including the four 1M-scale and
granularity-dependent defects (`BUG-009` through `BUG-012`) uncovered during
the **Benchmark Completion Patch**
(see [`AUDIT_YANTRA_BENCHMARKS.md`](AUDIT_YANTRA_BENCHMARKS.md)) and resolved
in the **Library Correctness & Performance Fix Patch**.

---

## 1. Resolved Bugs Discovered During Verification

### BUG-001: Chase-Lev Work-Stealing Deque Single-Owner Violation (`src/dag.cpp`)
- **Severity**: Critical (Deadlock / Lost Wakeup)
- **Symptom**: `test_property` (`prop_dag_topological_and_cycle_rejection`)
  hung intermittently during multi-trial random 30-node DAG executions with
  `state.remaining_tasks` stuck above zero.
- **Root Cause**: `DagScheduler::execute` and `DagScheduler::execute_indexed_dag`
  previously pushed initial root tasks directly onto worker deques via
  `deques[q_idx % num_workers]->push_bottom(...)` from the calling thread and
  invoked `deques[0]->pop_bottom(...)` from the calling thread while worker `0`
  concurrently executed `push_bottom` and `pop_bottom` on `deques[0]`. In the
  Chase-Lev work-stealing algorithm, `bottom_` is accessed with
  `std::memory_order_relaxed` and strictly requires a single owner thread per
  deque. Concurrent `push_bottom`/`pop_bottom` calls corrupted `bottom_` and
  dropped ready tasks.
- **Resolution**: Added per-worker spin-guarded `InjectorQueue` rings for
  submissions originating from external caller threads (`worker_id == SIZE_MAX`).
  Only pool worker thread `w` (`0 <= w < num_workers`) ever invokes
  `deques[w]->push_bottom` or `deques[w]->pop_bottom`; external caller threads
  steal via `steal_top` and `InjectorQueue::pop`.

### BUG-002: Non-Lock-Free 24-Byte `std::atomic<WorkItem>` Lowering to `libatomic` (`src/dag.cpp`)
- **Severity**: High (Build Failure & Hidden Spinlock Contention)
- **Symptom**: Shared library link failure (`undefined reference to
  '__atomic_load'` and `'__atomic_store'`) when linking `libyantra.so.1.0.0`.
- **Root Cause**: `WorkItem` is a 24-byte struct (`fn`, `ctx`, `node_idx`).
  On x86-64, `std::atomic<T>` is only hardware lock-free up to 16 bytes
  (`CMPXCHG16B`), so `std::atomic<WorkItem>` emitted calls to `libatomic`'s
  locked hash table, violating the lock-free hot-path mandate.
- **Resolution**: Replaced `std::atomic<WorkItem>` in `ChaseLevDeque` with
  `AtomicWorkSlot`, storing three 8-byte `std::atomic` fields (`fn`, `ctx`,
  `node_idx`), each satisfying `std::atomic<...>::is_always_lock_free == true`
  and synchronized by the Chase-Lev `release` and `seq_cst` fences on `bottom_`
  and `top_`.

### BUG-003: Recursive `std::shared_mutex` Acquisition (`EDEADLK`) on First `HnswIndex::add` (`src/index.cpp`)
- **Severity**: High (Runtime Exception / Abort)
- **Symptom**: `test_unit` terminated with `std::system_error: Resource
  deadlock avoided` (`EDEADLK`) when inserting the initial element (`new_idx == 0`)
  into `HnswIndex` via `HnswIndex::add`.
- **Root Cause**: `HnswIndex::add` acquired an exclusive lock
  `std::unique_lock<std::shared_mutex> lock(impl_->mu)` and, in the `new_idx == 0`
  early-return branch, called `this->memory_usage_bytes()`, which attempted to
  acquire a shared lock `std::shared_lock<std::shared_mutex> lock(impl_->mu)`
  on the same non-recursive `std::shared_mutex`.
- **Resolution**: Computed the byte footprint directly from the locked `Impl`
  containers in `HnswIndex::add` instead of calling the public lock-acquiring
  `memory_usage_bytes()` method.

### BUG-004: Unconditional Reverse-Edge Eviction in `HnswIndex::add_batch` & Per-Cell LUT Recomputation in `IvfPqIndex::search` (`src/index.cpp`)
- **Severity**: Medium (Recall Degradation on High-Dimensional Uniform Noise)
- **Symptom**: `prop_vector_index_recall_bound` failed the `Recall@10 >= 0.95`
  assertion on 2,000 uniform random 64-D vectors.
- **Root Cause**: During parallel `HnswIndex::add_batch`, when an existing
  hub node `nbr` reached `m0` connections, worker threads overwrote
  `n_links[(u + j) % (m0 - m) + m] = u` without checking whether `u` was
  closer to `nbr` than the existing neighbor at that slot. In `IvfPqIndex`,
  the asymmetric distance computation (ADC) lookup table was recomputed inside
  the per-cell loop rather than once per query.
- **Resolution**: Added distance comparison before replacing reverse edges in
  `HnswIndex::add_batch`, expanded multi-landmark seeding in
  `search_layer0_beam`, and hoisted the `IvfPqIndex` ADC lookup table
  computation outside the inverted-list scan loop.

### BUG-005: `-DNDEBUG` Stripping `assert(...)` Expressions in `Release` Test Binaries (`tests/*.cpp`)
- **Severity**: Medium (Test Coverage & Unused-Variable Warnings under `-Werror`)
- **Symptom**: Building `test_unit`, `test_property`, and `test_stress` under
  `CMAKE_BUILD_TYPE=Release` (`-O3 -DNDEBUG -Werror`) triggered
  `-Werror=unused-variable` and `-Werror=unused-but-set-variable` because
  `assert(...)` expanded to `((void)0)`.
- **Resolution**: Added `#undef NDEBUG` before `#include <cassert>` in all test
  executables so test assertions and side-effecting expressions are always
  evaluated across `Debug`, `Release`, `ASan`, and `TSan` builds.

### BUG-006: `/tmp` Mounted as `tmpfs` Invalidating Durable `fdatasync(2)` and Cold NVMe Benchmarks (`bench/bench_yantra.cpp`)
- **Severity**: Medium (Benchmark Physical Plausibility)
- **Symptom**: Running durable `fdatasync(2)` (`wal_commit_nvme_fdatasync`) or
  cold `POSIX_FADV_DONTNEED` (`registry_artifact_load_cold_nvme`) against `/tmp`
  on Linux systems where `/tmp` is mounted as `tmpfs` bypasses the physical
  NVMe block device and measures RAM speed, violating Check 1.1 of
  `YANTRA_BENCHMARK_SPEC.md`.
- **Resolution**: Separated `ram_dir` (`/tmp/yantra_bench_ram`) for page-cache
  throughput tests from `disk_dir` (`/var/tmp/yantra_bench_disk` on physical
  `ext4` NVMe storage) for durable `fdatasync(2)` and cold `POSIX_FADV_DONTNEED`
  benchmarks.

### BUG-007: Standalone `std::atomic_thread_fence` Rejected by GCC 14 `-Werror=tsan` (`src/dag.cpp`)
- **Severity**: High (ThreadSanitizer Build & Instrumentation)
- **Symptom**: Compiling `src/dag.cpp` under the `release-tsan` preset
  (`-fsanitize=thread -Werror`) failed with `error: 'atomic_thread_fence' is
  not supported with '-fsanitize=thread' [-Werror=tsan]`, and `pop_bottom`
  decrementing `bottom_` on empty deques caused heavy cache-line invalidation.
- **Root Cause**: GCC 14's ThreadSanitizer requires memory ordering to be
  attached directly to atomic load/store operations (`bottom_`, `top_`, and
  `AtomicWorkSlot`) rather than standalone `std::atomic_thread_fence` calls.
- **Resolution**: Replaced standalone fences in `ChaseLevDeque` with
  variable-attached `std::memory_order_release` and `std::memory_order_seq_cst`
  operations on `bottom_` and `top_`, and added an early empty-check
  (`cur_b <= cur_t`) in `pop_bottom`.

### BUG-008: Unsynchronized Forward-Link Write Racing with Reverse-Link Insertion in `HnswIndex::add_batch` (`src/index.cpp`)
- **Severity**: High (Data Race Detected by ThreadSanitizer)
- **Symptom**: ThreadSanitizer (`release-tsan`) reported a 4-byte write/write
  data race in `HnswIndex::add_batch` between a worker thread writing node `u`'s
  forward neighbor links (`u_links[u_cnt] = cp.node`) and a higher-numbered
  worker thread `u2 > u` simultaneously appending a reverse link to `u`'s
  adjacency list under `lock_node(u)`.
- **Root Cause**: `u_links` was populated before acquiring `lock_node(u32)`,
  while `u` was already visible as a candidate (`v < u2`) to threads processing
  indices `u2 > u`.
- **Resolution**: Assembled node `u`'s forward links into a stack-local buffer
  `local_u_links` and merged them into `impl_->layer0_links` strictly inside
  `lock_node(u32)`.

---

<a id="BUG-009"></a>
### BUG-009: `DagScheduler::execute` Per-Call Hash-Map Allocation & `results_mu` Contention on 100 Independent Noop Tasks (`src/dag.cpp`, `include/yantra/dag.hpp`) — **RESOLVED**
- **Severity**: Medium (Sub-Microsecond Noop-Task Orchestration Overhead)
- **Status**: **RESOLVED** (Regression test: [`tests/test_dag_overhead.cpp`](tests/test_dag_overhead.cpp))
- **Pre-Fix vs. Post-Fix Metrics**:
  - `dag_100_independent_overhead_w4`: `3.2184` (`321.84%`) $\to$ **`0.0622` (`6.22% <= 20%`, PASS)**
  - `dag_100_independent_overhead_w8` (primary): `7.8823` (`788.23%`) $\to$ **`0.0896` (`8.96% <= 20%`, PASS)**
  - `dag_100_independent_overhead_w12`: `13.2344` (`1323.44%`) $\to$ **`0.1335` (`13.35% <= 20%`, PASS)**
- **Root Cause**:
  1. **Per-Execute Dynamic Hash-Map Construction**: Every `DagScheduler::execute`
     invocation dynamically constructed four `std::unordered_map<std::string, ...>`
     tables over all $n$ task IDs (`validate()`'s `id_to_idx`, `execute()`'s
     `id_to_idx`, `report.results`, and `st->committed_map`), consuming `~45 us`
     of serial heap allocation on the calling thread before worker notification.
  2. **Global `st->results_mu` Contention**: Inside `run_node_cb`, every worker
     acquired the single shared mutex `st->results_mu` twice per task (`200`
     mutex acquisitions across 100 noop tasks), serializing workers on a single
     cache line.
- **Resolution**:
  - Added cached `CompiledPipeline` on `PipelineDefinition` (`compiled_cache_`)
    with pre-allocated `TaskMetadata` array and flat `TaskResultMap` view
    (`include/yantra/dag.hpp`, `src/dag.cpp`) so `DagScheduler::execute` performs
    zero `std::unordered_map` allocations on repeat executions.
  - Eliminated `st->results_mu` in favor of lock-free per-slot writes into
    `entries_ptr[idx]` synchronized by `meta[idx].status` (`memory_order_release`
    / `memory_order_acquire`), plus a lock-free chunked batch execution path
    (`BatchJob`, `batch_epoch`, `sleeping_workers`) for independent pipelines.

---

<a id="BUG-010"></a>
### BUG-010: `HnswIndex::add_batch` $O(N \cdot B)$ Strided Random DRAM Scan Exceeding 60 s Build Target at 1M Scale (`src/index.cpp`) — **RESOLVED**
- **Severity**: High (1M Index Build Latency $1.997\times$ Above Spec Target)
- **Status**: **RESOLVED** (Diagnostic: [`tests/diag_hnsw_1m.cpp`](tests/diag_hnsw_1m.cpp))
- **Pre-Fix vs. Post-Fix Metric**:
  - `hnsw_build_1m_128d`: `119.8073 s` (`8,347 vecs/s`) $\to$ **`5.2580 s` (`190,186 vecs/s <= 60.0 s`, PASS — $22.8\times$ faster)**
- **Root Cause**:
  - `HnswIndex::add_batch` evaluated `sample_budget = 6,000` prime-strided
    candidates per inserted node across the `512 MB` `impl_->vectors` buffer
    ($20.5\times$ larger than the `25 MB` L3 cache), causing $\approx 6 \times 10^9$
    random DRAM `l2_sq_f32` evaluations (`3.07 TB` of uncached reads).
- **Resolution**:
  - Replaced the prime-strided random scan in `HnswIndex::add_batch` with
    256-cell spherical coarse partitioning (`normalize_centroid`), transposed
    32-vector 2-bit Lloyd-Max FastScan blocks (`score_fastscan_block_avx2`), and
    cell-grouped parallel construction across 16 worker threads so each worker's
    probed FastScan blocks remain resident in L2/L3 cache (`5.26 s` total 1M
    build time).

---

<a id="BUG-011"></a>
### BUG-011: `HnswIndex` Recall@10 Degradation at 1M Scale (`0.2287` vs. `>= 0.95` Target) & Missing Exponential Layer Assignment (`src/index.cpp`, `include/yantra/index.hpp`) — **RESOLVED**
- **Severity**: Critical (Approximate Nearest-Neighbor Recall & Graph Connectivity at 1M Scale)
- **Status**: **RESOLVED** (Regression test: [`tests/test_hnsw_connectivity.cpp`](tests/test_hnsw_connectivity.cpp), Diagnostic: [`tests/diag_hnsw_1m.cpp`](tests/diag_hnsw_1m.cpp))
- **Pre-Fix vs. Post-Fix Metric**:
  - `hnsw_search_1m_ef64`: `p50 = 0.6068 ms`, `mean Recall@10 = 0.2287` (`worst = 0.0000`) $\to$ **`p50 = 1.7945 ms <= 2.0 ms`, `mean Recall@10 = 0.9869 >= 0.95`, `worst Recall@10 = 0.8000 >= 0.80` (PASS)**
  - Layer-1 fraction (`M = 16`): `0.0000` $\to$ **`0.062212` (within `0.46%` of theoretical $1/16 = 0.062500$)**
  - Layer-0 reachability from entry point at 1M: **`1,000,000 / 1,000,000` (`100.0% >= 99.9%`)**
- **Root Cause**:
  1. **Missing Exponential Level Assignment**: Nodes were not assigned levels via
     $\lfloor -\ln(\text{unif}) \cdot m_L \rfloor$ ($m_L = 1/\ln(M)$), leaving
     `max_level() == 0`.
  2. **Sparse Strided Graph Construction**: Sampling `6,000 / u` predecessors
     covered only `0.6%` of the graph at $N = 1{,}000{,}000$, wiring nodes to
     distant vectors instead of true local neighbors.
- **Resolution**:
  - Implemented exponential level assignment `compute_hnsw_level(i, m)` using
    `static_cast<size_t>(-std::log(u) * (1.0 / std::log(M)))` and persisted
    per-node levels in the `.yhnsw` v2 format (`HnswIndex::max_level()`,
    `HnswIndex::node_level()`).
  - Combined transposed 32-vector AVX2 `VPSHUFB` FastScan basin seeding (with
    exact 2-byte per-vector norm bias `bias_u16` at block offset `0`, radial
    in-cell norm ordering, and 48-D / 80-D `VPTEST` early exit) with Layer-0
    beam refinement and BFS tree back-edge strong connectivity enforcement.

---

<a id="BUG-012"></a>
### BUG-012: `IvfPqIndex::search` Latency (`8.35 ms` vs. `<= 0.50 ms`) and Recall (`0.6652` vs. `>= 0.90`) at 1M Scale (`src/index.cpp`) — **RESOLVED**
- **Severity**: High (1M IVF-PQ Search Latency & Recall)
- **Status**: **RESOLVED** (Regression test: [`tests/test_ivfpq_recall.cpp`](tests/test_ivfpq_recall.cpp))
- **Pre-Fix vs. Post-Fix Metric**:
  - `ivfpq_search_1m`: `p50 = 8.3488 ms/query`, `mean Recall@10 = 0.6652` (`worst = 0.1000`) $\to$ **`p50 = 0.4719 ms/query <= 0.50 ms`, `mean Recall@10 = 0.9051 >= 0.90`, `worst Recall@10 = 0.4000` (PASS)**
  - `ivfpq_build_1m_128d`: **`0.4737 s <= 30.0 s` (PASS)**
- **Root Cause**:
  1. **Scalar Scan + 750K-Element `std::partial_sort`**: `IvfPqIndex::search`
     scanned `750,000` vectors per query with a scalar 16-subspace loop and
     pushed all `750,000` `(dist, id)` pairs into a heap-allocated
     `std::vector<Cand>` followed by `std::partial_sort` (`~6 MB` per query).
  2. **Unbiased Clipping Quantizer**: Fixed thresholds `{-0.75, -0.25, 0.25, 0.75}`
     without per-vector squared-norm residual compensation distorted distances on
     high-norm coordinates.
- **Resolution**:
  - Trained `nlist = 256` coarse centroids using 4 iterations of spherical
    k-means (`normalize_coarse_centroid`) and sorted each inverted list in
    ascending order of vector squared norm $\|v\|^2$.
  - Encoded inverted lists into 32-vector transposed 2-bit Lloyd-Max FastScan
    blocks (`{-1.5104, -0.4528, +0.4528, +1.5104}`) with an exact 2-byte
    per-vector norm-residual bias `bias_u16` at block offset `0` (`kFastScanScale = 3.2f`).
  - Vectorized the probe scan with AVX2 `VPSHUFB` (`score_fastscan_block_avx2`)
    with 48-D and 80-D `VPTEST` early exit against a bounded max-heap of
    `kRerankCap = 248` candidates (`nprobe = 176` coarse cells with 3-tier
    radial norm-shell pruning), followed by software-prefetched exact L2
    re-ranking.

---

## 2. Current Architectural Notes & Hardware Dispatch

1. **HNSW `.yhnsw` v2 Binary Persistence Layout**:
   - The `.yhnsw` v2 format stores the 64-byte CRC32C-verified header, contiguous
     `fp32` vectors, Layer-0 adjacency lists, neighbor counts, external IDs,
     per-node exponential levels, coarse routing centroids, and transposed
     FastScan blocks in a flat, deterministic layout directly mapped via
     `mmap(2)` (`HnswIndex::load_mmap`) in `0.42 ms` with a `1.3406x` memory
     overhead ratio ($\le 1.50\times$).
2. **Declarative JSON Pipeline Parser Scope**:
   - `yantra::PipelineDefinition::from_json` implements a zero-dependency
     declarative parser for the Yantra pipeline schema (`tasks`, `id`, `cpu_cores`,
     `gpu_vram_bytes`, `max_retries`, `timeout_ns`, `dependencies`).
3. **Host AVX-512 vs. AVX2 Hardware Dispatch**:
   - Both AVX-512 and AVX2+FMA SIMD kernels are compiled into `libyantra.so` via
     GCC function multiversioning (`__attribute__((target(...)))`) and selected at
     runtime via `__builtin_cpu_supports`.

---

## 3. Found-by-Stack-Integration Bugs (`2026-11-01`, Yantra `v1.0.0` + Sutra `v1.0.0` + Darshana `v1.0.0`)

During full-stack composability verification (`stack-integration/stack_integration_test`), the following 3 cross-library seams in `Yantra` were identified and resolved:

1. **CHECK-1 / CHECK-6 (`ModelRegistry` File-Based Registration & Persistent Catalog)**:
   - **Symptom**: `ModelRegistry` only exposed `register_version(..., std::span<const std::byte>)` requiring callers to manually `mmap` multi-gigabyte GGUF files, and kept `model_versions` / `tag_index` in-memory without persisting `catalog.txt` across `ModelRegistry::open` sessions.
   - **Fix Applied**: Added `ModelRegistry::register_file(...)` (which `mmap`s the source file via RAII `ScopedMmap`, hashes it with BLAKE3 and SHA-256, writes the content-addressed artifact to `cas/`, and unmaps the source file immediately) and persistent `catalog.txt` save/load in `include/yantra/registry.hpp` and `src/registry.cpp`.
2. **CHECK-2 / CHECK-5 (`RunRecord` Per-Step Metric History Retention)**:
   - **Symptom**: `ExperimentStore::get_run()` and `RunRecord` only stored `MetricSummary` (`min_val`, `max_val`, `last_val`, `count`), discarding per-turn values for turns `1..9` needed for round-trip verification (`CHECK-2`) and Darshana per-turn latency plotting (`CHECK-5`).
   - **Fix Applied**: Added `MetricPoint` (`step`, `value`, `timestamp_ns`) and `std::unordered_map<std::string, std::vector<MetricPoint>> metric_history` to `RunRecord` in `include/yantra/track.hpp` and populated it during both live `Run::log_metric` and `WalEngine::recover` replay in `src/track.cpp`.
3. **CHECK-3 (`WalEngine` Lifecycle & Parameter Record Durability)**:
   - **Symptom**: `Run::start()`, `Run::log_param()`, and `Run::finish()` mutated in-memory `RunRecord` state and `runs_meta.txt` without writing `WalRecord` entries to `wal.bin`, resulting in `40` WAL records instead of the expected `44` (`10 turns * 4 metrics + run_start + run_end + 2 params`).
   - **Fix Applied**: Added `WalValueKind::RunStart = 3`, `WalValueKind::RunEnd = 4`, and `WalValueKind::Param = 5` in `include/yantra/track.hpp` and updated `Run::start()`, `Run::log_param()`, and `Run::finish()` (with atomic `finished` exchange to prevent duplicate `RunEnd` records in `~Run()`) in `src/track.cpp`.


---

## Issue 12: Found-by-Sphinx Phase 4 — `ModelRegistry::register_file` Zero-Copy Hardlink for Multi-GB GGUF Weights & `index_manifest` Catalog Persistence (`2026-11-01`)

- **Symptom**: Registering multi-GB GGUF model files (`Qwen2.5-7B-Instruct-Q4_K_M.gguf` at 4.4 GB and `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` at 1.1 GB) via `ModelRegistry::register_file` duplicated 5.5 GB of bytes into `cas/`, and `ModelRegistry::index_manifest` did not persist `catalog.txt` immediately.
- **Fix Applied**: Updated `ModelRegistry::register_file` in `src/registry.cpp` to pre-link large (`>= 64 MiB`) model artifacts into `cas/<prefix>/<blake3_hex>` via `std::filesystem::create_hard_link` (with symlink/parallel-write fallback) while still computing full BLAKE3 and SHA-256 digests, and added `impl_->save_catalog_locked()` to `ModelRegistry::index_manifest`.
