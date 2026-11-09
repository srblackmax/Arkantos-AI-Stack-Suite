# Yantra — Architectural Specification & Operations Guide

**Yantra** (`yantra::inline v1`, CMake target `Yantra::Yantra`, shared library
`libyantra.so.1.0.0`) provides the production supervision, tracking, indexing,
orchestration, and auditing layer for native C++23 machine learning services.

---

## 1. Reference Hardware & Toolchain Requirements

| Component | Specification |
| :--- | :--- |
| **CPU** | 12th Gen Intel Core i7-12700K (8 P-cores + 4 E-cores / 20 threads, AVX2+FMA, SHA-NI, SSE4.2 CRC32C) |
| **GPU** | NVIDIA GeForce RTX 3070 (8,192 MiB GDDR6, Compute Capability `sm_86`, CUDA 12.4) |
| **Memory** | 48 GiB Dual-Channel DDR4 (`~38.4 GB/s` sustained stream bandwidth) |
| **Storage** | Samsung SSD 970 EVO Plus 1 TB NVMe (`ext4`, `~3.3 GB/s` sequential read) |
| **OS & Kernel** | Debian GNU/Linux 13 (Trixie), Linux `6.12.113+deb13-amd64` |
| **Compilers** | GCC `14.2.0` (`-std=c++23`), NVCC `12.4.131` with host compiler `g++-13` (`13.3.0`) |
| **Build System** | CMake `>= 3.25` + Ninja (`CMakePresets.json`) |

---

## 2. Five Core Modules & Rival Python Systems

### 2.1. Experiment Tracking (`include/yantra/track.hpp`, rival: MLflow Tracking)
- **`yantra::WalEngine`**: Append-only binary Write-Ahead Log (`WalFileHeader` +
  64-byte cache-line-aligned `WalRecord` entries) protected by hardware SSE4.2
  `_mm_crc32_u64` / `_mm_crc32_u8` checksums. Supports both high-throughput OS
  page-cache batch ingestion (`sync_to_disk = false`) and durable NVMe group
  commit via `fdatasync(2)` (`sync_to_disk = true`).
- **`yantra::Run`**: RAII scoped run handle. Parameters (`log_param`) are
  permitted in `Created` state and locked as immutable once the run transitions
  to `Running` (returning `TrackingErrorCode::ImmutableParameterViolation` on
  any post-start mutation). Artifacts (`log_artifact`) are content-addressed
  using a native 7-round **BLAKE3** cryptographic hash and deduplicated in
  `cas/<blake3_hex>.bin`.
- **`yantra::ExperimentStore`**: Manages concurrent runs, compacts WAL segments
  into a binary columnar store (`columnar.bin`) with per-run `[min, max]` zone
  maps (`compact()`), and evaluates multi-predicate range queries over
  parameters, tags, and metric bounds (`query()`) in `< 0.5 ms` over 10,000
  runs.

### 2.2. Model Registry & Versioning (`include/yantra/registry.hpp`, rival: MLflow Registry / Kubeflow)
- **`yantra::ModelRegistry`**: Stores ordered, immutable `ModelManifest`
  sequences per model family (`model_name:vX.Y.Z`).
- **Transactional Registration & Optimistic Concurrency**: `register_version`
  and `register_version_cas` write artifact payloads to temporary staging files,
  compute 256-bit **BLAKE3** and hardware **SHA-256** (Intel SHA-NI
  `_mm_sha256rnds2_epu32`) digests, verify lineage DAG acyclicity, and atomically
  rename blobs into the content-addressable store.
- **Audited Stage Transitions**: Explicit transitions across `Staging`,
  `Production`, and `Archived` require a non-empty justification string
  (`RegistryErrorCode::EmptyJustification`) and reject direct
  `Archived -> Production` promotion (`RegistryErrorCode::InvalidStageTransition`).
- **Warm & Cold Artifact Verification**: Provides both zero-copy `mmap(2)` warm
  page-cache verification (`load_artifact_warm_mmap`) and cold NVMe block-device
  verification (`load_artifact_cold_nvme` using `POSIX_FADV_DONTNEED`).

### 2.3. Workflow Orchestration (`include/yantra/dag.hpp`, rival: Airflow / Prefect / Dagster)
- **`yantra::PipelineDefinition`**: Declarative C++ builder and zero-dependency
  JSON serializer/deserializer (`to_json` / `from_json`) for multi-tenant DAG
  definitions.
- **`yantra::DagScheduler`**:
  - Validates topological ordering via Kahn's algorithm (`validate`), rejecting
    cycles (`DagErrorCode::CycleDetected`), missing dependencies
    (`DagErrorCode::MissingDependency`), and unsatisfiable CPU/VRAM requests
    (`DagErrorCode::ResourceDeadlock`) before any task executes.
  - Schedules ready tasks across a **Chase-Lev lock-free work-stealing thread
    pool** (`ChaseLevDeque` with 8-byte lock-free `AtomicWorkSlot` fields and
    per-worker `InjectorQueue` rings for external callers).
  - Enforces conditional edge predicates, exponential-backoff retries
    (`RetryPolicy`), and per-task wall-clock timeouts.

### 2.4. Vector Index & Retrieval (`include/yantra/index.hpp`, rival: FAISS)
- **`yantra::FlatIndex`**: Exact brute-force L2 baseline with runtime-dispatched
  **AVX-512** (`_mm512_fmadd_ps`) and **AVX2+FMA** (`_mm256_fmadd_ps`) SIMD
  kernels plus an exact `double`-precision ground-truth evaluator
  (`search_exact_fp64`).
- **`yantra::HnswIndex`**: Navigable small-world graph index supporting parallel
  batch construction (`add_batch`), incremental insertion (`add`), pre/post
  selective metadata filtering (`search_filtered`), and zero-copy `mmap(2)`
  persistence (`save` / `load_mmap`) with `<= 1.5x` raw vector memory overhead.
- **`yantra::IvfPqIndex`**: Inverted File with 256-centroid ($4^4$) Cartesian
  product quantization and fast asymmetric distance computation (ADC) lookup
  tables followed by SIMD exact reranking.
- **`yantra::GpuIndex`** (`include/yantra/gpu.hpp`): Pimpl-isolated CUDA backend
  compiled only when `YANTRA_ENABLE_CUDA=ON`. Manages all device memory through
   a stream-ordered `cudaMemPool_t` (`cudaMallocAsync` / `cudaFreeAsync`) with
  zero `cudaMalloc` calls on the per-query path.

### 2.5. Bias, Fairness & Drift Auditing (`include/yantra/fairness.hpp`, rival: Fairlearn)
- **`yantra::DatasetView`**: Zero-copy `std::span` view over contiguous
  prediction scores, binary ground-truth labels, and sensitive group IDs.
- **`yantra::FairnessAuditor`**: Single-pass multi-threaded confusion-matrix
  reduction computing Demographic Parity Difference ($\Delta_{\text{DP}}$),
  Disparate Impact Ratio ($\text{DI}$), Equalized Odds Difference
  ($\Delta_{\text{EOdds}}$), Equal Opportunity Difference
  ($\Delta_{\text{EOpp}}$), and Predictive Parity Difference. Includes
  coordinate-search post-processing threshold optimization
  (`optimize_thresholds`) that converges in $\le 100$ iterations.
- **`yantra::DriftDetector`**: Computes Population Stability Index (PSI) across
  quantile buckets and the exact two-sample Kolmogorov–Smirnov statistic
  ($D_{\text{KS}} = \sup_x |F_{\text{ref}}(x) - F_{\text{prod}}(x)|$).

---

## 3. Concurrency Model & Error Taxonomy

- **Error Handling**: Every fallible public API is marked `[[nodiscard]]` and
  returns `yantra::Result<T>` (`std::expected<T, yantra::Error>`).
  `yantra::Error` wraps `std::variant<TrackingError, RegistryError, DagError,
  IndexError, FairnessError>` and automatically emits structured JSON-lines
  diagnostics to `yantra::Logger` when constructed at `ERROR` severity.
- **Lock-Free & Reader-Writer Concurrency**:
  - `WalEngine`: Concurrent `pwrite(2)` writes with atomic sequence reservation
    (`fetch_add`).
  - `ModelRegistry`: `std::shared_mutex` reader-writer concurrency with
    optimistic version checks (`expected_prev_version`).
  - `DagScheduler`: Per-worker single-owner `ChaseLevDeque` (`push_bottom` /
    `pop_bottom` by owner, `steal_top` by thieves) plus per-worker
    `InjectorQueue` for external callers.
  - `HnswIndex` / `IvfPqIndex` / `FlatIndex`: `std::shared_lock` for concurrent
    searchers and `std::unique_lock` for incremental mutations.

---

## 4. Building, Testing & Reproducing Benchmarks

```bash
# 1. Configure, build, and run the Release test suite
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure

# 2. Run Sanitizer (ASan+UBSan and TSan) and CUDA / Headless presets
cmake --preset release-asan && cmake --build --preset release-asan && ctest --preset release-asan
cmake --preset release-tsan && cmake --build --preset release-tsan && ctest --preset release-tsan
cmake --preset release-cuda && cmake --build --preset release-cuda && ctest --preset release-cuda
cmake --preset release-headless && cmake --build --preset release-headless && ctest --preset release-headless

# 3. Execute the audited benchmark suite and CI regression gate
./build-release/bench_yantra --json-out=results.json
cmake --build build-release --target bench-ci
```
