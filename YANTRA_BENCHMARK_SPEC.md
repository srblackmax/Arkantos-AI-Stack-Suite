# Yantra (Step 8 — MLOps) Benchmark Methodology & Integrity Specification (`YANTRA_BENCHMARK_SPEC.md`)

**Target Library:** Step 8 — **Yantra** (C++23 MLOps: Experiment Tracking WAL, Model Registry, HNSW Feature/Embedding Store, DAG Pipeline Orchestrator, and Drift/Fairness Auditor)  
**Status:** Pre-Implementation Binding Specification (Phase 3 of Ecosystem Benchmark Integrity Audit, `2026-11-01`)  

---

## 1. Binding Authoring Rules (Enforced from Commit 0)

1. **`BENCHMARKS_METHODOLOGY.md` First:** `Yantra/BENCHMARKS_METHODOLOGY.md` MUST be committed in the initial repository skeleton **before** any code in `Yantra/bench/` is written.
2. **No Unqualified Units:**
   - Every latency metric must report both **`p50` (median)** and **`p99` (tail)** across $\ge 10$ timed runs after explicit warmup.
   - Every throughput metric must state its exact unit and domain qualifier (`records/s [RAM append]`, `records/s [NVMe fdatasync]`, `nodes/s [noop-DAG orchestration]`, `queries/s [CPU AVX2]`).
   - Bare `"FPS"` is strictly forbidden in Yantra (Yantra has no swapchain presentation).
3. **Mandatory Post-Measurement Sanity Checks:** Every benchmark in `bench_yantra.cpp` must validate its computed output or on-disk artifact after the timed region closes and emit `"sanity_ok": true` in `results.json`. If any sanity check fails, `bench_yantra` must exit with non-zero status (`EXIT_FAILURE`).
4. **Automated Regression Target (`bench-ci`):** `Yantra/CMakeLists.txt` must define a `bench-ci` custom target that runs `bench_yantra --json-out=results.json --check-baseline=benchmarks/baseline.json` and fails if any sanity check fails or any `p50` metric regresses beyond tolerance (`20%`).

---

## 2. Pre-Specified Yantra Benchmark Metrics, Bounds & Sanity Checks

### Metric Group 1: Experiment Tracking Write-Ahead Log (`yantra::WalEngine`)

| Metric ID | Unit | Timed Interval Definition | Disclosed Exclusions | Physical Plausibility Bound (Reference Workstation) | Mandatory Post-Measurement Sanity Check |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `wal_append_pagecache` | `us/batch` (`p50`/`p99`) & `MB/s [page-cache write(2)]` | Serialization of $1{,}000$ metric records ($64\text{ B}$ each) + CRC32C checksum + `write(2)` syscall into OS page cache. | Excludes `fdatasync(2)` NVMe flush (explicitly labeled `[page-cache]`). | $\le 12{,}000\text{ MB/s}$ (single-core DDR4 copy + syscall ceiling on i7-12700K). | Verify `stat(2)` file byte size equals `header_bytes + N * record_bytes`. |
| `wal_commit_nvme_fdatasync` | `ms/commit` (`p50`/`p99`) & `MB/s [durable NVMe fdatasync]` | Serialization + CRC32C + `write(2)` + `fdatasync(2)` durable flush to `Samsung SSD 970 EVO Plus 1TB` NVMe (`nvme0n1`). | None (end-to-end durable persistence). | $\le 3{,}300\text{ MB/s}$ sequential write; $\ge 25\ \mu\text{s}$ per `fdatasync(2)` commit on Samsung 970 EVO Plus NVMe. | Re-open the WAL file from disk, verify every record's CRC32C checksum, and assert recovered record count == $N$ and last sequence ID == $N-1$. |

---

### Metric Group 2: HNSW Vector Feature Store (`yantra::HnswIndex`)

> **Binding Rule:** Reporting HNSW search latency or QPS without simultaneously reporting **`Recall@k`** against exact brute-force ground truth and disclosing `(N, D, M, efConstruction, efSearch)` is a Check 1.2 / Check 1.6 failure.

| Metric ID | Unit | Timed Interval Definition | Disclosed Exclusions | Physical Plausibility Bound | Mandatory Post-Measurement Sanity Check |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `hnsw_build_10k_128d` | `ms` (`p50`/`p99`) & `vecs/s` | Graph construction over $N = 10{,}000$, $D = 128$ `fp32` vectors (`M = 16`, `efConstruction = 200`). | Excludes synthetic vector RNG generation. | $> 10\text{ ms}$ (at least $10^4 \times 200$ distance evaluations). | Verify `index.size() == 10000` and non-empty neighbor lists at layer 0. |
| `hnsw_search_top10_ef64` | `us/query` (`p50`/`p99`), `queries/s`, and **`Recall@10`** | Single-query top-10 approximate nearest-neighbor search (`efSearch = 64`) on $N = 10{,}000, D = 128$. | Excludes index build time. | Latency bounded below by $L_1/L_2$ cache traversal of visited candidate set ($\ge 2\ \mu\text{s/query}$). | Compute exact brute-force `fp64` L2/cosine top-10 ground truth over 100 queries and assert measured **`Recall@10 >= 0.95`**. |

---

### Metric Group 3: DAG Pipeline Scheduler Overhead (`yantra::DagScheduler`)

> **Binding Rule (§4.6):** Any benchmark measuring DAG orchestration with lightweight/noop node functions MUST be labeled `[noop-node orchestration]` and paired with a **Theoretical Minimum C++ Direct Dispatch Baseline** (`std::atomic` counter + indirect function pointer call).

| Metric ID | Unit | Timed Interval Definition | Disclosed Exclusions | Physical Plausibility Bound | Mandatory Post-Measurement Sanity Check |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `dag_theoretical_min_direct` | `us/node` (`p50`/`p99`) | Direct loop executing indirect function pointer `noop_node(id)` + `std::atomic<uint64_t>::fetch_add(1)` without DAG topological scheduler. | No thread-pool handoff or dependency resolution. | `~0.005–0.050 us/node` on i7-12700K (`5.1 GHz`). | Verify final atomic accumulator equals exact node count. |
| `dag_100node_diamond_noop` | `us/node` (`p50`/`p99`) & `nodes/s [noop-node orchestration]` | End-to-end execution of a 100-node layered diamond DAG on Yantra's work-stealing thread pool, divided by 100 nodes. | Uses zero-wait `noop_node` to isolate scheduler + atomic in-degree resolution overhead. | Must be slower than `dag_theoretical_min_direct` (`ratio > 1.0x`). | Verify every child node's recorded start timestamp $\ge$ all parent nodes' completion timestamps (topological order invariant) and all 100 nodes executed exactly once. |

---

### Metric Group 4: Model Registry Lookup & Artifact Integrity Verification (`yantra::ModelRegistry`)

| Metric ID | Unit | Timed Interval Definition | Disclosed Exclusions | Physical Plausibility Bound | Mandatory Post-Measurement Sanity Check |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `registry_metadata_lookup` | `us/lookup` (`p50`/`p99`) | Resolving a model name + semantic version tag (`"basalt-resnet18:v1.2.0"`) to its manifest record in an index of $1{,}000$ registered models. | In-memory hash/B-tree index lookup only; excludes model weight file I/O. | $\ge 0.02\ \mu\text{s}$ (L1/L2 hash probe floor). | Verify returned `ModelManifest::sha256_hex` and `artifact_size_bytes` match registered ground truth. |
| `registry_artifact_load_warm_mmap` | `ms` (`p50`/`p99`) & `GB/s [warm page-cache]` | `mmap(2)` + full SHA-256 verification over a `64 MB` model checkpoint resident in OS page cache. | Warm OS page cache (no physical NVMe read; explicitly labeled `[warm page-cache]`). | $\le 40\text{ GB/s}$ (DDR4 dual-channel read bandwidth + SHA-256 instruction throughput on i7-12700K). | Verify computed SHA-256 digest equals expected 64-char hex digest and parsed tensor count $> 0$. |
| `registry_artifact_load_cold_nvme` | `ms` (`p50`/`p99`) & `MB/s [cold NVMe]` | `posix_fadvise(..., POSIX_FADV_DONTNEED)` followed by `read(2)` + SHA-256 verification of a `64 MB` model artifact from NVMe. | None (cold disk read + cryptographic hash verification). | $\le 3{,}500\text{ MB/s}$ (Samsung 970 EVO Plus sequential read spec). | Verify computed SHA-256 digest equals expected 64-char hex digest. |

---

### Metric Group 5: Data Drift & Algorithmic Fairness Audit (`yantra::FairnessAuditor` / `yantra::DriftDetector`)

| Metric ID | Unit | Timed Interval Definition | Disclosed Exclusions | Physical Plausibility Bound | Mandatory Post-Measurement Sanity Check |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `drift_psi_ks_100k_rows` | `ms` (`p50`/`p99`) & `Mrows/s` | Computing Population Stability Index (10 quantile bins) and exact two-sample Kolmogorov–Smirnov statistic $D_{\text{KS}}$ over $N = 100{,}000$ reference vs. production `fp64` scores. | Excludes synthetic array generation. | Bounded by `std::sort` + linear scan over $2 \times 10^5$ `fp64` elements ($\ge 0.5\text{ ms}$ on single core). | Assert $0 < D_{\text{KS}} \le 1.0$, $\text{PSI} > 0$, and $|\text{SIMD} - \text{scalar\_fp64}| < 10^{-12}$. |
| `fairness_audit_100k_rows` | `ms` (`p50`/`p99`) & `Mrows/s` | Computing Demographic Parity Difference ($\Delta_{\text{DP}}$), Equalized Odds Difference ($\Delta_{\text{EO}}$), and Disparate Impact Ratio ($DI$) over $N = 100{,}000$ predictions across binary/multi-group protected attributes. | Excludes synthetic dataset generation. | Bounded by memory read bandwidth of $10^5$ `(y_true, y_pred, group_id)` tuples ($\le 40\text{ GB/s}$). | Verify confusion matrix cell counts sum to $100{,}000$ and $|\Delta_{\text{DP}} - \Delta_{\text{DP,ref}}| < 10^{-12}$. |

---

### Metric Group 6: Three-Stage Process RSS Accounting (Check 1.4)

| Metric ID | Unit | Sampling Point | Mandatory Sanity Check |
| :--- | :--- | :--- | :--- |
| `rss_bare_process` | `MB` | First statement of `main()` in `/proc/self/statm` before any Yantra subsystem is constructed. | `rss_bare_process > 0.5 MB` |
| `rss_post_library_load` | `MB` | After initializing `WalEngine`, `ModelRegistry`, `HnswIndex`, and `DagScheduler` (prior to any CUDA/GPU context initialization). | `rss_post_library_load >= rss_bare_process` |
| `rss_peak_workload_hwm` | `MB` | Peak resident set size (`/proc/self/status` `VmHWM`) after all benchmarks complete + `1.0 s` idle. | `rss_peak_workload_hwm >= rss_post_library_load` |

---

## 3. Mandatory `Yantra/BENCHMARKS_METHODOLOGY.md` Template

The following file template MUST be instantiated at `Yantra/BENCHMARKS_METHODOLOGY.md` in Yantra's initial commit:

```markdown
# Yantra Benchmark Methodology (`BENCHMARKS_METHODOLOGY.md`)

**Library:** Yantra (`1.0.0`) — Native C++23 MLOps, Experiment WAL, Model Registry, HNSW Store & Drift/Fairness Engine
**Executable:** `./build/bench_yantra`
**Regression CI Target:** `cmake --build build --target bench-ci`

## 1. Verified Hardware & OS Baseline
- **CPU:** `12th Gen Intel(R) Core(TM) i7-12700K` (8 P-cores + 4 E-cores, 20 threads, AVX2 + FMA)
- **GPU:** `NVIDIA GeForce RTX 3070` (`8,192 MiB` GDDR6 VRAM, Driver `550.163.01`, CUDA `12.4.131`)
- **RAM:** `48 GiB` Dual-Channel DDR4
- **Storage:** `Samsung SSD 970 EVO Plus 1TB` NVMe (`nvme0n1`)
- **OS & Kernel:** Debian GNU/Linux 13 (trixie), Kernel `6.12.113+deb13-amd64` (`x86_64`)

## 2. Timing, Quantiles & Exclusions
- **Clock:** `std::chrono::steady_clock` (`CLOCK_MONOTONIC`).
- **Quantiles:** Every metric reports `p50` (median) and `p99` (tail) after warmup.
- **Disclosed Splits:**
  - WAL throughput is reported separately for `[page-cache write(2)]` and `[durable NVMe fdatasync(2)]`.
  - Model Registry load throughput is reported separately for `[warm page-cache mmap]` and `[cold NVMe POSIX_FADV_DONTNEED]`.
  - HNSW search latency is always paired with measured `Recall@10` against exact `fp64` brute-force k-NN.
  - DAG scheduler overhead is labeled `[noop-node orchestration]` and compared against `dag_theoretical_min_direct`.

## 3. Three-Stage Memory Accounting
- `rss_bare_process`: sampled at `main()` entry before Yantra object construction.
- `rss_post_library_load`: sampled after `WalEngine`, `ModelRegistry`, `HnswIndex`, and `DagScheduler` initialization.
- `rss_peak_workload_hwm`: sampled from `/proc/self/status` `VmHWM` after all benchmarks + 1s settle.

## 4. Post-Measurement Sanity Checks
Every benchmark verifies its output (`CRC32C` WAL replay, `Recall@10 >= 0.95`, DAG topological timestamps, `SHA-256` manifest match, and `1e-12` scalar equivalence for PSI/KS/Fairness) before setting `"sanity_ok": true` in `results.json`.
```

---

## 4. Pre-Release 10-Point Audit Gate for Yantra (§1 Checklist)

Before `Yantra/BENCHMARKS.md` is marked complete, every row must pass all 10 checks:

- [ ] **Check 1.1 (Physical Plausibility):** Durable NVMe WAL $\le 3{,}300\text{ MB/s}$; warm RAM hash $\le 40\text{ GB/s}$; no unphysical speedups.
- [ ] **Check 1.2 (Interval Definition):** All latencies have `p50`/`p99`; HNSW includes `Recall@10`; all throughputs have explicit domain units.
- [ ] **Check 1.3 (Exclusions Disclosed):** Page-cache vs `fdatasync(2)` and warm `mmap` vs cold NVMe are explicitly separated.
- [ ] **Check 1.4 (Memory Accounting):** 3-stage RSS (`Bare Process`, `Post-Library-Load`, `Peak Workload VmHWM`) reported.
- [ ] **Check 1.5 (Baseline Provenance):** Any comparison against Python MLOps tools (`MLflow`, `Weights & Biases`, `Feast`, `Evidently`, `Airflow`) explicitly states whether numbers are published reference ranges or same-host executions.
- [ ] **Check 1.6 (Sanity Check):** `sanity_ok == true` verified in `bench_yantra` for all metrics.
- [ ] **Check 1.7 (Reproducibility):** `cmake --build build --target bench-ci` reproduces `benchmarks/baseline.json` within `20%`.
- [ ] **Check 1.8 (No Null Sinks):** Every serialization/export benchmark writes to a real buffer or file and verifies non-zero byte size + checksum.
- [ ] **Check 1.9 (Hardware Verification):** CPU, GPU, RAM, NVMe, and Kernel (`6.12.113+deb13-amd64`) match system interrogation.
- [ ] **Check 1.10 (Unit Honesty):** Zero bare `"FPS"` labels; noop DAG benchmarks explicitly labeled `[noop-node orchestration]`.
