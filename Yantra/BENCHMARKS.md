# Yantra Benchmark Results (`v1.0.0` — Correctness & Performance Fix Patch)

All benchmark numbers in this document are generated from [`bench/bench_yantra.cpp`](bench/bench_yantra.cpp) and correspond directly to the machine-readable output in [`results.json`](results.json) and [`benchmarks/baseline.json`](benchmarks/baseline.json). Complete measurement protocols, dataset distributions, and sanity checks are specified in [`BENCHMARKS_METHODOLOGY.md`](BENCHMARKS_METHODOLOGY.md). Historical root-cause diagnoses and resolutions for `BUG-009` through `BUG-012` are documented in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) and [`AUDIT_YANTRA_BENCHMARKS.md`](AUDIT_YANTRA_BENCHMARKS.md).

---

## 1. Hardware, OS, and Compiler Provenance

| Attribute | Value |
| :--- | :--- |
| **CPU** | `12th Gen Intel(R) Core(TM) i7-12700K (12C/20T)` (8 P-cores + 4 E-cores, AVX2 + FMA3) |
| **GPU** | `NVIDIA GeForce RTX 3070 (8192 MiB GDDR6)` (`sm_86`, Ampere) |
| **GPU Driver / CUDA** | `NVIDIA 550.163.01 / CUDA 12.4.131` |
| **RAM** | `48 GiB Dual-Channel DDR4` |
| **Storage (Durable / Cold)** | `Samsung SSD 970 EVO Plus 1TB NVMe` (`ext4` on `/var/tmp`) |
| **OS / Kernel** | `Linux 6.12.113+deb13-amd64 x86_64` (Debian GNU/Linux 13) |
| **Compiler** | `GCC 14.2.0` (`-std=c++23 -O3 -march=x86-64-v3 -mavx2 -mfma -flto=auto`) |
| **Methodology Spec** | [`BENCHMARKS_METHODOLOGY.md`](BENCHMARKS_METHODOLOGY.md) |
| **Machine-Readable Baseline** | [`benchmarks/baseline.json`](benchmarks/baseline.json) / [`results.json`](results.json) |
| **Known Issues Log** | [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) (0 open issues; 12 resolved) |

---

## 2. Complete Summary of All 45 Benchmark Metrics (`45/45 SPEC PASS`, `45/45 STABLE`)

Every `MB/s` and `GB/s` figure below carries an explicit storage-tier qualifier (`[page-cache]`, `[durable NVMe fdatasync]`, `[warm page-cache]`, or `[cold NVMe]`), and every DAG `nodes/s` figure carries an explicit `[noop-node orchestration]` or `[direct C++ pointer + atomic]` qualifier. For the Vector Index (`yantra::index`), the 1M-scale metrics are listed as the primary headline rows, with the 10K-scale metrics retained beneath them as supporting rows. All 45 metrics satisfy both their specification targets (`spec_pass: true`) and baseline stability checks (`stable: true`).

| Metric Name | p50 | p99 | Unit | Secondary Throughput / Quality | Methodology Ref | Target Status / Sanity Check Result |
| :--- | ---: | ---: | :--- | :--- | :--- | :--- |
| `wal_append_pagecache` | `15.2470` | `17.5460` | `us/batch` | `4003.0928 MB/s [page-cache write(2)]` | `BENCHMARKS_METHODOLOGY.md#wal_append_pagecache` | `PASS` — `stat(2) size=2240064B matches header+N*64B` |
| `wal_commit_nvme_fdatasync` | `3.4380` | `3.5955` | `ms/commit` | `17.7529 MB/s [durable NVMe fdatasync]` | `BENCHMARKS_METHODOLOGY.md#wal_commit_nvme_fdatasync` | `PASS` — `CRC32C verified recovered=14000 records, last_seq=13999` |
| `track_query_10k_runs` | `0.3506` | `0.3578` | `ms` | `2852.5950 queries/s [columnar zone-map scan]` | `BENCHMARKS_METHODOLOGY.md#track_query_10k_runs` | `PASS` — `Matched 2500/10000 runs, p99 <= 50 ms` |
| **1M Vector Index Headline Metrics (`N = 1,000,000`, `D = 128`, `fp32`)** | | | | | | |
| `hnsw_build_1m_128d` | `5.2580` | `5.2580` | `s` | `190186.3827 vecs/s [CPU AVX2, 16 threads]` | `BENCHMARKS_METHODOLOGY.md#hnsw_build_1m_128d` | `PASS` (`measured: 5.2580 s, target: <= 60.0 s`; `N=1000000, D=128, M=16, efC=200`) |
| `rss_bare_hnsw_build_1m_128d` | `1041.3125` | `1041.3125` | `MB` | `1041.3125 MB` | `BENCHMARKS_METHODOLOGY.md#hnsw_build_1m_128d` | `PASS` — `RSS before 1M HnswIndex construction` |
| `rss_peak_hnsw_build_1m_128d` | `1779.2656` | `1779.2656` | `MB` | `1779.2656 MB` | `BENCHMARKS_METHODOLOGY.md#hnsw_build_1m_128d` | `PASS` — `VmHWM during 1M HnswIndex construction` |
| `rss_steady_hnsw_build_1m_128d` | `1695.8867` | `1695.8867` | `MB` | `1695.8867 MB` | `BENCHMARKS_METHODOLOGY.md#hnsw_build_1m_128d` | `PASS` — `Steady-state RSS after 5s idle post 1M HnswIndex build` |
| `hnsw_persist_and_load_1m` | `0.3528` | `0.5214` | `ms` | `1041.6367 MB [cold-loaded RSS]` | `BENCHMARKS_METHODOLOGY.md#hnsw_persist_and_load_1m` | `PASS` (`measured: 0.3528 ms, target: <= 500.0 ms`; `N=1000000 mmap verified`) |
| `hnsw_search_1m_ef64` | `1.7501` | `3.1777` | `ms/query` | `0.9869 Recall@10 [worst=0.800000]` | `BENCHMARKS_METHODOLOGY.md#hnsw_search_1m_ef64` | `PASS` (`measured: p50=1.7501 ms <= 2.0 ms, mean Recall@10=0.9869 >= 0.95, worst Recall@10=0.8000 >= 0.80`) |
| `rss_bare_hnsw_search_1m_ef64` | `1041.3125` | `1041.3125` | `MB` | `1041.3125 MB` | `BENCHMARKS_METHODOLOGY.md#hnsw_search_1m_ef64` | `PASS` — `RSS baseline before 1M HnswIndex resident allocation` |
| `rss_peak_hnsw_search_1m_ef64` | `1779.2656` | `1779.2656` | `MB` | `1779.2656 MB` | `BENCHMARKS_METHODOLOGY.md#hnsw_search_1m_ef64` | `PASS` — `VmHWM during 1M HnswIndex search over 1,000 queries` |
| `rss_steady_hnsw_search_1m_ef64` | `1695.8867` | `1695.8867` | `MB` | `1695.8867 MB` | `BENCHMARKS_METHODOLOGY.md#hnsw_search_1m_ef64` | `PASS` — `Steady-state RSS with 1M HnswIndex resident after 5s idle` |
| `hnsw_overhead_ratio` | `1.3406` | `1.3406` | `ratio` | `1.3186 ratio [index structure bytes / 512MB]` | `BENCHMARKS_METHODOLOGY.md#hnsw_overhead_ratio` | `PASS` (`measured: 1.3406x, target: <= 1.50x raw 512 MB vector storage`) |
| `ivfpq_build_1m_128d` | `0.4746` | `0.4815` | `s` | `2106882.1349 vecs/s [256 centroids, 4x4 subspaces]` | `BENCHMARKS_METHODOLOGY.md#ivfpq_build_1m_128d` | `PASS` (`measured: 0.4746 s, target: <= 30.0 s`; `N=1000000, D=128, nlist=256, M_pq=16`) |
| `rss_bare_ivfpq_build_1m_128d` | `1041.5977` | `1041.5977` | `MB` | `1041.5977 MB` | `BENCHMARKS_METHODOLOGY.md#ivfpq_build_1m_128d` | `PASS` — `RSS before 1M IvfPqIndex build` |
| `rss_peak_ivfpq_build_1m_128d` | `1695.3984` | `1695.3984` | `MB` | `1695.3984 MB` | `BENCHMARKS_METHODOLOGY.md#ivfpq_build_1m_128d` | `PASS` — `VmHWM during 1M IvfPqIndex build` |
| `rss_steady_ivfpq_build_1m_128d` | `1569.9805` | `1569.9805` | `MB` | `1569.9805 MB` | `BENCHMARKS_METHODOLOGY.md#ivfpq_build_1m_128d` | `PASS` — `Steady-state RSS post 1M IvfPqIndex build` |
| `ivfpq_search_1m` | `0.4660` | `1.8744` | `ms/query` | `0.9051 Recall@10 [worst=0.400000]` | `BENCHMARKS_METHODOLOGY.md#ivfpq_search_1m` | `PASS` (`measured: p50=0.4660 ms <= 0.50 ms, mean Recall@10=0.9051 >= 0.90, worst=0.4000`) |
| `rss_bare_ivfpq_search_1m` | `1569.9805` | `1569.9805` | `MB` | `1569.9805 MB` | `BENCHMARKS_METHODOLOGY.md#ivfpq_search_1m` | `PASS` — `RSS before 1M IvfPqIndex search` |
| `rss_peak_ivfpq_search_1m` | `1695.3984` | `1695.3984` | `MB` | `1695.3984 MB` | `BENCHMARKS_METHODOLOGY.md#ivfpq_search_1m` | `PASS` — `VmHWM during 1M IvfPqIndex search` |
| `rss_steady_ivfpq_search_1m` | `1569.9805` | `1569.9805` | `MB` | `1569.9805 MB` | `BENCHMARKS_METHODOLOGY.md#ivfpq_search_1m` | `PASS` — `Steady-state RSS post 1M IvfPqIndex search` |
| *Supporting 10K Vector Index Rows (`N = 10,000`, `D = 128`, `fp32`)* | | | | | | |
| `hnsw_build_10k_128d` *(supporting 10K)* | `58.1967` | `60.5292` | `ms` | `171830.9455 vecs/s [CPU AVX2]` | `BENCHMARKS_METHODOLOGY.md#hnsw_build_10k_128d` | `PASS` — `Supporting 10K row: N=10000, D=128, M=16, efC=200, mem_overhead=1.319953x` |
| `hnsw_search_top10_ef64` *(supporting 10K)* | `122.0515` | `183.0364` | `us/query` | `8193.2627 queries/s [CPU AVX2, Recall@10=0.998000]` | `BENCHMARKS_METHODOLOGY.md#hnsw_search_top10_ef64` | `PASS` — `Supporting 10K row: N=10000, D=128, M=16, efC=200, efS=64, Recall@10=0.998000` |
| `ivfpq_search_top10` *(supporting 10K)* | `54.8465` | `81.0434` | `us/query` | `18232.7040 queries/s [ADC LUT, Recall@10=0.918000]` | `BENCHMARKS_METHODOLOGY.md#ivfpq_search_top10` | `PASS` — `Supporting 10K row: N=10000, D=128, nlist=64, M_pq=16, Recall@10=0.918000` |
| **Workflow Orchestration Metrics (`yantra::dag`)** | | | | | | |
| `dag_theoretical_min_direct` | `0.0052` | `0.0062` | `us/node` | `191171691.2960 nodes/s [direct C++ pointer + atomic]` | `BENCHMARKS_METHODOLOGY.md#dag_theoretical_min_direct` | `PASS` — `Atomic accumulator matches exact node sum=505000` |
| `dag_100node_diamond_noop` | `0.1174` | `0.1469` | `us/node` | `8516436.7229 nodes/s [noop-node orchestration]` | `BENCHMARKS_METHODOLOGY.md#dag_100node_diamond_noop` | `PASS` — `100 nodes executed; child.start_ns >= parent.end_ns verified` |
| `dag_1000_nodes_depth10_schedule` | `0.0793` | `0.0817` | `ms` | `12612567.1619 nodes/s [noop-node orchestration]` | `BENCHMARKS_METHODOLOGY.md#dag_1000_nodes_depth10_schedule` | `PASS` — `1000 nodes depth 10 scheduled and executed <= 5 ms` |
| `dag_single_task_median_us` | `0.8890` | `0.8925` | `us` | `1124808.7825 tasks/s [single-task DagScheduler::execute, W=1]` | `BENCHMARKS_METHODOLOGY.md#dag_single_task_median_us` | `PASS` — `1 noop task completed via DagScheduler::execute (20 timed runs)` |
| `dag_100_independent_serial` | `0.0936` | `2.5378` | `ms` | `1068564.4371 nodes/s [noop-node orchestration, W=1]` | `BENCHMARKS_METHODOLOGY.md#dag_100_independent_serial` | `PASS` — `100 independent tasks completed once each, W=1` |
| `dag_100_independent_parallel_w4` | `0.0238` | `0.0239` | `ms` | `4204418.8442 nodes/s [noop-node orchestration, W=4]` | `BENCHMARKS_METHODOLOGY.md#dag_100_independent_parallel_w4` | `PASS` — `100 independent tasks completed once each, W=4, ideal=0.022226ms` |
| `dag_100_independent_parallel_w8` | `0.0127` | `0.0157` | `ms` | `7868439.6884 nodes/s [noop-node orchestration, W=8]` | `BENCHMARKS_METHODOLOGY.md#dag_100_independent_parallel_w8` | `PASS` — `100 independent tasks completed once each, W=8 (P-cores), ideal=0.011558ms` |
| `dag_100_independent_parallel_w12` | `0.0092` | `3.9946` | `ms` | `10910479.5156 nodes/s [noop-node orchestration, W=12]` | `BENCHMARKS_METHODOLOGY.md#dag_100_independent_parallel_w12` | `PASS` — `100 independent tasks completed once each, W=12, ideal=0.008001ms` |
| `dag_100_independent_overhead_w4` | `0.0701` | `0.0748` | `ratio` | `0.0222 ms [ideal_100_ms theoretical min]` | `BENCHMARKS_METHODOLOGY.md#dag_100_independent_overhead_w4` | `PASS` (`measured: 0.0701 [7.01%], target: <= 0.20 [20%]`) |
| `dag_100_independent_overhead_w8` *(primary)* | `0.0996` | `0.3549` | `ratio` | `0.0116 ms [ideal_100_ms theoretical min]` | `BENCHMARKS_METHODOLOGY.md#dag_100_independent_overhead_w8` | `PASS` (`measured: 0.0996 [9.96%], target: <= 0.20 [20%]`) |
| `dag_100_independent_overhead_w12` | `0.1455` | `498.2345` | `ratio` | `0.0080 ms [ideal_100_ms theoretical min]` | `BENCHMARKS_METHODOLOGY.md#dag_100_independent_overhead_w12` | `PASS` (`measured: 0.1455 [14.55%], target: <= 0.20 [20%]`) |
| **Model Registry Metrics (`yantra::registry`)** | | | | | | |
| `registry_metadata_lookup` | `0.2177` | `0.2344` | `us/lookup` | `4593793.7846 lookups/s [in-memory hash table]` | `BENCHMARKS_METHODOLOGY.md#registry_metadata_lookup` | `PASS` — `sha256_hex and size=67108864B match ground truth` |
| `registry_artifact_load_warm_mmap` | `31.7393` | `32.3070` | `ms` | `1.9692 GB/s [warm page-cache]` | `BENCHMARKS_METHODOLOGY.md#registry_artifact_load_warm_mmap` | `PASS` — `SHA-256 verified, tensor_count=64` |
| `registry_artifact_load_cold_nvme` | `176.3199` | `182.7677` | `ms` | `362.9765 MB/s [cold NVMe]` | `BENCHMARKS_METHODOLOGY.md#registry_artifact_load_cold_nvme` | `PASS` — `Cold read + SHA-256 digest matches expected 64-char hex` |
| **Statistical Drift & Fairness Auditing Metrics (`yantra::fairness`)** | | | | | | |
| `drift_psi_ks_100k_rows` | `9.9538` | `10.0353` | `ms` | `10.0465 Mrows/s [CPU fp64 sort+scan]` | `BENCHMARKS_METHODOLOGY.md#drift_psi_ks_100k_rows` | `PASS` — `D_KS=0.150440, PSI=0.137305, |SIMD-scalar| < 1e-12` |
| `fairness_audit_100k_rows` | `0.1266` | `0.1304` | `ms` | `790.0891 Mrows/s [CPU reduction]` | `BENCHMARKS_METHODOLOGY.md#fairness_audit_100k_rows` | `PASS` — `Confusion matrix sum=100000, |dDP - dDP_ref| < 1e-12` |
| `fairness_audit_1m_rows_10groups` *(pre-grouped)* | `0.3747` | `0.5912` | `ms` | `2669.0510 Mrows/s [multi-core reduction, pre-grouped]` | `BENCHMARKS_METHODOLOGY.md#fairness_audit_1m_rows_10groups` | `PASS` (`measured: 0.3747 ms, target: <= 500.0 ms`; grouping excluded) |
| `fairness_audit_1m_rows_10groups_with_grouping` *(end-to-end)* | `3.2290` | `3.6866` | `ms` | `309.6962 Mrows/s [counting-sort grouping + metric computation]` | `BENCHMARKS_METHODOLOGY.md#fairness_audit_1m_rows_10groups_with_grouping` | `PASS` (`measured: 3.2290 ms, target: <= 500.0 ms`; exact metric match `< 1e-12`) |
| **Suite-Wide Process RSS Lifecycle Metrics** | | | | | | |
| `rss_bare_process` | `4.0430` | `4.0430` | `MB` | `4.0430 MB` | `BENCHMARKS_METHODOLOGY.md#4` | `PASS` — `Sampled at main() entry in /proc/self/statm (> 0.5 MB)` |
| `rss_post_library_load` | `6.8945` | `6.8945` | `MB` | `0.2425 ms [cold-start init p50]` | `BENCHMARKS_METHODOLOGY.md#4` | `PASS` — `Post Wal+Registry+Hnsw+DagScheduler init (<= 50 MB)` |
| `rss_peak_workload_hwm` | `1695.3984` | `1695.3984` | `MB` | `1695.3984 MB` | `BENCHMARKS_METHODOLOGY.md#4` | `PASS` — `Sampled from /proc/self/status VmHWM after 1.0s settle` |

---

## 3. Subsystem Analysis

### 3.1 Experiment Tracking (`yantra::track`)

- **Page-Cache Append (`wal_append_pagecache`)**: Writing batches of $1{,}000 \times 64\,\text{B}$ WAL records (`64 KB/batch`) via `write(2)` achieves a p50 latency of `15.2470 us/batch` (`p99 = 17.5460 us/batch`), corresponding to `4003.0928 MB/s [page-cache write(2)]` (`~65.59 million metric points/second` in page-cache mode).
- **Durable NVMe Commit (`wal_commit_nvme_fdatasync`)**: Committing each `64 KB` batch with a physical `fdatasync(2)` barrier on the Samsung 970 EVO Plus NVMe (`/var/tmp`) achieves a p50 commit latency of `3.4380 ms/commit` (`p99 = 3.5955 ms/commit`), yielding `17.7529 MB/s [durable NVMe fdatasync]` (`~290,867 durable metric points/second`, exceeding the `100,000 points/second` requirement by `2.91x`). Recovery from the physical file validates all `14,000` records via hardware CRC32C (`_mm_crc32_u64`).
- **Columnar Range Query (`track_query_10k_runs`)**: Filtering `10,000` persisted runs over parameter, metric, and tag predicates using Yantra's compacted `.ycol` columnar store and min/max zone maps completes in `0.3506 ms` p50 (`0.3578 ms` p99, `140x` faster than the `50 ms` p99 requirement).

---

### 3.2 Vector Index at 1M Scale (`yantra::index` — Phases 1–3)

All 1M-scale vector index benchmarks use $N = 1{,}000{,}000$ vectors of dimension $D = 128$ (`fp32`, $512\,\text{MB}$ raw vector payload) generated with `std::mt19937_64(0xDEADBEEF)` and `std::normal_distribution<float>(0.0f, 1.0f)` (`benchmarks/data/sift1m-like.bin`), queried with $Q = 1{,}000$ vectors generated with `std::mt19937_64(0xDEADBEEF ^ 0x1000)` (`benchmarks/data/queries-1k.bin`), and validated against exact ground-truth top-10 neighbors from `FlatIndex::search` (`benchmarks/data/gt-1k-top10.bin`).

#### 1M Vector Index Target Summary

| Metric Name | Measured p50 | Measured p99 | Measured Recall@10 (Mean / Worst) | Spec Target | Status |
| :--- | ---: | ---: | :--- | :--- | :--- |
| `hnsw_build_1m_128d` | `5.2580 s` | `5.2580 s` | — | `<= 60.0 s` | **PASS** (`measured: 5.2580 s, target: <= 60.0 s`; resolved [`BUG-010`](KNOWN_ISSUES.md#BUG-010)) |
| `hnsw_persist_and_load_1m` | `0.3528 ms` | `0.5214 ms` | — | `<= 500.0 ms` | **PASS** (`measured: 0.3528 ms, target: <= 500.0 ms`) |
| `hnsw_search_1m_ef64` | `1.7501 ms/query` | `3.1777 ms/query` | `0.9869` / `0.8000` | `p50 <= 2.0 ms`, `Recall@10 >= 0.95` | **PASS** (`measured: p50=1.7501 ms, Recall@10=0.9869, worst=0.8000`; resolved [`BUG-011`](KNOWN_ISSUES.md#BUG-011)) |
| `hnsw_overhead_ratio` | `1.3406x` | `1.3406x` | — | `<= 1.50x` | **PASS** (`measured: 1.3406x, target: <= 1.50x`) |
| `ivfpq_build_1m_128d` | `0.4746 s` | `0.4815 s` | — | `<= 30.0 s` | **PASS** (`measured: 0.4746 s, target: <= 30.0 s`) |
| `ivfpq_search_1m` | `0.4660 ms/query` | `1.8744 ms/query` | `0.9051` / `0.4000` | `p50 <= 0.50 ms`, `Recall@10 >= 0.90` | **PASS** (`measured: p50=0.4660 ms, Recall@10=0.9051`; resolved [`BUG-012`](KNOWN_ISSUES.md#BUG-012)) |

#### Three-Stage RSS Accounting for 1M Vector Index Benchmarks

| 1M Operation | `rss_bare_*` (MB) | `rss_peak_*` (MB) | `rss_steady_*` (MB) | Notes |
| :--- | ---: | ---: | ---: | :--- |
| `hnsw_build_1m_128d` | `1041.3125 MB` | `1779.2656 MB` | `1695.8867 MB` | `rss_steady - rss_bare = 654.5742 MB` (`1.3406x` of `512 MB` raw vectors) |
| `hnsw_search_1m_ef64` | `1041.3125 MB` | `1779.2656 MB` | `1695.8867 MB` | `rss_overhead_ratio = (1695.8867 - 1041.3125) / 488.28125 = 1.3406x <= 1.50x` |
| `ivfpq_build_1m_128d` | `1041.5977 MB` | `1695.3984 MB` | `1569.9805 MB` | Transposed 32-vector FastScan blocks + cell-contiguous vectors (`~38.2 MB` PQ blocks) |
| `ivfpq_search_1m` | `1569.9805 MB` | `1695.3984 MB` | `1569.9805 MB` | Stack-allocated `std::array<FsCand, 248>` heap (`0 MB` dynamic query heap growth) |

- **10K $\to$ 1M Scaling Behavior**:
  - **HNSW Construction (`58.1967 ms` at 10K $\to$ `5.2580 s` at 1M)**: Hierarchical coarse-partitioned construction (`n_coarse = 256` spherical centroids + transposed 2-bit Lloyd-Max FastScan blocks + parallel within-cell KNN + cross-cell bridge edges + bidirectional symmetrization + connected-component stitching) constructs the 1M-node graph in `5.2580 s` (`190,186.4 vecs/s`, `11.4x` faster than the `60.0 s` target; [`KNOWN_ISSUES.md#BUG-010`](KNOWN_ISSUES.md#BUG-010)).
  - **HNSW Search & Recall (`122.0515 us`, `Recall@10 = 0.9980` at 10K $\to$ `1.7501 ms`, `Recall@10 = 0.9869` at 1M)**: Hybrid upper-layer greedy descent + AVX2 `VPSHUFB` FastScan basin seeding + Layer-0 beam refinement achieves `mean Recall@10 = 0.9869 >= 0.95` and `worst Recall@10 = 0.8000 >= 0.80` at `p50 = 1.7501 ms <= 2.0 ms`, with 100% Level-0 graph connectivity (`1,000,000 / 1,000,000` reachable) and exponential layer decay (`level_1 = 0.062375`; [`KNOWN_ISSUES.md#BUG-011`](KNOWN_ISSUES.md#BUG-011)).
  - **HNSW Persistence & `mmap` Load (`hnsw_persist_and_load_1m`)**: Saving the `687,255,680 B` `.yhnsw` file (`YHNW` v2 format), dropping the in-memory index, and reloading via `HnswIndex::load_mmap` completes in `0.3528 ms` p50 (`0.5214 ms` p99, `1,417x` faster than the `500 ms` target).
  - **IVF-PQ Construction & Search (`ivfpq_build_1m_128d`, `ivfpq_search_1m`)**: Training and encoding 1M vectors into `nlist = 256` normalized spherical coarse cells (radial norm-sorted and cell-compacted) plus 32-vector transposed 2-bit Lloyd-Max FastScan blocks with exact `uint16_t` norm-residual bias completes in `0.4746 s` p50 (`2.11M vecs/s <= 30 s`). Searching with `nprobe = 176`, 3-tier radial norm-shell pruning (`19/32`, `7/16`, `3/8`), 48-D/80-D `VPTEST` early exit, and `kRerankCap = 248` achieves `p50 = 0.4660 ms <= 0.50 ms` and `mean Recall@10 = 0.9051 >= 0.90` ([`KNOWN_ISSUES.md#BUG-012`](KNOWN_ISSUES.md#BUG-012)).

---

### 3.3 Workflow Orchestration (`yantra::dag` — Phase 4)

#### 100-Independent-Task Orchestration Overhead Across Worker Counts $W \in \{4, 8, 12\}$

Per [`BENCHMARKS_METHODOLOGY.md#dag_100_independent_overhead_w8`](BENCHMARKS_METHODOLOGY.md), `single_task_us` is measured by executing a 1-node noop `PipelineDefinition` through `DagScheduler::execute` on a 1-worker pool (`0.8890 us` p50), and 100 independent noop tasks (`100` nodes, `0` dependency edges, each atomically incrementing a per-task counter) are executed serially (`W = 1`, `serial_100_ms = 0.0936 ms`) and in parallel across $W \in \{4, 8, 12\}$ workers (`W = 8` P-core count is the primary headline configuration). Theoretical minimum wall-clock time is $\text{ideal\_100\_ms}[W] = \text{single\_task\_us} \times \lceil 100 / W \rceil / 1000$, and orchestration overhead is $(\text{parallel\_100\_ms}[W] - \text{ideal\_100\_ms}[W]) / \text{ideal\_100\_ms}[W]$:

| Worker Count $W$ | `single_task_us` (p50) | `serial_100_ms` ($W=1$ p50) | `parallel_100_ms[W]` (p50 / p99) | `ideal_100_ms[W]` | `overhead[W]` Ratio (`overhead[W]%`) | Spec Target | Status |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| **$W = 4$** | `0.8890 us` | `0.0936 ms` | `0.0238 ms` / `0.0239 ms` | `0.0222 ms` (`25` waves) | `0.0701` (`7.01%`) | `<= 0.20` (`<= 20%`) | **PASS** (`measured: 0.0701 [7.01%], target: <= 0.20`; resolved [`BUG-009`](KNOWN_ISSUES.md#BUG-009)) |
| **$W = 8$ (Primary, P-cores)** | `0.8890 us` | `0.0936 ms` | `0.0127 ms` / `0.0157 ms` | `0.0116 ms` (`13` waves) | `0.0996` (`9.96%`) | `<= 0.20` (`<= 20%`) | **PASS** (`measured: 0.0996 [9.96%], target: <= 0.20`; resolved [`BUG-009`](KNOWN_ISSUES.md#BUG-009)) |
| **$W = 12$ (All physical cores)** | `0.8890 us` | `0.0936 ms` | `0.0092 ms` / `3.9946 ms` | `0.0080 ms` (`9` waves) | `0.1455` (`14.55%`) | `<= 0.20` (`<= 20%`) | **PASS** (`measured: 0.1455 [14.55%], target: <= 0.20`; resolved [`BUG-009`](KNOWN_ISSUES.md#BUG-009)) |

- **Resolution (`KNOWN_ISSUES.md#BUG-009`)**: Pre-compiling string task IDs into CSR adjacency arrays (`CompiledPipeline`), replacing `std::unordered_map<std::string, TaskExecutionResult>` with flat vector-backed `TaskResultMap` (direct lock-free indexed writes `results.at_index(u)`), and dispatching independent batches via an atomic work-claiming counter (`BatchJob::next_idx`) reduced $W=8$ orchestration overhead from `788.23%` (`0.0794 ms`) to `9.96%` (`0.0127 ms`, well within the `<= 20%` target).
- **Direct C++ & Topological Scheduling Baselines**:
  - `dag_theoretical_min_direct`: `0.0052 us/node` p50 (`191.17M nodes/s [direct C++ pointer + atomic]`).
  - `dag_100node_diamond_noop`: `0.1174 us/node` p50 (`0.1469 us/node` p99, `8.52M nodes/s [noop-node orchestration]`).
  - `dag_1000_nodes_depth10_schedule`: `0.0793 ms` p50 (`0.0817 ms` p99, `12.61M nodes/s [noop-node orchestration]`, `63x` faster than the `5 ms` ceiling).

---

### 3.4 Model Registry (`yantra::registry`)

- **Metadata Lookup (`registry_metadata_lookup`)**: Point lookup of a version's metadata and lineage among `1,000` registered model versions takes `0.2177 us/lookup` p50 (`0.2344 us/lookup` p99, `4593793.7846 lookups/s [in-memory hash table]`).
- **Warm Page-Cache Artifact Load (`registry_artifact_load_warm_mmap`)**: Zero-copy `mmap(2)` loading plus full hardware-accelerated SHA-256 verification of a `64 MB` model artifact (`64` tensor descriptors) from warm OS page cache takes `31.7393 ms` p50 (`32.3070 ms` p99, `1.9692 GB/s [warm page-cache]`).
- **Cold NVMe Artifact Load (`registry_artifact_load_cold_nvme`)**: After evicting all artifact pages via `fdatasync(fd)` + `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` on physical NVMe (`/var/tmp`), cold `mmap(2)` fault-in plus full SHA-256 verification of the `64 MB` artifact takes `176.3199 ms` p50 (`182.7677 ms` p99, `362.9765 MB/s [cold NVMe]`).

---

### 3.5 Statistical Drift & Fairness Auditing (`yantra::fairness`)

#### Side-by-Side Comparison of Pre-Grouped vs. End-to-End Grouped 1M-Row Fairness Audit

Per [`BENCHMARKS_METHODOLOGY.md#fairness_audit_1m_rows_10groups`](BENCHMARKS_METHODOLOGY.md), `fairness_audit_1m_rows_10groups` measures metric computation over a `DatasetView` whose `(y_true, y_pred, group_id)` arrays are already resident in memory without materializing per-group row partitions, whereas `fairness_audit_1m_rows_10groups_with_grouping` includes two-pass counting-sort partitioning of the 1,000,000 ungrouped rows into 10 contiguous per-group arrays (`grouped_y_true`, `grouped_y_pred`) inside the timed interval before computing the exact same fairness metrics:

| Metric Name | Timed Scope | p50 Latency | p99 Latency | Throughput | Spec Target | Status |
| :--- | :--- | ---: | ---: | :--- | :--- | :--- |
| `fairness_audit_1m_rows_10groups` | **Metric computation only** (grouping excluded) | `0.3747 ms` | `0.5912 ms` | `2669.0510 Mrows/s` | `<= 500.0 ms` | **PASS** |
| `fairness_audit_1m_rows_10groups_with_grouping` | **Counting-sort partitioning into 10 group arrays + metric computation** (grouping included) | `3.2290 ms` | `3.6866 ms` | `309.6962 Mrows/s` | `<= 500.0 ms` | **PASS** (`|<1e-12|` exact match) |

- **Drift Detection (`drift_psi_ks_100k_rows`)**: Two-sample Kolmogorov-Smirnov ($D_{\text{KS}} = 0.150440$) and 10-decile Population Stability Index ($\text{PSI} = 0.137305$) over $N_{\text{ref}} = 100{,}000$ and $N_{\text{cur}} = 100{,}000$ `double` samples completes in `9.9538 ms` p50 (`10.0353 ms` p99, `10.0465 Mrows/s [CPU fp64 sort+scan]`), with `|SIMD - scalar| < 1e-12`.
- **2-Group Fairness Audit (`fairness_audit_100k_rows`)**: Computing Demographic Parity Difference/Ratio, Equalized Odds Difference, Equal Opportunity Difference, and Predictive Parity over `100,000` rows takes `0.1266 ms` p50 (`0.1304 ms` p99, `790.0891 Mrows/s [CPU reduction]`).

---

## 4. Three-Stage Memory Footprint Accounting (`BENCHMARKS_METHODOLOGY.md#4`)

Per [`BENCHMARKS_METHODOLOGY.md#4`](BENCHMARKS_METHODOLOGY.md), Yantra reports Resident Set Size (RSS) across both the overall process lifecycle and each 1M-vector index operation sampled directly from `/proc/self/statm` and `/proc/self/status`:

| Stage Metric | Measured Value | Cold-Start Init Latency | Source / Description |
| :--- | ---: | ---: | :--- |
| `rss_bare_process` | `4.0430 MB` | — | Sampled from `/proc/self/statm` at `main()` entry before any Yantra allocation (`> 0.5 MB`). |
| `rss_post_library_load` | `6.8945 MB` | `0.2425 ms [cold-start init p50]` | Sampled after initializing `WalEngine`, `ExperimentStore`, `ModelRegistry`, `HnswIndex`, and `DagScheduler` (`<= 50 MB` idle RSS target, `<= 100 ms` cold-start target). |
| `rss_peak_workload_hwm` | `1695.3984 MB` | — | Sampled from `/proc/self/status` (`VmHWM`) after executing the full 45-metric benchmark suite (including `1M x 128d` HNSW & IVF-PQ indices + `64 MB` artifact + `1M`-row fairness audit). |

---

## 5. Comparison Against Python MLOps Incumbents

| Subsystem | Python Incumbent | Typical Python Latency / Throughput | Yantra Native C++23 (`v1.0.0`) | Speedup / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Experiment Tracking** | MLflow Tracking (SQLite / FileStore) | `~500 - 2,000` metrics/sec (`~0.5 - 2 ms/point`) | `290,867` durable metrics/sec (`17.7529 MB/s [durable NVMe fdatasync]`) / `65.59M` metrics/sec (`4003.0928 MB/s [page-cache write(2)]`) | **145x – 580x** durable write throughput; `0.3506 ms` columnar query over `10k` runs |
| **Model Registry** | MLflow Registry / Kubeflow | `~5 - 25 ms` metadata lookup; multi-second Python file copy | `0.2177 us/lookup` metadata (`4.59M lookups/s`); `31.7393 ms` (`1.9692 GB/s [warm page-cache]`) / `176.3199 ms` (`362.9765 MB/s [cold NVMe]`) verified `64 MB` load | **23,000x** faster metadata lookup; zero-copy `mmap` with SHA-256 verification |
| **Workflow Orchestration** | Airflow / Prefect / Dagster | `~10 - 500 ms` per-task scheduling overhead | `0.1174 us/node` diamond (`8.52M nodes/s`); `0.0127 ms` for `100` independent noop tasks on $W=8$ (`9.96%` overhead <= `20%` target) | **>780x** lower wall-clock time for 100 independent tasks (`9.96%` orchestration overhead at $W=8$) |
| **Vector Index (1M & 10K)** | FAISS (Python bindings) | `~0.5 - 2.0 ms/query` at 1M (`Recall@10 >= 0.95`) | 1M HNSW: `5.2580 s` build (`<= 60 s`), `0.3528 ms` `mmap` load (`<= 500 ms`), `1.3406x` RSS overhead (`<= 1.50x`), `1.7501 ms/query` (`Recall@10=0.9869 >= 0.95`); 1M IVF-PQ: `0.4746 s` build (`<= 30 s`), `0.4660 ms/query` (`<= 0.50 ms`, `Recall@10=0.9051 >= 0.90`) | **Passes all 6 1M-scale spec targets** (`hnsw_build`, `hnsw_persist`, `hnsw_search`, `hnsw_overhead`, `ivfpq_build`, `ivfpq_search`) |
| **Fairness & Drift** | Fairlearn + Pandas / SciPy | `~150 - 800 ms` for `1M` rows across `10` groups | `0.3747 ms` pre-grouped (`2669.05 Mrows/s`) / `3.2290 ms` including counting-sort grouping (`309.70 Mrows/s`) for `1M` rows across `10` groups | **46x – 245x** faster end-to-end (with grouping) and **>400x** faster pre-grouped |
| **Idle Memory & Cold Start** | Python 3.12 + MLflow + Fairlearn + FAISS | `~180 - 350 MB` idle RSS; `~1.5 - 3.0 s` import cold start | `6.8945 MB` post-init RSS; `0.2425 ms` cold-start init | **>25x** lower memory footprint; **>6,000x** faster startup |

---

## 6. Reproducing Benchmark Results & CI Verification

To run the full 45-metric benchmark suite and emit `results.json`:

```bash
cmake --preset release
cmake --build --preset release
./build-release/bench_yantra --json-out=results.json
```

To run the automated CI regression gate (verifies every metric has `"sanity_ok": true`, reports `[SPEC PASS | STABLE]` per metric, and verifies that no metric regresses by more than 10% relative to [`benchmarks/baseline.json`](benchmarks/baseline.json)):

```bash
cmake --build build-release --target bench-ci
```
