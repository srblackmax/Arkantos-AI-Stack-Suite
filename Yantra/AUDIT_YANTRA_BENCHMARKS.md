# Yantra Benchmark Completeness Audit (`AUDIT_YANTRA_BENCHMARKS.md`)

**Date:** 2026-11-01  
**Library Version:** `Yantra v1.0.0`  
**Scope:** Audit of existing benchmarks in [`BENCHMARKS.md`](BENCHMARKS.md), [`BENCHMARKS_METHODOLOGY.md`](BENCHMARKS_METHODOLOGY.md), and [`bench/bench_yantra.cpp`](bench/bench_yantra.cpp) against the hard scale and methodology mandates of the Yantra architectural specification.

---

### GAP 1 — DAG 100-Independent-Task Overhead

- **Current metrics in [`BENCHMARKS.md`](BENCHMARKS.md):**
  - `dag_theoretical_min_direct` (`p50 = 0.0049 us/node`, `p99 = 0.0049 us/node`, `203169443.3157 nodes/s [direct C++ pointer + atomic]`)
  - `dag_100node_diamond_noop` (`p50 = 0.1319 us/node`, `p99 = 0.1585 us/node`, `7581501.1372 nodes/s [noop-node orchestration]`)
  - `dag_1000_nodes_depth10_schedule` (`p50 = 0.0934 ms`, `p99 = 0.0959 ms`, `10710192.8906 nodes/s [noop-node orchestration]`)
- **Spec target:**
  - *"DAG with 100 independent tasks: wall-clock time $\le 1.2\times$ the theoretical minimum (i.e., orchestration overhead $< 20\%$)."*
- **What is currently measured:**
  - `dag_100node_diamond_noop` measures a 10-layer $\times$ 10-node dependent diamond topology via `DagScheduler::execute_indexed_dag`, and `dag_theoretical_min_direct` measures a raw C++ function-pointer loop without `PipelineDefinition` or `DagScheduler::execute` dispatch machinery. Neither measures 100 independent tasks through `DagScheduler::execute` compared against the single-task serial cost of the same dispatch machinery.
- **What must be measured:**
  - 100 noop tasks with no dependencies (`PipelineDefinition` with 100 root `Task` nodes), executed on the work-stealing pool via `DagScheduler::execute` at `worker_count` $\in \{4, 8, 12\}$, compared to:
    - **Step A (serial baseline):** The same 100 independent noop tasks executed serially through `DagScheduler::execute` with `worker_count = 1` (`serial_100_ms`).
    - **Step B (single-task serial cost):** 1 noop task executed through the same `DagScheduler::execute` machinery (`worker_count = 1`, median of 20 runs, `single_task_us`), defining the theoretical minimum $\text{ideal\_100\_ms}[N] = \text{single\_task\_us} \times \lceil 100 / N \rceil / 1000$.
    - **Step C (parallel execution & overhead ratio):** $\text{overhead}[N] = (\text{parallel\_100\_ms}[N] / \text{ideal\_100\_ms}[N]) - 1.0$ for $N \in \{4, 8, 12\}$, emitting:
      - `dag_100_independent_serial`
      - `dag_100_independent_parallel_w4`
      - `dag_100_independent_parallel_w8`
      - `dag_100_independent_parallel_w12`
      - `dag_100_independent_overhead_w4`
      - `dag_100_independent_overhead_w8` (primary acceptance metric, target $\le 0.20$)
      - `dag_100_independent_overhead_w12`
      - `dag_single_task_median_us`
- **Sanity check required:**
  - Every task's status in `DagExecutionReport::results` is `TaskStatus::Completed`.
  - Total task completions $== 100$ (no task lost).
  - No task ID appears twice in the completion log (no task duplicated).
  - Sum of per-task wall-clock durations $\le 5 \times \text{serial\_100\_ms}$.

---

### GAP 2 — HNSW and IVF-PQ at 1M

- **Current metrics in [`BENCHMARKS.md`](BENCHMARKS.md):**
  - `hnsw_build_10k_128d` (`p50 = 155.6586 ms`, `p99 = 162.9763 ms`, `64243.1374 vecs/s [CPU AVX2]`, `mem_overhead = 1.251953x`)
  - `hnsw_search_top10_ef64` (`p50 = 190.6945 us/query`, `p99 = 272.6051 us/query`, `Recall@10 = 0.975000` at $N = 10{,}000$)
  - `ivfpq_search_top10` (`p50 = 116.9105 us/query`, `p99 = 136.7516 us/query`, `Recall@10 = 0.957000` at $N = 10{,}000$)
- **Spec targets:**
  1. *"HNSW: 1M vectors, recall@10 $\ge 95\%$, p50 latency $\le 2\text{ ms}$, p99 $\le 10\text{ ms}$."*
  2. *"IVF-PQ: 1M vectors, recall@10 $\ge 90\%$, p50 latency $\le 0.5\text{ ms}$."*
  3. *"Build time for 1M vectors: $\le 60\text{ seconds}$ (HNSW), $\le 30\text{ seconds}$ (IVF-PQ)."*
  4. *"Loading a persisted 1M-vector HNSW index from disk (mmap) in $\le 500\text{ ms}$."*
  5. *"HNSW index memory overhead: $\le 1.5\times$ the raw vector storage (including graph links)."*
- **What is currently measured:**
  - $N = 10{,}000$ vectors of dimension $D = 128$ (`fp32`). At $10\text{K}$ vectors, the entire raw vector dataset is $5.12\text{ MB}$, fitting comfortably inside the i7-12700K's $25\text{ MB}$ L3 cache. Scale-dependent behaviors (DRAM bandwidth saturation, page-cache behavior, candidate list growth in IVF-PQ, and $O(N \cdot \text{sample\_budget})$ candidate evaluation in `HnswIndex::add_batch`) only manifest at $N = 1{,}000{,}000$ ($512\text{ MB}$ raw vector footprint).
- **What must be measured:**
  - $1{,}000{,}000 \times 128\text{d}$ `fp32` vectors generated deterministically with `std::mt19937_64(0xDEADBEEF)` ($\mathcal{N}(0, 1)$), persisted in `benchmarks/data/sift1m-like.bin`, and $1{,}000$ query vectors persisted in `benchmarks/data/queries-1k.bin` (loaded into RAM outside the timed intervals):
    - `hnsw_build_1m_128d` (target $\le 60\text{ s}$)
    - `hnsw_search_1m_ef64` (target $\text{p50} \le 2\text{ ms}$, mean $\text{Recall@10} \ge 0.95$, plus worst-case $\text{Recall@10}$ across 1,000 queries)
    - `ivfpq_build_1m_128d` (target $\le 30\text{ s}$)
    - `ivfpq_search_1m` (target $\text{p50} \le 0.5\text{ ms}$, mean $\text{Recall@10} \ge 0.90$)
    - `hnsw_persist_and_load_1m` (save 1M HNSW to disk, drop in-memory index, reload via `HnswIndex::load_mmap`, target $\le 500\text{ ms}$)
    - `hnsw_overhead_ratio` ($(\text{rss\_steady\_hnsw\_search\_1m\_ef64} - \text{rss\_bare\_hnsw\_search\_1m\_ef64}) / (10^6 \times 128 \times 4)$, target $\le 1.5\times$)
- **Sanity checks required:**
  - Recall@10 validated against exact `FlatIndex` search over the same $1{,}000{,}000$ vectors for the same $1{,}000$ queries (hard sanity gate: mean $\text{Recall@10} \ge 0.90$ for HNSW, $\ge 0.85$ for IVF-PQ).
  - Three-phase RSS reported per benchmark (`rss_bare_<benchmark>`, `rss_peak_<benchmark>`, `rss_steady_<benchmark>`).
  - HNSW memory overhead vs. raw vector storage ($512\text{ MB}$) reported and compared to the $\le 1.5\times$ target.
  - Persistence load time measured via `HnswIndex::load_mmap` and verified to have `index.size() == 1000000`.

---

### GAP 3 — Fairness Methodology

- **Current claim in [`BENCHMARKS.md`](BENCHMARKS.md):**
  - `fairness_audit_1m_rows_10groups`: `p50 = 0.3551 ms`, `p99 = 0.3785 ms`, `2816.0883 Mrows/s [multi-core reduction]` ($\approx 2.82\text{ billion rows/s}$).
- **Ambiguity:**
  - Does the timed interval include partitioning/grouping or sorting rows by sensitive attribute, or does it operate directly on integer group indices `[0..9]` into thread-local confusion matrix accumulators without materializing per-group row partitions? Furthermore, how much of the apparent throughput comes from running 8 worker threads over L3-resident arrays without materializing grouped row arrays?
- **Required disclosure:**
  1. State explicitly in [`BENCHMARKS_METHODOLOGY.md`](BENCHMARKS_METHODOLOGY.md) under `fairness_audit_1m_rows_10groups`:
     > *"The timed interval includes only the metric computation (demographic parity difference, equalized odds difference, equal opportunity difference, predictive parity) over pre-grouped input. Grouping/sorting by sensitive attribute is excluded and measured separately (see `fairness_audit_1m_rows_10groups_with_grouping` below)."*
  2. Add a second benchmark `fairness_audit_1m_rows_10groups_with_grouping` in [`bench/bench_yantra.cpp`](bench/bench_yantra.cpp) that **includes** single-pass counting-sort partitioning of the $1{,}000{,}000$ raw rows into 10 contiguous per-group arrays inside the timed interval prior to computing the four fairness metrics per group (target $\le 500\text{ ms}$).
  3. **Sanity check required:** Verify that `fairness_audit_1m_rows_10groups` and `fairness_audit_1m_rows_10groups_with_grouping` produce identical fairness metric values ($\Delta_{\text{DP}}$, $DI$, $\Delta_{\text{EO}}$, $\Delta_{\text{EOD}}$, $\Delta_{\text{PP}}$ within $10^{-12}$) on the exact same 1M-row input dataset.
