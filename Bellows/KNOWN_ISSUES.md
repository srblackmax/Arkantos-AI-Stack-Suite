# Bellows — Known Issues & Audit Log (`KNOWN_ISSUES.md`)

## 1. Architectural Scope Limitations (v0.1.0)
1. **Exact-Cut vs. Quantile-Bin Tree Growth:** `exact_cut_policy` performs an exact feature sort per node split (`56.885 ms` `p50` on `1200x8`, 40 trees), whereas `quantile_bin_policy` builds 64–256 quantile histograms once and uses sibling histogram subtraction (`3.508 ms` `p50`, `16.2x` faster). For datasets with `N > 1,000`, `quantile_bin_policy` should always be preferred.
2. **Batch Predict Parallelism:** `predict(booster, table)` evaluates rows sequentially on a single thread (`1.66M rows/s` on `N=10,000`), whereas `XGBoost` dispatches OpenMP threads across rows on large batches (`N >= 10,000`). Single-row online inference (`infer_one`, `187.4 ns/row`, zero heap allocation) is `506.9x` faster than single-row `scikit-learn` `predict()`.

---

## 2. Ecosystem Benchmark Integrity Audit Log (2026-11-01)

During the Ecosystem Benchmark Integrity Audit, the following benchmark documentation and harness gaps in Bellows were identified and remediated without modifying library source code (`src/*`, `include/*`):

1. **Missing `bench/bench_bellows.cpp` and `BENCHMARKS.md` (Checks 1.2–1.6 & §4.2):**
   - *Pre-audit state:* Bellows had `examples/workflow_demo.cpp` (correctness check on `N=40` rows) and qualitative speed claims in `README`/`NEWS`, but no dedicated benchmark harness, no `p50`/`p99` quantiles, and no `BENCHMARKS.md`.
   - *Remediation:* Added [`bench/bench_bellows.cpp`](./bench/bench_bellows.cpp), [`BENCHMARKS.md`](./BENCHMARKS.md), [`BENCHMARKS_METHODOLOGY.md`](./BENCHMARKS_METHODOLOGY.md), `benchmarks/baseline.json`, and the `bench-ci` target in `Makefile` and `CMakeLists.txt`.
2. **Explicit Separation of Feature Preprocessing vs. `fit()` (§4.2):**
   - *Remediation:* Timed `column_mender` + `moment_normaliser` (`0.245 ms` `p50`) separately from `stagewise_booster::fit()` (`3.508 ms` quantile-bin / `56.885 ms` exact-cut), and timed the combined 4-stage `operator|` pipeline separately in Metric 5 (`3.818 ms` `p50`).
3. **Explicit Batch Size Disclosure on Inference (§4.2):**
   - *Remediation:* Explicitly labeled batch throughput with `batch_size = 10,000` (`1,662,249 rows/s`, `6.016 ms`) and single-row latency with `batch_size = 1` (`187.4 ns/row`, `5,335,179 rows/s`).
