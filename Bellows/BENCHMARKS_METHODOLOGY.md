# Bellows — Benchmark Methodology (`BENCHMARKS_METHODOLOGY.md`)

This document specifies the exact measurement protocol, hardware baseline, timed intervals, exclusions, competitor baselines (`scikit-learn 1.5.2`, `XGBoost 2.1.1`, `LightGBM 4.5.0`), and post-measurement sanity checks for every benchmark metric in [`BENCHMARKS.md`](./BENCHMARKS.md).

---

## 0. Hardware & Software Environment Baseline

- **CPU:** `12th Gen Intel(R) Core(TM) i7-12700K` (12 physical cores: 8 Performance-cores + 4 Efficient-cores, 20 hardware threads, 800 MHz – 5,100 MHz turbo, `30 MiB` L3 cache, AVX2 / FMA3)
- **GPU:** `NVIDIA GeForce RTX 3070` (`8,192 MiB` GDDR6 VRAM, Driver `550.163.01` — unused by Bellows, which is CPU-native)
- **RAM:** `48 GiB` Dual-Channel DDR4
- **Storage:** `Samsung SSD 970 EVO Plus 1TB` NVMe (`nvme0n1`)
- **OS & Kernel:** Debian GNU/Linux 13 (trixie), Kernel `Linux 6.12.113+deb13-amd64 x86_64`
- **Compiler & Flags:** `GCC 14.2.0` (`-std=c++20 -O2 -g -Wall -Wextra -Wpedantic -pthread`) linked against `libhypermath.a`
- **Statistical Protocol:** Every benchmark executes `2–4` untimed warm-up batches followed by `9–15` timed sample batches (`batch_iters = 1` to `20`), recording both `p50` (median) and `p99` latencies.

---

## 1. Stagewise Booster Fit (`exact_cut_policy` vs `quantile_bin_policy`)

- **Dataset (`1,200 x 8`):** Deterministic tabular dataset with `1,200` rows and `8` columns (`7` continuous features with periodic missing entries `NaN` every 31st row on column 1, plus `1` nominal categorical column with cardinality `4`).
- **Timed Interval Split (§4.2):**
  - **1a. Feature Preprocessing (`column_mender` + `moment_normaliser`):** Timed separately (`p50 = 0.245 ms`, `p99 = 0.268 ms`) to quantify imputation (`mending_strategy::median`) and z-score scaling overhead independently of tree induction.
  - **1b. Booster Fit (`quantile_bin_policy` + `best_gain_frontier_growth`):** Timed on `x_clean` (`p50 = 3.508 ms`, `p99 = 3.543 ms`). Hyperparameters: `n_estimators = 40`, `learning_rate = 0.15`, `max_depth = 4`, `max_leaves = 15`, `min_samples_leaf = 4`, `max_bins = 64`, `l1_reg = 0.01`, `l2_reg = 1.0`. Includes quantile histogram bin edge construction, per-node histogram accumulation, and sibling histogram subtraction.
  - **1c. Booster Fit (`exact_cut_policy` + `depth_synchronous_growth`):** Timed on `x_clean` (`p50 = 56.885 ms`, `p99 = 66.570 ms`) with matching hyperparameters (`40` trees, `max_depth = 4`, `l1_reg = 0.01`, `l2_reg = 1.0`), sorting candidate split thresholds per feature per depth level (`16.2x` slower than quantile-bin histogram subtraction).
- **Exclusions:** Initial dataset synthesis (`x_raw`, `y_reg`) is excluded.
- **Sanity Check:** Verifies `r2_metric(y_reg, predict(booster, x_clean)) > 0.85` (actual `R^2 = 0.990` for quantile-bin and `0.991` for exact-cut) and `tree_count() == 40`.

---

## 2. Stagewise Booster Batch Predict Throughput (`batch_size = 10,000`)

- **Timed Interval:** Evaluates `predict(q_booster, x_batch)` over a contiguous `10,000 x 8` `data_table` through the `40`-tree fitted `stagewise_booster` (`400,000` tree traversals per call).
- **Batch Size Disclosure (§4.2):** `batch_size = 10,000` rows (`p50 = 6.016 ms`, `p99 = 6.040 ms`, throughput = `1,662,249 rows/s`).
- **Exclusions:** `x_batch` table allocation is excluded; output `std::vector<double>` (`10,000` elements) allocation and all `400,000` root-to-leaf traversals are included.
- **Sanity Check:** Verifies `batch_preds.size() == 10000`, `std::isfinite(batch_preds[0])`, and non-constant predictions (`|batch_preds[0] - batch_preds[1]| > 1e-6`).

---

## 3. Single-Row Zero-Allocation Inference (`infer_one`, `batch_size = 1`)

- **Timed Interval:** Executes `4,000` back-to-back single-row `infer_one(q_booster, single_row)` calls (`batch_size = 1`, `8` features, `40` trees) over an already-allocated `std::vector<double>` row buffer, mutating `single_row[0]` on each call to prevent loop hoisting.
- **Exclusions:** None (zero heap allocations occur inside `infer_one`).
- **Sanity Check:** Accumulates all `4,000` scalar predictions and verifies `std::isfinite(acc_pred)` and `std::abs(acc_pred) > 1.0`.

---

## 4. Penalised Linear & Logistic Estimators (`1,200 x 8`)

- **Timed Interval:**
  - **4a. `penalised_linear_regressor` (`fit()`):** Fits an L2-penalised (`lambda_2 = 1e-3`) linear model on `1,200 x 8` via normal-equation Gram formation (`X^T X + lambda I`) and Cholesky/LU solve (`p50 = 0.038 ms`, `p99 = 0.039 ms`).
  - **4b. `penalised_logistic_classifier` (`fit()`):** Fits an L2-penalised binary logistic classifier on `1,200 x 8` via iterative gradient/Newton updates (`p50 = 4.567 ms`, `p99 = 4.591 ms`).
- **Exclusions:** Feature preprocessing (`x_clean`) is excluded and reported separately in Metric 1a.
- **Sanity Check:** Verifies `R^2 > 0.70` (actual `R^2 = 0.980`) for `penalised_linear_regressor` and classification accuracy `> 0.85` (actual `Acc = 0.951`) for `penalised_logistic_classifier`.

---

## 5. Composite Pipeline End-to-End Fit (`column_mender | moment_normaliser | indicator_expander | stagewise_booster`)

- **Timed Interval:** Executes `fit(pipe, x_raw, y_reg)` on the raw `1,200 x 8` dataset (containing missing values and a 4-level nominal categorical column) through the 4-stage compile-time C++20 `operator|` pipeline chain, expanding the 8 input columns to 11 dense indicator columns before fitting the `40`-tree quantile-bin booster (`p50 = 3.818 ms`, `p99 = 3.926 ms`).
- **Exclusions:** Raw dataset generation is excluded; all intermediate `data_table` projections (`mender -> normaliser -> indicator_expander -> booster`) are included.
- **Sanity Check:** Verifies end-to-end `r2_metric(y_reg, predict(pipe, x_raw)) > 0.85` (actual `R^2 = 0.989`).

---

## 6. Process Memory RSS Phases

- **Timed Interval:** Reads `/proc/self/status` at three checkpoints:
  1. **Bare Process RSS (`VmRSS`):** First statement in `main()` (`3.24 MB`).
  2. **Post-Library-Load RSS (`VmRSS`):** After constructing the `1,200 x 8` dataset (`3.51 MB`).
  3. **Peak Workload RSS (`VmHWM`):** High-water mark resident set size across all 5 benchmark suites (`4.92 MB`).
