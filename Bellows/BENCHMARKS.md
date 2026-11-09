# Bellows — Audited Benchmark Report (`BENCHMARKS.md`)

All benchmarks were executed on the reference workstation (`12th Gen Intel Core i7-12700K` 12C/20T, `48 GiB` Dual-Channel DDR4, Debian Linux `6.12.113+deb13-amd64`, `GCC 14.2.0 -O2 -std=c++20 -pthread`). Full timing protocols, hyperparameter disclosures, exclusions, and post-measurement sanity checks are documented in [`BENCHMARKS_METHODOLOGY.md`](./BENCHMARKS_METHODOLOGY.md).

---

## 1. Classical ML & Gradient-Boosted Decision Tree Benchmarks

| Benchmark Workload | `p50` Latency | `p99` Latency | Throughput / Quality | Warm Python Competitor Baseline (`scikit-learn 1.5.2` / `XGBoost 2.1.1` / `LightGBM 4.5.0` on same CPU)\* | Sanity Check |
| :--- | :---: | :---: | :--- | :--- | :---: |
| **1a. Feature Preprocessing Only** (`column_mender` + `moment_normaliser`, `1200x8`) | `0.245 ms` | `0.268 ms` | `4.90 Mrows/s` (median imputation + z-score) | `scikit-learn 1.5.2` (`SimpleImputer` + `StandardScaler` warm): `1.85 ms` (`7.55x` slower) | `PASS` (`0` NaNs remaining) |
| **1b. Booster Fit (`quantile_bin_policy`)** (`1200x8`, `40` trees, `depth=4`, `64` bins) | `3.508 ms` | `3.543 ms` | `R^2 = 0.990` (`16.2x` faster than `exact_cut_policy`) | `LightGBM 4.5.0` (`n_estimators=40, max_depth=4, max_bin=64`, warm `Dataset`): `5.40 ms` / `XGBoost 2.1.1` (`tree_method="hist"`): `7.10 ms` | `PASS` (`R^2 = 0.990 > 0.85`) |
| **1c. Booster Fit (`exact_cut_policy`)** (`1200x8`, `40` trees, `depth=4`, exact cuts) | `56.885 ms` | `66.570 ms` | `R^2 = 0.991` (full per-node feature sort) | `scikit-learn 1.5.2` (`GradientBoostingRegressor(n_estimators=40, max_depth=4)`): `38.20 ms` / `XGBoost 2.1.1` (`tree_method="exact"`): `29.50 ms` | `PASS` (`R^2 = 0.991 > 0.85`) |
| **2. Booster Batch Predict (`batch_size = 10,000`)** (`10000x8`, `40` trees) | `6.016 ms` | `6.040 ms` | `1,662,249 rows/s` (`601.6 ns/row` batch) | `XGBoost 2.1.1` (`inplace_predict`, `N=10,000` warm): `2.85 ms` / `scikit-learn 1.5.2` (`predict`, `N=10,000`): `8.90 ms` | `PASS` (`10,000` finite) |
| **3. Single-Row Zero-Alloc Inference (`infer_one`, `batch_size = 1`)** (`1x8`, `40` trees) | `187.4 ns/row` (`0.750 ms` / 4K rows) | `189.3 ns/row` (`0.757 ms` / 4K rows) | `5,335,179 rows/s` (`0` heap allocs/call)\*\* | `scikit-learn 1.5.2` (`predict(x[0:1])` single-row Python call): `95,000 ns/row` (`506.9x` slower due to Python/NumPy validation overhead) | `PASS` (`4,000` finite) |
| **4a. Penalised Linear Fit (`penalised_linear_regressor`)** (`1200x8`, Ridge `L2=1e-3`) | `0.038 ms` | `0.039 ms` | `R^2 = 0.980` (`X^T X` normal eq + Cholesky/LU) | `scikit-learn 1.5.2` (`Ridge(alpha=1e-3).fit` warm): `0.78 ms` (`20.5x` slower) | `PASS` (`R^2 = 0.980 > 0.70`) |
| **4b. Penalised Logistic Fit (`penalised_logistic_classifier`)** (`1200x8`, Binary `L2=1e-3`) | `4.567 ms` | `4.591 ms` | `Acc = 0.951` | `scikit-learn 1.5.2` (`LogisticRegression().fit` L-BFGS warm): `3.90 ms` | `PASS` (`Acc = 0.951 > 0.85`) |
| **5. Composite Pipeline Fit (`operator|` 4-Stage Chain)** (`1200x8` -> `11` expanded cols) | `3.818 ms` | `3.926 ms` | `R^2 = 0.989` (impute + scale + 1-hot + 40-tree hist GBDT) | `scikit-learn 1.5.2` (`Pipeline` + `ColumnTransformer` + `HistGradientBoostingRegressor` warm): `19.60 ms` (`5.13x` slower) | `PASS` (`R^2 = 0.989 > 0.85`) |

\* **Competitor Disclosure (§4.2):** All Python competitor baselines (`scikit-learn 1.5.2`, `XGBoost 2.1.1`, `LightGBM 4.5.0`) are measured **warm** (post-`import`, post-JIT/OpenMP thread-pool initialization) with identical dataset dimensions (`N=1,200, D=8` for fit; `N=10,000` or `N=1` for predict) and matching tree hyperparameters (`n_estimators=40, max_depth=4, learning_rate=0.15, max_bins=64`).

\*\* **Physical-Bound Footnote (`infer_one` Single-Row Latency):** Traversing `40` binary decision trees of depth `4` (`160` scalar branch comparisons + `40` leaf weight additions) in `187.4 ns` corresponds to `4.69 ns` per tree (`~23` CPU clock cycles per tree at `4.9 GHz`), as the entire `40`-tree node array (`~15 KiB`) resides in the i7-12700K's `32 KiB` L1D cache with zero dynamic memory allocation.

---

## 2. Process Memory Footprint (3-Phase RSS Split)

| Memory Phase | Measured RSS (`MB`) | Measurement Point & Disclosure |
| :--- | :---: | :--- |
| **6a. Bare Process RSS** | `3.24 MB` | `VmRSS` at `main()` entry before dataset allocation |
| **6b. Post-Library-Load RSS** | `3.51 MB` | `VmRSS` after `1,200 x 8` `data_table` construction |
| **6c. Peak Workload RSS** | `4.92 MB` | `VmHWM` peak resident set across all 5 benchmark suites (including `10,000 x 8` batch prediction table) |

---

## 3. Superseded Pre-Audit Status

Prior to the Ecosystem Benchmark Integrity Audit (2026-11-01), Bellows had no dedicated `BENCHMARKS.md` or `bench/bench_bellows.cpp` binary; performance characteristics were only described qualitatively in `README` and `NEWS`. See [`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md) for the complete audit log.

---

## 4. Reproducing & Verifying (`bench-ci`)

```bash
make check
make bench-ci
```
`make bench-ci` runs `./bench-bellows --json-out=results.json --check-baseline=benchmarks/baseline.json` and exits non-zero if any post-measurement sanity check fails or if any metric regresses by more than `15%` against `benchmarks/baseline.json`.
