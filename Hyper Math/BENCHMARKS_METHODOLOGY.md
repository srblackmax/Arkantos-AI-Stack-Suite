# Hyper Math — Benchmark Methodology (`BENCHMARKS_METHODOLOGY.md`)

This document specifies the exact measurement protocol, hardware baseline, timed intervals, exclusions, competitor baselines, and post-measurement sanity checks for every benchmark metric in [`BENCHMARKS.md`](file://. Math/BENCHMARKS.md).

---

## 0. Hardware & Software Environment Baseline

- **CPU:** `12th Gen Intel(R) Core(TM) i7-12700K` (12 physical cores: 8 Performance-cores + 4 Efficient-cores, 20 hardware threads, 800 MHz – 5,100 MHz turbo, `30 MiB` L3 cache, AVX2 / FMA3; thread pool configured to `8` worker threads)
- **GPU:** `NVIDIA GeForce RTX 3070` (`8,192 MiB` GDDR6 VRAM, Driver `550.163.01` — unused by Hyper Math, which is CPU-native)
- **RAM:** `48 GiB` Dual-Channel DDR4 (`~51.2 GB/s` theoretical peak memory bandwidth)
- **Storage:** `Samsung SSD 970 EVO Plus 1TB` NVMe (`nvme0n1`)
- **OS & Kernel:** Debian GNU/Linux 13 (trixie), Kernel `Linux 6.12.113+deb13-amd64 x86_64`
- **Compiler & Flags:** `GCC 14.2.0` (`-std=c++20 -O2 -g -Wall -Wextra -Wpedantic -pthread`)
- **Statistical Protocol:** Every benchmark executes `4–6` untimed warm-up batches followed by `15` timed sample batches (`batch_iters = 4` to `40`), recording both `p50` (median) and `p99` latencies in milliseconds (`ms`).

---

## 1. 1M Element Fused UFunc (`sin(x)*2.5 + cos(y) - 1.25`)

- **Timed Interval:** Evaluates the lazily fused CRTP expression template `out = sin(x) * 2.5 + cos(y) - 1.25` over two pre-allocated `1,000,000`-element `double` (`fp64`) 64-byte aligned `nd_tensor<double>` arrays (`8 MB` each, `24 MB` total R/W traffic per fused pass) across the 8-worker thread pool.
- **Unfused Baseline Measured:** In the same binary (`bench_main.cpp`), we time the explicit 5-temporary unfused evaluation (`t1 = sin(x); t2 = t1 * 2.5; t3 = cos(y); t4 = t2 + t3; unfused_out = t4 - 1.25;`) over the exact same `1,000,000`-element inputs (`p50 = 5.91 ms`, `p99 = 6.24 ms`), yielding a measured **1.79x speedup** from single-pass CRTP loop fusion (eliminating `32 MB` of intermediate heap allocations and cache writes).
- **Exclusions:** Input `linspace<double>` allocation is excluded from the timed interval; output `slab_buffer<double>` allocation is included.
- **Sanity Check:** Verifies `std::isfinite(sum(out))` and `std::abs(sum(out) - sum(unfused_out)) < 1e-6`.

---

## 2. 256x256 Parallel Matrix Multiply (`Blocked i-k-j`)

- **Timed Interval:** Computes `c = matmul(a, b)` for two `256 x 256` `fp64` dense matrices (`512 KiB` per matrix, `1.5 MiB` working set, `2 * 256^3 = 33,554,432` FLOPs per call) using cache-blocked (`BLOCK = 32`) `i-k-j` loop order dispatched across 8 worker threads.
- **Warm vs. Cold Cache Disclosure:**
  - **Warm L2/L3 Cache (`42.25 GFLOP/s`, `p50 = 0.794 ms`):** Measured when `a` and `b` (`1.0 MiB` total) remain resident in the i7-12700K's `1.25 MiB` per-P-core L2 / `30 MiB` shared L3 cache across consecutive calls.
  - **Cold Cache (`40.50 GFLOP/s`, `p50 = 0.828 ms`):** Measured after sweeping a `32 MiB` (`4,194,304` `fp64`) cache-thrash buffer before each `matmul(a, b)` invocation to evict `a` and `b` to DRAM.
- **Exclusions:** Random uniform input generation is excluded from the timed interval.
- **Sanity Check:** Verifies `c.size() == 65536` and `std::isfinite(sum(c))`.

---

## 3. 65,536-Point Radix-2 Complex FFT

- **Timed Interval:** Computes `spec = fft::fft(sig)` over a `65,536`-element `double` input signal using iterative bit-reversal permutation and Cooley-Tukey Radix-2 butterfly stages (`16` stages, `N log2(N) = 1,048,576` complex butterfly operations).
- **Exclusions:** Pseudo-random normal input signal synthesis (`random_normal`) is excluded.
- **Sanity Check:** Verifies `spec.size() == 65536` and `std::isfinite(spec[0].real())`.

---

## 4. 100K Row Lazy Filter + GroupBy + Projection Pushdown

- **Timed Interval:** Executes `lazy_frame(df).filter(col("revenue") > 100.0).group_by("region").agg({col("revenue").sum(), col("revenue").mean()}).collect()` over a `100,000`-row, 3-column `data_frame` (`region` string column, `revenue` `fp64` column, `unused_col` `fp64` column pruned by the query optimizer).
- **Exclusions:** Construction of the `100,000`-row input `data_frame` is excluded from the timed interval.
- **Sanity Check:** Verifies `res.nrows() == 4` and `res.ncols() == 3`.

---

## 5. 1000-D Reverse-Mode Autograd Gradient

- **Timed Interval:** Constructs the dynamic reverse-mode computation tape for a `1,000`-variable chained Rosenbrock objective (`999` coupled nonlinear terms) and executes the backward pass via `autograd::grad(..., x0)` returning a `1,000`-element `nd_tensor<double>` gradient vector.
- **Exclusions:** Initial seed vector allocation (`x0`) is excluded; tape node allocation, forward evaluation, and reverse accumulation are all included.
- **Sanity Check:** Verifies `g.size() == 1000`, `std::isfinite(sum(g))`, and `std::abs(sum(g)) > 1.0`.

---

## 6. 2000x2000 Sparse Conjugate Gradient (`cg_solve`)

- **Timed Interval:** Solves the `2,000 x 2,000` symmetric positive-definite (SPD) tridiagonal CSR linear system `A x = b` (`nnz = 5,998`, diagonal `2.05`, off-diagonal `-1.0`) to relative tolerance `1e-10` (`max_iter = 500`) using `sparse::cg_solve`.
- **Exclusions:** CSR matrix assembly (`indptr`, `indices`, `data`) is excluded.
- **Sanity Check:** Computes the true residual vector `r = A * x_sol - b` via `a_sp.spmv(x_sol)` and verifies `||A x - b||_2 / sqrt(N) < 1e-8`.

---

## 7. 10,000 x 8-D QMC Halton Sequence

- **Timed Interval:** Generates `10,000` points in `8` dimensions (`80,000` radical-inverse evaluations over the first 8 prime bases `2, 3, 5, 7, 11, 13, 17, 19`) via `stats::qmc::halton(10000, 8)`.
- **Exclusions:** None.
- **Sanity Check:** Verifies `pts.size() == 80000` and uniform sample mean `0.45 < mean(pts) < 0.55`.

---

## 8. 50K Row SQL Context Query (`SELECT + WHERE + ORDER BY + LIMIT`)

- **Timed Interval:** Parses and executes `"SELECT id, score FROM bench_tbl WHERE score > 500 ORDER BY score DESC LIMIT 100"` via `sql_context::execute()` over a `50,000`-row registered table.
- **Exclusions:** Table registration (`ctx.register_table`) is excluded; SQL lexing, AST parsing, filter execution, sorting, and `LIMIT 100` materialization are included.
- **Sanity Check:** Verifies `sql_out.nrows() == 100`.

---

## 9. Process Memory RSS Phases

- **Timed Interval:** Reads `/proc/self/status` at three well-defined lifecycle checkpoints:
  1. **Bare Process RSS (`VmRSS`):** First statement in `main()` before thread-pool initialization (`3.36 MB`).
  2. **Post-Library-Load RSS (`VmRSS`):** Immediately after `global_thread_pool()` spawns 8 worker threads (`3.62 MB`).
  3. **Peak Workload RSS (`VmHWM`):** High-water mark resident set size across all 8 benchmark suites including `32 MiB` cold-cache thrash buffers and `100K`-row tabular frames (`95.64 MB`).
