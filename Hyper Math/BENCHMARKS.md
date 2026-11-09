# Hyper Math — Audited Benchmark Report (`BENCHMARKS.md`)

All benchmarks were executed on the reference workstation (`12th Gen Intel Core i7-12700K` 12C/20T, `48 GiB` Dual-Channel DDR4, Debian Linux `6.12.113+deb13-amd64`, `GCC 14.2.0 -O2 -std=c++20 -pthread`, 8 worker threads). Full timing protocols, exclusions, and post-measurement sanity checks are documented in [`BENCHMARKS_METHODOLOGY.md`](file://. Math/BENCHMARKS_METHODOLOGY.md).

---

## 1. Core Tensor, Scientific & Tabular Benchmarks

| Benchmark Workload | `p50` Latency | `p99` Latency | Secondary Throughput / Speedup | Python Reference (`NumPy 2.1` / `SciPy 1.14` / `Pandas 2.2` on same CPU) | Sanity Check |
| :--- | :---: | :---: | :--- | :--- | :---: |
| **1. 1M Element Fused UFunc** (`sin(x)*2.5 + cos(y) - 1.25`) | `3.296 ms` | `3.483 ms` | `303.4 Melem/s` (`1.79x` vs unfused 5-pass C++ `5.91 ms`)\* | `NumPy 2.1` (5-pass eager): `14.80 ms` (`4.49x` slower) | `PASS` (`|sum_diff| < 1e-6`) |
| **2. 256x256 Parallel MatMul** (`Blocked i-k-j`, `fp64`) | `0.794 ms` (warm)<br>`0.828 ms` (cold) | `0.895 ms` (warm)<br>`0.912 ms` (cold) | `42.25 GFLOP/s` (warm L2/L3)<br>`40.50 GFLOP/s` (cold DRAM)\*\* | `NumPy 2.1` (OpenBLAS 8T `dgemm`): `0.29 ms` (`115.7 GFLOP/s`) | `PASS` (`65,536` finite) |
| **3. 65,536-Point Radix-2 Complex FFT** (`fp64` -> `complex<double>`) | `1.751 ms` | `2.484 ms` | `37.43 MSample/s` (`1.05M` butterflies/call) | `SciPy 1.14` (`scipy.fft.fft`, PocketFFT 1T): `1.42 ms` | `PASS` (`N=65536` finite) |
| **4. 100K Row Lazy Filter + GroupBy + Prune** (3 cols -> 2 used) | `3.228 ms` | `3.385 ms` | `30.98 Mrows/s` (`1` column pruned via pushdown) | `Pandas 2.2` (eager `df[df.rev>100].groupby(...)`): `11.40 ms` (`3.53x` slower) | `PASS` (`4x3` output) |
| **5. 1000-D Reverse-Mode Autograd Gradient** (Rosenbrock chain) | `0.050 ms` | `0.052 ms` | `20,000 grads/s` (`1,000` vars/tape) | `PyTorch 2.4` (CPU eager `backward()`): `0.62 ms` (`12.4x` slower due to Python dispatch) | `PASS` (`N=1000` finite) |
| **6. 2000x2000 Sparse Conjugate Gradient** (`csr_matrix`, `tol=1e-10`) | `1.245 ms` | `1.265 ms` | `rel_residual = 3.12e-11` (`nnz = 5,998`) | `SciPy 1.14` (`scipy.sparse.linalg.cg`): `4.85 ms` (`3.90x` slower due to Python callback per iter) | `PASS` (`res < 1e-8`) |
| **7. 10,000 x 8-D QMC Halton Sequence** (`80,000` radical inverses) | `0.976 ms` | `0.985 ms` | `81.97 Mvals/s` | `SciPy 1.14` (`scipy.stats.qmc.Halton(8).random(10000)`): `3.40 ms` (`3.48x` slower) | `PASS` (`mean = 0.4998`) |
| **8. 50K Row SQL Context Query** (`SELECT + WHERE + ORDER BY + LIMIT 100`) | `2.000 ms` | `2.025 ms` | `25.00 Mrows/s` (includes SQL parser + execution) | `DuckDB 1.1` (in-memory 50K scan+top-N): `1.65 ms` / `Pandas 2.2`: `6.90 ms` | `PASS` (`100` rows) |

\* **Physical-Bound Footnote (1M Fused UFunc):** At `p50 = 3.296 ms`, reading two `8 MB` (`1,000,000 x 8B`) input tensors and writing one `8 MB` output tensor transfers `24 MB` in `3.296 ms` = `7.28 GB/s` (~14.2% of Dual-Channel DDR4 peak bandwidth, bounded by vectorized `sin`/`cos` transcendental evaluation across 8 threads). By contrast, the unfused 5-temporary C++ baseline transfers `80 MB` (`5` intermediate passes) in `5.91 ms`.

\*\* **Physical-Bound Footnote (256x256 Blocked MatMul):** Two `256 x 256` `fp64` matrices occupy `1.0 MiB` (`1.5 MiB` including `C`), fitting entirely within the i7-12700K's `10 MiB` aggregate P-core L2 / `30 MiB` shared L3 cache. At `42.25 GFLOP/s` across 8 threads (`5.28 GFLOP/s` per thread), throughput is 16.5% of the theoretical 8-core AVX2 FMA ceiling (`8 cores * 8 FLOP/cycle * 4.9 GHz = 313.6 GFLOP/s`), reflecting portable C++20 auto-vectorized blocked `i-k-j` loops without hand-written assembly micro-kernels.

---

## 2. Process Memory Footprint (3-Phase RSS Split)

| Memory Phase | Measured RSS (`MB`) | Measurement Point & Disclosure |
| :--- | :---: | :--- |
| **9a. Bare Process RSS** | `3.36 MB` | `VmRSS` at `main()` entry before thread-pool initialization |
| **9b. Post-Library-Load RSS** | `3.62 MB` | `VmRSS` after `global_thread_pool()` spawns 8 worker threads |
| **9c. Peak Workload RSS** | `95.64 MB` | `VmHWM` peak resident set across all 8 benchmarks (including `32 MiB` cold-cache thrash buffer and `100K`-row tabular columns) |

---

## 3. Superseded Pre-Audit Numbers

In accordance with the Ecosystem Benchmark Integrity Audit rules, pre-audit single-pass console numbers (which lacked `p50`/`p99` quantiles, warm/cold cache separation, unfused baseline comparison, and `BENCHMARKS.md`) are preserved below as `superseded` and cross-referenced in [`KNOWN_ISSUES.md`](file://. Math/KNOWN_ISSUES.md):

| Pre-Audit Metric (`bench_main.cpp` v0.1.0 initial) | Pre-Audit Single-Sample Value | Status | Replacement Row |
| :--- | :---: | :---: | :--- |
| `1M Element Fused UFunc` | `~3.3 ms` (`~303 Melem/s`, single run, no unfused baseline) | **Superseded** | Section 1, Row 1 (`p50=3.296 ms`, `p99=3.483 ms`, vs unfused `5.91 ms`) |
| `256x256 Parallel MatMul` | `~0.8 ms` (`~42 GFLOP/s`, single run, warm/cold undisclosed) | **Superseded** | Section 1, Row 2 (Warm `42.25 GFLOP/s` vs Cold `40.50 GFLOP/s`) |
| `65,536-Point Radix-2 Complex FFT` | `~1.75 ms` (single run, no `p50`/`p99`) | **Superseded** | Section 1, Row 3 (`p50=1.751 ms`, `p99=2.484 ms`) |
| `100K Row Lazy Filter + GroupBy + Prune` | `~3.2 ms` (single run, no `p50`/`p99`) | **Superseded** | Section 1, Row 4 (`p50=3.228 ms`, `p99=3.385 ms`) |
| `1000-D Reverse-Mode Autograd Gradient` | `~0.05 ms` (single run, no `p50`/`p99`) | **Superseded** | Section 1, Row 5 (`p50=0.050 ms`, `p99=0.052 ms`) |
| `2000x2000 Sparse Conjugate Gradient` | `~1.25 ms` (single run, no residual check) | **Superseded** | Section 1, Row 6 (`p50=1.245 ms`, `p99=1.265 ms`, `rel_res < 1e-8`) |
| `10,000 x 8-D QMC Halton Sequence` | `~0.98 ms` (single run, no `p50`/`p99`) | **Superseded** | Section 1, Row 7 (`p50=0.976 ms`, `p99=0.985 ms`) |
| `50K Row SQL Context SELECT+WHERE+ORDER` | `~2.0 ms` (single run, no `p50`/`p99`) | **Superseded** | Section 1, Row 8 (`p50=2.000 ms`, `p99=2.025 ms`) |

---

## 4. Reproducing & Verifying (`bench-ci`)

```bash
make all check
make bench-ci
```
`make bench-ci` runs `./bin/bench-hyper-math --json-out=results.json --check-baseline=./benchmarks/baseline.json` and exits non-zero if any post-measurement sanity check fails or if any metric regresses by more than `15%` against `benchmarks/baseline.json`.
