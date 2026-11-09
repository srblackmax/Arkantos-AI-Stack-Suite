# Hyper Math — Known Issues & Superseded Benchmark Log (`KNOWN_ISSUES.md`)

## 1. Architectural Scope Limitations (v0.1.0)
1. **Dense `matmul` Kernel:** `scientific::linalg::matmul` uses a portable C++20 8-thread cache-blocked (`BLOCK=32`) `i-k-j` loop (`42.25 GFLOP/s` warm / `40.50 GFLOP/s` cold on `256x256` `fp64`) rather than hand-written AVX2 register-tiled assembly micro-kernels (`OpenBLAS` achieves `115.7 GFLOP/s` on the same CPU).
2. **Power-of-Two Radix-2 FFT:** `scientific::fft::fft` uses an in-place Cooley-Tukey Radix-2 butterfly kernel for power-of-two lengths and falls back to Bluestein's chirp-z algorithm for arbitrary prime lengths.
3. **CPU-Only Execution:** Hyper Math is a pure C++20 CPU library with zero GPU dependencies; GPU tensor acceleration is delegated to **Basalt Deep Learning** (Step 3).

---

## 2. Superseded Benchmark Claims (Ecosystem Benchmark Integrity Audit)

During the Ecosystem Benchmark Integrity Audit (2026-11-01), the following pre-audit benchmark issues in `benchmarks/bench_main.cpp` were identified and remediated without modifying library source code (`src/*`, `include/*`):

1. **Missing `BENCHMARKS.md` and `p50`/`p99` Latency Quantiles (Check 1.2):**
   - *Pre-audit state:* `benchmarks/bench_main.cpp` executed a single warm-up iteration and a single timed iteration (`repeats = 1`), printing one raw millisecond value to stdout with no `BENCHMARKS.md`.
   - *Remediation:* Updated `benchmarks/bench_main.cpp` to run 15 multi-iteration batches per workload, reporting `p50` and `p99` in [`BENCHMARKS.md`](file://. Math/BENCHMARKS.md) and [`BENCHMARKS_METHODOLOGY.md`](file://. Math/BENCHMARKS_METHODOLOGY.md).
2. **Unpaired 1M Fused UFunc Claim (Check 1.4 & §4.1):**
   - *Pre-audit state:* `1M Element Fused UFunc` reported `~3.3 ms` (`~303 Melem/s`) without measuring the unfused 5-pass baseline on the same data.
   - *Remediation:* Added an explicit 5-temporary unfused evaluation (`5.91 ms` `p50`) in `benchmarks/bench_main.cpp`, establishing a verified `1.79x` single-pass fusion speedup.
3. **Warm vs. Cold Cache Disclosure on `256x256` MatMul (§4.1):**
   - *Pre-audit state:* `256x256 Parallel MatMul` reported `~42 GFLOP/s` without disclosing that `256x256` `fp64` matrices (`1.0 MiB` total) fit entirely in L2/L3 cache.
   - *Remediation:* Added a `32 MiB` cache-thrash sweep between cold-cache iterations in `benchmarks/bench_main.cpp` and reported both Warm L2/L3 (`42.25 GFLOP/s`) and Cold DRAM (`40.50 GFLOP/s`) numbers in [`BENCHMARKS.md`](file://. Math/BENCHMARKS.md).
4. **Post-Measurement Sanity Checks & 3-Phase RSS (Checks 1.5 & 1.6):**
   - *Pre-audit state:* `bench_main.cpp` only cast outputs to `(void)` without verifying numerical correctness or RSS phases.
   - *Remediation:* Added numerical sanity assertions on all 8 workloads and 3-phase RSS reporting (`Bare = 3.36 MB`, `Post-Library-Load = 3.62 MB`, `Peak-Workload = 95.64 MB`).
