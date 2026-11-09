# Basalt Deep Learning — Known Issues & Audit Log (`KNOWN_ISSUES.md`)

## 1. Architectural Scope Limitations (v0.1.0)
1. **GGUF `Q4_K` Dequantizing GEMV vs. Dense `fp16` Tensor Core GEMM:** While Basalt achieves `34.71 TFLOP/s` on dense `2048x2048` `fp16` matrices via `libcublas.so.12` Tensor Cores (`cublasGemmEx`), quantized GGUF `Q4_K_M` weights loaded via downstream bridges (`Sutra::NativeEngineClient`) currently dequantize `Q4_K` super-blocks to `fp32` on the fly rather than executing fused integer `dp4a`/MMA kernels.
2. **CUDA Driver Footprint on Host RSS:** Calling `basalt::seed_everything()` probes CUDA availability via `libcuda.so.1`, mapping `~241 MB` of NVIDIA user-space driver pages into host virtual/resident memory (`Post-Library-Load RSS = 245.68 MB`, rising to `317.39 MB` after `libcublas.so.12` context initialization).

---

## 2. Superseded Benchmark Claims (Ecosystem Benchmark Integrity Audit)

During the Ecosystem Benchmark Integrity Audit (2026-11-01), the following pre-audit benchmark issues in `bench/bench_basalt.cpp` and `NEWS` were identified and remediated without modifying library source code (`src/*`, `include/*`):

1. **Mean-Only Timing Without `p50`/`p99` Quantiles or Per-Sample CUDA Synchronization (Check 1.2 & §4.3):**
   - *Pre-audit state:* `bench/bench_basalt.cpp` computed a single mean (`avg_ms = elapsed / iters`) with no `p50`/`p99` distribution, and only called `device_synchronize(gpu)` once after all 20 loop iterations.
   - *Remediation:* Updated [`bench/bench_basalt.cpp`](file://. Deep Learning/bench/bench_basalt.cpp) to record 15 synchronized multi-iteration batches per workload, reporting `p50` and `p99` in [`BENCHMARKS.md`](file://. Deep Learning/BENCHMARKS.md).
2. **Undisclosed Warm L2 Cache Residence on `256x256` `fp32` CPU MatMul (`303.38 GFLOP/s` in `NEWS` — Check 1.9 & §4.3):**
   - *Pre-audit state:* `NEWS` line 56 reported `303.380 GFLOP/s` on `256x256 fp32` without disclosing that the `512 KiB` input working set fits inside L2 cache or providing a cold-cache measurement.
   - *Remediation:* Added both warm L2/L3 (`373.3 GFLOP/s`, `p50 = 0.090 ms`) and cold DRAM (`221.1 GFLOP/s`, `p50 = 0.152 ms` after a `32 MiB` cache-thrash sweep) measurements plus theoretical AVX2 FMA ceiling calculations (`1,254.4 GFLOP/s`) in `BENCHMARKS.md`.
3. **Post-Measurement Sanity Checks & 3-Phase RSS (Checks 1.5 & 1.6):**
   - *Pre-audit state:* `bench_basalt.cpp` discarded tensor outputs with `(void)` and did not report RSS phases.
   - *Remediation:* Added finite element/loss/gradient assertions on all 6 benchmarks and 3-phase RSS tracking (`Bare = 4.00 MB`, `Post-Library-Load = 245.68 MB`, `Post-CUDA-Init = 317.39 MB`).
