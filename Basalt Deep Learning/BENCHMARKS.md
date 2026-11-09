# Basalt Deep Learning — Audited Benchmark Report (`BENCHMARKS.md`)

All benchmarks were executed on the reference workstation (`12th Gen Intel Core i7-12700K` 12C/20T + `NVIDIA GeForce RTX 3070` 8 GiB GDDR6, Driver `550.163.01`, CUDA `12.4`, Debian Linux `6.12.113+deb13-amd64`, `GCC 14.2.0 -O3 -march=native -fopenmp`). Full timing protocols, layer configurations, exclusions, and post-measurement sanity checks are documented in [`BENCHMARKS_METHODOLOGY.md`](file://. Deep Learning/BENCHMARKS_METHODOLOGY.md).

---

## 1. CPU & CUDA Deep Learning Benchmarks

| Benchmark Workload & Exact Configuration | `p50` Latency | `p99` Latency | Throughput (`GFLOP/s`) | Warm Competitor Baseline (`PyTorch 2.4` / `cuBLAS` / `cuDNN 9` on same CPU/GPU)\* | Sanity Check |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **1. CPU `matmul` (`256x256`, `fp32`, `No_Grad`)**<br>AVX2+FMA 8-thread register-tiled GEMM | `0.090 ms` (warm)<br>`0.152 ms` (cold) | `0.111 ms` (warm)<br>`0.168 ms` (cold) | `373.3 GFLOP/s` (warm L2/L3)<br>`221.1 GFLOP/s` (cold DRAM)\*\* | `PyTorch 2.4` (`torch.matmul` CPU MKL/OpenBLAS 8T warm): `0.082 ms` (`409.2 GFLOP/s`) | `PASS` (`65,536` finite) |
| **2. CPU `Conv2d` Forward (`batch=4, 8->16, 16x16, K=3, S=1, P=1, fp32`)** | `0.075 ms` | `0.084 ms` | `31.46 GFLOP/s` (`53,333 img/s` CPU) | `PyTorch 2.4` (`nn.Conv2d` CPU oneDNN warm): `0.095 ms` (`1.27x` slower due to dispatch overhead on small spatial tensor) | `PASS` (`16,384` finite) |
| **3. CPU `MLP` Forward + Backward (`batch=32, 64->128->GELU->32, fp32`)** | `0.198 ms` | `0.253 ms` | `5,050 steps/s` (`161,616 samples/s`) | `PyTorch 2.4` (CPU eager `forward + mean + backward` warm): `0.480 ms` (`2.42x` slower due to Python autograd dispatch) | `PASS` (`loss & grad` finite) |
| **4. CPU `Transformer_Encoder_Layer` (`batch=4, seq=16, d_model=32, h=4, d_ff=64, fp32 fwd+bwd`)** | `0.322 ms` | `0.325 ms` | `3,106 steps/s` (`198,758 tok/s` train) | `PyTorch 2.4` (`nn.TransformerEncoderLayer` CPU eager fwd+bwd warm): `1.150 ms` (`3.57x` slower) | `PASS` (`loss & grad` finite) |
| **5. CUDA `matmul` (`1024x1024`, `fp32`, RTX 3070 resident VRAM)** | `0.215 ms` | `0.343 ms` | `9,974.2 GFLOP/s` (`9.97 TFLOP/s`)\*\*\* | `PyTorch 2.4` (`torch.matmul` CUDA `1024x1024` `fp32` resident warm): `0.212 ms` (`10,129 GFLOP/s` — identical `cuBLAS` kernel) | `PASS` (`1,048,576` finite) |
| **6. CUDA `TensorCore` (`2048x2048`, `fp16`, RTX 3070 resident VRAM)** | `0.495 ms` | `0.611 ms` | `34,709.3 GFLOP/s` (`34.71 TFLOP/s`)\*\*\* | `PyTorch 2.4` (`torch.matmul` CUDA `2048x2048` `fp16` resident warm): `0.490 ms` (`35,061 GFLOP/s` — identical `cuBLAS` Tensor Core kernel) | `PASS` (`4,194,304` finite) |

\* **Competitor Disclosure (§4.3):** All `PyTorch 2.4` (CUDA `12.4` / `cuDNN 9`) numbers are measured **warm** on the same i7-12700K + RTX 3070 workstation with identical tensor shapes, dtypes, and synchronization (`torch.cuda.synchronize()`). Note on **NativeEngineClient Q4_K Bridge**: as documented in Step 6 (`Sutra`), Basalt's native GGUF inference bridge dequantizes `Q4_K` blocks to `fp32` during GEMV (`~2.4 tok/s` on `Qwen2.5-0.5B-Instruct-Q4_K_M`) rather than invoking fused `dp4a`/TensorCore integer-quantized MMA kernels; the `34.71 TFLOP/s` Tensor Core rate above applies strictly to dense `fp16` `2048x2048` GEMM via `libcublas.so.12`.

\*\* **Physical-Bound Footnote (CPU `256x256` `fp32` MatMul — Check 1.9):** Two `256x256` `fp32` matrices occupy `512 KiB` (`768 KiB` with output `C`), fitting completely inside the i7-12700K's `1.25 MiB` per-P-core L2 cache. The theoretical peak `fp32` AVX2 FMA throughput of 8 Performance-cores at `4.9 GHz` (2× 256-bit FMA units/core = 32 `fp32` FLOPs/cycle/core) is `8 * 32 * 4.9 GHz = 1,254.4 GFLOP/s`. Basalt's `373.3 GFLOP/s` warm (`29.8%` of theoretical AVX2 FMA peak) and `221.1 GFLOP/s` cold (DRAM-evicted via 32 MiB thrash buffer) are well within physical hardware bounds.

\*\*\* **Physical-Bound Footnote (RTX 3070 GA104 CUDA & Tensor Core GEMM — Check 1.9):** The NVIDIA GeForce RTX 3070 (GA104, 46 SMs, 1.73–1.95 GHz boost clock) has a theoretical peak of `~20.3 TFLOP/s` for dense `fp32` shader FMA and `~40.6 TFLOP/s` (`~81.2 TFLOP/s` with 2:4 sparsity) for dense `fp16` 3rd-Gen Tensor Core MMA. Basalt's measured `9.97 TFLOP/s` on `1024x1024` `fp32` (`49.1%` of peak on a moderate `1024^2` grid) and `34.71 TFLOP/s` on `2048x2048` `fp16` (`85.5%` of dense Tensor Core peak via `cublasGemmEx`) respect GA104 silicon limits.

---

## 2. Process Memory Footprint (3-Phase Host RSS Split)

| Memory Phase | Measured Host RSS (`MB`) | Measurement Point & Disclosure |
| :--- | :---: | :--- |
| **7a. Bare Process RSS** | `4.00 MB` | `VmRSS` at `main()` entry before any `basalt` call |
| **7b. Post-Library-Load RSS** | `245.68 MB` | `VmRSS` after `basalt::seed_everything(42)` + CPU tensor warm-up (includes `libcuda.so.1` driver mapping and OpenMP thread stacks) |
| **7c. Post-CUDA-Init RSS** | `317.39 MB` | `VmRSS` after `Device::cuda(0)` context creation, `libcublas.so.12` handle init, 1 warm-up GPU GEMM, and 1s idle settle |

---

## 3. Superseded Pre-Audit Numbers

In accordance with the Ecosystem Benchmark Integrity Audit rules, the pre-audit single-average benchmark numbers from `NEWS` (lines 55–62) are preserved below as `superseded` and cross-referenced in [`KNOWN_ISSUES.md`](file://. Deep Learning/KNOWN_ISSUES.md):

| Pre-Audit Metric (`NEWS` v0.1.0) | Pre-Audit Value (`avg_ms`, no `p50`/`p99`) | Status | Replacement Row |
| :--- | :---: | :---: | :--- |
| `matmul (256x256 fp32, CPU AVX2+FMA)` | `0.111 ms/iter (303.380 GFLOP/s)` (warm/cold undisclosed) | **Superseded** | Section 1, Row 1 (`p50=0.090 ms` warm `373.3 GFLOP/s` \| `0.152 ms` cold `221.1 GFLOP/s`) |
| `Conv2d (4x8x16x16 -> 16, k=3, CPU)` | `0.088 ms/iter` (no `p50`/`p99`) | **Superseded** | Section 1, Row 2 (`p50=0.075 ms`, `p99=0.084 ms`) |
| `MLP (32x64->128->32 fwd+bwd, CPU)` | `0.205 ms/iter` (no `p50`/`p99`) | **Superseded** | Section 1, Row 3 (`p50=0.198 ms`, `p99=0.253 ms`) |
| `Transformer_Encoder (4x16x32, CPU)` | `0.341 ms/iter` (no `p50`/`p99`) | **Superseded** | Section 1, Row 4 (`p50=0.322 ms`, `p99=0.325 ms`) |
| `CUDA matmul (1024x1024 fp32, RTX 3070)` | `0.209 ms/iter (10261.162 GFLOP/s)` (single loop sync) | **Superseded** | Section 1, Row 5 (`p50=0.215 ms`, `p99=0.343 ms`, `9,974.2 GFLOP/s`) |
| `CUDA TensorCore (2048x2048 fp16, RTX 3070)` | `0.504 ms/iter (34109.968 GFLOP/s)` (single loop sync) | **Superseded** | Section 1, Row 6 (`p50=0.495 ms`, `p99=0.611 ms`, `34,709.3 GFLOP/s`) |

---

## 4. Reproducing & Verifying (`bench-ci`)

```bash
make check
make bench-ci
```
`make bench-ci` runs `./bench/bench_basalt --json-out=results.json --check-baseline=./benchmarks/baseline.json` and exits non-zero if any post-measurement sanity check fails or if any metric regresses by more than `15%` against `benchmarks/baseline.json`.
