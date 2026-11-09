# Basalt Deep Learning — Benchmark Methodology (`BENCHMARKS_METHODOLOGY.md`)

This document specifies the exact measurement protocol, hardware baseline, timed intervals, exclusions, layer configurations, competitor baselines (`PyTorch 2.4` / `cuDNN 9`), and post-measurement sanity checks for every benchmark metric in [`BENCHMARKS.md`](file://. Deep Learning/BENCHMARKS.md).

---

## 0. Hardware & Software Environment Baseline

- **CPU:** `12th Gen Intel(R) Core(TM) i7-12700K` (12 physical cores: 8 Performance-cores + 4 Efficient-cores, 20 hardware threads, 800 MHz – 5,100 MHz turbo, `30 MiB` L3 cache, AVX2 / FMA3; OpenMP configured to 8 P-core threads)
- **GPU:** `NVIDIA GeForce RTX 3070` (GA104, `46` SMs, `5,888` FP32 CUDA Cores, `184` 3rd-Gen Tensor Cores, `8,192 MiB` GDDR6 VRAM, Driver `550.163.01`, CUDA `12.4`, `libcublas.so.12`)
- **RAM:** `48 GiB` Dual-Channel DDR4
- **Storage:** `Samsung SSD 970 EVO Plus 1TB` NVMe (`nvme0n1`)
- **OS & Kernel:** Debian GNU/Linux 13 (trixie), Kernel `Linux 6.12.113+deb13-amd64 x86_64`
- **Compiler & Flags:** `GCC 14.2.0` (`-O3 -march=native -funroll-loops -fopenmp -Wall -Wextra -Wpedantic`)
- **Statistical Protocol:** Every benchmark executes `4–6` untimed warm-up batches followed by `15` timed sample batches (`batch_iters = 6` to `20`), recording both `p50` (median) and `p99` latencies in milliseconds (`ms`). All CUDA benchmarks execute `basalt::device_synchronize(gpu)` inside each timed sample batch so `p50` and `p99` reflect true device completion rather than asynchronous host launch overhead.

---

## 1. CPU `matmul` (`256x256` `fp32`, AVX2+FMA + OpenMP)

- **Configuration (§4.3):** Dense matrix multiplication `C = A * B` for `A, B in R^{256 x 256}` (`Dtype::Float32`, `256 KiB` per matrix, `768 KiB` total working set, `2 * 256^3 = 33,554,432` FLOPs per call) under `No_Grad_Guard`.
- **Warm vs. Cold Cache Disclosure (Check 1.9):**
  - **Warm L2/L3 Cache (`373.3 GFLOP/s`, `p50 = 0.090 ms`, `p99 = 0.111 ms`):** Measured when `A` and `B` (`512 KiB` combined) remain resident in the i7-12700K's `10 MiB` aggregate P-core L2 / `30 MiB` shared L3 cache across consecutive calls.
  - **Cold DRAM Cache (`221.1 GFLOP/s`, `p50 = 0.152 ms`):** Measured after sweeping a `32 MiB` (`8,388,608` `fp32`) cache-thrash buffer before each `matmul(a, b)` invocation to evict `A` and `B` to main memory.
- **Exclusions:** Random normal tensor initialization (`randn`) is excluded; output tensor slab allocation (`slab_buffer`) is included.
- **Sanity Check:** Verifies `c.numel() == 65536` and `std::isfinite(c.read_scalar_at(0))`.

---

## 2. CPU `Conv2d` Forward (`4x8x16x16 -> 16, K=3`)

- **Configuration (§4.3):** `basalt::nn::Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)` evaluated in `fp32` under `No_Grad_Guard` on input shape `[batch=4, C_in=8, H=16, W=16]` producing `[4, 16, 16, 16]` via `im2col` unfolding + OpenMP AVX2 GEMM (`2 * 4 * 16 * 16 * 16 * (8 * 3 * 3) = 2,359,296` FLOPs).
- **Exclusions:** Module weight initialization and input synthesis are excluded.
- **Sanity Check:** Verifies `out.numel() == 16384` and `std::isfinite(out.read_scalar_at(0))`.

---

## 3. CPU `MLP` Forward + Reverse-Mode Autograd Backward (`32x64 -> 128 -> GELU -> 32`)

- **Configuration (§4.3):** 2-layer feed-forward network `Sequential(Linear(64, 128), GELU(), Linear(128, 32))` in `fp32` with `batch_size = 32` (`input [32, 64]`).
- **Timed Interval:** Includes `mlp.zero_grad()`, `out = mlp.forward(x)`, `l = ops::mean(out)`, `l.backward()`, and scalar loss extraction `l.item_double()`.
- **Exclusions:** Optimizer step (`SGD`/`AdamW`) is excluded to isolate forward + reverse-mode autograd tape construction and backward gradient accumulation.
- **Sanity Check:** Verifies `std::isfinite(last_loss)` and `mlp.parameters()[0].grad().defined()`.

---

## 4. CPU `Transformer_Encoder_Layer` Forward + Backward (`batch=4, seq=16, d_model=32, nhead=4, d_ff=64`)

- **Configuration (§4.3):** Single `basalt::nn::Transformer_Encoder_Layer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0)` in `fp32` on input tensor `[batch=4, seq_len=16, d_model=32]`.
- **Timed Interval:** Includes `enc.zero_grad()`, multi-head self-attention + residual `Layer_Norm` + FFN forward pass, `ops::mean(out)`, and full reverse-mode `backward()` pass through attention and FFN weights.
- **Exclusions:** Token embedding lookup is excluded.
- **Sanity Check:** Verifies `std::isfinite(last_loss)` and `enc.parameters()[0].grad().defined()`.

---

## 5. CUDA `matmul` (`1024x1024` `fp32`, RTX 3070 `cuBLAS` Resident VRAM)

- **Configuration (§4.3):** Dense matrix multiplication `C = A * B` for `A, B in R^{1024 x 1024}` (`Dtype::Float32`, `4 MiB` per matrix in GDDR6 VRAM, `2 * 1024^3 = 2,147,483,648` FLOPs per call) dispatched to `cublasSgemm` on `Device::cuda(0)` (`NVIDIA GeForce RTX 3070`).
- **Timed Interval:** Executes 10 `ops::matmul(a, b)` dispatches followed by `basalt::device_synchronize(gpu)` per timed sample, dividing elapsed time by 10.
- **Exclusions:** Host-to-device (`H2D`) and device-to-host (`D2H`) PCIe transfers and initial `cuBLAS` handle creation are excluded (`A` and `B` are resident on `cuda:0`).
- **Sanity Check:** Reads `c.read_scalar_at(0)` back to host after timing and verifies `c.numel() == 1048576` and `std::isfinite(c0)`.

---

## 6. CUDA `TensorCore` `matmul` (`2048x2048` `fp16`, RTX 3070 Tensor Core GEMM)

- **Configuration (§4.3):** Dense half-precision matrix multiplication `C = A * B` for `A, B in R^{2048 x 2048}` (`Dtype::Float16`, `8 MiB` per matrix in GDDR6 VRAM, `2 * 2048^3 = 17,179,869,184` FLOPs per call) dispatched to `cublasGemmEx` (`CUBLAS_GEMM_DEFAULT_TENSOR_OP`) on `Device::cuda(0)` (`NVIDIA GeForce RTX 3070`).
- **Timed Interval:** Executes 10 `ops::matmul(ah, bh)` dispatches followed by `basalt::device_synchronize(gpu)` per timed sample, dividing elapsed time by 10.
- **Exclusions:** PCIe H2D/D2H transfers and `fp32 -> fp16` input casting are excluded.
- **Sanity Check:** Verifies `ch.numel() == 4194304` and `std::isfinite(ch.read_scalar_at(0))`.

---

## 7. Process Memory RSS Phases

- **Timed Interval:** Reads `VmRSS` from `/proc/self/status` at three checkpoints:
  1. **Bare Process RSS (`4.00 MB`):** First statement in `main()` before any `basalt` call.
  2. **Post-Library-Load RSS (`245.68 MB`):** After `seed_everything(42)` and initial CPU tensor warm-up (note: `basalt::seed_everything` probes CUDA driver availability via `dlopen("libcuda.so.1")` and initializes OpenMP worker stacks).
  3. **Post-CUDA-Init RSS (`317.39 MB`):** After `Device::cuda(0)` allocation, `libcublas.so.12` handle initialization, 1 warm-up GPU GEMM, and a 1-second idle settle.
