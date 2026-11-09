# Surya Production Benchmark Report (`BENCHMARKS.md`)

**Library:** Surya (`1.0.0`) — Native C++23 Computer Vision & Hybrid Linear-Attention VLM Pipeline  
**Audit Date:** `2026-11-01` (see [BENCHMARKS_METHODOLOGY.md](BENCHMARKS_METHODOLOGY.md) and [KNOWN_ISSUES.md](KNOWN_ISSUES.md))  
**CPU:** `12th Gen Intel(R) Core(TM) i7-12700K` (8 P-cores + 4 E-cores, 20 logical threads, AVX2 + FMA)  
**GPU:** `NVIDIA GeForce RTX 3070` (`GA104`, `sm_86`, `8,192 MiB` GDDR6, Driver `550.163.01`, CUDA `12.4.131`)  
**OS & Kernel:** Debian GNU/Linux 13 (trixie), Kernel `6.12.113+deb13-amd64` (`x86_64`)  
**Toolchain:** GCC 14.2.0 (`-O3 -std=c++23`), CUDA 12.4.131 (`-O3 -arch=sm_86`), Intel oneTBB 2022.1.0, CMake 3.31.6  
**Benchmark Command:** `./build-release-cuda/bench_surya`  
**Regression CI Target:** `cmake --build build-release-cuda --target bench-ci`  

---

## 1. Single-Image VLM Latency (`batch = 1`): CPU-Only vs. CPU+GPU Hybrid

> **Architecture & Scope Disclosure:** Measured on `640x480` RGB document frames through Surya's compact 24-layer Qwen3.5-family Hybrid Linear-Attention VLM pipeline (`256.00 MiB` INT8 weights: preprocess + 24-layer hybrid Gated DeltaNet/GQA + unified OCR/Layout/Table heads + lightweight text-detection NMS + OCR-error head). All operations execute offscreen without swapchain presentation; throughputs are reported in `img/s`.

| Execution Mode | `p50` Latency ($\mu\text{s}$) | `p99` Latency ($\mu\text{s}$) | Throughput (`img/s`) | Speedup vs. CPU (`p50`) |
| :--- | :---: | :---: | :---: | :---: |
| **CPU-Only (`oneTBB + AVX2`)** | `5,253.25 us` | `5,371.97 us` | `190.36 img/s (CPU)` | `1.00x` |
| **CPU+GPU Hybrid (No CUDA Graph)** | `394.02 us` | `601.24 us` | `2,537.95 img/s (offscreen GPU)` | `13.33x` |
| **CPU+GPU Hybrid (CUDA Graph Replay)** | **`348.59 us`** | **`478.40 us`** | **`2,868.69 img/s (offscreen GPU)`** | **`15.07x`** |

---

## 2. Batched Hybrid VLM Throughput (`batch = 1, 4, 8, 16, 32`) & Overlapped Streaming

| Batch Size / Pipeline Mode | `p50` Batch Latency ($\mu\text{s}$) | `p99` Batch Latency ($\mu\text{s}$) | Per-Image `p50` ($\mu\text{s}/\text{img}$) | Throughput (`img/s [offscreen GPU]`) | Active GPU Utilization |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`batch = 1` (CUDA Graph)** | `356.06 us` | `519.12 us` | `356.06 us` | `2,808.48 img/s` | `74.2%` |
| **`batch = 4` (CUDA Graph)** | `998.06 us` | `1,213.53 us` | `249.52 us` | `4,007.76 img/s` | `86.5%` |
| **`batch = 8` (CUDA Graph)** | `1,761.35 us` | `2,007.46 us` | `220.17 us` | `4,541.97 img/s` | `91.4%` |
| **`batch = 16` (CUDA Graph)** | `3,279.06 us` | `3,622.46 us` | `204.94 us` | `4,879.45 img/s` | `94.1%` |
| **`batch = 32` (CUDA Graph)** | `6,460.74 us` | `6,803.48 us` | `201.90 us` | `4,952.99 img/s` | `96.3%` |
| **Overlapped Multi-Stream (`128 imgs, b=16`)** | `24,036.22 us` | `30,361.63 us` | **`187.78 us`** | **`5,325.30 img/s`** | **`98.0%`** ($\ge 85\%$ target) |

---

## 3. Host RSS (Three-Stage Split) & GPU VRAM Residency (8 GB RTX 3070)

### 3.1. Three-Stage Host Process RSS (`VmRSS` via `/proc/self/status`, Check 1.4)

| Host Memory Stage | Measured `VmRSS` (`MB`) | What Is Resident in Host Address Space |
| :--- | :---: | :--- |
| **1. Bare Process RSS (`rss_bare_process_mb`)** | **`8.41 MB`** | Sampled on the first line of `main()` before any Surya pool or pipeline allocation (`libsurya.so`, `libtbb.so`, `libcudart.so` mapped). |
| **2. Post-Library-Load RSS (`rss_post_library_load_mb`)** | **`20.21 MB`** | Sampled after `Aligned_Buffer_Pool` and `1080p` / `720p` CPU image processing buffers are initialized (prior to CUDA `SuryaPipeline` initialization). |
| **3. Post-GPU-Init RSS (`rss_post_gpu_init_mb`)** | **`1,827.18 MB`** | Sampled after `SuryaPipeline` initializes its CUDA primary context, 8-slot pinned host-staging ring buffers (`max_batch_size=32`), and CPU fallback weights + `1.0 s` idle. |

### 3.2. GPU Device VRAM Residency (`NVIDIA GeForce RTX 3070`, `8,192 MiB`)

| VRAM Metric | Measured Value | Budget / Target | Status |
| :--- | :---: | :---: | :--- |
| **Configured Hard VRAM Cap** | `6,656.00 MiB` (`6.50 GiB`) | `6.50 GiB` (leaves `1.50 GiB` OS/display headroom) | Enforced via `VramBudgetAllocator` |
| **Resident INT8 Model Weights** | `256.00 MiB` | Loaded once per process (`SuryaPipeline::create`) | Zero per-inference weight reloads |
| **Steady-State VRAM (`max_batch_size=32`, 8 slots)** | `2,090.89 MiB` (`2.04 GiB`) | $\le 6.50\text{ GiB}$ | `0 bytes` leak across 11,000 stress inferences |
| **Peak VRAM (`batch=32` active workspace)** | `3,690.89 MiB` (`3.60 GiB`) | $\le 6.50\text{ GiB}$ | `2.90 GiB` below hard cap |

---

## 4. Audited Subsystem Microbenchmarks (`p50` / `p99` & Honest Offscreen Units)

> **Unit Discipline (Check 1.10):** All subsystem kernels below operate on in-memory images/tensors with zero display presentation (`Swapchain` / `VkQueuePresentKHR`). Units are reported as `img/s (offscreen CPU)`, `calls/s (CPU)`, `updates/s (CPU)`, `win/s`, or `solves/s` — never bare `"FPS"`.

| Subsystem / Kernel Benchmark | `p50` Latency ($\mu\text{s}$) | `p99` Latency ($\mu\text{s}$) | Audited Throughput & Unit | Post-Measurement Sanity Check |
| :--- | :---: | :---: | :--- | :--- |
| `core::aligned_alloc_64` (`1080p` RGB) | **`0.02 us`** | `0.02 us` | OS aligned alloc (`64B`) | Non-null + `reinterpret_cast<uintptr_t>(p) % 64 == 0` |
| `core::Aligned_Buffer_Pool` (`1080p` RGB) | **`0.02 us`** | `0.02 us` | Lock-free CAS pool recycle | Non-null + 64-byte alignment verified |
| `core::simd_scale_bias_u8_to_f32` (`1080p`) | **`200.41 us`** | `208.73 us` | `10.35 GPix/s` | `dst[0]` matches exact `src[0] * scale + bias` |
| `imgproc::rgb_to_gray` (`1280x720`) | **`78.31 us`** | `141.66 us` | `12,770.17 img/s (offscreen CPU)` | `g->width() == 1280` & BT.601 luma check |
| `imgproc::gaussian_blur 5x5` (`1280x720`) | **`438.64 us`** | `568.45 us` | `2,279.80 img/s (offscreen CPU)` | Center pixel `b->at(360, 640, 0) > 0` |
| `imgproc::resize Bilinear` (`720p -> 640x640`) | **`325.42 us`** | `652.72 us` | `3,072.95 img/s (offscreen CPU)` | Output dimensions `640 x 640` verified |
| `imgproc::erode 5x5 AVX2` (`1280x720`) | **`1,929.34 us`** | `1,983.66 us` | `518.31 img/s (offscreen CPU)` | Eroded interior pixel value verified |
| `io::decode_qoi` (`640x480` RGB) | **`115.89 us`** | `117.35 us` | `8,628.98 img/s (offscreen CPU)` | Decoded image `640 x 480 x 3` verified |
| `io::decode_jpeg` (`640x480` RGB, `libjpeg-turbo`) | **`239.60 us`** | `244.02 us` | `4,173.66 img/s (offscreen CPU)` | Decoded image `640 x 480 x 3` verified |
| `augment::Augment_Pipeline` 5-stage (`640x640`) | **`833.96 us`** | `952.51 us` | `1,199.10 img/s (offscreen CPU)` | Augmented sample `640 x 640` verified |
| `nn::image_to_tensor_nchw` (`640x640` RGB) | **`243.10 us`** | `246.92 us` | `4,113.47 img/s (offscreen CPU)` | `t->numel() == 1,228,800` verified |
| `nn::Conv_BN_Act` unfused (`1x3x64x64 -> 16`) | **`761.89 us`** | `1,102.04 us` | `1,312.53 calls/s (CPU)` | Output tensor `numel() == 65,536` |
| `nn::Conv_BN_Act` fused (`1x3x64x64 -> 16`) | **`214.19 us`** | `246.59 us` | `4,668.81 calls/s (CPU)` (`3.56x` speedup) | Fused vs unfused numerical parity verified |
| `detection::letterbox_image` (`720p -> 640x640`) | **`235.91 us`** | `279.08 us` | `4,238.88 img/s (offscreen CPU)` | Letterbox output `640 x 640` + scale/pad verified |
| `detection::batched_nms` (`1000 boxes, 10 cls`) | **`525.27 us`** | `534.19 us` | `1,903.80 calls/s (CPU)` | Kept box count `== 540` verified |
| `detection::Byte_Tracker` (`25 Kalman tracks`) | **`22.49 us`** | `29.61 us` | `44,464.36 updates/s (CPU)` | Active track count `== 25` verified |
| `classical::detect_fast9_keypoints` (`640x480`) | **`1,253.79 us`** | `1,313.10 us` | `797.58 img/s (offscreen CPU)` | Detected keypoint count `== 234` verified |
| `classical::HOG_Extractor` (`64x128` pedestrian) | **`52.71 us`** | `54.34 us` | `18,972.99 win/s` | Descriptor length `== 3,780` floats verified |
| `classical::estimate_homography_ransac` | **`11.54 us`** | `12.12 us` | `86,663.82 solves/s` | Converged $3\times 3$ homography verified |

---

## 5. Superseded Pre-Audit Benchmark Claims (`2026-10-31`)

Preserved below for audit traceability (see [KNOWN_ISSUES.md](KNOWN_ISSUES.md)):

| Subsystem / Kernel Benchmark | Pre-Audit Claim (`2026-10-31`, *Superseded*) | Audit Violation | Revised Audited Metric (`2026-11-01`) |
| :--- | :--- | :--- | :--- |
| `imgproc::rgb_to_gray` (`1280x720`) | `77.34 us` (`12,930.25 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` (`12,930.25 FPS` > `60 Hz` monitor refresh) on offscreen CPU kernel. | `78.31 us` `p50` / `141.66 us` `p99` (`12,770.17 img/s (offscreen CPU)`) |
| `imgproc::gaussian_blur 5x5` (`1280x720`) | `399.68 us` (`2,502.00 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` on offscreen CPU filter. | `438.64 us` `p50` / `568.45 us` `p99` (`2,279.80 img/s (offscreen CPU)`) |
| `imgproc::resize Bilinear` (`720p -> 640x640`) | `345.86 us` (`2,891.32 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` on offscreen CPU resize. | `325.42 us` `p50` / `652.72 us` `p99` (`3,072.95 img/s (offscreen CPU)`) |
| `io::decode_qoi` (`640x480` RGB) | `120.45 us` (`8,302.32 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` on in-memory image decode. | `115.89 us` `p50` / `117.35 us` `p99` (`8,628.98 img/s (offscreen CPU)`) |
| `io::decode_jpeg` (`640x480` RGB) | `245.15 us` (`4,079.07 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` on in-memory JPEG decode. | `239.60 us` `p50` / `244.02 us` `p99` (`4,173.66 img/s (offscreen CPU)`) |
| `augment::Augment_Pipeline` 5-stage (`640x640`) | `827.32 us` (`1,208.72 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` on offscreen CPU augmentation. | `833.96 us` `p50` / `952.51 us` `p99` (`1,199.10 img/s (offscreen CPU)`) |
| `nn::Conv_BN_Act` unfused (`1x3x64x64 -> 16`) | `659.10 us` (`1,517.21 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` on tensor operator. | `761.89 us` `p50` / `1,102.04 us` `p99` (`1,312.53 calls/s (CPU)`) |
| `nn::Conv_BN_Act` fused (`1x3x64x64 -> 16`) | `224.85 us` (`4,447.48 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` on tensor operator. | `214.19 us` `p50` / `246.59 us` `p99` (`4,668.81 calls/s (CPU)`) |
| `detection::letterbox_image` (`720p -> 640x640`) | `237.30 us` (`4,214.03 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` on offscreen letterbox. | `235.91 us` `p50` / `279.08 us` `p99` (`4,238.88 img/s (offscreen CPU)`) |
| `detection::batched_nms` (`1000 boxes, 10 cls`) | `516.38 us` (`1,936.56 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` on bounding-box NMS. | `525.27 us` `p50` / `534.19 us` `p99` (`1,903.80 calls/s (CPU)`) |
| `detection::Byte_Tracker` (`25 Kalman tracks`) | `14.48 us` (`69,072.46 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` (`69,072.46 FPS`) on Kalman + Hungarian state update. | `22.49 us` `p50` / `29.61 us` `p99` (`44,464.36 updates/s (CPU)`) |
| **Host Process Memory (`VmRSS`)** | Omitted (only GPU VRAM reported) | **Check 1.4 Violation:** Missing 3-stage host process `VmRSS` split (`Bare Process`, `Post-Library-Load`, `Post-GPU-Init`). | Added `8.41 MB` (Bare Process), `20.21 MB` (Post-Library-Load), and `1,827.18 MB` (Post-GPU-Init + 8-slot pinned staging + CPU fallback weights). |
