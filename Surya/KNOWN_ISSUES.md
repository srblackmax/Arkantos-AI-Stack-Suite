# Surya Known Issues & Superseded Benchmark Records (`KNOWN_ISSUES.md`)

**Library:** Surya (`1.0.0`) — Native C++23 Computer Vision & Hybrid Linear-Attention VLM Pipeline  
**Audit Date:** `2026-11-01`  

---

## 1. Architectural & Scope Disclosures

### 1.1 Compact 24-Layer INT8 Document VLM vs. Multi-Billion-Parameter Autoregressive VLMs
- **Component:** `surya::gpu::SuryaPipeline` (`src/gpu/surya_pipeline.cpp`, `src/gpu/vlm_kernels.cu`)
- **Behavior:** The resident INT8 model executed by `bench_surya` (`256.00 MiB` INT8 weights, 24-layer hybrid Gated DeltaNet / GQA backbone + OCR/Layout/Table/NMS heads on `640x480` frames) completes a single-frame forward pass in **`348.59 us` `p50`** (`2,868.69 img/s` with CUDA Graph replay on RTX 3070) and **`187.78 us/img`** (`5,325.30 img/s` at `batch=16` overlapped multi-stream).
- **Scope Note:** This measures a single deterministic non-autoregressive encoder/head pass over a compact `256 MiB` INT8 document-vision backbone, NOT multi-hundred-token autoregressive text decoding on a 7B+ parameter VLM (such as Qwen2-VL-7B or LLaVA-1.5-7B).

### 1.2 Host Process RSS Increase After `SuryaPipeline` Multi-Stream GPU Initialization (`1,827.18 MB`)
- **Component:** `surya::gpu::SuryaPipeline::create` (`max_batch_size = 32`, `num_streams = 8`)
- **Behavior:** While Bare Process RSS is `8.41 MB` and Post-Library-Load RSS (after `1080p` CPU pools) is `20.21 MB`, initializing both a CPU `SuryaPipeline` instance and a GPU `SuryaPipeline` instance with `max_batch_size = 32` across 8 asynchronous CUDA streams + CUDA Graph capture allocates `1,827.18 MB` of host `VmRSS` (alongside `2,090.89 MiB` steady-state GPU VRAM).
- **Root Cause:** `cudaMallocHost` page-locked (pinned) host staging buffers for 8 concurrent `batch = 32` streams plus `libcuda.so.1` primary context and CPU fallback weight tables are mapped into the host process's resident set.

---

## 2. Superseded Pre-Audit Benchmark Claims (`2026-10-31`)

During the 2026-11-01 Ecosystem Benchmark Integrity Audit (see [AUDIT_REPORT.md](../AUDIT_REPORT.md), [BENCHMARKS_METHODOLOGY.md](BENCHMARKS_METHODOLOGY.md), and [BENCHMARKS.md](BENCHMARKS.md)), the following pre-audit rows from `Surya/BENCHMARKS.md` (`2026-10-31`) were flagged and marked **superseded**:

| Subsystem / Kernel Benchmark | Pre-Audit Claim (`2026-10-31`, *Superseded*) | Audit Violation & Root Cause | Revised Audited Metric (`2026-11-01`) |
| :--- | :--- | :--- | :--- |
| `imgproc::rgb_to_gray` (`1280x720`) | `77.34 us` (`12,930.25 FPS`) | **Check 1.1, 1.2 & 1.10:** Reported single-pass mean without `p50`/`p99` and labeled an offscreen CPU grayscale conversion as `12,930.25 FPS` (`> 60 Hz` physical monitor refresh rate) with zero display presentation. | `78.31 us` `p50` / `141.66 us` `p99` (`12,770.17 img/s (offscreen CPU)`) |
| `imgproc::gaussian_blur 5x5` (`1280x720`) | `399.68 us` (`2,502.00 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` label on offscreen CPU convolution. | `438.64 us` `p50` / `568.45 us` `p99` (`2,279.80 img/s (offscreen CPU)`) |
| `imgproc::resize Bilinear` (`720p -> 640x640`) | `345.86 us` (`2,891.32 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` label on offscreen CPU resize. | `325.42 us` `p50` / `652.72 us` `p99` (`3,072.95 img/s (offscreen CPU)`) |
| `io::decode_qoi` (`640x480` RGB) | `120.45 us` (`8,302.32 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` label on in-memory QOI decode. | `115.89 us` `p50` / `117.35 us` `p99` (`8,628.98 img/s (offscreen CPU)`) |
| `io::decode_jpeg` (`640x480` RGB) | `245.15 us` (`4,079.07 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` label on in-memory JPEG decode. | `239.60 us` `p50` / `244.02 us` `p99` (`4,173.66 img/s (offscreen CPU)`) |
| `augment::Augment_Pipeline` 5-stage (`640x640`) | `827.32 us` (`1,208.72 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` label on offscreen CPU augmentation. | `833.96 us` `p50` / `952.51 us` `p99` (`1,199.10 img/s (offscreen CPU)`) |
| `nn::Conv_BN_Act` unfused (`1x3x64x64 -> 16`) | `659.10 us` (`1,517.21 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` label on a neural-network layer operator. | `761.89 us` `p50` / `1,102.04 us` `p99` (`1,312.53 calls/s (CPU)`) |
| `nn::Conv_BN_Act` fused (`1x3x64x64 -> 16`) | `224.85 us` (`4,447.48 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` label on a neural-network layer operator. | `214.19 us` `p50` / `246.59 us` `p99` (`4,668.81 calls/s (CPU)`) |
| `detection::letterbox_image` (`720p -> 640x640`) | `237.30 us` (`4,214.03 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` label on offscreen CPU letterbox. | `235.91 us` `p50` / `279.08 us` `p99` (`4,238.88 img/s (offscreen CPU)`) |
| `detection::batched_nms` (`1000 boxes, 10 cls`) | `516.38 us` (`1,936.56 FPS`) | **Check 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` label on bounding-box NMS. | `525.27 us` `p50` / `534.19 us` `p99` (`1,903.80 calls/s (CPU)`) |
| `detection::Byte_Tracker` (`25 Kalman tracks`) | `14.48 us` (`69,072.46 FPS`) | **Check 1.1, 1.2 & 1.10:** Single-pass mean without `p50`/`p99`; bare `"FPS"` label (`69,072.46 FPS`) on a Kalman + Hungarian state update. | `22.49 us` `p50` / `29.61 us` `p99` (`44,464.36 updates/s (CPU)`) |
| **Host Process Memory (`VmRSS`)** | Omitted (only GPU VRAM reported) | **Check 1.4:** Did not report 3-stage host process `VmRSS` (`Bare Process`, `Post-Library-Load`, `Post-GPU-Init`). | Added `8.41 MB` (Bare Process), `20.21 MB` (Post-Library-Load), and `1,827.18 MB` (Post-GPU-Init + 8-slot pinned staging + CPU fallback weights). |
