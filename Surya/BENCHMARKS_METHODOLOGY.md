# Surya — Benchmark Methodology (`BENCHMARKS_METHODOLOGY.md`)

This document specifies the exact measurement protocol, hardware baseline, timed intervals, exclusions, warm weight residency & PCIe transfer disclosures, competitor baselines (`OpenCV 4.10`, `torchvision 0.19`, `Albumentations 1.4`, `PyTorch 2.4`), and post-measurement sanity checks for every benchmark metric in [`BENCHMARKS.md`](./BENCHMARKS.md).

---

## 0. Hardware & Software Environment Baseline

- **CPU:** `12th Gen Intel(R) Core(TM) i7-12700K` (12 physical cores: 8 Performance-cores + 4 Efficient-cores, 20 hardware threads, 800 MHz – 5,100 MHz turbo, `30 MiB` L3 cache, AVX2 / FMA3; Intel `oneTBB 2022.1.0` work-stealing scheduler)
- **GPU:** `NVIDIA GeForce RTX 3070` (`GA104`, `sm_86`, `46` SMs, `8,192 MiB` GDDR6 VRAM @ `448 GB/s`, Driver `550.163.01`, CUDA `12.4.131`)
- **RAM:** `48 GiB` Dual-Channel DDR4
- **Storage:** `Samsung SSD 970 EVO Plus 1TB` NVMe (`nvme0n1`)
- **OS & Kernel:** Debian GNU/Linux 13 (trixie), Kernel `Linux 6.12.113+deb13-amd64 x86_64`
- **Compiler & Flags:** `GCC 14.2.0` (`-std=c++23 -O3 -march=x86-64-v3 -Wall -Wextra -Wpedantic -Werror`), `NVCC 12.4.131` (`-O3 -arch=sm_86`)
- **Statistical Protocol:** Every benchmark executes `2–6` untimed warm-up batches followed by `9–21` timed sample batches (`batch_iters = 1` to `500`), recording both `p50` (median) and `p99` latencies in microseconds (`us`).

---

## 1. Single-Image Hybrid VLM Latency (`batch = 1`: CPU vs. CUDA vs. CUDA Graph)

- **Architecture & Warm Weight Residency (§4.5):** Evaluates `SuryaPipeline::infer_single` on a `640x480` 3-channel RGB document image (`921.6 KiB`) through Surya's compact 24-layer Qwen3.5-family Hybrid Linear-Attention document vision-language pipeline (`image_target_width = 320`, `image_target_height = 320`, `vlm_hidden_dim = 256`, `vlm_num_layers = 24`, `vlm_num_tokens = 32`).
- **Warm Weights vs. Cold Load Disclosure (§4.5):** The `256.00 MiB` INT8 quantized model weights and `cudaMallocAsync` workspace pools are pre-loaded once during `SuryaPipeline::create()`. The timed `infer_single` interval measures **warm steady-state inference** (`0` weight reload I/O).
- **PCIe H2D / D2H Inclusion Disclosure (§4.5):** Every timed `infer_single` and `infer_batch` call **includes**:
  1. Host-to-device (`H2D`) asynchronous DMA copy of the input `640x480` RGB frame from pinned host staging memory to device VRAM,
  2. Device-side bilinear letterbox + ImageNet normalization kernel (`320x320`),
  3. 24-layer hybrid Gated DeltaNet / GQA backbone + OCR/Layout/Table heads + GPU NMS,
  4. Device-to-host (`D2H`) copy of detected bounding boxes, layout labels, and token IDs, and
  5. `cudaStreamSynchronize` before returning `Result<DocumentResult>`.
- **Sanity Check:** Verifies `r.has_value() == true` on every inference.

---

## 2. Batched Hybrid VLM Throughput (`batch = 1, 4, 8, 16, 32` & Overlapped 128-Image Stream)

- **Timed Interval:**
  - `infer_batch(bvec)` for `batch_size in {1, 4, 8, 16, 32}` using captured CUDA Graph replay (`enable_cuda_graphs = true`), including H2D batch staging, graph launch, D2H result readback, and stream synchronization.
  - `run_overlapped_stream_pipeline(stream128, 16)` over `128` images partitioned into `8` batches of `16` across dual CUDA streams (overlapping Stream A H2D/D2H transfers with Stream B compute kernels).
- **Sanity Check:** Verifies `r.has_value() && r->size() == bs` (and `r->size() == 128` for overlapped streaming).

---

## 3. VRAM Residency & Budget Compliance (`VramBudgetAllocator`)

- **Timed Interval:** Queries `(*pipe_g)->vram_current_bytes()` (`2,090.89 MiB` steady-state across 8 pre-allocated ring slots for `max_batch_size = 32`) and `(*pipe_g)->vram_peak_bytes()` (`3,690.89 MiB` peak active workspace) against the hard `6,656.00 MiB` (`6.50 GiB`) cap on the `8 GiB` RTX 3070.
- **Sanity Check:** Verifies `steady_mb > 0.0 && peak_mb <= 6656.0`.

---

## 4. Subsystem CPU Microbenchmarks (`core`, `imgproc`, `io`, `augment`, `nn`, `detection`, `classical`)

- **Terminology Discipline (Check 1.10):** All 16 CPU subsystem microbenchmarks execute in **offscreen host RAM** with zero display swapchain involvement. Consequently, throughput is reported strictly in **`img/s (offscreen CPU)`**, **`calls/s (CPU)`**, **`updates/s (CPU)`**, **`win/s`**, **`solves/s`**, or **`GPix/s`** — never bare `FPS`.
- **Sanity Checks:** Every microbenchmark inspects its output container (`gray_w == 1280`, `gauss_p0 == 120`, `res_w == 640`, `erode_p0 == 120`, `qoi_w == 640`, `jpg_w == 640`, `aug_w == 640`, `bridge_numel == 1228800`, `unfused_numel == 65536`, `fused_numel == 65536`, `lb_w == 640`, `0 < nms_kept < 1000`, `trk_cnt > 0`, `fast_kps > 0`, `hog_dims == 3780`, `ransac_ok == true`).

---

## 5. Host Memory RSS Phases

- **Timed Interval:** Reads `VmRSS` from `/proc/self/status` at three checkpoints:
  1. **Bare Process RSS (`8.43 MB`):** At `main()` entry (includes dynamically linked `libsurya.so`, `libobsidian.so`, `libbasalt.so`, `libtbb.so`, `libjpeg.so`, `libpng.so`, `libwebp.so`, `libtiff.so`, `libcudart.so`).
  2. **Post-Library-Load RSS (`20.06 MB`):** After allocating the 1080p `Aligned_Buffer_Pool` and 1080p grayscale/f32 images on the CPU.
  3. **Post-GPU-Init RSS (`1,827.03 MB`):** After initializing `SuryaPipeline` on CUDA (`device_id = 0`), allocating pinned host staging buffers + CUDA Graph nodes, executing 1 warm-up inference, and idling for 1 second.
