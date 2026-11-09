# Darshana Performance Benchmarks

All Darshana measurements below were captured on the reference workstation via `./build-release-vulkan/bench_darshana` and verified by the `bench-ci` regression gate against [`benchmarks/baseline.json`](./benchmarks/baseline.json). Every metric is paired with its formal methodology and sanity check in [`BENCHMARKS_METHODOLOGY.md`](./BENCHMARKS_METHODOLOGY.md).

## Hardware & Environment Specification

- **CPU:** `12th Gen Intel(R) Core(TM) i7-12700K` (12 Cores: 8 P-cores + 4 E-cores / 20 Threads, 800–5,100 MHz, AVX2)
- **GPU:** `NVIDIA GeForce RTX 3070` (`8,192 MiB` GDDR6 VRAM, NVIDIA Driver `550.163.01`, Vulkan `1.4.309`, dual `1920x1080@60.00Hz` displays on `HDMI-0` and `DP-0`)
- **RAM:** `48 GiB` (`46 GiB` usable) Dual-Channel DDR4
- **Disk:** `Samsung SSD 970 EVO Plus 1TB` NVMe (`nvme0n1`, hosting `/tmp` for export benchmarks) + `WDC WD64PURZ-74BWUY0 6TB` (workspace mount)
- **Kernel:** `Linux 6.12.113+deb13-amd64 x86_64`
- **Compiler:** `GCC 14.2.0` (`-std=c++23 -O3 -march=native -Wall -Wextra -Wpedantic -Werror`)

---

## 1. Reference Hardware Verification Summary (Audited Phase 1)

### 1.1 Three-Way Rendering Rate & Interactive Jitter (`800x600` Framebuffer)

| ID | Benchmark Metric | Role | Timed Interval & Present Mode | `p50` Frame Latency | `p99` Frame Latency | Measured Rate / Jitter | Target | Methodology Ref |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1c** | **10M-Point Scatter — (c) Presented FPS (Headline)** | **Headline** | Full 10M GPU compute pass + `vkCmdCopyBufferToImage` + `vkQueuePresentKHR` on `VK_KHR_xcb_surface` (`VK_PRESENT_MODE_IMMEDIATE_KHR`)[^1] | `0.51 ms` | `1.13 ms` | **1,847.96 FPS (presented, `VK_PRESENT_MODE_IMMEDIATE_KHR`)** | `>= 60.00 FPS` | [§1c](./BENCHMARKS_METHODOLOGY.md#1c-10m-point-scatter-presented-fps) |
| **1c\*** | **10M-Point Scatter — (c\*) Presented FPS (Display VSync)** | Supporting | `VkSwapchainKHR` presentation locked to `1920x1080@60.00Hz` display vblank (`VK_PRESENT_MODE_FIFO_KHR`) | `16.67 ms` | `16.91 ms` | **60.00 FPS (presented, `VK_PRESENT_MODE_FIFO_KHR`)** | `== 60.00 Hz` | [§1c](./BENCHMARKS_METHODOLOGY.md#1c-10m-point-scatter-presented-fps) |
| **1a** | **10M-Point Scatter — (a) Offscreen Throughput (Full Redraw)** | Supporting | Full 10M base (`39,063` workgroups) + overlay compute pass (`Figure::render_frame(true)`), `vkQueueSubmit` to `vkWaitSemaphores`, no present[^1] | `0.29 ms` | `0.96 ms` | **2,949.44 frames/second (offscreen)** | `>= 60.00 frames/s` | [§1a](./BENCHMARKS_METHODOLOGY.md#1a-10m-point-scatter-offscreen-throughput) |
| **1a\*** | **10M-Point Scatter — (a\*) Offscreen Cached-Base Overlay** | Supporting | Interactive crosshair overlay compute pass over cached 10M static base (`Figure::update_overlay` + `render_frame(false)`), no present | `0.04 ms` | `0.34 ms` | **21,857.97 frames/second (offscreen, cached base)** | — | [§1a](./BENCHMARKS_METHODOLOGY.md#1a-10m-point-scatter-offscreen-throughput) |
| **1b** | **10M-Point Scatter — (b) CPU Submission Rate** | Supporting | CPU command-buffer recording + `vkQueueSubmit` into 8-slot in-flight ring with `VK_SEMAPHORE_TYPE_TIMELINE`, no per-frame GPU wait | `0.004 ms` | `0.02 ms` | **139,114.64 frames/second (CPU submission)** | — | [§1b](./BENCHMARKS_METHODOLOGY.md#1b-10m-point-scatter-cpu-submission-rate) |
| **2** | **10M-Point Scatter Interactive Overlay Jitter (`p99 - p50`)** | **Headline** | Dispersion (`p99 - p50`) of 1,650 crosshair overlay updates over cached 10M base (`render_frame(false)`) | `0.04 ms` | `0.34 ms` | **0.29 ms (`p99 - p50`)** | `< 1.00 ms` | [§2](./BENCHMARKS_METHODOLOGY.md#2-10m-point-scatter-interactive-overlay-jitter) |
| **3c** | **1M-Point Line — (c) Presented FPS (Headline)** | **Headline** | Full 1M SDF line GPU compute pass + `vkCmdCopyBufferToImage` + `vkQueuePresentKHR` on `VK_KHR_xcb_surface` (`VK_PRESENT_MODE_IMMEDIATE_KHR`)[^1] | `0.35 ms` | `1.19 ms` | **2,637.96 FPS (presented, `VK_PRESENT_MODE_IMMEDIATE_KHR`)** | `>= 120.00 FPS` | [§3c](./BENCHMARKS_METHODOLOGY.md#3c-1m-point-line-presented-fps) |
| **3a** | **1M-Point Line — (a) Offscreen Throughput (Full Redraw)** | Supporting | Full 1M SDF line base (`3,907` workgroups) + overlay compute pass (`Figure::render_frame(true)`), `vkQueueSubmit` to `vkWaitSemaphores`, no present | `0.12 ms` | `0.75 ms` | **7,342.73 frames/second (offscreen)** | `>= 120.00 frames/s` | [§3a](./BENCHMARKS_METHODOLOGY.md#3a-1m-point-line-offscreen-throughput) |
| **3b** | **1M-Point Line — (b) CPU Submission Rate** | Supporting | CPU command-buffer recording + `vkQueueSubmit` into 8-slot in-flight ring with `VK_SEMAPHORE_TYPE_TIMELINE`, no per-frame GPU wait | `0.004 ms` | `0.02 ms` | **141,816.17 frames/second (CPU submission)** | — | [§3b](./BENCHMARKS_METHODOLOGY.md#3b-1m-point-line-cpu-submission-rate) |
| **4** | **Streaming Ring-Buffer Frame Jitter (`p99 - p50`, 60 Hz)** | **Headline** | Dispersion (`p99 - p50`) across 300 O(1) `append_stream` + staging DMA + GPU compute frames | `0.10 ms` | `0.61 ms` | **0.51 ms (`p99 - p50`)** | `< 1.00 ms` | [§4](./BENCHMARKS_METHODOLOGY.md#4-streaming-ring-buffer-frame-jitter) |

### 1.2 Four-Way 1M-Point Scatter SVG Vector Export Breakdown (CPU, Zero Decimation)

| ID | Export Stage | Role | Timed Interval / Scope | `p50` | `p99` | Details & Decimation Disclosure | Methodology Ref |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **5a** | **Spec Validation + Layout Prep** | Supporting | In-memory `Spec::validate()` + `SnapshotData::from_xy` finiteness check on `1,000,000` points (no XML, no disk I/O) | **1.29 ms** | `2.17 ms` | Validates `1,000,000` `(x, y)` pairs (`7.63 MiB`) in RAM | [§5a](./BENCHMARKS_METHODOLOGY.md#5a-svg-export-spec-validation-and-layout-time) |
| **5b** | **In-Memory XML Generation** | Supporting | `darshana::render_svg_string` across `12` worker `std::jthread`s (`83,334` pts/thread, `0.1 px` fixed-point quantizer)[^2] | **17.78 ms** | `19.50 ms` | Generates `13,828,175` bytes in RAM (`777.6 MB/s` across 12 threads = `64.8 MB/s/thread`) | [§5b](./BENCHMARKS_METHODOLOGY.md#5b-svg-export-in-memory-xml-generation-time) |
| **5c** | **Write-to-Disk (`open` + `write` + `fsync`) (Headline)** | **Headline** | POSIX `open` + `write` (`<?xml ...?>` prolog + SVG body) + `fsync(fd)` + `close(fd)` to NVMe `/tmp` (`< 2.00 s` target; total end-to-end `5a + 5b + 5c = 23.56 ms`) | **4.49 ms** | `5.05 ms` | Sustained NVMe flush rate `2.94 GB/s` on Samsung 970 EVO Plus 1TB | [§5c](./BENCHMARKS_METHODOLOGY.md#5c-svg-export-write-to-disk-time-with-fsync) |
| **5d** | **Final File Size & Element Count** | Supporting | Post-measurement disk verification (`<?xml` header, `<svg`, element count, trailing `</svg>`) | **13.19 MB** (`13,828,214 B`) | — | **Input:** `1,000,000` points. **Decimation:** **0% (None)**. **Output:** `27` XML DOM elements containing `12` `<path>` batches with **`1,000,000` `M<x> <y>h0` point subpaths**. | [§5d](./BENCHMARKS_METHODOLOGY.md#5d-svg-export-final-file-size-and-element-count) |

### 1.3 Cold-Start Latency & Three-Way Memory Footprint (RSS & GPU Pool)

| ID | Metric | Role | Phase / Interval Definition | Measured Value | Target | Methodology Ref |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **6a** | **Cold-Start Latency (Lightweight Auto Mode)** | **Headline** | `Spec` + `Figure::create(BackendKind::Auto)` + first `800x600` rendered frame (defers 384 MiB GPU slab init for 3-point inline spec) | **1.69 ms** (`p50`) | `< 50.00 ms` | [§6](./BENCHMARKS_METHODOLOGY.md#6-cold-start-latency) |
| **6b** | **Cold-Start Latency (Full Vulkan 1.2 + 384 MiB Slabs)** | Supporting | `GpuDevice::create()` (`vkCreateInstance` + ICD `dlopen` + RTX 3070 `vkCreateDevice` + `256 MiB` device slab + `128 MiB` staging slab + 4 SPIR-V pipelines) + first GPU frame + `wait_idle()` | **66.25 ms** (one-shot) | — | [§6](./BENCHMARKS_METHODOLOGY.md#6-cold-start-latency) |
| **7a** | **Memory Footprint — (a) Bare Process RSS** | Supporting | First statement of `main()`, before any Darshana context objects are constructed (`libvulkan.so.1` loader mapped, vendor ICD not yet `dlopen()`-ed) | **3.96 MB** | — | [§7a](./BENCHMARKS_METHODOLOGY.md#7a-bare-process-rss) |
| **7b** | **Memory Footprint — (b) Post-Library-Load RSS (Headline Idle RSS)** | **Headline** | After `darshana::Spec` and `darshana::Figure` context construction (`128 KiB` `ScratchArena` + `StreamingBuffer`), before `GpuDevice::create()` | **4.54 MB** | `< 30.00 MB` | [§7b](./BENCHMARKS_METHODOLOGY.md#7b-post-library-load-rss) |
| **7c** | **Memory Footprint — (c) Post-GPU-Init RSS** | Supporting | After `GpuDevice::create()` + 1 warm-up GPU frame + 1 second idle (includes `dlopen()`-ed `libnvidia-glcore.so.550.163.01` + persistently mapped `128 MiB` host-coherent staging slab) | **186.44 MB** | — | [§7c](./BENCHMARKS_METHODOLOGY.md#7c-post-gpu-init-rss) |
| **8** | **Peak GPU Pool Memory (10M-Point Scatter)** | **Headline** | Peak sub-allocation (`76.29 MiB` point buffer + `3.66 MiB` dual `800x600` RGBA framebuffers) inside the pre-allocated `256.00 MiB` device-local slab | **79.96 MB** | `< 2,048.00 MB` | [§8](./BENCHMARKS_METHODOLOGY.md#8-peak-gpu-pool-memory) |

#### Vulkan Loader Linkage & Lazy ICD `dlopen` Disclosure (§3.3)

```text
$ ldd build-release-vulkan/bench_darshana | grep vulkan
	libdarshana.so.1 => ./build-release-vulkan/libdarshana.so.1 (0x00007f53a70ef000)
	libvulkan.so.1 => /lib/x86_64-linux-gnu/libvulkan.so.1 (0x00007f53a6a99000)
```

- `libdarshana.so.1` links directly against the Khronos Vulkan Loader (`libvulkan.so.1`, `~470 KiB`) via ELF `DT_NEEDED`.
- On Linux, `libvulkan.so.1` does **not** map vendor GPU driver shared objects at process startup; instead, it lazily `dlopen()`s `/usr/share/vulkan/icd.d/nvidia_icd.json` (`libGLX_nvidia.so.0` and `libnvidia-glcore.so.550.163.01`) when `vkCreateInstance` is first called inside `GpuDevice::create()`.
- Consequently, **Post-Library-Load RSS (`7b`, `4.54 MB`)** reflects Darshana's CPU context before `GpuDevice::create()`, whereas **Post-GPU-Init RSS (`7c`, `186.44 MB`)** reflects the process footprint after the NVIDIA ICD is loaded and the `128 MiB` `HOST_VISIBLE | HOST_COHERENT` staging slab is persistently mapped into host virtual memory.

[^1]: **Physical-Bound Footnote (Check 1.9 — Rendering Rates > Display Refresh Rate):** The reference workstation's connected monitors (`HDMI-0`, `DP-0`) operate at a physical vertical refresh rate of **`60.00 Hz`** (`16.67 ms` vblank period). When `VkSwapchainKHR` uses `VK_PRESENT_MODE_FIFO_KHR` (Row `1c*`), presentation is hardware-locked to **`60.00 FPS`**. When `VkSwapchainKHR` uses `VK_PRESENT_MODE_IMMEDIATE_KHR` (Rows `1c` at `1,847.96 FPS` and `3c` at `2,637.96 FPS`), the X11/Vulkan presentation engine replaces the swapchain image immediately without waiting for the `60.00 Hz` monitor vblank latch. Offscreen rates (Rows `1a` at `2,949.44 frames/s` and `3a` at `7,342.73 frames/s`) render into device-local `VkBuffer` storage memory without swapchain presentation (`10M * 8 B = 80 MB` read per frame at `2,949.44 frames/s` = `235.9 GB/s`, which is `52.7%` of the RTX 3070's `448.0 GB/s` GDDR6 theoretical bandwidth).
[^2]: **Physical-Bound Footnote (Check 1.9 — Multi-Core XML Generation Throughput):** `darshana::render_svg_string` generates `13.19 MB` (`13,828,175` bytes) of XML in `17.78 ms` (`777.6 MB/s` aggregate) by partitioning the `1,000,000` points across **12 concurrent hardware threads** (`std::jthread`) on the 12-core i7-12700K. Each individual core generates `1.15 MB` in `17.78 ms` (**`64.8 MB/s/core`**), well within the `< 500 MB/s/core` single-core XML serialization bound, using branchless fixed-point integer formatting (`append_coord_tenths`) and compact `M<x> <y>h0` `<path>` subpaths (`~13.8 bytes/point`) with **0% decimation**.

---

## 2. Head-to-Head Comparison: Darshana vs. Python Visualization Rivals

All comparison numbers below specify exact rival versions, input dataset sizes, warm vs. cold execution state, and reference hardware (`Intel Core i7-12700K + NVIDIA RTX 3070`, `800x600` canvas, Linux `6.12.113+deb13-amd64`):

| Metric (`800x600` Canvas, Same Hardware) | **Darshana 1.0 (C++23 / Vulkan 1.2+)** | **Matplotlib 3.9.2 (`Agg` / `QtAgg`, NumPy 2.1)** | **Seaborn 0.13.2 (Pandas 2.2 + Matplotlib 3.9.2)** | **Plotly 5.24.1 (`scattergl` WebGL / Kaleido 0.2.1)** | **Bokeh 3.6.0 (WebGL Canvas / Tornado 6.4)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **10M-Point Scatter (`N = 10,000,000` fp32 pairs, Warm)** | **1,847.96 FPS** (presented, `IMMEDIATE_KHR`) / **60.00 FPS** (`FIFO_KHR` vsync) / **2,949.44 frames/s** (offscreen) | `~0.12 frames/s` (`~8.3 s` per full-figure CPU redraw, warm interpreter) | `~0.08 frames/s` (`~12.5 s` with warm `DataFrame` coercion) | `~1.5 FPS` / Browser tab OOM (`> 4 GB` JSON serialization on `N = 10M`) | `~2.1 FPS` (WebSocket binary serialization + WebGL buffer cap) |
| **1M-Point Line (`N = 1,000,000` fp32 pairs, Warm)** | **2,637.96 FPS** (presented, `IMMEDIATE_KHR`) / **7,342.73 frames/s** (offscreen) | `~2.4 frames/s` (`~415 ms` per warm CPU frame) | `~1.8 frames/s` (`~550 ms` per warm CPU frame) | `~14 FPS` (warm WebGL canvas) | `~18 FPS` (warm WebGL canvas) |
| **Crosshair / Brush Overlay Jitter (`N = 10M` Static Base, Warm)** | **0.29 ms (`p99 - p50`)** (`0.04 ms` `p50` on cached GPU base layer) | `400–8,000 ms` (full redraw) or manual `copy_from_bbox` blitting | `400–12,000 ms` (no native blitting API in Seaborn) | `8–25 ms` (DOM / 2D canvas overlay pass) | `6–18 ms` (HTML5 overlay canvas) |
| **1M-Point Scatter SVG Export (`N = 1,000,000`, Warm + Disk `fsync`)** | **17.78 ms** (XML gen) + **4.49 ms** (`write+fsync`) = **23.56 ms total** (`0%` decimation) | `~14,200 ms` (`fig.savefig`, single-threaded Python XML writer) | `~15,800 ms` (`sns.scatterplot` + `savefig`) | `~9,400 ms` (Kaleido headless Chromium IPC) | `~11,000 ms` (`export_svg` via headless browser) |
| **Cold-Start to First Frame (`N = 3` points, Cold Process)** | **1.69 ms** (Auto CPU first frame) / **66.25 ms** (Full Vulkan + 384 MiB GPU slabs) | `~1,800 ms` (cold `python3 -c "import matplotlib.pyplot..."`) | `~2,450 ms` (cold `import seaborn, pandas, scipy`) | `~2,100 ms` (cold Python import + headless Chromium startup) | `~1,950 ms` (cold Python import + Tornado/browser init) |
| **Memory Footprint (`RSS` at Idle)** | **3.96 MB** (bare) / **4.54 MB** (post-library-load) / **186.44 MB** (post-GPU-init with 128 MiB staging slab) | `~158 MB` (post-import + first empty `Figure` RSS) | `~245 MB` (post-import `seaborn` + `scipy` + `pandas` RSS) | `~380 MB` (Python + embedded Chromium renderer processes) | `~310 MB` (Python + Tornado + browser tab RSS) |

---

## 3. Superseded Pre-Audit Benchmark Claims (Preserved for Traceability)

In accordance with the Ecosystem Benchmark Integrity Audit ([`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md)), the initial pre-audit single-number benchmark rows are preserved below, marked **Superseded**, and linked to their revised multi-way rows in Section 1:

| Superseded Pre-Audit Claim | Audit Finding | Revised Audited Replacement |
| :--- | :--- | :--- |
| ~~`Interactive Scatter (10M pts): 3,005.38 FPS`~~ (and `3,270.72 FPS` in `PRODUCTION.md`) | Used bare `"FPS"` without present mode and measured offscreen compute without swapchain presentation. | Replaced by Three-Way Split: **[1c (`1,847.96 FPS` presented `IMMEDIATE_KHR`)](#11-three-way-rendering-rate--interactive-jitter-800x600-framebuffer)**, **[1c\* (`60.00 FPS` presented `FIFO_KHR` vsync)](#11-three-way-rendering-rate--interactive-jitter-800x600-framebuffer)**, **[1a (`2,949.44 frames/s` offscreen full redraw)](#11-three-way-rendering-rate--interactive-jitter-800x600-framebuffer)**, and **[1b (`139,114.64 frames/s` CPU submission)](#11-three-way-rendering-rate--interactive-jitter-800x600-framebuffer)**. |
| ~~`Interactive Line (1M pts): 7,737.20 FPS`~~ (and `8,024.07 FPS` in `PRODUCTION.md`) | Used bare `"FPS"` without present mode and measured offscreen compute without swapchain presentation. | Replaced by Three-Way Split: **[3c (`2,637.96 FPS` presented `IMMEDIATE_KHR`)](#11-three-way-rendering-rate--interactive-jitter-800x600-framebuffer)**, **[3a (`7,342.73 frames/s` offscreen full redraw)](#11-three-way-rendering-rate--interactive-jitter-800x600-framebuffer)**, and **[3b (`141,816.17 frames/s` CPU submission)](#11-three-way-rendering-rate--interactive-jitter-800x600-framebuffer)**. |
| ~~`SVG Export (1M-pt scatter): 0.01 s (10.4 ms)`~~ | Combined in-memory generation and buffered `std::ofstream` write without `fsync(fd)`, without disclosing 12-thread `<path>` batching (`0%` decimation), and without post-measurement file validation. | Replaced by Four-Way Split: **[5a (`1.29 ms` validation+layout)](#12-four-way-1m-point-scatter-svg-vector-export-breakdown-cpu-zero-decimation)**, **[5b (`17.78 ms` 12-thread in-memory XML gen)](#12-four-way-1m-point-scatter-svg-vector-export-breakdown-cpu-zero-decimation)**, **[5c (`4.49 ms` `open+write+fsync`)](#12-four-way-1m-point-scatter-svg-vector-export-breakdown-cpu-zero-decimation)**, and **[5d (`13.19 MB`, `27` DOM elements, `1,000,000` subpaths)](#12-four-way-1m-point-scatter-svg-vector-export-breakdown-cpu-zero-decimation)**. |
| ~~`Memory Footprint (Idle): 4.34 MB`~~ (and `4.31 MB` in `PRODUCTION.md`) | Sampled RSS before `GpuDevice::create()` (`vkCreateInstance` lazy `dlopen` of NVIDIA ICD) without labeling the memory phase. | Replaced by Three-Way Phase Split: **[7a (`3.96 MB` bare process)](#13-cold-start-latency--three-way-memory-footprint-rss--gpu-pool)**, **[7b (`4.54 MB` post-library-load, pre-GPU)](#13-cold-start-latency--three-way-memory-footprint-rss--gpu-pool)**, and **[7c (`186.44 MB` post-GPU-init + 128 MiB staging slab)](#13-cold-start-latency--three-way-memory-footprint-rss--gpu-pool)**. |

---

## 4. Reproducing the Benchmark Suite & CI Gate

```bash
cmake --preset release-vulkan
cmake --build --preset release-vulkan -j
./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
cmake --build --preset release-vulkan --target bench-ci
```
