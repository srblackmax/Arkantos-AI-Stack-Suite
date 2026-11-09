# Darshana Benchmark Methodology

Every metric reported in [`BENCHMARKS.md`](./BENCHMARKS.md) and emitted to `results.json` / [`benchmarks/baseline.json`](./benchmarks/baseline.json) by [`bench/bench_darshana.cpp`](./bench/bench_darshana.cpp) follows the strict interval definitions, exclusion disclosures, hardware baselines, and post-measurement sanity checks documented below.

---

### 1a. 10M-Point Scatter Offscreen Throughput

**What is measured:** Full static base-layer GPU compute pass (`kClearGridSpv` + `kScatterSplatSpv` dispatching `39,063` compute workgroups of `256` threads across `10,000,000` device-local `DataPoint2D` vertices) plus interactive overlay compositing compute pass (`kOverlayCompSpv`) into an offscreen `800x600` RGBA storage buffer (`Figure::render_frame(true)`), timed from `vkQueueSubmit` through `vkWaitSemaphores` completion (`p50` across 11 batches of 30 frames after a 60-frame warm-up). Also reports the supporting cached-base interactive overlay offscreen rate (`Figure::update_overlay` + `Figure::render_frame(false)`), where only `kOverlayCompSpv` executes over the pre-rendered static base buffer.
**What is excluded:** Host dataset generation, initial host-to-device DMA upload (`Figure::bind_xy` / `GpuBuffer::upload_points_async`), host RGBA readback (`Figure::readback_image`), and window-system `VkSwapchainKHR` presentation.
**Instrumentation:** `darshana::monotonic_time_ns()` (`CLOCK_MONOTONIC` via `std::chrono::steady_clock`) around `Figure::render_frame(true)` with `VK_SEMAPHORE_TYPE_TIMELINE` synchronization (`vkWaitSemaphores`).
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe.
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Post-measurement `Figure::readback_image()` copies the `800x600` RGBA framebuffer from device memory to host memory and verifies `width == 800`, `height == 600`, and 64-bit FNV-1a checksum `fnv1a_hash() == 13884735082708671537` (`!= 0`), proving all 10M points and overlay pixels were rendered to the offscreen target.

---

### 1b. 10M-Point Scatter CPU Submission Rate

**What is measured:** Rate at which the CPU records and submits Vulkan command buffers (`vkResetCommandBuffer` + `vkBeginCommandBuffer` + command recording + `vkEndCommandBuffer` + `vkQueueSubmit`) into an 8-slot in-flight command-buffer ring synchronized with a Vulkan 1.2 timeline semaphore (`VK_SEMAPHORE_TYPE_TIMELINE`) without blocking on per-frame GPU execution (`p50` across 11 batches of 500 submissions after a 200-submission warm-up).
**What is excluded:** Per-frame synchronous GPU completion wait (bounded only when the 8-slot in-flight ring wraps via `vkWaitSemaphores` on `next_val - 8`), dataset upload, and swapchain presentation.
**Instrumentation:** `darshana::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around the command-buffer ring recording and `vkQueueSubmit` loop, followed by a final `vkWaitSemaphores` drain.
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe.
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Verifies that all `11 * 500 = 5,500` submitted command buffers advance and retire the Vulkan timeline semaphore monotonically to the final target value without `VK_ERROR_DEVICE_LOST`.

---

### 1c. 10M-Point Scatter Presented FPS

**What is measured:** End-to-end presented frame rate on a real `VkSwapchainKHR` (`800x600` `VK_FORMAT_B8G8R8A8_UNORM`), timed between consecutive `vkQueuePresentKHR` completions across 100 presented frames (after 5 warm-up frames), including `vkAcquireNextImageKHR`, full 10M-point GPU compute redraw (`Figure::render_frame(true)`), `vkCmdCopyBufferToImage` into the swapchain `VkImage`, image layout transitions (`UNDEFINED -> TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR`), `vkQueueSubmit`, `vkQueuePresentKHR`, and `vkWaitForFences`. Measured in both unthrottled `VK_PRESENT_MODE_IMMEDIATE_KHR` and vblank-locked `VK_PRESENT_MODE_FIFO_KHR` (`60.00 Hz` display refresh rate). The harness attempts `VK_EXT_headless_surface` first and falls back to a hidden X11 window via `VK_KHR_xcb_surface` when the discrete NVIDIA driver does not expose swapchain formats on headless surfaces.
**What is excluded:** Swapchain/window creation (`xcb_create_window`, `vkCreateSwapchainKHR`) and initial 10M-point host-to-device staging upload.
**Instrumentation:** `darshana::monotonic_time_ns()` (`CLOCK_MONOTONIC`) recording inter-present intervals between consecutive `vkQueuePresentKHR` + `vkWaitForFences` completions.
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe, dual `1920x1080@60.00Hz` displays (`HDMI-0`, `DP-0`).
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Verifies `vkCreateSwapchainKHR`, `vkAcquireNextImageKHR`, and `vkQueuePresentKHR` return `VK_SUCCESS` across all presented frames and that the rendered `800x600` framebuffer has non-zero FNV-1a checksum (`13884735082708671537`).

---

### 2. 10M-Point Scatter Interactive Overlay Jitter

**What is measured:** Frame-time dispersion (`p99 - p50`, in milliseconds) across `11 * 150 = 1,650` consecutive interactive crosshair overlay updates (`Figure::update_overlay` + `Figure::render_frame(false)`) over a pre-rendered 10M-point static base layer on the GPU.
**What is excluded:** Static base-layer redraw (`base_dirty == false`; the 10M-point static layer remains cached in device-local VRAM) and host pixel readback.
**Instrumentation:** `darshana::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around each `update_overlay` + `render_frame(false)` call, sorting all per-frame durations to extract `p50` and `p99`.
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe.
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Verifies `p99 - p50 < 1.0 ms` and confirms via `Figure::base_redraw_count()` and `Figure::readback_image()` that the overlay crosshair is composited without redrawing the static base layer.

---

### 3a. 1M-Point Line Offscreen Throughput

**What is measured:** Full static base-layer GPU compute pass (`kClearGridSpv` + `kLineSdfSpv` dispatching `3,907` compute workgroups of `256` threads across `1,000,000` connected line vertices with analytical capsule SDF anti-aliasing) plus `kOverlayCompSpv` into an offscreen `800x600` RGBA storage buffer (`Figure::render_frame(true)`), timed from `vkQueueSubmit` through `vkWaitSemaphores` completion (`p50` across 11 batches of 40 frames after a 60-frame warm-up).
**What is excluded:** Host dataset generation, initial host-to-device DMA upload (`Figure::bind_xy`), host RGBA readback, and swapchain presentation.
**Instrumentation:** `darshana::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around `Figure::render_frame(true)` with `VK_SEMAPHORE_TYPE_TIMELINE` synchronization.
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe.
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Post-measurement `Figure::readback_image()` verifies `width == 800`, `height == 600`, and 64-bit FNV-1a checksum `fnv1a_hash() == 17812521895849381041` (`!= 0`).

---

### 3b. 1M-Point Line CPU Submission Rate

**What is measured:** Rate at which the CPU records and submits Vulkan command buffers for the 1M-point line pass (`3,907` workgroups) into an 8-slot in-flight command-buffer ring with timeline semaphore synchronization (`VK_SEMAPHORE_TYPE_TIMELINE`), without blocking on per-frame GPU completion (`p50` across 11 batches of 500 submissions after a 200-submission warm-up).
**What is excluded:** Per-frame synchronous GPU wait (bounded only at ring wrap `next_val - 8`), dataset upload, and swapchain presentation.
**Instrumentation:** `darshana::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around the command-buffer recording and `vkQueueSubmit` loop.
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe.
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Verifies all `5,500` submitted command buffers complete and advance the timeline semaphore to the final expected counter value.

---

### 3c. 1M-Point Line Presented FPS

**What is measured:** End-to-end presented frame rate for the 1M-point SDF line plot on a real `VkSwapchainKHR` (`800x600` `VK_FORMAT_B8G8R8A8_UNORM` on `VK_KHR_xcb_surface` with `VK_PRESENT_MODE_IMMEDIATE_KHR`), timed between consecutive `vkQueuePresentKHR` completions across 150 presented frames (after 5 warm-up frames), including `vkAcquireNextImageKHR`, full 1M-point GPU compute redraw (`Figure::render_frame(true)`), `vkCmdCopyBufferToImage` into the swapchain image, layout barriers, `vkQueueSubmit`, `vkQueuePresentKHR`, and `vkWaitForFences`.
**What is excluded:** Swapchain/window initialization and initial 1M-point host-to-device staging upload.
**Instrumentation:** `darshana::monotonic_time_ns()` (`CLOCK_MONOTONIC`) across consecutive `vkQueuePresentKHR` + `vkWaitForFences` completions.
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe, dual `1920x1080@60.00Hz` displays.
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Verifies `VK_SUCCESS` on all 150 `vkQueuePresentKHR` calls and non-zero FNV-1a framebuffer checksum (`17812521895849381041`).

---

### 4. Streaming Ring-Buffer Frame Jitter

**What is measured:** Frame-time dispersion (`p99 - p50`, in milliseconds) across 300 consecutive streaming updates (`Figure::append_stream(x, y)` into the O(1) lock-free ring buffer followed by `Figure::render_frame(false)` uploading the ring buffer via host-coherent staging DMA and executing the GPU compute pass).
**What is excluded:** Initial `Figure::create` ring-buffer allocation (`capacity = 8,192`) and post-loop image readback.
**Instrumentation:** `darshana::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around each `append_stream` + `render_frame(false)` tick.
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe.
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Verifies `p99 - p50 < 1.0 ms`, zero per-frame GPU pool allocations (`gpu_allocations_in_frame == 0`), and non-zero `readback_image().fnv1a_hash()`.

---

### 5a. SVG Export Spec Validation and Layout Time

**What is measured:** In-memory declarative `Spec::validate()` verification plus `SnapshotData::from_xy` finiteness validation and coordinate layout preparation for `1,000,000` 2D points (`p50` and `p99` across 9 timed runs after 2 warm-up runs).
**What is excluded:** XML string formatting, file descriptor creation, disk writes, and `fsync`.
**Instrumentation:** `darshana::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around `scatter_spec.validate()` and `SnapshotData::from_xy(x_1m, y_1m)`.
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe.
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Verifies `val_res.has_value() && snap_1m.has_value() && snap_1m->point_count() == 1,000,000`.

---

### 5b. SVG Export In-Memory XML Generation Time

**What is measured:** Time to serialize all `1,000,000` 2D scatter points and plot chrome into a complete `13.19 MB` (`13,828,175`-byte) in-memory SVG 1.1 document string via `darshana::render_svg_string(scatter_spec, &*snap_1m)` (`p50` and `p99` across 9 timed runs after 2 warm-up runs). Uses `12` worker `std::jthread`s (`83,334` points per thread) and branchless `0.1 px` fixed-point integer formatting (`append_coord_tenths`) emitting `12` `<path>` elements containing `1,000,000` `M<x> <y>h0` round-capped point subpaths (`0%` decimation).
**What is excluded:** Initial `SnapshotData::from_xy` construction and POSIX disk I/O (`open` + `write` + `fsync`).
**Instrumentation:** `darshana::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around `darshana::render_svg_string`.
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe.
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Verifies in-memory string size `> 13,000,000` bytes and exact presence of `1,000,000` `M<x> <y>h0` subpath point commands.

---

### 5c. SVG Export Write-to-Disk Time with fsync

**What is measured:** Time to open `/tmp/darshana_bench_1m.svg` (`::open` with `O_WRONLY | O_CREAT | O_TRUNC`), write the XML prolog (`<?xml version="1.0" encoding="UTF-8"?>\n`) and the `13.19 MB` SVG payload (`::write`), flush kernel page-cache buffers to the physical NVMe storage media via `::fsync(fd)`, and close the descriptor (`::close(fd)`) (`p50` and `p99` across 9 timed runs after 2 warm-up runs).
**What is excluded:** In-memory `Spec` validation and in-memory XML string generation (reported separately in 5a and 5b).
**Instrumentation:** `darshana::monotonic_time_ns()` (`CLOCK_MONOTONIC`) around `::open` + `write_all_fd` + `::fsync` + `::close`.
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe (`/tmp` mounted on `nvme0n1`).
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Reads `/tmp/darshana_bench_1m.svg` back from disk and asserts: (1) file exists and `file_size == 13,828,214` bytes (`> 0`), (2) first 4 KB begins with `"<?xml"` and contains `"<svg"`, (3) XML DOM element count equals `27` (`1 <svg>`, `2 <rect>`, `1 <g>`, `8 <line>`, `3 <text>`, `12 <path>`) and point subpath count equals `1,000,000` (`±1%`), and (4) trailing bytes end with `"</svg>"`.

---

### 5d. SVG Export Final File Size and Element Count

**What is measured:** Exact byte size on disk (`std::filesystem::file_size`), XML DOM element count (`27` elements), and encoded point subpath count (`1,000,000` `M<x> <y>h0` commands) of the exported 1M-point scatter SVG document.
**What is excluded:** Timing intervals (covered in 5a, 5b, and 5c).
**Instrumentation:** `std::filesystem::file_size` and byte-level lexical scan of the written disk file prior to cleanup.
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe.
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Fails the benchmark executable (`exit 1`) if `final_svg_bytes != 13,828,214`, `svg_dom_elements != 27`, or `svg_point_subpaths != 1,000,000`.

---

### 6. Cold-Start Latency

**What is measured:** Two distinct cold-start intervals:
1. **Lightweight Auto Mode (`cold_start_auto_ms`):** Time to construct a 3-point `Spec`, create a `Figure` (`BackendKind::Auto`, which defers heavy 384 MiB GPU slab initialization for small inline plots), and render its first `800x600` frame (`p50` across 15 fresh `Figure` lifecycles).
2. **Full Vulkan 1.2 GPU + 384 MiB Slab Init (`cold_start_vulkan_gpu_init_ms`):** One-shot cold initialization time across `GpuDevice::create()` (`vkCreateInstance`, `dlopen` of `libGLX_nvidia.so.0` / `libnvidia-glcore.so.550.163.01`, `vkCreateDevice` on RTX 3070, 4 SPIR-V compute pipelines, `256 MiB` device-local slab allocation, `128 MiB` host-coherent staging slab allocation and persistent mapping) plus `Figure::create_with_device`, first GPU compute frame `render_frame(true)`, and `wait_idle()`.
**What is excluded:** OS process `execve()` loader startup prior to `main()`.
**Instrumentation:** `darshana::monotonic_time_ns()` (`CLOCK_MONOTONIC`).
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe.
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Reads back the rendered `800x600` first frame (`cold_fig->readback_image()`) and verifies `width == 800`, `height == 600`, and `fnv1a_hash() != 0`.

---

### 7a. Bare Process RSS

**What is measured:** Resident Set Size (`VmRSS` from `/proc/self/status`) sampled at the very first statement of `main()`, after the dynamic linker (`ld-linux-x86-64.so.2`) maps `libdarshana.so.1` and the lightweight Khronos loader `libvulkan.so.1`, but before any `darshana::Spec`, `darshana::Figure`, or `darshana::GpuDevice` objects are constructed and before `vkCreateInstance` `dlopen()`s the NVIDIA ICD (`libGLX_nvidia.so.0` / `libnvidia-glcore.so.550.163.01`).
**What is excluded:** Darshana context allocations (`ScratchArena`, `StreamingBuffer`) and Vulkan ICD driver/slab memory.
**Instrumentation:** Parsing `VmRSS:` in `/proc/self/status` (in KiB, divided by `1024.0` to MiB).
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe.
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Verifies `rss_bare_process_mb > 0.5 MB` and `rss_bare_process_mb <= rss_post_library_load_mb`.

---

### 7b. Post-Library-Load RSS

**What is measured:** Resident Set Size (`VmRSS` from `/proc/self/status`) sampled after constructing `darshana::Spec` and `darshana::Figure` (`128 KiB` `ScratchArena` + `StreamingBuffer` ring buffer), prior to invoking `GpuDevice::create()`. This is the headline idle CPU RSS of a Darshana figure before the Vulkan loader lazily `dlopen()`s the vendor ICD shared libraries inside `vkCreateInstance`.
**What is excluded:** Vendor Vulkan ICD (`libnvidia-glcore.so.550.163.01`) and the `128 MiB` host-mapped Vulkan staging pool (which are measured in 7c).
**Instrumentation:** Parsing `VmRSS:` in `/proc/self/status` immediately after `Figure::create(cold_spec, BackendKind::Auto, 1024U)`.
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe.
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Verifies `rss_post_library_load_mb >= rss_bare_process_mb` and `rss_post_library_load_mb < 30.0 MB`.

---

### 7c. Post-GPU-Init RSS

**What is measured:** Resident Set Size (`VmRSS` from `/proc/self/status`) sampled after `GpuDevice::create()`, one warm-up GPU compute frame (`Figure::render_frame(true)` + `GpuDevice::wait_idle()`), and a 1-second idle settle (`std::this_thread::sleep_for(std::chrono::seconds(1))`). Includes the `dlopen()`-ed NVIDIA Vulkan ICD (`libGLX_nvidia.so.0` + `libnvidia-glcore.so.550.163.01`) and the persistently mapped `128 MiB` (`VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`) staging slab.
**What is excluded:** Bulk 10M-point host `std::vector<float>` test arrays (allocated after 7c is sampled).
**Instrumentation:** Parsing `VmRSS:` in `/proc/self/status` after `GpuDevice::create()` + 1 warm-up frame + 1s sleep.
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe.
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Verifies `rss_post_gpu_init_mb > rss_post_library_load_mb` (`~186.44 MB` vs `~4.54 MB`).

---

### 8. Peak GPU Pool Memory

**What is measured:** Peak sub-allocated bytes (`FrameStats::global().snapshot().peak_gpu_bytes`) from Darshana's pre-allocated `256.00 MiB` device-local `VkBuffer` slab during the 10M-point scatter benchmark (`10,000,000 * 8 bytes = 76.29 MiB` point buffer + two `800x600 * 4 bytes = 1.83 MiB` base/output RGBA framebuffers = `79.96 MiB`).
**What is excluded:** Unallocated free blocks remaining inside the pre-allocated `256.00 MiB` device-local slab and `128.00 MiB` host-coherent staging slab.
**Instrumentation:** Atomic high-water-mark counter `GpuDevice::peak_allocated_bytes()` recorded in `GpuBuffer::allocate`.
**Hardware:** 12th Gen Intel(R) Core(TM) i7-12700K (12C/20T), NVIDIA GeForce RTX 3070 (8192 MiB GDDR6), NVIDIA Driver 550.163.01 / Vulkan 1.4.309, Linux 6.12.113+deb13-amd64 x86_64, 48 GiB Dual-Channel DDR4, Samsung SSD 970 EVO Plus 1TB NVMe.
**Command to reproduce:**
    ./build-release-vulkan/bench_darshana --json build-release-vulkan/results.json
**Sanity check:** Verifies `peak_gpu_mb > 75.0 MB` and `peak_gpu_mb < 2048.0 MB` (`79.96 MB` exact).
