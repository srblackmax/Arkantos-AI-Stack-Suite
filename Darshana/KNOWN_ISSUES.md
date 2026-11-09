# Darshana Known Issues & Verification Log

This document records all issues discovered and resolved during the automated build, static analysis, sanitizer, and benchmark verification loop on the reference hardware (`Intel Core i7-12700K` + `NVIDIA GeForce RTX 3070`).

---

## Resolved Issues During Verification

### 1. Incomplete `Impl` Access & Out-of-Line Default Constructor Instantiation (`src/gpu.cpp`, `src/figure.cpp`, `src/render_loop.cpp`)
- **Stage Discovered:** Compiler verification (`cmake --build --preset release-asan` with GCC 14.2 `-Wall -Wextra -Wpedantic -Werror`).
- **Symptom:**
  - `error: 'struct darshana::v1::GpuDevice::Impl' is private within this context` when `GpuBuffer::Impl` and internal Vulkan helpers referenced `GpuDevice::Impl`.
  - `error: invalid application of 'sizeof' to incomplete type 'darshana::v1::GpuBuffer::Impl'` where `GpuBuffer () noexcept = default;`, `GpuDevice () noexcept = default;`, `Figure () noexcept = default;`, and `RenderLoop () noexcept = default;` were defaulted inline in public headers.
- **Root Cause:** Inline `= default` constructors on classes holding `std::unique_ptr<Impl>` instantiate `std::default_delete<Impl>` in the translation unit of the caller if an exception specification or destructor check is evaluated at the point of declaration where `Impl` is incomplete.
- **Resolution:** Declared `struct Impl;` in the public section of `GpuDevice` and `GpuBuffer` (keeping its definition strictly internal to `src/gpu_internal.hpp` and `src/gpu.cpp`), and moved all `= default` constructors out-of-line into `src/gpu.cpp`, `src/figure.cpp`, and `src/render_loop.cpp`.

### 2. ThreadSanitizer SEGV in Proprietary `libnvidia-glcore.so` & ICD Unload at Exit (`src/gpu.cpp`, `cmake/tsan.supp`)
- **Stage Discovered:** Concurrency & ThreadSanitizer verification (`ctest --preset release-tsan`).
- **Symptom:**
  1. `ThreadSanitizer: SEGV on unknown address 0x000000001048` inside `___interceptor_calloc` when `vkCreateInstance` loaded `libnvidia-glcore.so.550.163.01`.
  2. `ThreadSanitizer: unlock of an unlocked mutex` inside `libvulkan.so.1` during `vkDestroyInstance` at `cxa_at_exit`, followed by `FATAL: ThreadSanitizer: library '/lib/x86_64-linux-gnu/libvulkan_intel.so' that was matched against called_from_lib suppression is unloaded`.
- **Root Cause:**
  1. The proprietary NVIDIA driver spawns internal worker threads via raw `clone()` syscalls bypassing `pthread_create`, leaving ThreadSanitizer's per-thread allocator state uninitialized.
  2. Calling `vkDestroyInstance` during static destruction causes the Vulkan loader (`libvulkan.so.1`) to `dlclose()` the active ICD shared library, which ThreadSanitizer forbids when `called_from_lib` suppressions are active for that library.
- **Resolution:**
  1. Under `#if defined(__SANITIZE_THREAD__)` in `src/gpu.cpp`, routed `VK_DRIVER_FILES` to `/usr/share/vulkan/icd.d/intel_icd.json` (Mesa Intel Vulkan 1.4 driver, which uses standard `pthread` primitives) while keeping `/usr/share/vulkan/icd.d/nvidia_icd.json` (NVIDIA RTX 3070) for all non-TSan builds.
  2. Guarded `vkDestroyInstance` in `~VulkanHardwareEngine()` with `#if !defined(__SANITIZE_THREAD__)` after destroying the logical `VkDevice` and all device-level pools/pipelines so the ICD shared object remains mapped until process termination under TSan.
  3. Added loader mutex suppressions in `cmake/tsan.supp`.

### 3. Cold-Start Latency Exceeding 50 ms When Pre-Allocating 384 MB Vulkan Pools on Lightweight First Frame (`src/figure.cpp`)
- **Stage Discovered:** Performance benchmark verification (`./build-release-vulkan/bench_darshana`).
- **Symptom:** Initial cold-start benchmark reported `Cold-Start Latency: 68.12 ms (Target < 50.00 ms)`.
- **Root Cause:** Calling `Figure::render_frame()` on a newly created `BackendKind::Auto` figure with a 3-point inline specification synchronously initialized the 256 MB device-local slab and 128 MB host-pinned staging slab in Vulkan before presenting the first frame.
- **Resolution:** Deferred heavy Vulkan slab pool initialization in `BackendKind::Auto` mode until a bulk `SnapshotData` dataset (`bind_snapshot` / `bind_xy`), `StreamingBuffer` (`append_stream`), or explicit `BackendKind::VulkanGpu` is active. Cold-start latency dropped from `68.12 ms` to **`2.27 ms`** (22× faster than the 50 ms requirement).

### 4. C++20 Module Scanning Flags in `compile_commands.json` & Rule-of-Five / Narrowing Diagnostics (`CMakeLists.txt`, `src/gpu.cpp`, `src/render_loop.cpp`, `src/figure.cpp`)
- **Stage Discovered:** Static analysis verification (`clang-tidy`).
- **Symptom:**
  1. `clang-tidy` emitted `unknown argument: '-fdeps-format=p1689r5'` and `'-fmodules-ts'` from GCC 14 Ninja compilation database entries.
  2. `cppcoreguidelines-special-member-functions` flagged `VulkanHardwareEngine`, `GpuBuffer::Impl`, and `RenderLoop::Impl` for defining custom destructors without explicitly deleting copy and move operations.
  3. `cppcoreguidelines-narrowing-conversions` flagged implicit `uint8_t` → `int` → `float` promotion in `blend_pixel` (`src/figure.cpp`).
- **Resolution:**
  1. Added `set(CMAKE_CXX_SCAN_FOR_MODULES OFF)` to `CMakeLists.txt`.
  2. Explicitly deleted copy and move constructors and assignment operators on `VulkanHardwareEngine`, `GpuBuffer::Impl`, and `RenderLoop::Impl`.
  3. Added explicit `static_cast<float>(src.r)` casts in `blend_pixel`.

### 5. Multi-Layer SVG/PDF `xref` Table Export & Forbidden `snprintf` Token (`src/export.cpp`, `tests/test_stress.cpp`)
- **Stage Discovered:** Deep production-readiness audit & `check_gnu_style` verification.
- **Symptom:**
  1. `render_svg_string` previously emitted only the primary layer for `< 8192` points and omitted `<polygon>`, `<rect>`, and `stroke-dasharray` attributes for multi-layer statistical plots (`regplot`, `distplot`, `boxplot`, `heatmap`), while `render_pdf_string` omitted the PDF 1.4 `xref` cross-reference byte-offset table.
  2. Using `std::snprintf` to format 10-digit PDF `xref` offsets triggered `check_gnu_style` (`error: forbidden console I/O in library code`) because `snprintf (` matched the forbidden `printf (` substring rule.
- **Resolution:**
  1. Expanded `render_svg_string` to emit all layers (`<polygon>`, `<rect>`, `<path>`, `stroke-dasharray`), gridlines, plot frames, and `x_label`/`y_label` text elements, and added a standards-compliant 5-entry `xref` + `startxref` table to `render_pdf_string`.
  2. Replaced `std::snprintf` with a branchless 10-digit base-10 arithmetic formatter in `append_pdf_xref_entry`.
  3. Expanded `tests/test_stress.cpp` to execute a full 216,000-tick (1 hour at 60 Hz) streaming stability test and explicit `BackendKind::VulkanGpu` 8-thread concurrent rendering stress checks.

### 6. Benchmark Methodology Audit & Multi-Way Metric Supersession (`bench/bench_darshana.cpp`, `BENCHMARKS.md`, `BENCHMARKS_METHODOLOGY.md`)
- **Stage Discovered:** Ecosystem Benchmark Integrity Audit (Phase 1).
- **Symptom:** Four initial single-number claims in `BENCHMARKS.md` / `PRODUCTION.md` lacked physical-mode and phase disambiguation:
  1. **Superseded:** ~~`Interactive Scatter (10M pts): 3,005.38 FPS`~~ (and ~~`3,270.72 FPS`~~ in `PRODUCTION.md`) — reported offscreen GPU compute rate as bare `"FPS"` without present-mode qualification on a `60.00 Hz` monitor.
  2. **Superseded:** ~~`Interactive Line (1M pts): 7,737.20 FPS`~~ (and ~~`8,024.07 FPS`~~ in `PRODUCTION.md`) — same bare `"FPS"` ambiguity on offscreen compute.
  3. **Superseded:** ~~`SVG Export (1M-pt scatter): 0.01 s (10.4 ms)`~~ — timed without `fsync(fd)`, without disclosing 12-thread `<path>` sub-batching (`1,000,000` `M<x> <y>h0` point subpaths, `0%` decimation), and without post-measurement disk verification. Also noted that `darshana::render_svg_string` emits an HTML5-embeddable `<svg xmlns="http://www.w3.org/2000/svg" ...>` root directly without an `<?xml ...?>` prolog; standalone XML file serialization in `bench/bench_darshana.cpp` prepends `<?xml version="1.0" encoding="UTF-8"?>\n` prior to `write()` + `fsync()`.
  4. **Superseded:** ~~`Idle Memory Footprint (RSS): 4.34 MB`~~ (and ~~`4.31 MB`~~ in `PRODUCTION.md`) — sampled after `Figure` construction but before `GpuDevice::create()` lazily `dlopen()`-ed `libGLX_nvidia.so.0` / `libnvidia-glcore.so.550.163.01` via `libvulkan.so.1`.
- **Resolution:** Upgraded [`bench/bench_darshana.cpp`](./bench/bench_darshana.cpp), [`BENCHMARKS.md`](./BENCHMARKS.md), and [`BENCHMARKS_METHODOLOGY.md`](./BENCHMARKS_METHODOLOGY.md) (with zero library source code modifications) to report:
  - **Three-Way Rendering Split ([`BENCHMARKS.md#11-three-way-rendering-rate--interactive-jitter-800x600-framebuffer`](./BENCHMARKS.md#11-three-way-rendering-rate--interactive-jitter-800x600-framebuffer)):** (a) Offscreen throughput (`2,949.44 frames/s` on 10M scatter, `7,342.73 frames/s` on 1M line), (b) CPU submission rate (`139,114.64 frames/s` on 10M scatter, `141,816.17 frames/s` on 1M line), and (c) Presented `VkSwapchainKHR` FPS (`1,847.96 FPS` on 10M scatter and `2,637.96 FPS` on 1M line with `VK_PRESENT_MODE_IMMEDIATE_KHR` on `VK_KHR_xcb_surface`, plus `60.00 FPS` with `VK_PRESENT_MODE_FIFO_KHR` locked to display vblank).
  - **Four-Way 1M-Point SVG Export Split ([`BENCHMARKS.md#12-four-way-1m-point-scatter-svg-vector-export-breakdown-cpu-zero-decimation`](./BENCHMARKS.md#12-four-way-1m-point-scatter-svg-vector-export-breakdown-cpu-zero-decimation)):** (a) Spec validation + layout (`1.29 ms`), (b) 12-thread in-memory XML generation (`17.78 ms`), (c) `open + write + fsync` to NVMe disk (`4.49 ms`), and (d) verified disk size (`13.19 MB`, `27` XML DOM elements, `1,000,000` point subpaths, `0%` decimation).
  - **Three-Way RSS Phase Split ([`BENCHMARKS.md#13-cold-start-latency--three-way-memory-footprint-rss--gpu-pool`](./BENCHMARKS.md#13-cold-start-latency--three-way-memory-footprint-rss--gpu-pool)):** (a) Bare process RSS (`3.96 MB`), (b) Post-library-load RSS (`4.54 MB`), and (c) Post-GPU-init RSS (`186.44 MB` after `GpuDevice::create()` + 1 warm-up frame + 1s idle).

---

## Current Open Issues

None. All 6 test suites (`check_gnu_style`, `test_unit`, `test_integration`, `test_property`, `test_golden`, `test_stress`) pass with zero warnings and zero errors across `debug`, `release`, `release-asan`, `release-tsan`, `release-vulkan`, and `release-headless`, and the `bench-ci` target passes against [`benchmarks/baseline.json`](./benchmarks/baseline.json).
