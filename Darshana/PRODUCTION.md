# Darshana Production Operations & Architecture Guide

**Darshana** (`namespace darshana::inline v1`, CMake target `Darshana::Darshana`) is a production-grade C++23 visualization and experimentation engine with zero Python dependencies, built for real-time GPU rendering and high-throughput CPU vector export.

---

## 1. Hardware & System Requirements

### Reference & Target Hardware
- **GPU (Interactive Vulkan Backend):**
  - Vulkan 1.2+ capable discrete or integrated GPU with Timeline Semaphore (`VK_KHR_timeline_semaphore` / Vulkan 1.2 core `VkPhysicalDeviceVulkan12Features::timelineSemaphore`) and storage buffer compute support.
  - Reference GPU: **NVIDIA GeForce RTX 3070 (8 GB GDDR6 VRAM)** using Vulkan 1.4 (`550.163.01`).
  - Automatic Graceful Fallback: If no Vulkan GPU is available or `DARSHANA_ENABLE_VULKAN=OFF` (`release-headless` preset), `darshana::Figure` and `darshana::GpuDevice` automatically degrade to the multi-threaded SIMD CPU renderer with a structured warning log, never crashing or aborting.
- **CPU (Vector Export & Statistical Compute):**
  - x86-64-v3 (AVX2 + FMA) baseline by default; `-march=native` (AVX-512 / Alder Lake optimization) when `DARSHANA_BUILD_FAT_NATIVE=ON`.
  - Reference CPU: **Intel Core i7-12700K (12 cores / 20 threads)**.
- **Compiler & OS:**
  - Linux (`x86_64`), GCC 14+ or Clang 18+ with full C++23 support (`std::expected`, `std::span`, `std::jthread`, `std::pmr`).
  - CMake 3.28+ and Ninja.

---

## 2. Memory Budgeting & Allocation Model

Darshana enforces a strict zero-raw-`new`/`delete` policy and eliminates heap allocations from the per-frame hot path:

### Per-Frame CPU Scratch Arena (`darshana::ScratchArena`)
- Backed by a pre-allocated buffer and `std::pmr::monotonic_buffer_resource` teamed with `std::pmr::null_memory_resource()` (preventing upstream heap fallback on exhaustion).
- Default per-figure scratch capacity: **4 MiB** (`4 * 1024 * 1024` bytes).
- Reclaimed in $O(1)$ time at the start of every frame via `ScratchArena::reset()`.
- Used for tessellation intermediates, layout boxes, and per-frame coordinate transformations.

### Vulkan GPU Pool Allocator (`darshana::GpuDevice` & `darshana::GpuBuffer`)
- Pre-allocates two contiguous memory slabs at device initialization:
  - **Device-Local Slab (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`):** `256 MiB` default (`GpuPoolConfig::device_local_pool_bytes`), sub-allocated on 256-byte alignment boundaries with deterministic coalescing free-list reclamation on `GpuBuffer` destruction.
  - **Host-Pinned Staging Slab (`VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`):** `128 MiB` default (`GpuPoolConfig::staging_pool_bytes`), persistently mapped via `vkMapMemory` for zero-allocation asynchronous DMA uploads.
- **Budget Enforcement:** Peak GPU memory usage for a 10M-point `DataPoint2D` scatter plot is **79.96 MiB** (`80,000,000` bytes for points + framebuffer slabs), well within the **2,048 MiB** (`2 GiB`) budget on an 8 GB RTX 3070.
- **Idle RSS Footprint:** **4.34 MiB** (Target `< 30.00 MiB`).

### Dual Data Path (`StreamingBuffer` vs. `SnapshotData`)
- **`darshana::StreamingBuffer`:** Lock-free $O(1)$ circular ring buffer pre-allocated to maximum capacity at creation. Zero allocations occur during `push()` or `push_batch()`.
- **`darshana::SnapshotData`:** Immutable or batch-updated columnar store with contiguous `DataPoint2D` packing for single-transfer GPU staging uploads.

---

## 3. Concurrency & Synchronization Model

### Zero Global Mutable State & Re-Entrant Figures
- Every `darshana::Figure` instance owns its private state, `ScratchArena`, cached base-layer RGBA buffer (`base_rgba`), composited output buffer (`final_rgba`), and GPU sub-allocations (`gpu_points_buf`, `gpu_base_buf`, `gpu_out_buf`).
- `N` threads can concurrently construct, mutate, render, and export independent `Figure` objects without contention.

### Multi-Slot Vulkan Command Recording & Timeline Semaphores
- `VulkanHardwareEngine` maintains **16 independent worker slots** (`WorkerSlot`), each owning a dedicated `VkCommandPool`, `VkCommandBuffer`, `VkDescriptorPool`, and `VkDescriptorSet`.
- Concurrent threads acquire a worker slot via atomic round-robin + `try_lock`, recording compute dispatches (`clear_grid`, `scatter_splat`, `line_sdf`, `overlay_comp`) in parallel without locking other recording threads.
- Queue submissions synchronize via a Vulkan 1.2 **Timeline Semaphore** (`VK_SEMAPHORE_TYPE_TIMELINE`) with monotonically increasing 64-bit counter values (`vkWaitSemaphores`).

### Dedicated Asynchronous `RenderLoop`
- `darshana::RenderLoop` wraps a `std::jthread` worker and two lock-free single-producer/single-consumer (SPSC) ring queues (`cmd_queue_` and `done_queue_`, 1024 slots each):
  - **UI / Producer Thread:** Submits non-blocking `submit_overlay()`, `submit_stream_point()`, `submit_viewport()`, and `request_frame()` commands in $O(1)$ time.
  - **Render Thread:** Drains pending updates, executes layered rendering (skipping the static data pass whenever `base_dirty == false`), and pushes `CompletedFrame` telemetry to the completion queue.

### Asynchronous Vector Export
- `darshana::export_file_async()` and `Figure::export_async()` clone the declarative `Spec` and snapshot data, launching multi-threaded CPU SVG/PDF generation on a background task returning `std::future<Result<ExportResult>>` and invoking an optional `ExportCallback` on completion without stalling interactive rendering.

---

## 4. Error Taxonomy & Observability

### Typed `std::expected<T, darshana::Error>` API
All fallible public methods return `darshana::Result<T>` (`std::expected<T, darshana::Error>`). `Error::payload` is a `std::variant<GpuError, DataError, ExportError, FrameworkError>`:

| Category | Variant Type | Representative Error Codes | Recovery Strategy |
| :--- | :--- | :--- | :--- |
| **GPU** | `darshana::GpuError` | `DeviceUnavailable`, `DeviceLost`, `OutOfPoolMemory`, `PipelineCreationFailed`, `SynchronizationFailed` | Automatic fallback to CPU renderer on `DeviceUnavailable`; resize pool or downsample on `OutOfPoolMemory`. |
| **Data** | `darshana::DataError` | `EmptyDataset`, `InvalidSpec`, `DimensionMismatch`, `NonFiniteValue`, `DomainRangeInvalid`, `ParseError` | Reject invalid update while preserving previous valid `Figure` state. |
| **Export** | `darshana::ExportError` | `UnsupportedFormat`, `IoOpenFailed`, `IoWriteFailed`, `RasterizationFailed` | Report filesystem path and `errno` message to caller via `Result<ExportResult>`. |
| **Framework** | `darshana::FrameworkError` | `InvalidState`, `ScratchArenaExhausted`, `QueueFull`, `ThreadPoolShutdown` | Increase `ScratchArena` capacity or drain `RenderLoop` completion queue. |

### Structured Logging & Frame Telemetry
- Library code emits zero `std::cout`, `std::cerr`, `printf`, or `std::endl` calls.
- Consumers register a thread-safe structured logger via `darshana::set_logger_sink()` (with JSON formatting via `darshana::format_log_record_json()`) and a zero-overhead profiler hook via `darshana::set_profiler_hook()`.
- Every rendered frame populates `darshana::FrameStats` (`frame_time_us`, `gpu_submit_us`, `upload_bytes`, `points_rendered`, `gpu_pool_used_bytes`, `scratch_used_bytes`, `reused_static_base_layer`).

---

## 5. Build Presets, Sanitizer Verification & Benchmark Reproduction

### Configure, Build, and Test All Presets
```bash
# Release with Vulkan 1.2+ and native CPU optimizations (-O3 -march=native)
cmake --preset release-vulkan
cmake --build --preset release-vulkan -j
ctest --preset release-vulkan --output-on-failure

# Portable Release (-O3 -march=x86-64-v3)
cmake --preset release
cmake --build --preset release -j
ctest --preset release --output-on-failure

# Headless CPU-Only Build (DARSHANA_ENABLE_VULKAN=OFF)
cmake --preset release-headless
cmake --build --preset release-headless -j
ctest --preset release-headless --output-on-failure

# AddressSanitizer + UndefinedBehaviorSanitizer
cmake --preset release-asan
cmake --build --preset release-asan -j
ctest --preset release-asan --output-on-failure

# ThreadSanitizer (8-thread concurrent rendering & 60 Hz streaming stress suite)
cmake --preset release-tsan
cmake --build --preset release-tsan -j
ctest --preset release-tsan --output-on-failure

# Debug Build
cmake --preset debug
cmake --build --preset debug -j
ctest --preset debug --output-on-failure
```

### Reproduce Hardware Benchmarks
```bash
./build-release-vulkan/bench_darshana
```

### Downstream CMake Package Consumption
```bash
cmake --install build-release-vulkan --prefix /usr/local
```
In downstream `CMakeLists.txt`:
```cmake
find_package(Darshana 1.0.0 REQUIRED)
target_link_libraries(my_app PRIVATE Darshana::Darshana)
```
