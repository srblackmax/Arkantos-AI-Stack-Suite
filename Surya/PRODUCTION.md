# Surya Production Operations & Deployment Guide (`PRODUCTION.md`)

**Version**: `1.0.0` (`namespace surya::inline v1`)  
**License**: `GPL-3.0-or-later` (Non-GNU Project)  
**Target Architecture**: Hybrid x86-64 CPU (`AVX-512` / `AVX2` + Intel oneTBB) + NVIDIA Ampere/Ada/Hopper GPU (`CUDA 12.x`, `sm_86+`)

---

## 1. Hardware Requirements & Reference Platform

Surya is engineered and verified on the following reference production workstation/server node:

| Component | Reference Specification | Minimum Production Requirement |
| :--- | :--- | :--- |
| **CPU** | 12th Gen Intel Core i7-12700K (8 P-cores / 16 threads + 4 E-cores / 4 threads, 20 logical CPUs) | x86-64-v3 (`AVX2` + `FMA`); runtime-dispatched `AVX-512F` when supported |
| **GPU** | NVIDIA GeForce RTX 3070 (`GA104`, `sm_86`, 8 GB GDDR6 @ 448 GB/s) | NVIDIA GPU with Compute Capability $\ge 8.0$ (`sm_80+`), $\ge 8\text{ GB}$ VRAM |
| **System RAM** | $\ge 32\text{ GB}$ DDR4/DDR5 Dual-Channel | $\ge 16\text{ GB}$ (for `cudaHostAllocMapped` page-locked staging pools) |
| **Toolchain** | GCC 14.2.0 (host C++23), CUDA 12.4 (`nvcc` + GCC 13.3 host compiler), CMake 3.31.6, Intel oneTBB 2022.1 | CMake $\ge 3.28$, C++23 compiler, CUDA $\ge 12.0$, oneTBB $\ge 2021.5$ |

---

## 2. Recommended BIOS, OS, and NVIDIA Driver Settings

1. **PCIe Resizable BAR (`ReBAR`)**: Enable **Above 4G Decoding** and **Resizable BAR** in BIOS to reduce host-to-device (`H2D`) staging latency over PCIe 4.0 x16.
2. **NVIDIA Persistence Mode & Clocks**:
   ```bash
   sudo nvidia-smi -pm 1
   sudo nvidia-smi -pl 220
   ```
   Persistence mode keeps the CUDA driver initialized across process lifetimes and eliminates cold-start context creation latency.
3. **Locked Page Memory Limits (`ulimit -l`)**:
   Surya allocates page-locked host staging buffers via `cudaHostAlloc(..., cudaHostAllocMapped)`. Ensure `/etc/security/limits.conf` permits at least `4 GB` of locked memory (`memlock unlimited`).
4. **CPU Scaling Governor**:
   ```bash
   sudo cpupower frequency-set -g performance
   ```

---

## 3. Sizing Batch Count, Execution Slots, and Streams within the 6.5 GiB VRAM Budget

Surya enforces a hard default VRAM ceiling of **6.5 GiB** (`SURYA_DEFAULT_VRAM_BUDGET_BYTES = 6,656 MiB`) on an 8 GB RTX 3070, reserving $\sim 1.5\text{ GiB}$ of headroom for the display server, kernel driver, and primary CUDA context.

### VRAM Accounting Formula

$$\text{VRAM}_{\text{total}} = \text{VRAM}_{\text{weights}} + N_{\text{slots}} \times \text{VRAM}_{\text{slot}} (B_{\max}) + K_{\text{inflight}} \times B \times \text{VRAM}_{\text{recognition\_item}}$$

- **Resident Model Weights ($\text{VRAM}_{\text{weights}}$)**: Loaded **once** per `SuryaPipeline` instance at `INT8` (or `INT4` when `quantization_bits = 4`). Default: `256 MiB` (`INT8`) or `128 MiB` (`INT4`).
- **Pre-Allocated Multi-Stream Execution Slots ($N_{\text{slots}} = 8$)**: Each slot holds 3 non-blocking CUDA streams (`h2d_stream`, high-priority `compute_stream`, `d2h_stream`) and stream-ordered device buffers allocated via `cudaMallocAsync`.
- **Per-Batch Recognition Workspace ($\text{VRAM}_{\text{recognition\_item}} \approx 50\text{ MiB}$)**: Reserved atomically via `VramBudgetAllocator::try_reserve` on entry to `SuryaPipeline::infer_batch` and released via RAII (`VramReservationGuard`) on scope exit.

### Recommended `SuryaPipelineConfig` Profiles (8 GB RTX 3070)

| Workload Profile | `max_batch_size` | `quantization_bits` | `enable_cuda_graphs` | Steady-State VRAM | Peak VRAM (Full Load) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Low-Latency Interactive OCR** | `4` | `8` (`INT8`) | `true` | `512 MiB` | `912 MiB` |
| **Balanced Production Server (Default)** | `16` | `8` (`INT8`) | `true` | `1,150 MiB` | `2,750 MiB` |
| **High-Throughput Batch Ingestion** | `32` | `8` (`INT8`) | `true` | `2,091 MiB` | `3,691 MiB` |

---

## 4. Thread-Affinity Guidance for Intel Core i7-12700K (Alder Lake Hybrid)

The Intel i7-12700K combines **8 Golden Cove P-cores** (logical CPUs `0..15` with Hyper-Threading) and **4 Gracemont E-cores** (logical CPUs `16..19`):

1. **Pin Surya Compute & TBB Workers to P-Cores (`0..15`)**:
   ```bash
   taskset -c 0-15 ./your_surya_service
   ```
   Preventing TBB image-preprocessing tasks from migrating onto slower E-cores eliminates tail-latency (`p99`) spikes during barrier synchronization before H2D DMA submission.
2. **Isolate Background I/O and Log Consumption on E-Cores (`16..19`)**:
   If your application runs dedicated disk/network ingestion threads or custom `SuryaLogSink` writers, bind those threads to CPUs `16..19` via `pthread_setaffinity_np`.

---

## 5. Observability: `SuryaStats`, `SuryaLogger`, and `SuryaProfiler`

### 5.1 Interpreting `SuryaStats` (`SuryaStatsSnapshot`)
Query `surya::SuryaStats::global_stats().snapshot()` at any time (lock-free `std::memory_order_relaxed` atomic reads):

- **`preprocess_ns` vs `inference_ns`**:
  - If `preprocess_ns > inference_ns`, the pipeline is CPU-bound on host image packing or decoding; increase batch size or use `run_overlapped_stream_pipeline`.
  - If `inference_ns >> preprocess_ns` at `batch=1` and `cuda_graph_launches == 0`, verify `enable_cuda_graphs = true` is set in `SuryaPipelineConfig`.
- **`vram_current_bytes` and `vram_peak_bytes`**:
  - Tracks live and high-water-mark device allocations across `VramBudgetAllocator`. In steady state after warmup, `vram_current_bytes` must remain flat (`delta == 0`).
- **`cpu_fallback_count`**:
  - Non-zero values indicate that a burst of concurrent batches exceeded the `6.5 GiB` VRAM cap and triggered automatic CPU fallback. Reduce `max_batch_size` or limit upstream concurrency.
- **`cuda_graph_captures` and `cuda_graph_launches`**:
  - `cuda_graph_captures` increments once per `(slot, batch_size)` pair during warmup; `cuda_graph_launches` increments on every subsequent graph replay.

### 5.2 Structured Logging (`SuryaLogger`)
Surya never writes to `std::cout` or `printf`. All internal events emit structured JSON-lines to `stderr` by default:
```json
{"ts_ns":44099280810954,"level":"WARN","subsystem":"gpu","msg":"Batch VRAM workspace exceeded 6.5 GB cap; graceful CPU fallback"}
```
Register a custom sink via `surya::SuryaLogger::set_sink(std::make_shared<MySink>())` or silence logs below a threshold via `surya::SuryaLogger::set_level(surya::Log_Level::Error)`.

### 5.3 External Profiler Integration (`SuryaProfiler`)
Attach Nsight Systems (`NVTX`), Tracy, or Perfetto without recompiling Surya by implementing `surya::SuryaProfilerHook` and calling `surya::SuryaProfiler::set_hook(&my_hook)`.

---

## 6. Known Limitations & Failure Modes

1. **VRAM Exhaustion (`Error_Code::Vram_Exhausted`)**:
   - When `fallback_to_cpu_on_oom = true` (default), exceeding the 6.5 GiB budget logs a `WARN` event, increments `cpu_fallback_count`, and transparently executes the batch on the CPU using oneTBB + SIMD.
   - When `fallback_to_cpu_on_oom = false`, `SuryaPipeline::create` or `SuryaPipeline::infer_batch` fails fast and returns `std::unexpected(SuryaError(Error_Code::Vram_Exhausted, ...))`.
2. **CUDA Graph Static Topology per Batch Size**:
   - `SuryaPipeline` caches one instantiated `cudaGraphExec_t` per batch size (`1 .. max_batch_size`) per execution slot (`kMaxBatchCap = 32`). Batch sizes $> 32$ should be chunked into sub-batches of $\le 32$ or processed via `run_overlapped_stream_pipeline`.
3. **AddressSanitizer + CUDA Runtime Shadow Memory**:
   - When running binaries compiled with `-fsanitize=address` alongside the NVIDIA CUDA driver, set `ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0` (automatically configured by the `release-asan` CMake preset).
