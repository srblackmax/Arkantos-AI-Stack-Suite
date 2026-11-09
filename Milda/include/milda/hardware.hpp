/*
 * Copyright (C) 2026 Kival
 * This file is part of the Arkantos AI Stack Suite.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef MILDA_HARDWARE_HPP
#define MILDA_HARDWARE_HPP

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

struct ggml_backend_buffer;

namespace milda
{

/* Discovered Linux CPU topology separating physical Performance Cores
   (P-cores) from Efficiency Cores (E-cores) and SMT siblings.  */
struct cpu_topology_info
{
  std::vector<int32_t> p_core_cpus;
  std::vector<int32_t> e_core_cpus;
  std::vector<int32_t> all_physical_cpus;
  int32_t logical_cpu_count = 1;
  bool has_avx2 = false;
  bool has_avx_vnni = false;
  bool has_fma = false;
};

/* Real-time hardware & inference telemetry snapshot.  */
struct telemetry_snapshot
{
  double prompt_eval_tok_per_s = 0.0;
  double token_gen_tok_per_s = 0.0;
  double ttft_ms = 0.0;
  std::size_t prompt_tokens = 0U;
  std::size_t generated_tokens = 0U;
  int32_t kv_cached_tokens = 0;
  int32_t kv_pinned_tokens = 0;
  int32_t n_ctx = 0;
  int32_t active_threads_decode = 0;
  int32_t active_threads_prefill = 0;
  int32_t p_cores_pinned = 0;
  int32_t e_cores_pinned = 0;
  int32_t offloaded_gpu_layers = 0;
  int32_t total_model_layers = 0;
  int32_t hybrid_cpu_ffn_layers = 0;
  std::size_t vram_allocated_bytes = 0U;
  std::size_t vram_total_bytes = 0U;
  std::size_t rss_bytes = 0U;
  std::size_t pinned_host_bytes = 0U;
};

/* RAII zero-copy model weight mapping using POSIX mmap (MAP_SHARED) with
   madvise (MADV_WILLNEED | MADV_HUGEPAGE) and optional mlock.  */
class mapped_weight_file
{
public:
  mapped_weight_file () noexcept = default;
  mapped_weight_file (const mapped_weight_file &) = delete;
  mapped_weight_file &
  operator= (const mapped_weight_file &) = delete;

  mapped_weight_file (mapped_weight_file &&other) noexcept;
  mapped_weight_file &
  operator= (mapped_weight_file &&other) noexcept;

  ~mapped_weight_file () noexcept;

  [[nodiscard]] bool
  map_file (const char *path, bool lock_in_ram) noexcept;

  void
  unmap_file () noexcept;

  [[nodiscard]] const void *
  data () const noexcept;

  [[nodiscard]] std::size_t
  size () const noexcept;

  [[nodiscard]] bool
  is_locked () const noexcept;

private:
  void *mapped_ptr_ = nullptr;
  std::size_t mapped_size_ = 0U;
  std::size_t locked_size_ = 0U;
  int fd_ = -1;
  bool is_locked_ = false;
};

/* RAII page-locked (pinned) host memory buffer for zero-staging PCIe
   transfers between CPU and GPU during split-layer forward passes.  */
class pinned_host_buffer
{
public:
  pinned_host_buffer () noexcept = default;
  pinned_host_buffer (const pinned_host_buffer &) = delete;
  pinned_host_buffer &
  operator= (const pinned_host_buffer &) = delete;

  pinned_host_buffer (pinned_host_buffer &&other) noexcept;
  pinned_host_buffer &
  operator= (pinned_host_buffer &&other) noexcept;

  ~pinned_host_buffer () noexcept;

  [[nodiscard]] bool
  allocate (std::size_t bytes) noexcept;

  void
  release () noexcept;

  [[nodiscard]] void *
  data () const noexcept;

  [[nodiscard]] std::size_t
  size () const noexcept;

  [[nodiscard]] bool
  is_pinned () const noexcept;

private:
  ggml_backend_buffer *backend_buf_ = nullptr;
  void *raw_ptr_ = nullptr;
  std::size_t size_ = 0U;
  bool is_pinned_ = false;
  bool mmap_fallback_ = false;
};

/* Detects Linux hybrid CPU topology and SIMD capabilities.  */
[[nodiscard]] cpu_topology_info
detect_cpu_topology () noexcept;

/* Pins the calling thread strictly to physical P-cores via
   pthread_setaffinity_np to eliminate E-core and SMT contention.  */
[[nodiscard]] bool
pin_calling_thread_to_pcores (const cpu_topology_info &topo,
                              int32_t max_threads) noexcept;

/* Reads Linux Resident Set Size (RSS) in bytes directly from
   /proc/self/statm with zero heap allocations.  */
[[nodiscard]] std::size_t
read_linux_rss_bytes () noexcept;

/* Queries available and total system RAM in bytes from /proc/meminfo.  */
[[nodiscard]] std::size_t
query_available_ram_bytes () noexcept;

/* Queries GPU VRAM allocated and total bytes from GGML backend device.  */
[[nodiscard]] bool
probe_vram_metrics (std::size_t *allocated_bytes,
                    std::size_t *total_bytes) noexcept;

[[nodiscard]] bool
query_vram_usage (std::size_t *allocated_bytes,
                  std::size_t *total_bytes) noexcept;

/* Monotonic nanosecond clock for high-precision TTFT and TG telemetry.  */
[[nodiscard]] uint64_t
monotonic_time_ns () noexcept;

/* Formats and prints real-time hardware and inference telemetry in
   Esperanto directly to unbuffered stdout.  */
void
print_telemetry (const telemetry_snapshot &snap) noexcept;

} /* namespace milda */

#endif /* MILDA_HARDWARE_HPP */
