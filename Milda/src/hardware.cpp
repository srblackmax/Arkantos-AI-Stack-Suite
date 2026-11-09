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

#include "milda/hardware.hpp"

#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>

#if defined (__linux__)
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

namespace milda
{

namespace
{

constexpr int32_t k_max_scan_cpus = 256;
constexpr std::size_t k_safety_ram_reserve_bytes = 2560ULL << 20U;

struct discovered_core
{
  int32_t cpu_id = -1;
  uint64_t max_freq_khz = 0ULL;
  bool has_smt_sibling = false;
};

#if defined (__linux__)
/* Reads a small sysfs/procfs file into buf without heap allocation.  */
[[nodiscard]] ssize_t
read_small_sys_file (const char *path, char *buf, std::size_t cap) noexcept
{
  if (cap == 0U)
    {
      return -1;
    }
  const int fd = open (path, O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    {
      return -1;
    }
  const ssize_t n_read = read (fd, buf, cap - 1U);
  (void)close (fd);
  if (n_read < 0)
    {
      return -1;
    }
  buf[static_cast<std::size_t> (n_read)] = '\0';
  return n_read;
}

/* Parses an unsigned 64-bit integer from the beginning of a C string.  */
[[nodiscard]] bool
parse_u64_prefix (const char *str, uint64_t *out_val) noexcept
{
  if (str == nullptr || out_val == nullptr)
    {
      return false;
    }
  while (*str == ' ' || *str == '\t' || *str == '\n')
    {
      ++str;
    }
  if (*str < '0' || *str > '9')
    {
      return false;
    }
  uint64_t val = 0ULL;
  while (*str >= '0' && *str <= '9')
    {
      val = (val * 10ULL) + static_cast<uint64_t> (*str - '0');
      ++str;
    }
  *out_val = val;
  return true;
}
#endif

} /* namespace */

mapped_weight_file::mapped_weight_file (mapped_weight_file &&other) noexcept
    : mapped_ptr_ (other.mapped_ptr_), mapped_size_ (other.mapped_size_),
      locked_size_ (other.locked_size_), fd_ (other.fd_),
      is_locked_ (other.is_locked_)
{
  other.mapped_ptr_ = nullptr;
  other.mapped_size_ = 0U;
  other.locked_size_ = 0U;
  other.fd_ = -1;
  other.is_locked_ = false;
}

mapped_weight_file &
mapped_weight_file::operator= (mapped_weight_file &&other) noexcept
{
  if (this != &other)
    {
      unmap_file ();
      mapped_ptr_ = other.mapped_ptr_;
      mapped_size_ = other.mapped_size_;
      locked_size_ = other.locked_size_;
      fd_ = other.fd_;
      is_locked_ = other.is_locked_;

      other.mapped_ptr_ = nullptr;
      other.mapped_size_ = 0U;
      other.locked_size_ = 0U;
      other.fd_ = -1;
      other.is_locked_ = false;
    }
  return *this;
}

mapped_weight_file::~mapped_weight_file () noexcept
{
  unmap_file ();
}

bool
mapped_weight_file::map_file (const char *path, bool lock_in_ram) noexcept
{
  unmap_file ();
  if (path == nullptr || path[0] == '\0')
    {
      return false;
    }

#if defined (__linux__)
  const int fd = open (path, O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    {
      return false;
    }

  struct stat st;
  std::memset (&st, 0, sizeof (st));
  if (fstat (fd, &st) != 0 || st.st_size <= 0)
    {
      (void)close (fd);
      return false;
    }

  const auto file_bytes = static_cast<std::size_t> (st.st_size);
  const std::size_t avail_ram = query_available_ram_bytes ();
  const std::size_t safe_budget
      = (avail_ram > k_safety_ram_reserve_bytes)
            ? (avail_ram - k_safety_ram_reserve_bytes)
            : 0U;

  /* Only create a secondary whole-file pre-fault mapping when the entire
     model file fits comfortably within available RAM with at least 6 GiB of
     headroom.  For larger hybrid models (e.g., 42.5 GiB 80B MoE on 48 GiB
     RAM), llama.cpp owns the single mmap and unmapped GPU tensor gaps are
     evicted via POSIX_FADV_DONTNEED after model load so CPU MoE experts fit
     100% in RAM without swap or disk thrashing.  */
  if (file_bytes + (6ULL << 30U) > safe_budget)
    {
      (void)close (fd);
      return false;
    }

#if defined (POSIX_FADV_WILLNEED)
  (void)posix_fadvise (fd, 0, 0, POSIX_FADV_WILLNEED);
#endif
  void *addr = mmap (nullptr, file_bytes, PROT_READ, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED || addr == nullptr)
    {
      (void)close (fd);
      return false;
    }

  fd_ = fd;
  mapped_ptr_ = addr;
  mapped_size_ = file_bytes;

  /* Advise transparent huge pages and sequential prefetching on the shared
     page-cache mapping.  */
#if defined(MADV_HUGEPAGE)
  (void)madvise (mapped_ptr_, mapped_size_, MADV_WILLNEED | MADV_HUGEPAGE);
  (void)madvise (mapped_ptr_, mapped_size_, MADV_HUGEPAGE);
#endif

  const std::size_t warm_bytes = std::min (mapped_size_, safe_budget);
  if (warm_bytes > 0U)
    {
      (void)madvise (mapped_ptr_, warm_bytes, MADV_WILLNEED);
    }

  if (lock_in_ram && warm_bytes > 0U)
    {
      if (mlock (mapped_ptr_, warm_bytes) == 0)
        {
          locked_size_ = warm_bytes;
          is_locked_ = true;
        }
    }
  return true;
#else
  (void)lock_in_ram;
  return false;
#endif
}

void
mapped_weight_file::unmap_file () noexcept
{
#if defined (__linux__)
  if (mapped_ptr_ != nullptr && mapped_size_ > 0U)
    {
      if (is_locked_ && locked_size_ > 0U)
        {
          (void)munlock (mapped_ptr_, locked_size_);
        }
      (void)munmap (mapped_ptr_, mapped_size_);
    }
  if (fd_ >= 0)
    {
      (void)close (fd_);
    }
#endif
  mapped_ptr_ = nullptr;
  mapped_size_ = 0U;
  locked_size_ = 0U;
  fd_ = -1;
  is_locked_ = false;
}

const void *
mapped_weight_file::data () const noexcept
{
  return mapped_ptr_;
}

std::size_t
mapped_weight_file::size () const noexcept
{
  return mapped_size_;
}

bool
mapped_weight_file::is_locked () const noexcept
{
  return is_locked_;
}

pinned_host_buffer::pinned_host_buffer (pinned_host_buffer &&other) noexcept
    : backend_buf_ (other.backend_buf_), raw_ptr_ (other.raw_ptr_),
      size_ (other.size_), is_pinned_ (other.is_pinned_),
      mmap_fallback_ (other.mmap_fallback_)
{
  other.backend_buf_ = nullptr;
  other.raw_ptr_ = nullptr;
  other.size_ = 0U;
  other.is_pinned_ = false;
  other.mmap_fallback_ = false;
}

pinned_host_buffer &
pinned_host_buffer::operator= (pinned_host_buffer &&other) noexcept
{
  if (this != &other)
    {
      release ();
      backend_buf_ = other.backend_buf_;
      raw_ptr_ = other.raw_ptr_;
      size_ = other.size_;
      is_pinned_ = other.is_pinned_;
      mmap_fallback_ = other.mmap_fallback_;

      other.backend_buf_ = nullptr;
      other.raw_ptr_ = nullptr;
      other.size_ = 0U;
      other.is_pinned_ = false;
      other.mmap_fallback_ = false;
    }
  return *this;
}

pinned_host_buffer::~pinned_host_buffer () noexcept
{
  release ();
}

bool
pinned_host_buffer::allocate (std::size_t bytes) noexcept
{
  release ();
  if (bytes == 0U)
    {
      return false;
    }

#if defined (MILDA_ENABLE_CUDA)
  /* Prefer CUDA/GPU page-locked host buffer via
     ggml_backend_dev_host_buffer_type for zero-copy DMA across PCIe.  */
  ggml_backend_dev_t gpu_dev
      = ggml_backend_dev_by_type (GGML_BACKEND_DEVICE_TYPE_GPU);
  if (gpu_dev != nullptr)
    {
      ggml_backend_buffer_type_t host_buft
          = ggml_backend_dev_host_buffer_type (gpu_dev);
      if (host_buft != nullptr)
        {
          ggml_backend_buffer_t buf
              = ggml_backend_buft_alloc_buffer (host_buft, bytes);
          if (buf != nullptr)
            {
              backend_buf_ = buf;
              raw_ptr_ = ggml_backend_buffer_get_base (buf);
              size_ = ggml_backend_buffer_get_size (buf);
              is_pinned_ = true;
              mmap_fallback_ = false;
              return true;
            }
        }
    }
#endif

#if defined (__linux__)
  void *ptr = mmap (nullptr, bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr != MAP_FAILED && ptr != nullptr)
    {
      raw_ptr_ = ptr;
      size_ = bytes;
      mmap_fallback_ = true;
#if defined (MADV_HUGEPAGE)
      (void)madvise (raw_ptr_, size_, MADV_HUGEPAGE);
#endif
      is_pinned_ = (mlock (raw_ptr_, size_) == 0);
      return true;
    }
#endif

  return false;
}

void
pinned_host_buffer::release () noexcept
{
  if (backend_buf_ != nullptr)
    {
      ggml_backend_buffer_free (backend_buf_);
      backend_buf_ = nullptr;
    }
#if defined (__linux__)
  else if (mmap_fallback_ && raw_ptr_ != nullptr && size_ > 0U)
    {
      if (is_pinned_)
        {
          (void)munlock (raw_ptr_, size_);
        }
      (void)munmap (raw_ptr_, size_);
    }
#endif
  raw_ptr_ = nullptr;
  size_ = 0U;
  is_pinned_ = false;
  mmap_fallback_ = false;
}

void *
pinned_host_buffer::data () const noexcept
{
  return raw_ptr_;
}

std::size_t
pinned_host_buffer::size () const noexcept
{
  return size_;
}

bool
pinned_host_buffer::is_pinned () const noexcept
{
  return is_pinned_;
}

cpu_topology_info
detect_cpu_topology () noexcept
{
  cpu_topology_info info = cpu_topology_info ();
  info.has_avx2 = (ggml_cpu_has_avx2 () != 0);
  info.has_avx_vnni = (ggml_cpu_has_avx_vnni () != 0);
  info.has_fma = (ggml_cpu_has_fma () != 0);

#if defined (__linux__)
  const long n_conf = sysconf (_SC_NPROCESSORS_ONLN);
  info.logical_cpu_count = (n_conf > 0L) ? static_cast<int32_t> (n_conf) : 1;

  std::vector<discovered_core> cores;
  cores.reserve (64U);
  uint64_t global_max_freq = 0ULL;
  bool any_has_smt = false;

  std::array<char, 128U> path_buf;
  std::array<char, 64U> text_buf;

  for (int32_t cpu = 0; cpu < k_max_scan_cpus; ++cpu)
    {
      (void)std::snprintf (
          path_buf.data (), path_buf.size (),
          "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
      const ssize_t sib_len = read_small_sys_file (
          path_buf.data (), text_buf.data (), text_buf.size ());
      if (sib_len <= 0)
        {
          continue;
        }

      uint64_t first_sib = 0ULL;
      if (!parse_u64_prefix (text_buf.data (), &first_sib))
        {
          continue;
        }
      if (static_cast<int32_t> (first_sib) != cpu)
        {
          /* Skip secondary SMT sibling threads.  */
          continue;
        }

      bool has_smt = false;
      for (ssize_t idx = 0; idx < sib_len; ++idx)
        {
          const char ch = text_buf[static_cast<std::size_t> (idx)];
          if (ch == '-' || ch == ',')
            {
              has_smt = true;
              any_has_smt = true;
              break;
            }
        }

      uint64_t max_freq = 0ULL;
      (void)std::snprintf (
          path_buf.data (), path_buf.size (),
          "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
      if (read_small_sys_file (path_buf.data (), text_buf.data (),
                               text_buf.size ())
          > 0)
        {
          (void)parse_u64_prefix (text_buf.data (), &max_freq);
        }
      if (max_freq > global_max_freq)
        {
          global_max_freq = max_freq;
        }

      discovered_core dc = discovered_core ();
      dc.cpu_id = cpu;
      dc.max_freq_khz = max_freq;
      dc.has_smt_sibling = has_smt;
      cores.push_back (dc);
    }

  const uint64_t p_core_freq_threshold
      = (global_max_freq > 0ULL) ? ((global_max_freq * 9ULL) / 10ULL) : 0ULL;

  std::vector<int32_t> e_cores;
  info.p_core_cpus.reserve (cores.size ());
  info.e_core_cpus.reserve (cores.size ());
  info.all_physical_cpus.reserve (cores.size ());

  for (const discovered_core &dc : cores)
    {
      /* Primary thread of multi-sibling core = P-core; single-sibling core
         on hybrid CPU = E-core; flat VM without SMT/cpufreq = P-core.  */
      const bool is_pcore
          = any_has_smt
                ? dc.has_smt_sibling
                : ((global_max_freq > 0ULL)
                       ? (dc.max_freq_khz >= p_core_freq_threshold)
                       : true);
      if (is_pcore)
        {
          info.p_core_cpus.push_back (dc.cpu_id);
          info.all_physical_cpus.push_back (dc.cpu_id);
        }
      else
        {
          e_cores.push_back (dc.cpu_id);
          info.e_core_cpus.push_back (dc.cpu_id);
        }
    }

  for (const int32_t e_cpu : e_cores)
    {
      info.all_physical_cpus.push_back (e_cpu);
    }

  if (info.all_physical_cpus.empty ())
    {
      const int32_t n_vcpus = std::max (1, info.logical_cpu_count);
      info.p_core_cpus.reserve (static_cast<std::size_t> (n_vcpus));
      info.all_physical_cpus.reserve (static_cast<std::size_t> (n_vcpus));
      for (int32_t c = 0; c < n_vcpus; ++c)
        {
          info.p_core_cpus.push_back (c);
          info.all_physical_cpus.push_back (c);
        }
    }
  else if (info.p_core_cpus.empty ())
    {
      info.p_core_cpus = info.all_physical_cpus;
    }
#endif

  if (info.p_core_cpus.empty ())
    {
      info.p_core_cpus.push_back (0);
      info.all_physical_cpus.push_back (0);
    }

  return info;
}

bool
pin_calling_thread_to_pcores (const cpu_topology_info &topo,
                              int32_t max_threads) noexcept
{
#if defined (__linux__)
  const std::vector<int32_t> &phys_list
      = !topo.all_physical_cpus.empty () ? topo.all_physical_cpus
                                         : topo.p_core_cpus;
  if (phys_list.empty ())
    {
      return false;
    }

  (void)setpriority (PRIO_PROCESS, 0, -20);

  cpu_set_t cpuset;
  CPU_ZERO (&cpuset);

  const std::size_t limit
      = (max_threads > 0)
            ? std::min (phys_list.size (),
                        static_cast<std::size_t> (max_threads))
            : phys_list.size ();

  std::size_t count = 0U;
  for (std::size_t i = 0U; i < limit; ++i)
    {
      const int32_t cpu_id = phys_list[i];
      if (cpu_id >= 0 && cpu_id < CPU_SETSIZE)
        {
          CPU_SET (cpu_id, &cpuset);
          ++count;
        }
    }

  if (max_threads > static_cast<int32_t> (phys_list.size ()))
    {
      const int32_t max_logical
          = std::min (topo.logical_cpu_count, CPU_SETSIZE);
      for (int32_t cpu_id = 0;
           cpu_id < max_logical
           && count < static_cast<std::size_t> (max_threads);
           ++cpu_id)
        {
          if (!CPU_ISSET (cpu_id, &cpuset))
            {
              CPU_SET (cpu_id, &cpuset);
              ++count;
            }
        }
    }

  if (count == 0U)
    {
      return false;
    }

  return pthread_setaffinity_np (pthread_self (), sizeof (cpu_set_t), &cpuset)
         == 0;
#else
  (void)topo;
  (void)max_threads;
  return false;
#endif
}

std::size_t
read_linux_rss_bytes () noexcept
{
#if defined (__linux__)
  std::array<char, 128U> buf;
  const ssize_t n_read
      = read_small_sys_file ("/proc/self/statm", buf.data (), buf.size ());
  if (n_read <= 0)
    {
      return 0U;
    }

  /* /proc/self/statm format: "<size> <resident> <shared> ..."  */
  const char *cursor = buf.data ();
  while (*cursor == ' ')
    {
      ++cursor;
    }
  while (*cursor >= '0' && *cursor <= '9')
    {
      ++cursor;
    }
  uint64_t resident_pages = 0ULL;
  if (!parse_u64_prefix (cursor, &resident_pages))
    {
      return 0U;
    }

  const long page_sz = sysconf (_SC_PAGESIZE);
  const std::size_t page_bytes
      = (page_sz > 0L) ? static_cast<std::size_t> (page_sz) : 4096U;
  return static_cast<std::size_t> (resident_pages) * page_bytes;
#else
  return 0U;
#endif
}

std::size_t
query_available_ram_bytes () noexcept
{
#if defined (__linux__)
  std::array<char, 1024U> buf;
  const ssize_t n_read
      = read_small_sys_file ("/proc/meminfo", buf.data (), buf.size ());
  if (n_read <= 0)
    {
      return 0U;
    }

  const char *found = std::strstr (buf.data (), "MemAvailable:");
  if (found == nullptr)
    {
      return 0U;
    }
  found += std::strlen ("MemAvailable:");
  uint64_t avail_kb = 0ULL;
  if (!parse_u64_prefix (found, &avail_kb))
    {
      return 0U;
    }
  return static_cast<std::size_t> (avail_kb) * 1024ULL;
#else
  return 0U;
#endif
}

bool
probe_vram_metrics (std::size_t *allocated_bytes,
                    std::size_t *total_bytes) noexcept
{
  if (allocated_bytes != nullptr)
    {
      *allocated_bytes = 0U;
    }
  if (total_bytes != nullptr)
    {
      *total_bytes = 0U;
    }

#if defined (MILDA_ENABLE_CUDA)
  ggml_backend_dev_t gpu_dev
      = ggml_backend_dev_by_type (GGML_BACKEND_DEVICE_TYPE_GPU);
  if (gpu_dev == nullptr)
    {
      return false;
    }

  std::size_t free_b = 0U;
  std::size_t total_b = 0U;
  ggml_backend_dev_memory (gpu_dev, &free_b, &total_b);
  if (total_b == 0U)
    {
      return false;
    }

  if (allocated_bytes != nullptr)
    {
      *allocated_bytes = (total_b >= free_b) ? (total_b - free_b) : 0U;
    }
  if (total_bytes != nullptr)
    {
      *total_bytes = total_b;
    }
  return true;
#else
  return false;
#endif
}

bool
query_vram_usage (std::size_t *allocated_bytes,
                  std::size_t *total_bytes) noexcept
{
  return probe_vram_metrics (allocated_bytes, total_bytes);
}

uint64_t
monotonic_time_ns () noexcept
{
#if defined (__linux__)
  struct timespec ts;
  ts.tv_sec = 0;
  ts.tv_nsec = 0;
  (void)clock_gettime (CLOCK_MONOTONIC, &ts);
  return (static_cast<uint64_t> (ts.tv_sec) * 1000000000ULL)
         + static_cast<uint64_t> (ts.tv_nsec);
#else
  return 0ULL;
#endif
}

void
print_telemetry (const telemetry_snapshot &snap) noexcept
{
  const double rss_mib
      = static_cast<double> (snap.rss_bytes) / (1024.0 * 1024.0);
  const double vram_used_mib
      = static_cast<double> (snap.vram_allocated_bytes) / (1024.0 * 1024.0);
  const double vram_total_mib
      = static_cast<double> (snap.vram_total_bytes) / (1024.0 * 1024.0);
  const double pinned_mib
      = static_cast<double> (snap.pinned_host_bytes) / (1024.0 * 1024.0);

  std::array<char, 96U> core_desc;
  if (snap.e_cores_pinned > 0)
    {
      (void)std::snprintf (core_desc.data (), core_desc.size (),
                           "%d P-kernoj + %d E-kernoj", snap.p_cores_pinned,
                           snap.e_cores_pinned);
    }
  else
    {
      (void)std::snprintf (core_desc.data (), core_desc.size (),
                           "%d P-kernoj", snap.p_cores_pinned);
    }

  std::array<char, 640U> out_buf;
  int n_written = 0;
  if (snap.offloaded_gpu_layers <= 0)
    {
      n_written = std::snprintf (
          out_buf.data (), out_buf.size (),
          "\n[telemetrio] Prompt-rapideco (PP): %.2f tok/s (%zu tok, "
          "TTFT: %.2f ms)"
          "\n[telemetrio] Generado-rapideco (TG): %.2f tok/s (%zu tok)"
          "\n[telemetrio] Kunteksto (KV): %d / %d tok (fiksitaj: %d tok)"
          "\n[telemetrio] CPU-fadenoj: %d (TG, %s) / %d (PP)"
          "\n[telemetrio] Tavoloj: CPU-nur (0 GPU-tavoloj, %d totalaj)"
          "\n[telemetrio] Memoro: VRAM %.1f / %.1f MiB | RSS %.1f MiB "
          "(statm) | Pinned %.1f MiB\n",
          snap.prompt_eval_tok_per_s, snap.prompt_tokens, snap.ttft_ms,
          snap.token_gen_tok_per_s, snap.generated_tokens,
          snap.kv_cached_tokens, snap.n_ctx, snap.kv_pinned_tokens,
          snap.active_threads_decode, core_desc.data (),
          snap.active_threads_prefill, snap.total_model_layers, vram_used_mib,
          vram_total_mib, rss_mib, pinned_mib);
    }
  else
    {
      n_written = std::snprintf (
          out_buf.data (), out_buf.size (),
          "\n[telemetrio] Prompt-rapideco (PP): %.2f tok/s (%zu tok, "
          "TTFT: %.2f ms)"
          "\n[telemetrio] Generado-rapideco (TG): %.2f tok/s (%zu tok)"
          "\n[telemetrio] Kunteksto (KV): %d / %d tok (fiksitaj: %d tok)"
          "\n[telemetrio] CPU-fadenoj: %d (TG, %s) / %d (PP)"
          "\n[telemetrio] Tavoloj: %d / %d GPU (hibridaj CPU FFN/MoE: %d)"
          "\n[telemetrio] Memoro: VRAM %.1f / %.1f MiB | RSS %.1f MiB "
          "(statm) | Pinned %.1f MiB\n",
          snap.prompt_eval_tok_per_s, snap.prompt_tokens, snap.ttft_ms,
          snap.token_gen_tok_per_s, snap.generated_tokens,
          snap.kv_cached_tokens, snap.n_ctx, snap.kv_pinned_tokens,
          snap.active_threads_decode, core_desc.data (),
          snap.active_threads_prefill, snap.offloaded_gpu_layers,
          snap.total_model_layers, snap.hybrid_cpu_ffn_layers, vram_used_mib,
          vram_total_mib, rss_mib, pinned_mib);
    }

  if (n_written > 0)
    {
      const auto valid_len = std::min (out_buf.size () - 1U,
                                       static_cast<std::size_t> (n_written));
      (void)std::fwrite (out_buf.data (), 1U, valid_len, stdout);
      (void)std::fflush (stdout);
    }
}

} /* namespace milda */
