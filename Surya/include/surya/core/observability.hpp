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

#ifndef SURYA_CORE_OBSERVABILITY_HPP
#define SURYA_CORE_OBSERVABILITY_HPP

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <surya/core/config.hpp>

namespace surya::inline v1::core
{

/**
 * @brief Severity levels for Surya structured JSON-lines logging.
 * @note Thread-safety: Value type.
 */
enum class Log_Level : uint8_t
{
  Debug = 0U,
  Info = 1U,
  Warn = 2U,
  Error = 3U,
  Off = 4U
};

/**
 * @brief Pluggable sink interface for Surya structured logging.
 * @note Thread-safety: Implementations must be safe for concurrent calls.
 */
class SURYA_API SuryaLogSink
{
public:
  virtual ~SuryaLogSink (void) noexcept = default;

  /**
   * @brief Emit a structured log record.
   * @pre SUBSYSTEM and MESSAGE are valid string views.
   * @post Record is written to the backing sink without throwing.
   * @note Thread-safety: Must be safe to call concurrently from any thread.
   */
  virtual void
  write (Log_Level level,
         std::string_view subsystem,
         std::string_view message,
         uint64_t timestamp_ns) noexcept = 0;
};

/**
 * @brief Process-wide structured logger with pluggable sink and level filter.
 * @note Thread-safety: Lock-free atomic level check; sink dispatch is
 *       thread-safe.
 */
class SURYA_API SuryaLogger
{
public:
  /**
   * @brief Install a custom log sink (or nullptr to restore default stderr
   *        JSON-lines sink).
   * @pre None.
   * @post Subsequent log records route to SINK.
   * @note Thread-safety: Safe to call concurrently.
   */
  static void
  set_sink (std::shared_ptr<SuryaLogSink> sink) noexcept;

  /**
   * @brief Set the minimum severity level that will be emitted.
   * @pre None.
   * @post Records below LEVEL are discarded with relaxed atomic load.
   * @note Thread-safety: Lock-free atomic store.
   */
  static void
  set_level (Log_Level level) noexcept;

  /**
   * @brief Query the current minimum log level.
   * @pre None.
   * @post Returns current Log_Level.
   * @returns Active minimum Log_Level.
   * @note Thread-safety: Lock-free atomic load.
   */
  static Log_Level
  level (void) noexcept;

  /**
   * @brief Emit a structured log event if LEVEL >= active threshold.
   * @pre None.
   * @post Emits JSON-lines record to active sink.
   * @note Thread-safety: Safe to call concurrently from any thread.
   */
  static void
  log (Log_Level level,
       std::string_view subsystem,
       std::string_view message) noexcept;
};

/**
 * @brief Point-in-time value snapshot of SuryaStats counters.
 * @note Thread-safety: Plain value struct; safe to copy across threads.
 */
struct SuryaStatsSnapshot
{
  uint64_t preprocess_ns = 0ULL;
  uint64_t h2d_ns = 0ULL;
  uint64_t inference_ns = 0ULL;
  uint64_t d2h_ns = 0ULL;
  uint64_t postprocess_ns = 0ULL;
  uint64_t total_requests = 0ULL;
  uint64_t total_images = 0ULL;
  uint64_t vram_current_bytes = 0ULL;
  uint64_t vram_peak_bytes = 0ULL;
  uint64_t cpu_fallback_count = 0ULL;
  uint64_t cuda_graph_captures = 0ULL;
  uint64_t cuda_graph_launches = 0ULL;
};

/**
 * @brief Lock-free per-stage telemetry counters using relaxed atomics.
 *        Zero-cost when compiled with SURYA_DISABLE_STATS.
 * @note Thread-safety: All mutations use std::memory_order_relaxed atomics.
 */
class SURYA_API SuryaStats
{
public:
  SuryaStats (void) noexcept = default;
  ~SuryaStats (void) noexcept = default;

  SuryaStats (const SuryaStats &) = delete;
  SuryaStats &operator= (const SuryaStats &) = delete;

  void
  add_preprocess_ns (uint64_t ns) noexcept
  {
#if !defined(SURYA_DISABLE_STATS)
    preprocess_ns_.fetch_add (ns, std::memory_order_relaxed);
#else
    (void) ns;
#endif
  }

  void
  add_h2d_ns (uint64_t ns) noexcept
  {
#if !defined(SURYA_DISABLE_STATS)
    h2d_ns_.fetch_add (ns, std::memory_order_relaxed);
#else
    (void) ns;
#endif
  }

  void
  add_inference_ns (uint64_t ns) noexcept
  {
#if !defined(SURYA_DISABLE_STATS)
    inference_ns_.fetch_add (ns, std::memory_order_relaxed);
#else
    (void) ns;
#endif
  }

  void
  add_d2h_ns (uint64_t ns) noexcept
  {
#if !defined(SURYA_DISABLE_STATS)
    d2h_ns_.fetch_add (ns, std::memory_order_relaxed);
#else
    (void) ns;
#endif
  }

  void
  add_postprocess_ns (uint64_t ns) noexcept
  {
#if !defined(SURYA_DISABLE_STATS)
    postprocess_ns_.fetch_add (ns, std::memory_order_relaxed);
#else
    (void) ns;
#endif
  }

  void
  record_batch (uint64_t images) noexcept
  {
#if !defined(SURYA_DISABLE_STATS)
    total_requests_.fetch_add (1ULL, std::memory_order_relaxed);
    total_images_.fetch_add (images, std::memory_order_relaxed);
#else
    (void) images;
#endif
  }

  void
  update_vram_bytes (uint64_t current_bytes) noexcept
  {
#if !defined(SURYA_DISABLE_STATS)
    vram_current_bytes_.store (current_bytes, std::memory_order_relaxed);
    uint64_t prev_peak = vram_peak_bytes_.load (std::memory_order_relaxed);
    while (current_bytes > prev_peak
           && !vram_peak_bytes_.compare_exchange_weak (
             prev_peak, current_bytes, std::memory_order_relaxed))
      {
      }
#else
    (void) current_bytes;
#endif
  }

  void
  inc_cpu_fallback (void) noexcept
  {
#if !defined(SURYA_DISABLE_STATS)
    cpu_fallback_count_.fetch_add (1ULL, std::memory_order_relaxed);
#endif
  }

  void
  inc_graph_capture (void) noexcept
  {
#if !defined(SURYA_DISABLE_STATS)
    cuda_graph_captures_.fetch_add (1ULL, std::memory_order_relaxed);
#endif
  }

  void
  inc_graph_launch (void) noexcept
  {
#if !defined(SURYA_DISABLE_STATS)
    cuda_graph_launches_.fetch_add (1ULL, std::memory_order_relaxed);
#endif
  }

  /**
   * @brief Capture a consistent relaxed snapshot of all counters.
   * @pre None.
   * @post Returns SuryaStatsSnapshot.
   * @returns Copy of current counter values.
   * @note Thread-safety: Lock-free.
   */
  SuryaStatsSnapshot
  snapshot (void) const noexcept;

  /**
   * @brief Reset all timing and event counters to zero.
   * @pre None.
   * @post Counters (except current VRAM residency) are zeroed.
   * @note Thread-safety: Lock-free.
   */
  void
  reset (void) noexcept;

  /**
   * @brief Return the process-wide default telemetry registry.
   * @pre None.
   * @post Returns valid reference.
   * @returns Reference to global SuryaStats.
   * @note Thread-safety: Lock-free.
   */
  static SuryaStats &
  global_stats (void) noexcept;

private:
  alignas (SURYA_CACHELINE_BYTES) std::atomic<uint64_t> preprocess_ns_ {0ULL};
  std::atomic<uint64_t> h2d_ns_ {0ULL};
  std::atomic<uint64_t> inference_ns_ {0ULL};
  std::atomic<uint64_t> d2h_ns_ {0ULL};
  std::atomic<uint64_t> postprocess_ns_ {0ULL};
  std::atomic<uint64_t> total_requests_ {0ULL};
  std::atomic<uint64_t> total_images_ {0ULL};
  std::atomic<uint64_t> vram_current_bytes_ {0ULL};
  std::atomic<uint64_t> vram_peak_bytes_ {0ULL};
  std::atomic<uint64_t> cpu_fallback_count_ {0ULL};
  std::atomic<uint64_t> cuda_graph_captures_ {0ULL};
  std::atomic<uint64_t> cuda_graph_launches_ {0ULL};
};

/**
 * @brief Pluggable external profiler callback interface for Nsight NVTX,
 *        Tracy, or Perfetto integration without recompiling Surya.
 * @note Thread-safety: Hook implementations must be thread-safe.
 */
class SURYA_API SuryaProfilerHook
{
public:
  virtual ~SuryaProfilerHook (void) noexcept = default;

  virtual void
  on_range_push (const char *name) noexcept = 0;

  virtual void
  on_range_pop (void) noexcept = 0;

  virtual void
  on_mark (const char *name) noexcept = 0;
};

/**
 * @brief Global profiler dispatcher and RAII range guard.
 * @note Thread-safety: Atomic pointer load on hot path; zero cost when unset.
 */
class SURYA_API SuryaProfiler
{
public:
  static void
  set_hook (SuryaProfilerHook *hook) noexcept;

  static SuryaProfilerHook *
  hook (void) noexcept;

  static void
  range_push (const char *name) noexcept
  {
    SuryaProfilerHook *h = hook ();
    if (h != nullptr)
      {
        h->on_range_push (name);
      }
  }

  static void
  range_pop (void) noexcept
  {
    SuryaProfilerHook *h = hook ();
    if (h != nullptr)
      {
        h->on_range_pop ();
      }
  }

  static void
  mark (const char *name) noexcept
  {
    SuryaProfilerHook *h = hook ();
    if (h != nullptr)
      {
        h->on_mark (name);
      }
  }
};

/**
 * @brief RAII scope guard for SuryaProfiler ranges.
 */
class ScopedProfileRange
{
public:
  explicit ScopedProfileRange (const char *name) noexcept
    : active_ (SuryaProfiler::hook () != nullptr)
  {
    if (active_)
      {
        SuryaProfiler::range_push (name);
      }
  }

  ~ScopedProfileRange (void) noexcept
  {
    if (active_)
      {
        SuryaProfiler::range_pop ();
      }
  }

  ScopedProfileRange (const ScopedProfileRange &) = delete;
  ScopedProfileRange &operator= (const ScopedProfileRange &) = delete;

private:
  bool active_ = false;
};

} /* namespace surya::inline v1::core */

namespace surya::inline v1
{
using Log_Level = core::Log_Level;
using SuryaLogSink = core::SuryaLogSink;
using SuryaLogger = core::SuryaLogger;
using SuryaStatsSnapshot = core::SuryaStatsSnapshot;
using SuryaStats = core::SuryaStats;
using SuryaProfilerHook = core::SuryaProfilerHook;
using SuryaProfiler = core::SuryaProfiler;
using ScopedProfileRange = core::ScopedProfileRange;
} /* namespace surya::inline v1 */

#endif /* SURYA_CORE_OBSERVABILITY_HPP */
