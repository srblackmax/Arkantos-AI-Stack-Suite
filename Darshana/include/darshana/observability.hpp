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

#ifndef DARSHANA_OBSERVABILITY_HPP
#define DARSHANA_OBSERVABILITY_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <string_view>

#ifndef DARSHANA_ENABLE_STATS
#define DARSHANA_ENABLE_STATS 1
#endif

namespace darshana::inline v1
{

/**
 * @brief Severity levels for Darshana structured JSON-lines logging.
 * @pre None.
 * @post Immutable enum type.
 * @note Thread-safety: Safe for concurrent reads.
 */
enum class LogLevel : uint8_t
{
  Trace = 0,
  Debug = 1,
  Info = 2,
  Warn = 3,
  Error = 4
};

/**
 * @brief Pluggable interface for receiving structured log records.
 * @pre Implementations of write() must not throw exceptions.
 * @post Log entries are delivered in real time.
 * @note Thread-safety: Implementations must be safe for concurrent write().
 */
class LogSink
{
public:
  LogSink () noexcept = default;
  LogSink (const LogSink &) noexcept = default;
  auto operator= (const LogSink &) noexcept -> LogSink & = default;
  LogSink (LogSink &&) noexcept = default;
  auto operator= (LogSink &&) noexcept -> LogSink & = default;
  virtual ~LogSink () noexcept = default;

  /**
   * @brief Writes a structured log event to the underlying destination.
   * @param level Severity level of the event.
   * @param subsystem Name of the emitting Darshana subsystem.
   * @param message Diagnostic message payload.
   * @param timestamp_ns Monotonic timestamp in nanoseconds.
   * @pre None.
   * @post Event is written or buffered by the sink.
   */
  virtual void write (LogLevel level, std::string_view subsystem,
                      std::string_view message, uint64_t timestamp_ns) noexcept
      = 0;
};

/**
 * @brief Global structured logger dispatching JSON-lines to a configurable
 * sink.
 * @pre None.
 * @post Emits JSON-lines to stderr by default.
 * @note Thread-safety: All static methods are thread-safe.
 */
class Logger
{
public:
  /**
   * @brief Sets the minimum severity threshold for emitted logs.
   * @param level Minimum LogLevel to emit.
   * @pre None.
   * @post Subsequent log() calls below level are filtered out.
   * @note Thread-safety: Lock-free atomic store.
   */
  static void set_level (LogLevel level) noexcept;

  /**
   * @brief Queries the current minimum log severity threshold.
   * @pre None.
   * @post State is unchanged.
   * @returns Current minimum LogLevel.
   * @note Thread-safety: Lock-free atomic load.
   */
  [[nodiscard]] static auto level () noexcept -> LogLevel;

  /**
   * @brief Installs a custom LogSink or restores default stderr JSON sink.
   * @param sink Shared pointer to custom sink, or nullptr for default stderr.
   * @pre None.
   * @post Future log messages route through sink.
   * @note Thread-safety: Synchronized via internal mutex on cold config path.
   */
  static void set_sink (std::shared_ptr<LogSink> sink) noexcept;

  /**
   * @brief Emits a structured log event if level meets threshold.
   * @param level Severity level of the message.
   * @param subsystem Subsystem identifier string.
   * @param message Diagnostic text.
   * @pre None.
   * @post Sink receives formatted event if level >= Logger::level().
   * @note Thread-safety: Safe for concurrent calls across all threads.
   */
  static void log (LogLevel level, std::string_view subsystem,
                   std::string_view message) noexcept;
};

/**
 * @brief Identifies per-frame execution stages tracked by FrameStats.
 * @pre None.
 * @post Immutable enum type.
 * @note Thread-safety: Safe for concurrent reads.
 */
enum class FrameStage : uint8_t
{
  CpuPrep = 0,
  GpuUpload = 1,
  GpuDraw = 2,
  Present = 3
};

/**
 * @brief Point-in-time snapshot of per-frame timing and memory counters.
 * @pre None.
 * @post Value type containing counter snapshot.
 * @note Thread-safety: Value type; safe for independent access.
 */
struct FrameStatsSnapshot
{
  uint64_t cpu_prep_ns = 0U;
  uint64_t gpu_upload_ns = 0U;
  uint64_t gpu_draw_ns = 0U;
  uint64_t present_ns = 0U;
  uint64_t last_frame_ns = 0U;
  uint64_t total_frame_ns = 0U;
  uint64_t frame_count = 0U;
  uint64_t base_layer_redraws = 0U;
  uint64_t overlay_layer_redraws = 0U;
  uint64_t gpu_allocations_in_frame = 0U;
  uint64_t active_gpu_bytes = 0U;
  uint64_t peak_gpu_bytes = 0U;
  uint64_t active_arena_bytes = 0U;
  uint64_t peak_arena_bytes = 0U;
};

/**
 * @brief Lock-free per-frame telemetry accumulator for Darshana figures.
 * @pre None.
 * @post Maintains monotonic timing totals and memory watermarks.
 * @note Thread-safety: Fully lock-free using std::atomic operations.
 */
class FrameStats
{
public:
  FrameStats () noexcept = default;
  FrameStats (const FrameStats &) = delete;
  auto operator= (const FrameStats &) -> FrameStats & = delete;
  FrameStats (FrameStats &&) = delete;
  auto operator= (FrameStats &&) -> FrameStats & = delete;
  ~FrameStats () noexcept = default;

  /**
   * @brief Records an execution duration for the specified frame stage.
   * @param stage FrameStage that completed.
   * @param duration_ns Elapsed wall-clock time in nanoseconds.
   * @pre None.
   * @post Increments stage duration atomically when stats are enabled.
   * @note Thread-safety: Lock-free atomic fetch_add.
   */
  void record_stage (FrameStage stage, uint64_t duration_ns) noexcept;

  /**
   * @brief Records completion of a rendered frame and its total duration.
   * @param frame_ns Total elapsed frame duration in nanoseconds.
   * @pre None.
   * @post Updates last_frame_ns, total_frame_ns, and frame_count.
   * @note Thread-safety: Lock-free atomic operations.
   */
  void record_frame (uint64_t frame_ns) noexcept;

  /**
   * @brief Increments the static base data layer redraw counter.
   * @pre None.
   * @post Increments base_layer_redraws atomically.
   * @note Thread-safety: Lock-free atomic fetch_add.
   */
  void record_base_redraw () noexcept;

  /**
   * @brief Increments the interactive overlay layer redraw counter.
   * @pre None.
   * @post Increments overlay_layer_redraws atomically.
   * @note Thread-safety: Lock-free atomic fetch_add.
   */
  void record_overlay_redraw () noexcept;

  /**
   * @brief Records any GPU allocation occurring in the frame hot path.
   * @pre Should never be called during steady-state rendering (must remain 0).
   * @post Increments gpu_allocations_in_frame atomically.
   * @note Thread-safety: Lock-free atomic fetch_add.
   */
  void record_gpu_alloc_in_frame () noexcept;

  /**
   * @brief Updates active and peak GPU memory pool usage gauges.
   * @param active_bytes Currently sub-allocated GPU memory bytes.
   * @param peak_bytes Peak sub-allocated GPU memory bytes.
   * @pre None.
   * @post Stores active_gpu_bytes and updates peak_gpu_bytes high-water mark.
   * @note Thread-safety: Lock-free atomic operations.
   */
  void record_gpu_memory (uint64_t active_bytes, uint64_t peak_bytes) noexcept;

  /**
   * @brief Records scratch arena allocation bytes and updates peak watermark.
   * @param active_bytes Active bytes in the PMR scratch arena.
   * @pre None.
   * @post Updates active_arena_bytes and peak_arena_bytes.
   * @note Thread-safety: Lock-free atomic operations.
   */
  void record_arena_usage (uint64_t active_bytes) noexcept;

  /**
   * @brief Captures a point-in-time snapshot of all telemetry counters.
   * @pre None.
   * @post State is unchanged.
   * @returns FrameStatsSnapshot containing current counter values.
   * @note Thread-safety: Lock-free atomic loads.
   */
  [[nodiscard]] auto snapshot () const noexcept -> FrameStatsSnapshot;

  /**
   * @brief Resets all counters to zero.
   * @pre None.
   * @post All counters equal zero.
   * @note Thread-safety: Lock-free atomic stores.
   */
  void reset () noexcept;

  /**
   * @brief Accesses the process-wide default FrameStats instance.
   * @pre None.
   * @post Returns valid reference for process lifetime.
   * @returns Reference to global FrameStats singleton.
   * @note Thread-safety: C++23 magic-static initialization is thread-safe.
   */
  [[nodiscard]] static auto global () noexcept -> FrameStats &;

private:
  std::atomic<uint64_t> cpu_prep_ns_{ 0U };
  std::atomic<uint64_t> gpu_upload_ns_{ 0U };
  std::atomic<uint64_t> gpu_draw_ns_{ 0U };
  std::atomic<uint64_t> present_ns_{ 0U };
  std::atomic<uint64_t> last_frame_ns_{ 0U };
  std::atomic<uint64_t> total_frame_ns_{ 0U };
  std::atomic<uint64_t> frame_count_{ 0U };
  std::atomic<uint64_t> base_layer_redraws_{ 0U };
  std::atomic<uint64_t> overlay_layer_redraws_{ 0U };
  std::atomic<uint64_t> gpu_allocations_in_frame_{ 0U };
  std::atomic<uint64_t> active_gpu_bytes_{ 0U };
  std::atomic<uint64_t> peak_gpu_bytes_{ 0U };
  std::atomic<uint64_t> active_arena_bytes_{ 0U };
  std::atomic<uint64_t> peak_arena_bytes_{ 0U };
};

/**
 * @brief Hook interface for attaching external profilers (Tracy, Perfetto,
 * RenderDoc) without recompilation.
 * @pre Implementations must not throw exceptions.
 * @post Receives zone begin/end, counter, and frame events.
 * @note Thread-safety: Implementations must be safe for concurrent callbacks.
 */
class ProfilerHook
{
public:
  ProfilerHook () noexcept = default;
  ProfilerHook (const ProfilerHook &) noexcept = default;
  auto operator= (const ProfilerHook &) noexcept -> ProfilerHook & = default;
  ProfilerHook (ProfilerHook &&) noexcept = default;
  auto operator= (ProfilerHook &&) noexcept -> ProfilerHook & = default;
  virtual ~ProfilerHook () noexcept = default;

  /**
   * @brief Invoked when an instrumented Darshana zone begins.
   * @param zone_name Static identifier of the zone.
   * @pre zone_name is non-empty.
   * @post External profiler records range start.
   */
  virtual void on_zone_begin (std::string_view zone_name) noexcept = 0;

  /**
   * @brief Invoked when an instrumented Darshana zone ends.
   * @param zone_name Static identifier of the zone.
   * @param elapsed_ns Duration of the zone in nanoseconds.
   * @pre Matching on_zone_begin was called.
   * @post External profiler records range end.
   */
  virtual void on_zone_end (std::string_view zone_name,
                            uint64_t elapsed_ns) noexcept
      = 0;

  /**
   * @brief Invoked when a numeric telemetry counter updates.
   * @param counter_name Name of the counter.
   * @param value New counter value.
   * @pre None.
   * @post External profiler records counter sample.
   */
  virtual void on_counter (std::string_view counter_name,
                           int64_t value) noexcept
      = 0;

  /**
   * @brief Invoked at frame boundaries for Tracy/RenderDoc frame markers.
   * @param frame_index Monotonic frame number.
   * @pre None.
   * @post External profiler marks frame boundary.
   */
  virtual void on_frame_mark (uint64_t frame_index) noexcept;
};

/**
 * @brief Global registry for attaching an external ProfilerHook.
 * @pre None.
 * @post Dispatches profiler events with zero overhead when detached.
 * @note Thread-safety: Lock-free atomic pointer exchange/load.
 */
class Profiler
{
public:
  /**
   * @brief Attaches or detaches a profiler hook.
   * @param hook Pointer to user ProfilerHook, or nullptr to detach.
   * @pre If non-null, *hook must remain valid while attached.
   * @post Subsequent profiling events invoke *hook.
   * @note Thread-safety: Lock-free atomic store.
   */
  static void set_hook (ProfilerHook *hook) noexcept;

  /**
   * @brief Retrieves the currently attached profiler hook.
   * @pre None.
   * @post State is unchanged.
   * @returns Pointer to active ProfilerHook or nullptr.
   * @note Thread-safety: Lock-free atomic load.
   */
  [[nodiscard]] static auto hook () noexcept -> ProfilerHook *;

  /**
   * @brief Emits a counter value to the attached profiler hook if present.
   * @param counter_name Name of the counter.
   * @param value Counter value to report.
   * @pre None.
   * @post Invokes hook()->on_counter if hook is non-null.
   * @note Thread-safety: Lock-free atomic load and callback dispatch.
   */
  static void emit_counter (std::string_view counter_name,
                            int64_t value) noexcept;

  /**
   * @brief Emits a frame boundary marker to the attached profiler hook.
   * @param frame_index Monotonic frame number.
   * @pre None.
   * @post Invokes hook()->on_frame_mark if hook is non-null.
   * @note Thread-safety: Lock-free atomic load and callback dispatch.
   */
  static void mark_frame (uint64_t frame_index) noexcept;
};

/**
 * @brief RAII scope guard that records frame stage timing and profiler zones.
 * @pre None.
 * @post Records elapsed nanoseconds on destruction.
 * @note Thread-safety: Stack-allocated per scope; thread-safe.
 */
class ScopedFrameTimer
{
public:
  /**
   * @brief Begins timing a frame stage on the given FrameStats object.
   * @param stats Target FrameStats accumulator.
   * @param stage FrameStage being timed.
   * @param zone_name Name forwarded to Profiler hook.
   * @pre None.
   * @post Captures start timestamp when DARSHANA_ENABLE_STATS is enabled.
   */
  ScopedFrameTimer (FrameStats &stats, FrameStage stage,
                    std::string_view zone_name) noexcept;

  ScopedFrameTimer (const ScopedFrameTimer &) = delete;
  auto operator= (const ScopedFrameTimer &) -> ScopedFrameTimer & = delete;
  ScopedFrameTimer (ScopedFrameTimer &&) = delete;
  auto operator= (ScopedFrameTimer &&) -> ScopedFrameTimer & = delete;

  /**
   * @brief Stops timing and commits elapsed nanoseconds to FrameStats.
   */
  ~ScopedFrameTimer () noexcept;

#if DARSHANA_ENABLE_STATS
private:
  FrameStats *stats_ = nullptr;
  FrameStage stage_ = FrameStage::CpuPrep;
  std::string_view zone_name_;
  uint64_t start_ns_ = 0U;
#endif
};

/**
 * @brief Returns monotonic clock time in nanoseconds.
 * @pre None.
 * @post State is unchanged.
 * @returns Current steady_clock timestamp in nanoseconds.
 * @note Thread-safety: Thread-safe OS monotonic clock query.
 */
[[nodiscard]] auto monotonic_time_ns () noexcept -> uint64_t;

} /* namespace darshana::inline v1 */

#endif /* DARSHANA_OBSERVABILITY_HPP */
