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

#ifndef SUTRA_OBSERVABILITY_HPP
#define SUTRA_OBSERVABILITY_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <string_view>

#ifndef SUTRA_ENABLE_STATS
#define SUTRA_ENABLE_STATS 1
#endif

namespace sutra::inline v1
{

/**
 * @brief Severity levels for Sutra structured JSON-lines logging.
 * @note Thread-safety: Immutable enum type.
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
   * @param subsystem Name of the emitting Sutra subsystem.
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
 * @brief Identifies execution stages tracked by AgentStats.
 * @note Thread-safety: Immutable enum type.
 */
enum class StageKind : uint8_t
{
  Planning = 0,
  ToolDispatch = 1,
  LlmInference = 2,
  ToolExecution = 3,
  CheckpointWrite = 4
};

/**
 * @brief Point-in-time snapshot of per-stage timing and resource counters.
 * @note Thread-safety: Value type; safe for independent access.
 */
struct AgentStatsSnapshot
{
  uint64_t planning_ns = 0U;
  uint64_t planning_calls = 0U;
  uint64_t tool_dispatch_ns = 0U;
  uint64_t tool_dispatch_calls = 0U;
  uint64_t llm_inference_ns = 0U;
  uint64_t llm_inference_calls = 0U;
  uint64_t tool_execution_ns = 0U;
  uint64_t tool_execution_calls = 0U;
  uint64_t checkpoint_write_ns = 0U;
  uint64_t checkpoint_write_calls = 0U;
  uint64_t overlapped_tool_tokens = 0U;
  uint64_t active_arena_bytes = 0U;
  uint64_t peak_arena_bytes = 0U;
  uint64_t last_ttft_ns = 0U;
  uint64_t total_ttft_ns = 0U;
  uint64_t ttft_samples = 0U;
  uint64_t last_generation_ns = 0U;
  uint64_t last_tokens_generated = 0U;
  uint64_t total_tokens_generated = 0U;
  double last_tokens_per_sec = 0.0;
  uint64_t sentence_to_tts_ns = 0U;
  uint64_t sentence_to_tts_samples = 0U;
  uint64_t tts_queue_depth = 0U;
  uint64_t playback_buffer_occupancy = 0U;
  uint64_t stt_partial_latency_ns = 0U;
  uint64_t barge_in_count = 0U;
  uint64_t xrun_count = 0U;
};

/**
 * @brief Lock-free per-stage telemetry accumulator for agent orchestration.
 * @pre None.
 * @post Maintains monotonic timing totals and arena memory high-water marks.
 * @note Thread-safety: Fully lock-free using std::atomic relaxed/acq-rel ops.
 */
class AgentStats
{
public:
  AgentStats () noexcept = default;
  AgentStats (const AgentStats &) = delete;
  auto operator= (const AgentStats &) -> AgentStats & = delete;
  AgentStats (AgentStats &&) = delete;
  auto operator= (AgentStats &&) -> AgentStats & = delete;
  ~AgentStats () noexcept = default;

  /**
   * @brief Records an execution duration for the specified stage.
   * @param stage The pipeline stage that completed.
   * @param duration_ns Elapsed wall-clock time in nanoseconds.
   * @pre None.
   * @post Increments stage duration and invocation count atomically.
   * @note Thread-safety: Lock-free atomic fetch_add.
   */
  void record_stage (StageKind stage, uint64_t duration_ns) noexcept;

  /**
   * @brief Records LLM tokens generated concurrently with tool execution.
   * @param token_count Number of overlapped tokens generated.
   * @pre None.
   * @post Increments overlapped_tool_tokens counter atomically.
   * @note Thread-safety: Lock-free atomic fetch_add.
   */
  void record_overlapped_tokens (uint64_t token_count) noexcept;

  /**
   * @brief Records scratch arena allocation bytes and updates peak watermark.
   * @param bytes Number of bytes allocated from scratch arena.
   * @pre None.
   * @post Updates active_arena_bytes and peak_arena_bytes.
   * @note Thread-safety: Lock-free atomic operations.
   */
  void record_arena_alloc (uint64_t bytes) noexcept;

  /**
   * @brief Records scratch arena release bytes on turn reset.
   * @param bytes Number of bytes released back to scratch arena.
   * @pre None.
   * @post Decrements active_arena_bytes atomically.
   * @note Thread-safety: Lock-free atomic operations.
   */
  void record_arena_release (uint64_t bytes) noexcept;

  /**
   * @brief Records time-to-first-token (TTFT) latency for a completion turn.
   */
  void record_ttft (uint64_t ttft_ns) noexcept;

  /**
   * @brief Records per-turn token generation count and total generation time.
   */
  void record_generation (uint64_t tokens_generated,
                          uint64_t generation_ns) noexcept;

  /**
   * @brief Records sentence-boundary detection to TTS enqueue latency.
   */
  void record_sentence_to_tts (uint64_t latency_ns) noexcept;

  /**
   * @brief Updates the current TTS sentence queue depth gauge.
   */
  void set_tts_queue_depth (uint64_t depth) noexcept;

  /**
   * @brief Updates the current audio playback ring buffer occupancy gauge.
   */
  void set_playback_buffer_occupancy (uint64_t samples) noexcept;

  /**
   * @brief Records STT partial or final hypothesis latency in nanoseconds.
   */
  void record_stt_latency (uint64_t latency_ns) noexcept;

  /**
   * @brief Increments the voice barge-in interruption counter.
   */
  void record_barge_in () noexcept;

  /**
   * @brief Increments the ALSA PCM overrun/underrun (xrun) counter.
   */
  void record_xrun () noexcept;

  /**
   * @brief Returns the current ALSA PCM xrun count.
   */
  [[nodiscard]] auto xrun_count () const noexcept -> uint64_t;

  /**
   * @brief Captures a snapshot of all telemetry counters.
   * @pre None.
   * @post State is unchanged.
   * @returns AgentStatsSnapshot containing current counter values.
   * @note Thread-safety: Lock-free atomic loads.
   */
  [[nodiscard]] auto snapshot () const noexcept -> AgentStatsSnapshot;

  /**
   * @brief Resets all counters to zero.
   * @pre None.
   * @post All counters equal zero.
   * @note Thread-safety: Lock-free atomic stores.
   */
  void reset () noexcept;

  /**
   * @brief Accesses the process-wide default AgentStats instance.
   * @pre None.
   * @post Returns valid reference for process lifetime.
   * @returns Reference to global AgentStats singleton.
   * @note Thread-safety: C++23 magic-static initialization is thread-safe.
   */
  [[nodiscard]] static auto global () noexcept -> AgentStats &;

private:
  std::atomic<uint64_t> planning_ns_{ 0U };
  std::atomic<uint64_t> planning_calls_{ 0U };
  std::atomic<uint64_t> tool_dispatch_ns_{ 0U };
  std::atomic<uint64_t> tool_dispatch_calls_{ 0U };
  std::atomic<uint64_t> llm_inference_ns_{ 0U };
  std::atomic<uint64_t> llm_inference_calls_{ 0U };
  std::atomic<uint64_t> tool_execution_ns_{ 0U };
  std::atomic<uint64_t> tool_execution_calls_{ 0U };
  std::atomic<uint64_t> checkpoint_write_ns_{ 0U };
  std::atomic<uint64_t> checkpoint_write_calls_{ 0U };
  std::atomic<uint64_t> overlapped_tool_tokens_{ 0U };
  std::atomic<uint64_t> active_arena_bytes_{ 0U };
  std::atomic<uint64_t> peak_arena_bytes_{ 0U };
  std::atomic<uint64_t> last_ttft_ns_{ 0U };
  std::atomic<uint64_t> total_ttft_ns_{ 0U };
  std::atomic<uint64_t> ttft_samples_{ 0U };
  std::atomic<uint64_t> last_generation_ns_{ 0U };
  std::atomic<uint64_t> last_tokens_generated_{ 0U };
  std::atomic<uint64_t> total_tokens_generated_{ 0U };
  std::atomic<uint64_t> sentence_to_tts_ns_{ 0U };
  std::atomic<uint64_t> sentence_to_tts_samples_{ 0U };
  std::atomic<uint64_t> tts_queue_depth_{ 0U };
  std::atomic<uint64_t> playback_buffer_occupancy_{ 0U };
  std::atomic<uint64_t> stt_partial_latency_ns_{ 0U };
  std::atomic<uint64_t> barge_in_count_{ 0U };
  std::atomic<uint64_t> xrun_count_{ 0U };
};

/**
 * @brief Hook interface for attaching external profilers without recompiling.
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
   * @brief Invoked when an instrumented Sutra zone begins.
   * @param zone_name Null-terminated static name of the zone.
   * @pre zone_name is non-null.
   * @post External profiler records range start.
   */
  virtual void on_zone_begin (std::string_view zone_name) noexcept = 0;

  /**
   * @brief Invoked when an instrumented Sutra zone ends.
   * @param zone_name Null-terminated static name of the zone.
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
};

/**
 * @brief Global registry for attaching an external ProfilerHook.
 * @note Thread-safety: Lock-free atomic pointer exchange/load.
 */
class AgentProfiler
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
   */
  static void emit_counter (std::string_view counter_name,
                            int64_t value) noexcept;
};

/**
 * @brief Unified Tracy/timeline profiler hook alias for Sphinx + Vach + Sutra.
 */
using SphinxProfiler = AgentProfiler;

/**
 * @brief RAII scope guard that records stage timing and profiler zones.
 * @note Thread-safety: Stack-allocated per scope; thread-safe.
 */
class ScopedStageTimer
{
public:
  /**
   * @brief Begins timing a stage on the given stats object.
   * @param stats Target AgentStats accumulator.
   * @param stage StageKind being timed.
   * @param zone_name Name forwarded to AgentProfiler hook.
   * @pre None.
   * @post Captures start timestamp when SUTRA_ENABLE_STATS is enabled.
   */
  ScopedStageTimer (AgentStats &stats, StageKind stage,
                    std::string_view zone_name) noexcept;

  ScopedStageTimer (const ScopedStageTimer &) = delete;
  auto operator= (const ScopedStageTimer &) -> ScopedStageTimer & = delete;
  ScopedStageTimer (ScopedStageTimer &&) = delete;
  auto operator= (ScopedStageTimer &&) -> ScopedStageTimer & = delete;

  /**
   * @brief Stops timing and commits elapsed nanoseconds to AgentStats.
   */
  ~ScopedStageTimer () noexcept;

#if SUTRA_ENABLE_STATS
private:
  AgentStats *stats_ = nullptr;
  StageKind stage_ = StageKind::Planning;
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

} /* namespace sutra::inline v1 */

#endif /* SUTRA_OBSERVABILITY_HPP */
