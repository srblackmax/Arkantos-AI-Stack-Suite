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

#ifndef YANTRA_OBSERVABILITY_HPP
#define YANTRA_OBSERVABILITY_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#ifndef YANTRA_ENABLE_STATS
#define YANTRA_ENABLE_STATS 1
#endif

namespace yantra::inline v1
{

/**
 * @brief Severity levels for Yantra structured JSON-lines logging.
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
   * @param subsystem Name of the emitting Yantra subsystem.
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
 * @brief Snapshot of experiment tracking telemetry counters.
 * @note Thread-safety: Value type; safe for independent access.
 */
struct TrackingStatsSnapshot
{
  uint64_t metrics_logged_total = 0U;
  double metrics_logged_per_sec = 0.0;
  uint64_t query_p50_ns = 0U;
  uint64_t query_p99_ns = 0U;
  uint64_t wal_size_bytes = 0U;
  uint64_t compactions_completed = 0U;
};

/**
 * @brief Lock-free telemetry accumulator for yantra::track.
 * @note Thread-safety: Fully lock-free using std::atomic operations.
 */
class TrackingStats
{
public:
  TrackingStats () noexcept = default;
  TrackingStats (const TrackingStats &) = delete;
  auto operator= (const TrackingStats &) -> TrackingStats & = delete;

  /**
   * @brief Records metric points appended to the WAL.
   * @param count Number of metric points logged.
   * @param elapsed_ns Duration of the write batch in nanoseconds.
   * @param wal_bytes Total WAL byte size after append.
   */
  void record_metrics_logged (uint64_t count, uint64_t elapsed_ns,
                              uint64_t wal_bytes) noexcept;

  /**
   * @brief Records a tracking store query latency sample.
   * @param latency_ns Elapsed query latency in nanoseconds.
   */
  void record_query_latency (uint64_t latency_ns) noexcept;

  /**
   * @brief Increments the completed WAL compaction counter.
   */
  void record_compaction () noexcept;

  /**
   * @brief Captures a point-in-time snapshot of tracking statistics.
   * @returns TrackingStatsSnapshot with current counters.
   */
  [[nodiscard]] auto snapshot () const noexcept -> TrackingStatsSnapshot;

  /**
   * @brief Resets all counters to zero.
   */
  void reset () noexcept;

  /**
   * @brief Returns the process-wide TrackingStats instance.
   */
  [[nodiscard]] static auto global () noexcept -> TrackingStats &;

private:
  std::atomic<uint64_t> metrics_logged_total_{ 0U };
  std::atomic<uint64_t> logging_time_ns_{ 0U };
  std::atomic<uint64_t> last_query_ns_{ 0U };
  std::atomic<uint64_t> max_query_ns_{ 0U };
  std::atomic<uint64_t> wal_size_bytes_{ 0U };
  std::atomic<uint64_t> compactions_completed_{ 0U };
};

/**
 * @brief Snapshot of model registry telemetry counters.
 * @note Thread-safety: Value type; safe for independent access.
 */
struct RegistryStatsSnapshot
{
  uint64_t registrations_total = 0U;
  double registrations_per_sec = 0.0;
  uint64_t stage_transitions_total = 0U;
  uint64_t lineage_query_p50_ns = 0U;
  uint64_t lineage_query_p99_ns = 0U;
};

/**
 * @brief Lock-free telemetry accumulator for yantra::registry.
 * @note Thread-safety: Fully lock-free using std::atomic operations.
 */
class RegistryStats
{
public:
  RegistryStats () noexcept = default;
  RegistryStats (const RegistryStats &) = delete;
  auto operator= (const RegistryStats &) -> RegistryStats & = delete;

  /**
   * @brief Records a model version registration event.
   * @param duration_ns Duration of the registration in nanoseconds.
   */
  void record_registration (uint64_t duration_ns) noexcept;

  /**
   * @brief Records a model stage transition event.
   */
  void record_stage_transition () noexcept;

  /**
   * @brief Records a lineage graph query latency sample.
   * @param latency_ns Elapsed lineage query duration in nanoseconds.
   */
  void record_lineage_query (uint64_t latency_ns) noexcept;

  /**
   * @brief Captures a point-in-time snapshot of registry statistics.
   * @returns RegistryStatsSnapshot with current counters.
   */
  [[nodiscard]] auto snapshot () const noexcept -> RegistryStatsSnapshot;

  /**
   * @brief Resets all counters to zero.
   */
  void reset () noexcept;

  /**
   * @brief Returns the process-wide RegistryStats instance.
   */
  [[nodiscard]] static auto global () noexcept -> RegistryStats &;

private:
  std::atomic<uint64_t> registrations_total_{ 0U };
  std::atomic<uint64_t> registration_time_ns_{ 0U };
  std::atomic<uint64_t> stage_transitions_total_{ 0U };
  std::atomic<uint64_t> last_lineage_ns_{ 0U };
  std::atomic<uint64_t> max_lineage_ns_{ 0U };
};

/**
 * @brief Snapshot of DAG orchestrator telemetry counters.
 * @note Thread-safety: Value type; safe for independent access.
 */
struct DagStatsSnapshot
{
  uint64_t tasks_completed_total = 0U;
  double tasks_per_sec = 0.0;
  uint64_t scheduling_latency_ns = 0U;
  uint64_t max_worker_queue_depth = 0U;
  uint64_t retry_count_total = 0U;
  uint64_t steal_count_total = 0U;
};

/**
 * @brief Lock-free telemetry accumulator for yantra::dag.
 * @note Thread-safety: Fully lock-free using std::atomic operations.
 */
class DagStats
{
public:
  DagStats () noexcept = default;
  DagStats (const DagStats &) = delete;
  auto operator= (const DagStats &) -> DagStats & = delete;

  /**
   * @brief Records completion of a batch of DAG tasks.
   * @param task_count Number of tasks executed.
   * @param wall_ns Elapsed wall time in nanoseconds.
   * @param schedule_ns Topological scheduling latency in nanoseconds.
   */
  void record_dag_run (uint64_t task_count, uint64_t wall_ns,
                       uint64_t schedule_ns) noexcept;

  /**
   * @brief Records a task retry attempt.
   */
  void record_retry () noexcept;

  /**
   * @brief Records a successful work-stealing operation between workers.
   */
  void record_steal () noexcept;

  /**
   * @brief Updates the observed worker deque depth watermark.
   * @param depth Queue depth observed on a worker deque.
   */
  void record_queue_depth (uint64_t depth) noexcept;

  /**
   * @brief Captures a point-in-time snapshot of DAG statistics.
   * @returns DagStatsSnapshot with current counters.
   */
  [[nodiscard]] auto snapshot () const noexcept -> DagStatsSnapshot;

  /**
   * @brief Resets all counters to zero.
   */
  void reset () noexcept;

  /**
   * @brief Returns the process-wide DagStats instance.
   */
  [[nodiscard]] static auto global () noexcept -> DagStats &;

private:
  std::atomic<uint64_t> tasks_completed_total_{ 0U };
  std::atomic<uint64_t> total_wall_ns_{ 0U };
  std::atomic<uint64_t> last_schedule_ns_{ 0U };
  std::atomic<uint64_t> max_queue_depth_{ 0U };
  std::atomic<uint64_t> retry_count_total_{ 0U };
  std::atomic<uint64_t> steal_count_total_{ 0U };
};

/**
 * @brief Snapshot of vector index telemetry counters.
 * @note Thread-safety: Value type; safe for independent access.
 */
struct IndexStatsSnapshot
{
  uint64_t queries_total = 0U;
  uint64_t search_p50_ns = 0U;
  uint64_t search_p99_ns = 0U;
  double recall_estimate = 1.0;
  uint64_t memory_usage_bytes = 0U;
};

/**
 * @brief Lock-free telemetry accumulator for yantra::index.
 * @note Thread-safety: Fully lock-free using std::atomic operations.
 */
class IndexStats
{
public:
  IndexStats () noexcept = default;
  IndexStats (const IndexStats &) = delete;
  auto operator= (const IndexStats &) -> IndexStats & = delete;

  /**
   * @brief Records a vector search latency sample.
   * @param latency_ns Search latency in nanoseconds.
   */
  void record_search (uint64_t latency_ns) noexcept;

  /**
   * @brief Updates the periodic brute-force recall estimate.
   * @param recall Measured recall in [0.0, 1.0].
   */
  void set_recall_estimate (double recall) noexcept;

  /**
   * @brief Updates the current index memory footprint in bytes.
   * @param bytes Resident index memory usage in bytes.
   */
  void set_memory_usage (uint64_t bytes) noexcept;

  /**
   * @brief Captures a point-in-time snapshot of index statistics.
   * @returns IndexStatsSnapshot with current counters.
   */
  [[nodiscard]] auto snapshot () const noexcept -> IndexStatsSnapshot;

  /**
   * @brief Resets all counters to zero.
   */
  void reset () noexcept;

  /**
   * @brief Returns the process-wide IndexStats instance.
   */
  [[nodiscard]] static auto global () noexcept -> IndexStats &;

private:
  std::atomic<uint64_t> queries_total_{ 0U };
  std::atomic<uint64_t> last_search_ns_{ 0U };
  std::atomic<uint64_t> max_search_ns_{ 0U };
  std::atomic<uint64_t> recall_fixed_pt_{ 1000000U };
  std::atomic<uint64_t> memory_usage_bytes_{ 0U };
};

/**
 * @brief Snapshot of fairness and drift auditing telemetry counters.
 * @note Thread-safety: Value type; safe for independent access.
 */
struct FairnessStatsSnapshot
{
  uint64_t audits_performed = 0U;
  double avg_constraint_violation = 0.0;
  double avg_threshold_adjustment = 0.0;
  uint64_t last_audit_duration_ns = 0U;
};

/**
 * @brief Lock-free telemetry accumulator for yantra::fairness.
 * @note Thread-safety: Fully lock-free using std::atomic operations.
 */
class FairnessStats
{
public:
  FairnessStats () noexcept = default;
  FairnessStats (const FairnessStats &) = delete;
  auto operator= (const FairnessStats &) -> FairnessStats & = delete;

  /**
   * @brief Records a completed fairness audit and optional threshold shift.
   * @param violation_mag Observed fairness constraint violation magnitude.
   * @param threshold_adj Magnitude of post-processing threshold adjustment.
   * @param duration_ns Audit execution duration in nanoseconds.
   */
  void record_audit (double violation_mag, double threshold_adj,
                     uint64_t duration_ns) noexcept;

  /**
   * @brief Captures a point-in-time snapshot of fairness statistics.
   * @returns FairnessStatsSnapshot with current counters.
   */
  [[nodiscard]] auto snapshot () const noexcept -> FairnessStatsSnapshot;

  /**
   * @brief Resets all counters to zero.
   */
  void reset () noexcept;

  /**
   * @brief Returns the process-wide FairnessStats instance.
   */
  [[nodiscard]] static auto global () noexcept -> FairnessStats &;

private:
  std::atomic<uint64_t> audits_performed_{ 0U };
  std::atomic<uint64_t> violation_sum_fixed_{ 0U };
  std::atomic<uint64_t> adjustment_sum_fixed_{ 0U };
  std::atomic<uint64_t> last_duration_ns_{ 0U };
};

/**
 * @brief Hook interface for attaching external profilers (Tracy/Perfetto/NVTX)
 * without recompiling Yantra.
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
   * @brief Invoked when an instrumented Yantra zone begins.
   * @param zone_name Static name of the zone.
   */
  virtual void on_zone_begin (std::string_view zone_name) noexcept = 0;

  /**
   * @brief Invoked when an instrumented Yantra zone ends.
   * @param zone_name Static name of the zone.
   * @param elapsed_ns Duration of the zone in nanoseconds.
   */
  virtual void on_zone_end (std::string_view zone_name,
                            uint64_t elapsed_ns) noexcept
      = 0;

  /**
   * @brief Invoked when a numeric telemetry counter updates.
   * @param counter_name Name of the counter.
   * @param value New counter value.
   */
  virtual void on_counter (std::string_view counter_name,
                           int64_t value) noexcept
      = 0;
};

/**
 * @brief Global registry for attaching an external ProfilerHook.
 * @note Thread-safety: Lock-free atomic pointer exchange/load.
 */
class YantraProfiler
{
public:
  /**
   * @brief Attaches or detaches a profiler hook.
   * @param hook Pointer to user ProfilerHook, or nullptr to detach.
   */
  static void set_hook (ProfilerHook *hook) noexcept;

  /**
   * @brief Retrieves the currently attached profiler hook.
   * @returns Pointer to active ProfilerHook or nullptr.
   */
  [[nodiscard]] static auto hook () noexcept -> ProfilerHook *;

  /**
   * @brief Emits a counter value to the attached profiler hook if present.
   * @param counter_name Name of the counter.
   * @param value Counter value to report.
   */
  static void emit_counter (std::string_view counter_name,
                            int64_t value) noexcept;
};

/**
 * @brief RAII scope guard that notifies YantraProfiler on scope entry/exit.
 * @note Thread-safety: Stack-allocated per scope; thread-safe.
 */
class ScopedZoneTimer
{
public:
  explicit ScopedZoneTimer (std::string_view zone_name) noexcept;
  ScopedZoneTimer (const ScopedZoneTimer &) = delete;
  auto operator= (const ScopedZoneTimer &) -> ScopedZoneTimer & = delete;
  ~ScopedZoneTimer () noexcept;

#if YANTRA_ENABLE_STATS
private:
  std::string_view zone_name_;
  uint64_t start_ns_ = 0U;
#endif
};

/**
 * @brief Returns monotonic clock time in nanoseconds (CLOCK_MONOTONIC).
 * @returns Current steady_clock timestamp in nanoseconds.
 * @note Thread-safety: Thread-safe OS monotonic clock query.
 */
[[nodiscard]] auto monotonic_time_ns () noexcept -> uint64_t;

/**
 * @brief Computes hardware-accelerated Castagnoli CRC32C checksum over data.
 * @param data Byte span to checksum.
 * @param initial_crc Initial seed CRC value (default 0).
 * @returns 32-bit CRC32C checksum.
 * @note Thread-safety: Pure function; thread-safe.
 */
[[nodiscard]] auto
compute_crc32c (std::span<const std::byte> data,
                uint32_t initial_crc = 0U) noexcept -> uint32_t;

/**
 * @brief Computes 256-bit cryptographic SHA-256 hex digest (64 lowercase hex
 * chars).
 * @param data Byte span to hash.
 * @returns 64-character lowercase hexadecimal SHA-256 digest string.
 * @note Thread-safety: Pure function; thread-safe.
 */
[[nodiscard]] auto
compute_sha256_hex (std::span<const std::byte> data) noexcept -> std::string;

/**
 * @brief Computes 256-bit cryptographic BLAKE3 hex digest (64 lowercase hex
 * chars), parallelizing across hardware threads for large buffers (>= 4 MB).
 * @param data Byte span to hash.
 * @returns 64-character lowercase hexadecimal BLAKE3 digest string.
 * @note Thread-safety: Pure function; thread-safe.
 */
[[nodiscard]] auto
compute_blake3_hex (std::span<const std::byte> data) noexcept -> std::string;

} /* namespace yantra::inline v1 */

#endif /* YANTRA_OBSERVABILITY_HPP */
