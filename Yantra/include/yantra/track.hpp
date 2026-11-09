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

#ifndef YANTRA_TRACK_HPP
#define YANTRA_TRACK_HPP

#include <yantra/error.hpp>
#include <yantra/observability.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace yantra::inline v1
{

/**
 * @brief Distinguishes payload types stored in a 64-byte WalRecord.
 * @note Thread-safety: Immutable enum type.
 */
enum class WalValueKind : uint8_t
{
  Scalar = 0,
  Histogram = 1,
  BlobRef = 2,
  RunStart = 3,
  RunEnd = 4,
  Param = 5
};

/**
 * @brief Fixed 64-byte Write-Ahead Log record protected by CRC32C.
 * @note Thread-safety: Trivially copyable value type.
 */
struct alignas (8) WalRecord
{
  uint64_t seq_id = 0U;
  uint64_t run_id_num = 0U;
  uint64_t step = 0U;
  uint64_t timestamp_ns = 0U;
  double scalar_value = 0.0;
  uint32_t key_id = 0U;
  uint8_t value_kind = static_cast<uint8_t> (WalValueKind::Scalar);
  uint8_t flags = 0U;
  uint16_t reserved = 0U;
  uint64_t payload_aux = 0U;
  uint32_t crc32c = 0U;
  uint32_t magic = 0x5957414CU; /* "YWAL" */
};

static_assert (sizeof (WalRecord) == 64U,
               "WalRecord must be exactly 64 bytes");

/**
 * @brief Fixed 64-byte Write-Ahead Log file header protected by CRC32C.
 * @note Thread-safety: Trivially copyable value type.
 */
struct alignas (8) WalFileHeader
{
  std::array<char, 8> magic
      = { 'Y', 'A', 'N', 'T', 'R', 'A', 'W', '1' };
  uint32_t version = 1U;
  uint32_t record_bytes = 64U;
  uint64_t created_at_ns = 0U;
  std::array<uint64_t, 4> reserved = { 0U, 0U, 0U, 0U };
  uint32_t header_crc32c = 0U;
  uint32_t tail_magic = 0x57414C48U;
};

static_assert (sizeof (WalFileHeader) == 64U,
               "WalFileHeader must be exactly 64 bytes");

/**
 * @brief Crash-safe Write-Ahead Log engine supporting high-throughput OS
 * page-cache appends and durable NVMe fdatasync(2) commits.
 * @pre File path parent directory must exist and be writable.
 * @post File descriptor is managed via RAII and closed on destruction.
 * @note Thread-safety: Safe for concurrent multi-writer appends via internal
 * lock-free sequence counter and pwrite(2) / serialized flush.
 */
class WalEngine
{
public:
  WalEngine () noexcept;
  WalEngine (const WalEngine &) = delete;
  auto operator= (const WalEngine &) -> WalEngine & = delete;
  WalEngine (WalEngine &&other) noexcept;
  auto operator= (WalEngine &&other) noexcept -> WalEngine &;
  ~WalEngine () noexcept;

  /**
   * @brief Opens or creates a WAL segment file at the specified path.
   * @param wal_path Filesystem path for the WAL binary file.
   * @param truncate_existing If true, truncates and writes a fresh header.
   * @returns Result<WalEngine> or TrackingErrorCode::WalWriteFailure.
   */
  [[nodiscard]] static auto
  open (const std::filesystem::path &wal_path,
        bool truncate_existing = true) noexcept -> Result<WalEngine>;

  /**
   * @brief Serializes a batch of metric records, computes per-record hardware
   * CRC32C checksums, writes them via write(2)/pwrite(2), and optionally
   * flushes to non-volatile storage via fdatasync(2).
   * @param records Span of records to append (seq_id and crc32c are populated
   * automatically).
   * @param durable_fdatasync If true, invokes fdatasync(2) before returning.
   * @returns Result<uint64_t> containing the last assigned sequence ID.
   */
  [[nodiscard]] auto
  append_batch (std::span<const WalRecord> records,
                bool durable_fdatasync = false) noexcept -> Result<uint64_t>;

  /**
   * @brief Appends a single metric record to the WAL.
   * @param record Record to append.
   * @param durable_fdatasync If true, invokes fdatasync(2) before returning.
   * @returns Result<uint64_t> containing the assigned sequence ID.
   */
  [[nodiscard]] auto
  append_record (const WalRecord &record,
                 bool durable_fdatasync = false) noexcept -> Result<uint64_t>;

  /**
   * @brief Forces a durable flush of all written WAL records via fdatasync(2).
   * @returns Result<void> or TrackingErrorCode::WalWriteFailure.
   */
  [[nodiscard]] auto sync_durable () noexcept -> Result<void>;

  /**
   * @brief Queries the on-disk byte size of the WAL file via stat(2)/fstat(2).
   * @returns Result<uint64_t> byte size on disk.
   */
  [[nodiscard]] auto file_size_bytes () const noexcept -> Result<uint64_t>;

  /**
   * @brief Re-opens a WAL file from disk, verifies the header and every
   * record's CRC32C checksum, and returns all valid recovered records.
   * @param wal_path Path to the WAL file on disk.
   * @returns Result<std::vector<WalRecord>> or
   * TrackingErrorCode::CorruptedSegment if any checksum or framing fails.
   */
  [[nodiscard]] static auto
  recover (const std::filesystem::path &wal_path) noexcept
      -> Result<std::vector<WalRecord>>;

  /**
   * @brief Queries the filesystem path of this WAL segment.
   */
  [[nodiscard]] auto path () const noexcept -> const std::filesystem::path &;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Summary representation of a logged metric histogram.
 * @note Thread-safety: Value type.
 */
struct HistogramSummary
{
  uint64_t count = 0U;
  double min_value = 0.0;
  double max_value = 0.0;
  double sum_value = 0.0;
  std::array<uint32_t, 8> buckets = { 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U };
};

/**
 * @brief Lifecycle state of an experiment tracking Run.
 */
enum class RunState : uint8_t
{
  Created = 0,
  Running = 1,
  Completed = 2,
  Failed = 3
};

/**
 * @brief Min/max/last summary of a metric key within a run for fast columnar
 * range queries.
 */
struct MetricSummary
{
  double min_val = 0.0;
  double max_val = 0.0;
  double last_val = 0.0;
  uint64_t count = 0U;
};

/**
 * @brief Single logged scalar metric observation at a specific step.
 */
struct MetricPoint
{
  uint64_t step = 0U;
  double value = 0.0;
  uint64_t timestamp_ns = 0U;
};

/**
 * @brief Snapshot of a tracked run returned by ExperimentStore::query().
 */
struct RunRecord
{
  std::string run_id;
  uint64_t numeric_id = 0U;
  uint64_t start_timestamp_ns = 0U;
  RunState state = RunState::Created;
  std::unordered_map<std::string, std::string> params;
  std::unordered_map<std::string, std::string> tags;
  std::unordered_map<std::string, MetricSummary> metrics;
  std::unordered_map<std::string, std::vector<MetricPoint>> metric_history;
  std::unordered_map<std::string, std::string> artifacts;
};

/**
 * @brief Predicate filter over parameters, metrics, and tags for querying
 * runs in ExperimentStore.
 */
struct QueryFilter
{
  std::unordered_map<std::string, std::string> required_params;
  std::unordered_map<std::string, std::string> required_tags;
  std::string metric_key;
  double metric_min = -1.0e300;
  double metric_max = 1.0e300;
  bool has_metric_filter = false;
  std::string param_range_key;
  double param_min = -1.0e300;
  double param_max = 1.0e300;
  bool has_param_range = false;
};

class ExperimentStore;

/**
 * @brief Scoped RAII experiment run object that logs parameters, tags,
 * crash-safe WAL metrics, and content-addressed BLAKE3 artifacts.
 * @pre Created via ExperimentStore::create_run().
 * @post Destructor flushes buffered WAL records and marks run Completed.
 * @note Thread-safety: Safe for concurrent log_metric() calls across threads.
 */
class Run
{
public:
  Run (const Run &) = delete;
  auto operator= (const Run &) -> Run & = delete;
  Run (Run &&other) noexcept;
  auto operator= (Run &&other) noexcept -> Run &;
  ~Run () noexcept;

  /**
   * @brief Retrieves the string identifier of this run.
   */
  [[nodiscard]] auto run_id () const noexcept -> std::string_view;

  /**
   * @brief Retrieves the numeric identifier of this run.
   */
  [[nodiscard]] auto numeric_id () const noexcept -> uint64_t;

  /**
   * @brief Retrieves the start timestamp in nanoseconds.
   */
  [[nodiscard]] auto start_timestamp_ns () const noexcept -> uint64_t;

  /**
   * @brief Retrieves the current lifecycle state of this run.
   */
  [[nodiscard]] auto state () const noexcept -> RunState;

  /**
   * @brief Transitions the run from Created to Running, locking parameters.
   * @returns Result<void>.
   */
  [[nodiscard]] auto start () noexcept -> Result<void>;

  /**
   * @brief Logs a hyperparameter key-value pair before the run enters the
   * Running state.
   * @param key Hyperparameter name.
   * @param value Hyperparameter value string.
   * @returns Result<void> or TrackingErrorCode::ImmutableParameterViolation
   * if the run has already entered Running or Completed state.
   */
  [[nodiscard]] auto
  log_param (std::string_view key,
             std::string_view value) noexcept -> Result<void>;

  /**
   * @brief Logs a mutable metadata tag on the run.
   * @param key Tag key.
   * @param value Tag value.
   * @returns Result<void>.
   */
  [[nodiscard]] auto
  log_tag (std::string_view key,
           std::string_view value) noexcept -> Result<void>;

  /**
   * @brief Logs a scalar metric point (step, key, value, timestamp) to the
   * crash-safe WAL. Transitions the run to Running if still in Created state.
   * @param step Training step or epoch index.
   * @param key Metric name (e.g., "loss", "val_accuracy").
   * @param value Scalar floating-point metric value.
   * @param timestamp_ns Optional timestamp (0 uses monotonic_time_ns()).
   * @returns Result<void> or TrackingErrorCode::WalWriteFailure.
   */
  [[nodiscard]] auto
  log_metric (uint64_t step, std::string_view key, double value,
              uint64_t timestamp_ns = 0U) noexcept -> Result<void>;

  /**
   * @brief Logs a histogram summary at (step, key, timestamp) to the WAL.
   * @param step Training step index.
   * @param key Histogram metric name.
   * @param hist Histogram summary statistics and bucket counts.
   * @param timestamp_ns Optional timestamp (0 uses monotonic_time_ns()).
   * @returns Result<void> or TrackingErrorCode::WalWriteFailure.
   */
  [[nodiscard]] auto
  log_histogram (uint64_t step, std::string_view key,
                 const HistogramSummary &hist,
                 uint64_t timestamp_ns = 0U) noexcept -> Result<void>;

  /**
   * @brief Logs an arbitrary binary blob at (step, key, timestamp), storing
   * payload in the content-addressable store and recording its reference in
   * the WAL.
   * @param step Step index.
   * @param key Blob key name.
   * @param payload Binary data span.
   * @param timestamp_ns Optional timestamp.
   * @returns Result<std::string> containing the 64-char BLAKE3 hex digest.
   */
  [[nodiscard]] auto
  log_blob (uint64_t step, std::string_view key,
            std::span<const std::byte> payload,
            uint64_t timestamp_ns = 0U) noexcept -> Result<std::string>;

  /**
   * @brief Stores an artifact in the content-addressable store (BLAKE3 hash)
   * and associates it with this run.
   * @param name Logical artifact name (e.g., "checkpoint.bin").
   * @param data Raw artifact payload bytes.
   * @returns Result<std::string> containing the 64-char BLAKE3 hex digest.
   */
  [[nodiscard]] auto
  log_artifact (std::string_view name,
                std::span<const std::byte> data) noexcept
      -> Result<std::string>;

  /**
   * @brief Flushes any buffered metric points to the WAL and marks the run
   * Completed.
   * @returns Result<void> or TrackingErrorCode::WalWriteFailure.
   */
  [[nodiscard]] auto finish () noexcept -> Result<void>;

private:
  friend class ExperimentStore;
  struct Impl;
  explicit Run (std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Content-addressed experiment tracking store managing runs, multi-
 * writer WAL segments, background/explicit columnar compaction, and fast
 * vectorized predicate queries.
 * @note Thread-safety: Safe for concurrent run creation, metric logging,
 * compaction, and querying across threads.
 */
class ExperimentStore
{
public:
  ExperimentStore () noexcept;
  ExperimentStore (const ExperimentStore &) = delete;
  auto operator= (const ExperimentStore &) -> ExperimentStore & = delete;
  ExperimentStore (ExperimentStore &&other) noexcept;
  auto operator= (ExperimentStore &&other) noexcept -> ExperimentStore &;
  ~ExperimentStore () noexcept;

  /**
   * @brief Opens or initializes an ExperimentStore at the given directory,
   * automatically replaying any uncompacted WAL segments and loading columnar
   * segments.
   * @param store_dir Directory path for WAL, columnar segments, and CAS blobs.
   * @returns Result<ExperimentStore> or TrackingError on corruption/IO error.
   */
  [[nodiscard]] static auto
  open (const std::filesystem::path &store_dir) noexcept
      -> Result<ExperimentStore>;

  /**
   * @brief Creates a fresh scoped Run in the tracking store.
   * @param custom_run_id Optional run identifier (auto-generated if empty).
   * @returns Result<Run> scoped object.
   */
  [[nodiscard]] auto
  create_run (std::string_view custom_run_id = "") noexcept -> Result<Run>;

  /**
   * @brief Merges active WAL segments into the binary columnar store format
   * with per-run zone maps for fast range queries without blocking writers.
   * @returns Result<std::size_t> number of metric records compacted.
   */
  [[nodiscard]] auto compact () noexcept -> Result<std::size_t>;

  /**
   * @brief Queries runs matching a predicate filter over parameters, metrics,
   * and tags using columnar zone maps.
   * @param filter Query predicate specification.
   * @returns Result<std::vector<RunRecord>> matching runs.
   */
  [[nodiscard]] auto query (const QueryFilter &filter) const noexcept
      -> Result<std::vector<RunRecord>>;

  /**
   * @brief Retrieves a specific run record by run_id.
   * @param run_id Target run identifier.
   * @returns Result<RunRecord> or TrackingErrorCode::RunNotFound.
   */
  [[nodiscard]] auto
  get_run (std::string_view run_id) const noexcept -> Result<RunRecord>;

  /**
   * @brief Retrieves an artifact's raw bytes from the content-addressable
   * store by its 64-character BLAKE3 hex hash.
   * @param blake3_hex 64-character hex digest.
   * @returns Result<std::vector<std::byte>> or TrackingError.
   */
  [[nodiscard]] auto
  load_artifact (std::string_view blake3_hex) const noexcept
      -> Result<std::vector<std::byte>>;

  /**
   * @brief Queries the total number of runs tracked in this store.
   */
  [[nodiscard]] auto run_count () const noexcept -> std::size_t;

  /**
   * @brief Queries the total number of metric points logged across all runs.
   */
  [[nodiscard]] auto total_metrics_logged () const noexcept -> uint64_t;

private:
  friend class Run;
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

namespace track
{
using yantra::v1::ExperimentStore;
using yantra::v1::HistogramSummary;
using yantra::v1::MetricPoint;
using yantra::v1::MetricSummary;
using yantra::v1::QueryFilter;
using yantra::v1::Run;
using yantra::v1::RunRecord;
using yantra::v1::RunState;
using yantra::v1::WalEngine;
using yantra::v1::WalFileHeader;
using yantra::v1::WalRecord;
using yantra::v1::WalValueKind;
} /* namespace track */

} /* namespace yantra::inline v1 */

#endif /* YANTRA_TRACK_HPP */
