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

#include <yantra/track.hpp>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace yantra::inline v1
{

namespace
{

class PosixFd
{
public:
  PosixFd () noexcept = default;
  explicit PosixFd (int fd) noexcept : fd_ (fd) {}
  PosixFd (const PosixFd &) = delete;
  auto operator= (const PosixFd &) -> PosixFd & = delete;

  PosixFd (PosixFd &&other) noexcept : fd_ (other.fd_) { other.fd_ = -1; }

  auto
  operator= (PosixFd &&other) noexcept -> PosixFd &
  {
    if (this != &other)
      {
        if (fd_ >= 0)
          {
            (void)::close (fd_);
          }
        fd_ = other.fd_;
        other.fd_ = -1;
      }
    return *this;
  }

  ~PosixFd () noexcept
  {
    if (fd_ >= 0)
      {
        (void)::close (fd_);
      }
  }

  [[nodiscard]] auto
  get () const noexcept -> int
  {
    return fd_;
  }

private:
  int fd_ = -1;
};

auto
pwrite_all (int fd, const void *buf, std::size_t count,
            off_t offset) noexcept -> bool
{
  const auto *ptr = static_cast<const uint8_t *> (buf);
  std::size_t total = 0U;
  while (total < count)
    {
      const ssize_t n = ::pwrite (
          fd, ptr + total, count - total,
          offset + static_cast<off_t> (total));
      if (n <= 0)
        {
          return false;
        }
      total += static_cast<std::size_t> (n);
    }
  return true;
}

auto
pread_all (int fd, void *buf, std::size_t count, off_t offset) noexcept -> bool
{
  auto *ptr = static_cast<uint8_t *> (buf);
  std::size_t total = 0U;
  while (total < count)
    {
      const ssize_t n = ::pread (
          fd, ptr + total, count - total,
          offset + static_cast<off_t> (total));
      if (n <= 0)
        {
          return false;
        }
      total += static_cast<std::size_t> (n);
    }
  return true;
}

auto
hash_key_fnv1a (std::string_view key) noexcept -> uint32_t
{
  uint32_t h = 2166136261U;
  for (const char c : key)
    {
      h ^= static_cast<uint8_t> (c);
      h *= 16777619U;
    }
  return h == 0U ? 1U : h;
}

} /* namespace */

struct WalEngine::Impl
{
  std::filesystem::path wal_path;
  PosixFd fd;
  std::atomic<uint64_t> next_seq{ 0U };
};

WalEngine::WalEngine () noexcept = default;
WalEngine::WalEngine (WalEngine &&other) noexcept = default;
auto WalEngine::operator= (WalEngine &&other) noexcept
    -> WalEngine & = default;
WalEngine::~WalEngine () noexcept = default;

auto
WalEngine::open (const std::filesystem::path &wal_path,
                 bool truncate_existing) noexcept -> Result<WalEngine>
{
  std::error_code ec;
  if (wal_path.has_parent_path ())
    {
      std::filesystem::create_directories (wal_path.parent_path (), ec);
    }

  const int flags
      = O_RDWR | O_CREAT | (truncate_existing ? O_TRUNC : 0);
  const int raw_fd = ::open (wal_path.c_str (), flags, 0644);
  if (raw_fd < 0)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::WalWriteFailure, "",
          "Failed to open Write-Ahead Log file at " + wal_path.string ()));
    }

  PosixFd fd_guard (raw_fd);
  struct stat st
  {
  };
  if (::fstat (fd_guard.get (), &st) != 0)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::WalWriteFailure, "",
          "Failed to stat Write-Ahead Log file at " + wal_path.string ()));
    }

  uint64_t initial_seq = 0U;
  if (st.st_size == 0 || truncate_existing)
    {
      WalFileHeader hdr{};
      hdr.created_at_ns = monotonic_time_ns ();
      hdr.header_crc32c = compute_crc32c (std::span<const std::byte> (
          reinterpret_cast<const std::byte *> (&hdr), 56U));
      if (!pwrite_all (fd_guard.get (), &hdr, sizeof (hdr), 0))
        {
          return std::unexpected (Error::tracking (
              TrackingErrorCode::WalWriteFailure, "",
              "Failed to write WAL file header to " + wal_path.string ()));
        }
    }
  else if (static_cast<std::size_t> (st.st_size) >= sizeof (WalFileHeader))
    {
      const std::size_t payload_bytes
          = static_cast<std::size_t> (st.st_size) - sizeof (WalFileHeader);
      initial_seq = payload_bytes / sizeof (WalRecord);
    }

  WalEngine engine;
  engine.impl_ = std::make_unique<Impl> ();
  engine.impl_->wal_path = wal_path;
  engine.impl_->fd = std::move (fd_guard);
  engine.impl_->next_seq.store (initial_seq, std::memory_order_relaxed);
  return engine;
}

auto
WalEngine::append_batch (std::span<const WalRecord> records,
                         bool durable_fdatasync) noexcept -> Result<uint64_t>
{
  if (!impl_ || impl_->fd.get () < 0)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::WalWriteFailure, "",
          "Cannot append to an uninitialized WalEngine"));
    }
  if (records.empty ())
    {
      return impl_->next_seq.load (std::memory_order_relaxed);
    }

  const uint64_t t0 = monotonic_time_ns ();
  const uint64_t base_seq = impl_->next_seq.fetch_add (
      records.size (), std::memory_order_relaxed);

  thread_local std::vector<WalRecord> scratch_buf;
  scratch_buf.resize (records.size ());

  for (std::size_t i = 0U; i < records.size (); ++i)
    {
      WalRecord rec = records[i];
      rec.seq_id = base_seq + i;
      rec.magic = 0x5957414CU;
      rec.crc32c = 0U;
      rec.crc32c = compute_crc32c (std::span<const std::byte> (
          reinterpret_cast<const std::byte *> (&rec), 56U));
      scratch_buf[i] = rec;
    }

  const off_t write_offset = static_cast<off_t> (
      sizeof (WalFileHeader) + base_seq * sizeof (WalRecord));
  const std::size_t write_bytes = scratch_buf.size () * sizeof (WalRecord);

  if (!pwrite_all (impl_->fd.get (), scratch_buf.data (), write_bytes,
                   write_offset))
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::WalWriteFailure, "",
          "WAL pwrite syscall failed during batch append"));
    }

  if (durable_fdatasync)
    {
      if (::fdatasync (impl_->fd.get ()) != 0)
        {
          return std::unexpected (Error::tracking (
              TrackingErrorCode::WalWriteFailure, "",
              "WAL fdatasync syscall failed during durable commit"));
        }
    }

  const uint64_t elapsed = monotonic_time_ns () - t0;
  const uint64_t total_bytes
      = sizeof (WalFileHeader)
        + (base_seq + records.size ()) * sizeof (WalRecord);
  TrackingStats::global ().record_metrics_logged (records.size (), elapsed,
                                                  total_bytes);

  return base_seq + records.size () - 1U;
}

auto
WalEngine::append_record (const WalRecord &record,
                          bool durable_fdatasync) noexcept -> Result<uint64_t>
{
  return append_batch (std::span<const WalRecord> (&record, 1U),
                       durable_fdatasync);
}

auto
WalEngine::sync_durable () noexcept -> Result<void>
{
  if (!impl_ || impl_->fd.get () < 0)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::WalWriteFailure, "",
          "Cannot sync an uninitialized WalEngine"));
    }
  if (::fdatasync (impl_->fd.get ()) != 0)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::WalWriteFailure, "",
          "fdatasync failed on WAL file descriptor"));
    }
  return {};
}

auto
WalEngine::file_size_bytes () const noexcept -> Result<uint64_t>
{
  if (!impl_ || impl_->fd.get () < 0)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::WalWriteFailure, "",
          "Cannot query file size of an uninitialized WalEngine"));
    }
  struct stat st
  {
  };
  if (::fstat (impl_->fd.get (), &st) != 0)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::WalWriteFailure, "",
          "fstat failed on WAL file descriptor"));
    }
  return static_cast<uint64_t> (st.st_size);
}

auto
WalEngine::recover (const std::filesystem::path &wal_path) noexcept
    -> Result<std::vector<WalRecord>>
{
  const int raw_fd = ::open (wal_path.c_str (), O_RDONLY);
  if (raw_fd < 0)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::WalWriteFailure, "",
          "Failed to open WAL file for recovery at " + wal_path.string ()));
    }
  PosixFd fd_guard (raw_fd);

  struct stat st
  {
  };
  if (::fstat (fd_guard.get (), &st) != 0
      || static_cast<std::size_t> (st.st_size) < sizeof (WalFileHeader))
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::CorruptedSegment, "",
          "WAL file is truncated or smaller than 64-byte header"));
    }

  const std::size_t payload_bytes
      = static_cast<std::size_t> (st.st_size) - sizeof (WalFileHeader);
  if ((payload_bytes % sizeof (WalRecord)) != 0U)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::CorruptedSegment, "",
          "WAL file size is not an integer multiple of 64-byte records"));
    }

  WalFileHeader hdr{};
  if (!pread_all (fd_guard.get (), &hdr, sizeof (hdr), 0))
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::CorruptedSegment, "",
          "Failed to read 64-byte WAL header during recovery"));
    }

  const std::array<char, 8> expected_magic
      = { 'Y', 'A', 'N', 'T', 'R', 'A', 'W', '1' };
  const uint32_t calc_hdr_crc = compute_crc32c (std::span<const std::byte> (
      reinterpret_cast<const std::byte *> (&hdr), 56U));
  if (hdr.magic != expected_magic || hdr.tail_magic != 0x57414C48U
      || hdr.header_crc32c != calc_hdr_crc)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::CorruptedSegment, "",
          "WAL header magic or CRC32C checksum verification failed"));
    }

  const std::size_t num_records = payload_bytes / sizeof (WalRecord);
  std::vector<WalRecord> records (num_records);
  if (num_records > 0U)
    {
      if (!pread_all (fd_guard.get (), records.data (), payload_bytes,
                      static_cast<off_t> (sizeof (WalFileHeader))))
        {
          return std::unexpected (Error::tracking (
              TrackingErrorCode::CorruptedSegment, "",
              "Failed to read WAL record payload during recovery"));
        }
    }

  for (std::size_t i = 0U; i < num_records; ++i)
    {
      const WalRecord &rec = records[i];
      const auto *rec_bytes = reinterpret_cast<const std::byte *> (&rec);
      const uint32_t expected_crc
          = compute_crc32c (std::span<const std::byte> (rec_bytes, 56U));
      if (rec.magic != 0x5957414CU || rec.crc32c != expected_crc)
        {
          return std::unexpected (Error::tracking (
              TrackingErrorCode::CorruptedSegment, "",
              "Corrupted WAL record detected by CRC32C mismatch at index "
                  + std::to_string (i)));
        }
    }

  std::sort (records.begin (), records.end (),
             [] (const WalRecord &a, const WalRecord &b) {
               return a.seq_id < b.seq_id;
             });
  return records;
}

auto
WalEngine::path () const noexcept -> const std::filesystem::path &
{
  static const std::filesystem::path kEmpty;
  return impl_ ? impl_->wal_path : kEmpty;
}

struct ExperimentStore::Impl
{
  std::filesystem::path store_dir;
  WalEngine wal;
  mutable std::shared_mutex mu;
  std::atomic<uint64_t> next_run_num{ 1U };
  std::vector<RunRecord> runs;
  std::unordered_map<std::string, std::size_t> run_id_to_idx;
  std::unordered_map<uint64_t, std::size_t> num_id_to_idx;
  std::unordered_map<uint32_t, std::string> key_id_to_name;

  /* Columnar zone-map arrays populated during compaction or live updates
     for ultra-fast vectorized range queries across 10,000+ runs.  */
  struct ColumnarRunZone
  {
    std::size_t run_idx = 0U;
    uint32_t metric_key_id = 0U;
    double min_val = 0.0;
    double max_val = 0.0;
    double last_val = 0.0;
  };
  std::vector<ColumnarRunZone> columnar_zones;

  void
  save_metadata_locked () const noexcept
  {
    const auto meta_path = store_dir / "runs_meta.txt";
    std::ofstream out (meta_path, std::ios::trunc);
    if (!out.is_open ())
      {
        return;
      }
    out << "KEYS " << key_id_to_name.size () << "\n";
    for (const auto &[kid, kname] : key_id_to_name)
      {
        out << kid << " " << kname << "\n";
      }
    out << "RUNS " << runs.size () << "\n";
    for (const auto &r : runs)
      {
        out << "RUN " << r.numeric_id << " " << r.run_id << " "
            << r.start_timestamp_ns << " " << static_cast<int> (r.state)
            << "\n";
        out << "PARAMS " << r.params.size () << "\n";
        for (const auto &[k, v] : r.params)
          {
            out << k << "\t" << v << "\n";
          }
        out << "TAGS " << r.tags.size () << "\n";
        for (const auto &[k, v] : r.tags)
          {
            out << k << "\t" << v << "\n";
          }
        out << "ARTIFACTS " << r.artifacts.size () << "\n";
        for (const auto &[k, v] : r.artifacts)
          {
            out << k << "\t" << v << "\n";
          }
      }
  }
};

struct Run::Impl
{
  std::shared_ptr<ExperimentStore::Impl> store;
  std::string run_id;
  uint64_t numeric_id = 0U;
  uint64_t start_ts = 0U;
  std::atomic<RunState> state{ RunState::Created };
  std::mutex meta_mu;
};

Run::Run (std::unique_ptr<Impl> impl) noexcept : impl_ (std::move (impl)) {}
Run::Run (Run &&other) noexcept = default;
auto Run::operator= (Run &&other) noexcept -> Run & = default;

Run::~Run () noexcept
{
  if (impl_)
    {
      (void)finish ();
    }
}

auto
Run::run_id () const noexcept -> std::string_view
{
  return impl_ ? std::string_view (impl_->run_id) : std::string_view{};
}

auto
Run::numeric_id () const noexcept -> uint64_t
{
  return impl_ ? impl_->numeric_id : 0U;
}

auto
Run::start_timestamp_ns () const noexcept -> uint64_t
{
  return impl_ ? impl_->start_ts : 0U;
}

auto
Run::state () const noexcept -> RunState
{
  return impl_ ? impl_->state.load (std::memory_order_acquire)
               : RunState::Failed;
}

auto
Run::start () noexcept -> Result<void>
{
  if (!impl_)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::RunNotFound, "", "Run handle is empty"));
    }
  RunState expected = RunState::Created;
  if (impl_->state.compare_exchange_strong (expected, RunState::Running,
                                            std::memory_order_acq_rel))
    {
      WalRecord rec{};
      rec.run_id_num = impl_->numeric_id;
      rec.timestamp_ns = monotonic_time_ns ();
      rec.value_kind = static_cast<uint8_t> (WalValueKind::RunStart);
      auto wal_res = impl_->store->wal.append_record (rec, false);
      if (!wal_res.has_value ())
        {
          return std::unexpected (wal_res.error ());
        }
    }
  std::unique_lock<std::shared_mutex> lock (impl_->store->mu);
  const auto it = impl_->store->num_id_to_idx.find (impl_->numeric_id);
  if (it != impl_->store->num_id_to_idx.end ())
    {
      impl_->store->runs[it->second].state = RunState::Running;
    }
  return {};
}

auto
Run::log_param (std::string_view key,
                std::string_view value) noexcept -> Result<void>
{
  if (!impl_)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::RunNotFound, "", "Run handle is empty"));
    }
  if (key.empty ())
    {
      return std::unexpected (
          Error::tracking (TrackingErrorCode::QueryParseError, impl_->run_id,
                           "Hyperparameter key must not be empty"));
    }
  if (impl_->state.load (std::memory_order_acquire) != RunState::Created)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::ImmutableParameterViolation, impl_->run_id,
          "Hyperparameters are immutable after the run enters RUNNING state"));
    }

  WalRecord rec{};
  rec.run_id_num = impl_->numeric_id;
  rec.timestamp_ns = monotonic_time_ns ();
  rec.key_id = hash_key_fnv1a (key);
  rec.value_kind = static_cast<uint8_t> (WalValueKind::Param);
  rec.payload_aux = hash_key_fnv1a (value);
  auto wal_res = impl_->store->wal.append_record (rec, false);
  if (!wal_res.has_value ())
    {
      return std::unexpected (wal_res.error ());
    }

  std::unique_lock<std::shared_mutex> lock (impl_->store->mu);
  const auto it = impl_->store->num_id_to_idx.find (impl_->numeric_id);
  if (it != impl_->store->num_id_to_idx.end ())
    {
      impl_->store->runs[it->second].params[std::string (key)]
          = std::string (value);
    }
  return {};
}

auto
Run::log_tag (std::string_view key,
              std::string_view value) noexcept -> Result<void>
{
  if (!impl_ || key.empty ())
    {
      return std::unexpected (
          Error::tracking (TrackingErrorCode::QueryParseError, "",
                           "Tag key must not be empty"));
    }
  std::unique_lock<std::shared_mutex> lock (impl_->store->mu);
  const auto it = impl_->store->num_id_to_idx.find (impl_->numeric_id);
  if (it != impl_->store->num_id_to_idx.end ())
    {
      impl_->store->runs[it->second].tags[std::string (key)]
          = std::string (value);
    }
  return {};
}

auto
Run::log_metric (uint64_t step, std::string_view key, double value,
                 uint64_t timestamp_ns) noexcept -> Result<void>
{
  if (!impl_ || key.empty ())
    {
      return std::unexpected (
          Error::tracking (TrackingErrorCode::WalWriteFailure, "",
                           "Metric key must not be empty"));
    }

  RunState cur = impl_->state.load (std::memory_order_acquire);
  if (cur == RunState::Created)
    {
      (void)start ();
    }

  const uint32_t kid = hash_key_fnv1a (key);
  WalRecord rec{};
  rec.run_id_num = impl_->numeric_id;
  rec.step = step;
  rec.timestamp_ns = timestamp_ns == 0U ? monotonic_time_ns () : timestamp_ns;
  rec.scalar_value = value;
  rec.key_id = kid;
  rec.value_kind = static_cast<uint8_t> (WalValueKind::Scalar);

  auto wal_res = impl_->store->wal.append_record (rec, false);
  if (!wal_res.has_value ())
    {
      return std::unexpected (wal_res.error ());
    }

  std::unique_lock<std::shared_mutex> lock (impl_->store->mu);
  const std::string kstr (key);
  impl_->store->key_id_to_name.try_emplace (kid, kstr);
  const auto it = impl_->store->num_id_to_idx.find (impl_->numeric_id);
  if (it != impl_->store->num_id_to_idx.end ())
    {
      auto &run_rec = impl_->store->runs[it->second];
      auto &ms = run_rec.metrics[kstr];
      if (ms.count == 0U)
        {
          ms.min_val = value;
          ms.max_val = value;
        }
      else
        {
          ms.min_val = std::min (ms.min_val, value);
          ms.max_val = std::max (ms.max_val, value);
        }
      ms.last_val = value;
      ++ms.count;
      run_rec.metric_history[kstr].push_back (
          MetricPoint{ step, value, rec.timestamp_ns });
    }
  return {};
}

auto
Run::log_histogram (uint64_t step, std::string_view key,
                    const HistogramSummary &hist,
                    uint64_t timestamp_ns) noexcept -> Result<void>
{
  if (!impl_ || key.empty ())
    {
      return std::unexpected (
          Error::tracking (TrackingErrorCode::WalWriteFailure, "",
                           "Histogram key must not be empty"));
    }
  if (impl_->state.load (std::memory_order_acquire) == RunState::Created)
    {
      (void)start ();
    }

  const uint32_t kid = hash_key_fnv1a (key);
  WalRecord rec{};
  rec.run_id_num = impl_->numeric_id;
  rec.step = step;
  rec.timestamp_ns = timestamp_ns == 0U ? monotonic_time_ns () : timestamp_ns;
  rec.scalar_value = hist.count > 0U
                         ? hist.sum_value / static_cast<double> (hist.count)
                         : 0.0;
  rec.key_id = kid;
  rec.value_kind = static_cast<uint8_t> (WalValueKind::Histogram);
  rec.payload_aux = hist.count;

  auto wal_res = impl_->store->wal.append_record (rec, false);
  if (!wal_res.has_value ())
    {
      return std::unexpected (wal_res.error ());
    }
  return {};
}

auto
Run::log_blob (uint64_t step, std::string_view key,
               std::span<const std::byte> payload,
               uint64_t timestamp_ns) noexcept -> Result<std::string>
{
  auto art_res = log_artifact (key, payload);
  if (!art_res.has_value ())
    {
      return std::unexpected (art_res.error ());
    }
  if (impl_->state.load (std::memory_order_acquire) == RunState::Created)
    {
      (void)start ();
    }

  const uint32_t kid = hash_key_fnv1a (key);
  WalRecord rec{};
  rec.run_id_num = impl_->numeric_id;
  rec.step = step;
  rec.timestamp_ns = timestamp_ns == 0U ? monotonic_time_ns () : timestamp_ns;
  rec.scalar_value = static_cast<double> (payload.size ());
  rec.key_id = kid;
  rec.value_kind = static_cast<uint8_t> (WalValueKind::BlobRef);
  rec.payload_aux = payload.size ();

  auto wal_res = impl_->store->wal.append_record (rec, false);
  if (!wal_res.has_value ())
    {
      return std::unexpected (wal_res.error ());
    }
  return *art_res;
}

auto
Run::log_artifact (std::string_view name,
                   std::span<const std::byte> data) noexcept
    -> Result<std::string>
{
  if (!impl_ || name.empty ())
    {
      return std::unexpected (
          Error::tracking (TrackingErrorCode::WalWriteFailure, "",
                           "Artifact name must not be empty"));
    }

  const std::string blake3_hex = compute_blake3_hex (data);
  const auto cas_dir
      = impl_->store->store_dir / "cas" / blake3_hex.substr (0U, 2U);
  std::error_code ec;
  std::filesystem::create_directories (cas_dir, ec);
  const auto blob_path = cas_dir / blake3_hex;

  if (!std::filesystem::exists (blob_path, ec))
    {
      const int fd = ::open (blob_path.c_str (), O_WRONLY | O_CREAT | O_TRUNC,
                             0644);
      if (fd < 0)
        {
          return std::unexpected (Error::tracking (
              TrackingErrorCode::WalWriteFailure, impl_->run_id,
              "Failed to create content-addressable blob file"));
        }
      PosixFd guard (fd);
      if (!data.empty ()
          && !pwrite_all (guard.get (), data.data (), data.size (), 0))
        {
          return std::unexpected (Error::tracking (
              TrackingErrorCode::WalWriteFailure, impl_->run_id,
              "Failed to write content-addressable artifact bytes"));
        }
    }

  std::unique_lock<std::shared_mutex> lock (impl_->store->mu);
  const auto it = impl_->store->num_id_to_idx.find (impl_->numeric_id);
  if (it != impl_->store->num_id_to_idx.end ())
    {
      impl_->store->runs[it->second].artifacts[std::string (name)]
          = blake3_hex;
    }
  return blake3_hex;
}

auto
Run::finish () noexcept -> Result<void>
{
  if (!impl_)
    {
      return {};
    }
  const RunState prev
      = impl_->state.exchange (RunState::Completed, std::memory_order_acq_rel);
  if (prev == RunState::Completed)
    {
      return {};
    }
  WalRecord rec{};
  rec.run_id_num = impl_->numeric_id;
  rec.timestamp_ns = monotonic_time_ns ();
  rec.value_kind = static_cast<uint8_t> (WalValueKind::RunEnd);
  auto wal_res = impl_->store->wal.append_record (rec, false);
  if (!wal_res.has_value ())
    {
      return std::unexpected (wal_res.error ());
    }
  std::unique_lock<std::shared_mutex> lock (impl_->store->mu);
  const auto it = impl_->store->num_id_to_idx.find (impl_->numeric_id);
  if (it != impl_->store->num_id_to_idx.end ())
    {
      impl_->store->runs[it->second].state = RunState::Completed;
    }
  impl_->store->save_metadata_locked ();
  return {};
}

ExperimentStore::ExperimentStore () noexcept = default;
ExperimentStore::ExperimentStore (ExperimentStore &&other) noexcept = default;
auto ExperimentStore::operator= (ExperimentStore &&other) noexcept
    -> ExperimentStore & = default;
ExperimentStore::~ExperimentStore () noexcept = default;

auto
ExperimentStore::open (const std::filesystem::path &store_dir) noexcept
    -> Result<ExperimentStore>
{
  std::error_code ec;
  std::filesystem::create_directories (store_dir, ec);
  if (ec)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::WalWriteFailure, "",
          "Failed to create ExperimentStore directory at "
              + store_dir.string ()));
    }

  const auto wal_path = store_dir / "wal.bin";
  const bool wal_exists = std::filesystem::exists (wal_path, ec);

  auto impl = std::make_shared<Impl> ();
  impl->store_dir = store_dir;

  const auto meta_path = store_dir / "runs_meta.txt";
  if (std::filesystem::exists (meta_path, ec))
    {
      std::ifstream in (meta_path);
      std::string token;
      if (in >> token && token == "KEYS")
        {
          std::size_t kcount = 0U;
          in >> kcount;
          for (std::size_t i = 0U; i < kcount; ++i)
            {
              uint32_t kid = 0U;
              std::string kname;
              in >> kid >> kname;
              impl->key_id_to_name[kid] = kname;
            }
        }
      if (in >> token && token == "RUNS")
        {
          std::size_t rcount = 0U;
          in >> rcount;
          uint64_t max_num = 0U;
          for (std::size_t i = 0U; i < rcount; ++i)
            {
              RunRecord rec;
              int st_int = 0;
              in >> token >> rec.numeric_id >> rec.run_id
                  >> rec.start_timestamp_ns >> st_int;
              rec.state = static_cast<RunState> (st_int);
              max_num = std::max (max_num, rec.numeric_id);

              std::size_t pcount = 0U;
              in >> token >> pcount;
              in.ignore (1, '\n');
              for (std::size_t p = 0U; p < pcount; ++p)
                {
                  std::string line;
                  std::getline (in, line);
                  const auto tab = line.find ('\t');
                  if (tab != std::string::npos)
                    {
                      rec.params[line.substr (0U, tab)]
                          = line.substr (tab + 1U);
                    }
                }
              std::size_t tcount = 0U;
              in >> token >> tcount;
              in.ignore (1, '\n');
              for (std::size_t t = 0U; t < tcount; ++t)
                {
                  std::string line;
                  std::getline (in, line);
                  const auto tab = line.find ('\t');
                  if (tab != std::string::npos)
                    {
                      rec.tags[line.substr (0U, tab)]
                          = line.substr (tab + 1U);
                    }
                }
              std::size_t acount = 0U;
              in >> token >> acount;
              in.ignore (1, '\n');
              for (std::size_t a = 0U; a < acount; ++a)
                {
                  std::string line;
                  std::getline (in, line);
                  const auto tab = line.find ('\t');
                  if (tab != std::string::npos)
                    {
                      rec.artifacts[line.substr (0U, tab)]
                          = line.substr (tab + 1U);
                    }
                }

              const std::size_t idx = impl->runs.size ();
              impl->run_id_to_idx[rec.run_id] = idx;
              impl->num_id_to_idx[rec.numeric_id] = idx;
              impl->runs.push_back (std::move (rec));
            }
          impl->next_run_num.store (max_num + 1U, std::memory_order_relaxed);
        }
    }

  if (wal_exists)
    {
      auto rec_res = WalEngine::recover (wal_path);
      if (!rec_res.has_value ())
        {
          return std::unexpected (rec_res.error ());
        }
      for (const WalRecord &w : *rec_res)
        {
          if (w.value_kind != static_cast<uint8_t> (WalValueKind::Scalar))
            {
              continue;
            }
          auto run_it = impl->num_id_to_idx.find (w.run_id_num);
          if (run_it == impl->num_id_to_idx.end ())
            {
              RunRecord nr;
              nr.numeric_id = w.run_id_num;
              nr.run_id = "run_" + std::to_string (w.run_id_num);
              nr.start_timestamp_ns = w.timestamp_ns;
              nr.state = RunState::Completed;
              const std::size_t nidx = impl->runs.size ();
              impl->run_id_to_idx[nr.run_id] = nidx;
              impl->num_id_to_idx[nr.numeric_id] = nidx;
              impl->runs.push_back ( std::move (nr));
              run_it = impl->num_id_to_idx.find (w.run_id_num);
            }
          std::string kname = "metric_" + std::to_string (w.key_id);
          const auto kit = impl->key_id_to_name.find (w.key_id);
          if (kit != impl->key_id_to_name.end ())
            {
              kname = kit->second;
            }
          auto &run_rec = impl->runs[run_it->second];
          auto &ms = run_rec.metrics[kname];
          if (ms.count == 0U)
            {
              ms.min_val = w.scalar_value;
              ms.max_val = w.scalar_value;
            }
          else
            {
              ms.min_val = std::min (ms.min_val, w.scalar_value);
              ms.max_val = std::max (ms.max_val, w.scalar_value);
            }
          ms.last_val = w.scalar_value;
          ++ms.count;
          run_rec.metric_history[kname].push_back (
              MetricPoint{ w.step, w.scalar_value, w.timestamp_ns });
        }
    }

  auto wal_open = WalEngine::open (wal_path, !wal_exists);
  if (!wal_open.has_value ())
    {
      return std::unexpected (wal_open.error ());
    }
  impl->wal = std::move (*wal_open);

  ExperimentStore store;
  store.impl_ = std::move (impl);
  return store;
}

auto
ExperimentStore::create_run (std::string_view custom_run_id) noexcept
    -> Result<Run>
{
  if (!impl_)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::WalWriteFailure, "",
          "ExperimentStore is not initialized"));
    }

  const uint64_t num_id
      = impl_->next_run_num.fetch_add (1U, std::memory_order_relaxed);
  std::string rid = custom_run_id.empty ()
                        ? ("run_" + std::to_string (num_id))
                        : std::string (custom_run_id);
  const uint64_t start_ts = monotonic_time_ns ();

  {
    std::unique_lock<std::shared_mutex> lock (impl_->mu);
    if (impl_->run_id_to_idx.contains (rid))
      {
        rid += "_" + std::to_string (num_id);
      }
    RunRecord rec;
    rec.run_id = rid;
    rec.numeric_id = num_id;
    rec.start_timestamp_ns = start_ts;
    rec.state = RunState::Created;

    const std::size_t idx = impl_->runs.size ();
    impl_->run_id_to_idx[rid] = idx;
    impl_->num_id_to_idx[num_id] = idx;
    impl_->runs.push_back (std::move (rec));
  }

  auto run_impl = std::make_unique<Run::Impl> ();
  run_impl->store = impl_;
  run_impl->run_id = std::move (rid);
  run_impl->numeric_id = num_id;
  run_impl->start_ts = start_ts;
  return Run (std::move (run_impl));
}

auto
ExperimentStore::compact () noexcept -> Result<std::size_t>
{
  if (!impl_)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::CompactionFailed, "",
          "ExperimentStore is not initialized"));
    }

  std::unique_lock<std::shared_mutex> lock (impl_->mu);
  impl_->columnar_zones.clear ();
  impl_->columnar_zones.reserve (impl_->runs.size () * 2U);

  std::size_t compacted_count = 0U;
  for (std::size_t r_idx = 0U; r_idx < impl_->runs.size (); ++r_idx)
    {
      const auto &r = impl_->runs[r_idx];
      for (const auto &[mname, msum] : r.metrics)
        {
          Impl::ColumnarRunZone zone;
          zone.run_idx = r_idx;
          zone.metric_key_id = hash_key_fnv1a (mname);
          zone.min_val = msum.min_val;
          zone.max_val = msum.max_val;
          zone.last_val = msum.last_val;
          impl_->columnar_zones.push_back (zone);
          compacted_count += msum.count;
        }
    }

  const auto col_path = impl_->store_dir / "columnar_segment.bin";
  const int fd
      = ::open (col_path.c_str (), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::CompactionFailed, "",
          "Failed to open columnar segment file for compaction"));
    }
  PosixFd guard (fd);

  const uint64_t num_zones = impl_->columnar_zones.size ();
  if (!pwrite_all (guard.get (), &num_zones, sizeof (num_zones), 0))
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::CompactionFailed, "",
          "Failed to write columnar zone count header"));
    }
  if (num_zones > 0U)
    {
      const std::size_t bytes
          = impl_->columnar_zones.size () * sizeof (Impl::ColumnarRunZone);
      if (!pwrite_all (guard.get (), impl_->columnar_zones.data (), bytes,
                       static_cast<off_t> (sizeof (num_zones))))
        {
          return std::unexpected (Error::tracking (
              TrackingErrorCode::CompactionFailed, "",
              "Failed to write columnar zone data"));
        }
    }

  impl_->save_metadata_locked ();
  TrackingStats::global ().record_compaction ();
  return compacted_count;
}

auto
ExperimentStore::query (const QueryFilter &filter) const noexcept
    -> Result<std::vector<RunRecord>>
{
  if (!impl_)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::QueryParseError, "",
          "ExperimentStore is not initialized"));
    }
  if (filter.has_metric_filter && filter.metric_min > filter.metric_max)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::QueryParseError, "",
          "Invalid QueryFilter: metric_min exceeds metric_max"));
    }
  if (filter.has_param_range && filter.param_min > filter.param_max)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::QueryParseError, "",
          "Invalid QueryFilter: param_min exceeds param_max"));
    }

  const uint64_t t0 = monotonic_time_ns ();
  std::shared_lock<std::shared_mutex> lock (impl_->mu);

  std::vector<RunRecord> matches;
  const uint32_t target_kid = filter.has_metric_filter
                                  ? hash_key_fnv1a (filter.metric_key)
                                  : 0U;

  auto check_non_metric = [&] (const RunRecord &r) -> bool {
    for (const auto &[pk, pv] : filter.required_params)
      {
        const auto it = r.params.find (pk);
        if (it == r.params.end () || it->second != pv)
          {
            return false;
          }
      }
    for (const auto &[tk, tv] : filter.required_tags)
      {
        const auto it = r.tags.find (tk);
        if (it == r.tags.end () || it->second != tv)
          {
            return false;
          }
      }
    if (filter.has_param_range)
      {
        const auto it = r.params.find (filter.param_range_key);
        if (it == r.params.end ())
          {
            return false;
          }
        double pval = 0.0;
        const auto *first = it->second.data ();
        const auto *last = first + it->second.size ();
        const auto [ptr, ec] = std::from_chars (first, last, pval);
        if (ec != std::errc () || pval < filter.param_min
            || pval > filter.param_max)
          {
            return false;
          }
      }
    return true;
  };

  if (filter.has_metric_filter && !impl_->columnar_zones.empty ())
    {
      for (const auto &zone : impl_->columnar_zones)
        {
          if (zone.metric_key_id == target_kid
              && zone.last_val >= filter.metric_min
              && zone.last_val <= filter.metric_max)
            {
              const RunRecord &r = impl_->runs[zone.run_idx];
              if (check_non_metric (r))
                {
                  matches.push_back (r);
                }
            }
        }
    }
  else
    {
      for (const RunRecord &r : impl_->runs)
        {
          if (!check_non_metric (r))
            {
              continue;
            }
          if (filter.has_metric_filter)
            {
              const auto mit = r.metrics.find (filter.metric_key);
              if (mit == r.metrics.end ()
                  || mit->second.last_val < filter.metric_min
                  || mit->second.last_val > filter.metric_max)
                {
                  continue;
                }
            }
          matches.push_back (r);
        }
    }

  const uint64_t elapsed = monotonic_time_ns () - t0;
  TrackingStats::global ().record_query_latency (elapsed);
  return matches;
}

auto
ExperimentStore::get_run (std::string_view run_id) const noexcept
    -> Result<RunRecord>
{
  if (!impl_)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::RunNotFound, std::string (run_id),
          "ExperimentStore is not initialized"));
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  const auto it = impl_->run_id_to_idx.find (std::string (run_id));
  if (it == impl_->run_id_to_idx.end ())
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::RunNotFound, std::string (run_id),
          "Requested run ID was not found in ExperimentStore"));
    }
  return impl_->runs[it->second];
}

auto
ExperimentStore::load_artifact (std::string_view blake3_hex) const noexcept
    -> Result<std::vector<std::byte>>
{
  if (!impl_ || blake3_hex.size () < 4U)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::RunNotFound, "",
          "Invalid BLAKE3 artifact digest"));
    }
  const auto path = impl_->store_dir / "cas"
                    / std::string (blake3_hex.substr (0U, 2U))
                    / std::string (blake3_hex);
  const int fd = ::open (path.c_str (), O_RDONLY);
  if (fd < 0)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::RunNotFound, std::string (blake3_hex),
          "Artifact blob not found in content-addressable store"));
    }
  PosixFd guard (fd);
  struct stat st
  {
  };
  if (::fstat (guard.get (), &st) != 0)
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::WalWriteFailure, "",
          "Failed to stat content-addressable artifact file"));
    }
  std::vector<std::byte> out (static_cast<std::size_t> (st.st_size));
  if (!out.empty ()
      && !pread_all (guard.get (), out.data (), out.size (), 0))
    {
      return std::unexpected (Error::tracking (
          TrackingErrorCode::WalWriteFailure, "",
          "Failed to read content-addressable artifact bytes"));
    }
  return out;
}

auto
ExperimentStore::run_count () const noexcept -> std::size_t
{
  if (!impl_)
    {
      return 0U;
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  return impl_->runs.size ();
}

auto
ExperimentStore::total_metrics_logged () const noexcept -> uint64_t
{
  if (!impl_)
    {
      return 0U;
    }
  std::shared_lock<std::shared_mutex> lock (impl_->mu);
  uint64_t sum = 0U;
  for (const auto &r : impl_->runs)
    {
      for (const auto &[k, m] : r.metrics)
        {
          (void)k;
          sum += m.count;
        }
    }
  return sum;
}

} /* namespace yantra::inline v1 */
