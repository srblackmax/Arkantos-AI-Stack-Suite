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

#include <sutra/gpu.hpp>
#include <sutra/memory.hpp>
#include <sutra/observability.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <mutex>
#include <numeric>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

namespace sutra::inline v1
{

namespace
{

constexpr std::size_t kSimdWidthAvx512 = 16U;
constexpr std::size_t kSimdWidthAvx2 = 8U;
constexpr std::size_t kHalfSimdWidthAvx2 = 4U;
constexpr std::size_t kAlignAvx512Bytes = 64U;
constexpr std::size_t kAlignAvx2Bytes = 32U;
constexpr uint32_t kLtmFileMagic = 0x53554C54U; // "SULT"
constexpr uint32_t kLtmFileVersion = 1U;
constexpr std::size_t kMaxSerializedStringBytes = 16U * 1024U * 1024U;

#if (defined(__x86_64__) || defined(_M_X64))                                  \
    && (defined(__GNUC__) || defined(__clang__))

/* Computes fp32 dot product using AVX-512F instructions with fp64 lane
   accumulation to guarantee <= 1e-6 numerical equivalence with scalar/GPU.  */
__attribute__ ((target ("avx512f"))) auto
simd_dot_f32_avx512 (std::span<const float> a,
                     std::span<const float> b) noexcept -> float
{
  const std::size_t n = std::min (a.size (), b.size ());
  alignas (kAlignAvx512Bytes) std::array<float, kSimdWidthAvx512> stage_a{};
  alignas (kAlignAvx512Bytes) std::array<float, kSimdWidthAvx512> stage_b{};
  const std::span<const float> span_a (stage_a);
  const std::span<const float> span_b (stage_b);
  __m512d acc_lo = _mm512_setzero_pd ();
  __m512d acc_hi = _mm512_setzero_pd ();

  std::size_t i = 0U;
  for (; i + kSimdWidthAvx512 <= n; i += kSimdWidthAvx512)
    {
      std::memcpy (stage_a.data (), a.subspan (i, kSimdWidthAvx512).data (),
                   kSimdWidthAvx512 * sizeof (float));
      std::memcpy (stage_b.data (), b.subspan (i, kSimdWidthAvx512).data (),
                   kSimdWidthAvx512 * sizeof (float));
      const __m256 va_lo = _mm256_load_ps (span_a.data ());
      const __m256 vb_lo = _mm256_load_ps (span_b.data ());
      const __m256 va_hi
          = _mm256_load_ps (span_a.subspan (kSimdWidthAvx2).data ());
      const __m256 vb_hi
          = _mm256_load_ps (span_b.subspan (kSimdWidthAvx2).data ());
      acc_lo = _mm512_fmadd_pd (_mm512_cvtps_pd (va_lo),
                                _mm512_cvtps_pd (vb_lo), acc_lo);
      acc_hi = _mm512_fmadd_pd (_mm512_cvtps_pd (va_hi),
                                _mm512_cvtps_pd (vb_hi), acc_hi);
    }

  const __m512d acc = _mm512_add_pd (acc_lo, acc_hi);
  alignas (kAlignAvx512Bytes) std::array<double, kSimdWidthAvx2> lanes{};
  _mm512_store_pd (lanes.data (), acc);
  double sum = 0.0;
  for (const double lane_val : lanes)
    {
      sum += lane_val;
    }
  for (; i < n; ++i)
    {
      sum += static_cast<double> (a[i]) * static_cast<double> (b[i]);
    }
  return static_cast<float> (sum);
}

/* Computes fp32 dot product using AVX2 + FMA instructions with fp64 lane
   accumulation to guarantee <= 1e-6 numerical equivalence with scalar/GPU.  */
__attribute__ ((target ("avx2,fma"))) auto
simd_dot_f32_avx2 (std::span<const float> a,
                   std::span<const float> b) noexcept -> float
{
  const std::size_t n = std::min (a.size (), b.size ());
  alignas (kAlignAvx2Bytes) std::array<float, kSimdWidthAvx2> stage_a{};
  alignas (kAlignAvx2Bytes) std::array<float, kSimdWidthAvx2> stage_b{};
  const std::span<const float> span_a (stage_a);
  const std::span<const float> span_b (stage_b);
  __m256d acc_lo = _mm256_setzero_pd ();
  __m256d acc_hi = _mm256_setzero_pd ();

  std::size_t i = 0U;
  for (; i + kSimdWidthAvx2 <= n; i += kSimdWidthAvx2)
    {
      std::memcpy (stage_a.data (), a.subspan (i, kSimdWidthAvx2).data (),
                   kSimdWidthAvx2 * sizeof (float));
      std::memcpy (stage_b.data (), b.subspan (i, kSimdWidthAvx2).data (),
                   kSimdWidthAvx2 * sizeof (float));
      const __m128 va_lo = _mm_load_ps (span_a.data ());
      const __m128 vb_lo = _mm_load_ps (span_b.data ());
      const __m128 va_hi
          = _mm_load_ps (span_a.subspan (kHalfSimdWidthAvx2).data ());
      const __m128 vb_hi
          = _mm_load_ps (span_b.subspan (kHalfSimdWidthAvx2).data ());
      acc_lo = _mm256_fmadd_pd (_mm256_cvtps_pd (va_lo),
                                _mm256_cvtps_pd (vb_lo), acc_lo);
      acc_hi = _mm256_fmadd_pd (_mm256_cvtps_pd (va_hi),
                                _mm256_cvtps_pd (vb_hi), acc_hi);
    }

  const __m256d acc = _mm256_add_pd (acc_lo, acc_hi);
  alignas (kAlignAvx2Bytes) std::array<double, kHalfSimdWidthAvx2> lanes{};
  _mm256_store_pd (lanes.data (), acc);
  double sum = 0.0;
  for (const double lane_val : lanes)
    {
      sum += lane_val;
    }
  for (; i < n; ++i)
    {
      sum += static_cast<double> (a[i]) * static_cast<double> (b[i]);
    }
  return static_cast<float> (sum);
}

#endif

} /* namespace */

/* Initializes a monotonic buffer arena with pre-allocated storage.  */
InferenceArena::InferenceArena (std::size_t capacity_bytes)
    : buffer_ (capacity_bytes), pool_ (buffer_.data (), buffer_.size (),
                                       std::pmr::new_delete_resource ())
{
  AgentStats::global ().record_arena_alloc (capacity_bytes);
}

/* Releases arena tracking metrics on destruction.  */
InferenceArena::~InferenceArena () noexcept
{
  pool_.release ();
  AgentStats::global ().record_arena_release (buffer_.size ());
}

/* Returns the underlying PMR memory_resource pointer.  */
auto
InferenceArena::resource () noexcept -> std::pmr::memory_resource *
{
  return &pool_;
}

/* Rewinds the monotonic buffer in O(1) time.  */
void
InferenceArena::reset () noexcept
{
  pool_.release ();
}

/* Returns the size of the pre-allocated arena buffer in bytes.  */
auto
InferenceArena::capacity_bytes () const noexcept -> std::size_t
{
  return buffer_.size ();
}

/* Private constructor pre-allocating the ring buffer vector.  */
ShortTermMemory::ShortTermMemory (std::size_t capacity)
    : ring_ (capacity), capacity_ (capacity)
{
}

/* Private delegating copy constructor with lock held.  */
ShortTermMemory::ShortTermMemory (
    const ShortTermMemory &other,
    const std::shared_lock<std::shared_mutex> & /*lock*/)
    : ring_ (other.ring_), capacity_ (other.capacity_), head_ (other.head_),
      count_ (other.count_), next_seq_ (other.next_seq_)
{
}

/* Copy constructor acquiring shared lock on source.  */
ShortTermMemory::ShortTermMemory (const ShortTermMemory &other)
    : ShortTermMemory (other, std::shared_lock (other.mtx_))
{
}

/* Copy assignment operator acquiring locks safely.  */
auto
ShortTermMemory::operator= (const ShortTermMemory &other) -> ShortTermMemory &
{
  if (this != &other)
    {
      std::unique_lock lhs_lock (mtx_, std::defer_lock);
      std::shared_lock rhs_lock (other.mtx_, std::defer_lock);
      std::lock (lhs_lock, rhs_lock);
      ring_ = other.ring_;
      capacity_ = other.capacity_;
      head_ = other.head_;
      count_ = other.count_;
      next_seq_ = other.next_seq_;
    }
  return *this;
}

/* Private delegating move constructor with lock held.  */
ShortTermMemory::ShortTermMemory (
    ShortTermMemory &&other,
    const std::unique_lock<std::shared_mutex> & /*lock*/) noexcept
    : ring_ (std::move (other).ring_), capacity_ (other.capacity_),
      head_ (other.head_), count_ (other.count_), next_seq_ (other.next_seq_)
{
  other.capacity_ = 0U;
  other.head_ = 0U;
  other.count_ = 0U;
}

/* Move constructor acquiring exclusive lock on source.  */
ShortTermMemory::ShortTermMemory (ShortTermMemory &&other) noexcept
    : ShortTermMemory (std::move (other), std::unique_lock (other.mtx_))
{
}

/* Move assignment operator acquiring exclusive locks on both instances.  */
auto
ShortTermMemory::operator= (ShortTermMemory &&other) noexcept
    -> ShortTermMemory &
{
  if (this != &other)
    {
      std::scoped_lock lock (mtx_, other.mtx_);
      ring_ = std::move (other.ring_);
      capacity_ = other.capacity_;
      head_ = other.head_;
      count_ = other.count_;
      next_seq_ = other.next_seq_;
      other.capacity_ = 0U;
      other.head_ = 0U;
      other.count_ = 0U;
    }
  return *this;
}

/* Factory method validating non-zero ring buffer capacity.  */
auto
ShortTermMemory::create (std::size_t capacity) noexcept
    -> Result<ShortTermMemory>
{
  if (capacity == 0U)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "ShortTermMemory capacity must be greater than zero"));
    }
  try
    {
      return ShortTermMemory (capacity);
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Failed to allocate fixed-capacity ShortTermMemory buffer"));
    }
}

/* Appends a message in O(1) without reallocating ring_.  */
auto
ShortTermMemory::append (Message msg) noexcept -> Result<uint64_t>
{
  const std::unique_lock lock (mtx_);
  if (capacity_ == 0U || ring_.size () != capacity_)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "Cannot append to an uninitialized ShortTermMemory ring buffer"));
    }

  const uint64_t seq = next_seq_++;
  msg.sequence_id = seq;
  if (msg.timestamp_ns == 0U)
    {
      msg.timestamp_ns = monotonic_time_ns ();
    }

  if (count_ < capacity_)
    {
      const std::size_t write_idx = (head_ + count_) % capacity_;
      ring_.at (write_idx) = std::move (msg);
      ++count_;
    }
  else
    {
      ring_.at (head_) = std::move (msg);
      head_ = (head_ + 1U) % capacity_;
    }
  return seq;
}

/* Constructs a zero-copy span view of the ring buffer segments.  */
auto
ShortTermMemory::borrow_view () const noexcept -> ConversationSpanView
{
  const std::shared_lock lock (mtx_);
  ConversationSpanView view;
  view.total_turns = static_cast<std::size_t> (next_seq_ - 1U);
  if (count_ == 0U || capacity_ == 0U)
    {
      return view;
    }

  const std::span<const Message> full_span (ring_.data (), ring_.size ());
  if (head_ + count_ <= capacity_)
    {
      view.first_segment = full_span.subspan (head_, count_);
    }
  else
    {
      const std::size_t first_len = capacity_ - head_;
      const std::size_t second_len = count_ - first_len;
      view.first_segment = full_span.subspan (head_, first_len);
      view.second_segment = full_span.subspan (0U, second_len);
    }
  return view;
}

/* Extracts only messages with sequence_id > since_seq for incremental
   checkpointing.  */
auto
ShortTermMemory::delta_since (uint64_t since_seq) const -> std::vector<Message>
{
  const std::shared_lock lock (mtx_);
  std::vector<Message> delta;
  if (count_ == 0U || capacity_ == 0U)
    {
      return delta;
    }

  for (std::size_t i = 0U; i < count_; ++i)
    {
      const std::size_t idx = (head_ + i) % capacity_;
      const Message &msg = ring_.at (idx);
      if (msg.sequence_id > since_seq)
        {
          delta.push_back (msg);
        }
    }
  return delta;
}

/* Returns current ring occupancy.  */
auto
ShortTermMemory::size () const noexcept -> std::size_t
{
  const std::shared_lock lock (mtx_);
  return count_;
}

/* Returns fixed ring capacity.  */
auto
ShortTermMemory::capacity () const noexcept -> std::size_t
{
  const std::shared_lock lock (mtx_);
  return capacity_;
}

/* Returns total messages appended since creation or clear.  */
auto
ShortTermMemory::total_appended () const noexcept -> uint64_t
{
  const std::shared_lock lock (mtx_);
  return next_seq_ - 1U;
}

/* Resets ring pointers without deallocating storage.  */
void
ShortTermMemory::clear () noexcept
{
  const std::unique_lock lock (mtx_);
  head_ = 0U;
  count_ = 0U;
  next_seq_ = 1U;
}

/* Private delegating copy constructor for LongTermMemory.  */
LongTermMemory::LongTermMemory (
    const LongTermMemory &other,
    const std::shared_lock<std::shared_mutex> & /*lock*/)
    : store_ (other.store_)
{
}

/* Copy constructor for LongTermMemory.  */
LongTermMemory::LongTermMemory (const LongTermMemory &other)
    : LongTermMemory (other, std::shared_lock (other.mtx_))
{
}

/* Copy assignment for LongTermMemory.  */
auto
LongTermMemory::operator= (const LongTermMemory &other) -> LongTermMemory &
{
  if (this != &other)
    {
      std::unique_lock lhs_lock (mtx_, std::defer_lock);
      std::shared_lock rhs_lock (other.mtx_, std::defer_lock);
      std::lock (lhs_lock, rhs_lock);
      store_ = other.store_;
    }
  return *this;
}

/* Private delegating move constructor for LongTermMemory.  */
LongTermMemory::LongTermMemory (
    LongTermMemory &&other,
    const std::unique_lock<std::shared_mutex> & /*lock*/) noexcept
    : store_ (std::move (other).store_)
{
}

/* Move constructor for LongTermMemory.  */
LongTermMemory::LongTermMemory (LongTermMemory &&other) noexcept
    : LongTermMemory (std::move (other), std::unique_lock (other.mtx_))
{
}

/* Move assignment for LongTermMemory.  */
auto
LongTermMemory::operator= (LongTermMemory &&other) noexcept -> LongTermMemory &
{
  if (this != &other)
    {
      std::scoped_lock lock (mtx_, other.mtx_);
      store_ = std::move (other.store_);
    }
  return *this;
}

/* Inserts or updates a key-value entry in LongTermMemory.  */
auto
LongTermMemory::put (std::string key,
                     std::string value) noexcept -> Result<void>
{
  if (key.empty ())
    {
      return std::unexpected (
          AgentError::framework (FrameworkErrorCode::InvalidConfiguration, 0,
                                 "LongTermMemory key must not be empty"));
    }
  try
    {
      const std::unique_lock lock (mtx_);
      store_.insert_or_assign (std::move (key), std::move (value));
      return {};
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory while inserting into LongTermMemory"));
    }
}

/* Looks up a value by key in LongTermMemory.  */
auto
LongTermMemory::get (std::string_view key) const noexcept
    -> Result<std::string>
{
  try
    {
      const std::shared_lock lock (mtx_);
      const auto it = store_.find (std::string (key));
      if (it == store_.end ())
        {
          return std::unexpected (AgentError::framework (
              FrameworkErrorCode::InvalidConfiguration, 0,
              "Requested key was not found in LongTermMemory"));
        }
      return it->second;
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory while reading from LongTermMemory"));
    }
}

/* Removes a key from LongTermMemory.  */
auto
LongTermMemory::erase (std::string_view key) noexcept -> bool
{
  try
    {
      const std::unique_lock lock (mtx_);
      return store_.erase (std::string (key)) > 0U;
    }
  catch (...)
    {
      return false;
    }
}

/* Returns the number of entries in LongTermMemory.  */
auto
LongTermMemory::size () const noexcept -> std::size_t
{
  const std::shared_lock lock (mtx_);
  return store_.size ();
}

/* Saves LongTermMemory to a binary file with RAII stream cleanup.  */
auto
LongTermMemory::save_to_file (std::string_view filepath) const noexcept
    -> Result<void>
{
  if (filepath.empty ())
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "Filepath for LongTermMemory::save_to_file must not be empty"));
    }

  try
    {
      const std::string path_str (filepath);
      std::ofstream out_stream (path_str, std::ios::binary | std::ios::trunc);
      if (!out_stream.is_open ())
        {
          return std::unexpected (AgentError::framework (
              FrameworkErrorCode::InvalidConfiguration, 0,
              "Failed to open LongTermMemory persistence file for writing"));
        }

      const std::shared_lock lock (mtx_);
      const uint32_t magic = kLtmFileMagic;
      const uint32_t version = kLtmFileVersion;
      const auto count = static_cast<uint64_t> (store_.size ());

      out_stream.write (std::bit_cast<const char *> (&magic), sizeof (magic));
      out_stream.write (std::bit_cast<const char *> (&version),
                        sizeof (version));
      out_stream.write (std::bit_cast<const char *> (&count), sizeof (count));
      if (!out_stream.good ())
        {
          return std::unexpected (AgentError::framework (
              FrameworkErrorCode::CheckpointCorrupted, 0,
              "Failed to write binary header to LongTermMemory file"));
        }

      for (const auto &[k, v] : store_)
        {
          const auto k_len = static_cast<uint64_t> (k.size ());
          const auto v_len = static_cast<uint64_t> (v.size ());
          out_stream.write (std::bit_cast<const char *> (&k_len),
                            sizeof (k_len));
          out_stream.write (k.data (),
                            static_cast<std::streamsize> (k.size ()));
          out_stream.write (std::bit_cast<const char *> (&v_len),
                            sizeof (v_len));
          out_stream.write (v.data (),
                            static_cast<std::streamsize> (v.size ()));
          if (!out_stream.good ())
            {
              return std::unexpected (AgentError::framework (
                  FrameworkErrorCode::CheckpointCorrupted, 0,
                  "Failed to write key-value record to LongTermMemory file"));
            }
        }
      return {};
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Exception during LongTermMemory::save_to_file"));
    }
}

/* Loads LongTermMemory from a binary file.  */
auto
LongTermMemory::load_from_file (std::string_view filepath) noexcept
    -> Result<void>
{
  if (filepath.empty ())
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "Filepath for LongTermMemory::load_from_file must not be empty"));
    }

  try
    {
      const std::string path_str (filepath);
      std::ifstream in_stream (path_str, std::ios::binary);
      if (!in_stream.is_open ())
        {
          return std::unexpected (AgentError::framework (
              FrameworkErrorCode::InvalidConfiguration, 0,
              "Failed to open LongTermMemory persistence file for reading"));
        }

      uint32_t magic = 0U;
      uint32_t version = 0U;
      uint64_t count = 0U;
      in_stream.read (std::bit_cast<char *> (&magic), sizeof (magic));
      in_stream.read (std::bit_cast<char *> (&version), sizeof (version));
      in_stream.read (std::bit_cast<char *> (&count), sizeof (count));
      if (!in_stream.good () || magic != kLtmFileMagic
          || version != kLtmFileVersion)
        {
          return std::unexpected (AgentError::framework (
              FrameworkErrorCode::CheckpointCorrupted, 0,
              "Invalid binary header in LongTermMemory persistence file"));
        }

      std::unordered_map<std::string, std::string> loaded;
      for (uint64_t i = 0U; i < count; ++i)
        {
          uint64_t k_len = 0U;
          in_stream.read (std::bit_cast<char *> (&k_len), sizeof (k_len));
          if (!in_stream.good () || k_len > kMaxSerializedStringBytes)
            {
              return std::unexpected (AgentError::framework (
                  FrameworkErrorCode::CheckpointCorrupted, 0,
                  "Corrupted key length in LongTermMemory file"));
            }
          std::string key (static_cast<std::size_t> (k_len), '\0');
          in_stream.read (key.data (),
                          static_cast<std::streamsize> (key.size ()));
          if (!in_stream.good ())
            {
              return std::unexpected (AgentError::framework (
                  FrameworkErrorCode::CheckpointCorrupted, 0,
                  "Truncated key payload in LongTermMemory file"));
            }

          uint64_t v_len = 0U;
          in_stream.read (std::bit_cast<char *> (&v_len), sizeof (v_len));
          if (!in_stream.good () || v_len > kMaxSerializedStringBytes)
            {
              return std::unexpected (AgentError::framework (
                  FrameworkErrorCode::CheckpointCorrupted, 0,
                  "Corrupted value length in LongTermMemory file"));
            }
          std::string val (static_cast<std::size_t> (v_len), '\0');
          in_stream.read (val.data (),
                          static_cast<std::streamsize> (val.size ()));
          if (!in_stream.good ())
            {
              return std::unexpected (AgentError::framework (
                  FrameworkErrorCode::CheckpointCorrupted, 0,
                  "Truncated value payload in LongTermMemory file"));
            }
          loaded.insert_or_assign (std::move (key), std::move (val));
        }

      const std::unique_lock lock (mtx_);
      store_ = std::move (loaded);
      return {};
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory while loading LongTermMemory file"));
    }
}

/* Clears all entries from LongTermMemory.  */
void
LongTermMemory::clear () noexcept
{
  const std::unique_lock lock (mtx_);
  store_.clear ();
}

/* Detects CPU SIMD capability via __builtin_cpu_supports.  */
auto
detect_simd_level () noexcept -> SimdLevel
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init ();
  if (__builtin_cpu_supports ("avx512f"))
    {
      return SimdLevel::AVX512;
    }
  if (__builtin_cpu_supports ("avx2") && __builtin_cpu_supports ("fma"))
    {
      return SimdLevel::AVX2;
    }
  if (__builtin_cpu_supports ("sse4.2"))
    {
      return SimdLevel::SSE4_2;
    }
#endif
#endif
  return SimdLevel::Scalar;
}

/* Scalar reference fp32 dot product using double accumulator.  */
auto
simd_dot_f32_scalar (std::span<const float> a,
                     std::span<const float> b) noexcept -> float
{
  const std::size_t n = std::min (a.size (), b.size ());
  double sum = 0.0;
  for (std::size_t i = 0U; i < n; ++i)
    {
      sum += static_cast<double> (a[i]) * static_cast<double> (b[i]);
    }
  return static_cast<float> (sum);
}

/* Runtime-dispatched SIMD fp32 dot product.  */
auto
simd_dot_f32 (std::span<const float> a,
              std::span<const float> b) noexcept -> float
{
#if (defined(__x86_64__) || defined(_M_X64))                                  \
    && (defined(__GNUC__) || defined(__clang__))
  const SimdLevel lvl = detect_simd_level ();
  if (lvl == SimdLevel::AVX512)
    {
      return simd_dot_f32_avx512 (a, b);
    }
  if (lvl == SimdLevel::AVX2)
    {
      return simd_dot_f32_avx2 (a, b);
    }
#endif
  return simd_dot_f32_scalar (a, b);
}

/* Private constructor setting embedding dimension.  */
SemanticMemory::SemanticMemory (std::size_t dimension) noexcept
    : dimension_ (dimension)
{
}

/* Private delegating copy constructor for SemanticMemory.  */
SemanticMemory::SemanticMemory (
    const SemanticMemory &other,
    const std::shared_lock<std::shared_mutex> & /*lock*/)
    : dimension_ (other.dimension_), docs_ (other.docs_),
      flat_embeddings_ (other.flat_embeddings_)
{
}

/* Copy constructor for SemanticMemory.  */
SemanticMemory::SemanticMemory (const SemanticMemory &other)
    : SemanticMemory (other, std::shared_lock (other.mtx_))
{
}

/* Copy assignment for SemanticMemory.  */
auto
SemanticMemory::operator= (const SemanticMemory &other) -> SemanticMemory &
{
  if (this != &other)
    {
      std::unique_lock lhs_lock (mtx_, std::defer_lock);
      std::shared_lock rhs_lock (other.mtx_, std::defer_lock);
      std::lock (lhs_lock, rhs_lock);
      dimension_ = other.dimension_;
      docs_ = other.docs_;
      flat_embeddings_ = other.flat_embeddings_;
    }
  return *this;
}

/* Private delegating move constructor for SemanticMemory.  */
SemanticMemory::SemanticMemory (
    SemanticMemory &&other,
    const std::unique_lock<std::shared_mutex> & /*lock*/) noexcept
    : dimension_ (other.dimension_), docs_ (std::move (other).docs_),
      flat_embeddings_ (std::move (other.flat_embeddings_))
{
}

/* Move constructor for SemanticMemory.  */
SemanticMemory::SemanticMemory (SemanticMemory &&other) noexcept
    : SemanticMemory (std::move (other), std::unique_lock (other.mtx_))
{
}

/* Move assignment for SemanticMemory.  */
auto
SemanticMemory::operator= (SemanticMemory &&other) noexcept -> SemanticMemory &
{
  if (this != &other)
    {
      std::scoped_lock lock (mtx_, other.mtx_);
      dimension_ = other.dimension_;
      docs_ = std::move (other.docs_);
      flat_embeddings_ = std::move (other.flat_embeddings_);
    }
  return *this;
}

/* Creates a SemanticMemory instance after validating dimension > 0.  */
auto
SemanticMemory::create (std::size_t dimension) noexcept
    -> Result<SemanticMemory>
{
  if (dimension == 0U)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "SemanticMemory embedding dimension must be greater than zero"));
    }
  return SemanticMemory (dimension);
}

/* Adds a document and appends its embedding to the contiguous flat matrix.  */
auto
SemanticMemory::add_document (std::string id, std::string text,
                              std::span<const float> embedding) noexcept
    -> Result<void>
{
  if (id.empty ())
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "SemanticMemory document id must not be empty"));
    }

  try
    {
      const std::unique_lock lock (mtx_);
      if (embedding.size () != dimension_)
        {
          return std::unexpected (AgentError::framework (
              FrameworkErrorCode::InvalidConfiguration, 0,
              "Document embedding length does not match SemanticMemory "
              "dimension"));
        }

      docs_.push_back (DocumentMeta{ std::move (id), std::move (text) });
      flat_embeddings_.insert (flat_embeddings_.end (), embedding.begin (),
                               embedding.end ());
      return {};
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory while adding document to SemanticMemory"));
    }
}

/* Searches top_k documents by dot product using SIMD or CUDA GPU offload.  */
auto
SemanticMemory::search (std::span<const float> query, std::size_t top_k,
                        bool prefer_gpu) const noexcept
    -> Result<std::vector<SemanticMatch>>
{
  if (top_k == 0U)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "SemanticMemory::search requires top_k greater than zero"));
    }

  try
    {
      const std::shared_lock lock (mtx_);
      if (query.size () != dimension_)
        {
          return std::unexpected (AgentError::framework (
              FrameworkErrorCode::InvalidConfiguration, 0,
              "Query embedding length does not match SemanticMemory "
              "dimension"));
        }

      const std::size_t num_docs = docs_.size ();
      if (num_docs == 0U)
        {
          return std::vector<SemanticMatch>{};
        }

      std::vector<float> scores (num_docs, 0.0F);
      bool gpu_completed = false;

      if (prefer_gpu && is_cuda_available ())
        {
          auto stream_res = Stream::create ();
          auto mat_res = Tensor::allocate (num_docs * dimension_);
          auto qry_res = Tensor::allocate (dimension_);
          auto out_res = Tensor::allocate (num_docs);

          if (stream_res && mat_res && qry_res && out_res)
            {
              const std::span<const float> mat_span (flat_embeddings_.data (),
                                                     flat_embeddings_.size ());
              if (mat_res->copy_from_host (mat_span, *stream_res)
                  && qry_res->copy_from_host (query, *stream_res)
                  && gpu_batch_dot_f32 (*mat_res, *qry_res, *out_res, num_docs,
                                        dimension_, *stream_res)
                  && out_res->copy_to_host (scores, *stream_res)
                  && stream_res->synchronize ())
                {
                  gpu_completed = true;
                }
            }
        }

      if (!gpu_completed)
        {
          const std::span<const float> full_mat (flat_embeddings_.data (),
                                                 flat_embeddings_.size ());
          for (std::size_t i = 0U; i < num_docs; ++i)
            {
              const std::span<const float> row
                  = full_mat.subspan (i * dimension_, dimension_);
              scores.at (i) = simd_dot_f32 (row, query);
            }
        }

      std::vector<std::size_t> indices (num_docs);
      std::iota (indices.begin (), indices.end (), 0U);
      const std::size_t limit = std::min (top_k, num_docs);

      std::partial_sort (
          indices.begin (),
          indices.begin () + static_cast<std::ptrdiff_t> (limit),
          indices.end (), [&scores] (std::size_t lhs, std::size_t rhs) {
            return scores.at (lhs) > scores.at (rhs);
          });

      std::vector<SemanticMatch> results;
      results.reserve (limit);
      for (std::size_t i = 0U; i < limit; ++i)
        {
          const std::size_t idx = indices.at (i);
          const DocumentMeta &meta = docs_.at (idx);
          results.push_back (
              SemanticMatch{ meta.id, meta.text, scores.at (idx) });
        }
      return results;
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory during SemanticMemory::search"));
    }
}

/* Returns the number of indexed documents.  */
auto
SemanticMemory::size () const noexcept -> std::size_t
{
  const std::shared_lock lock (mtx_);
  return docs_.size ();
}

/* Returns the configured embedding dimension.  */
auto
SemanticMemory::dimension () const noexcept -> std::size_t
{
  const std::shared_lock lock (mtx_);
  return dimension_;
}

/* Clears all documents and embeddings.  */
void
SemanticMemory::clear () noexcept
{
  const std::unique_lock lock (mtx_);
  docs_.clear ();
  flat_embeddings_.clear ();
}

/* Private constructor composing the three memory tiers.  */
MemorySystem::MemorySystem (ShortTermMemory st, LongTermMemory lt,
                            SemanticMemory sem) noexcept
    : short_term_ (std::move (st)), long_term_ (std::move (lt)),
      semantic_ (std::move (sem))
{
}

/* Factory method initializing all three isolated memory tiers.  */
auto
MemorySystem::create (std::size_t short_term_capacity,
                      std::size_t semantic_dimension) noexcept
    -> Result<MemorySystem>
{
  auto st_res = ShortTermMemory::create (short_term_capacity);
  if (!st_res)
    {
      return std::unexpected (st_res.error ());
    }
  auto sem_res = SemanticMemory::create (semantic_dimension);
  if (!sem_res)
    {
      return std::unexpected (sem_res.error ());
    }
  return MemorySystem (std::move (*st_res), LongTermMemory{},
                       std::move (*sem_res));
}

/* Returns mutable reference to Tier 1 ShortTermMemory.  */
auto
MemorySystem::short_term () noexcept -> ShortTermMemory &
{
  return short_term_;
}

/* Returns const reference to Tier 1 ShortTermMemory.  */
auto
MemorySystem::short_term () const noexcept -> const ShortTermMemory &
{
  return short_term_;
}

/* Returns mutable reference to Tier 2 LongTermMemory.  */
auto
MemorySystem::long_term () noexcept -> LongTermMemory &
{
  return long_term_;
}

/* Returns const reference to Tier 2 LongTermMemory.  */
auto
MemorySystem::long_term () const noexcept -> const LongTermMemory &
{
  return long_term_;
}

/* Returns mutable reference to Tier 3 SemanticMemory.  */
auto
MemorySystem::semantic () noexcept -> SemanticMemory &
{
  return semantic_;
}

/* Returns const reference to Tier 3 SemanticMemory.  */
auto
MemorySystem::semantic () const noexcept -> const SemanticMemory &
{
  return semantic_;
}

} /* namespace sutra::inline v1 */
