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

#include <sutra/checkpoint.hpp>
#include <sutra/observability.hpp>

#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <utility>

namespace sutra::inline v1
{

namespace
{

constexpr uint64_t kCheckpointMagic = 0x535554524143504BULL; // "SUTRACPK"
constexpr uint32_t kCheckpointVersion = 1U;
constexpr uint32_t kCrc32Polynomial = 0xEDB88320U;
constexpr uint32_t kCrc32Initial = 0xFFFFFFFFU;
constexpr int kCrcBitsPerByte = 8;
constexpr uint32_t kMaxFramePayloadBytes = 64U * 1024U * 1024U;
constexpr std::size_t kBytesPerMessageEstimate = 128U;

/* Aligned binary header preceding each incremental delta frame.  */
struct BinaryFrameHeader
{
  uint64_t magic = kCheckpointMagic;
  uint32_t version = kCheckpointVersion;
  uint32_t message_count = 0U;
  uint64_t run_id = 0U;
  uint64_t prev_sequence_id = 0U;
  uint64_t high_sequence_id = 0U;
  uint32_t payload_bytes = 0U;
  uint32_t crc32 = 0U;
};

/* Computes standard IEEE 802.3 CRC32 checksum over a byte span.  */
[[nodiscard]] auto
compute_crc32 (std::span<const std::byte> data) noexcept -> uint32_t
{
  uint32_t crc = kCrc32Initial;
  for (const std::byte b : data)
    {
      crc ^= static_cast<uint32_t> (std::to_integer<uint8_t> (b));
      for (int bit = 0; bit < kCrcBitsPerByte; ++bit)
        {
          const uint32_t mask = -(crc & 1U);
          crc = (crc >> 1U) ^ (kCrc32Polynomial & mask);
        }
    }
  return ~crc;
}

/* Appends a trivial scalar value in native layout using std::bit_cast.  */
template <typename ScalarT>
void
append_pod (std::vector<std::byte> &buf, ScalarT val)
{
  const auto bytes
      = std::bit_cast<std::array<std::byte, sizeof (ScalarT)>> (val);
  buf.insert (buf.end (), bytes.begin (), bytes.end ());
}

/* Appends a length-prefixed string to buf without pointer arithmetic.  */
void
append_string (std::vector<std::byte> &buf, std::string_view str)
{
  const auto len = static_cast<uint32_t> (str.size ());
  append_pod (buf, len);
  for (const char ch : str)
    {
      buf.push_back (static_cast<std::byte> (ch));
    }
}

/* Reads a trivial scalar from a byte span at cursor.  */
template <typename ScalarT>
[[nodiscard]] auto
read_pod (std::span<const std::byte> buf, std::size_t &cursor,
          ScalarT &out) noexcept -> bool
{
  if (cursor + sizeof (ScalarT) > buf.size ())
    {
      return false;
    }
  std::memcpy (&out, buf.subspan (cursor, sizeof (ScalarT)).data (),
               sizeof (ScalarT));
  cursor += sizeof (ScalarT);
  return true;
}

/* Reads a length-prefixed string from a byte span at cursor.  */
[[nodiscard]] auto
read_string (std::span<const std::byte> buf, std::size_t &cursor,
             std::string &out) -> bool
{
  uint32_t len = 0U;
  if (!read_pod (buf, cursor, len))
    {
      return false;
    }
  if (cursor + len > buf.size ())
    {
      return false;
    }
  out.resize (len);
  if (len > 0U)
    {
      std::memcpy (out.data (), buf.subspan (cursor, len).data (), len);
    }
  cursor += len;
  return true;
}

} /* namespace */

/* Constructs CheckpointManager and launches background writer jthread.  */
CheckpointManager::CheckpointManager (PrivateTag /*tag*/, std::string filepath,
                                      std::size_t queue_capacity)
    : filepath_ (std::move (filepath)), queue_ (queue_capacity),
      writer_thread_ ([this] { writer_loop (); })
{
}

/* Stops background writer thread and flushes all queued deltas.  */
CheckpointManager::~CheckpointManager () noexcept
{
  stop_.store (true, std::memory_order_release);
  wake_signal_.fetch_add (1U, std::memory_order_acq_rel);
  wake_signal_.notify_all ();
}

/* Validates path, truncates/initializes binary log file, and creates manager.
 */
auto
CheckpointManager::create (std::string filepath,
                           std::size_t queue_capacity) noexcept
    -> Result<std::unique_ptr<CheckpointManager>>
{
  if (filepath.empty ())
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "CheckpointManager filepath must not be empty"));
    }

  try
    {
      std::ofstream out (filepath, std::ios::binary | std::ios::trunc);
      if (!out.is_open ())
        {
          return std::unexpected (AgentError::framework (
              FrameworkErrorCode::InvalidConfiguration, 0,
              "Cannot open checkpoint file for binary writing"));
        }
      out.close ();

      return std::make_unique<CheckpointManager> (
          PrivateTag{}, std::move (filepath), queue_capacity);
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Failed to allocate CheckpointManager instance"));
    }
}

/* Enqueues an incremental delta onto the lock-free ring buffer in O(1).  */
auto
CheckpointManager::enqueue_delta (CheckpointDelta delta) noexcept
    -> Result<void>
{
  if (delta.messages.empty ())
    {
      return {};
    }

  enqueued_.fetch_add (1U, std::memory_order_release);
  if (!queue_.try_push (std::move (delta)))
    {
      enqueued_.fetch_sub (1U, std::memory_order_relaxed);
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Asynchronous checkpoint ring buffer is full; increase "
          "queue_capacity"));
    }

  wake_signal_.fetch_add (1U, std::memory_order_release);
  wake_signal_.notify_one ();
  return {};
}

/* Waits until flushed_ catches up with enqueued_.  */
auto
CheckpointManager::flush () noexcept -> Result<void>
{
  const uint64_t target = enqueued_.load (std::memory_order_acquire);
  while (flushed_.load (std::memory_order_acquire) < target)
    {
      wake_signal_.fetch_add (1U, std::memory_order_release);
      wake_signal_.notify_one ();
      std::this_thread::yield ();
    }

  if (io_error_.load (std::memory_order_acquire))
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::CheckpointCorrupted, 0,
          "Background checkpoint writer encountered a disk I/O error"));
    }
  return {};
}

/* Returns total enqueued delta frames.  */
auto
CheckpointManager::enqueued_count () const noexcept -> uint64_t
{
  return enqueued_.load (std::memory_order_acquire);
}

/* Returns total flushed delta frames.  */
auto
CheckpointManager::flushed_count () const noexcept -> uint64_t
{
  return flushed_.load (std::memory_order_acquire);
}

/* Background thread draining the lock-free delta queue to disk.  */
void
CheckpointManager::writer_loop () noexcept
{
  CheckpointDelta delta;
  while (!stop_.load (std::memory_order_acquire))
    {
      if (queue_.try_pop (delta))
        {
          if (!write_binary_frame (delta))
            {
              io_error_.store (true, std::memory_order_release);
            }
          flushed_.fetch_add (1U, std::memory_order_release);
        }
      else
        {
          const uint32_t observed
              = wake_signal_.load (std::memory_order_acquire);
          if (!stop_.load (std::memory_order_acquire))
            {
              wake_signal_.wait (observed, std::memory_order_relaxed);
            }
        }
    }

  while (queue_.try_pop (delta))
    {
      if (!write_binary_frame (delta))
        {
          io_error_.store (true, std::memory_order_release);
        }
      flushed_.fetch_add (1U, std::memory_order_release);
    }
}

/* Serializes one incremental delta frame in binary format with CRC32.  */
auto
CheckpointManager::write_binary_frame (const CheckpointDelta &delta) noexcept
    -> bool
{
  const ScopedStageTimer timer (AgentStats::global (),
                                StageKind::CheckpointWrite,
                                "checkpoint_write_binary");
  try
    {
      std::vector<std::byte> payload;
      payload.reserve (delta.messages.size () * kBytesPerMessageEstimate);

      for (const Message &msg : delta.messages)
        {
          append_pod (payload, static_cast<uint8_t> (msg.role));
          append_pod (payload, msg.sequence_id);
          append_pod (payload, msg.timestamp_ns);
          append_string (payload, msg.content);
          append_string (payload, msg.name);
          append_string (payload, msg.tool_call_id);
        }

      BinaryFrameHeader hdr;
      hdr.message_count = static_cast<uint32_t> (delta.messages.size ());
      hdr.run_id = delta.run_id;
      hdr.prev_sequence_id = delta.prev_sequence_id;
      hdr.high_sequence_id = delta.high_sequence_id;
      hdr.payload_bytes = static_cast<uint32_t> (payload.size ());
      hdr.crc32 = compute_crc32 (payload);

      std::ofstream out (filepath_, std::ios::binary | std::ios::app);
      if (!out.is_open ())
        {
          return false;
        }

      const auto hdr_bytes
          = std::bit_cast<std::array<char, sizeof (hdr)>> (hdr);
      out.write (hdr_bytes.data (), sizeof (hdr));
      if (!payload.empty ())
        {
          std::string pay_chars (payload.size (), '\0');
          std::memcpy (pay_chars.data (), payload.data (), payload.size ());
          out.write (pay_chars.data (),
                     static_cast<std::streamsize> (pay_chars.size ()));
        }
      out.flush ();
      return out.good ();
    }
  catch (...)
    {
      return false;
    }
}

/* Replays all incremental binary frames from filepath and verifies CRC32.  */
auto
CheckpointManager::replay (std::string_view filepath) noexcept
    -> Result<std::vector<Message>>
{
  if (filepath.empty ())
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "Filepath for CheckpointManager::replay must not be empty"));
    }

  try
    {
      std::ifstream in (std::string (filepath), std::ios::binary);
      if (!in.is_open ())
        {
          return std::unexpected (AgentError::framework (
              FrameworkErrorCode::InvalidConfiguration, 0,
              "Cannot open checkpoint file for replay"));
        }

      std::vector<Message> history;
      std::array<char, sizeof (BinaryFrameHeader)> hdr_buf{};

      while (in.read (hdr_buf.data (), sizeof (BinaryFrameHeader)))
        {
          const auto hdr = std::bit_cast<BinaryFrameHeader> (hdr_buf);
          if (hdr.magic != kCheckpointMagic
              || hdr.version != kCheckpointVersion
              || hdr.payload_bytes > kMaxFramePayloadBytes)
            {
              return std::unexpected (AgentError::framework (
                  FrameworkErrorCode::CheckpointCorrupted, 0,
                  "Invalid frame header magic or version in checkpoint file"));
            }

          std::vector<std::byte> payload (hdr.payload_bytes);
          if (hdr.payload_bytes > 0U)
            {
              std::string pay_chars (hdr.payload_bytes, '\0');
              if (!in.read (pay_chars.data (),
                            static_cast<std::streamsize> (hdr.payload_bytes)))
                {
                  return std::unexpected (AgentError::framework (
                      FrameworkErrorCode::CheckpointCorrupted, 0,
                      "Truncated binary frame payload in checkpoint file"));
                }
              std::memcpy (payload.data (), pay_chars.data (),
                           hdr.payload_bytes);
            }

          const uint32_t actual_crc = compute_crc32 (payload);
          if (actual_crc != hdr.crc32)
            {
              return std::unexpected (AgentError::framework (
                  FrameworkErrorCode::CheckpointCorrupted, 0,
                  "CRC32 checksum mismatch in incremental checkpoint frame"));
            }

          std::size_t cursor = 0U;
          const std::span<const std::byte> pay_span (payload.data (),
                                                     payload.size ());
          for (uint32_t m = 0U; m < hdr.message_count; ++m)
            {
              uint8_t role_u8 = 0U;
              Message msg;
              if (!read_pod (pay_span, cursor, role_u8)
                  || !read_pod (pay_span, cursor, msg.sequence_id)
                  || !read_pod (pay_span, cursor, msg.timestamp_ns)
                  || !read_string (pay_span, cursor, msg.content)
                  || !read_string (pay_span, cursor, msg.name)
                  || !read_string (pay_span, cursor, msg.tool_call_id))
                {
                  return std::unexpected (AgentError::framework (
                      FrameworkErrorCode::CheckpointCorrupted, 0,
                      "Malformed message record inside checkpoint frame"));
                }
              msg.role = static_cast<MessageRole> (role_u8);
              history.push_back (std::move (msg));
            }
        }

      return history;
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory during CheckpointManager::replay"));
    }
}

} /* namespace sutra::inline v1 */
