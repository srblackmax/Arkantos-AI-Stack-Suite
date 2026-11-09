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

#ifndef SUTRA_CHECKPOINT_HPP
#define SUTRA_CHECKPOINT_HPP

#include <sutra/error.hpp>
#include <sutra/memory.hpp>
#include <sutra/scheduler.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace sutra::inline v1
{

/**
 * @brief Default capacity of the lock-free asynchronous checkpoint queue.
 */
inline constexpr std::size_t kDefaultCheckpointQueueCapacity = 1024U;

/**
 * @brief Represents an incremental state delta since the previous checkpoint.
 * @note Thread-safety: Value type transferred across LockFreeQueue.
 */
struct CheckpointDelta
{
  uint64_t run_id = 0U;
  uint64_t prev_sequence_id = 0U;
  uint64_t high_sequence_id = 0U;
  std::vector<Message> messages;
};

/**
 * @brief Asynchronous incremental binary checkpointer with CRC32 verification.
 * @pre filepath must be writable when checkpointing is active.
 * @post Serializes only per-turn deltas in binary format on a background
 *       thread; never blocks the hot agent inference loop.
 * @note Thread-safety: Hot-path enqueue_delta() is lock-free via
 * LockFreeQueue.
 */
class CheckpointManager
{
private:
  struct PrivateTag
  {
    explicit PrivateTag () = default;
  };

public:
  CheckpointManager (PrivateTag tag, std::string filepath,
                     std::size_t queue_capacity);

  /**
   * @brief Creates and starts an asynchronous CheckpointManager.
   * @param filepath Destination binary checkpoint log path.
   * @param queue_capacity Capacity of the lock-free delta ring queue.
   * @pre !filepath.empty().
   * @post Background writer thread is running and ready to flush deltas.
   * @returns Result<std::unique_ptr<CheckpointManager>> or AgentError.
   */
  [[nodiscard]] static auto create (std::string filepath,
                                    std::size_t queue_capacity
                                    = kDefaultCheckpointQueueCapacity) noexcept
      -> Result<std::unique_ptr<CheckpointManager>>;

  CheckpointManager (const CheckpointManager &) = delete;
  auto operator= (const CheckpointManager &) -> CheckpointManager & = delete;
  CheckpointManager (CheckpointManager &&) = delete;
  auto operator= (CheckpointManager &&) -> CheckpointManager & = delete;

  /**
   * @brief Flushes pending deltas and joins the background writer thread.
   */
  ~CheckpointManager () noexcept;

  /**
   * @brief Enqueues an incremental delta without blocking the agent loop.
   * @param delta Incremental messages appended since last checkpoint.
   * @pre None.
   * @post Delta is queued in O(1) for asynchronous binary serialization.
   * @returns Result<void> or FrameworkErrorCode::ResourceExhaustion if full.
   * @note Thread-safety: Lock-free via LockFreeQueue::try_push.
   */
  [[nodiscard]] auto
  enqueue_delta (CheckpointDelta delta) noexcept -> Result<void>;

  /**
   * @brief Waits until all enqueued deltas have been persisted to disk.
   * @pre Do not call on the hot path if non-blocking behavior is desired.
   * @post flushed_count() >= enqueued_count().
   * @returns Result<void> or FrameworkErrorCode::CheckpointCorrupted on I/O
   * err.
   * @note Thread-safety: Lock-free atomic wait.
   */
  [[nodiscard]] auto flush () noexcept -> Result<void>;

  /**
   * @brief Returns the total number of incremental delta frames enqueued.
   * @returns Enqueued frame count.
   */
  [[nodiscard]] auto enqueued_count () const noexcept -> uint64_t;

  /**
   * @brief Returns the total number of incremental delta frames written to
   * disk.
   * @returns Flushed frame count.
   */
  [[nodiscard]] auto flushed_count () const noexcept -> uint64_t;

  /**
   * @brief Replays a binary checkpoint file and reconstructs message history.
   * @param filepath Path to the binary checkpoint file.
   * @pre File was written by CheckpointManager.
   * @post Verifies magic bytes, format version, and CRC32 on every frame.
   * @returns Result<std::vector<Message>> or
   * FrameworkErrorCode::CheckpointCorrupted.
   * @note Thread-safety: Safe when no writer is concurrently modifying
   * filepath.
   */
  [[nodiscard]] static auto
  replay (std::string_view filepath) noexcept -> Result<std::vector<Message>>;

private:
  void writer_loop () noexcept;
  auto write_binary_frame (const CheckpointDelta &delta) noexcept -> bool;

  std::string filepath_;
  LockFreeQueue<CheckpointDelta> queue_;
  std::atomic<uint64_t> enqueued_{ 0U };
  std::atomic<uint64_t> flushed_{ 0U };
  std::atomic<uint32_t> wake_signal_{ 0U };
  std::atomic<bool> stop_{ false };
  std::atomic<bool> io_error_{ false };
  std::jthread writer_thread_;
};

} /* namespace sutra::inline v1 */

#endif /* SUTRA_CHECKPOINT_HPP */
