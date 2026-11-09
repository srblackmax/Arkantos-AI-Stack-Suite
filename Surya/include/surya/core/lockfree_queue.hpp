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

#ifndef SURYA_CORE_LOCKFREE_QUEUE_HPP
#define SURYA_CORE_LOCKFREE_QUEUE_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <surya/core/config.hpp>

namespace surya::inline v1::core
{

/**
 * @brief Bounded lock-free MPSC/MPMC ring queue based on Vyukov sequence
 *        tickets.  Zero std::mutex and zero dynamic allocation in hot path.
 * @tparam T Move-constructible payload type.
 * @tparam Capacity Power-of-two ring capacity (must be >= 2).
 * @note Thread-safety: Safe for concurrent producers and consumers.
 */
template <typename T, size_t Capacity = 256U>
class Lock_Free_Queue
{
  static_assert (Capacity >= 2U && (Capacity & (Capacity - 1U)) == 0U,
                 "Lock_Free_Queue Capacity must be a power of two >= 2");

public:
  Lock_Free_Queue (void) noexcept
  {
    for (size_t i = 0U; i < Capacity; ++i)
      {
        slots_[i].sequence.store (i, std::memory_order_relaxed);
      }
    enqueue_pos_.store (0U, std::memory_order_relaxed);
    dequeue_pos_.store (0U, std::memory_order_relaxed);
  }

  ~Lock_Free_Queue (void) noexcept = default;

  Lock_Free_Queue (const Lock_Free_Queue &) = delete;
  Lock_Free_Queue &operator= (const Lock_Free_Queue &) = delete;

  /**
   * @brief Attempt to enqueue ITEM without blocking.
   * @pre None.
   * @post If true is returned, ITEM was moved into the ring.
   * @returns True on success, false if the queue is currently full.
   * @note Thread-safety: Lock-free CAS loop.
   */
  bool
  try_enqueue (T item) noexcept
  {
    Slot *slot = nullptr;
    size_t pos = enqueue_pos_.load (std::memory_order_relaxed);
    for (;;)
      {
        slot = &slots_[pos & index_mask];
        const size_t seq = slot->sequence.load (std::memory_order_acquire);
        const intptr_t diff =
          static_cast<intptr_t> (seq) - static_cast<intptr_t> (pos);
        if (diff == 0)
          {
            if (enqueue_pos_.compare_exchange_weak (
                  pos, pos + 1U, std::memory_order_relaxed))
              {
                break;
              }
          }
        else if (diff < 0)
          {
            return false;
          }
        else
          {
            pos = enqueue_pos_.load (std::memory_order_relaxed);
          }
      }

    slot->storage = std::move (item);
    slot->sequence.store (pos + 1U, std::memory_order_release);
    return true;
  }

  /**
   * @brief Attempt to dequeue an item without blocking.
   * @pre None.
   * @post If engaged optional is returned, the front item was moved out.
   * @returns std::optional<T> containing the dequeued item, or std::nullopt.
   * @note Thread-safety: Lock-free CAS loop.
   */
  std::optional<T>
  try_dequeue (void) noexcept
  {
    Slot *slot = nullptr;
    size_t pos = dequeue_pos_.load (std::memory_order_relaxed);
    for (;;)
      {
        slot = &slots_[pos & index_mask];
        const size_t seq = slot->sequence.load (std::memory_order_acquire);
        const intptr_t diff =
          static_cast<intptr_t> (seq) - static_cast<intptr_t> (pos + 1U);
        if (diff == 0)
          {
            if (dequeue_pos_.compare_exchange_weak (
                  pos, pos + 1U, std::memory_order_relaxed))
              {
                break;
              }
          }
        else if (diff < 0)
          {
            return std::nullopt;
          }
        else
          {
            pos = dequeue_pos_.load (std::memory_order_relaxed);
          }
      }

    T result = std::move (*slot->storage);
    slot->storage.reset ();
    slot->sequence.store (pos + Capacity, std::memory_order_release);
    return result;
  }

  /**
   * @brief Return approximate number of queued items.
   * @pre None.
   * @post Returns approximate count in [0, Capacity].
   * @returns Approximate queue depth.
   * @note Thread-safety: Lock-free relaxed load.
   */
  size_t
  approx_size (void) const noexcept
  {
    const size_t enq = enqueue_pos_.load (std::memory_order_relaxed);
    const size_t deq = dequeue_pos_.load (std::memory_order_relaxed);
    return (enq >= deq) ? (enq - deq) : 0U;
  }

  static constexpr size_t
  capacity (void) noexcept
  {
    return Capacity;
  }

private:
  static constexpr size_t index_mask = Capacity - 1U;

  struct alignas (SURYA_CACHELINE_BYTES) Slot
  {
    std::atomic<size_t> sequence {0U};
    std::optional<T> storage {};
  };

  alignas (SURYA_CACHELINE_BYTES) std::atomic<size_t> enqueue_pos_ {0U};
  alignas (SURYA_CACHELINE_BYTES) std::atomic<size_t> dequeue_pos_ {0U};
  std::array<Slot, Capacity> slots_ {};
};

} /* namespace surya::inline v1::core */

#endif /* SURYA_CORE_LOCKFREE_QUEUE_HPP */
