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

#ifndef SURYA_CORE_MEMORY_HPP
#define SURYA_CORE_MEMORY_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <vector>
#include <surya/core/config.hpp>
#include <surya/core/error.hpp>

namespace surya::inline v1::core
{

/**
 * @brief Round ROW_BYTES up to the nearest multiple of ALIGNMENT (power of 2).
 * @pre ALIGNMENT is a non-zero power of two.
 * @post Returned value >= ROW_BYTES and is divisible by ALIGNMENT.
 * @returns Aligned byte pitch.
 * @note Thread-safety: Pure constexpr function; safe across all threads.
 */
constexpr size_t
compute_aligned_pitch (size_t row_bytes,
                       size_t alignment = SURYA_CACHELINE_BYTES) noexcept
{
  return (row_bytes + (alignment - 1U)) & ~(alignment - 1U);
}

/**
 * @brief Allocate a 64-byte cache-line aligned heap block of BYTES size.
 * @pre None.
 * @post Returns 64-byte aligned pointer or nullptr on allocation failure.
 * @returns Pointer to aligned heap block.
 * @note Thread-safety: Safe to call concurrently from any thread.
 */
SURYA_API void *
aligned_alloc_64 (size_t bytes) noexcept;

/**
 * @brief Free a memory block previously returned by aligned_alloc_64.
 * @pre PTR was returned by aligned_alloc_64 or is nullptr.
 * @post Backing memory is released to the OS allocator.
 * @note Thread-safety: Safe to call concurrently from any thread.
 */
SURYA_API void
aligned_free_64 (void *ptr) noexcept;

/**
 * @brief Monotonic bump-pointer arena allocator for zero-syscall scratch
 *        allocations inside image processing and detection kernels.
 * @note Thread-safety: Instances are thread-local or per-request (not shared
 *       without external synchronization).
 */
class SURYA_API Arena_Allocator
{
public:
  explicit Arena_Allocator (size_t initial_capacity_bytes = 1048576U);

  ~Arena_Allocator (void) noexcept;

  Arena_Allocator (const Arena_Allocator &) = delete;
  Arena_Allocator &operator= (const Arena_Allocator &) = delete;

  Arena_Allocator (Arena_Allocator &&other) noexcept;
  Arena_Allocator &operator= (Arena_Allocator &&other) noexcept;

  /**
   * @brief Allocate BYTES aligned to ALIGNMENT from the arena.
   * @pre ALIGNMENT is a power of two <= 64.
   * @post Advances offset or allocates overflow block.
   * @returns Aligned pointer or nullptr if OS allocation fails.
   * @note Thread-safety: Single-thread / per-request owned.
   */
  void *
  allocate (size_t bytes, size_t alignment = SURYA_CACHELINE_BYTES) noexcept;

  /**
   * @brief Allocate a typed array of COUNT elements aligned to 64 bytes.
   */
  template <typename T>
  T *
  allocate_array (size_t count) noexcept
  {
    return static_cast<T *> (
      allocate (count * sizeof (T), SURYA_CACHELINE_BYTES));
  }

  /**
   * @brief Reset the primary arena cursor to 0 without freeing backing memory.
   * @pre None.
   * @post used_bytes() == 0.
   */
  void
  reset (void) noexcept;

  size_t
  used_bytes (void) const noexcept
  {
    return offset_;
  }

  size_t
  capacity_bytes (void) const noexcept
  {
    return capacity_;
  }

  void
  rewind_to (size_t mark) noexcept;

private:
  uint8_t *base_ = nullptr;
  size_t capacity_ = 0;
  size_t offset_ = 0;
  std::vector<void *> overflow_blocks_;
};

/**
 * @brief C++23 std::pmr::memory_resource adapter backed by Arena_Allocator
 *        for zero-heap-churn std::pmr::vector / tokenizers / scratch buffers.
 * @note Thread-safety: Per-request monotonic resource (not thread-shared).
 */
class SURYA_API Pmr_Arena_Resource final : public std::pmr::memory_resource
{
public:
  explicit Pmr_Arena_Resource (Arena_Allocator &arena) noexcept
    : arena_ (arena)
  {
  }

  ~Pmr_Arena_Resource (void) noexcept override = default;

protected:
  void *
  do_allocate (size_t bytes, size_t alignment) override
  {
    return arena_.allocate (bytes, alignment);
  }

  void
  do_deallocate (void *p, size_t bytes, size_t alignment) noexcept override
  {
    (void) p;
    (void) bytes;
    (void) alignment;
  }

  bool
  do_is_equal (
    const std::pmr::memory_resource &other) const noexcept override
  {
    return this == &other;
  }

private:
  Arena_Allocator &arena_;
};

/**
 * @brief RAII scope guard that rewinds an Arena_Allocator on destruction.
 */
class Scoped_Arena_Mark
{
public:
  explicit Scoped_Arena_Mark (Arena_Allocator &arena) noexcept
    : arena_ (arena), mark_ (arena.used_bytes ())
  {
  }

  ~Scoped_Arena_Mark (void) noexcept
  {
    arena_.rewind_to (mark_);
  }

  Scoped_Arena_Mark (const Scoped_Arena_Mark &) = delete;
  Scoped_Arena_Mark &operator= (const Scoped_Arena_Mark &) = delete;

private:
  Arena_Allocator &arena_;
  size_t mark_;
};

/**
 * @brief Lock-free size-bucketed pool of 64-byte aligned memory buffers.
 *        Uses atomic CAS slot tables per power-of-two bucket with zero
 *        std::mutex on the hot path.
 * @note Thread-safety: Lock-free and safe for concurrent acquire/release.
 */
class SURYA_API Aligned_Buffer_Pool
{
public:
  static constexpr size_t num_buckets = 28U;
  static constexpr size_t min_bucket_shift = 6U; /* 64 bytes */
  static constexpr size_t max_slots_per_bucket = 32U;

  struct Allocation
  {
    void *ptr = nullptr;
    size_t capacity_bytes = 0;
    bool from_pool = false;
  };

  explicit Aligned_Buffer_Pool (
    size_t max_buffers_per_bucket = 16U) noexcept;

  ~Aligned_Buffer_Pool (void) noexcept;

  Aligned_Buffer_Pool (const Aligned_Buffer_Pool &) = delete;
  Aligned_Buffer_Pool &operator= (const Aligned_Buffer_Pool &) = delete;

  /**
   * @brief Acquire a 64-byte aligned buffer of at least REQUESTED_BYTES.
   * @pre None.
   * @post Returns aligned buffer >= REQUESTED_BYTES or nullptr on OOM.
   * @returns Allocation descriptor.
   * @note Thread-safety: Lock-free atomic exchange.
   */
  Allocation
  acquire (size_t requested_bytes) noexcept;

  /**
   * @brief Return an allocation to the pool or free it if the bucket is full.
   * @pre ALLOC was returned by acquire().
   * @post Memory is recycled in lock-free slot or freed to OS.
   * @note Thread-safety: Lock-free atomic CAS.
   */
  void
  release (Allocation alloc) noexcept;

  /**
   * @brief Release all cached buffers back to the OS heap.
   * @pre None.
   * @post All slots in all buckets are nullptr.
   * @note Thread-safety: Lock-free atomic exchange per slot.
   */
  void
  clear (void) noexcept;

  size_t
  cache_hits (void) const noexcept;

  size_t
  cache_misses (void) const noexcept;

  /**
   * @brief Process-wide default buffer pool singleton.
   */
  static Aligned_Buffer_Pool &
  global_pool (void) noexcept;

private:
  static size_t
  bucket_index_for (size_t bytes, size_t &rounded_capacity) noexcept;

  struct Bucket
  {
    std::array<std::atomic<void *>, max_slots_per_bucket> slots {};
  };

  size_t max_per_bucket_;
  std::array<Bucket, num_buckets> buckets_ {};
  std::atomic<size_t> hits_ {0U};
  std::atomic<size_t> misses_ {0U};
};

} /* namespace surya::inline v1::core */

#endif /* SURYA_CORE_MEMORY_HPP */
