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

#include <surya/core/memory.hpp>

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>

namespace surya::inline v1::core
{

void *
aligned_alloc_64 (size_t bytes) noexcept
{
  if (bytes == 0U)
    {
      return nullptr;
    }
  const size_t padded =
    compute_aligned_pitch (bytes, SURYA_CACHELINE_BYTES);
  return std::aligned_alloc (SURYA_CACHELINE_BYTES, padded);
}

void
aligned_free_64 (void *ptr) noexcept
{
  std::free (ptr);
}

Arena_Allocator::Arena_Allocator (size_t initial_capacity_bytes)
{
  const size_t cap =
    compute_aligned_pitch (initial_capacity_bytes > 0U
                             ? initial_capacity_bytes
                             : 4096U,
                           SURYA_CACHELINE_BYTES);
  base_ = static_cast<uint8_t *> (aligned_alloc_64 (cap));
  if (base_ != nullptr)
    {
      capacity_ = cap;
    }
}

Arena_Allocator::~Arena_Allocator (void) noexcept
{
  for (void *blk : overflow_blocks_)
    {
      aligned_free_64 (blk);
    }
  aligned_free_64 (base_);
}

Arena_Allocator::Arena_Allocator (Arena_Allocator &&other) noexcept
  : base_ (other.base_),
    capacity_ (other.capacity_),
    offset_ (other.offset_),
    overflow_blocks_ (std::move (other.overflow_blocks_))
{
  other.base_ = nullptr;
  other.capacity_ = 0U;
  other.offset_ = 0U;
}

Arena_Allocator &
Arena_Allocator::operator= (Arena_Allocator &&other) noexcept
{
  if (this != &other)
    {
      for (void *blk : overflow_blocks_)
        {
          aligned_free_64 (blk);
        }
      aligned_free_64 (base_);
      base_ = other.base_;
      capacity_ = other.capacity_;
      offset_ = other.offset_;
      overflow_blocks_ = std::move (other.overflow_blocks_);
      other.base_ = nullptr;
      other.capacity_ = 0U;
      other.offset_ = 0U;
    }
  return *this;
}

void *
Arena_Allocator::allocate (size_t bytes, size_t alignment) noexcept
{
  if (bytes == 0U)
    {
      return nullptr;
    }
  const size_t align =
    alignment < SURYA_CACHELINE_BYTES ? SURYA_CACHELINE_BYTES : alignment;
  const size_t aligned_offset = compute_aligned_pitch (offset_, align);
  const size_t padded_bytes = compute_aligned_pitch (bytes, align);

  if (base_ != nullptr && aligned_offset + padded_bytes <= capacity_)
    {
      void *ptr = base_ + aligned_offset;
      offset_ = aligned_offset + padded_bytes;
      return ptr;
    }

  void *overflow = aligned_alloc_64 (padded_bytes);
  if (overflow != nullptr)
    {
      try
        {
          overflow_blocks_.push_back (overflow);
        }
      catch (...)
        {
          aligned_free_64 (overflow);
          return nullptr;
        }
    }
  return overflow;
}

void
Arena_Allocator::reset (void) noexcept
{
  offset_ = 0U;
  for (void *blk : overflow_blocks_)
    {
      aligned_free_64 (blk);
    }
  overflow_blocks_.clear ();
}

void
Arena_Allocator::rewind_to (size_t mark) noexcept
{
  if (mark <= offset_)
    {
      offset_ = mark;
    }
}

Aligned_Buffer_Pool::Aligned_Buffer_Pool (
  size_t max_buffers_per_bucket) noexcept
  : max_per_bucket_ (
      std::min (max_buffers_per_bucket, max_slots_per_bucket))
{
  for (auto &b : buckets_)
    {
      for (auto &s : b.slots)
        {
          s.store (nullptr, std::memory_order_relaxed);
        }
    }
}

Aligned_Buffer_Pool::~Aligned_Buffer_Pool (void) noexcept
{
  clear ();
}

size_t
Aligned_Buffer_Pool::bucket_index_for (size_t bytes,
                                       size_t &rounded_capacity) noexcept
{
  const size_t min_size = static_cast<size_t> (1U) << min_bucket_shift;
  const size_t target = bytes < min_size ? min_size : bytes;
  const size_t pow2 = std::bit_ceil (target);
  const size_t shift = static_cast<size_t> (std::countr_zero (pow2));
  const size_t idx =
    shift >= min_bucket_shift ? (shift - min_bucket_shift) : 0U;
  if (idx >= num_buckets)
    {
      rounded_capacity =
        compute_aligned_pitch (bytes, SURYA_CACHELINE_BYTES);
      return num_buckets;
    }
  rounded_capacity = pow2;
  return idx;
}

Aligned_Buffer_Pool::Allocation
Aligned_Buffer_Pool::acquire (size_t requested_bytes) noexcept
{
  Allocation res {};
  if (requested_bytes == 0U)
    {
      return res;
    }

  size_t cap = 0U;
  const size_t idx = bucket_index_for (requested_bytes, cap);
  if (idx < num_buckets)
    {
      auto &bucket = buckets_[idx];
      for (size_t s = 0U; s < max_per_bucket_; ++s)
        {
          void *cached =
            bucket.slots[s].exchange (nullptr, std::memory_order_acquire);
          if (cached != nullptr)
            {
              res.ptr = cached;
              res.capacity_bytes = cap;
              res.from_pool = true;
              hits_.fetch_add (1U, std::memory_order_relaxed);
              return res;
            }
        }
    }

  misses_.fetch_add (1U, std::memory_order_relaxed);
  res.ptr = aligned_alloc_64 (cap);
  if (res.ptr != nullptr)
    {
      res.capacity_bytes = cap;
      res.from_pool = true;
    }
  return res;
}

void
Aligned_Buffer_Pool::release (Allocation alloc) noexcept
{
  if (alloc.ptr == nullptr)
    {
      return;
    }
  if (!alloc.from_pool || alloc.capacity_bytes == 0U)
    {
      aligned_free_64 (alloc.ptr);
      return;
    }

  size_t cap = 0U;
  const size_t idx = bucket_index_for (alloc.capacity_bytes, cap);
  if (idx < num_buckets && cap == alloc.capacity_bytes)
    {
      auto &bucket = buckets_[idx];
      for (size_t s = 0U; s < max_per_bucket_; ++s)
        {
          void *expected = nullptr;
          if (bucket.slots[s].compare_exchange_strong (
                expected, alloc.ptr, std::memory_order_release,
                std::memory_order_relaxed))
            {
              return;
            }
        }
    }
  aligned_free_64 (alloc.ptr);
}

void
Aligned_Buffer_Pool::clear (void) noexcept
{
  for (auto &bucket : buckets_)
    {
      for (auto &slot : bucket.slots)
        {
          void *ptr = slot.exchange (nullptr, std::memory_order_acquire);
          if (ptr != nullptr)
            {
              aligned_free_64 (ptr);
            }
        }
    }
}

size_t
Aligned_Buffer_Pool::cache_hits (void) const noexcept
{
  return hits_.load (std::memory_order_relaxed);
}

size_t
Aligned_Buffer_Pool::cache_misses (void) const noexcept
{
  return misses_.load (std::memory_order_relaxed);
}

Aligned_Buffer_Pool &
Aligned_Buffer_Pool::global_pool (void) noexcept
{
  static Aligned_Buffer_Pool instance (32U);
  return instance;
}

} /* namespace surya::inline v1::core */
