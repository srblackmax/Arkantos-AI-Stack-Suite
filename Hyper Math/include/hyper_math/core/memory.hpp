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

#ifndef HYPER_MATH_CORE_MEMORY_HPP
#define HYPER_MATH_CORE_MEMORY_HPP

#include "hyper_math/core/config.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <span>

namespace hyper_math
{

/* Clean-room 64-byte cache-line aligned reference-counted memory block.
   Supports zero-copy view sharing across tensor slices, transpositions,
   and immutable columnar series while guaranteeing SIMD alignment.  */
template <typename T>
class slab_buffer
{
private:
  struct control_block
  {
    std::atomic<std::size_t> ref_count;
    std::size_t capacity;
    T *data_ptr;

    explicit
    control_block (std::size_t count)
      : ref_count (1), capacity (count), data_ptr (nullptr)
    {
      if (count > 0)
        {
          const std::size_t byte_size = count * sizeof (T);
          void *raw = ::operator new (byte_size,
                                      std::align_val_t (cache_line_bytes));
          data_ptr = static_cast<T *> (raw);
          if constexpr (!std::is_trivially_default_constructible_v<T>)
            {
              try
                {
                  std::uninitialized_value_construct_n (data_ptr, count);
                }
              catch (...)
                {
                  ::operator delete (raw,
                                     std::align_val_t (cache_line_bytes));
                  throw;
                }
            }
        }
    }

    ~control_block ()
    {
      if (data_ptr != nullptr)
        {
          if constexpr (!std::is_trivially_destructible_v<T>)
            {
              std::destroy_n (data_ptr, capacity);
            }
          ::operator delete ( static_cast<void *> (data_ptr),
                              std::align_val_t (cache_line_bytes));
        }
    }
  };

  control_block *block_;

  void
  retain () noexcept
  {
    if (block_ != nullptr)
      {
        block_->ref_count.fetch_add (1, std::memory_order_relaxed);
      }
  }

  void
  release () noexcept
  {
    if (block_ != nullptr)
      {
        if (block_->ref_count.fetch_sub (1, std::memory_order_acq_rel) == 1)
          {
            delete block_;
          }
        block_ = nullptr;
      }
  }

public:
  using value_type = T;

  slab_buffer () noexcept
    : block_ (nullptr)
  {
  }

  explicit
  slab_buffer (std::size_t count)
    : block_ (count > 0 ? new control_block (count) : nullptr)
  {
  }

  slab_buffer (std::size_t count, const T &fill_value)
    : slab_buffer (count)
  {
    if (block_ != nullptr && block_->data_ptr != nullptr)
      {
        std::fill_n (block_->data_ptr, count, fill_value);
      }
  }

  slab_buffer (const slab_buffer &other) noexcept
    : block_ (other.block_)
  {
    retain ();
  }

  slab_buffer (slab_buffer &&other) noexcept
    : block_ (other.block_)
  {
    other.block_ = nullptr;
  }

  slab_buffer &
  operator= (const slab_buffer &other) noexcept
  {
    if (this != &other)
      {
        release ();
        block_ = other.block_;
        retain ();
      }
    return *this;
  }

  slab_buffer &
  operator= (slab_buffer &&other) noexcept
  {
    if (this != &other)
      {
        release ();
        block_ = other.block_;
        other.block_ = nullptr;
      }
    return *this;
  }

  ~slab_buffer ()
  {
    release ();
  }

  HYPER_MATH_ALWAYS_INLINE T *
  data () noexcept
  {
    return block_ != nullptr ? block_->data_ptr : nullptr;
  }

  HYPER_MATH_ALWAYS_INLINE const T *
  data () const noexcept
  {
    return block_ != nullptr ? block_->data_ptr : nullptr;
  }

  HYPER_MATH_ALWAYS_INLINE std::size_t
  size () const noexcept
  {
    return block_ != nullptr ? block_->capacity : 0;
  }

  HYPER_MATH_ALWAYS_INLINE bool
  empty () const noexcept
  {
    return size () == 0;
  }

  std::size_t
  use_count () const noexcept
  {
    return block_ != nullptr
           ? block_->ref_count.load (std::memory_order_relaxed)
           : 0;
  }

  HYPER_MATH_ALWAYS_INLINE T &
  operator[] (std::size_t idx) noexcept
  {
    return block_->data_ptr[idx];
  }

  HYPER_MATH_ALWAYS_INLINE const T &
  operator[] (std::size_t idx) const noexcept
  {
    return block_->data_ptr[idx];
  }

  std::span<T>
  as_span () noexcept
  {
    return std::span<T> (data (), size ());
  }

  std::span<const T>
  as_span () const noexcept
  {
    return std::span<const T> (data (), size ());
  }

  slab_buffer
  clone () const
  {
    const std::size_t n = size ();
    slab_buffer copy (n);
    if (n > 0)
      {
        std::copy_n (data (), n, copy.data ());
      }
    return copy;
  }
};

} // namespace hyper_math

#endif /* HYPER_MATH_CORE_MEMORY_HPP */
