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

#ifndef HYPER_MATH_CORE_BITMASK_HPP
#define HYPER_MATH_CORE_BITMASK_HPP

#include "hyper_math/core/config.hpp"
#include "hyper_math/core/memory.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>

namespace hyper_math
{

/* Dual-Bitmask Algebraic Engine component:
   Stores 1 bit per row/element in 64-bit words. A bit value of 1 indicates
   a valid (non-null / active) entry, while 0 indicates null / masked-out.
   Bitwise operations process 64 elements per CPU instruction and use
   hardware std::popcount for O(N/64) cardinality computation.  */
class validity_bitmap
{
private:
  slab_buffer<std::uint64_t> words_;
  std::size_t bit_length_;

  static constexpr std::size_t
  words_for_bits (std::size_t n_bits) noexcept
  {
    return (n_bits + 63) / 64;
  }

  void
  mask_trailing_bits () noexcept
  {
    const std::size_t rem = bit_length_ & 63;
    if (rem != 0 && words_.size () > 0)
      {
        const std::uint64_t tail_mask = (std::uint64_t (1) << rem) - 1;
        words_[words_.size () - 1] &= tail_mask;
      }
  }

public:
  validity_bitmap () noexcept
    : words_ (), bit_length_ (0)
  {
  }

  explicit
  validity_bitmap (std::size_t n_bits, bool initial_valid = true)
    : words_ (words_for_bits (n_bits),
              initial_valid ? ~std::uint64_t (0) : std::uint64_t (0)),
      bit_length_ (n_bits)
  {
    mask_trailing_bits ();
  }

  std::size_t
  size () const noexcept
  {
    return bit_length_;
  }

  std::size_t
  word_count () const noexcept
  {
    return words_.size ();
  }

  HYPER_MATH_ALWAYS_INLINE bool
  is_valid (std::size_t index) const noexcept
  {
    const std::size_t word_idx = index >> 6;
    const std::size_t bit_idx = index & 63;
    return ((words_[word_idx] >> bit_idx) & std::uint64_t (1)) != 0;
  }

  HYPER_MATH_ALWAYS_INLINE bool
  is_null (std::size_t index) const noexcept
  {
    return !is_valid (index);
  }

  HYPER_MATH_ALWAYS_INLINE void
  set_valid (std::size_t index, bool valid = true)
  {
    if (words_.use_count () > 1)
      {
        words_ = words_.clone ();
      }
    const std::size_t word_idx = index >> 6;
    const std::size_t bit_idx = index & 63;
    const std::uint64_t mask = std::uint64_t (1) << bit_idx;
    if (valid)
      {
        words_[word_idx] |= mask;
      }
    else
      {
        words_[word_idx] &= ~mask;
      }
  }

  std::size_t
  valid_count () const noexcept
  {
    std::size_t total = 0;
    const std::size_t n_words = words_.size ();
    for (std::size_t i = 0; i < n_words; ++i)
      {
        total += static_cast<std::size_t> (std::popcount (words_[i]));
      }
    return total;
  }

  std::size_t
  null_count () const noexcept
  {
    return bit_length_ - valid_count ();
  }

  bool
  all_valid () const noexcept
  {
    return valid_count () == bit_length_;
  }

  bool
  any_valid () const noexcept
  {
    return valid_count () > 0;
  }

  validity_bitmap
  clone () const
  {
    validity_bitmap out;
    out.words_ = words_.clone ();
    out.bit_length_ = bit_length_;
    return out;
  }

  validity_bitmap
  operator& (const validity_bitmap &other) const
  {
    if (bit_length_ != other.bit_length_)
      {
        throw shape_error ("validity_bitmap bitwise AND length mismatch");
      }
    validity_bitmap out (bit_length_, false);
    const std::size_t n_words = words_.size ();
    for (std::size_t i = 0; i < n_words; ++i)
      {
        out.words_[i] = words_[i] & other.words_[i];
      }
    out.mask_trailing_bits ();
    return out;
  }

  validity_bitmap
  operator| (const validity_bitmap &other) const
  {
    if (bit_length_ != other.bit_length_)
      {
        throw shape_error ("validity_bitmap bitwise OR length mismatch");
      }
    validity_bitmap out (bit_length_, false);
    const std::size_t n_words = words_.size ();
    for (std::size_t i = 0; i < n_words; ++i)
      {
        out.words_[i] = words_[i] | other.words_[i];
      }
    out.mask_trailing_bits ();
    return out;
  }

  validity_bitmap
  operator~ () const
  {
    validity_bitmap out (bit_length_, false);
    const std::size_t n_words = words_.size ();
    for (std::size_t i = 0; i < n_words; ++i)
      {
        out.words_[i] = ~words_[i];
      }
    out.mask_trailing_bits ();
    return out;
  }
};

} // namespace hyper_math

#endif /* HYPER_MATH_CORE_BITMASK_HPP */
