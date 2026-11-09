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

#ifndef HYPER_MATH_TENSOR_LATTICE_HPP
#define HYPER_MATH_TENSOR_LATTICE_HPP

#include "hyper_math/core/config.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <vector>

namespace hyper_math::tensor
{

using shape_type = std::vector<std::size_t>;
using stride_type = std::vector<std::ptrdiff_t>;

/* Represents a 1-D slice [start : stop : step] along an axis, supporting
   open bounds and negative indices like NumPy.  */
struct slice_range
{
  std::optional<std::ptrdiff_t> start;
  std::optional<std::ptrdiff_t> stop;
  std::ptrdiff_t step;

  constexpr
  slice_range () noexcept
    : start (std::nullopt), stop (std::nullopt), step (1)
  {
  }

  constexpr
  slice_range (std::optional<std::ptrdiff_t> s_start,
               std::optional<std::ptrdiff_t> s_stop,
               std::ptrdiff_t s_step = 1) noexcept
    : start (s_start), stop (s_stop), step (s_step)
  {
  }

  static constexpr slice_range
  all () noexcept
  {
    return slice_range ();
  }

  static constexpr slice_range
  single (std::ptrdiff_t idx) noexcept
  {
    return slice_range (idx, idx + 1, 1);
  }
};

/* Rank-agnostic affine lattice descriptor mapping logical N-D coordinates
   (i_0, i_1, ..., i_{R-1}) to physical storage offset:
     offset + sum_{k=0}^{R-1} i_k * stride_k.  */
class lattice_descriptor
{
private:
  shape_type shape_;
  stride_type strides_;
  std::ptrdiff_t offset_;
  std::size_t total_size_;
  bool contiguous_;

  void
  recompute_metadata () noexcept
  {
    if (shape_.empty ())
      {
        total_size_ = 0;
        contiguous_ = true;
        return;
      }

    total_size_ = 1;
    for (std::size_t dim : shape_)
      {
        total_size_ *= dim;
      }

    std::ptrdiff_t expected_stride = 1;
    contiguous_ = (offset_ == 0);
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t> (shape_.size ()) - 1;
         i >= 0; --i)
      {
        const std::size_t idx = static_cast<std::size_t> (i);
        if (shape_[idx] > 1 && strides_[idx] != expected_stride)
          {
            contiguous_ = false;
          }
        expected_stride *= static_cast<std::ptrdiff_t> (shape_[idx]);
      }
  }

public:
  lattice_descriptor () noexcept
    : shape_ (), strides_ (), offset_ (0), total_size_ (0), contiguous_ (true)
  {
  }

  explicit
  lattice_descriptor (shape_type dims)
    : shape_ (std::move (dims)),
      strides_ (shape_.size (), 1),
      offset_ (0),
      total_size_ (0),
      contiguous_ (true)
  {
    if (!shape_.empty ())
      {
        std::ptrdiff_t running = 1;
        for (std::ptrdiff_t i =
               static_cast<std::ptrdiff_t> (shape_.size ()) - 1;
             i >= 0; --i)
          {
            const std::size_t idx = static_cast<std::size_t> (i);
            strides_[idx] = running;
            running *= static_cast<std::ptrdiff_t> (shape_[idx]);
          }
      }
    recompute_metadata ();
  }

  lattice_descriptor (shape_type dims,
                      stride_type str,
                      std::ptrdiff_t off = 0)
    : shape_ (std::move (dims)),
      strides_ (std::move (str)),
      offset_ (off),
      total_size_ (0),
      contiguous_ (false)
  {
    if (shape_.size () != strides_.size ())
      {
        throw shape_error ("shape and strides rank mismatch in lattice");
      }
    recompute_metadata ();
  }

  const shape_type &
  shape () const noexcept
  {
    return shape_;
  }

  const stride_type &
  strides () const noexcept
  {
    return strides_;
  }

  std::ptrdiff_t
  offset () const noexcept
  {
    return offset_;
  }

  std::size_t
  ndim () const noexcept
  {
    return shape_.size ();
  }

  std::size_t
  size () const noexcept
  {
    return total_size_;
  }

  bool
  is_contiguous () const noexcept
  {
    return contiguous_;
  }

  /* Translate flat logical index in [0, size()) into physical offset.  */
  HYPER_MATH_ALWAYS_INLINE std::ptrdiff_t
  flat_to_physical (std::size_t flat_idx) const noexcept
  {
    if (contiguous_)
      {
        return static_cast<std::ptrdiff_t> (flat_idx);
      }
    std::ptrdiff_t phys = offset_;
    std::size_t rem = flat_idx;
    for (std::ptrdiff_t d = static_cast<std::ptrdiff_t> (shape_.size ()) - 1;
         d >= 0; --d)
      {
        const std::size_t axis = static_cast<std::size_t> (d);
        const std::size_t extent = shape_[axis];
        const std::size_t coord = extent > 0 ? (rem % extent) : 0;
        rem = extent > 0 ? (rem / extent) : 0;
        phys += static_cast<std::ptrdiff_t> (coord) * strides_[axis];
      }
    return phys;
  }

  HYPER_MATH_ALWAYS_INLINE std::ptrdiff_t
  coords_to_physical_2d (std::size_t r, std::size_t c) const
  {
    if (shape_.size () != 2 || r >= shape_[0] || c >= shape_[1])
      {
        throw index_error ("2D tensor coordinate out of bounds");
      }
    return offset_
           + static_cast<std::ptrdiff_t> (r) * strides_[0]
           + static_cast<std::ptrdiff_t> (c) * strides_[1];
  }

  std::ptrdiff_t
  coords_to_physical (const std::vector<std::size_t> &coords) const
  {
    if (coords.size () != shape_.size ())
      {
        throw index_error ("coordinate count does not match tensor rank");
      }
    std::ptrdiff_t phys = offset_;
    for (std::size_t i = 0; i < shape_.size (); ++i)
      {
        if (coords[i] >= shape_[i])
          {
            throw index_error ("tensor coordinate out of bounds along axis "
                               + std::to_string (i));
          }
        phys += static_cast<std::ptrdiff_t> (coords[i]) * strides_[i];
      }
    return phys;
  }

  /* Zero-copy slice transformation across arbitrary axes.  */
  lattice_descriptor
  slice (const std::vector<slice_range> &ranges) const
  {
    if (ranges.size () > shape_.size ())
      {
        throw index_error ("too many slice specifications for tensor rank");
      }

    shape_type new_shape = shape_;
    stride_type new_strides = strides_;
    std::ptrdiff_t new_offset = offset_;

    for (std::size_t d = 0; d < ranges.size (); ++d)
      {
        const std::ptrdiff_t dim_len =
          static_cast<std::ptrdiff_t> (shape_[d]);
        const std::ptrdiff_t step = ranges[d].step;
        if (step == 0)
          {
            throw index_error ("slice step cannot be zero");
          }

        std::ptrdiff_t start =
          ranges[d].start.value_or (step > 0 ? 0 : dim_len - 1);
        std::ptrdiff_t stop =
          ranges[d].stop.value_or (step > 0 ? dim_len : -dim_len - 1);

        if (start < 0)
          {
            start += dim_len;
          }
        if (stop < 0 && ranges[d].stop.has_value ())
          {
            stop += dim_len;
          }

        if (step > 0)
          {
            start = std::clamp<std::ptrdiff_t> (start, 0, dim_len);
            stop = std::clamp<std::ptrdiff_t> (stop, 0, dim_len);
            const std::ptrdiff_t span = std::max<std::ptrdiff_t> (0, stop - start);
            new_shape[d] = static_cast<std::size_t> ((span + step - 1) / step);
          }
        else
          {
            start = std::clamp<std::ptrdiff_t> (start, -1, dim_len - 1);
            stop = std::clamp<std::ptrdiff_t> (stop, -1, dim_len - 1);
            const std::ptrdiff_t span = std::max<std::ptrdiff_t> (0, start - stop);
            const std::ptrdiff_t abs_step = -step;
            new_shape[d] =
              static_cast<std::size_t> ((span + abs_step - 1) / abs_step);
          }

        if (new_shape[d] > 0)
          {
            new_offset += start * strides_[d];
          }
        new_strides[d] = strides_[d] * step;
      }

    for (std::size_t s : new_shape)
      {
        if (s == 0)
          {
            new_offset = 0;
            break;
          }
      }

    return lattice_descriptor (std::move (new_shape),
                               std::move (new_strides),
                               new_offset);
  }

  /* Zero-copy axis permutation / transposition.  */
  lattice_descriptor
  transpose (const std::vector<std::size_t> &axes = {}) const
  {
    const std::size_t r = shape_.size ();
    if (r <= 1)
      {
        return *this;
      }

    shape_type new_shape (r);
    stride_type new_strides (r);

    if (axes.empty ())
      {
        for (std::size_t i = 0; i < r; ++i)
          {
            new_shape[i] = shape_[r - 1 - i];
            new_strides[i] = strides_[r - 1 - i];
          }
      }
    else
      {
        if (axes.size () != r)
          {
            throw shape_error ("transpose axes permutation length mismatch");
          }
        std::vector<bool> seen (r, false);
        for (std::size_t i = 0; i < r; ++i)
          {
            if (axes[i] >= r || seen[axes[i]])
              {
                throw shape_error ("invalid transpose permutation axis");
              }
            seen[axes[i]] = true;
            new_shape[i] = shape_[axes[i]];
            new_strides[i] = strides_[axes[i]];
          }
      }

    return lattice_descriptor (std::move (new_shape),
                               std::move (new_strides),
                               offset_);
  }

  /* Broadcast this lattice to target_shape by setting strides of singleton
     dimensions to 0 (zero-copy replication).  */
  lattice_descriptor
  broadcast_to (const shape_type &target_shape) const
  {
    if (target_shape.size () < shape_.size ())
      {
        throw shape_error ("cannot broadcast lattice to lower rank");
      }

    const std::size_t rank_diff = target_shape.size () - shape_.size ();
    stride_type new_strides (target_shape.size (), 0);

    for (std::size_t i = 0; i < shape_.size (); ++i)
      {
        const std::size_t src_dim = shape_[i];
        const std::size_t dst_dim = target_shape[i + rank_diff];
        if (src_dim == dst_dim)
          {
            new_strides[i + rank_diff] = strides_[i];
          }
        else if (src_dim == 1)
          {
            new_strides[i + rank_diff] = 0;
          }
        else
          {
            throw shape_error ("incompatible dimensions in broadcast_to");
          }
      }

    return lattice_descriptor (target_shape, std::move (new_strides), offset_);
  }
};

/* Clean-room Canonical Contradiction-Free Stride Broadcast Resolver:
   Computes the unified broadcasted shape for two arbitrary N-D shapes.  */
inline shape_type
broadcast_shapes (const shape_type &a, const shape_type &b)
{
  const std::size_t max_rank = std::max (a.size (), b.size ());
  shape_type result (max_rank, 1);

  for (std::size_t i = 0; i < max_rank; ++i)
    {
      const std::size_t dim_a =
        (i < a.size ()) ? a[a.size () - 1 - i] : 1;
      const std::size_t dim_b =
        (i < b.size ()) ? b[b.size () - 1 - i] : 1;

      if (dim_a != dim_b && dim_a != 1 && dim_b != 1)
        {
          std::ostringstream oss;
          oss << "shapes cannot be broadcast together ("
              << dim_a << " vs " << dim_b << ")";
          throw shape_error (oss.str ());
        }
      result[max_rank - 1 - i] = std::max (dim_a, dim_b);
    }

  return result;
}

inline std::string
format_shape (const shape_type &s)
{
  std::ostringstream oss;
  oss << "(";
  for (std::size_t i = 0; i < s.size (); ++i)
    {
      oss << s[i];
      if (i + 1 < s.size () || s.size () == 1)
        {
          oss << (i + 1 < s.size () ? ", " : ",");
        }
    }
  oss << ")";
  return oss.str ();
}

} // namespace hyper_math::tensor

#endif /* HYPER_MATH_TENSOR_LATTICE_HPP */
