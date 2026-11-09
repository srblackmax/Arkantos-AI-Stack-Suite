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

#ifndef BASALT_INTERNAL_TENSOR_INTERNAL_HPP
#define BASALT_INTERNAL_TENSOR_INTERNAL_HPP

/* Internal tensor metadata, stride calculation, and multi-dtype element
   access helpers.  */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <basalt/autograd.hpp>
#include <basalt/tensor.hpp>
#include "bellows-compat/bellows_bridge.hpp"

namespace basalt
{

struct Autograd_Meta
{
  bool requires_grad = false;
  bool retain_grad = false;
  Tensor grad;
  std::shared_ptr<Autograd_Node> grad_fn;
  std::shared_ptr<Autograd_Node> grad_accumulator;
  size_t output_nr = 0;
  std::mutex mutex;
};

namespace detail
{

/* Normalize a possibly-negative dimension index in [-ndim, ndim - 1].  */
inline size_t
normalize_dim (ptrdiff_t dim, size_t ndim, bool allow_endpoint = false)
{
  ptrdiff_t rank = static_cast<ptrdiff_t> (ndim);
  ptrdiff_t limit = allow_endpoint ? (rank + 1) : rank;
  if (rank == 0 && !allow_endpoint && (dim == 0 || dim == -1))
    {
      return 0;
    }
  ptrdiff_t d = (dim < 0) ? (dim + limit) : dim;
  BASALT_ASSERT_SHAPE (d >= 0 && d < limit,
                       "dimension index out of range");
  return static_cast<size_t> (d);
}

/* Compute contiguous row-major strides for SHAPE.  */
inline Stride_Vector
contiguous_strides (const Shape_Vector &shape)
{
  Stride_Vector strides (shape.size (), 1);
  if (shape.empty ())
    {
      return strides;
    }
  ptrdiff_t running = 1;
  for (size_t i = shape.size (); i > 0; --i)
    {
      strides[i - 1] = running;
      size_t dim_sz = shape[i - 1];
      if (dim_sz > 1)
        {
          running *= static_cast<ptrdiff_t> (dim_sz);
        }
    }
  return strides;
}

/* Compute total number of elements for SHAPE.  */
inline size_t
compute_numel (const Shape_Vector &shape) noexcept
{
  size_t total = 1;
  for (size_t s : shape)
    {
      total *= s;
    }
  return total;
}

/* Return true if SHAPE and STRIDES describe a contiguous row-major layout.  */
inline bool
compute_is_contiguous (const Shape_Vector &shape,
                       const Stride_Vector &strides) noexcept
{
  if (compute_numel (shape) <= 1)
    return true;
  ptrdiff_t expected = 1;
  for (size_t i = shape.size (); i > 0; --i)
    {
      size_t d = i - 1;
      if (shape[d] > 1)
        {
          if (strides[d] != expected)
            return false;
          expected *= static_cast<ptrdiff_t> (shape[d]);
        }
    }
  return true;
}

/* Map a flat row-major logical index (0 <= linear < numel) to a signed
   element offset relative to storage_offset using SHAPE and STRIDES.  */
inline ptrdiff_t
linear_to_strided_offset (size_t linear,
                          const Shape_Vector &shape,
                          const Stride_Vector &strides) noexcept
{
  const size_t rank = shape.size ();
  if (rank == 1)
    return static_cast<ptrdiff_t> (linear) * strides[0];
  if (rank == 2)
    {
      size_t d1 = shape[1];
      size_t c0 = linear / d1;
      size_t c1 = linear % d1;
      return static_cast<ptrdiff_t> (c0) * strides[0]
             + static_cast<ptrdiff_t> (c1) * strides[1];
    }
  if (rank == 3)
    {
      size_t d2 = shape[2];
      size_t d1 = shape[1];
      size_t c2 = linear % d2;
      size_t rem = linear / d2;
      size_t c1 = rem % d1;
      size_t c0 = rem / d1;
      return static_cast<ptrdiff_t> (c0) * strides[0]
             + static_cast<ptrdiff_t> (c1) * strides[1]
             + static_cast<ptrdiff_t> (c2) * strides[2];
    }

  ptrdiff_t offset = 0;
  size_t rem = linear;
  for (size_t i = rank; i > 0; --i)
    {
      size_t d = i - 1;
      size_t extent = shape[d];
      if (extent > 0)
        {
          size_t coord = rem % extent;
          rem /= extent;
          offset += static_cast<ptrdiff_t> (coord) * strides[d];
        }
    }
  return offset;
}

/* Read one element at base pointer + signed element offset as double.  */
inline double
load_element_as_double (const void *base, ptrdiff_t elem_offset,
                        Dtype dtype) noexcept
{
  switch (dtype)
    {
    case Dtype::Float32:
      return static_cast<double> (
        static_cast<const float *> (base)[elem_offset]);
    case Dtype::Float64:
      return static_cast<const double *> (base)[elem_offset];
    case Dtype::Float16:
      return static_cast<double> (static_cast<float> (
        static_cast<const Float16 *> (base)[elem_offset]));
    case Dtype::BFloat16:
      return static_cast<double> (static_cast<float> (
        static_cast<const BFloat16 *> (base)[elem_offset]));
    case Dtype::Int32:
      return static_cast<double> (
        static_cast<const int32_t *> (base)[elem_offset]);
    case Dtype::Int64:
      return static_cast<double> (
        static_cast<const int64_t *> (base)[elem_offset]);
    case Dtype::UInt8:
      return static_cast<double> (
        static_cast<const uint8_t *> (base)[elem_offset]);
    case Dtype::Bool:
      return static_cast<const uint8_t *> (base)[elem_offset] != 0
             ? 1.0
             : 0.0;
    }
  return 0.0;
}

/* Write one double value into base pointer + signed element offset.  */
inline void
store_double_as_element (void *base, ptrdiff_t elem_offset,
                         Dtype dtype, double value) noexcept
{
  switch (dtype)
    {
    case Dtype::Float32:
      static_cast<float *> (base)[elem_offset] = static_cast<float> (value);
      break;
    case Dtype::Float64:
      static_cast<double *> (base)[elem_offset] = value;
      break;
    case Dtype::Float16:
      static_cast<Float16 *> (base)[elem_offset]
        = Float16 (static_cast<float> (value));
      break;
    case Dtype::BFloat16:
      static_cast<BFloat16 *> (base)[elem_offset]
        = BFloat16 (static_cast<float> (value));
      break;
    case Dtype::Int32:
      static_cast<int32_t *> (base)[elem_offset]
        = static_cast<int32_t> (value);
      break;
    case Dtype::Int64:
      static_cast<int64_t *> (base)[elem_offset]
        = static_cast<int64_t> (value);
      break;
    case Dtype::UInt8:
      static_cast<uint8_t *> (base)[elem_offset]
        = static_cast<uint8_t> (value);
      break;
    case Dtype::Bool:
      static_cast<uint8_t *> (base)[elem_offset]
        = (value != 0.0) ? uint8_t (1) : uint8_t (0);
      break;
    }
}

} /* namespace detail */
} /* namespace basalt */

#endif /* BASALT_INTERNAL_TENSOR_INTERNAL_HPP */
