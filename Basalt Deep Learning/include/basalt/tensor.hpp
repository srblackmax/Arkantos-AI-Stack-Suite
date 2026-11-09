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

#ifndef BASALT_TENSOR_HPP
#define BASALT_TENSOR_HPP

/* N-dimensional strided Tensor built on Bellows and Hyper Math storage,
   supporting zero-copy strided views, multi-dimensional broadcasting,
   mixed-precision dtypes, CPU/CUDA device dispatch, and dynamic reverse-mode
   automatic differentiation.  */

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>
#include <basalt/config.hpp>
#include <basalt/device.hpp>
#include <basalt/dtype.hpp>
#include <basalt/error.hpp>

namespace bellows
{
class data_table;
}

namespace basalt
{

namespace bellows_compat
{
class Bellows_Storage;
}

struct Autograd_Meta;
class Autograd_Node;

using Shape_Vector = std::vector<size_t>;
using Stride_Vector = std::vector<ptrdiff_t>;

/* Options controlling tensor construction (dtype, device, requires_grad).  */
struct BASALT_API Tensor_Options
{
  Dtype dtype = Dtype::Float32;
  Device device = Device::cpu ();
  bool requires_grad = false;

  constexpr Tensor_Options (void) noexcept = default;

  constexpr Tensor_Options
  with_dtype (Dtype dt) const noexcept
  {
    Tensor_Options out = *this;
    out.dtype = dt;
    return out;
  }

  constexpr Tensor_Options
  with_device (Device dev) const noexcept
  {
    Tensor_Options out = *this;
    out.device = dev;
    return out;
  }

  constexpr Tensor_Options
  with_requires_grad (bool rg) const noexcept
  {
    Tensor_Options out = *this;
    out.requires_grad = rg;
    return out;
  }
};

/* Primary N-dimensional strided tensor handle.
   Copying a Tensor is a lightweight O(1) shallow handle copy sharing the
   underlying Bellows_Storage and Autograd_Meta; use clone() for a deep copy
   and detach() for a view detached from the autograd graph.  */
class BASALT_API Tensor
{
public:
  Tensor (void) noexcept = default;

  Tensor (const Shape_Vector &shape,
          Tensor_Options options = Tensor_Options ());

  Tensor (std::shared_ptr<bellows_compat::Bellows_Storage> storage,
          size_t offset_elems,
          Shape_Vector shape,
          Stride_Vector strides,
          Dtype dtype,
          Device device,
          bool requires_grad = false);

  ~Tensor (void) noexcept = default;

  Tensor (const Tensor &) noexcept = default;
  Tensor &operator= (const Tensor &) noexcept = default;
  Tensor (Tensor &&) noexcept = default;
  Tensor &operator= (Tensor &&) noexcept = default;

  /* Return true if this handle refers to allocated storage.  */
  bool
  defined (void) const noexcept;

  /* Number of tensor dimensions (rank).  */
  size_t
  ndim (void) const noexcept;

  /* Total number of logical elements across all dimensions.  */
  size_t
  numel (void) const noexcept;

  /* Size of dimension DIM (supports negative indexing from the end).  */
  size_t
  size (ptrdiff_t dim) const;

  const Shape_Vector &
  shape (void) const noexcept;

  const Stride_Vector &
  strides (void) const noexcept;

  size_t
  storage_offset (void) const noexcept;

  Dtype
  dtype (void) const noexcept;

  Device
  device (void) const noexcept;

  Tensor_Options
  options (void) const noexcept;

  /* Return true if elements are laid out contiguously in row-major order
     with positive non-overlapping strides.  */
  bool
  is_contiguous (void) const noexcept;

  /* Return a contiguous tensor containing the same values (returns *this if
     already contiguous).  */
  Tensor
  contiguous (void) const;

  /* Deep-copy tensor storage (recorded in autograd if requires_grad).  */
  Tensor
  clone (void) const;

  /* Return a tensor sharing storage but detached from the autograd graph.  */
  Tensor
  detach (void) const;

  /* Copy or cast tensor to target device and/or dtype (recorded in autograd
     when floating-point gradients are active).  */
  Tensor
  to (Device target_device) const;

  Tensor
  to (Dtype target_dtype) const;

  Tensor
  to (Device target_device, Dtype target_dtype) const;

  Tensor
  cpu (void) const
  {
    return to (Device::cpu ());
  }

  Tensor
  cuda (size_t device_index = 0) const
  {
    return to (Device::cuda (device_index));
  }

  /* Strided view and shape manipulation operations.  */
  Tensor
  view (const Shape_Vector &new_shape) const;

  Tensor
  reshape (const Shape_Vector &new_shape) const;

  Tensor
  transpose (ptrdiff_t dim0, ptrdiff_t dim1) const;

  Tensor
  t (void) const;

  Tensor
  permute (const std::vector<ptrdiff_t> &dims) const;

  Tensor
  slice (ptrdiff_t dim, ptrdiff_t start, ptrdiff_t stop,
         ptrdiff_t step = 1) const;

  Tensor
  narrow (ptrdiff_t dim, ptrdiff_t start, size_t length) const
  {
    return slice (dim, start, start + static_cast<ptrdiff_t> (length), 1);
  }

  Tensor
  select (ptrdiff_t dim, ptrdiff_t index) const;

  Tensor
  squeeze (ptrdiff_t dim = -9999) const;

  Tensor
  unsqueeze (ptrdiff_t dim) const;

  Tensor
  expand (const Shape_Vector &target_shape) const;

  Tensor
  broadcast_to (const Shape_Vector &target_shape) const
  {
    return expand (target_shape);
  }

  Tensor
  flatten (ptrdiff_t start_dim = 0, ptrdiff_t end_dim = -1) const;

  /* Raw and typed pointer access (accounting for storage_offset).  */
  void *
  raw_data_ptr (void);

  const void *
  raw_data_ptr (void) const;

  template <typename T>
  T *
  data_ptr (void)
  {
    BASALT_ASSERT_DTYPE (dtype () == dtype_of<T> (),
                         "data_ptr<T> dtype mismatch");
    return static_cast<T *> (raw_data_ptr ());
  }

  template <typename T>
  const T *
  data_ptr (void) const
  {
    BASALT_ASSERT_DTYPE (dtype () == dtype_of<T> (),
                         "data_ptr<T> const dtype mismatch");
    return static_cast<const T *> (raw_data_ptr ());
  }

  /* Read or write an element at flat logical index (0 <= idx < numel) as
     double regardless of underlying dtype or device.  */
  double
  read_scalar_at (size_t linear_index) const;

  void
  write_scalar_at (size_t linear_index, double value);

  /* Copy all logical elements in row-major order into a host vector.  */
  std::vector<double>
  to_vector_double (void) const;

  std::vector<float>
  to_vector_float (void) const;

  std::vector<int64_t>
  to_vector_int64 (void) const;

  /* Extract scalar value of a 1-element tensor.  */
  double
  item_double (void) const;

  float
  item_float (void) const
  {
    return static_cast<float> (item_double ());
  }

  int64_t
  item_int64 (void) const
  {
    return static_cast<int64_t> (item_double ());
  }

  template <typename T = double>
  T
  item (void) const
  {
    return static_cast<T> (item_double ());
  }

  /* In-place fill and mutation helpers (increment version counter).  */
  Tensor &
  fill_ (double value);

  Tensor &
  zero_ (void)
  {
    return fill_ (0.0);
  }

  Tensor &
  copy_ (const Tensor &src);

  Tensor &
  add_ (const Tensor &other, double alpha = 1.0);

  Tensor &
  mul_ (double scale);

  uint64_t
  version (void) const noexcept;

  void
  bump_version (void) noexcept;

  /* Autograd interface.  */
  bool
  requires_grad (void) const noexcept;

  Tensor &
  set_requires_grad (bool rg);

  bool
  is_leaf (void) const noexcept;

  void
  retain_grad (void);

  bool
  retains_grad (void) const noexcept;

  Tensor
  grad (void) const noexcept;

  void
  set_grad (const Tensor &g);

  void
  zero_grad (void);

  std::shared_ptr<Autograd_Node>
  grad_fn (void) const noexcept;

  void
  set_grad_fn (std::shared_ptr<Autograd_Node> node);

  std::shared_ptr<Autograd_Meta>
  autograd_meta (void) const noexcept;

  void
  backward (const Tensor &grad_output = Tensor (),
            bool retain_graph = false,
            bool create_graph = false) const;

  std::string
  to_string (void) const;

  /* Internal storage access.  */
  const std::shared_ptr<bellows_compat::Bellows_Storage> &
  storage (void) const noexcept;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

/* Factory functions.  */
BASALT_API Tensor
empty (const Shape_Vector &shape,
       Tensor_Options options = Tensor_Options ());

BASALT_API Tensor
zeros (const Shape_Vector &shape,
       Tensor_Options options = Tensor_Options ());

BASALT_API Tensor
ones (const Shape_Vector &shape,
      Tensor_Options options = Tensor_Options ());

BASALT_API Tensor
full (const Shape_Vector &shape, double fill_value,
      Tensor_Options options = Tensor_Options ());

BASALT_API Tensor
zeros_like (const Tensor &other);

BASALT_API Tensor
ones_like (const Tensor &other);

BASALT_API Tensor
full_like (const Tensor &other, double fill_value);

BASALT_API Tensor
eye (size_t n, size_t m = 0,
     Tensor_Options options = Tensor_Options ());

BASALT_API Tensor
arange (double start, double stop, double step = 1.0,
        Tensor_Options options = Tensor_Options ());

BASALT_API Tensor
linspace (double start, double stop, size_t steps,
          Tensor_Options options = Tensor_Options ());

BASALT_API Tensor
rand (const Shape_Vector &shape,
      Tensor_Options options = Tensor_Options ());

BASALT_API Tensor
randn (const Shape_Vector &shape,
       Tensor_Options options = Tensor_Options ());

BASALT_API Tensor
randint (int64_t low, int64_t high, const Shape_Vector &shape,
         Tensor_Options options = Tensor_Options ().with_dtype (Dtype::Int64));

BASALT_API Tensor
from_blob (const void *data, const Shape_Vector &shape,
           Tensor_Options options = Tensor_Options ());

BASALT_API Tensor
tensor (std::initializer_list<double> values,
        Tensor_Options options = Tensor_Options ());

BASALT_API Tensor
scalar_tensor (double value,
               Tensor_Options options = Tensor_Options ());

/* Bridge functions to and from Bellows data_table.  */
BASALT_API Tensor
from_bellows_table (const bellows::data_table &table,
                    Tensor_Options options = Tensor_Options ());

BASALT_API bellows::data_table
to_bellows_table (const Tensor &t);

/* Compute the broadcasted output shape of A and B according to standard
   right-aligned broadcasting rules.  */
BASALT_API Shape_Vector
broadcast_shapes (const Shape_Vector &a, const Shape_Vector &b);

} /* namespace basalt */

#endif /* BASALT_TENSOR_HPP */
