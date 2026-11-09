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

#include "config.h"
#include <basalt/tensor.hpp>
#include <basalt/autograd.hpp>
#include <basalt/ops.hpp>
#include <basalt/random.hpp>
#include "tensor/tensor_internal.hpp"
#include <bellows/core/data_table.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

namespace basalt
{

struct Tensor::Impl
{
  std::shared_ptr<bellows_compat::Bellows_Storage> storage;
  size_t offset_elems = 0;
  Shape_Vector shape;
  Stride_Vector strides;
  size_t numel = 0;
  bool is_contiguous = true;
  Dtype dtype = Dtype::Float32;
  Device device = Device::cpu ();
  std::shared_ptr< std::atomic<uint64_t> > version_counter;
  std::shared_ptr<Autograd_Meta> autograd_meta;
};

namespace
{

class Reshape_Backward : public Autograd_Node
{
public:
  explicit Reshape_Backward (Shape_Vector orig_shape)
    : orig_shape_ (std::move (orig_shape))
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Reshape_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    return {grads[0].reshape (orig_shape_)};
  }

private:
  Shape_Vector orig_shape_;
};

class Transpose_Backward : public Autograd_Node
{
public:
  Transpose_Backward (ptrdiff_t d0, ptrdiff_t d1)
    : d0_ (d0),
      d1_ (d1)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Transpose_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    return {grads[0].transpose (d0_, d1_)};
  }

private:
  ptrdiff_t d0_;
  ptrdiff_t d1_;
};

class Permute_Backward : public Autograd_Node
{
public:
  explicit Permute_Backward (std::vector<ptrdiff_t> inv_dims)
    : inv_dims_ (std::move (inv_dims))
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Permute_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    return {grads[0].permute (inv_dims_)};
  }

private:
  std::vector<ptrdiff_t> inv_dims_;
};

class Expand_Backward : public Autograd_Node
{
public:
  explicit Expand_Backward (Shape_Vector orig_shape)
    : orig_shape_ (std::move (orig_shape))
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Expand_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    return {ops::sum_to_shape (grads[0], orig_shape_)};
  }

private:
  Shape_Vector orig_shape_;
};

class Slice_Grad_Node : public Autograd_Node
{
public:
  Slice_Grad_Node (Shape_Vector orig_shape, Tensor_Options opts,
                   ptrdiff_t dim, ptrdiff_t start, ptrdiff_t stop,
                   ptrdiff_t step)
    : orig_shape_ (std::move (orig_shape)),
      opts_ (opts.with_requires_grad (false)),
      dim_ (dim),
      start_ (start),
      stop_ (stop),
      step_ (step)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Slice_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    No_Grad_Guard no_grad;
    Tensor gx = zeros (orig_shape_, opts_);
    Tensor view_slice = gx.slice (dim_, start_, stop_, step_);
    view_slice.copy_ (grads[0]);
    return {gx};
  }

private:
  Shape_Vector orig_shape_;
  Tensor_Options opts_;
  ptrdiff_t dim_;
  ptrdiff_t start_;
  ptrdiff_t stop_;
  ptrdiff_t step_;
};

class Select_Grad_Node : public Autograd_Node
{
public:
  Select_Grad_Node (Shape_Vector orig_shape, Tensor_Options opts,
                    ptrdiff_t dim, ptrdiff_t index)
    : orig_shape_ (std::move (orig_shape)),
      opts_ (opts.with_requires_grad (false)),
      dim_ (dim),
      index_ (index)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Select_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    No_Grad_Guard no_grad;
    Tensor gx = zeros (orig_shape_, opts_);
    Tensor view_sel = gx.select (dim_, index_);
    view_sel.copy_ (grads[0]);
    return {gx};
  }

private:
  Shape_Vector orig_shape_;
  Tensor_Options opts_;
  ptrdiff_t dim_;
  ptrdiff_t index_;
};

class To_Backward : public Autograd_Node
{
public:
  To_Backward (Device dev, Dtype dt)
    : dev_ (dev),
      dt_ (dt)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "To_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    return {grads[0].to (dev_, dt_)};
  }

private:
  Device dev_;
  Dtype dt_;
};

void
attach_unary_view_grad (const Tensor &input, Tensor &output,
                        std::shared_ptr<Autograd_Node> node)
{
  if (Grad_Mode::is_enabled () && input.requires_grad ()
      && is_floating_point (output.dtype ()))
    {
      node->set_next_edges (autograd::collect_next_edges ({input}));
      output.set_requires_grad (true);
      output.set_grad_fn (std::move (node));
    }
}

} /* namespace */

Tensor::Tensor (const Shape_Vector &shape, Tensor_Options options)
  : impl_ (std::make_shared<Impl> ())
{
  impl_->shape = shape;
  impl_->strides = detail::contiguous_strides (shape);
  impl_->numel = detail::compute_numel (shape);
  impl_->is_contiguous = true;
  impl_->dtype = options.dtype;
  impl_->device = options.device;
  impl_->offset_elems = 0;
  size_t bytes = impl_->numel * dtype_size (options.dtype);
  impl_->storage = std::make_shared<bellows_compat::Bellows_Storage> (
    bytes, options.device);
  impl_->version_counter = std::make_shared< std::atomic<uint64_t> > (0ULL);
  if (options.requires_grad)
    {
      BASALT_ASSERT_DTYPE (is_floating_point (options.dtype),
                           "only floating-point tensors can require_grad");
      impl_->autograd_meta = std::make_shared<Autograd_Meta> ();
      impl_->autograd_meta->requires_grad = true;
    }
}

Tensor::Tensor (std::shared_ptr<bellows_compat::Bellows_Storage> storage,
                size_t offset_elems,
                Shape_Vector shape,
                Stride_Vector strides,
                Dtype dtype,
                Device device,
                bool requires_grad)
  : impl_ (std::make_shared<Impl> ())
{
  impl_->storage = std::move (storage);
  impl_->offset_elems = offset_elems;
  impl_->shape = std::move (shape);
  impl_->strides = std::move (strides);
  impl_->numel = detail::compute_numel (impl_->shape);
  impl_->is_contiguous =
    detail::compute_is_contiguous (impl_->shape, impl_->strides);
  impl_->dtype = dtype;
  impl_->device = device;
  impl_->version_counter = std::make_shared< std::atomic<uint64_t> > (0ULL);
  if (requires_grad)
    {
      impl_->autograd_meta = std::make_shared<Autograd_Meta> ();
      impl_->autograd_meta->requires_grad = true;
    }
}

bool
Tensor::defined (void) const noexcept
{
  return impl_ != nullptr && impl_->storage != nullptr;
}

size_t
Tensor::ndim (void) const noexcept
{
  return impl_ ? impl_->shape.size () : 0;
}

size_t
Tensor::numel (void) const noexcept
{
  if (!defined ())
    {
      return 0;
    }
  return impl_->numel;
}

size_t
Tensor::size (ptrdiff_t dim) const
{
  BASALT_ASSERT (defined (), "size() called on undefined Tensor");
  size_t d = detail::normalize_dim (dim, impl_->shape.size ());
  return impl_->shape[d];
}

const Shape_Vector &
Tensor::shape (void) const noexcept
{
  static const Shape_Vector empty_shape;
  return impl_ ? impl_->shape : empty_shape;
}

const Stride_Vector &
Tensor::strides (void) const noexcept
{
  static const Stride_Vector empty_strides;
  return impl_ ? impl_->strides : empty_strides;
}

size_t
Tensor::storage_offset (void) const noexcept
{
  return impl_ ? impl_->offset_elems : 0;
}

Dtype
Tensor::dtype (void) const noexcept
{
  return impl_ ? impl_->dtype : Dtype::Float32;
}

Device
Tensor::device (void) const noexcept
{
  return impl_ ? impl_->device : Device::cpu ();
}

Tensor_Options
Tensor::options (void) const noexcept
{
  return Tensor_Options ()
    .with_dtype (dtype ())
    .with_device (device ())
    .with_requires_grad (requires_grad ());
}

bool
Tensor::is_contiguous (void) const noexcept
{
  if (!defined ())
    {
      return false;
    }
  return impl_->is_contiguous;
}

Tensor
Tensor::contiguous (void) const
{
  BASALT_ASSERT (defined (), "contiguous() called on undefined Tensor");
  if (is_contiguous ())
    {
      return *this;
    }
  return clone ();
}

Tensor
Tensor::clone (void) const
{
  BASALT_ASSERT (defined (), "clone() called on undefined Tensor");
  Tensor out (impl_->shape,
              Tensor_Options ().with_dtype (dtype ()).with_device (device ()));
  size_t n = numel ();
  if (is_contiguous () && n > 0)
    {
      size_t bytes = n * dtype_size (dtype ());
      std::memcpy (out.raw_data_ptr (), raw_data_ptr (), bytes);
    }
  else if (dtype () == Dtype::Float32 && n > 0)
    {
      if (impl_->device.is_cuda ())
        {
          bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
        }
      const float *src_base
        = static_cast<const float *> (impl_->storage->data ())
          + impl_->offset_elems;
      float *dst = out.data_ptr<float> ();
      if (ndim () == 2)
        {
          size_t rows = impl_->shape[0];
          size_t cols = impl_->shape[1];
          ptrdiff_t s0 = impl_->strides[0];
          ptrdiff_t s1 = impl_->strides[1];
          for (size_t r = 0; r < rows; ++r)
            {
              const float *row_p = src_base + static_cast<ptrdiff_t> (r) * s0;
              float *dst_row = dst + r * cols;
              for (size_t c = 0; c < cols; ++c)
                {
                  dst_row[c] = row_p[static_cast<ptrdiff_t> (c) * s1];
                }
            }
        }
      else if (ndim () == 3)
        {
          size_t d0 = impl_->shape[0];
          size_t d1 = impl_->shape[1];
          size_t d2 = impl_->shape[2];
          ptrdiff_t s0 = impl_->strides[0];
          ptrdiff_t s1 = impl_->strides[1];
          ptrdiff_t s2 = impl_->strides[2];
          for (size_t i0 = 0; i0 < d0; ++i0)
            {
              const float *p0 = src_base + static_cast<ptrdiff_t> (i0) * s0;
              for (size_t i1 = 0; i1 < d1; ++i1)
                {
                  const float *p1 = p0 + static_cast<ptrdiff_t> (i1) * s1;
                  float *dst_row = dst + (i0 * d1 + i1) * d2;
                  if (s2 == 1)
                    {
                      std::memcpy (dst_row, p1, d2 * sizeof (float));
                    }
                  else
                    {
                      for (size_t i2 = 0; i2 < d2; ++i2)
                        {
                          dst_row[i2] = p1[static_cast<ptrdiff_t> (i2) * s2];
                        }
                    }
                }
            }
        }
      else if (ndim () == 4)
        {
          size_t d0 = impl_->shape[0];
          size_t d1 = impl_->shape[1];
          size_t d2 = impl_->shape[2];
          size_t d3 = impl_->shape[3];
          ptrdiff_t s0 = impl_->strides[0];
          ptrdiff_t s1 = impl_->strides[1];
          ptrdiff_t s2 = impl_->strides[2];
          ptrdiff_t s3 = impl_->strides[3];
          for (size_t i0 = 0; i0 < d0; ++i0)
            {
              const float *p0 = src_base + static_cast<ptrdiff_t> (i0) * s0;
              for (size_t i1 = 0; i1 < d1; ++i1)
                {
                  const float *p1 = p0 + static_cast<ptrdiff_t> (i1) * s1;
                  for (size_t i2 = 0; i2 < d2; ++i2)
                    {
                      const float *p2 = p1 + static_cast<ptrdiff_t> (i2) * s2;
                      float *dst_row
                        = dst + ((i0 * d1 + i1) * d2 + i2) * d3;
                      if (s3 == 1)
                        {
                          std::memcpy (dst_row, p2, d3 * sizeof (float));
                        }
                      else
                        {
                          for (size_t i3 = 0; i3 < d3; ++i3)
                            {
                              dst_row[i3]
                                = p2[static_cast<ptrdiff_t> (i3) * s3];
                            }
                        }
                    }
                }
            }
        }
      else
        {
          for (size_t i = 0; i < n; ++i)
            {
              ptrdiff_t rel = detail::linear_to_strided_offset (
                i, impl_->shape, impl_->strides);
              dst[i] = src_base[rel];
            }
        }
    }
  else
    {
      for (size_t i = 0; i < n; ++i)
        {
          out.write_scalar_at (i, read_scalar_at (i));
        }
    }
  attach_unary_view_grad (
    *this, out, std::make_shared<To_Backward> (device (), dtype ()));
  return out;
}

Tensor
Tensor::detach (void) const
{
  if (!defined ())
    {
      return Tensor ();
    }
  Tensor out (impl_->storage, impl_->offset_elems, impl_->shape,
              impl_->strides, impl_->dtype, impl_->device, false);
  out.impl_->version_counter = impl_->version_counter;
  return out;
}

Tensor
Tensor::to (Device target_device) const
{
  return to (target_device, dtype ());
}

Tensor
Tensor::to (Dtype target_dtype) const
{
  return to (device (), target_dtype);
}

Tensor
Tensor::to (Device target_device, Dtype target_dtype) const
{
  BASALT_ASSERT (defined (), "to() called on undefined Tensor");
  if (target_device == device () && target_dtype == dtype ())
    {
      return *this;
    }
  Tensor out (impl_->shape,
              Tensor_Options ()
                .with_device (target_device)
                .with_dtype (target_dtype));
  size_t n = numel ();
  if (device ().is_cuda () || target_device.is_cuda ())
    {
      bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
    }
  if (is_contiguous () && target_dtype == dtype () && n > 0)
    {
      std::memcpy (out.raw_data_ptr (), raw_data_ptr (),
                   n * dtype_size (target_dtype));
    }
  else if (is_contiguous () && dtype () == Dtype::Float32
           && target_dtype == Dtype::Float16 && n > 0)
    {
      const float *src = data_ptr<float> ();
      uint16_t *dst = static_cast<uint16_t *> (out.raw_data_ptr ());
      for (size_t i = 0; i < n; ++i)
        {
          dst[i] = Float16::from_float32 (src[i]);
        }
    }
  else if (is_contiguous () && dtype () == Dtype::Float16
           && target_dtype == Dtype::Float32 && n > 0)
    {
      const uint16_t *src = static_cast<const uint16_t *> (raw_data_ptr ());
      float *dst = out.data_ptr<float> ();
      for (size_t i = 0; i < n; ++i)
        {
          dst[i] = Float16::to_float32 (src[i]);
        }
    }
  else if (is_contiguous () && dtype () == Dtype::Float32
           && target_dtype == Dtype::BFloat16 && n > 0)
    {
      const float *src = data_ptr<float> ();
      uint16_t *dst = static_cast<uint16_t *> (out.raw_data_ptr ());
      for (size_t i = 0; i < n; ++i)
        {
          dst[i] = BFloat16::from_float32 (src[i]);
        }
    }
  else if (is_contiguous () && dtype () == Dtype::BFloat16
           && target_dtype == Dtype::Float32 && n > 0)
    {
      const uint16_t *src = static_cast<const uint16_t *> (raw_data_ptr ());
      float *dst = out.data_ptr<float> ();
      for (size_t i = 0; i < n; ++i)
        {
          dst[i] = BFloat16::to_float32 (src[i]);
        }
    }
  else
    {
      for (size_t i = 0; i < n; ++i)
        {
          out.write_scalar_at (i, read_scalar_at (i));
        }
    }
  attach_unary_view_grad (
    *this, out, std::make_shared<To_Backward> (device (), dtype ()));
  return out;
}

Tensor
Tensor::view (const Shape_Vector &new_shape) const
{
  BASALT_ASSERT (defined (), "view() called on undefined Tensor");
  BASALT_ASSERT_SHAPE (is_contiguous (),
                       "view() requires a contiguous tensor; use reshape()");
  BASALT_ASSERT_SHAPE (detail::compute_numel (new_shape) == numel (),
                       "view() total element count mismatch");
  Tensor out (impl_->storage, impl_->offset_elems, new_shape,
              detail::contiguous_strides (new_shape), impl_->dtype,
              impl_->device, false);
  out.impl_->version_counter = impl_->version_counter;
  attach_unary_view_grad (
    *this, out, std::make_shared<Reshape_Backward> (impl_->shape));
  return out;
}

Tensor
Tensor::reshape (const Shape_Vector &new_shape) const
{
  BASALT_ASSERT (defined (), "reshape() called on undefined Tensor");
  if (new_shape == impl_->shape)
    {
      return *this;
    }
  if (is_contiguous ())
    {
      return view (new_shape);
    }
  return contiguous ().view (new_shape);
}

Tensor
Tensor::transpose (ptrdiff_t dim0, ptrdiff_t dim1) const
{
  BASALT_ASSERT (defined (), "transpose() called on undefined Tensor");
  size_t d0 = detail::normalize_dim (dim0, ndim ());
  size_t d1 = detail::normalize_dim (dim1, ndim ());
  if (d0 == d1)
    {
      return *this;
    }
  Shape_Vector new_shape = impl_->shape;
  Stride_Vector new_strides = impl_->strides;
  std::swap (new_shape[d0], new_shape[d1]);
  std::swap (new_strides[d0], new_strides[d1]);
  Tensor out (impl_->storage, impl_->offset_elems, std::move (new_shape),
              std::move (new_strides), impl_->dtype, impl_->device, false);
  out.impl_->version_counter = impl_->version_counter;
  attach_unary_view_grad (
    *this, out, std::make_shared<Transpose_Backward> (dim0, dim1));
  return out;
}

Tensor
Tensor::t (void) const
{
  BASALT_ASSERT_SHAPE (ndim () == 2, "t() expects a 2-D tensor");
  return transpose (0, 1);
}

Tensor
Tensor::permute (const std::vector<ptrdiff_t> &dims) const
{
  BASALT_ASSERT (defined (), "permute() called on undefined Tensor");
  BASALT_ASSERT_SHAPE (dims.size () == ndim (),
                       "permute() rank mismatch");
  size_t r = ndim ();
  Shape_Vector new_shape (r);
  Stride_Vector new_strides (r);
  std::vector<bool> seen (r, false);
  std::vector<ptrdiff_t> inv_dims (r, 0);

  for (size_t i = 0; i < r; ++i)
    {
      size_t d = detail::normalize_dim (dims[i], r);
      BASALT_ASSERT_SHAPE (!seen[d], "duplicate dimension in permute()");
      seen[d] = true;
      new_shape[i] = impl_->shape[d];
      new_strides[i] = impl_->strides[d];
      inv_dims[d] = static_cast<ptrdiff_t> (i);
    }
  Tensor out (impl_->storage, impl_->offset_elems, std::move (new_shape),
              std::move (new_strides), impl_->dtype, impl_->device, false);
  out.impl_->version_counter = impl_->version_counter;
  attach_unary_view_grad (
    *this, out, std::make_shared<Permute_Backward> (std::move (inv_dims)));
  return out;
}

Tensor
Tensor::slice (ptrdiff_t dim, ptrdiff_t start, ptrdiff_t stop,
               ptrdiff_t step) const
{
  BASALT_ASSERT (defined (), "slice() called on undefined Tensor");
  BASALT_ASSERT_ARG (step > 0, "slice() step must be positive");
  size_t d = detail::normalize_dim (dim, ndim ());
  ptrdiff_t len = static_cast<ptrdiff_t> (impl_->shape[d]);
  ptrdiff_t s = (start < 0) ? (start + len) : start;
  ptrdiff_t e = (stop < 0) ? (stop + len) : stop;
  s = std::clamp<ptrdiff_t> (s, 0, len);
  e = std::clamp<ptrdiff_t> (e, s, len);

  size_t slice_len = static_cast<size_t> ((e - s + step - 1) / step);
  Shape_Vector new_shape = impl_->shape;
  Stride_Vector new_strides = impl_->strides;
  new_shape[d] = slice_len;
  new_strides[d] *= step;
  size_t new_offset = static_cast<size_t> (
    static_cast<ptrdiff_t> (impl_->offset_elems) + s * impl_->strides[d]);

  Tensor out (impl_->storage, new_offset, std::move (new_shape),
              std::move (new_strides), impl_->dtype, impl_->device, false);
  out.impl_->version_counter = impl_->version_counter;
  attach_unary_view_grad (
    *this, out,
    std::make_shared<Slice_Grad_Node> (
      impl_->shape, options (), static_cast<ptrdiff_t> (d), s, e, step));
  return out;
}

Tensor
Tensor::select (ptrdiff_t dim, ptrdiff_t index) const
{
  BASALT_ASSERT (defined (), "select() called on undefined Tensor");
  BASALT_ASSERT_SHAPE (ndim () >= 1, "select() requires ndim >= 1");
  size_t d = detail::normalize_dim (dim, ndim ());
  ptrdiff_t len = static_cast<ptrdiff_t> (impl_->shape[d]);
  ptrdiff_t idx = (index < 0) ? (index + len) : index;
  BASALT_ASSERT_SHAPE (idx >= 0 && idx < len, "select() index out of bounds");

  Shape_Vector new_shape;
  Stride_Vector new_strides;
  for (size_t i = 0; i < ndim (); ++i)
    {
      if (i != d)
        {
          new_shape.push_back (impl_->shape[i]);
          new_strides.push_back (impl_->strides[i]);
        }
    }
  size_t new_offset = static_cast<size_t> (
    static_cast<ptrdiff_t> (impl_->offset_elems) + idx * impl_->strides[d]);

  Tensor out (impl_->storage, new_offset, std::move (new_shape),
              std::move (new_strides), impl_->dtype, impl_->device, false);
  out.impl_->version_counter = impl_->version_counter;
  attach_unary_view_grad (
    *this, out,
    std::make_shared<Select_Grad_Node> (
      impl_->shape, options (), static_cast<ptrdiff_t> (d), idx));
  return out;
}

Tensor
Tensor::squeeze (ptrdiff_t dim) const
{
  BASALT_ASSERT (defined (), "squeeze() called on undefined Tensor");
  Shape_Vector new_shape;
  Stride_Vector new_strides;
  if (dim == -9999)
    {
      for (size_t i = 0; i < ndim (); ++i)
        {
          if (impl_->shape[i] != 1)
            {
              new_shape.push_back (impl_->shape[i]);
              new_strides.push_back (impl_->strides[i]);
            }
        }
    }
  else
    {
      size_t d = detail::normalize_dim (dim, ndim ());
      for (size_t i = 0; i < ndim (); ++i)
        {
          if (i == d && impl_->shape[i] == 1)
            {
              continue;
            }
          new_shape.push_back (impl_->shape[i]);
          new_strides.push_back (impl_->strides[i]);
        }
    }
  Tensor out (impl_->storage, impl_->offset_elems, std::move (new_shape),
              std::move (new_strides), impl_->dtype, impl_->device, false);
  out.impl_->version_counter = impl_->version_counter;
  attach_unary_view_grad (
    *this, out, std::make_shared<Reshape_Backward> (impl_->shape));
  return out;
}

Tensor
Tensor::unsqueeze (ptrdiff_t dim) const
{
  BASALT_ASSERT (defined (), "unsqueeze() called on undefined Tensor");
  size_t d = detail::normalize_dim (dim, ndim (), true);
  Shape_Vector new_shape = impl_->shape;
  Stride_Vector new_strides = impl_->strides;
  ptrdiff_t st = (d < impl_->strides.size ())
                 ? (impl_->strides[d]
                    * static_cast<ptrdiff_t> (impl_->shape[d]))
                 : 1;
  new_shape.insert (new_shape.begin () + static_cast<ptrdiff_t> (d), 1U);
  new_strides.insert (new_strides.begin () + static_cast<ptrdiff_t> (d), st);
  Tensor out (impl_->storage, impl_->offset_elems, std::move (new_shape),
              std::move (new_strides), impl_->dtype, impl_->device, false);
  out.impl_->version_counter = impl_->version_counter;
  attach_unary_view_grad (
    *this, out, std::make_shared<Reshape_Backward> (impl_->shape));
  return out;
}

Tensor
Tensor::expand (const Shape_Vector &target_shape) const
{
  BASALT_ASSERT (defined (), "expand() called on undefined Tensor");
  BASALT_ASSERT_SHAPE (target_shape.size () >= ndim (),
                       "expand() target rank cannot be smaller than ndim");
  if (target_shape == impl_->shape)
    {
      return *this;
    }
  size_t out_ndim = target_shape.size ();
  size_t pad = out_ndim - ndim ();
  Stride_Vector new_strides (out_ndim, 0);
  for (size_t i = 0; i < out_ndim; ++i)
    {
      if (i < pad)
        {
          new_strides[i] = 0;
        }
      else
        {
          size_t src_dim = impl_->shape[i - pad];
          ptrdiff_t src_stride = impl_->strides[i - pad];
          if (src_dim == target_shape[i])
            {
              new_strides[i] = src_stride;
            }
          else if (src_dim == 1)
            {
              new_strides[i] = 0;
            }
          else
            {
              BASALT_THROW (Shape_Error,
                            "expand() incompatible dimension size");
            }
        }
    }
  Tensor out (impl_->storage, impl_->offset_elems, target_shape,
              std::move (new_strides), impl_->dtype, impl_->device, false);
  out.impl_->version_counter = impl_->version_counter;
  attach_unary_view_grad (
    *this, out, std::make_shared<Expand_Backward> (impl_->shape));
  return out;
}

Tensor
Tensor::flatten (ptrdiff_t start_dim, ptrdiff_t end_dim) const
{
  BASALT_ASSERT (defined (), "flatten() called on undefined Tensor");
  if (ndim () == 0)
    {
      return reshape ({1});
    }
  size_t s = detail::normalize_dim (start_dim, ndim ());
  size_t e = detail::normalize_dim (end_dim, ndim ());
  BASALT_ASSERT_SHAPE (s <= e, "flatten() requires start_dim <= end_dim");

  Shape_Vector new_shape;
  for (size_t i = 0; i < s; ++i)
    {
      new_shape.push_back (impl_->shape[i]);
    }
  size_t merged = 1;
  for (size_t i = s; i <= e; ++i)
    {
      merged *= impl_->shape[i];
    }
  new_shape.push_back (merged);
  for (size_t i = e + 1; i < ndim (); ++i)
    {
      new_shape.push_back (impl_->shape[i]);
    }
  return reshape (new_shape);
}

void *
Tensor::raw_data_ptr (void)
{
  BASALT_ASSERT (defined (), "raw_data_ptr() called on undefined Tensor");
  uint8_t *base = static_cast<uint8_t *> (impl_->storage->data ());
  return base + impl_->offset_elems * dtype_size (impl_->dtype);
}

const void *
Tensor::raw_data_ptr (void) const
{
  BASALT_ASSERT (defined (), "raw_data_ptr() called on undefined Tensor");
  const uint8_t *base = static_cast<const uint8_t *> (impl_->storage->data ());
  return base + impl_->offset_elems * dtype_size (impl_->dtype);
}

double
Tensor::read_scalar_at (size_t linear_index) const
{
  BASALT_ASSERT (defined (), "read_scalar_at() called on undefined Tensor");
  if (impl_->device.is_cuda ())
    {
      bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
    }
  ptrdiff_t rel = impl_->is_contiguous
                  ? static_cast<ptrdiff_t> (linear_index)
                  : detail::linear_to_strided_offset (
                      linear_index, impl_->shape, impl_->strides);
  ptrdiff_t abs_elem = static_cast<ptrdiff_t> (impl_->offset_elems) + rel;
  return detail::load_element_as_double (
    impl_->storage->data (), abs_elem, impl_->dtype);
}

void
Tensor::write_scalar_at (size_t linear_index, double value)
{
  BASALT_ASSERT (defined (), "write_scalar_at() called on undefined Tensor");
  if (impl_->device.is_cuda ())
    {
      bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
    }
  ptrdiff_t rel = impl_->is_contiguous
                  ? static_cast<ptrdiff_t> (linear_index)
                  : detail::linear_to_strided_offset (
                      linear_index, impl_->shape, impl_->strides);
  ptrdiff_t abs_elem = static_cast<ptrdiff_t> (impl_->offset_elems) + rel;
  detail::store_double_as_element (
    impl_->storage->data (), abs_elem, impl_->dtype, value);
}

std::vector<double>
Tensor::to_vector_double (void) const
{
  size_t n = numel ();
  std::vector<double> out (n);
  for (size_t i = 0; i < n; ++i)
    {
      out[i] = read_scalar_at (i);
    }
  return out;
}

std::vector<float>
Tensor::to_vector_float (void) const
{
  size_t n = numel ();
  std::vector<float> out (n);
  if (is_contiguous () && dtype () == Dtype::Float32 && n > 0)
    {
      if (impl_->device.is_cuda ())
        {
          bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
        }
      std::memcpy (out.data (), data_ptr<float> (), n * sizeof (float));
      return out;
    }
  for (size_t i = 0; i < n; ++i)
    {
      out[i] = static_cast<float> (read_scalar_at (i));
    }
  return out;
}

std::vector<int64_t>
Tensor::to_vector_int64 (void) const
{
  size_t n = numel ();
  std::vector<int64_t> out (n);
  for (size_t i = 0; i < n; ++i)
    {
      out[i] = static_cast<int64_t> (read_scalar_at (i));
    }
  return out;
}

double
Tensor::item_double (void) const
{
  BASALT_ASSERT_SHAPE (numel () == 1,
                       "item() requires a single-element tensor");
  return read_scalar_at (0);
}

Tensor &
Tensor::fill_ (double value)
{
  BASALT_ASSERT (defined (), "fill_() called on undefined Tensor");
  size_t n = numel ();
  if (is_contiguous () && dtype () == Dtype::Float32 && n > 0)
    {
      float fv = static_cast<float> (value);
      float *dst = static_cast<float *> (impl_->storage->data ())
                   + impl_->offset_elems;
      if (impl_->device.is_cuda ())
        {
          auto &cuda_drv = bellows_compat::Cuda_Runtime_Driver::instance ();
          if (cuda_drv.launch_fill_f32 (impl_->device.index (), dst, fv, n))
            {
              bump_version ();
              return *this;
            }
          cuda_drv.sync_if_busy ();
        }
      float *host_dst = data_ptr<float> ();
      std::fill_n (host_dst, n, fv);
      bump_version ();
      return *this;
    }
  for (size_t i = 0; i < n; ++i)
    {
      write_scalar_at (i, value);
    }
  bump_version ();
  return *this;
}

Tensor &
Tensor::copy_ (const Tensor &src)
{
  BASALT_ASSERT (defined () && src.defined (),
                 "copy_() requires defined tensors");
  size_t n = numel ();
  if (is_contiguous () && src.is_contiguous ()
      && shape () == src.shape () && dtype () == src.dtype () && n > 0)
    {
      if (impl_->device.is_cuda () || src.device ().is_cuda ())
        {
          bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
        }
      std::memcpy (raw_data_ptr (), src.raw_data_ptr (),
                   n * dtype_size (dtype ()));
      bump_version ();
      return *this;
    }
  Tensor bcast = (src.shape () == shape ()) ? src : src.expand (shape ());
  for (size_t i = 0; i < n; ++i)
    {
      write_scalar_at (i, bcast.read_scalar_at (i));
    }
  bump_version ();
  return *this;
}

Tensor &
Tensor::add_ (const Tensor &other, double alpha)
{
  BASALT_ASSERT (defined () && other.defined (),
                 "add_() requires defined tensors");
  size_t n = numel ();
  if (is_contiguous () && other.is_contiguous ()
      && shape () == other.shape ()
      && dtype () == Dtype::Float32 && other.dtype () == Dtype::Float32
      && n > 0)
    {
      if (impl_->device.is_cuda () || other.device ().is_cuda ())
        {
          bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
        }
      float *dst = data_ptr<float> ();
      const float *src = other.data_ptr<float> ();
      float fa = static_cast<float> (alpha);
      for (size_t i = 0; i < n; ++i)
        {
          dst[i] += fa * src[i];
        }
      bump_version ();
      return *this;
    }
  Tensor bcast = (other.shape () == shape ())
                 ? other
                 : other.expand (shape ());
  for (size_t i = 0; i < n; ++i)
    {
      double v = read_scalar_at (i) + alpha * bcast.read_scalar_at (i);
      write_scalar_at (i, v);
    }
  bump_version ();
  return *this;
}

Tensor &
Tensor::mul_ (double scale)
{
  BASALT_ASSERT (defined (), "mul_() called on undefined Tensor");
  size_t n = numel ();
  if (is_contiguous () && dtype () == Dtype::Float32 && n > 0)
    {
      float fs = static_cast<float> (scale);
      float *dst = static_cast<float *> (impl_->storage->data ())
                   + impl_->offset_elems;
      auto &cuda_drv = bellows_compat::Cuda_Runtime_Driver::instance ();
      if (impl_->device.is_cuda ()
          && cuda_drv.launch_unary_f32 (
               impl_->device.index (), dst, dst, fs,
               bellows_compat::Cuda_Unary_Op::Mul_Scalar, n))
        {
          bump_version ();
          return *this;
        }
      float *host_dst = data_ptr<float> ();
      for (size_t i = 0; i < n; ++i)
        {
          host_dst[i] *= fs;
        }
      bump_version ();
      return *this;
    }
  for (size_t i = 0; i < n; ++i)
    {
      write_scalar_at (i, read_scalar_at (i) * scale);
    }
  bump_version ();
  return *this;
}

uint64_t
Tensor::version (void) const noexcept
{
  if (!impl_ || !impl_->version_counter)
    {
      return 0;
    }
  return impl_->version_counter->load (std::memory_order_relaxed);
}

void
Tensor::bump_version (void) noexcept
{
  if (impl_ && impl_->version_counter)
    {
      impl_->version_counter->fetch_add (1ULL, std::memory_order_relaxed);
    }
}

bool
Tensor::requires_grad (void) const noexcept
{
  return impl_ && impl_->autograd_meta && impl_->autograd_meta->requires_grad;
}

Tensor &
Tensor::set_requires_grad (bool rg)
{
  BASALT_ASSERT (defined (), "set_requires_grad() on undefined Tensor");
  if (rg)
    {
      BASALT_ASSERT_DTYPE (is_floating_point (dtype ()),
                           "only floating-point tensors can require_grad");
      if (!impl_->autograd_meta)
        {
          impl_->autograd_meta = std::make_shared<Autograd_Meta> ();
        }
      impl_->autograd_meta->requires_grad = true;
    }
  else if (impl_->autograd_meta)
    {
      impl_->autograd_meta->requires_grad = false;
    }
  return *this;
}

bool
Tensor::is_leaf (void) const noexcept
{
  if (!requires_grad ())
    {
      return true;
    }
  return impl_->autograd_meta->grad_fn == nullptr;
}

void
Tensor::retain_grad (void)
{
  BASALT_ASSERT (requires_grad (),
                 "retain_grad() requires requires_grad == true");
  impl_->autograd_meta->retain_grad = true;
}

bool
Tensor::retains_grad (void) const noexcept
{
  return is_leaf ()
         || (impl_ && impl_->autograd_meta
             && impl_->autograd_meta->retain_grad);
}

Tensor
Tensor::grad (void) const noexcept
{
  if (!impl_ || !impl_->autograd_meta)
    {
      return Tensor ();
    }
  std::lock_guard<std::mutex> lock (impl_->autograd_meta->mutex);
  return impl_->autograd_meta->grad;
}

void
Tensor::set_grad (const Tensor &g)
{
  BASALT_ASSERT (defined (), "set_grad() on undefined Tensor");
  if (!impl_->autograd_meta)
    {
      impl_->autograd_meta = std::make_shared<Autograd_Meta> ();
    }
  std::lock_guard<std::mutex> lock (impl_->autograd_meta->mutex);
  impl_->autograd_meta->grad = g;
}

void
Tensor::zero_grad (void)
{
  if (impl_ && impl_->autograd_meta)
    {
      std::lock_guard<std::mutex> lock (impl_->autograd_meta->mutex);
      impl_->autograd_meta->grad = Tensor ();
    }
}

std::shared_ptr<Autograd_Node>
Tensor::grad_fn (void) const noexcept
{
  if (!impl_ || !impl_->autograd_meta)
    {
      return nullptr;
    }
  return impl_->autograd_meta->grad_fn;
}

void
Tensor::set_grad_fn (std::shared_ptr<Autograd_Node> node)
{
  if (!impl_->autograd_meta)
    {
      impl_->autograd_meta = std::make_shared<Autograd_Meta> ();
    }
  if (node)
    {
      node->set_output_meta (impl_->autograd_meta);
    }
  impl_->autograd_meta->grad_fn = std::move (node);
  impl_->autograd_meta->requires_grad = (impl_->autograd_meta->grad_fn
                                         != nullptr);
}

std::shared_ptr<Autograd_Meta>
Tensor::autograd_meta (void) const noexcept
{
  return impl_ ? impl_->autograd_meta : nullptr;
}

void
Tensor::backward (const Tensor &grad_output,
                  bool retain_graph,
                  bool create_graph) const
{
  autograd::backward (*this, grad_output, retain_graph, create_graph);
}

std::string
Tensor::to_string (void) const
{
  if (!defined ())
    {
      return "Tensor(undefined)";
    }
  std::ostringstream oss;
  oss << "Tensor(shape=[";
  for (size_t i = 0; i < impl_->shape.size (); ++i)
    {
      if (i > 0)
        {
          oss << ", ";
        }
      oss << impl_->shape[i];
    }
  oss << "], dtype=" << dtype_name (dtype ())
      << ", device=" << device ().str ()
      << ", values=[";
  size_t n = numel ();
  size_t limit = std::min<size_t> (n, 8);
  for (size_t i = 0; i < limit; ++i)
    {
      if (i > 0)
        {
          oss << ", ";
        }
      oss << read_scalar_at (i);
    }
  if (n > limit)
    {
      oss << ", ...";
    }
  oss << "])";
  return oss.str ();
}

const std::shared_ptr<bellows_compat::Bellows_Storage> &
Tensor::storage (void) const noexcept
{
  static const std::shared_ptr<bellows_compat::Bellows_Storage> null_s;
  return impl_ ? impl_->storage : null_s;
}

Tensor
empty (const Shape_Vector &shape, Tensor_Options options)
{
  return Tensor (shape, options);
}

Tensor
zeros (const Shape_Vector &shape, Tensor_Options options)
{
  return Tensor (shape, options);
}

Tensor
ones (const Shape_Vector &shape, Tensor_Options options)
{
  return full (shape, 1.0, options);
}

Tensor
full (const Shape_Vector &shape, double fill_value, Tensor_Options options)
{
  Tensor out (shape, options.with_requires_grad (false));
  out.fill_ (fill_value);
  out.bump_version ();
  if (options.requires_grad)
    {
      out.set_requires_grad (true);
    }
  return out;
}

Tensor
zeros_like (const Tensor &other)
{
  return zeros (other.shape (), other.options ().with_requires_grad (false));
}

Tensor
ones_like (const Tensor &other)
{
  return ones (other.shape (), other.options ().with_requires_grad (false));
}

Tensor
full_like (const Tensor &other, double fill_value)
{
  return full (other.shape (), fill_value,
               other.options ().with_requires_grad (false));
}

Tensor
eye (size_t n, size_t m, Tensor_Options options)
{
  size_t cols = (m == 0) ? n : m;
  Tensor out ({n, cols}, options.with_requires_grad (false));
  size_t diag = std::min (n, cols);
  for (size_t i = 0; i < diag; ++i)
    {
      out.write_scalar_at (i * cols + i, 1.0);
    }
  if (options.requires_grad)
    {
      out.set_requires_grad (true);
    }
  return out;
}

Tensor
arange (double start, double stop, double step, Tensor_Options options)
{
  BASALT_ASSERT_ARG (step != 0.0, "arange() step cannot be zero");
  BASALT_ASSERT_ARG ((step > 0.0 && stop >= start)
                     || (step < 0.0 && stop <= start),
                     "arange() invalid range for given step");
  size_t count = static_cast<size_t> (std::ceil ((stop - start) / step));
  Tensor out ({count}, options.with_requires_grad (false));
  for (size_t i = 0; i < count; ++i)
    {
      out.write_scalar_at (i, start + static_cast<double> (i) * step);
    }
  if (options.requires_grad)
    {
      out.set_requires_grad (true);
    }
  return out;
}

Tensor
linspace (double start, double stop, size_t steps, Tensor_Options options)
{
  BASALT_ASSERT_ARG (steps >= 1, "linspace() requires steps >= 1");
  Tensor out ({steps}, options.with_requires_grad (false));
  if (steps == 1)
    {
      out.write_scalar_at (0, start);
    }
  else
    {
      double delta = (stop - start) / static_cast<double> (steps - 1);
      for (size_t i = 0; i < steps; ++i)
        {
          out.write_scalar_at (i, start + static_cast<double> (i) * delta);
        }
    }
  if (options.requires_grad)
    {
      out.set_requires_grad (true);
    }
  return out;
}

Tensor
rand (const Shape_Vector &shape, Tensor_Options options)
{
  Tensor out (shape, options.with_requires_grad (false));
  size_t n = out.numel ();
  Random_Generator &gen = default_generator ();
  for (size_t i = 0; i < n; ++i)
    {
      out.write_scalar_at (i, gen.uniform (0.0, 1.0));
    }
  if (options.requires_grad)
    {
      out.set_requires_grad (true);
    }
  return out;
}

Tensor
randn (const Shape_Vector &shape, Tensor_Options options)
{
  Tensor out (shape, options.with_requires_grad (false));
  size_t n = out.numel ();
  Random_Generator &gen = default_generator ();
  for (size_t i = 0; i < n; ++i)
    {
      out.write_scalar_at (i, gen.normal (0.0, 1.0));
    }
  if (options.requires_grad)
    {
      out.set_requires_grad (true);
    }
  return out;
}

Tensor
randint (int64_t low, int64_t high, const Shape_Vector &shape,
         Tensor_Options options)
{
  Tensor out (shape, options.with_requires_grad (false));
  size_t n = out.numel ();
  Random_Generator &gen = default_generator ();
  for (size_t i = 0; i < n; ++i)
    {
      out.write_scalar_at (i, static_cast<double> (gen.randint (low, high)));
    }
  if (options.requires_grad)
    {
      out.set_requires_grad (true);
    }
  return out;
}

Tensor
from_blob (const void *data, const Shape_Vector &shape, Tensor_Options options)
{
  BASALT_ASSERT_ARG (data != nullptr, "from_blob() data pointer is null");
  Tensor out (shape, options.with_requires_grad (false));
  size_t bytes = out.numel () * dtype_size (options.dtype);
  out.storage ()->copy_from_host (data, 0, bytes);
  if (options.requires_grad)
    {
      out.set_requires_grad (true);
    }
  return out;
}

Tensor
tensor (std::initializer_list<double> values, Tensor_Options options)
{
  Tensor out ({values.size ()}, options.with_requires_grad (false));
  size_t i = 0;
  for (double v : values)
    {
      out.write_scalar_at (i++, v);
    }
  if (options.requires_grad)
    {
      out.set_requires_grad (true);
    }
  return out;
}

Tensor
scalar_tensor (double value, Tensor_Options options)
{
  Tensor out ({1}, options.with_requires_grad (false));
  out.write_scalar_at (0, value);
  if (options.requires_grad)
    {
      out.set_requires_grad (true);
    }
  return out;
}

Tensor
from_bellows_table (const bellows::data_table &table, Tensor_Options options)
{
  size_t rows = table.row_count ();
  size_t cols = table.column_count ();
  Tensor out ({rows, cols}, options.with_requires_grad (false));
  for (size_t r = 0; r < rows; ++r)
    {
      for (size_t c = 0; c < cols; ++c)
        {
          out.write_scalar_at (r * cols + c, table.at (r, c));
        }
    }
  if (options.requires_grad)
    {
      out.set_requires_grad (true);
    }
  return out;
}

bellows::data_table
to_bellows_table (const Tensor &t)
{
  BASALT_ASSERT_SHAPE (t.ndim () == 2,
                       "to_bellows_table() requires a 2-D tensor");
  size_t rows = t.size (0);
  size_t cols = t.size (1);
  hyper_math::tensor::nd_tensor<double> mat (
    hyper_math::tensor::shape_type{rows, cols});
  for (size_t r = 0; r < rows; ++r)
    {
      for (size_t c = 0; c < cols; ++c)
        {
          mat (r, c) = t.read_scalar_at (r * cols + c);
        }
    }
  return bellows::data_table (std::move (mat));
}

Shape_Vector
broadcast_shapes (const Shape_Vector &a, const Shape_Vector &b)
{
  try
    {
      return hyper_math::tensor::broadcast_shapes (a, b);
    }
  catch (const std::exception &e)
    {
      BASALT_THROW (Shape_Error,
                    std::string ("broadcast_shapes failed: ") + e.what ());
    }
}

} /* namespace basalt */
