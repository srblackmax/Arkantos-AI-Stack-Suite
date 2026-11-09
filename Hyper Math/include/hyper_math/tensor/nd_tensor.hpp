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

#ifndef HYPER_MATH_TENSOR_ND_TENSOR_HPP
#define HYPER_MATH_TENSOR_ND_TENSOR_HPP

#include "hyper_math/core/config.hpp"
#include "hyper_math/core/memory.hpp"
#include "hyper_math/core/parallel.hpp"
#include "hyper_math/core/prng.hpp"
#include "hyper_math/tensor/lattice.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <vector>

namespace hyper_math::tensor
{

/* CRTP base class for compile-time lazily fused tensor expressions.  */
template <typename Derived, typename ValueType>
class tensor_expr_base
{
public:
  using value_type = ValueType;

  const Derived &
  derived () const noexcept
  {
    return static_cast<const Derived &> (*this);
  }

  const shape_type &
  shape () const noexcept
  {
    return derived ().shape ();
  }

  std::size_t
  size () const noexcept
  {
    return derived ().size ();
  }

  HYPER_MATH_ALWAYS_INLINE value_type
  eval_at (std::size_t flat_idx) const noexcept
  {
    return derived ().eval_at (flat_idx);
  }

  HYPER_MATH_ALWAYS_INLINE value_type
  eval_with_lattice (std::size_t flat_idx,
                     const lattice_descriptor &bcast_lat) const noexcept
  {
    return derived ().eval_with_lattice (flat_idx, bcast_lat);
  }

  lattice_descriptor
  broadcast_lattice (const shape_type &target_shape) const
  {
    return derived ().broadcast_lattice (target_shape);
  }
};

/* Primary N-Dimensional Tensor class (NumPy ndarray equivalent).
   Combines a 64-byte aligned reference-counted slab_buffer<T> with a
   rank-agnostic affine lattice_descriptor.  */
template <typename T = double>
class nd_tensor : public tensor_expr_base<nd_tensor<T>, T>
{
private:
  slab_buffer<T> storage_;
  lattice_descriptor lattice_;

public:
  using value_type = T;

  nd_tensor () noexcept
    : storage_ (), lattice_ ()
  {
  }

  explicit
  nd_tensor (shape_type dims)
    : storage_ (), lattice_ ( std::move (dims))
  {
    storage_ = slab_buffer<T> (lattice_.size (), T {});
  }

  nd_tensor (shape_type dims, const T &fill_val)
    : storage_ (), lattice_ (std::move (dims))
  {
    storage_ = slab_buffer<T> (lattice_.size (), fill_val);
  }

  nd_tensor (shape_type dims, const std::vector<T> &values)
    : storage_ (), lattice_ (std::move (dims))
  {
    if (values.size () != lattice_.size ())
      {
        throw shape_error ("initializer vector size ("
                           + std::to_string (values.size ())
                           + ") does not match shape size ("
                           + std::to_string (lattice_.size ()) + ")");
      }
    storage_ = slab_buffer<T> (lattice_.size ());
    std::copy (values.begin (), values.end (), storage_.data ());
  }

  /* 1-D initializer_list constructor.  */
  nd_tensor (std::initializer_list<T> list)
    : storage_ (list.size ()), lattice_ (shape_type {list.size ()})
  {
    std::copy (list.begin (), list.end (), storage_.data ());
  }

  /* 2-D nested initializer_list constructor for matrices.  */
  nd_tensor (std::initializer_list<std::initializer_list<T>> rows)
    : storage_ (), lattice_ ()
  {
    const std::size_t n_rows = rows.size ();
    const std::size_t n_cols = n_rows > 0 ? rows.begin ()->size () : 0;
    lattice_ = lattice_descriptor (shape_type {n_rows, n_cols});
    storage_ = slab_buffer<T> (lattice_.size ());
    std::size_t idx = 0;
    for (const auto &row : rows)
      {
        if (row.size () != n_cols)
          {
            throw shape_error ("ragged 2D initializer list in nd_tensor");
          }
        for (const T &val : row)
          {
            storage_[idx++] = val;
          }
      }
  }

  /* Zero-copy view constructor sharing an existing slab_buffer.  */
  nd_tensor (slab_buffer<T> buf, lattice_descriptor lat) noexcept
    : storage_ (std::move (buf)), lattice_ (std::move (lat))
  {
  }

  /* Materialize any lazy tensor_expr_base into a contiguous nd_tensor.  */
  template <typename ExprDerived, typename ExprVal>
  nd_tensor (const tensor_expr_base<ExprDerived, ExprVal> &expr)
    : storage_ (expr.size ()), lattice_ (expr.shape ())
  {
    const std::size_t n = expr.size ();
    T *HYPER_MATH_RESTRICT dst = storage_.data ();
    if (n >= 262144)
      {
        parallel_for (0, n, 65536, [&] (std::size_t i) {
          dst[i] = static_cast<T> (expr.eval_at (i));
        });
      }
    else
      {
        for (std::size_t i = 0; i < n; ++i)
          {
            dst[i] = static_cast<T> (expr.eval_at (i));
          }
      }
  }

  template <typename ExprDerived, typename ExprVal>
  nd_tensor &
  operator= (const tensor_expr_base<ExprDerived, ExprVal> &expr)
  {
    nd_tensor materialized (expr);
    storage_ = std::move (materialized.storage_);
    lattice_ = std::move (materialized.lattice_);
    return *this;
  }

  const shape_type &
  shape () const noexcept
  {
    return lattice_.shape ();
  }

  const stride_type &
  strides () const noexcept
  {
    return lattice_.strides ();
  }

  const lattice_descriptor &
  lattice () const noexcept
  {
    return lattice_;
  }

  std::size_t
  ndim () const noexcept
  {
    return lattice_.ndim ();
  }

  std::size_t
  size () const noexcept
  {
    return lattice_.size ();
  }

  bool
  empty () const noexcept
  {
    return lattice_.size () == 0;
  }

  bool
  is_contiguous () const noexcept
  {
    return lattice_.is_contiguous ();
  }

  T *
  raw_data () noexcept
  {
    return storage_.data () + lattice_.offset ();
  }

  const T *
  raw_data () const noexcept
  {
    return storage_.data () + lattice_.offset ();
  }

  const slab_buffer<T> &
  storage () const noexcept
  {
    return storage_;
  }

  /* Expression template interface methods.  */
  HYPER_MATH_ALWAYS_INLINE T
  eval_at (std::size_t flat_idx) const noexcept
  {
    return storage_[static_cast<std::size_t> (
      lattice_.flat_to_physical (flat_idx))];
  }

  HYPER_MATH_ALWAYS_INLINE T
  eval_with_lattice (std::size_t flat_idx,
                     const lattice_descriptor &bcast_lat) const noexcept
  {
    return storage_[static_cast<std::size_t> (
      bcast_lat.flat_to_physical (flat_idx))];
  }

  lattice_descriptor
  broadcast_lattice (const shape_type &target_shape) const
  {
    return lattice_.broadcast_to (target_shape);
  }

  /* Flat logical element access (supports non-contiguous & broadcast views). */
  HYPER_MATH_ALWAYS_INLINE T &
  flat (std::size_t flat_idx)
  {
    return storage_[static_cast<std::size_t> (
      lattice_.flat_to_physical (flat_idx))];
  }

  HYPER_MATH_ALWAYS_INLINE const T &
  flat (std::size_t flat_idx) const
  {
    return storage_[static_cast<std::size_t> (
      lattice_.flat_to_physical (flat_idx))];
  }

  HYPER_MATH_ALWAYS_INLINE T &
  operator[] (std::size_t flat_idx)
  {
    return flat (flat_idx);
  }

  HYPER_MATH_ALWAYS_INLINE const T &
  operator[] (std::size_t flat_idx) const
  {
    return flat (flat_idx);
  }

  /* Zero-allocation 2-D coordinate indexing fast path: tensor (r, c).  */
  HYPER_MATH_ALWAYS_INLINE T &
  operator() (std::size_t r, std::size_t c)
  {
    return storage_[static_cast<std::size_t> (
      lattice_.coords_to_physical_2d (r, c))];
  }

  HYPER_MATH_ALWAYS_INLINE const T &
  operator() (std::size_t r, std::size_t c) const
  {
    return storage_[static_cast<std::size_t> (
      lattice_.coords_to_physical_2d (r, c))];
  }

  /* Variadic N-D coordinate indexing: tensor (i, j, k).  */
  template <integral_scalar... Indices>
  T &
  operator() (Indices... indices)
  {
    std::vector<std::size_t> coords {static_cast<std::size_t> (indices)...};
    const std::ptrdiff_t phys = lattice_.coords_to_physical (coords);
    return storage_[static_cast<std::size_t> (phys)];
  }

  template <integral_scalar... Indices>
  const T &
  operator() (Indices... indices) const
  {
    std::vector<std::size_t> coords {static_cast<std::size_t> (indices)...};
    const std::ptrdiff_t phys = lattice_.coords_to_physical (coords);
    return storage_[static_cast<std::size_t> (phys)];
  }

  T &
  at (const std::vector<std::size_t> &coords)
  {
    const std::ptrdiff_t phys = lattice_.coords_to_physical (coords);
    return storage_[static_cast<std::size_t> (phys)];
  }

  const T &
  at (const std::vector<std::size_t> &coords) const
  {
    const std::ptrdiff_t phys = lattice_.coords_to_physical (coords);
    return storage_[static_cast<std::size_t> (phys)];
  }

  /* Return a deep contiguous copy.  */
  nd_tensor
  copy () const
  {
    nd_tensor out (shape ());
    const std::size_t n = size ();
    for (std::size_t i = 0; i < n; ++i)
      {
        out.storage_[i] = eval_at (i);
      }
    return out;
  }

  nd_tensor
  as_contiguous () const
  {
    if (is_contiguous ())
      {
        return *this;
      }
    return copy ();
  }

  template <typename U>
  nd_tensor<U>
  astype () const
  {
    nd_tensor<U> out (shape ());
    const std::size_t n = size ();
    for (std::size_t i = 0; i < n; ++i)
      {
        out[i] = static_cast<U> (eval_at (i));
      }
    return out;
  }

  /* Zero-copy view operations (or contiguous copy if non-contiguous reshape). */
  nd_tensor
  slice (const std::vector<slice_range> &ranges) const
  {
    return nd_tensor (storage_, lattice_.slice (ranges));
  }

  nd_tensor
  transpose (const std::vector<std::size_t> &axes = {}) const
  {
    return nd_tensor (storage_, lattice_.transpose (axes));
  }

  nd_tensor
  t () const
  {
    return transpose ();
  }

  nd_tensor
  broadcast_to (const shape_type &target_shape) const
  {
    return nd_tensor (storage_, lattice_.broadcast_to (target_shape));
  }

  nd_tensor
  reshape (shape_type new_shape) const
  {
    std::size_t new_total = 1;
    for (std::size_t d : new_shape)
      {
        new_total *= d;
      }
    if (new_total != size ())
      {
        throw shape_error ("cannot reshape tensor of size "
                           + std::to_string (size ()) + " into shape "
                           + format_shape (new_shape));
      }
    if (is_contiguous ())
      {
        return nd_tensor (storage_, lattice_descriptor (std::move (new_shape)));
      }
    nd_tensor contig = copy ();
    return nd_tensor (contig.storage_,
                      lattice_descriptor (std::move (new_shape)));
  }

  nd_tensor
  ravel () const
  {
    return reshape (shape_type {size ()});
  }

  nd_tensor
  flatten () const
  {
    return copy ().reshape (shape_type {size ()});
  }

  nd_tensor
  squeeze () const
  {
    shape_type new_shape;
    for (std::size_t d : shape ())
      {
        if (d != 1)
          {
            new_shape.push_back (d);
          }
      }
    if (new_shape.empty ())
      {
        new_shape.push_back (1);
      }
    return reshape (std::move (new_shape));
  }

  nd_tensor
  expand_dims (std::size_t axis) const
  {
    if (axis > ndim ())
      {
        throw index_error ("expand_dims axis out of bounds");
      }
    shape_type new_shape = shape ();
    new_shape.insert (new_shape.begin () + static_cast<std::ptrdiff_t> (axis),
                      1);
    return reshape (std::move (new_shape));
  }

  /* Boolean mask filtering (NumPy arr[mask]).  */
  template <typename MaskT>
  nd_tensor
  filter (const nd_tensor<MaskT> &mask) const
  {
    if (mask.size () != size ())
      {
        throw shape_error ("boolean mask size does not match tensor size");
      }
    std::vector<T> selected;
    selected.reserve (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (static_cast<bool> (mask.eval_at (i)))
          {
            selected.push_back (eval_at (i));
          }
      }
    return nd_tensor (shape_type {selected.size ()}, selected);
  }

  /* Fancy indexing: take elements by flat indices.  */
  nd_tensor
  take (const std::vector<std::size_t> &indices) const
  {
    nd_tensor out (shape_type {indices.size ()});
    for (std::size_t i = 0; i < indices.size (); ++i)
      {
        if (indices[i] >= size ())
          {
            throw index_error ("take index out of bounds");
          }
        out[i] = eval_at (indices[i]);
      }
    return out;
  }

  void
  fill (const T &val)
  {
    for (std::size_t i = 0; i < size (); ++i)
      {
        flat (i) = val;
      }
  }

  std::vector<T>
  to_vector () const
  {
    std::vector<T> out (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        out[i] = eval_at (i);
      }
    return out;
  }

  /* In-place compound assignment operators with broadcasting and
     self-aliasing protection.  */
  template <typename ExprDerived, typename ExprVal>
  nd_tensor &
  operator+= (const tensor_expr_base<ExprDerived, ExprVal> &rhs)
  {
    const auto bcast_lat =
      lattice_descriptor (rhs.shape ()).broadcast_to (shape ());
    std::vector<T> staged (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        const std::size_t ri =
          static_cast<std::size_t> (bcast_lat.flat_to_physical (i));
        staged[i] = eval_at (i) + static_cast<T> (rhs.eval_at (ri));
      }
    for (std::size_t i = 0; i < size (); ++i)
      {
        flat (i) = staged[i];
      }
    return *this;
  }

  template <typename ExprDerived, typename ExprVal>
  nd_tensor &
  operator-= (const tensor_expr_base<ExprDerived, ExprVal> &rhs)
  {
    const auto bcast_lat =
      lattice_descriptor (rhs.shape ()).broadcast_to (shape ());
    std::vector<T> staged (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        const std::size_t ri =
          static_cast<std::size_t> (bcast_lat.flat_to_physical (i));
        staged[i] = eval_at (i) - static_cast<T> (rhs.eval_at (ri));
      }
    for (std::size_t i = 0; i < size (); ++i)
      {
        flat (i) = staged[i];
      }
    return *this;
  }

  template <typename ExprDerived, typename ExprVal>
  nd_tensor &
  operator*= (const tensor_expr_base<ExprDerived, ExprVal> &rhs)
  {
    const auto bcast_lat =
      lattice_descriptor (rhs.shape ()).broadcast_to (shape ());
    std::vector<T> staged (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        const std::size_t ri =
          static_cast<std::size_t> (bcast_lat.flat_to_physical (i));
        staged[i] = eval_at (i) * static_cast<T> (rhs.eval_at (ri));
      }
    for (std::size_t i = 0; i < size (); ++i)
      {
        flat (i) = staged[i];
      }
    return *this;
  }

  template <typename ExprDerived, typename ExprVal>
  nd_tensor &
  operator/= (const tensor_expr_base<ExprDerived, ExprVal> &rhs)
  {
    const auto bcast_lat =
      lattice_descriptor (rhs.shape ()).broadcast_to (shape ());
    std::vector<T> staged (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        const std::size_t ri =
          static_cast<std::size_t> (bcast_lat.flat_to_physical (i));
        staged[i] = eval_at (i) / static_cast<T> (rhs.eval_at (ri));
      }
    for (std::size_t i = 0; i < size (); ++i)
      {
        flat (i) = staged[i];
      }
    return *this;
  }

  nd_tensor &
  operator+= (const T &scalar)
  {
    std::vector<T> staged (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        staged[i] = eval_at (i) + scalar;
      }
    for (std::size_t i = 0; i < size (); ++i)
      {
        flat (i) = staged[i];
      }
    return *this;
  }

  nd_tensor &
  operator-= (const T &scalar)
  {
    std::vector<T> staged (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        staged[i] = eval_at (i) - scalar;
      }
    for (std::size_t i = 0; i < size (); ++i)
      {
        flat (i) = staged[i];
      }
    return *this;
  }

  nd_tensor &
  operator*= (const T &scalar)
  {
    std::vector<T> staged (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        staged[i] = eval_at (i) * scalar;
      }
    for (std::size_t i = 0; i < size (); ++i)
      {
        flat (i) = staged[i];
      }
    return *this;
  }

  nd_tensor &
  operator/= (const T &scalar)
  {
    std::vector<T> staged (size ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        staged[i] = eval_at (i) / scalar;
      }
    for (std::size_t i = 0; i < size (); ++i)
      {
        flat (i) = staged[i];
      }
    return *this;
  }

  /* Static creation helpers (NumPy equivalents).  */
  static nd_tensor
  zeros (shape_type dims)
  {
    return nd_tensor (std::move (dims), T {0});
  }

  static nd_tensor
  ones (shape_type dims)
  {
    return nd_tensor (std::move (dims), T {1});
  }

  static nd_tensor
  full (shape_type dims, const T &val)
  {
    return nd_tensor (std::move (dims), val);
  }

  static nd_tensor
  arange (T start, T stop, T step = T {1})
  {
    if (step == T {0})
      {
        throw hyper_math_error ("arange step cannot be zero");
      }
    const double span =
      (static_cast<double> (stop) - static_cast<double> (start))
      / static_cast<double> (step);
    const std::size_t count =
      span > 0.0
      ? static_cast<std::size_t> (std::ceil (span * (1.0 - 1e-12)))
      : 0;
    nd_tensor out (shape_type {count});
    for (std::size_t i = 0; i < count; ++i)
      {
        out[i] = start + static_cast<T> (i) * step;
      }
    return out;
  }

  static nd_tensor
  linspace (T start, T stop, std::size_t num = 50, bool endpoint = true)
  {
    nd_tensor out (shape_type {num});
    if (num == 0)
      {
        return out;
      }
    if (num == 1)
      {
        out[0] = start;
        return out;
      }
    const double denom =
      endpoint ? static_cast<double> (num - 1) : static_cast<double> (num);
    const double step =
      (static_cast<double> (stop) - static_cast<double> (start)) / denom;
    for (std::size_t i = 0; i < num; ++i)
      {
        out[i] = static_cast<T> (static_cast<double> (start)
                                 + static_cast<double> (i) * step);
      }
    return out;
  }

  static nd_tensor
  logspace (T start_exp, T stop_exp, std::size_t num = 50, T base = T {10})
  {
    nd_tensor exponents = linspace (start_exp, stop_exp, num, true);
    for (std::size_t i = 0; i < num; ++i)
      {
        exponents[i] = static_cast<T> (
          std::pow (static_cast<double> (base),
                    static_cast<double> (exponents[i])));
      }
    return exponents;
  }

  static nd_tensor
  eye (std::size_t n, std::size_t m = 0, std::ptrdiff_t k = 0)
  {
    if (m == 0)
      {
        m = n;
      }
    nd_tensor out (shape_type {n, m}, T {0});
    for (std::size_t r = 0; r < n; ++r)
      {
        const std::ptrdiff_t c = static_cast<std::ptrdiff_t> (r) + k;
        if (c >= 0 && static_cast<std::size_t> (c) < m)
          {
            out (r, static_cast<std::size_t> (c)) = T {1};
          }
      }
    return out;
  }

  static nd_tensor
  diag (const nd_tensor &v)
  {
    if (v.ndim () == 1)
      {
        const std::size_t n = v.size ();
        nd_tensor out (shape_type {n, n}, T {0});
        for (std::size_t i = 0; i < n; ++i)
          {
            out (i, i) = v[i];
          }
        return out;
      }
    else if (v.ndim () == 2)
      {
        const std::size_t n = std::min (v.shape ()[0], v.shape ()[1]);
        nd_tensor out (shape_type {n});
        for (std::size_t i = 0; i < n; ++i)
          {
            out[i] = v (i, i);
          }
        return out;
      }
    throw shape_error ("diag requires a 1-D or 2-D tensor");
  }

  static nd_tensor
  random_uniform (shape_type dims,
                  double low = 0.0,
                  double high = 1.0,
                  random_engine &rng = global_rng ())
  {
    nd_tensor out (std::move (dims));
    for (std::size_t i = 0; i < out.size (); ++i)
      {
        out.storage_[i] = static_cast<T> (rng.uniform (low, high));
      }
    return out;
  }

  static nd_tensor
  random_normal (shape_type dims,
                 double mean = 0.0,
                 double stddev = 1.0,
                 random_engine &rng = global_rng ())
  {
    nd_tensor out (std::move (dims));
    for (std::size_t i = 0; i < out.size (); ++i)
      {
        out.storage_[i] = static_cast<T> (rng.normal (mean, stddev));
      }
    return out;
  }
};

/* Free-standing NumPy-style creation wrappers.  */
template <typename T = double>
nd_tensor<T>
zeros (shape_type dims)
{
  return nd_tensor<T>::zeros (std::move (dims));
}

template <typename T = double>
nd_tensor<T>
ones (shape_type dims)
{
  return nd_tensor<T>::ones (std::move (dims));
}

template <typename T = double>
nd_tensor<T>
full (shape_type dims, const T &val)
{
  return nd_tensor<T>::full (std::move (dims), val);
}

template <typename T = double>
nd_tensor<T>
arange (T start, T stop, T step = T {1})
{
  return nd_tensor<T>::arange (start, stop, step);
}

template <typename T = double>
nd_tensor<T>
linspace (T start, T stop, std::size_t num = 50, bool endpoint = true)
{
  return nd_tensor<T>::linspace (start, stop, num, endpoint);
}

template <typename T = double>
nd_tensor<T>
logspace (T start_exp, T stop_exp, std::size_t num = 50, T base = T {10})
{
  return nd_tensor<T>::logspace (start_exp, stop_exp, num, base);
}

template <typename T = double>
nd_tensor<T>
eye (std::size_t n, std::size_t m = 0, std::ptrdiff_t k = 0)
{
  return nd_tensor<T>::eye (n, m, k);
}

template <typename T>
nd_tensor<T>
diag (const nd_tensor<T> &v)
{
  return nd_tensor<T>::diag (v);
}

template <typename T = double>
nd_tensor<T>
random_uniform (shape_type dims,
                double low = 0.0,
                double high = 1.0,
                random_engine &rng = global_rng ())
{
  return nd_tensor<T>::random_uniform (std::move (dims), low, high, rng);
}

template <typename T = double>
nd_tensor<T>
random_normal (shape_type dims,
               double mean = 0.0,
               double stddev = 1.0,
               random_engine &rng = global_rng ())
{
  return nd_tensor<T>::random_normal (std::move (dims), mean, stddev, rng);
}

/* Extended NumPy-style random distributions and sampling utilities
   (random_randint, random_exponential, random_poisson, random_binomial,
    random_gamma, random_beta, random_chisquare, random_permutation,
    random_shuffle, random_choice).  */
inline nd_tensor<std::int64_t>
random_randint (shape_type dims,
                std::int64_t low,
                std::int64_t high,
                random_engine &rng = global_rng ())
{
  if (low >= high)
    {
      throw hyper_math_error ("random_randint requires low < high");
    }
  nd_tensor<std::int64_t> out (std::move (dims));
  const std::uint64_t span =
    static_cast<std::uint64_t> (high - low);
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      out[i] = low + static_cast<std::int64_t> (rng.next_u64 () % span);
    }
  return out;
}

inline nd_tensor<double>
random_exponential (shape_type dims,
                    double scale = 1.0,
                    random_engine &rng = global_rng ())
{
  if (scale <= 0.0)
    {
      throw hyper_math_error ("random_exponential requires scale > 0");
    }
  nd_tensor<double> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      out[i] = -scale * std::log (std::max (1e-300, rng.uniform (0.0, 1.0)));
    }
  return out;
}

inline nd_tensor<double>
random_poisson (shape_type dims,
                double lam = 1.0,
                random_engine &rng = global_rng ())
{
  if (lam < 0.0)
    {
      throw hyper_math_error ("random_poisson requires lam >= 0");
    }
  nd_tensor<double> out (std::move (dims));
  const double l_bound = std::exp (-lam);
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      std::size_t k = 0;
      double p = 1.0;
      do
        {
          ++k;
          p *= rng.uniform (0.0, 1.0);
        }
      while (p > l_bound);
      out[i] = static_cast<double> (k - 1);
    }
  return out;
}

inline nd_tensor<double>
random_binomial (shape_type dims,
                 std::size_t n_trials,
                 double p,
                 random_engine &rng = global_rng ())
{
  if (p < 0.0 || p > 1.0)
    {
      throw hyper_math_error ("random_binomial requires p in [0, 1]");
    }
  nd_tensor<double> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      std::size_t successes = 0;
      for (std::size_t t = 0; t < n_trials; ++t)
        {
          if (rng.uniform (0.0, 1.0) < p)
            {
              ++successes;
            }
        }
      out[i] = static_cast<double> (successes);
    }
  return out;
}

namespace detail
{
inline double
sample_standard_gamma (double alpha, random_engine &rng)
{
  if (alpha < 1.0)
    {
      const double u = std::max (1e-15, rng.uniform (0.0, 1.0));
      return sample_standard_gamma (alpha + 1.0, rng)
             * std::pow (u, 1.0 / alpha);
    }
  const double d = alpha - 1.0 / 3.0;
  const double c = 1.0 / std::sqrt (9.0 * d);
  for (;;)
    {
      double x = 0.0;
      double v = 0.0;
      do
        {
          x = rng.normal (0.0, 1.0);
          v = 1.0 + c * x;
        }
      while (v <= 0.0);
      v = v * v * v;
      const double u = std::max (1e-15, rng.uniform (0.0, 1.0));
      if (u < 1.0 - 0.0331 * (x * x) * (x * x))
        {
          return d * v;
        }
      if (std::log (u) < 0.5 * x * x + d * (1.0 - v + std::log (v)))
        {
          return d * v;
        }
    }
}
} // namespace detail

inline nd_tensor<double>
random_gamma (shape_type dims,
              double k_shape,
              double theta_scale = 1.0,
              random_engine &rng = global_rng ())
{
  if (k_shape <= 0.0 || theta_scale <= 0.0)
    {
      throw hyper_math_error ("random_gamma requires shape, scale > 0");
    }
  nd_tensor<double> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      out[i] = theta_scale * detail::sample_standard_gamma (k_shape, rng);
    }
  return out;
}

inline nd_tensor<double>
random_beta (shape_type dims,
             double a,
             double b,
             random_engine &rng = global_rng ())
{
  if (a <= 0.0 || b <= 0.0)
    {
      throw hyper_math_error ("random_beta requires a, b > 0");
    }
  nd_tensor<double> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      const double ga = detail::sample_standard_gamma (a, rng);
      const double gb = detail::sample_standard_gamma (b, rng);
      out[i] = (ga + gb > 0.0) ? (ga / (ga + gb)) : 0.5;
    }
  return out;
}

inline nd_tensor<double>
random_chisquare (shape_type dims,
                  double df,
                  random_engine &rng = global_rng ())
{
  if (df <= 0.0)
    {
      throw hyper_math_error ("random_chisquare requires df > 0");
    }
  return random_gamma (std::move (dims), 0.5 * df, 2.0, rng);
}

template <typename T>
void
random_shuffle (nd_tensor<T> &a, random_engine &rng = global_rng ())
{
  const std::size_t n = a.size ();
  for (std::size_t i = n; i > 1; --i)
    {
      const std::size_t j = static_cast<std::size_t> (rng.next_u64 () % i);
      std::swap (a[i - 1], a[j]);
    }
}

inline nd_tensor<double>
random_permutation (std::size_t n, random_engine &rng = global_rng ())
{
  nd_tensor<double> out (shape_type {n});
  for (std::size_t i = 0; i < n; ++i)
    {
      out[i] = static_cast<double> (i);
    }
  random_shuffle (out, rng);
  return out;
}

template <typename T>
nd_tensor<T>
random_choice (const nd_tensor<T> &a,
               std::size_t size,
               bool replace = true,
               random_engine &rng = global_rng ())
{
  if (a.size () == 0 || (!replace && size > a.size ()))
    {
      throw shape_error ("random_choice invalid population or sample size");
    }
  nd_tensor<T> out (shape_type {size});
  if (replace)
    {
      for (std::size_t i = 0; i < size; ++i)
        {
          const std::size_t idx =
            static_cast<std::size_t> (rng.next_u64 () % a.size ());
          out[i] = a.eval_at (idx);
        }
    }
  else
    {
      std::vector<std::size_t> perm (a.size ());
      for (std::size_t i = 0; i < a.size (); ++i)
        {
          perm[i] = i;
        }
      for (std::size_t i = 0; i < size; ++i)
        {
          const std::size_t j =
            i + static_cast<std::size_t> (rng.next_u64 () % (a.size () - i));
          std::swap (perm[i], perm[j]);
          out[i] = a.eval_at (perm[i]);
        }
    }
  return out;
}

/* Concatenate tensors along an axis (default axis 0).  */
template <typename T>
nd_tensor<T>
concatenate (const std::vector<nd_tensor<T>> &tensors, std::size_t axis = 0)
{
  if (tensors.empty ())
    {
      throw shape_error ("concatenate requires at least one tensor");
    }
  const std::size_t rank = tensors[0].ndim ();
  if (axis >= rank)
    {
      throw index_error ("concatenate axis out of bounds");
    }

  shape_type out_shape = tensors[0].shape ();
  out_shape[axis] = 0;
  for (const auto &t : tensors)
    {
      if (t.ndim () != rank)
        {
          throw shape_error ("all tensors in concatenate must share rank");
        }
      for (std::size_t d = 0; d < rank; ++d)
        {
          if (d != axis && t.shape ()[d] != out_shape[d])
            {
              throw shape_error ("non-concatenated axis dimension mismatch");
            }
        }
      out_shape[axis] += t.shape ()[axis];
    }

  nd_tensor<T> result (out_shape);
  std::size_t outer_extent = 1;
  for (std::size_t d = 0; d < axis; ++d)
    {
      outer_extent *= out_shape[d];
    }
  std::size_t inner_extent = 1;
  for (std::size_t d = axis + 1; d < rank; ++d)
    {
      inner_extent *= out_shape[d];
    }

  std::size_t dst_axis_offset = 0;
  for (const auto &t : tensors)
    {
      const std::size_t t_axis_len = t.shape ()[axis];
      for (std::size_t o = 0; o < outer_extent; ++o)
        {
          for (std::size_t a = 0; a < t_axis_len; ++a)
            {
              for (std::size_t in = 0; in < inner_extent; ++in)
                {
                  const std::size_t src_flat =
                    (o * t_axis_len + a) * inner_extent + in;
                  const std::size_t dst_flat =
                    (o * out_shape[axis] + (dst_axis_offset + a))
                    * inner_extent + in;
                  result[dst_flat] = t.eval_at (src_flat);
                }
            }
        }
      dst_axis_offset += t_axis_len;
    }

  return result;
}

template <typename T>
nd_tensor<T>
vstack (const std::vector<nd_tensor<T>> &tensors)
{
  std::vector<nd_tensor<T>> promoted;
  promoted.reserve (tensors.size ());
  for (const auto &t : tensors)
    {
      if (t.ndim () == 1)
        {
          promoted.push_back (t.reshape ({1, t.size ()}));
        }
      else
        {
          promoted.push_back (t);
        }
    }
  return concatenate (promoted, 0);
}

template <typename T>
nd_tensor<T>
hstack (const std::vector<nd_tensor<T>> &tensors)
{
  if (tensors.empty ())
    {
      throw shape_error ("hstack requires at least one tensor");
    }
  if (tensors[0].ndim () == 1)
    {
      return concatenate (tensors, 0);
    }
  return concatenate (tensors, 1);
}

/* Stream output for human-readable N-D tensor inspection.  */
template <typename T>
std::ostream &
operator<< (std::ostream &os, const nd_tensor<T> &t)
{
  os << "nd_tensor(shape=" << format_shape (t.shape ()) << ", data=[";
  const std::size_t n = t.size ();
  const std::size_t limit = std::min<std::size_t> (n, 32);
  for (std::size_t i = 0; i < limit; ++i)
    {
      if (i > 0)
        {
          os << ", ";
        }
      os << t.eval_at (i);
    }
  if (n > limit)
    {
      os << ", ...";
    }
  os << "])";
  return os;
}

/* Extended random distributions and grid constructors (Phase 20).  */
inline nd_tensor<double>
random_lognormal (shape_type dims,
                  double mean = 0.0,
                  double sigma = 1.0,
                  random_engine &rng = global_rng ())
{
  if (sigma <= 0.0)
    {
      throw hyper_math_error ("random_lognormal requires sigma > 0");
    }
  nd_tensor<double> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      out[i] = std::exp (rng.normal (mean, sigma));
    }
  return out;
}

inline nd_tensor<double>
random_weibull (shape_type dims,
                double a = 1.0,
                random_engine &rng = global_rng ())
{
  if (a <= 0.0)
    {
      throw hyper_math_error ("random_weibull requires shape a > 0");
    }
  nd_tensor<double> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      const double u = std::max (1e-300, rng.uniform (0.0, 1.0));
      out[i] = std::pow (-std::log (u), 1.0 / a);
    }
  return out;
}

inline nd_tensor<double>
random_pareto (shape_type dims,
               double a = 1.0,
               random_engine &rng = global_rng ())
{
  if (a <= 0.0)
    {
      throw hyper_math_error ("random_pareto requires shape a > 0");
    }
  nd_tensor<double> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      const double u = std::max (1e-300, rng.uniform (0.0, 1.0));
      out[i] = std::pow (u, -1.0 / a) - 1.0;
    }
  return out;
}

inline nd_tensor<double>
random_laplace (shape_type dims,
                double loc = 0.0,
                double scale = 1.0,
                random_engine &rng = global_rng ())
{
  if (scale <= 0.0)
    {
      throw hyper_math_error ("random_laplace requires scale > 0");
    }
  nd_tensor<double> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      const double u = rng.uniform (-0.5, 0.5);
      const double sgn = (u < 0.0) ? -1.0 : 1.0;
      out[i] = loc - scale * sgn
                     * std::log (std::max (1e-300, 1.0 - 2.0 * std::abs (u)));
    }
  return out;
}

inline nd_tensor<double>
random_logistic (shape_type dims,
                 double loc = 0.0,
                 double scale = 1.0,
                 random_engine &rng = global_rng ())
{
  if (scale <= 0.0)
    {
      throw hyper_math_error ("random_logistic requires scale > 0");
    }
  nd_tensor<double> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      const double u = std::clamp (rng.uniform (0.0, 1.0), 1e-15, 1.0 - 1e-15);
      out[i] = loc + scale * std::log (u / (1.0 - u));
    }
  return out;
}

inline nd_tensor<double>
random_gumbel (shape_type dims,
               double loc = 0.0,
               double scale = 1.0,
               random_engine &rng = global_rng ())
{
  if (scale <= 0.0)
    {
      throw hyper_math_error ("random_gumbel requires scale > 0");
    }
  nd_tensor<double> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      const double u = std::clamp (rng.uniform (0.0, 1.0), 1e-15, 1.0 - 1e-15);
      out[i] = loc - scale * std::log (-std::log (u));
    }
  return out;
}

inline nd_tensor<double>
random_rayleigh (shape_type dims,
                 double scale = 1.0,
                 random_engine &rng = global_rng ())
{
  if (scale <= 0.0)
    {
      throw hyper_math_error ("random_rayleigh requires scale > 0");
    }
  nd_tensor<double> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      const double u = std::max (1e-300, rng.uniform (0.0, 1.0));
      out[i] = scale * std::sqrt (-2.0 * std::log (u));
    }
  return out;
}

inline nd_tensor<double>
random_triangular (shape_type dims,
                   double left,
                   double mode,
                   double right,
                   random_engine &rng = global_rng ())
{
  if (!(left <= mode && mode <= right && left < right))
    {
      throw hyper_math_error ("random_triangular requires left <= mode <= right");
    }
  nd_tensor<double> out (std::move (dims));
  const double fc = (mode - left) / (right - left);
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      const double u = rng.uniform (0.0, 1.0);
      if (u <= fc)
        {
          out[i] = left + std::sqrt (u * (right - left) * (mode - left));
        }
      else
        {
          out[i] =
            right - std::sqrt ((1.0 - u) * (right - left) * (right - mode));
        }
    }
  return out;
}

inline nd_tensor<double>
random_wald (shape_type dims,
             double mean = 1.0,
             double scale = 1.0,
             random_engine &rng = global_rng ())
{
  if (mean <= 0.0 || scale <= 0.0)
    {
      throw hyper_math_error ("random_wald requires mean > 0 and scale > 0");
    }
  nd_tensor<double> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      const double v = rng.normal (0.0, 1.0);
      const double y = v * v;
      const double x =
        mean + (mean * mean * y) / (2.0 * scale)
        - (mean / (2.0 * scale))
            * std::sqrt (4.0 * mean * scale * y + mean * mean * y * y);
      const double u = rng.uniform (0.0, 1.0);
      out[i] = (u <= mean / (mean + x)) ? x : (mean * mean) / x;
    }
  return out;
}

inline nd_tensor<double>
random_vonmises (shape_type dims,
                 double mu = 0.0,
                 double kappa = 1.0,
                 random_engine &rng = global_rng ())
{
  if (kappa < 0.0)
    {
      throw hyper_math_error ("random_vonmises requires kappa >= 0");
    }
  nd_tensor<double> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      if (kappa < 1e-8)
        {
          out[i] = rng.uniform (-constants::pi, constants::pi);
          continue;
        }
      const double tau = 1.0 + std::sqrt (1.0 + 4.0 * kappa * kappa);
      const double rho = (tau - std::sqrt (2.0 * tau)) / (2.0 * kappa);
      const double r = (1.0 + rho * rho) / (2.0 * rho);
      for (;;)
        {
          const double u1 = rng.uniform (0.0, 1.0);
          const double z = std::cos (constants::pi * u1);
          const double f = (1.0 + r * z) / (r + z);
          const double c = kappa * (r - f);
          const double u2 = rng.uniform (0.0, 1.0);
          if (u2 < c * (2.0 - c) || u2 <= c * std::exp (1.0 - c))
            {
              const double u3 = rng.uniform (0.0, 1.0);
              const double theta =
                (u3 > 0.5 ? 1.0 : -1.0) * std::acos (std::clamp (f, -1.0, 1.0));
              out[i] = std::remainder (mu + theta, constants::two_pi);
              break;
            }
        }
    }
  return out;
}

inline nd_tensor<std::int64_t>
random_geometric (shape_type dims,
                  double p = 0.5,
                  random_engine &rng = global_rng ())
{
  if (p <= 0.0 || p > 1.0)
    {
      throw hyper_math_error ("random_geometric requires p in (0, 1]");
    }
  nd_tensor<std::int64_t> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      if (p == 1.0)
        {
          out[i] = 1;
        }
      else
        {
          const double u = std::max (1e-300, rng.uniform (0.0, 1.0));
          out[i] =
            static_cast<std::int64_t> (std::floor (std::log (u)
                                                   / std::log (1.0 - p)))
            + 1;
        }
    }
  return out;
}

inline nd_tensor<std::int64_t>
random_hypergeometric (shape_type dims,
                       std::int64_t ngood,
                       std::int64_t nbad,
                       std::int64_t nsample,
                       random_engine &rng = global_rng ())
{
  if (ngood < 0 || nbad < 0 || nsample < 0 || nsample > ngood + nbad)
    {
      throw hyper_math_error ("invalid parameters in random_hypergeometric");
    }
  nd_tensor<std::int64_t> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      std::int64_t g = ngood;
      std::int64_t b = nbad;
      std::int64_t k = 0;
      for (std::int64_t draw = 0; draw < nsample; ++draw)
        {
          const double prob =
            static_cast<double> (g) / static_cast<double> (g + b);
          if (rng.uniform (0.0, 1.0) < prob)
            {
              ++k;
              --g;
            }
          else
            {
              --b;
            }
        }
      out[i] = k;
    }
  return out;
}

inline nd_tensor<std::int64_t>
random_zipf (shape_type dims,
             double a = 2.0,
             random_engine &rng = global_rng ())
{
  if (a <= 1.0)
    {
      throw hyper_math_error ("random_zipf requires a > 1");
    }
  nd_tensor<std::int64_t> out (std::move (dims));
  const double am1 = a - 1.0;
  const double b = std::pow (2.0, am1);
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      for (;;)
        {
          const double u = std::max (1e-300, rng.uniform (0.0, 1.0));
          const double v = rng.uniform (0.0, 1.0);
          const double x = std::floor (std::pow (u, -1.0 / am1));
          const double t = std::pow (1.0 + 1.0 / x, am1);
          if (v * x * (t - 1.0) / (b - 1.0) <= t / b)
            {
              out[i] = static_cast<std::int64_t> (x);
              break;
            }
        }
    }
  return out;
}

inline nd_tensor<double>
random_standard_cauchy (shape_type dims, random_engine &rng = global_rng ())
{
  nd_tensor<double> out (std::move (dims));
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      const double u = rng.uniform (-0.5, 0.5);
      out[i] = std::tan (constants::pi * u);
    }
  return out;
}

inline nd_tensor<double>
random_standard_t (shape_type dims,
                   double df,
                   random_engine &rng = global_rng ())
{
  if (df <= 0.0)
    {
      throw hyper_math_error ("random_standard_t requires df > 0");
    }
  const nd_tensor<double> z = random_normal (dims, 0.0, 1.0, rng);
  const nd_tensor<double> v = random_chisquare ( std::move (dims), df, rng);
  nd_tensor<double> out (z.shape ());
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      out[i] = z[i] / std::sqrt (v[i] / df);
    }
  return out;
}

inline nd_tensor<double>
random_f (shape_type dims,
          double dfnum,
          double dfden,
          random_engine &rng = global_rng ())
{
  if (dfnum <= 0.0 || dfden <= 0.0)
    {
      throw hyper_math_error ("random_f requires dfnum > 0 and dfden > 0");
    }
  const nd_tensor<double> c1 = random_chisquare (dims, dfnum, rng);
  const nd_tensor<double> c2 = random_chisquare (std::move (dims), dfden, rng);
  nd_tensor<double> out (c1.shape ());
  for (std::size_t i = 0; i < out.size (); ++i)
    {
      out[i] = (c1[i] / dfnum) / std::max (1e-300, c2[i] / dfden);
    }
  return out;
}

inline nd_tensor<double>
random_dirichlet (const std::vector<double> &alpha,
                  std::size_t n_samples = 1,
                  random_engine &rng = global_rng ())
{
  const std::size_t k = alpha.size ();
  if (k < 2)
    {
      throw shape_error ("random_dirichlet requires at least 2 alpha parameters");
    }
  nd_tensor<double> out (shape_type {n_samples, k}, 0.0);
  for (std::size_t s = 0; s < n_samples; ++s)
    {
      double sum_g = 0.0;
      for (std::size_t j = 0; j < k; ++j)
        {
          const double g = random_gamma ({1}, alpha[j], 1.0, rng)[0];
          out (s, j) = g;
          sum_g += g;
        }
      for (std::size_t j = 0; j < k; ++j)
        {
          out (s, j) /= std::max (1e-300, sum_g);
        }
    }
  return out;
}

inline nd_tensor<std::int64_t>
random_multinomial (std::int64_t n,
                    const std::vector<double> &pvals,
                    std::size_t n_samples = 1,
                    random_engine &rng = global_rng ())
{
  const std::size_t k = pvals.size ();
  if (k == 0 || n < 0)
    {
      throw shape_error ("invalid parameters in random_multinomial");
    }
  nd_tensor<std::int64_t> out (shape_type {n_samples, k}, 0);
  for (std::size_t s = 0; s < n_samples; ++s)
    {
      std::int64_t rem_n = n;
      double rem_p = 1.0;
      for (std::size_t j = 0; j + 1 < k; ++j)
        {
          const double p = std::clamp (pvals[j] / std::max (1e-30, rem_p), 0.0, 1.0);
          std::int64_t cnt = 0;
          for (std::int64_t t = 0; t < rem_n; ++t)
            {
              if (rng.uniform (0.0, 1.0) < p)
                {
                  ++cnt;
                }
            }
          out (s, j) = cnt;
          rem_n -= cnt;
          rem_p -= pvals[j];
        }
      out (s, k - 1) = rem_n;
    }
  return out;
}

/* Construct tensor by evaluating a callable f(coords_vec) at each N-D index. */
template <typename Func>
auto
fromfunction (Func &&f, const shape_type &dims)
{
  const std::size_t nd = dims.size ();
  std::vector<std::size_t> zero_coord (nd, 0);
  using OutT = std::decay_t<decltype (f (zero_coord))>;
  nd_tensor<OutT> out (dims);
  const std::size_t total = out.size ();
  for (std::size_t flat = 0; flat < total; ++flat)
    {
      std::vector<std::size_t> coords (nd, 0);
      std::size_t rem = flat;
      for (std::size_t d = nd; d > 0; --d)
        {
          coords[d - 1] = rem % dims[d - 1];
          rem /= dims[d - 1];
        }
      out[flat] = f (coords);
    }
  return out;
}

/* Return an (N+1)-D dense grid tensor of indices for a given shape.  */
inline nd_tensor<std::int64_t>
indices (const shape_type &dims)
{
  const std::size_t nd = dims.size ();
  shape_type out_shape;
  out_shape.reserve (nd + 1);
  out_shape.push_back (nd);
  std::size_t slice_sz = 1;
  for (std::size_t d : dims)
    {
      out_shape.push_back (d);
      slice_sz *= d;
    }
  nd_tensor<std::int64_t> out (out_shape, 0);
  for (std::size_t flat = 0; flat < slice_sz; ++flat)
    {
      std::size_t rem = flat;
      for (std::size_t d = nd; d > 0; --d)
        {
          const std::size_t c = rem % dims[d - 1];
          rem /= dims[d - 1];
          out[(d - 1) * slice_sz + flat] = static_cast<std::int64_t> (c);
        }
    }
  return out;
}

/* Return (rows, cols) indices of an (n, n) array where mask_fn(r, c) is true. */
template <typename MaskFunc>
std::pair<nd_tensor<std::size_t>, nd_tensor<std::size_t>>
mask_indices (std::size_t n, MaskFunc &&mask_fn)
{
  std::vector<std::size_t> rows;
  std::vector<std::size_t> cols;
  for (std::size_t r = 0; r < n; ++r)
    {
      for (std::size_t c = 0; c < n; ++c)
        {
          if (mask_fn (r, c))
            {
              rows.push_back (r);
              cols.push_back (c);
            }
        }
    }
  return {
    nd_tensor<std::size_t> (shape_type {rows.size ()}, std::move (rows)),
    nd_tensor<std::size_t> (shape_type {cols.size ()}, std::move (cols))
  };
}

} // namespace hyper_math::tensor

#endif /* HYPER_MATH_TENSOR_ND_TENSOR_HPP */
