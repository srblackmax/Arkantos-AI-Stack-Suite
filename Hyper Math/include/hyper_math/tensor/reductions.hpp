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

#ifndef HYPER_MATH_TENSOR_REDUCTIONS_HPP
#define HYPER_MATH_TENSOR_REDUCTIONS_HPP

#include "hyper_math/core/parallel.hpp"
#include "hyper_math/tensor/ufunc.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace hyper_math::tensor
{

/* Generic axis reduction helper. Collapses `axis` using `reducer`.  */
template <typename T, typename AccT, typename InitFn, typename StepFn, typename FinalFn>
nd_tensor<AccT>
reduce_along_axis (const nd_tensor<T> &input,
                   std::size_t axis,
                   InitFn init_fn,
                   StepFn step_fn,
                   FinalFn final_fn,
                   bool keepdims = false)
{
  const shape_type &in_shape = input.shape ();
  if (axis >= in_shape.size ())
    {
      throw index_error ("reduction axis out of bounds");
    }

  const std::size_t axis_extent = in_shape[axis];
  if (axis_extent == 0)
    {
      throw shape_error ("cannot reduce along zero-length axis");
    }

  shape_type out_shape;
  for (std::size_t d = 0; d < in_shape.size (); ++d)
    {
      if (d == axis)
        {
          if (keepdims)
            {
              out_shape.push_back (1);
            }
        }
      else
        {
          out_shape.push_back (in_shape[d]);
        }
    }
  if (out_shape.empty ())
    {
      out_shape.push_back (1);
    }

  std::size_t outer_size = 1;
  for (std::size_t d = 0; d < axis; ++d)
    {
      outer_size *= in_shape[d];
    }
  std::size_t inner_size = 1;
  for (std::size_t d = axis + 1; d < in_shape.size (); ++d)
    {
      inner_size *= in_shape[d];
    }

  nd_tensor<AccT> result (out_shape);
  for (std::size_t o = 0; o < outer_size; ++o)
    {
      for (std::size_t in = 0; in < inner_size; ++in)
        {
          auto state = init_fn ();
          for (std::size_t k = 0; k < axis_extent; ++k)
            {
              const std::size_t src_flat =
                (o * axis_extent + k) * inner_size + in;
              step_fn (state, input.eval_at (src_flat), k);
            }
          result[o * inner_size + in] = final_fn (state, axis_extent);
        }
    }

  return result;
}

/* Global and axis-specific reductions.  */
template <typename Derived, typename T>
T
sum (const tensor_expr_base<Derived, T> &expr)
{
  T acc = T {0};
  const std::size_t n = expr.size ();
  for (std::size_t i = 0; i < n; ++i)
    {
      acc += expr.eval_at (i);
    }
  return acc;
}

template <typename T>
nd_tensor<T>
sum (const nd_tensor<T> &t, std::size_t axis, bool keepdims = false)
{
  return reduce_along_axis<T, T> (
    t, axis,
    [] () { return T {0}; },
    [] (T &acc, T val, std::size_t) { acc += val; },
    [] (T acc, std::size_t) { return acc; },
    keepdims);
}

template <typename Derived, typename T>
T
prod (const tensor_expr_base<Derived, T> &expr)
{
  T acc = T {1};
  const std::size_t n = expr.size ();
  for (std::size_t i = 0; i < n; ++i)
    {
      acc *= expr.eval_at (i);
    }
  return acc;
}

template <typename T>
nd_tensor<T>
prod (const nd_tensor<T> &t, std::size_t axis, bool keepdims = false)
{
  return reduce_along_axis<T, T> (
    t, axis,
    [] () { return T {1}; },
    [] (T &acc, T val, std::size_t) { acc *= val; },
    [] (T acc, std::size_t) { return acc; },
    keepdims);
}

template <typename Derived, typename T>
double
mean (const tensor_expr_base<Derived, T> &expr)
{
  if (expr.size () == 0)
    {
      throw shape_error ("cannot compute mean of empty tensor");
    }
  double total = 0.0;
  const std::size_t n = expr.size ();
  for (std::size_t i = 0; i < n; ++i)
    {
      total += static_cast<double> (expr.eval_at (i));
    }
  return total / static_cast<double> (n);
}

template <typename T>
nd_tensor<double>
mean (const nd_tensor<T> &t, std::size_t axis, bool keepdims = false)
{
  return reduce_along_axis<T, double> (
    t, axis,
    [] () { return 0.0; },
    [] (double &acc, T val, std::size_t) {
      acc += static_cast<double> (val);
    },
    [] (double acc, std::size_t count) {
      return acc / static_cast<double> (count);
    },
    keepdims);
}

/* Welford single-pass numerically stable variance & standard deviation.  */
template <typename Derived, typename T>
double
sample_var (const tensor_expr_base<Derived, T> &expr, std::size_t ddof = 1)
{
  const std::size_t n = expr.size ();
  if (n <= ddof)
    {
      throw shape_error ("tensor size must exceed ddof in variance");
    }
  double m = 0.0;
  double m2 = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      const double x = static_cast<double> (expr.eval_at (i));
      const double delta = x - m;
      m += delta / static_cast<double> (i + 1);
      const double delta2 = x - m;
      m2 += delta * delta2;
    }
  return m2 / static_cast<double> (n - ddof);
}

template <typename Derived, typename T>
double
var (const tensor_expr_base<Derived, T> &expr)
{
  return sample_var (expr, 0);
}

template <typename T>
nd_tensor<double>
var (const nd_tensor<T> &t,
     std::size_t axis,
     std::size_t ddof = 0,
     bool keepdims = false)
{
  if (axis < t.ndim () && t.shape ()[axis] <= ddof)
    {
      throw shape_error ("axis extent must exceed ddof in variance");
    }
  struct welford_state
  {
    double mean = 0.0;
    double m2 = 0.0;
    std::size_t count = 0;
  };

  return reduce_along_axis<T, double> (
    t, axis,
    [] () { return welford_state {}; },
    [] (welford_state &st, T val, std::size_t) {
      st.count += 1;
      const double x = static_cast<double> (val);
      const double delta = x - st.mean;
      st.mean += delta / static_cast<double> (st.count);
      st.m2 += delta * (x - st.mean);
    },
    [ddof] (const welford_state &st, std::size_t count) {
      return st.m2 / static_cast<double> (count - ddof);
    },
    keepdims);
}

template <typename Derived, typename T>
double
sample_std_dev (const tensor_expr_base<Derived, T> &expr, std::size_t ddof = 1)
{
  return std::sqrt (sample_var (expr, ddof));
}

template <typename Derived, typename T>
double
std_dev (const tensor_expr_base<Derived, T> &expr)
{
  return std::sqrt (sample_var (expr, 0));
}

template <typename T>
nd_tensor<double>
std_dev (const nd_tensor<T> &t,
         std::size_t axis,
         std::size_t ddof = 0,
         bool keepdims = false)
{
  return nd_tensor<double> (sqrt (var (t, axis, ddof, keepdims)));
}

template <typename Derived, typename T>
T
min (const tensor_expr_base<Derived, T> &expr)
{
  if (expr.size () == 0)
    {
      throw shape_error ("min of empty tensor");
    }
  T best = expr.eval_at (0);
  for (std::size_t i = 1; i < expr.size (); ++i)
    {
      const T val = expr.eval_at (i);
      if constexpr (std::is_floating_point_v<T>)
        {
          if (std::isnan (static_cast<double> (val)))
            {
              return val;
            }
        }
      best = std::min (best, val);
    }
  return best;
}

template <typename T>
nd_tensor<T>
min (const nd_tensor<T> &t, std::size_t axis, bool keepdims = false)
{
  return reduce_along_axis<T, T> (
    t, axis,
    [] () { return T {0}; },
    [] (T &acc, T val, std::size_t k) {
      if (k == 0)
        {
          acc = val;
          return;
        }
      if constexpr (std::is_floating_point_v<T>)
        {
          if (std::isnan (static_cast<double> (val)))
            {
              acc = val;
              return;
            }
          if (std::isnan (static_cast<double> (acc)))
            {
              return;
            }
        }
      acc = std::min (acc, val);
    },
    [] (T acc, std::size_t) { return acc; },
    keepdims);
}

template <typename Derived, typename T>
T
max (const tensor_expr_base<Derived, T> &expr)
{
  if (expr.size () == 0)
    {
      throw shape_error ("max of empty tensor");
    }
  T best = expr.eval_at (0);
  for (std::size_t i = 1; i < expr.size (); ++i)
    {
      const T val = expr.eval_at (i);
      if constexpr (std::is_floating_point_v<T>)
        {
          if (std::isnan (static_cast<double> (val)))
            {
              return val;
            }
        }
      best = std::max (best, val);
    }
  return best;
}

template <typename T>
nd_tensor<T>
max (const nd_tensor<T> &t, std::size_t axis, bool keepdims = false)
{
  return reduce_along_axis<T, T> (
    t, axis,
    [] () { return T {0}; },
    [] (T &acc, T val, std::size_t k) {
      if (k == 0)
        {
          acc = val;
          return;
        }
      if constexpr (std::is_floating_point_v<T>)
        {
          if (std::isnan (static_cast<double> (val)))
            {
              acc = val;
              return;
            }
          if (std::isnan (static_cast<double> (acc)))
            {
              return;
            }
        }
      acc = std::max (acc, val);
    },
    [] (T acc, std::size_t) { return acc; },
    keepdims);
}

template <typename Derived, typename T>
std::size_t
argmin (const tensor_expr_base<Derived, T> &expr)
{
  if (expr.size () == 0)
    {
      throw shape_error ("argmin of empty tensor");
    }
  T best = expr.eval_at (0);
  std::size_t best_idx = 0;
  for (std::size_t i = 1; i < expr.size (); ++i)
    {
      const T v = expr.eval_at (i);
      if (v < best)
        {
          best = v;
          best_idx = i;
        }
    }
  return best_idx;
}

template <typename Derived, typename T>
std::size_t
argmax (const tensor_expr_base<Derived, T> &expr)
{
  if (expr.size () == 0)
    {
      throw shape_error ("argmax of empty tensor");
    }
  T best = expr.eval_at (0);
  std::size_t best_idx = 0;
  for (std::size_t i = 1; i < expr.size (); ++i)
    {
      const T v = expr.eval_at (i);
      if (v > best)
        {
          best = v;
          best_idx = i;
        }
    }
  return best_idx;
}

template <typename Derived, typename T>
bool
all (const tensor_expr_base<Derived, T> &expr)
{
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      if (!static_cast<bool> (expr.eval_at (i)))
        {
          return false;
        }
    }
  return true;
}

template <typename Derived, typename T>
bool
any (const tensor_expr_base<Derived, T> &expr)
{
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      if (static_cast<bool> (expr.eval_at (i)))
        {
          return true;
        }
    }
  return false;
}

template <typename Derived, typename T>
std::size_t
count_nonzero (const tensor_expr_base<Derived, T> &expr)
{
  std::size_t cnt = 0;
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      if (expr.eval_at (i) != T {0})
        {
          ++cnt;
        }
    }
  return cnt;
}

/* Cumulative sum and product.  */
template <typename Derived, typename T>
nd_tensor<T>
cumsum (const tensor_expr_base<Derived, T> &expr)
{
  nd_tensor<T> out (shape_type {expr.size ()});
  T running = T {0};
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      running += expr.eval_at (i);
      out[i] = running;
    }
  return out;
}

template <typename Derived, typename T>
nd_tensor<T>
cumprod (const tensor_expr_base<Derived, T> &expr)
{
  nd_tensor<T> out (shape_type {expr.size ()});
  T running = T {1};
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      running *= expr.eval_at (i);
      out[i] = running;
    }
  return out;
}

template <typename Derived, typename T>
nd_tensor<T>
diff (const tensor_expr_base<Derived, T> &expr, std::size_t n_order = 1)
{
  nd_tensor<T> current (expr);
  current = current.ravel ();
  for (std::size_t k = 0; k < n_order; ++k)
    {
      if (current.size () <= 1)
        {
          return nd_tensor<T> (shape_type {0});
        }
      nd_tensor<T> next (shape_type {current.size () - 1});
      for (std::size_t i = 0; i + 1 < current.size (); ++i)
        {
          next[i] = current[i + 1] - current[i];
        }
      current = std::move (next);
    }
  return current;
}

/* Sorting, argsort, unique, percentile, and median.  */
template <typename Derived, typename T>
nd_tensor<T>
sort (const tensor_expr_base<Derived, T> &expr)
{
  nd_tensor<T> out (expr);
  std::sort (out.raw_data (), out.raw_data () + out.size ());
  return out;
}

template <typename Derived, typename T>
nd_tensor<std::size_t>
argsort (const tensor_expr_base<Derived, T> &expr)
{
  nd_tensor<std::size_t> indices (shape_type {expr.size ()});
  std::iota (indices.raw_data (), indices.raw_data () + indices.size (), 0);
  std::stable_sort (indices.raw_data (),
                    indices.raw_data () + indices.size (),
                    [&expr] (std::size_t a, std::size_t b) {
                      return expr.eval_at (a) < expr.eval_at (b);
                    });
  return indices;
}

template <typename Derived, typename T>
nd_tensor<T>
unique (const tensor_expr_base<Derived, T> &expr)
{
  std::vector<T> vals (expr.size ());
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      vals[i] = expr.eval_at (i);
    }
  std::sort (vals.begin (), vals.end ());
  vals.erase (std::unique (vals.begin (), vals.end ()), vals.end ());
  return nd_tensor<T> (shape_type {vals.size ()}, vals);
}

template <typename Derived, typename T>
double
percentile (const tensor_expr_base<Derived, T> &expr, double q_percent)
{
  if (expr.size () == 0)
    {
      throw shape_error ("percentile of empty tensor");
    }
  if (q_percent < 0.0 || q_percent > 100.0)
    {
      throw hyper_math_error ("percentile q must lie in [0, 100]");
    }
  std::vector<double> sorted (expr.size ());
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      sorted[i] = static_cast<double> (expr.eval_at (i));
    }
  std::sort (sorted.begin (), sorted.end ());
  const double pos = (q_percent / 100.0) * static_cast<double> (sorted.size () - 1);
  const std::size_t lo = static_cast<std::size_t> (std::floor (pos));
  const std::size_t hi = static_cast<std::size_t> (std::ceil (pos));
  const double frac = pos - static_cast<double> (lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

template <typename Derived, typename T>
double
median (const tensor_expr_base<Derived, T> &expr)
{
  return percentile (expr, 50.0);
}

/* Matrix multiplication, dot product, outer product, Kronecker product,
   and trace.  */
template <typename T>
nd_tensor<T>
matmul (const nd_tensor<T> &a, const nd_tensor<T> &b)
{
  if (a.ndim () == 1 && b.ndim () == 1)
    {
      if (a.size () != b.size ())
        {
          throw shape_error ("1D dot product length mismatch");
        }
      T s = T {0};
      for (std::size_t i = 0; i < a.size (); ++i)
        {
          s += a.eval_at (i) * b.eval_at (i);
        }
      return nd_tensor<T> (shape_type {1}, s);
    }

  if (a.ndim () == 2 && b.ndim () == 1)
    {
      const std::size_t m = a.shape ()[0];
      const std::size_t k = a.shape ()[1];
      if (b.size () != k)
        {
          throw shape_error ("matrix-vector inner dimension mismatch");
        }
      nd_tensor<T> out (shape_type {m}, T {0});
      for (std::size_t i = 0; i < m; ++i)
        {
          T acc = T {0};
          for (std::size_t p = 0; p < k; ++p)
            {
              acc += a (i, p) * b[p];
            }
          out[i] = acc;
        }
      return out;
    }

  if (a.ndim () == 1 && b.ndim () == 2)
    {
      const std::size_t k = a.size ();
      if (b.shape ()[0] != k)
        {
          throw shape_error ("vector-matrix inner dimension mismatch");
        }
      const std::size_t n = b.shape ()[1];
      nd_tensor<T> out (shape_type {n}, T {0});
      for (std::size_t p = 0; p < k; ++p)
        {
          const T ap = a[p];
          for (std::size_t j = 0; j < n; ++j)
            {
              out[j] += ap * b (p, j);
            }
        }
      return out;
    }

  if (a.ndim () == 2 && b.ndim () == 2)
    {
      const std::size_t m = a.shape ()[0];
      const std::size_t k = a.shape ()[1];
      if (b.shape ()[0] != k)
        {
          throw shape_error ("matmul inner dimensions do not match");
        }
      const std::size_t n = b.shape ()[1];
      nd_tensor<T> out (shape_type {m, n}, T {0});
      const std::size_t ops = m * k * n;
      const std::size_t grain =
        (k * n > 0) ? std::max<std::size_t> (1, 16384 / (k * n)) : m;

      /* L1/L2 Cache-Tiled & SIMD-Vectorizable Fast Path for Contiguous 2D
         Matrices (eliminates stride arithmetic in the hot inner loop).  */
      if (a.is_contiguous () && b.is_contiguous ())
        {
          const T *HYPER_MATH_RESTRICT a_ptr = a.raw_data ();
          const T *HYPER_MATH_RESTRICT b_ptr = b.raw_data ();
          T *HYPER_MATH_RESTRICT c_ptr = out.raw_data ();
          constexpr std::size_t tile_k = 64;

          auto compute_row = [&] (std::size_t i) {
            const T *HYPER_MATH_RESTRICT a_row = a_ptr + i * k;
            T *HYPER_MATH_RESTRICT c_row = c_ptr + i * n;
            for (std::size_t p0 = 0; p0 < k; p0 += tile_k)
              {
                const std::size_t p_end = std::min (p0 + tile_k, k);
                for (std::size_t p = p0; p < p_end; ++p)
                  {
                    const T a_ip = a_row[p];
                    const T *HYPER_MATH_RESTRICT b_row = b_ptr + p * n;
                    for (std::size_t j = 0; j < n; ++j)
                      {
                        c_row[j] += a_ip * b_row[j];
                      }
                  }
              }
          };

          auto compute_4rows = [&] (std::size_t i0) {
            const T *HYPER_MATH_RESTRICT a0 = a_ptr + (i0 + 0) * k;
            const T *HYPER_MATH_RESTRICT a1 = a_ptr + (i0 + 1) * k;
            const T *HYPER_MATH_RESTRICT a2 = a_ptr + (i0 + 2) * k;
            const T *HYPER_MATH_RESTRICT a3 = a_ptr + (i0 + 3) * k;
            T *HYPER_MATH_RESTRICT c0 = c_ptr + (i0 + 0) * n;
            T *HYPER_MATH_RESTRICT c1 = c_ptr + (i0 + 1) * n;
            T *HYPER_MATH_RESTRICT c2 = c_ptr + (i0 + 2) * n;
            T *HYPER_MATH_RESTRICT c3 = c_ptr + (i0 + 3) * n;
            for (std::size_t p0 = 0; p0 < k; p0 += tile_k)
              {
                const std::size_t p_end = std::min (p0 + tile_k, k);
                for (std::size_t p = p0; p < p_end; ++p)
                  {
                    const T v0 = a0[p];
                    const T v1 = a1[p];
                    const T v2 = a2[p];
                    const T v3 = a3[p];
                    const T *HYPER_MATH_RESTRICT b_row = b_ptr + p * n;
                    for (std::size_t j = 0; j < n; ++j)
                      {
                        const T bv = b_row[j];
                        c0[j] += v0 * bv;
                        c1[j] += v1 * bv;
                        c2[j] += v2 * bv;
                        c3[j] += v3 * bv;
                      }
                  }
              }
          };

          const std::size_t n_blocks4 = m / 4;
          if (ops >= 32768 && n_blocks4 > 0)
            {
              const std::size_t block_grain
                = std::max<std::size_t> (1, grain / 4);
              parallel_for (0, n_blocks4, block_grain, [&] (std::size_t blk) {
                compute_4rows (blk * 4);
              });
              for (std::size_t i = n_blocks4 * 4; i < m; ++i)
                {
                  compute_row (i);
                }
            }
          else
            {
              std::size_t i = 0;
              for (; i + 4 <= m; i += 4)
                {
                  compute_4rows (i);
                }
              for (; i < m; ++i)
                {
                  compute_row (i);
                }
            }
          return out;
        }

      if (ops >= 32768)
        {
          parallel_for (0, m, grain, [&] (std::size_t i) {
            for (std::size_t p = 0; p < k; ++p)
              {
                const T a_ip = a (i, p);
                for (std::size_t j = 0; j < n; ++j)
                  {
                    out (i, j) += a_ip * b (p, j);
                  }
              }
          });
        }
      else
        {
          for (std::size_t i = 0; i < m; ++i)
            {
              for (std::size_t p = 0; p < k; ++p)
                {
                  const T a_ip = a (i, p);
                  for (std::size_t j = 0; j < n; ++j)
                    {
                      out (i, j) += a_ip * b (p, j);
                    }
                }
            }
        }
      return out;
    }

  throw shape_error ("matmul requires 1D or 2D tensors");
}

template <typename T>
T
dot (const nd_tensor<T> &a, const nd_tensor<T> &b)
{
  if (a.size () != b.size ())
    {
      throw shape_error ("dot product requires tensors of equal size");
    }
  const std::size_t n = a.size ();
  if (a.is_contiguous () && b.is_contiguous ())
    {
      const T *HYPER_MATH_RESTRICT pa = a.raw_data ();
      const T *HYPER_MATH_RESTRICT pb = b.raw_data ();
      T acc = T {0};
      for (std::size_t i = 0; i < n; ++i)
        {
          acc += pa[i] * pb[i];
        }
      return acc;
    }
  T acc = T {0};
  for (std::size_t i = 0; i < n; ++i)
    {
      acc += a.eval_at (i) * b.eval_at (i);
    }
  return acc;
}

template <typename T>
nd_tensor<T>
outer (const nd_tensor<T> &a, const nd_tensor<T> &b)
{
  const std::size_t m = a.size ();
  const std::size_t n = b.size ();
  nd_tensor<T> out (shape_type {m, n});
  for (std::size_t i = 0; i < m; ++i)
    {
      const T ai = a.eval_at (i);
      for (std::size_t j = 0; j < n; ++j)
        {
          out (i, j) = ai * b.eval_at (j);
        }
    }
  return out;
}

template <typename T>
T
trace (const nd_tensor<T> &m)
{
  if (m.ndim () != 2)
    {
      throw shape_error ("trace requires a 2D tensor");
    }
  const std::size_t d = std::min (m.shape ()[0], m.shape ()[1]);
  T s = T {0};
  for (std::size_t i = 0; i < d; ++i)
    {
      s += m (i, i);
    }
  return s;
}

/* Clean-room Generalized Einstein Summation Engine (einsum):
   Parses subscript strings such as:
     "ij,jk->ik"   (matrix multiplication)
     "ii->i"       (diagonal extraction)
     "ii->"        (matrix trace)
     "ij->ji"      (transpose)
     "i,j->ij"     (outer product)
     "ijk,ikl->ijl" (batched tensor contraction)
   Supports an arbitrary number of input tensors and arbitrary index labels.  */
template <typename T>
nd_tensor<T>
einsum (const std::string &subscripts,
        const std::vector<nd_tensor<T>> &operands)
{
  /* Strip spaces.  */
  std::string spec;
  for (char c : subscripts)
    {
      if (c != ' ')
        {
          spec.push_back (c);
        }
    }

  const std::size_t arrow_pos = spec.find ("->");
  std::string lhs_spec =
    (arrow_pos == std::string::npos) ? spec : spec.substr (0, arrow_pos);
  std::string rhs_spec =
    (arrow_pos == std::string::npos) ? "" : spec.substr (arrow_pos + 2);

  /* Split lhs_spec by comma.  */
  std::vector<std::string> in_labels;
  std::size_t start = 0;
  while (true)
    {
      const std::size_t comma = lhs_spec.find (',', start);
      if (comma == std::string::npos)
        {
          in_labels.push_back (lhs_spec.substr (start));
          break;
        }
      in_labels.push_back (lhs_spec.substr (start, comma - start));
      start = comma + 1;
    }

  if (in_labels.size () != operands.size ())
    {
      throw shape_error ("einsum operand count does not match subscript list");
    }

  /* Discover dimension extent for each index character.  */
  std::map<char, std::size_t> index_dim;
  std::map<char, std::size_t> index_freq;
  std::vector<char> all_indices;

  for (std::size_t op_idx = 0; op_idx < operands.size (); ++op_idx)
    {
      const std::string &lbl = in_labels[op_idx];
      const auto &ten = operands[op_idx];
      if (lbl.size () != ten.ndim ())
        {
          throw shape_error ("einsum subscript rank mismatch for operand "
                             + std::to_string (op_idx));
        }
      for (std::size_t axis = 0; axis < lbl.size (); ++axis)
        {
          const char c = lbl[axis];
          const std::size_t dim = ten.shape ()[axis];
          auto it = index_dim.find (c);
          if (it == index_dim.end ())
            {
              index_dim[c] = dim;
              all_indices.push_back (c);
            }
          else if (it->second != dim)
            {
              throw shape_error (std::string ("einsum dimension mismatch on index '")
                                 + c + "'");
            }
          index_freq[c] += 1;
        }
    }

  /* If implicit mode (no "->"), output indices are those appearing once,
     sorted alphabetically.  */
  if (arrow_pos == std::string::npos)
    {
      for (const auto &kv : index_freq)
        {
          if (kv.second == 1)
            {
              rhs_spec.push_back (kv.first);
            }
        }
    }

  /* Partition all_indices into free (in rhs_spec) and contracted (not in rhs). */
  std::vector<char> contracted_indices;
  for (char c : all_indices)
    {
      if (rhs_spec.find (c) == std::string::npos)
        {
          contracted_indices.push_back (c);
        }
    }

  shape_type out_shape;
  for (std::size_t idx_pos = 0; idx_pos < rhs_spec.size (); ++idx_pos)
    {
      const char c = rhs_spec[idx_pos];
      if (rhs_spec.find (c) < idx_pos)
        {
          throw shape_error (std::string ("einsum duplicate output index '")
                             + c + "'");
        }
      auto it = index_dim.find (c);
      if (it == index_dim.end ())
        {
          throw shape_error (std::string ("einsum output index '") + c
                             + "' not found in inputs");
        }
      out_shape.push_back (it->second);
    }

  const bool scalar_output = out_shape.empty ();
  if (scalar_output)
    {
      out_shape.push_back (1);
    }

  nd_tensor<T> output (out_shape, T {0});

  /* Compute total iteration counts for free and contracted spaces.  */
  std::size_t free_total = 1;
  for (char c : rhs_spec)
    {
      free_total *= index_dim[c];
    }
  std::size_t contracted_total = 1;
  for (char c : contracted_indices)
    {
      contracted_total *= index_dim[c];
    }

  std::map<char, std::size_t> coord_map;
  for (std::size_t f_idx = 0; f_idx < free_total; ++f_idx)
    {
      std::size_t rem_f = f_idx;
      for (std::ptrdiff_t d =
             static_cast<std::ptrdiff_t> (rhs_spec.size ()) - 1;
           d >= 0; --d)
        {
          const char c = rhs_spec[static_cast<std::size_t> (d)];
          const std::size_t ext = index_dim[c];
          coord_map[c] = rem_f % ext;
          rem_f /= ext;
        }

      T sum_val = T {0};
      for (std::size_t c_idx = 0; c_idx < contracted_total; ++c_idx)
        {
          std::size_t rem_c = c_idx;
          for (std::ptrdiff_t d =
                 static_cast<std::ptrdiff_t> (contracted_indices.size ()) - 1;
               d >= 0; --d)
            {
              const char c = contracted_indices[static_cast<std::size_t> (d)];
              const std::size_t ext = index_dim[c];
              coord_map[c] = rem_c % ext;
              rem_c /= ext;
            }

          T prod_val = T {1};
          for (std::size_t op_i = 0; op_i < operands.size (); ++op_i)
            {
              const std::string &lbl = in_labels[op_i];
              std::vector<std::size_t> op_coords (lbl.size ());
              for (std::size_t axis = 0; axis < lbl.size (); ++axis)
                {
                  op_coords[axis] = coord_map[lbl[axis]];
                }
              prod_val *= operands[op_i].at (op_coords);
            }
          sum_val += prod_val;
        }

      output[f_idx] = sum_val;
    }

  return output;
}

/* NaN-safe reductions (NumPy nansum, nanprod, nanmean, nanvar, nanstd,
   nanmin, nanmax, nanpercentile, nanmedian).  */
template <typename Derived, typename T>
double
nansum (const tensor_expr_base<Derived, T> &expr)
{
  double acc = 0.0;
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      const double v = static_cast<double> (expr.eval_at (i));
      if (!std::isnan (v))
        {
          acc += v;
        }
    }
  return acc;
}

template <typename Derived, typename T>
double
nanprod (const tensor_expr_base<Derived, T> &expr)
{
  double acc = 1.0;
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      const double v = static_cast<double> (expr.eval_at (i));
      if (!std::isnan (v))
        {
          acc *= v;
        }
    }
  return acc;
}

template <typename Derived, typename T>
double
nanmean (const tensor_expr_base<Derived, T> &expr)
{
  double acc = 0.0;
  std::size_t cnt = 0;
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      const double v = static_cast<double> (expr.eval_at (i));
      if (!std::isnan (v))
        {
          acc += v;
          ++cnt;
        }
    }
  return (cnt > 0) ? (acc / static_cast<double> (cnt)) : constants::nan;
}

template <typename Derived, typename T>
double
nanvar (const tensor_expr_base<Derived, T> &expr, std::size_t ddof = 0)
{
  const double m = nanmean (expr);
  if (std::isnan (m))
    {
      return constants::nan;
    }
  double sq = 0.0;
  std::size_t cnt = 0;
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      const double v = static_cast<double> (expr.eval_at (i));
      if (!std::isnan (v))
        {
          const double d = v - m;
          sq += d * d;
          ++cnt;
        }
    }
  return (cnt > ddof) ? (sq / static_cast<double> (cnt - ddof))
                      : constants::nan;
}

template <typename Derived, typename T>
double
nanstd (const tensor_expr_base<Derived, T> &expr, std::size_t ddof = 0)
{
  return std::sqrt (nanvar (expr, ddof));
}

template <typename Derived, typename T>
double
nanmin (const tensor_expr_base<Derived, T> &expr)
{
  double best = constants::inf;
  bool found = false;
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      const double v = static_cast<double> (expr.eval_at (i));
      if (!std::isnan (v))
        {
          best = std::min (best, v);
          found = true;
        }
    }
  return found ? best : constants::nan;
}

template <typename Derived, typename T>
double
nanmax (const tensor_expr_base<Derived, T> &expr)
{
  double best = -constants::inf;
  bool found = false;
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      const double v = static_cast<double> (expr.eval_at (i));
      if (!std::isnan (v))
        {
          best = std::max (best, v);
          found = true;
        }
    }
  return found ? best : constants::nan;
}

template <typename Derived, typename T>
double
nanpercentile (const tensor_expr_base<Derived, T> &expr, double q_percent)
{
  std::vector<double> valid;
  valid.reserve (expr.size ());
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      const double v = static_cast<double> (expr.eval_at (i));
      if (!std::isnan (v))
        {
          valid.push_back (v);
        }
    }
  if (valid.empty ())
    {
      return constants::nan;
    }
  return percentile (nd_tensor<double> (shape_type {valid.size ()}, valid),
                     q_percent);
}

template <typename Derived, typename T>
double
nanmedian (const tensor_expr_base<Derived, T> &expr)
{
  return nanpercentile (expr, 50.0);
}

/* NaN-ignoring flat index extrema: nanargmin and nanargmax (NumPy equivalents). */
template <typename Derived, typename T>
std::size_t
nanargmin (const tensor_expr_base<Derived, T> &expr)
{
  std::size_t best_idx = expr.size ();
  double best_val = constants::inf;
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      const double v = static_cast<double> (expr.eval_at (i));
      if (!std::isnan (v) && (v < best_val || best_idx == expr.size ()))
        {
          best_val = v;
          best_idx = i;
        }
    }
  if (best_idx == expr.size ())
    {
      throw hyper_math_error ("nanargmin encountered an all-NaN tensor");
    }
  return best_idx;
}

template <typename Derived, typename T>
std::size_t
nanargmax (const tensor_expr_base<Derived, T> &expr)
{
  std::size_t best_idx = expr.size ();
  double best_val = -constants::inf;
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      const double v = static_cast<double> (expr.eval_at (i));
      if (!std::isnan (v) && (v > best_val || best_idx == expr.size ()))
        {
          best_val = v;
          best_idx = i;
        }
    }
  if (best_idx == expr.size ())
    {
      throw hyper_math_error ("nanargmax encountered an all-NaN tensor");
    }
  return best_idx;
}

/* Peak-to-peak range ptp(a) = max(a) - min(a) and 1-D trapezoidal rule
   trapz(y, dx) / trapz(y, x) (NumPy ptp and trapz equivalents).  */
template <typename Derived, typename T>
T
ptp (const tensor_expr_base<Derived, T> &expr)
{
  return max (expr) - min (expr);
}

template <typename Derived, typename T>
double
trapz (const tensor_expr_base<Derived, T> &y, double dx = 1.0)
{
  const std::size_t n = y.size ();
  if (n < 2)
    {
      return 0.0;
    }
  double sum_v =
    0.5 * (static_cast<double> (y.eval_at (0))
           + static_cast<double> (y.eval_at (n - 1)));
  for (std::size_t i = 1; i + 1 < n; ++i)
    {
      sum_v += static_cast<double> (y.eval_at (i));
    }
  return sum_v * dx;
}

template <typename YDerived, typename YVal, typename XDerived, typename XVal>
double
trapz (const tensor_expr_base<YDerived, YVal> &y,
       const tensor_expr_base<XDerived, XVal> &x)
{
  const std::size_t n = y.size ();
  if (x.size () != n)
    {
      throw shape_error ("trapz requires y and x to have identical length");
    }
  if (n < 2)
    {
      return 0.0;
    }
  double area = 0.0;
  for (std::size_t i = 0; i + 1 < n; ++i)
    {
      const double dx =
        static_cast<double> (x.eval_at (i + 1))
        - static_cast<double> (x.eval_at (i));
      const double avg_y =
        0.5 * (static_cast<double> (y.eval_at (i))
               + static_cast<double> (y.eval_at (i + 1)));
      area += dx * avg_y;
    }
  return area;
}

/* Kronecker product of two 1D or 2D tensors (NumPy kron).  */
template <typename T>
nd_tensor<T>
kron (const nd_tensor<T> &a, const nd_tensor<T> &b)
{
  if (a.ndim () == 1 && b.ndim () == 1)
    {
      const std::size_t m = a.size ();
      const std::size_t n = b.size ();
      nd_tensor<T> out (shape_type {m * n});
      for (std::size_t i = 0; i < m; ++i)
        {
          for (std::size_t j = 0; j < n; ++j)
            {
              out[i * n + j] = a.eval_at (i) * b.eval_at (j);
            }
        }
      return out;
    }

  const nd_tensor<T> ma =
    (a.ndim () == 1) ? a.reshape ({1, a.size ()}) : a;
  const nd_tensor<T> mb =
    (b.ndim () == 1) ? b.reshape ({1, b.size ()}) : b;
  if (ma.ndim () != 2 || mb.ndim () != 2)
    {
      throw shape_error ("kron requires 1D or 2D tensors");
    }

  const std::size_t ra = ma.shape ()[0];
  const std::size_t ca = ma.shape ()[1];
  const std::size_t rb = mb.shape ()[0];
  const std::size_t cb = mb.shape ()[1];
  nd_tensor<T> out (shape_type {ra * rb, ca * cb});

  for (std::size_t ia = 0; ia < ra; ++ia)
    {
      for (std::size_t ja = 0; ja < ca; ++ja)
        {
          const T v = ma (ia, ja);
          for (std::size_t ib = 0; ib < rb; ++ib)
            {
              for (std::size_t jb = 0; jb < cb; ++jb)
                {
                  out (ia * rb + ib, ja * cb + jb) = v * mb (ib, jb);
                }
            }
        }
    }
  return out;
}

/* General multi-axis tensor dot contraction (NumPy tensordot).
   Contracts the last `n_axes` of `a` with the first `n_axes` of `b`.  */
template <typename T>
nd_tensor<T>
tensordot (const nd_tensor<T> &a,
           const nd_tensor<T> &b,
           std::size_t n_axes = 2)
{
  if (n_axes > a.ndim () || n_axes > b.ndim ())
    {
      throw shape_error ("tensordot n_axes exceeds operand rank");
    }

  std::size_t contract_len = 1;
  for (std::size_t k = 0; k < n_axes; ++k)
    {
      const std::size_t da = a.shape ()[a.ndim () - n_axes + k];
      const std::size_t db = b.shape ()[k];
      if (da != db)
        {
          throw shape_error ("tensordot contracted axis dimension mismatch");
        }
      contract_len *= da;
    }

  shape_type out_shape;
  std::size_t free_a = 1;
  for (std::size_t d = 0; d < a.ndim () - n_axes; ++d)
    {
      out_shape.push_back (a.shape ()[d]);
      free_a *= a.shape ()[d];
    }
  std::size_t free_b = 1;
  for (std::size_t d = n_axes; d < b.ndim (); ++d)
    {
      out_shape.push_back (b.shape ()[d]);
      free_b *= b.shape ()[d];
    }
  if (out_shape.empty ())
    {
      out_shape.push_back (1);
    }

  const nd_tensor<T> mat_a = a.reshape ({free_a, contract_len});
  const nd_tensor<T> mat_b = b.reshape ({contract_len, free_b});
  return matmul (mat_a, mat_b).reshape (std::move (out_shape));
}

/* Weighted arithmetic mean `average(a, weights)` and 3-D vector cross product
   `cross_3d(a, b)` (NumPy average and cross equivalents).  */
template <typename ADerived, typename AVal, typename WDerived, typename WVal>
double
average (const tensor_expr_base<ADerived, AVal> &a,
         const tensor_expr_base<WDerived, WVal> &weights)
{
  if (a.size () == 0 || a.size () != weights.size ())
    {
      throw shape_error ("average requires non-empty matching a and weights");
    }
  double w_sum = 0.0;
  double aw_sum = 0.0;
  for (std::size_t i = 0; i < a.size (); ++i)
    {
      const double w = static_cast<double> (weights.eval_at (i));
      w_sum += w;
      aw_sum += static_cast<double> (a.eval_at (i)) * w;
    }
  if (std::abs (w_sum) < 1e-15)
    {
      throw hyper_math_error ("average weights sum to zero");
    }
  return aw_sum / w_sum;
}

template <typename T>
nd_tensor<T>
cross_3d (const nd_tensor<T> &a, const nd_tensor<T> &b)
{
  if (a.size () != 3 || b.size () != 3)
    {
      throw shape_error ("cross_3d requires 3-element vectors");
    }
  const T a0 = a.eval_at (0);
  const T a1 = a.eval_at (1);
  const T a2 = a.eval_at (2);
  const T b0 = b.eval_at (0);
  const T b1 = b.eval_at (1);
  const T b2 = b.eval_at (2);
  return nd_tensor<T> {
    a1 * b2 - a2 * b1,
    a2 * b0 - a0 * b2,
    a0 * b1 - a1 * b0
  };
}

/* NaN-safe cumulative reductions `nancumsum`, `nancumprod`, and peak-to-peak
   range `nanptp` (NumPy nancumsum, nancumprod, and nanmax - nanmin).  */
template <typename Derived, typename Val>
nd_tensor<double>
nancumsum (const tensor_expr_base<Derived, Val> &expr)
{
  nd_tensor<double> out (shape_type {expr.size ()});
  double acc = 0.0;
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      const double v = static_cast<double> (expr.eval_at (i));
      if (!std::isnan (v))
        {
          acc += v;
        }
      out[i] = acc;
    }
  return out;
}

template <typename Derived, typename Val>
nd_tensor<double>
nancumprod (const tensor_expr_base<Derived, Val> &expr)
{
  nd_tensor<double> out (shape_type {expr.size ()});
  double acc = 1.0;
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      const double v = static_cast<double> (expr.eval_at (i));
      if (!std::isnan (v))
        {
          acc *= v;
        }
      out[i] = acc;
    }
  return out;
}

template <typename Derived, typename Val>
double
nanptp (const tensor_expr_base<Derived, Val> &expr)
{
  return static_cast<double> (nanmax (expr))
         - static_cast<double> (nanmin (expr));
}

template <typename Derived, typename T>
double
quantile (const tensor_expr_base<Derived, T> &expr, double q)
{
  if (q < 0.0 || q > 1.0)
    {
      throw hyper_math_error ("quantile requires q in [0, 1]");
    }
  return percentile (expr, q * 100.0);
}

template <typename Derived, typename T>
double
nanquantile (const tensor_expr_base<Derived, T> &expr, double q)
{
  if (q < 0.0 || q > 1.0)
    {
      throw hyper_math_error ("nanquantile requires q in [0, 1]");
    }
  return nanpercentile (expr, q * 100.0);
}

/* Axis-specific quantile and nanquantile reductions for 2D tensors.  */
template <typename T>
nd_tensor<double>
quantile (const nd_tensor<T> &a, double q, std::size_t axis)
{
  if (a.ndim () != 2 || axis > 1)
    {
      throw shape_error ("axis quantile requires 2D tensor and axis in {0, 1}");
    }
  const std::size_t m = a.shape ()[0];
  const std::size_t n = a.shape ()[1];
  const std::size_t out_len = (axis == 0) ? n : m;
  const std::size_t slice_len = (axis == 0) ? m : n;
  nd_tensor<double> out (shape_type {out_len});
  for (std::size_t k = 0; k < out_len; ++k)
    {
      nd_tensor<double> line (shape_type {slice_len});
      for (std::size_t s = 0; s < slice_len; ++s)
        {
          line[s] = static_cast<double> ((axis == 0) ? a (s, k) : a (k, s));
        }
      out[k] = quantile (line, q);
    }
  return out;
}

template <typename T>
nd_tensor<double>
nanquantile (const nd_tensor<T> &a, double q, std::size_t axis)
{
  if (a.ndim () != 2 || axis > 1)
    {
      throw shape_error (
        "axis nanquantile requires 2D tensor and axis in {0, 1}");
    }
  const std::size_t m = a.shape ()[0];
  const std::size_t n = a.shape ()[1];
  const std::size_t out_len = (axis == 0) ? n : m;
  const std::size_t slice_len = (axis == 0) ? m : n;
  nd_tensor<double> out (shape_type {out_len});
  for (std::size_t k = 0; k < out_len; ++k)
    {
      nd_tensor<double> line (shape_type {slice_len});
      for (std::size_t s = 0; s < slice_len; ++s)
        {
          line[s] = static_cast<double> ((axis == 0) ? a (s, k) : a (k, s));
        }
      out[k] = nanquantile (line, q);
    }
  return out;
}

/* Optimal matrix-chain multiplication `multi_dot` via O(k^3) dynamic
   programming parenthesization (NumPy linalg.multi_dot parity).  */
template <typename T>
nd_tensor<T>
multi_dot (const std::vector<nd_tensor<T>> &arrays)
{
  const std::size_t k = arrays.size ();
  if (k < 2)
    {
      throw shape_error ("multi_dot requires at least 2 matrices");
    }
  for (std::size_t i = 0; i < k; ++i)
    {
      if (arrays[i].ndim () != 2)
        {
          throw shape_error ("multi_dot requires 2D matrices");
        }
      if (i > 0 && arrays[i - 1].shape ()[1] != arrays[i].shape ()[0])
        {
          throw shape_error ("incompatible inner dimensions in multi_dot");
        }
    }
  std::vector<std::size_t> p (k + 1);
  for (std::size_t i = 0; i < k; ++i)
    {
      p[i] = arrays[i].shape ()[0];
    }
  p[k] = arrays[k - 1].shape ()[1];

  std::vector<std::vector<double>> cost (k, std::vector<double> (k, 0.0));
  std::vector<std::vector<std::size_t>> split (k, std::vector<std::size_t> (k, 0));

  for (std::size_t len = 2; len <= k; ++len)
    {
      for (std::size_t i = 0; i <= k - len; ++i)
        {
          const std::size_t j = i + len - 1;
          cost[i][j] = constants::inf;
          for (std::size_t s = i; s < j; ++s)
            {
              const double c =
                cost[i][s] + cost[s + 1][j]
                + static_cast<double> (p[i]) * static_cast<double> (p[s + 1])
                    * static_cast<double> (p[j + 1]);
              if (c < cost[i][j])
                {
                  cost[i][j] = c;
                  split[i][j] = s;
                }
            }
        }
    }

  auto eval_chain = [&] (auto &self, std::size_t i, std::size_t j) -> nd_tensor<T> {
    if (i == j)
      {
        return arrays[i];
      }
    const std::size_t s = split[i][j];
    return matmul (self (self, i, s), self (self, s + 1, j));
  };

  return eval_chain (eval_chain, 0, k - 1);
}

} // namespace hyper_math::tensor

#endif /* HYPER_MATH_TENSOR_REDUCTIONS_HPP */
