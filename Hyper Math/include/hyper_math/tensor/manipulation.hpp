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

#ifndef HYPER_MATH_TENSOR_MANIPULATION_HPP
#define HYPER_MATH_TENSOR_MANIPULATION_HPP

#include "hyper_math/scientific/linalg.hpp"
#include "hyper_math/tensor/reductions.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hyper_math::tensor
{

/* Generate 2-D coordinate matrices from 1-D coordinate vectors
   (NumPy meshgrid equivalent, supporting "xy" Cartesian and "ij" indexing). */
template <typename T>
std::pair<nd_tensor<T>, nd_tensor<T>>
meshgrid (const nd_tensor<T> &x,
          const nd_tensor<T> &y,
          const std::string &indexing = "xy")
{
  const std::size_t nx = x.size ();
  const std::size_t ny = y.size ();
  if (indexing == "xy")
    {
      nd_tensor<T> xx (shape_type {ny, nx});
      nd_tensor<T> yy (shape_type {ny, nx});
      for (std::size_t r = 0; r < ny; ++r)
        {
          for (std::size_t c = 0; c < nx; ++c)
            {
              xx (r, c) = x.eval_at (c);
              yy (r, c) = y.eval_at (r);
            }
        }
      return {std::move (xx), std::move (yy)};
    }
  else if (indexing == "ij")
    {
      nd_tensor<T> xx (shape_type {nx, ny});
      nd_tensor<T> yy (shape_type {nx, ny});
      for (std::size_t r = 0; r < nx; ++r)
        {
          for (std::size_t c = 0; c < ny; ++c)
            {
              xx (r, c) = x.eval_at (r);
              yy (r, c) = y.eval_at (c);
            }
        }
      return {std::move (xx), std::move (yy)};
    }
  throw hyper_math_error ("meshgrid indexing must be 'xy' or 'ij'");
}

/* Repeat elements of a tensor `repeats` times (NumPy repeat).  */
template <typename T>
nd_tensor<T>
repeat (const nd_tensor<T> &a, std::size_t repeats)
{
  nd_tensor<T> out (shape_type {a.size () * repeats});
  std::size_t idx = 0;
  for (std::size_t i = 0; i < a.size (); ++i)
    {
      const T val = a.eval_at (i);
      for (std::size_t r = 0; r < repeats; ++r)
        {
          out[idx++] = val;
        }
    }
  return out;
}

/* Construct a tensor by repeating `a` the number of times given by `reps`
   (NumPy tile for 1D and 2D).  */
template <typename T>
nd_tensor<T>
tile (const nd_tensor<T> &a, const std::vector<std::size_t> &reps)
{
  if (reps.empty ())
    {
      return a.copy ();
    }
  if (a.ndim () == 1 && reps.size () == 1)
    {
      const std::size_t n = a.size ();
      const std::size_t k = reps[0];
      nd_tensor<T> out (shape_type {n * k});
      for (std::size_t r = 0; r < k; ++r)
        {
          for (std::size_t i = 0; i < n; ++i)
            {
              out[r * n + i] = a.eval_at (i);
            }
        }
      return out;
    }

  const nd_tensor<T> mat =
    (a.ndim () == 1) ? a.reshape ({1, a.size ()}) : a;
  if (mat.ndim () != 2)
    {
      throw shape_error ("tile currently supports 1D and 2D tensors");
    }
  const std::size_t r_rep = reps.size () >= 2 ? reps[0] : 1;
  const std::size_t c_rep = reps.size () >= 2 ? reps[1] : reps[0];
  const std::size_t rows = mat.shape ()[0];
  const std::size_t cols = mat.shape ()[1];

  nd_tensor<T> out (shape_type {rows * r_rep, cols * c_rep});
  for (std::size_t br = 0; br < r_rep; ++br)
    {
      for (std::size_t bc = 0; bc < c_rep; ++bc)
        {
          for (std::size_t r = 0; r < rows; ++r)
            {
              for (std::size_t c = 0; c < cols; ++c)
                {
                  out (br * rows + r, bc * cols + c) = mat (r, c);
                }
            }
        }
    }
  return out;
}

/* Pad a 1D tensor with boundary modes: "constant", "edge", "reflect", "wrap". */
template <typename T>
nd_tensor<T>
pad (const nd_tensor<T> &a,
     std::size_t pad_before,
     std::size_t pad_after,
     const std::string &mode = "constant",
     T constant_val = T {0})
{
  const std::size_t n = a.size ();
  if (n == 0)
    {
      return nd_tensor<T> (shape_type {pad_before + pad_after}, constant_val);
    }
  const std::size_t total = pad_before + n + pad_after;
  nd_tensor<T> out (shape_type {total}, constant_val);

  for (std::size_t i = 0; i < n; ++i)
    {
      out[pad_before + i] = a.eval_at (i);
    }

  if (mode == "constant")
    {
      return out;
    }
  else if (mode == "edge")
    {
      for (std::size_t i = 0; i < pad_before; ++i)
        {
          out[i] = a.eval_at (0);
        }
      for (std::size_t i = 0; i < pad_after; ++i)
        {
          out[pad_before + n + i] = a.eval_at (n - 1);
        }
    }
  else if (mode == "wrap")
    {
      for (std::size_t i = 0; i < pad_before; ++i)
        {
          const std::size_t src = (n - ((pad_before - i) % n)) % n;
          out[i] = a.eval_at (src);
        }
      for (std::size_t i = 0; i < pad_after; ++i)
        {
          out[pad_before + n + i] = a.eval_at (i % n);
        }
    }
  else if (mode == "reflect")
    {
      const std::size_t period = (n > 1) ? (2 * (n - 1)) : 1;
      for (std::size_t i = 0; i < pad_before; ++i)
        {
          const std::size_t dist = pad_before - i;
          const std::size_t rem = (n > 1) ? (dist % period) : 0;
          const std::size_t src =
            (rem < n) ? rem : (period - rem);
          out[i] = a.eval_at (src);
        }
      for (std::size_t i = 0; i < pad_after; ++i)
        {
          const std::size_t dist = i + 1;
          const std::size_t pos = (n > 1) ? ((n - 1 + dist) % period) : 0;
          const std::size_t src =
            (pos < n) ? pos : (period - pos);
          out[pad_before + n + i] = a.eval_at (src);
        }
    }
  else if (mode == "symmetric")
    {
      const std::size_t period = 2 * n;
      for (std::size_t i = 0; i < pad_before; ++i)
        {
          const std::size_t dist = pad_before - 1 - i;
          const std::size_t rem = dist % period;
          const std::size_t src = (rem < n) ? rem : (period - 1 - rem);
          out[i] = a.eval_at (src);
        }
      for (std::size_t i = 0; i < pad_after; ++i)
        {
          const std::size_t rem = i % period;
          const std::size_t src =
            (rem < n) ? (n - 1 - rem) : (rem - n);
          out[pad_before + n + i] = a.eval_at (src);
        }
    }
  else if (mode == "linear_ramp")
    {
      const double start_v = static_cast<double> (a.eval_at (0));
      const double end_v = static_cast<double> (a.eval_at (n - 1));
      const double target_v = static_cast<double> (constant_val);
      for (std::size_t i = 0; i < pad_before; ++i)
        {
          const double t =
            static_cast<double> (i) / static_cast<double> (pad_before);
          out[i] = static_cast<T> (target_v + t * (start_v - target_v));
        }
      for (std::size_t i = 0; i < pad_after; ++i)
        {
          const double t =
            static_cast<double> (i + 1) / static_cast<double> (pad_after);
          out[pad_before + n + i] =
            static_cast<T> (end_v + t * (target_v - end_v));
        }
    }
  else if (mode == "mean" || mode == "median"
           || mode == "minimum" || mode == "maximum")
    {
      T fill_v {};
      if (mode == "mean")
        {
          fill_v = static_cast<T> (mean (a));
        }
      else if (mode == "median")
        {
          fill_v = static_cast<T> (median (a));
        }
      else if (mode == "minimum")
        {
          fill_v = min (a);
        }
      else
        {
          fill_v = max (a);
        }
      for (std::size_t i = 0; i < pad_before; ++i)
        {
          out[i] = fill_v;
        }
      for (std::size_t i = 0; i < pad_after; ++i)
        {
          out[pad_before + n + i] = fill_v;
        }
    }
  else
    {
      throw hyper_math_error ("unknown pad mode: " + mode);
    }
  return out;
}

/* Reverse order of elements along flattened or 2D axes: flip, fliplr, flipud,
   roll, rot90.  */
template <typename T>
nd_tensor<T>
flip (const nd_tensor<T> &a)
{
  nd_tensor<T> out (a.shape ());
  const std::size_t n = a.size ();
  for (std::size_t i = 0; i < n; ++i)
    {
      out[i] = a.eval_at (n - 1 - i);
    }
  return out;
}

template <typename T>
nd_tensor<T>
fliplr (const nd_tensor<T> &m)
{
  if (m.ndim () != 2)
    {
      throw shape_error ("fliplr requires a 2D tensor");
    }
  const std::size_t r = m.shape ()[0];
  const std::size_t c = m.shape ()[1];
  nd_tensor<T> out (shape_type {r, c});
  for (std::size_t i = 0; i < r; ++i)
    {
      for (std::size_t j = 0; j < c; ++j)
        {
          out (i, j) = m (i, c - 1 - j);
        }
    }
  return out;
}

template <typename T>
nd_tensor<T>
flipud (const nd_tensor<T> &m)
{
  if (m.ndim () != 2)
    {
      throw shape_error ("flipud requires a 2D tensor");
    }
  const std::size_t r = m.shape ()[0];
  const std::size_t c = m.shape ()[1];
  nd_tensor<T> out (shape_type {r, c});
  for (std::size_t i = 0; i < r; ++i)
    {
      for (std::size_t j = 0; j < c; ++j)
        {
          out (i, j) = m (r - 1 - i, j);
        }
    }
  return out;
}

template <typename T>
nd_tensor<T>
roll (const nd_tensor<T> &a, std::ptrdiff_t shift)
{
  const std::size_t n = a.size ();
  if (n == 0)
    {
      return a.copy ();
    }
  nd_tensor<T> out (a.shape ());
  const std::ptrdiff_t sn = static_cast<std::ptrdiff_t> (n);
  const std::ptrdiff_t norm_shift = ((shift % sn) + sn) % sn;
  for (std::size_t i = 0; i < n; ++i)
    {
      const std::size_t dst =
        (i + static_cast<std::size_t> (norm_shift)) % n;
      out[dst] = a.eval_at (i);
    }
  return out;
}

template <typename T>
nd_tensor<T>
rot90 (const nd_tensor<T> &m, int k = 1)
{
  if (m.ndim () != 2)
    {
      throw shape_error ("rot90 requires a 2D tensor");
    }
  int turns = ((k % 4) + 4) % 4;
  nd_tensor<T> cur = m.copy ();
  for (int t = 0; t < turns; ++t)
    {
      cur = flipud (cur.t ());
    }
  return cur;
}

/* Lower and upper triangle extraction (tril, triu) & Vandermonde matrix.  */
template <typename T>
nd_tensor<T>
tril (const nd_tensor<T> &m, std::ptrdiff_t k = 0)
{
  if (m.ndim () != 2)
    {
      throw shape_error ("tril requires a 2D tensor");
    }
  const std::size_t rows = m.shape ()[0];
  const std::size_t cols = m.shape ()[1];
  nd_tensor<T> out (shape_type {rows, cols}, T {0});
  for (std::size_t r = 0; r < rows; ++r)
    {
      for (std::size_t c = 0; c < cols; ++c)
        {
          if (static_cast<std::ptrdiff_t> (c)
              <= static_cast<std::ptrdiff_t> (r) + k)
            {
              out (r, c) = m (r, c);
            }
        }
    }
  return out;
}

template <typename T>
nd_tensor<T>
triu (const nd_tensor<T> &m, std::ptrdiff_t k = 0)
{
  if (m.ndim () != 2)
    {
      throw shape_error ("triu requires a 2D tensor");
    }
  const std::size_t rows = m.shape ()[0];
  const std::size_t cols = m.shape ()[1];
  nd_tensor<T> out (shape_type {rows, cols}, T {0});
  for (std::size_t r = 0; r < rows; ++r)
    {
      for (std::size_t c = 0; c < cols; ++c)
        {
          if (static_cast<std::ptrdiff_t> (c)
              >= static_cast<std::ptrdiff_t> (r) + k)
            {
              out (r, c) = m (r, c);
            }
        }
    }
  return out;
}

template <typename T = double>
nd_tensor<T>
vander (const nd_tensor<T> &x, std::size_t n_cols = 0, bool increasing = false)
{
  const std::size_t m = x.size ();
  if (n_cols == 0)
    {
      n_cols = m;
    }
  nd_tensor<T> out (shape_type {m, n_cols});
  for (std::size_t i = 0; i < m; ++i)
    {
      const T xi = x.eval_at (i);
      for (std::size_t j = 0; j < n_cols; ++j)
        {
          const std::size_t p = increasing ? j : (n_cols - 1 - j);
          out (i, j) = static_cast<T> (
            std::pow (static_cast<double> (xi), static_cast<double> (p)));
        }
    }
  return out;
}

/* Nonzero flat indices, searchsorted, bincount, histogram, and digitize.  */
template <typename Derived, typename T>
nd_tensor<std::size_t>
nonzero (const tensor_expr_base<Derived, T> &expr)
{
  std::vector<std::size_t> idx;
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      if (expr.eval_at (i) != T {0})
        {
          idx.push_back (i);
        }
    }
  return nd_tensor<std::size_t> (shape_type {idx.size ()}, idx);
}

template <typename T>
nd_tensor<std::size_t>
searchsorted (const nd_tensor<T> &sorted_arr,
              const nd_tensor<T> &values,
              const std::string &side = "left")
{
  const std::vector<T> s = sorted_arr.to_vector ();
  nd_tensor<std::size_t> out (values.shape ());
  for (std::size_t i = 0; i < values.size (); ++i)
    {
      const T v = values.eval_at (i);
      auto it = (side == "right")
                ? std::upper_bound (s.begin (), s.end (), v)
                : std::lower_bound (s.begin (), s.end (), v);
      out[i] = static_cast<std::size_t> (std::distance (s.begin (), it));
    }
  return out;
}

inline nd_tensor<std::size_t>
bincount (const nd_tensor<std::int64_t> &x, std::size_t minlength = 0)
{
  std::int64_t max_val = -1;
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      const std::int64_t v = x.eval_at (i);
      if (v < 0)
        {
          throw hyper_math_error ("bincount requires non-negative integers");
        }
      max_val = std::max (max_val, v);
    }
  const std::size_t n_bins =
    std::max (minlength, static_cast<std::size_t> (max_val + 1));
  nd_tensor<std::size_t> counts (shape_type {n_bins}, 0);
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      counts[static_cast<std::size_t> (x.eval_at (i))] += 1;
    }
  return counts;
}

struct histogram_result
{
  nd_tensor<std::size_t> counts;
  nd_tensor<double> bin_edges;
};

inline histogram_result
histogram (const nd_tensor<double> &a,
           std::size_t bins = 10,
           std::optional<std::pair<double, double>> range = std::nullopt)
{
  if (bins == 0)
    {
      throw hyper_math_error ("histogram bins must be >= 1");
    }
  const double lo = range.has_value () ? range->first : min (a);
  const double hi = range.has_value () ? range->second : max (a);
  nd_tensor<double> edges = linspace<double> (lo, hi, bins + 1, true);
  nd_tensor<std::size_t> counts (shape_type {bins}, 0);

  const double span = (hi > lo) ? (hi - lo) : 1.0;
  for (std::size_t i = 0; i < a.size (); ++i)
    {
      const double v = a.eval_at (i);
      if (v < lo || v > hi)
        {
          continue;
        }
      std::size_t idx = static_cast<std::size_t> (
        std::floor (((v - lo) / span) * static_cast<double> (bins)));
      if (idx >= bins)
        {
          idx = bins - 1;
        }
      counts[idx] += 1;
    }
  return histogram_result {std::move (counts), std::move (edges)};
}

/* Numerical gradient via 2nd-order central differences in the interior and
   1st-order differences at the boundaries (NumPy gradient).  */
inline nd_tensor<double>
gradient (const nd_tensor<double> &f, double dx = 1.0)
{
  const std::size_t n = f.size ();
  if (n < 2)
    {
      throw shape_error ("gradient requires tensor of size >= 2");
    }
  nd_tensor<double> g (f.shape ());
  g[0] = (f.eval_at (1) - f.eval_at (0)) / dx;
  for (std::size_t i = 1; i + 1 < n; ++i)
    {
      g[i] = (f.eval_at (i + 1) - f.eval_at (i - 1)) / (2.0 * dx);
    }
  g[n - 1] = (f.eval_at (n - 1) - f.eval_at (n - 2)) / dx;
  return g;
}

/* Covariance and Pearson Correlation Matrix estimators (NumPy cov, corrcoef).
   For 2D input X of shape (n_vars, n_obs), each row is a variable and each
   column is an observation.  */
inline nd_tensor<double>
cov (const nd_tensor<double> &m, std::size_t ddof = 1)
{
  const nd_tensor<double> mat =
    (m.ndim () == 1) ? m.reshape ({1, m.size ()}) : m;
  if (mat.ndim () != 2)
    {
      throw shape_error ("cov requires a 1D or 2D tensor");
    }
  const std::size_t n_vars = mat.shape ()[0];
  const std::size_t n_obs = mat.shape ()[1];
  if (n_obs <= ddof)
    {
      throw shape_error ("number of observations must exceed ddof in cov");
    }

  nd_tensor<double> centered (shape_type {n_vars, n_obs});
  for (std::size_t v = 0; v < n_vars; ++v)
    {
      double row_sum = 0.0;
      for (std::size_t j = 0; j < n_obs; ++j)
        {
          row_sum += mat (v, j);
        }
      const double row_mean = row_sum / static_cast<double> (n_obs);
      for (std::size_t j = 0; j < n_obs; ++j)
        {
          centered (v, j) = mat (v, j) - row_mean;
        }
    }

  return matmul (centered, centered.t ())
         / static_cast<double> (n_obs - ddof);
}

inline nd_tensor<double>
corrcoef (const nd_tensor<double> &m)
{
  nd_tensor<double> c = cov (m, 1);
  const std::size_t n = c.shape ()[0];
  nd_tensor<double> r (shape_type {n, n});
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          const double denom = std::sqrt (c (i, i) * c (j, j));
          r (i, j) = (denom > 0.0) ? (c (i, j) / denom) : 0.0;
        }
    }
  return r;
}

/* Polynomial least-squares fitting and Horner evaluation (NumPy polyfit,
   polyval equivalents). Coeffs are ordered highest power first:
   p(x) = coeffs[0]*x^deg + coeffs[1]*x^(deg-1) + ... + coeffs[deg].  */
inline nd_tensor<double>
polyfit (const nd_tensor<double> &x,
         const nd_tensor<double> &y,
         std::size_t deg)
{
  const nd_tensor<double> v = vander (x, deg + 1, false);
  return scientific::linalg::lstsq (v, y);
}

inline double
polyval (const nd_tensor<double> &coeffs, double x) noexcept
{
  double acc = 0.0;
  for (std::size_t i = 0; i < coeffs.size (); ++i)
    {
      acc = acc * x + coeffs.eval_at (i);
    }
  return acc;
}

inline nd_tensor<double>
polyval (const nd_tensor<double> &coeffs, const nd_tensor<double> &x)
{
  nd_tensor<double> out (x.shape ());
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      out[i] = polyval (coeffs, x.eval_at (i));
    }
  return out;
}

/* Join a sequence of tensors along a new axis (NumPy stack, column_stack).  */
template <typename T>
nd_tensor<T>
stack (const std::vector<nd_tensor<T>> &tensors, std::size_t axis = 0)
{
  if (tensors.empty ())
    {
      throw shape_error ("stack requires at least one tensor");
    }
  std::vector<nd_tensor<T>> expanded;
  expanded.reserve (tensors.size ());
  for (const auto &t : tensors)
    {
      expanded.push_back (t.expand_dims (axis));
    }
  return concatenate (expanded, axis);
}

template <typename T>
nd_tensor<T>
column_stack (const std::vector<nd_tensor<T>> &tensors)
{
  if (tensors.empty ())
    {
      throw shape_error ("column_stack requires at least one tensor");
    }
  std::vector<nd_tensor<T>> cols;
  cols.reserve (tensors.size ());
  for (const auto &t : tensors)
    {
      cols.push_back ((t.ndim () == 1) ? t.reshape ({t.size (), 1}) : t);
    }
  return concatenate (cols, 1);
}

/* Split a tensor into `sections` sub-tensors along `axis` (NumPy split,
   array_split, hsplit, vsplit).  */
template <typename T>
std::vector<nd_tensor<T>>
array_split (const nd_tensor<T> &a,
             std::size_t sections,
             std::size_t axis = 0)
{
  if (sections == 0)
    {
      throw shape_error ("array_split sections must be >= 1");
    }
  if (axis >= a.ndim ())
    {
      throw index_error ("array_split axis out of bounds");
    }
  const std::size_t total = a.shape ()[axis];
  const std::size_t base_len = total / sections;
  const std::size_t rem = total % sections;

  std::vector<nd_tensor<T>> parts;
  parts.reserve (sections);
  std::size_t cur = 0;
  for (std::size_t s = 0; s < sections; ++s)
    {
      const std::size_t len = base_len + (s < rem ? 1 : 0);
      std::vector<slice_range> ranges (a.ndim (), slice_range::all ());
      ranges[axis] =
        slice_range (static_cast<std::ptrdiff_t> (cur),
                     static_cast<std::ptrdiff_t> (cur + len),
                     1);
      parts.push_back (a.slice (ranges).copy ());
      cur += len;
    }
  return parts;
}

template <typename T>
std::vector<nd_tensor<T>>
split (const nd_tensor<T> &a, std::size_t sections, std::size_t axis = 0)
{
  if (sections == 0 || axis >= a.ndim () || (a.shape ()[axis] % sections) != 0)
    {
      throw shape_error ("split does not result in equal division");
    }
  return array_split (a, sections, axis);
}

template <typename T>
std::vector<nd_tensor<T>>
vsplit (const nd_tensor<T> &a, std::size_t sections)
{
  return split (a, sections, 0);
}

template <typename T>
std::vector<nd_tensor<T>>
hsplit (const nd_tensor<T> &a, std::size_t sections)
{
  return split (a, sections, a.ndim () > 1 ? 1 : 0);
}

/* Indexed mutation and bin digitization: put, putmask, digitize.  */
template <typename T>
void
put (nd_tensor<T> &a,
     const std::vector<std::size_t> &indices,
     const std::vector<T> &values)
{
  if (values.empty ())
    {
      return;
    }
  for (std::size_t i = 0; i < indices.size (); ++i)
    {
      if (indices[i] >= a.size ())
        {
          throw index_error ("put index out of bounds");
        }
      a[indices[i]] = values[i % values.size ()];
    }
}

template <typename T, typename MaskT>
void
putmask (nd_tensor<T> &a,
         const nd_tensor<MaskT> &mask,
         const nd_tensor<T> &values)
{
  if (mask.size () != a.size () || values.size () == 0)
    {
      throw shape_error ("putmask shape mismatch");
    }
  std::size_t v_idx = 0;
  for (std::size_t i = 0; i < a.size (); ++i)
    {
      if (static_cast<bool> (mask.eval_at (i)))
        {
          a[i] = values.eval_at (v_idx % values.size ());
        }
      ++v_idx;
    }
}

inline nd_tensor<std::size_t>
digitize (const nd_tensor<double> &x,
          const nd_tensor<double> &bins,
          bool right = false)
{
  return searchsorted (bins, x, right ? "left" : "right");
}

/* Fancy indexing & boolean mask extraction (NumPy take, compress, extract). */
template <typename T>
nd_tensor<T>
take (const nd_tensor<T> &a, const nd_tensor<std::size_t> &indices)
{
  nd_tensor<T> out (indices.shape ());
  for (std::size_t i = 0; i < indices.size (); ++i)
    {
      const std::size_t idx = indices.eval_at (i);
      if (idx >= a.size ())
        {
          throw index_error ("take index out of bounds");
        }
      out[i] = a.eval_at (idx);
    }
  return out;
}

template <typename T, typename MaskT>
nd_tensor<T>
compress (const nd_tensor<MaskT> &condition, const nd_tensor<T> &a)
{
  const std::size_t limit = std::min (condition.size (), a.size ());
  std::vector<T> selected;
  for (std::size_t i = 0; i < limit; ++i)
    {
      if (static_cast<bool> (condition.eval_at (i)))
        {
          selected.push_back (a.eval_at (i));
        }
    }
  return nd_tensor<T> (shape_type {selected.size ()}, selected);
}

template <typename T, typename MaskT>
nd_tensor<T>
extract (const nd_tensor<MaskT> &condition, const nd_tensor<T> &a)
{
  return compress (condition, a);
}

/* Replace NaN, +Inf, and -Inf with finite numbers (NumPy nan_to_num).  */
inline nd_tensor<double>
nan_to_num (const nd_tensor<double> &a,
            double nan_val = 0.0,
            double posinf_val = 1e308,
            double neginf_val = -1e308)
{
  nd_tensor<double> out (a.shape ());
  for (std::size_t i = 0; i < a.size (); ++i)
    {
      const double v = a.eval_at (i);
      if (std::isnan (v))
        {
          out[i] = nan_val;
        }
      else if (std::isinf (v))
        {
          out[i] = (v > 0.0) ? posinf_val : neginf_val;
        }
      else
        {
          out[i] = v;
        }
    }
  return out;
}

/* Phase unwrapping by changing jumps > discont to their period complement
   (NumPy unwrap equivalent).  */
inline nd_tensor<double>
unwrap (const nd_tensor<double> &p,
        double discont = constants::pi,
        double period = constants::two_pi)
{
  const std::size_t n = p.size ();
  if (n == 0)
    {
      return p.copy ();
    }
  nd_tensor<double> out (p.shape ());
  out[0] = p.eval_at (0);
  double cum_corr = 0.0;
  const double half_period = 0.5 * period;

  for (std::size_t i = 1; i < n; ++i)
    {
      const double dd = p.eval_at (i) - p.eval_at (i - 1);
      double ddmod = std::fmod (dd + half_period, period);
      if (ddmod < 0.0)
        {
          ddmod += period;
        }
      ddmod -= half_period;
      if (ddmod == -half_period && dd > 0.0)
        {
          ddmod = half_period;
        }
      const double ph_corr = ddmod - dd;
      if (std::abs (dd) >= discont)
        {
          cum_corr += ph_corr;
        }
      out[i] = p.eval_at (i) + cum_corr;
    }
  return out;
}

/* Multi-key indirect stable lexicographical sort (NumPy lexsort).
   `keys.back()` is the primary sort key, `keys.front()` is the most minor.  */
template <typename T>
nd_tensor<std::size_t>
lexsort (const std::vector<nd_tensor<T>> &keys)
{
  if (keys.empty ())
    {
      return nd_tensor<std::size_t> (shape_type {0});
    }
  const std::size_t n = keys[0].size ();
  for (const auto &k : keys)
    {
      if (k.size () != n)
        {
          throw shape_error ("lexsort requires all key tensors to have equal length");
        }
    }

  std::vector<std::size_t> idx (n);
  std::iota (idx.begin (), idx.end (), 0);
  std::stable_sort (idx.begin (), idx.end (),
                    [&] (std::size_t a, std::size_t b) {
                      for (std::ptrdiff_t k =
                             static_cast<std::ptrdiff_t> (keys.size ()) - 1;
                           k >= 0; --k)
                        {
                          const auto &kt = keys[static_cast<std::size_t> (k)];
                          const T va = kt.eval_at (a);
                          const T vb = kt.eval_at (b);
                          if (va < vb)
                            {
                              return true;
                            }
                          if (vb < va)
                            {
                              return false;
                            }
                        }
                      return false;
                    });
  return nd_tensor<std::size_t> (shape_type {n}, idx);
}

/* 1-D consecutive differences and 1-D set operations: ediff1d, intersect1d,
   union1d, setdiff1d, isin (NumPy set routines).  */
template <typename T>
nd_tensor<T>
ediff1d (const nd_tensor<T> &a)
{
  const std::size_t n = a.size ();
  if (n < 2)
    {
      return nd_tensor<T> (shape_type {0});
    }
  nd_tensor<T> out (shape_type {n - 1});
  for (std::size_t i = 0; i + 1 < n; ++i)
    {
      out[i] = a.eval_at (i + 1) - a.eval_at (i);
    }
  return out;
}

template <typename T>
nd_tensor<T>
intersect1d (const nd_tensor<T> &ar1, const nd_tensor<T> &ar2)
{
  const nd_tensor<T> u1 = unique (ar1);
  const nd_tensor<T> u2 = unique (ar2);
  std::vector<T> v1 = u1.to_vector ();
  std::vector<T> v2 = u2.to_vector ();
  std::vector<T> res;
  std::set_intersection (v1.begin (), v1.end (),
                         v2.begin (), v2.end (),
                         std::back_inserter (res));
  return nd_tensor<T> (shape_type {res.size ()}, res);
}

template <typename T>
nd_tensor<T>
union1d (const nd_tensor<T> &ar1, const nd_tensor<T> &ar2)
{
  const nd_tensor<T> u1 = unique (ar1);
  const nd_tensor<T> u2 = unique (ar2);
  std::vector<T> v1 = u1.to_vector ();
  std::vector<T> v2 = u2.to_vector ();
  std::vector<T> res;
  std::set_union (v1.begin (), v1.end (),
                  v2.begin (), v2.end (),
                  std::back_inserter (res));
  return nd_tensor<T> (shape_type {res.size ()}, res);
}

template <typename T>
nd_tensor<T>
setdiff1d (const nd_tensor<T> &ar1, const nd_tensor<T> &ar2)
{
  const nd_tensor<T> u1 = unique (ar1);
  const nd_tensor<T> u2 = unique (ar2);
  std::vector<T> v1 = u1.to_vector ();
  std::vector<T> v2 = u2.to_vector ();
  std::vector<T> res;
  std::set_difference (v1.begin (), v1.end (),
                       v2.begin (), v2.end (),
                       std::back_inserter (res));
  return nd_tensor<T> (shape_type {res.size ()}, res);
}

template <typename T>
nd_tensor<std::uint8_t>
isin (const nd_tensor<T> &element, const nd_tensor<T> &test_elements)
{
  const nd_tensor<T> u_test = unique (test_elements);
  const std::vector<T> tv = u_test.to_vector ();
  nd_tensor<std::uint8_t> out (element.shape (), 0);
  for (std::size_t i = 0; i < element.size (); ++i)
    {
      if (std::binary_search (tv.begin (), tv.end (), element.eval_at (i)))
        {
          out[i] = 1;
        }
    }
  return out;
}

/* Polynomial ring algebra & symbolic calculus: polyadd, polysub, polymul,
   polyder, polyint (NumPy polynomial routines, highest-power-first).  */
inline nd_tensor<double>
polyadd (const nd_tensor<double> &p, const nd_tensor<double> &q)
{
  const std::size_t np = p.size ();
  const std::size_t nq = q.size ();
  const std::size_t n = std::max (np, nq);
  nd_tensor<double> out (shape_type {n}, 0.0);
  for (std::size_t i = 0; i < np; ++i)
    {
      out[n - np + i] += p.eval_at (i);
    }
  for (std::size_t i = 0; i < nq; ++i)
    {
      out[n - nq + i] += q.eval_at (i);
    }
  return out;
}

inline nd_tensor<double>
polysub (const nd_tensor<double> &p, const nd_tensor<double> &q)
{
  const std::size_t np = p.size ();
  const std::size_t nq = q.size ();
  const std::size_t n = std::max (np, nq);
  nd_tensor<double> out (shape_type {n}, 0.0);
  for (std::size_t i = 0; i < np; ++i)
    {
      out[n - np + i] += p.eval_at (i);
    }
  for (std::size_t i = 0; i < nq; ++i)
    {
      out[n - nq + i] -= q.eval_at (i);
    }
  return out;
}

inline nd_tensor<double>
polymul (const nd_tensor<double> &p, const nd_tensor<double> &q)
{
  if (p.empty () || q.empty ())
    {
      return nd_tensor<double> (shape_type {1}, 0.0);
    }
  const std::size_t np = p.size ();
  const std::size_t nq = q.size ();
  nd_tensor<double> out (shape_type {np + nq - 1}, 0.0);
  for (std::size_t i = 0; i < np; ++i)
    {
      for (std::size_t j = 0; j < nq; ++j)
        {
          out[i + j] += p.eval_at (i) * q.eval_at (j);
        }
    }
  return out;
}

inline nd_tensor<double>
polyder (const nd_tensor<double> &p, std::size_t m = 1)
{
  nd_tensor<double> cur = p.copy ();
  for (std::size_t step = 0; step < m; ++step)
    {
      if (cur.size () <= 1)
        {
          return nd_tensor<double> (shape_type {1}, 0.0);
        }
      const std::size_t deg = cur.size () - 1;
      nd_tensor<double> next (shape_type {deg});
      for (std::size_t i = 0; i < deg; ++i)
        {
          next[i] =
            cur.eval_at (i) * static_cast<double> (deg - i);
        }
      cur = std::move (next);
    }
  return cur;
}

inline nd_tensor<double>
polyint (const nd_tensor<double> &p, std::size_t m = 1, double k = 0.0)
{
  nd_tensor<double> cur = p.copy ();
  for (std::size_t step = 0; step < m; ++step)
    {
      const std::size_t n = cur.size ();
      nd_tensor<double> next (shape_type {n + 1}, 0.0);
      for (std::size_t i = 0; i < n; ++i)
        {
          const double power = static_cast<double> (n - i);
          next[i] = cur.eval_at (i) / power;
        }
      next[n] = (step == 0) ? k : 0.0;
      cur = std::move (next);
    }
  return cur;
}

/* Geometric progression and window functions: geomspace, bartlett, kaiser,
   and flatnonzero (NumPy equivalents).  */
inline nd_tensor<double>
geomspace (double start,
           double stop,
           std::size_t num = 50,
           bool endpoint = true)
{
  if (start <= 0.0 || stop <= 0.0)
    {
      throw hyper_math_error ("geomspace requires positive start and stop");
    }
  const nd_tensor<double> logs =
    linspace<double> (std::log10 (start), std::log10 (stop), num, endpoint);
  nd_tensor<double> out (shape_type {num});
  for (std::size_t i = 0; i < num; ++i)
    {
      out[i] = std::pow (10.0, logs[i]);
    }
  return out;
}

inline nd_tensor<double>
bartlett (std::size_t m)
{
  if (m == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  if (m == 1)
    {
      return nd_tensor<double> (shape_type {1}, 1.0);
    }
  nd_tensor<double> w (shape_type {m});
  const double dm = static_cast<double> (m - 1);
  for (std::size_t n = 0; n < m; ++n)
    {
      w[n] = 1.0 - std::abs ((2.0 * static_cast<double> (n) - dm) / dm);
    }
  return w;
}

inline nd_tensor<double>
kaiser (std::size_t m, double beta)
{
  if (m == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  if (m == 1)
    {
      return nd_tensor<double> (shape_type {1}, 1.0);
    }
  auto bessel_i0 = [] (double x) noexcept {
    double sum = 1.0;
    double term = 1.0;
    const double half_x = 0.5 * x;
    for (int k = 1; k <= 40; ++k)
      {
        const double r = half_x / static_cast<double> (k);
        term *= r * r;
        sum += term;
        if (term < 1e-16 * sum)
          {
            break;
          }
      }
    return sum;
  };

  nd_tensor<double> w (shape_type {m});
  const double den = bessel_i0 (beta);
  const double alpha = 0.5 * static_cast<double> (m - 1);
  for (std::size_t n = 0; n < m; ++n)
    {
      const double ratio = (static_cast<double> (n) - alpha) / alpha;
      const double arg = beta * std::sqrt (std::max (0.0, 1.0 - ratio * ratio));
      w[n] = bessel_i0 (arg) / den;
    }
  return w;
}

template <typename T>
nd_tensor<std::size_t>
flatnonzero (const nd_tensor<T> &a)
{
  return nonzero (a);
}

/* Clean-room NumPy .npy Binary Format v1.0 Reader & Writer (save_npy,
   load_npy). Writes standard "\x93NUMPY" v1.0 header + contiguous C-order
   little-endian payload so files can be exchanged with Python NumPy.  */
inline void
save_npy (const std::string &filepath, const nd_tensor<double> &arr)
{
  std::ofstream ofs (filepath, std::ios::binary);
  if (!ofs)
    {
      throw parse_error ("unable to open .npy file for writing: " + filepath);
    }

  std::ostringstream dict;
  dict << "{'descr': '<f8', 'fortran_order': False, 'shape': "
       << format_shape (arr.shape ()) << ", }";
  std::string header_str = dict.str ();

  /* Pad header with spaces to a multiple of 64 bytes (including 10-byte
     preamble and trailing '\n').  */
  const std::size_t preamble_len = 10;
  std::size_t total_hdr = preamble_len + header_str.size () + 1;
  const std::size_t rem = total_hdr % 64;
  if (rem != 0)
    {
      header_str.append (64 - rem, ' ');
    }
  header_str.push_back ('\n');

  const char magic[8] = {'\x93', 'N', 'U', 'M', 'P', 'Y', '\x01', '\x00'};
  ofs.write (magic, 8);
  const std::uint16_t hdr_len = static_cast<std::uint16_t> (header_str.size ());
  ofs.write (reinterpret_cast<const char *> (&hdr_len), 2);
  ofs.write (header_str.data (), static_cast<std::streamsize> (header_str.size ()));

  const nd_tensor<double> contig = arr.as_contiguous ();
  ofs.write (reinterpret_cast<const char *> (contig.raw_data ()),
             static_cast<std::streamsize> (contig.size () * sizeof (double)));
}

inline nd_tensor<double>
load_npy (const std::string &filepath)
{
  std::ifstream ifs (filepath, std::ios::binary);
  if (!ifs)
    {
      throw parse_error ("unable to open .npy file for reading: " + filepath);
    }

  char magic[8] = {};
  ifs.read (magic, 8);
  if (!ifs || magic[0] != '\x93' || magic[1] != 'N' || magic[2] != 'U'
      || magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y')
    {
      throw parse_error ("invalid .npy magic header in " + filepath);
    }

  std::uint16_t hdr_len = 0;
  ifs.read (reinterpret_cast<char *> (&hdr_len), 2);
  if (!ifs)
    {
      throw parse_error ("truncated .npy header length in " + filepath);
    }
  std::string header_str (hdr_len, '\0');
  ifs.read (header_str.data (), hdr_len);
  if (!ifs)
    {
      throw parse_error ("truncated .npy header body in " + filepath);
    }

  const std::size_t shape_pos = header_str.find ("'shape':");
  if (shape_pos == std::string::npos)
    {
      throw parse_error ("missing 'shape' field in .npy header");
    }
  const std::size_t lparen = header_str.find ('(', shape_pos);
  const std::size_t rparen = header_str.find (')', lparen);
  if (lparen == std::string::npos || rparen == std::string::npos)
    {
      throw parse_error ("malformed shape tuple in .npy header");
    }

  const std::string tuple_body =
    header_str.substr (lparen + 1, rparen - lparen - 1);
  shape_type dims;
  std::stringstream ss (tuple_body);
  std::string item;
  while (std::getline (ss, item, ','))
    {
      std::size_t p = 0;
      while (p < item.size () && item[p] == ' ')
        {
          ++p;
        }
      if (p < item.size ())
        {
          dims.push_back (static_cast<std::size_t> (std::stoull (item.substr (p))));
        }
    }
  if (dims.empty ())
    {
      dims.push_back (1);
    }

  nd_tensor<double> out (std::move (dims));
  const std::streamsize byte_count =
    static_cast<std::streamsize> (out.size () * sizeof (double));
  if (byte_count > 0)
    {
      ifs.read (reinterpret_cast<char *> (out.raw_data ()), byte_count);
      if (!ifs)
        {
          throw parse_error ("truncated .npy payload in " + filepath);
        }
    }
  return out;
}

/* Axis & Dimension Manipulation: squeeze, expand_dims, swapaxes, moveaxis,
   atleast_1d, atleast_2d, atleast_3d, broadcast_to, and 1-D piecewise linear
   interp (NumPy equivalents).  */
template <typename T>
nd_tensor<T>
squeeze (const nd_tensor<T> &a)
{
  shape_type new_shape;
  for (std::size_t d : a.shape ())
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
  return a.reshape (new_shape);
}

template <typename T>
nd_tensor<T>
expand_dims (const nd_tensor<T> &a, std::size_t axis)
{
  if (axis > a.ndim ())
    {
      throw shape_error ("expand_dims axis out of bounds");
    }
  shape_type new_shape = a.shape ();
  new_shape.insert (new_shape.begin () + static_cast<std::ptrdiff_t> (axis), 1);
  return a.reshape (new_shape);
}

template <typename T>
nd_tensor<T>
swapaxes (const nd_tensor<T> &a, std::size_t axis1, std::size_t axis2)
{
  if (axis1 >= a.ndim () || axis2 >= a.ndim ())
    {
      throw shape_error ("swapaxes axis out of bounds");
    }
  std::vector<std::size_t> axes (a.ndim ());
  std::iota (axes.begin (), axes.end (), 0);
  std::swap (axes[axis1], axes[axis2]);
  return a.transpose (axes);
}

template <typename T>
nd_tensor<T>
moveaxis (const nd_tensor<T> &a, std::size_t source, std::size_t destination)
{
  if (source >= a.ndim () || destination >= a.ndim ())
    {
      throw shape_error ("moveaxis axis out of bounds");
    }
  std::vector<std::size_t> axes;
  axes.reserve (a.ndim ());
  for (std::size_t i = 0; i < a.ndim (); ++i)
    {
      if (i != source)
        {
          axes.push_back (i);
        }
    }
  axes.insert (axes.begin () + static_cast<std::ptrdiff_t> (destination),
               source);
  return a.transpose (axes);
}

template <typename T>
nd_tensor<T>
atleast_1d (const nd_tensor<T> &a)
{
  if (a.ndim () >= 1)
    {
      return a;
    }
  return nd_tensor<T> (shape_type {a.size ()});
}

template <typename T>
nd_tensor<T>
atleast_2d (const nd_tensor<T> &a)
{
  if (a.ndim () == 0)
    {
      return nd_tensor<T> (shape_type {1, a.size ()});
    }
  if (a.ndim () == 1)
    {
      return a.reshape (shape_type {1, a.size ()});
    }
  return a;
}

template <typename T>
nd_tensor<T>
atleast_3d (const nd_tensor<T> &a)
{
  if (a.ndim () == 0)
    {
      return nd_tensor<T> (shape_type {1, a.size (), 1});
    }
  if (a.ndim () == 1)
    {
      return a.reshape (shape_type {1, a.size (), 1});
    }
  if (a.ndim () == 2)
    {
      return a.reshape (shape_type {a.shape ()[0], a.shape ()[1], 1});
    }
  return a;
}

template <typename T>
nd_tensor<T>
broadcast_to (const nd_tensor<T> &a, const shape_type &target_shape)
{
  return a.broadcast_to (target_shape);
}

/* 1-D Piecewise Linear Interpolation with boundary clamping (NumPy interp). */
inline nd_tensor<double>
interp (const nd_tensor<double> &x,
        const nd_tensor<double> &xp,
        const nd_tensor<double> &fp)
{
  const std::size_t n = xp.size ();
  if (n == 0 || fp.size () != n)
    {
      throw shape_error ("interp requires matching non-empty xp and fp");
    }
  nd_tensor<double> out (x.shape ());
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      const double xv = x.eval_at (i);
      if (xv <= xp.eval_at (0) || n == 1)
        {
          out[i] = fp.eval_at (0);
          continue;
        }
      if (xv >= xp.eval_at (n - 1))
        {
          out[i] = fp.eval_at (n - 1);
          continue;
        }
      /* Binary search for interval [xp[lo], xp[lo + 1]].  */
      std::size_t lo = 0;
      std::size_t hi = n - 1;
      while (lo + 1 < hi)
        {
          const std::size_t mid = lo + (hi - lo) / 2;
          if (xp.eval_at (mid) <= xv)
            {
              lo = mid;
            }
          else
            {
              hi = mid;
            }
        }
      const double x0 = xp.eval_at (lo);
      const double x1 = xp.eval_at (lo + 1);
      const double y0 = fp.eval_at (lo);
      const double y1 = fp.eval_at (lo + 1);
      const double t = (x1 > x0) ? ((xv - x0) / (x1 - x0)) : 0.0;
      out[i] = y0 + t * (y1 - y0);
    }
  return out;
}

/* Multi-index coordinate conversion (`unravel_index`, `ravel_multi_index`)
   and diagonal/triangular matrix builders (`diagflat`, `fill_diagonal`, `tri`)
   (NumPy equivalents).  */
inline shape_type
unravel_index (std::size_t flat_idx, const shape_type &dims)
{
  if (dims.empty ())
    {
      throw shape_error ("unravel_index requires non-empty shape");
    }
  std::size_t total = 1;
  for (std::size_t d : dims)
    {
      total *= d;
    }
  if (flat_idx >= total)
    {
      throw index_error ("unravel_index flat_idx out of bounds for shape");
    }
  shape_type coords (dims.size (), 0);
  std::size_t rem = flat_idx;
  for (std::size_t k = dims.size (); k > 0; --k)
    {
      const std::size_t d = dims[k - 1];
      coords[k - 1] = rem % d;
      rem /= d;
    }
  return coords;
}

inline std::size_t
ravel_multi_index (const shape_type &coords, const shape_type &dims)
{
  if (coords.size () != dims.size () || dims.empty ())
    {
      throw shape_error ("ravel_multi_index rank mismatch");
    }
  std::size_t flat = 0;
  for (std::size_t i = 0; i < dims.size (); ++i)
    {
      if (coords[i] >= dims[i])
        {
          throw index_error ("ravel_multi_index coordinate out of bounds");
        }
      flat = flat * dims[i] + coords[i];
    }
  return flat;
}

template <typename T>
nd_tensor<T>
diagflat (const nd_tensor<T> &v, std::ptrdiff_t k = 0)
{
  const std::size_t n = v.size ();
  const std::size_t offset =
    static_cast<std::size_t> (std::abs (k));
  const std::size_t dim = n + offset;
  nd_tensor<T> out (shape_type {dim, dim}, T {0});
  for (std::size_t i = 0; i < n; ++i)
    {
      const std::size_t r = (k >= 0) ? i : (i + offset);
      const std::size_t c = (k >= 0) ? (i + offset) : i;
      out (r, c) = v.eval_at (i);
    }
  return out;
}

template <typename T>
void
fill_diagonal (nd_tensor<T> &a, T val)
{
  if (a.ndim () < 2)
    {
      throw shape_error ("fill_diagonal requires at least a 2D tensor");
    }
  std::size_t min_dim = a.shape ()[0];
  for (std::size_t d = 1; d < a.ndim (); ++d)
    {
      min_dim = std::min (min_dim, a.shape ()[d]);
    }
  for (std::size_t i = 0; i < min_dim; ++i)
    {
      if (a.ndim () == 2)
        {
          a (i, i) = val;
        }
      else
        {
          shape_type coords (a.ndim (), i);
          a.at (coords) = val;
        }
    }
}

template <typename T = double>
nd_tensor<T>
tri (std::size_t n, std::size_t m = 0, std::ptrdiff_t k = 0)
{
  if (m == 0)
    {
      m = n;
    }
  nd_tensor<T> out (shape_type {n, m}, T {0});
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < m; ++j)
        {
          if (static_cast<std::ptrdiff_t> (j)
              <= static_cast<std::ptrdiff_t> (i) + k)
            {
              out (i, j) = T {1};
            }
        }
    }
  return out;
}

/* 2-D Block Matrix Assembly from a 2-D grid of sub-matrices (NumPy block).  */
template <typename T>
nd_tensor<T>
block_2d (const std::vector<std::vector<nd_tensor<T>>> &blocks)
{
  if (blocks.empty () || blocks[0].empty ())
    {
      throw shape_error ("block_2d requires non-empty 2D grid of blocks");
    }
  std::vector<nd_tensor<T>> row_strips;
  row_strips.reserve (blocks.size ());
  for (const auto &row : blocks)
    {
      if (row.empty ())
        {
          throw shape_error ("block_2d encountered an empty block row");
        }
      row_strips.push_back (hstack (row));
    }
  return vstack (row_strips);
}

/* Lower and Upper Triangular Coordinate Index Generators:
   tril_indices and triu_indices (NumPy equivalents).  */
inline std::pair<nd_tensor<std::size_t>, nd_tensor<std::size_t>>
tril_indices (std::size_t n, std::ptrdiff_t k = 0, std::size_t m = 0)
{
  if (m == 0)
    {
      m = n;
    }
  std::vector<std::size_t> rows;
  std::vector<std::size_t> cols;
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < m; ++j)
        {
          if (static_cast<std::ptrdiff_t> (j)
              <= static_cast<std::ptrdiff_t> (i) + k)
            {
              rows.push_back (i);
              cols.push_back (j);
            }
        }
    }
  return {
    nd_tensor<std::size_t> (shape_type {rows.size ()}, rows),
    nd_tensor<std::size_t> (shape_type {cols.size ()}, cols)
  };
}

inline std::pair<nd_tensor<std::size_t>, nd_tensor<std::size_t>>
triu_indices (std::size_t n, std::ptrdiff_t k = 0, std::size_t m = 0)
{
  if (m == 0)
    {
      m = n;
    }
  std::vector<std::size_t> rows;
  std::vector<std::size_t> cols;
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < m; ++j)
        {
          if (static_cast<std::ptrdiff_t> (j)
              >= static_cast<std::ptrdiff_t> (i) + k)
            {
              rows.push_back (i);
              cols.push_back (j);
            }
        }
    }
  return {
    nd_tensor<std::size_t> (shape_type {rows.size ()}, rows),
    nd_tensor<std::size_t> (shape_type {cols.size ()}, cols)
  };
}

/* Main diagonal index generator `diag_indices(n)`, axis rotation `rollaxis`,
   and boundary zero trimming `trim_zeros` (NumPy equivalents).  */
inline std::pair<nd_tensor<std::size_t>, nd_tensor<std::size_t>>
diag_indices (std::size_t n)
{
  std::vector<std::size_t> idx (n);
  for (std::size_t i = 0; i < n; ++i)
    {
      idx[i] = i;
    }
  return {
    nd_tensor<std::size_t> (shape_type {n}, idx),
    nd_tensor<std::size_t> (shape_type {n}, idx)
  };
}

template <typename T>
nd_tensor<T>
rollaxis (const nd_tensor<T> &t, std::size_t axis, std::size_t start = 0)
{
  const std::size_t nd = t.ndim ();
  if (axis >= nd || start > nd)
    {
      throw shape_error ("rollaxis axis or start out of bounds");
    }
  if (axis < start)
    {
      --start;
    }
  return moveaxis (t, axis, start);
}

template <typename T>
nd_tensor<T>
trim_zeros (const nd_tensor<T> &t, const std::string &trim = "fb")
{
  const std::size_t n = t.size ();
  std::size_t first = 0;
  if (trim.find ('f') != std::string::npos
      || trim.find ('F') != std::string::npos)
    {
      while (first < n && t.eval_at (first) == T {0})
        {
          ++first;
        }
    }
  std::size_t last = n;
  if (trim.find ('b') != std::string::npos
      || trim.find ('B') != std::string::npos)
    {
      while (last > first && t.eval_at (last - 1) == T {0})
        {
          --last;
        }
    }
  nd_tensor<T> out (shape_type {last - first});
  for (std::size_t i = first; i < last; ++i)
    {
      out[i - first] = t.eval_at (i);
    }
  return out;
}

/* 1-D structural element deletion `delete_elements`, insertion
   `insert_elements`, and appending `append_elements`
   (NumPy delete, insert, and append equivalents).  */
template <typename T>
nd_tensor<T>
delete_elements (const nd_tensor<T> &t,
                 const std::vector<std::size_t> &indices)
{
  std::set<std::size_t> drop_set (indices.begin (), indices.end ());
  std::vector<T> kept;
  kept.reserve (t.size ());
  for (std::size_t i = 0; i < t.size (); ++i)
    {
      if (drop_set.find (i) == drop_set.end ())
        {
          kept.push_back (t.eval_at (i));
        }
    }
  return nd_tensor<T> (shape_type {kept.size ()}, kept);
}

template <typename T>
nd_tensor<T>
insert_elements (const nd_tensor<T> &t,
                 std::size_t index,
                 const nd_tensor<T> &values)
{
  if (index > t.size ())
    {
      throw shape_error ("insert_elements index out of bounds");
    }
  const std::size_t n_out = t.size () + values.size ();
  nd_tensor<T> out (shape_type {n_out});
  for (std::size_t i = 0; i < index; ++i)
    {
      out[i] = t.eval_at (i);
    }
  for (std::size_t j = 0; j < values.size (); ++j)
    {
      out[index + j] = values.eval_at (j);
    }
  for (std::size_t i = index; i < t.size (); ++i)
    {
      out[values.size () + i] = t.eval_at (i);
    }
  return out;
}

template <typename T>
nd_tensor<T>
append_elements (const nd_tensor<T> &t, const nd_tensor<T> &values)
{
  return insert_elements (t, t.size (), values);
}

/* 2-D joint histogram `histogram2d` and 1-D symmetric set difference
   `setxor1d` (NumPy histogram2d and setxor1d equivalents).  */
struct histogram2d_result
{
  nd_tensor<double> counts;
  nd_tensor<double> x_edges;
  nd_tensor<double> y_edges;
};

inline histogram2d_result
histogram2d (const nd_tensor<double> &x,
             const nd_tensor<double> &y,
             std::size_t bins_x = 10,
             std::size_t bins_y = 10)
{
  if (x.size () != y.size () || x.size () == 0 || bins_x == 0 || bins_y == 0)
    {
      throw shape_error ("histogram2d requires non-empty matching x, y and bins >= 1");
    }
  const double xmin = min (x);
  const double xmax = max (x);
  const double ymin = min (y);
  const double ymax = max (y);
  const double xspan = (xmax > xmin) ? (xmax - xmin) : 1.0;
  const double yspan = (ymax > ymin) ? (ymax - ymin) : 1.0;

  nd_tensor<double> x_edges = linspace (xmin, xmin + xspan, bins_x + 1);
  nd_tensor<double> y_edges = linspace (ymin, ymin + yspan, bins_y + 1);
  nd_tensor<double> counts (shape_type {bins_x, bins_y}, 0.0);

  for (std::size_t i = 0; i < x.size (); ++i)
    {
      const double xv = x.eval_at (i);
      const double yv = y.eval_at (i);
      std::size_t bx = static_cast<std::size_t> (
        std::floor (((xv - xmin) / xspan) * static_cast<double> (bins_x)));
      std::size_t by = static_cast<std::size_t> (
        std::floor (((yv - ymin) / yspan) * static_cast<double> (bins_y)));
      if (bx >= bins_x)
        {
          bx = bins_x - 1;
        }
      if (by >= bins_y)
        {
          by = bins_y - 1;
        }
      counts (bx, by) += 1.0;
    }

  return histogram2d_result {
    std::move (counts), std::move (x_edges), std::move (y_edges)
  };
}

template <typename T>
nd_tensor<T>
setxor1d (const nd_tensor<T> &a, const nd_tensor<T> &b)
{
  const nd_tensor<T> diff_ab = setdiff1d (a, b);
  const nd_tensor<T> diff_ba = setdiff1d (b, a);
  return union1d (diff_ab, diff_ba);
}

/* 1-D sliding window view `sliding_window_view_1d` and spectral window
   generators `blackman`, `hamming`, `hanning`
   (NumPy lib.stride_tricks.sliding_window_view and window equivalents).  */
template <typename T>
nd_tensor<T>
sliding_window_view_1d (const nd_tensor<T> &t, std::size_t window_size)
{
  const std::size_t n = t.size ();
  if (window_size == 0 || window_size > n)
    {
      throw shape_error ("sliding_window_view_1d requires 1 <= window_size <= n");
    }
  const std::size_t n_win = n - window_size + 1;
  nd_tensor<T> out (shape_type {n_win, window_size});
  for (std::size_t i = 0; i < n_win; ++i)
    {
      for (std::size_t j = 0; j < window_size; ++j)
        {
          out (i, j) = t.eval_at (i + j);
        }
    }
  return out;
}

inline nd_tensor<double>
hanning (std::size_t m)
{
  if (m == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  if (m == 1)
    {
      return nd_tensor<double> {1.0};
    }
  nd_tensor<double> out (shape_type {m});
  const double dm1 = static_cast<double> (m - 1);
  for (std::size_t i = 0; i < m; ++i)
    {
      out[i] =
        0.5 - 0.5 * std::cos ((2.0 * constants::pi * static_cast<double> (i)) / dm1);
    }
  return out;
}

inline nd_tensor<double>
hamming (std::size_t m)
{
  if (m == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  if (m == 1)
    {
      return nd_tensor<double> {1.0};
    }
  nd_tensor<double> out (shape_type {m});
  const double dm1 = static_cast<double> (m - 1);
  for (std::size_t i = 0; i < m; ++i)
    {
      out[i] =
        0.54
        - 0.46 * std::cos ((2.0 * constants::pi * static_cast<double> (i)) / dm1);
    }
  return out;
}

inline nd_tensor<double>
blackman (std::size_t m)
{
  if (m == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  if (m == 1)
    {
      return nd_tensor<double> {1.0};
    }
  nd_tensor<double> out (shape_type {m});
  const double dm1 = static_cast<double> (m - 1);
  for (std::size_t i = 0; i < m; ++i)
    {
      const double phase =
        (2.0 * constants::pi * static_cast<double> (i)) / dm1;
      out[i] =
        0.42 - 0.5 * std::cos (phase) + 0.08 * std::cos (2.0 * phase);
    }
  return out;
}

/* Matrix-shape index helpers: `diag_indices_from`, `tril_indices_from`,
   `triu_indices_from`, and open mesh builder `ix_` (NumPy equivalents).  */
template <typename T>
std::pair<nd_tensor<std::size_t>, nd_tensor<std::size_t>>
diag_indices_from (const nd_tensor<T> &a)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("diag_indices_from requires a 2D square tensor");
    }
  return diag_indices (a.shape ()[0]);
}

template <typename T>
std::pair<nd_tensor<std::size_t>, nd_tensor<std::size_t>>
tril_indices_from (const nd_tensor<T> &a, std::ptrdiff_t k = 0)
{
  if (a.ndim () != 2)
    {
      throw shape_error ("tril_indices_from requires a 2D tensor");
    }
  return tril_indices (a.shape ()[0], k, a.shape ()[1]);
}

template <typename T>
std::pair<nd_tensor<std::size_t>, nd_tensor<std::size_t>>
triu_indices_from (const nd_tensor<T> &a, std::ptrdiff_t k = 0)
{
  if (a.ndim () != 2)
    {
      throw shape_error ("triu_indices_from requires a 2D tensor");
    }
  return triu_indices (a.shape ()[0], k, a.shape ()[1]);
}

template <typename T>
std::pair<nd_tensor<T>, nd_tensor<T>>
ix_ (const nd_tensor<T> &r, const nd_tensor<T> &c)
{
  return {
    r.reshape (shape_type {r.size (), 1}),
    c.reshape (shape_type {1, c.size ()})
  };
}

template <typename T>
std::vector<nd_tensor<T>>
ix_ (const std::vector<nd_tensor<T>> &seqs)
{
  const std::size_t nd = seqs.size ();
  std::vector<nd_tensor<T>> grids;
  grids.reserve (nd);
  for (std::size_t d = 0; d < nd; ++d)
    {
      shape_type sh (nd, 1);
      sh[d] = seqs[d].size ();
      grids.push_back (seqs[d].reshape (sh));
    }
  return grids;
}

/* Multi-condition selection `select`, functional `piecewise`, and in-place
   masked replacement `place` (NumPy select, piecewise, place equivalents).  */
template <typename T>
nd_tensor<T>
select (const std::vector<nd_tensor<std::uint8_t>> &condlist,
        const std::vector<nd_tensor<T>> &choicelist,
        T default_val = T {})
{
  if (condlist.empty () || condlist.size () != choicelist.size ())
    {
      throw shape_error ("select requires equal non-empty condlist and choicelist");
    }
  const shape_type sh = condlist[0].shape ();
  const std::size_t n = condlist[0].size ();
  for (std::size_t k = 0; k < condlist.size (); ++k)
    {
      if (condlist[k].shape () != sh || choicelist[k].shape () != sh)
        {
          throw shape_error ("select condition and choice shapes must match");
        }
    }

  nd_tensor<T> out (sh, default_val);
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t k = 0; k < condlist.size (); ++k)
        {
          if (condlist[k].eval_at (i) != 0)
            {
              out[i] = choicelist[k].eval_at (i);
              break;
            }
        }
    }
  return out;
}

template <typename T>
nd_tensor<T>
piecewise (const nd_tensor<T> &x,
           const std::vector<nd_tensor<std::uint8_t>> &condlist,
           const std::vector<std::function<T (T)>> &funclist)
{
  if (condlist.empty ()
      || (funclist.size () != condlist.size ()
          && funclist.size () != condlist.size () + 1))
    {
      throw shape_error ("piecewise requires funclist of size N or N+1");
    }
  for (const auto &c : condlist)
    {
      if (c.shape () != x.shape ())
        {
          throw shape_error ("piecewise condition shape must match x");
        }
    }

  nd_tensor<T> out (x.shape (), T {});
  const bool has_default = (funclist.size () == condlist.size () + 1);
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      bool matched = false;
      for (std::size_t k = 0; k < condlist.size (); ++k)
        {
          if (condlist[k].eval_at (i) != 0)
            {
              out[i] = funclist[k] (x.eval_at (i));
              matched = true;
              break;
            }
        }
      if (!matched && has_default)
        {
          out[i] = funclist.back () (x.eval_at (i));
        }
    }
  return out;
}

template <typename T>
void
place (nd_tensor<T> &arr,
       const nd_tensor<std::uint8_t> &mask,
       const nd_tensor<T> &vals)
{
  if (arr.shape () != mask.shape ())
    {
      throw shape_error ("place requires arr and mask to have identical shapes");
    }
  const std::size_t nv = vals.size ();
  std::size_t v_idx = 0;
  for (std::size_t i = 0; i < arr.size (); ++i)
    {
      if (mask.eval_at (i) != 0)
        {
          if (nv == 0)
            {
              throw shape_error ("place cannot insert from empty vals tensor");
            }
          arr[i] = vals.eval_at (v_idx % nv);
          ++v_idx;
        }
    }
}

/* Polynomial synthetic division `polydiv(u, v)` returning {quotient, remainder}
   in descending-power order (NumPy polydiv equivalent).  */
inline std::pair<nd_tensor<double>, nd_tensor<double>>
polydiv (const nd_tensor<double> &u, const nd_tensor<double> &v)
{
  if (u.size () == 0 || v.size () == 0)
    {
      throw shape_error ("polydiv requires non-empty coefficient tensors");
    }
  /* Strip leading zeros from divisor v.  */
  std::size_t v_start = 0;
  while (v_start + 1 < v.size () && std::abs (v.eval_at (v_start)) == 0.0)
    {
      ++v_start;
    }
  const double v0 = v.eval_at (v_start);
  if (std::abs (v0) == 0.0)
    {
      throw convergence_error ("polydiv division by zero polynomial");
    }
  const std::size_t m = u.size ();
  const std::size_t n = v.size () - v_start;
  if (m < n)
    {
      return {nd_tensor<double> {0.0}, u};
    }

  std::vector<double> rem = u.to_vector ();
  const std::size_t q_len = m - n + 1;
  nd_tensor<double> quot (shape_type {q_len}, 0.0);
  for (std::size_t k = 0; k < q_len; ++k)
    {
      const double coeff = rem[k] / v0;
      quot[k] = coeff;
      rem[k] = 0.0;
      for (std::size_t j = 1; j < n; ++j)
        {
          rem[k + j] -= coeff * v.eval_at (v_start + j);
        }
    }

  std::size_t r_start = q_len;
  while (r_start + 1 < m && std::abs (rem[r_start]) < 1e-15)
    {
      ++r_start;
    }
  if (r_start >= m)
    {
      return {quot, nd_tensor<double> {0.0}};
    }
  std::vector<double> r_vec (rem.begin () + static_cast<std::ptrdiff_t> (r_start),
                             rem.end ());
  return {quot, nd_tensor<double> (shape_type {r_vec.size ()}, r_vec)};
}

/* Delimited ASCII/CSV text tensor I/O `savetxt` and `loadtxt`
   (NumPy savetxt / loadtxt equivalents).  */
inline void
savetxt (const std::string &filepath,
         const nd_tensor<double> &t,
         char delimiter = ' ')
{
  if (t.ndim () == 0 || t.ndim () > 2)
    {
      throw shape_error ("savetxt supports 1D or 2D tensors");
    }
  std::ofstream ofs (filepath);
  if (!ofs)
    {
      throw parse_error ("savetxt unable to open file for writing: " + filepath);
    }
  ofs << std::setprecision (17);
  if (t.ndim () == 1)
    {
      for (std::size_t i = 0; i < t.size (); ++i)
        {
          ofs << t.eval_at (i) << "\n";
        }
    }
  else
    {
      const std::size_t rows = t.shape ()[0];
      const std::size_t cols = t.shape ()[1];
      for (std::size_t r = 0; r < rows; ++r)
        {
          for (std::size_t c = 0; c < cols; ++c)
            {
              if (c > 0)
                {
                  ofs << delimiter;
                }
              ofs << t (r, c);
            }
          ofs << "\n";
        }
    }
}

inline nd_tensor<double>
loadtxt (const std::string &filepath, char delimiter = ' ')
{
  std::ifstream ifs (filepath);
  if (!ifs)
    {
      throw parse_error ("loadtxt unable to open file for reading: " + filepath);
    }
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline (ifs, line))
    {
      if (!line.empty () && line.back () == '\r')
        {
          line.pop_back ();
        }
      const std::size_t comment_pos = line.find ('#');
      if (comment_pos != std::string::npos)
        {
          line = line.substr (0, comment_pos);
        }
      std::vector<double> row;
      if (delimiter == ' ')
        {
          std::istringstream iss (line);
          double val = 0.0;
          while (iss >> val)
            {
              row.push_back (val);
            }
        }
      else
        {
          std::stringstream ss (line);
          std::string cell;
          while (std::getline (ss, cell, delimiter))
            {
              std::istringstream cs (cell);
              double val = 0.0;
              if (cs >> val)
                {
                  row.push_back (val);
                }
            }
        }
      if (!row.empty ())
        {
          if (!rows.empty () && row.size () != rows[0].size ())
            {
              throw parse_error ("loadtxt inconsistent column count across rows");
            }
          rows.push_back (std::move (row));
        }
    }
  if (rows.empty ())
    {
      return nd_tensor<double> (shape_type {0});
    }
  const std::size_t nr = rows.size ();
  const std::size_t nc = rows[0].size ();
  if (nc == 1)
    {
      nd_tensor<double> out (shape_type {nr});
      for (std::size_t r = 0; r < nr; ++r)
        {
          out[r] = rows[r][0];
        }
      return out;
    }
  nd_tensor<double> out (shape_type {nr, nc});
  for (std::size_t r = 0; r < nr; ++r)
    {
      for (std::size_t c = 0; c < nc; ++c)
        {
          out (r, c) = rows[r][c];
        }
    }
  return out;
}

/* Masked Tensor `masked_tensor<T>` (`numpy.ma.MaskedArray` equivalent)
   and factory helpers `masked_where`, `masked_invalid`, `masked_equal`,
   plus Multi-Tensor Archive I/O `save_npz` and `load_npz`.  */
template <typename T = double>
class masked_tensor
{
private:
  nd_tensor<T> data_;
  nd_tensor<std::uint8_t> mask_;

public:
  masked_tensor (nd_tensor<T> data, nd_tensor<std::uint8_t> mask)
    : data_ (std::move (data)), mask_ (std::move (mask))
  {
    if (data_.shape () != mask_.shape ())
      {
        throw shape_error ("masked_tensor data and mask shapes must match");
      }
  }

  const nd_tensor<T> &
  data () const noexcept
  {
    return data_;
  }

  const nd_tensor<std::uint8_t> &
  mask () const noexcept
  {
    return mask_;
  }

  const shape_type &
  shape () const noexcept
  {
    return data_.shape ();
  }

  std::size_t
  size () const noexcept
  {
    return data_.size ();
  }

  bool
  is_masked (std::size_t idx) const
  {
    return mask_.eval_at (idx) != 0;
  }

  std::size_t
  count () const noexcept
  {
    std::size_t valid = 0;
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (mask_.eval_at (i) == 0)
          {
            ++valid;
          }
      }
    return valid;
  }

  nd_tensor<T>
  filled (T fill_value) const
  {
    nd_tensor<T> out (data_.shape ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        out[i] = (mask_.eval_at (i) != 0) ? fill_value : data_.eval_at (i);
      }
    return out;
  }

  nd_tensor<T>
  compressed () const
  {
    std::vector<T> vals;
    vals.reserve (count ());
    for (std::size_t i = 0; i < size (); ++i)
      {
        if (mask_.eval_at (i) == 0)
          {
            vals.push_back (data_.eval_at (i));
          }
      }
    return nd_tensor<T> (shape_type {vals.size ()}, vals);
  }

  double
  sum () const
  {
    return tensor::sum (compressed ());
  }

  double
  mean () const
  {
    const nd_tensor<T> comp = compressed ();
    return (comp.size () > 0) ? tensor::mean (comp) : constants::nan;
  }

  double
  var () const
  {
    const nd_tensor<T> comp = compressed ();
    return (comp.size () > 0) ? tensor::var (comp) : constants::nan;
  }

  double
  std_dev () const
  {
    const nd_tensor<T> comp = compressed ();
    return (comp.size () > 0) ? tensor::std_dev (comp) : constants::nan;
  }

  T
  min () const
  {
    return tensor::min (compressed ());
  }

  T
  max () const
  {
    return tensor::max (compressed ());
  }
};

template <typename T>
masked_tensor<T>
masked_where (const nd_tensor<std::uint8_t> &condition,
              const nd_tensor<T> &data)
{
  return masked_tensor<T> (data, condition);
}

inline masked_tensor<double>
masked_invalid (const nd_tensor<double> &data)
{
  nd_tensor<std::uint8_t> m (data.shape (), 0);
  for (std::size_t i = 0; i < data.size (); ++i)
    {
      m[i] = std::isfinite (data.eval_at (i)) ? 0 : 1;
    }
  return masked_tensor<double> (data, std::move (m));
}

template <typename T>
masked_tensor<T>
masked_equal (const nd_tensor<T> &data, T value)
{
  nd_tensor<std::uint8_t> m (data.shape (), 0);
  for (std::size_t i = 0; i < data.size (); ++i)
    {
      m[i] = (data.eval_at (i) == value) ? 1 : 0;
    }
  return masked_tensor<T> (data, std::move (m));
}

/* Multi-tensor binary archive container `save_npz` and `load_npz`
   (NumPy savez / load equivalents for named nd_tensor<double> collections).  */
inline void
save_npz (const std::string &filepath,
          const std::map<std::string, nd_tensor<double>> &tensors)
{
  std::ofstream ofs (filepath, std::ios::binary);
  if (!ofs)
    {
      throw parse_error ("save_npz unable to open file for writing: " + filepath);
    }
  const char magic[4] = {'H', 'M', 'Z', '\x01'};
  ofs.write (magic, 4);
  const std::uint64_t count = static_cast<std::uint64_t> (tensors.size ());
  ofs.write (reinterpret_cast<const char *> (&count), sizeof (count));

  for (const auto &[name, t] : tensors)
    {
      const std::uint64_t name_len = static_cast<std::uint64_t> (name.size ());
      ofs.write (reinterpret_cast<const char *> (&name_len), sizeof (name_len));
      ofs.write (name.data (), static_cast<std::streamsize> (name_len));

      const std::uint64_t ndim = static_cast<std::uint64_t> (t.ndim ());
      ofs.write (reinterpret_cast<const char *> (&ndim), sizeof (ndim));
      for (std::size_t d = 0; d < t.ndim (); ++d)
        {
          const std::uint64_t dim_sz = static_cast<std::uint64_t> (t.shape ()[d]);
          ofs.write (reinterpret_cast<const char *> (&dim_sz), sizeof (dim_sz));
        }
      for (std::size_t i = 0; i < t.size (); ++i)
        {
          const double v = t.eval_at (i);
          ofs.write (reinterpret_cast<const char *> (&v), sizeof (v));
        }
    }
}

inline std::map<std::string, nd_tensor<double>>
load_npz (const std::string &filepath)
{
  std::ifstream ifs (filepath, std::ios::binary);
  if (!ifs)
    {
      throw parse_error ("load_npz unable to open file for reading: " + filepath);
    }
  char magic[4] = {};
  ifs.read (magic, 4);
  if (!ifs || magic[0] != 'H' || magic[1] != 'M'
      || magic[2] != 'Z' || magic[3] != '\x01')
    {
      throw parse_error ("load_npz invalid archive header");
    }
  std::uint64_t count = 0;
  ifs.read (reinterpret_cast<char *> (&count), sizeof (count));

  std::map<std::string, nd_tensor<double>> out;
  for (std::uint64_t k = 0; k < count; ++k)
    {
      std::uint64_t name_len = 0;
      ifs.read (reinterpret_cast<char *> (&name_len), sizeof (name_len));
      std::string name (static_cast<std::size_t> (name_len), '\0');
      ifs.read (name.data (), static_cast<std::streamsize> (name_len));

      std::uint64_t ndim = 0;
      ifs.read (reinterpret_cast<char *> (&ndim), sizeof (ndim));
      shape_type sh (static_cast<std::size_t> (ndim));
      for (std::size_t d = 0; d < sh.size (); ++d)
        {
          std::uint64_t dim_sz = 0;
          ifs.read (reinterpret_cast<char *> (&dim_sz), sizeof (dim_sz));
          sh[d] = static_cast<std::size_t> (dim_sz);
        }
      nd_tensor<double> t (sh);
      for (std::size_t i = 0; i < t.size (); ++i)
        {
          double v = 0.0;
          ifs.read (reinterpret_cast<char *> (&v), sizeof (v));
          t[i] = v;
        }
      out.emplace (std::move (name), std::move (t));
    }
  return out;
}

/* O(N) k-th element partitioning (`partition`, `argpartition`), axis-indexed
   gather/scatter (`take_along_axis`, `put_along_axis`), and binary bitpacking
   (`packbits`, `unpackbits`) (NumPy equivalents).  */
template <typename T>
nd_tensor<T>
partition (const nd_tensor<T> &a, std::size_t kth)
{
  if (kth >= a.size ())
    {
      throw index_error ("partition kth index out of bounds");
    }
  std::vector<T> vals = a.to_vector ();
  std::nth_element (vals.begin (),
                    vals.begin () + static_cast<std::ptrdiff_t> (kth),
                    vals.end ());
  return nd_tensor<T> (a.shape (), std::move (vals));
}

template <typename T>
nd_tensor<std::size_t>
argpartition (const nd_tensor<T> &a, std::size_t kth)
{
  if (kth >= a.size ())
    {
      throw index_error ("argpartition kth index out of bounds");
    }
  std::vector<std::size_t> idx (a.size ());
  std::iota (idx.begin (), idx.end (), 0);
  std::nth_element (idx.begin (),
                    idx.begin () + static_cast<std::ptrdiff_t> (kth),
                    idx.end (),
                    [&] (std::size_t i, std::size_t j) {
                      return a.eval_at (i) < a.eval_at (j);
                    });
  return nd_tensor<std::size_t> (a.shape (), std::move (idx));
}

template <typename T>
nd_tensor<T>
take_along_axis (const nd_tensor<T> &arr,
                 const nd_tensor<std::size_t> &indices,
                 std::size_t axis = 0)
{
  if (arr.ndim () != indices.ndim () || axis >= arr.ndim ())
    {
      throw shape_error ("take_along_axis requires matching rank and valid axis");
    }
  if (arr.ndim () == 1)
    {
      nd_tensor<T> out (indices.shape ());
      for (std::size_t i = 0; i < indices.size (); ++i)
        {
          const std::size_t k = indices.eval_at (i);
          if (k >= arr.size ())
            {
              throw index_error ("take_along_axis index out of bounds");
            }
          out[i] = arr.eval_at (k);
        }
      return out;
    }
  if (arr.ndim () == 2)
    {
      const std::size_t r = indices.shape ()[0];
      const std::size_t c = indices.shape ()[1];
      nd_tensor<T> out (shape_type {r, c});
      for (std::size_t i = 0; i < r; ++i)
        {
          for (std::size_t j = 0; j < c; ++j)
            {
              const std::size_t k = indices (i, j);
              if (axis == 0)
                {
                  if (k >= arr.shape ()[0] || j >= arr.shape ()[1])
                    {
                      throw index_error ("take_along_axis index out of bounds");
                    }
                  out (i, j) = arr (k, j);
                }
              else
                {
                  if (i >= arr.shape ()[0] || k >= arr.shape ()[1])
                    {
                      throw index_error ("take_along_axis index out of bounds");
                    }
                  out (i, j) = arr (i, k);
                }
            }
        }
      return out;
    }
  throw shape_error ("take_along_axis currently supports 1D and 2D tensors");
}

template <typename T>
void
put_along_axis (nd_tensor<T> &arr,
                const nd_tensor<std::size_t> &indices,
                const nd_tensor<T> &values,
                std::size_t axis = 0)
{
  if (arr.ndim () != indices.ndim () || axis >= arr.ndim ())
    {
      throw shape_error ("put_along_axis requires matching rank and valid axis");
    }
  if (arr.ndim () == 1)
    {
      for (std::size_t i = 0; i < indices.size (); ++i)
        {
          const std::size_t k = indices.eval_at (i);
          if (k >= arr.size ())
            {
              throw index_error ("put_along_axis index out of bounds");
            }
          arr[k] = values.eval_at (i % values.size ());
        }
      return;
    }
  if (arr.ndim () == 2)
    {
      const std::size_t r = indices.shape ()[0];
      const std::size_t c = indices.shape ()[1];
      for (std::size_t i = 0; i < r; ++i)
        {
          for (std::size_t j = 0; j < c; ++j)
            {
              const std::size_t k = indices (i, j);
              const T val = values.eval_at ((i * c + j) % values.size ());
              if (axis == 0)
                {
                  if (k >= arr.shape ()[0] || j >= arr.shape ()[1])
                    {
                      throw index_error ("put_along_axis index out of bounds");
                    }
                  arr (k, j) = val;
                }
              else
                {
                  if (i >= arr.shape ()[0] || k >= arr.shape ()[1])
                    {
                      throw index_error ("put_along_axis index out of bounds");
                    }
                  arr (i, k) = val;
                }
            }
        }
      return;
    }
  throw shape_error ("put_along_axis currently supports 1D and 2D tensors");
}

inline nd_tensor<std::uint8_t>
packbits (const nd_tensor<std::uint8_t> &a)
{
  const std::size_t n_bits = a.size ();
  const std::size_t n_bytes = (n_bits + 7) / 8;
  nd_tensor<std::uint8_t> out (shape_type {n_bytes}, 0);
  for (std::size_t i = 0; i < n_bits; ++i)
    {
      if (a.eval_at (i) != 0)
        {
          out[i / 8] |= static_cast<std::uint8_t> (1u << (7 - (i % 8)));
        }
    }
  return out;
}

inline nd_tensor<std::uint8_t>
unpackbits (const nd_tensor<std::uint8_t> &a)
{
  const std::size_t n_bytes = a.size ();
  nd_tensor<std::uint8_t> out (shape_type {n_bytes * 8}, 0);
  for (std::size_t i = 0; i < n_bytes; ++i)
    {
      const std::uint8_t byte = a.eval_at (i);
      for (std::size_t b = 0; b < 8; ++b)
        {
          out[i * 8 + b] =
            static_cast<std::uint8_t> ((byte >> (7 - b)) & 1u);
        }
    }
  return out;
}

/* =========================================================================
   Orthogonal Polynomial Series (`chebval`, `chebfit`, `chebder`, `chebint`,
   `legval`, `legfit`, `legder`, `legint`, `hermval`, `hermfit`,
   `lagval`, `lagfit`) with rising-order coefficients c[0..deg].
   ========================================================================= */

namespace detail
{
inline std::vector<double>
solve_normal_equations (const std::vector<std::vector<double>> &v_mat,
                        const nd_tensor<double> &y)
{
  const std::size_t m = v_mat.size ();
  const std::size_t n = v_mat[0].size ();
  std::vector<std::vector<double>> ata (n, std::vector<double> (n + 1, 0.0));
  for (std::size_t r = 0; r < m; ++r)
    {
      const double yr = y.eval_at (r);
      for (std::size_t i = 0; i < n; ++i)
        {
          ata[i][n] += v_mat[r][i] * yr;
          for (std::size_t j = 0; j < n; ++j)
            {
              ata[i][j] += v_mat[r][i] * v_mat[r][j];
            }
        }
    }
  for (std::size_t k = 0; k < n; ++k)
    {
      std::size_t piv = k;
      double max_v = std::abs (ata[k][k]);
      for (std::size_t i = k + 1; i < n; ++i)
        {
          if (std::abs (ata[i][k]) > max_v)
            {
              max_v = std::abs (ata[i][k]);
              piv = i;
            }
        }
      if (piv != k)
        {
          std::swap (ata[k], ata[piv]);
        }
      const double diag = (std::abs (ata[k][k]) < 1e-15) ? 1e-15 : ata[k][k];
      for (std::size_t j = k; j <= n; ++j)
        {
          ata[k][j] /= diag;
        }
      for (std::size_t i = 0; i < n; ++i)
        {
          if (i != k)
            {
              const double f = ata[i][k];
              for (std::size_t j = k; j <= n; ++j)
                {
                  ata[i][j] -= f * ata[k][j];
                }
            }
        }
    }
  std::vector<double> c (n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    {
      c[i] = ata[i][n];
    }
  return c;
}
} // namespace detail

inline double
chebval (double x, const nd_tensor<double> &c)
{
  const std::size_t n = c.size ();
  if (n == 0)
    {
      return 0.0;
    }
  if (n == 1)
    {
      return c.eval_at (0);
    }
  double b_k1 = 0.0;
  double b_k2 = 0.0;
  for (std::size_t idx = n - 1; idx >= 1; --idx)
    {
      const double b_k = c.eval_at (idx) + 2.0 * x * b_k1 - b_k2;
      b_k2 = b_k1;
      b_k1 = b_k;
    }
  return c.eval_at (0) + x * b_k1 - b_k2;
}

inline nd_tensor<double>
chebval (const nd_tensor<double> &x, const nd_tensor<double> &c)
{
  nd_tensor<double> out (x.shape ());
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      out[i] = chebval (x.eval_at (i), c);
    }
  return out;
}

inline nd_tensor<double>
chebfit (const nd_tensor<double> &x, const nd_tensor<double> &y,
         std::size_t deg)
{
  if (x.size () != y.size () || x.size () == 0)
    {
      throw shape_error ("chebfit requires non-empty matching x and y");
    }
  const std::size_t m = x.size ();
  const std::size_t n = deg + 1;
  std::vector<std::vector<double>> v_mat (m, std::vector<double> (n, 1.0));
  for (std::size_t r = 0; r < m; ++r)
    {
      const double xv = x.eval_at (r);
      if (n > 1)
        {
          v_mat[r][1] = xv;
        }
      for (std::size_t k = 2; k < n; ++k)
        {
          v_mat[r][k] = 2.0 * xv * v_mat[r][k - 1] - v_mat[r][k - 2];
        }
    }
  return nd_tensor<double> (shape_type {n},
                            detail::solve_normal_equations (v_mat, y));
}

inline nd_tensor<double>
chebder (const nd_tensor<double> &c, std::size_t m = 1)
{
  std::vector<double> cur = c.to_vector ();
  for (std::size_t step = 0; step < m; ++step)
    {
      if (cur.size () <= 1)
        {
          cur = {0.0};
          break;
        }
      const std::size_t n = cur.size () - 1;
      std::vector<double> der (n, 0.0);
      for (std::size_t j = n; j >= 1; --j)
        {
          der[j - 1] = 2.0 * static_cast<double> (j) * cur[j];
          if (j + 1 < n)
            {
              der[j - 1] += der[j + 1];
            }
        }
      der[0] *= 0.5;
      cur = std::move (der);
    }
  return nd_tensor<double> (shape_type {cur.size ()}, std::move (cur));
}

inline nd_tensor<double>
chebint (const nd_tensor<double> &c, double k = 0.0)
{
  const std::size_t n = c.size ();
  std::vector<double> out (n + 1, 0.0);
  if (n >= 1)
    {
      out[1] = c.eval_at (0);
    }
  if (n >= 2)
    {
      out[2] = c.eval_at (1) / 4.0;
    }
  for (std::size_t j = 2; j < n; ++j)
    {
      const double cj = c.eval_at (j);
      out[j + 1] += cj / (2.0 * static_cast<double> (j + 1));
      out[j - 1] -= cj / (2.0 * static_cast<double> (j - 1));
    }
  double val_zero = 0.0;
  for (std::size_t j = 1; j <= n; ++j)
    {
      if (j % 2 == 0)
        {
          const double sign = ((j / 2) % 2 == 0) ? 1.0 : -1.0;
          val_zero += sign * out[j];
        }
    }
  out[0] = k - val_zero;
  return nd_tensor<double> (shape_type {out.size ()}, std::move (out));
}

inline double
legval (double x, const nd_tensor<double> &c)
{
  const std::size_t n = c.size ();
  if (n == 0)
    {
      return 0.0;
    }
  if (n == 1)
    {
      return c.eval_at (0);
    }
  double p_km1 = 1.0;
  double p_k = x;
  double sum = c.eval_at (0) + c.eval_at (1) * x;
  for (std::size_t k = 2; k < n; ++k)
    {
      const double kd = static_cast<double> (k);
      const double p_kp1 =
        ((2.0 * kd - 1.0) * x * p_k - (kd - 1.0) * p_km1) / kd;
      sum += c.eval_at (k) * p_kp1;
      p_km1 = p_k;
      p_k = p_kp1;
    }
  return sum;
}

inline nd_tensor<double>
legval (const nd_tensor<double> &x, const nd_tensor<double> &c)
{
  nd_tensor<double> out (x.shape ());
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      out[i] = legval (x.eval_at (i), c);
    }
  return out;
}

inline nd_tensor<double>
legfit (const nd_tensor<double> &x, const nd_tensor<double> &y,
        std::size_t deg)
{
  if (x.size () != y.size () || x.size () == 0)
    {
      throw shape_error ("legfit requires non-empty matching x and y");
    }
  const std::size_t m = x.size ();
  const std::size_t n = deg + 1;
  std::vector<std::vector<double>> v_mat (m, std::vector<double> (n, 1.0));
  for (std::size_t r = 0; r < m; ++r)
    {
      const double xv = x.eval_at (r);
      if (n > 1)
        {
          v_mat[r][1] = xv;
        }
      for (std::size_t k = 2; k < n; ++k)
        {
          const double kd = static_cast<double> (k);
          v_mat[r][k] =
            ((2.0 * kd - 1.0) * xv * v_mat[r][k - 1]
             - (kd - 1.0) * v_mat[r][k - 2]) / kd;
        }
    }
  return nd_tensor<double> (shape_type {n},
                            detail::solve_normal_equations (v_mat, y));
}

inline nd_tensor<double>
legder (const nd_tensor<double> &c, std::size_t m = 1)
{
  std::vector<double> cur = c.to_vector ();
  for (std::size_t step = 0; step < m; ++step)
    {
      if (cur.size () <= 1)
        {
          cur = {0.0};
          break;
        }
      const std::size_t n = cur.size () - 1;
      std::vector<double> der (n, 0.0);
      for (std::size_t j = n; j >= 1; --j)
        {
          for (std::size_t k = j - 1; ; k -= 2)
            {
              der[k] += (2.0 * static_cast<double> (k) + 1.0) * cur[j];
              if (k < 2)
                {
                  break;
                }
            }
        }
      cur = std::move (der);
    }
  return nd_tensor<double> (shape_type {cur.size ()}, std::move (cur));
}

inline nd_tensor<double>
legint (const nd_tensor<double> &c, double k = 0.0)
{
  const std::size_t n = c.size ();
  std::vector<double> out (n + 1, 0.0);
  if (n >= 1)
    {
      out[1] += c.eval_at (0);
    }
  for (std::size_t j = 1; j < n; ++j)
    {
      const double cj = c.eval_at (j);
      const double denom = 2.0 * static_cast<double> (j) + 1.0;
      out[j + 1] += cj / denom;
      out[j - 1] -= cj / denom;
    }
  nd_tensor<double> tmp (shape_type {out.size ()}, out);
  out[0] += k - legval (0.0, tmp);
  return nd_tensor<double> (shape_type {out.size ()}, std::move (out));
}

inline double
hermval (double x, const nd_tensor<double> &c)
{
  const std::size_t n = c.size ();
  if (n == 0)
    {
      return 0.0;
    }
  if (n == 1)
    {
      return c.eval_at (0);
    }
  double h_km1 = 1.0;
  double h_k = 2.0 * x;
  double sum = c.eval_at (0) + c.eval_at (1) * h_k;
  for (std::size_t k = 2; k < n; ++k)
    {
      const double h_kp1 =
        2.0 * x * h_k - 2.0 * static_cast<double> (k - 1) * h_km1;
      sum += c.eval_at (k) * h_kp1;
      h_km1 = h_k;
      h_k = h_kp1;
    }
  return sum;
}

inline nd_tensor<double>
hermval (const nd_tensor<double> &x, const nd_tensor<double> &c)
{
  nd_tensor<double> out (x.shape ());
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      out[i] = hermval (x.eval_at (i), c);
    }
  return out;
}

inline nd_tensor<double>
hermfit (const nd_tensor<double> &x, const nd_tensor<double> &y,
         std::size_t deg)
{
  if (x.size () != y.size () || x.size () == 0)
    {
      throw shape_error ("hermfit requires non-empty matching x and y");
    }
  const std::size_t m = x.size ();
  const std::size_t n = deg + 1;
  std::vector<std::vector<double>> v_mat (m, std::vector<double> (n, 1.0));
  for (std::size_t r = 0; r < m; ++r)
    {
      const double xv = x.eval_at (r);
      if (n > 1)
        {
          v_mat[r][1] = 2.0 * xv;
        }
      for (std::size_t k = 2; k < n; ++k)
        {
          v_mat[r][k] =
            2.0 * xv * v_mat[r][k - 1]
            - 2.0 * static_cast<double> (k - 1) * v_mat[r][k - 2];
        }
    }
  return nd_tensor<double> (shape_type {n},
                            detail::solve_normal_equations (v_mat, y));
}

inline double
lagval (double x, const nd_tensor<double> &c)
{
  const std::size_t n = c.size ();
  if (n == 0)
    {
      return 0.0;
    }
  if (n == 1)
    {
      return c.eval_at (0);
    }
  double l_km1 = 1.0;
  double l_k = 1.0 - x;
  double sum = c.eval_at (0) + c.eval_at (1) * l_k;
  for (std::size_t k = 2; k < n; ++k)
    {
      const double kd = static_cast<double> (k);
      const double l_kp1 =
        ((2.0 * kd - 1.0 - x) * l_k - (kd - 1.0) * l_km1) / kd;
      sum += c.eval_at (k) * l_kp1;
      l_km1 = l_k;
      l_k = l_kp1;
    }
  return sum;
}

inline nd_tensor<double>
lagval (const nd_tensor<double> &x, const nd_tensor<double> &c)
{
  nd_tensor<double> out (x.shape ());
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      out[i] = lagval (x.eval_at (i), c);
    }
  return out;
}

inline nd_tensor<double>
lagfit (const nd_tensor<double> &x, const nd_tensor<double> &y,
        std::size_t deg)
{
  if (x.size () != y.size () || x.size () == 0)
    {
      throw shape_error ("lagfit requires non-empty matching x and y");
    }
  const std::size_t m = x.size ();
  const std::size_t n = deg + 1;
  std::vector<std::vector<double>> v_mat (m, std::vector<double> (n, 1.0));
  for (std::size_t r = 0; r < m; ++r)
    {
      const double xv = x.eval_at (r);
      if (n > 1)
        {
          v_mat[r][1] = 1.0 - xv;
        }
      for (std::size_t k = 2; k < n; ++k)
        {
          const double kd = static_cast<double> (k);
          v_mat[r][k] =
            ((2.0 * kd - 1.0 - xv) * v_mat[r][k - 1]
             - (kd - 1.0) * v_mat[r][k - 2]) / kd;
        }
    }
  return nd_tensor<double> (shape_type {n},
                            detail::solve_normal_equations (v_mat, y));
}

/* =========================================================================
   Functional & N-D Windowing / Histogram Utilities:
   `apply_along_axis`, `sliding_window_view_2d`, `histogram_bin_edges`,
   and `histogramdd`.
   ========================================================================= */

template <typename T, typename Func>
nd_tensor<double>
apply_along_axis (Func &&fn, std::size_t axis, const nd_tensor<T> &arr)
{
  if (arr.ndim () != 2 || axis > 1)
    {
      throw shape_error ("apply_along_axis requires a 2D tensor and axis 0 or 1");
    }
  const std::size_t r = arr.shape ()[0];
  const std::size_t c = arr.shape ()[1];
  if (axis == 0)
    {
      nd_tensor<double> out (shape_type {c});
      for (std::size_t j = 0; j < c; ++j)
        {
          nd_tensor<T> col (shape_type {r});
          for (std::size_t i = 0; i < r; ++i)
            {
              col[i] = arr (i, j);
            }
          out[j] = static_cast<double> (fn (col));
        }
      return out;
    }
  nd_tensor<double> out (shape_type {r});
  for (std::size_t i = 0; i < r; ++i)
    {
      nd_tensor<T> row (shape_type {c});
      for (std::size_t j = 0; j < c; ++j)
        {
          row[j] = arr (i, j);
        }
      out[i] = static_cast<double> (fn (row));
    }
  return out;
}

template <typename T>
nd_tensor<T>
sliding_window_view_2d (const nd_tensor<T> &a,
                        std::size_t win_r, std::size_t win_c)
{
  if (a.ndim () != 2)
    {
      throw shape_error ("sliding_window_view_2d requires a 2D tensor");
    }
  const std::size_t r = a.shape ()[0];
  const std::size_t c = a.shape ()[1];
  if (win_r == 0 || win_c == 0 || win_r > r || win_c > c)
    {
      throw value_error ("sliding_window_view_2d window size invalid");
    }
  const std::size_t out_r = r - win_r + 1;
  const std::size_t out_c = c - win_c + 1;
  nd_tensor<T> out (shape_type {out_r, out_c, win_r, win_c});
  std::size_t idx = 0;
  for (std::size_t i = 0; i < out_r; ++i)
    {
      for (std::size_t j = 0; j < out_c; ++j)
        {
          for (std::size_t wi = 0; wi < win_r; ++wi)
            {
              for (std::size_t wj = 0; wj < win_c; ++wj)
                {
                  out[idx++] = a (i + wi, j + wj);
                }
            }
        }
    }
  return out;
}

inline nd_tensor<double>
histogram_bin_edges (const nd_tensor<double> &a,
                     const std::string &method = "auto")
{
  if (a.size () == 0)
    {
      return nd_tensor<double> (shape_type {2}, {0.0, 1.0});
    }
  const double lo = min (a);
  double hi = max (a);
  if (hi <= lo)
    {
      hi = lo + 1.0;
    }
  const double n = static_cast<double> (a.size ());
  std::size_t n_bins = 10;
  const std::size_t sturges_bins =
    static_cast<std::size_t> (std::ceil (std::log2 (n) + 1.0));
  const double iqr = percentile (a, 75.0) - percentile (a, 25.0);
  const double fd_width = 2.0 * iqr * std::pow (n, -1.0 / 3.0);
  const std::size_t fd_bins =
    (fd_width > 1e-12)
      ? static_cast<std::size_t> (std::ceil ((hi - lo) / fd_width))
      : sturges_bins;

  if (method == "auto")
    {
      n_bins = std::max (sturges_bins, fd_bins);
    }
  else if (method == "fd")
    {
      n_bins = std::max<std::size_t> (1, fd_bins);
    }
  else if (method == "sturges")
    {
      n_bins = std::max<std::size_t> (1, sturges_bins);
    }
  else if (method == "scott")
    {
      const double s = std_dev (a);
      const double w = 3.5 * s * std::pow (n, -1.0 / 3.0);
      n_bins = (w > 1e-12)
        ? static_cast<std::size_t> (std::ceil ((hi - lo) / w))
        : sturges_bins;
    }
  else if (method == "rice")
    {
      n_bins = static_cast<std::size_t> (std::ceil (2.0 * std::cbrt (n)));
    }
  else if (method == "sqrt")
    {
      n_bins = static_cast<std::size_t> (std::ceil (std::sqrt (n)));
    }
  else
    {
      throw value_error ("histogram_bin_edges unknown method: " + method);
    }
  n_bins = std::max<std::size_t> (1, n_bins);
  return linspace (lo, hi, n_bins + 1);
}

struct histogramdd_result
{
  nd_tensor<double> hist;
  std::vector<nd_tensor<double>> edges;
};

inline histogramdd_result
histogramdd (const nd_tensor<double> &sample,
             const std::vector<std::size_t> &bins)
{
  if (sample.ndim () != 2)
    {
      throw shape_error ("histogramdd requires a 2D (N, D) sample tensor");
    }
  const std::size_t n_pts = sample.shape ()[0];
  const std::size_t d_dim = sample.shape ()[1];
  if (bins.size () != d_dim || d_dim == 0)
    {
      throw value_error ("histogramdd bins size must match sample columns D");
    }
  std::vector<nd_tensor<double>> edges (d_dim, nd_tensor<double> (shape_type {2}));
  for (std::size_t d = 0; d < d_dim; ++d)
    {
      if (bins[d] == 0)
        {
          throw value_error ("histogramdd bin count must be positive");
        }
      double lo = sample (0, d);
      double hi = sample (0, d);
      for (std::size_t i = 1; i < n_pts; ++i)
        {
          lo = std::min (lo, sample (i, d));
          hi = std::max (hi, sample (i, d));
        }
      if (hi <= lo)
        {
          hi = lo + 1.0;
        }
      edges[d] = linspace (lo, hi, bins[d] + 1);
    }

  shape_type hist_shape (bins.begin (), bins.end ());
  nd_tensor<double> hist (hist_shape, 0.0);
  std::vector<std::size_t> strides (d_dim, 1);
  for (std::size_t d = d_dim - 1; d > 0; --d)
    {
      strides[d - 1] = strides[d] * bins[d];
    }

  for (std::size_t i = 0; i < n_pts; ++i)
    {
      std::size_t flat_idx = 0;
      for (std::size_t d = 0; d < d_dim; ++d)
        {
          const double v = sample (i, d);
          const double lo = edges[d].eval_at (0);
          const double hi = edges[d].eval_at (bins[d]);
          double ratio = (v - lo) / (hi - lo);
          std::size_t b =
            static_cast<std::size_t> (std::floor (ratio * static_cast<double> (bins[d])));
          if (b >= bins[d])
            {
              b = bins[d] - 1;
            }
          flat_idx += b * strides[d];
        }
      hist[flat_idx] += 1.0;
    }
  return {std::move (hist), std::move (edges)};
}

} // namespace hyper_math::tensor

#endif /* HYPER_MATH_TENSOR_MANIPULATION_HPP */
