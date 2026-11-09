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

#ifndef HYPER_MATH_SCIENTIFIC_INTERPOLATE_HPP
#define HYPER_MATH_SCIENTIFIC_INTERPOLATE_HPP

#include "hyper_math/scientific/linalg.hpp"
#include "hyper_math/tensor/nd_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace hyper_math::scientific::interpolate
{

using tensor::nd_tensor;
using tensor::shape_type;

/* Natural Cubic Spline Interpolator with O(N) Thomas tridiagonal construction,
   analytic first/second derivative evaluation, and exact piecewise integration
   (SciPy CubicSpline equivalent).  */
class cubic_spline
{
private:
  std::vector<double> x_;
  std::vector<double> a_;
  std::vector<double> b_;
  std::vector<double> c_;
  std::vector<double> d_;

  std::size_t
  find_segment (double xq) const noexcept
  {
    if (xq <= x_.front ())
      {
        return 0;
      }
    if (xq >= x_.back ())
      {
        return x_.size () - 2;
      }
    auto it = std::upper_bound (x_.begin (), x_.end (), xq);
    const std::size_t idx =
      static_cast<std::size_t> (std::distance (x_.begin (), it)) - 1;
    return std::min (idx, x_.size () - 2);
  }

public:
  cubic_spline (const nd_tensor<double> &x, const nd_tensor<double> &y)
    : x_ (x.to_vector ()),
      a_ (y.to_vector ()),
      b_ (),
      c_ (),
      d_ ()
  {
    const std::size_t n = x_.size ();
    if (n < 2 || a_.size () != n)
      {
        throw shape_error ("cubic_spline requires matching x, y with n >= 2");
      }
    for (std::size_t i = 0; i + 1 < n; ++i)
      {
        if (x_[i + 1] <= x_[i])
          {
            throw hyper_math_error ("cubic_spline x knots must be strictly increasing");
          }
      }

    const std::size_t m = n - 1;
    b_.resize (m, 0.0);
    c_.resize (n, 0.0);
    d_.resize (m, 0.0);

    std::vector<double> h (m);
    for (std::size_t i = 0; i < m; ++i)
      {
        h[i] = x_[i + 1] - x_[i];
      }

    std::vector<double> alpha (n, 0.0);
    for (std::size_t i = 1; i < m; ++i)
      {
        alpha[i] = (3.0 / h[i]) * (a_[i + 1] - a_[i])
                   - (3.0 / h[i - 1]) * (a_[i] - a_[i - 1]);
      }

    /* Solve tridiagonal system for natural spline second derivatives.  */
    std::vector<double> l (n, 1.0);
    std::vector<double> mu (n, 0.0);
    std::vector<double> z (n, 0.0);

    for (std::size_t i = 1; i < m; ++i)
      {
        l[i] = 2.0 * (x_[i + 1] - x_[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
      }

    for (std::ptrdiff_t j = static_cast<std::ptrdiff_t> (m) - 1; j >= 0; --j)
      {
        const std::size_t uj = static_cast<std::size_t> (j);
        c_[uj] = z[uj] - mu[uj] * c_[uj + 1];
        b_[uj] = (a_[uj + 1] - a_[uj]) / h[uj]
                 - h[uj] * (c_[uj + 1] + 2.0 * c_[uj]) / 3.0;
        d_[uj] = (c_[uj + 1] - c_[uj]) / (3.0 * h[uj]);
      }
  }

  double
  operator() (double xq) const noexcept
  {
    const std::size_t i = find_segment (xq);
    const double dx = xq - x_[i];
    return a_[i] + dx * (b_[i] + dx * (c_[i] + dx * d_[i]));
  }

  nd_tensor<double>
  operator() (const nd_tensor<double> &xq) const
  {
    nd_tensor<double> out (xq.shape ());
    for (std::size_t i = 0; i < xq.size (); ++i)
      {
        out[i] = (*this) (xq.eval_at (i));
      }
    return out;
  }

  double
  derivative (double xq, std::size_t order = 1) const
  {
    const std::size_t i = find_segment (xq);
    const double dx = xq - x_[i];
    if (order == 1)
      {
        return b_[i] + dx * (2.0 * c_[i] + 3.0 * d_[i] * dx);
      }
    else if (order == 2)
      {
        return 2.0 * c_[i] + 6.0 * d_[i] * dx;
      }
    else if (order == 3)
      {
        return 6.0 * d_[i];
      }
    return 0.0;
  }

  nd_tensor<double>
  derivative (const nd_tensor<double> &xq, std::size_t order = 1) const
  {
    nd_tensor<double> out (xq.shape ());
    for (std::size_t i = 0; i < xq.size (); ++i)
      {
        out[i] = derivative (xq.eval_at (i), order);
      }
    return out;
  }

  /* Exact analytical definite integral \int_a^b S(x) dx over the spline.  */
  double
  integrate (double a, double b) const noexcept
  {
    if (a == b)
      {
        return 0.0;
      }
    double sign = 1.0;
    if (a > b)
      {
        std::swap (a, b);
        sign = -1.0;
      }

    auto antideriv_seg = [this] (std::size_t i, double dx) noexcept {
      return dx
             * (a_[i]
                + dx
                    * (0.5 * b_[i]
                       + dx * ((c_[i] / 3.0) + 0.25 * d_[i] * dx)));
    };

    const std::size_t seg_a = find_segment (a);
    const std::size_t seg_b = find_segment (b);
    if (seg_a == seg_b)
      {
        return sign
               * (antideriv_seg (seg_a, b - x_[seg_a])
                  - antideriv_seg (seg_a, a - x_[seg_a]));
      }

    double total =
      antideriv_seg (seg_a, x_[seg_a + 1] - x_[seg_a])
      - antideriv_seg (seg_a, a - x_[seg_a]);
    for (std::size_t i = seg_a + 1; i < seg_b; ++i)
      {
        total += antideriv_seg (i, x_[i + 1] - x_[i]);
      }
    total += antideriv_seg (seg_b, b - x_[seg_b]);
    return sign * total;
  }
};

/* 1-D Linear and Nearest-Neighbor Interpolator (SciPy interp1d equivalent). */
inline nd_tensor<double>
interp1d (const nd_tensor<double> &x,
          const nd_tensor<double> &y,
          const nd_tensor<double> &xq,
          const std::string &kind = "linear")
{
  if (kind == "cubic")
    {
      cubic_spline cs (x, y);
      return cs (xq);
    }
  if (kind != "linear" && kind != "nearest")
    {
      throw hyper_math_error ("interp1d unknown kind: " + kind);
    }

  const std::vector<double> xv = x.to_vector ();
  const std::vector<double> yv = y.to_vector ();
  if (xv.size () < 2 || xv.size () != yv.size ())
    {
      throw shape_error ("interp1d requires matching x and y of size >= 2");
    }
  for (std::size_t i = 0; i + 1 < xv.size (); ++i)
    {
      if (xv[i + 1] <= xv[i])
        {
          throw hyper_math_error ("interp1d requires strictly increasing x");
        }
    }

  nd_tensor<double> out (xq.shape ());
  for (std::size_t k = 0; k < xq.size (); ++k)
    {
      const double q = xq.eval_at (k);
      if (q <= xv.front ())
        {
          out[k] = yv.front ();
          continue;
        }
      if (q >= xv.back ())
        {
          out[k] = yv.back ();
          continue;
        }

      auto it = std::upper_bound (xv.begin (), xv.end (), q);
      const std::size_t i =
        static_cast<std::size_t> (std::distance (xv.begin (), it)) - 1;
      const double t = (q - xv[i]) / (xv[i + 1] - xv[i]);

      if (kind == "nearest")
        {
          out[k] = (t < 0.5) ? yv[i] : yv[i + 1];
        }
      else
        {
          out[k] = yv[i] + t * (yv[i + 1] - yv[i]);
        }
    }
  return out;
}

/* Shape-Preserving Piecewise Cubic Hermite Interpolating Polynomial (PCHIP,
   SciPy PchipInterpolator equivalent). Uses the Fritsch-Carlson weighted
   harmonic mean derivative formula to preserve monotonicity and prevent
   overshoot near step changes.  */
class pchip_interpolator
{
private:
  std::vector<double> x_;
  std::vector<double> y_;
  std::vector<double> d_;

  static bool
  same_sign (double a, double b) noexcept
  {
    return (a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0);
  }

  static double
  endpoint_slope (double h0, double h1, double del0, double del1) noexcept
  {
    double d = ((2.0 * h0 + h1) * del0 - h0 * del1) / (h0 + h1);
    if (!same_sign (d, del0))
      {
        return 0.0;
      }
    if (!same_sign (del0, del1) && std::abs (d) > 3.0 * std::abs (del0))
      {
        d = 3.0 * del0;
      }
    return d;
  }

public:
  pchip_interpolator (const nd_tensor<double> &x, const nd_tensor<double> &y)
    : x_ (x.to_vector ()), y_ (y.to_vector ()), d_ ()
  {
    const std::size_t n = x_.size ();
    if (n < 2 || y_.size () != n)
      {
        throw shape_error ("pchip_interpolator requires matching x, y (n >= 2)");
      }
    const std::size_t m = n - 1;
    std::vector<double> h (m);
    std::vector<double> delta (m);
    for (std::size_t k = 0; k < m; ++k)
      {
        h[k] = x_[k + 1] - x_[k];
        if (h[k] <= 0.0)
          {
            throw hyper_math_error ("pchip_interpolator x must be strictly increasing");
          }
        delta[k] = (y_[k + 1] - y_[k]) / h[k];
      }

    d_.assign (n, 0.0);
    if (n == 2)
      {
        d_[0] = delta[0];
        d_[1] = delta[0];
        return;
      }

    for (std::size_t k = 1; k < m; ++k)
      {
        if (same_sign (delta[k - 1], delta[k]))
          {
            const double w1 = 2.0 * h[k] + h[k - 1];
            const double w2 = h[k] + 2.0 * h[k - 1];
            d_[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k]);
          }
        else
          {
            d_[k] = 0.0;
          }
      }

    d_[0] = endpoint_slope (h[0], h[1], delta[0], delta[1]);
    d_[n - 1] =
      endpoint_slope (h[m - 1], h[m - 2], delta[m - 1], delta[m - 2]);
  }

  double
  operator() (double xq) const noexcept
  {
    std::size_t k = 0;
    if (xq <= x_.front ())
      {
        k = 0;
      }
    else if (xq >= x_.back ())
      {
        k = x_.size () - 2;
      }
    else
      {
        auto it = std::upper_bound (x_.begin (), x_.end (), xq);
        k = static_cast<std::size_t> (std::distance (x_.begin (), it)) - 1;
      }

    const double hk = x_[k + 1] - x_[k];
    const double t = (xq - x_[k]) / hk;
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    const double h10 = t3 - 2.0 * t2 + t;
    const double h01 = -2.0 * t3 + 3.0 * t2;
    const double h11 = t3 - t2;
    return h00 * y_[k] + h10 * hk * d_[k]
           + h01 * y_[k + 1] + h11 * hk * d_[k + 1];
  }

  nd_tensor<double>
  operator() (const nd_tensor<double> &xq) const
  {
    nd_tensor<double> out (xq.shape ());
    for (std::size_t i = 0; i < xq.size (); ++i)
      {
        out[i] = (*this) (xq.eval_at (i));
      }
    return out;
  }
};

/* Second-Form Barycentric Lagrange Polynomial Interpolator
   (SciPy BarycentricInterpolator equivalent).  */
class barycentric_interpolator
{
private:
  std::vector<double> x_;
  std::vector<double> y_;
  std::vector<double> w_;

public:
  barycentric_interpolator (const nd_tensor<double> &x,
                            const nd_tensor<double> &y)
    : x_ (x.to_vector ()), y_ (y.to_vector ()), w_ (x.size (), 1.0)
  {
    const std::size_t n = x_.size ();
    if (n == 0 || y_.size () != n)
      {
        throw shape_error ("barycentric_interpolator requires matching x, y");
      }
    for (std::size_t j = 0; j < n; ++j)
      {
        double prod = 1.0;
        for (std::size_t k = 0; k < n; ++k)
          {
            if (k != j)
              {
                prod *= (x_[j] - x_[k]);
              }
          }
        if (std::abs (prod) == 0.0)
          {
            throw hyper_math_error (
              "barycentric_interpolator requires distinct x knots");
          }
        w_[j] = 1.0 / prod;
      }
  }

  double
  operator() (double xq) const noexcept
  {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t j = 0; j < x_.size (); ++j)
      {
        const double diff = xq - x_[j];
        if (std::abs (diff) < 1e-15)
          {
            return y_[j];
          }
        const double term = w_[j] / diff;
        num += term * y_[j];
        den += term;
      }
    return num / den;
  }

  nd_tensor<double>
  operator() (const nd_tensor<double> &xq) const
  {
    nd_tensor<double> out (xq.shape ());
    for (std::size_t i = 0; i < xq.size (); ++i)
      {
        out[i] = (*this) (xq.eval_at (i));
      }
    return out;
  }
};

/* Scattered N-Dimensional Radial Basis Function Interpolator
   (SciPy RBFInterpolator equivalent).  */
class rbf_interpolator
{
private:
  nd_tensor<double> centers_;
  nd_tensor<double> weights_;
  std::string kernel_;
  double epsilon_;

  double
  eval_kernel (double r) const noexcept
  {
    const double er = epsilon_ * r;
    if (kernel_ == "gaussian")
      {
        return std::exp (-er * er);
      }
    if (kernel_ == "inverse_multiquadric")
      {
        return 1.0 / std::sqrt (1.0 + er * er);
      }
    if (kernel_ == "linear")
      {
        return r;
      }
    /* Default: "multiquadric".  */
    return std::sqrt (1.0 + er * er);
  }

public:
  rbf_interpolator (const nd_tensor<double> &points,
                    const nd_tensor<double> &values,
                    std::string kernel = "multiquadric",
                    double epsilon = 1.0,
                    double smoothing = 1e-10)
    : centers_ (points.ndim () == 1
                ? points.reshape ({points.size (), 1})
                : points.copy ()),
      weights_ (),
      kernel_ (std::move (kernel)),
      epsilon_ (epsilon)
  {
    const std::size_t n = centers_.shape ()[0];
    const std::size_t d = centers_.shape ()[1];
    if (values.size () != n)
      {
        throw shape_error ("rbf_interpolator point count must match values");
      }

    nd_tensor<double> phi (shape_type {n, n}, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      {
        phi (i, i) = eval_kernel (0.0) + smoothing;
        for (std::size_t j = i + 1; j < n; ++j)
          {
            double dist_sq = 0.0;
            for (std::size_t k = 0; k < d; ++k)
              {
                const double diff = centers_ (i, k) - centers_ (j, k);
                dist_sq += diff * diff;
              }
            const double kij = eval_kernel (std::sqrt (dist_sq));
            phi (i, j) = kij;
            phi (j, i) = kij;
          }
      }
    weights_ = linalg::solve (phi, values);
  }

  nd_tensor<double>
  operator() (const nd_tensor<double> &query_pts) const
  {
    const nd_tensor<double> q =
      (query_pts.ndim () == 1)
      ? query_pts.reshape ({query_pts.size (), 1})
      : query_pts;
    const std::size_t m = q.shape ()[0];
    const std::size_t n = centers_.shape ()[0];
    const std::size_t d = centers_.shape ()[1];
    if (q.shape ()[1] != d)
      {
        throw shape_error ("rbf_interpolator query dimension mismatch");
      }

    nd_tensor<double> out (shape_type {m}, 0.0);
    for (std::size_t i = 0; i < m; ++i)
      {
        double acc = 0.0;
        for (std::size_t j = 0; j < n; ++j)
          {
            double dist_sq = 0.0;
            for (std::size_t k = 0; k < d; ++k)
              {
                const double diff = q (i, k) - centers_ (j, k);
                dist_sq += diff * diff;
              }
            acc += weights_[j] * eval_kernel (std::sqrt (dist_sq));
          }
        out[i] = acc;
      }
    return out;
  }
};

/* 2-D Rectilinear Grid Interpolator supporting "linear" (bilinear) and
   "nearest" interpolation (SciPy interpolate.RegularGridInterpolator).  */
class regular_grid_interpolator_2d
{
private:
  std::vector<double> x_;
  std::vector<double> y_;
  nd_tensor<double> values_;
  std::string method_;

  static std::size_t
  find_interval (const std::vector<double> &grid, double q) noexcept
  {
    if (q <= grid.front ())
      {
        return 0;
      }
    if (q >= grid.back ())
      {
        return grid.size () - 2;
      }
    auto it = std::upper_bound (grid.begin (), grid.end (), q);
    return static_cast<std::size_t> (std::distance (grid.begin (), it)) - 1;
  }

public:
  regular_grid_interpolator_2d (const nd_tensor<double> &x,
                                const nd_tensor<double> &y,
                                const nd_tensor<double> &values,
                                std::string method = "linear")
    : x_ (x.to_vector ()),
      y_ (y.to_vector ()),
      values_ (values.copy ()),
      method_ (std::move (method))
  {
    if (x_.size () < 2 || y_.size () < 2)
      {
        throw shape_error ("regular_grid_interpolator_2d requires grid sizes >= 2");
      }
    if (values_.ndim () != 2
        || values_.shape ()[0] != x_.size ()
        || values_.shape ()[1] != y_.size ())
      {
        throw shape_error ("regular_grid_interpolator_2d values shape mismatch");
      }
  }

  double
  operator() (double xq, double yq) const noexcept
  {
    const double xc = std::clamp (xq, x_.front (), x_.back ());
    const double yc = std::clamp (yq, y_.front (), y_.back ());
    const std::size_t ix = find_interval (x_, xc);
    const std::size_t iy = find_interval (y_, yc);
    const double tx = (xc - x_[ix]) / (x_[ix + 1] - x_[ix]);
    const double ty = (yc - y_[iy]) / (y_[iy + 1] - y_[iy]);

    if (method_ == "nearest")
      {
        const std::size_t rx = (tx < 0.5) ? ix : (ix + 1);
        const std::size_t ry = (ty < 0.5) ? iy : (iy + 1);
        return values_ (rx, ry);
      }

    /* Bilinear interpolation on cell [x_ix, x_{ix+1}] x [y_iy, y_{iy+1}].  */
    const double v00 = values_ (ix, iy);
    const double v10 = values_ (ix + 1, iy);
    const double v01 = values_ (ix, iy + 1);
    const double v11 = values_ (ix + 1, iy + 1);
    return (1.0 - tx) * (1.0 - ty) * v00
           + tx * (1.0 - ty) * v10
           + (1.0 - tx) * ty * v01
           + tx * ty * v11;
  }

  nd_tensor<double>
  operator() (const nd_tensor<double> &query_pts) const
  {
    if (query_pts.ndim () != 2 || query_pts.shape ()[1] != 2)
      {
        throw shape_error ("regular_grid_interpolator_2d expects (M, 2) query points");
      }
    const std::size_t m = query_pts.shape ()[0];
    nd_tensor<double> out (shape_type {m});
    for (std::size_t i = 0; i < m; ++i)
      {
        out[i] = (*this) (query_pts (i, 0), query_pts (i, 1));
      }
    return out;
  }
};

/* Akima 1-D Oscillation-Suppressing C^1 Piecewise Cubic Interpolator
   `akima1d_interpolator`, 2-D Separable Bicubic Spline `rect_bivariate_spline`,
   and 2-D Scattered Data Interpolator `griddata`
   (SciPy interpolate equivalents).  */
class akima1d_interpolator
{
private:
  std::vector<double> x_;
  std::vector<double> y_;
  std::vector<double> t_;

public:
  akima1d_interpolator (const nd_tensor<double> &x, const nd_tensor<double> &y)
  {
    const std::size_t n = x.size ();
    if (n < 2 || y.size () != n)
      {
        throw shape_error ("akima1d_interpolator requires equal x, y of size >= 2");
      }
    x_ = x.to_vector ();
    y_ = y.to_vector ();
    for (std::size_t i = 0; i + 1 < n; ++i)
      {
        if (x_[i + 1] <= x_[i])
          {
            throw hyper_math_error ("akima1d_interpolator requires strictly increasing x");
          }
      }

    /* Padded secant slopes m[0 .. n + 2], where m[k + 2] = (y[k+1]-y[k])/(x[k+1]-x[k]). */
    std::vector<double> m (n + 3, 0.0);
    for (std::size_t k = 0; k + 1 < n; ++k)
      {
        m[k + 2] = (y_[k + 1] - y_[k]) / (x_[k + 1] - x_[k]);
      }
    if (n >= 3)
      {
        m[1] = 2.0 * m[2] - m[3];
        m[0] = 2.0 * m[1] - m[2];
        m[n + 1] = 2.0 * m[n] - m[n - 1];
        m[n + 2] = 2.0 * m[n + 1] - m[n];
      }
    else
      {
        m[1] = m[2];
        m[0] = m[2];
        m[3] = m[2];
        m[4] = m[2];
      }

    t_.resize (n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      {
        const double w1 = std::abs (m[i + 3] - m[i + 2]);
        const double w2 = std::abs (m[i + 1] - m[i]);
        if (w1 + w2 > 1e-15)
          {
            t_[i] = (w1 * m[i + 1] + w2 * m[i + 2]) / (w1 + w2);
          }
        else
          {
            t_[i] = 0.5 * (m[i + 1] + m[i + 2]);
          }
      }
  }

  double
  operator() (double xq) const noexcept
  {
    const std::size_t n = x_.size ();
    std::size_t k = 0;
    if (xq <= x_.front ())
      {
        k = 0;
      }
    else if (xq >= x_.back ())
      {
        k = n - 2;
      }
    else
      {
        auto it = std::upper_bound (x_.begin (), x_.end (), xq);
        k = static_cast<std::size_t> (std::distance (x_.begin (), it) - 1);
      }
    const double h = x_[k + 1] - x_[k];
    const double dx = xq - x_[k];
    const double m_k = (y_[k + 1] - y_[k]) / h;
    const double c2 = (3.0 * m_k - 2.0 * t_[k] - t_[k + 1]) / h;
    const double c3 = (t_[k] + t_[k + 1] - 2.0 * m_k) / (h * h);
    return y_[k] + dx * (t_[k] + dx * (c2 + dx * c3));
  }

  nd_tensor<double>
  operator() (const nd_tensor<double> &xq) const
  {
    nd_tensor<double> out (xq.shape ());
    for (std::size_t i = 0; i < xq.size (); ++i)
      {
        out[i] = (*this) (xq.eval_at (i));
      }
    return out;
  }
};

class rect_bivariate_spline
{
private:
  nd_tensor<double> x_;
  nd_tensor<double> y_;
  std::vector<cubic_spline> row_splines_;

public:
  rect_bivariate_spline (const nd_tensor<double> &x,
                         const nd_tensor<double> &y,
                         const nd_tensor<double> &z)
    : x_ (x), y_ (y), row_splines_ ()
  {
    if (z.ndim () != 2
        || z.shape ()[0] != x.size ()
        || z.shape ()[1] != y.size ()
        || x.size () < 2 || y.size () < 2)
      {
        throw shape_error ("rect_bivariate_spline requires z of shape (|x|, |y|) with |x|,|y| >= 2");
      }
    const std::size_t nx = x.size ();
    const std::size_t ny = y.size ();
    row_splines_.reserve (nx);
    for (std::size_t i = 0; i < nx; ++i)
      {
        nd_tensor<double> z_row (shape_type {ny});
        for (std::size_t j = 0; j < ny; ++j)
          {
            z_row[j] = z (i, j);
          }
        row_splines_.emplace_back (y_, z_row);
      }
  }

  double
  operator() (double xq, double yq) const
  {
    const std::size_t nx = x_.size ();
    nd_tensor<double> col_vals (shape_type {nx});
    for (std::size_t i = 0; i < nx; ++i)
      {
        col_vals[i] = row_splines_[i] (yq);
      }
    const cubic_spline col_sp (x_, col_vals);
    return col_sp (xq);
  }
};

inline nd_tensor<double>
griddata (const nd_tensor<double> &points,
          const nd_tensor<double> &values,
          const nd_tensor<double> &xi,
          const std::string &method = "linear")
{
  if (points.ndim () != 2 || xi.ndim () != 2
      || points.shape ()[1] != xi.shape ()[1]
      || points.shape ()[0] != values.size ()
      || points.shape ()[0] == 0)
    {
      throw shape_error ("griddata requires matching point dimensions and non-empty points");
    }
  const std::size_t n_pts = points.shape ()[0];
  const std::size_t dim = points.shape ()[1];
  const std::size_t m_q = xi.shape ()[0];

  if (method == "nearest")
    {
      nd_tensor<double> out (shape_type {m_q});
      for (std::size_t q = 0; q < m_q; ++q)
        {
          double best_d2 = constants::inf;
          std::size_t best_i = 0;
          for (std::size_t i = 0; i < n_pts; ++i)
            {
              double d2 = 0.0;
              for (std::size_t d = 0; d < dim; ++d)
                {
                  const double diff = xi (q, d) - points (i, d);
                  d2 += diff * diff;
                }
              if (d2 < best_d2)
                {
                  best_d2 = d2;
                  best_i = i;
                }
            }
          out[q] = values.eval_at (best_i);
        }
      return out;
    }
  if (method == "linear")
    {
      nd_tensor<double> out (shape_type {m_q});
      for (std::size_t q = 0; q < m_q; ++q)
        {
          /* Check if query point coincides with a sample point.  */
          bool exact_hit = false;
          for (std::size_t i = 0; i < n_pts; ++i)
            {
              double d2 = 0.0;
              for (std::size_t d = 0; d < dim; ++d)
                {
                  const double diff = xi (q, d) - points (i, d);
                  d2 += diff * diff;
                }
              if (d2 < 1e-24)
                {
                  out[q] = values.eval_at (i);
                  exact_hit = true;
                  break;
                }
            }
          if (exact_hit)
            {
              continue;
            }

          /* Inverse-distance weighted local affine plane fit:
             min_c sum_i w_i (c_0 + sum_d c_{d+1} (p_{i,d} - x_{q,d}) - v_i)^2.
             Exact for all linear functions and C^0 interpolating at nodes.  */
          const std::size_t p_dim = dim + 1;
          nd_tensor<double> ata (shape_type {p_dim, p_dim}, 0.0);
          nd_tensor<double> atb (shape_type {p_dim}, 0.0);
          for (std::size_t i = 0; i < n_pts; ++i)
            {
              double d2 = 0.0;
              for (std::size_t d = 0; d < dim; ++d)
                {
                  const double diff = points (i, d) - xi (q, d);
                  d2 += diff * diff;
                }
              const double w = 1.0 / (d2 + 1e-12);
              std::vector<double> row (p_dim, 1.0);
              for (std::size_t d = 0; d < dim; ++d)
                {
                  row[d + 1] = points (i, d) - xi (q, d);
                }
              const double vi = values.eval_at (i);
              for (std::size_t r = 0; r < p_dim; ++r)
                {
                  atb[r] += w * row[r] * vi;
                  for (std::size_t c = 0; c < p_dim; ++c)
                    {
                      ata (r, c) += w * row[r] * row[c];
                    }
                }
            }
          for (std::size_t d = 1; d < p_dim; ++d)
            {
              ata (d, d) += 1e-12;
            }
          const nd_tensor<double> coeff = linalg::solve (ata, atb);
          out[q] = coeff[0];
        }
      return out;
    }
  if (method == "cubic")
    {
      const rbf_interpolator rbf (points, values, "thin_plate_spline", 1.0);
      return rbf (xi);
    }
  throw hyper_math_error ("unknown griddata method: " + method);
}

/* =========================================================================
   Arbitrary-Degree B-Splines (`bspline`, `make_interp_spline`,
   `make_lsq_spline`), 2D Smooth Scattered Interpolator
   (`clough_tocher_2d_interpolator`), Polynomial Interpolators
   (`lagrange_interpolator`, `krogh_interpolator`), and Padé Rational
   Approximation (`pade`).
   ========================================================================= */

class bspline
{
private:
  std::vector<double> t_;
  std::vector<double> c_;
  std::size_t k_;

  static double
  basis_eval (const std::vector<double> &t, std::size_t i, std::size_t k,
              double x) noexcept
  {
    if (k == 0)
      {
        const bool last_interval =
          (i + 2 == t.size () || t[i + 1] == t.back ());
        if (last_interval)
          {
            return (x >= t[i] && x <= t[i + 1]) ? 1.0 : 0.0;
          }
        return (x >= t[i] && x < t[i + 1]) ? 1.0 : 0.0;
      }
    double val = 0.0;
    const double d1 = t[i + k] - t[i];
    if (d1 > 1e-15)
      {
        val += ((x - t[i]) / d1) * basis_eval (t, i, k - 1, x);
      }
    const double d2 = t[i + k + 1] - t[i + 1];
    if (d2 > 1e-15)
      {
        val += ((t[i + k + 1] - x) / d2) * basis_eval (t, i + 1, k - 1, x);
      }
    return val;
  }

public:
  bspline (const nd_tensor<double> &t, const nd_tensor<double> &c,
           std::size_t k)
    : t_ (t.to_vector ()), c_ (c.to_vector ()), k_ (k)
  {
    if (t_.size () < c_.size () + k_ + 1 || c_.empty ())
      {
        throw shape_error ("bspline requires len(t) >= len(c) + k + 1");
      }
  }

  double
  operator() (double x) const noexcept
  {
    const double xc = std::clamp (x, t_[k_], t_[c_.size ()]);
    double sum = 0.0;
    for (std::size_t i = 0; i < c_.size (); ++i)
      {
        sum += c_[i] * basis_eval (t_, i, k_, xc);
      }
    return sum;
  }

  nd_tensor<double>
  operator() (const nd_tensor<double> &x) const
  {
    nd_tensor<double> out (x.shape ());
    for (std::size_t i = 0; i < x.size (); ++i)
      {
        out[i] = (*this) (x.eval_at (i));
      }
    return out;
  }

  bspline
  derivative () const
  {
    if (k_ == 0 || c_.size () <= 1)
      {
        return bspline (
          nd_tensor<double> (shape_type {t_.size ()}, t_),
          nd_tensor<double> (shape_type {c_.size ()}, 0.0),
          0);
      }
    const std::size_t n_new = c_.size () - 1;
    std::vector<double> c_new (n_new, 0.0);
    for (std::size_t i = 0; i < n_new; ++i)
      {
        const double dt = t_[i + k_ + 1] - t_[i + 1];
        c_new[i] =
          (dt > 1e-15) ? (static_cast<double> (k_) * (c_[i + 1] - c_[i]) / dt)
                       : 0.0;
      }
    std::vector<double> t_new (t_.begin () + 1, t_.end () - 1);
    return bspline (
      nd_tensor<double> (shape_type {t_new.size ()}, std::move (t_new)),
      nd_tensor<double> (shape_type {c_new.size ()}, std::move (c_new)),
      k_ - 1);
  }
};

inline bspline
make_interp_spline (const nd_tensor<double> &x, const nd_tensor<double> &y,
                    std::size_t k = 3)
{
  const std::size_t n = x.size ();
  if (n != y.size () || n <= k)
    {
      throw shape_error ("make_interp_spline requires len(x) == len(y) > k");
    }
  std::vector<double> t (n + k + 1, 0.0);
  for (std::size_t i = 0; i <= k; ++i)
    {
      t[i] = x.eval_at (0);
      t[n + i] = x.eval_at (n - 1);
    }
  for (std::size_t j = 1; j < n - k; ++j)
    {
      double s = 0.0;
      for (std::size_t r = j; r < j + k; ++r)
        {
          s += x.eval_at (r);
        }
      t[j + k] = s / static_cast<double> (k);
    }
  nd_tensor<double> t_ten (shape_type {t.size ()}, t);
  nd_tensor<double> a_mat (shape_type {n, n}, 0.0);
  for (std::size_t r = 0; r < n; ++r)
    {
      for (std::size_t c = 0; c < n; ++c)
        {
          std::vector<double> unit_c (n, 0.0);
          unit_c[c] = 1.0;
          bspline basis_c (t_ten, nd_tensor<double> (shape_type {n}, unit_c), k);
          a_mat (r, c) = basis_c (x.eval_at (r));
        }
    }
  const nd_tensor<double> coeffs = linalg::solve (a_mat, y);
  return bspline (t_ten, coeffs, k);
}

inline bspline
make_lsq_spline (const nd_tensor<double> &x, const nd_tensor<double> &y,
                 const nd_tensor<double> &t, std::size_t k = 3)
{
  const std::size_t m = x.size ();
  if (m != y.size () || t.size () <= k + 1)
    {
      throw shape_error ("make_lsq_spline invalid inputs");
    }
  const std::size_t n = t.size () - k - 1;
  nd_tensor<double> a_mat (shape_type {m, n}, 0.0);
  for (std::size_t c = 0; c < n; ++c)
    {
      std::vector<double> unit_c (n, 0.0);
      unit_c[c] = 1.0;
      bspline basis_c (t, nd_tensor<double> (shape_type {n}, unit_c), k);
      for (std::size_t r = 0; r < m; ++r)
        {
          a_mat (r, c) = basis_c (x.eval_at (r));
        }
    }
  const nd_tensor<double> coeffs = linalg::lstsq (a_mat, y);
  return bspline (t, coeffs, k);
}

class clough_tocher_2d_interpolator
{
private:
  rbf_interpolator rbf_;

public:
  clough_tocher_2d_interpolator (const nd_tensor<double> &points,
                                 const nd_tensor<double> &values)
    : rbf_ (points, values, "thin_plate_spline", 1.0)
  {
  }

  double
  operator() (double xq, double yq) const
  {
    nd_tensor<double> q (shape_type {1, 2}, {xq, yq});
    return rbf_ (q)[0];
  }

  nd_tensor<double>
  operator() (const nd_tensor<double> &xi) const
  {
    return rbf_ (xi);
  }
};

class lagrange_interpolator
{
private:
  std::vector<double> x_;
  std::vector<double> y_;

public:
  lagrange_interpolator (const nd_tensor<double> &x, const nd_tensor<double> &y)
    : x_ (x.to_vector ()), y_ (y.to_vector ())
  {
    if (x_.empty () || x_.size () != y_.size ())
      {
        throw shape_error ("lagrange_interpolator requires non-empty matching x, y");
      }
  }

  double
  operator() (double xq) const noexcept
  {
    double sum = 0.0;
    const std::size_t n = x_.size ();
    for (std::size_t i = 0; i < n; ++i)
      {
        double li = 1.0;
        for (std::size_t j = 0; j < n; ++j)
          {
            if (i != j)
              {
                li *= (xq - x_[j]) / (x_[i] - x_[j]);
              }
          }
        sum += y_[i] * li;
      }
    return sum;
  }

  nd_tensor<double>
  operator() (const nd_tensor<double> &xq) const
  {
    nd_tensor<double> out (xq.shape ());
    for (std::size_t i = 0; i < xq.size (); ++i)
      {
        out[i] = (*this) (xq.eval_at (i));
      }
    return out;
  }
};

class krogh_interpolator
{
private:
  std::vector<double> x_;
  std::vector<double> c_;

public:
  krogh_interpolator (const nd_tensor<double> &x, const nd_tensor<double> &y)
    : x_ (x.to_vector ()), c_ (y.to_vector ())
  {
    if (x_.empty () || x_.size () != c_.size ())
      {
        throw shape_error ("krogh_interpolator requires non-empty matching x, y");
      }
    const std::size_t n = x_.size ();
    for (std::size_t j = 1; j < n; ++j)
      {
        for (std::size_t i = n - 1; i >= j; --i)
          {
            c_[i] = (c_[i] - c_[i - 1]) / (x_[i] - x_[i - j]);
          }
      }
  }

  double
  operator() (double xq) const noexcept
  {
    const std::size_t n = x_.size ();
    double p = c_[n - 1];
    for (std::size_t k = n - 1; k > 0; --k)
      {
        p = c_[k - 1] + (xq - x_[k - 1]) * p;
      }
    return p;
  }
};

struct pade_result
{
  nd_tensor<double> p;
  nd_tensor<double> q;
};

inline pade_result
pade (const nd_tensor<double> &an, std::size_t m, std::size_t n = 0)
{
  if (n == 0)
    {
      if (an.size () <= m + 1)
        {
          throw value_error ("pade requires len(an) >= m + n + 1");
        }
      n = an.size () - 1 - m;
    }
  if (an.size () < m + n + 1)
    {
      throw value_error ("pade requires len(an) >= m + n + 1");
    }
  nd_tensor<double> q_coeffs (shape_type {n + 1}, 0.0);
  q_coeffs[n] = 1.0;
  if (n > 0)
    {
      nd_tensor<double> mat (shape_type {n, n}, 0.0);
      nd_tensor<double> rhs (shape_type {n}, 0.0);
      for (std::size_t r = 0; r < n; ++r)
        {
          const std::size_t k = m + 1 + r;
          rhs[r] = -an.eval_at (k);
          for (std::size_t c = 0; c < n; ++c)
            {
              const std::size_t j = c + 1;
              mat (r, c) = (k >= j) ? an.eval_at (k - j) : 0.0;
            }
        }
      const nd_tensor<double> b_vec = linalg::solve (mat, rhs);
      for (std::size_t j = 1; j <= n; ++j)
        {
          q_coeffs[n - j] = b_vec[j - 1];
        }
    }
  nd_tensor<double> p_coeffs (shape_type {m + 1}, 0.0);
  for (std::size_t k = 0; k <= m; ++k)
    {
      double sum = an.eval_at (k);
      for (std::size_t j = 1; j <= std::min (k, n); ++j)
        {
          sum += q_coeffs[n - j] * an.eval_at (k - j);
        }
      p_coeffs[m - k] = sum;
    }
  return {std::move (p_coeffs), std::move (q_coeffs)};
}

} // namespace hyper_math::scientific::interpolate

#endif /* HYPER_MATH_SCIENTIFIC_INTERPOLATE_HPP */
