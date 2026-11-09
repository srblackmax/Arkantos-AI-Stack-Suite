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

#ifndef HYPER_MATH_SCIENTIFIC_INTEGRATE_HPP
#define HYPER_MATH_SCIENTIFIC_INTEGRATE_HPP

#include "hyper_math/scientific/linalg.hpp"
#include "hyper_math/tensor/ufunc.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <vector>

namespace hyper_math::scientific::integrate
{

using tensor::nd_tensor;
using tensor::shape_type;

struct quad_result
{
  double value;
  double error_estimate;
  std::size_t evaluations;
};

namespace detail
{

  /* 7-point Gauss / 15-point Kronrod nodes and weights on [-1, 1].  */
  inline constexpr std::array<double, 8> gk15_nodes = {
    0.00000000000000000000,
    0.20778495500789846760,
    0.40584515137739716691,
    0.58608723546769113029,
    0.74153118559939443986,
    0.86486442335976907279,
    0.94910791234275852453,
    0.99145537112081263921
  };

  inline constexpr std::array<double, 8> gk15_weights = {
    0.20948214108472782801,
    0.20443294007529889241,
    0.19035057806478540991,
    0.16900472663926790283,
    0.14065325971552591875,
    0.10479001032225018384,
    0.06309209262997855329,
    0.02293532201052922496
  };

  inline constexpr std::array<double, 4> g7_weights = {
    0.41795918367346938776,
    0.38183005050511894495,
    0.27970539148927666790,
    0.12948496616886969327
  };

  template <typename Func>
  quad_result
  gk15_panel (Func &&f, double a, double b)
  {
    const double center = 0.5 * (a + b);
    const double half_len = 0.5 * (b - a);
    const double f_center = f (center);

    double res_k15 = f_center * gk15_weights[0];
    double res_g7 = f_center * g7_weights[0];

    for (std::size_t j = 1; j < 8; ++j)
      {
        const double dx = half_len * gk15_nodes[j];
        const double f_sum = f (center - dx) + f (center + dx);
        res_k15 += f_sum * gk15_weights[j];
        if ((j & 1) == 0)
          {
            res_g7 += f_sum * g7_weights[j / 2];
          }
      }

    res_k15 *= half_len;
    res_g7 *= half_len;
    const double err = std::abs (res_k15 - res_g7);
    return quad_result {res_k15, err, 15};
  }

  template <typename Func>
  quad_result
  adaptive_gk15 (Func &&f,
                 double a,
                 double b,
                 double epsabs,
                 double epsrel,
                 std::size_t depth,
                 std::size_t max_depth)
  {
    const quad_result panel = gk15_panel (f, a, b);
    const double tol =
      std::max (epsabs, epsrel * std::abs (panel.value));
    if (panel.error_estimate <= tol || depth >= max_depth)
      {
        return panel;
      }

    const double mid = 0.5 * (a + b);
    const quad_result left =
      adaptive_gk15 (f, a, mid, epsabs * 0.5, epsrel, depth + 1, max_depth);
    const quad_result right =
      adaptive_gk15 (f, mid, b, epsabs * 0.5, epsrel, depth + 1, max_depth);

    return quad_result {
      left.value + right.value,
      left.error_estimate + right.error_estimate,
      panel.evaluations + left.evaluations + right.evaluations
    };
  }

} // namespace detail

/* Adaptive Gauss-Kronrod G7-K15 numerical integration on [a, b].  */
template <typename Func>
quad_result
quad (Func &&f,
      double a,
      double b,
      double epsabs = 1e-10,
      double epsrel = 1e-10,
      std::size_t limit = 20)
{
  return detail::adaptive_gk15 (std::forward<Func> (f),
                                a, b, epsabs, epsrel, 0, limit);
}

/* Composite Trapezoidal rule over sampled 1D arrays.  */
inline double
trapezoid (const nd_tensor<double> &y, double dx = 1.0)
{
  if (y.size () < 2)
    {
      return 0.0;
    }
  double total = 0.5 * (y.eval_at (0) + y.eval_at (y.size () - 1));
  for (std::size_t i = 1; i + 1 < y.size (); ++i)
    {
      total += y.eval_at (i);
    }
  return total * dx;
}

inline double
trapezoid (const nd_tensor<double> &y, const nd_tensor<double> &x)
{
  if (y.size () != x.size ())
    {
      throw shape_error ("trapezoid y and x size mismatch");
    }
  double total = 0.0;
  for (std::size_t i = 0; i + 1 < y.size (); ++i)
    {
      total += 0.5 * (y.eval_at (i) + y.eval_at (i + 1))
               * (x.eval_at (i + 1) - x.eval_at (i));
    }
  return total;
}

/* Composite Simpson's 1/3 + 3/8 rule over uniformly spaced samples.  */
inline double
simpson (const nd_tensor<double> &y, double dx = 1.0)
{
  const std::size_t n = y.size ();
  if (n < 2)
    {
      return 0.0;
    }
  if (n == 2)
    {
      return 0.5 * dx * (y.eval_at (0) + y.eval_at (1));
    }

  const std::size_t end_simpson = (n % 2 == 1) ? (n - 1) : (n - 4);
  double s = 0.0;
  if (end_simpson > 0)
    {
      s = y.eval_at (0) + y.eval_at (end_simpson);
      for (std::size_t i = 1; i < end_simpson; ++i)
        {
          s += (i % 2 == 1 ? 4.0 : 2.0) * y.eval_at (i);
        }
      s *= (dx / 3.0);
    }

  if (n % 2 == 0)
    {
      s += (3.0 * dx / 8.0)
           * (y.eval_at (n - 4) + 3.0 * y.eval_at (n - 3)
              + 3.0 * y.eval_at (n - 2) + y.eval_at (n - 1));
    }
  return s;
}

/* Romberg Integration via Richardson Extrapolation over composite trapezoidal
   refinements (SciPy integrate.romberg equivalent).  */
template <typename Func>
double
romberg (Func &&f,
         double a,
         double b,
         double tol = 1e-11,
         std::size_t divmax = 12)
{
  std::vector<double> r_prev (divmax + 1, 0.0);
  std::vector<double> r_curr (divmax + 1, 0.0);
  double h = b - a;
  r_prev[0] = 0.5 * h * (f (a) + f (b));

  for (std::size_t i = 1; i <= divmax; ++i)
    {
      h *= 0.5;
      double sum_mid = 0.0;
      const std::size_t num_new_pts = static_cast<std::size_t> (1) << (i - 1);
      for (std::size_t k = 1; k <= num_new_pts; ++k)
        {
          sum_mid += f (a + static_cast<double> (2 * k - 1) * h);
        }
      r_curr[0] = 0.5 * r_prev[0] + h * sum_mid;

      double pow4 = 1.0;
      for (std::size_t m = 1; m <= i; ++m)
        {
          pow4 *= 4.0;
          r_curr[m] =
            r_curr[m - 1] + (r_curr[m - 1] - r_prev[m - 1]) / (pow4 - 1.0);
        }

      if (i >= 2 && std::abs (r_curr[i] - r_prev[i - 1]) <= tol)
        {
          return r_curr[i];
        }
      std::swap (r_prev, r_curr);
    }

  return r_prev[divmax];
}

/* Cumulative trapezoidal integration (SciPy cumulative_trapezoid).  */
template <typename Derived, typename T>
nd_tensor<double>
cumulative_trapezoid (const tensor::tensor_expr_base<Derived, T> &y,
                      double dx = 1.0)
{
  const std::size_t n = y.size ();
  if (n < 2)
    {
      return nd_tensor<double> (shape_type {0});
    }
  nd_tensor<double> out (shape_type {n - 1});
  double acc = 0.0;
  for (std::size_t i = 0; i + 1 < n; ++i)
    {
      acc += 0.5 * dx * (static_cast<double> (y.eval_at (i))
                         + static_cast<double> (y.eval_at (i + 1)));
      out[i] = acc;
    }
  return out;
}

/* Adaptive Dormand-Prince RK4(5) Initial Value Problem ODE Solver
   (SciPy solve_ivp equivalent).
   Solves dy/dt = f(t, y) for t in [t0, tf] starting from y(t0) = y0.  */
struct ivp_solution
{
  nd_tensor<double> t;
  nd_tensor<double> y; /* Shape: (state_dim, num_steps).  */
  std::size_t nfev;
  bool success;
};

template <typename OdeFunc>
ivp_solution
solve_ivp (OdeFunc &&fun,
           double t0,
           double tf,
           const nd_tensor<double> &y0,
           double rtol = 1e-6,
           double atol = 1e-9,
           double max_step = constants::inf)
{
  const std::size_t dim = y0.size ();
  if (dim == 0)
    {
      throw shape_error ("solve_ivp requires non-empty initial state y0");
    }
  const double direction = (tf >= t0) ? 1.0 : -1.0;
  double t = t0;
  nd_tensor<double> y = y0.copy ();

  std::vector<double> t_hist {t0};
  std::vector<std::vector<double>> y_hist {y.to_vector ()};

  double h = std::min (std::abs (tf - t0) * 0.02, max_step);
  if (h <= 0.0)
    {
      h = 1e-4;
    }

  std::size_t nfev = 0;
  const std::size_t max_iters = 100000;

  /* Dormand-Prince RK4(5) Butcher tableau coefficients.  */
  for (std::size_t iter = 0; iter < max_iters; ++iter)
    {
      const double rem = std::abs (tf - t);
      if (rem <= 1e-14)
        {
          break;
        }
      if (h > rem)
        {
          h = rem;
        }
      const double dt = direction * h;

      const nd_tensor<double> k1 = fun (t, y);
      const nd_tensor<double> k2 =
        fun (t + 0.2 * dt, nd_tensor<double> (y + (0.2 * dt) * k1));
      const nd_tensor<double> k3 =
        fun (t + 0.3 * dt,
             nd_tensor<double> (y + (3.0 / 40.0 * dt) * k1
                                + (9.0 / 40.0 * dt) * k2));
      const nd_tensor<double> k4 =
        fun (t + 0.8 * dt,
             nd_tensor<double> (y + (44.0 / 45.0 * dt) * k1
                                - (56.0 / 15.0 * dt) * k2
                                + (32.0 / 9.0 * dt) * k3));
      const nd_tensor<double> k5 =
        fun (t + (8.0 / 9.0) * dt,
             nd_tensor<double> (y + (19372.0 / 6561.0 * dt) * k1
                                - (25360.0 / 2187.0 * dt) * k2
                                + (64448.0 / 6561.0 * dt) * k3
                                - (212.0 / 729.0 * dt) * k4));
      const nd_tensor<double> k6 =
        fun (t + dt,
             nd_tensor<double> (y + (9017.0 / 3168.0 * dt) * k1
                                - (355.0 / 33.0 * dt) * k2
                                + (46732.0 / 5247.0 * dt) * k3
                                + (49.0 / 176.0 * dt) * k4
                                - (5103.0 / 18656.0 * dt) * k5));

      /* 5th-order solution candidate.  */
      const nd_tensor<double> y_next =
        y + dt * ((35.0 / 384.0) * k1
                  + (500.0 / 1113.0) * k3
                  + (125.0 / 192.0) * k4
                  - (2187.0 / 6784.0) * k5
                  + (11.0 / 84.0) * k6);

      const nd_tensor<double> k7 = fun (t + dt, y_next);
      nfev += 7;

      /* 4th-order embedded error estimate.  */
      const nd_tensor<double> err_vec =
        dt * ((71.0 / 57600.0) * k1
              - (71.0 / 16695.0) * k3
              + (71.0 / 1920.0) * k4
              - (17253.0 / 339200.0) * k5
              + (22.0 / 525.0) * k6
              - (1.0 / 40.0) * k7);

      double err_norm = 0.0;
      for (std::size_t i = 0; i < dim; ++i)
        {
          const double sc =
            atol + rtol * std::max (std::abs (y[i]), std::abs (y_next[i]));
          const double ratio = err_vec[i] / sc;
          err_norm += ratio * ratio;
        }
      err_norm = std::sqrt (err_norm / static_cast<double> (dim));

      if (err_norm <= 1.0)
        {
          t += dt;
          y = y_next;
          t_hist.push_back (t);
          y_hist.push_back (y.to_vector ());
        }

      const double factor =
        (err_norm == 0.0)
        ? 5.0
        : std::clamp (0.9 * std::pow (err_norm, -0.2), 0.2, 5.0);
      h = std::min (h * factor, max_step);
    }

  const std::size_t n_steps = t_hist.size ();
  nd_tensor<double> t_out (shape_type {n_steps}, t_hist);
  nd_tensor<double> y_out (shape_type {dim, n_steps});
  for (std::size_t s = 0; s < n_steps; ++s)
    {
      for (std::size_t d = 0; d < dim; ++d)
        {
          y_out (d, s) = y_hist[s][d];
        }
    }

  return ivp_solution {
    std::move (t_out),
    std::move (y_out),
    nfev,
    std::abs (tf - t) <= 1e-9
  };
}

/* 2-D Adaptive Gauss-Kronrod Double Integral (SciPy integrate.dblquad):
   Computes \int_{a}^{b} \int_{gfun(x)}^{hfun(x)} func(y, x) dy dx.
   Note: Following SciPy convention, func takes (y, x) where y is the inner
   variable and x is the outer variable.  */
template <typename Func, typename GFunc, typename HFunc>
quad_result
dblquad (Func &&func,
         double a,
         double b,
         GFunc &&gfun,
         HFunc &&hfun,
         double epsabs = 1e-8,
         double epsrel = 1e-8)
{
  std::size_t total_evals = 0;
  auto outer_integrand = [&] (double x) -> double {
    const double y_lo = gfun (x);
    const double y_hi = hfun (x);
    auto inner_res = quad (
      [&] (double y) { return func (y, x); },
      y_lo,
      y_hi,
      epsabs,
      epsrel);
    total_evals += inner_res.evaluations;
    return inner_res.value;
  };
  auto res = quad (outer_integrand, a, b, epsabs, epsrel);
  res.evaluations += total_evals;
  return res;
}

/* Golub-Welsch n-point Gauss-Legendre nodes and weights on [-1, 1] (`leggauss`)
   and Fixed-Order Gauss-Legendre Quadrature (`fixed_quad`)
   (NumPy polynomial.legendre.leggauss & SciPy integrate.fixed_quad).  */
inline std::pair<nd_tensor<double>, nd_tensor<double>>
leggauss (std::size_t n)
{
  if (n == 0)
    {
      throw shape_error ("leggauss requires degree n >= 1");
    }
  if (n == 1)
    {
      return {nd_tensor<double> {0.0}, nd_tensor<double> {2.0}};
    }
  /* Construct symmetric Jacobi tridiagonal matrix J where
     J_{i, i+1} = J_{i+1, i} = (i + 1) / sqrt(4 * (i + 1)^2 - 1).  */
  nd_tensor<double> jac (shape_type {n, n}, 0.0);
  for (std::size_t i = 0; i + 1 < n; ++i)
    {
      const double k = static_cast<double> (i + 1);
      const double beta = k / std::sqrt (4.0 * k * k - 1.0);
      jac (i, i + 1) = beta;
      jac (i + 1, i) = beta;
    }
  const auto eg = linalg::eigh (jac);
  nd_tensor<double> weights (shape_type {n});
  for (std::size_t j = 0; j < n; ++j)
    {
      const double v0 = eg.eigenvectors (0, j);
      weights[j] = 2.0 * v0 * v0;
    }
  return {eg.eigenvalues, std::move (weights)};
}

template <typename Func>
quad_result
fixed_quad (Func &&func, double a, double b, std::size_t n = 5)
{
  const auto [nodes, weights] = leggauss (n);
  const double half_span = 0.5 * (b - a);
  const double mid = 0.5 * (a + b);
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      sum += weights[i] * func (mid + half_span * nodes[i]);
    }
  return quad_result {half_span * sum, 0.0, n};
}

/* Romberg integration of sampled 1-D data `y` of length `2^k + 1` with
   uniform step `dx` via Richardson extrapolation (SciPy integrate.romb).  */
inline double
romb (const nd_tensor<double> &y, double dx = 1.0)
{
  const std::size_t n = y.size ();
  if (n < 2 || ((n - 1) & (n - 2)) != 0)
    {
      throw shape_error ("romb requires y.size() == 2^k + 1 with k >= 0");
    }
  std::size_t k_max = 0;
  std::size_t intervals = n - 1;
  while (intervals > 1)
    {
      intervals >>= 1;
      ++k_max;
    }

  std::vector<std::vector<double>> r (k_max + 1,
                                      std::vector<double> (k_max + 1, 0.0));
  for (std::size_t k = 0; k <= k_max; ++k)
    {
      const std::size_t step = static_cast<std::size_t> (1) << (k_max - k);
      const double h = dx * static_cast<double> (step);
      double sum = 0.5 * (y.eval_at (0) + y.eval_at (n - 1));
      for (std::size_t i = step; i < n - 1; i += step)
        {
          sum += y.eval_at (i);
        }
      r[k][0] = h * sum;
      double four_m = 4.0;
      for (std::size_t m = 1; m <= k; ++m)
        {
          r[k][m] = r[k][m - 1]
                    + (r[k][m - 1] - r[k - 1][m - 1]) / (four_m - 1.0);
          four_m *= 4.0;
        }
    }
  return r[k_max][k_max];
}

/* 3-D Adaptive Gauss-Kronrod Triple Integral `tplquad`, Implicit BDF-2 Stiff
   ODE Solver `solve_ivp_bdf`, and Two-Point Boundary Value Problem Collocation
   Solver `solve_bvp` (SciPy integrate equivalents).  */
template <typename Func, typename GFunc, typename HFunc,
          typename QFunc, typename RFunc>
quad_result
tplquad (Func &&func,
         double a,
         double b,
         GFunc &&gfun,
         HFunc &&hfun,
         QFunc &&qfun,
         RFunc &&rfun,
         double epsabs = 1e-6,
         double epsrel = 1e-6)
{
  std::size_t total_evals = 0;
  auto outer_x = [&] (double x) -> double {
    const auto inner_yz = dblquad (
      [&] (double z, double y) { return func (z, y, x); },
      gfun (x),
      hfun (x),
      [&] (double y) { return qfun (x, y); },
      [&] (double y) { return rfun (x, y); },
      epsabs,
      epsrel);
    total_evals += inner_yz.evaluations;
    return inner_yz.value;
  };
  auto res = quad (outer_x, a, b, epsabs, epsrel);
  res.evaluations += total_evals;
  return res;
}

template <typename OdeFunc>
ivp_solution
solve_ivp_bdf (OdeFunc &&fun,
               double t0,
               double tf,
               const nd_tensor<double> &y0,
               std::size_t n_steps = 200,
               double newton_tol = 1e-10)
{
  const std::size_t dim = y0.size ();
  if (dim == 0 || n_steps == 0)
    {
      throw shape_error ("solve_ivp_bdf requires non-empty y0 and n_steps >= 1");
    }
  const double dt = (tf - t0) / static_cast<double> (n_steps);
  nd_tensor<double> t_out (shape_type {n_steps + 1});
  nd_tensor<double> y_out (shape_type {dim, n_steps + 1});
  for (std::size_t s = 0; s <= n_steps; ++s)
    {
      t_out[s] = t0 + static_cast<double> (s) * dt;
    }
  for (std::size_t d = 0; d < dim; ++d)
    {
      y_out (d, 0) = y0[d];
    }

  std::size_t nfev = 0;
  nd_tensor<double> y_prev2 = y0.copy ();
  nd_tensor<double> y_prev1 = y0.copy ();

  for (std::size_t step = 1; step <= n_steps; ++step)
    {
      const double t_next = t_out[step];
      nd_tensor<double> y_guess = y_prev1.copy ();

      /* Step 1 uses implicit Backward Euler (BDF-1); steps >= 2 use BDF-2:
         y_{n+1} - (4/3) y_n + (1/3) y_{n-1} = (2/3) dt f(t_{n+1}, y_{n+1}). */
      const double beta = (step == 1) ? dt : (2.0 / 3.0) * dt;
      nd_tensor<double> rhs_hist (shape_type {dim});
      for (std::size_t d = 0; d < dim; ++d)
        {
          rhs_hist[d] =
            (step == 1)
            ? y_prev1[d]
            : ((4.0 / 3.0) * y_prev1[d] - (1.0 / 3.0) * y_prev2[d]);
        }

      for (int nr_it = 0; nr_it < 15; ++nr_it)
        {
          const nd_tensor<double> f_val = fun (t_next, y_guess);
          ++nfev;
          nd_tensor<double> res (shape_type {dim});
          double max_res = 0.0;
          for (std::size_t d = 0; d < dim; ++d)
            {
              res[d] = y_guess[d] - rhs_hist[d] - beta * f_val[d];
              max_res = std::max (max_res, std::abs (res[d]));
            }
          if (max_res <= newton_tol)
            {
              break;
            }

          /* Finite-difference Jacobian J = I - beta * df/dy.  */
          nd_tensor<double> jac = nd_tensor<double>::eye (dim);
          for (std::size_t j = 0; j < dim; ++j)
            {
              const double eps = 1e-8 * std::max (1.0, std::abs (y_guess[j]));
              nd_tensor<double> y_p = y_guess.copy ();
              y_p[j] += eps;
              const nd_tensor<double> f_p = fun (t_next, y_p);
              ++nfev;
              for (std::size_t i = 0; i < dim; ++i)
                {
                  jac (i, j) -= beta * (f_p[i] - f_val[i]) / eps;
                }
            }

          const nd_tensor<double> dy = linalg::solve (jac, res);
          double max_dy = 0.0;
          for (std::size_t d = 0; d < dim; ++d)
            {
              y_guess[d] -= dy[d];
              max_dy = std::max (max_dy, std::abs (dy[d]));
            }
          if (max_dy <= newton_tol)
            {
              break;
            }
        }

      y_prev2 = y_prev1;
      y_prev1 = y_guess;
      for (std::size_t d = 0; d < dim; ++d)
        {
          y_out (d, step) = y_guess[d];
        }
    }

  return ivp_solution {std::move (t_out), std::move (y_out), nfev, true};
}

/* Two-point Boundary Value Problem collocation solver:
   Solves dy/dx = fun(x, y) on mesh `x` of length M with boundary condition
   residual `bc(y_a, y_b) == 0` of length D, starting from initial guess `y_init`
   of shape (D, M) (SciPy integrate.solve_bvp equivalent).  */
template <typename OdeFunc, typename BcFunc>
nd_tensor<double>
solve_bvp (OdeFunc &&fun,
           BcFunc &&bc,
           const nd_tensor<double> &x,
           const nd_tensor<double> &y_init,
           std::size_t max_iter = 40,
           double tol = 1e-8)
{
  const std::size_t m = x.size ();
  if (m < 2 || y_init.ndim () != 2 || y_init.shape ()[1] != m)
    {
      throw shape_error ("solve_bvp requires x of length M >= 2 and y_init of shape (D, M)");
    }
  const std::size_t d_dim = y_init.shape ()[0];
  const std::size_t n_total = d_dim * m;

  auto pack_state = [&] (const nd_tensor<double> &y_mat) {
    nd_tensor<double> u (shape_type {n_total});
    for (std::size_t k = 0; k < m; ++k)
      {
        for (std::size_t d = 0; d < d_dim; ++d)
          {
            u[k * d_dim + d] = y_mat (d, k);
          }
      }
    return u;
  };

  auto unpack_state = [&] (const nd_tensor<double> &u) {
    nd_tensor<double> y_mat (shape_type {d_dim, m});
    for (std::size_t k = 0; k < m; ++k)
      {
        for (std::size_t d = 0; d < d_dim; ++d)
          {
            y_mat (d, k) = u[k * d_dim + d];
          }
      }
    return y_mat;
  };

  auto eval_residual = [&] (const nd_tensor<double> &u) {
    nd_tensor<double> res (shape_type {n_total}, 0.0);
    nd_tensor<double> ya (shape_type {d_dim});
    nd_tensor<double> yb (shape_type {d_dim});
    for (std::size_t d = 0; d < d_dim; ++d)
      {
        ya[d] = u[d];
        yb[d] = u[(m - 1) * d_dim + d];
      }
    const nd_tensor<double> bc_res = bc (ya, yb);
    for (std::size_t d = 0; d < d_dim; ++d)
      {
        res[d] = bc_res[d];
      }

    nd_tensor<double> yk (shape_type {d_dim});
    nd_tensor<double> yk1 (shape_type {d_dim});
    for (std::size_t k = 0; k + 1 < m; ++k)
      {
        const double h = x[k + 1] - x[k];
        for (std::size_t d = 0; d < d_dim; ++d)
          {
            yk[d] = u[k * d_dim + d];
            yk1[d] = u[(k + 1) * d_dim + d];
          }
        const nd_tensor<double> fk = fun (x[k], yk);
        const nd_tensor<double> fk1 = fun (x[k + 1], yk1);
        for (std::size_t d = 0; d < d_dim; ++d)
          {
            res[(k + 1) * d_dim + d] =
              yk1[d] - yk[d] - 0.5 * h * (fk[d] + fk1[d]);
          }
      }
    return res;
  };

  nd_tensor<double> u = pack_state (y_init);
  for (std::size_t it = 0; it < max_iter; ++it)
    {
      const nd_tensor<double> r0 = eval_residual (u);
      double max_r = 0.0;
      for (std::size_t i = 0; i < n_total; ++i)
        {
          max_r = std::max (max_r, std::abs (r0[i]));
        }
      if (max_r <= tol)
        {
          break;
        }

      nd_tensor<double> jac (shape_type {n_total, n_total}, 0.0);
      for (std::size_t j = 0; j < n_total; ++j)
        {
          const double eps = 1e-7 * std::max (1.0, std::abs (u[j]));
          nd_tensor<double> up = u.copy ();
          up[j] += eps;
          const nd_tensor<double> rp = eval_residual (up);
          for (std::size_t i = 0; i < n_total; ++i)
            {
              jac (i, j) = (rp[i] - r0[i]) / eps;
            }
        }

      const nd_tensor<double> du = linalg::solve (jac, r0);
      for (std::size_t i = 0; i < n_total; ++i)
        {
          u[i] -= du[i];
        }
    }

  return unpack_state (u);
}

/* Arbitrary N-D Recursive Adaptive Gauss-Kronrod Quadrature `nquad` and
   Closed Newton-Cotes Weights `newton_cotes` (SciPy integrate equivalents).  */
namespace detail
{
template <typename Func>
quad_result
nquad_impl (Func &func,
            const std::vector<std::pair<double, double>> &ranges,
            std::size_t dim_idx,
            std::vector<double> &coords,
            double epsabs,
            double epsrel)
{
  if (dim_idx + 1 == ranges.size ())
    {
      auto leaf_fn = [&] (double x_last) -> double {
        coords[dim_idx] = x_last;
        return func (nd_tensor<double> (shape_type {coords.size ()}, coords));
      };
      return quad (leaf_fn, ranges[dim_idx].first, ranges[dim_idx].second,
                   epsabs, epsrel);
    }

  std::size_t sub_evals = 0;
  auto slice_fn = [&] (double x_cur) -> double {
    coords[dim_idx] = x_cur;
    auto sub_res =
      nquad_impl (func, ranges, dim_idx + 1, coords, epsabs, epsrel);
    sub_evals += sub_res.evaluations;
    return sub_res.value;
  };
  auto res = quad (slice_fn, ranges[dim_idx].first, ranges[dim_idx].second,
                   epsabs, epsrel);
  res.evaluations += sub_evals;
  return res;
}
} // namespace detail

template <typename Func>
quad_result
nquad (Func &&func,
       const std::vector<std::pair<double, double>> &ranges,
       double epsabs = 1e-8,
       double epsrel = 1e-8)
{
  if (ranges.empty ())
    {
      throw shape_error ("nquad requires at least one integration range");
    }
  std::vector<double> coords (ranges.size (), 0.0);
  return detail::nquad_impl (func, ranges, 0, coords, epsabs, epsrel);
}

inline nd_tensor<double>
newton_cotes (std::size_t rn)
{
  if (rn == 0)
    {
      throw shape_error ("newton_cotes requires order rn >= 1");
    }
  /* Construct Vandermonde system sum_{j=0}^{rn} w_j * j^k = rn^{k+1} / (k + 1)
     so that sum_j w_j f(x_0 + j h) * h integrates polynomials of degree <= rn
     on [x_0, x_0 + rn * h] exactly.  */
  const std::size_t n = rn + 1;
  nd_tensor<double> v_mat (shape_type {n, n}, 0.0);
  nd_tensor<double> rhs (shape_type {n}, 0.0);
  const double drn = static_cast<double> (rn);
  for (std::size_t k = 0; k < n; ++k)
    {
      const double dk = static_cast<double> (k);
      rhs[k] = std::pow (drn, dk + 1.0) / (dk + 1.0);
      for (std::size_t j = 0; j < n; ++j)
        {
          v_mat (k, j) =
            (k == 0) ? 1.0 : std::pow (static_cast<double> (j), dk);
        }
    }
  return linalg::solve (v_mat, rhs);
}

/* =========================================================================
   Bogacki-Shampine Adaptive RK2(3) (`solve_ivp_rk23`), Implicit Radau IIA
   Stiff Solver (`solve_ivp_radau`), Composite Cumulative Simpson
   (`cumulative_simpson`), and N-D Quasi-Monte Carlo Quadrature (`qmc_quad`).
   ========================================================================= */

template <typename OdeFunc>
ivp_solution
solve_ivp_rk23 (OdeFunc &&fun,
                double t0,
                double tf,
                const nd_tensor<double> &y0,
                std::size_t num_steps = 100)
{
  if (num_steps == 0 || y0.ndim () != 1)
    {
      throw value_error ("solve_ivp_rk23 requires 1D y0 and num_steps > 0");
    }
  const std::size_t dim = y0.size ();
  const double h = (tf - t0) / static_cast<double> (num_steps);
  nd_tensor<double> t_out (shape_type {num_steps + 1});
  nd_tensor<double> y_out (shape_type {dim, num_steps + 1});
  t_out[0] = t0;
  nd_tensor<double> y_cur = y0.copy ();
  for (std::size_t d = 0; d < dim; ++d)
    {
      y_out (d, 0) = y_cur[d];
    }
  std::size_t nfev = 0;
  for (std::size_t step = 1; step <= num_steps; ++step)
    {
      const double t = t0 + static_cast<double> (step - 1) * h;
      const nd_tensor<double> k1 = fun (t, y_cur);
      nd_tensor<double> y2 (shape_type {dim});
      for (std::size_t d = 0; d < dim; ++d)
        {
          y2[d] = y_cur[d] + 0.5 * h * k1[d];
        }
      const nd_tensor<double> k2 = fun (t + 0.5 * h, y2);
      nd_tensor<double> y3 (shape_type {dim});
      for (std::size_t d = 0; d < dim; ++d)
        {
          y3[d] = y_cur[d] + 0.75 * h * k2[d];
        }
      const nd_tensor<double> k3 = fun (t + 0.75 * h, y3);
      nfev += 3;
      for (std::size_t d = 0; d < dim; ++d)
        {
          y_cur[d] += h * ((2.0 / 9.0) * k1[d]
                           + (1.0 / 3.0) * k2[d]
                           + (4.0 / 9.0) * k3[d]);
          y_out (d, step) = y_cur[d];
        }
      t_out[step] = t0 + static_cast<double> (step) * h;
    }
  return {std::move (t_out), std::move (y_out), nfev, true};
}

template <typename OdeFunc>
ivp_solution
solve_ivp_radau (OdeFunc &&fun,
                 double t0,
                 double tf,
                 const nd_tensor<double> &y0,
                 std::size_t num_steps = 100)
{
  if (num_steps == 0 || y0.ndim () != 1)
    {
      throw value_error ("solve_ivp_radau requires 1D y0 and num_steps > 0");
    }
  const std::size_t dim = y0.size ();
  const double h = (tf - t0) / static_cast<double> (num_steps);
  nd_tensor<double> t_out (shape_type {num_steps + 1});
  nd_tensor<double> y_out (shape_type {dim, num_steps + 1});
  t_out[0] = t0;
  nd_tensor<double> y_cur = y0.copy ();
  for (std::size_t d = 0; d < dim; ++d)
    {
      y_out (d, 0) = y_cur[d];
    }
  std::size_t nfev = 0;
  /* 2-stage Radau IIA (order 3 L-stable implicit Runge-Kutta):
     c1 = 1/3, c2 = 1;
     A = [[5/12, -1/12], [3/4, 1/4]].  */
  for (std::size_t step = 1; step <= num_steps; ++step)
    {
      const double t = t0 + static_cast<double> (step - 1) * h;
      const double t1 = t + h / 3.0;
      const double t2 = t + h;
      nd_tensor<double> z1 = y_cur.copy ();
      nd_tensor<double> z2 = y_cur.copy ();
      for (int it = 0; it < 8; ++it)
        {
          const nd_tensor<double> f1 = fun (t1, z1);
          const nd_tensor<double> f2 = fun (t2, z2);
          nfev += 2;
          double max_diff = 0.0;
          for (std::size_t d = 0; d < dim; ++d)
            {
              const double nz1 =
                y_cur[d] + h * ((5.0 / 12.0) * f1[d] - (1.0 / 12.0) * f2[d]);
              const double nz2 =
                y_cur[d] + h * (0.75 * f1[d] + 0.25 * f2[d]);
              max_diff = std::max (max_diff, std::abs (nz1 - z1[d]));
              max_diff = std::max (max_diff, std::abs (nz2 - z2[d]));
              z1[d] = 0.5 * z1[d] + 0.5 * nz1;
              z2[d] = 0.5 * z2[d] + 0.5 * nz2;
            }
          if (max_diff < 1e-11)
            {
              break;
            }
        }
      /* Refine final implicit stage at t2 via damped Newton step.  */
      for (int it = 0; it < 4; ++it)
        {
          const nd_tensor<double> f1 = fun (t1, z1);
          const nd_tensor<double> f2 = fun (t2, z2);
          nfev += 2;
          nd_tensor<double> res (shape_type {dim});
          for (std::size_t d = 0; d < dim; ++d)
            {
              res[d] = z2[d] - y_cur[d] - h * (0.75 * f1[d] + 0.25 * f2[d]);
            }
          nd_tensor<double> jac = nd_tensor<double>::eye (dim);
          for (std::size_t j = 0; j < dim; ++j)
            {
              const double eps = 1e-8 * std::max (1.0, std::abs (z2[j]));
              nd_tensor<double> zp = z2.copy ();
              zp[j] += eps;
              const nd_tensor<double> fp = fun (t2, zp);
              ++nfev;
              for (std::size_t i = 0; i < dim; ++i)
                {
                  jac (i, j) -= 0.25 * h * (fp[i] - f2[i]) / eps;
                }
            }
          const nd_tensor<double> dz = linalg::solve (jac, res);
          double max_dz = 0.0;
          for (std::size_t d = 0; d < dim; ++d)
            {
              z2[d] -= dz[d];
              max_dz = std::max (max_dz, std::abs (dz[d]));
            }
          if (max_dz < 1e-11)
            {
              break;
            }
        }
      y_cur = z2;
      for (std::size_t d = 0; d < dim; ++d)
        {
          y_out (d, step) = y_cur[d];
        }
      t_out[step] = t2;
    }
  return {std::move (t_out), std::move (y_out), nfev, true};
}

inline nd_tensor<double>
cumulative_simpson (const nd_tensor<double> &y, double dx = 1.0,
                    double initial = 0.0)
{
  const std::size_t n = y.size ();
  if (n < 2)
    {
      throw value_error ("cumulative_simpson requires at least 2 samples");
    }
  nd_tensor<double> out (shape_type {n}, 0.0);
  out[0] = initial;
  if (n == 2)
    {
      out[1] = initial + 0.5 * dx * (y.eval_at (0) + y.eval_at (1));
      return out;
    }
  out[1] = initial + (dx / 12.0) * (5.0 * y.eval_at (0)
                                    + 8.0 * y.eval_at (1)
                                    - y.eval_at (2));
  for (std::size_t i = 2; i < n; ++i)
    {
      out[i] = out[i - 2]
               + (dx / 3.0) * (y.eval_at (i - 2)
                               + 4.0 * y.eval_at (i - 1)
                               + y.eval_at (i));
    }
  return out;
}

template <typename Func>
quad_result
qmc_quad (Func &&func,
          const nd_tensor<double> &a,
          const nd_tensor<double> &b,
          std::size_t n_points = 1024)
{
  if (a.size () == 0 || a.size () != b.size () || n_points == 0)
    {
      throw value_error ("qmc_quad requires non-empty matching bounds a and b");
    }
  const std::size_t dim = a.size ();
  static const std::size_t primes[16] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53
  };
  double vol = 1.0;
  for (std::size_t d = 0; d < dim; ++d)
    {
      vol *= (b.eval_at (d) - a.eval_at (d));
    }
  double sum = 0.0;
  double sum_sq = 0.0;
  nd_tensor<double> pt (shape_type {dim});
  for (std::size_t i = 1; i <= n_points; ++i)
    {
      for (std::size_t d = 0; d < dim; ++d)
        {
          const std::size_t base = primes[d % 16];
          double f = 1.0;
          double r = 0.0;
          std::size_t idx = i;
          while (idx > 0)
            {
              f /= static_cast<double> (base);
              r += f * static_cast<double> (idx % base);
              idx /= base;
            }
          pt[d] = a.eval_at (d) + r * (b.eval_at (d) - a.eval_at (d));
        }
      const double v = func (pt);
      sum += v;
      sum_sq += v * v;
    }
  const double mean = sum / static_cast<double> (n_points);
  const double var =
    std::max (0.0, sum_sq / static_cast<double> (n_points) - mean * mean);
  const double err =
    std::abs (vol) * std::sqrt (var / static_cast<double> (n_points));
  return {vol * mean, err, n_points};
}

} // namespace hyper_math::scientific::integrate

#endif /* HYPER_MATH_SCIENTIFIC_INTEGRATE_HPP */
