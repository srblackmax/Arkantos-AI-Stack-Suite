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

#ifndef HYPER_MATH_SCIENTIFIC_OPTIMIZE_HPP
#define HYPER_MATH_SCIENTIFIC_OPTIMIZE_HPP

#include "hyper_math/scientific/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

namespace hyper_math::scientific::optimize
{

using tensor::nd_tensor;
using tensor::shape_type;

/* Brent-Dekker bracketed root-finding algorithm on [a, b].  */
template <typename Func>
double
brentq (Func &&f,
        double a,
        double b,
        double xtol = 1e-12,
        std::size_t max_iter = 200)
{
  double fa = f (a);
  double fb = f (b);
  if (fa * fb > 0.0)
    {
      throw convergence_error ("brentq requires f(a) and f(b) to have opposite signs");
    }
  if (std::abs (fa) < std::abs (fb))
    {
      std::swap (a, b);
      std::swap (fa, fb);
    }

  double c = a;
  double fc = fa;
  double d = b - a;
  bool mflag = true;

  for (std::size_t iter = 0; iter < max_iter; ++iter)
    {
      if (std::abs (fb) == 0.0 || std::abs (b - a) < xtol)
        {
          return b;
        }

      double s = 0.0;
      if (fa != fc && fb != fc)
        {
          /* Inverse quadratic interpolation.  */
          s = (a * fb * fc) / ((fa - fb) * (fa - fc))
              + (b * fa * fc) / ((fb - fa) * (fb - fc))
              + (c * fa * fb) / ((fc - fa) * (fc - fb));
        }
      else
        {
          /* Secant step.  */
          s = b - fb * (b - a) / (fb - fa);
        }

      const double cond1_lo = (3.0 * a + b) * 0.25;
      const bool outside_interval =
        !((s > std::min (cond1_lo, b)) && (s < std::max (cond1_lo, b)));
      const bool step_too_large =
        (mflag && std::abs (s - b) >= std::abs (b - c) * 0.5)
        || (!mflag && std::abs (s - b) >= std::abs (c - d) * 0.5);
      const bool step_too_small =
        (mflag && std::abs (b - c) < xtol)
        || (!mflag && std::abs (c - d) < xtol);

      if (outside_interval || step_too_large || step_too_small)
        {
          s = 0.5 * (a + b);
          mflag = true;
        }
      else
        {
          mflag = false;
        }

      const double fs = f (s);
      d = c;
      c = b;
      fc = fb;

      if (fa * fs < 0.0)
        {
          b = s;
          fb = fs;
        }
      else
        {
          a = s;
          fa = fs;
        }

      if (std::abs (fa) < std::abs (fb))
        {
          std::swap (a, b);
          std::swap (fa, fb);
        }
    }

  return b;
}

/* Newton-Raphson root solver with finite-difference fallback derivative.  */
template <typename Func>
double
newton (Func &&f,
        double x0,
        double tol = 1e-11,
        std::size_t max_iter = 100)
{
  double x = x0;
  for (std::size_t i = 0; i < max_iter; ++i)
    {
      const double fx = f (x);
      if (std::abs (fx) < tol)
        {
          return x;
        }
      const double h = 1e-7 * std::max (1.0, std::abs (x));
      const double dfx = (f (x + h) - f (x - h)) / (2.0 * h);
      if (std::abs (dfx) < 1e-15)
        {
          throw convergence_error ("zero derivative encountered in newton");
        }
      const double step = fx / dfx;
      x -= step;
      if (std::abs (step) < tol)
        {
          return x;
        }
    }
  throw convergence_error ("newton failed to converge within max_iter");
}

/* Golden-Section 1D Scalar Minimization on [a, b].  */
struct scalar_opt_result
{
  double x;
  double fun;
  std::size_t iterations;
};

template <typename Func>
scalar_opt_result
minimize_scalar (Func &&f,
                 double a,
                 double b,
                 double tol = 1e-10,
                 std::size_t max_iter = 500)
{
  constexpr double inv_phi = 0.61803398874989484820;
  constexpr double inv_phi_sq = 0.38196601125010515180;

  double h = b - a;
  double c = a + inv_phi_sq * h;
  double d = a + inv_phi * h;
  double yc = f (c);
  double yd = f (d);

  std::size_t iter = 0;
  for (; iter < max_iter && std::abs (b - a) > tol; ++iter)
    {
      if (yc < yd)
        {
          b = d;
          d = c;
          yd = yc;
          h = b - a;
          c = a + inv_phi_sq * h;
          yc = f (c);
        }
      else
        {
          a = c;
          c = d;
          yc = yd;
          h = b - a;
          d = a + inv_phi * h;
          yd = f (d);
        }
    }

  const double x_min = (yc < yd) ? c : d;
  const double f_min = (yc < yd) ? yc : yd;
  return scalar_opt_result {x_min, f_min, iter};
}

/* Result of multivariable optimization.  */
struct optimize_result
{
  nd_tensor<double> x;
  double fun;
  nd_tensor<double> grad;
  std::size_t nit;
  bool success;
};

namespace detail
{

  template <typename Func>
  nd_tensor<double>
  finite_diff_gradient (Func &&f, const nd_tensor<double> &x, double eps = 1e-7)
  {
    const std::size_t n = x.size ();
    nd_tensor<double> g (shape_type {n});
    nd_tensor<double> x_pert = x.copy ();
    for (std::size_t i = 0; i < n; ++i)
      {
        const double orig = x_pert[i];
        const double h = eps * std::max (1.0, std::abs (orig));
        x_pert[i] = orig + h;
        const double fp = f (x_pert);
        x_pert[i] = orig - h;
        const double fm = f (x_pert);
        x_pert[i] = orig;
        g[i] = (fp - fm) / (2.0 * h);
      }
    return g;
  }

} // namespace detail

/* Multivariable BFGS Quasi-Newton Optimizer with Armijo Backtracking Line
   Search (SciPy optimize.minimize(method='BFGS') equivalent).  */
template <typename Func>
optimize_result
minimize_bfgs (Func &&f,
               const nd_tensor<double> &x0,
               double gtol = 1e-7,
               std::size_t max_iter = 500)
{
  const std::size_t n = x0.size ();
  if (n == 0)
    {
      throw shape_error ("minimize_bfgs requires non-empty x0");
    }
  nd_tensor<double> x = x0.copy ();
  double fx = f (x);
  nd_tensor<double> g = detail::finite_diff_gradient (f, x);
  nd_tensor<double> h_inv = tensor::eye<double> (n);

  std::size_t iter = 0;
  for (; iter < max_iter; ++iter)
    {
      if (linalg::norm (g) < gtol)
        {
          return optimize_result {x, fx, g, iter, true};
        }

      /* Search direction p = -H * g.  */
      nd_tensor<double> p = -tensor::matmul (h_inv, g);
      double dir_deriv = tensor::dot (g, p);
      if (dir_deriv >= 0.0)
        {
          h_inv = tensor::eye<double> (n);
          p = -g;
          dir_deriv = -tensor::dot (g, g);
        }

      /* Armijo backtracking line search.  */
      double alpha = 1.0;
      constexpr double c1 = 1e-4;
      nd_tensor<double> x_next = x + alpha * p;
      double fx_next = f (x_next);
      for (int ls = 0; ls < 40 && fx_next > fx + c1 * alpha * dir_deriv; ++ls)
        {
          alpha *= 0.5;
          x_next = x + alpha * p;
          fx_next = f (x_next);
        }

      nd_tensor<double> g_next = detail::finite_diff_gradient (f, x_next);
      const nd_tensor<double> s = x_next - x;
      const nd_tensor<double> y = g_next - g;
      const double ys = tensor::dot (y, s);

      if (ys > 1e-14)
        {
          const double rho = 1.0 / ys;
          const nd_tensor<double> i_mat = tensor::eye<double> (n);
          const nd_tensor<double> v_left = i_mat - rho * tensor::outer (s, y);
          const nd_tensor<double> v_right = i_mat - rho * tensor::outer (y, s);
          h_inv = tensor::matmul (tensor::matmul (v_left, h_inv), v_right)
                  + rho * tensor::outer (s, s);
        }

      x = std::move (x_next);
      fx = fx_next;
      g = std::move (g_next);
    }

  return optimize_result {x, fx, g, iter, linalg::norm (g) < gtol};
}

/* Derivative-free Nelder-Mead Downhill Simplex Optimizer.  */
template <typename Func>
optimize_result
minimize_nelder_mead (Func &&f,
                      const nd_tensor<double> &x0,
                      double tol = 1e-8,
                      std::size_t max_iter = 2000)
{
  const std::size_t n = x0.size ();
  if (n == 0)
    {
      throw shape_error ("minimize_nelder_mead requires non-empty x0");
    }
  std::vector<nd_tensor<double>> simplex (n + 1);
  std::vector<double> fvals (n + 1);
  simplex[0] = x0.copy ();
  fvals[0] = f (simplex[0]);

  for (std::size_t i = 0; i < n; ++i)
    {
      simplex[i + 1] = x0.copy ();
      const double step = (std::abs (x0[i]) > 1e-5) ? (0.05 * x0[i]) : 0.25;
      simplex[i + 1][i] += step;
      fvals[i + 1] = f (simplex[i + 1]);
    }

  std::vector<std::size_t> order (n + 1);
  std::size_t iter = 0;
  for (; iter < max_iter; ++iter)
    {
      std::iota (order.begin (), order.end (), 0);
      std::sort (order.begin (), order.end (), [&fvals] (std::size_t a, std::size_t b) {
        return fvals[a] < fvals[b];
      });

      double max_dist = 0.0;
      for (std::size_t i = 1; i <= n; ++i)
        {
          max_dist = std::max (
            max_dist,
            linalg::norm (nd_tensor<double> (simplex[order[i]] - simplex[order[0]])));
        }

      if (std::abs (fvals[order[n]] - fvals[order[0]]) < tol
          && max_dist < tol)
        {
          break;
        }

      nd_tensor<double> centroid (shape_type {n}, 0.0);
      for (std::size_t i = 0; i < n; ++i)
        {
          centroid += simplex[order[i]];
        }
      centroid /= static_cast<double> (n);

      const nd_tensor<double> &worst = simplex[order[n]];
      const nd_tensor<double> xr = centroid + 1.0 * (centroid - worst);
      const double fr = f (xr);

      if (fr < fvals[order[0]])
        {
          const nd_tensor<double> xe = centroid + 2.0 * (xr - centroid);
          const double fe = f (xe);
          if (fe < fr)
            {
              simplex[order[n]] = xe;
              fvals[order[n]] = fe;
            }
          else
            {
              simplex[order[n]] = xr;
              fvals[order[n]] = fr;
            }
        }
      else if (fr < fvals[order[n - 1]])
        {
          simplex[order[n]] = xr;
          fvals[order[n]] = fr;
        }
      else
        {
          const nd_tensor<double> xc =
            (fr < fvals[order[n]])
            ? nd_tensor<double> (centroid + 0.5 * (xr - centroid))
            : nd_tensor<double> (centroid + 0.5 * (worst - centroid));
          const double fc = f (xc);
          if (fc < std::min (fr, fvals[order[n]]))
            {
              simplex[order[n]] = xc;
              fvals[order[n]] = fc;
            }
          else
            {
              for (std::size_t i = 1; i <= n; ++i)
                {
                  simplex[order[i]] =
                    simplex[order[0]]
                    + 0.5 * (simplex[order[i]] - simplex[order[0]]);
                  fvals[order[i]] = f (simplex[order[i]]);
                }
            }
        }
    }

  const nd_tensor<double> &best = simplex[order[0]];
  return optimize_result {
    best,
    fvals[order[0]],
    detail::finite_diff_gradient (f, best),
    iter,
    iter < max_iter
  };
}

/* Levenberg-Marquardt Nonlinear Least Squares Curve Fitting
   (SciPy optimize.curve_fit equivalent).
   Fits model(x_i, params) to y_data[i].  */
template <typename ModelFunc>
nd_tensor<double>
curve_fit (ModelFunc &&model,
           const nd_tensor<double> &x_data,
           const nd_tensor<double> &y_data,
           const nd_tensor<double> &p0,
           std::size_t max_iter = 200,
           double tol = 1e-10)
{
  const std::size_t m = x_data.size ();
  if (y_data.size () != m)
    {
      throw shape_error ("curve_fit x_data and y_data size mismatch");
    }
  const std::size_t p_dim = p0.size ();
  nd_tensor<double> params = p0.copy ();
  double lambda = 1e-3;

  auto compute_residuals = [&] (const nd_tensor<double> &p) {
    nd_tensor<double> r (shape_type {m});
    for (std::size_t i = 0; i < m; ++i)
      {
        r[i] = y_data.eval_at (i) - model (x_data.eval_at (i), p);
      }
    return r;
  };

  nd_tensor<double> r = compute_residuals (params);
  double cost = 0.5 * tensor::dot (r, r);

  for (std::size_t iter = 0; iter < max_iter; ++iter)
    {
      /* Compute Jacobian J (m x p_dim) of model wrt params.  */
      nd_tensor<double> J (shape_type {m, p_dim});
      nd_tensor<double> p_pert = params.copy ();
      for (std::size_t j = 0; j < p_dim; ++j)
        {
          const double orig = p_pert[j];
          const double h = 1e-7 * std::max (1.0, std::abs (orig));
          p_pert[j] = orig + h;
          for (std::size_t i = 0; i < m; ++i)
            {
              const double vp = model (x_data.eval_at (i), p_pert);
              p_pert[j] = orig - h;
              const double vm = model (x_data.eval_at (i), p_pert);
              p_pert[j] = orig + h;
              J (i, j) = (vp - vm) / (2.0 * h);
            }
          p_pert[j] = orig;
        }

      const nd_tensor<double> jt = J.t ();
      nd_tensor<double> jtj = tensor::matmul (jt, J);
      const nd_tensor<double> jtr = tensor::matmul (jt, r);

      if (linalg::norm (jtr) < tol)
        {
          break;
        }

      for (std::size_t j = 0; j < p_dim; ++j)
        {
          jtj (j, j) += lambda * std::max (1e-6, jtj (j, j));
        }

      nd_tensor<double> delta = linalg::solve (jtj, jtr);
      if (linalg::norm (delta) < tol)
        {
          break;
        }

      const nd_tensor<double> cand_params = params + delta;
      const nd_tensor<double> cand_r = compute_residuals (cand_params);
      const double cand_cost = 0.5 * tensor::dot (cand_r, cand_r);

      if (cand_cost < cost)
        {
          params = cand_params;
          r = cand_r;
          cost = cand_cost;
          lambda = std::max (1e-9, lambda * 0.3);
        }
      else
        {
          lambda = std::min (1e9, lambda * 5.0);
        }
    }

  return params;
}

/* Box-Constrained Limited-Memory BFGS (L-BFGS-B) optimizer
   (SciPy optimize.minimize(..., method='L-BFGS-B') equivalent).
   Minimizes f(x) subject to bounds[i].first <= x[i] <= bounds[i].second
   using Nocedal's two-loop L-BFGS recursion and projected backtracking.  */
template <typename ObjFunc>
optimize_result
minimize_lbfgsb (ObjFunc &&f,
                 const nd_tensor<double> &x0,
                 const std::vector<std::pair<double, double>> &bounds,
                 double gtol = 1e-6,
                 std::size_t max_iter = 300,
                 std::size_t history_size = 10)
{
  const std::size_t n = x0.size ();
  if (!bounds.empty () && bounds.size () != n)
    {
      throw shape_error ("minimize_lbfgsb bounds length must match x0 size");
    }

  auto project = [&] (const nd_tensor<double> &v) {
    nd_tensor<double> out = v.copy ();
    if (!bounds.empty ())
      {
        for (std::size_t i = 0; i < n; ++i)
          {
            out[i] = std::clamp (out[i], bounds[i].first, bounds[i].second);
          }
      }
    return out;
  };

  nd_tensor<double> x = project (x0);
  double fx = f (x);
  nd_tensor<double> g = detail::finite_diff_gradient (f, x);

  std::vector<nd_tensor<double>> s_hist;
  std::vector<nd_tensor<double>> y_hist;
  std::vector<double> rho_hist;

  std::size_t iter = 0;
  for (; iter < max_iter; ++iter)
    {
      /* Check projected gradient norm ||P(x - g) - x||_inf.  */
      const nd_tensor<double> pg = project (x - g) - x;
      if (linalg::norm (pg) < gtol)
        {
          return optimize_result {x, fx, g, iter, true};
        }

      /* Two-loop L-BFGS recursion to compute search direction d = -H_k g.  */
      nd_tensor<double> q = g.copy ();
      const std::size_t m = s_hist.size ();
      std::vector<double> alpha_vec (m, 0.0);
      for (std::ptrdiff_t i = static_cast<std::ptrdiff_t> (m) - 1; i >= 0; --i)
        {
          const std::size_t ui = static_cast<std::size_t> (i);
          alpha_vec[ui] = rho_hist[ui] * tensor::dot (s_hist[ui], q);
          q = q - alpha_vec[ui] * y_hist[ui];
        }

      double gamma = 1.0;
      if (m > 0)
        {
          const double yy = tensor::dot (y_hist.back (), y_hist.back ());
          if (yy > 1e-15)
            {
              gamma = tensor::dot (s_hist.back (), y_hist.back ()) / yy;
            }
        }
      nd_tensor<double> r = gamma * q;

      for (std::size_t i = 0; i < m; ++i)
        {
          const double beta_val = rho_hist[i] * tensor::dot (y_hist[i], r);
          r = r + (alpha_vec[i] - beta_val) * s_hist[i];
        }

      nd_tensor<double> d = -r;
      auto mask_active_bounds = [&] (nd_tensor<double> &dir) {
        if (!bounds.empty ())
          {
            for (std::size_t i = 0; i < n; ++i)
              {
                if ((x[i] <= bounds[i].first && dir[i] < 0.0)
                    || (x[i] >= bounds[i].second && dir[i] > 0.0))
                  {
                    dir[i] = 0.0;
                  }
              }
          }
      };
      mask_active_bounds (d);
      if (tensor::dot (g, d) >= 0.0)
        {
          d = -g;
          mask_active_bounds (d);
        }

      /* Projected Armijo backtracking line search.  */
      double step = 1.0;
      nd_tensor<double> x_next = project (x + step * d);
      double f_next = f (x_next);
      for (int ls = 0; ls < 30; ++ls)
        {
          const nd_tensor<double> actual_step = x_next - x;
          if (f_next <= fx + 1e-4 * tensor::dot (g, actual_step))
            {
              break;
            }
          step *= 0.5;
          x_next = project (x + step * d);
          f_next = f (x_next);
        }

      nd_tensor<double> g_next = detail::finite_diff_gradient (f, x_next);
      nd_tensor<double> s = x_next - x;
      nd_tensor<double> y = g_next - g;
      const double sy = tensor::dot (s, y);
      if (sy > 1e-12 && history_size > 0)
        {
          if (s_hist.size () >= history_size)
            {
              s_hist.erase (s_hist.begin ());
              y_hist.erase (y_hist.begin ());
              rho_hist.erase (rho_hist.begin ());
            }
          s_hist.push_back (std::move (s));
          y_hist.push_back (std::move (y));
          rho_hist.push_back (1.0 / sy);
        }

      if (linalg::norm (x_next - x) < gtol * 1e-2)
        {
          x = std::move (x_next);
          fx = f_next;
          g = std::move (g_next);
          return optimize_result {x, fx, g, iter + 1, true};
        }

      x = std::move (x_next);
      fx = f_next;
      g = std::move (g_next);
    }

  return optimize_result {x, fx, g, iter, false};
}

/* General Vector-Residual Levenberg-Marquardt Nonlinear Least Squares
   (SciPy optimize.least_squares equivalent).
   Minimizes 0.5 * ||res_fn(x)||_2^2.  */
template <typename ResFunc>
optimize_result
least_squares_lm (ResFunc &&res_fn,
                  const nd_tensor<double> &x0,
                  double tol = 1e-10,
                  std::size_t max_iter = 200)
{
  nd_tensor<double> x = x0.copy ();
  const std::size_t n = x.size ();
  nd_tensor<double> r = res_fn (x);
  const std::size_t m = r.size ();
  double cost = 0.5 * tensor::dot (r, r);
  double lambda = 1e-3;

  std::size_t iter = 0;
  for (; iter < max_iter; ++iter)
    {
      nd_tensor<double> J (shape_type {m, n});
      nd_tensor<double> x_pert = x.copy ();
      for (std::size_t j = 0; j < n; ++j)
        {
          const double orig = x_pert[j];
          const double h = 1e-7 * std::max (1.0, std::abs (orig));
          x_pert[j] = orig + h;
          const nd_tensor<double> rp = res_fn (x_pert);
          x_pert[j] = orig - h;
          const nd_tensor<double> rm = res_fn (x_pert);
          x_pert[j] = orig;
          for (std::size_t i = 0; i < m; ++i)
            {
              J (i, j) = (rp[i] - rm[i]) / (2.0 * h);
            }
        }

      const nd_tensor<double> jt = J.t ();
      nd_tensor<double> jtj = tensor::matmul (jt, J);
      const nd_tensor<double> g = tensor::matmul (jt, r);
      if (linalg::norm (g) < tol)
        {
          return optimize_result {x, cost, g, iter, true};
        }

      for (std::size_t j = 0; j < n; ++j)
        {
          jtj (j, j) += lambda * std::max (1e-6, jtj (j, j));
        }

      const nd_tensor<double> step = linalg::solve (jtj, -g);
      if (linalg::norm (step) < tol)
        {
          return optimize_result {x, cost, g, iter, true};
        }

      const nd_tensor<double> x_cand = x + step;
      const nd_tensor<double> r_cand = res_fn (x_cand);
      const double cost_cand = 0.5 * tensor::dot (r_cand, r_cand);
      if (cost_cand < cost)
        {
          x = x_cand;
          r = r_cand;
          cost = cost_cand;
          lambda = std::max (1e-9, lambda * 0.3);
        }
      else
        {
          lambda = std::min (1e9, lambda * 5.0);
        }
    }

  return optimize_result {x, cost, nd_tensor<double> (shape_type {n}, 0.0), iter, false};
}

/* Linear Programming Solver via Dantzig Tableau Simplex Algorithm
   (SciPy optimize.linprog equivalent):
     Minimize    c^T * x
     Subject to  A_ub * x <= b_ub  (with b_ub >= 0, x >= 0).  */
struct linprog_result
{
  nd_tensor<double> x;
  double fun;
  std::size_t nit;
  bool success;
};

inline linprog_result
linprog (const nd_tensor<double> &c,
         const nd_tensor<double> &a_ub,
         const nd_tensor<double> &b_ub,
         std::size_t max_iter = 500)
{
  const std::size_t n_vars = c.size ();
  if (a_ub.ndim () != 2 || a_ub.shape ()[1] != n_vars
      || b_ub.size () != a_ub.shape ()[0])
    {
      throw shape_error ("linprog dimension mismatch between c, a_ub, and b_ub");
    }
  const std::size_t m_cons = a_ub.shape ()[0];
  const std::size_t n_cols = n_vars + m_cons + 1;

  /* Construct Simplex tableau of size (m_cons + 1) x (n_vars + m_cons + 1):
     Rows 0..m_cons-1: [ A_ub | I_slack | b_ub ]
     Row m_cons:       [  c^T |   0     |  0   ].  */
  nd_tensor<double> tab (shape_type {m_cons + 1, n_cols}, 0.0);
  std::vector<std::size_t> basis (m_cons);
  for (std::size_t i = 0; i < m_cons; ++i)
    {
      if (b_ub.eval_at (i) < 0.0)
        {
          throw hyper_math_error ("linprog requires non-negative b_ub in standard form");
        }
      for (std::size_t j = 0; j < n_vars; ++j)
        {
          tab (i, j) = a_ub (i, j);
        }
      tab (i, n_vars + i) = 1.0;
      tab (i, n_cols - 1) = b_ub.eval_at (i);
      basis[i] = n_vars + i;
    }
  for (std::size_t j = 0; j < n_vars; ++j)
    {
      tab (m_cons, j) = c.eval_at (j);
    }

  std::size_t iter = 0;
  for (; iter < max_iter; ++iter)
    {
      /* Select entering variable with most negative reduced cost.  */
      std::size_t pivot_col = n_cols;
      double most_neg = -1e-12;
      for (std::size_t j = 0; j < n_vars + m_cons; ++j)
        {
          if (tab (m_cons, j) < most_neg)
            {
              most_neg = tab (m_cons, j);
              pivot_col = j;
            }
        }
      if (pivot_col == n_cols)
        {
          break;
        }

      /* Minimum ratio test to select leaving variable.  */
      std::size_t pivot_row = m_cons;
      double min_ratio = constants::inf;
      for (std::size_t i = 0; i < m_cons; ++i)
        {
          const double a_ij = tab (i, pivot_col);
          if (a_ij > 1e-12)
            {
              const double ratio = tab (i, n_cols - 1) / a_ij;
              if (ratio < min_ratio)
                {
                  min_ratio = ratio;
                  pivot_row = i;
                }
            }
        }
      if (pivot_row == m_cons)
        {
          /* Unbounded objective.  */
          return linprog_result {
            nd_tensor<double> (shape_type {n_vars}, constants::nan),
            -constants::inf,
            iter,
            false
          };
        }

      /* Gauss-Jordan pivot around (pivot_row, pivot_col).  */
      const double pv = tab (pivot_row, pivot_col);
      for (std::size_t j = 0; j < n_cols; ++j)
        {
          tab (pivot_row, j) /= pv;
        }
      for (std::size_t i = 0; i <= m_cons; ++i)
        {
          if (i != pivot_row)
            {
              const double factor = tab (i, pivot_col);
              for (std::size_t j = 0; j < n_cols; ++j)
                {
                  tab (i, j) -= factor * tab (pivot_row, j);
                }
            }
        }
      basis[pivot_row] = pivot_col;
    }

  nd_tensor<double> x_sol (shape_type {n_vars}, 0.0);
  for (std::size_t i = 0; i < m_cons; ++i)
    {
      if (basis[i] < n_vars)
        {
          x_sol[basis[i]] = tab (i, n_cols - 1);
        }
    }
  const double obj = -tab (m_cons, n_cols - 1);
  return linprog_result {std::move (x_sol), obj, iter, iter < max_iter};
}

/* Dual Annealing Global Optimizer (SciPy optimize.dual_annealing equivalent).
   Combines simulated annealing Metropolis-Hastings acceptance over box bounds
   [lower_bounds, upper_bounds] with L-BFGS-B local polishing.  */
template <typename Func>
optimize_result
dual_annealing (Func &&f,
                const nd_tensor<double> &lower_bounds,
                const nd_tensor<double> &upper_bounds,
                std::size_t max_iter = 600,
                std::uint64_t seed = 42)
{
  const std::size_t n = lower_bounds.size ();
  if (n == 0 || upper_bounds.size () != n)
    {
      throw shape_error ("dual_annealing requires matching non-empty bounds");
    }

  random_engine rng (seed);
  nd_tensor<double> x_curr (shape_type {n});
  for (std::size_t i = 0; i < n; ++i)
    {
      x_curr[i] = rng.uniform (lower_bounds.eval_at (i),
                               upper_bounds.eval_at (i));
    }
  double f_curr = f (x_curr);

  nd_tensor<double> x_best = x_curr.copy ();
  double f_best = f_curr;

  const double t_init = 5230.0;
  for (std::size_t step = 1; step <= max_iter; ++step)
    {
      const double temp =
        t_init / std::pow (1.0 + static_cast<double> (step), 1.5);
      const double scale =
        std::clamp (temp / t_init, 0.02, 1.0);

      nd_tensor<double> x_cand (shape_type {n});
      for (std::size_t i = 0; i < n; ++i)
        {
          const double lo = lower_bounds.eval_at (i);
          const double hi = upper_bounds.eval_at (i);
          const double span = hi - lo;
          const double delta = rng.normal (0.0, scale * span * 0.35);
          x_cand[i] = std::clamp (x_curr[i] + delta, lo, hi);
        }

      const double f_cand = f (x_cand);
      if (f_cand < f_curr)
        {
          x_curr = std::move (x_cand);
          f_curr = f_cand;
          if (f_curr < f_best)
            {
              x_best = x_curr.copy ();
              f_best = f_curr;
            }
        }
      else
        {
          const double prob =
            std::exp (-(f_cand - f_curr) / std::max (temp * 1e-3, 1e-9));
          if (rng.uniform (0.0, 1.0) < prob)
            {
              x_curr = std::move (x_cand);
              f_curr = f_cand;
            }
        }
    }

  /* Polish best global candidate using L-BFGS-B box-constrained local search. */
  std::vector<std::pair<double, double>> box (n);
  for (std::size_t i = 0; i < n; ++i)
    {
      box[i] = {lower_bounds.eval_at (i), upper_bounds.eval_at (i)};
    }
  auto polished =
    minimize_lbfgsb (std::forward<Func> (f), x_best, box);
  if (polished.fun <= f_best)
    {
      return polished;
    }
  return optimize_result {
    std::move (x_best),
    f_best,
    nd_tensor<double> (shape_type {n}, 0.0),
    max_iter,
    true
  };
}

/* Multidimensional Nonlinear System Root-Finder F(x) = 0 (SciPy optimize.fsolve
   / optimize.root equivalent). Uses damped Newton-Raphson with finite-difference
   Jacobian and Armijo backtracking on ||F(x)||_2^2.  */
template <typename SysFunc>
optimize_result
fsolve (SysFunc &&fun,
        const nd_tensor<double> &x0,
        double xtol = 1e-9,
        std::size_t max_iter = 200)
{
  const std::size_t n = x0.size ();
  if (n == 0)
    {
      throw shape_error ("fsolve requires non-empty initial guess x0");
    }

  nd_tensor<double> x = x0.copy ();
  nd_tensor<double> fx = fun (x);
  if (fx.size () != n)
    {
      throw shape_error ("fsolve system output dimension must match x0.size()");
    }

  auto norm_sq = [] (const nd_tensor<double> &v) noexcept {
    double s = 0.0;
    for (std::size_t i = 0; i < v.size (); ++i)
      {
        const double val = v.eval_at (i);
        s += val * val;
      }
    return s;
  };

  std::size_t iter = 0;
  bool converged = false;
  const double eps = 1e-7;

  for (; iter < max_iter; ++iter)
    {
      const double f_norm2 = norm_sq (fx);
      if (std::sqrt (f_norm2) <= xtol)
        {
          converged = true;
          break;
        }

      /* Finite-difference Jacobian J_{i,j} = dF_i / dx_j.  */
      nd_tensor<double> jac (shape_type {n, n});
      for (std::size_t j = 0; j < n; ++j)
        {
          const double h = eps * std::max (1.0, std::abs (x[j]));
          nd_tensor<double> x_pert = x.copy ();
          x_pert[j] += h;
          const nd_tensor<double> f_pert = fun (x_pert);
          for (std::size_t i = 0; i < n; ++i)
            {
              jac (i, j) = (f_pert.eval_at (i) - fx.eval_at (i)) / h;
            }
        }

      /* Solve J * dx = -F(x) via lstsq for robustness against near-singular J. */
      const nd_tensor<double> rhs (-fx);
      const nd_tensor<double> dx = linalg::lstsq (jac, rhs);

      /* Backtracking line search to reduce ||F(x + alpha * dx)||^2.  */
      double alpha = 1.0;
      nd_tensor<double> x_next = x.copy ();
      nd_tensor<double> f_next = fx.copy ();
      for (int ls = 0; ls < 20; ++ls)
        {
          x_next = nd_tensor<double> (x + alpha * dx);
          f_next = fun (x_next);
          if (norm_sq (f_next) < f_norm2 || alpha < 1e-4)
            {
              break;
            }
          alpha *= 0.5;
        }

      x = std::move (x_next);
      fx = std::move (f_next);
    }

  const double final_res = std::sqrt (norm_sq (fx));
  if (final_res <= xtol)
    {
      converged = true;
    }

  return optimize_result {
    std::move (x), final_res, std::move (fx), iter, converged
  };
}

/* Exact O(N^3) Hungarian / Jonker-Volgenant Minimum-Weight Bipartite
   Assignment Algorithm (SciPy optimize.linear_sum_assignment equivalent).
   Solves min \sum_i C(row_ind[i], col_ind[i]) for rectangular cost matrix C. */
struct assignment_result
{
  nd_tensor<std::size_t> row_ind;
  nd_tensor<std::size_t> col_ind;
  double total_cost;
};

inline assignment_result
linear_sum_assignment (const nd_tensor<double> &cost_matrix,
                       bool maximize = false)
{
  if (cost_matrix.ndim () != 2)
    {
      throw shape_error ("linear_sum_assignment requires a 2D cost matrix");
    }
  const std::size_t nr = cost_matrix.shape ()[0];
  const std::size_t nc = cost_matrix.shape ()[1];
  if (nr == 0 || nc == 0)
    {
      return assignment_result {
        nd_tensor<std::size_t> (shape_type {0}),
        nd_tensor<std::size_t> (shape_type {0}),
        0.0
      };
    }

  const bool transposed = (nr > nc);
  const std::size_t n = transposed ? nc : nr;
  const std::size_t m = transposed ? nr : nc;
  const double sign = maximize ? -1.0 : 1.0;

  /* 1-indexed potentials u[0..n], v[0..m], and matching p[0..m], way[0..m].  */
  std::vector<double> u (n + 1, 0.0);
  std::vector<double> v (m + 1, 0.0);
  std::vector<std::size_t> p (m + 1, 0);
  std::vector<std::size_t> way (m + 1, 0);

  auto cost_at = [&] (std::size_t i1, std::size_t j1) noexcept -> double {
    const std::size_t r = transposed ? (j1 - 1) : (i1 - 1);
    const std::size_t c = transposed ? (i1 - 1) : (j1 - 1);
    return sign * cost_matrix (r, c);
  };

  for (std::size_t i = 1; i <= n; ++i)
    {
      p[0] = i;
      std::size_t j0 = 0;
      std::vector<double> minv (m + 1, constants::inf);
      std::vector<bool> used (m + 1, false);
      do
        {
          used[j0] = true;
          const std::size_t i0 = p[j0];
          double delta = constants::inf;
          std::size_t j1 = 0;
          for (std::size_t j = 1; j <= m; ++j)
            {
              if (!used[j])
                {
                  const double cur = cost_at (i0, j) - u[i0] - v[j];
                  if (cur < minv[j])
                    {
                      minv[j] = cur;
                      way[j] = j0;
                    }
                  if (minv[j] < delta)
                    {
                      delta = minv[j];
                      j1 = j;
                    }
                }
            }
          for (std::size_t j = 0; j <= m; ++j)
            {
              if (used[j])
                {
                  u[p[j]] += delta;
                  v[j] -= delta;
                }
              else
                {
                  minv[j] -= delta;
                }
            }
          j0 = j1;
        }
      while (p[j0] != 0);

      do
        {
          const std::size_t j1 = way[j0];
          p[j0] = p[j1];
          j0 = j1;
        }
      while (j0 != 0);
    }

  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  pairs.reserve (n);
  for (std::size_t j = 1; j <= m; ++j)
    {
      if (p[j] > 0)
        {
          const std::size_t r = transposed ? (j - 1) : (p[j] - 1);
          const std::size_t c = transposed ? (p[j] - 1) : (j - 1);
          pairs.emplace_back (r, c);
        }
    }
  std::sort (pairs.begin (), pairs.end ());

  nd_tensor<std::size_t> row_out (shape_type {n});
  nd_tensor<std::size_t> col_out (shape_type {n});
  double total = 0.0;
  for (std::size_t k = 0; k < n; ++k)
    {
      row_out[k] = pairs[k].first;
      col_out[k] = pairs[k].second;
      total += cost_matrix (pairs[k].first, pairs[k].second);
    }
  return assignment_result {std::move (row_out), std::move (col_out), total};
}

/* Additional Scalar Root Finders (`bisect`, `ridder`, `secant`), Minimum
   Bracketing (`bracket`), and Global Non-Convex Optimizers
   (`differential_evolution`, `basinhopping`) (SciPy optimize equivalents).  */
template <typename Func>
double
bisect (Func &&f,
        double a,
        double b,
        double xtol = 1e-12,
        std::size_t max_iter = 200)
{
  double fa = f (a);
  double fb = f (b);
  if (fa * fb > 0.0)
    {
      throw convergence_error ("bisect requires f(a) and f(b) to have opposite signs");
    }
  if (std::abs (fa) == 0.0)
    {
      return a;
    }
  if (std::abs (fb) == 0.0)
    {
      return b;
    }
  for (std::size_t iter = 0; iter < max_iter; ++iter)
    {
      const double mid = 0.5 * (a + b);
      const double fm = f (mid);
      if (std::abs (fm) == 0.0 || 0.5 * std::abs (b - a) < xtol)
        {
          return mid;
        }
      if (fa * fm < 0.0)
        {
          b = mid;
          fb = fm;
        }
      else
        {
          a = mid;
          fa = fm;
        }
    }
  return 0.5 * (a + b);
}

template <typename Func>
double
ridder (Func &&f,
        double a,
        double b,
        double xtol = 1e-12,
        std::size_t max_iter = 200)
{
  double fa = f (a);
  double fb = f (b);
  if (fa * fb > 0.0)
    {
      throw convergence_error ("ridder requires f(a) and f(b) to have opposite signs");
    }
  if (std::abs (fa) == 0.0)
    {
      return a;
    }
  if (std::abs (fb) == 0.0)
    {
      return b;
    }

  double ans = 0.5 * (a + b);
  for (std::size_t iter = 0; iter < max_iter; ++iter)
    {
      const double xm = 0.5 * (a + b);
      const double fm = f (xm);
      const double s = std::sqrt (fm * fm - fa * fb);
      if (s == 0.0)
        {
          return ans;
        }
      const double sign = (fa >= fb) ? 1.0 : -1.0;
      const double xnew = xm + (xm - a) * sign * fm / s;
      if (std::abs (xnew - ans) <= xtol)
        {
          return xnew;
        }
      ans = xnew;
      const double fnew = f (ans);
      if (std::abs (fnew) == 0.0)
        {
          return ans;
        }
      if (fm * fnew < 0.0)
        {
          a = xm;
          fa = fm;
          b = ans;
          fb = fnew;
        }
      else if (fa * fnew < 0.0)
        {
          b = ans;
          fb = fnew;
        }
      else
        {
          a = ans;
          fa = fnew;
        }
      if (std::abs (b - a) <= xtol)
        {
          return ans;
        }
    }
  return ans;
}

template <typename Func>
double
secant (Func &&f,
        double x0,
        double x1,
        double xtol = 1e-12,
        std::size_t max_iter = 200)
{
  double f0 = f (x0);
  double f1 = f (x1);
  for (std::size_t iter = 0; iter < max_iter; ++iter)
    {
      if (std::abs (f1) < xtol || std::abs (x1 - x0) < xtol)
        {
          return x1;
        }
      const double denom = f1 - f0;
      if (std::abs (denom) < 1e-16)
        {
          throw convergence_error ("secant encountered zero slope");
        }
      const double x2 = x1 - f1 * (x1 - x0) / denom;
      x0 = x1;
      f0 = f1;
      x1 = x2;
      f1 = f (x1);
    }
  if (std::abs (f1) < 1e-6)
    {
      return x1;
    }
  throw convergence_error ("secant failed to converge within max_iter");
}

struct bracket_result
{
  double xa;
  double xb;
  double xc;
  double fa;
  double fb;
  double fc;
  std::size_t funcalls;
};

template <typename Func>
bracket_result
bracket (Func &&f,
         double xa = 0.0,
         double xb = 1.0,
         double grow_limit = 110.0,
         std::size_t max_iter = 1000)
{
  constexpr double gold = 1.6180339887498948482;
  double fa = f (xa);
  double fb = f (xb);
  std::size_t funcalls = 2;
  if (fa < fb)
    {
      std::swap (xa, xb);
      std::swap (fa, fb);
    }
  double xc = xb + gold * (xb - xa);
  double fc = f (xc);
  ++funcalls;

  std::size_t iter = 0;
  while (fc < fb && iter < max_iter)
    {
      const double tmp1 = (xb - xa) * (fb - fc);
      const double tmp2 = (xb - xc) * (fb - fa);
      const double val = tmp2 - tmp1;
      const double denom =
        (std::abs (val) < 1e-21) ? (2.0 * 1e-21) : (2.0 * val);
      double w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
      const double wlim = xb + grow_limit * (xc - xb);
      double fw = 0.0;
      ++iter;

      if ((w - xc) * (xb - w) > 0.0)
        {
          fw = f (w);
          ++funcalls;
          if (fw < fc)
            {
              xa = xb;
              xb = w;
              fa = fb;
              fb = fw;
              return bracket_result {xa, xb, xc, fa, fb, fc, funcalls};
            }
          else if (fw > fb)
            {
              xc = w;
              fc = fw;
              return bracket_result {xa, xb, xc, fa, fb, fc, funcalls};
            }
          w = xc + gold * (xc - xb);
          fw = f (w);
          ++funcalls;
        }
      else if ((w - wlim) * (wlim - xc) >= 0.0)
        {
          w = wlim;
          fw = f (w);
          ++funcalls;
        }
      else if ((w - wlim) * (xc - w) > 0.0)
        {
          fw = f (w);
          ++funcalls;
          if (fw < fc)
            {
              xb = xc;
              xc = w;
              w = xc + gold * (xc - xb);
              fb = fc;
              fc = fw;
              fw = f (w);
              ++funcalls;
            }
        }
      else
        {
          w = xc + gold * (xc - xb);
          fw = f (w);
          ++funcalls;
        }
      xa = xb;
      xb = xc;
      xc = w;
      fa = fb;
      fb = fc;
      fc = fw;
    }

  return bracket_result {xa, xb, xc, fa, fb, fc, funcalls};
}

template <typename Func>
optimize_result
differential_evolution (
  Func &&f,
  const std::vector<std::pair<double, double>> &bounds,
  std::size_t popsize = 15,
  std::size_t max_iter = 200,
  double mutation = 0.7,
  double recombination = 0.9,
  std::uint64_t seed = 42)
{
  const std::size_t dim = bounds.size ();
  if (dim == 0 || popsize < 4)
    {
      throw shape_error ("differential_evolution requires dim >= 1 and popsize >= 4");
    }
  const std::size_t np = popsize * dim;
  std::mt19937_64 rng (seed);
  std::uniform_real_distribution<double> u01 (0.0, 1.0);
  std::uniform_int_distribution<std::size_t> rand_dim (0, dim - 1);

  std::vector<nd_tensor<double>> pop (np, nd_tensor<double> (shape_type {dim}));
  std::vector<double> energies (np, 0.0);
  std::size_t best_idx = 0;

  for (std::size_t i = 0; i < np; ++i)
    {
      for (std::size_t d = 0; d < dim; ++d)
        {
          const double lo = bounds[d].first;
          const double hi = bounds[d].second;
          pop[i][d] = lo + u01 (rng) * (hi - lo);
        }
      energies[i] = f (pop[i]);
      if (energies[i] < energies[best_idx])
        {
          best_idx = i;
        }
    }

  std::uniform_int_distribution<std::size_t> rand_pop (0, np - 1);
  nd_tensor<double> trial (shape_type {dim});

  for (std::size_t gen = 0; gen < max_iter; ++gen)
    {
      for (std::size_t i = 0; i < np; ++i)
        {
          std::size_t r1 = rand_pop (rng);
          while (r1 == i)
            {
              r1 = rand_pop (rng);
            }
          std::size_t r2 = rand_pop (rng);
          while (r2 == i || r2 == r1)
            {
              r2 = rand_pop (rng);
            }

          const std::size_t j_rand = rand_dim (rng);
          for (std::size_t d = 0; d < dim; ++d)
            {
              if (u01 (rng) < recombination || d == j_rand)
                {
                  const double v =
                    pop[best_idx][d] + mutation * (pop[r1][d] - pop[r2][d]);
                  trial[d] = std::clamp (v, bounds[d].first, bounds[d].second);
                }
              else
                {
                  trial[d] = pop[i][d];
                }
            }

          const double e_trial = f (trial);
          if (e_trial <= energies[i])
            {
              pop[i] = trial;
              energies[i] = e_trial;
              if (e_trial < energies[best_idx])
                {
                  best_idx = i;
                }
            }
        }
    }

  /* Local Nelder-Mead polish clamped to box bounds.  */
  auto clamped_f = [&] (const nd_tensor<double> &x) -> double {
    nd_tensor<double> xc (shape_type {dim});
    for (std::size_t d = 0; d < dim; ++d)
      {
        xc[d] = std::clamp (x[d], bounds[d].first, bounds[d].second);
      }
    return f (xc);
  };
  optimize_result polished = minimize_nelder_mead (clamped_f, pop[best_idx]);
  for (std::size_t d = 0; d < dim; ++d)
    {
      polished.x[d] =
        std::clamp (polished.x[d], bounds[d].first, bounds[d].second);
    }
  polished.fun = f (polished.x);
  return polished;
}

template <typename Func>
optimize_result
basinhopping (Func &&f,
              const nd_tensor<double> &x0,
              std::size_t niter = 50,
              double T = 1.0,
              double stepsize = 0.5,
              std::uint64_t seed = 42)
{
  const std::size_t dim = x0.size ();
  if (dim == 0)
    {
      throw shape_error ("basinhopping requires non-empty x0");
    }
  std::mt19937_64 rng (seed);
  std::uniform_real_distribution<double> step_dist (-stepsize, stepsize);
  std::uniform_real_distribution<double> u01 (0.0, 1.0);

  optimize_result cur = minimize_bfgs (f, x0);
  optimize_result best = cur;

  for (std::size_t it = 0; it < niter; ++it)
    {
      nd_tensor<double> x_trial = cur.x.copy ();
      for (std::size_t d = 0; d < dim; ++d)
        {
          x_trial[d] += step_dist (rng);
        }
      optimize_result cand = minimize_bfgs (f, x_trial);
      if (cand.fun < best.fun)
        {
          best = cand;
        }
      const double dE = cand.fun - cur.fun;
      if (dE <= 0.0 || (T > 0.0 && u01 (rng) < std::exp (-dE / T)))
        {
          cur = std::move (cand);
        }
    }
  return best;
}

/* Nonlinear Conjugate Gradient `minimize_cg`, Powell's Direction-Set Method
   `minimize_powell`, Constrained Nonlinear Optimizer `minimize_slsqp`,
   Lawson-Hanson Non-Negative Least Squares `nnls`, Bounded Linear Least
   Squares `lsq_linear`, Pool-Adjacent-Violators `isotonic_regression`, and
   Branch-and-Bound Mixed-Integer Linear Programming `milp`
   (SciPy optimize equivalents).  */
template <typename Func>
optimize_result
minimize_cg (Func &&f,
             const nd_tensor<double> &x0,
             double gtol = 1e-6,
             std::size_t max_iter = 300)
{
  const std::size_t n = x0.size ();
  if (n == 0)
    {
      throw shape_error ("minimize_cg requires non-empty x0");
    }
  nd_tensor<double> x = x0.copy ();
  std::size_t nfev = 0;
  auto grad_fn = [&] (const nd_tensor<double> &pt, double f_pt) {
    nd_tensor<double> g (shape_type {n});
    for (std::size_t j = 0; j < n; ++j)
      {
        const double h = 1e-7 * std::max (1.0, std::abs (pt[j]));
        nd_tensor<double> p2 = pt.copy ();
        p2[j] += h;
        g[j] = (f (p2) - f_pt) / h;
        ++nfev;
      }
    return g;
  };

  double fx = f (x);
  ++nfev;
  nd_tensor<double> g = grad_fn (x, fx);
  nd_tensor<double> d = -1.0 * g;

  for (std::size_t it = 0; it < max_iter; ++it)
    {
      double g_norm = 0.0;
      for (std::size_t i = 0; i < n; ++i)
        {
          g_norm = std::max (g_norm, std::abs (g[i]));
        }
      if (g_norm <= gtol)
        {
          return optimize_result {x, fx, g, it, true};
        }

      /* Backtracking Armijo line search along direction d.  */
      double dg = tensor::dot (g, d);
      if (dg >= 0.0)
        {
          d = -1.0 * g;
          dg = tensor::dot (g, d);
        }
      double alpha = 1.0;
      nd_tensor<double> x_next = x;
      double f_next = fx;
      for (int ls = 0; ls < 30; ++ls)
        {
          x_next = x + alpha * d;
          f_next = f (x_next);
          ++nfev;
          if (f_next <= fx + 1e-4 * alpha * dg)
            {
              break;
            }
          alpha *= 0.5;
        }

      nd_tensor<double> g_next = grad_fn (x_next, f_next);
      /* Polak-Ribiere+ beta = max(0, g_{k+1}^T (g_{k+1} - g_k) / ||g_k||^2). */
      const double denom = std::max (1e-30, tensor::dot (g, g));
      const nd_tensor<double> dg_vec = g_next - g;
      const double beta =
        std::max (0.0, tensor::dot (g_next, dg_vec) / denom);
      d = -1.0 * g_next + beta * d;
      x = std::move (x_next);
      fx = f_next;
      g = std::move (g_next);
    }

  (void) nfev;
  return optimize_result {x, fx, g, max_iter, false};
}

template <typename Func>
optimize_result
minimize_powell (Func &&f,
                 const nd_tensor<double> &x0,
                 double ftol = 1e-8,
                 std::size_t max_iter = 200)
{
  const std::size_t n = x0.size ();
  if (n == 0)
    {
      throw shape_error ("minimize_powell requires non-empty x0");
    }
  nd_tensor<double> x = x0.copy ();
  double fx = f (x);
  std::vector<nd_tensor<double>> dirs (n);
  for (std::size_t i = 0; i < n; ++i)
    {
      dirs[i] = nd_tensor<double> (shape_type {n}, 0.0);
      dirs[i][i] = 1.0;
    }

  for (std::size_t it = 0; it < max_iter; ++it)
    {
      const double fx_start = fx;
      for (std::size_t i = 0; i < n; ++i)
        {
          const nd_tensor<double> &dir = dirs[i];
          auto line_f = [&] (double alpha) {
            return f (x + alpha * dir);
          };
          const auto res_1d = minimize_scalar (line_f, -2.0, 2.0, 1e-7, 60);
          x = x + res_1d.x * dir;
          fx = res_1d.fun;
        }
      if (std::abs (fx_start - fx) <= ftol * (1.0 + std::abs (fx)))
        {
          return optimize_result {
            x, fx, nd_tensor<double> (shape_type {n}, 0.0), it + 1, true
          };
        }
    }
  return optimize_result {
    x, fx, nd_tensor<double> (shape_type {n}, 0.0), max_iter, true
  };
}

/* Constrained nonlinear optimization `minimize_slsqp` solving:
     min_x f(x)  s.t.  eq_constraints[i](x) == 0,  ineq_constraints[j](x) >= 0
   via Augmented Lagrangian / Sequential Quadratic Penalty with BFGS subproblems. */
template <typename Func>
optimize_result
minimize_slsqp (
  Func &&f,
  const nd_tensor<double> &x0,
  const std::vector<std::function<double (const nd_tensor<double> &)>> &eq_constraints = {},
  const std::vector<std::function<double (const nd_tensor<double> &)>> &ineq_constraints = {},
  double tol = 1e-6)
{
  nd_tensor<double> x = x0.copy ();
  double mu = 10.0;
  std::vector<double> lam_eq (eq_constraints.size (), 0.0);
  std::vector<double> lam_ineq (ineq_constraints.size (), 0.0);
  std::size_t total_iters = 0;

  for (int outer = 0; outer < 10; ++outer)
    {
      auto aug_lag = [&] (const nd_tensor<double> &pt) -> double {
        double val = f (pt);
        for (std::size_t i = 0; i < eq_constraints.size (); ++i)
          {
            const double c = eq_constraints[i] (pt);
            val += -lam_eq[i] * c + 0.5 * mu * c * c;
          }
        for (std::size_t j = 0; j < ineq_constraints.size (); ++j)
          {
            const double g = ineq_constraints[j] (pt);
            const double viol = std::min (0.0, g - lam_ineq[j] / mu);
            val += 0.5 * mu * viol * viol;
          }
        return val;
      };

      const auto sub = minimize_bfgs (aug_lag, x, tol, 150);
      x = sub.x;
      total_iters += sub.nit;

      double max_viol = 0.0;
      for (std::size_t i = 0; i < eq_constraints.size (); ++i)
        {
          const double c = eq_constraints[i] (x);
          lam_eq[i] -= mu * c;
          max_viol = std::max (max_viol, std::abs (c));
        }
      for (std::size_t j = 0; j < ineq_constraints.size (); ++j)
        {
          const double g = ineq_constraints[j] (x);
          lam_ineq[j] = std::max (0.0, lam_ineq[j] - mu * g);
          max_viol = std::max (max_viol, std::max (0.0, -g));
        }
      if (max_viol <= tol)
        {
          break;
        }
      mu *= 10.0;
    }

  return optimize_result {
    x, f (x), detail::finite_diff_gradient (f, x), total_iters, true
  };
}

inline nd_tensor<double>
lsq_linear (const nd_tensor<double> &a,
            const nd_tensor<double> &b,
            const nd_tensor<double> &lb,
            const nd_tensor<double> &ub)
{
  if (a.ndim () != 2 || a.shape ()[0] != b.size ()
      || lb.size () != a.shape ()[1] || ub.size () != a.shape ()[1])
    {
      throw shape_error ("lsq_linear dimension mismatch");
    }
  const std::size_t n = a.shape ()[1];
  std::vector<std::pair<double, double>> bounds (n);
  nd_tensor<double> x0 (shape_type {n});
  for (std::size_t j = 0; j < n; ++j)
    {
      bounds[j] = {lb[j], ub[j]};
      x0[j] = std::clamp (0.0, lb[j], ub[j]);
    }
  auto obj = [&] (const nd_tensor<double> &x) -> double {
    const nd_tensor<double> r = tensor::matmul (a, x) - b;
    return 0.5 * tensor::dot (r, r);
  };
  return minimize_lbfgsb (obj, x0, bounds).x;
}

inline std::pair<nd_tensor<double>, double>
nnls (const nd_tensor<double> &a, const nd_tensor<double> &b)
{
  if (a.ndim () != 2 || a.shape ()[0] != b.size ())
    {
      throw shape_error ("nnls dimension mismatch");
    }
  const std::size_t n = a.shape ()[1];
  nd_tensor<double> lb (shape_type {n}, 0.0);
  nd_tensor<double> ub (shape_type {n}, 1e12);
  nd_tensor<double> x = lsq_linear (a, b, lb, ub);
  for (std::size_t j = 0; j < n; ++j)
    {
      if (x[j] < 1e-12)
        {
          x[j] = 0.0;
        }
    }
  const nd_tensor<double> r = tensor::matmul (a, x) - b;
  return {std::move (x), linalg::norm (r)};
}

inline nd_tensor<double>
isotonic_regression (const nd_tensor<double> &y, bool increasing = true)
{
  const std::size_t n = y.size ();
  if (n == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  struct block
  {
    double sum;
    std::size_t count;
    double mean () const noexcept { return sum / static_cast<double> (count); }
  };
  std::vector<block> blocks;
  blocks.reserve (n);
  for (std::size_t i = 0; i < n; ++i)
    {
      const double v = increasing ? y.eval_at (i) : -y.eval_at (i);
      blocks.push_back (block {v, 1});
      while (blocks.size () >= 2
             && blocks[blocks.size () - 2].mean () > blocks.back ().mean ())
        {
          blocks[blocks.size () - 2].sum += blocks.back ().sum;
          blocks[blocks.size () - 2].count += blocks.back ().count;
          blocks.pop_back ();
        }
    }
  nd_tensor<double> out (shape_type {n});
  std::size_t pos = 0;
  for (const auto &blk : blocks)
    {
      const double val = increasing ? blk.mean () : -blk.mean ();
      for (std::size_t k = 0; k < blk.count; ++k)
        {
          out[pos++] = val;
        }
    }
  return out;
}

/* Branch-and-Bound Mixed-Integer Linear Programming `milp`
   solving min c^T x s.t. A_ub x <= b_ub, x >= 0, and x_j in Z for integrality[j]. */
inline linprog_result
milp (const nd_tensor<double> &c,
      const nd_tensor<double> &a_ub,
      const nd_tensor<double> &b_ub,
      const std::vector<bool> &integrality)
{
  const std::size_t n = c.size ();
  if (integrality.size () != n)
    {
      throw shape_error ("milp integrality vector size must match c.size()");
    }

  linprog_result best_sol {
    nd_tensor<double> (shape_type {n}, 0.0),
    constants::inf,
    0,
    false
  };

  const std::size_t m_orig = a_ub.shape ()[0];
  auto solve_node = [&] (auto &self,
                         const std::vector<double> &lb,
                         const std::vector<double> &ub,
                         std::size_t depth) -> void {
    if (depth > 40)
      {
        return;
      }
    for (std::size_t j = 0; j < n; ++j)
      {
        if (lb[j] > ub[j] + 1e-9)
          {
            return;
          }
      }

    /* Count finite upper bounds and build shifted LP in y = x - lb >= 0.  */
    std::size_t n_ub_finite = 0;
    for (std::size_t j = 0; j < n; ++j)
      {
        if (std::isfinite (ub[j]))
          {
            ++n_ub_finite;
          }
      }
    nd_tensor<double> sub_a (shape_type {m_orig + n_ub_finite, n}, 0.0);
    nd_tensor<double> sub_b (shape_type {m_orig + n_ub_finite}, 0.0);
    for (std::size_t r = 0; r < m_orig; ++r)
      {
        double rhs = b_ub[r];
        for (std::size_t j = 0; j < n; ++j)
          {
            sub_a (r, j) = a_ub (r, j);
            rhs -= a_ub (r, j) * lb[j];
          }
        if (rhs < 0.0)
          {
            return;
          }
        sub_b[r] = rhs;
      }
    std::size_t row_idx = m_orig;
    for (std::size_t j = 0; j < n; ++j)
      {
        if (std::isfinite (ub[j]))
          {
            sub_a (row_idx, j) = 1.0;
            sub_b[row_idx] = std::max (0.0, ub[j] - lb[j]);
            ++row_idx;
          }
      }

    linprog_result lp;
    try
      {
        lp = linprog (c, sub_a, sub_b);
      }
    catch (...)
      {
        return;
      }
    if (!lp.success)
      {
        return;
      }

    nd_tensor<double> x_cand (shape_type {n}, 0.0);
    for (std::size_t j = 0; j < n; ++j)
      {
        x_cand[j] = lp.x[j] + lb[j];
      }
    const double obj_cand = tensor::dot (c, x_cand);
    if (obj_cand >= best_sol.fun - 1e-9)
      {
        return;
      }

    /* Check if all integer-constrained variables are integral.  */
    std::size_t branch_var = n;
    double max_frac = 0.0;
    for (std::size_t j = 0; j < n; ++j)
      {
        if (integrality[j])
          {
            const double val = x_cand[j];
            const double frac = std::abs (val - std::round (val));
            if (frac > 1e-6 && frac > max_frac)
              {
                max_frac = frac;
                branch_var = j;
              }
          }
      }

    if (branch_var == n)
      {
        for (std::size_t j = 0; j < n; ++j)
          {
            if (integrality[j])
              {
                x_cand[j] = std::round (x_cand[j]);
              }
          }
        best_sol.x = x_cand;
        best_sol.fun = tensor::dot (c, best_sol.x);
        best_sol.nit = lp.nit;
        best_sol.success = true;
        return;
      }

    const double fl = std::floor (x_cand[branch_var]);
    const double cl = std::ceil (x_cand[branch_var]);

    /* Left branch: x[branch_var] <= floor(val).  */
    if (fl >= lb[branch_var] - 1e-9)
      {
        std::vector<double> next_ub = ub;
        next_ub[branch_var] = std::min (next_ub[branch_var], fl);
        self (self, lb, next_ub, depth + 1);
      }

    /* Right branch: x[branch_var] >= ceil(val).  */
    if (cl <= ub[branch_var] + 1e-9)
      {
        std::vector<double> next_lb = lb;
        next_lb[branch_var] = std::max (next_lb[branch_var], cl);
        self (self, next_lb, ub, depth + 1);
      }
  };

  solve_node (solve_node,
              std::vector<double> (n, 0.0),
              std::vector<double> (n, constants::inf),
              0);
  return best_sol;
}

/* =========================================================================
   Constrained Derivative-Free Optimization (`minimize_cobyla`), Trust-Region
   Dogleg (`minimize_dogleg`), Global Optimization (`shgo`, `direct`, `brute`),
   Quasi-Newton Root Finding (`broyden1`), and Combinatorial Quadratic
   Assignment (`quadratic_assignment`).
   ========================================================================= */

template <typename Func>
optimize_result
minimize_cobyla (
  Func &&f,
  const nd_tensor<double> &x0,
  const std::vector<std::function<double (const nd_tensor<double> &)>> &ineq_constraints = {},
  double rhobeg = 0.5,
  double rhoend = 1e-5,
  std::size_t max_iter = 400)
{
  const std::size_t n = x0.size ();
  if (n == 0)
    {
      throw shape_error ("minimize_cobyla requires non-empty x0");
    }
  double rho = rhobeg;
  double penalty = 10.0;
  nd_tensor<double> x = x0.copy ();
  auto merit = [&] (const nd_tensor<double> &pt) -> double {
    double val = f (pt);
    for (const auto &g : ineq_constraints)
      {
        const double c = g (pt);
        if (c < 0.0)
          {
            val -= penalty * c;
          }
      }
    return val;
  };
  double best_m = merit (x);
  std::size_t it = 0;
  while (rho > rhoend && it < max_iter)
    {
      bool improved = false;
      for (std::size_t d = 0; d < n; ++d)
        {
          for (double sgn : {1.0, -1.0})
            {
              nd_tensor<double> trial = x.copy ();
              trial[d] += sgn * rho;
              const double m_val = merit (trial);
              if (m_val < best_m)
                {
                  best_m = m_val;
                  x = std::move (trial);
                  improved = true;
                }
            }
        }
      if (!improved)
        {
          rho *= 0.5;
          penalty *= 2.0;
          best_m = merit (x);
        }
      ++it;
    }
  auto polished = minimize_nelder_mead (merit, x, rhoend, 200);
  return {polished.x, f (polished.x),
          nd_tensor<double> (shape_type {n}, 0.0), it + polished.nit, true};
}

template <typename Func>
optimize_result
minimize_dogleg (Func &&f,
                 const nd_tensor<double> &x0,
                 double gtol = 1e-6,
                 std::size_t max_iter = 100)
{
  const std::size_t n = x0.size ();
  if (n == 0)
    {
      throw shape_error ("minimize_dogleg requires non-empty x0");
    }
  nd_tensor<double> x = x0.copy ();
  double delta = 1.0;
  const double delta_max = 10.0;
  for (std::size_t it = 0; it < max_iter; ++it)
    {
      const double fx = f (x);
      const nd_tensor<double> g = detail::finite_diff_gradient (f, x);
      double g_norm = linalg::norm (g);
      if (g_norm <= gtol)
        {
          return {x, fx, g, it, true};
        }
      nd_tensor<double> h (shape_type {n, n}, 0.0);
      const double eps_h = 1e-5;
      for (std::size_t i = 0; i < n; ++i)
        {
          nd_tensor<double> xp = x.copy ();
          nd_tensor<double> xm = x.copy ();
          xp[i] += eps_h;
          xm[i] -= eps_h;
          const nd_tensor<double> gp = detail::finite_diff_gradient (f, xp);
          const nd_tensor<double> gm = detail::finite_diff_gradient (f, xm);
          for (std::size_t j = 0; j < n; ++j)
            {
              h (i, j) = (gp[j] - gm[j]) / (2.0 * eps_h);
            }
          h (i, i) += 1e-8;
        }
      nd_tensor<double> p_u (shape_type {n});
      const nd_tensor<double> hg = tensor::matmul (h, g);
      const double g_hg = tensor::dot (g, hg);
      const double alpha_u =
        (g_hg > 1e-14) ? (g_norm * g_norm / g_hg) : (delta / g_norm);
      for (std::size_t i = 0; i < n; ++i)
        {
          p_u[i] = -alpha_u * g[i];
        }
      nd_tensor<double> p_b = p_u.copy ();
      try
        {
          p_b = -1.0 * linalg::solve (h, g);
        }
      catch (...)
        {
        }
      nd_tensor<double> p (shape_type {n});
      if (linalg::norm (p_b) <= delta)
        {
          p = p_b;
        }
      else if (linalg::norm (p_u) >= delta)
        {
          p = (delta / linalg::norm (p_u)) * p_u;
        }
      else
        {
          const nd_tensor<double> diff = p_b - p_u;
          const double a = tensor::dot (diff, diff);
          const double b = 2.0 * tensor::dot (p_u, diff);
          const double c = tensor::dot (p_u, p_u) - delta * delta;
          const double tau =
            (-b + std::sqrt (std::max (0.0, b * b - 4.0 * a * c)))
            / (2.0 * std::max (1e-15, a));
          p = p_u + std::clamp (tau, 0.0, 1.0) * diff;
        }
      const nd_tensor<double> x_trial = x + p;
      const double f_trial = f (x_trial);
      const double actual = fx - f_trial;
      const double pred =
        -(tensor::dot (g, p) + 0.5 * tensor::dot (p, tensor::matmul (h, p)));
      const double rho = (pred > 1e-15) ? (actual / pred) : 0.0;
      if (rho < 0.25)
        {
          delta *= 0.25;
        }
      else if (rho > 0.75 && std::abs (linalg::norm (p) - delta) < 1e-8)
        {
          delta = std::min (2.0 * delta, delta_max);
        }
      if (rho > 0.05)
        {
          x = x_trial;
        }
    }
  return {x, f (x), detail::finite_diff_gradient (f, x), max_iter, true};
}

template <typename Func>
optimize_result
shgo (Func &&f,
      const std::vector<std::pair<double, double>> &bounds,
      std::size_t n_points = 64)
{
  const std::size_t dim = bounds.size ();
  if (dim == 0)
    {
      throw shape_error ("shgo requires non-empty bounds");
    }
  static const std::size_t primes[16] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53
  };
  std::vector<nd_tensor<double>> pts;
  pts.reserve (n_points);
  std::vector<double> vals (n_points, 0.0);
  std::size_t best_idx = 0;
  for (std::size_t i = 0; i < n_points; ++i)
    {
      nd_tensor<double> pt (shape_type {dim});
      for (std::size_t d = 0; d < dim; ++d)
        {
          const std::size_t base = primes[d % 16];
          double frac = 1.0;
          double r = 0.0;
          std::size_t idx = i + 1;
          while (idx > 0)
            {
              frac /= static_cast<double> (base);
              r += frac * static_cast<double> (idx % base);
              idx /= base;
            }
          pt[d] = bounds[d].first + r * (bounds[d].second - bounds[d].first);
        }
      vals[i] = f (pt);
      if (vals[i] < vals[best_idx])
        {
          best_idx = i;
        }
      pts.push_back (std::move (pt));
    }
  return minimize_lbfgsb (f, pts[best_idx], bounds);
}

template <typename Func>
optimize_result
direct (Func &&f,
        const std::vector<std::pair<double, double>> &bounds,
        std::size_t max_iter = 60)
{
  const std::size_t dim = bounds.size ();
  if (dim == 0)
    {
      throw shape_error ("direct requires non-empty bounds");
    }
  struct box
  {
    std::vector<double> center;
    std::vector<double> half;
    double fval;
  };
  std::vector<double> c0 (dim);
  std::vector<double> h0 (dim);
  for (std::size_t d = 0; d < dim; ++d)
    {
      c0[d] = 0.5 * (bounds[d].first + bounds[d].second);
      h0[d] = 0.5 * (bounds[d].second - bounds[d].first);
    }
  std::vector<box> boxes;
  boxes.push_back ({c0, h0, f (nd_tensor<double> (shape_type {dim}, c0))});
  for (std::size_t it = 0; it < max_iter; ++it)
    {
      std::size_t best_b = 0;
      for (std::size_t i = 1; i < boxes.size (); ++i)
        {
          if (boxes[i].fval < boxes[best_b].fval)
            {
              best_b = i;
            }
        }
      box cur = boxes[best_b];
      std::size_t split_d = 0;
      for (std::size_t d = 1; d < dim; ++d)
        {
          if (cur.half[d] > cur.half[split_d])
            {
              split_d = d;
            }
        }
      const double step = cur.half[split_d] / 3.0;
      cur.half[split_d] = step;
      boxes[best_b] = cur;
      for (double sgn : {-2.0, 2.0})
        {
          box child = cur;
          child.center[split_d] += sgn * step;
          child.fval = f (nd_tensor<double> (shape_type {dim}, child.center));
          boxes.push_back (std::move (child));
        }
    }
  std::size_t best_b = 0;
  for (std::size_t i = 1; i < boxes.size (); ++i)
    {
      if (boxes[i].fval < boxes[best_b].fval)
        {
          best_b = i;
        }
    }
  return minimize_lbfgsb (
    f, nd_tensor<double> (shape_type {dim}, boxes[best_b].center), bounds);
}

template <typename Func>
optimize_result
brute (Func &&f,
       const std::vector<std::pair<double, double>> &ranges,
       std::size_t ns = 15)
{
  const std::size_t dim = ranges.size ();
  if (dim == 0 || ns < 2)
    {
      throw value_error ("brute requires non-empty ranges and ns >= 2");
    }
  std::size_t total = 1;
  for (std::size_t d = 0; d < dim; ++d)
    {
      total *= ns;
    }
  nd_tensor<double> best_x (shape_type {dim}, 0.0);
  double best_f = constants::inf;
  nd_tensor<double> pt (shape_type {dim});
  for (std::size_t idx = 0; idx < total; ++idx)
    {
      std::size_t rem = idx;
      for (std::size_t d = 0; d < dim; ++d)
        {
          const std::size_t k = rem % ns;
          rem /= ns;
          const double t =
            static_cast<double> (k) / static_cast<double> (ns - 1);
          pt[d] = ranges[d].first + t * (ranges[d].second - ranges[d].first);
        }
      const double val = f (pt);
      if (val < best_f)
        {
          best_f = val;
          best_x = pt.copy ();
        }
    }
  return minimize_nelder_mead (f, best_x);
}

template <typename VecFunc>
nd_tensor<double>
broyden1 (VecFunc &&fun,
          const nd_tensor<double> &x0,
          double tol = 1e-8,
          std::size_t max_iter = 100)
{
  const std::size_t n = x0.size ();
  if (n == 0)
    {
      throw shape_error ("broyden1 requires non-empty x0");
    }
  nd_tensor<double> x = x0.copy ();
  nd_tensor<double> fx = fun (x);
  nd_tensor<double> b_inv = nd_tensor<double>::eye (n);
  for (std::size_t j = 0; j < n; ++j)
    {
      const double h = 1e-7 * std::max (1.0, std::abs (x[j]));
      nd_tensor<double> xp = x.copy ();
      xp[j] += h;
      const nd_tensor<double> fp = fun (xp);
      for (std::size_t i = 0; i < n; ++i)
        {
          b_inv (i, j) = (fp[i] - fx[i]) / h;
        }
    }
  b_inv = linalg::inv (b_inv);
  for (std::size_t it = 0; it < max_iter; ++it)
    {
      if (linalg::norm (fx) <= tol)
        {
          return x;
        }
      const nd_tensor<double> dx = -1.0 * tensor::matmul (b_inv, fx);
      const nd_tensor<double> x_next = x + dx;
      const nd_tensor<double> f_next = fun (x_next);
      const nd_tensor<double> df = f_next - fx;
      const nd_tensor<double> b_df = tensor::matmul (b_inv, df);
      const double denom = tensor::dot (dx, b_df);
      if (std::abs (denom) > 1e-14)
        {
          const nd_tensor<double> num_vec = dx - b_df;
          const nd_tensor<double> dx_t_b = tensor::matmul (dx, b_inv);
          for (std::size_t i = 0; i < n; ++i)
            {
              for (std::size_t j = 0; j < n; ++j)
                {
                  b_inv (i, j) += (num_vec[i] * dx_t_b[j]) / denom;
                }
            }
        }
      x = x_next;
      fx = f_next;
    }
  return x;
}

struct qap_result
{
  nd_tensor<std::size_t> col_ind;
  double fun {};
};

inline qap_result
quadratic_assignment (const nd_tensor<double> &a, const nd_tensor<double> &b)
{
  if (a.ndim () != 2 || b.ndim () != 2
      || a.shape () != b.shape ()
      || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("quadratic_assignment requires matching square matrices A and B");
    }
  const std::size_t n = a.shape ()[0];
  std::vector<std::size_t> perm (n);
  std::iota (perm.begin (), perm.end (), 0);
  auto eval_obj = [&] (const std::vector<std::size_t> &p) -> double {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      {
        for (std::size_t j = 0; j < n; ++j)
          {
            sum += a (i, j) * b (p[i], p[j]);
          }
      }
    return sum;
  };
  double best_val = eval_obj (perm);
  bool improved = true;
  while (improved)
    {
      improved = false;
      for (std::size_t i = 0; i < n; ++i)
        {
          for (std::size_t j = i + 1; j < n; ++j)
            {
              std::swap (perm[i], perm[j]);
              const double cand = eval_obj (perm);
              if (cand + 1e-12 < best_val)
                {
                  best_val = cand;
                  improved = true;
                }
              else
                {
                  std::swap (perm[i], perm[j]);
                }
            }
        }
    }
  return {nd_tensor<std::size_t> (shape_type {n}, std::move (perm)), best_val};
}

} // namespace hyper_math::scientific::optimize

#endif /* HYPER_MATH_SCIENTIFIC_OPTIMIZE_HPP */
