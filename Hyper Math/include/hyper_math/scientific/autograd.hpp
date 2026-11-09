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

#ifndef HYPER_MATH_SCIENTIFIC_AUTOGRAD_HPP
#define HYPER_MATH_SCIENTIFIC_AUTOGRAD_HPP

#include "hyper_math/core/config.hpp"
#include "hyper_math/scientific/linalg.hpp"
#include "hyper_math/scientific/optimize.hpp"
#include "hyper_math/tensor/nd_tensor.hpp"

#include <cmath>
#include <cstddef>
#include <ostream>
#include <vector>

namespace hyper_math::scientific::autograd
{

using tensor::nd_tensor;
using tensor::shape_type;

/* =========================================================================
   1. Forward-Mode Automatic Differentiation via Dual Numbers (a + b * eps)
   ========================================================================= */

class dual_number
{
public:
  constexpr
  dual_number (double val = 0.0, double dot = 0.0) noexcept
    : val_ (val), dot_ (dot)
  {
  }

  HYPER_MATH_NODISCARD constexpr double
  val () const noexcept
  {
    return val_;
  }

  HYPER_MATH_NODISCARD constexpr double
  dot () const noexcept
  {
    return dot_;
  }

  constexpr dual_number
  operator- () const noexcept
  {
    return dual_number (-val_, -dot_);
  }

  constexpr dual_number &
  operator+= (const dual_number &rhs) noexcept
  {
    val_ += rhs.val_;
    dot_ += rhs.dot_;
    return *this;
  }

  constexpr dual_number &
  operator-= (const dual_number &rhs) noexcept
  {
    val_ -= rhs.val_;
    dot_ -= rhs.dot_;
    return *this;
  }

  constexpr dual_number &
  operator*= (const dual_number &rhs) noexcept
  {
    dot_ = val_ * rhs.dot_ + dot_ * rhs.val_;
    val_ *= rhs.val_;
    return *this;
  }

  constexpr dual_number &
  operator/= (const dual_number &rhs) noexcept
  {
    const double inv = 1.0 / rhs.val_;
    dot_ = (dot_ * rhs.val_ - val_ * rhs.dot_) * inv * inv;
    val_ *= inv;
    return *this;
  }

private:
  double val_;
  double dot_;
};

inline constexpr dual_number
operator+ (const dual_number &a, const dual_number &b) noexcept
{
  return dual_number (a.val () + b.val (), a.dot () + b.dot ());
}

inline constexpr dual_number
operator- (const dual_number &a, const dual_number &b) noexcept
{
  return dual_number (a.val () - b.val (), a.dot () - b.dot ());
}

inline constexpr dual_number
operator* (const dual_number &a, const dual_number &b) noexcept
{
  return dual_number (a.val () * b.val (),
                      a.val () * b.dot () + a.dot () * b.val ());
}

inline constexpr dual_number
operator/ (const dual_number &a, const dual_number &b) noexcept
{
  const double inv = 1.0 / b.val ();
  return dual_number (a.val () * inv,
                      (a.dot () * b.val () - a.val () * b.dot ()) * inv * inv);
}

inline constexpr bool
operator== (const dual_number &a, const dual_number &b) noexcept
{
  return a.val () == b.val ();
}

inline constexpr bool
operator< (const dual_number &a, const dual_number &b) noexcept
{
  return a.val () < b.val ();
}

inline constexpr bool
operator> (const dual_number &a, const dual_number &b) noexcept
{
  return a.val () > b.val ();
}

inline dual_number
sin (const dual_number &x) noexcept
{
  return dual_number (std::sin (x.val ()), x.dot () * std::cos (x.val ()));
}

inline dual_number
cos (const dual_number &x) noexcept
{
  return dual_number (std::cos (x.val ()), -x.dot () * std::sin (x.val ()));
}

inline dual_number
tan (const dual_number &x) noexcept
{
  const double c = std::cos (x.val ());
  return dual_number (std::tan (x.val ()), x.dot () / (c * c));
}

inline dual_number
exp (const dual_number &x) noexcept
{
  const double e = std::exp (x.val ());
  return dual_number (e, x.dot () * e);
}

inline dual_number
log (const dual_number &x) noexcept
{
  return dual_number (std::log (x.val ()), x.dot () / x.val ());
}

inline dual_number
sqrt (const dual_number &x) noexcept
{
  const double s = std::sqrt (x.val ());
  return dual_number (s, 0.5 * x.dot () / s);
}

inline dual_number
pow (const dual_number &x, double p) noexcept
{
  return dual_number (std::pow (x.val (), p),
                      p * std::pow (x.val (), p - 1.0) * x.dot ());
}

inline dual_number
pow (const dual_number &x, const dual_number &y) noexcept
{
  const double v = std::pow (x.val (), y.val ());
  const double dx_term =
    (x.dot () == 0.0)
    ? 0.0
    : (y.val () * std::pow (x.val (), y.val () - 1.0) * x.dot ());
  const double dy_term =
    (y.dot () == 0.0) ? 0.0 : (v * std::log (x.val ()) * y.dot ());
  return dual_number (v, dx_term + dy_term);
}

inline dual_number
abs (const dual_number &x) noexcept
{
  const double sgn = (x.val () >= 0.0) ? 1.0 : -1.0;
  return dual_number (std::abs (x.val ()), sgn * x.dot ());
}

inline dual_number
tanh (const dual_number &x) noexcept
{
  const double th = std::tanh (x.val ());
  return dual_number (th, x.dot () * (1.0 - th * th));
}

inline std::ostream &
operator<< (std::ostream &os, const dual_number &d)
{
  os << d.val () << (d.dot () >= 0.0 ? "+" : "") << d.dot () << "eps";
  return os;
}

/* Exact scalar derivative f'(x) via forward-mode dual numbers.  */
template <typename Fn>
double
derivative (Fn &&fn, double x)
{
  const dual_number res = fn (dual_number (x, 1.0));
  return res.dot ();
}

/* Exact gradient of f : R^n -> R via forward-mode dual seeding.  */
template <typename Fn>
nd_tensor<double>
forward_grad (Fn &&fn, const nd_tensor<double> &x)
{
  const std::size_t n = x.size ();
  std::vector<dual_number> dual_x (n);
  for (std::size_t i = 0; i < n; ++i)
    {
      dual_x[i] = dual_number (x.eval_at (i), 0.0);
    }

  nd_tensor<double> g (shape_type {n});
  for (std::size_t i = 0; i < n; ++i)
    {
      dual_x[i] = dual_number (x.eval_at (i), 1.0);
      const dual_number y = fn (dual_x);
      g[i] = y.dot ();
      dual_x[i] = dual_number (x.eval_at (i), 0.0);
    }
  return g;
}

/* Exact Jacobian matrix J_ij = dF_i / dx_j for F : R^n -> R^m.  */
template <typename Fn>
nd_tensor<double>
jacobian (Fn &&fn, const nd_tensor<double> &x)
{
  const std::size_t n = x.size ();
  std::vector<dual_number> dual_x (n);
  for (std::size_t j = 0; j < n; ++j)
    {
      dual_x[j] = dual_number (x.eval_at (j), 0.0);
    }

  /* Probe output dimension on first directional pass.  */
  if (n == 0)
    {
      throw shape_error ("jacobian requires non-empty input vector");
    }
  dual_x[0] = dual_number (x.eval_at (0), 1.0);
  const std::vector<dual_number> col0 = fn (dual_x);
  dual_x[0] = dual_number (x.eval_at (0), 0.0);

  const std::size_t m = col0.size ();
  nd_tensor<double> jac (shape_type {m, n}, 0.0);
  for (std::size_t i = 0; i < m; ++i)
    {
      jac (i, 0) = col0[i].dot ();
    }

  for (std::size_t j = 1; j < n; ++j)
    {
      dual_x[j] = dual_number (x.eval_at (j), 1.0);
      const std::vector<dual_number> col_j = fn (dual_x);
      for (std::size_t i = 0; i < m; ++i)
        {
          jac (i, j) = col_j[i].dot ();
        }
      dual_x[j] = dual_number (x.eval_at (j), 0.0);
    }
  return jac;
}

/* =========================================================================
   2. Reverse-Mode Automatic Differentiation (Wengert Computation Tape)
   ========================================================================= */

struct tape_node
{
  double val = 0.0;
  double adj = 0.0;
  std::ptrdiff_t p0 = -1;
  std::ptrdiff_t p1 = -1;
  double w0 = 0.0;
  double w1 = 0.0;
};

class tape
{
public:
  tape () = default;

  std::size_t
  push_leaf (double v)
  {
    const std::size_t idx = nodes_.size ();
    nodes_.push_back (tape_node {v, 0.0, -1, -1, 0.0, 0.0});
    return idx;
  }

  std::size_t
  push_unary (double v, std::size_t p0, double w0)
  {
    const std::size_t idx = nodes_.size ();
    nodes_.push_back (tape_node {
      v, 0.0, static_cast<std::ptrdiff_t> (p0), -1, w0, 0.0
    });
    return idx;
  }

  std::size_t
  push_binary (double v, std::size_t p0, double w0, std::size_t p1, double w1)
  {
    const std::size_t idx = nodes_.size ();
    nodes_.push_back (tape_node {
      v,
      0.0,
      static_cast<std::ptrdiff_t> (p0),
      static_cast<std::ptrdiff_t> (p1),
      w0,
      w1
    });
    return idx;
  }

  void
  backward (std::size_t root_idx)
  {
    for (auto &nd : nodes_)
      {
        nd.adj = 0.0;
      }
    if (root_idx >= nodes_.size ())
      {
        throw index_error ("autograd::tape::backward root index out of range");
      }
    nodes_[root_idx].adj = 1.0;

    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t> (root_idx);
         i >= 0; --i)
      {
        const tape_node &cur = nodes_[static_cast<std::size_t> (i)];
        if (cur.adj == 0.0)
          {
            continue;
          }
        if (cur.p0 >= 0)
          {
            nodes_[static_cast<std::size_t> (cur.p0)].adj += cur.adj * cur.w0;
          }
        if (cur.p1 >= 0)
          {
            nodes_[static_cast<std::size_t> (cur.p1)].adj += cur.adj * cur.w1;
          }
      }
  }

  HYPER_MATH_NODISCARD double
  val (std::size_t idx) const
  {
    return nodes_[idx].val;
  }

  HYPER_MATH_NODISCARD double
  adj (std::size_t idx) const
  {
    return nodes_[idx].adj;
  }

  HYPER_MATH_NODISCARD std::size_t
  size () const noexcept
  {
    return nodes_.size ();
  }

private:
  std::vector<tape_node> nodes_;
};

class var
{
public:
  var () noexcept
    : tape_ (nullptr), idx_ (0)
  {
  }

  var (tape &t, double v)
    : tape_ (&t), idx_ (t.push_leaf (v))
  {
  }

  var (tape *t, std::size_t idx) noexcept
    : tape_ (t), idx_ (idx)
  {
  }

  HYPER_MATH_NODISCARD double
  val () const
  {
    return tape_->val (idx_);
  }

  HYPER_MATH_NODISCARD double
  grad () const
  {
    return tape_->adj (idx_);
  }

  HYPER_MATH_NODISCARD tape *
  tape_ptr () const noexcept
  {
    return tape_;
  }

  HYPER_MATH_NODISCARD std::size_t
  index () const noexcept
  {
    return idx_;
  }

  void
  backward () const
  {
    tape_->backward (idx_);
  }

  var
  operator- () const
  {
    return var (tape_, tape_->push_unary (-val (), idx_, -1.0));
  }

  var &
  operator+= (const var &rhs);

  var &
  operator-= (const var &rhs);

  var &
  operator*= (const var &rhs);

  var &
  operator/= (const var &rhs);

private:
  tape *tape_;
  std::size_t idx_;
};

inline var
operator+ (const var &a, const var &b)
{
  tape *t = a.tape_ptr ();
  return var (t, t->push_binary (a.val () + b.val (),
                                 a.index (), 1.0,
                                 b.index (), 1.0));
}

inline var
operator+ (const var &a, double b)
{
  tape *t = a.tape_ptr ();
  return var (t, t->push_unary (a.val () + b, a.index (), 1.0));
}

inline var
operator+ (double a, const var &b)
{
  return b + a;
}

inline var
operator- (const var &a, const var &b)
{
  tape *t = a.tape_ptr ();
  return var (t, t->push_binary (a.val () - b.val (),
                                 a.index (), 1.0,
                                 b.index (), -1.0));
}

inline var
operator- (const var &a, double b)
{
  tape *t = a.tape_ptr ();
  return var (t, t->push_unary (a.val () - b, a.index (), 1.0));
}

inline var
operator- (double a, const var &b)
{
  tape *t = b.tape_ptr ();
  return var (t, t->push_unary (a - b.val (), b.index (), -1.0));
}

inline var
operator* (const var &a, const var &b)
{
  tape *t = a.tape_ptr ();
  return var (t, t->push_binary (a.val () * b.val (),
                                 a.index (), b.val (),
                                 b.index (), a.val ()));
}

inline var
operator* (const var &a, double b)
{
  tape *t = a.tape_ptr ();
  return var (t, t->push_unary (a.val () * b, a.index (), b));
}

inline var
operator* (double a, const var &b)
{
  return b * a;
}

inline var
operator/ (const var &a, const var &b)
{
  tape *t = a.tape_ptr ();
  const double inv = 1.0 / b.val ();
  return var (t, t->push_binary (a.val () * inv,
                                 a.index (), inv,
                                 b.index (), -a.val () * inv * inv));
}

inline var
operator/ (const var &a, double b)
{
  return a * (1.0 / b);
}

inline var
operator/ (double a, const var &b)
{
  tape *t = b.tape_ptr ();
  const double inv = 1.0 / b.val ();
  return var (t, t->push_unary (a * inv, b.index (), -a * inv * inv));
}

inline var &
var::operator+= (const var &rhs)
{
  *this = *this + rhs;
  return *this;
}

inline var &
var::operator-= (const var &rhs)
{
  *this = *this - rhs;
  return *this;
}

inline var &
var::operator*= (const var &rhs)
{
  *this = *this * rhs;
  return *this;
}

inline var &
var::operator/= (const var &rhs)
{
  *this = *this / rhs;
  return *this;
}

inline var
sin (const var &x)
{
  tape *t = x.tape_ptr ();
  return var (t, t->push_unary (std::sin (x.val ()),
                                x.index (),
                                std::cos (x.val ())));
}

inline var
cos (const var &x)
{
  tape *t = x.tape_ptr ();
  return var (t, t->push_unary (std::cos (x.val ()),
                                x.index (),
                                -std::sin (x.val ())));
}

inline var
tan (const var &x)
{
  tape *t = x.tape_ptr ();
  const double c = std::cos (x.val ());
  return var (t, t->push_unary (std::tan (x.val ()),
                                x.index (),
                                1.0 / (c * c)));
}

inline var
exp (const var &x)
{
  tape *t = x.tape_ptr ();
  const double e = std::exp (x.val ());
  return var (t, t->push_unary (e, x.index (), e));
}

inline var
log (const var &x)
{
  tape *t = x.tape_ptr ();
  return var (t, t->push_unary (std::log (x.val ()),
                                x.index (),
                                1.0 / x.val ()));
}

inline var
sqrt (const var &x)
{
  tape *t = x.tape_ptr ();
  const double s = std::sqrt (x.val ());
  return var (t, t->push_unary (s, x.index (), 0.5 / s));
}

inline var
pow (const var &x, double p)
{
  tape *t = x.tape_ptr ();
  return var (t, t->push_unary (std::pow (x.val (), p),
                                x.index (),
                                p * std::pow (x.val (), p - 1.0)));
}

inline var
pow (const var &x, const var &y)
{
  tape *t = x.tape_ptr ();
  const double v = std::pow (x.val (), y.val ());
  const double dx = y.val () * std::pow (x.val (), y.val () - 1.0);
  const double dy = (x.val () > 0.0) ? (v * std::log (x.val ())) : 0.0;
  return var (t, t->push_binary (v, x.index (), dx, y.index (), dy));
}

inline var
abs (const var &x)
{
  tape *t = x.tape_ptr ();
  const double sgn = (x.val () >= 0.0) ? 1.0 : -1.0;
  return var (t, t->push_unary (std::abs (x.val ()), x.index (), sgn));
}

inline var
tanh (const var &x)
{
  tape *t = x.tape_ptr ();
  const double th = std::tanh (x.val ());
  return var (t, t->push_unary (th, x.index (), 1.0 - th * th));
}

inline var
sigmoid (const var &x)
{
  tape *t = x.tape_ptr ();
  const double s = 1.0 / (1.0 + std::exp (-x.val ()));
  return var (t, t->push_unary (s, x.index (), s * (1.0 - s)));
}

inline var
relu (const var &x)
{
  tape *t = x.tape_ptr ();
  const double v = (x.val () > 0.0) ? x.val () : 0.0;
  const double w = (x.val () > 0.0) ? 1.0 : 0.0;
  return var (t, t->push_unary (v, x.index (), w));
}

struct value_and_grad_result
{
  double value = 0.0;
  nd_tensor<double> gradient;
};

/* Evaluate scalar objective f(x) and its exact reverse-mode gradient in one
   forward + backward pass over the Wengert computation tape.  */
template <typename Fn>
value_and_grad_result
value_and_grad (Fn &&fn, const nd_tensor<double> &x)
{
  tape tp;
  const std::size_t n = x.size ();
  std::vector<var> vars;
  vars.reserve (n);
  for (std::size_t i = 0; i < n; ++i)
    {
      vars.emplace_back (tp, x.eval_at (i));
    }

  const var out = fn (vars);
  out.backward ();

  nd_tensor<double> g (shape_type {n});
  for (std::size_t i = 0; i < n; ++i)
    {
      g[i] = vars[i].grad ();
    }
  return value_and_grad_result {out.val (), std::move (g)};
}

template <typename Fn>
nd_tensor<double>
grad (Fn &&fn, const nd_tensor<double> &x)
{
  return value_and_grad (std::forward<Fn> (fn), x).gradient;
}

/* Exact Hessian matrix H_ij = d^2 f / (dx_i dx_j) via central directional
   differences of exact reverse-mode analytical gradients.  */
template <typename Fn>
nd_tensor<double>
hessian (Fn &&fn, const nd_tensor<double> &x, double eps = 1e-6)
{
  const std::size_t n = x.size ();
  nd_tensor<double> h (shape_type {n, n}, 0.0);
  nd_tensor<double> xp = x.copy ();
  nd_tensor<double> xm = x.copy ();

  for (std::size_t j = 0; j < n; ++j)
    {
      const double xj = x.eval_at (j);
      xp[j] = xj + eps;
      xm[j] = xj - eps;
      const nd_tensor<double> gp = grad (fn, xp);
      const nd_tensor<double> gm = grad (fn, xm);
      const double inv_2eps = 0.5 / eps;
      for (std::size_t i = 0; i < n; ++i)
        {
          h (i, j) = (gp[i] - gm[i]) * inv_2eps;
        }
      xp[j] = xj;
      xm[j] = xj;
    }

  /* Symmetrize numerical rounding at ~1e-13 level.  */
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = i + 1; j < n; ++j)
        {
          const double avg = 0.5 * (h (i, j) + h (j, i));
          h (i, j) = avg;
          h (j, i) = avg;
        }
    }
  return h;
}

/* Multivariable BFGS Quasi-Newton Optimizer powered by exact Reverse-Mode
   Automatic Differentiation gradients (zero finite-difference error).  */
template <typename Fn>
optimize::optimize_result
minimize_bfgs_autograd (Fn &&fn,
                        const nd_tensor<double> &x0,
                        double gtol = 1e-8,
                        std::size_t max_iter = 500)
{
  const std::size_t n = x0.size ();
  nd_tensor<double> x = x0.copy ();
  auto vg = value_and_grad (fn, x);
  double fx = vg.value;
  nd_tensor<double> g = std::move (vg.gradient);
  nd_tensor<double> h_inv = tensor::eye<double> (n);

  std::size_t iter = 0;
  for (; iter < max_iter; ++iter)
    {
      if (linalg::norm (g) < gtol)
        {
          return optimize::optimize_result {x, fx, g, iter, true};
        }

      nd_tensor<double> p = -tensor::matmul (h_inv, g);
      double dir_deriv = tensor::dot (g, p);
      if (dir_deriv >= 0.0)
        {
          h_inv = tensor::eye<double> (n);
          p = -g;
          dir_deriv = -tensor::dot (g, g);
        }

      double alpha = 1.0;
      constexpr double c1 = 1e-4;
      nd_tensor<double> x_next = x + alpha * p;
      auto vg_next = value_and_grad (fn, x_next);
      for (int ls = 0;
           ls < 40 && vg_next.value > fx + c1 * alpha * dir_deriv;
           ++ls)
        {
          alpha *= 0.5;
          x_next = x + alpha * p;
          vg_next = value_and_grad (fn, x_next);
        }

      const nd_tensor<double> s = x_next - x;
      const nd_tensor<double> y = vg_next.gradient - g;
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
      fx = vg_next.value;
      g = std::move (vg_next.gradient);
    }

  return optimize::optimize_result {x, fx, g, iter, linalg::norm (g) < gtol};
}

} // namespace hyper_math::scientific::autograd

#endif /* HYPER_MATH_SCIENTIFIC_AUTOGRAD_HPP */
