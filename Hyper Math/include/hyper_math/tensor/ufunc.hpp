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

#ifndef HYPER_MATH_TENSOR_UFUNC_HPP
#define HYPER_MATH_TENSOR_UFUNC_HPP

#include "hyper_math/tensor/nd_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numeric>
#include <type_traits>

namespace hyper_math::tensor
{

/* Scalar broadcast node for lazy expression templates.  */
template <typename T>
class scalar_expr : public tensor_expr_base<scalar_expr<T>, T>
{
private:
  T value_;
  shape_type shape_;

public:
  using value_type = T;

  explicit
  scalar_expr (T val)
    : value_ (val), shape_ {1}
  {
  }

  const shape_type &
  shape () const noexcept
  {
    return shape_;
  }

  std::size_t
  size () const noexcept
  {
    return 1;
  }

  HYPER_MATH_ALWAYS_INLINE T
  eval_at (std::size_t) const noexcept
  {
    return value_;
  }

  HYPER_MATH_ALWAYS_INLINE T
  eval_with_lattice (std::size_t, const lattice_descriptor &) const noexcept
  {
    return value_;
  }

  lattice_descriptor
  broadcast_lattice (const shape_type &target_shape) const
  {
    return lattice_descriptor (target_shape,
                               stride_type (target_shape.size (), 0),
                               0);
  }
};

/* Unary lazily fused expression node.  */
template <typename Expr, typename UnaryOp>
class unary_tensor_expr
  : public tensor_expr_base<
      unary_tensor_expr<Expr, UnaryOp>,
      std::invoke_result_t<UnaryOp, typename Expr::value_type>>
{
public:
  using value_type =
    std::invoke_result_t<UnaryOp, typename Expr::value_type>;

private:
  Expr expr_;
  UnaryOp op_;

public:
  unary_tensor_expr (Expr e, UnaryOp op)
    : expr_ (std::move (e)), op_ (std::move (op))
  {
  }

  const shape_type &
  shape () const noexcept
  {
    return expr_.shape ();
  }

  std::size_t
  size () const noexcept
  {
    return expr_.size ();
  }

  HYPER_MATH_ALWAYS_INLINE value_type
  eval_at (std::size_t flat_idx) const noexcept
  {
    return op_ (expr_.eval_at (flat_idx));
  }

  HYPER_MATH_ALWAYS_INLINE value_type
  eval_with_lattice (std::size_t flat_idx,
                     const lattice_descriptor &bcast_lat) const noexcept
  {
    return op_ (expr_.eval_with_lattice (flat_idx, bcast_lat));
  }

  lattice_descriptor
  broadcast_lattice (const shape_type &target_shape) const
  {
    return expr_.broadcast_lattice (target_shape);
  }
};

/* Binary lazily fused expression node with automatic N-D broadcasting.  */
template <typename LhsExpr, typename RhsExpr, typename BinaryOp>
class binary_tensor_expr
  : public tensor_expr_base<
      binary_tensor_expr<LhsExpr, RhsExpr, BinaryOp>,
      std::invoke_result_t<BinaryOp,
                           typename LhsExpr::value_type,
                           typename RhsExpr::value_type>>
{
public:
  using value_type =
    std::invoke_result_t<BinaryOp,
                         typename LhsExpr::value_type,
                         typename RhsExpr::value_type>;

private:
  LhsExpr lhs_;
  RhsExpr rhs_;
  BinaryOp op_;
  shape_type shape_;
  lattice_descriptor lhs_lat_;
  lattice_descriptor rhs_lat_;
  std::size_t total_size_;
  bool same_shape_;

public:
  binary_tensor_expr (LhsExpr l, RhsExpr r, BinaryOp op)
    : lhs_ (std::move (l)),
      rhs_ (std::move (r)),
      op_ (std::move (op)),
      shape_ (broadcast_shapes (lhs_.shape (), rhs_.shape ())),
      lhs_lat_ (lattice_descriptor (lhs_.shape ()).broadcast_to (shape_)),
      rhs_lat_ (lattice_descriptor (rhs_.shape ()).broadcast_to (shape_)),
      total_size_ (lhs_lat_.size ()),
      same_shape_ (lhs_.shape () == rhs_.shape ())
  {
  }

  const shape_type &
  shape () const noexcept
  {
    return shape_;
  }

  std::size_t
  size () const noexcept
  {
    return total_size_;
  }

  HYPER_MATH_ALWAYS_INLINE value_type
  eval_at (std::size_t flat_idx) const noexcept
  {
    if (same_shape_)
      {
        return op_ (lhs_.eval_at (flat_idx), rhs_.eval_at (flat_idx));
      }
    const std::size_t l_idx =
      static_cast<std::size_t> (lhs_lat_.flat_to_physical (flat_idx));
    const std::size_t r_idx =
      static_cast<std::size_t> (rhs_lat_.flat_to_physical (flat_idx));
    return op_ (lhs_.eval_at (l_idx), rhs_.eval_at (r_idx));
  }

  HYPER_MATH_ALWAYS_INLINE value_type
  eval_with_lattice (std::size_t flat_idx,
                     const lattice_descriptor &bcast_lat) const noexcept
  {
    const std::size_t mapped =
      static_cast<std::size_t> (bcast_lat.flat_to_physical (flat_idx));
    return eval_at (mapped);
  }

  lattice_descriptor
  broadcast_lattice (const shape_type &target_shape) const
  {
    return lattice_descriptor (shape_).broadcast_to (target_shape);
  }
};

/* Helper factory functions for expression nodes.  */
template <typename Derived, typename Val, typename UnaryOp>
auto
make_unary_expr (const tensor_expr_base<Derived, Val> &expr, UnaryOp op)
{
  return unary_tensor_expr<Derived, UnaryOp> (expr.derived (), std::move (op));
}

template <typename LDerived, typename LVal,
          typename RDerived, typename RVal,
          typename BinaryOp>
auto
make_binary_expr (const tensor_expr_base<LDerived, LVal> &lhs,
                  const tensor_expr_base<RDerived, RVal> &rhs,
                  BinaryOp op)
{
  return binary_tensor_expr<LDerived, RDerived, BinaryOp> (
    lhs.derived (), rhs.derived (), std::move (op));
}

/* Arithmetic operators (tensor-tensor, tensor-scalar, scalar-tensor).  */
template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
operator+ (const tensor_expr_base<LDerived, LVal> &lhs,
           const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (auto a, auto b) { return a + b; });
}

template <typename LDerived, typename LVal, arithmetic_scalar S>
auto
operator+ (const tensor_expr_base<LDerived, LVal> &lhs, S rhs)
{
  return make_unary_expr (lhs, [rhs] (auto a) { return a + rhs; });
}

template <arithmetic_scalar S, typename RDerived, typename RVal>
auto
operator+ (S lhs, const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_unary_expr (rhs, [lhs] (auto b) { return lhs + b; });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
operator- (const tensor_expr_base<LDerived, LVal> &lhs,
           const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (auto a, auto b) { return a - b; });
}

template <typename LDerived, typename LVal, arithmetic_scalar S>
auto
operator- (const tensor_expr_base<LDerived, LVal> &lhs, S rhs)
{
  return make_unary_expr (lhs, [rhs] (auto a) { return a - rhs; });
}

template <arithmetic_scalar S, typename RDerived, typename RVal>
auto
operator- (S lhs, const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_unary_expr (rhs, [lhs] (auto b) { return lhs - b; });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
operator* (const tensor_expr_base<LDerived, LVal> &lhs,
           const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (auto a, auto b) { return a * b; });
}

template <typename LDerived, typename LVal, arithmetic_scalar S>
auto
operator* (const tensor_expr_base<LDerived, LVal> &lhs, S rhs)
{
  return make_unary_expr (lhs, [rhs] (auto a) { return a * rhs; });
}

template <arithmetic_scalar S, typename RDerived, typename RVal>
auto
operator* (S lhs, const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_unary_expr (rhs, [lhs] (auto b) { return lhs * b; });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
operator/ (const tensor_expr_base<LDerived, LVal> &lhs,
           const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (auto a, auto b) { return a / b; });
}

template <typename LDerived, typename LVal, arithmetic_scalar S>
auto
operator/ (const tensor_expr_base<LDerived, LVal> &lhs, S rhs)
{
  return make_unary_expr (lhs, [rhs] (auto a) { return a / rhs; });
}

template <arithmetic_scalar S, typename RDerived, typename RVal>
auto
operator/ (S lhs, const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_unary_expr (rhs, [lhs] (auto b) { return lhs / b; });
}

template <typename Derived, typename Val>
auto
operator- (const tensor_expr_base<Derived, Val> &expr)
{
  return make_unary_expr (expr, [] (auto a) { return -a; });
}

/* Comparison operators returning materialized boolean (uint8_t) masks.  */
#define HYPER_MATH_DEFINE_COMPARISON_OP(OP_TOKEN, EXPR_BODY)                  \
  template <typename LDerived, typename LVal, typename RDerived, typename RVal> \
  nd_tensor<std::uint8_t>                                                     \
  operator OP_TOKEN (const tensor_expr_base<LDerived, LVal> &lhs,             \
                     const tensor_expr_base<RDerived, RVal> &rhs)             \
  {                                                                           \
    return nd_tensor<std::uint8_t> (make_binary_expr (                        \
      lhs, rhs, [] (auto a, auto b) -> std::uint8_t { return (EXPR_BODY); }));\
  }                                                                           \
  template <typename LDerived, typename LVal, arithmetic_scalar S>            \
  nd_tensor<std::uint8_t>                                                     \
  operator OP_TOKEN (const tensor_expr_base<LDerived, LVal> &lhs, S b)        \
  {                                                                           \
    return nd_tensor<std::uint8_t> (make_unary_expr (                         \
      lhs, [b] (auto a) -> std::uint8_t { return (EXPR_BODY); }));            \
  }

HYPER_MATH_DEFINE_COMPARISON_OP (==, a == b ? 1 : 0)
HYPER_MATH_DEFINE_COMPARISON_OP (!=, a != b ? 1 : 0)
HYPER_MATH_DEFINE_COMPARISON_OP (<,  a < b ? 1 : 0)
HYPER_MATH_DEFINE_COMPARISON_OP (<=, a <= b ? 1 : 0)
HYPER_MATH_DEFINE_COMPARISON_OP (>,  a > b ? 1 : 0)
HYPER_MATH_DEFINE_COMPARISON_OP (>=, a >= b ? 1 : 0)

#undef HYPER_MATH_DEFINE_COMPARISON_OP

/* Universal unary mathematical functions (NumPy ufunc equivalents).  */
#define HYPER_MATH_DEFINE_UNARY_UFUNC(FUNC_NAME, STD_EXPR)                    \
  template <typename Derived, typename Val>                                   \
  auto                                                                        \
  FUNC_NAME (const tensor_expr_base<Derived, Val> &expr)                      \
  {                                                                           \
    return make_unary_expr (expr, [] (auto x) {                               \
      const double d = static_cast<double> (x);                               \
      return (STD_EXPR);                                                      \
    });                                                                       \
  }

HYPER_MATH_DEFINE_UNARY_UFUNC (sin, std::sin (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (cos, std::cos (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (tan, std::tan (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (asin, std::asin (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (acos, std::acos (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (atan, std::atan (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (sinh, std::sinh (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (cosh, std::cosh (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (tanh, std::tanh (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (asinh, std::asinh (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (acosh, std::acosh (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (atanh, std::atanh (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (rint, std::rint (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (exp, std::exp (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (expm1, std::expm1 (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (log, std::log (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (log2, std::log2 (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (log10, std::log10 (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (log1p, std::log1p (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (sqrt, std::sqrt (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (cbrt, std::cbrt (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (abs, std::abs (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (floor, std::floor (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (ceil, std::ceil (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (round, std::round (d))
HYPER_MATH_DEFINE_UNARY_UFUNC (deg2rad, d * (constants::pi / 180.0))
HYPER_MATH_DEFINE_UNARY_UFUNC (rad2deg, d * (180.0 / constants::pi))
HYPER_MATH_DEFINE_UNARY_UFUNC (sign, std::isnan (d) ? constants::nan : ((d > 0.0) ? 1.0 : ((d < 0.0) ? -1.0 : 0.0)))

#undef HYPER_MATH_DEFINE_UNARY_UFUNC

/* Binary mathematical ufuncs: pow, atan2, hypot, maximum, minimum.  */
template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
pow (const tensor_expr_base<LDerived, LVal> &lhs,
     const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (auto a, auto b) {
    return std::pow (static_cast<double> (a), static_cast<double> (b));
  });
}

template <typename LDerived, typename LVal, arithmetic_scalar S>
auto
pow (const tensor_expr_base<LDerived, LVal> &lhs, S exponent)
{
  return make_unary_expr (lhs, [exponent] (auto a) {
    return std::pow (static_cast<double> (a), static_cast<double> (exponent));
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
atan2 (const tensor_expr_base<LDerived, LVal> &y,
       const tensor_expr_base<RDerived, RVal> &x)
{
  return make_binary_expr (y, x, [] (auto a, auto b) {
    return std::atan2 (static_cast<double> (a), static_cast<double> (b));
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
hypot (const tensor_expr_base<LDerived, LVal> &x,
       const tensor_expr_base<RDerived, RVal> &y)
{
  return make_binary_expr (x, y, [] (auto a, auto b) {
    return std::hypot (static_cast<double> (a), static_cast<double> (b));
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
maximum (const tensor_expr_base<LDerived, LVal> &lhs,
         const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (auto a, auto b) {
    using common_t = std::common_type_t<decltype (a), decltype (b)>;
    if constexpr (std::is_floating_point_v<common_t>)
      {
        if (std::isnan (static_cast<double> (a))
            || std::isnan (static_cast<double> (b)))
          {
            return static_cast<common_t> (constants::nan);
          }
      }
    return std::max (static_cast<common_t> (a), static_cast<common_t> (b));
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
minimum (const tensor_expr_base<LDerived, LVal> &lhs,
         const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (auto a, auto b) {
    using common_t = std::common_type_t<decltype (a), decltype (b)>;
    if constexpr (std::is_floating_point_v<common_t>)
      {
        if (std::isnan (static_cast<double> (a))
            || std::isnan (static_cast<double> (b)))
          {
            return static_cast<common_t> (constants::nan);
          }
      }
    return std::min (static_cast<common_t> (a), static_cast<common_t> (b));
  });
}

/* Clip values to [low, high].  */
template <typename Derived, typename Val>
nd_tensor<Val>
clip (const tensor_expr_base<Derived, Val> &expr, Val low, Val high)
{
  return nd_tensor<Val> (make_unary_expr (expr, [low, high] (Val x) {
    return std::clamp (x, low, high);
  }));
}

/* Ternary conditional selection: where (cond, x, y).  */
template <typename CondDerived, typename CondVal,
          typename XDerived, typename XVal,
          typename YDerived, typename YVal>
auto
where (const tensor_expr_base<CondDerived, CondVal> &cond,
       const tensor_expr_base<XDerived, XVal> &x,
       const tensor_expr_base<YDerived, YVal> &y)
{
  using result_type = std::common_type_t<XVal, YVal>;
  const shape_type s_xy = broadcast_shapes (x.shape (), y.shape ());
  const shape_type out_shape = broadcast_shapes (cond.shape (), s_xy);
  const auto c_lat =
    lattice_descriptor (cond.shape ()).broadcast_to (out_shape);
  const auto x_lat =
    lattice_descriptor (x.shape ()).broadcast_to (out_shape);
  const auto y_lat =
    lattice_descriptor (y.shape ()).broadcast_to (out_shape);

  nd_tensor<result_type> out (out_shape);
  const std::size_t n = out.size ();
  for (std::size_t i = 0; i < n; ++i)
    {
      const std::size_t ci =
        static_cast<std::size_t> (c_lat.flat_to_physical (i));
      const bool flag = static_cast<bool> (cond.eval_at (ci));
      if (flag)
        {
          const std::size_t xi =
            static_cast<std::size_t> (x_lat.flat_to_physical (i));
          out[i] = static_cast<result_type> (x.eval_at (xi));
        }
      else
        {
          const std::size_t yi =
            static_cast<std::size_t> (y_lat.flat_to_physical (i));
          out[i] = static_cast<result_type> (y.eval_at (yi));
        }
    }
  return out;
}

template <typename CondDerived, typename CondVal, arithmetic_scalar T>
nd_tensor<T>
where (const tensor_expr_base<CondDerived, CondVal> &cond, T x_val, T y_val)
{
  nd_tensor<T> out (cond.shape ());
  const std::size_t n = out.size ();
  for (std::size_t i = 0; i < n; ++i)
    {
      out[i] = static_cast<bool> (cond.eval_at (i)) ? x_val : y_val;
    }
  return out;
}

/* IEEE-754 classification ufuncs.  */
template <typename Derived, typename Val>
nd_tensor<std::uint8_t>
isnan (const tensor_expr_base<Derived, Val> &expr)
{
  return nd_tensor<std::uint8_t> (make_unary_expr (expr, [] (Val x) {
    return static_cast<std::uint8_t> (std::isnan (static_cast<double> (x)));
  }));
}

template <typename Derived, typename Val>
nd_tensor<std::uint8_t>
isinf (const tensor_expr_base<Derived, Val> &expr)
{
  return nd_tensor<std::uint8_t> (make_unary_expr (expr, [] (Val x) {
    return static_cast<std::uint8_t> (std::isinf (static_cast<double> (x)));
  }));
}

template <typename Derived, typename Val>
nd_tensor<std::uint8_t>
isfinite (const tensor_expr_base<Derived, Val> &expr)
{
  return nd_tensor<std::uint8_t> (make_unary_expr (expr, [] (Val x) {
    return static_cast<std::uint8_t> (std::isfinite (static_cast<double> (x)));
  }));
}

/* Tolerance-based tensor equality check (NumPy allclose / isclose).  */
template <typename LDerived, typename LVal, typename RDerived, typename RVal>
bool
allclose (const tensor_expr_base<LDerived, LVal> &a,
          const tensor_expr_base<RDerived, RVal> &b,
          double rtol = 1e-5,
          double atol = 1e-8)
{
  if (a.shape () != b.shape ())
    {
      return false;
    }
  const std::size_t n = a.size ();
  for (std::size_t i = 0; i < n; ++i)
    {
      const double va = static_cast<double> (a.eval_at (i));
      const double vb = static_cast<double> (b.eval_at (i));
      if (std::isnan (va) || std::isnan (vb))
        {
          return false;
        }
      if (std::isinf (va) || std::isinf (vb))
        {
          if (va != vb)
            {
              return false;
            }
          continue;
        }
      if (std::abs (va - vb) > (atol + rtol * std::abs (vb)))
        {
          return false;
        }
    }
  return true;
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
nd_tensor<std::uint8_t>
isclose (const tensor_expr_base<LDerived, LVal> &a,
         const tensor_expr_base<RDerived, RVal> &b,
         double rtol = 1e-5,
         double atol = 1e-8)
{
  return nd_tensor<std::uint8_t> (
    make_binary_expr (a, b, [rtol, atol] (LVal va, RVal vb) {
      const double da = static_cast<double> (va);
      const double db = static_cast<double> (vb);
      if (std::isnan (da) || std::isnan (db))
        {
          return static_cast<std::uint8_t> (0);
        }
      if (std::isinf (da) || std::isinf (db))
        {
          return static_cast<std::uint8_t> (da == db ? 1 : 0);
        }
      return static_cast<std::uint8_t> (
        std::abs (da - db) <= (atol + rtol * std::abs (db)));
    }));
}

/* Normalized sinc(x) = sin(pi * x) / (pi * x), deg2rad, rad2deg, copysign,
   and fmod ufuncs (NumPy equivalents).  */
template <typename Derived, typename Val>
auto
sinc (const tensor_expr_base<Derived, Val> &expr)
{
  return make_unary_expr (expr, [] (Val x) {
    const double d = static_cast<double> (x);
    if (std::abs (d) < 1e-15)
      {
        return 1.0;
      }
    const double pix = constants::pi * d;
    return std::sin (pix) / pix;
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
copysign (const tensor_expr_base<LDerived, LVal> &lhs,
          const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    return std::copysign (static_cast<double> (a), static_cast<double> (b));
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
fmod (const tensor_expr_base<LDerived, LVal> &lhs,
      const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    return std::fmod (static_cast<double> (a), static_cast<double> (b));
  });
}

/* High-precision exponential/logarithmic ufuncs and Heaviside step function:
   exp2, logaddexp, heaviside (NumPy equivalents).  */
template <typename Derived, typename Val>
auto
exp2 (const tensor_expr_base<Derived, Val> &expr)
{
  return make_unary_expr (expr, [] (Val x) {
    return std::exp2 (static_cast<double> (x));
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
logaddexp (const tensor_expr_base<LDerived, LVal> &lhs,
           const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    const double da = static_cast<double> (a);
    const double db = static_cast<double> (b);
    if (std::isnan (da) || std::isnan (db))
      {
        return constants::nan;
      }
    if (da == db)
      {
        return da + constants::ln2;
      }
    const double hi = std::max (da, db);
    const double lo = std::min (da, db);
    return hi + std::log1p (std::exp (lo - hi));
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
heaviside (const tensor_expr_base<LDerived, LVal> &x,
           const tensor_expr_base<RDerived, RVal> &h0)
{
  return make_binary_expr (x, h0, [] (LVal vx, RVal vh0) {
    const double dx = static_cast<double> (vx);
    const double dh0 = static_cast<double> (vh0);
    if (std::isnan (dx) || std::isnan (dh0))
      {
        return constants::nan;
      }
    if (dx < 0.0)
      {
        return 0.0;
      }
    if (dx == 0.0)
      {
        return dh0;
      }
    return 1.0;
  });
}

template <typename Derived, typename Val>
auto
heaviside (const tensor_expr_base<Derived, Val> &x, double h0)
{
  return make_unary_expr (x, [h0] (Val vx) {
    const double dx = static_cast<double> (vx);
    if (std::isnan (dx) || std::isnan (h0))
      {
        return constants::nan;
      }
    if (dx < 0.0)
      {
        return 0.0;
      }
    if (dx == 0.0)
      {
        return h0;
      }
    return 1.0;
  });
}

/* Base-2 logaddexp2 and IEEE-754 floating-point classification ufuncs:
   isnan, isinf, isfinite, signbit (NumPy equivalents).  */
template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
logaddexp2 (const tensor_expr_base<LDerived, LVal> &lhs,
            const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    const double da = static_cast<double> (a);
    const double db = static_cast<double> (b);
    if (std::isnan (da) || std::isnan (db))
      {
        return constants::nan;
      }
    if (da == db)
      {
        return da + 1.0;
      }
    const double hi = std::max (da, db);
    const double lo = std::min (da, db);
    return hi + std::log2 (1.0 + std::exp2 (lo - hi));
  });
}

template <typename Derived, typename Val>
nd_tensor<std::uint8_t>
signbit (const tensor_expr_base<Derived, Val> &expr)
{
  return nd_tensor<std::uint8_t> (make_unary_expr (expr, [] (Val x) {
    return static_cast<std::uint8_t> (
      std::signbit (static_cast<double> (x)) ? 1 : 0);
  }));
}

/* NumPy-semantics floor_divide, remainder (modulo with divisor sign),
   nextafter, ldexp, and frexp.  */
template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
floor_divide (const tensor_expr_base<LDerived, LVal> &lhs,
              const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    return std::floor (static_cast<double> (a) / static_cast<double> (b));
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
remainder (const tensor_expr_base<LDerived, LVal> &lhs,
           const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    const double da = static_cast<double> (a);
    const double db = static_cast<double> (b);
    return da - db * std::floor (da / db);
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
nextafter (const tensor_expr_base<LDerived, LVal> &lhs,
           const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    return std::nextafter (static_cast<double> (a), static_cast<double> (b));
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
ldexp (const tensor_expr_base<LDerived, LVal> &lhs,
       const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (LVal x, RVal exp_val) {
    return std::ldexp (static_cast<double> (x), static_cast<int> (exp_val));
  });
}

template <typename Derived, typename Val>
std::pair<nd_tensor<double>, nd_tensor<std::int64_t>>
frexp (const tensor_expr_base<Derived, Val> &expr)
{
  nd_tensor<double> mant (expr.shape ());
  nd_tensor<std::int64_t> exps (expr.shape ());
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      int e = 0;
      mant[i] = std::frexp (static_cast<double> (expr.eval_at (i)), &e);
      exps[i] = static_cast<std::int64_t> (e);
    }
  return {std::move (mant), std::move (exps)};
}

/* Fractional/integral split `modf`, combined quotient/remainder `divmod`,
   and elementwise `reciprocal`, `square`, `positive`, `negative` ufuncs
   (NumPy equivalents).  */
template <typename Derived, typename Val>
std::pair<nd_tensor<double>, nd_tensor<double>>
modf (const tensor_expr_base<Derived, Val> &expr)
{
  nd_tensor<double> frac_part (expr.shape ());
  nd_tensor<double> int_part (expr.shape ());
  for (std::size_t i = 0; i < expr.size (); ++i)
    {
      double ip = 0.0;
      frac_part[i] = std::modf (static_cast<double> (expr.eval_at (i)), &ip);
      int_part[i] = ip;
    }
  return {std::move (frac_part), std::move (int_part)};
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
std::pair<nd_tensor<double>, nd_tensor<double>>
divmod (const tensor_expr_base<LDerived, LVal> &lhs,
        const tensor_expr_base<RDerived, RVal> &rhs)
{
  return {
    nd_tensor<double> (floor_divide (lhs, rhs)),
    nd_tensor<double> (remainder (lhs, rhs))
  };
}

template <typename Derived, typename Val>
auto
reciprocal (const tensor_expr_base<Derived, Val> &expr)
{
  return make_unary_expr (expr, [] (Val x) {
    return static_cast<Val> (1.0 / static_cast<double> (x));
  });
}

template <typename Derived, typename Val>
auto
square (const tensor_expr_base<Derived, Val> &expr)
{
  return make_unary_expr (expr, [] (Val x) { return x * x; });
}

template <typename Derived, typename Val>
auto
positive (const tensor_expr_base<Derived, Val> &expr)
{
  return make_unary_expr (expr, [] (Val x) { return +x; });
}

template <typename Derived, typename Val>
auto
negative (const tensor_expr_base<Derived, Val> &expr)
{
  return make_unary_expr (expr, [] (Val x) { return -x; });
}

/* IEEE-754 distance to the next adjacent representable floating-point number
   (NumPy spacing equivalent).  */
template <typename Derived, typename Val>
auto
spacing (const tensor_expr_base<Derived, Val> &expr)
{
  return make_unary_expr (expr, [] (Val x) {
    const double d = static_cast<double> (x);
    const double target =
      (d >= 0.0) ? constants::inf : -constants::inf;
    return std::nextafter (d, target) - d;
  });
}

/* Round toward zero `trunc` / `fix`, integer greatest common divisor `gcd`,
   least common multiple `lcm`, and elementwise boolean logic ufuncs
   `logical_and`, `logical_or`, `logical_xor`, `logical_not`.  */
template <typename Derived, typename Val>
auto
trunc (const tensor_expr_base<Derived, Val> &expr)
{
  return make_unary_expr (expr, [] (Val x) {
    return std::trunc (static_cast<double> (x));
  });
}

template <typename Derived, typename Val>
auto
fix (const tensor_expr_base<Derived, Val> &expr)
{
  return trunc (expr);
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
gcd (const tensor_expr_base<LDerived, LVal> &lhs,
     const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    return static_cast<double> (
      std::gcd (static_cast<std::int64_t> (a),
                static_cast<std::int64_t> (b)));
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
lcm (const tensor_expr_base<LDerived, LVal> &lhs,
     const tensor_expr_base<RDerived, RVal> &rhs)
{
  return make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    return static_cast<double> (
      std::lcm (static_cast<std::int64_t> (a),
                static_cast<std::int64_t> (b)));
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
nd_tensor<std::uint8_t>
logical_and (const tensor_expr_base<LDerived, LVal> &lhs,
             const tensor_expr_base<RDerived, RVal> &rhs)
{
  return nd_tensor<std::uint8_t> (make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    return static_cast<std::uint8_t> (
      (static_cast<double> (a) != 0.0 && static_cast<double> (b) != 0.0)
      ? 1 : 0);
  }));
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
nd_tensor<std::uint8_t>
logical_or (const tensor_expr_base<LDerived, LVal> &lhs,
            const tensor_expr_base<RDerived, RVal> &rhs)
{
  return nd_tensor<std::uint8_t> (make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    return static_cast<std::uint8_t> (
      (static_cast<double> (a) != 0.0 || static_cast<double> (b) != 0.0)
      ? 1 : 0);
  }));
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
nd_tensor<std::uint8_t>
logical_xor (const tensor_expr_base<LDerived, LVal> &lhs,
             const tensor_expr_base<RDerived, RVal> &rhs)
{
  return nd_tensor<std::uint8_t> (make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    const bool ba = (static_cast<double> (a) != 0.0);
    const bool bb = (static_cast<double> (b) != 0.0);
    return static_cast<std::uint8_t> ((ba != bb) ? 1 : 0);
  }));
}

template <typename Derived, typename Val>
nd_tensor<std::uint8_t>
logical_not (const tensor_expr_base<Derived, Val> &expr)
{
  return nd_tensor<std::uint8_t> (make_unary_expr (expr, [] (Val x) {
    return static_cast<std::uint8_t> (
      (static_cast<double> (x) == 0.0) ? 1 : 0);
  }));
}

/* Bitwise Integer Universal Functions: bitwise_and, bitwise_or, bitwise_xor,
   bitwise_not, left_shift, right_shift (NumPy bitwise ufunc equivalents).  */
template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
bitwise_and (const tensor_expr_base<LDerived, LVal> &lhs,
             const tensor_expr_base<RDerived, RVal> &rhs)
{
  using Ret = std::common_type_t<LVal, RVal>;
  return make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    return static_cast<Ret> (static_cast<std::int64_t> (a)
                             & static_cast<std::int64_t> (b));
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
bitwise_or (const tensor_expr_base<LDerived, LVal> &lhs,
            const tensor_expr_base<RDerived, RVal> &rhs)
{
  using Ret = std::common_type_t<LVal, RVal>;
  return make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    return static_cast<Ret> (static_cast<std::int64_t> (a)
                             | static_cast<std::int64_t> (b));
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
bitwise_xor (const tensor_expr_base<LDerived, LVal> &lhs,
             const tensor_expr_base<RDerived, RVal> &rhs)
{
  using Ret = std::common_type_t<LVal, RVal>;
  return make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    return static_cast<Ret> (static_cast<std::int64_t> (a)
                             ^ static_cast<std::int64_t> (b));
  });
}

template <typename Derived, typename Val>
auto
bitwise_not (const tensor_expr_base<Derived, Val> &expr)
{
  return make_unary_expr (expr, [] (Val x) {
    return static_cast<Val> (~static_cast<std::int64_t> (x));
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
left_shift (const tensor_expr_base<LDerived, LVal> &lhs,
            const tensor_expr_base<RDerived, RVal> &rhs)
{
  using Ret = std::common_type_t<LVal, RVal>;
  return make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    const auto shift = static_cast<int> (b);
    if (shift <= 0)
      {
        return static_cast<Ret> (static_cast<std::int64_t> (a));
      }
    return static_cast<Ret> (
      static_cast<std::int64_t> (
        static_cast<std::uint64_t> (static_cast<std::int64_t> (a)) << shift));
  });
}

template <typename LDerived, typename LVal, typename RDerived, typename RVal>
auto
right_shift (const tensor_expr_base<LDerived, LVal> &lhs,
             const tensor_expr_base<RDerived, RVal> &rhs)
{
  using Ret = std::common_type_t<LVal, RVal>;
  return make_binary_expr (lhs, rhs, [] (LVal a, RVal b) {
    const auto shift = static_cast<int> (b);
    if (shift <= 0)
      {
        return static_cast<Ret> (static_cast<std::int64_t> (a));
      }
    return static_cast<Ret> (static_cast<std::int64_t> (a) >> shift);
  });
}

/* Complex Tensor Elementwise Operations: real, imag, conj, angle, complex_abs
   (NumPy complex ufunc equivalents).  */
inline nd_tensor<double>
real (const nd_tensor<std::complex<double>> &z)
{
  nd_tensor<double> out (z.shape ());
  for (std::size_t i = 0; i < z.size (); ++i)
    {
      out[i] = z.eval_at (i).real ();
    }
  return out;
}

inline nd_tensor<double>
imag (const nd_tensor<std::complex<double>> &z)
{
  nd_tensor<double> out (z.shape ());
  for (std::size_t i = 0; i < z.size (); ++i)
    {
      out[i] = z.eval_at (i).imag ();
    }
  return out;
}

inline nd_tensor<std::complex<double>>
conj (const nd_tensor<std::complex<double>> &z)
{
  nd_tensor<std::complex<double>> out (z.shape ());
  for (std::size_t i = 0; i < z.size (); ++i)
    {
      out[i] = std::conj (z.eval_at (i));
    }
  return out;
}

inline nd_tensor<double>
angle (const nd_tensor<std::complex<double>> &z, bool deg = false)
{
  nd_tensor<double> out (z.shape ());
  const double scale = deg ? (180.0 / constants::pi) : 1.0;
  for (std::size_t i = 0; i < z.size (); ++i)
    {
      out[i] = std::arg (z.eval_at (i)) * scale;
    }
  return out;
}

inline nd_tensor<double>
complex_abs (const nd_tensor<std::complex<double>> &z)
{
  nd_tensor<double> out (z.shape ());
  for (std::size_t i = 0; i < z.size (); ++i)
    {
      out[i] = std::abs (z.eval_at (i));
    }
  return out;
}

inline nd_tensor<std::uint8_t>
isreal (const nd_tensor<std::complex<double>> &z)
{
  nd_tensor<std::uint8_t> out (z.shape ());
  for (std::size_t i = 0; i < z.size (); ++i)
    {
      out[i] = (z.eval_at (i).imag () == 0.0) ? 1 : 0;
    }
  return out;
}

inline nd_tensor<std::uint8_t>
iscomplex (const nd_tensor<std::complex<double>> &z)
{
  nd_tensor<std::uint8_t> out (z.shape ());
  for (std::size_t i = 0; i < z.size (); ++i)
    {
      out[i] = (z.eval_at (i).imag () != 0.0) ? 1 : 0;
    }
  return out;
}

inline nd_tensor<double>
real_if_close (const nd_tensor<std::complex<double>> &z, double tol = 1e-13)
{
  nd_tensor<double> out (z.shape ());
  for (std::size_t i = 0; i < z.size (); ++i)
    {
      const auto val = z.eval_at (i);
      if (std::abs (val.imag ()) > tol)
        {
          throw hyper_math_error (
            "real_if_close encountered imaginary component exceeding tol");
        }
      out[i] = val.real ();
    }
  return out;
}

} // namespace hyper_math::tensor

#endif /* HYPER_MATH_TENSOR_UFUNC_HPP */
