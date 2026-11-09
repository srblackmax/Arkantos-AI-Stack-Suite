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

#ifndef BASALT_OPS_HPP
#define BASALT_OPS_HPP

/* Differentiable tensor operations dispatched through the Bellows / Hyper Math
   compute kernels on CPU and the runtime-loaded CUDA driver backend, with
   full reverse-mode and higher-order autograd support.  */

#include <cstddef>
#include <vector>
#include <basalt/config.hpp>
#include <basalt/tensor.hpp>

namespace basalt
{
namespace ops
{

/* Binary elementwise arithmetic with broadcasting.  */
BASALT_API Tensor
add (const Tensor &a, const Tensor &b);

BASALT_API Tensor
sub (const Tensor &a, const Tensor &b);

BASALT_API Tensor
mul (const Tensor &a, const Tensor &b);

BASALT_API Tensor
div (const Tensor &a, const Tensor &b);

BASALT_API Tensor
pow (const Tensor &base, const Tensor &exponent);

/* Tensor-scalar arithmetic.  */
BASALT_API Tensor
add_scalar (const Tensor &a, double scalar);

BASALT_API Tensor
sub_scalar (const Tensor &a, double scalar);

BASALT_API Tensor
rsub_scalar (double scalar, const Tensor &a);

BASALT_API Tensor
mul_scalar (const Tensor &a, double scalar);

BASALT_API Tensor
div_scalar (const Tensor &a, double scalar);

BASALT_API Tensor
rdiv_scalar (double scalar, const Tensor &a);

BASALT_API Tensor
pow_scalar (const Tensor &a, double exponent);

/* Unary mathematical operations.  */
BASALT_API Tensor
neg (const Tensor &x);

BASALT_API Tensor
abs (const Tensor &x);

BASALT_API Tensor
sign (const Tensor &x);

BASALT_API Tensor
square (const Tensor &x);

BASALT_API Tensor
sqrt (const Tensor &x);

BASALT_API Tensor
rsqrt (const Tensor &x);

BASALT_API Tensor
exp (const Tensor &x);

BASALT_API Tensor
log (const Tensor &x);

BASALT_API Tensor
log1p (const Tensor &x);

BASALT_API Tensor
expm1 (const Tensor &x);

BASALT_API Tensor
sin (const Tensor &x);

BASALT_API Tensor
cos (const Tensor &x);

BASALT_API Tensor
tanh (const Tensor &x);

BASALT_API Tensor
sigmoid (const Tensor &x);

BASALT_API Tensor
relu (const Tensor &x);

BASALT_API Tensor
leaky_relu (const Tensor &x, double negative_slope = 0.01);

BASALT_API Tensor
elu (const Tensor &x, double alpha = 1.0);

BASALT_API Tensor
silu (const Tensor &x);

BASALT_API Tensor
gelu (const Tensor &x);

BASALT_API Tensor
mish (const Tensor &x);

BASALT_API Tensor
softplus (const Tensor &x, double beta = 1.0);

BASALT_API Tensor
clamp (const Tensor &x, double min_val, double max_val);

/* Conditional selection and comparisons.  */
BASALT_API Tensor
where (const Tensor &condition, const Tensor &x, const Tensor &y);

BASALT_API Tensor
masked_fill (const Tensor &x, const Tensor &mask, double value);

BASALT_API Tensor
eq (const Tensor &a, const Tensor &b);

BASALT_API Tensor
ne (const Tensor &a, const Tensor &b);

BASALT_API Tensor
lt (const Tensor &a, const Tensor &b);

BASALT_API Tensor
le (const Tensor &a, const Tensor &b);

BASALT_API Tensor
gt (const Tensor &a, const Tensor &b);

BASALT_API Tensor
ge (const Tensor &a, const Tensor &b);

BASALT_API bool
allclose (const Tensor &a, const Tensor &b,
          double rtol = 1e-4, double atol = 1e-5);

/* Reductions.  */
BASALT_API Tensor
sum (const Tensor &x);

BASALT_API Tensor
sum (const Tensor &x, ptrdiff_t dim, bool keepdim = false);

BASALT_API Tensor
sum_to_shape (const Tensor &x, const Shape_Vector &target_shape);

BASALT_API Tensor
mean (const Tensor &x);

BASALT_API Tensor
mean (const Tensor &x, ptrdiff_t dim, bool keepdim = false);

BASALT_API Tensor
var (const Tensor &x, bool unbiased = true);

BASALT_API Tensor
var (const Tensor &x, ptrdiff_t dim, bool unbiased = true,
     bool keepdim = false);

BASALT_API Tensor
max (const Tensor &x);

BASALT_API Tensor
max (const Tensor &x, ptrdiff_t dim, bool keepdim = false);

BASALT_API Tensor
min (const Tensor &x);

BASALT_API Tensor
min (const Tensor &x, ptrdiff_t dim, bool keepdim = false);

BASALT_API Tensor
argmax (const Tensor &x, ptrdiff_t dim = -1, bool keepdim = false);

BASALT_API Tensor
argmin (const Tensor &x, ptrdiff_t dim = -1, bool keepdim = false);

BASALT_API Tensor
logsumexp (const Tensor &x, ptrdiff_t dim = -1, bool keepdim = false);

BASALT_API Tensor
softmax (const Tensor &x, ptrdiff_t dim = -1);

BASALT_API Tensor
log_softmax (const Tensor &x, ptrdiff_t dim = -1);

/* Matrix multiplication and linear algebra.  */
BASALT_API Tensor
matmul (const Tensor &a, const Tensor &b);

BASALT_API Tensor
bmm (const Tensor &a, const Tensor &b);

BASALT_API Tensor
addmm (const Tensor &bias, const Tensor &input, const Tensor &weight,
       double beta = 1.0, double alpha = 1.0);

BASALT_API Tensor
dot (const Tensor &a, const Tensor &b);

BASALT_API Tensor
outer (const Tensor &a, const Tensor &b);

BASALT_API Tensor
norm (const Tensor &x, double p = 2.0);

/* Concatenation, stacking, indexing, and padding.  */
BASALT_API Tensor
cat (const std::vector<Tensor> &tensors, ptrdiff_t dim = 0);

BASALT_API Tensor
stack (const std::vector<Tensor> &tensors, ptrdiff_t dim = 0);

BASALT_API std::vector<Tensor>
split (const Tensor &x, size_t split_size, ptrdiff_t dim = 0);

BASALT_API std::vector<Tensor>
chunk (const Tensor &x, size_t num_chunks, ptrdiff_t dim = 0);

BASALT_API Tensor
index_select (const Tensor &x, ptrdiff_t dim, const Tensor &indices);

BASALT_API Tensor
gather (const Tensor &x, ptrdiff_t dim, const Tensor &index);

BASALT_API Tensor
tril (const Tensor &x, ptrdiff_t diagonal = 0);

BASALT_API Tensor
triu (const Tensor &x, ptrdiff_t diagonal = 0);

BASALT_API Tensor
pad2d (const Tensor &x, size_t pad_top, size_t pad_bottom,
       size_t pad_left, size_t pad_right, double value = 0.0);

} /* namespace ops */

/* Bring core operations and operator overloads into namespace basalt.  */
using ops::abs;
using ops::add;
using ops::addmm;
using ops::allclose;
using ops::argmax;
using ops::argmin;
using ops::bmm;
using ops::cat;
using ops::chunk;
using ops::clamp;
using ops::cos;
using ops::div;
using ops::dot;
using ops::elu;
using ops::eq;
using ops::exp;
using ops::expm1;
using ops::gather;
using ops::ge;
using ops::gelu;
using ops::gt;
using ops::index_select;
using ops::le;
using ops::leaky_relu;
using ops::log;
using ops::log1p;
using ops::log_softmax;
using ops::logsumexp;
using ops::lt;
using ops::masked_fill;
using ops::matmul;
using ops::max;
using ops::mean;
using ops::min;
using ops::mish;
using ops::mul;
using ops::ne;
using ops::neg;
using ops::norm;
using ops::outer;
using ops::pow;
using ops::relu;
using ops::rsqrt;
using ops::sigmoid;
using ops::sign;
using ops::silu;
using ops::sin;
using ops::softmax;
using ops::softplus;
using ops::split;
using ops::sqrt;
using ops::square;
using ops::stack;
using ops::sub;
using ops::sum;
using ops::sum_to_shape;
using ops::tanh;
using ops::tril;
using ops::triu;
using ops::var;
using ops::where;

inline Tensor
operator+ (const Tensor &a, const Tensor &b)
{
  return ops::add (a, b);
}

inline Tensor
operator+ (const Tensor &a, double s)
{
  return ops::add_scalar (a, s);
}

inline Tensor
operator+ (double s, const Tensor &a)
{
  return ops::add_scalar (a, s);
}

inline Tensor
operator- (const Tensor &a, const Tensor &b)
{
  return ops::sub (a, b);
}

inline Tensor
operator- (const Tensor &a, double s)
{
  return ops::sub_scalar (a, s);
}

inline Tensor
operator- (double s, const Tensor &a)
{
  return ops::rsub_scalar (s, a);
}

inline Tensor
operator- (const Tensor &a)
{
  return ops::neg (a);
}

inline Tensor
operator* (const Tensor &a, const Tensor &b)
{
  return ops::mul (a, b);
}

inline Tensor
operator* (const Tensor &a, double s)
{
  return ops::mul_scalar (a, s);
}

inline Tensor
operator* (double s, const Tensor &a)
{
  return ops::mul_scalar (a, s);
}

inline Tensor
operator/ (const Tensor &a, const Tensor &b)
{
  return ops::div (a, b);
}

inline Tensor
operator/ (const Tensor &a, double s)
{
  return ops::div_scalar (a, s);
}

inline Tensor
operator/ (double s, const Tensor &a)
{
  return ops::rdiv_scalar (s, a);
}

} /* namespace basalt */

#endif /* BASALT_OPS_HPP */
