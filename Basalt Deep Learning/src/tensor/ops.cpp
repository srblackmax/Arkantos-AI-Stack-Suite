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
#include <basalt/ops.hpp>
#include <basalt/autograd.hpp>
#include "tensor/tensor_internal.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace basalt::ops
{

namespace
{

void
check_same_device (const Tensor &a, const Tensor &b, const char *op_name)
{
  BASALT_ASSERT (a.defined () && b.defined (),
                 std::string (op_name) + ": undefined input tensor");
  BASALT_ASSERT_DEVICE (
    a.device () == b.device (),
    std::string (op_name) + ": tensors must be on the same device ("
    + a.device ().str () + " vs " + b.device ().str () + ")");
}

template <typename Func>
Tensor
unary_kernel (
  const Tensor &x, Dtype out_dtype, Func fn,
  bellows_compat::Cuda_Unary_Op cuda_op = bellows_compat::Cuda_Unary_Op::None,
  float cuda_param = 0.0f)
{
  BASALT_ASSERT (x.defined (), "unary op called on undefined Tensor");
  Tensor out (x.shape (),
              Tensor_Options ().with_dtype (out_dtype).with_device (
                x.device ()));
  size_t n = x.numel ();
  if (x.is_contiguous () && x.dtype () == Dtype::Float32
      && out_dtype == Dtype::Float32)
    {
      auto &cuda_drv = bellows_compat::Cuda_Runtime_Driver::instance ();
      if (x.device ().is_cuda ()
          && cuda_op != bellows_compat::Cuda_Unary_Op::None
          && cuda_drv.launch_unary_f32 (
               x.device ().index (), x.data_ptr<float> (),
               out.data_ptr<float> (), cuda_param, cuda_op, n))
        {
          return out;
        }
      if (x.device ().is_cuda ())
        {
          cuda_drv.sync_if_busy ();
        }
      const float *src = x.data_ptr<float> ();
      float *dst = out.data_ptr<float> ();
      if (n >= 16384)
        {
          hyper_math::parallel_for (
            size_t (0), n, size_t (2048), [&] (size_t i)
            {
              dst[i] = static_cast<float> (fn (static_cast<double> (src[i])));
            });
        }
      else
        {
          for (size_t i = 0; i < n; ++i)
            dst[i] = static_cast<float> (fn (static_cast<double> (src[i])));
        }
    }
  else if (n >= 16384)
    {
      hyper_math::parallel_for (size_t (0), n, size_t (2048), [&] (size_t i)
        {
          out.write_scalar_at (i, fn (x.read_scalar_at (i)));
        });
    }
  else
    {
      for (size_t i = 0; i < n; ++i)
        {
          out.write_scalar_at (i, fn (x.read_scalar_at (i)));
        }
    }
  return out;
}

template <typename Func>
Tensor
binary_kernel (
  const Tensor &a, const Tensor &b, Dtype out_dtype, Func fn,
  bellows_compat::Cuda_Binary_Op cuda_op
    = bellows_compat::Cuda_Binary_Op::None)
{
  check_same_device (a, b, "binary_op");
  Shape_Vector out_shape = broadcast_shapes (a.shape (), b.shape ());
  Tensor out (out_shape,
              Tensor_Options ().with_dtype (out_dtype).with_device (
                a.device ()));
  size_t n = out.numel ();

  if (a.dtype () == Dtype::Float32 && b.dtype () == Dtype::Float32
      && out_dtype == Dtype::Float32 && a.is_contiguous ()
      && b.is_contiguous () && n > 0)
    {
      const float *ap = a.data_ptr<float> ();
      const float *bp = b.data_ptr<float> ();
      float *cp = out.data_ptr<float> ();
      size_t na = a.numel ();
      size_t nb = b.numel ();
      auto &cuda_drv = bellows_compat::Cuda_Runtime_Driver::instance ();

      if (na == n && nb == n)
        {
          if (a.device ().is_cuda ()
              && cuda_op != bellows_compat::Cuda_Binary_Op::None
              && cuda_drv.launch_binary_f32 (a.device ().index (), ap, bp, cp,
                                             cuda_op, n))
            {
              return out;
            }
          if (a.device ().is_cuda ())
            {
              cuda_drv.sync_if_busy ();
            }
          if (n >= 16384)
            {
              hyper_math::parallel_for (
                size_t (0), n, size_t (2048), [&] (size_t i)
                {
                  cp[i] = static_cast<float> (
                    fn (static_cast<double> (ap[i]),
                        static_cast<double> (bp[i])));
                });
            }
          else
            {
              for (size_t i = 0; i < n; ++i)
                {
                  cp[i] = static_cast<float> (
                    fn (static_cast<double> (ap[i]),
                        static_cast<double> (bp[i])));
                }
            }
          return out;
        }
      size_t cols = out_shape.empty () ? 1 : out_shape.back ();
      if (cols > 0 && na == n && nb == cols && b.size (-1) == cols
          && a.device ().is_cuda ()
          && cuda_op != bellows_compat::Cuda_Binary_Op::None
          && cuda_drv.launch_row_bcast_f32 (a.device ().index (), ap, bp, cp,
                                            cuda_op, n, cols))
        {
          return out;
        }
      if (a.device ().is_cuda ())
        {
          cuda_drv.sync_if_busy ();
        }
      if (nb == 1 && na == n)
        {
          double b0 = static_cast<double> (bp[0]);
          for (size_t i = 0; i < n; ++i)
            {
              cp[i] = static_cast<float> (
                fn (static_cast<double> (ap[i]), b0));
            }
          return out;
        }
      if (na == 1 && nb == n)
        {
          double a0 = static_cast<double> (ap[0]);
          for (size_t i = 0; i < n; ++i)
            {
              cp[i] = static_cast<float> (
                fn (a0, static_cast<double> (bp[i])));
            }
          return out;
        }
      if (cols > 0 && na == n && nb == cols && b.size (-1) == cols)
        {
          size_t rows = n / cols;
          for (size_t r = 0; r < rows; ++r)
            {
              const float *ar = ap + r * cols;
              float *cr = cp + r * cols;
              for (size_t c = 0; c < cols; ++c)
                {
                  cr[c] = static_cast<float> (
                    fn (static_cast<double> (ar[c]),
                        static_cast<double> (bp[c])));
                }
            }
          return out;
        }
      if (cols > 1 && na == n && b.ndim () == a.ndim ()
          && b.size (-1) == 1 && nb * cols == n)
        {
          for (size_t r = 0; r < nb; ++r)
            {
              const float *ar = ap + r * cols;
              float *cr = cp + r * cols;
              double br = static_cast<double> (bp[r]);
              for (size_t c = 0; c < cols; ++c)
                {
                  cr[c] = static_cast<float> (
                    fn (static_cast<double> (ar[c]), br));
                }
            }
          return out;
        }
    }

  Tensor a_exp = (a.shape () == out_shape) ? a : a.expand (out_shape);
  Tensor b_exp = (b.shape () == out_shape) ? b : b.expand (out_shape);
  if (n >= 16384)
    {
      hyper_math::parallel_for (size_t (0), n, size_t (2048), [&] (size_t i)
        {
          out.write_scalar_at (
            i, fn (a_exp.read_scalar_at (i), b_exp.read_scalar_at (i)));
        });
    }
  else
    {
      for (size_t i = 0; i < n; ++i)
        {
          out.write_scalar_at (
            i, fn (a_exp.read_scalar_at (i), b_exp.read_scalar_at (i)));
        }
    }
  return out;
}

void
attach_grad (const std::vector<Tensor> &inputs, Tensor &output,
             std::shared_ptr<Autograd_Node> node)
{
  if (!Grad_Mode::is_enabled () || !is_floating_point (output.dtype ()))
    {
      return;
    }
  bool any_req = false;
  for (const Tensor &inp : inputs)
    {
      if (inp.defined () && inp.requires_grad ())
        {
          any_req = true;
          break;
        }
    }
  if (any_req)
    {
      node->set_next_edges (autograd::collect_next_edges (inputs));
      output.set_requires_grad (true);
      output.set_grad_fn (std::move (node));
    }
}

/* Autograd backward nodes implemented using differentiable ops::* calls
   so higher-order derivatives (create_graph = true) work seamlessly.  */

class Add_Backward : public Autograd_Node
{
public:
  Add_Backward (Shape_Vector sa, Shape_Vector sb)
    : sa_ (std::move (sa)),
      sb_ (std::move (sb))
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Add_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    const Tensor &g = grads[0];
    return {sum_to_shape (g, sa_), sum_to_shape (g, sb_)};
  }

private:
  Shape_Vector sa_;
  Shape_Vector sb_;
};

class Sub_Backward : public Autograd_Node
{
public:
  Sub_Backward (Shape_Vector sa, Shape_Vector sb)
    : sa_ (std::move (sa)),
      sb_ (std::move (sb))
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Sub_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    const Tensor &g = grads[0];
    return {sum_to_shape (g, sa_), sum_to_shape (neg (g), sb_)};
  }

private:
  Shape_Vector sa_;
  Shape_Vector sb_;
};

class Mul_Backward : public Autograd_Node
{
public:
  Mul_Backward (const Tensor &a, const Tensor &b)
    : a_ (a),
      b_ (b),
      sa_ (a.shape ()),
      sb_ (b.shape ())
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Mul_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    Tensor a = a_.unpack ();
    Tensor b = b_.unpack ();
    const Tensor &g = grads[0];
    return {sum_to_shape (mul (g, b), sa_), sum_to_shape (mul (g, a), sb_)};
  }

  void
  release_saved_variables (void) noexcept override
  {
    a_ = Saved_Variable ();
    b_ = Saved_Variable ();
  }

private:
  Saved_Variable a_;
  Saved_Variable b_;
  Shape_Vector sa_;
  Shape_Vector sb_;
};

class Div_Backward : public Autograd_Node
{
public:
  Div_Backward (const Tensor &a, const Tensor &b)
    : a_ (a),
      b_ (b),
      sa_ (a.shape ()),
      sb_ (b.shape ())
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Div_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    Tensor a = a_.unpack ();
    Tensor b = b_.unpack ();
    const Tensor &g = grads[0];
    Tensor ga = sum_to_shape (div (g, b), sa_);
    Tensor gb = sum_to_shape (neg (div (mul (g, a), mul (b, b))), sb_);
    return {ga, gb};
  }

  void
  release_saved_variables (void) noexcept override
  {
    a_ = Saved_Variable ();
    b_ = Saved_Variable ();
  }

private:
  Saved_Variable a_;
  Saved_Variable b_;
  Shape_Vector sa_;
  Shape_Vector sb_;
};

class Pow_Backward : public Autograd_Node
{
public:
  Pow_Backward (const Tensor &base, const Tensor &exp_t)
    : base_ (base),
      exp_ (exp_t),
      sb_ (base.shape ()),
      se_ (exp_t.shape ())
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Pow_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    Tensor b = base_.unpack ();
    Tensor e = exp_.unpack ();
    const Tensor &g = grads[0];
    Tensor gb = sum_to_shape (
      mul (g, mul (e, pow (b, sub_scalar (e, 1.0)))), sb_);
    Tensor safe_b = clamp (b, 1e-12, std::numeric_limits<double>::max ());
    Tensor ge = sum_to_shape (mul (g, mul (pow (b, e), log (safe_b))), se_);
    return {gb, ge};
  }

  void
  release_saved_variables (void) noexcept override
  {
    base_ = Saved_Variable ();
    exp_ = Saved_Variable ();
  }

private:
  Saved_Variable base_;
  Saved_Variable exp_;
  Shape_Vector sb_;
  Shape_Vector se_;
};

class Lambda_Unary_Backward : public Autograd_Node
{
public:
  using Fn = std::function<Tensor (const Tensor &, const Tensor &)>;

  Lambda_Unary_Backward (const Tensor &saved, Fn fn,
                         const char *node_name = "Unary_Backward",
                         bool is_output = false)
    : saved_ (saved, is_output),
      fn_ (std::move (fn)),
      name_ (node_name)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return name_;
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    Tensor s = saved_.unpack ();
    return {fn_ (grads[0], s)};
  }

  void
  release_saved_variables (void) noexcept override
  {
    saved_ = Saved_Variable ();
  }

private:
  Saved_Variable saved_;
  Fn fn_;
  const char *name_;
};

} /* namespace */

Tensor
add (const Tensor &a, const Tensor &b)
{
  Dtype dt = promote_dtypes (a.dtype (), b.dtype ());
  Tensor out = binary_kernel (a, b, dt, [] (double x, double y)
    {
      return x + y;
    }, bellows_compat::Cuda_Binary_Op::Add);
  attach_grad ({a, b}, out,
               std::make_shared<Add_Backward> (a.shape (), b.shape ()));
  return out;
}

Tensor
sub (const Tensor &a, const Tensor &b)
{
  Dtype dt = promote_dtypes (a.dtype (), b.dtype ());
  Tensor out = binary_kernel (a, b, dt, [] (double x, double y)
    {
      return x - y;
    }, bellows_compat::Cuda_Binary_Op::Sub);
  attach_grad ({a, b}, out,
               std::make_shared<Sub_Backward> (a.shape (), b.shape ()));
  return out;
}

Tensor
mul (const Tensor &a, const Tensor &b)
{
  Dtype dt = promote_dtypes (a.dtype (), b.dtype ());
  Tensor out = binary_kernel (a, b, dt, [] (double x, double y)
    {
      return x * y;
    }, bellows_compat::Cuda_Binary_Op::Mul);
  attach_grad ({a, b}, out, std::make_shared<Mul_Backward> (a, b));
  return out;
}

Tensor
div (const Tensor &a, const Tensor &b)
{
  Dtype dt = promote_dtypes (a.dtype (), b.dtype ());
  Tensor out = binary_kernel (a, b, dt, [] (double x, double y)
    {
      return x / y;
    }, bellows_compat::Cuda_Binary_Op::Div);
  attach_grad ({a, b}, out, std::make_shared<Div_Backward> (a, b));
  return out;
}

Tensor
pow (const Tensor &base, const Tensor &exponent)
{
  Dtype dt = promote_dtypes (base.dtype (), exponent.dtype ());
  Tensor out = binary_kernel (base, exponent, dt, [] (double x, double y)
    {
      return std::pow (x, y);
    });
  attach_grad ({base, exponent}, out,
               std::make_shared<Pow_Backward> (base, exponent));
  return out;
}

Tensor
add_scalar (const Tensor &a, double scalar)
{
  Tensor out = unary_kernel (a, a.dtype (), [scalar] (double x)
    {
      return x + scalar;
    }, bellows_compat::Cuda_Unary_Op::Add_Scalar,
       static_cast<float> (scalar));
  attach_grad ({a}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 Tensor (),
                 [] (const Tensor &g, const Tensor &) { return g; },
                 "Add_Scalar_Backward"));
  return out;
}

Tensor
sub_scalar (const Tensor &a, double scalar)
{
  return add_scalar (a, -scalar);
}

Tensor
rsub_scalar (double scalar, const Tensor &a)
{
  return add_scalar (neg (a), scalar);
}

Tensor
mul_scalar (const Tensor &a, double scalar)
{
  Tensor out = unary_kernel (a, a.dtype (), [scalar] (double x)
    {
      return x * scalar;
    }, bellows_compat::Cuda_Unary_Op::Mul_Scalar,
       static_cast<float> (scalar));
  attach_grad ({a}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 Tensor (),
                 [scalar] (const Tensor &g, const Tensor &)
                   {
                     return mul_scalar (g, scalar);
                   },
                 "Mul_Scalar_Backward"));
  return out;
}

Tensor
div_scalar (const Tensor &a, double scalar)
{
  return mul_scalar (a, 1.0 / scalar);
}

Tensor
rdiv_scalar (double scalar, const Tensor &a)
{
  Tensor s = full_like (a, scalar);
  return div (s, a);
}

Tensor
pow_scalar (const Tensor &a, double exponent)
{
  Tensor out = unary_kernel (a, a.dtype (), [exponent] (double x)
    {
      return std::pow (x, exponent);
    });
  attach_grad ({a}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 a,
                 [exponent] (const Tensor &g, const Tensor &x)
                   {
                     return mul (g, mul_scalar (
                       pow_scalar (x, exponent - 1.0), exponent));
                   },
                 "Pow_Scalar_Backward"));
  return out;
}

Tensor
neg (const Tensor &x)
{
  Tensor out = unary_kernel (x, x.dtype (), [] (double v) { return -v; },
                             bellows_compat::Cuda_Unary_Op::Neg);
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 Tensor (),
                 [] (const Tensor &g, const Tensor &) { return neg (g); },
                 "Neg_Backward"));
  return out;
}

Tensor
abs (const Tensor &x)
{
  Tensor out = unary_kernel (x, x.dtype (), [] (double v)
    {
      return std::fabs (v);
    }, bellows_compat::Cuda_Unary_Op::Abs);
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [] (const Tensor &g, const Tensor &s)
                   {
                     return mul (g, sign (s));
                   },
                 "Abs_Backward"));
  return out;
}

Tensor
sign (const Tensor &x)
{
  return unary_kernel (x, x.dtype (), [] (double v)
    {
      return (v > 0.0) ? 1.0 : ((v < 0.0) ? -1.0 : 0.0);
    });
}

Tensor
square (const Tensor &x)
{
  Tensor out = unary_kernel (x, x.dtype (), [] (double v) { return v * v; },
                             bellows_compat::Cuda_Unary_Op::Square);
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [] (const Tensor &g, const Tensor &s)
                   {
                     return mul (g, mul_scalar (s, 2.0));
                   },
                 "Square_Backward"));
  return out;
}

Tensor
sqrt (const Tensor &x)
{
  Tensor out = unary_kernel (x, x.dtype (), [] (double v)
    {
      return std::sqrt (v);
    }, bellows_compat::Cuda_Unary_Op::Sqrt);
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [] (const Tensor &g, const Tensor &s)
                   {
                     return div (g, mul_scalar (sqrt (s), 2.0));
                   },
                 "Sqrt_Backward"));
  return out;
}

Tensor
rsqrt (const Tensor &x)
{
  return pow_scalar (x, -0.5);
}

Tensor
exp (const Tensor &x)
{
  Tensor out = unary_kernel (x, x.dtype (), [] (double v)
    {
      return std::exp (v);
    }, bellows_compat::Cuda_Unary_Op::Exp);
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [] (const Tensor &g, const Tensor &s)
                   {
                     return mul (g, exp (s));
                   },
                 "Exp_Backward"));
  return out;
}

Tensor
log (const Tensor &x)
{
  Tensor out = unary_kernel (x, x.dtype (), [] (double v)
    {
      return std::log (v);
    }, bellows_compat::Cuda_Unary_Op::Log);
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [] (const Tensor &g, const Tensor &s)
                   {
                     return div (g, s);
                   },
                 "Log_Backward"));
  return out;
}

Tensor
log1p (const Tensor &x)
{
  Tensor out = unary_kernel (x, x.dtype (), [] (double v)
    {
      return std::log1p (v);
    });
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [] (const Tensor &g, const Tensor &s)
                   {
                     return div (g, add_scalar (s, 1.0));
                   },
                 "Log1p_Backward"));
  return out;
}

Tensor
expm1 (const Tensor &x)
{
  Tensor out = unary_kernel (x, x.dtype (), [] (double v)
    {
      return std::expm1 (v);
    });
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [] (const Tensor &g, const Tensor &s)
                   {
                     return mul (g, exp (s));
                   },
                 "Expm1_Backward"));
  return out;
}

Tensor
sin (const Tensor &x)
{
  Tensor out = unary_kernel (x, x.dtype (), [] (double v)
    {
      return std::sin (v);
    });
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [] (const Tensor &g, const Tensor &s)
                   {
                     return mul (g, cos (s));
                   },
                 "Sin_Backward"));
  return out;
}

Tensor
cos (const Tensor &x)
{
  Tensor out = unary_kernel (x, x.dtype (), [] (double v)
    {
      return std::cos (v);
    });
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [] (const Tensor &g, const Tensor &s)
                   {
                     return mul (g, neg (sin (s)));
                   },
                 "Cos_Backward"));
  return out;
}

Tensor
tanh (const Tensor &x)
{
  Tensor out = unary_kernel (x, x.dtype (), [] (double v)
    {
      return std::tanh (v);
    }, bellows_compat::Cuda_Unary_Op::Tanh);
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [] (const Tensor &g, const Tensor &s)
                   {
                     Tensor t = tanh (s);
                     return mul (g, rsub_scalar (1.0, mul (t, t)));
                   },
                 "Tanh_Backward"));
  return out;
}

Tensor
sigmoid (const Tensor &x)
{
  Tensor out = unary_kernel (x, x.dtype (), [] (double v)
    {
      return (v >= 0.0)
             ? (1.0 / (1.0 + std::exp (-v)))
             : (std::exp (v) / (1.0 + std::exp (v)));
    }, bellows_compat::Cuda_Unary_Op::Sigmoid);
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [] (const Tensor &g, const Tensor &s)
                   {
                     Tensor sig = sigmoid (s);
                     return mul (g, mul (sig, rsub_scalar (1.0, sig)));
                   },
                 "Sigmoid_Backward"));
  return out;
}

Tensor
relu (const Tensor &x)
{
  Tensor out = unary_kernel (x, x.dtype (), [] (double v)
    {
      return (v > 0.0) ? v : 0.0;
    }, bellows_compat::Cuda_Unary_Op::Relu);
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [] (const Tensor &g, const Tensor &s)
                   {
                     Tensor mask = gt (s, zeros_like (s)).to (g.dtype ());
                     return mul (g, mask);
                   },
                 "ReLU_Backward"));
  return out;
}

Tensor
leaky_relu (const Tensor &x, double negative_slope)
{
  Tensor out = unary_kernel (x, x.dtype (), [negative_slope] (double v)
    {
      return (v > 0.0) ? v : (negative_slope * v);
    });
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [negative_slope] (const Tensor &g, const Tensor &s)
                   {
                     Tensor pos = gt (s, zeros_like (s));
                     Tensor scale = where (
                       pos, ones_like (s), full_like (s, negative_slope));
                     return mul (g, scale);
                   },
                 "LeakyReLU_Backward"));
  return out;
}

Tensor
elu (const Tensor &x, double alpha)
{
  Tensor out = unary_kernel (x, x.dtype (), [alpha] (double v)
    {
      return (v > 0.0) ? v : (alpha * std::expm1 (v));
    });
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [alpha] (const Tensor &g, const Tensor &s)
                   {
                     Tensor pos = gt (s, zeros_like (s));
                     Tensor d = where (
                       pos, ones_like (s), mul_scalar (exp (s), alpha));
                     return mul (g, d);
                   },
                 "ELU_Backward"));
  return out;
}

Tensor
silu (const Tensor &x)
{
  return mul (x, sigmoid (x));
}

Tensor
gelu (const Tensor &x)
{
  /* Fused Gaussian Error Linear Unit forward pass with higher-order
     differentiable backward.  */
  constexpr double k_sqrt_2_over_pi = 0.7978845608028654;
  constexpr double k_coeff = 0.044715;
  Tensor out = unary_kernel (x, x.dtype (), [] (double v)
    {
      double inner = k_sqrt_2_over_pi * (v + k_coeff * v * v * v);
      return 0.5 * v * (1.0 + std::tanh (inner));
    }, bellows_compat::Cuda_Unary_Op::Gelu);
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [] (const Tensor &g, const Tensor &s)
                   {
                     if (!Grad_Mode::is_enabled ())
                       {
                         return binary_kernel (
                           g, s, g.dtype (), [] (double gv, double xv)
                           {
                             double u = k_sqrt_2_over_pi
                                        * (xv + k_coeff * xv * xv * xv);
                             double th = std::tanh (u);
                             double sech2 = 1.0 - th * th;
                             double du = k_sqrt_2_over_pi
                                         * (1.0 + 3.0 * k_coeff * xv * xv);
                             return gv * (0.5 * (1.0 + th)
                                          + 0.5 * xv * sech2 * du);
                           });
                       }
                     Tensor x2 = mul (s, s);
                     Tensor x3 = mul (s, x2);
                     Tensor u = mul_scalar (
                       add (s, mul_scalar (x3, k_coeff)), k_sqrt_2_over_pi);
                     Tensor th = tanh (u);
                     Tensor sech2 = rsub_scalar (1.0, mul (th, th));
                     Tensor du = mul_scalar (
                       add_scalar (mul_scalar (x2, 3.0 * k_coeff), 1.0),
                       k_sqrt_2_over_pi);
                     Tensor dgelu = add (
                       mul_scalar (add_scalar (th, 1.0), 0.5),
                       mul (mul_scalar (s, 0.5), mul (sech2, du)));
                     return mul (g, dgelu);
                   },
                 "GELU_Backward"));
  return out;
}

Tensor
mish (const Tensor &x)
{
  return mul (x, tanh (softplus (x, 1.0)));
}

Tensor
softplus (const Tensor &x, double beta)
{
  Tensor bx = mul_scalar (x, beta);
  Tensor out = unary_kernel (bx, x.dtype (), [beta] (double v)
    {
      double sp = (v > 20.0) ? v : std::log1p (std::exp (v));
      return sp / beta;
    });
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [beta] (const Tensor &g, const Tensor &s)
                   {
                     return mul (g, sigmoid (mul_scalar (s, beta)));
                   },
                 "Softplus_Backward"));
  return out;
}

Tensor
clamp (const Tensor &x, double min_val, double max_val)
{
  Tensor out = unary_kernel (x, x.dtype (), [min_val, max_val] (double v)
    {
      return std::clamp (v, min_val, max_val);
    });
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 x,
                 [min_val, max_val] (const Tensor &g, const Tensor &s)
                   {
                     Tensor mask = unary_kernel (
                       s, g.dtype (), [min_val, max_val] (double v)
                         {
                           return (v >= min_val && v <= max_val) ? 1.0 : 0.0;
                         });
                     return mul (g, mask);
                   },
                 "Clamp_Backward"));
  return out;
}

namespace
{

class Where_Backward : public Autograd_Node
{
public:
  Where_Backward (const Tensor &cond, Shape_Vector sx, Shape_Vector sy,
                  Dtype dt)
    : cond_ (cond.detach ()),
      sx_ (std::move (sx)),
      sy_ (std::move (sy)),
      dt_ (dt)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Where_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    const Tensor &g = grads[0];
    Tensor z = zeros (g.shape (),
                      Tensor_Options ().with_dtype (dt_).with_device (
                        g.device ()));
    Tensor gx = sum_to_shape (where (cond_, g, z), sx_);
    Tensor gy = sum_to_shape (where (cond_, z, g), sy_);
    return {gx, gy};
  }

private:
  Tensor cond_;
  Shape_Vector sx_;
  Shape_Vector sy_;
  Dtype dt_;
};

} /* namespace */

Tensor
where (const Tensor &condition, const Tensor &x, const Tensor &y)
{
  check_same_device (condition, x, "where");
  check_same_device (x, y, "where");
  Shape_Vector out_shape = broadcast_shapes (
    condition.shape (), broadcast_shapes (x.shape (), y.shape ()));
  Tensor c_exp = (condition.shape () == out_shape)
                 ? condition
                 : condition.expand (out_shape);
  Tensor x_exp = (x.shape () == out_shape) ? x : x.expand (out_shape);
  Tensor y_exp = (y.shape () == out_shape) ? y : y.expand (out_shape);
  Dtype dt = promote_dtypes (x.dtype (), y.dtype ());

  Tensor out (out_shape,
              Tensor_Options ().with_dtype (dt).with_device (x.device ()));
  size_t n = out.numel ();
  for (size_t i = 0; i < n; ++i)
    {
      double cond_v = c_exp.read_scalar_at (i);
      out.write_scalar_at (
        i, (cond_v != 0.0) ? x_exp.read_scalar_at (i)
                           : y_exp.read_scalar_at (i));
    }
  attach_grad ({x, y}, out,
               std::make_shared<Where_Backward> (
                 c_exp, x.shape (), y.shape (), dt));
  return out;
}

Tensor
masked_fill (const Tensor &x, const Tensor &mask, double value)
{
  Tensor val_t = full (x.shape (), value,
                       x.options ().with_requires_grad (false));
  return where (mask, val_t, x);
}

Tensor
eq (const Tensor &a, const Tensor &b)
{
  return binary_kernel (a, b, Dtype::Bool, [] (double x, double y)
    {
      return (x == y) ? 1.0 : 0.0;
    });
}

Tensor
ne (const Tensor &a, const Tensor &b)
{
  return binary_kernel (a, b, Dtype::Bool, [] (double x, double y)
    {
      return (x != y) ? 1.0 : 0.0;
    });
}

Tensor
lt (const Tensor &a, const Tensor &b)
{
  return binary_kernel (a, b, Dtype::Bool, [] (double x, double y)
    {
      return (x < y) ? 1.0 : 0.0;
    });
}

Tensor
le (const Tensor &a, const Tensor &b)
{
  return binary_kernel (a, b, Dtype::Bool, [] (double x, double y)
    {
      return (x <= y) ? 1.0 : 0.0;
    });
}

Tensor
gt (const Tensor &a, const Tensor &b)
{
  return binary_kernel (a, b, Dtype::Bool, [] (double x, double y)
    {
      return (x > y) ? 1.0 : 0.0;
    });
}

Tensor
ge (const Tensor &a, const Tensor &b)
{
  return binary_kernel (a, b, Dtype::Bool, [] (double x, double y)
    {
      return (x >= y) ? 1.0 : 0.0;
    });
}

bool
allclose (const Tensor &a, const Tensor &b, double rtol, double atol)
{
  if (!a.defined () || !b.defined () || a.shape () != b.shape ())
    {
      return false;
    }
  size_t n = a.numel ();
  for (size_t i = 0; i < n; ++i)
    {
      double va = a.read_scalar_at (i);
      double vb = b.read_scalar_at (i);
      if (std::isnan (va) || std::isnan (vb))
        {
          return false;
        }
      if (std::fabs (va - vb) > atol + rtol * std::fabs (vb))
        {
          return false;
        }
    }
  return true;
}

/* Reductions.  */

namespace
{

class Sum_All_Backward : public Autograd_Node
{
public:
  explicit Sum_All_Backward (Shape_Vector orig_shape)
    : orig_shape_ (std::move (orig_shape))
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Sum_All_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    Tensor g = grads[0];
    while (g.ndim () < orig_shape_.size ())
      {
        g = g.unsqueeze (0);
      }
    return {g.expand (orig_shape_)};
  }

private:
  Shape_Vector orig_shape_;
};

class Sum_Dim_Backward : public Autograd_Node
{
public:
  Sum_Dim_Backward (Shape_Vector orig_shape, size_t dim, bool keepdim)
    : orig_shape_ (std::move (orig_shape)),
      dim_ (dim),
      keepdim_ (keepdim)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Sum_Dim_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    Tensor g = grads[0];
    if (!keepdim_)
      {
        g = g.unsqueeze (static_cast<ptrdiff_t> (dim_));
      }
    return {g.expand (orig_shape_)};
  }

private:
  Shape_Vector orig_shape_;
  size_t dim_;
  bool keepdim_;
};

} /* namespace */

Tensor
sum (const Tensor &x)
{
  BASALT_ASSERT (x.defined (), "sum() called on undefined Tensor");
  double acc = 0.0;
  size_t n = x.numel ();
  if (x.is_contiguous () && x.dtype () == Dtype::Float32)
    {
      if (x.device ().is_cuda ())
        {
          bellows_compat::Cuda_Runtime_Driver::instance ().sync_if_busy ();
        }
      const float *xp = x.data_ptr<float> ();
      for (size_t i = 0; i < n; ++i)
        {
          acc += static_cast<double> (xp[i]);
        }
    }
  else
    {
      for (size_t i = 0; i < n; ++i)
        {
          acc += x.read_scalar_at (i);
        }
    }
  Dtype out_dt = is_floating_point (x.dtype ()) ? x.dtype () : Dtype::Int64;
  Tensor out ({1},
              Tensor_Options ().with_dtype (out_dt).with_device (x.device ()));
  out.write_scalar_at (0, acc);
  attach_grad ({x}, out, std::make_shared<Sum_All_Backward> (x.shape ()));
  return out;
}

Tensor
sum (const Tensor &x, ptrdiff_t dim, bool keepdim)
{
  BASALT_ASSERT (x.defined (), "sum(dim) called on undefined Tensor");
  size_t d = detail::normalize_dim (dim, x.ndim ());
  const Shape_Vector &in_shape = x.shape ();

  Shape_Vector out_shape;
  for (size_t i = 0; i < in_shape.size (); ++i)
    {
      if (i == d)
        {
          if (keepdim)
            {
              out_shape.push_back (1);
            }
        }
      else
        {
          out_shape.push_back (in_shape[i]);
        }
    }
  if (out_shape.empty ())
    {
      out_shape.push_back (1);
    }

  size_t outer_ext = 1;
  for (size_t i = 0; i < d; ++i)
    {
      outer_ext *= in_shape[i];
    }
  size_t reduce_ext = in_shape[d];
  size_t inner_ext = 1;
  for (size_t i = d + 1; i < in_shape.size (); ++i)
    {
      inner_ext *= in_shape[i];
    }

  Dtype out_dt = is_floating_point (x.dtype ()) ? x.dtype () : Dtype::Int64;
  Tensor out (out_shape,
              Tensor_Options ().with_dtype (out_dt).with_device (x.device ()));

  if (x.is_contiguous () && x.dtype () == Dtype::Float32
      && out_dt == Dtype::Float32)
    {
      auto &cuda_drv = bellows_compat::Cuda_Runtime_Driver::instance ();
      if (inner_ext == 1 && x.device ().is_cuda ()
          && cuda_drv.launch_sum_last_f32 (x.device ().index (),
                                           x.data_ptr<float> (),
                                           out.data_ptr<float> (),
                                           outer_ext, reduce_ext))
        {
          attach_grad (
            {x}, out,
            std::make_shared<Sum_Dim_Backward> (x.shape (), d, keepdim));
          return out;
        }
      if (x.device ().is_cuda ())
        {
          cuda_drv.sync_if_busy ();
        }
      const float *xp = x.data_ptr<float> ();
      float *op = out.data_ptr<float> ();
      if (inner_ext == 1)
        {
          for (size_t o = 0; o < outer_ext; ++o)
            {
              const float *row = xp + o * reduce_ext;
              float acc = 0.0f;
              for (size_t r = 0; r < reduce_ext; ++r)
                {
                  acc += row[r];
                }
              op[o] = acc;
            }
        }
      else
        {
          for (size_t o = 0; o < outer_ext; ++o)
            {
              float *out_row = op + o * inner_ext;
              for (size_t r = 0; r < reduce_ext; ++r)
                {
                  const float *in_row = xp + (o * reduce_ext + r) * inner_ext;
                  for (size_t in = 0; in < inner_ext; ++in)
                    {
                      out_row[in] += in_row[in];
                    }
                }
            }
        }
    }
  else
    {
      for (size_t o = 0; o < outer_ext; ++o)
        {
          for (size_t in = 0; in < inner_ext; ++in)
            {
              double acc = 0.0;
              for (size_t r = 0; r < reduce_ext; ++r)
                {
                  size_t idx = (o * reduce_ext + r) * inner_ext + in;
                  acc += x.read_scalar_at (idx);
                }
              out.write_scalar_at (o * inner_ext + in, acc);
            }
        }
    }

  attach_grad (
    {x}, out,
    std::make_shared<Sum_Dim_Backward> (x.shape (), d, keepdim));
  return out;
}

Tensor
sum_to_shape (const Tensor &x, const Shape_Vector &target_shape)
{
  if (x.shape () == target_shape)
    {
      return x;
    }
  if (target_shape.empty ()
      || (target_shape.size () == 1 && target_shape[0] == 1
          && x.numel () > 1 && x.ndim () > 1))
    {
      return sum (x).reshape (target_shape);
    }

  Tensor cur = x;
  while (cur.ndim () > target_shape.size ())
    {
      cur = sum (cur, 0, false);
    }
  for (size_t i = 0; i < target_shape.size (); ++i)
    {
      if (target_shape[i] == 1 && cur.size (static_cast<ptrdiff_t> (i)) > 1)
        {
          cur = sum (cur, static_cast<ptrdiff_t> (i), true);
        }
    }
  if (cur.shape () != target_shape)
    {
      cur = cur.reshape (target_shape);
    }
  return cur;
}

Tensor
mean (const Tensor &x)
{
  double denom = static_cast<double> (std::max<size_t> (1, x.numel ()));
  return div_scalar (sum (x), denom);
}

Tensor
mean (const Tensor &x, ptrdiff_t dim, bool keepdim)
{
  size_t d = detail::normalize_dim (dim, x.ndim ());
  double denom = static_cast<double> (
    std::max<size_t> (1, x.size (static_cast<ptrdiff_t> (d))));
  return div_scalar (sum (x, dim, keepdim), denom);
}

Tensor
var (const Tensor &x, bool unbiased)
{
  Tensor m = mean (x);
  Tensor diff = sub (x, m);
  size_t n = x.numel ();
  double denom = static_cast<double> (
    (unbiased && n > 1) ? (n - 1) : std::max<size_t> (1, n));
  return div_scalar (sum (mul (diff, diff)), denom);
}

Tensor
var (const Tensor &x, ptrdiff_t dim, bool unbiased, bool keepdim)
{
  size_t d = detail::normalize_dim (dim, x.ndim ());
  Tensor m = mean (x, static_cast<ptrdiff_t> (d), true);
  Tensor diff = sub (x, m);
  size_t n = x.size (static_cast<ptrdiff_t> (d));
  double denom = static_cast<double> (
    (unbiased && n > 1) ? (n - 1) : std::max<size_t> (1, n));
  return div_scalar (
    sum (mul (diff, diff), static_cast<ptrdiff_t> (d), keepdim), denom);
}

Tensor
max (const Tensor &x)
{
  BASALT_ASSERT (x.defined () && x.numel () > 0,
                 "max() requires a non-empty tensor");
  size_t best_idx = 0;
  double best_val = x.read_scalar_at (0);
  size_t n = x.numel ();
  for (size_t i = 1; i < n; ++i)
    {
      double v = x.read_scalar_at (i);
      if (v > best_val)
        {
          best_val = v;
          best_idx = i;
        }
    }
  Tensor out ({1},
              Tensor_Options ().with_dtype (x.dtype ()).with_device (
                x.device ()));
  out.write_scalar_at (0, best_val);

  Shape_Vector orig_shape = x.shape ();
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 Tensor (),
                 [orig_shape, best_idx] (const Tensor &g, const Tensor &)
                   {
                     Tensor gx = zeros (
                       orig_shape,
                       g.options ().with_requires_grad (false));
                     gx.write_scalar_at (best_idx, 1.0);
                     return mul (gx, g.expand (orig_shape));
                   },
                 "Max_All_Backward"));
  return out;
}

Tensor
max (const Tensor &x, ptrdiff_t dim, bool keepdim)
{
  BASALT_ASSERT (x.defined () && x.numel () > 0,
                 "max(dim) requires a non-empty tensor");
  size_t d = detail::normalize_dim (dim, x.ndim ());
  const Shape_Vector &in_shape = x.shape ();

  Shape_Vector out_shape;
  for (size_t i = 0; i < in_shape.size (); ++i)
    {
      if (i == d)
        {
          if (keepdim)
            {
              out_shape.push_back (1);
            }
        }
      else
        {
          out_shape.push_back (in_shape[i]);
        }
    }
  if (out_shape.empty ())
    {
      out_shape.push_back (1);
    }

  size_t outer_ext = 1;
  for (size_t i = 0; i < d; ++i)
    {
      outer_ext *= in_shape[i];
    }
  size_t reduce_ext = in_shape[d];
  size_t inner_ext = 1;
  for (size_t i = d + 1; i < in_shape.size (); ++i)
    {
      inner_ext *= in_shape[i];
    }

  Tensor out (out_shape,
              Tensor_Options ().with_dtype (x.dtype ()).with_device (
                x.device ()));
  std::vector<size_t> argmax_flat (outer_ext * inner_ext, 0);

  for (size_t o = 0; o < outer_ext; ++o)
    {
      for (size_t in = 0; in < inner_ext; ++in)
        {
          size_t best_flat = (o * reduce_ext) * inner_ext + in;
          double best_val = x.read_scalar_at (best_flat);
          for (size_t r = 1; r < reduce_ext; ++r)
            {
              size_t idx = (o * reduce_ext + r) * inner_ext + in;
              double v = x.read_scalar_at (idx);
              if (v > best_val)
                {
                  best_val = v;
                  best_flat = idx;
                }
            }
          out.write_scalar_at (o * inner_ext + in, best_val);
          argmax_flat[o * inner_ext + in] = best_flat;
        }
    }

  Shape_Vector orig_shape = x.shape ();
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 Tensor (),
                 [orig_shape, d, keepdim, argmax_flat] (
                   const Tensor &g, const Tensor &)
                   {
                     Tensor g_exp = keepdim
                       ? g.expand (orig_shape)
                       : g.unsqueeze (static_cast<ptrdiff_t> (d))
                           .expand (orig_shape);
                     Tensor mask = zeros (
                       orig_shape, g.options ().with_requires_grad (false));
                     for (size_t flat_i : argmax_flat)
                       {
                         mask.write_scalar_at (flat_i, 1.0);
                       }
                     return mul (g_exp, mask);
                   },
                 "Max_Dim_Backward"));
  return out;
}

Tensor
min (const Tensor &x)
{
  return neg (max (neg (x)));
}

Tensor
min (const Tensor &x, ptrdiff_t dim, bool keepdim)
{
  return neg (max (neg (x), dim, keepdim));
}

Tensor
argmax (const Tensor &x, ptrdiff_t dim, bool keepdim)
{
  BASALT_ASSERT (x.defined () && x.numel () > 0,
                 "argmax() requires a non-empty tensor");
  size_t d = detail::normalize_dim (dim, x.ndim ());
  const Shape_Vector &in_shape = x.shape ();

  Shape_Vector out_shape;
  for (size_t i = 0; i < in_shape.size (); ++i)
    {
      if (i == d)
        {
          if (keepdim)
            {
              out_shape.push_back (1);
            }
        }
      else
        {
          out_shape.push_back (in_shape[i]);
        }
    }
  if (out_shape.empty ())
    {
      out_shape.push_back (1);
    }

  size_t outer_ext = 1;
  for (size_t i = 0; i < d; ++i)
    {
      outer_ext *= in_shape[i];
    }
  size_t reduce_ext = in_shape[d];
  size_t inner_ext = 1;
  for (size_t i = d + 1; i < in_shape.size (); ++i)
    {
      inner_ext *= in_shape[i];
    }

  Tensor out (out_shape,
              Tensor_Options ().with_dtype (Dtype::Int64).with_device (
                x.device ()));
  for (size_t o = 0; o < outer_ext; ++o)
    {
      for (size_t in = 0; in < inner_ext; ++in)
        {
          size_t best_r = 0;
          double best_val = x.read_scalar_at (
            (o * reduce_ext) * inner_ext + in);
          for (size_t r = 1; r < reduce_ext; ++r)
            {
              double v = x.read_scalar_at (
                (o * reduce_ext + r) * inner_ext + in);
              if (v > best_val)
                {
                  best_val = v;
                  best_r = r;
                }
            }
          out.write_scalar_at (o * inner_ext + in,
                               static_cast<double> (best_r));
        }
    }
  return out;
}

Tensor
argmin (const Tensor &x, ptrdiff_t dim, bool keepdim)
{
  return argmax (neg (x), dim, keepdim);
}

Tensor
logsumexp (const Tensor &x, ptrdiff_t dim, bool keepdim)
{
  Tensor m = max (x, dim, true).detach ();
  Tensor shifted = sub (x, m);
  Tensor s = sum (exp (shifted), dim, keepdim);
  Tensor m_out = keepdim ? m : m.squeeze (dim);
  return add (m_out, log (s));
}

namespace
{

class Softmax_Last_Dim_Backward : public Autograd_Node
{
public:
  explicit Softmax_Last_Dim_Backward (const Tensor &y)
    : y_ (y)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Softmax_Last_Dim_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    Tensor y = y_.unpack ();
    const Tensor &g = grads[0];
    if (!Grad_Mode::is_enabled () && g.is_contiguous () && y.is_contiguous ()
        && g.dtype () == Dtype::Float32 && y.dtype () == Dtype::Float32
        && g.shape () == y.shape () && y.ndim () >= 1)
      {
        Tensor gx (y.shape (), y.options ().with_requires_grad (false));
        size_t cols = y.size (-1);
        size_t rows = (cols > 0) ? (y.numel () / cols) : 0;
        const float *gp = g.data_ptr<float> ();
        const float *yp = y.data_ptr<float> ();
        float *gxp = gx.data_ptr<float> ();
        auto &cuda_drv = bellows_compat::Cuda_Runtime_Driver::instance ();
        if (y.device ().is_cuda ()
            && cuda_drv.launch_softmax_bwd_f32 (y.device ().index (),
                                                gp, yp, gxp, rows, cols))
          {
            return {gx};
          }
        if (y.device ().is_cuda ())
          {
            cuda_drv.sync_if_busy ();
          }
        for (size_t r = 0; r < rows; ++r)
          {
            const float *gr = gp + r * cols;
            const float *yr = yp + r * cols;
            float *gxr = gxp + r * cols;
            float dot_gy = 0.0f;
            for (size_t c = 0; c < cols; ++c)
              {
                dot_gy += gr[c] * yr[c];
              }
            for (size_t c = 0; c < cols; ++c)
              {
                gxr[c] = yr[c] * (gr[c] - dot_gy);
              }
          }
        return {gx};
      }
    Tensor dot_gy = sum (mul (g, y), -1, true);
    return {mul (y, sub (g, dot_gy))};
  }

  void
  release_saved_variables (void) noexcept override
  {
    y_ = Saved_Variable ();
  }

private:
  Saved_Variable y_;
};

} /* namespace */

Tensor
softmax (const Tensor &x, ptrdiff_t dim)
{
  BASALT_ASSERT (x.defined (), "softmax() called on undefined Tensor");
  size_t d = detail::normalize_dim (dim, x.ndim ());
  if (d + 1 == x.ndim () && x.is_contiguous ()
      && x.dtype () == Dtype::Float32 && x.numel () > 0)
    {
      Tensor out (x.shape (), x.options ().with_requires_grad (false));
      size_t cols = x.size (-1);
      size_t rows = x.numel () / cols;
      const float *xp = x.data_ptr<float> ();
      float *yp = out.data_ptr<float> ();
      auto &cuda_drv = bellows_compat::Cuda_Runtime_Driver::instance ();
      if (x.device ().is_cuda ()
          && cuda_drv.launch_softmax_fwd_f32 (x.device ().index (),
                                              xp, yp, rows, cols))
        {
          attach_grad ({x}, out,
                       std::make_shared<Softmax_Last_Dim_Backward> (out));
          return out;
        }
      if (x.device ().is_cuda ())
        {
          cuda_drv.sync_if_busy ();
        }
      auto row_fn = [&] (size_t r)
      {
        const float *xr = xp + r * cols;
        float *yr = yp + r * cols;
        float max_v = xr[0];
        for (size_t c = 1; c < cols; ++c)
          {
            if (xr[c] > max_v)
              {
                max_v = xr[c];
              }
          }
        float sum_exp = 0.0f;
        for (size_t c = 0; c < cols; ++c)
          {
            float e = std::exp (xr[c] - max_v);
            yr[c] = e;
            sum_exp += e;
          }
        float inv_sum = 1.0f / sum_exp;
        for (size_t c = 0; c < cols; ++c)
          {
            yr[c] *= inv_sum;
          }
      };
      if (rows * cols >= 65536 && rows >= 8)
        {
          hyper_math::parallel_for (size_t (0), rows, size_t (16), row_fn);
        }
      else
        {
          for (size_t r = 0; r < rows; ++r)
            {
              row_fn (r);
            }
        }
      attach_grad ({x}, out,
                   std::make_shared<Softmax_Last_Dim_Backward> (out));
      return out;
    }
  Tensor m = max (x, dim, true).detach ();
  Tensor ex = exp (sub (x, m));
  Tensor denom = sum (ex, dim, true);
  return div (ex, denom);
}

Tensor
log_softmax (const Tensor &x, ptrdiff_t dim)
{
  Tensor lse = logsumexp (x, dim, true);
  return sub (x, lse);
}

/* Matrix multiplication and linear algebra.  */

namespace
{

class Matmul_2D_Backward : public Autograd_Node
{
public:
  Matmul_2D_Backward (const Tensor &a, const Tensor &b)
    : a_ (a),
      b_ (b)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Matmul_2D_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    Tensor a = a_.unpack ();
    Tensor b = b_.unpack ();
    const Tensor &g = grads[0];
    Tensor ga = matmul (g, b.transpose (-2, -1));
    Tensor gb = matmul (a.transpose (-2, -1), g);
    return {sum_to_shape (ga, a.shape ()), sum_to_shape (gb, b.shape ())};
  }

  void
  release_saved_variables (void) noexcept override
  {
    a_ = Saved_Variable ();
    b_ = Saved_Variable ();
  }

private:
  Saved_Variable a_;
  Saved_Variable b_;
};

} /* namespace */

Tensor
matmul (const Tensor &a, const Tensor &b)
{
  check_same_device (a, b, "matmul");
  BASALT_ASSERT_SHAPE (a.ndim () >= 1 && b.ndim () >= 1,
                       "matmul() requires rank >= 1 tensors");

  if (a.ndim () == 1 && b.ndim () == 1)
    {
      return dot (a, b);
    }
  if (a.ndim () == 1)
    {
      return matmul (a.unsqueeze (0), b).squeeze (-2);
    }
  if (b.ndim () == 1)
    {
      return matmul (a, b.unsqueeze (-1)).squeeze (-1);
    }

  size_t m = a.size (-2);
  size_t k = a.size (-1);
  BASALT_ASSERT_SHAPE (b.size (-2) == k,
                       "matmul() inner dimension mismatch");
  size_t n = b.size (-1);

  Shape_Vector a_batch (a.shape ().begin (), a.shape ().end () - 2);
  Shape_Vector b_batch (b.shape ().begin (), b.shape ().end () - 2);
  Shape_Vector out_batch = broadcast_shapes (a_batch, b_batch);

  Shape_Vector a_target = out_batch;
  a_target.push_back (m);
  a_target.push_back (k);
  Shape_Vector b_target = out_batch;
  b_target.push_back (k);
  b_target.push_back (n);

  Tensor a_exp = (a.shape () == a_target) ? a : a.expand (a_target);
  Tensor b_exp = (b.shape () == b_target) ? b : b.expand (b_target);

  Shape_Vector out_shape = out_batch;
  out_shape.push_back (m);
  out_shape.push_back (n);

  size_t batch_count = detail::compute_numel (out_batch);
  Dtype dt = promote_dtypes (a.dtype (), b.dtype ());
  Tensor out = zeros (
    out_shape, Tensor_Options ().with_dtype (dt).with_device (a.device ()));

  if (dt == Dtype::Float32 && a_exp.dtype () == Dtype::Float32
      && b_exp.dtype () == Dtype::Float32 && m > 0 && n > 0 && k > 0)
    {
      No_Grad_Guard ng;
      bool done = false;
      auto &cuda_drv = bellows_compat::Cuda_Runtime_Driver::instance ();
      if (a.device ().is_cuda ()
          && cuda_drv.has_cublas (a.device ().index ()))
        {
          size_t dev_idx = a.device ().index ();
          if (batch_count == 1)
            {
              bool trans_a = false;
              size_t lda = k;
              Tensor a_holder;
              const float *a_ptr = nullptr;
              if (a_exp.is_contiguous ())
                {
                  a_ptr = a_exp.data_ptr<float> ();
                }
              else if (a_exp.ndim () == 2 && a_exp.strides ()[0] == 1
                       && a_exp.strides ()[1] >= static_cast<ptrdiff_t> (m))
                {
                  trans_a = true;
                  lda = static_cast<size_t> (a_exp.strides ()[1]);
                  a_ptr = a_exp.data_ptr<float> ();
                }
              else
                {
                  a_holder = a_exp.contiguous ();
                  a_ptr = a_holder.data_ptr<float> ();
                }

              bool trans_b = false;
              size_t ldb = n;
              Tensor b_holder;
              const float *b_ptr = nullptr;
              if (b_exp.is_contiguous ())
                {
                  b_ptr = b_exp.data_ptr<float> ();
                }
              else if (b_exp.ndim () == 2 && b_exp.strides ()[0] == 1
                       && b_exp.strides ()[1] >= static_cast<ptrdiff_t> (k))
                {
                  trans_b = true;
                  ldb = static_cast<size_t> (b_exp.strides ()[1]);
                  b_ptr = b_exp.data_ptr<float> ();
                }
              else
                {
                  b_holder = b_exp.contiguous ();
                  b_ptr = b_holder.data_ptr<float> ();
                }

              float *c_ptr = out.data_ptr<float> ();
              if (cuda_drv.cublas_sgemm (dev_idx, trans_a, trans_b, m, n, k,
                                         1.0f, a_ptr, lda, b_ptr, ldb, 0.0f,
                                         c_ptr, n))
                {
                  done = true;
                }
            }
          else
            {
              Tensor a_c = a_exp.is_contiguous ()
                           ? a_exp
                           : a_exp.contiguous ();
              Tensor b_c = b_exp.is_contiguous ()
                           ? b_exp
                           : b_exp.contiguous ();
              if (cuda_drv.cublas_sgemm_batched (
                    dev_idx, false, false, m, n, k, 1.0f,
                    a_c.data_ptr<float> (), k,
                    static_cast<long long> (m * k),
                    b_c.data_ptr<float> (), n,
                    static_cast<long long> (k * n), 0.0f,
                    out.data_ptr<float> (), n,
                    static_cast<long long> (m * n), batch_count))
                {
                  done = true;
                }
            }
        }

      /* Fast CPU Float32 path when B is a 2-D transposed matrix [k, n]
         backed by contiguous [n, k] storage (e.g. Linear forward & Matmul
         backward).  Avoids transposing B and uses contiguous dot products.  */
      if (!done && batch_count == 1 && a_exp.is_contiguous ()
          && b_exp.ndim () == 2
          && b_exp.strides ()[0] == 1
          && b_exp.strides ()[1] == static_cast<ptrdiff_t> (k))
        {
          if (a.device ().is_cuda ())
            {
              cuda_drv.sync_if_busy ();
            }
          const float *__restrict__ a_ptr = a_exp.data_ptr<float> ();
          const float *__restrict__ bt_ptr = b_exp.data_ptr<float> ();
          float *__restrict__ c_ptr = out.data_ptr<float> ();
          size_t m_blocks = (m + 3) / 4;
          auto blk_fn = [&] (size_t blk)
          {
            size_t i0 = blk * 4;
            size_t i_end = std::min (m, i0 + 4);
            if (i_end - i0 == 4)
              {
                const float *__restrict__ a0 = a_ptr + (i0 + 0) * k;
                const float *__restrict__ a1 = a_ptr + (i0 + 1) * k;
                const float *__restrict__ a2 = a_ptr + (i0 + 2) * k;
                const float *__restrict__ a3 = a_ptr + (i0 + 3) * k;
                float *__restrict__ c0 = c_ptr + (i0 + 0) * n;
                float *__restrict__ c1 = c_ptr + (i0 + 1) * n;
                float *__restrict__ c2 = c_ptr + (i0 + 2) * n;
                float *__restrict__ c3 = c_ptr + (i0 + 3) * n;
                for (size_t j = 0; j < n; ++j)
                  {
                    const float *__restrict__ bt_row = bt_ptr + j * k;
                    float s0 = 0.0f;
                    float s1 = 0.0f;
                    float s2 = 0.0f;
                    float s3 = 0.0f;
                    for (size_t p = 0; p < k; ++p)
                      {
                        float bv = bt_row[p];
                        s0 += a0[p] * bv;
                        s1 += a1[p] * bv;
                        s2 += a2[p] * bv;
                        s3 += a3[p] * bv;
                      }
                    c0[j] = s0;
                    c1[j] = s1;
                    c2[j] = s2;
                    c3[j] = s3;
                  }
              }
            else
              {
                for (size_t i = i0; i < i_end; ++i)
                  {
                    const float *__restrict__ a_row = a_ptr + i * k;
                    float *__restrict__ c_row = c_ptr + i * n;
                    for (size_t j = 0; j < n; ++j)
                      {
                        const float *__restrict__ bt_row = bt_ptr + j * k;
                        float acc = 0.0f;
                        for (size_t p = 0; p < k; ++p)
                          {
                            acc += a_row[p] * bt_row[p];
                          }
                        c_row[j] = acc;
                      }
                  }
              }
          };
          if (m * n * k >= 262144 && m_blocks >= 4)
            {
              size_t grain = (k * n > 0)
                             ? std::max<size_t> (1, 8192 / (k * n))
                             : 1;
              hyper_math::parallel_for (size_t (0), m_blocks, grain, blk_fn);
            }
          else
            {
              for (size_t blk = 0; blk < m_blocks; ++blk)
                {
                  blk_fn (blk);
                }
            }
          done = true;
        }

      if (!done)
        {
          Tensor a_c = a_exp.is_contiguous () ? a_exp : a_exp.contiguous ();
          Tensor b_c = b_exp.is_contiguous () ? b_exp : b_exp.contiguous ();
          if (a.device ().is_cuda ())
            {
              cuda_drv.sync_if_busy ();
            }
          const float *__restrict__ a_ptr = a_c.data_ptr<float> ();
          const float *__restrict__ b_ptr = b_c.data_ptr<float> ();
          float *__restrict__ c_ptr = out.data_ptr<float> ();

          /* 4-row register-blocked and K-cache-tiled AVX2/FMA GEMM kernel.  */
          size_t m_blocks = (m + 3) / 4;
          size_t total_blocks = batch_count * m_blocks;
          auto blk_fn = [&] (size_t blk_idx)
          {
            size_t batch_idx = blk_idx / m_blocks;
            size_t i0 = (blk_idx % m_blocks) * 4;
            size_t i_end = std::min (m, i0 + 4);
            const float *__restrict__ a_mat = a_ptr + batch_idx * m * k;
            const float *__restrict__ b_mat = b_ptr + batch_idx * k * n;
            float *__restrict__ c_mat = c_ptr + batch_idx * m * n;

            if (i_end - i0 == 4)
              {
                const float *__restrict__ a0 = a_mat + (i0 + 0) * k;
                const float *__restrict__ a1 = a_mat + (i0 + 1) * k;
                const float *__restrict__ a2 = a_mat + (i0 + 2) * k;
                const float *__restrict__ a3 = a_mat + (i0 + 3) * k;
                float *__restrict__ c0 = c_mat + (i0 + 0) * n;
                float *__restrict__ c1 = c_mat + (i0 + 1) * n;
                float *__restrict__ c2 = c_mat + (i0 + 2) * n;
                float *__restrict__ c3 = c_mat + (i0 + 3) * n;

                size_t j0 = 0;
                for (; j0 + 16 <= n; j0 += 16)
                  {
                    float acc0[16] = {0.0f};
                    float acc1[16] = {0.0f};
                    float acc2[16] = {0.0f};
                    float acc3[16] = {0.0f};
                    for (size_t p = 0; p < k; ++p)
                      {
                        float v0 = a0[p];
                        float v1 = a1[p];
                        float v2 = a2[p];
                        float v3 = a3[p];
                        const float *__restrict__ bp = b_mat + p * n + j0;
                        for (size_t jj = 0; jj < 16; ++jj)
                          {
                            float bv = bp[jj];
                            acc0[jj] += v0 * bv;
                            acc1[jj] += v1 * bv;
                            acc2[jj] += v2 * bv;
                            acc3[jj] += v3 * bv;
                          }
                      }
                    for (size_t jj = 0; jj < 16; ++jj)
                      {
                        c0[j0 + jj] = acc0[jj];
                        c1[j0 + jj] = acc1[jj];
                        c2[j0 + jj] = acc2[jj];
                        c3[j0 + jj] = acc3[jj];
                      }
                  }
                if (j0 < n)
                  {
                    for (size_t p = 0; p < k; ++p)
                      {
                        float v0 = a0[p];
                        float v1 = a1[p];
                        float v2 = a2[p];
                        float v3 = a3[p];
                        const float *__restrict__ b_row = b_mat + p * n;
                        for (size_t j = j0; j < n; ++j)
                          {
                            float bv = b_row[j];
                            c0[j] += v0 * bv;
                            c1[j] += v1 * bv;
                            c2[j] += v2 * bv;
                            c3[j] += v3 * bv;
                          }
                      }
                  }
              }
            else
              {
                for (size_t i = i0; i < i_end; ++i)
                  {
                    const float *__restrict__ a_row = a_mat + i * k;
                    float *__restrict__ c_row = c_mat + i * n;
                    for (size_t p = 0; p < k; ++p)
                      {
                        float a_ip = a_row[p];
                        const float *__restrict__ b_row = b_mat + p * n;
                        for (size_t j = 0; j < n; ++j)
                          {
                            c_row[j] += a_ip * b_row[j];
                          }
                      }
                  }
              }
          };
          if (batch_count * m * n * k >= 262144 && total_blocks >= 4)
            {
              size_t grain = (k * n > 0)
                             ? std::max<size_t> (1, 8192 / (k * n))
                             : 1;
              hyper_math::parallel_for (
                size_t (0), total_blocks, grain, blk_fn);
            }
          else
            {
              for (size_t blk_idx = 0; blk_idx < total_blocks; ++blk_idx)
                {
                  blk_fn (blk_idx);
                }
            }
        }
    }
  else if ((dt == Dtype::Float16 || dt == Dtype::BFloat16)
           && a_exp.dtype () == dt && b_exp.dtype () == dt
           && m > 0 && n > 0 && k > 0)
    {
      No_Grad_Guard ng;
      bool done = false;
      bool is_bf16 = (dt == Dtype::BFloat16);
      auto &cuda_drv = bellows_compat::Cuda_Runtime_Driver::instance ();
      if (a.device ().is_cuda ()
          && cuda_drv.has_cublas (a.device ().index ()))
        {
          size_t dev_idx = a.device ().index ();
          if (batch_count == 1)
            {
              bool trans_a = false;
              size_t lda = k;
              Tensor a_holder;
              const void *a_ptr = nullptr;
              if (a_exp.is_contiguous ())
                {
                  a_ptr = a_exp.raw_data_ptr ();
                }
              else if (a_exp.ndim () == 2 && a_exp.strides ()[0] == 1
                       && a_exp.strides ()[1] >= static_cast<ptrdiff_t> (m))
                {
                  trans_a = true;
                  lda = static_cast<size_t> (a_exp.strides ()[1]);
                  a_ptr = a_exp.raw_data_ptr ();
                }
              else
                {
                  a_holder = a_exp.contiguous ();
                  a_ptr = a_holder.raw_data_ptr ();
                }

              bool trans_b = false;
              size_t ldb = n;
              Tensor b_holder;
              const void *b_ptr = nullptr;
              if (b_exp.is_contiguous ())
                {
                  b_ptr = b_exp.raw_data_ptr ();
                }
              else if (b_exp.ndim () == 2 && b_exp.strides ()[0] == 1
                       && b_exp.strides ()[1] >= static_cast<ptrdiff_t> (k))
                {
                  trans_b = true;
                  ldb = static_cast<size_t> (b_exp.strides ()[1]);
                  b_ptr = b_exp.raw_data_ptr ();
                }
              else
                {
                  b_holder = b_exp.contiguous ();
                  b_ptr = b_holder.raw_data_ptr ();
                }

              if (cuda_drv.cublas_gemm_half (dev_idx, is_bf16, trans_a,
                                             trans_b, m, n, k, 1.0f, a_ptr,
                                             lda, b_ptr, ldb, 0.0f,
                                             out.raw_data_ptr (), n))
                {
                  done = true;
                }
            }
          else
            {
              Tensor a_c = a_exp.is_contiguous ()
                           ? a_exp
                           : a_exp.contiguous ();
              Tensor b_c = b_exp.is_contiguous ()
                           ? b_exp
                           : b_exp.contiguous ();
              if (cuda_drv.cublas_gemm_half_batched (
                    dev_idx, is_bf16, false, false, m, n, k, 1.0f,
                    a_c.raw_data_ptr (), k,
                    static_cast<long long> (m * k),
                    b_c.raw_data_ptr (), n,
                    static_cast<long long> (k * n), 0.0f,
                    out.raw_data_ptr (), n,
                    static_cast<long long> (m * n), batch_count))
                {
                  done = true;
                }
            }
        }
      if (!done)
        {
          Tensor a_f32 = a_exp.contiguous ().to (Dtype::Float32);
          Tensor b_f32 = b_exp.contiguous ().to (Dtype::Float32);
          out = matmul (a_f32, b_f32).to (dt);
        }
    }
  else
    {
      hyper_math::parallel_for (
        size_t (0), batch_count, size_t (1), [&] (size_t batch_idx)
        {
          size_t a_base = batch_idx * m * k;
          size_t b_base = batch_idx * k * n;
          size_t c_base = batch_idx * m * n;
          for (size_t i = 0; i < m; ++i)
            {
              for (size_t j = 0; j < n; ++j)
                {
                  double acc = 0.0;
                  for (size_t p = 0; p < k; ++p)
                    {
                      acc += a_exp.read_scalar_at (a_base + i * k + p)
                             * b_exp.read_scalar_at (b_base + p * n + j);
                    }
                  out.write_scalar_at (c_base + i * n + j, acc);
                }
            }
        });
    }

  attach_grad ({a, b}, out, std::make_shared<Matmul_2D_Backward> (a, b));
  return out;
}

Tensor
bmm (const Tensor &a, const Tensor &b)
{
  BASALT_ASSERT_SHAPE (a.ndim () == 3 && b.ndim () == 3,
                       "bmm() requires 3-D tensors");
  BASALT_ASSERT_SHAPE (a.size (0) == b.size (0),
                       "bmm() batch size mismatch");
  return matmul (a, b);
}

Tensor
addmm (const Tensor &bias, const Tensor &input, const Tensor &weight,
       double beta, double alpha)
{
  Tensor mm = matmul (input, weight);
  if (alpha != 1.0)
    {
      mm = mul_scalar (mm, alpha);
    }
  Tensor b_term = (beta == 1.0) ? bias : mul_scalar (bias, beta);
  return add (b_term, mm);
}

Tensor
dot (const Tensor &a, const Tensor &b)
{
  BASALT_ASSERT_SHAPE (a.ndim () == 1 && b.ndim () == 1
                       && a.numel () == b.numel (),
                       "dot() requires 1-D tensors of equal length");
  return sum (mul (a, b));
}

Tensor
outer (const Tensor &a, const Tensor &b)
{
  BASALT_ASSERT_SHAPE (a.ndim () == 1 && b.ndim () == 1,
                       "outer() requires 1-D tensors");
  return matmul (a.unsqueeze (1), b.unsqueeze (0));
}

Tensor
norm (const Tensor &x, double p)
{
  BASALT_ASSERT_ARG (p > 0.0, "norm() requires p > 0");
  if (p == 2.0)
    {
      return sqrt (sum (square (x)));
    }
  return pow_scalar (sum (pow_scalar (abs (x), p)), 1.0 / p);
}

/* Concatenation, stacking, splitting, indexing, tril/triu, and padding.  */

namespace
{

class Cat_Backward : public Autograd_Node
{
public:
  Cat_Backward (std::vector<size_t> sizes, size_t dim)
    : sizes_ (std::move (sizes)),
      dim_ (dim)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Cat_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    const Tensor &g = grads[0];
    std::vector<Tensor> out;
    out.reserve (sizes_.size ());
    ptrdiff_t offset = 0;
    for (size_t sz : sizes_)
      {
        ptrdiff_t end = offset + static_cast<ptrdiff_t> (sz);
        out.push_back (
          g.slice (static_cast<ptrdiff_t> (dim_), offset, end, 1));
        offset = end;
      }
    return out;
  }

private:
  std::vector<size_t> sizes_;
  size_t dim_;
};

class Index_Select_Backward : public Autograd_Node
{
public:
  Index_Select_Backward (Shape_Vector orig_shape, Tensor_Options opts,
                         size_t dim, Tensor indices)
    : orig_shape_ (std::move (orig_shape)),
      opts_ (opts.with_requires_grad (false)),
      dim_ (dim),
      indices_ (std::move (indices))
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "Index_Select_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    No_Grad_Guard no_grad;
    const Tensor &g = grads[0];
    Tensor gx = zeros (orig_shape_, opts_);
    size_t num_idx = indices_.numel ();
    for (size_t i = 0; i < num_idx; ++i)
      {
        ptrdiff_t dst_i = static_cast<ptrdiff_t> (indices_.read_scalar_at (i));
        Tensor dst_slice = gx.select (static_cast<ptrdiff_t> (dim_), dst_i);
        Tensor src_slice = g.select (
          static_cast<ptrdiff_t> (dim_), static_cast<ptrdiff_t> (i));
        dst_slice.add_ (src_slice);
      }
    return {gx};
  }

private:
  Shape_Vector orig_shape_;
  Tensor_Options opts_;
  size_t dim_;
  Tensor indices_;
};

} /* namespace */

Tensor
cat (const std::vector<Tensor> &tensors, ptrdiff_t dim)
{
  BASALT_ASSERT_ARG (!tensors.empty (), "cat() tensor list cannot be empty");
  const Tensor &first = tensors[0];
  size_t d = detail::normalize_dim (dim, first.ndim ());
  Shape_Vector out_shape = first.shape ();
  size_t total_dim = 0;
  std::vector<size_t> dim_sizes;

  for (const Tensor &t : tensors)
    {
      BASALT_ASSERT_DEVICE (t.device () == first.device (),
                            "cat() device mismatch");
      BASALT_ASSERT_SHAPE (t.ndim () == first.ndim (), "cat() rank mismatch");
      for (size_t i = 0; i < first.ndim (); ++i)
        {
          if (i != d)
            {
              BASALT_ASSERT_SHAPE (t.shape ()[i] == first.shape ()[i],
                                   "cat() non-concat dimension mismatch");
            }
        }
      total_dim += t.shape ()[d];
      dim_sizes.push_back (t.shape ()[d]);
    }
  out_shape[d] = total_dim;

  Tensor out (out_shape,
              first.options ().with_requires_grad (false));
  ptrdiff_t offset = 0;
  {
    No_Grad_Guard no_grad;
    for (const Tensor &t : tensors)
      {
        ptrdiff_t next_off = offset + static_cast<ptrdiff_t> (t.shape ()[d]);
        Tensor dst = out.slice (
          static_cast<ptrdiff_t> (d), offset, next_off, 1);
        dst.copy_ (t);
        offset = next_off;
      }
  }
  attach_grad (tensors, out,
               std::make_shared<Cat_Backward> (std::move (dim_sizes), d));
  return out;
}

Tensor
stack (const std::vector<Tensor> &tensors, ptrdiff_t dim)
{
  BASALT_ASSERT_ARG (!tensors.empty (), "stack() tensor list cannot be empty");
  std::vector<Tensor> unsqueezed;
  unsqueezed.reserve (tensors.size ());
  for (const Tensor &t : tensors)
    {
      unsqueezed.push_back (t.unsqueeze (dim));
    }
  return cat (unsqueezed, dim);
}

std::vector<Tensor>
split (const Tensor &x, size_t split_size, ptrdiff_t dim)
{
  BASALT_ASSERT_ARG (split_size > 0, "split() requires split_size > 0");
  size_t d = detail::normalize_dim (dim, x.ndim ());
  size_t total = x.size (static_cast<ptrdiff_t> (d));
  std::vector<Tensor> chunks;
  for (size_t start = 0; start < total; start += split_size)
    {
      size_t end = std::min (start + split_size, total);
      chunks.push_back (
        x.slice (static_cast<ptrdiff_t> (d),
                 static_cast<ptrdiff_t> (start),
                 static_cast<ptrdiff_t> (end), 1));
    }
  return chunks;
}

std::vector<Tensor>
chunk (const Tensor &x, size_t num_chunks, ptrdiff_t dim)
{
  BASALT_ASSERT_ARG (num_chunks > 0, "chunk() requires num_chunks > 0");
  size_t d = detail::normalize_dim (dim, x.ndim ());
  size_t total = x.size (static_cast<ptrdiff_t> (d));
  size_t split_sz = std::max<size_t> (1, (total + num_chunks - 1)
                                         / num_chunks);
  return split (x, split_sz, dim);
}

Tensor
index_select (const Tensor &x, ptrdiff_t dim, const Tensor &indices)
{
  BASALT_ASSERT (x.defined () && indices.defined (),
                 "index_select() requires defined tensors");
  BASALT_ASSERT_SHAPE (indices.ndim () == 1,
                       "index_select() requires 1-D indices tensor");
  size_t d = detail::normalize_dim (dim, x.ndim ());
  Shape_Vector out_shape = x.shape ();
  out_shape[d] = indices.numel ();

  Tensor out (out_shape, x.options ().with_requires_grad (false));
  {
    No_Grad_Guard no_grad;
    size_t count = indices.numel ();
    for (size_t i = 0; i < count; ++i)
      {
        ptrdiff_t src_i = static_cast<ptrdiff_t> (indices.read_scalar_at (i));
        Tensor dst_sel = out.select (
          static_cast<ptrdiff_t> (d), static_cast<ptrdiff_t> (i));
        Tensor src_sel = x.select (static_cast<ptrdiff_t> (d), src_i);
        dst_sel.copy_ (src_sel);
      }
  }
  attach_grad ({x}, out,
               std::make_shared<Index_Select_Backward> (
                 x.shape (), x.options (), d, indices.detach ()));
  return out;
}

Tensor
gather (const Tensor &x, ptrdiff_t dim, const Tensor &index)
{
  BASALT_ASSERT (x.defined () && index.defined (),
                 "gather() requires defined tensors");
  BASALT_ASSERT_SHAPE (x.ndim () == index.ndim (),
                       "gather() rank mismatch");
  size_t d = detail::normalize_dim (dim, x.ndim ());
  const Shape_Vector &idx_shape = index.shape ();
  Tensor out (idx_shape, x.options ().with_requires_grad (false));

  size_t total = index.numel ();
  size_t r = x.ndim ();
  for (size_t flat = 0; flat < total; ++flat)
    {
      std::vector<size_t> coords (r, 0);
      size_t rem = flat;
      for (size_t k = r; k > 0; --k)
        {
          coords[k - 1] = rem % idx_shape[k - 1];
          rem /= idx_shape[k - 1];
        }
      size_t gather_idx = static_cast<size_t> (index.read_scalar_at (flat));
      BASALT_ASSERT_SHAPE (gather_idx < x.shape ()[d],
                           "gather() index out of bounds");
      coords[d] = gather_idx;
      size_t src_flat = 0;
      for (size_t k = 0; k < r; ++k)
        {
          src_flat = src_flat * x.shape ()[k] + coords[k];
        }
      out.write_scalar_at (flat, x.read_scalar_at (src_flat));
    }
  return out;
}

Tensor
tril (const Tensor &x, ptrdiff_t diagonal)
{
  BASALT_ASSERT_SHAPE (x.ndim () >= 2, "tril() requires ndim >= 2");
  size_t rows = x.size (-2);
  size_t cols = x.size (-1);
  Tensor mask ({rows, cols},
               Tensor_Options ().with_dtype (Dtype::Bool).with_device (
                 x.device ()));
  for (size_t r = 0; r < rows; ++r)
    {
      for (size_t c = 0; c < cols; ++c)
        {
          bool keep = (static_cast<ptrdiff_t> (c)
                       <= static_cast<ptrdiff_t> (r) + diagonal);
          mask.write_scalar_at (r * cols + c, keep ? 1.0 : 0.0);
        }
    }
  return where (mask, x, zeros_like (x));
}

Tensor
triu (const Tensor &x, ptrdiff_t diagonal)
{
  BASALT_ASSERT_SHAPE (x.ndim () >= 2, "triu() requires ndim >= 2");
  size_t rows = x.size (-2);
  size_t cols = x.size (-1);
  Tensor mask ({rows, cols},
               Tensor_Options ().with_dtype (Dtype::Bool).with_device (
                 x.device ()));
  for (size_t r = 0; r < rows; ++r)
    {
      for (size_t c = 0; c < cols; ++c)
        {
          bool keep = (static_cast<ptrdiff_t> (c)
                       >= static_cast<ptrdiff_t> (r) + diagonal);
          mask.write_scalar_at (r * cols + c, keep ? 1.0 : 0.0);
        }
    }
  return where (mask, x, zeros_like (x));
}

Tensor
pad2d (const Tensor &x, size_t pad_top, size_t pad_bottom,
       size_t pad_left, size_t pad_right, double value)
{
  BASALT_ASSERT_SHAPE (x.ndim () >= 2, "pad2d() requires ndim >= 2");
  if (pad_top == 0 && pad_bottom == 0 && pad_left == 0 && pad_right == 0)
    {
      return x;
    }
  Shape_Vector out_shape = x.shape ();
  size_t h_dim = out_shape.size () - 2;
  size_t w_dim = out_shape.size () - 1;
  size_t in_h = out_shape[h_dim];
  size_t in_w = out_shape[w_dim];
  out_shape[h_dim] = in_h + pad_top + pad_bottom;
  out_shape[w_dim] = in_w + pad_left + pad_right;

  Tensor out = full (out_shape, value,
                     x.options ().with_requires_grad (false));
  {
    No_Grad_Guard no_grad;
    Tensor sub_h = out.slice (
      static_cast<ptrdiff_t> (h_dim),
      static_cast<ptrdiff_t> (pad_top),
      static_cast<ptrdiff_t> (pad_top + in_h), 1);
    Tensor sub_hw = sub_h.slice (
      static_cast<ptrdiff_t> (w_dim),
      static_cast<ptrdiff_t> (pad_left),
      static_cast<ptrdiff_t> (pad_left + in_w), 1);
    sub_hw.copy_ (x);
  }
  attach_grad ({x}, out,
               std::make_shared<Lambda_Unary_Backward> (
                 Tensor (),
                 [h_dim, w_dim, pad_top, in_h, pad_left, in_w] (
                   const Tensor &g, const Tensor &)
                   {
                     Tensor gh = g.slice (
                       static_cast<ptrdiff_t> (h_dim),
                       static_cast<ptrdiff_t> (pad_top),
                       static_cast<ptrdiff_t> (pad_top + in_h), 1);
                     return gh.slice (
                       static_cast<ptrdiff_t> (w_dim),
                       static_cast<ptrdiff_t> (pad_left),
                       static_cast<ptrdiff_t> (pad_left + in_w), 1);
                   },
                 "Pad2d_Backward"));
  return out;
}

} /* namespace basalt::ops */
