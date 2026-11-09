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

#include <basalt/autograd.hpp>
#include <basalt/error.hpp>
#include <basalt/ops.hpp>
#include <basalt/random.hpp>
#include <basalt/tensor.hpp>
#include <cmath>
#include <vector>

namespace
{

void
test_reverse_mode_and_gradcheck (void)
{
  using namespace basalt;

  seed_everything (2026ULL);
  Tensor_Options opts = Tensor_Options ()
                          .with_dtype (Dtype::Float64)
                          .with_requires_grad (true);

  Tensor x = randn ({3, 4}, opts);
  Tensor w = randn ({4, 2}, opts);
  Tensor b = randn ({1, 2}, opts);

  auto fn = [] (const std::vector<Tensor> &inps) -> Tensor
    {
      Tensor h = matmul (inps[0], inps[1]) + inps[2];
      Tensor a = gelu (h) + silu (h) * tanh (h);
      return mean (log_softmax (a, -1));
    };

  BASALT_ASSERT (gradcheck (fn, {x, w, b}, 1e-6, 1e-5, 1e-4),
                 "finite-difference gradcheck failed on composite graph");
}

void
test_higher_order_derivatives (void)
{
  using namespace basalt;

  Tensor_Options opts = Tensor_Options ()
                          .with_dtype (Dtype::Float64)
                          .with_requires_grad (true);
  Tensor x = tensor ({1.5, -0.5, 2.0}, opts);

  /* f(x) = sum(x^3 + sin(x))
     f'(x) = 3*x^2 + cos(x)
     f''(x) = 6*x - sin(x).  */
  Tensor y = sum (ops::pow_scalar (x, 3.0) + sin (x));
  std::vector<Tensor> g1 = grad ({y}, {x}, {}, true, true);
  BASALT_ASSERT (g1[0].requires_grad (),
                 "create_graph=true must produce differentiable gradient");

  std::vector<Tensor> g2 = grad ({sum (g1[0])}, {x}, {}, false, false);
  for (size_t i = 0; i < x.numel (); ++i)
    {
      double xv = x.read_scalar_at (i);
      double expected_d1 = 3.0 * xv * xv + std::cos (xv);
      double expected_d2 = 6.0 * xv - std::sin (xv);
      BASALT_ASSERT (
        std::fabs (g1[0].read_scalar_at (i) - expected_d1) < 1e-9,
        "first derivative mismatch");
      BASALT_ASSERT (
        std::fabs (g2[0].read_scalar_at (i) - expected_d2) < 1e-9,
        "second derivative mismatch");
    }

  /* Verify numerical gradgradcheck on cubic + trig function.  */
  auto cubic_fn = [] (const std::vector<Tensor> &inps) -> Tensor
    {
      return sum (ops::pow_scalar (inps[0], 3.0) + cos (inps[0]));
    };
  BASALT_ASSERT (gradgradcheck (cubic_fn, {x}, 1e-5, 1e-4, 1e-3),
                 "gradgradcheck failed");
}

void
test_scopes_retain_grad_and_checkpoint (void)
{
  using namespace basalt;

  Tensor_Options opts = Tensor_Options ()
                          .with_dtype (Dtype::Float64)
                          .with_requires_grad (true);
  Tensor x = tensor ({2.0, 3.0}, opts);

  {
    No_Grad_Guard no_grad;
    Tensor y_ng = x * 5.0;
    BASALT_ASSERT (!y_ng.requires_grad (), "No_Grad_Guard must disable grad");
    {
      Enable_Grad_Guard en_grad;
      Tensor y_eg = x * 5.0;
      BASALT_ASSERT (y_eg.requires_grad (),
                     "Enable_Grad_Guard must re-enable grad");
    }
  }

  /* Verify retain_grad on non-leaf tensor.  */
  Tensor mid = x * x;
  mid.retain_grad ();
  Tensor loss = sum (mid * 3.0);
  loss.backward ();
  BASALT_ASSERT (mid.grad ().defined (), "non-leaf retain_grad must populate");
  BASALT_ASSERT (std::fabs (mid.grad ().read_scalar_at (0) - 3.0) < 1e-12,
                 "non-leaf grad value");

  /* Verify activation checkpointing produces identical gradients.  */
  x.zero_grad ();
  auto block = [] (const std::vector<Tensor> &inps) -> Tensor
    {
      return tanh (inps[0] * inps[0] + 1.0);
    };
  Tensor ckpt_out = sum (checkpoint (block, {x}));
  ckpt_out.backward ();
  Tensor g_ckpt = x.grad ().clone ();

  x.zero_grad ();
  Tensor direct_out = sum (block ({x}));
  direct_out.backward ();
  BASALT_ASSERT (allclose (g_ckpt, x.grad (), 1e-12, 1e-12),
                 "checkpoint backward must match direct backward");

  /* Verify in-place modification detection raises Autograd_Error.  */
  Tensor z = tensor ({1.0, 2.0}, opts);
  Tensor w = z * z;
  z.add_ (ones_like (z));
  bool caught_inplace = false;
  try
    {
      sum (w).backward ();
    }
  catch (const Autograd_Error &)
    {
      caught_inplace = true;
    }
  BASALT_ASSERT (
    caught_inplace,
    "in-place mutation of saved tensor must throw Autograd_Error");
}

} /* namespace */

int
main (void)
{
  test_reverse_mode_and_gradcheck ();
  test_higher_order_derivatives ();
  test_scopes_retain_grad_and_checkpoint ();
  return 0;
}
