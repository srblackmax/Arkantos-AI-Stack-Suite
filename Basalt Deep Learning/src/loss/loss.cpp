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
#include <basalt/loss.hpp>
#include <basalt/autograd.hpp>
#include <basalt/ops.hpp>
#include <algorithm>
#include <cmath>

namespace basalt::loss
{

namespace
{

Tensor
apply_reduction (const Tensor &unreduced, Reduction reduction)
{
  switch (reduction)
    {
    case Reduction::None:
      return unreduced;
    case Reduction::Mean:
      return ops::mean (unreduced);
    case Reduction::Sum:
      return ops::sum (unreduced);
    }
  return ops::mean (unreduced);
}

class NLL_Loss_Backward : public Autograd_Node
{
public:
  NLL_Loss_Backward (Shape_Vector log_probs_shape, Tensor_Options opts,
                     Tensor target, ptrdiff_t ignore_index,
                     size_t valid_count, Reduction reduction)
    : shape_ (std::move (log_probs_shape)),
      opts_ (opts.with_requires_grad (false)),
      target_ (std::move (target)),
      ignore_index_ (ignore_index),
      valid_count_ (valid_count),
      reduction_ (reduction)
  {
  }

  const char *
  name (void) const noexcept override
  {
    return "NLL_Loss_Backward";
  }

  std::vector<Tensor>
  apply (const std::vector<Tensor> &grads) override
  {
    No_Grad_Guard no_grad;
    const Tensor &gy = grads[0];
    size_t batch = shape_[0];
    size_t num_classes = shape_[1];
    Tensor_Options cpu_opts
      = opts_.with_device (Device::cpu ()).with_requires_grad (false);
    Tensor gx_cpu = zeros (shape_, cpu_opts);
    Tensor tgt_cpu = target_.cpu ();
    Tensor gy_cpu = gy.cpu ();

    double scale = 0.0;
    if (reduction_ == Reduction::Mean)
      {
        scale = (valid_count_ > 0)
                ? (gy_cpu.item_double () / static_cast<double> (valid_count_))
                : 0.0;
      }
    else if (reduction_ == Reduction::Sum)
      {
        scale = gy_cpu.item_double ();
      }

    for (size_t i = 0; i < batch; ++i)
      {
        ptrdiff_t cls = static_cast<ptrdiff_t> (tgt_cpu.read_scalar_at (i));
        if (cls == ignore_index_)
          {
            continue;
          }
        double g_i = (reduction_ == Reduction::None)
                     ? gy_cpu.read_scalar_at (i)
                     : scale;
        gx_cpu.write_scalar_at (i * num_classes + static_cast<size_t> (cls),
                                -g_i);
      }
    return {gx_cpu.to (opts_.device)};
  }

private:
  Shape_Vector shape_;
  Tensor_Options opts_;
  Tensor target_;
  ptrdiff_t ignore_index_;
  size_t valid_count_;
  Reduction reduction_;
};

} /* namespace */

Tensor
mse_loss (const Tensor &input, const Tensor &target, Reduction reduction)
{
  Tensor diff = ops::sub (input, target);
  return apply_reduction (ops::square (diff), reduction);
}

Tensor
l1_loss (const Tensor &input, const Tensor &target, Reduction reduction)
{
  return apply_reduction (ops::abs (ops::sub (input, target)), reduction);
}

Tensor
smooth_l1_loss (const Tensor &input, const Tensor &target,
                double beta, Reduction reduction)
{
  BASALT_ASSERT_ARG (beta > 0.0, "smooth_l1_loss() requires beta > 0");
  Tensor diff = ops::abs (ops::sub (input, target));
  Tensor cond = ops::lt (diff, full_like (diff, beta));
  Tensor quad = ops::mul_scalar (ops::square (diff), 0.5 / beta);
  Tensor lin = ops::sub_scalar (diff, 0.5 * beta);
  return apply_reduction (ops::where (cond, quad, lin), reduction);
}

Tensor
nll_loss (const Tensor &log_probs, const Tensor &target,
          Reduction reduction, ptrdiff_t ignore_index)
{
  BASALT_ASSERT_SHAPE (log_probs.ndim () == 2 && target.ndim () == 1
                       && log_probs.size (0) == target.size (0),
                       "nll_loss() expects (N, C) log_probs and (N) target");
  size_t batch = log_probs.size (0);
  size_t num_classes = log_probs.size (1);

  Tensor lp_cpu = log_probs.detach ().cpu ();
  Tensor tgt_cpu = target.detach ().cpu ();
  Tensor_Options cpu_opts
    = log_probs.options ().with_device (Device::cpu ())
                          .with_requires_grad (false);
  Tensor per_sample ({batch}, cpu_opts);
  size_t valid_count = 0;
  double total_loss = 0.0;

  for (size_t i = 0; i < batch; ++i)
    {
      ptrdiff_t cls = static_cast<ptrdiff_t> (tgt_cpu.read_scalar_at (i));
      if (cls == ignore_index)
        {
          per_sample.write_scalar_at (i, 0.0);
          continue;
        }
      BASALT_ASSERT_SHAPE (cls >= 0
                           && static_cast<size_t> (cls) < num_classes,
                           "nll_loss() target class index out of range");
      double lp = lp_cpu.read_scalar_at (
        i * num_classes + static_cast<size_t> (cls));
      per_sample.write_scalar_at (i, -lp);
      total_loss += -lp;
      ++valid_count;
    }

  Tensor out_cpu;
  if (reduction == Reduction::None)
    {
      out_cpu = per_sample;
    }
  else
    {
      out_cpu = Tensor ({1}, cpu_opts);
      double val = 0.0;
      if (reduction == Reduction::Mean)
        {
          val = (valid_count > 0)
                ? (total_loss / static_cast<double> (valid_count))
                : 0.0;
        }
      else
        {
          val = total_loss;
        }
      out_cpu.write_scalar_at (0, val);
    }
  Tensor out = out_cpu.to (log_probs.device ());

  if (Grad_Mode::is_enabled () && log_probs.requires_grad ())
    {
      auto node = std::make_shared<NLL_Loss_Backward> (
        log_probs.shape (), log_probs.options (), tgt_cpu,
        ignore_index, valid_count, reduction);
      node->set_next_edges (autograd::collect_next_edges ({log_probs}));
      out.set_requires_grad (true);
      out.set_grad_fn (std::move (node));
    }
  return out;
}

Tensor
cross_entropy_loss (const Tensor &logits, const Tensor &target,
                    Reduction reduction, double label_smoothing,
                    ptrdiff_t ignore_index)
{
  BASALT_ASSERT_ARG (label_smoothing >= 0.0 && label_smoothing < 1.0,
                     "cross_entropy_loss() label_smoothing must be in [0, 1)");
  Tensor log_probs = ops::log_softmax (logits, -1);

  /* Support both 1-D class index targets and 2-D soft probability targets.  */
  if (target.ndim () == logits.ndim ())
    {
      Tensor prod = ops::neg (ops::sum (ops::mul (target, log_probs), -1));
      return apply_reduction (prod, reduction);
    }

  Tensor hard_loss = nll_loss (log_probs, target, reduction, ignore_index);
  if (label_smoothing == 0.0)
    {
      return hard_loss;
    }
  Tensor smooth_per_sample = ops::neg (ops::mean (log_probs, -1, false));
  size_t batch = logits.size (0);
  Tensor tgt_cpu = target.detach ().cpu ();
  Tensor mask_cpu ({batch},
                   logits.options ().with_device (Device::cpu ())
                                    .with_requires_grad (false));
  size_t valid_count = 0;
  for (size_t i = 0; i < batch; ++i)
    {
      ptrdiff_t cls = static_cast<ptrdiff_t> (tgt_cpu.read_scalar_at (i));
      if (cls == ignore_index)
        {
          mask_cpu.write_scalar_at (i, 0.0);
        }
      else
        {
          mask_cpu.write_scalar_at (i, 1.0);
          ++valid_count;
        }
    }
  if (valid_count < batch)
    {
      smooth_per_sample
        = ops::mul (smooth_per_sample, mask_cpu.to (logits.device ()));
    }
  Tensor smooth_loss;
  if (reduction == Reduction::None)
    {
      smooth_loss = smooth_per_sample;
    }
  else if (reduction == Reduction::Sum)
    {
      smooth_loss = ops::sum (smooth_per_sample);
    }
  else
    {
      smooth_loss = (valid_count > 0)
                    ? ops::div_scalar (ops::sum (smooth_per_sample),
                                       static_cast<double> (valid_count))
                    : ops::mul_scalar (ops::sum (smooth_per_sample), 0.0);
    }
  return ops::add (
    ops::mul_scalar (hard_loss, 1.0 - label_smoothing),
    ops::mul_scalar (smooth_loss, label_smoothing));
}

Tensor
bce_with_logits_loss (const Tensor &logits, const Tensor &target,
                      Reduction reduction)
{
  /* Numerically stable binary cross-entropy from logits:
     max(x, 0) - x * z + log(1 + exp(-|x|)).  */
  Tensor max_val = ops::relu (logits);
  Tensor prod = ops::mul (logits, target);
  Tensor penalty = ops::log1p (ops::exp (ops::neg (ops::abs (logits))));
  Tensor elem = ops::add (ops::sub (max_val, prod), penalty);
  return apply_reduction (elem, reduction);
}

Tensor
kl_div_loss (const Tensor &log_probs, const Tensor &target_probs,
             Reduction reduction, bool log_target)
{
  Tensor p = log_target ? ops::exp (target_probs) : target_probs;
  Tensor log_p = log_target
                 ? target_probs
                 : ops::log (ops::clamp (target_probs, 1e-12, 1e30));
  Tensor elem = ops::mul (p, ops::sub (log_p, log_probs));
  return apply_reduction (elem, reduction);
}

Tensor
huber_loss (const Tensor &input, const Tensor &target,
            double delta, Reduction reduction)
{
  BASALT_ASSERT_ARG (delta > 0.0, "huber_loss() requires delta > 0");
  Tensor diff = ops::abs (ops::sub (input, target));
  Tensor cond = ops::lt (diff, full_like (diff, delta));
  Tensor quad = ops::mul_scalar (ops::square (diff), 0.5);
  Tensor lin = ops::mul_scalar (ops::sub_scalar (diff, 0.5 * delta), delta);
  return apply_reduction (ops::where (cond, quad, lin), reduction);
}

Tensor
cosine_embedding_loss (const Tensor &input1, const Tensor &input2,
                       const Tensor &target, double margin,
                       Reduction reduction)
{
  BASALT_ASSERT_SHAPE (input1.shape () == input2.shape ()
                       && input1.ndim () >= 1,
                       "cosine_embedding_loss() requires matching shapes");
  ptrdiff_t dim = (input1.ndim () == 1) ? 0 : 1;
  Tensor dot = ops::sum (ops::mul (input1, input2), dim, false);
  Tensor n1 = ops::sqrt (ops::add_scalar (
    ops::sum (ops::square (input1), dim, false), 1e-16));
  Tensor n2 = ops::sqrt (ops::add_scalar (
    ops::sum (ops::square (input2), dim, false), 1e-16));
  Tensor cos_sim = ops::div (dot, ops::mul (n1, n2));

  Tensor pos_loss = ops::sub (ones_like (cos_sim), cos_sim);
  Tensor neg_loss = ops::relu (ops::sub_scalar (cos_sim, margin));
  Tensor is_pos = ops::gt (target, zeros_like (target));
  Tensor elem = ops::where (is_pos, pos_loss, neg_loss);
  return apply_reduction (elem, reduction);
}

} /* namespace basalt::loss */
