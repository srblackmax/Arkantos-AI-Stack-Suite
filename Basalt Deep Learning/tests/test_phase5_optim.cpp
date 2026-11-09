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
#include <basalt/loss.hpp>
#include <basalt/metrics.hpp>
#include <basalt/nn.hpp>
#include <basalt/ops.hpp>
#include <basalt/optim.hpp>
#include <basalt/random.hpp>
#include <basalt/scheduler.hpp>
#include <basalt/tensor.hpp>
#include <cmath>
#include <memory>
#include <vector>

namespace
{

void
test_losses_and_bellows_metrics (void)
{
  using namespace basalt;

  Tensor_Options f64 = Tensor_Options ()
                         .with_dtype (Dtype::Float64)
                         .with_requires_grad (true);

  Tensor logits = tensor ({2.0, -1.0, 0.5, -0.5, 3.0, 0.1},
                          f64.with_requires_grad (false))
                    .reshape ({2, 3})
                    .set_requires_grad (true);
  Tensor targets = tensor ({0, 1},
                           Tensor_Options ().with_dtype (Dtype::Int64));

  nn::Cross_Entropy_Loss ce (Reduction::Mean, 0.1);
  Tensor l_ce = ce.forward (logits, targets);
  l_ce.backward ();
  BASALT_ASSERT (logits.grad ().defined (), "CrossEntropyLoss backward");

  /* Verify gradcheck on cross_entropy_loss and bce_with_logits_loss.  */
  auto ce_fn = [&] (const std::vector<Tensor> &inps) -> Tensor
    {
      return loss::cross_entropy_loss (
        inps[0], targets, Reduction::Mean, 0.05);
    };
  BASALT_ASSERT (gradcheck (ce_fn, {logits}, 1e-6, 1e-5, 1e-4),
                 "cross_entropy_loss gradcheck");

  Tensor bin_logits = tensor ({-1.2, 0.8, 2.5}, f64);
  Tensor bin_targets = tensor ({0.0, 1.0, 1.0},
                               Tensor_Options ().with_dtype (Dtype::Float64));
  auto bce_fn = [&] (const std::vector<Tensor> &inps) -> Tensor
    {
      return loss::bce_with_logits_loss (inps[0], bin_targets);
    };
  BASALT_ASSERT (gradcheck (bce_fn, {bin_logits}, 1e-6, 1e-5, 1e-4),
                 "bce_with_logits_loss gradcheck");

  /* Verify Bellows metric bridges.  */
  double acc = metrics::accuracy (logits, targets);
  BASALT_ASSERT (std::fabs (acc - 1.0) < 1e-9, "expected 100% accuracy");
  double top2 = metrics::top_k_accuracy (logits, targets, 2);
  BASALT_ASSERT (std::fabs (top2 - 1.0) < 1e-9, "expected 100% top-2");

  Tensor p_reg = tensor ({1.0, 2.0, 3.0});
  Tensor t_reg = tensor ({1.5, 2.0, 2.5});
  double mse = metrics::mean_squared_error (p_reg, t_reg);
  BASALT_ASSERT (std::fabs (mse - (0.5 / 3.0)) < 1e-6, "Bellows MSE bridge");
}

void
test_all_optimizers_converge (void)
{
  using namespace basalt;

  /* Minimize f(w) = sum((w - target)^2) using SGD+Nesterov, Adam, AdamW,
     RMSprop, Adagrad, and LAMB.  */
  Tensor target = tensor ({3.0, -2.0, 1.5, -0.5},
                          Tensor_Options ().with_dtype (Dtype::Float64));

  auto check_opt = [&] (auto make_opt, size_t steps)
    {
      Tensor w = zeros ({4}, Tensor_Options ()
                               .with_dtype (Dtype::Float64)
                               .with_requires_grad (true));
      auto opt = make_opt (std::vector<Tensor>{w});
      double initial_loss = loss::mse_loss (w, target).item_double ();
      for (size_t s = 0; s < steps; ++s)
        {
          opt->zero_grad ();
          Tensor l = loss::mse_loss (w, target);
          l.backward ();
          opt->step ();
        }
      double final_loss = loss::mse_loss (w, target).item_double ();
      BASALT_ASSERT (final_loss < initial_loss * 0.05,
                     "optimizer failed to reduce quadratic objective");
    };

  check_opt ([] (std::vector<Tensor> p)
    {
      return std::make_unique<optim::SGD> (p, 0.1, 0.8, 0.0, 0.0, true);
    }, 60);

  check_opt ([] (std::vector<Tensor> p)
    {
      return std::make_unique<optim::Adam> (p, 0.2);
    }, 60);

  check_opt ([] (std::vector<Tensor> p)
    {
      return std::make_unique<optim::AdamW> (p, 0.2, 0.9, 0.999, 1e-8, 1e-4);
    }, 60);

  check_opt ([] (std::vector<Tensor> p)
    {
      return std::make_unique<optim::RMSprop> (p, 0.1, 0.9, 1e-8, 0.0,
                                               0.5, true);
    }, 60);

  check_opt ([] (std::vector<Tensor> p)
    {
      return std::make_unique<optim::Adagrad> (p, 0.8);
    }, 80);

  check_opt ([] (std::vector<Tensor> p)
    {
      return std::make_unique<optim::LAMB> (p, 0.2, 0.9, 0.999, 1e-6, 0.0);
    }, 60);
}

void
test_schedulers_and_grad_clipping (void)
{
  using namespace basalt;

  Tensor w = ones ({4}, Tensor_Options ().with_requires_grad (true));
  optim::Adam opt ({w}, 0.1);

  opt.set_lr (0.1);
  optim::Step_LR step_sched (opt, 2, 0.5);
  step_sched.step ();
  BASALT_ASSERT (std::fabs (opt.lr () - 0.1) < 1e-9, "StepLR epoch 1");
  step_sched.step ();
  BASALT_ASSERT (std::fabs (opt.lr () - 0.05) < 1e-9, "StepLR epoch 2");

  opt.set_lr (0.1);
  optim::Exponential_LR exp_sched (opt, 0.5);
  exp_sched.step ();
  BASALT_ASSERT (std::fabs (opt.lr () - 0.05) < 1e-9, "ExponentialLR epoch 1");
  exp_sched.step ();
  BASALT_ASSERT (std::fabs (opt.lr () - 0.025) < 1e-9,
                 "ExponentialLR epoch 2");

  opt.set_lr (0.1);
  optim::Multi_Step_LR ms_sched (opt, {2, 4}, 0.5);
  ms_sched.step ();
  BASALT_ASSERT (std::fabs (opt.lr () - 0.1) < 1e-9, "MultiStepLR epoch 1");
  ms_sched.step ();
  BASALT_ASSERT (std::fabs (opt.lr () - 0.05) < 1e-9, "MultiStepLR epoch 2");
  ms_sched.step ();
  ms_sched.step ();
  BASALT_ASSERT (std::fabs (opt.lr () - 0.025) < 1e-9, "MultiStepLR epoch 4");

  /* Huber_Loss, Cosine_Embedding_Loss, and smoothed CE ignore_index.  */
  nn::Huber_Loss huber (1.0);
  Tensor h_in = tensor ({0.0, 2.0},
                        Tensor_Options ().with_requires_grad (true));
  Tensor h_tgt = tensor ({0.0, 0.0});
  Tensor h_loss = huber.forward (h_in, h_tgt);
  BASALT_ASSERT (std::fabs (h_loss.item_double () - 0.75) < 1e-5,
                 "Huber_Loss forward value");
  h_loss.backward ();
  BASALT_ASSERT (h_in.grad ().defined (), "Huber_Loss backward grad");

  nn::Cosine_Embedding_Loss cos_emb (0.0);
  Tensor ce1 = tensor ({1.0, 0.0, 0.0, 1.0}).reshape ({2, 2});
  Tensor ce2 = tensor ({1.0, 0.0, 1.0, 0.0}).reshape ({2, 2});
  Tensor ce_y = tensor ({1.0, -1.0});
  Tensor ce_loss = cos_emb.forward (ce1, ce2, ce_y);
  BASALT_ASSERT (ce_loss.item_double () < 1e-5,
                 "Cosine_Embedding_Loss orthogonal negative & aligned pos");

  Tensor logits = tensor ({2.0, -1.0, 0.5, 99.0, -99.0, 0.0}).reshape ({2, 3});
  Tensor targets = tensor ({0.0, -100.0});
  Tensor ce_ign = loss::cross_entropy_loss (
    logits, targets, Reduction::Mean, 0.1, -100);
  Tensor logits_single = tensor ({2.0, -1.0, 0.5}).reshape ({1, 3});
  Tensor targets_single = tensor ({0.0});
  Tensor ce_ref = loss::cross_entropy_loss (
    logits_single, targets_single, Reduction::Mean, 0.1, -100);
  BASALT_ASSERT (
    std::fabs (ce_ign.item_double () - ce_ref.item_double ()) < 1e-5,
    "cross_entropy_loss with label_smoothing masks ignore_index");

  opt.set_lr (0.1);
  optim::Cosine_Annealing_LR cos_sched (opt, 4, 0.0);
  cos_sched.step ();
  cos_sched.step ();
  BASALT_ASSERT (std::fabs (opt.lr () - 0.05) < 1e-9, "Cosine midpoint");

  optim::One_Cycle_LR oc_sched (opt, 0.1, 10, 0.2);
  oc_sched.step ();
  oc_sched.step ();
  BASALT_ASSERT (std::fabs (opt.lr () - 0.1) < 1e-6, "OneCycle peak");

  opt.set_lr (0.1);
  optim::Linear_Warmup_LR warm_sched (opt, 2, 0.5, 1.0);
  warm_sched.step ();
  BASALT_ASSERT (std::fabs (opt.lr () - 0.075) < 1e-9, "Warmup mid");

  opt.set_lr (0.1);
  optim::Reduce_LR_On_Plateau plat_sched (opt, true, 0.5, 1);
  plat_sched.step (1.0);
  plat_sched.step (1.0);
  plat_sched.step (1.0);
  BASALT_ASSERT (std::fabs (opt.lr () - 0.05) < 1e-9, "Plateau reduction");

  /* Gradient norm clipping.  */
  w.zero_grad ();
  Tensor loss = sum (w * 10.0);
  loss.backward ();
  double orig_norm = optim::clip_grad_norm_ ({w}, 2.0, 2.0);
  BASALT_ASSERT ( std::fabs (orig_norm - 20.0) < 1e-4, "pre-clipped norm");
  double new_norm = ops::norm (w.grad (), 2.0).item_double ();
  BASALT_ASSERT (new_norm <= 2.001, "post-clipped norm must be <= max_norm");
}

} /* namespace */

int
main (void)
{
  test_losses_and_bellows_metrics ();
  test_all_optimizers_converge ();
  test_schedulers_and_grad_clipping ();
  return 0;
}
