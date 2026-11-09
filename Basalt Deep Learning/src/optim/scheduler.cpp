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
#include <basalt/scheduler.hpp>
#include <basalt/error.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace basalt::optim
{

namespace
{
constexpr double k_pi = 3.14159265358979323846;
}

LR_Scheduler::LR_Scheduler (Optimizer &optimizer)
  : optimizer_ (optimizer),
    base_lr_ (optimizer.lr ()),
    last_epoch_ (0)
{
}

LR_Scheduler::~LR_Scheduler (void) noexcept = default;

Step_LR::Step_LR (Optimizer &optimizer, size_t step_size, double gamma)
  : LR_Scheduler (optimizer),
    step_size_ (step_size),
    gamma_ (gamma)
{
  BASALT_ASSERT_ARG (step_size >= 1, "Step_LR requires step_size >= 1");
}

void
Step_LR::step (void)
{
  ++last_epoch_;
  size_t num_decays = last_epoch_ / step_size_;
  double new_lr
    = base_lr_ * std::pow (gamma_, static_cast<double> (num_decays));
  optimizer_.set_lr (new_lr);
}

Exponential_LR::Exponential_LR (Optimizer &optimizer, double gamma)
  : LR_Scheduler (optimizer),
    gamma_ (gamma)
{
  BASALT_ASSERT_ARG (gamma > 0.0, "Exponential_LR requires gamma > 0");
}

void
Exponential_LR::step (void)
{
  ++last_epoch_;
  double new_lr
    = base_lr_ * std::pow (gamma_, static_cast<double> (last_epoch_));
  optimizer_.set_lr (new_lr);
}

Multi_Step_LR::Multi_Step_LR (Optimizer &optimizer,
                              std::vector<size_t> milestones,
                              double gamma)
  : LR_Scheduler (optimizer),
    milestones_ (std::move (milestones)),
    gamma_ (gamma)
{
  BASALT_ASSERT_ARG (gamma > 0.0, "Multi_Step_LR requires gamma > 0");
  std::sort (milestones_.begin (), milestones_.end ());
}

void
Multi_Step_LR::step (void)
{
  ++last_epoch_;
  auto it = std::upper_bound (milestones_.begin (), milestones_.end (),
                              last_epoch_);
  size_t num_decays
    = static_cast<size_t> (std::distance (milestones_.begin (), it));
  double new_lr
    = base_lr_ * std::pow (gamma_, static_cast<double> (num_decays));
  optimizer_.set_lr (new_lr);
}

Cosine_Annealing_LR::Cosine_Annealing_LR (Optimizer &optimizer, size_t t_max,
                                          double eta_min)
  : LR_Scheduler (optimizer),
    t_max_ (t_max),
    eta_min_ (eta_min)
{
  BASALT_ASSERT_ARG (t_max >= 1, "Cosine_Annealing_LR requires t_max >= 1");
}

void
Cosine_Annealing_LR::step (void)
{
  ++last_epoch_;
  size_t cur = std::min (last_epoch_, t_max_);
  double cosine = std::cos (
    k_pi * static_cast<double> (cur) / static_cast<double> (t_max_));
  double new_lr = eta_min_ + 0.5 * (base_lr_ - eta_min_) * (1.0 + cosine);
  optimizer_.set_lr (new_lr);
}

One_Cycle_LR::One_Cycle_LR (Optimizer &optimizer,
                            double max_lr,
                            size_t total_steps,
                            double pct_start,
                            double div_factor,
                            double final_div_factor)
  : LR_Scheduler (optimizer),
    max_lr_ (max_lr),
    total_steps_ (total_steps),
    pct_start_ (pct_start),
    initial_lr_ (max_lr / div_factor),
    min_lr_ (max_lr / (div_factor * final_div_factor))
{
  BASALT_ASSERT_ARG (total_steps >= 2 && pct_start > 0.0 && pct_start < 1.0,
                     "One_Cycle_LR invalid total_steps or pct_start");
  optimizer_.set_lr (initial_lr_);
}

void
One_Cycle_LR::step (void)
{
  ++last_epoch_;
  size_t cur = std::min (last_epoch_, total_steps_);
  double warmup_end = pct_start_ * static_cast<double> (total_steps_);
  double new_lr = min_lr_;

  if (static_cast<double> (cur) <= warmup_end)
    {
      double frac = static_cast<double> (cur) / std::max (1.0, warmup_end);
      new_lr = initial_lr_ + frac * (max_lr_ - initial_lr_);
    }
  else
    {
      double down_steps = static_cast<double> (total_steps_) - warmup_end;
      double frac = (static_cast<double> (cur) - warmup_end)
                    / std::max (1.0, down_steps);
      double cosine = std::cos (k_pi * frac);
      new_lr = min_lr_ + 0.5 * (max_lr_ - min_lr_) * (1.0 + cosine);
    }
  optimizer_.set_lr (new_lr);
}

Linear_Warmup_LR::Linear_Warmup_LR (Optimizer &optimizer,
                                    size_t warmup_steps,
                                    double start_factor,
                                    double end_factor)
  : LR_Scheduler (optimizer),
    warmup_steps_ (warmup_steps),
    start_factor_ (start_factor),
    end_factor_ (end_factor)
{
  BASALT_ASSERT_ARG (warmup_steps >= 1,
                     "Linear_Warmup_LR requires warmup_steps >= 1");
  optimizer_.set_lr (base_lr_ * start_factor_);
}

void
Linear_Warmup_LR::step (void)
{
  ++last_epoch_;
  double frac = std::min (
    1.0, static_cast<double> (last_epoch_)
         / static_cast<double> (warmup_steps_));
  double factor = start_factor_ + frac * (end_factor_ - start_factor_);
  optimizer_.set_lr (base_lr_ * factor);
}

Reduce_LR_On_Plateau::Reduce_LR_On_Plateau (Optimizer &optimizer,
                                            bool minimize,
                                            double factor,
                                            size_t patience,
                                            double threshold,
                                            double min_lr)
  : LR_Scheduler (optimizer),
    minimize_ (minimize),
    factor_ (factor),
    patience_ (patience),
    threshold_ (threshold),
    min_lr_ (min_lr),
    best_metric_ (minimize ? std::numeric_limits<double>::infinity ()
                           : -std::numeric_limits<double>::infinity ())
{
  BASALT_ASSERT_ARG (factor > 0.0 && factor < 1.0,
                     "Reduce_LR_On_Plateau factor must be in (0, 1)");
}

void
Reduce_LR_On_Plateau::step (void)
{
  step (best_metric_);
}

void
Reduce_LR_On_Plateau::step (double metric)
{
  ++last_epoch_;
  if (!initialized_)
    {
      best_metric_ = metric;
      initialized_ = true;
      num_bad_epochs_ = 0;
      return;
    }

  bool improved = minimize_
                  ? (metric < best_metric_ - threshold_)
                  : (metric > best_metric_ + threshold_);
  if (improved)
    {
      best_metric_ = metric;
      num_bad_epochs_ = 0;
    }
  else
    {
      ++num_bad_epochs_;
      if (num_bad_epochs_ > patience_)
        {
          double new_lr = std::max (min_lr_, optimizer_.lr () * factor_);
          optimizer_.set_lr (new_lr);
          num_bad_epochs_ = 0;
        }
    }
}

} /* namespace basalt::optim */
