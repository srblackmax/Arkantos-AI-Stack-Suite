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

#ifndef BASALT_SCHEDULER_HPP
#define BASALT_SCHEDULER_HPP

/* Learning-rate scheduling family: Step_LR, Exponential_LR, Multi_Step_LR,
   Cosine_Annealing_LR, One_Cycle_LR, Linear_Warmup_LR, and
   Reduce_LR_On_Plateau.  */

#include <cstddef>
#include <vector>
#include <basalt/config.hpp>
#include <basalt/optim.hpp>

namespace basalt::optim
{

/* Abstract base class for learning-rate schedulers.  */
class BASALT_API LR_Scheduler
{
public:
  explicit LR_Scheduler (Optimizer &optimizer);
  virtual ~LR_Scheduler (void) noexcept;

  virtual void
  step (void) = 0;

  double
  get_lr (void) const noexcept
  {
    return optimizer_.lr ();
  }

  size_t
  last_epoch (void) const noexcept
  {
    return last_epoch_;
  }

protected:
  Optimizer &optimizer_;
  double base_lr_;
  size_t last_epoch_ = 0;
};

using LRScheduler = LR_Scheduler;

/* Multiply learning rate by GAMMA every STEP_SIZE epochs.  */
class BASALT_API Step_LR : public LR_Scheduler
{
public:
  Step_LR (Optimizer &optimizer, size_t step_size, double gamma = 0.1);

  void
  step (void) override;

private:
  size_t step_size_;
  double gamma_;
};

using StepLR = Step_LR;

/* Multiply learning rate by GAMMA every epoch.  */
class BASALT_API Exponential_LR : public LR_Scheduler
{
public:
  Exponential_LR (Optimizer &optimizer, double gamma);

  void
  step (void) override;

private:
  double gamma_;
};

using ExponentialLR = Exponential_LR;

/* Multiply learning rate by GAMMA once the epoch reaches each milestone in
   MILESTONES.  */
class BASALT_API Multi_Step_LR : public LR_Scheduler
{
public:
  Multi_Step_LR (Optimizer &optimizer,
                 std::vector<size_t> milestones,
                 double gamma = 0.1);

  void
  step (void) override;

private:
  std::vector<size_t> milestones_;
  double gamma_;
};

using MultiStepLR = Multi_Step_LR;

/* Cosine annealing schedule from base_lr down to ETA_MIN over T_MAX steps.  */
class BASALT_API Cosine_Annealing_LR : public LR_Scheduler
{
public:
  Cosine_Annealing_LR (Optimizer &optimizer, size_t t_max,
                       double eta_min = 0.0);

  void
  step (void) override;

private:
  size_t t_max_;
  double eta_min_;
};

using CosineAnnealingLR = Cosine_Annealing_LR;

/* Super-convergence 1-cycle learning rate schedule with linear warmup to
   MAX_LR followed by cosine decay to MAX_LR / (DIV_FACTOR * FINAL_DIV).  */
class BASALT_API One_Cycle_LR : public LR_Scheduler
{
public:
  One_Cycle_LR (Optimizer &optimizer,
                double max_lr,
                size_t total_steps,
                double pct_start = 0.3,
                double div_factor = 25.0,
                double final_div_factor = 1e4);

  void
  step (void) override;

private:
  double max_lr_;
  size_t total_steps_;
  double pct_start_;
  double initial_lr_;
  double min_lr_;
};

using OneCycleLR = One_Cycle_LR;

/* Linear warmup schedule interpolating from START_FACTOR * base_lr to
   END_FACTOR * base_lr over WARMUP_STEPS.  */
class BASALT_API Linear_Warmup_LR : public LR_Scheduler
{
public:
  Linear_Warmup_LR (Optimizer &optimizer,
                    size_t warmup_steps,
                    double start_factor = 0.1,
                    double end_factor = 1.0);

  void
  step (void) override;

private:
  size_t warmup_steps_;
  double start_factor_;
  double end_factor_;
};

using LinearWarmupLR = Linear_Warmup_LR;

/* Reduce learning rate by FACTOR when a monitored metric stops improving for
   PATIENCE calls to step(metric).  */
class BASALT_API Reduce_LR_On_Plateau : public LR_Scheduler
{
public:
  Reduce_LR_On_Plateau (Optimizer &optimizer,
                        bool minimize = true,
                        double factor = 0.1,
                        size_t patience = 10,
                        double threshold = 1e-4,
                        double min_lr = 0.0);

  void
  step (void) override;

  void
  step (double metric);

private:
  bool minimize_;
  double factor_;
  size_t patience_;
  double threshold_;
  double min_lr_;
  double best_metric_;
  size_t num_bad_epochs_ = 0;
  bool initialized_ = false;
};

using ReduceLROnPlateau = Reduce_LR_On_Plateau;

} /* namespace basalt::optim */

#endif /* BASALT_SCHEDULER_HPP */
