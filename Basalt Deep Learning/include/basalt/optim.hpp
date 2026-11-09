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

#ifndef BASALT_OPTIM_HPP
#define BASALT_OPTIM_HPP

/* First-order and adaptive optimization algorithms (SGD with momentum and
   Nesterov acceleration, Adam, AdamW, RMSprop, Adagrad, and LAMB) plus
   gradient norm/value clipping utilities.  */

#include <cstddef>
#include <map>
#include <string>
#include <vector>
#include <basalt/config.hpp>
#include <basalt/tensor.hpp>

namespace basalt::optim
{

/* Abstract base class for parameter optimizers.  */
class BASALT_API Optimizer
{
public:
  Optimizer (std::vector<Tensor> params, double lr);
  virtual ~Optimizer (void) noexcept;

  virtual void
  step (void) = 0;

  void
  zero_grad (void);

  double
  lr (void) const noexcept
  {
    return lr_;
  }

  void
  set_lr (double new_lr) noexcept
  {
    lr_ = new_lr;
  }

  size_t
  step_count (void) const noexcept
  {
    return step_count_;
  }

  const std::vector<Tensor> &
  parameters (void) const noexcept
  {
    return params_;
  }

  virtual std::map<std::string, Tensor>
  state_dict (void) const;

  virtual void
  load_state_dict (const std::map<std::string, Tensor> &state);

protected:
  std::vector<Tensor> params_;
  double lr_;
  size_t step_count_ = 0;
  std::map<std::string, Tensor> state_buffers_;
};

/* Stochastic Gradient Descent with optional momentum, dampening, L2 weight
   decay, and Nesterov acceleration.  */
class BASALT_API SGD : public Optimizer
{
public:
  SGD (std::vector<Tensor> params,
       double lr,
       double momentum = 0.0,
       double dampening = 0.0,
       double weight_decay = 0.0,
       bool nesterov = false);

  void
  step (void) override;

private:
  double momentum_;
  double dampening_;
  double weight_decay_;
  bool nesterov_;
};

/* Adaptive Moment Estimation (Adam) with L2 weight decay.  */
class BASALT_API Adam : public Optimizer
{
public:
  Adam (std::vector<Tensor> params,
        double lr = 1e-3,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double eps = 1e-8,
        double weight_decay = 0.0);

  void
  step (void) override;

private:
  double beta1_;
  double beta2_;
  double eps_;
  double weight_decay_;
};

/* Adam with decoupled weight decay (AdamW).  */
class BASALT_API AdamW : public Optimizer
{
public:
  AdamW (std::vector<Tensor> params,
         double lr = 1e-3,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double eps = 1e-8,
         double weight_decay = 1e-2);

  void
  step (void) override;

private:
  double beta1_;
  double beta2_;
  double eps_;
  double weight_decay_;
};

/* Root Mean Square Propagation (RMSprop) with optional momentum and centered
   second moments.  */
class BASALT_API RMSprop : public Optimizer
{
public:
  RMSprop (std::vector<Tensor> params,
           double lr = 1e-2,
           double alpha = 0.99,
           double eps = 1e-8,
           double weight_decay = 0.0,
           double momentum = 0.0,
           bool centered = false);

  void
  step (void) override;

private:
  double alpha_;
  double eps_;
  double weight_decay_;
  double momentum_;
  bool centered_;
};

/* Adaptive Subgradient Method (Adagrad).  */
class BASALT_API Adagrad : public Optimizer
{
public:
  Adagrad (std::vector<Tensor> params,
           double lr = 1e-2,
           double lr_decay = 0.0,
           double weight_decay = 0.0,
           double eps = 1e-10);

  void
  step (void) override;

private:
  double lr_decay_;
  double weight_decay_;
  double eps_;
};

/* Layer-wise Adaptive Moments optimizer for Batch training (LAMB).  */
class BASALT_API LAMB : public Optimizer
{
public:
  LAMB (std::vector<Tensor> params,
        double lr = 1e-3,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double eps = 1e-6,
        double weight_decay = 0.01);

  void
  step (void) override;

private:
  double beta1_;
  double beta2_;
  double eps_;
  double weight_decay_;
};

/* Clip total gradient norm across PARAMETERS in-place and return the
   pre-clipped norm.  */
BASALT_API double
clip_grad_norm_ (const std::vector<Tensor> &parameters,
                 double max_norm,
                 double norm_type = 2.0);

/* Clip individual gradient values across PARAMETERS to [-clip_value,
   +clip_value] in-place.  */
BASALT_API void
clip_grad_value_ (const std::vector<Tensor> &parameters,
                  double clip_value);

} /* namespace basalt::optim */

#endif /* BASALT_OPTIM_HPP */
