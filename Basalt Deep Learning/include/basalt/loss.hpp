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

#ifndef BASALT_LOSS_HPP
#define BASALT_LOSS_HPP

/* Numerically stable loss functions and stateful Module wrappers for
   regression, classification, and divergence minimization.  */

#include <cstddef>
#include <cstdint>
#include <basalt/config.hpp>
#include <basalt/nn.hpp>
#include <basalt/tensor.hpp>

namespace basalt
{

/* Reduction mode applied to elementwise or per-sample loss values.  */
enum class Reduction : uint8_t
{
  None = 0,
  Mean = 1,
  Sum = 2
};

namespace loss
{

BASALT_API Tensor
mse_loss (const Tensor &input, const Tensor &target,
          Reduction reduction = Reduction::Mean);

BASALT_API Tensor
l1_loss (const Tensor &input, const Tensor &target,
         Reduction reduction = Reduction::Mean);

BASALT_API Tensor
smooth_l1_loss (const Tensor &input, const Tensor &target,
                double beta = 1.0,
                Reduction reduction = Reduction::Mean);

BASALT_API Tensor
nll_loss (const Tensor &log_probs, const Tensor &target,
          Reduction reduction = Reduction::Mean,
          ptrdiff_t ignore_index = -100);

BASALT_API Tensor
cross_entropy_loss (const Tensor &logits, const Tensor &target,
                    Reduction reduction = Reduction::Mean,
                    double label_smoothing = 0.0,
                    ptrdiff_t ignore_index = -100);

BASALT_API Tensor
bce_with_logits_loss (const Tensor &logits, const Tensor &target,
                      Reduction reduction = Reduction::Mean);

BASALT_API Tensor
kl_div_loss (const Tensor &log_probs, const Tensor &target_probs,
             Reduction reduction = Reduction::Mean,
             bool log_target = false);

BASALT_API Tensor
huber_loss (const Tensor &input, const Tensor &target,
            double delta = 1.0,
            Reduction reduction = Reduction::Mean);

BASALT_API Tensor
cosine_embedding_loss (const Tensor &input1, const Tensor &input2,
                       const Tensor &target, double margin = 0.0,
                       Reduction reduction = Reduction::Mean);

} /* namespace loss */

namespace nn
{

class BASALT_API MSE_Loss : public Module
{
public:
  using Module::forward;

  explicit MSE_Loss (Reduction reduction = Reduction::Mean) noexcept
    : reduction_ (reduction)
  {
  }

  Tensor
  forward (const Tensor &input, const Tensor &target) const
  {
    return loss::mse_loss (input, target, reduction_);
  }

private:
  Reduction reduction_;
};

using MSELoss = MSE_Loss;

class BASALT_API L1_Loss : public Module
{
public:
  using Module::forward;

  explicit L1_Loss (Reduction reduction = Reduction::Mean) noexcept
    : reduction_ (reduction)
  {
  }

  Tensor
  forward (const Tensor &input, const Tensor &target) const
  {
    return loss::l1_loss (input, target, reduction_);
  }

private:
  Reduction reduction_;
};

using L1Loss = L1_Loss;

class BASALT_API Smooth_L1_Loss : public Module
{
public:
  using Module::forward;

  explicit Smooth_L1_Loss (double beta = 1.0,
                           Reduction reduction = Reduction::Mean) noexcept
    : beta_ (beta),
      reduction_ (reduction)
  {
  }

  Tensor
  forward (const Tensor &input, const Tensor &target) const
  {
    return loss::smooth_l1_loss (input, target, beta_, reduction_);
  }

private:
  double beta_;
  Reduction reduction_;
};

using SmoothL1Loss = Smooth_L1_Loss;

class BASALT_API NLL_Loss : public Module
{
public:
  using Module::forward;

  explicit NLL_Loss (Reduction reduction = Reduction::Mean,
                     ptrdiff_t ignore_index = -100) noexcept
    : reduction_ (reduction),
      ignore_index_ (ignore_index)
  {
  }

  Tensor
  forward (const Tensor &log_probs, const Tensor &target) const
  {
    return loss::nll_loss (log_probs, target, reduction_, ignore_index_);
  }

private:
  Reduction reduction_;
  ptrdiff_t ignore_index_;
};

using NLLLoss = NLL_Loss;

class BASALT_API Cross_Entropy_Loss : public Module
{
public:
  using Module::forward;

  explicit Cross_Entropy_Loss (Reduction reduction = Reduction::Mean,
                               double label_smoothing = 0.0,
                               ptrdiff_t ignore_index = -100) noexcept
    : reduction_ (reduction),
      label_smoothing_ (label_smoothing),
      ignore_index_ (ignore_index)
  {
  }

  Tensor
  forward (const Tensor &logits, const Tensor &target) const
  {
    return loss::cross_entropy_loss (
      logits, target, reduction_, label_smoothing_, ignore_index_);
  }

private:
  Reduction reduction_;
  double label_smoothing_;
  ptrdiff_t ignore_index_;
};

using CrossEntropyLoss = Cross_Entropy_Loss;

class BASALT_API BCE_With_Logits_Loss : public Module
{
public:
  using Module::forward;

  explicit BCE_With_Logits_Loss (
    Reduction reduction = Reduction::Mean) noexcept
    : reduction_ (reduction)
  {
  }

  Tensor
  forward (const Tensor &logits, const Tensor &target) const
  {
    return loss::bce_with_logits_loss (logits, target, reduction_);
  }

private:
  Reduction reduction_;
};

using BCEWithLogitsLoss = BCE_With_Logits_Loss;

class BASALT_API KL_Div_Loss : public Module
{
public:
  using Module::forward;

  explicit KL_Div_Loss (Reduction reduction = Reduction::Mean,
                        bool log_target = false) noexcept
    : reduction_ (reduction),
      log_target_ (log_target)
  {
  }

  Tensor
  forward (const Tensor &log_probs, const Tensor &target_probs) const
  {
    return loss::kl_div_loss (log_probs, target_probs,
                              reduction_, log_target_);
  }

private:
  Reduction reduction_;
  bool log_target_;
};

using KLDivLoss = KL_Div_Loss;

class BASALT_API Huber_Loss : public Module
{
public:
  using Module::forward;

  explicit Huber_Loss (double delta = 1.0,
                       Reduction reduction = Reduction::Mean) noexcept
    : delta_ (delta),
      reduction_ (reduction)
  {
  }

  Tensor
  forward (const Tensor &input, const Tensor &target) const
  {
    return loss::huber_loss (input, target, delta_, reduction_);
  }

private:
  double delta_;
  Reduction reduction_;
};

using HuberLoss = Huber_Loss;

class BASALT_API Cosine_Embedding_Loss : public Module
{
public:
  using Module::forward;

  explicit Cosine_Embedding_Loss (
    double margin = 0.0, Reduction reduction = Reduction::Mean) noexcept
    : margin_ (margin),
      reduction_ (reduction)
  {
  }

  Tensor
  forward (const Tensor &input1, const Tensor &input2,
           const Tensor &target) const
  {
    return loss::cosine_embedding_loss (input1, input2, target,
                                        margin_, reduction_);
  }

private:
  double margin_;
  Reduction reduction_;
};

using CosineEmbeddingLoss = Cosine_Embedding_Loss;

} /* namespace nn */
} /* namespace basalt */

#endif /* BASALT_LOSS_HPP */
