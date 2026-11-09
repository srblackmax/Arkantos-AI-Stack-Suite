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

#ifndef BASALT_METRICS_HPP
#define BASALT_METRICS_HPP

/* Evaluation metrics bridging Bellows (<bellows/metrics/metrics.hpp>) with
   Basalt tensors, plus streaming epoch metric accumulators.  */

#include <cstddef>
#include <basalt/config.hpp>
#include <basalt/tensor.hpp>

namespace basalt::metrics
{

/* Regression metrics delegated to Bellows.  */
BASALT_API double
mean_squared_error (const Tensor &predictions, const Tensor &targets);

BASALT_API double
root_mean_squared_error (const Tensor &predictions, const Tensor &targets);

BASALT_API double
mean_absolute_error (const Tensor &predictions, const Tensor &targets);

BASALT_API double
explained_variance (const Tensor &predictions, const Tensor &targets);

/* Classification metrics delegated to Bellows (accepts either 1-D class
   labels/probabilities or 2-D (N, C) logits).  */
BASALT_API double
accuracy (const Tensor &predictions, const Tensor &targets);

BASALT_API double
top_k_accuracy (const Tensor &logits, const Tensor &targets, size_t k = 5);

BASALT_API double
balanced_f1_score (const Tensor &predictions, const Tensor &targets);

BASALT_API double
roc_auc_score (const Tensor &probabilities, const Tensor &targets);

BASALT_API double
perplexity (double cross_entropy_loss_value) noexcept;

/* Streaming accumulator for tracking running averages across training and
   validation batches.  */
class BASALT_API Metric_Tracker
{
public:
  Metric_Tracker (void) noexcept = default;

  void
  update (double value, size_t count = 1) noexcept;

  void
  reset (void) noexcept;

  double
  average (void) const noexcept;

  size_t
  count (void) const noexcept
  {
    return total_count_;
  }

private:
  double weighted_sum_ = 0.0;
  size_t total_count_ = 0;
};

} /* namespace basalt::metrics */

#endif /* BASALT_METRICS_HPP */
