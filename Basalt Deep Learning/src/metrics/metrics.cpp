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
#include <basalt/metrics.hpp>
#include <basalt/ops.hpp>
#include <bellows/metrics/metrics.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace basalt::metrics
{

double
mean_squared_error (const Tensor &predictions, const Tensor &targets)
{
  return bellows::mean_squared_deviation{} (
    targets.to_vector_double (), predictions.to_vector_double ());
}

double
root_mean_squared_error (const Tensor &predictions, const Tensor &targets)
{
  return bellows::root_mean_squared_deviation{} (
    targets.to_vector_double (), predictions.to_vector_double ());
}

double
mean_absolute_error (const Tensor &predictions, const Tensor &targets)
{
  return bellows::mean_absolute_deviation{} (
    targets.to_vector_double (), predictions.to_vector_double ());
}

double
explained_variance (const Tensor &predictions, const Tensor &targets)
{
  return bellows::explained_variance_ratio{} (
    targets.to_vector_double (), predictions.to_vector_double ());
}

double
accuracy (const Tensor &predictions, const Tensor &targets)
{
  Tensor pred_labels = (predictions.ndim () >= 2)
                       ? ops::argmax (predictions, -1, false)
                       : predictions;
  return bellows::classification_accuracy{} (
    targets.to_vector_double (), pred_labels.to_vector_double ());
}

double
top_k_accuracy (const Tensor &logits, const Tensor &targets, size_t k)
{
  BASALT_ASSERT_SHAPE (logits.ndim () == 2 && targets.ndim () == 1
                       && logits.size (0) == targets.size (0),
                       "top_k_accuracy() expects (N, C) logits and (N) "
                       "targets");
  size_t n = logits.size (0);
  size_t c = logits.size (1);
  size_t eff_k = std::min (k, c);
  size_t hits = 0;

  std::vector<size_t> order (c);
  for (size_t i = 0; i < n; ++i)
    {
      std::iota (order.begin (), order.end (), size_t (0));
      std::partial_sort (
        order.begin (),
        order.begin () + static_cast<ptrdiff_t> (eff_k),
        order.end (),
        [&] (size_t a, size_t b)
          {
            return logits.read_scalar_at (i * c + a)
                   > logits.read_scalar_at (i * c + b);
          });
      size_t truth = static_cast<size_t> (targets.read_scalar_at (i));
      for (size_t j = 0; j < eff_k; ++j)
        {
          if (order[j] == truth)
            {
              ++hits;
              break;
            }
        }
    }
  return (n > 0) ? (static_cast<double> (hits) / static_cast<double> (n))
                 : 0.0;
}

double
balanced_f1_score (const Tensor &predictions, const Tensor &targets)
{
  Tensor pred_labels = (predictions.ndim () >= 2)
                       ? ops::argmax (predictions, -1, false)
                       : predictions;
  return bellows::balanced_f1_score{} (
    targets.to_vector_double (), pred_labels.to_vector_double ());
}

double
roc_auc_score (const Tensor &probabilities, const Tensor &targets)
{
  return bellows::roc_area_under_curve{} (
    targets.to_vector_double (), probabilities.to_vector_double ());
}

double
perplexity (double cross_entropy_loss_value) noexcept
{
  return std::exp (cross_entropy_loss_value);
}

void
Metric_Tracker::update (double value, size_t count) noexcept
{
  weighted_sum_ += value * static_cast<double> (count);
  total_count_ += count;
}

void
Metric_Tracker::reset (void) noexcept
{
  weighted_sum_ = 0.0;
  total_count_ = 0;
}

double
Metric_Tracker::average (void) const noexcept
{
  if (total_count_ == 0)
    {
      return 0.0;
    }
  return weighted_sum_ / static_cast<double> (total_count_);
}

} /* namespace basalt::metrics */
