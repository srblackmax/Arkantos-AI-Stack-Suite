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

#ifndef BELLOWS_BOOSTING_LOSSES_HPP
#define BELLOWS_BOOSTING_LOSSES_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <bellows/core/concepts.hpp>

namespace bellows
{

// Second-order squared residual objective for continuous regression.
struct squared_residual_loss
{
  static constexpr std::uint32_t loss_code = 1;
  static constexpr bool is_classification = false;
  static constexpr bool is_multiclass = false;

  double
  base_score (const std::vector<double> &targets) const noexcept
  {
    if (targets.empty ())
      {
        return 0.0;
      }
    double sum = 0.0;
    for (double y : targets)
      {
        sum += y;
      }
    return sum / static_cast<double> (targets.size ());
  }

  derivative_pair
  compute_derivatives (double y, double raw_score) const noexcept
  {
    return derivative_pair{raw_score - y, 1.0};
  }

  double
  map_prediction (double raw_score) const noexcept
  {
    return raw_score;
  }

  double
  map_probability (double raw_score) const noexcept
  {
    return raw_score;
  }
};

// Twice-differentiable pseudo-Huber objective for outlier-robust regression.
struct huber_pseudo_loss
{
  static constexpr std::uint32_t loss_code = 2;
  static constexpr bool is_classification = false;
  static constexpr bool is_multiclass = false;

  double delta = 1.0;

  double
  base_score (const std::vector<double> &targets) const noexcept
  {
    if (targets.empty ())
      {
        return 0.0;
      }
    std::vector<double> sorted = targets;
    std::size_t mid = sorted.size () / 2;
    std::nth_element (sorted.begin (),
                      sorted.begin () + static_cast<std::ptrdiff_t> (mid),
                      sorted.end ());
    return sorted[mid];
  }

  derivative_pair
  compute_derivatives (double y, double raw_score) const noexcept
  {
    double r = raw_score - y;
    double scaled = r / delta;
    double denom = std::sqrt (1.0 + scaled * scaled);
    double grad = r / denom;
    double hess = std::max (1.0 / (denom * denom * denom), 1.0e-6);
    return derivative_pair{grad, hess};
  }

  double
  map_prediction (double raw_score) const noexcept
  {
    return raw_score;
  }

  double
  map_probability (double raw_score) const noexcept
  {
    return raw_score;
  }
};

// Second-order Bernoulli logistic deviance objective for binary classification.
struct logistic_bernoulli_loss
{
  static constexpr std::uint32_t loss_code = 3;
  static constexpr bool is_classification = true;
  static constexpr bool is_multiclass = false;

  static double
  sigmoid (double z) noexcept
  {
    double clamped = std::clamp (z, -20.0, 20.0);
    return 1.0 / (1.0 + std::exp (-clamped));
  }

  double
  base_score (const std::vector<double> &targets) const noexcept
  {
    if (targets.empty ())
      {
        return 0.0;
      }
    double pos = 0.0;
    for (double y : targets)
      {
        if (y > 0.5)
          {
            pos += 1.0;
          }
      }
    double p = std::clamp (
      pos / static_cast<double> (targets.size ()), 1.0e-4, 1.0 - 1.0e-4);
    return std::log (p / (1.0 - p));
  }

  derivative_pair
  compute_derivatives (double y, double raw_score) const noexcept
  {
    double prob = sigmoid (raw_score);
    double label = y > 0.5 ? 1.0 : 0.0;
    double grad = prob - label;
    double hess = std::max (prob * (1.0 - prob), 1.0e-6);
    return derivative_pair{grad, hess};
  }

  double
  map_prediction (double raw_score) const noexcept
  {
    return sigmoid (raw_score) >= 0.5 ? 1.0 : 0.0;
  }

  double
  map_probability (double raw_score) const noexcept
  {
    return sigmoid (raw_score);
  }
};

// Second-order multinomial cross-entropy objective for K-class classification.
struct multinomial_deviance_loss
{
  static constexpr std::uint32_t loss_code = 4;
  static constexpr bool is_classification = true;
  static constexpr bool is_multiclass = true;

  double
  base_score (const std::vector<double> &) const noexcept
  {
    return 0.0;
  }

  derivative_pair
  compute_derivatives (double target_indicator, double class_prob) const noexcept
  {
    double grad = class_prob - target_indicator;
    double hess = std::max (2.0 * class_prob * (1.0 - class_prob), 1.0e-6);
    return derivative_pair{grad, hess};
  }

  double
  map_prediction (double raw_score) const noexcept
  {
    return raw_score;
  }

  double
  map_probability (double raw_score) const noexcept
  {
    return raw_score;
  }
};

} // namespace bellows

#endif // BELLOWS_BOOSTING_LOSSES_HPP
