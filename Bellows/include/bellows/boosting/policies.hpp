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

#ifndef BELLOWS_BOOSTING_POLICIES_HPP
#define BELLOWS_BOOSTING_POLICIES_HPP

#include <cmath>
#include <cstdint>

namespace bellows
{

// Split-candidate cut policy evaluating exact sorted midpoints.
struct exact_cut_policy
{
  static constexpr std::uint32_t policy_code = 1;
  static constexpr bool uses_quantile_bins = false;
};

// Split-candidate cut policy using quantile buckets and histogram subtraction.
struct quantile_bin_policy
{
  static constexpr std::uint32_t policy_code = 2;
  static constexpr bool uses_quantile_bins = true;
};

// Tree topology growth policy expanding all nodes level by level.
struct depth_synchronous_growth
{
  static constexpr std::uint32_t policy_code = 1;
};

// Tree topology growth policy expanding the highest-gain frontier leaf first.
struct best_gain_frontier_growth
{
  static constexpr std::uint32_t policy_code = 2;
};

// Tree topology growth policy enforcing identical splits across each depth.
struct oblivious_symmetric_growth
{
  static constexpr std::uint32_t policy_code = 3;
};

// Observation sampling policy using all rows uniformly per stage.
struct uniform_full_sample
{
  static constexpr std::uint32_t policy_code = 1;
  static constexpr bool filters_by_gradient = false;
};

// Observation sampling policy keeping high-gradient rows and weighted tail rows.
struct gradient_magnitude_filter
{
  static constexpr std::uint32_t policy_code = 2;
  static constexpr bool filters_by_gradient = true;
};

// Regularised Newton leaf weight and split gain formulas.
struct newton_regulariser
{
  static double
  soft_threshold (double grad_sum, double l1_alpha) noexcept
  {
    if (l1_alpha <= 0.0)
      {
        return grad_sum;
      }
    if (grad_sum > l1_alpha)
      {
        return grad_sum - l1_alpha;
      }
    if (grad_sum < -l1_alpha)
      {
        return grad_sum + l1_alpha;
      }
    return 0.0;
  }

  static double
  leaf_weight (double grad_sum,
                double hess_sum,
                double l1_alpha,
                double l2_lambda) noexcept
  {
    double shrunk = soft_threshold (grad_sum, l1_alpha);
    return -shrunk / (hess_sum + l2_lambda);
  }

  static double
  node_objective (double grad_sum,
                  double hess_sum,
                  double l1_alpha,
                  double l2_lambda) noexcept
  {
    double shrunk = soft_threshold (grad_sum, l1_alpha);
    return (shrunk * shrunk) / (hess_sum + l2_lambda);
  }

  static double
  split_gain (double grad_left,
              double hess_left,
              double grad_right,
              double hess_right,
              double l1_alpha,
              double l2_lambda,
              double complexity_gamma) noexcept
  {
    double grad_parent = grad_left + grad_right;
    double hess_parent = hess_left + hess_right;
    double score_left =
      node_objective (grad_left, hess_left, l1_alpha, l2_lambda);
    double score_right =
      node_objective (grad_right, hess_right, l1_alpha, l2_lambda);
    double score_parent =
      node_objective (grad_parent, hess_parent, l1_alpha, l2_lambda);
    return 0.5 * (score_left + score_right - score_parent) - complexity_gamma;
  }
};

} // namespace bellows

#endif // BELLOWS_BOOSTING_POLICIES_HPP
