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

#ifndef BELLOWS_CORE_PARAMETERS_HPP
#define BELLOWS_CORE_PARAMETERS_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bellows
{

// Descriptor of a single tunable hyperparameter knob.
struct parameter_knob
{
  std::string name;
  double current_value = 0.0;
  double min_value = 0.0;
  double max_value = 1.0;
};

// Compile-time / declarative modifier for L1, L2, and complexity penalties.
struct with_regularisation
{
  double l1 = 0.0;
  double l2 = 1.0;
  double complexity = 0.0;
};

// Modifier specifying the shrinkage step rate for stagewise learners.
struct with_step_rate
{
  double value = 0.1;
};

// Modifier specifying the number of boosting or ensemble stages.
struct with_stage_count
{
  std::size_t count = 50;
};

// Modifier specifying structural bounds on decision tree learners.
struct with_tree_limits
{
  std::size_t max_depth = 6;
  std::size_t max_leaves = 31;
  std::size_t min_samples_leaf = 2;
  std::size_t max_bins = 64;
};

// Modifier configuring ordered target statistics for nominal columns.
struct with_ordered_categories
{
  double prior_weight = 10.0;
  bool enabled = true;
};

// Modifier configuring gradient-magnitude observation filtering.
struct with_gradient_sampling
{
  double top_ratio = 0.25;
  double tail_ratio = 0.25;
};

// Modifier specifying a deterministic pseudo-random seed.
struct with_seed
{
  std::uint64_t seed = 42;
};

// Modifier specifying per-feature monotonic direction constraints (-1, 0, +1).
struct with_monotonic_constraints
{
  std::vector<int> directions;
};

// Compile-time policy selector tag for stagewise boosting learners.
template <typename Cut_Policy,
          typename Growth_Policy,
          typename Sampling_Policy = void>
struct with_booster
{
  using cut_policy_type = Cut_Policy;
  using growth_policy_type = Growth_Policy;
  using sampling_policy_type = Sampling_Policy;
};

} // namespace bellows

#endif // BELLOWS_CORE_PARAMETERS_HPP
