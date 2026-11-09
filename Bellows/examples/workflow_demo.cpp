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

#include <iostream>
#include <vector>

#include <bellows/bellows.hpp>

int
main ()
{
  bellows::data_table features (40, 3);
  features.set_scale (2, bellows::feature_scale::nominal, 3);
  std::vector<double> targets (40, 0.0);

  for (std::size_t i = 0; i < 40; ++i)
    {
      double x0 = static_cast<double> (i) * 0.1;
      double x1 = (i % 5 == 0) ? bellows::missing_value ()
                                : static_cast<double> (i % 7) * 0.5;
      double cat = static_cast<double> (i % 3);
      features.at (i, 0) = x0;
      features.at (i, 1) = x1;
      features.at (i, 2) = cat;
      targets[i] = 2.5 * x0 + (cat == 2.0 ? 3.0 : -1.0);
    }

  auto pipeline =
    bellows::column_mender (bellows::mending_strategy::median)
    | bellows::moment_normaliser ()
    | bellows::make_stagewise_booster (
        bellows::with_booster<bellows::quantile_bin_policy,
                              bellows::best_gain_frontier_growth,
                              bellows::gradient_magnitude_filter>{},
        bellows::with_stage_count{35},
        bellows::with_step_rate{0.15},
        bellows::with_regularisation{0.01, 1.0, 0.0});

  bellows::fit (pipeline, features, targets);
  auto predictions = bellows::predict (pipeline, features);

  bellows::explained_variance_ratio r2_metric;
  double r2 = r2_metric (targets, predictions);
  std::cout << "Bellows workflow demo R^2: " << r2 << "\n";
  return r2 > 0.85 ? 0 : 1;
}
