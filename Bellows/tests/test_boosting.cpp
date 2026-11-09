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

#include <cmath>
#include <iostream>
#include <vector>

#include <bellows/bellows.hpp>

namespace bellows_test
{

int
run_boosting_tests ()
{
  using namespace bellows;

  // Synthetic dataset with continuous, missing (NaN), and nominal features.
  const std::size_t n = 60;
  data_table x_reg (n, 3);
  x_reg.set_scale (2, feature_scale::nominal, 4);
  std::vector<double> y_reg (n, 0.0);

  for (std::size_t i = 0; i < n; ++i)
    {
      double f0 = static_cast<double> (i) * 0.1;
      double f1 = (i % 6 == 0) ? missing_value ()
                                : (i < 30 ? -2.0 : 2.0);
      double cat = static_cast<double> (i % 4);
      x_reg.at (i, 0) = f0;
      x_reg.at (i, 1) = f1;
      x_reg.at (i, 2) = cat;
      double cat_effect = (cat == 3.0 ? 4.0 : (cat == 2.0 ? 1.5 : -1.5));
      double f1_effect = is_missing (f1) ? 2.5 : f1;
      y_reg[i] = 1.8 * f0 + f1_effect + cat_effect;
    }

  explained_variance_ratio r2;

  // 1. Exact cuts + depth-synchronous growth (classic exact regularised GBDT).
  auto exact_booster = make_stagewise_booster<squared_residual_loss> (
    with_booster<exact_cut_policy,
                  depth_synchronous_growth,
                  uniform_full_sample>{},
    with_stage_count{45},
    with_step_rate{0.15},
    with_tree_limits{4, 16, 2, 32},
    with_regularisation{0.0, 0.5, 0.0});
  fit (exact_booster, x_reg, y_reg);
  if (r2 (y_reg, predict (exact_booster, x_reg)) < 0.90)
    {
      std::cerr << "test_boosting: exact + depth_synchronous booster failed\n";
      return 1;
    }

  // 2. Quantile-bin histograms (with sibling subtraction) + leaf-wise growth
  //    + gradient-magnitude filtering (GOSS).
  auto leafwise_booster = make_stagewise_booster<huber_pseudo_loss> (
    with_booster<quantile_bin_policy,
                  best_gain_frontier_growth,
                  gradient_magnitude_filter>{},
    with_stage_count{55},
    with_step_rate{0.15},
    with_tree_limits{6, 15, 2, 32},
    with_gradient_sampling{0.4, 0.4},
    with_regularisation{0.01, 0.5, 0.0});
  fit (leafwise_booster, x_reg, y_reg);
  if (r2 (y_reg, predict (leafwise_booster, x_reg)) < 0.85)
    {
      std::cerr << "test_boosting: quantile_bin + best_gain_frontier + GOSS failed\n";
      return 1;
    }

  // 3. Oblivious symmetric tree growth + ordered categorical statistics.
  auto oblivious_booster = make_stagewise_booster<squared_residual_loss> (
    with_booster<quantile_bin_policy,
                  oblivious_symmetric_growth,
                  uniform_full_sample>{},
    with_stage_count{50},
    with_step_rate{0.15},
    with_tree_limits{4, 16, 2, 32},
    with_ordered_categories{5.0, true},
    with_regularisation{0.0, 0.5, 0.0});
  fit (oblivious_booster, x_reg, y_reg);
  if (r2 (y_reg, predict (oblivious_booster, x_reg)) < 0.88)
    {
      std::cerr << "test_boosting: oblivious_symmetric_growth booster failed\n";
      return 1;
    }

  // Verify stagewise_booster also satisfies transformer (leaf index embeddings).
  data_table leaf_embed = transform (oblivious_booster, x_reg);
  if (leaf_embed.row_count () != n || leaf_embed.column_count () != 50)
    {
      std::cerr << "test_boosting: leaf-index transformer projection failed\n";
      return 1;
    }

  // 4. Binary classification with logistic_bernoulli_loss.
  std::vector<double> y_bin (n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    {
      y_bin[i] = (y_reg[i] > 5.0) ? 1.0 : 0.0;
    }
  stagewise_booster<logistic_bernoulli_loss,
                    quantile_bin_policy,
                    best_gain_frontier_growth>
    bin_booster (with_stage_count{40}, with_step_rate{0.15});
  fit (bin_booster, x_reg, y_bin);
  classification_accuracy acc;
  if (acc (y_bin, predict (bin_booster, x_reg)) < 0.90)
    {
      std::cerr << "test_boosting: binary classification booster failed\n";
      return 1;
    }
  data_table bin_beliefs = predict_beliefs (bin_booster, x_reg);
  if (bin_beliefs.column_count () != 2)
    {
      std::cerr << "test_boosting: binary beliefs shape mismatch\n";
      return 1;
    }

  // 5. Multiclass classification with multinomial_deviance_loss (3 classes).
  std::vector<double> y_multi (n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    {
      if (i < 20)
        {
          y_multi[i] = 0.0;
        }
      else if (i < 40)
        {
          y_multi[i] = 1.0;
        }
      else
        {
          y_multi[i] = 2.0;
        }
    }
  stagewise_booster<multinomial_deviance_loss,
                    quantile_bin_policy,
                    depth_synchronous_growth>
    multi_booster (with_stage_count{30}, with_step_rate{0.2});
  fit (multi_booster, x_reg, y_multi);
  if (acc (y_multi, predict (multi_booster, x_reg)) < 0.90)
    {
      std::cerr << "test_boosting: multiclass booster failed\n";
      return 1;
    }
  data_table multi_beliefs = predict_beliefs (multi_booster, x_reg);
  if (multi_beliefs.column_count () != 3
      || std::abs ((multi_beliefs.at (0, 0) + multi_beliefs.at (0, 1)
                    + multi_beliefs.at (0, 2))
                    - 1.0)
            > 1.0e-5)
    {
      std::cerr << "test_boosting: multiclass beliefs failed\n";
      return 1;
    }

  // 6. Edge-case test: heavily skewed binary feature with quantile_bin_policy
  //    and constant-feature table with oblivious_symmetric_growth.
  data_table x_skew (40, 2, 1.0);
  std::vector<double> y_skew (40, 10.0);
  x_skew.at (0, 0) = 0.0;
  y_skew[0] = -10.0;

  stagewise_booster<squared_residual_loss,
                    quantile_bin_policy,
                    oblivious_symmetric_growth>
    skew_booster (with_stage_count{25},
                  with_step_rate{0.3},
                  with_tree_limits{3, 8, 1, 16},
                  with_regularisation{0.0, 0.01, 0.0});
  fit (skew_booster, x_skew, y_skew);
  auto skew_preds = predict (skew_booster, x_skew);
  if (skew_preds[0] >= 0.0 || skew_preds[1] <= 5.0)
    {
      std::cerr << "test_boosting: skewed binary quantile binning failed\n";
      return 1;
    }

  // 7. Stress test: all-NaN table, 1-row table, and short-row infer_one.
  data_table x_all_nan (10, 3, missing_value ());
  std::vector<double> y_all_nan (10, 3.5);
  stagewise_booster<squared_residual_loss,
                    quantile_bin_policy,
                    best_gain_frontier_growth>
    nan_booster (with_stage_count{10});
  fit (nan_booster, x_all_nan, y_all_nan);
  auto nan_preds = predict (nan_booster, x_all_nan);
  if (std::isnan (nan_preds[0]) || std::abs (nan_preds[0] - 3.5) > 1.0e-5)
    {
      std::cerr << "test_boosting: all-NaN table booster failed\n";
      return 1;
    }

  data_table x_one (1, 2, 2.0);
  std::vector<double> y_one = {7.25};
  stagewise_booster<squared_residual_loss,
                    exact_cut_policy,
                    oblivious_symmetric_growth>
    one_booster (with_stage_count{5});
  fit (one_booster, x_one, y_one);
  if (std::abs (predict (one_booster, x_one)[0] - 7.25) > 1.0e-5)
    {
      std::cerr << "test_boosting: 1-row booster failed\n";
      return 1;
    }

  std::vector<double> short_row = {1.0};
  double short_pred = infer_one (oblivious_booster, short_row);
  if (std::isnan (short_pred))
    {
      std::cerr << "test_boosting: short-row infer_one returned NaN\n";
      return 1;
    }

  // 8. Test monotonic feature constraints (+1 and -1), feature importances,
  //    and validation-monitored early stopping.
  data_table mono_x (40, 2);
  std::vector<double> mono_y (40, 0.0);
  for (std::size_t i = 0; i < 40; ++i)
    {
      double f0 = static_cast<double> (i) * 0.25;
      double f1 = static_cast<double> (i % 10) * 0.5;
      mono_x.at (i, 0) = f0;
      mono_x.at (i, 1) = f1;
      double noise = (i % 3 == 0) ? -3.0 : 2.0;
      mono_y[i] = 2.0 * f0 - 1.5 * f1 + noise;
    }

  stagewise_booster<squared_residual_loss,
                    exact_cut_policy,
                    depth_synchronous_growth>
    mono_booster (with_stage_count{25},
                  with_step_rate{0.15},
                  with_tree_limits{4, 15, 2, 32},
                  with_monotonic_constraints{{1, -1}});
  fit (mono_booster, mono_x, mono_y);

  // Check monotonic non-decreasing in feature 0 and non-increasing in feature 1.
  for (int step = 0; step < 30; ++step)
    {
      double a = static_cast<double> (step) * 0.3;
      double b = a + 0.3;
      std::vector<double> r_a0 = {a, 2.0};
      std::vector<double> r_b0 = {b, 2.0};
      if (infer_one (mono_booster, r_a0) > infer_one (mono_booster, r_b0) + 1e-9)
        {
          std::cerr << "test_boosting: +1 monotonic constraint violated\n";
          return 1;
        }
      std::vector<double> r_a1 = {3.0, a};
      std::vector<double> r_b1 = {3.0, b};
      if (infer_one (mono_booster, r_a1) < infer_one (mono_booster, r_b1) - 1e-9)
        {
          std::cerr << "test_boosting: -1 monotonic constraint violated\n";
          return 1;
        }
    }

  const auto &imps = mono_booster.feature_importances ();
  const auto &scounts = mono_booster.split_counts ();
  if (imps.size () != 2 || scounts.size () != 2
      || std::abs ((imps[0] + imps[1]) - 1.0) > 1.0e-6
      || imps[0] <= imps[1])
    {
      std::cerr << "test_boosting: feature_importances failed\n";
      return 1;
    }

  stagewise_booster<squared_residual_loss,
                    quantile_bin_policy,
                    best_gain_frontier_growth>
    monitored_booster (with_stage_count{150}, with_step_rate{0.3});
  monitored_booster.calibrate_with_monitor (mono_x, mono_y, mono_x, mono_y, 3);
  if (!monitored_booster.is_calibrated ()
      || monitored_booster.tree_count () == 0)
    {
      std::cerr << "test_boosting: calibrate_with_monitor failed\n";
      return 1;
    }

  std::cout << "  [PASS] test_boosting (exact, quantile-hist, depth/leaf/oblivious, GOSS, ordered cats, multiclass, monotonic, importances)\n";
  return 0;
}

} // namespace bellows_test
