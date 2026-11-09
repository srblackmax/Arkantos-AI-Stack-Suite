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
#include <string>
#include <vector>

#include <bellows/bellows.hpp>

namespace bellows_test
{

namespace
{

bool
approx_equal (double a, double b, double tol = 1.0e-6)
{
  return std::abs (a - b) <= tol;
}

} // namespace

int
run_core_tests ()
{
  using namespace bellows;

  // Compile-time verification of the C++20 Concept Lattice.
  static_assert (tabular_source<data_table>);
  static_assert (tabular_source<table_view>);
  static_assert (target_sequence<std::vector<double>>);

  static_assert (unsupervised_fittable<moment_normaliser>);
  static_assert (supervised_fittable<moment_normaliser>);
  static_assert (transformer<moment_normaliser>);
  static_assert (categorical_aware<moment_normaliser>);
  static_assert (missing_aware<moment_normaliser>);

  static_assert (supervised_fittable<penalised_linear_regressor>);
  static_assert (predictor<penalised_linear_regressor>);
  static_assert (tunable<penalised_linear_regressor>);
  static_assert (serialisable<penalised_linear_regressor>);
  static_assert (!belief_predictor<penalised_linear_regressor>);

  static_assert (supervised_fittable<penalised_logistic_classifier>);
  static_assert (predictor<penalised_logistic_classifier>);
  static_assert (belief_predictor<penalised_logistic_classifier>);
  static_assert (serialisable<penalised_logistic_classifier>);

  static_assert (serialisable<moment_normaliser>);
  static_assert (serialisable<interval_rescaler>);
  static_assert (serialisable<column_mender>);
  static_assert (serialisable<indicator_expander>);
  static_assert (serialisable<ordered_target_encoder>);
  static_assert (serialisable<quantile_bucketer>);
  static_assert (serialisable<dispersion_filter>);
  static_assert (serialisable<relevance_selector>);
  static_assert (serialisable<proximity_learner>);
  static_assert (serialisable<gaussian_density_classifier>);
  static_assert (serialisable<centroid_partitioner>);
  static_assert (serialisable<subspace_tree_ensemble>);

  using serialisable_chain =
    composite_chain<moment_normaliser, penalised_linear_regressor>;
  static_assert (serialisable<serialisable_chain>);

  // Verify centroid_partitioner satisfies both predictor and transformer.
  static_assert (unsupervised_fittable<centroid_partitioner>);
  static_assert (predictor<centroid_partitioner>);
  static_assert (transformer<centroid_partitioner>);

  // Verify stagewise_booster satisfies predictor, transformer, tunable, etc.
  using reg_booster = stagewise_booster<squared_residual_loss>;
  using clf_booster = stagewise_booster<logistic_bernoulli_loss>;
  static_assert (supervised_fittable<reg_booster>);
  static_assert (predictor<reg_booster>);
  static_assert (transformer<reg_booster>);
  static_assert (categorical_aware<reg_booster>);
  static_assert (missing_aware<reg_booster>);
  static_assert (tunable<reg_booster>);
  static_assert (serialisable<reg_booster>);
  static_assert (belief_predictor<clf_booster>);

  static_assert (second_order_loss<squared_residual_loss>);
  static_assert (second_order_loss<huber_pseudo_loss>);
  static_assert (second_order_loss<logistic_bernoulli_loss>);
  static_assert (second_order_loss<multinomial_deviance_loss>);

  static_assert (evaluation_metric<mean_squared_deviation>);
  static_assert (evaluation_metric<explained_variance_ratio>);
  static_assert (evaluation_metric<classification_accuracy>);
  static_assert (evaluation_metric<roc_area_under_curve>);

  // Verify Hyper Math nd_tensor and data_frame foundations in data_table.
  hyper_math::tensor::nd_tensor<double> hm_mat = {
    {1.0, 2.0, missing_value ()},
    {4.0, 0.0, 1.0},
    {9.0, 1.0, 2.0}};
  data_table tbl (hm_mat);
  tbl.set_scale (1, feature_scale::nominal);

  if (tbl.row_count () != 3 || tbl.column_count () != 3)
    {
      std::cerr << "test_core: data_table dimensions mismatch\n";
      return 1;
    }
  if (!tbl.has_missing_entries () || !tbl.has_nominal_columns ())
    {
      std::cerr << "test_core: missing/nominal detection failed\n";
      return 1;
    }
  if (tbl.cardinality (1) != 3)
    {
      std::cerr << "test_core: nominal cardinality inference failed\n";
      return 1;
    }
  if (tbl.validity ().null_count () != 1)
    {
      std::cerr << "test_core: Hyper Math validity_bitmap count mismatch\n";
      return 1;
    }

  // Verify deep copy value semantics of data_table (no shared slab aliasing).
  data_table tbl_copy = tbl;
  tbl_copy.at (0, 0) = 999.0;
  tbl_copy.at (1, 1) = missing_value ();
  if (approx_equal (tbl.at (0, 0), 999.0)
      || tbl_copy.validity ().null_count () != 2
      || tbl.validity ().null_count () != 1)
    {
      std::cerr << "test_core: data_table deep copy or validity sync failed\n";
      return 1;
    }

  // Test conversion from Hyper Math tabular::data_frame with string column.
  hyper_math::tabular::series s_num ("x", std::vector<double>{1.5, 2.5, 3.5});
  hyper_math::tabular::series s_cat (
    "color", std::vector<std::string>{"red", "blue", "red"});
  hyper_math::tabular::data_frame df ({s_num, s_cat});
  data_table from_df (df);

  if (from_df.scale (1) != feature_scale::nominal
      || from_df.cardinality (1) != 2
      || !approx_equal (from_df.at (0, 1), from_df.at (2, 1)))
    {
      std::cerr << "test_core: Hyper Math data_frame nominal conversion failed\n";
      return 1;
    }

  // Verify evaluation metrics.
  std::vector<double> y_true = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> y_pred = {1.0, 2.0, 3.0, 4.0};
  mean_squared_deviation msd;
  root_mean_squared_deviation rmsd;
  mean_absolute_deviation mad;
  explained_variance_ratio r2;

  if (!approx_equal (msd (y_true, y_pred), 0.0)
      || !approx_equal (rmsd (y_true, y_pred), 0.0)
      || !approx_equal (mad (y_true, y_pred), 0.0)
      || !approx_equal (r2 (y_true, y_pred), 1.0))
    {
      std::cerr << "test_core: regression metrics failed on exact match\n";
      return 1;
    }

  std::vector<double> cls_true = {0.0, 0.0, 1.0, 1.0};
  std::vector<double> cls_pred = {0.0, 0.0, 1.0, 1.0};
  std::vector<double> cls_prob = {0.05, 0.10, 0.85, 0.95};
  classification_accuracy acc;
  balanced_f1_score f1;
  logarithmic_loss logloss;
  roc_area_under_curve auc;

  if (!approx_equal (acc (cls_true, cls_pred), 1.0)
      || !approx_equal (f1 (cls_true, cls_pred), 1.0)
      || logloss (cls_true, cls_prob) >= 0.15
      || !approx_equal (auc (cls_true, cls_prob), 1.0))
    {
      std::cerr << "test_core: classification metrics failed\n";
      return 1;
    }

  // Verify fit_and_transform CPO and out-of-bounds / mismatched-target guards.
  moment_normaliser ft_norm;
  data_table ft_out = fit_and_transform (ft_norm, from_df);
  if (ft_out.row_count () != 3 || !ft_norm.is_calibrated ())
    {
      std::cerr << "test_core: fit_and_transform failed\n";
      return 1;
    }

  bool caught_oob = false;
  try
    {
      (void) tbl.scale (99);
    }
  catch (const std::out_of_range &)
    {
      caught_oob = true;
    }
  bool caught_mismatch = false;
  try
    {
      penalised_linear_regressor lr;
      std::vector<double> bad_y = {1.0};
      fit (lr, from_df, bad_y);
    }
  catch (const std::invalid_argument &)
    {
      caught_mismatch = true;
    }
  if (!caught_oob || !caught_mismatch)
    {
      std::cerr << "test_core: bounds or target-size validation failed\n";
      return 1;
    }

  std::cout << "  [PASS] test_core (concept lattice, Hyper Math foundations, metrics)\n";
  return 0;
}

} // namespace bellows_test
