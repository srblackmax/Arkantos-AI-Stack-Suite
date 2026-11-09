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
#include <sstream>
#include <vector>

#include <bellows/bellows.hpp>

namespace bellows_test
{

int
run_estimator_tests ()
{
  using namespace bellows;

  // 1. Test penalised_linear_regressor (both L2 Cholesky/LU and L1/L2 elastic).
  data_table reg_x (30, 2);
  std::vector<double> reg_y (30, 0.0);
  for (std::size_t i = 0; i < 30; ++i)
    {
      double x0 = static_cast<double> (i) * 0.2;
      double x1 = static_cast<double> (i % 5) * 0.5;
      reg_x.at (i, 0) = x0;
      reg_x.at (i, 1) = x1;
      reg_y[i] = 3.0 * x0 - 1.5 * x1 + 2.0;
    }

  penalised_linear_regressor ridge_reg (with_regularisation{0.0, 1.0e-4, 0.0});
  fit (ridge_reg, reg_x, reg_y);
  auto ridge_preds = predict (ridge_reg, reg_x);
  explained_variance_ratio r2;
  if (r2 (reg_y, ridge_preds) < 0.99)
    {
      std::cerr << "test_estimators: penalised_linear_regressor (L2) failed\n";
      return 1;
    }

  penalised_linear_regressor elastic_reg (
    with_regularisation{1.0e-3, 1.0e-3, 0.0});
  fit (elastic_reg, reg_x, reg_y);
  if (r2 (reg_y, predict (elastic_reg, reg_x)) < 0.98)
    {
      std::cerr << "test_estimators: penalised_linear_regressor (L1/L2) failed\n";
      return 1;
    }

  // 2. Test penalised_logistic_classifier, proximity_learner, density, forest.
  data_table clf_x (40, 2);
  std::vector<double> clf_y (40, 0.0);
  for (std::size_t i = 0; i < 40; ++i)
    {
      double label = (i >= 20) ? 1.0 : 0.0;
      double shift = label > 0.5 ? 3.0 : -3.0;
      clf_x.at (i, 0) = shift + static_cast<double> (i % 4) * 0.2;
      clf_x.at (i, 1) = shift * 0.8 + static_cast<double> (i % 3) * 0.1;
      clf_y[i] = label;
    }

  penalised_logistic_classifier log_clf;
  fit (log_clf, clf_x, clf_y);
  classification_accuracy acc;
  if (acc (clf_y, predict (log_clf, clf_x)) < 0.95)
    {
      std::cerr << "test_estimators: penalised_logistic_classifier failed\n";
      return 1;
    }
  data_table log_beliefs = predict_beliefs (log_clf, clf_x);
  if (log_beliefs.column_count () != 2
      || std::abs ((log_beliefs.at (0, 0) + log_beliefs.at (0, 1)) - 1.0)
            > 1.0e-6)
    {
      std::cerr << "test_estimators: logistic beliefs do not sum to 1\n";
      return 1;
    }

  // 3. Test proximity_learner (k-NN classification).
  proximity_learner knn (3, proximity_mode::classification);
  fit (knn, clf_x, clf_y);
  if (acc (clf_y, predict (knn, clf_x)) < 0.95)
    {
      std::cerr << "test_estimators: proximity_learner failed\n";
      return 1;
    }

  // 4. Test gaussian_density_classifier.
  gaussian_density_classifier gnb;
  fit (gnb, clf_x, clf_y);
  if (acc (clf_y, predict (gnb, clf_x)) < 0.95)
    {
      std::cerr << "test_estimators: gaussian_density_classifier failed\n";
      return 1;
    }

  // 5. Test subspace_tree_ensemble (bagged random subspace forest).
  subspace_tree_ensemble forest (20, 4, proximity_mode::classification, 42);
  fit (forest, clf_x, clf_y);
  if (acc (clf_y, predict (forest, clf_x)) < 0.95)
    {
      std::cerr << "test_estimators: subspace_tree_ensemble failed\n";
      return 1;
    }

  // Test multiclass subspace_tree_ensemble with non-monotonic class regions.
  data_table band_x (45, 1);
  std::vector<double> band_y (45, 0.0);
  for (std::size_t i = 0; i < 45; ++i)
    {
      if (i < 15)
        {
          band_x.at (i, 0) = -3.0 + static_cast<double> (i) * 0.1;
          band_y[i] = 0.0;
        }
      else if (i < 30)
        {
          band_x.at (i, 0) = -0.5 + static_cast<double> (i - 15) * 0.05;
          band_y[i] = 1.0;
        }
      else
        {
          band_x.at (i, 0) = 2.0 + static_cast<double> (i - 30) * 0.1;
          band_y[i] = 2.0;
        }
    }
  subspace_tree_ensemble mc_forest (25, 4, proximity_mode::classification, 7);
  fit (mc_forest, band_x, band_y);
  if (acc (band_y, predict (mc_forest, band_x)) < 0.95)
    {
      std::cerr << "test_estimators: multiclass subspace_tree_ensemble failed\n";
      return 1;
    }

  // 6. Test centroid_partitioner in both predictor and transformer roles.
  centroid_partitioner clusterer (2, 50, 42);
  fit (clusterer, clf_x);
  auto cluster_ids = predict (clusterer, clf_x);
  data_table dist_space = transform (clusterer, clf_x);
  silhouette_cohesion sil;
  double sil_score = sil.evaluate_clusters (clf_x, cluster_ids);
  if (dist_space.column_count () != 2 || sil_score < 0.70)
    {
      std::cerr << "test_estimators: centroid_partitioner failed (sil="
                << sil_score << ")\n";
      return 1;
    }

  // 7. Test multiclass penalised_logistic_classifier (K = 3).
  data_table mc_x (45, 2);
  std::vector<double> mc_y (45, 0.0);
  for (std::size_t i = 0; i < 45; ++i)
    {
      if (i < 15)
        {
          mc_x.at (i, 0) = -4.0 + static_cast<double> (i % 3) * 0.2;
          mc_x.at (i, 1) = -3.0 + static_cast<double> (i % 4) * 0.1;
          mc_y[i] = 0.0;
        }
      else if (i < 30)
        {
          mc_x.at (i, 0) = 4.0 + static_cast<double> (i % 3) * 0.2;
          mc_x.at (i, 1) = -3.0 + static_cast<double> (i % 4) * 0.1;
          mc_y[i] = 1.0;
        }
      else
        {
          mc_x.at (i, 0) = 0.0 + static_cast<double> (i % 3) * 0.2;
          mc_x.at (i, 1) = 4.0 + static_cast<double> (i % 4) * 0.1;
          mc_y[i] = 2.0;
        }
    }

  penalised_logistic_classifier mc_log_clf;
  fit (mc_log_clf, mc_x, mc_y);
  if (acc (mc_y, predict (mc_log_clf, mc_x)) < 0.95)
    {
      std::cerr << "test_estimators: multiclass logistic classifier failed\n";
      return 1;
    }
  data_table mc_beliefs = predict_beliefs (mc_log_clf, mc_x);
  if (mc_beliefs.column_count () != 3
      || std::abs ((mc_beliefs.at (0, 0) + mc_beliefs.at (0, 1)
                    + mc_beliefs.at (0, 2))
                    - 1.0)
            > 1.0e-5)
    {
      std::cerr << "test_estimators: multiclass logistic beliefs failed\n";
      return 1;
    }

  // 8. Test binary serialisation round-trips for all classical estimators.
  {
    std::stringstream ss;
    save_artifact (mc_log_clf, ss, 10);
    penalised_logistic_classifier loaded_log;
    load_artifact (loaded_log, ss);
    if (acc (mc_y, predict (loaded_log, mc_x)) < 0.95)
      {
        std::cerr << "test_estimators: logistic serialisation failed\n";
        return 1;
      }
  }
  {
    std::stringstream ss;
    save_artifact (knn, ss, 11);
    proximity_learner loaded_knn;
    load_artifact (loaded_knn, ss);
    if (acc (clf_y, predict (loaded_knn, clf_x)) < 0.95)
      {
        std::cerr << "test_estimators: proximity serialisation failed\n";
        return 1;
      }
  }
  {
    std::stringstream ss;
    save_artifact (gnb, ss, 12);
    gaussian_density_classifier loaded_gnb;
    load_artifact (loaded_gnb, ss);
    if (acc (clf_y, predict (loaded_gnb, clf_x)) < 0.95)
      {
        std::cerr << "test_estimators: density serialisation failed\n";
        return 1;
      }
  }
  {
    std::stringstream ss;
    save_artifact (forest, ss, 13);
    subspace_tree_ensemble loaded_forest;
    load_artifact (loaded_forest, ss);
    if (acc (clf_y, predict (loaded_forest, clf_x)) < 0.95)
      {
        std::cerr << "test_estimators: forest serialisation failed\n";
        return 1;
      }
  }
  {
    std::stringstream ss;
    save_artifact (clusterer, ss, 14);
    centroid_partitioner loaded_clusterer;
    load_artifact (loaded_clusterer, ss);
    if (loaded_clusterer.centroids ().row_count () != 2)
      {
        std::cerr << "test_estimators: clusterer serialisation failed\n";
        return 1;
      }
  }

  // 8. Test max_margin_classifier (binary and multiclass + serialisation).
  max_margin_classifier mmc (0.01, 0.08, 300);
  static_assert (belief_predictor<max_margin_classifier>);
  static_assert (tunable<max_margin_classifier>);
  static_assert (serialisable<max_margin_classifier>);
  fit (mmc, clf_x, clf_y);
  if (acc (clf_y, predict (mmc, clf_x)) < 0.95)
    {
      std::cerr << "test_estimators: max_margin_classifier binary failed\n";
      return 1;
    }
  {
    std::stringstream ss;
    save_artifact (mmc, ss, 15);
    max_margin_classifier loaded_mmc;
    load_artifact (loaded_mmc, ss);
    if (acc (clf_y, predict (loaded_mmc, clf_x)) < 0.95)
      {
        std::cerr << "test_estimators: max_margin_classifier io failed\n";
        return 1;
      }
  }

  // 9. Test kernel_ridge_regressor (RBF kernel on nonlinear curve).
  data_table nl_x (25, 1);
  std::vector<double> nl_y (25, 0.0);
  for (std::size_t i = 0; i < 25; ++i)
    {
      double xv = -2.0 + 4.0 * static_cast<double> (i) / 24.0;
      nl_x.at (i, 0) = xv;
      nl_y[i] = std::sin (xv) + 0.25 * xv * xv;
    }
  kernel_ridge_regressor krr (1.0e-3, 1.5, kernel_kind::radial_basis);
  static_assert (tunable<kernel_ridge_regressor>);
  static_assert (serialisable<kernel_ridge_regressor>);
  fit (krr, nl_x, nl_y);
  mean_squared_deviation msd;
  if (msd (nl_y, predict (krr, nl_x)) > 1.0e-2)
    {
      std::cerr << "test_estimators: kernel_ridge_regressor failed\n";
      return 1;
    }
  {
    std::stringstream ss;
    save_artifact (krr, ss, 16);
    kernel_ridge_regressor loaded_krr;
    load_artifact (loaded_krr, ss);
    if (msd (nl_y, predict (loaded_krr, nl_x)) > 1.0e-2)
      {
        std::cerr << "test_estimators: kernel_ridge_regressor io failed\n";
        return 1;
      }
  }

  // 10. Test hierarchical_partitioner and density_reachability_partitioner.
  hierarchical_partitioner hpart (2, "ward");
  fit (hpart, clf_x);
  if (hpart.labels ().size () != clf_x.row_count ()
      || hpart.labels ()[0] == hpart.labels ()[clf_x.row_count () - 1]
      || hpart.linkage_matrix ().row_count () != clf_x.row_count () - 1)
    {
      std::cerr << "test_estimators: hierarchical_partitioner failed\n";
      return 1;
    }

  data_table db_x = {
    {0.0, 0.0}, {0.1, 0.0}, {0.0, 0.1}, {0.1, 0.1},
    {10.0, 10.0}, {10.1, 10.0}, {10.0, 10.1}, {10.1, 10.1},
    {100.0, -100.0}};
  density_reachability_partitioner dbpart (0.5, 3);
  fit (dbpart, db_x);
  if (dbpart.cluster_count () != 2
      || dbpart.labels ()[8] != -1.0
      || dbpart.labels ()[0] == dbpart.labels ()[4])
    {
      std::cerr << "test_estimators: density_reachability_partitioner failed\n";
      return 1;
    }
  {
    std::stringstream ss;
    save_artifact (dbpart, ss, 17);
    density_reachability_partitioner loaded_db;
    load_artifact (loaded_db, ss);
    std::vector<double> noise_q = {100.0, -100.0};
    if (infer_one (loaded_db, noise_q) != -1.0)
      {
        std::cerr << "test_estimators: density partitioner io failed\n";
        return 1;
      }
  }

  std::cout << "  [PASS] test_estimators (linear, logistic, margin, kernel, proximity, density, clustering, forest)\n";
  return 0;
}

} // namespace bellows_test
