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
#include <sstream>
#include <vector>

#include <bellows/bellows.hpp>

namespace bellows_test
{

int
run_composition_tests ()
{
  using namespace bellows;

  const std::size_t n = 50;
  data_table x (n, 4);
  std::vector<double> y (n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    {
      double f0 = static_cast<double> (i) * 0.15;
      double f1 = (i % 7 == 0) ? missing_value ()
                                : static_cast<double> (i % 5) * 0.4;
      double f2 = 42.0; // Constant column to be removed by dispersion_filter.
      double f3 = static_cast<double> (i % 3) * 0.1;
      x.at (i, 0) = f0;
      x.at (i, 1) = f1;
      x.at (i, 2) = f2;
      x.at (i, 3) = f3;
      y[i] = (f0 > 3.5) ? 1.0 : 0.0;
    }

  // 1. Test compile-time pipe chain (`operator|`).
  auto static_chain =
    column_mender (mending_strategy::median)
    | dispersion_filter (1.0e-5)
    | moment_normaliser ()
    | penalised_logistic_classifier ();

  fit (static_chain, x, y);
  classification_accuracy acc;
  if (acc (y, predict (static_chain, x)) < 0.90)
    {
      std::cerr << "test_composition: static composite_chain failed\n";
      return 1;
    }
  data_table beliefs = predict_beliefs (static_chain, x);
  if (beliefs.row_count () != n || beliefs.column_count () != 2)
    {
      std::cerr << "test_composition: composite_chain predict_beliefs failed\n";
      return 1;
    }

  // Verify binary serialisation of a multi-stage composite_chain.
  static_assert (serialisable<decltype (static_chain)>);
  {
    std::stringstream ss;
    save_artifact (static_chain, ss, 99U);
    decltype (static_chain) loaded_chain;
    load_artifact (loaded_chain, ss);
    if (acc (y, predict (loaded_chain, x)) < 0.90)
      {
        std::cerr << "test_composition: composite_chain serialisation failed\n";
        return 1;
      }
  }

  // 2. Test type-erased runtime dynamic_chain.
  dynamic_chain dyn;
  dyn.append_transformer (column_mender (mending_strategy::mean));
  dyn.append_transformer (interval_rescaler (0.0, 1.0));
  dyn.set_predictor (stagewise_booster<logistic_bernoulli_loss> (
    with_stage_count{25}, with_step_rate{0.15}));
  fit (dyn, x, y);
  if (acc (y, predict (dyn, x)) < 0.90)
    {
      std::cerr << "test_composition: dynamic_chain failed\n";
      return 1;
    }

  // 3. Test cross_validate with fold_splitter and stratified_splitter.
  stratified_splitter strat (5, 42);
  validation_summary cv_res =
    cross_validate ( static_chain, x, y, strat, acc);
  if (cv_res.fold_scores.size () != 5 || cv_res.mean_score < 0.80)
    {
      std::cerr << "test_composition: stratified cross_validate failed\n";
      return 1;
    }

  // 4. Test parameter_sweep over a tunable composite_chain pipeline.
  auto tunable_chain =
    column_mender (mending_strategy::median)
    | moment_normaliser ()
    | stagewise_booster<logistic_bernoulli_loss> (with_stage_count{20});
  static_assert (tunable<decltype (tunable_chain)>);

  std::vector<parameter_assignment> grid = {
    {{"step_rate", 0.05}, {"max_depth", 2.0}},
    {{"step_rate", 0.20}, {"max_depth", 4.0}}};
  auto sweep = parameter_sweep (tunable_chain, x, y, strat, acc, grid);
  if (!sweep.best_estimator.is_calibrated () || sweep.best_score < 0.85
      || sweep.trial_scores.size () != 2)
    {
      std::cerr << "test_composition: parameter_sweep failed\n";
      return 1;
    }

  // 5. Test grid_explorer, random_latin_explorer, bellows::evaluate, and bellows::tune.
  random_latin_explorer lhs (4, 123);
  auto lhs_res = tune (tunable_chain, x, y, strat, acc, lhs);
  if (!lhs_res.best_estimator.is_calibrated ()
      || lhs_res.trial_scores.size () != 4
      || evaluate (lhs_res.best_estimator, x, y, acc) < 0.85)
    {
      std::cerr << "test_composition: random_latin_explorer tune failed\n";
      return 1;
    }

  grid_explorer gexp ({{"step_rate", {0.08, 0.2}}, {"max_depth", {2.0, 3.0}}});
  auto g_res = tune (tunable_chain, x, y, strat, acc, gexp);
  if (g_res.trial_scores.size () != 4 || g_res.best_score < 0.85)
    {
      std::cerr << "test_composition: grid_explorer tune failed\n";
      return 1;
    }

  // 6. Test consensus_ensemble (majority_vote) and stacked_generaliser.
  column_mender cm_ens;
  data_table clean_x = fit_and_transform (cm_ens, x);
  consensus_ensemble voter (aggregation_mode::majority_vote);
  voter.add_member (penalised_logistic_classifier (), 1.0);
  voter.add_member (max_margin_classifier (), 1.0);
  voter.add_member (
    subspace_tree_ensemble (15, 4, proximity_mode::classification, 42), 1.5);
  fit (voter, clean_x, y);
  if (evaluate (voter, clean_x, y, acc) < 0.90)
    {
      std::cerr << "test_composition: consensus_ensemble failed\n";
      return 1;
    }

  stacked_generaliser<penalised_logistic_classifier> stacker (
    penalised_logistic_classifier (), 4, false, 42);
  stacker.add_base_learner (penalised_logistic_classifier ());
  stacker.add_base_learner (max_margin_classifier ());
  stacker.add_base_learner (
    subspace_tree_ensemble (15, 4, proximity_mode::classification, 99));
  fit (stacker, clean_x, y);
  if (evaluate (stacker, clean_x, y, acc) < 0.90)
    {
      std::cerr << "test_composition: stacked_generaliser failed\n";
      return 1;
    }

  std::cout << "  [PASS] test_composition (pipe chains, dynamic_chain, CV, explorers, consensus & stacking)\n";
  return 0;
}

} // namespace bellows_test
