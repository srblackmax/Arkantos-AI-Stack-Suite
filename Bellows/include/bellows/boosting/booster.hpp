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

#ifndef BELLOWS_BOOSTING_BOOSTER_HPP
#define BELLOWS_BOOSTING_BOOSTER_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <bellows/boosting/losses.hpp>
#include <bellows/boosting/policies.hpp>
#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>
#include <bellows/core/parameters.hpp>

namespace bellows
{

// Compact decision node supporting native missing-value routing.
struct stage_tree_node
{
  std::int32_t feature_index = -1;
  double threshold = 0.0;
  bool missing_goes_right = false;
  std::int32_t left_child = -1;
  std::int32_t right_child = -1;
  double leaf_value = 0.0;

  bool
  is_leaf () const noexcept
  {
    return feature_index < 0;
  }
};

// Single additive decision tree (supports both standard and oblivious trees).
struct additive_stage_tree
{
  bool is_oblivious = false;
  std::vector<stage_tree_node> nodes;
  std::vector<std::int32_t> oblivious_features;
  std::vector<double> oblivious_thresholds;
  std::vector<std::uint8_t> oblivious_missing_right;
  std::vector<double> oblivious_leaves;

  double
  evaluate_row (std::span<const double> row) const noexcept
  {
    if (is_oblivious && !oblivious_leaves.empty ())
      {
        std::size_t depth = oblivious_features.size ();
        std::size_t mask = 0;
        for (std::size_t d = 0; d < depth; ++d)
          {
            std::size_t f = static_cast<std::size_t> (oblivious_features[d]);
            double v = f < row.size () ? row[f] : missing_value ();
            bool go_right = false;
            if (is_missing (v))
              {
                go_right = (oblivious_missing_right[d] != 0);
              }
            else
              {
                go_right = (v > oblivious_thresholds[d]);
              }
            if (go_right)
              {
                mask |= (std::size_t{1} << d);
              }
          }
        return oblivious_leaves[mask];
      }

    if (nodes.empty ())
      {
        return 0.0;
      }
    std::int32_t idx = 0;
    while (idx >= 0 && !nodes[static_cast<std::size_t> (idx)].is_leaf ())
      {
        const auto &node = nodes[static_cast<std::size_t> (idx)];
        std::size_t f = static_cast<std::size_t> (node.feature_index);
        double v = f < row.size () ? row[f] : missing_value ();
        bool go_right = false;
        if (is_missing (v))
          {
            go_right = node.missing_goes_right;
          }
        else
          {
            go_right = (v > node.threshold);
          }
        idx = go_right ? node.right_child : node.left_child;
      }
    return idx >= 0 ? nodes[static_cast<std::size_t> (idx)].leaf_value : 0.0;
  }

  double
  leaf_index_for_row (std::span<const double> row) const noexcept
  {
    if (is_oblivious && !oblivious_leaves.empty ())
      {
        std::size_t depth = oblivious_features.size ();
        std::size_t mask = 0;
        for (std::size_t d = 0; d < depth; ++d)
          {
            std::size_t f = static_cast<std::size_t> (oblivious_features[d]);
            double v = f < row.size () ? row[f] : missing_value ();
            bool go_right = is_missing (v)
                              ? (oblivious_missing_right[d] != 0)
                              : (v > oblivious_thresholds[d]);
            if (go_right)
              {
                mask |= (std::size_t{1} << d);
              }
          }
        return static_cast<double> (mask);
      }
    if (nodes.empty ())
      {
        return 0.0;
      }
    std::int32_t idx = 0;
    while (idx >= 0 && !nodes[static_cast<std::size_t> (idx)].is_leaf ())
      {
        const auto &node = nodes[static_cast<std::size_t> (idx)];
        std::size_t f = static_cast<std::size_t> (node.feature_index);
        double v = f < row.size () ? row[f] : missing_value ();
        bool go_right =
          is_missing (v) ? node.missing_goes_right : (v > node.threshold);
        idx = go_right ? node.right_child : node.left_child;
      }
    return static_cast<double> (idx);
  }
};

// Unified second-order stagewise gradient boosting estimator.
template <typename Loss_Type = squared_residual_loss,
          typename Cut_Policy = quantile_bin_policy,
          typename Growth_Policy = depth_synchronous_growth,
          typename Sampling_Policy = uniform_full_sample>
class stagewise_booster
{
public:
  static constexpr bool supports_nominal_columns = true;
  static constexpr bool supports_missing_entries = true;

  stagewise_booster () = default;

  template <typename... Modifiers>
    requires (sizeof...(Modifiers) > 0
              && !(sizeof...(Modifiers) == 1
                    && (std::is_same_v<std::remove_cvref_t<Modifiers>,
                                      stagewise_booster>
                        && ...)))
  explicit stagewise_booster (Modifiers... mods)
  {
    (configure (mods), ...);
  }

  void
  configure (with_regularisation reg) noexcept
  {
    reg_ = reg;
  }

  void
  configure (with_step_rate rate) noexcept
  {
    step_rate_ = rate.value;
  }

  void
  configure (with_stage_count stages) noexcept
  {
    stage_count_ = stages.count;
  }

  void
  configure (with_tree_limits limits) noexcept
  {
    limits_ = limits;
  }

  void
  configure (with_ordered_categories ord) noexcept
  {
    ordered_cats_ = ord;
  }

  void
  configure (with_gradient_sampling samp) noexcept
  {
    grad_sampling_ = samp;
  }

  void
  configure (with_seed s) noexcept
  {
    seed_ = s.seed;
  }

  void
  configure (with_monotonic_constraints mc)
  {
    monotonic_ = std::move (mc);
  }

  void
  configure (Loss_Type loss) noexcept
  {
    loss_ = loss;
  }

  bool
  is_calibrated () const noexcept
  {
    return calibrated_;
  }

  std::size_t
  tree_count () const noexcept
  {
    return trees_.size ();
  }

  std::size_t
  class_count () const noexcept
  {
    return class_count_;
  }

  std::size_t
  best_stage () const noexcept
  {
    return best_stage_;
  }

  const std::vector<double> &
  feature_importances () const noexcept
  {
    return feature_importances_;
  }

  const std::vector<std::size_t> &
  split_counts () const noexcept
  {
    return split_counts_;
  }

  std::vector<parameter_knob>
  parameter_space () const
  {
    return {
      {"step_rate", step_rate_, 0.02, 0.5},
      {"stage_count", static_cast<double> (stage_count_), 10.0, 100.0},
      {"max_depth", static_cast<double> (limits_.max_depth), 2.0, 8.0},
      {"max_leaves", static_cast<double> (limits_.max_leaves), 4.0, 64.0},
      {"l1", reg_.l1, 0.0, 1.0},
      {"l2", reg_.l2, 1.0e-2, 10.0},
      {"complexity", reg_.complexity, 0.0, 0.5}};
  }

  bool
  apply_parameter (const std::string &name, double val)
  {
    if (name == "step_rate")
      {
        step_rate_ = std::clamp (val, 1.0e-4, 2.0);
        return true;
      }
    if (name == "stage_count")
      {
        stage_count_ =
          static_cast<std::size_t> (std::max (1L, std::lround (val)));
        return true;
      }
    if (name == "max_depth")
      {
        limits_.max_depth =
          static_cast<std::size_t> (std::max (1L, std::lround (val)));
        return true;
      }
    if (name == "max_leaves")
      {
        limits_.max_leaves =
          static_cast<std::size_t> (std::max (2L, std::lround (val)));
        return true;
      }
    if (name == "l1")
      {
        reg_.l1 = std::max (0.0, val);
        return true;
      }
    if (name == "l2")
      {
        reg_.l2 = std::max (1.0e-6, val);
        return true;
      }
    if (name == "complexity")
      {
        reg_.complexity = std::max (0.0, val);
        return true;
      }
    return false;
  }

  template <tabular_source Data, target_sequence Targets>
  void
  calibrate (const Data &data, const Targets &targets)
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    feature_count_ = d;
    trees_.clear ();
    base_scores_.clear ();
    raw_gain_importances_.assign (d, 0.0);
    feature_importances_.assign (d, 0.0);
    split_counts_.assign (d, 0);
    best_stage_ = 0;

    if (n == 0 || d == 0)
      {
        calibrated_ = false;
        return;
      }
    if (targets.size () != n)
      {
        throw std::invalid_argument (
          "bellows::stagewise_booster::calibrate: target size mismatch");
      }

    std::vector<double> y_vec (n);
    for (std::size_t i = 0; i < n; ++i)
      {
        y_vec[i] = targets[i];
      }

    data_table work_table = prepare_ordered_categories (data, y_vec);

    quantile_bins qbins;
    if constexpr (Cut_Policy::uses_quantile_bins)
      {
        qbins = build_quantile_bins (work_table);
      }

    if constexpr (Loss_Type::is_multiclass)
      {
        long max_label = 0;
        for (double y : y_vec)
          {
            max_label = std::max (max_label, std::lround (y));
          }
        class_count_ =
          static_cast<std::size_t> (std::max (2L, max_label + 1));
        base_scores_.assign (class_count_, 0.0);

        std::vector<double> raw_scores (n * class_count_, 0.0);
        std::vector<double> grads (n, 0.0);
        std::vector<double> hesses (n, 1.0);

        for (std::size_t stage = 0; stage < stage_count_; ++stage)
          {
            std::vector<double> probs (n * class_count_, 0.0);
            for (std::size_t i = 0; i < n; ++i)
              {
                double max_s = raw_scores[i * class_count_];
                for (std::size_t k = 1; k < class_count_; ++k)
                  {
                    max_s = std::max (max_s, raw_scores[i * class_count_ + k]);
                  }
                double sum_exp = 0.0;
                for (std::size_t k = 0; k < class_count_; ++k)
                  {
                    double e = std::exp (
                      std::clamp (raw_scores[i * class_count_ + k] - max_s,
                                  -25.0,
                                  25.0));
                    probs[i * class_count_ + k] = e;
                    sum_exp += e;
                  }
                for (std::size_t k = 0; k < class_count_; ++k)
                  {
                    probs[i * class_count_ + k] /= sum_exp;
                  }
              }

            for (std::size_t k = 0; k < class_count_; ++k)
              {
                for (std::size_t i = 0; i < n; ++i)
                  {
                    double ind =
                      (static_cast<std::size_t> (std::lround (y_vec[i])) == k)
                        ? 1.0
                        : 0.0;
                    derivative_pair dp = loss_.compute_derivatives (
                      ind, probs[i * class_count_ + k]);
                    grads[i] = dp.gradient;
                    hesses[i] = dp.curvature;
                  }

                additive_stage_tree tree = build_single_stage_tree (
                  work_table, qbins, grads, hesses, stage * class_count_ + k);
                for (std::size_t i = 0; i < n; ++i)
                  {
                    raw_scores[i * class_count_ + k] +=
                      step_rate_ * tree.evaluate_row (work_table.row_span (i));
                  }
                trees_.push_back (std::move (tree));
              }
          }
      }
    else
      {
        class_count_ = Loss_Type::is_classification ? 2 : 1;
        double init_score = loss_.base_score (y_vec);
        base_scores_.assign (1, init_score);

        std::vector<double> raw_scores (n, init_score);
        std::vector<double> grads (n, 0.0);
        std::vector<double> hesses (n, 1.0);

        for (std::size_t stage = 0; stage < stage_count_; ++stage)
          {
            for (std::size_t i = 0; i < n; ++i)
              {
                derivative_pair dp =
                  loss_.compute_derivatives (y_vec[i], raw_scores[i]);
                grads[i] = dp.gradient;
                hesses[i] = dp.curvature;
              }

            additive_stage_tree tree = build_single_stage_tree (
              work_table, qbins, grads, hesses, stage);
            for (std::size_t i = 0; i < n; ++i)
              {
                raw_scores[i] +=
                  step_rate_ * tree.evaluate_row (work_table.row_span (i));
              }
            trees_.push_back (std::move (tree));
          }
      }

    finalise_feature_importances ();
    best_stage_ = stage_count_ > 0 ? (stage_count_ - 1) : 0;
    calibrated_ = true;
  }

  // Calibrate with validation monitoring and early stopping.
  template <tabular_source TrainData,
            target_sequence TrainTargets,
            tabular_source ValData,
            target_sequence ValTargets>
  void
  calibrate_with_monitor (const TrainData &train_data,
                          const TrainTargets &train_targets,
                          const ValData &val_data,
                          const ValTargets &val_targets,
                          std::size_t patience = 5)
  {
    std::size_t n = train_data.row_count ();
    std::size_t d = train_data.column_count ();
    std::size_t n_val = val_data.row_count ();
    if (n_val == 0 || val_targets.size () < n_val)
      {
        calibrate (train_data, train_targets);
        return;
      }

    feature_count_ = d;
    trees_.clear ();
    base_scores_.clear ();
    raw_gain_importances_.assign (d, 0.0);
    feature_importances_.assign (d, 0.0);
    split_counts_.assign (d, 0);
    best_stage_ = 0;

    if (n == 0 || d == 0)
      {
        calibrated_ = false;
        return;
      }
    if (train_targets.size () != n)
      {
        throw std::invalid_argument (
          "bellows::stagewise_booster::calibrate_with_monitor: target size");
      }

    std::vector<double> y_vec (n);
    for (std::size_t i = 0; i < n; ++i)
      {
        y_vec[i] = train_targets[i];
      }

    data_table work_table = prepare_ordered_categories (train_data, y_vec);
    quantile_bins qbins;
    if constexpr (Cut_Policy::uses_quantile_bins)
      {
        qbins = build_quantile_bins (work_table);
      }

    data_table val_work (n_val, d);
    for (std::size_t i = 0; i < n_val; ++i)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            val_work.at (i, c) = map_nominal_cell (c, val_data.at (i, c));
          }
      }

    double best_val_loss = std::numeric_limits<double>::infinity ();
    std::size_t rounds_without_improvement = 0;

    if constexpr (!Loss_Type::is_multiclass)
      {
        class_count_ = Loss_Type::is_classification ? 2 : 1;
        double init_score = loss_.base_score (y_vec);
        base_scores_.assign (1, init_score);

        std::vector<double> raw_scores (n, init_score);
        std::vector<double> val_scores (n_val, init_score);
        std::vector<double> grads (n, 0.0);
        std::vector<double> hesses (n, 1.0);
        std::vector<double> best_gains (d, 0.0);
        std::vector<std::size_t> best_counts (d, 0);

        for (std::size_t stage = 0; stage < stage_count_; ++stage)
          {
            for (std::size_t i = 0; i < n; ++i)
              {
                derivative_pair dp =
                  loss_.compute_derivatives (y_vec[i], raw_scores[i]);
                grads[i] = dp.gradient;
                hesses[i] = dp.curvature;
              }

            additive_stage_tree tree = build_single_stage_tree (
              work_table, qbins, grads, hesses, stage);
            for (std::size_t i = 0; i < n; ++i)
              {
                raw_scores[i] +=
                  step_rate_ * tree.evaluate_row (work_table.row_span (i));
              }

            double val_loss = 0.0;
            for (std::size_t i = 0; i < n_val; ++i)
              {
                val_scores[i] +=
                  step_rate_ * tree.evaluate_row (val_work.row_span (i));
                double pred = loss_.map_prediction (val_scores[i]);
                double diff = static_cast<double> (val_targets[i]) - pred;
                val_loss += diff * diff;
              }
            val_loss /= static_cast<double> (n_val);
            trees_.push_back (std::move (tree));

            if (val_loss + 1.0e-12 < best_val_loss)
              {
                best_val_loss = val_loss;
                best_stage_ = stage;
                best_gains = raw_gain_importances_;
                best_counts = split_counts_;
                rounds_without_improvement = 0;
              }
            else
              {
                ++rounds_without_improvement;
                if (rounds_without_improvement >= patience)
                  {
                    trees_.resize (best_stage_ + 1);
                    raw_gain_importances_ = std::move (best_gains);
                    split_counts_ = std::move (best_counts);
                    break;
                  }
              }
          }
      }
    else
      {
        calibrate (train_data, train_targets);
        return;
      }

    finalise_feature_importances ();
    calibrated_ = true;
  }

  double
  evaluate_one (std::span<const double> raw_row) const
  {
    if (!calibrated_)
      {
        return 0.0;
      }

    std::vector<double> mapped_buf;
    std::span<const double> row = raw_row;
    if (has_nominal_encodings_ || raw_row.size () < feature_count_)
      {
        mapped_buf.assign (feature_count_, missing_value ());
        std::size_t limit = std::min (raw_row.size (), feature_count_);
        for (std::size_t c = 0; c < limit; ++c)
          {
            mapped_buf[c] = map_nominal_cell (c, raw_row[c]);
          }
        row = std::span<const double> (mapped_buf.data (), feature_count_);
      }

    if constexpr (Loss_Type::is_multiclass)
      {
        std::vector<double> scores = base_scores_;
        std::size_t stages = trees_.size () / class_count_;
        for (std::size_t s = 0; s < stages; ++s)
          {
            for (std::size_t k = 0; k < class_count_; ++k)
              {
                scores[k] +=
                  step_rate_
                  * trees_[s * class_count_ + k].evaluate_row (row);
              }
          }
        auto best_it = std::max_element (scores.begin (), scores.end ());
        return static_cast<double> (std::distance (scores.begin (), best_it));
      }
    else
      {
        double raw = base_scores_.empty () ? 0.0 : base_scores_[0];
        for (const auto &tree : trees_)
          {
            raw += step_rate_ * tree.evaluate_row (row);
          }
        return loss_.map_prediction (raw);
      }
  }

  template <tabular_source Data>
  std::vector<double>
  evaluate (const Data &data) const
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    std::vector<double> out (n, 0.0);
    std::vector<double> row_buf (d, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            row_buf[c] = data.at (i, c);
          }
        out[i] = evaluate_one (row_buf);
      }
    return out;
  }

  template <tabular_source Data>
  data_table
  evaluate_beliefs (const Data &data) const
    requires (Loss_Type::is_classification)
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    std::size_t k_cols = std::max<std::size_t> (2, class_count_);
    data_table beliefs (n, k_cols, 1.0 / static_cast<double> (k_cols));
    if (!calibrated_ || d < feature_count_)
      {
        return beliefs;
      }
    std::vector<double> row_buf (d, 0.0);

    for (std::size_t i = 0; i < n; ++i)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            row_buf[c] = map_nominal_cell (c, data.at (i, c));
          }
        std::span<const double> row (row_buf.data (), d);

        if constexpr (Loss_Type::is_multiclass)
          {
            std::vector<double> scores = base_scores_;
            std::size_t stages = trees_.size () / class_count_;
            for (std::size_t s = 0; s < stages; ++s)
              {
                for (std::size_t k = 0; k < class_count_; ++k)
                  {
                    scores[k] +=
                      step_rate_
                      * trees_[s * class_count_ + k].evaluate_row (row);
                  }
              }
            double max_s = *std::max_element (scores.begin (), scores.end ());
            double sum_exp = 0.0;
            for (std::size_t k = 0; k < class_count_; ++k)
              {
                double e =
                  std::exp (std::clamp (scores[k] - max_s, -25.0, 25.0));
                beliefs.at (i, k) = e;
                sum_exp += e;
              }
            for (std::size_t k = 0; k < class_count_; ++k)
              {
                beliefs.at (i, k) /= sum_exp;
              }
          }
        else
          {
            double raw = base_scores_.empty () ? 0.0 : base_scores_[0];
            for (const auto &tree : trees_)
              {
                raw += step_rate_ * tree.evaluate_row (row);
              }
            double p1 = loss_.map_probability (raw);
            beliefs.at (i, 0) = 1.0 - p1;
            beliefs.at (i, 1) = p1;
          }
      }
    return beliefs;
  }

  // Project observations into stagewise leaf-index embeddings (transformer role).
  template <tabular_source Data>
  data_table
  project (const Data &data) const
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    std::size_t m = trees_.size ();
    data_table embedding (n, m);
    for (std::size_t t = 0; t < m; ++t)
      {
        embedding.set_scale (t, feature_scale::nominal);
        embedding.set_column_name (t, "stage_leaf_" + std::to_string (t));
      }

    std::vector<double> row_buf (d, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            row_buf[c] = map_nominal_cell (c, data.at (i, c));
          }
        std::span<const double> row (row_buf.data (), d);
        for (std::size_t t = 0; t < m; ++t)
          {
            embedding.at (i, t) = trees_[t].leaf_index_for_row (row);
          }
      }
    return embedding;
  }

  void
  write_state (std::ostream &out) const
  {
    auto write_pod = [&] (const auto &val)
    {
      out.write (reinterpret_cast<const char *> (&val), sizeof (val));
    };

    std::uint8_t cal = calibrated_ ? 1 : 0;
    write_pod (cal);
    write_pod (feature_count_);
    write_pod (class_count_);
    write_pod (step_rate_);
    write_pod (stage_count_);

    std::size_t nb = base_scores_.size ();
    write_pod (nb);
    for (double b : base_scores_)
      {
        write_pod (b);
      }

    std::uint8_t has_nom = has_nominal_encodings_ ? 1 : 0;
    write_pod (has_nom);
    std::size_t n_cols = is_nominal_col_.size ();
    write_pod (n_cols);
    for (std::size_t c = 0; c < n_cols; ++c)
      {
        std::uint8_t flag = is_nominal_col_[c] ? 1 : 0;
        write_pod (flag);
        write_pod (nominal_priors_[c]);
        std::size_t map_sz = nominal_maps_[c].size ();
        write_pod (map_sz);
        for (double v : nominal_maps_[c])
          {
            write_pod (v);
          }
      }

    std::size_t nt = trees_.size ();
    write_pod (nt);
    for (const auto &tree : trees_)
      {
        std::uint8_t ob = tree.is_oblivious ? 1 : 0;
        write_pod (ob);
        std::size_t nn = tree.nodes.size ();
        write_pod (nn);
        for (const auto &node : tree.nodes)
          {
            write_pod (node.feature_index);
            write_pod (node.threshold);
            std::uint8_t mr = node.missing_goes_right ? 1 : 0;
            write_pod (mr);
            write_pod (node.left_child);
            write_pod (node.right_child);
            write_pod (node.leaf_value);
          }
        std::size_t od = tree.oblivious_features.size ();
        write_pod (od);
        for (std::size_t d = 0; d < od; ++d)
          {
            write_pod (tree.oblivious_features[d]);
            write_pod (tree.oblivious_thresholds[d]);
            write_pod (tree.oblivious_missing_right[d]);
          }
        std::size_t ol = tree.oblivious_leaves.size ();
        write_pod (ol);
        for (double lv : tree.oblivious_leaves)
          {
            write_pod (lv);
          }
      }

    write_pod (best_stage_);
    std::size_t n_imp = feature_importances_.size ();
    write_pod (n_imp);
    for (double imp : feature_importances_)
      {
        write_pod (imp);
      }
    std::size_t n_sc = split_counts_.size ();
    write_pod (n_sc);
    for (std::size_t sc : split_counts_)
      {
        write_pod (sc);
      }
  }

  void
  read_state (std::istream &in)
  {
    auto read_pod = [&] (auto &val)
    {
      in.read (reinterpret_cast<char *> (&val), sizeof (val));
    };

    std::uint8_t cal = 0;
    read_pod (cal);
    calibrated_ = (cal != 0);
    read_pod (feature_count_);
    read_pod (class_count_);
    read_pod (step_rate_);
    read_pod (stage_count_);

    std::size_t nb = 0;
    read_pod (nb);
    base_scores_.resize (nb);
    for (std::size_t i = 0; i < nb; ++i)
      {
        read_pod (base_scores_[i]);
      }

    std::uint8_t has_nom = 0;
    read_pod (has_nom);
    has_nominal_encodings_ = (has_nom != 0);
    std::size_t n_cols = 0;
    read_pod (n_cols);
    is_nominal_col_.assign (n_cols, false);
    nominal_priors_.assign (n_cols, 0.0);
    nominal_maps_.assign (n_cols, {});
    for (std::size_t c = 0; c < n_cols; ++c)
      {
        std::uint8_t flag = 0;
        read_pod (flag);
        is_nominal_col_[c] = (flag != 0);
        read_pod (nominal_priors_[c]);
        std::size_t map_sz = 0;
        read_pod (map_sz);
        nominal_maps_[c].resize (map_sz);
        for (std::size_t k = 0; k < map_sz; ++k)
          {
            read_pod (nominal_maps_[c][k]);
          }
      }

    std::size_t nt = 0;
    read_pod (nt);
    trees_.resize (nt);
    for (std::size_t t = 0; t < nt; ++t)
      {
        auto &tree = trees_[t];
        std::uint8_t ob = 0;
        read_pod (ob);
        tree.is_oblivious = (ob != 0);
        std::size_t nn = 0;
        read_pod (nn);
        tree.nodes.resize (nn);
        for (std::size_t j = 0; j < nn; ++j)
          {
            auto &node = tree.nodes[j];
            read_pod (node.feature_index);
            read_pod (node.threshold);
            std::uint8_t mr = 0;
            read_pod (mr);
            node.missing_goes_right = (mr != 0);
            read_pod (node.left_child);
            read_pod (node.right_child);
            read_pod (node.leaf_value);
          }
        std::size_t od = 0;
        read_pod (od);
        tree.oblivious_features.resize (od);
        tree.oblivious_thresholds.resize (od);
        tree.oblivious_missing_right.resize (od);
        for (std::size_t d = 0; d < od; ++d)
          {
            read_pod (tree.oblivious_features[d]);
            read_pod (tree.oblivious_thresholds[d]);
            read_pod (tree.oblivious_missing_right[d]);
          }
        std::size_t ol = 0;
        read_pod (ol);
        tree.oblivious_leaves.resize (ol);
        for (std::size_t k = 0; k < ol; ++k)
          {
            read_pod (tree.oblivious_leaves[k]);
          }
      }

    read_pod (best_stage_);
    std::size_t n_imp = 0;
    read_pod (n_imp);
    feature_importances_.resize (n_imp, 0.0);
    for (std::size_t i = 0; i < n_imp; ++i)
      {
        read_pod (feature_importances_[i]);
      }
    std::size_t n_sc = 0;
    read_pod (n_sc);
    split_counts_.resize (n_sc, 0);
    for (std::size_t i = 0; i < n_sc; ++i)
      {
        read_pod (split_counts_[i]);
      }
  }

private:
  struct split_proposal
  {
    bool valid = false;
    std::int32_t feature = -1;
    double threshold = 0.0;
    bool missing_goes_right = false;
    double gain = -1.0;
    double left_grad = 0.0;
    double left_hess = 0.0;
    double right_grad = 0.0;
    double right_hess = 0.0;
  };

  struct hist_bin
  {
    double grad = 0.0;
    double hess = 0.0;
    std::size_t count = 0;
  };

  struct node_histograms
  {
    bool ready = false;
    std::vector<std::vector<hist_bin>> feature_bins;
    std::vector<hist_bin> missing_bins;
  };

  struct quantile_bins
  {
    std::size_t rows = 0;
    std::size_t cols = 0;
    static constexpr std::uint16_t missing_bin_id = 0xFFFF;
    std::vector<std::vector<double>> upper_bounds;
    std::vector<std::uint16_t> binned_cells;
  };

  template <tabular_source Data>
  data_table
  prepare_ordered_categories (const Data &data,
                              const std::vector<double> &targets)
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    data_table result (n, d);
    is_nominal_col_.assign (d, false);
    nominal_priors_.assign (d, 0.0);
    nominal_maps_.assign (d, {});
    has_nominal_encodings_ = false;

    double global_mean = 0.0;
    for (double y : targets)
      {
        global_mean += y;
      }
    global_mean /= static_cast<double> (std::max<std::size_t> (1, n));

    for (std::size_t c = 0; c < d; ++c)
      {
        result.set_column_name (c, "col_" + std::to_string (c));
        if (ordered_cats_.enabled && data.scale (c) == feature_scale::nominal)
          {
            has_nominal_encodings_ = true;
            is_nominal_col_[c] = true;
            nominal_priors_[c] = global_mean;
            std::size_t card = std::max<std::size_t> (2, data.cardinality (c));
            double m_weight = std::max (0.5, ordered_cats_.prior_weight);

            constexpr std::size_t num_perms = 4;
            std::vector<double> accum_stat (n, 0.0);
            std::vector<double> final_sum_y (card, 0.0);
            std::vector<double> final_cnt_y (card, 0.0);

            for (std::size_t p = 0; p < num_perms; ++p)
              {
                hyper_math::random_engine perm_rng (
                  seed_ + (c + 1) * 1009ULL + p * 7919ULL);
                std::vector<std::size_t> p_order = perm_rng.permutation (n);
                std::vector<double> sum_y (card, 0.0);
                std::vector<double> cnt_y (card, 0.0);

                for (std::size_t idx = 0; idx < n; ++idx)
                  {
                    std::size_t r = p_order[idx];
                    double raw_v = data.at (r, c);
                    if (!is_missing (raw_v) && raw_v >= 0.0)
                      {
                        std::size_t cat = static_cast<std::size_t> (raw_v);
                        if (cat >= card)
                          {
                            card = cat + 1;
                            sum_y.resize (card, 0.0);
                            cnt_y.resize (card, 0.0);
                            final_sum_y.resize (card, 0.0);
                            final_cnt_y.resize (card, 0.0);
                          }
                        double stat =
                          (sum_y[cat] + m_weight * global_mean)
                          / (cnt_y[cat] + m_weight);
                        accum_stat[r] += stat;
                        sum_y[cat] += targets[r];
                        cnt_y[cat] += 1.0;
                      }
                  }
                if (p == 0)
                  {
                    final_sum_y = sum_y;
                    final_cnt_y = cnt_y;
                  }
              }

            for (std::size_t r = 0; r < n; ++r)
              {
                double raw_v = data.at (r, c);
                if (is_missing (raw_v) || raw_v < 0.0)
                  {
                    result.at (r, c) = missing_value ();
                  }
                else
                  {
                    result.at (r, c) =
                      accum_stat[r] / static_cast<double> (num_perms);
                  }
              }

            nominal_maps_[c].resize (card, global_mean);
            for (std::size_t cat = 0; cat < card; ++cat)
              {
                nominal_maps_[c][cat] =
                  (final_sum_y[cat] + m_weight * global_mean)
                  / (final_cnt_y[cat] + m_weight);
              }
            result.set_scale (c, feature_scale::real);
          }
        else
          {
            result.set_scale (c, data.scale (c), data.cardinality (c));
            for (std::size_t r = 0; r < n; ++r)
              {
                result.at (r, c) = data.at (r, c);
              }
          }
      }
    return result;
  }

  double
  map_nominal_cell (std::size_t col, double val) const noexcept
  {
    if (!has_nominal_encodings_ || col >= is_nominal_col_.size ()
        || !is_nominal_col_[col])
      {
        return val;
      }
    if (is_missing (val) || val < 0.0)
      {
        return missing_value ();
      }
    std::size_t cat = static_cast<std::size_t> (val);
    if (cat < nominal_maps_[col].size ())
      {
        return nominal_maps_[col][cat];
      }
    return nominal_priors_[col];
  }

  quantile_bins
  build_quantile_bins (const data_table &table) const
  {
    quantile_bins qb;
    qb.rows = table.row_count ();
    qb.cols = table.column_count ();
    qb.upper_bounds.resize (qb.cols);
    qb.binned_cells.assign (qb.rows * qb.cols, quantile_bins::missing_bin_id);

    std::size_t max_b = std::clamp<std::size_t> (limits_.max_bins, 4, 255);

    for (std::size_t c = 0; c < qb.cols; ++c)
      {
        std::vector<double> vals;
        vals.reserve (qb.rows);
        for (std::size_t r = 0; r < qb.rows; ++r)
          {
            double v = table.at (r, c);
            if (!is_missing (v))
              {
                vals.push_back (v);
              }
          }
        std::sort (vals.begin (), vals.end ());
        std::vector<double> unique_vals;
        unique_vals.reserve (vals.size ());
        for (double v : vals)
          {
            if (unique_vals.empty () || v > unique_vals.back () + 1.0e-12)
              {
                unique_vals.push_back (v);
              }
          }

        std::vector<double> edges;
        if (unique_vals.size () >= 2)
          {
            if (unique_vals.size () <= max_b)
              {
                for (std::size_t i = 0; i + 1 < unique_vals.size (); ++i)
                  {
                    edges.push_back (
                      0.5 * (unique_vals[i] + unique_vals[i + 1]));
                  }
              }
            else
              {
                for (std::size_t b = 1; b < max_b; ++b)
                  {
                    std::size_t pos = (b * vals.size ()) / max_b;
                    if (pos < vals.size ())
                      {
                        double candidate = vals[pos];
                        if (candidate < unique_vals.back () - 1.0e-12
                            && (edges.empty ()
                                || candidate > edges.back () + 1.0e-12))
                          {
                            edges.push_back (candidate);
                          }
                      }
                  }
                if (edges.empty ())
                  {
                    edges.push_back (
                      0.5 * (unique_vals.front () + unique_vals.back ()));
                  }
              }
            edges.push_back (unique_vals.back () + 1.0);
          }
        else if (!unique_vals.empty ())
          {
            edges.push_back (unique_vals.back () + 1.0);
          }
        else
          {
            edges.push_back (0.0);
          }
        qb.upper_bounds[c] = std::move (edges);

        const auto &bounds = qb.upper_bounds[c];
        for (std::size_t r = 0; r < qb.rows; ++r)
          {
            double v = table.at (r, c);
            if (is_missing (v))
              {
                qb.binned_cells[r * qb.cols + c] =
                  quantile_bins::missing_bin_id;
              }
            else
              {
                auto it = std::lower_bound (bounds.begin (), bounds.end (), v);
                std::size_t idx = static_cast<std::size_t> (
                  std::distance (bounds.begin (), it));
                if (idx >= bounds.size ())
                  {
                    idx = bounds.size () - 1;
                  }
                qb.binned_cells[r * qb.cols + c] =
                  static_cast<std::uint16_t> (idx);
              }
          }
      }
    return qb;
  }

  void
  select_stage_rows (const std::vector<double> &grads,
                      const std::vector<double> &hesses,
                      std::size_t stage_seed,
                      std::vector<std::size_t> &active_rows,
                      std::vector<double> &work_grads,
                      std::vector<double> &work_hesses) const
  {
    std::size_t n = grads.size ();
    work_grads = grads;
    work_hesses = hesses;

    if constexpr (!Sampling_Policy::filters_by_gradient)
      {
        active_rows.resize (n);
        std::iota (active_rows.begin (), active_rows.end (), 0);
        return;
      }
    else
      {
        std::vector<std::size_t> order (n);
        std::iota (order.begin (), order.end (), 0);
        std::sort (order.begin (), order.end (),
                    [&] (std::size_t a, std::size_t b)
                    {
                      return std::abs (grads[a]) > std::abs (grads[b]);
                    });

        double a_ratio = std::clamp (grad_sampling_.top_ratio, 0.05, 1.0);
        double b_ratio =
          std::clamp (grad_sampling_.tail_ratio, 0.05, 1.0 - a_ratio + 1.0e-9);
        std::size_t top_n = std::max<std::size_t> (
          1, static_cast<std::size_t> (a_ratio * static_cast<double> (n)));
        std::size_t tail_n = std::min<std::size_t> (
          n - top_n,
          std::max<std::size_t> (
            1, static_cast<std::size_t> (b_ratio * static_cast<double> (n))));

        active_rows.clear ();
        active_rows.reserve (top_n + tail_n);
        for (std::size_t i = 0; i < top_n; ++i)
          {
            active_rows.push_back (order[i]);
          }

        if (top_n < n && tail_n > 0)
          {
            hyper_math::random_engine rng (
              seed_ + stage_seed * 1315423911ULL);
            std::vector<std::size_t> tail_perm = rng.permutation (n - top_n);
            double weight = (1.0 - a_ratio) / b_ratio;
            for (std::size_t i = 0; i < tail_n; ++i)
              {
                std::size_t r = order[top_n + tail_perm[i]];
                active_rows.push_back (r);
                work_grads[r] *= weight;
                work_hesses[r] *= weight;
              }
          }
      }
  }

  node_histograms
  compute_node_histograms (const quantile_bins &qbins,
                            const std::vector<std::size_t> &rows,
                            const std::vector<double> &grads,
                            const std::vector<double> &hesses) const
  {
    node_histograms nh;
    nh.ready = true;
    nh.feature_bins.resize (qbins.cols);
    nh.missing_bins.resize (qbins.cols);
    for (std::size_t c = 0; c < qbins.cols; ++c)
      {
        nh.feature_bins[c].resize (qbins.upper_bounds[c].size ());
      }

    for (std::size_t r : rows)
      {
        double g = grads[r];
        double h = hesses[r];
        const std::uint16_t *row_bins =
          qbins.binned_cells.data () + r * qbins.cols;
        for (std::size_t c = 0; c < qbins.cols; ++c)
          {
            std::uint16_t b = row_bins[c];
            if (b == quantile_bins::missing_bin_id)
              {
                nh.missing_bins[c].grad += g;
                nh.missing_bins[c].hess += h;
                nh.missing_bins[c].count += 1;
              }
            else
              {
                auto &slot = nh.feature_bins[c][b];
                slot.grad += g;
                slot.hess += h;
                slot.count += 1;
              }
          }
      }
    return nh;
  }

  // Sibling histogram subtraction: Hist_large = Hist_parent - Hist_small.
  node_histograms
  subtract_node_histograms (const node_histograms &parent,
                            const node_histograms &small_sibling) const
  {
    node_histograms large = parent;
    for (std::size_t c = 0; c < large.feature_bins.size (); ++c)
      {
        large.missing_bins[c].grad -= small_sibling.missing_bins[c].grad;
        large.missing_bins[c].hess -= small_sibling.missing_bins[c].hess;
        large.missing_bins[c].count -= small_sibling.missing_bins[c].count;
        for (std::size_t b = 0; b < large.feature_bins[c].size (); ++b)
          {
            large.feature_bins[c][b].grad -=
              small_sibling.feature_bins[c][b].grad;
            large.feature_bins[c][b].hess -=
              small_sibling.feature_bins[c][b].hess;
            large.feature_bins[c][b].count -=
              small_sibling.feature_bins[c][b].count;
          }
      }
    return large;
  }

  int
  feature_monotonic_dir (std::size_t col) const noexcept
  {
    if (col < monotonic_.directions.size ())
      {
        return monotonic_.directions[col];
      }
    return 0;
  }

  bool
  satisfies_monotonic_constraint (std::size_t col,
                                  double gl,
                                  double hl,
                                  double gr,
                                  double hr,
                                  double min_bound,
                                  double max_bound) const noexcept
  {
    int dir = feature_monotonic_dir (col);
    if (dir == 0)
      {
        return true;
      }
    double wl = std::clamp (
      newton_regulariser::leaf_weight (gl, hl, reg_.l1, reg_.l2),
      min_bound,
      max_bound);
    double wr = std::clamp (
      newton_regulariser::leaf_weight (gr, hr, reg_.l1, reg_.l2),
      min_bound,
      max_bound);
    return dir > 0 ? (wl <= wr) : (wl >= wr);
  }

  void
  finalise_feature_importances ()
  {
    double sum_gain = 0.0;
    for (double g : raw_gain_importances_)
      {
        sum_gain += std::max (0.0, g);
      }
    feature_importances_.assign (feature_count_, 0.0);
    if (sum_gain > 1.0e-15)
      {
        for (std::size_t c = 0; c < feature_count_; ++c)
          {
            feature_importances_[c] =
              std::max (0.0, raw_gain_importances_[c]) / sum_gain;
          }
      }
  }

  split_proposal
  propose_best_split_from_histograms (const quantile_bins &qbins,
                                      const node_histograms &nh,
                                      double total_grad,
                                      double total_hess,
                                      std::size_t total_count,
                                      double min_bound,
                                      double max_bound) const
  {
    split_proposal best;
    if (total_count < 2 * limits_.min_samples_leaf)
      {
        return best;
      }

    for (std::size_t c = 0; c < qbins.cols; ++c)
      {
        const auto &bins = nh.feature_bins[c];
        const auto &bounds = qbins.upper_bounds[c];
        const auto &miss = nh.missing_bins[c];

        double prefix_g = 0.0;
        double prefix_h = 0.0;
        std::size_t prefix_n = 0;

        for (std::size_t b = 0; b + 1 < bins.size (); ++b)
          {
            prefix_g += bins[b].grad;
            prefix_h += bins[b].hess;
            prefix_n += bins[b].count;
            if (prefix_n == 0)
              {
                continue;
              }

            for (int miss_right = 0; miss_right <= 1; ++miss_right)
              {
                if (miss.count == 0 && miss_right == 1)
                  {
                    continue;
                  }
                double gl = prefix_g + (miss_right == 0 ? miss.grad : 0.0);
                double hl = prefix_h + (miss_right == 0 ? miss.hess : 0.0);
                std::size_t nl =
                  prefix_n + (miss_right == 0 ? miss.count : 0);

                if (nl < limits_.min_samples_leaf
                    || (total_count - nl) < limits_.min_samples_leaf)
                  {
                    continue;
                  }
                double gr = total_grad - gl;
                double hr = total_hess - hl;
                if (!satisfies_monotonic_constraint (
                      c, gl, hl, gr, hr, min_bound, max_bound))
                  {
                    continue;
                  }
                double gain = newton_regulariser::split_gain (
                  gl, hl, gr, hr, reg_.l1, reg_.l2, reg_.complexity);
                if (gain > best.gain && gain > 1.0e-12)
                  {
                    best.valid = true;
                    best.feature = static_cast<std::int32_t> (c);
                    best.threshold = bounds[b];
                    best.missing_goes_right = (miss_right == 1);
                    best.gain = gain;
                    best.left_grad = gl;
                    best.left_hess = hl;
                    best.right_grad = gr;
                    best.right_hess = hr;
                  }
              }
          }
      }
    return best;
  }

  split_proposal
  propose_best_split_exact (const data_table &table,
                            const std::vector<std::size_t> &rows,
                            const std::vector<double> &grads,
                            const std::vector<double> &hesses,
                            double total_grad,
                            double total_hess,
                            double min_bound,
                            double max_bound) const
  {
    split_proposal best;
    std::size_t total_count = rows.size ();
    if (total_count < 2 * limits_.min_samples_leaf)
      {
        return best;
      }

    std::size_t cols = table.column_count ();
    std::vector<std::size_t> present;
    present.reserve (total_count);

    for (std::size_t c = 0; c < cols; ++c)
      {
        present.clear ();
        double miss_g = 0.0;
        double miss_h = 0.0;
        std::size_t miss_n = 0;

        for (std::size_t r : rows)
          {
            double v = table.at (r, c);
            if (is_missing (v))
              {
                miss_g += grads[r];
                miss_h += hesses[r];
                miss_n += 1;
              }
            else
              {
                present.push_back (r);
              }
          }
        if (present.size () < 2)
          {
            continue;
          }

        std::sort (present.begin (), present.end (),
                    [&] (std::size_t a, std::size_t b)
                    {
                      return table.at (a, c) < table.at (b, c);
                    });

        double prefix_g = 0.0;
        double prefix_h = 0.0;
        for (std::size_t i = 0; i + 1 < present.size (); ++i)
          {
            std::size_t r = present[i];
            prefix_g += grads[r];
            prefix_h += hesses[r];
            double v_curr = table.at (r, c);
            double v_next = table.at (present[i + 1], c);
            if (v_next <= v_curr + 1.0e-12)
              {
                continue;
              }
            double mid = 0.5 * (v_curr + v_next);

            for (int miss_right = 0; miss_right <= 1; ++miss_right)
              {
                if (miss_n == 0 && miss_right == 1)
                  {
                    continue;
                  }
                double gl = prefix_g + (miss_right == 0 ? miss_g : 0.0);
                double hl = prefix_h + (miss_right == 0 ? miss_h : 0.0);
                std::size_t nl = (i + 1) + (miss_right == 0 ? miss_n : 0);
                if (nl < limits_.min_samples_leaf
                    || (total_count - nl) < limits_.min_samples_leaf)
                  {
                    continue;
                  }
                double gr = total_grad - gl;
                double hr = total_hess - hl;
                if (!satisfies_monotonic_constraint (
                      c, gl, hl, gr, hr, min_bound, max_bound))
                  {
                    continue;
                  }
                double gain = newton_regulariser::split_gain (
                  gl, hl, gr, hr, reg_.l1, reg_.l2, reg_.complexity);
                if (gain > best.gain && gain > 1.0e-12)
                  {
                    best.valid = true;
                    best.feature = static_cast<std::int32_t> (c);
                    best.threshold = mid;
                    best.missing_goes_right = (miss_right == 1);
                    best.gain = gain;
                    best.left_grad = gl;
                    best.left_hess = hl;
                    best.right_grad = gr;
                    best.right_hess = hr;
                  }
              }
          }
      }
    return best;
  }

  additive_stage_tree
  build_oblivious_tree (const data_table &table,
                        const quantile_bins &qbins,
                        const std::vector<std::size_t> &active_rows,
                        const std::vector<double> &grads,
                        const std::vector<double> &hesses)
  {
    additive_stage_tree tree;
    tree.is_oblivious = true;
    std::size_t depth = std::clamp<std::size_t> (limits_.max_depth, 1, 8);
    std::size_t cols = table.column_count ();
    std::size_t n_active = active_rows.size ();

    std::vector<std::size_t> bucket_of_row (n_active, 0);

    for (std::size_t d = 0; d < depth; ++d)
      {
        std::size_t num_buckets = std::size_t{1} << d;
        double best_level_gain = -1.0;
        std::int32_t best_f = 0;
        double best_thr = 0.0;
        bool best_miss_right = false;

        std::vector<double> b_tot_g (num_buckets, 0.0);
        std::vector<double> b_tot_h (num_buckets, 0.0);
        std::vector<std::size_t> b_tot_n (num_buckets, 0);
        for (std::size_t idx = 0; idx < n_active; ++idx)
          {
            std::size_t r = active_rows[idx];
            std::size_t b = bucket_of_row[idx];
            b_tot_g[b] += grads[r];
            b_tot_h[b] += hesses[r];
            b_tot_n[b] += 1;
          }

        for (std::size_t c = 0; c < cols; ++c)
          {
            std::vector<double> candidates;
            if constexpr (Cut_Policy::uses_quantile_bins)
              {
                for (std::size_t b = 0;
                      b + 1 < qbins.upper_bounds[c].size ();
                      ++b)
                  {
                    candidates.push_back (qbins.upper_bounds[c][b]);
                  }
              }
            else
              {
                std::vector<double> vals;
                vals.reserve (n_active);
                for (std::size_t r : active_rows)
                  {
                    double v = table.at (r, c);
                    if (!is_missing (v))
                      {
                        vals.push_back (v);
                      }
                  }
                std::sort (vals.begin (), vals.end ());
                vals.erase (std::unique (vals.begin (),
                                          vals.end (),
                                          [] (double a, double b)
                                          {
                                            return std::abs (a - b) <= 1.0e-12;
                                          }),
                            vals.end ());
                if (vals.size () >= 2)
                  {
                    std::size_t step =
                      std::max<std::size_t> (1, (vals.size () - 1) / 32);
                    for (std::size_t i = 0; i + 1 < vals.size (); i += step)
                      {
                        candidates.push_back (0.5 * (vals[i] + vals[i + 1]));
                      }
                  }
              }

            for (double thr : candidates)
              {
                for (int miss_right = 0; miss_right <= 1; ++miss_right)
                  {
                    std::vector<double> gl (num_buckets, 0.0);
                    std::vector<double> hl (num_buckets, 0.0);
                    std::vector<std::size_t> nl (num_buckets, 0);

                    for (std::size_t idx = 0; idx < n_active; ++idx)
                      {
                        std::size_t r = active_rows[idx];
                        std::size_t b = bucket_of_row[idx];
                        double v = table.at (r, c);
                        bool go_right = is_missing (v) ? (miss_right == 1)
                                                        : (v > thr);
                        if (!go_right)
                          {
                            gl[b] += grads[r];
                            hl[b] += hesses[r];
                            nl[b] += 1;
                          }
                      }

                    double level_gain = 0.0;
                    bool any_valid = false;
                    bool mono_ok = true;
                    for (std::size_t b = 0; b < num_buckets; ++b)
                      {
                        std::size_t nr = b_tot_n[b] - nl[b];
                        if (nl[b] >= limits_.min_samples_leaf
                            && nr >= limits_.min_samples_leaf)
                          {
                            any_valid = true;
                          }
                        double gr = b_tot_g[b] - gl[b];
                        double hr = b_tot_h[b] - hl[b];
                        if (nl[b] > 0 && nr > 0
                            && !satisfies_monotonic_constraint (
                              c,
                              gl[b],
                              hl[b],
                              gr,
                              hr,
                              -std::numeric_limits<double>::infinity (),
                              std::numeric_limits<double>::infinity ()))
                          {
                            mono_ok = false;
                            break;
                          }
                        level_gain += newton_regulariser::split_gain (
                          gl[b], hl[b], gr, hr, reg_.l1, reg_.l2, 0.0);
                      }
                    level_gain -= reg_.complexity;

                    if (mono_ok && any_valid && level_gain > best_level_gain)
                      {
                        best_level_gain = level_gain;
                        best_f = static_cast<std::int32_t> (c);
                        best_thr = thr;
                        best_miss_right = (miss_right == 1);
                      }
                  }
              }
          }

        if (best_level_gain <= 0.0)
          {
            break;
          }

        std::size_t f_col = static_cast<std::size_t> (best_f);
        if (f_col < raw_gain_importances_.size ())
          {
            raw_gain_importances_[f_col] += best_level_gain;
            split_counts_[f_col] += 1;
          }

        tree.oblivious_features.push_back (best_f);
        tree.oblivious_thresholds.push_back (best_thr);
        tree.oblivious_missing_right.push_back (best_miss_right ? 1 : 0);

        for (std::size_t idx = 0; idx < n_active; ++idx)
          {
            std::size_t r = active_rows[idx];
            double v =
              table.at (r, static_cast<std::size_t> (best_f));
            bool go_right =
              is_missing (v) ? best_miss_right : (v > best_thr);
            if (go_right)
              {
                bucket_of_row[idx] |= (std::size_t{1} << d);
              }
          }
      }

    std::size_t final_depth = tree.oblivious_features.size ();
    std::size_t total_leaves = std::size_t{1} << final_depth;
    std::vector<double> leaf_g (total_leaves, 0.0);
    std::vector<double> leaf_h (total_leaves, 0.0);
    for (std::size_t idx = 0; idx < n_active; ++idx)
      {
        std::size_t r = active_rows[idx];
        std::size_t b = bucket_of_row[idx];
        leaf_g[b] += grads[r];
        leaf_h[b] += hesses[r];
      }

    tree.oblivious_leaves.resize (total_leaves, 0.0);
    for (std::size_t b = 0; b < total_leaves; ++b)
      {
        tree.oblivious_leaves[b] = newton_regulariser::leaf_weight (
          leaf_g[b], leaf_h[b], reg_.l1, reg_.l2);
      }
    return tree;
  }

  struct build_task
  {
    std::int32_t node_index = 0;
    std::size_t depth = 0;
    double grad_sum = 0.0;
    double hess_sum = 0.0;
    double min_bound = -std::numeric_limits<double>::infinity ();
    double max_bound = std::numeric_limits<double>::infinity ();
    std::vector<std::size_t> rows;
    node_histograms hists;
    split_proposal best_split;
  };

  additive_stage_tree
  build_single_stage_tree (const data_table &table,
                            const quantile_bins &qbins,
                            const std::vector<double> &grads,
                            const std::vector<double> &hesses,
                            std::size_t stage_seed)
  {
    std::vector<std::size_t> active_rows;
    std::vector<double> work_grads;
    std::vector<double> work_hesses;
    select_stage_rows (
      grads, hesses, stage_seed, active_rows, work_grads, work_hesses);

    if constexpr (std::is_same_v<Growth_Policy, oblivious_symmetric_growth>)
      {
        return build_oblivious_tree (
          table, qbins, active_rows, work_grads, work_hesses);
      }

    additive_stage_tree tree;
    tree.is_oblivious = false;

    double root_g = 0.0;
    double root_h = 0.0;
    for (std::size_t r : active_rows)
      {
        root_g += work_grads[r];
        root_h += work_hesses[r];
      }

    stage_tree_node root_node;
    root_node.leaf_value =
      newton_regulariser::leaf_weight (root_g, root_h, reg_.l1, reg_.l2);
    tree.nodes.push_back (root_node);

    build_task root_task;
    root_task.node_index = 0;
    root_task.depth = 0;
    root_task.grad_sum = root_g;
    root_task.hess_sum = root_h;
    root_task.rows = std::move (active_rows);

    if constexpr (Cut_Policy::uses_quantile_bins)
      {
        root_task.hists = compute_node_histograms (
          qbins, root_task.rows, work_grads, work_hesses);
        root_task.best_split = propose_best_split_from_histograms (
          qbins,
          root_task.hists,
          root_task.grad_sum,
          root_task.hess_sum,
          root_task.rows.size (),
          root_task.min_bound,
          root_task.max_bound);
      }
    else
      {
        root_task.best_split = propose_best_split_exact (table,
                                                          root_task.rows,
                                                          work_grads,
                                                          work_hesses,
                                                          root_task.grad_sum,
                                                          root_task.hess_sum,
                                                          root_task.min_bound,
                                                          root_task.max_bound);
      }

    std::vector<build_task> frontier;
    frontier.push_back (std::move (root_task));
    std::size_t current_leaves = 1;

    while (!frontier.empty () && current_leaves < limits_.max_leaves)
      {
        std::size_t pick_idx = 0;
        if constexpr (std::is_same_v<Growth_Policy, best_gain_frontier_growth>)
          {
            double max_gain = -1.0;
            for (std::size_t i = 0; i < frontier.size (); ++i)
              {
                if (frontier[i].best_split.valid
                    && frontier[i].depth < limits_.max_depth
                    && frontier[i].best_split.gain > max_gain)
                  {
                    max_gain = frontier[i].best_split.gain;
                    pick_idx = i;
                  }
              }
            if (max_gain <= 0.0)
              {
                break;
              }
          }

        build_task task = std::move (frontier[pick_idx]);
        frontier.erase (frontier.begin ()
                        + static_cast<std::ptrdiff_t> (pick_idx));

        if (!task.best_split.valid || task.depth >= limits_.max_depth)
          {
            continue;
          }

        const auto &sp = task.best_split;
        std::vector<std::size_t> left_rows;
        std::vector<std::size_t> right_rows;
        left_rows.reserve (task.rows.size ());
        right_rows.reserve (task.rows.size ());

        std::size_t f_col = static_cast<std::size_t> (sp.feature);
        for (std::size_t r : task.rows)
          {
            double v = table.at (r, f_col);
            bool go_right =
              is_missing (v) ? sp.missing_goes_right : (v > sp.threshold);
            if (go_right)
              {
                right_rows.push_back (r);
              }
            else
              {
                left_rows.push_back (r);
              }
          }

        if (left_rows.empty () || right_rows.empty ())
          {
            continue;
          }

        if (f_col < raw_gain_importances_.size ())
          {
            raw_gain_importances_[f_col] += sp.gain;
            split_counts_[f_col] += 1;
          }

        std::int32_t left_id = static_cast<std::int32_t> (tree.nodes.size ());
        std::int32_t right_id = left_id + 1;

        double raw_left = std::clamp (
          newton_regulariser::leaf_weight (
            sp.left_grad, sp.left_hess, reg_.l1, reg_.l2),
          task.min_bound,
          task.max_bound);
        double raw_right = std::clamp (
          newton_regulariser::leaf_weight (
            sp.right_grad, sp.right_hess, reg_.l1, reg_.l2),
          task.min_bound,
          task.max_bound);

        stage_tree_node left_node;
        left_node.leaf_value = raw_left;

        stage_tree_node right_node;
        right_node.leaf_value = raw_right;

        tree.nodes.push_back (left_node);
        tree.nodes.push_back (right_node);

        auto &parent_node =
          tree.nodes[static_cast<std::size_t> (task.node_index)];
        parent_node.feature_index = sp.feature;
        parent_node.threshold = sp.threshold;
        parent_node.missing_goes_right = sp.missing_goes_right;
        parent_node.left_child = left_id;
        parent_node.right_child = right_id;

        current_leaves += 1;

        double mid_val = 0.5 * (raw_left + raw_right);
        int dir = feature_monotonic_dir (f_col);

        build_task left_task;
        left_task.node_index = left_id;
        left_task.depth = task.depth + 1;
        left_task.grad_sum = sp.left_grad;
        left_task.hess_sum = sp.left_hess;
        left_task.min_bound = (dir < 0) ? mid_val : task.min_bound;
        left_task.max_bound = (dir > 0) ? mid_val : task.max_bound;
        left_task.rows = std::move (left_rows);

        build_task right_task;
        right_task.node_index = right_id;
        right_task.depth = task.depth + 1;
        right_task.grad_sum = sp.right_grad;
        right_task.hess_sum = sp.right_hess;
        right_task.min_bound = (dir > 0) ? mid_val : task.min_bound;
        right_task.max_bound = (dir < 0) ? mid_val : task.max_bound;
        right_task.rows = std::move (right_rows);

        if (left_task.depth < limits_.max_depth)
          {
            if constexpr (Cut_Policy::uses_quantile_bins)
              {
                if (left_task.rows.size () <= right_task.rows.size ())
                  {
                    left_task.hists = compute_node_histograms (
                      qbins, left_task.rows, work_grads, work_hesses);
                    right_task.hists = subtract_node_histograms (
                      task.hists, left_task.hists);
                  }
                else
                  {
                    right_task.hists = compute_node_histograms (
                      qbins, right_task.rows, work_grads, work_hesses);
                    left_task.hists = subtract_node_histograms (
                      task.hists, right_task.hists);
                  }
                left_task.best_split = propose_best_split_from_histograms (
                  qbins,
                  left_task.hists,
                  left_task.grad_sum,
                  left_task.hess_sum,
                  left_task.rows.size (),
                  left_task.min_bound,
                  left_task.max_bound);
                right_task.best_split = propose_best_split_from_histograms (
                  qbins,
                  right_task.hists,
                  right_task.grad_sum,
                  right_task.hess_sum,
                  right_task.rows.size (),
                  right_task.min_bound,
                  right_task.max_bound);
              }
            else
              {
                left_task.best_split =
                  propose_best_split_exact (table,
                                            left_task.rows,
                                            work_grads,
                                            work_hesses,
                                            left_task.grad_sum,
                                            left_task.hess_sum,
                                            left_task.min_bound,
                                            left_task.max_bound);
                right_task.best_split =
                  propose_best_split_exact (table,
                                            right_task.rows,
                                            work_grads,
                                            work_hesses,
                                            right_task.grad_sum,
                                            right_task.hess_sum,
                                            right_task.min_bound,
                                            right_task.max_bound);
              }
            frontier.push_back (std::move (left_task));
            frontier.push_back (std::move (right_task));
          }
      }

    return tree;
  }

  Loss_Type loss_{};
  with_regularisation reg_{};
  double step_rate_ = 0.1;
  std::size_t stage_count_ = 50;
  with_tree_limits limits_{};
  with_ordered_categories ordered_cats_{};
  with_gradient_sampling grad_sampling_{};
  with_monotonic_constraints monotonic_{};
  std::uint64_t seed_ = 42;

  bool calibrated_ = false;
  std::size_t feature_count_ = 0;
  std::size_t class_count_ = 1;
  std::size_t best_stage_ = 0;
  std::vector<double> base_scores_;
  std::vector<additive_stage_tree> trees_;
  std::vector<double> raw_gain_importances_;
  std::vector<double> feature_importances_;
  std::vector<std::size_t> split_counts_;

  bool has_nominal_encodings_ = false;
  std::vector<bool> is_nominal_col_;
  std::vector<double> nominal_priors_;
  std::vector<std::vector<double>> nominal_maps_;
};

// Factory helper constructing a stagewise_booster from a with_booster tag.
template <typename Loss_Type = squared_residual_loss,
          typename Cut_Policy,
          typename Growth_Policy,
          typename Sampling_Policy,
          typename... Modifiers>
auto
make_stagewise_booster (
  with_booster<Cut_Policy, Growth_Policy, Sampling_Policy>,
  Modifiers... mods)
{
  using resolved_sampling = std::conditional_t<
    std::is_void_v<Sampling_Policy>,
    uniform_full_sample,
    Sampling_Policy>;
  return stagewise_booster<Loss_Type,
                            Cut_Policy,
                            Growth_Policy,
                            resolved_sampling> (mods...);
}

} // namespace bellows

#endif // BELLOWS_BOOSTING_BOOSTER_HPP
