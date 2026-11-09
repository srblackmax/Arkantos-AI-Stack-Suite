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

#ifndef BELLOWS_COMPOSITION_VALIDATION_HPP
#define BELLOWS_COMPOSITION_VALIDATION_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>
#include <bellows/core/operations.hpp>

namespace bellows
{

// Single train/validation row index partition.
struct fold_partition
{
  std::vector<std::size_t> train_rows;
  std::vector<std::size_t> validation_rows;
};

// Standard randomized k-fold partition generator.
class fold_splitter
{
public:
  explicit fold_splitter (std::size_t fold_count = 5,
                          bool shuffle = true,
                          std::uint64_t seed = 42) noexcept
    : fold_count_ (std::max<std::size_t> (2, fold_count)),
      shuffle_ (shuffle),
      seed_ (seed)
  {
  }

  template <tabular_source Data, target_sequence Targets>
  std::vector<fold_partition>
  split (const Data &data, const Targets &) const
  {
    std::size_t n = data.row_count ();
    if (n < 2)
      {
        return {};
      }
    std::size_t k = std::min (n, fold_count_);
    std::vector<std::size_t> indices (n);
    if (shuffle_)
      {
        hyper_math::random_engine rng (seed_);
        indices = rng.permutation (n);
      }
    else
      {
        std::iota (indices.begin (), indices.end (), 0);
      }

    std::vector<fold_partition> folds (k);
    for (std::size_t i = 0; i < n; ++i)
      {
        std::size_t fold_id = i % k;
        for (std::size_t f = 0; f < k; ++f)
          {
            if (f == fold_id)
              {
                folds[f].validation_rows.push_back (indices[i]);
              }
            else
              {
                folds[f].train_rows.push_back (indices[i]);
              }
          }
      }
    return folds;
  }

private:
  std::size_t fold_count_ = 5;
  bool shuffle_ = true;
  std::uint64_t seed_ = 42;
};

// Class-balanced stratified k-fold partition generator.
class stratified_splitter
{
public:
  explicit stratified_splitter (std::size_t fold_count = 5,
                                std::uint64_t seed = 42) noexcept
    : fold_count_ (std::max<std::size_t> (2, fold_count)),
      seed_ (seed)
  {
  }

  template <tabular_source Data, target_sequence Targets>
  std::vector<fold_partition>
  split (const Data &data, const Targets &targets) const
  {
    std::size_t n = data.row_count ();
    if (n < 2 || targets.size () < n)
      {
        return {};
      }
    std::size_t k = std::min (n, fold_count_);
    std::map<long, std::vector<std::size_t>> by_class;
    for (std::size_t i = 0; i < n; ++i)
      {
        by_class[std::lround (targets[i])].push_back (i);
      }

    hyper_math::random_engine rng (seed_);
    std::vector<fold_partition> folds (k);
    std::size_t cursor = 0;

    for (auto &kv : by_class)
      {
        auto &bucket = kv.second;
        std::vector<std::size_t> perm = rng.permutation (bucket.size ());
        for (std::size_t p_idx : perm)
          {
            std::size_t r = bucket[p_idx];
            std::size_t fold_id = cursor % k;
            ++cursor;
            for (std::size_t f = 0; f < k; ++f)
              {
                if (f == fold_id)
                  {
                    folds[f].validation_rows.push_back (r);
                  }
                else
                  {
                    folds[f].train_rows.push_back (r);
                  }
              }
          }
      }
    return folds;
  }

private:
  std::size_t fold_count_ = 5;
  std::uint64_t seed_ = 42;
};

// Aggregated cross-validation statistics across folds.
struct validation_summary
{
  std::vector<double> fold_scores;
  double mean_score = 0.0;
  double std_dev_score = 0.0;
};

// Evaluate a supervised predictor across all partitions of a splitter.
template <typename Learner,
          typename Splitter,
          evaluation_metric Metric>
  requires supervised_fittable<Learner, data_table, std::vector<double>>
            && predictor<Learner, data_table>
validation_summary
cross_validate (const Learner &prototype,
                const data_table &data,
                const std::vector<double> &targets,
                const Splitter &splitter,
                const Metric &metric)
{
  if (data.row_count () != targets.size ())
    {
      throw std::invalid_argument (
        "bellows::cross_validate: row count and target size mismatch");
    }
  validation_summary summary;
  auto partitions = splitter.split (data, targets);
  if (partitions.empty ())
    {
      return summary;
    }

  for (const auto &part : partitions)
    {
      data_table x_train = data.select_rows (part.train_rows);
      data_table x_val = data.select_rows (part.validation_rows);
      std::vector<double> y_train;
      std::vector<double> y_val;
      y_train.reserve (part.train_rows.size ());
      y_val.reserve (part.validation_rows.size ());

      for (std::size_t r : part.train_rows)
        {
          y_train.push_back (targets[r]);
        }
      for (std::size_t r : part.validation_rows)
        {
          y_val.push_back (targets[r]);
        }

      Learner fold_model = prototype;
      fold_model.calibrate (x_train, y_train);
      std::vector<double> preds = fold_model.evaluate (x_val);
      summary.fold_scores.push_back (metric (y_val, preds));
    }

  double sum = 0.0;
  for (double s : summary.fold_scores)
    {
      sum += s;
    }
  summary.mean_score =
    sum / static_cast<double> (summary.fold_scores.size ());

  double sq = 0.0;
  for (double s : summary.fold_scores)
    {
      double d = s - summary.mean_score;
      sq += d * d;
    }
  summary.std_dev_score =
    std::sqrt (sq / static_cast<double> (summary.fold_scores.size ()));
  return summary;
}

// Single hyperparameter setting assignment.
using parameter_assignment = std::vector<std::pair<std::string, double>>;

// Result of a hyperparameter sweep over candidate assignments.
template <typename Learner>
struct sweep_outcome
{
  Learner best_estimator;
  double best_score = 0.0;
  parameter_assignment best_assignment;
  std::vector<double> trial_scores;
};

// Explore candidate hyperparameter assignments using cross-validation.
template <typename Learner,
          typename Splitter,
          evaluation_metric Metric>
  requires tunable<Learner>
            && supervised_fittable<Learner, data_table, std::vector<double>>
            && predictor<Learner, data_table>
sweep_outcome<Learner>
parameter_sweep (const Learner &base_learner,
                  const data_table &data,
                  const std::vector<double> &targets,
                  const Splitter &splitter,
                  const Metric &metric,
                  const std::vector<parameter_assignment> &candidates)
{
  sweep_outcome<Learner> outcome;
  outcome.best_estimator = base_learner;
  outcome.best_score = Metric::maximise
                          ? -std::numeric_limits<double>::infinity ()
                          : std::numeric_limits<double>::infinity ();

  for (const auto &assign : candidates)
    {
      Learner candidate = base_learner;
      for (const auto &kv : assign)
        {
          candidate.apply_parameter (kv.first, kv.second);
        }
      validation_summary cv =
        cross_validate (candidate, data, targets, splitter, metric);
      outcome.trial_scores.push_back (cv.mean_score);

      bool is_better = Metric::maximise
                          ? (cv.mean_score > outcome.best_score)
                          : (cv.mean_score < outcome.best_score);
      if (is_better)
        {
          outcome.best_score = cv.mean_score;
          outcome.best_assignment = assign;
          outcome.best_estimator = candidate;
        }
    }

  outcome.best_estimator.calibrate (data, targets);
  return outcome;
}

// Cartesian grid hyperparameter design generator for tunable learners.
class grid_explorer
{
public:
  explicit grid_explorer (std::size_t levels_per_knob = 3) noexcept
    : levels_per_knob_ (std::max<std::size_t> (2, levels_per_knob))
  {
  }

  explicit grid_explorer (
    std::vector<std::pair<std::string, std::vector<double>>> axes)
    : explicit_axes_ (std::move (axes))
  {
  }

  template <tunable Learner>
  std::vector<parameter_assignment>
  candidates (const Learner &base_learner) const
  {
    std::vector<std::pair<std::string, std::vector<double>>> axes =
      explicit_axes_;
    if (axes.empty ())
      {
        auto space = base_learner.parameter_space ();
        for (const auto &knob : space)
          {
            std::vector<double> vals;
            vals.reserve (levels_per_knob_);
            for (std::size_t i = 0; i < levels_per_knob_; ++i)
              {
                double frac = static_cast<double> (i)
                              / static_cast<double> (levels_per_knob_ - 1);
                vals.push_back (
                  knob.min_value + frac * (knob.max_value - knob.min_value));
              }
            axes.push_back ({knob.name, std::move (vals)});
          }
      }

    std::vector<parameter_assignment> grid = {{}};
    for (const auto &axis : axes)
      {
        std::vector<parameter_assignment> next_grid;
        for (const auto &prefix : grid)
          {
            for (double v : axis.second)
              {
                parameter_assignment extended = prefix;
                extended.push_back ({axis.first, v});
                next_grid.push_back (std::move (extended));
              }
          }
        grid = std::move (next_grid);
      }
    return grid;
  }

private:
  std::size_t levels_per_knob_ = 3;
  std::vector<std::pair<std::string, std::vector<double>>> explicit_axes_;
};

// Space-filling Latin Hypercube design generator backed by Hyper Math PRNG.
class random_latin_explorer
{
public:
  explicit random_latin_explorer (
    std::size_t trial_count = 8,
    std::uint64_t seed = 42,
    std::vector<parameter_knob> custom_knobs = {}) noexcept
    : trial_count_ (std::max<std::size_t> (1, trial_count)),
      seed_ (seed),
      custom_knobs_ (std::move (custom_knobs))
  {
  }

  template <tunable Learner>
  std::vector<parameter_assignment>
  candidates (const Learner &base_learner) const
  {
    auto space = custom_knobs_.empty () ? base_learner.parameter_space ()
                                        : custom_knobs_;
    std::vector<parameter_assignment> trials (trial_count_);
    if (space.empty ())
      {
        return trials;
      }

    hyper_math::random_engine rng (seed_);
    for (const auto &knob : space)
      {
        std::vector<std::size_t> perm = rng.permutation (trial_count_);
        double span = knob.max_value - knob.min_value;
        for (std::size_t t = 0; t < trial_count_; ++t)
          {
            double jitter = rng.uniform (0.0, 1.0);
            double frac = (static_cast<double> (perm[t]) + jitter)
                          / static_cast<double> (trial_count_);
            double val = knob.min_value + frac * span;
            trials[t].push_back ({knob.name, val});
          }
      }
    return trials;
  }

private:
  std::size_t trial_count_ = 8;
  std::uint64_t seed_ = 42;
  std::vector<parameter_knob> custom_knobs_;
};

// Free-function CPO overload of evaluate performing cross-validation.
template <typename Learner,
          typename Splitter,
          evaluation_metric Metric>
  requires supervised_fittable<Learner, data_table, std::vector<double>>
            && predictor<Learner, data_table>
validation_summary
evaluate (const Learner &prototype,
          const data_table &data,
          const std::vector<double> &targets,
          const Splitter &splitter,
          const Metric &metric)
{
  return cross_validate (prototype, data, targets, splitter, metric);
}

// Free-function CPO tuning a learner using an explorer strategy.
template <typename Learner,
          typename Splitter,
          evaluation_metric Metric,
          typename Explorer>
  requires tunable<Learner>
            && supervised_fittable<Learner, data_table, std::vector<double>>
            && predictor<Learner, data_table>
sweep_outcome<Learner>
tune (const Learner &base_learner,
      const data_table &data,
      const std::vector<double> &targets,
      const Splitter &splitter,
      const Metric &metric,
      const Explorer &explorer)
{
  return parameter_sweep (base_learner,
                          data,
                          targets,
                          splitter,
                          metric,
                          explorer.candidates (base_learner));
}

} // namespace bellows

#endif // BELLOWS_COMPOSITION_VALIDATION_HPP
