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

#ifndef BELLOWS_CORE_OPERATIONS_HPP
#define BELLOWS_CORE_OPERATIONS_HPP

#include <span>
#include <stdexcept>
#include <vector>

#include <bellows/core/concepts.hpp>

namespace bellows
{

// Calibrate an unsupervised estimator in place and return a reference to it.
template <typename E, tabular_source Data>
  requires unsupervised_fittable<E, Data>
E &
fit (E &est, const Data &data)
{
  est.calibrate (data);
  return est;
}

// Calibrate a supervised estimator in place and return a reference to it.
template <typename E, tabular_source Data, target_sequence Targets>
  requires supervised_fittable<E, Data, Targets>
E &
fit (E &est, const Data &data, const Targets &targets)
{
  if (data.row_count () != targets.size ())
    {
      throw std::invalid_argument (
        "bellows::fit: data.row_count() does not match targets.size()");
    }
  est.calibrate (data, targets);
  return est;
}

// Functional convenience returning a calibrated copy of an unsupervised spec.
template <typename E, tabular_source Data>
  requires unsupervised_fittable<E, Data>
E
fitted (E spec, const Data &data)
{
  spec.calibrate (data);
  return spec;
}

// Functional convenience returning a calibrated copy of a supervised spec.
template <typename E, tabular_source Data, target_sequence Targets>
  requires supervised_fittable<E, Data, Targets>
E
fitted (E spec, const Data &data, const Targets &targets)
{
  if (data.row_count () != targets.size ())
    {
      throw std::invalid_argument (
        "bellows::fitted: data.row_count() does not match targets.size()");
    }
  spec.calibrate (data, targets);
  return spec;
}

// Generate point predictions for a batch of tabular observations.
template <typename M, tabular_source Data>
  requires predictor<M, Data>
std::vector<double>
predict (const M &model, const Data &data)
{
  return model.evaluate (data);
}

// Generate low-latency single-row prediction without dynamic allocation.
template <typename M>
  requires predictor<M, data_table>
double
infer_one (const M &model, std::span<const double> single_row)
{
  return model.evaluate_one (single_row);
}

// Generate normalized class probability beliefs for each observation.
template <typename M, tabular_source Data>
  requires belief_predictor<M, Data>
data_table
predict_beliefs (const M &model, const Data &data)
{
  return model.evaluate_beliefs (data);
}

// Project a tabular dataset through a calibrated transformer.
template <typename T, tabular_source Data>
  requires transformer<T, Data>
data_table
transform (const T &trans, const Data &data)
{
  return trans.project (data);
}

// Calibrate an unsupervised transformer and immediately project the dataset.
template <typename T, tabular_source Data>
  requires unsupervised_fittable<T, Data> && transformer<T, Data>
data_table
fit_and_transform (T &trans, const Data &data)
{
  trans.calibrate (data);
  return trans.project (data);
}

// Calibrate a supervised transformer and immediately project the dataset.
template <typename T, tabular_source Data, target_sequence Targets>
  requires supervised_fittable<T, Data, Targets> && transformer<T, Data>
data_table
fit_and_transform (T &trans, const Data &data, const Targets &targets)
{
  if (data.row_count () != targets.size ())
    {
      throw std::invalid_argument (
        "bellows::fit_and_transform: row count and target size mismatch");
    }
  trans.calibrate (data, targets);
  if constexpr (requires { trans.project_ordered (data, targets); })
    {
      return trans.project_ordered (data, targets);
    }
  else
    {
      return trans.project (data);
    }
}

// Evaluate a calibrated predictor on a dataset using an evaluation metric.
template <typename M,
          tabular_source Data,
          target_sequence Targets,
          evaluation_metric Metric>
  requires predictor<M, Data>
double
evaluate (const M &model,
          const Data &data,
          const Targets &targets,
          const Metric &metric)
{
  std::vector<double> preds = model.evaluate (data);
  std::vector<double> truth (targets.size ());
  for (std::size_t i = 0; i < targets.size (); ++i)
    {
      truth[i] = static_cast<double> (targets[i]);
    }
  return metric (truth, preds);
}

} // namespace bellows

#endif // BELLOWS_CORE_OPERATIONS_HPP
