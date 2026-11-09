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

#ifndef BELLOWS_CORE_CONCEPTS_HPP
#define BELLOWS_CORE_CONCEPTS_HPP

#include <concepts>
#include <cstddef>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include <bellows/core/data_table.hpp>
#include <bellows/core/parameters.hpp>

namespace bellows
{

// Concept for any 2D tabular observation source with column scale metadata.
template <typename T>
concept tabular_source = requires (const T &src, std::size_t r, std::size_t c)
{
  { src.row_count () } -> std::convertible_to<std::size_t>;
  { src.column_count () } -> std::convertible_to<std::size_t>;
  { src.at (r, c) } -> std::convertible_to<double>;
  { src.scale (c) } -> std::same_as<feature_scale>;
  { src.cardinality (c) } -> std::convertible_to<std::size_t>;
};

// Concept for a 1D target response sequence.
template <typename V>
concept target_sequence = requires (const V &seq, std::size_t i)
{
  { seq.size () } -> std::convertible_to<std::size_t>;
  { seq[i] } -> std::convertible_to<double>;
};

// Base concept satisfied by all learning and preprocessing components.
template <typename E>
concept estimator = std::copy_constructible<E> && requires (const E &est)
{
  { est.is_calibrated () } -> std::convertible_to<bool>;
};

// Concept for components calibrated on unlabeled tabular observations.
template <typename E, typename Data = data_table>
concept unsupervised_fittable =
  estimator<E> && tabular_source<Data> && requires (E &est, const Data &data)
{
  est.calibrate (data);
};

// Concept for components calibrated on labeled tabular observations.
template <typename E,
          typename Data = data_table,
          typename Targets = std::vector<double>>
concept supervised_fittable =
  estimator<E> && tabular_source<Data> && target_sequence<Targets>
  && requires (E &est, const Data &data, const Targets &targets)
{
  est.calibrate (data, targets);
};

// Concept for calibrated components that produce point predictions.
template <typename M, typename Data = data_table>
concept predictor =
  estimator<M> && tabular_source<Data>
  && requires (const M &model,
                const Data &data,
                std::span<const double> single_row)
{
  { model.evaluate (data) } -> std::same_as<std::vector<double>>;
  { model.evaluate_one (single_row) } -> std::convertible_to<double>;
};

// Concept for calibrated classifiers that produce class probability beliefs.
template <typename M, typename Data = data_table>
concept belief_predictor =
  predictor<M, Data> && requires (const M &model, const Data &data)
{
  { model.evaluate_beliefs (data) } -> std::same_as<data_table>;
};

// Concept for components that map a tabular source to a new data_table.
template <typename T, typename Data = data_table>
concept transformer =
  estimator<T> && tabular_source<Data>
  && requires (const T &trans, const Data &data)
{
  { trans.project (data) } -> std::same_as<data_table>;
};

// Compile-time capability concept for native nominal feature support.
template <typename E>
concept categorical_aware = estimator<E> && requires
{
  requires (E::supports_nominal_columns == true);
};

// Compile-time capability concept for native missing-value routing/handling.
template <typename E>
concept missing_aware = estimator<E> && requires
{
  requires (E::supports_missing_entries == true);
};

// Concept for estimators exposing tunable hyperparameter knobs.
template <typename E>
concept tunable =
  estimator<E>
  && requires (E &est, const E &cest, const std::string &name, double val)
{
  { cest.parameter_space () } -> std::same_as<std::vector<parameter_knob>>;
  { est.apply_parameter (name, val) } -> std::convertible_to<bool>;
};

// Concept for models supporting binary stream persistence.
template <typename M>
concept serialisable =
  estimator<M>
  && requires (const M &cmodel, M &model, std::ostream &out, std::istream &in)
{
  cmodel.write_state (out);
  model.read_state (in);
};

// Pair of first- and second-order loss derivatives for a single sample.
struct derivative_pair
{
  double gradient = 0.0;
  double curvature = 1.0;
};

// Concept for twice-differentiable stagewise boosting loss objectives.
template <typename L>
concept second_order_loss =
  requires (const L &loss,
            const std::vector<double> &targets,
            double y,
            double raw_score)
{
  { loss.base_score (targets) } -> std::convertible_to<double>;
  { loss.compute_derivatives (y, raw_score) } -> std::same_as<derivative_pair>;
  { loss.map_prediction (raw_score) } -> std::convertible_to<double>;
};

// Concept for composable evaluation metric function objects.
template <typename M>
concept evaluation_metric =
  requires (const M &metric,
            const std::vector<double> &truth,
            const std::vector<double> &pred)
{
  { metric (truth, pred) } -> std::convertible_to<double>;
  { M::maximise } -> std::convertible_to<bool>;
};

} // namespace bellows

#endif // BELLOWS_CORE_CONCEPTS_HPP
