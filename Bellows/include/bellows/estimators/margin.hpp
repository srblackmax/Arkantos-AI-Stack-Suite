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

#ifndef BELLOWS_ESTIMATORS_MARGIN_HPP
#define BELLOWS_ESTIMATORS_MARGIN_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>
#include <bellows/core/parameters.hpp>

namespace bellows
{

// Regularised maximum-margin classifier supporting binary and multiclass targets.
class max_margin_classifier
{
public:
  static constexpr bool supports_nominal_columns = false;
  static constexpr bool supports_missing_entries = true;

  explicit max_margin_classifier (double penalty = 0.01,
                                  double learning_rate = 0.05,
                                  std::size_t max_epochs = 250) noexcept
    : penalty_ (std::max (1.0e-8, penalty)),
      learning_rate_ (std::max (1.0e-6, learning_rate)),
      max_epochs_ (std::max<std::size_t> (1, max_epochs))
  {
  }

  bool
  is_calibrated () const noexcept
  {
    return calibrated_;
  }

  std::size_t
  class_count () const noexcept
  {
    return classes_.size ();
  }

  std::vector<parameter_knob>
  parameter_space () const
  {
    return {
      {"penalty", penalty_, 1.0e-5, 10.0},
      {"learning_rate", learning_rate_, 1.0e-4, 1.0}
    };
  }

  bool
  apply_parameter (const std::string &name, double val) noexcept
  {
    if (name == "penalty")
      {
        penalty_ = std::max (1.0e-8, val);
        return true;
      }
    if (name == "learning_rate")
      {
        learning_rate_ = std::max (1.0e-6, val);
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
    if (targets.size () < n)
      {
        throw std::invalid_argument (
          "max_margin_classifier::calibrate: targets shorter than data");
      }
    if (n == 0 || d == 0)
      {
        calibrated_ = false;
        return;
      }

    feature_count_ = d;
    col_means_.assign (d, 0.0);
    for (std::size_t c = 0; c < d; ++c)
      {
        double sum = 0.0;
        std::size_t cnt = 0;
        for (std::size_t r = 0; r < n; ++r)
          {
            double v = data.at (r, c);
            if (!is_missing (v))
              {
                sum += v;
                ++cnt;
              }
          }
        col_means_[c] = cnt > 0 ? sum / static_cast<double> (cnt) : 0.0;
      }

    classes_.clear ();
    for (std::size_t r = 0; r < n; ++r)
      {
        double lbl = static_cast<double> (targets[r]);
        if (std::find (classes_.begin (), classes_.end (), lbl)
            == classes_.end ())
          {
            classes_.push_back (lbl);
          }
      }
    std::sort (classes_.begin (), classes_.end ());
    if (classes_.size () < 2)
      {
        classes_ = {0.0, 1.0};
      }

    std::size_t n_planes = classes_.size () == 2 ? 1 : classes_.size ();
    weights_.assign (n_planes * d, 0.0);
    intercepts_.assign (n_planes, 0.0);

    for (std::size_t p = 0; p < n_planes; ++p)
      {
        double pos_label = classes_.size () == 2 ? classes_[1] : classes_[p];
        double *w = weights_.data () + p * d;
        double b = 0.0;

        for (std::size_t epoch = 0; epoch < max_epochs_; ++epoch)
          {
            double eta =
              learning_rate_ / (1.0 + 0.02 * static_cast<double> (epoch));
            std::vector<double> grad_w (d, 0.0);
            double grad_b = 0.0;

            for (std::size_t r = 0; r < n; ++r)
              {
                double y =
                  (static_cast<double> (targets[r]) == pos_label) ? 1.0 : -1.0;
                double margin = b;
                for (std::size_t c = 0; c < d; ++c)
                  {
                    double v = data.at (r, c);
                    double xv = is_missing (v) ? col_means_[c] : v;
                    margin += w[c] * xv;
                  }
                if (y * margin < 1.0)
                  {
                    for (std::size_t c = 0; c < d; ++c)
                      {
                        double v = data.at (r, c);
                        double xv = is_missing (v) ? col_means_[c] : v;
                        grad_w[c] -= y * xv;
                      }
                    grad_b -= y;
                  }
              }

            double inv_n = 1.0 / static_cast<double> (n);
            for (std::size_t c = 0; c < d; ++c)
              {
                double total_grad = grad_w[c] * inv_n + penalty_ * w[c];
                w[c] -= eta * total_grad;
              }
            b -= eta * (grad_b * inv_n);
          }
        intercepts_[p] = b;
      }

    calibrated_ = true;
  }

  double
  decision_margin (std::span<const double> row,
                    std::size_t plane = 0) const noexcept
  {
    if (!calibrated_ || intercepts_.empty ())
      {
        return 0.0;
      }
    std::size_t p = std::min (plane, intercepts_.size () - 1);
    std::size_t d = std::min (row.size (), feature_count_);
    const double *w = weights_.data () + p * feature_count_;
    double score = intercepts_[p];
    for (std::size_t c = 0; c < d; ++c)
      {
        double xv = is_missing (row[c]) ? col_means_[c] : row[c];
        score += w[c] * xv;
      }
    return score;
  }

  std::vector<double>
  evaluate_beliefs_one (std::span<const double> row) const
  {
    if (!calibrated_ || classes_.empty ())
      {
        return {0.5, 0.5};
      }
    if (classes_.size () == 2)
      {
        double m = std::clamp (decision_margin (row, 0), -50.0, 50.0);
        double p1 = 1.0 / (1.0 + std::exp (-m));
        return {1.0 - p1, p1};
      }

    std::size_t k = classes_.size ();
    std::vector<double> margins (k, 0.0);
    double max_m = -1.0e300;
    for (std::size_t p = 0; p < k; ++p)
      {
        margins[p] = decision_margin (row, p);
        if (margins[p] > max_m)
          {
            max_m = margins[p];
          }
      }

    std::vector<double> probs (k, 0.0);
    double sum_exp = 0.0;
    for (std::size_t p = 0; p < k; ++p)
      {
        probs[p] = std::exp (std::clamp (margins[p] - max_m, -60.0, 0.0));
        sum_exp += probs[p];
      }
    if (sum_exp > 0.0)
      {
        for (double &pr : probs)
          {
            pr /= sum_exp;
          }
      }
    return probs;
  }

  template <tabular_source Data>
  data_table
  evaluate_beliefs (const Data &data) const
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    std::size_t k = std::max<std::size_t> (2, classes_.size ());
    data_table beliefs (n, k, 0.0);
    std::vector<double> buf (d, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            buf[c] = data.at (r, c);
          }
        auto pr = evaluate_beliefs_one (buf);
        for (std::size_t j = 0; j < std::min (k, pr.size ()); ++j)
          {
            beliefs.at (r, j) = pr[j];
          }
      }
    return beliefs;
  }

  double
  evaluate_one (std::span<const double> row) const noexcept
  {
    if (!calibrated_ || classes_.empty ())
      {
        return 0.0;
      }
    if (classes_.size () == 2)
      {
        return decision_margin (row, 0) >= 0.0 ? classes_[1] : classes_[0];
      }
    std::size_t best_p = 0;
    double best_m = decision_margin (row, 0);
    for (std::size_t p = 1; p < classes_.size (); ++p)
      {
        double m = decision_margin (row, p);
        if (m > best_m)
          {
            best_m = m;
            best_p = p;
          }
      }
    return classes_[best_p];
  }

  template <tabular_source Data>
  std::vector<double>
  evaluate (const Data &data) const
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    std::vector<double> out (n, 0.0);
    std::vector<double> buf (d, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            buf[c] = data.at (r, c);
          }
        out[r] = evaluate_one (buf);
      }
    return out;
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
    write_pod (penalty_);
    write_pod (learning_rate_);
    write_pod (max_epochs_);
    write_pod (feature_count_);

    std::size_t k = classes_.size ();
    std::size_t np = intercepts_.size ();
    write_pod (k);
    write_pod (np);

    for (double cl : classes_)
      {
        write_pod (cl);
      }
    for (double m : col_means_)
      {
        write_pod (m);
      }
    for (double w : weights_)
      {
        write_pod (w);
      }
    for (double b : intercepts_)
      {
        write_pod (b);
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
    read_pod (penalty_);
    read_pod (learning_rate_);
    read_pod (max_epochs_);
    read_pod (feature_count_);

    std::size_t k = 0;
    std::size_t np = 0;
    read_pod (k);
    read_pod (np);

    classes_.resize (k, 0.0);
    for (std::size_t i = 0; i < k; ++i)
      {
        read_pod (classes_[i]);
      }
    col_means_.resize (feature_count_, 0.0);
    for (std::size_t c = 0; c < feature_count_; ++c)
      {
        read_pod (col_means_[c]);
      }
    weights_.resize (np * feature_count_, 0.0);
    for (std::size_t i = 0; i < weights_.size (); ++i)
      {
        read_pod (weights_[i]);
      }
    intercepts_.resize (np, 0.0);
    for (std::size_t i = 0; i < np; ++i)
      {
        read_pod (intercepts_[i]);
      }
  }

private:
  double penalty_ = 0.01;
  double learning_rate_ = 0.05;
  std::size_t max_epochs_ = 250;
  std::size_t feature_count_ = 0;
  bool calibrated_ = false;
  std::vector<double> classes_;
  std::vector<double> col_means_;
  std::vector<double> weights_;
  std::vector<double> intercepts_;
};

enum class kernel_kind : std::uint8_t
{
  radial_basis = 0,
  polynomial = 1,
  linear = 2
};

// Dual kernelised regularised regressor backed by Hyper Math linear solver.
class kernel_ridge_regressor
{
public:
  static constexpr bool supports_nominal_columns = false;
  static constexpr bool supports_missing_entries = true;

  explicit kernel_ridge_regressor (
    double penalty = 1.0,
    double bandwidth = 1.0,
    kernel_kind kind = kernel_kind::radial_basis,
    std::size_t degree = 2) noexcept
    : penalty_ (std::max (1.0e-10, penalty)),
      bandwidth_ (std::max (1.0e-10, bandwidth)),
      kind_ (kind),
      degree_ (std::max<std::size_t> (1, degree))
  {
  }

  bool
  is_calibrated () const noexcept
  {
    return calibrated_;
  }

  std::vector<parameter_knob>
  parameter_space () const
  {
    return {
      {"penalty", penalty_, 1.0e-6, 100.0},
      {"bandwidth", bandwidth_, 1.0e-4, 100.0}
    };
  }

  bool
  apply_parameter (const std::string &name, double val) noexcept
  {
    if (name == "penalty")
      {
        penalty_ = std::max (1.0e-10, val);
        return true;
      }
    if (name == "bandwidth")
      {
        bandwidth_ = std::max (1.0e-10, val);
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
    if (targets.size () < n)
      {
        throw std::invalid_argument (
          "kernel_ridge_regressor::calibrate: targets shorter than data");
      }
    if (n == 0 || d == 0)
      {
        calibrated_ = false;
        return;
      }

    col_means_.assign (d, 0.0);
    for (std::size_t c = 0; c < d; ++c)
      {
        double sum = 0.0;
        std::size_t cnt = 0;
        for (std::size_t r = 0; r < n; ++r)
          {
            double v = data.at (r, c);
            if (!is_missing (v))
              {
                sum += v;
                ++cnt;
              }
          }
        col_means_[c] = cnt > 0 ? sum / static_cast<double> (cnt) : 0.0;
      }

    support_points_ = data_table (n, d, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = data.at (r, c);
            support_points_.at (r, c) = is_missing (v) ? col_means_[c] : v;
          }
      }

    hyper_math::tensor::nd_tensor<double> gram (
      hyper_math::tensor::shape_type{n, n}, 0.0);
    hyper_math::tensor::nd_tensor<double> rhs (
      hyper_math::tensor::shape_type{n}, 0.0);

    for (std::size_t i = 0; i < n; ++i)
      {
        rhs[i] = static_cast<double> (targets[i]);
        auto row_i = support_points_.row_span (i);
        for (std::size_t j = i; j < n; ++j)
          {
            double kij = kernel_eval (row_i, support_points_.row_span (j));
            gram (i, j) = kij;
            gram (j, i) = kij;
          }
        gram (i, i) += penalty_;
      }

    auto dual = hyper_math::scientific::linalg::solve (gram, rhs);
    dual_weights_.assign (n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      {
        dual_weights_[i] = dual[i];
      }
    calibrated_ = true;
  }

  double
  evaluate_one (std::span<const double> row) const noexcept
  {
    if (!calibrated_ || support_points_.row_count () == 0)
      {
        return 0.0;
      }
    std::size_t n = support_points_.row_count ();
    std::size_t d = std::min (row.size (), support_points_.column_count ());
    std::vector<double> clean_row (d, 0.0);
    for (std::size_t c = 0; c < d; ++c)
      {
        clean_row[c] = is_missing (row[c]) ? col_means_[c] : row[c];
      }

    double pred = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      {
        pred += dual_weights_[i]
                * kernel_eval (clean_row, support_points_.row_span (i));
      }
    return pred;
  }

  template <tabular_source Data>
  std::vector<double>
  evaluate (const Data &data) const
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    std::vector<double> out (n, 0.0);
    std::vector<double> buf (d, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            buf[c] = data.at (r, c);
          }
        out[r] = evaluate_one (buf);
      }
    return out;
  }

  void
  write_state (std::ostream &out) const
  {
    auto write_pod = [&] (const auto &val)
    {
      out.write (reinterpret_cast<const char *> (&val), sizeof (val));
    };
    std::uint8_t cal = calibrated_ ? 1 : 0;
    std::uint8_t k_val = static_cast<std::uint8_t> (kind_);
    write_pod (cal);
    write_pod (k_val);
    write_pod (penalty_);
    write_pod (bandwidth_);
    write_pod (degree_);

    std::size_t n = support_points_.row_count ();
    std::size_t d = support_points_.column_count ();
    write_pod (n);
    write_pod (d);

    for (double m : col_means_)
      {
        write_pod (m);
      }
    for (double alpha : dual_weights_)
      {
        write_pod (alpha);
      }
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            write_pod (support_points_.at (r, c));
          }
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
    std::uint8_t k_val = 0;
    read_pod (cal);
    read_pod (k_val);
    calibrated_ = (cal != 0);
    kind_ = static_cast<kernel_kind> (k_val);
    read_pod (penalty_);
    read_pod (bandwidth_);
    read_pod (degree_);

    std::size_t n = 0;
    std::size_t d = 0;
    read_pod (n);
    read_pod (d);

    col_means_.resize (d, 0.0);
    for (std::size_t c = 0; c < d; ++c)
      {
        read_pod (col_means_[c]);
      }
    dual_weights_.resize (n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      {
        read_pod (dual_weights_[i]);
      }
    support_points_ = data_table (n, d, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = 0.0;
            read_pod (v);
            support_points_.at (r, c) = v;
          }
      }
  }

private:
  double
  kernel_eval (std::span<const double> u,
                std::span<const double> v) const noexcept
  {
    std::size_t d = std::min (u.size (), v.size ());
    if (kind_ == kernel_kind::radial_basis)
      {
        double sq = 0.0;
        for (std::size_t c = 0; c < d; ++c)
          {
            double diff = u[c] - v[c];
            sq += diff * diff;
          }
        return std::exp (-bandwidth_ * sq);
      }
    double dot = 0.0;
    for (std::size_t c = 0; c < d; ++c)
      {
        dot += u[c] * v[c];
      }
    if (kind_ == kernel_kind::linear)
      {
        return dot + 1.0;
      }
    return std::pow (bandwidth_ * dot + 1.0, static_cast<double> (degree_));
  }

  double penalty_ = 1.0;
  double bandwidth_ = 1.0;
  kernel_kind kind_ = kernel_kind::radial_basis;
  std::size_t degree_ = 2;
  bool calibrated_ = false;
  std::vector<double> col_means_;
  std::vector<double> dual_weights_;
  data_table support_points_;
};

} // namespace bellows

#endif // BELLOWS_ESTIMATORS_MARGIN_HPP
