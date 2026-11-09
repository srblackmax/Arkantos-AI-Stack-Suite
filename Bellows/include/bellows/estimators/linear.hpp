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

#ifndef BELLOWS_ESTIMATORS_LINEAR_HPP
#define BELLOWS_ESTIMATORS_LINEAR_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include <bellows/boosting/policies.hpp>
#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>
#include <bellows/core/parameters.hpp>

namespace bellows
{

// Penalised linear regressor with L1/L2 regularisation via coordinate descent.
class penalised_linear_regressor
{
public:
  static constexpr bool supports_nominal_columns = false;
  static constexpr bool supports_missing_entries = true;

  explicit penalised_linear_regressor (
    with_regularisation reg = {0.0, 1.0e-2, 0.0},
    std::size_t max_iterations = 200) noexcept
    : reg_ (reg),
      max_iterations_ (max_iterations)
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
    return {{"l1", reg_.l1, 0.0, 100.0}, {"l2", reg_.l2, 0.0, 100.0}};
  }

  bool
  apply_parameter (const std::string &name, double val)
  {
    if (name == "l1")
      {
        reg_.l1 = std::max (0.0, val);
        return true;
      }
    if (name == "l2")
      {
        reg_.l2 = std::max (0.0, val);
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
    if (n == 0 || d == 0)
      {
        calibrated_ = false;
        return;
      }
    if (targets.size () != n)
      {
        throw std::invalid_argument (
          "bellows::penalised_linear_regressor::calibrate: target size mismatch");
      }

    col_means_.assign (d, 0.0);
    weights_.assign (d, 0.0);
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

    double y_mean = 0.0;
    for (std::size_t r = 0; r < n; ++r)
      {
        y_mean += targets[r];
      }
    y_mean /= static_cast<double> (n);
    intercept_ = y_mean;

    std::vector<double> x_centered (n * d, 0.0);
    std::vector<double> col_sq (d, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = data.at (r, c);
            double xc = is_missing (v) ? 0.0 : (v - col_means_[c]);
            x_centered[r * d + c] = xc;
            col_sq[c] += xc * xc;
          }
      }

    std::vector<double> residuals (n, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        residuals[r] = targets[r] - y_mean;
      }

    double n_dbl = static_cast<double> (n);
    if (reg_.l1 <= 0.0)
      {
        hyper_math::tensor::nd_tensor<double> gram (
          hyper_math::tensor::shape_type{d, d}, 0.0);
        hyper_math::tensor::nd_tensor<double> rhs (
          hyper_math::tensor::shape_type{d}, 0.0);
        for (std::size_t i = 0; i < d; ++i)
          {
            double r_sum = 0.0;
            for (std::size_t r = 0; r < n; ++r)
              {
                r_sum += x_centered[r * d + i] * residuals[r];
              }
            rhs[i] = r_sum;
            for (std::size_t j = i; j < d; ++j)
              {
                double g_sum = 0.0;
                for (std::size_t r = 0; r < n; ++r)
                  {
                    g_sum += x_centered[r * d + i] * x_centered[r * d + j];
                  }
                if (i == j)
                  {
                    g_sum += std::max (1.0e-8, reg_.l2 * n_dbl);
                  }
                gram (i, j) = g_sum;
                gram (j, i) = g_sum;
              }
          }
        auto w_tensor = hyper_math::scientific::linalg::solve (gram, rhs);
        for (std::size_t c = 0; c < d; ++c)
          {
            weights_[c] = w_tensor[c];
          }
      }
    else
      {
        for (std::size_t iter = 0; iter < max_iterations_; ++iter)
          {
            double max_delta = 0.0;
            for (std::size_t c = 0; c < d; ++c)
              {
                if (col_sq[c] <= 1.0e-15)
                  {
                    continue;
                  }
                double old_w = weights_[c];
                double rho = 0.0;
                for (std::size_t r = 0; r < n; ++r)
                  {
                    double x_rc = x_centered[r * d + c];
                    rho += x_rc * (residuals[r] + old_w * x_rc);
                  }
                double shrunk =
                  newton_regulariser::soft_threshold (rho, reg_.l1 * n_dbl);
                double new_w = shrunk / (col_sq[c] + reg_.l2 * n_dbl);
                double diff = new_w - old_w;
                if (std::abs (diff) > 0.0)
                  {
                    for (std::size_t r = 0; r < n; ++r)
                      {
                        residuals[r] -= diff * x_centered[r * d + c];
                      }
                    weights_[c] = new_w;
                    max_delta = std::max (max_delta, std::abs (diff));
                  }
              }
            if (max_delta < 1.0e-7)
              {
                break;
              }
          }
      }

    double shift = 0.0;
    for (std::size_t c = 0; c < d; ++c)
      {
        shift += weights_[c] * col_means_[c];
      }
    intercept_ = y_mean - shift;
    calibrated_ = true;
  }

  double
  evaluate_one (std::span<const double> row) const noexcept
  {
    if (!calibrated_)
      {
        return 0.0;
      }
    double sum = intercept_;
    std::size_t d = std::min (row.size (), weights_.size ());
    for (std::size_t c = 0; c < d; ++c)
      {
        double v = is_missing (row[c]) ? col_means_[c] : row[c];
        sum += weights_[c] * v;
      }
    return sum;
  }

  template <tabular_source Data>
  std::vector<double>
  evaluate (const Data &data) const
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    std::vector<double> preds (n, 0.0);
    std::vector<double> buf (d, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            buf[c] = data.at (r, c);
          }
        preds[r] = evaluate_one (buf);
      }
    return preds;
  }

  void
  write_state (std::ostream &out) const
  {
    std::uint8_t cal = calibrated_ ? 1 : 0;
    out.write (reinterpret_cast<const char *> (&cal), sizeof (cal));
    out.write (reinterpret_cast<const char *> (&intercept_),
                sizeof (intercept_));
    std::size_t d = weights_.size ();
    out.write (reinterpret_cast<const char *> (&d), sizeof (d));
    for (std::size_t c = 0; c < d; ++c)
      {
        out.write (reinterpret_cast<const char *> (&weights_[c]),
                    sizeof (double));
        out.write (reinterpret_cast<const char *> (&col_means_[c]),
                    sizeof (double));
      }
  }

  void
  read_state (std::istream &in)
  {
    std::uint8_t cal = 0;
    in.read (reinterpret_cast<char *> (&cal), sizeof (cal));
    calibrated_ = (cal != 0);
    in.read (reinterpret_cast<char *> (&intercept_), sizeof (intercept_));
    std::size_t d = 0;
    in.read (reinterpret_cast<char *> (&d), sizeof (d));
    weights_.resize (d);
    col_means_.resize (d);
    for (std::size_t c = 0; c < d; ++c)
      {
        in.read (reinterpret_cast<char *> (&weights_[c]), sizeof (double));
        in.read (reinterpret_cast<char *> (&col_means_[c]), sizeof (double));
      }
  }

  const std::vector<double> &
  weights () const noexcept
  {
    return weights_;
  }

  double
  intercept () const noexcept
  {
    return intercept_;
  }

private:
  with_regularisation reg_;
  std::size_t max_iterations_ = 200;
  bool calibrated_ = false;
  double intercept_ = 0.0;
  std::vector<double> weights_;
  std::vector<double> col_means_;
};

// Penalised logistic classifier trained via proximal coordinate Newton steps.
class penalised_logistic_classifier
{
public:
  static constexpr bool supports_nominal_columns = false;
  static constexpr bool supports_missing_entries = true;

  explicit penalised_logistic_classifier (
    with_regularisation reg = {0.0, 1.0e-2, 0.0},
    std::size_t max_iterations = 150) noexcept
    : reg_ (reg),
      max_iterations_ (max_iterations)
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
    return {{"l1", reg_.l1, 0.0, 100.0}, {"l2", reg_.l2, 0.0, 100.0}};
  }

  bool
  apply_parameter (const std::string &name, double val)
  {
    if (name == "l1")
      {
        reg_.l1 = std::max (0.0, val);
        return true;
      }
    if (name == "l2")
      {
        reg_.l2 = std::max (0.0, val);
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
    if (n == 0 || d == 0)
      {
        calibrated_ = false;
        return;
      }
    if (targets.size () != n)
      {
        throw std::invalid_argument (
          "bellows::penalised_logistic_classifier::calibrate: target size mismatch");
      }

    feature_count_ = d;
    long max_cls = 1;
    for (std::size_t r = 0; r < n; ++r)
      {
        max_cls = std::max (max_cls, std::lround (targets[r]));
      }
    class_count_ = static_cast<std::size_t> (std::max (2L, max_cls + 1));
    std::size_t num_models = (class_count_ <= 2) ? 1 : class_count_;

    col_means_.assign (d, 0.0);
    weights_.assign (num_models * d, 0.0);
    intercepts_.assign (num_models, 0.0);
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

    std::vector<double> x_mat (n * d, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = data.at (r, c);
            x_mat[r * d + c] = is_missing (v) ? col_means_[c] : v;
          }
      }

    double n_dbl = static_cast<double> (n);
    for (std::size_t m = 0; m < num_models; ++m)
      {
        std::size_t pos_class = (num_models == 1) ? 1 : m;
        std::vector<double> logits (n, 0.0);
        double *w_ptr = weights_.data () + m * d;

        for (std::size_t iter = 0; iter < max_iterations_; ++iter)
          {
            double max_delta = 0.0;
            double g_b = 0.0;
            double h_b = 0.0;
            for (std::size_t r = 0; r < n; ++r)
              {
                double p =
                  1.0
                  / (1.0 + std::exp (-std::clamp (logits[r], -20.0, 20.0)));
                double y =
                  (static_cast<std::size_t> (std::lround (targets[r]))
                    == pos_class)
                    ? 1.0
                    : 0.0;
                g_b += (p - y);
                h_b += std::max (p * (1.0 - p), 1.0e-5);
              }
            double step_b = -g_b / (h_b + 1.0e-6);
            intercepts_[m] += step_b;
            for (std::size_t r = 0; r < n; ++r)
              {
                logits[r] += step_b;
              }
            max_delta = std::max (max_delta, std::abs (step_b));

            for (std::size_t c = 0; c < d; ++c)
              {
                double grad = 0.0;
                double hess = 0.0;
                for (std::size_t r = 0; r < n; ++r)
                  {
                    double p =
                      1.0
                      / (1.0 + std::exp (-std::clamp (logits[r], -20.0, 20.0)));
                    double y =
                      (static_cast<std::size_t> (std::lround (targets[r]))
                        == pos_class)
                        ? 1.0
                        : 0.0;
                    double x_rc = x_mat[r * d + c];
                    grad += (p - y) * x_rc;
                    hess += std::max (p * (1.0 - p), 1.0e-5) * x_rc * x_rc;
                  }
                if (hess <= 1.0e-12)
                  {
                    continue;
                  }
                double old_w = w_ptr[c];
                double z = hess * old_w - grad;
                double shrunk =
                  newton_regulariser::soft_threshold (z, reg_.l1 * n_dbl);
                double new_w = shrunk / (hess + reg_.l2 * n_dbl);
                double diff = new_w - old_w;
                if (std::abs (diff) > 0.0)
                  {
                    for (std::size_t r = 0; r < n; ++r)
                      {
                        logits[r] += diff * x_mat[r * d + c];
                      }
                    w_ptr[c] = new_w;
                    max_delta = std::max (max_delta, std::abs (diff));
                  }
              }
            if (max_delta < 1.0e-6)
              {
                break;
              }
          }
      }
    calibrated_ = true;
  }

  std::vector<double>
  probabilities_one (std::span<const double> row) const
  {
    std::size_t k_cols = std::max<std::size_t> (2, class_count_);
    if (!calibrated_ || feature_count_ == 0)
      {
        return std::vector<double> (k_cols, 1.0 / static_cast<double> (k_cols));
      }
    std::size_t d = std::min (row.size (), feature_count_);
    if (class_count_ <= 2)
      {
        double z = intercepts_.empty () ? 0.0 : intercepts_[0];
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = is_missing (row[c]) ? col_means_[c] : row[c];
            z += weights_[c] * v;
          }
        double p1 = 1.0 / (1.0 + std::exp (-std::clamp (z, -20.0, 20.0)));
        return {1.0 - p1, p1};
      }

    std::vector<double> logits (class_count_, 0.0);
    for (std::size_t m = 0; m < class_count_; ++m)
      {
        double z = intercepts_[m];
        const double *w_ptr = weights_.data () + m * feature_count_;
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = is_missing (row[c]) ? col_means_[c] : row[c];
            z += w_ptr[c] * v;
          }
        logits[m] = z;
      }
    double max_z = *std::max_element (logits.begin (), logits.end ());
    std::vector<double> probs (class_count_, 0.0);
    double sum = 0.0;
    for (std::size_t m = 0; m < class_count_; ++m)
      {
        probs[m] = std::exp (std::clamp (logits[m] - max_z, -25.0, 25.0));
        sum += probs[m];
      }
    for (std::size_t m = 0; m < class_count_; ++m)
      {
        probs[m] /= sum;
      }
    return probs;
  }

  double
  evaluate_one (std::span<const double> row) const noexcept
  {
    auto probs = probabilities_one (row);
    auto best = std::max_element (probs.begin (), probs.end ());
    return static_cast<double> (std::distance (probs.begin (), best));
  }

  template <tabular_source Data>
  std::vector<double>
  evaluate (const Data &data) const
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    std::vector<double> preds (n, 0.0);
    std::vector<double> buf (d, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            buf[c] = data.at (r, c);
          }
        preds[r] = evaluate_one (buf);
      }
    return preds;
  }

  template <tabular_source Data>
  data_table
  evaluate_beliefs (const Data &data) const
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    std::size_t k_cols = std::max<std::size_t> (2, class_count_);
    data_table beliefs (n, k_cols);
    std::vector<double> buf (d, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            buf[c] = data.at (r, c);
          }
        auto probs = probabilities_one (buf);
        for (std::size_t k = 0; k < k_cols; ++k)
          {
            beliefs.at (r, k) = probs[k];
          }
      }
    return beliefs;
  }

  void
  write_state (std::ostream &out) const
  {
    std::uint8_t cal = calibrated_ ? 1 : 0;
    out.write (reinterpret_cast<const char *> (&cal), sizeof (cal));
    out.write (reinterpret_cast<const char *> (&feature_count_),
                sizeof (feature_count_));
    out.write (reinterpret_cast<const char *> (&class_count_),
                sizeof (class_count_));
    std::size_t ni = intercepts_.size ();
    out.write (reinterpret_cast<const char *> (&ni), sizeof (ni));
    for (double b : intercepts_)
      {
        out.write (reinterpret_cast<const char *> (&b), sizeof (double));
      }
    std::size_t nw = weights_.size ();
    out.write (reinterpret_cast<const char *> (&nw), sizeof (nw));
    for (double w : weights_)
      {
        out.write (reinterpret_cast<const char *> (&w), sizeof (double));
      }
    std::size_t nm = col_means_.size ();
    out.write (reinterpret_cast<const char *> (&nm), sizeof (nm));
    for (double m : col_means_)
      {
        out.write (reinterpret_cast<const char *> (&m), sizeof (double));
      }
  }

  void
  read_state (std::istream &in)
  {
    std::uint8_t cal = 0;
    in.read (reinterpret_cast<char *> (&cal), sizeof (cal));
    calibrated_ = (cal != 0);
    in.read (reinterpret_cast<char *> (&feature_count_),
              sizeof (feature_count_));
    in.read (reinterpret_cast<char *> (&class_count_), sizeof (class_count_));
    std::size_t ni = 0;
    in.read (reinterpret_cast<char *> (&ni), sizeof (ni));
    intercepts_.resize (ni);
    for (std::size_t i = 0; i < ni; ++i)
      {
        in.read (reinterpret_cast<char *> (&intercepts_[i]), sizeof (double));
      }
    std::size_t nw = 0;
    in.read (reinterpret_cast<char *> (&nw), sizeof (nw));
    weights_.resize (nw);
    for (std::size_t i = 0; i < nw; ++i)
      {
        in.read (reinterpret_cast<char *> (&weights_[i]), sizeof (double));
      }
    std::size_t nm = 0;
    in.read (reinterpret_cast<char *> (&nm), sizeof (nm));
    col_means_.resize (nm);
    for (std::size_t i = 0; i < nm; ++i)
      {
        in.read (reinterpret_cast<char *> (&col_means_[i]), sizeof (double));
      }
  }

private:
  with_regularisation reg_;
  std::size_t max_iterations_ = 150;
  bool calibrated_ = false;
  std::size_t feature_count_ = 0;
  std::size_t class_count_ = 2;
  std::vector<double> intercepts_;
  std::vector<double> weights_;
  std::vector<double> col_means_;
};

} // namespace bellows

#endif // BELLOWS_ESTIMATORS_LINEAR_HPP
