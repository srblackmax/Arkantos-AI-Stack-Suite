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

#ifndef BELLOWS_ESTIMATORS_DENSITY_HPP
#define BELLOWS_ESTIMATORS_DENSITY_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <span>
#include <string>
#include <vector>

#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>
#include <bellows/core/parameters.hpp>

namespace bellows
{

// Parametric Gaussian class-conditional density classifier.
class gaussian_density_classifier
{
public:
  static constexpr bool supports_nominal_columns = false;
  static constexpr bool supports_missing_entries = true;

  explicit gaussian_density_classifier (double variance_floor = 1.0e-6) noexcept
    : variance_floor_ (variance_floor)
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
    return {{"variance_floor", variance_floor_, 1.0e-12, 1.0}};
  }

  bool
  apply_parameter (const std::string &name, double val)
  {
    if (name == "variance_floor")
      {
        variance_floor_ = std::max (1.0e-12, val);
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
          "bellows::gaussian_density_classifier::calibrate: target size mismatch");
      }

    feature_count_ = d;
    long max_c = 1;
    for (std::size_t r = 0; r < n; ++r)
      {
        max_c = std::max (max_c, std::lround (targets[r]));
      }
    class_count_ = static_cast<std::size_t> (max_c + 1);

    log_priors_.assign (class_count_, -std::log (static_cast<double> (class_count_)));
    means_.assign (class_count_ * d, 0.0);
    vars_.assign (class_count_ * d, 1.0);

    std::vector<std::size_t> cls_counts (class_count_, 0);
    for (std::size_t r = 0; r < n; ++r)
      {
        std::size_t k = static_cast<std::size_t> (
          std::max (0L, std::lround (targets[r])));
        cls_counts[k] += 1;
      }

    for (std::size_t k = 0; k < class_count_; ++k)
      {
        double p = static_cast<double> (std::max<std::size_t> (1, cls_counts[k]))
                    / static_cast<double> (n);
        log_priors_[k] = std::log (p);
      }

    for (std::size_t k = 0; k < class_count_; ++k)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double sum = 0.0;
            std::size_t cnt = 0;
            for (std::size_t r = 0; r < n; ++r)
              {
                if (static_cast<std::size_t> (
                      std::max (0L, std::lround (targets[r])))
                    != k)
                  {
                    continue;
                  }
                double v = data.at (r, c);
                if (!is_missing (v))
                  {
                    sum += v;
                    ++cnt;
                  }
              }
            double mean = cnt > 0 ? sum / static_cast<double> (cnt) : 0.0;
            means_[k * d + c] = mean;

            double sq = 0.0;
            for (std::size_t r = 0; r < n; ++r)
              {
                if (static_cast<std::size_t> (
                      std::max (0L, std::lround (targets[r])))
                    != k)
                  {
                    continue;
                  }
                double v = data.at (r, c);
                if (!is_missing (v))
                  {
                    double diff = v - mean;
                    sq += diff * diff;
                  }
              }
            double var =
              cnt > 0 ? (sq / static_cast<double> (cnt)) + variance_floor_
                      : 1.0;
            vars_[k * d + c] = std::max (var, variance_floor_);
          }
      }
    calibrated_ = true;
  }

  double
  evaluate_one (std::span<const double> row) const
  {
    auto probs = compute_row_beliefs (row);
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
    data_table beliefs (n, class_count_);
    std::vector<double> buf (d, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            buf[c] = data.at (r, c);
          }
        auto probs = compute_row_beliefs (buf);
        for (std::size_t k = 0; k < class_count_; ++k)
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
    for (double lp : log_priors_)
      {
        out.write (reinterpret_cast<const char *> (&lp), sizeof (double));
      }
    for (double m : means_)
      {
        out.write (reinterpret_cast<const char *> (&m), sizeof (double));
      }
    for (double v : vars_)
      {
        out.write (reinterpret_cast<const char *> (&v), sizeof (double));
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
    log_priors_.resize (class_count_);
    for (std::size_t k = 0; k < class_count_; ++k)
      {
        in.read (reinterpret_cast<char *> (&log_priors_[k]), sizeof (double));
      }
    means_.resize (class_count_ * feature_count_);
    vars_.resize (class_count_ * feature_count_);
    for (std::size_t i = 0; i < means_.size (); ++i)
      {
        in.read (reinterpret_cast<char *> (&means_[i]), sizeof (double));
      }
    for (std::size_t i = 0; i < vars_.size (); ++i)
      {
        in.read (reinterpret_cast<char *> (&vars_[i]), sizeof (double));
      }
  }

private:
  std::vector<double>
  compute_row_beliefs (std::span<const double> row) const
  {
    std::vector<double> log_lik (class_count_, 0.0);
    std::size_t d = std::min (row.size (), feature_count_);
    for (std::size_t k = 0; k < class_count_; ++k)
      {
        double ll = log_priors_[k];
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = row[c];
            if (is_missing (v))
              {
                continue;
              }
            double mu = means_[k * feature_count_ + c];
            double var = vars_[k * feature_count_ + c];
            double diff = v - mu;
            ll -= 0.5 * (std::log (2.0 * std::numbers::pi * var)
                          + (diff * diff) / var);
          }
        log_lik[k] = ll;
      }

    double max_ll = *std::max_element (log_lik.begin (), log_lik.end ());
    double sum = 0.0;
    std::vector<double> probs (class_count_, 0.0);
    for (std::size_t k = 0; k < class_count_; ++k)
      {
        probs[k] = std::exp (std::clamp (log_lik[k] - max_ll, -30.0, 30.0));
        sum += probs[k];
      }
    for (std::size_t k = 0; k < class_count_; ++k)
      {
        probs[k] /= sum;
      }
    return probs;
  }

  double variance_floor_ = 1.0e-6;
  bool calibrated_ = false;
  std::size_t feature_count_ = 0;
  std::size_t class_count_ = 2;
  std::vector<double> log_priors_;
  std::vector<double> means_;
  std::vector<double> vars_;
};

} // namespace bellows

#endif // BELLOWS_ESTIMATORS_DENSITY_HPP
