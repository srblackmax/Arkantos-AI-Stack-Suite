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

#ifndef BELLOWS_PREPROCESS_SELECTION_HPP
#define BELLOWS_PREPROCESS_SELECTION_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>

namespace bellows
{

// Filters out low-dispersion (near-constant) feature columns.
class dispersion_filter
{
public:
  static constexpr bool supports_nominal_columns = true;
  static constexpr bool supports_missing_entries = true;

  explicit dispersion_filter (double min_variance = 1.0e-5) noexcept
    : min_variance_ (min_variance)
  {
  }

  bool
  is_calibrated () const noexcept
  {
    return calibrated_;
  }

  template <tabular_source Data>
  void
  calibrate (const Data &data)
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    retained_columns_.clear ();
    variances_.assign (d, 0.0);

    for (std::size_t c = 0; c < d; ++c)
      {
        double sum = 0.0;
        std::size_t count = 0;
        for (std::size_t r = 0; r < n; ++r)
          {
            double v = data.at (r, c);
            if (!is_missing (v))
              {
                sum += v;
                ++count;
              }
          }
        if (count <= 1)
          {
            continue;
          }
        double mean = sum / static_cast<double> (count);
        double sq = 0.0;
        for (std::size_t r = 0; r < n; ++r)
          {
            double v = data.at (r, c);
            if (!is_missing (v))
              {
                double diff = v - mean;
                sq += diff * diff;
              }
          }
        double var = sq / static_cast<double> (count);
        variances_[c] = var;
        if (var >= min_variance_)
          {
            retained_columns_.push_back (c);
          }
      }

    if (retained_columns_.empty () && d > 0)
      {
        retained_columns_.push_back (0);
      }
    calibrated_ = true;
  }

  template <tabular_source Data, target_sequence Targets>
  void
  calibrate (const Data &data, const Targets &)
  {
    calibrate (data);
  }

  template <tabular_source Data>
  data_table
  project (const Data &data) const
  {
    std::size_t n = data.row_count ();
    std::size_t k = retained_columns_.size ();
    data_table result (n, k);
    for (std::size_t j = 0; j < k; ++j)
      {
        std::size_t src = retained_columns_[j];
        result.set_scale (j, data.scale (src), data.cardinality (src));
        for (std::size_t r = 0; r < n; ++r)
          {
            result.set_cell (r, j, data.at (r, src));
          }
      }
    return result;
  }

  const std::vector<std::size_t> &
  retained_columns () const noexcept
  {
    return retained_columns_;
  }

  void
  write_state (std::ostream &out) const
  {
    std::uint8_t cal = calibrated_ ? 1 : 0;
    out.write (reinterpret_cast<const char *> (&cal), sizeof (cal));
    out.write (reinterpret_cast<const char *> (&min_variance_),
                sizeof (min_variance_));
    std::size_t nv = variances_.size ();
    out.write (reinterpret_cast<const char *> (&nv), sizeof (nv));
    if (nv > 0)
      {
        out.write (reinterpret_cast<const char *> (variances_.data ()),
                    static_cast<std::streamsize> (nv * sizeof (double)));
      }
    std::size_t nr = retained_columns_.size ();
    out.write (reinterpret_cast<const char *> (&nr), sizeof (nr));
    if (nr > 0)
      {
        out.write (reinterpret_cast<const char *> (retained_columns_.data ()),
                    static_cast<std::streamsize> (nr * sizeof (std::size_t)));
      }
  }

  void
  read_state (std::istream &in)
  {
    std::uint8_t cal = 0;
    in.read (reinterpret_cast<char *> (&cal), sizeof (cal));
    calibrated_ = (cal != 0);
    in.read (reinterpret_cast<char *> (&min_variance_),
              sizeof (min_variance_));
    std::size_t nv = 0;
    in.read (reinterpret_cast<char *> (&nv), sizeof (nv));
    variances_.assign (nv, 0.0);
    if (nv > 0)
      {
        in.read (reinterpret_cast<char *> (variances_.data ()),
                  static_cast<std::streamsize> (nv * sizeof (double)));
      }
    std::size_t nr = 0;
    in.read (reinterpret_cast<char *> (&nr), sizeof (nr));
    retained_columns_.assign (nr, 0);
    if (nr > 0)
      {
        in.read (reinterpret_cast<char *> (retained_columns_.data ()),
                  static_cast<std::streamsize> (nr * sizeof (std::size_t)));
      }
  }

private:
  double min_variance_ = 1.0e-5;
  bool calibrated_ = false;
  std::vector<double> variances_;
  std::vector<std::size_t> retained_columns_;
};

// Selects top-K feature columns ranked by target correlation relevance.
class relevance_selector
{
public:
  static constexpr bool supports_nominal_columns = true;
  static constexpr bool supports_missing_entries = true;

  explicit relevance_selector (std::size_t retain_count = 5) noexcept
    : retain_count_ (std::max<std::size_t> (1, retain_count))
  {
  }

  bool
  is_calibrated () const noexcept
  {
    return calibrated_;
  }

  template <tabular_source Data, target_sequence Targets>
  void
  calibrate (const Data &data, const Targets &targets)
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    if (targets.size () != n)
      {
        throw std::invalid_argument (
          "bellows::relevance_selector::calibrate: target size mismatch");
      }
    scores_.assign (d, 0.0);

    double mean_y = 0.0;
    for (std::size_t r = 0; r < n; ++r)
      {
        mean_y += targets[r];
      }
    mean_y /= static_cast<double> (std::max<std::size_t> (1, n));

    for (std::size_t c = 0; c < d; ++c)
      {
        double sum_x = 0.0;
        std::size_t cnt = 0;
        for (std::size_t r = 0; r < n; ++r)
          {
            double v = data.at (r, c);
            if (!is_missing (v))
              {
                sum_x += v;
                ++cnt;
              }
          }
        if (cnt <= 1)
          {
            continue;
          }
        double mean_x = sum_x / static_cast<double> (cnt);
        double cov = 0.0;
        double var_x = 0.0;
        double var_y = 0.0;
        for (std::size_t r = 0; r < n; ++r)
          {
            double v = data.at (r, c);
            if (!is_missing (v))
              {
                double dx = v - mean_x;
                double dy = targets[r] - mean_y;
                cov += dx * dy;
                var_x += dx * dx;
                var_y += dy * dy;
              }
          }
        double denom = std::sqrt (var_x * var_y);
        scores_[c] = denom > 1.0e-12 ? std::abs (cov / denom) : 0.0;
      }

    std::vector<std::size_t> order (d);
    std::iota (order.begin (), order.end (), 0);
    std::sort (order.begin (), order.end (),
                [&] (std::size_t a, std::size_t b)
                {
                  return scores_[a] > scores_[b];
                });

    std::size_t k = std::min (d, retain_count_);
    retained_columns_.assign (order.begin (),
                              order.begin () + static_cast<std::ptrdiff_t> (k));
    std::sort (retained_columns_.begin (), retained_columns_.end ());
    calibrated_ = true;
  }

  template <tabular_source Data>
  data_table
  project (const Data &data) const
  {
    std::size_t n = data.row_count ();
    std::size_t k = retained_columns_.size ();
    data_table result (n, k);
    for (std::size_t j = 0; j < k; ++j)
      {
        std::size_t src = retained_columns_[j];
        result.set_scale (j, data.scale (src), data.cardinality (src));
        for (std::size_t r = 0; r < n; ++r)
          {
            result.set_cell (r, j, data.at (r, src));
          }
      }
    return result;
  }

  const std::vector<std::size_t> &
  retained_columns () const noexcept
  {
    return retained_columns_;
  }

  const std::vector<double> &
  scores () const noexcept
  {
    return scores_;
  }

  void
  write_state (std::ostream &out) const
  {
    std::uint8_t cal = calibrated_ ? 1 : 0;
    out.write (reinterpret_cast<const char *> (&cal), sizeof (cal));
    out.write (reinterpret_cast<const char *> (&retain_count_),
                sizeof (retain_count_));
    std::size_t ns = scores_.size ();
    out.write (reinterpret_cast<const char *> (&ns), sizeof (ns));
    if (ns > 0)
      {
        out.write (reinterpret_cast<const char *> (scores_.data ()),
                    static_cast<std::streamsize> (ns * sizeof (double)));
      }
    std::size_t nr = retained_columns_.size ();
    out.write (reinterpret_cast<const char *> (&nr), sizeof (nr));
    if (nr > 0)
      {
        out.write (reinterpret_cast<const char *> (retained_columns_.data ()),
                    static_cast<std::streamsize> (nr * sizeof (std::size_t)));
      }
  }

  void
  read_state (std::istream &in)
  {
    std::uint8_t cal = 0;
    in.read (reinterpret_cast<char *> (&cal), sizeof (cal));
    calibrated_ = (cal != 0);
    in.read (reinterpret_cast<char *> (&retain_count_),
              sizeof (retain_count_));
    std::size_t ns = 0;
    in.read (reinterpret_cast<char *> (&ns), sizeof (ns));
    scores_.assign (ns, 0.0);
    if (ns > 0)
      {
        in.read (reinterpret_cast<char *> (scores_.data ()),
                  static_cast<std::streamsize> (ns * sizeof (double)));
      }
    std::size_t nr = 0;
    in.read (reinterpret_cast<char *> (&nr), sizeof (nr));
    retained_columns_.assign (nr, 0);
    if (nr > 0)
      {
        in.read (reinterpret_cast<char *> (retained_columns_.data ()),
                  static_cast<std::streamsize> (nr * sizeof (std::size_t)));
      }
  }

private:
  std::size_t retain_count_ = 5;
  bool calibrated_ = false;
  std::vector<double> scores_;
  std::vector<std::size_t> retained_columns_;
};

} // namespace bellows

#endif // BELLOWS_PREPROCESS_SELECTION_HPP
