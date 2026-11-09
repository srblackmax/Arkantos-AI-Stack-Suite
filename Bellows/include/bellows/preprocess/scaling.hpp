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

#ifndef BELLOWS_PREPROCESS_SCALING_HPP
#define BELLOWS_PREPROCESS_SCALING_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>

namespace bellows
{

// Centers real-valued columns to zero mean and scales to unit variance.
class moment_normaliser
{
public:
  static constexpr bool supports_nominal_columns = true;
  static constexpr bool supports_missing_entries = true;

  moment_normaliser () = default;

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
    means_.assign (d, 0.0);
    scales_.assign (d, 1.0);
    scales_meta_.assign (d, feature_scale::real);
    cards_meta_.assign (d, 0);

    for (std::size_t c = 0; c < d; ++c)
      {
        scales_meta_[c] = data.scale (c);
        cards_meta_[c] = data.cardinality (c);
        if (data.scale (c) == feature_scale::nominal)
          {
            continue;
          }

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
        if (count == 0)
          {
            continue;
          }
        double mean = sum / static_cast<double> (count);
        means_[c] = mean;

        double sq_sum = 0.0;
        for (std::size_t r = 0; r < n; ++r)
          {
            double v = data.at (r, c);
            if (!is_missing (v))
              {
                double diff = v - mean;
                sq_sum += diff * diff;
              }
          }
        double std_dev = std::sqrt (sq_sum / static_cast<double> (count));
        scales_[c] = std_dev > 1.0e-12 ? std_dev : 1.0;
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
    std::size_t d = data.column_count ();
    data_table result (n, d);
    for (std::size_t c = 0; c < d; ++c)
      {
        feature_scale col_scale =
          (c < scales_meta_.size ()
            && scales_meta_[c] == feature_scale::nominal)
            ? feature_scale::nominal
            : data.scale (c);
        std::size_t col_card =
          (c < cards_meta_.size () && cards_meta_[c] > 0)
            ? cards_meta_[c]
            : data.cardinality (c);
        result.set_scale (c, col_scale, col_card);
        bool skip = (col_scale == feature_scale::nominal
                      || c >= means_.size ());
        for (std::size_t r = 0; r < n; ++r)
          {
            double v = data.at (r, c);
            if (skip || is_missing (v))
              {
                result.set_cell (r, c, v);
              }
            else
              {
                result.set_cell (r, c, (v - means_[c]) / scales_[c]);
              }
          }
      }
    return result;
  }

  const std::vector<double> &
  means () const noexcept
  {
    return means_;
  }

  const std::vector<double> &
  scales () const noexcept
  {
    return scales_;
  }

  void
  write_state (std::ostream &out) const
  {
    std::uint8_t cal = calibrated_ ? 1 : 0;
    out.write (reinterpret_cast<const char *> (&cal), sizeof (cal));
    std::size_t d = means_.size ();
    out.write (reinterpret_cast<const char *> (&d), sizeof (d));
    if (d > 0)
      {
        out.write (reinterpret_cast<const char *> (means_.data ()),
                    static_cast<std::streamsize> (d * sizeof (double)));
        out.write (reinterpret_cast<const char *> (scales_.data ()),
                    static_cast<std::streamsize> (d * sizeof (double)));
        for (std::size_t c = 0; c < d; ++c)
          {
            std::uint8_t sc = static_cast<std::uint8_t> (scales_meta_[c]);
            out.write (reinterpret_cast<const char *> (&sc), sizeof (sc));
            std::size_t card = cards_meta_[c];
            out.write (reinterpret_cast<const char *> (&card), sizeof (card));
          }
      }
  }

  void
  read_state (std::istream &in)
  {
    std::uint8_t cal = 0;
    in.read (reinterpret_cast<char *> (&cal), sizeof (cal));
    calibrated_ = (cal != 0);
    std::size_t d = 0;
    in.read (reinterpret_cast<char *> (&d), sizeof (d));
    means_.assign (d, 0.0);
    scales_.assign (d, 1.0);
    scales_meta_.assign (d, feature_scale::real);
    cards_meta_.assign (d, 0);
    if (d > 0)
      {
        in.read (reinterpret_cast<char *> (means_.data ()),
                  static_cast<std::streamsize> (d * sizeof (double)));
        in.read (reinterpret_cast<char *> (scales_.data ()),
                  static_cast<std::streamsize> (d * sizeof (double)));
        for (std::size_t c = 0; c < d; ++c)
          {
            std::uint8_t sc = 0;
            in.read (reinterpret_cast<char *> (&sc), sizeof (sc));
            scales_meta_[c] = static_cast<feature_scale> (sc);
            std::size_t card = 0;
            in.read (reinterpret_cast<char *> (&card), sizeof (card));
            cards_meta_[c] = card;
          }
      }
  }

private:
  bool calibrated_ = false;
  std::vector<double> means_;
  std::vector<double> scales_;
  std::vector<feature_scale> scales_meta_;
  std::vector<std::size_t> cards_meta_;
};

// Linearly maps real-valued columns onto a bounded interval [lower, upper].
class interval_rescaler
{
public:
  static constexpr bool supports_nominal_columns = true;
  static constexpr bool supports_missing_entries = true;

  explicit interval_rescaler (double lower = 0.0, double upper = 1.0) noexcept
    : lower_ (lower),
      upper_ (upper)
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
    mins_.assign (d, 0.0);
    spans_.assign (d, 1.0);
    scales_meta_.assign (d, feature_scale::real);
    cards_meta_.assign (d, 0);

    for (std::size_t c = 0; c < d; ++c)
      {
        scales_meta_[c] = data.scale (c);
        cards_meta_[c] = data.cardinality (c);
        if (data.scale (c) == feature_scale::nominal)
          {
            continue;
          }
        double lo = std::numeric_limits<double>::infinity ();
        double hi = -std::numeric_limits<double>::infinity ();
        for (std::size_t r = 0; r < n; ++r)
          {
            double v = data.at (r, c);
            if (!is_missing (v))
              {
                lo = std::min (lo, v);
                hi = std::max (hi, v);
              }
          }
        if (std::isfinite (lo) && std::isfinite (hi))
          {
            mins_[c] = lo;
            double span = hi - lo;
            spans_[c] = span > 1.0e-12 ? span : 1.0;
          }
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
    std::size_t d = data.column_count ();
    data_table result (n, d);
    double width = upper_ - lower_;
    for (std::size_t c = 0; c < d; ++c)
      {
        feature_scale col_scale =
          (c < scales_meta_.size ()
            && scales_meta_[c] == feature_scale::nominal)
            ? feature_scale::nominal
            : data.scale (c);
        std::size_t col_card =
          (c < cards_meta_.size () && cards_meta_[c] > 0)
            ? cards_meta_[c]
            : data.cardinality (c);
        result.set_scale (c, col_scale, col_card);
        bool skip = (col_scale == feature_scale::nominal
                      || c >= mins_.size ());
        for (std::size_t r = 0; r < n; ++r)
          {
            double v = data.at (r, c);
            if (skip || is_missing (v))
              {
                result.set_cell (r, c, v);
              }
            else
              {
                double unit = (v - mins_[c]) / spans_[c];
                result.set_cell (r, c, lower_ + unit * width);
              }
          }
      }
    return result;
  }

  void
  write_state (std::ostream &out) const
  {
    std::uint8_t cal = calibrated_ ? 1 : 0;
    out.write (reinterpret_cast<const char *> (&cal), sizeof (cal));
    out.write (reinterpret_cast<const char *> (&lower_), sizeof (lower_));
    out.write (reinterpret_cast<const char *> (&upper_), sizeof (upper_));
    std::size_t d = mins_.size ();
    out.write (reinterpret_cast<const char *> (&d), sizeof (d));
    if (d > 0)
      {
        out.write (reinterpret_cast<const char *> (mins_.data ()),
                    static_cast<std::streamsize> (d * sizeof (double)));
        out.write (reinterpret_cast<const char *> (spans_.data ()),
                    static_cast<std::streamsize> (d * sizeof (double)));
        for (std::size_t c = 0; c < d; ++c)
          {
            std::uint8_t sc = static_cast<std::uint8_t> (scales_meta_[c]);
            out.write (reinterpret_cast<const char *> (&sc), sizeof (sc));
            std::size_t card = cards_meta_[c];
            out.write (reinterpret_cast<const char *> (&card), sizeof (card));
          }
      }
  }

  void
  read_state (std::istream &in)
  {
    std::uint8_t cal = 0;
    in.read (reinterpret_cast<char *> (&cal), sizeof (cal));
    calibrated_ = (cal != 0);
    in.read (reinterpret_cast<char *> (&lower_), sizeof (lower_));
    in.read (reinterpret_cast<char *> (&upper_), sizeof (upper_));
    std::size_t d = 0;
    in.read (reinterpret_cast<char *> (&d), sizeof (d));
    mins_.assign (d, 0.0);
    spans_.assign (d, 1.0);
    scales_meta_.assign (d, feature_scale::real);
    cards_meta_.assign (d, 0);
    if (d > 0)
      {
        in.read (reinterpret_cast<char *> (mins_.data ()),
                  static_cast<std::streamsize> (d * sizeof (double)));
        in.read (reinterpret_cast<char *> (spans_.data ()),
                  static_cast<std::streamsize> (d * sizeof (double)));
        for (std::size_t c = 0; c < d; ++c)
          {
            std::uint8_t sc = 0;
            in.read (reinterpret_cast<char *> (&sc), sizeof (sc));
            scales_meta_[c] = static_cast<feature_scale> (sc);
            std::size_t card = 0;
            in.read (reinterpret_cast<char *> (&card), sizeof (card));
            cards_meta_[c] = card;
          }
      }
  }

private:
  double lower_ = 0.0;
  double upper_ = 1.0;
  bool calibrated_ = false;
  std::vector<double> mins_;
  std::vector<double> spans_;
  std::vector<feature_scale> scales_meta_;
  std::vector<std::size_t> cards_meta_;
};

} // namespace bellows

#endif // BELLOWS_PREPROCESS_SCALING_HPP
