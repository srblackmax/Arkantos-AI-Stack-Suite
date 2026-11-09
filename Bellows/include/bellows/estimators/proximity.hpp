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

#ifndef BELLOWS_ESTIMATORS_PROXIMITY_HPP
#define BELLOWS_ESTIMATORS_PROXIMITY_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>
#include <bellows/core/parameters.hpp>

namespace bellows
{

// Task mode for the non-parametric proximity learner.
enum class proximity_mode : std::uint8_t
{
  regression = 0,
  classification = 1
};

// Non-parametric k-nearest-neighbour estimator with missing/nominal support.
class proximity_learner
{
public:
  static constexpr bool supports_nominal_columns = true;
  static constexpr bool supports_missing_entries = true;

  explicit proximity_learner (
    std::size_t neighbour_count = 5,
    proximity_mode mode = proximity_mode::classification,
    bool inverse_distance_weighting = true) noexcept
    : neighbour_count_ (std::max<std::size_t> (1, neighbour_count)),
      mode_ (mode),
      inverse_distance_weighting_ (inverse_distance_weighting)
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
    return {{"neighbour_count",
              static_cast<double> (neighbour_count_),
              1.0,
              50.0}};
  }

  bool
  apply_parameter (const std::string &name, double val)
  {
    if (name == "neighbour_count")
      {
        neighbour_count_ =
          static_cast<std::size_t> (std::max (1L, std::lround (val)));
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
    if (targets.size () != n)
      {
        throw std::invalid_argument (
          "bellows::proximity_learner::calibrate: target size mismatch");
      }
    exemplars_ = data_table (n, d);
    for (std::size_t c = 0; c < d; ++c)
      {
        exemplars_.set_scale (c, data.scale (c), data.cardinality (c));
        for (std::size_t r = 0; r < n; ++r)
          {
            exemplars_.set_cell (r, c, data.at (r, c));
          }
      }
    targets_.resize (n);
    long max_c = 1;
    for (std::size_t r = 0; r < n; ++r)
      {
        targets_[r] = targets[r];
        if (mode_ == proximity_mode::classification)
          {
            max_c = std::max (max_c, std::lround (targets[r]));
          }
      }
    class_count_ = static_cast<std::size_t> (max_c + 1);
    calibrated_ = (n > 0 && d > 0);
  }

  double
  evaluate_one (std::span<const double> query) const
  {
    if (!calibrated_)
      {
        return 0.0;
      }
    auto neighbours = find_nearest (query);
    if (mode_ == proximity_mode::regression)
      {
        double num = 0.0;
        double den = 0.0;
        for (const auto &nb : neighbours)
          {
            double w = inverse_distance_weighting_
                          ? 1.0 / (nb.distance + 1.0e-6)
                          : 1.0;
            num += w * targets_[nb.index];
            den += w;
          }
        return den > 0.0 ? num / den : 0.0;
      }

    std::vector<double> votes (class_count_, 0.0);
    for (const auto &nb : neighbours)
      {
        double w =
          inverse_distance_weighting_ ? 1.0 / (nb.distance + 1.0e-6) : 1.0;
        std::size_t cls =
          static_cast<std::size_t> (std::max (0L, std::lround (targets_[nb.index])));
        if (cls < votes.size ())
          {
            votes[cls] += w;
          }
      }
    auto best = std::max_element (votes.begin (), votes.end ());
    return static_cast<double> (std::distance (votes.begin (), best));
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
    data_table beliefs (n, class_count_, 0.0);
    std::vector<double> buf (d, 0.0);

    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            buf[c] = data.at (r, c);
          }
        auto neighbours = find_nearest (buf);
        double total_w = 0.0;
        for (const auto &nb : neighbours)
          {
            double w = inverse_distance_weighting_
                          ? 1.0 / (nb.distance + 1.0e-6)
                          : 1.0;
            std::size_t cls = static_cast<std::size_t> (
              std::max (0L, std::lround (targets_[nb.index])));
            if (cls < class_count_)
              {
                beliefs.at (r, cls) += w;
                total_w += w;
              }
          }
        if (total_w > 0.0)
          {
            for (std::size_t k = 0; k < class_count_; ++k)
              {
                beliefs.at (r, k) /= total_w;
              }
          }
      }
    return beliefs;
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
    write_pod (neighbour_count_);
    std::uint8_t m_code = static_cast<std::uint8_t> (mode_);
    write_pod (m_code);
    std::uint8_t idw = inverse_distance_weighting_ ? 1 : 0;
    write_pod (idw);
    write_pod (class_count_);
    std::size_t n = exemplars_.row_count ();
    std::size_t d = exemplars_.column_count ();
    write_pod (n);
    write_pod (d);
    for (std::size_t c = 0; c < d; ++c)
      {
        std::uint8_t sc = static_cast<std::uint8_t> (exemplars_.scale (c));
        std::size_t card = exemplars_.cardinality (c);
        write_pod (sc);
        write_pod (card);
      }
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = exemplars_.at (r, c);
            write_pod (v);
          }
      }
    for (double y : targets_)
      {
        write_pod (y);
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
    read_pod (neighbour_count_);
    std::uint8_t m_code = 0;
    read_pod (m_code);
    mode_ = static_cast<proximity_mode> (m_code);
    std::uint8_t idw = 0;
    read_pod (idw);
    inverse_distance_weighting_ = (idw != 0);
    read_pod (class_count_);
    std::size_t n = 0;
    std::size_t d = 0;
    read_pod (n);
    read_pod (d);
    exemplars_ = data_table (n, d);
    for (std::size_t c = 0; c < d; ++c)
      {
        std::uint8_t sc = 0;
        std::size_t card = 0;
        read_pod (sc);
        read_pod (card);
        exemplars_.set_scale (c, static_cast<feature_scale> (sc), card);
      }
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = 0.0;
            read_pod (v);
            exemplars_.set_cell (r, c, v);
          }
      }
    targets_.resize (n);
    for (std::size_t r = 0; r < n; ++r)
      {
        read_pod (targets_[r]);
      }
  }

private:
  struct neighbour_pair
  {
    std::size_t index = 0;
    double distance = 0.0;
  };

  std::vector<neighbour_pair>
  find_nearest (std::span<const double> query) const
  {
    std::size_t n = exemplars_.row_count ();
    std::size_t d = std::min (query.size (), exemplars_.column_count ());
    std::vector<neighbour_pair> all (n);

    for (std::size_t r = 0; r < n; ++r)
      {
        double sq_sum = 0.0;
        std::size_t valid_dims = 0;
        for (std::size_t c = 0; c < d; ++c)
          {
            double qv = query[c];
            double ev = exemplars_.at (r, c);
            if (is_missing (qv) || is_missing (ev))
              {
                continue;
              }
            ++valid_dims;
            if (exemplars_.scale (c) == feature_scale::nominal)
              {
                sq_sum += (std::lround (qv) == std::lround (ev)) ? 0.0 : 1.0;
              }
            else
              {
                double diff = qv - ev;
                sq_sum += diff * diff;
              }
          }
        double scale =
          valid_dims > 0
            ? static_cast<double> (d) / static_cast<double> (valid_dims)
            : 1.0;
        all[r] = neighbour_pair{r, std::sqrt (sq_sum * scale)};
      }

    std::size_t k = std::min (n, neighbour_count_);
    std::partial_sort (all.begin (),
                        all.begin () + static_cast<std::ptrdiff_t> (k),
                        all.end (),
                        [] (const neighbour_pair &a, const neighbour_pair &b)
                        {
                          return a.distance < b.distance;
                        });
    all.resize (k);
    return all;
  }

  std::size_t neighbour_count_ = 5;
  proximity_mode mode_ = proximity_mode::classification;
  bool inverse_distance_weighting_ = true;
  bool calibrated_ = false;
  std::size_t class_count_ = 2;
  data_table exemplars_;
  std::vector<double> targets_;
};

} // namespace bellows

#endif // BELLOWS_ESTIMATORS_PROXIMITY_HPP
