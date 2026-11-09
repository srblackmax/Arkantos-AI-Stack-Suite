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

#ifndef BELLOWS_PREPROCESS_MENDING_HPP
#define BELLOWS_PREPROCESS_MENDING_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>

#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>

namespace bellows
{

// Statistical summary rule used to mend missing entries in a column.
enum class mending_strategy : std::uint8_t
{
  mean = 0,
  median = 1,
  most_frequent = 2
};

// Transformer that replaces missing cells with calibrated column statistics.
class column_mender
{
public:
  static constexpr bool supports_nominal_columns = true;
  static constexpr bool supports_missing_entries = true;

  explicit column_mender (
    mending_strategy strategy = mending_strategy::median) noexcept
    : strategy_ (strategy)
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
    fill_values_.assign (d, 0.0);

    for (std::size_t c = 0; c < d; ++c)
      {
        std::vector<double> present;
        present.reserve (n);
        for (std::size_t r = 0; r < n; ++r)
          {
            double v = data.at (r, c);
            if (!is_missing (v))
              {
                present.push_back (v);
              }
          }
        if (present.empty ())
          {
            fill_values_[c] = 0.0;
            continue;
          }

        mending_strategy effective =
          data.scale (c) == feature_scale::nominal
            ? mending_strategy::most_frequent
            : strategy_;

        if (effective == mending_strategy::mean)
          {
            double sum = 0.0;
            for (double v : present)
              {
                sum += v;
              }
            fill_values_[c] = sum / static_cast<double> (present.size ());
          }
        else if (effective == mending_strategy::median)
          {
            std::sort (present.begin (), present.end ());
            std::size_t mid = present.size () / 2;
            if (present.size () % 2 == 1)
              {
                fill_values_[c] = present[mid];
              }
            else
              {
                fill_values_[c] = 0.5 * (present[mid - 1] + present[mid]);
              }
          }
        else
          {
            std::map<long, std::size_t> counts;
            for (double v : present)
              {
                counts[std::lround (v)] += 1;
              }
            long best_val = 0;
            std::size_t best_cnt = 0;
            for (const auto &kv : counts)
              {
                if (kv.second > best_cnt)
                  {
                    best_cnt = kv.second;
                    best_val = kv.first;
                  }
              }
            fill_values_[c] = static_cast<double> (best_val);
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
    for (std::size_t c = 0; c < d; ++c)
      {
        result.set_scale (c, data.scale (c), data.cardinality (c));
        double replacement = c < fill_values_.size () ? fill_values_[c] : 0.0;
        for (std::size_t r = 0; r < n; ++r)
          {
            double v = data.at (r, c);
            result.at (r, c) = is_missing (v) ? replacement : v;
          }
      }
    return result;
  }

  const std::vector<double> &
  fill_values () const noexcept
  {
    return fill_values_;
  }

  void
  write_state (std::ostream &out) const
  {
    std::uint8_t cal = calibrated_ ? 1 : 0;
    std::uint8_t strat = static_cast<std::uint8_t> (strategy_);
    out.write (reinterpret_cast<const char *> (&cal), sizeof (cal));
    out.write (reinterpret_cast<const char *> (&strat), sizeof (strat));
    std::size_t d = fill_values_.size ();
    out.write (reinterpret_cast<const char *> (&d), sizeof (d));
    if (d > 0)
      {
        out.write (reinterpret_cast<const char *> (fill_values_.data ()),
                    static_cast<std::streamsize> (d * sizeof (double)));
      }
  }

  void
  read_state (std::istream &in)
  {
    std::uint8_t cal = 0;
    std::uint8_t strat = 0;
    in.read (reinterpret_cast<char *> (&cal), sizeof (cal));
    in.read (reinterpret_cast<char *> (&strat), sizeof (strat));
    calibrated_ = (cal != 0);
    strategy_ = static_cast<mending_strategy> (strat);
    std::size_t d = 0;
    in.read (reinterpret_cast<char *> (&d), sizeof (d));
    fill_values_.assign (d, 0.0);
    if (d > 0)
      {
        in.read (reinterpret_cast<char *> (fill_values_.data ()),
                  static_cast<std::streamsize> (d * sizeof (double)));
      }
  }

private:
  mending_strategy strategy_ = mending_strategy::median;
  bool calibrated_ = false;
  std::vector<double> fill_values_;
};

} // namespace bellows

#endif // BELLOWS_PREPROCESS_MENDING_HPP
