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

#ifndef BELLOWS_PREPROCESS_ENCODING_HPP
#define BELLOWS_PREPROCESS_ENCODING_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>

namespace bellows
{

// Expands nominal columns into binary (0/1) indicator columns.
class indicator_expander
{
public:
  static constexpr bool supports_nominal_columns = true;
  static constexpr bool supports_missing_entries = true;

  indicator_expander () = default;

  bool
  is_calibrated () const noexcept
  {
    return calibrated_;
  }

  template <tabular_source Data>
  void
  calibrate (const Data &data)
  {
    std::size_t d = data.column_count ();
    is_nominal_.assign (d, false);
    cardinalities_.assign (d, 1);
    output_cols_ = 0;

    for (std::size_t c = 0; c < d; ++c)
      {
        if (data.scale (c) == feature_scale::nominal)
          {
            is_nominal_[c] = true;
            std::size_t card = std::max<std::size_t> (2, data.cardinality (c));
            cardinalities_[c] = card;
            output_cols_ += card;
          }
        else
          {
            output_cols_ += 1;
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
    data_table result (n, output_cols_, 0.0);

    for (std::size_t r = 0; r < n; ++r)
      {
        std::size_t out_c = 0;
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = data.at (r, c);
            if (c < is_nominal_.size () && is_nominal_[c])
              {
                std::size_t card = cardinalities_[c];
                if (!is_missing (v) && v >= 0.0)
                  {
                    std::size_t cat = static_cast<std::size_t> (v);
                    if (cat < card)
                      {
                        result.at (r, out_c + cat) = 1.0;
                      }
                  }
                out_c += card;
              }
            else
              {
                result.set_cell (r, out_c, v);
                out_c += 1;
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
    out.write (reinterpret_cast<const char *> (&output_cols_),
                sizeof (output_cols_));
    std::size_t d = is_nominal_.size ();
    out.write (reinterpret_cast<const char *> (&d), sizeof (d));
    for (std::size_t c = 0; c < d; ++c)
      {
        std::uint8_t nom = is_nominal_[c] ? 1 : 0;
        std::size_t card = cardinalities_[c];
        out.write (reinterpret_cast<const char *> (&nom), sizeof (nom));
        out.write (reinterpret_cast<const char *> (&card), sizeof (card));
      }
  }

  void
  read_state (std::istream &in)
  {
    std::uint8_t cal = 0;
    in.read (reinterpret_cast<char *> (&cal), sizeof (cal));
    calibrated_ = (cal != 0);
    in.read (reinterpret_cast<char *> (&output_cols_), sizeof (output_cols_));
    std::size_t d = 0;
    in.read (reinterpret_cast<char *> (&d), sizeof (d));
    is_nominal_.assign (d, false);
    cardinalities_.assign (d, 1);
    for (std::size_t c = 0; c < d; ++c)
      {
        std::uint8_t nom = 0;
        std::size_t card = 1;
        in.read (reinterpret_cast<char *> (&nom), sizeof (nom));
        in.read (reinterpret_cast<char *> (&card), sizeof (card));
        is_nominal_[c] = (nom != 0);
        cardinalities_[c] = card;
      }
  }

private:
  bool calibrated_ = false;
  std::size_t output_cols_ = 0;
  std::vector<bool> is_nominal_;
  std::vector<std::size_t> cardinalities_;
};

// Maps nominal columns to smoothed target statistics using ordered priors.
class ordered_target_encoder
{
public:
  static constexpr bool supports_nominal_columns = true;
  static constexpr bool supports_missing_entries = true;

  explicit ordered_target_encoder (double prior_weight = 10.0,
                                    std::uint64_t seed = 42) noexcept
    : prior_weight_ (prior_weight),
      seed_ (seed)
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
          "bellows::ordered_target_encoder::calibrate: target size mismatch");
      }
    is_nominal_.assign (d, false);
    category_posteriors_.assign (d, {});

    double sum_all = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      {
        sum_all += targets[i];
      }
    global_prior_ =
      n > 0 ? sum_all / static_cast<double> (n) : 0.0;

    for (std::size_t c = 0; c < d; ++c)
      {
        if (data.scale (c) != feature_scale::nominal)
          {
            continue;
          }
        is_nominal_[c] = true;
        std::size_t card = std::max<std::size_t> (2, data.cardinality (c));
        std::vector<double> sum_y (card, 0.0);
        std::vector<double> cnt_y (card, 0.0);
        for (std::size_t r = 0; r < n; ++r)
          {
            double v = data.at (r, c);
            if (!is_missing (v) && v >= 0.0)
              {
                std::size_t cat = static_cast<std::size_t> (v);
                if (cat >= card)
                  {
                    card = cat + 1;
                    sum_y.resize (card, 0.0);
                    cnt_y.resize (card, 0.0);
                  }
                sum_y[cat] += targets[r];
                cnt_y[cat] += 1.0;
              }
          }
        category_posteriors_[c].resize (card, global_prior_);
        for (std::size_t k = 0; k < card; ++k)
          {
            category_posteriors_[c][k] =
              (sum_y[k] + prior_weight_ * global_prior_)
              / (cnt_y[k] + prior_weight_);
          }
      }
    calibrated_ = true;
  }

  // Produce leakage-free ordered encodings for a training table.
  template <tabular_source Data, target_sequence Targets>
  data_table
  project_ordered (const Data &data, const Targets &targets) const
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    data_table result (n, d);

    hyper_math::random_engine rng (seed_);
    std::vector<std::size_t> perm = rng.permutation (n);

    for (std::size_t c = 0; c < d; ++c)
      {
        if (data.scale (c) != feature_scale::nominal)
          {
            result.set_scale (c, data.scale (c), data.cardinality (c));
            for (std::size_t r = 0; r < n; ++r)
              {
                result.set_cell (r, c, data.at (r, c));
              }
            continue;
          }
        result.set_scale (c, feature_scale::real);
        std::size_t card = std::max<std::size_t> (2, data.cardinality (c));
        std::vector<double> sum_y (card, 0.0);
        std::vector<double> cnt_y (card, 0.0);

        for (std::size_t idx = 0; idx < n; ++idx)
          {
            std::size_t r = perm[idx];
            double v = data.at (r, c);
            if (is_missing (v) || v < 0.0)
              {
                result.at (r, c) = global_prior_;
              }
            else
              {
                std::size_t cat = static_cast<std::size_t> (v);
                if (cat >= card)
                  {
                    card = cat + 1;
                    sum_y.resize (card, 0.0);
                    cnt_y.resize (card, 0.0);
                  }
                result.at (r, c) =
                  (sum_y[cat] + prior_weight_ * global_prior_)
                  / (cnt_y[cat] + prior_weight_);
                sum_y[cat] += targets[r];
                cnt_y[cat] += 1.0;
              }
          }
      }
    return result;
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
        if (c < is_nominal_.size () && is_nominal_[c])
          {
            result.set_scale (c, feature_scale::real);
            const auto &post = category_posteriors_[c];
            for (std::size_t r = 0; r < n; ++r)
              {
                double v = data.at (r, c);
                if (is_missing (v) || v < 0.0)
                  {
                    result.at (r, c) = global_prior_;
                  }
                else
                  {
                    std::size_t cat = static_cast<std::size_t> (v);
                    result.at (r, c) =
                      cat < post.size () ? post[cat] : global_prior_;
                  }
              }
          }
        else
          {
            result.set_scale (c, data.scale (c), data.cardinality (c));
            for (std::size_t r = 0; r < n; ++r)
              {
                result.set_cell (r, c, data.at (r, c));
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
    out.write (reinterpret_cast<const char *> (&prior_weight_),
                sizeof (prior_weight_));
    out.write (reinterpret_cast<const char *> (&seed_), sizeof (seed_));
    out.write (reinterpret_cast<const char *> (&global_prior_),
                sizeof (global_prior_));
    std::size_t d = is_nominal_.size ();
    out.write (reinterpret_cast<const char *> (&d), sizeof (d));
    for (std::size_t c = 0; c < d; ++c)
      {
        std::uint8_t nom = is_nominal_[c] ? 1 : 0;
        out.write (reinterpret_cast<const char *> (&nom), sizeof (nom));
        std::size_t k = category_posteriors_[c].size ();
        out.write (reinterpret_cast<const char *> (&k), sizeof (k));
        if (k > 0)
          {
            out.write (
              reinterpret_cast<const char *> (category_posteriors_[c].data ()),
              static_cast<std::streamsize> (k * sizeof (double)));
          }
      }
  }

  void
  read_state (std::istream &in)
  {
    std::uint8_t cal = 0;
    in.read (reinterpret_cast<char *> (&cal), sizeof (cal));
    calibrated_ = (cal != 0);
    in.read (reinterpret_cast<char *> (&prior_weight_),
              sizeof (prior_weight_));
    in.read (reinterpret_cast<char *> (&seed_), sizeof (seed_));
    in.read (reinterpret_cast<char *> (&global_prior_),
              sizeof (global_prior_));
    std::size_t d = 0;
    in.read (reinterpret_cast<char *> (&d), sizeof (d));
    is_nominal_.assign (d, false);
    category_posteriors_.assign (d, {});
    for (std::size_t c = 0; c < d; ++c)
      {
        std::uint8_t nom = 0;
        in.read (reinterpret_cast<char *> (&nom), sizeof (nom));
        is_nominal_[c] = (nom != 0);
        std::size_t k = 0;
        in.read (reinterpret_cast<char *> (&k), sizeof (k));
        category_posteriors_[c].resize (k, 0.0);
        if (k > 0)
          {
            in.read (
              reinterpret_cast<char *> (category_posteriors_[c].data ()),
              static_cast<std::streamsize> (k * sizeof (double)));
          }
      }
  }

private:
  double prior_weight_ = 10.0;
  std::uint64_t seed_ = 42;
  bool calibrated_ = false;
  double global_prior_ = 0.0;
  std::vector<bool> is_nominal_;
  std::vector<std::vector<double>> category_posteriors_;
};

// Discretises real-valued columns into ordinal quantile buckets.
class quantile_bucketer
{
public:
  static constexpr bool supports_nominal_columns = true;
  static constexpr bool supports_missing_entries = true;

  explicit quantile_bucketer (std::size_t bucket_count = 10) noexcept
    : bucket_count_ (std::max<std::size_t> (2, bucket_count))
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
    cut_points_.assign (d, {});
    is_real_.assign (d, false);

    for (std::size_t c = 0; c < d; ++c)
      {
        if (data.scale (c) != feature_scale::real)
          {
            continue;
          }
        is_real_[c] = true;
        std::vector<double> vals;
        vals.reserve (n);
        for (std::size_t r = 0; r < n; ++r)
          {
            double v = data.at (r, c);
            if (!is_missing (v))
              {
                vals.push_back (v);
              }
          }
        std::sort (vals.begin (), vals.end ());
        if (vals.empty ())
          {
            continue;
          }
        for (std::size_t b = 1; b < bucket_count_; ++b)
          {
            std::size_t idx = (b * vals.size ()) / bucket_count_;
            if (idx < vals.size ())
              {
                double edge = vals[idx];
                if (cut_points_[c].empty ()
                    || edge > cut_points_[c].back () + 1.0e-12)
                  {
                    cut_points_[c].push_back (edge);
                  }
              }
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
        if (c < is_real_.size () && is_real_[c] && c < cut_points_.size ())
          {
            const auto &edges = cut_points_[c];
            result.set_scale (c, feature_scale::ordinal, edges.size () + 1);
            for (std::size_t r = 0; r < n; ++r)
              {
                double v = data.at (r, c);
                if (is_missing (v))
                  {
                    result.at (r, c) = missing_value ();
                  }
                else
                  {
                    auto it =
                      std::lower_bound (edges.begin (), edges.end (), v);
                    std::size_t bucket = static_cast<std::size_t> (
                      std::distance (edges.begin (), it));
                    result.at (r, c) = static_cast<double> (bucket);
                  }
              }
          }
        else
          {
            result.set_scale (c, data.scale (c), data.cardinality (c));
            for (std::size_t r = 0; r < n; ++r)
              {
                result.set_cell (r, c, data.at (r, c));
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
    out.write (reinterpret_cast<const char *> (&bucket_count_),
                sizeof (bucket_count_));
    std::size_t d = is_real_.size ();
    out.write (reinterpret_cast<const char *> (&d), sizeof (d));
    for (std::size_t c = 0; c < d; ++c)
      {
        std::uint8_t rl = is_real_[c] ? 1 : 0;
        out.write (reinterpret_cast<const char *> (&rl), sizeof (rl));
        std::size_t k = cut_points_[c].size ();
        out.write (reinterpret_cast<const char *> (&k), sizeof (k));
        if (k > 0)
          {
            out.write (reinterpret_cast<const char *> (cut_points_[c].data ()),
                        static_cast<std::streamsize> (k * sizeof (double)));
          }
      }
  }

  void
  read_state (std::istream &in)
  {
    std::uint8_t cal = 0;
    in.read (reinterpret_cast<char *> (&cal), sizeof (cal));
    calibrated_ = (cal != 0);
    in.read (reinterpret_cast<char *> (&bucket_count_),
              sizeof (bucket_count_));
    std::size_t d = 0;
    in.read (reinterpret_cast<char *> (&d), sizeof (d));
    is_real_.assign (d, false);
    cut_points_.assign (d, {});
    for (std::size_t c = 0; c < d; ++c)
      {
        std::uint8_t rl = 0;
        in.read (reinterpret_cast<char *> (&rl), sizeof (rl));
        is_real_[c] = (rl != 0);
        std::size_t k = 0;
        in.read (reinterpret_cast<char *> (&k), sizeof (k));
        cut_points_[c].resize (k, 0.0);
        if (k > 0)
          {
            in.read (reinterpret_cast<char *> (cut_points_[c].data ()),
                      static_cast<std::streamsize> (k * sizeof (double)));
          }
      }
  }

private:
  std::size_t bucket_count_ = 10;
  bool calibrated_ = false;
  std::vector<bool> is_real_;
  std::vector<std::vector<double>> cut_points_;
};

} // namespace bellows

#endif // BELLOWS_PREPROCESS_ENCODING_HPP
