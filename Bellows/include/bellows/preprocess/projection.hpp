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

#ifndef BELLOWS_PREPROCESS_PROJECTION_HPP
#define BELLOWS_PREPROCESS_PROJECTION_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>

namespace bellows
{

// Orthogonal principal-subspace dimensionality reducer backed by Hyper Math eigh.
class principal_subspace_projector
{
public:
  static constexpr bool supports_nominal_columns = false;
  static constexpr bool supports_missing_entries = true;

  explicit principal_subspace_projector (
    std::size_t component_count = 2) noexcept
    : requested_components_ (std::max<std::size_t> (1, component_count))
  {
  }

  bool
  is_calibrated () const noexcept
  {
    return calibrated_;
  }

  std::size_t
  component_count () const noexcept
  {
    return actual_components_;
  }

  const std::vector<double> &
  explained_variances () const noexcept
  {
    return explained_variances_;
  }

  const std::vector<double> &
  explained_variance_ratios () const noexcept
  {
    return variance_ratios_;
  }

  template <tabular_source Data>
  void
  calibrate (const Data &data)
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    if (n == 0 || d == 0)
      {
        calibrated_ = false;
        return;
      }

    feature_count_ = d;
    actual_components_ = std::min (d, requested_components_);
    means_.assign (d, 0.0);

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
        means_[c] = cnt > 0 ? sum / static_cast<double> (cnt) : 0.0;
      }

    std::vector<double> centered (n * d, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = data.at (r, c);
            centered[r * d + c] = (is_missing (v) ? means_[c] : v) - means_[c];
          }
      }

    double denom = static_cast<double> (std::max<std::size_t> (1, n - 1));
    hyper_math::tensor::nd_tensor<double> cov (
      hyper_math::tensor::shape_type{d, d}, 0.0);
    for (std::size_t i = 0; i < d; ++i)
      {
        for (std::size_t j = i; j < d; ++j)
          {
            double dot = 0.0;
            for (std::size_t r = 0; r < n; ++r)
              {
                dot += centered[r * d + i] * centered[r * d + j];
              }
            double val = dot / denom;
            cov (i, j) = val;
            cov (j, i) = val;
          }
      }

    auto eig = hyper_math::scientific::linalg::eigh (cov);
    double total_var = 0.0;
    for (std::size_t j = 0; j < d; ++j)
      {
        total_var += std::max (0.0, eig.eigenvalues[j]);
      }

    basis_.assign (actual_components_ * d, 0.0);
    explained_variances_.assign (actual_components_, 0.0);
    variance_ratios_.assign (actual_components_, 0.0);

    for (std::size_t k = 0; k < actual_components_; ++k)
      {
        std::size_t eig_col = d - 1 - k;
        double ev = std::max (0.0, eig.eigenvalues[eig_col]);
        explained_variances_[k] = ev;
        variance_ratios_[k] = total_var > 1.0e-15 ? ev / total_var : 0.0;
        for (std::size_t c = 0; c < d; ++c)
          {
            basis_[k * d + c] = eig.eigenvectors (c, eig_col);
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
    std::size_t d = std::min (data.column_count (), feature_count_);
    data_table out (n, actual_components_, 0.0);
    if (!calibrated_)
      {
        return out;
      }

    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t k = 0; k < actual_components_; ++k)
          {
            double score = 0.0;
            const double *axis = basis_.data () + k * feature_count_;
            for (std::size_t c = 0; c < d; ++c)
              {
                double v = data.at (r, c);
                double centered = (is_missing (v) ? means_[c] : v) - means_[c];
                score += centered * axis[c];
              }
            out.at (r, k) = score;
          }
      }
    return out;
  }

  // Reconstruct observations in the original feature space from coordinates.
  template <tabular_source Data>
  data_table
  reconstruct (const Data &projected) const
  {
    std::size_t n = projected.row_count ();
    std::size_t k_in = std::min (projected.column_count (), actual_components_);
    data_table recon (n, feature_count_, 0.0);
    if (!calibrated_)
      {
        return recon;
      }

    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < feature_count_; ++c)
          {
            double acc = means_[c];
            for (std::size_t k = 0; k < k_in; ++k)
              {
                acc += projected.at (r, k) * basis_[k * feature_count_ + c];
              }
            recon.at (r, c) = acc;
          }
      }
    return recon;
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
    write_pod (requested_components_);
    write_pod (actual_components_);
    write_pod (feature_count_);
    for (double m : means_)
      {
        write_pod (m);
      }
    for (double b : basis_)
      {
        write_pod (b);
      }
    for (double ev : explained_variances_)
      {
        write_pod (ev);
      }
    for (double vr : variance_ratios_)
      {
        write_pod (vr);
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
    read_pod (requested_components_);
    read_pod (actual_components_);
    read_pod (feature_count_);
    means_.resize (feature_count_, 0.0);
    for (std::size_t c = 0; c < feature_count_; ++c)
      {
        read_pod (means_[c]);
      }
    basis_.resize (actual_components_ * feature_count_, 0.0);
    for (std::size_t i = 0; i < basis_.size (); ++i)
      {
        read_pod (basis_[i]);
      }
    explained_variances_.resize (actual_components_, 0.0);
    variance_ratios_.resize (actual_components_, 0.0);
    for (std::size_t k = 0; k < actual_components_; ++k)
      {
        read_pod (explained_variances_[k]);
      }
    for (std::size_t k = 0; k < actual_components_; ++k)
      {
        read_pod (variance_ratios_[k]);
      }
  }

private:
  std::size_t requested_components_ = 2;
  std::size_t actual_components_ = 0;
  std::size_t feature_count_ = 0;
  bool calibrated_ = false;
  std::vector<double> means_;
  std::vector<double> basis_;
  std::vector<double> explained_variances_;
  std::vector<double> variance_ratios_;
};

// Synthesises degree-2 pairwise feature interactions and quadratic terms.
class interaction_expander
{
public:
  static constexpr bool supports_nominal_columns = false;
  static constexpr bool supports_missing_entries = true;

  explicit interaction_expander (bool include_quadratic = true) noexcept
    : include_quadratic_ (include_quadratic)
  {
  }

  bool
  is_calibrated () const noexcept
  {
    return calibrated_;
  }

  std::size_t
  output_column_count () const noexcept
  {
    return output_cols_;
  }

  template <tabular_source Data>
  void
  calibrate (const Data &data)
  {
    feature_count_ = data.column_count ();
    std::size_t pairs =
      feature_count_ > 1 ? (feature_count_ * (feature_count_ - 1)) / 2 : 0;
    std::size_t quads = include_quadratic_ ? feature_count_ : 0;
    output_cols_ = feature_count_ + pairs + quads;
    calibrated_ = (feature_count_ > 0);
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
    std::size_t d = std::min (data.column_count (), feature_count_);
    data_table out (n, output_cols_, 0.0);
    if (!calibrated_)
      {
        return out;
      }

    for (std::size_t r = 0; r < n; ++r)
      {
        std::size_t col_idx = 0;
        for (std::size_t c = 0; c < d; ++c)
          {
            out.at (r, col_idx++) = data.at (r, c);
          }
        for (std::size_t i = 0; i < d; ++i)
          {
            double vi = data.at (r, i);
            if (include_quadratic_)
              {
                out.at (r, col_idx++) =
                  is_missing (vi) ? missing_value () : (vi * vi);
              }
            for (std::size_t j = i + 1; j < d; ++j)
              {
                double vj = data.at (r, j);
                out.at (r, col_idx++) =
                  (is_missing (vi) || is_missing (vj))
                    ? missing_value ()
                    : (vi * vj);
              }
          }
      }
    return out;
  }

  void
  write_state (std::ostream &out) const
  {
    std::uint8_t cal = calibrated_ ? 1 : 0;
    std::uint8_t quad = include_quadratic_ ? 1 : 0;
    out.write (reinterpret_cast<const char *> (&cal), sizeof (cal));
    out.write (reinterpret_cast<const char *> (&quad), sizeof (quad));
    out.write (reinterpret_cast<const char *> (&feature_count_),
                sizeof (feature_count_));
    out.write (reinterpret_cast<const char *> (&output_cols_),
                sizeof (output_cols_));
  }

  void
  read_state (std::istream &in)
  {
    std::uint8_t cal = 0;
    std::uint8_t quad = 0;
    in.read (reinterpret_cast<char *> (&cal), sizeof (cal));
    in.read (reinterpret_cast<char *> (&quad), sizeof (quad));
    calibrated_ = (cal != 0);
    include_quadratic_ = (quad != 0);
    in.read (reinterpret_cast<char *> (&feature_count_),
              sizeof (feature_count_));
    in.read (reinterpret_cast<char *> (&output_cols_), sizeof (output_cols_));
  }

private:
  bool include_quadratic_ = true;
  bool calibrated_ = false;
  std::size_t feature_count_ = 0;
  std::size_t output_cols_ = 0;
};

} // namespace bellows

#endif // BELLOWS_PREPROCESS_PROJECTION_HPP
