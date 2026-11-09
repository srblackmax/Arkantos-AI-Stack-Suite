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

#ifndef BELLOWS_ESTIMATORS_CLUSTERING_HPP
#define BELLOWS_ESTIMATORS_CLUSTERING_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>
#include <bellows/core/parameters.hpp>

namespace bellows
{

// Centroid-based partitioning learner backed by Hyper Math kmeans2 & spatial.
class centroid_partitioner
{
public:
  static constexpr bool supports_nominal_columns = false;
  static constexpr bool supports_missing_entries = true;

  explicit centroid_partitioner (std::size_t cluster_count = 3,
                                  std::size_t max_iterations = 100,
                                  std::uint64_t seed = 42) noexcept
    : cluster_count_ (std::max<std::size_t> (1, cluster_count)),
      max_iterations_ (max_iterations),
      seed_ (seed)
  {
  }

  bool
  is_calibrated () const noexcept
  {
    return calibrated_;
  }

  double
  inertia () const noexcept
  {
    return inertia_;
  }

  const data_table &
  centroids () const noexcept
  {
    return centroids_;
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

    std::size_t k = std::min (n, cluster_count_);
    std::vector<double> col_means (d, 0.0);
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
        col_means[c] = cnt > 0 ? sum / static_cast<double> (cnt) : 0.0;
      }

    hyper_math::tensor::nd_tensor<double> clean_mat (
      hyper_math::tensor::shape_type{n, d}, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = data.at (r, c);
            clean_mat (r, c) = is_missing (v) ? col_means[c] : v;
          }
      }

    hyper_math::random_engine rng (seed_);
    auto km = hyper_math::scientific::cluster::kmeans2 (
      clean_mat, k, max_iterations_, 1.0e-6, rng);

    centroids_ = data_table (km.centroids);
    inertia_ = km.inertia;
    calibrated_ = true;
  }

  template <tabular_source Data, target_sequence Targets>
  void
  calibrate (const Data &data, const Targets &)
  {
    calibrate (data);
  }

  // Predictor role: assign observation to nearest centroid index.
  double
  evaluate_one (std::span<const double> row) const noexcept
  {
    if (!calibrated_ || centroids_.row_count () == 0)
      {
        return 0.0;
      }
    std::size_t k = centroids_.row_count ();
    std::size_t d = std::min (row.size (), centroids_.column_count ());
    std::size_t best_j = 0;
    double best_sq = std::numeric_limits<double>::infinity ();
    for (std::size_t j = 0; j < k; ++j)
      {
        double sq = 0.0;
        for (std::size_t c = 0; c < d; ++c)
          {
            if (!is_missing (row[c]))
              {
                double diff = row[c] - centroids_.at (j, c);
                sq += diff * diff;
              }
          }
        if (sq < best_sq)
          {
            best_sq = sq;
            best_j = j;
          }
      }
    return static_cast<double> (best_j);
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

  // Transformer role: project observations into centroid-distance space.
  template <tabular_source Data>
  data_table
  project (const Data &data) const
  {
    std::size_t n = data.row_count ();
    std::size_t d = std::min (data.column_count (), centroids_.column_count ());
    std::size_t k = centroids_.row_count ();
    data_table dist_table (n, k, 0.0);

    std::vector<double> row_clean (d, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t j = 0; j < k; ++j)
          {
            for (std::size_t c = 0; c < d; ++c)
              {
                double v = data.at (r, c);
                row_clean[c] = is_missing (v) ? centroids_.at (j, c) : v;
              }
            dist_table.at (r, j) =
              hyper_math::scientific::spatial::point_distance (
                row_clean.data (),
                centroids_.row_span (j).data (),
                d,
                "euclidean");
          }
      }
    return dist_table;
  }

  void
  write_state (std::ostream &out) const
  {
    std::uint8_t cal = calibrated_ ? 1 : 0;
    out.write (reinterpret_cast<const char *> (&cal), sizeof (cal));
    out.write (reinterpret_cast<const char *> (&inertia_), sizeof (inertia_));
    std::size_t k = centroids_.row_count ();
    std::size_t d = centroids_.column_count ();
    out.write (reinterpret_cast<const char *> (&k), sizeof (k));
    out.write (reinterpret_cast<const char *> (&d), sizeof (d));
    for (std::size_t r = 0; r < k; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = centroids_.at (r, c);
            out.write (reinterpret_cast<const char *> (&v), sizeof (double));
          }
      }
  }

  void
  read_state (std::istream &in)
  {
    std::uint8_t cal = 0;
    in.read (reinterpret_cast<char *> (&cal), sizeof (cal));
    calibrated_ = (cal != 0);
    in.read (reinterpret_cast<char *> (&inertia_), sizeof (inertia_));
    std::size_t k = 0;
    std::size_t d = 0;
    in.read (reinterpret_cast<char *> (&k), sizeof (k));
    in.read (reinterpret_cast<char *> (&d), sizeof (d));
    centroids_ = data_table (k, d, 0.0);
    for (std::size_t r = 0; r < k; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = 0.0;
            in.read (reinterpret_cast<char *> (&v), sizeof (double));
            centroids_.at (r, c) = v;
          }
      }
  }

private:
  std::size_t cluster_count_ = 3;
  std::size_t max_iterations_ = 100;
  std::uint64_t seed_ = 42;
  bool calibrated_ = false;
  double inertia_ = 0.0;
  data_table centroids_;
};

// Agglomerative hierarchical partitioner backed by Hyper Math linkage/fcluster.
class hierarchical_partitioner
{
public:
  static constexpr bool supports_nominal_columns = false;
  static constexpr bool supports_missing_entries = true;

  explicit hierarchical_partitioner (std::size_t cluster_count = 2,
                                      std::string linkage_kind = "ward")
    : cluster_count_ (std::max<std::size_t> (1, cluster_count)),
      linkage_kind_ (std::move (linkage_kind))
  {
  }

  bool
  is_calibrated () const noexcept
  {
    return calibrated_;
  }

  const std::vector<double> &
  labels () const noexcept
  {
    return labels_;
  }

  const data_table &
  linkage_matrix () const noexcept
  {
    return linkage_mat_;
  }

  const data_table &
  centroids () const noexcept
  {
    return centroids_;
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

    std::vector<double> col_means (d, 0.0);
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
        col_means[c] = cnt > 0 ? sum / static_cast<double> (cnt) : 0.0;
      }

    hyper_math::tensor::nd_tensor<double> clean_mat (
      hyper_math::tensor::shape_type{n, d}, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = data.at (r, c);
            clean_mat (r, c) = is_missing (v) ? col_means[c] : v;
          }
      }

    std::size_t k = std::min (n, cluster_count_);
    labels_.assign (n, 0.0);
    if (n == 1)
      {
        linkage_mat_ = data_table (0, 4, 0.0);
        centroids_ = data_table (clean_mat);
        calibrated_ = true;
        return;
      }

    auto z = hyper_math::scientific::cluster::linkage (clean_mat, linkage_kind_);
    linkage_mat_ = data_table (z);
    auto flat = hyper_math::scientific::cluster::fcluster (
      z, static_cast<double> (k), "maxclust");

    std::size_t max_label = 0;
    for (std::size_t r = 0; r < n; ++r)
      {
        std::size_t zero_based =
          flat[r] > 0 ? static_cast<std::size_t> (flat[r] - 1) : 0;
        labels_[r] = static_cast<double> (zero_based);
        if (zero_based + 1 > max_label)
          {
            max_label = zero_based + 1;
          }
      }

    std::size_t actual_k = std::max<std::size_t> (1, max_label);
    centroids_ = data_table (actual_k, d, 0.0);
    std::vector<std::size_t> counts (actual_k, 0);
    for (std::size_t r = 0; r < n; ++r)
      {
        std::size_t cid = static_cast<std::size_t> (labels_[r]);
        ++counts[cid];
        for (std::size_t c = 0; c < d; ++c)
          {
            centroids_.at (cid, c) += clean_mat (r, c);
          }
      }
    for (std::size_t j = 0; j < actual_k; ++j)
      {
        if (counts[j] > 0)
          {
            for (std::size_t c = 0; c < d; ++c)
              {
                centroids_.at (j, c) /= static_cast<double> (counts[j]);
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

  double
  evaluate_one (std::span<const double> row) const noexcept
  {
    if (!calibrated_ || centroids_.row_count () == 0)
      {
        return 0.0;
      }
    std::size_t k = centroids_.row_count ();
    std::size_t d = std::min (row.size (), centroids_.column_count ());
    std::size_t best_j = 0;
    double best_sq = std::numeric_limits<double>::infinity ();
    for (std::size_t j = 0; j < k; ++j)
      {
        double sq = 0.0;
        for (std::size_t c = 0; c < d; ++c)
          {
            if (!is_missing (row[c]))
              {
                double diff = row[c] - centroids_.at (j, c);
                sq += diff * diff;
              }
          }
        if (sq < best_sq)
          {
            best_sq = sq;
            best_j = j;
          }
      }
    return static_cast<double> (best_j);
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
    std::uint8_t cal = calibrated_ ? 1 : 0;
    out.write (reinterpret_cast<const char *> (&cal), sizeof (cal));
    out.write (reinterpret_cast<const char *> (&cluster_count_),
                sizeof (cluster_count_));
    std::size_t n_lbl = labels_.size ();
    out.write (reinterpret_cast<const char *> (&n_lbl), sizeof (n_lbl));
    for (double lbl : labels_)
      {
        out.write (reinterpret_cast<const char *> (&lbl), sizeof (double));
      }
    std::size_t k = centroids_.row_count ();
    std::size_t d = centroids_.column_count ();
    out.write (reinterpret_cast<const char *> (&k), sizeof (k));
    out.write (reinterpret_cast<const char *> (&d), sizeof (d));
    for (std::size_t r = 0; r < k; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = centroids_.at (r, c);
            out.write (reinterpret_cast<const char *> (&v), sizeof (double));
          }
      }
  }

  void
  read_state (std::istream &in)
  {
    std::uint8_t cal = 0;
    in.read (reinterpret_cast<char *> (&cal), sizeof (cal));
    calibrated_ = (cal != 0);
    in.read (reinterpret_cast<char *> (&cluster_count_),
              sizeof (cluster_count_));
    std::size_t n_lbl = 0;
    in.read (reinterpret_cast<char *> (&n_lbl), sizeof (n_lbl));
    labels_.resize (n_lbl, 0.0);
    for (std::size_t i = 0; i < n_lbl; ++i)
      {
        in.read (reinterpret_cast<char *> (&labels_[i]), sizeof (double));
      }
    std::size_t k = 0;
    std::size_t d = 0;
    in.read (reinterpret_cast<char *> (&k), sizeof (k));
    in.read (reinterpret_cast<char *> (&d), sizeof (d));
    centroids_ = data_table (k, d, 0.0);
    for (std::size_t r = 0; r < k; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = 0.0;
            in.read (reinterpret_cast<char *> (&v), sizeof (double));
            centroids_.at (r, c) = v;
          }
      }
  }

private:
  std::size_t cluster_count_ = 2;
  std::string linkage_kind_ = "ward";
  bool calibrated_ = false;
  std::vector<double> labels_;
  data_table linkage_mat_;
  data_table centroids_;
};

// Density-reachability partitioner backed by Hyper Math dbscan.
class density_reachability_partitioner
{
public:
  static constexpr bool supports_nominal_columns = false;
  static constexpr bool supports_missing_entries = true;

  explicit density_reachability_partitioner (
    double radius = 0.5,
    std::size_t min_neighbours = 3) noexcept
    : radius_ (std::max (1.0e-9, radius)),
      min_neighbours_ (std::max<std::size_t> (1, min_neighbours))
  {
  }

  bool
  is_calibrated () const noexcept
  {
    return calibrated_;
  }

  std::size_t
  cluster_count () const noexcept
  {
    return discovered_clusters_;
  }

  const std::vector<double> &
  labels () const noexcept
  {
    return labels_;
  }

  const std::vector<std::size_t> &
  core_indices () const noexcept
  {
    return core_indices_;
  }

  std::vector<parameter_knob>
  parameter_space () const
  {
    return {
      {"radius", radius_, 1.0e-3, 50.0},
      {"min_neighbours", static_cast<double> (min_neighbours_), 1.0, 50.0}
    };
  }

  bool
  apply_parameter (const std::string &name, double val) noexcept
  {
    if (name == "radius")
      {
        radius_ = std::max (1.0e-9, val);
        return true;
      }
    if (name == "min_neighbours")
      {
        min_neighbours_ =
          static_cast<std::size_t> (std::max (1L, std::lround (val)));
        return true;
      }
    return false;
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

    hyper_math::tensor::nd_tensor<double> clean_mat (
      hyper_math::tensor::shape_type{n, d}, 0.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = data.at (r, c);
            clean_mat (r, c) = is_missing (v) ? col_means_[c] : v;
          }
      }

    auto res = hyper_math::scientific::cluster::dbscan (
      clean_mat, radius_, min_neighbours_);
    discovered_clusters_ = res.n_clusters;
    core_indices_ = std::move (res.core_sample_indices);
    labels_.assign (n, -1.0);
    for (std::size_t r = 0; r < n; ++r)
      {
        labels_[r] = static_cast<double> (res.labels[r]);
      }

    core_points_ = data_table (core_indices_.size (), d, 0.0);
    core_labels_.assign (core_indices_.size (), -1.0);
    for (std::size_t i = 0; i < core_indices_.size (); ++i)
      {
        std::size_t r = core_indices_[i];
        core_labels_[i] = labels_[r];
        for (std::size_t c = 0; c < d; ++c)
          {
            core_points_.at (i, c) = clean_mat (r, c);
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

  double
  evaluate_one (std::span<const double> row) const noexcept
  {
    if (!calibrated_ || core_points_.row_count () == 0)
      {
        return -1.0;
      }
    std::size_t m = core_points_.row_count ();
    std::size_t d = std::min (row.size (), core_points_.column_count ());
    double best_sq = radius_ * radius_;
    double assigned = -1.0;

    for (std::size_t i = 0; i < m; ++i)
      {
        double sq = 0.0;
        for (std::size_t c = 0; c < d; ++c)
          {
            double xv = is_missing (row[c]) ? col_means_[c] : row[c];
            double diff = xv - core_points_.at (i, c);
            sq += diff * diff;
          }
        if (sq <= best_sq)
          {
            best_sq = sq;
            assigned = core_labels_[i];
          }
      }
    return assigned;
  }

  template <tabular_source Data>
  std::vector<double>
  evaluate (const Data &data) const
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    std::vector<double> out (n, -1.0);
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
    write_pod (radius_);
    write_pod (min_neighbours_);
    write_pod (discovered_clusters_);

    std::size_t m = core_points_.row_count ();
    std::size_t d = col_means_.size ();
    write_pod (m);
    write_pod (d);
    for (double cm : col_means_)
      {
        write_pod (cm);
      }
    for (double cl : core_labels_)
      {
        write_pod (cl);
      }
    for (std::size_t i = 0; i < m; ++i)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            write_pod (core_points_.at (i, c));
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
    read_pod (cal);
    calibrated_ = (cal != 0);
    read_pod (radius_);
    read_pod (min_neighbours_);
    read_pod (discovered_clusters_);

    std::size_t m = 0;
    std::size_t d = 0;
    read_pod (m);
    read_pod (d);
    col_means_.resize (d, 0.0);
    for (std::size_t c = 0; c < d; ++c)
      {
        read_pod (col_means_[c]);
      }
    core_labels_.resize (m, -1.0);
    for (std::size_t i = 0; i < m; ++i)
      {
        read_pod (core_labels_[i]);
      }
    core_points_ = data_table (m, d, 0.0);
    for (std::size_t i = 0; i < m; ++i)
      {
        for (std::size_t c = 0; c < d; ++c)
          {
            double v = 0.0;
            read_pod (v);
            core_points_.at (i, c) = v;
          }
      }
  }

private:
  double radius_ = 0.5;
  std::size_t min_neighbours_ = 3;
  std::size_t discovered_clusters_ = 0;
  bool calibrated_ = false;
  std::vector<double> col_means_;
  std::vector<double> labels_;
  std::vector<std::size_t> core_indices_;
  std::vector<double> core_labels_;
  data_table core_points_;
};

} // namespace bellows

#endif // BELLOWS_ESTIMATORS_CLUSTERING_HPP
