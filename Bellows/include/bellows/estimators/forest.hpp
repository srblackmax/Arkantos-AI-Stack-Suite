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

#ifndef BELLOWS_ESTIMATORS_FOREST_HPP
#define BELLOWS_ESTIMATORS_FOREST_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <bellows/boosting/booster.hpp>
#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>
#include <bellows/core/parameters.hpp>
#include <bellows/estimators/proximity.hpp>

namespace bellows
{

// Bagged ensemble of bootstrap and feature-subspace decision trees.
class subspace_tree_ensemble
{
public:
  static constexpr bool supports_nominal_columns = true;
  static constexpr bool supports_missing_entries = true;

  explicit subspace_tree_ensemble (
    std::size_t tree_count = 25,
    std::size_t max_depth = 6,
    proximity_mode mode = proximity_mode::classification,
    std::uint64_t seed = 42) noexcept
    : tree_count_ (std::max<std::size_t> (1, tree_count)),
      max_depth_ (std::max<std::size_t> (1, max_depth)),
      mode_ (mode),
      seed_ (seed)
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
      {"tree_count", static_cast<double> (tree_count_), 1.0, 200.0},
      {"max_depth", static_cast<double> (max_depth_), 1.0, 16.0}};
  }

  bool
  apply_parameter (const std::string &name, double val)
  {
    if (name == "tree_count")
      {
        tree_count_ =
          static_cast<std::size_t> (std::max (1L, std::lround (val)));
        return true;
      }
    if (name == "max_depth")
      {
        max_depth_ =
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
    trees_.clear ();
    if (n == 0 || d == 0)
      {
        calibrated_ = false;
        return;
      }
    if (targets.size () != n)
      {
        throw std::invalid_argument (
          "bellows::subspace_tree_ensemble::calibrate: target size mismatch");
      }

    feature_count_ = d;
    std::vector<double> y_vec (n);
    long max_c = 1;
    for (std::size_t i = 0; i < n; ++i)
      {
        y_vec[i] = targets[i];
        if (mode_ == proximity_mode::classification)
          {
            max_c = std::max (max_c, std::lround (targets[i]));
          }
      }
    class_count_ =
      mode_ == proximity_mode::classification
        ? static_cast<std::size_t> (max_c + 1)
        : 1;

    std::size_t sub_features = std::max<std::size_t> (
      1, static_cast<std::size_t> (std::ceil (std::sqrt (static_cast<double> (d)))));

    trees_.reserve (tree_count_);
    for (std::size_t t = 0; t < tree_count_; ++t)
      {
        hyper_math::random_engine rng (seed_ + t * 2654435761ULL);
        std::vector<std::size_t> boot_rows (n);
        for (std::size_t i = 0; i < n; ++i)
          {
            boot_rows[i] = static_cast<std::size_t> (
              rng.randint (0, static_cast<std::int64_t> (n)));
          }

        additive_stage_tree tree;
        build_subspace_node (
          data, y_vec, boot_rows, 0, sub_features, rng, tree);
        trees_.push_back (std::move (tree));
      }
    calibrated_ = true;
  }

  double
  evaluate_one (std::span<const double> row) const
  {
    if (!calibrated_ || trees_.empty ())
      {
        return 0.0;
      }
    if (mode_ == proximity_mode::regression)
      {
        double sum = 0.0;
        for (const auto &tree : trees_)
          {
            sum += tree.evaluate_row (row);
          }
        return sum / static_cast<double> (trees_.size ());
      }

    std::vector<double> votes (class_count_, 0.0);
    for (const auto &tree : trees_)
      {
        double pred = tree.evaluate_row (row);
        if (class_count_ == 2)
          {
            votes[1] += std::clamp (pred, 0.0, 1.0);
            votes[0] += 1.0 - std::clamp (pred, 0.0, 1.0);
          }
        else
          {
            std::size_t cls = static_cast<std::size_t> (
              std::clamp<long> (std::lround (pred), 0, static_cast<long> (class_count_ - 1)));
            votes[cls] += 1.0;
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
        for (const auto &tree : trees_)
          {
            double pred = tree.evaluate_row (buf);
            if (class_count_ == 2)
              {
                double p1 = std::clamp (pred, 0.0, 1.0);
                beliefs.at (r, 1) += p1;
                beliefs.at (r, 0) += 1.0 - p1;
              }
            else
              {
                std::size_t cls = static_cast<std::size_t> (
                  std::clamp<long> (std::lround (pred),
                                    0,
                                    static_cast<long> (class_count_ - 1)));
                beliefs.at (r, cls) += 1.0;
              }
          }
        double denom = static_cast<double> ( std::max<std::size_t> (1, trees_.size ()));
        for (std::size_t k = 0; k < class_count_; ++k)
          {
            beliefs.at (r, k) /= denom;
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
    write_pod (feature_count_);
    write_pod (class_count_);
    std::uint8_t m_code = static_cast<std::uint8_t> (mode_);
    write_pod (m_code);
    std::size_t nt = trees_.size ();
    write_pod (nt);
    for (const auto &tree : trees_)
      {
        std::size_t nn = tree.nodes.size ();
        write_pod (nn);
        for (const auto &node : tree.nodes)
          {
            write_pod (node.feature_index);
            write_pod (node.threshold);
            std::uint8_t mr = node.missing_goes_right ? 1 : 0;
            write_pod (mr);
            write_pod (node.left_child);
            write_pod (node.right_child);
            write_pod (node.leaf_value);
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
    read_pod (feature_count_);
    read_pod (class_count_);
    std::uint8_t m_code = 0;
    read_pod (m_code);
    mode_ = static_cast<proximity_mode> (m_code);
    std::size_t nt = 0;
    read_pod (nt);
    trees_.resize (nt);
    for (std::size_t t = 0; t < nt; ++t)
      {
        std::size_t nn = 0;
        read_pod (nn);
        trees_[t].nodes.resize (nn);
        for (std::size_t j = 0; j < nn; ++j)
          {
            auto &node = trees_[t].nodes[j];
            read_pod (node.feature_index);
            read_pod (node.threshold);
            std::uint8_t mr = 0;
            read_pod (mr);
            node.missing_goes_right = (mr != 0);
            read_pod (node.left_child);
            read_pod (node.right_child);
            read_pod (node.leaf_value);
          }
      }
  }

private:
  static double
  leaf_prediction (const std::vector<double> &targets,
                    const std::vector<std::size_t> &rows,
                    proximity_mode mode,
                    std::size_t class_count)
  {
    if (rows.empty ())
      {
        return 0.0;
      }
    if (mode == proximity_mode::regression || class_count == 2)
      {
        double sum = 0.0;
        for (std::size_t r : rows)
          {
            sum += targets[r];
          }
        return sum / static_cast<double> (rows.size ());
      }
    std::vector<std::size_t> counts (class_count, 0);
    for (std::size_t r : rows)
      {
        std::size_t c = static_cast<std::size_t> (
          std::clamp<long> (std::lround (targets[r]),
                            0,
                            static_cast<long> (class_count - 1)));
        counts[c] += 1;
      }
    auto best = std::max_element (counts.begin (), counts.end ());
    return static_cast<double> (std::distance (counts.begin (), best));
  }

  template <tabular_source Data>
  std::int32_t
  build_subspace_node (const Data &data,
                        const std::vector<double> &targets,
                        const std::vector<std::size_t> &rows,
                        std::size_t depth,
                        std::size_t sub_features,
                        hyper_math::random_engine &rng,
                        additive_stage_tree &tree) const
  {
    std::int32_t node_id = static_cast<std::int32_t> (tree.nodes.size ());
    stage_tree_node node;
    node.leaf_value =
      leaf_prediction (targets, rows, mode_, class_count_);
    tree.nodes.push_back (node);

    if (depth >= max_depth_ || rows.size () < 4)
      {
        return node_id;
      }

    double sum_y = 0.0;
    for (std::size_t r : rows)
      {
        sum_y += targets[r];
      }
    double parent_mean = sum_y / static_cast<double> (rows.size ());
    double parent_var = 0.0;
    for (std::size_t r : rows)
      {
        double diff = targets[r] - parent_mean;
        parent_var += diff * diff;
      }

    std::vector<double> parent_cls_cnt;
    double parent_gini_score = 0.0;
    if (mode_ == proximity_mode::classification)
      {
        parent_cls_cnt.assign (class_count_, 0.0);
        for (std::size_t r : rows)
          {
            std::size_t k = static_cast<std::size_t> (
              std::clamp<long> (std::lround (targets[r]),
                                0,
                                static_cast<long> (class_count_ - 1)));
            parent_cls_cnt[k] += 1.0;
          }
        double sq_c = 0.0;
        for (double ck : parent_cls_cnt)
          {
            sq_c += ck * ck;
          }
        parent_gini_score = sq_c / static_cast<double> (rows.size ());
      }

    if (parent_var <= 1.0e-12)
      {
        return node_id;
      }

    std::vector<std::size_t> cols = rng.permutation (feature_count_);
    cols.resize (std::min (feature_count_, sub_features));

    double best_gain = 0.0;
    std::int32_t best_col = -1;
    double best_thr = 0.0;
    bool best_miss_right = false;

    for (std::size_t c : cols)
      {
        std::vector<double> vals;
        vals.reserve (rows.size ());
        for (std::size_t r : rows)
          {
            double v = data.at (r, c);
            if (!is_missing (v))
              {
                vals.push_back (v);
              }
          }
        std::sort (vals.begin (), vals.end ());
        vals.erase (std::unique (vals.begin (),
                                  vals.end (),
                                  [] (double a, double b)
                                  {
                                    return std::abs (a - b) <= 1.0e-12;
                                  }),
                    vals.end ());
        if (vals.size () < 2)
          {
            continue;
          }

        std::size_t step = std::max<std::size_t> (1, (vals.size () - 1) / 16);
        for (std::size_t i = 0; i + 1 < vals.size (); i += step)
          {
            double thr = 0.5 * (vals[i] + vals[i + 1]);
            for (int mr = 0; mr <= 1; ++mr)
              {
                double gain = 0.0;
                if (mode_ == proximity_mode::classification)
                  {
                    std::vector<double> left_cls (class_count_, 0.0);
                    std::size_t cnt_l = 0;
                    for (std::size_t r : rows)
                      {
                        double v = data.at (r, c);
                        bool right = is_missing (v) ? (mr == 1) : (v > thr);
                        if (!right)
                          {
                            std::size_t k = static_cast<std::size_t> (
                              std::clamp<long> (
                                std::lround (targets[r]),
                                0,
                                static_cast<long> (class_count_ - 1)));
                            left_cls[k] += 1.0;
                            ++cnt_l;
                          }
                      }
                    std::size_t cnt_r = rows.size () - cnt_l;
                    if (cnt_l < 2 || cnt_r < 2)
                      {
                        continue;
                      }
                    double sq_l = 0.0;
                    double sq_r = 0.0;
                    for (std::size_t k = 0; k < class_count_; ++k)
                      {
                        double cl = left_cls[k];
                        double cr = parent_cls_cnt[k] - cl;
                        sq_l += cl * cl;
                        sq_r += cr * cr;
                      }
                    gain = sq_l / static_cast<double> (cnt_l)
                            + sq_r / static_cast<double> (cnt_r)
                            - parent_gini_score;
                  }
                else
                  {
                    double sum_l = 0.0;
                    std::size_t cnt_l = 0;
                    for (std::size_t r : rows)
                      {
                        double v = data.at (r, c);
                        bool right = is_missing (v) ? (mr == 1) : (v > thr);
                        if (!right)
                          {
                            sum_l += targets[r];
                            ++cnt_l;
                          }
                      }
                    std::size_t cnt_r = rows.size () - cnt_l;
                    if (cnt_l < 2 || cnt_r < 2)
                      {
                        continue;
                      }
                    double sum_r = sum_y - sum_l;
                    gain =
                      (sum_l * sum_l) / static_cast<double> (cnt_l)
                      + (sum_r * sum_r) / static_cast<double> (cnt_r)
                      - (sum_y * sum_y) / static_cast<double> (rows.size ());
                  }
                if (gain > best_gain)
                  {
                    best_gain = gain;
                    best_col = static_cast<std::int32_t> (c);
                    best_thr = thr;
                    best_miss_right = (mr == 1);
                  }
              }
          }
      }

    if (best_col < 0)
      {
        return node_id;
      }

    std::vector<std::size_t> left_rows;
    std::vector<std::size_t> right_rows;
    for (std::size_t r : rows)
      {
        double v = data.at (r, static_cast<std::size_t> (best_col));
        bool right = is_missing (v) ? best_miss_right : (v > best_thr);
        if (right)
          {
            right_rows.push_back (r);
          }
        else
          {
            left_rows.push_back (r);
          }
      }

    std::int32_t left_child = build_subspace_node (
      data, targets, left_rows, depth + 1, sub_features, rng, tree);
    std::int32_t right_child = build_subspace_node (
      data, targets, right_rows, depth + 1, sub_features, rng, tree);

    auto &curr = tree.nodes[static_cast<std::size_t> (node_id)];
    curr.feature_index = best_col;
    curr.threshold = best_thr;
    curr.missing_goes_right = best_miss_right;
    curr.left_child = left_child;
    curr.right_child = right_child;
    return node_id;
  }

  std::size_t tree_count_ = 25;
  std::size_t max_depth_ = 6;
  proximity_mode mode_ = proximity_mode::classification;
  std::uint64_t seed_ = 42;
  bool calibrated_ = false;
  std::size_t feature_count_ = 0;
  std::size_t class_count_ = 2;
  std::vector<additive_stage_tree> trees_;
};

} // namespace bellows

#endif // BELLOWS_ESTIMATORS_FOREST_HPP
