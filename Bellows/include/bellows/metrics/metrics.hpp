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

#ifndef BELLOWS_METRICS_METRICS_HPP
#define BELLOWS_METRICS_METRICS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

#include <bellows/core/concepts.hpp>
#include <bellows/core/data_table.hpp>

namespace bellows
{

// Mean squared deviation between ground truth and predicted values.
struct mean_squared_deviation
{
  static constexpr bool maximise = false;

  double
  operator() (const std::vector<double> &truth,
              const std::vector<double> &pred) const
  {
    if (truth.empty () || truth.size () != pred.size ())
      {
        return 0.0;
      }
    double sum = 0.0;
    for (std::size_t i = 0; i < truth.size (); ++i)
      {
        double diff = truth[i] - pred[i];
        sum += diff * diff;
      }
    return sum / static_cast<double> (truth.size ());
  }
};

// Root mean squared deviation metric functor.
struct root_mean_squared_deviation
{
  static constexpr bool maximise = false;

  double
  operator() (const std::vector<double> &truth,
              const std::vector<double> &pred) const
  {
    mean_squared_deviation msd;
    return std::sqrt (msd (truth, pred));
  }
};

// Mean absolute deviation metric functor.
struct mean_absolute_deviation
{
  static constexpr bool maximise = false;

  double
  operator() (const std::vector<double> &truth,
              const std::vector<double> &pred) const
  {
    if (truth.empty () || truth.size () != pred.size ())
      {
        return 0.0;
      }
    double sum = 0.0;
    for (std::size_t i = 0; i < truth.size (); ++i)
      {
        sum += std::abs (truth[i] - pred[i]);
      }
    return sum / static_cast<double> (truth.size ());
  }
};

// Proportion of explained target variance (coefficient of determination R^2).
struct explained_variance_ratio
{
  static constexpr bool maximise = true;

  double
  operator() (const std::vector<double> &truth,
              const std::vector<double> &pred) const
  {
    if (truth.empty () || truth.size () != pred.size ())
      {
        return 0.0;
      }
    double mean_y = 0.0;
    for (double y : truth)
      {
        mean_y += y;
      }
    mean_y /= static_cast<double> (truth.size ());

    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (std::size_t i = 0; i < truth.size (); ++i)
      {
        double res = truth[i] - pred[i];
        double tot = truth[i] - mean_y;
        ss_res += res * res;
        ss_tot += tot * tot;
      }
    if (ss_tot <= 1.0e-15)
      {
        return ss_res <= 1.0e-15 ? 1.0 : 0.0;
      }
    return 1.0 - (ss_res / ss_tot);
  }
};

// Discrete classification accuracy metric functor.
struct classification_accuracy
{
  static constexpr bool maximise = true;

  double
  operator() (const std::vector<double> &truth,
              const std::vector<double> &pred) const
  {
    if (truth.empty () || truth.size () != pred.size ())
      {
        return 0.0;
      }
    std::size_t correct = 0;
    for (std::size_t i = 0; i < truth.size (); ++i)
      {
        long t_label = std::lround (truth[i]);
        long p_label = std::lround (pred[i]);
        if (t_label == p_label)
          {
            ++correct;
          }
      }
    return static_cast<double> (correct)
            / static_cast<double> (truth.size ());
  }
};

// Harmonic precision-recall balance (F1 score, macro-averaged for multiclass).
struct balanced_f1_score
{
  static constexpr bool maximise = true;

  double
  operator() (const std::vector<double> &truth,
              const std::vector<double> &pred) const
  {
    if (truth.empty () || truth.size () != pred.size ())
      {
        return 0.0;
      }
    long max_class = 1;
    for (std::size_t i = 0; i < truth.size (); ++i)
      {
        max_class = std::max (max_class, std::lround (truth[i]));
        max_class = std::max (max_class, std::lround (pred[i]));
      }
    if (max_class <= 1)
      {
        double tp = 0.0;
        double fp = 0.0;
        double fn = 0.0;
        for (std::size_t i = 0; i < truth.size (); ++i)
          {
            long t = std::lround (truth[i]);
            long p = std::lround (pred[i]);
            if (t == 1 && p == 1)
              {
                tp += 1.0;
              }
            else if (t == 0 && p == 1)
              {
                fp += 1.0;
              }
            else if (t == 1 && p == 0)
              {
                fn += 1.0;
              }
          }
        double denom = 2.0 * tp + fp + fn;
        return denom > 0.0 ? (2.0 * tp) / denom : 1.0;
      }

    double f1_sum = 0.0;
    long num_classes = max_class + 1;
    for (long c = 0; c < num_classes; ++c)
      {
        double tp = 0.0;
        double fp = 0.0;
        double fn = 0.0;
        for (std::size_t i = 0; i < truth.size (); ++i)
          {
            long t = std::lround (truth[i]);
            long p = std::lround (pred[i]);
            if (t == c && p == c)
              {
                tp += 1.0;
              }
            else if (t != c && p == c)
              {
                fp += 1.0;
              }
            else if (t == c && p != c)
              {
                fn += 1.0;
              }
          }
        double denom = 2.0 * tp + fp + fn;
        f1_sum += denom > 0.0 ? (2.0 * tp) / denom : 1.0;
      }
    return f1_sum / static_cast<double> (num_classes);
  }
};

// Binary cross-entropy (logarithmic loss) metric functor.
struct logarithmic_loss
{
  static constexpr bool maximise = false;

  double
  operator() (const std::vector<double> &truth,
              const std::vector<double> &pred_prob) const
  {
    if (truth.empty () || truth.size () != pred_prob.size ())
      {
        return 0.0;
      }
    constexpr double eps = 1.0e-12;
    double sum = 0.0;
    for (std::size_t i = 0; i < truth.size (); ++i)
      {
        double p = std::clamp (pred_prob[i], eps, 1.0 - eps);
        double y = truth[i] > 0.5 ? 1.0 : 0.0;
        sum -= y * std::log (p) + (1.0 - y) * std::log (1.0 - p);
      }
    return sum / static_cast<double> (truth.size ());
  }
};

// Area under the receiver operating characteristic curve (ROC-AUC).
struct roc_area_under_curve
{
  static constexpr bool maximise = true;

  double
  operator() (const std::vector<double> &truth,
              const std::vector<double> &scores) const
  {
    if (truth.empty () || truth.size () != scores.size ())
      {
        return 0.5;
      }
    std::vector<std::size_t> order (truth.size ());
    std::iota (order.begin (), order.end (), 0);
    std::sort (order.begin (), order.end (),
                [&] (std::size_t a, std::size_t b)
                {
                  return scores[a] < scores[b];
                });

    double n_pos = 0.0;
    double n_neg = 0.0;
    for (double y : truth)
      {
        if (y > 0.5)
          {
            n_pos += 1.0;
          }
        else
          {
            n_neg += 1.0;
          }
      }
    if (n_pos == 0.0 || n_neg == 0.0)
      {
        return 0.5;
      }

    double rank_sum = 0.0;
    std::size_t i = 0;
    while (i < order.size ())
      {
        std::size_t j = i;
        while (j + 1 < order.size ()
                && std::abs (scores[order[j + 1]] - scores[order[i]]) <= 1.0e-15)
          {
            ++j;
          }
        double avg_rank = 0.5 * static_cast<double> ((i + 1) + (j + 1));
        for (std::size_t k = i; k <= j; ++k)
          {
            if (truth[order[k]] > 0.5)
              {
                rank_sum += avg_rank;
              }
          }
        i = j + 1;
      }

    double u_stat = rank_sum - 0.5 * n_pos * (n_pos + 1.0);
    return u_stat / (n_pos * n_neg);
  }
};

// Mean silhouette cohesion coefficient for cluster assignments on a table.
struct silhouette_cohesion
{
  static constexpr bool maximise = true;

  template <tabular_source Data>
  double
  evaluate_clusters (const Data &data,
                      const std::vector<double> &assignments) const
  {
    std::size_t n = data.row_count ();
    std::size_t d = data.column_count ();
    if (n <= 1 || assignments.size () != n)
      {
        return 0.0;
      }

    auto sq_dist = [&] (std::size_t i, std::size_t j) -> double
    {
      double sum = 0.0;
      for (std::size_t c = 0; c < d; ++c)
        {
          double vi = data.at (i, c);
          double vj = data.at (j, c);
          if (!is_missing (vi) && !is_missing (vj))
            {
              double diff = vi - vj;
              sum += diff * diff;
            }
        }
      return std::sqrt (sum);
    };

    long max_cluster = 0;
    for (double a : assignments)
      {
        max_cluster = std::max (max_cluster, std::lround (a));
      }
    std::size_t k_count = static_cast<std::size_t> (max_cluster + 1);
    if (k_count <= 1)
      {
        return 0.0;
      }

    double total_sil = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      {
        std::size_t ci = static_cast<std::size_t> (std::lround (assignments[i]));
        std::vector<double> sum_dist (k_count, 0.0);
        std::vector<std::size_t> count_cluster (k_count, 0);
        for (std::size_t j = 0; j < n; ++j)
          {
            if (i == j)
              {
                continue;
              }
            std::size_t cj =
              static_cast<std::size_t> (std::lround (assignments[j]));
            sum_dist[cj] += sq_dist (i, j);
            count_cluster[cj] += 1;
          }

        double a_i = count_cluster[ci] > 0
                        ? sum_dist[ci] / static_cast<double> (count_cluster[ci])
                        : 0.0;
        double b_i = std::numeric_limits<double>::infinity ();
        for (std::size_t c = 0; c < k_count; ++c)
          {
            if (c != ci && count_cluster[c] > 0)
              {
                double mean_c =
                  sum_dist[c] / static_cast<double> (count_cluster[c]);
                b_i = std::min (b_i, mean_c);
              }
          }
        if (std::isinf (b_i))
          {
            b_i = 0.0;
          }
        double denom = std::max (a_i, b_i);
        if (denom > 1.0e-15)
          {
            total_sil += (b_i - a_i) / denom;
          }
      }
    return total_sil / static_cast<double> (n);
  }
};

} // namespace bellows

#endif // BELLOWS_METRICS_METRICS_HPP
