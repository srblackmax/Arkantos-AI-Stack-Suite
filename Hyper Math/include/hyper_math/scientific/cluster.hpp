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

#ifndef HYPER_MATH_SCIENTIFIC_CLUSTER_HPP
#define HYPER_MATH_SCIENTIFIC_CLUSTER_HPP

#include "hyper_math/core/prng.hpp"
#include "hyper_math/scientific/spatial.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace hyper_math::scientific::cluster
{

using tensor::nd_tensor;
using tensor::shape_type;

struct kmeans_result
{
  nd_tensor<double> centroids;
  nd_tensor<std::size_t> labels;
  double inertia;
};

/* Clean-room k-Means++ Clustering with Arthur-Vassilvitskii D^2 seeding and
   Lloyd's expectation-maximization refinement (SciPy cluster.vq.kmeans2).  */
inline kmeans_result
kmeans2 (const nd_tensor<double> &data,
         std::size_t k,
         std::size_t max_iter = 100,
         double tol = 1e-6,
         random_engine &rng = global_rng ())
{
  if (data.ndim () != 2 || data.shape ()[0] < k || k == 0)
    {
      throw shape_error ("kmeans2 requires (N, D) tensor with N >= k >= 1");
    }
  const std::size_t n = data.shape ()[0];
  const std::size_t dim = data.shape ()[1];

  nd_tensor<double> centroids (shape_type {k, dim});
  const std::size_t first_idx =
    static_cast<std::size_t> (rng.randint (0, static_cast<std::int64_t> (n)));
  for (std::size_t d = 0; d < dim; ++d)
    {
      centroids (0, d) = data (first_idx, d);
    }

  std::vector<double> min_dist_sq (n, std::numeric_limits<double>::max ());
  for (std::size_t c = 1; c < k; ++c)
    {
      double sum_sq = 0.0;
      for (std::size_t i = 0; i < n; ++i)
        {
          double d_sq = 0.0;
          for (std::size_t d = 0; d < dim; ++d)
            {
              const double diff = data (i, d) - centroids (c - 1, d);
              d_sq += diff * diff;
            }
          min_dist_sq[i] = std::min (min_dist_sq[i], d_sq);
          sum_sq += min_dist_sq[i];
        }

      double target = rng.uniform (0.0, sum_sq);
      std::size_t chosen = n - 1;
      for (std::size_t i = 0; i < n; ++i)
        {
          target -= min_dist_sq[i];
          if (target <= 0.0)
            {
              chosen = i;
              break;
            }
        }
      for (std::size_t d = 0; d < dim; ++d)
        {
          centroids (c, d) = data (chosen, d);
        }
    }

  nd_tensor<std::size_t> labels (shape_type {n}, 0);
  double inertia = 0.0;

  for (std::size_t iter = 0; iter < max_iter; ++iter)
    {
      inertia = 0.0;
      for (std::size_t i = 0; i < n; ++i)
        {
          double best_d2 = std::numeric_limits<double>::max ();
          std::size_t best_c = 0;
          for (std::size_t c = 0; c < k; ++c)
            {
              double d2 = 0.0;
              for (std::size_t d = 0; d < dim; ++d)
                {
                  const double diff = data (i, d) - centroids (c, d);
                  d2 += diff * diff;
                }
              if (d2 < best_d2)
                {
                  best_d2 = d2;
                  best_c = c;
                }
            }
          labels[i] = best_c;
          inertia += best_d2;
        }

      nd_tensor<double> new_centroids (shape_type {k, dim}, 0.0);
      std::vector<std::size_t> counts (k, 0);
      for (std::size_t i = 0; i < n; ++i)
        {
          const std::size_t cl = labels[i];
          counts[cl] += 1;
          for (std::size_t d = 0; d < dim; ++d)
            {
              new_centroids (cl, d) += data (i, d);
            }
        }

      double max_shift = 0.0;
      for (std::size_t c = 0; c < k; ++c)
        {
          if (counts[c] > 0)
            {
              double shift_sq = 0.0;
              for (std::size_t d = 0; d < dim; ++d)
                {
                  new_centroids (c, d) /= static_cast<double> (counts[c]);
                  const double diff = new_centroids (c, d) - centroids (c, d);
                  shift_sq += diff * diff;
                }
              max_shift = std::max (max_shift, std::sqrt (shift_sq));
            }
          else
            {
              for (std::size_t d = 0; d < dim; ++d)
                {
                  new_centroids (c, d) = centroids (c, d);
                }
            }
        }

      centroids = std::move (new_centroids);
      if (max_shift < tol)
        {
          break;
        }
    }

  inertia = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      double best_d2 = std::numeric_limits<double>::max ();
      std::size_t best_c = 0;
      for (std::size_t c = 0; c < k; ++c)
        {
          double d2 = 0.0;
          for (std::size_t d = 0; d < dim; ++d)
            {
              const double diff = data (i, d) - centroids (c, d);
              d2 += diff * diff;
            }
          if (d2 < best_d2)
            {
              best_d2 = d2;
              best_c = c;
            }
        }
      labels[i] = best_c;
      inertia += best_d2;
    }

  return kmeans_result {std::move (centroids), std::move (labels), inertia};
}

/* Agglomerative Hierarchical Clustering linkage matrix (N - 1 x 4)
   supporting "single", "complete", "average" (UPGMA), and "ward" linkage
   via Lance-Williams distance updates (SciPy cluster.hierarchy.linkage).  */
inline nd_tensor<double>
linkage (const nd_tensor<double> &y, const std::string &method = "ward")
{
  const nd_tensor<double> dist_mat =
    (y.ndim () == 2) ? spatial::cdist (y, y) : spatial::squareform (y);
  const std::size_t n = dist_mat.shape ()[0];
  if (n < 2)
    {
      throw shape_error ("linkage requires at least 2 observations");
    }

  nd_tensor<double> d = dist_mat.copy ();
  std::vector<bool> active (n, true);
  std::vector<std::size_t> cluster_id (n);
  std::iota (cluster_id.begin (), cluster_id.end (), 0);
  std::vector<double> cluster_size (n, 1.0);

  nd_tensor<double> z (shape_type {n - 1, 4});

  for (std::size_t step = 0; step < n - 1; ++step)
    {
      double min_d = std::numeric_limits<double>::max ();
      std::size_t best_u = 0;
      std::size_t best_v = 1;
      for (std::size_t i = 0; i < n; ++i)
        {
          if (!active[i])
            {
              continue;
            }
          for (std::size_t j = i + 1; j < n; ++j)
            {
              if (active[j] && d (i, j) < min_d)
                {
                  min_d = d (i, j);
                  best_u = i;
                  best_v = j;
                }
            }
        }

      const double c1 =
        static_cast<double> (std::min (cluster_id[best_u], cluster_id[best_v]));
      const double c2 =
        static_cast<double> (std::max (cluster_id[best_u], cluster_id[best_v]));
      const double nu = cluster_size[best_u];
      const double nv = cluster_size[best_v];
      const double new_sz = nu + nv;

      z (step, 0) = c1;
      z (step, 1) = c2;
      z (step, 2) = min_d;
      z (step, 3) = new_sz;

      /* Update distances from merged cluster (stored at best_u) to all other
         active clusters k using the Lance-Williams recurrence.  */
      for (std::size_t k = 0; k < n; ++k)
        {
          if (!active[k] || k == best_u || k == best_v)
            {
              continue;
            }
          const double du = d (best_u, k);
          const double dv = d (best_v, k);
          double dk = 0.0;
          if (method == "single")
            {
              dk = std::min (du, dv);
            }
          else if (method == "complete")
            {
              dk = std::max (du, dv);
            }
          else if (method == "average")
            {
              dk = (nu * du + nv * dv) / new_sz;
            }
          else if (method == "ward")
            {
              const double nk = cluster_size[k];
              const double tot = nu + nv + nk;
              dk = std::sqrt (((nu + nk) * du * du
                               + (nv + nk) * dv * dv
                               - nk * min_d * min_d) / tot);
            }
          else
            {
              throw hyper_math_error ("unknown linkage method: " + method);
            }
          d (best_u, k) = dk;
          d (k, best_u) = dk;
        }

      active[best_v] = false;
      cluster_id[best_u] = n + step;
      cluster_size[best_u] = new_sz;
    }

  return z;
}

/* Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
   Assigns cluster IDs 0..K-1 to density-reachable points and -1 to noise.  */
struct dbscan_result
{
  nd_tensor<std::int64_t> labels;
  std::vector<std::size_t> core_sample_indices;
  std::size_t n_clusters;
};

inline dbscan_result
dbscan (const nd_tensor<double> &data,
        double eps = 0.5,
        std::size_t min_samples = 5)
{
  if (data.ndim () != 2 || data.shape ()[0] == 0)
    {
      throw shape_error ("dbscan requires a non-empty 2D (N, D) data matrix");
    }
  const std::size_t n = data.shape ()[0];
  const nd_tensor<double> dists = spatial::cdist (data, data, "euclidean");

  std::vector<std::vector<std::size_t>> neighborhoods (n);
  std::vector<bool> is_core (n, false);
  std::vector<std::size_t> core_indices;
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          if (dists (i, j) <= eps)
            {
              neighborhoods[i].push_back (j);
            }
        }
      if (neighborhoods[i].size () >= min_samples)
        {
          is_core[i] = true;
          core_indices.push_back (i);
        }
    }

  nd_tensor<std::int64_t> labels (shape_type {n}, -1);
  std::vector<bool> visited (n, false);
  std::int64_t cluster_id = 0;

  for (std::size_t i = 0; i < n; ++i)
    {
      if (visited[i] || !is_core[i])
        {
          continue;
        }
      visited[i] = true;
      labels[i] = cluster_id;
      std::vector<std::size_t> queue = neighborhoods[i];
      std::size_t head = 0;
      while (head < queue.size ())
        {
          const std::size_t q = queue[head++];
          if (!visited[q])
            {
              visited[q] = true;
              if (is_core[q])
                {
                  for (std::size_t nb : neighborhoods[q])
                    {
                      queue.push_back (nb);
                    }
                }
            }
          if (labels[q] == -1)
            {
              labels[q] = cluster_id;
            }
        }
      ++cluster_id;
    }

  return dbscan_result {
    std::move (labels),
    std::move (core_indices),
    static_cast<std::size_t> (cluster_id)
  };
}

/* Flat Cluster Extraction `fcluster` ("distance" or "maxclust" criteria,
   returning 1-based cluster labels) and Cophenetic Correlation Distance
   `cophenet` (SciPy cluster.hierarchy.fcluster and cophenet equivalents).  */
inline nd_tensor<std::int64_t>
fcluster (const nd_tensor<double> &z,
          double t,
          const std::string &criterion = "distance")
{
  if (z.ndim () != 2 || z.shape ()[1] != 4 || z.shape ()[0] == 0)
    {
      throw shape_error ("fcluster requires (N - 1, 4) linkage matrix");
    }
  const std::size_t n_merges = z.shape ()[0];
  const std::size_t n = n_merges + 1;
  const std::size_t total_nodes = 2 * n - 1;

  std::vector<std::size_t> parent (total_nodes);
  std::iota (parent.begin (), parent.end (), 0);

  auto find_root = [&] (std::size_t u) {
    while (parent[u] != u)
      {
        parent[u] = parent[parent[u]];
        u = parent[u];
      }
    return u;
  };

  std::size_t max_steps = n_merges;
  if (criterion == "maxclust")
    {
      const std::size_t k_target =
        std::max<std::size_t> (1, static_cast<std::size_t> (std::round (t)));
      max_steps = (k_target >= n) ? 0 : (n - k_target);
    }
  else if (criterion != "distance")
    {
      throw hyper_math_error ("unknown fcluster criterion: " + criterion);
    }

  for (std::size_t step = 0; step < max_steps; ++step)
    {
      if (criterion == "distance" && z (step, 2) > t)
        {
          break;
        }
      const std::size_t c1 = static_cast<std::size_t> (z (step, 0));
      const std::size_t c2 = static_cast<std::size_t> (z (step, 1));
      const std::size_t merged = n + step;
      parent[find_root (c1)] = merged;
      parent[find_root (c2)] = merged;
    }

  std::map<std::size_t, std::int64_t> root_to_label;
  std::int64_t next_label = 1;
  nd_tensor<std::int64_t> labels (shape_type {n});
  for (std::size_t i = 0; i < n; ++i)
    {
      const std::size_t r = find_root (i);
      auto it = root_to_label.find (r);
      if (it == root_to_label.end ())
        {
          root_to_label[r] = next_label;
          labels[i] = next_label++;
        }
      else
        {
          labels[i] = it->second;
        }
    }
  return labels;
}

inline std::pair<double, nd_tensor<double>>
cophenet (const nd_tensor<double> &z, const nd_tensor<double> &y_pdist)
{
  if (z.ndim () != 2 || z.shape ()[1] != 4 || z.shape ()[0] == 0)
    {
      throw shape_error ("cophenet requires (N - 1, 4) linkage matrix");
    }
  const std::size_t n_merges = z.shape ()[0];
  const std::size_t n = n_merges + 1;
  const std::size_t n_pairs = (n * (n - 1)) / 2;
  if (y_pdist.size () != n_pairs)
    {
      throw shape_error ("cophenet requires condensed distance vector of length N*(N-1)/2");
    }

  std::vector<std::vector<std::size_t>> members (2 * n - 1);
  for (std::size_t i = 0; i < n; ++i)
    {
      members[i] = {i};
    }

  nd_tensor<double> c_mat (shape_type {n, n}, 0.0);
  for (std::size_t step = 0; step < n_merges; ++step)
    {
      const std::size_t c1 = static_cast<std::size_t> (z (step, 0));
      const std::size_t c2 = static_cast<std::size_t> (z (step, 1));
      const double dist = z (step, 2);
      const std::size_t merged = n + step;

      for (std::size_t u : members[c1])
        {
          for (std::size_t v : members[c2])
            {
              c_mat (u, v) = dist;
              c_mat (v, u) = dist;
            }
        }
      members[merged] = members[c1];
      members[merged].insert (members[merged].end (),
                              members[c2].begin (), members[c2].end ());
    }

  nd_tensor<double> c_vec (shape_type {n_pairs});
  std::size_t idx = 0;
  double mean_y = 0.0;
  double mean_c = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = i + 1; j < n; ++j)
        {
          c_vec[idx] = c_mat (i, j);
          mean_y += y_pdist[idx];
          mean_c += c_vec[idx];
          ++idx;
        }
    }
  mean_y /= static_cast<double> (n_pairs);
  mean_c /= static_cast<double> (n_pairs);

  double cov_yc = 0.0;
  double var_y = 0.0;
  double var_c = 0.0;
  for (std::size_t k = 0; k < n_pairs; ++k)
    {
      const double dy = y_pdist[k] - mean_y;
      const double dc = c_vec[k] - mean_c;
      cov_yc += dy * dc;
      var_y += dy * dy;
      var_c += dc * dc;
    }
  const double corr =
    (var_y > 0.0 && var_c > 0.0) ? (cov_yc / std::sqrt (var_y * var_c)) : 0.0;
  return {corr, std::move (c_vec)};
}

/* Vector Quantization `whiten` & `vq`, and Linkage Hierarchy `inconsistent`
   & `leaders` (SciPy cluster.vq and cluster.hierarchy equivalents).  */
inline nd_tensor<double>
whiten (const nd_tensor<double> &obs)
{
  if (obs.ndim () != 2 || obs.shape ()[0] == 0)
    {
      throw shape_error ("whiten requires a non-empty 2D observation matrix");
    }
  const std::size_t m = obs.shape ()[0];
  const std::size_t d = obs.shape ()[1];
  nd_tensor<double> out (shape_type {m, d});
  for (std::size_t j = 0; j < d; ++j)
    {
      double mean = 0.0;
      for (std::size_t i = 0; i < m; ++i)
        {
          mean += obs (i, j);
        }
      mean /= static_cast<double> (m);
      double var = 0.0;
      for (std::size_t i = 0; i < m; ++i)
        {
          const double diff = obs (i, j) - mean;
          var += diff * diff;
        }
      const double std_j = std::sqrt (var / static_cast<double> (m));
      const double scale = (std_j > 1e-15) ? (1.0 / std_j) : 1.0;
      for (std::size_t i = 0; i < m; ++i)
        {
          out (i, j) = obs (i, j) * scale;
        }
    }
  return out;
}

inline std::pair<nd_tensor<std::size_t>, nd_tensor<double>>
vq (const nd_tensor<double> &obs, const nd_tensor<double> &code_book)
{
  if (obs.ndim () != 2 || code_book.ndim () != 2
      || obs.shape ()[1] != code_book.shape ()[1]
      || code_book.shape ()[0] == 0)
    {
      throw shape_error ("vq requires 2D obs and code_book with matching columns");
    }
  const std::size_t m = obs.shape ()[0];
  const std::size_t k = code_book.shape ()[0];
  const std::size_t dim = obs.shape ()[1];
  nd_tensor<std::size_t> codes (shape_type {m});
  nd_tensor<double> dists (shape_type {m});
  for (std::size_t i = 0; i < m; ++i)
    {
      double best_d2 = constants::inf;
      std::size_t best_c = 0;
      for (std::size_t c = 0; c < k; ++c)
        {
          double d2 = 0.0;
          for (std::size_t j = 0; j < dim; ++j)
            {
              const double diff = obs (i, j) - code_book (c, j);
              d2 += diff * diff;
            }
          if (d2 < best_d2)
            {
              best_d2 = d2;
              best_c = c;
            }
        }
      codes[i] = best_c;
      dists[i] = std::sqrt (best_d2);
    }
  return {std::move (codes), std::move (dists)};
}

inline nd_tensor<double>
inconsistent (const nd_tensor<double> &z)
{
  if (z.ndim () != 2 || z.shape ()[1] != 4 || z.shape ()[0] == 0)
    {
      throw shape_error ("inconsistent requires (N - 1, 4) linkage matrix");
    }
  const std::size_t m = z.shape ()[0];
  const std::size_t n = m + 1;
  nd_tensor<double> r (shape_type {m, 4}, 0.0);
  for (std::size_t i = 0; i < m; ++i)
    {
      std::vector<double> heights = {z (i, 2)};
      const std::size_t c1 = static_cast<std::size_t> (z (i, 0));
      const std::size_t c2 = static_cast<std::size_t> (z (i, 1));
      if (c1 >= n)
        {
          heights.push_back (z (c1 - n, 2));
        }
      if (c2 >= n)
        {
          heights.push_back (z (c2 - n, 2));
        }
      double mean_h = 0.0;
      for (double h : heights)
        {
          mean_h += h;
        }
      mean_h /= static_cast<double> (heights.size ());
      double var_h = 0.0;
      for (double h : heights)
        {
          const double d = h - mean_h;
          var_h += d * d;
        }
      const double std_h =
        (heights.size () > 1)
        ? std::sqrt (var_h / static_cast<double> (heights.size () - 1))
        : 0.0;
      r (i, 0) = mean_h;
      r (i, 1) = std_h;
      r (i, 2) = static_cast<double> (heights.size ());
      r (i, 3) = (std_h > 1e-15) ? ((z (i, 2) - mean_h) / std_h) : 0.0;
    }
  return r;
}

inline std::pair<nd_tensor<std::size_t>, nd_tensor<std::int64_t>>
leaders (const nd_tensor<double> &z, const nd_tensor<std::int64_t> &T)
{
  const std::size_t m = z.shape ()[0];
  const std::size_t n = m + 1;
  if (T.size () != n)
    {
      throw shape_error ("leaders requires flat cluster vector of length N");
    }
  std::vector<std::int64_t> node_label (2 * n - 1, -1);
  for (std::size_t i = 0; i < n; ++i)
    {
      node_label[i] = T[i];
    }
  for (std::size_t i = 0; i < m; ++i)
    {
      const std::size_t c1 = static_cast<std::size_t> (z (i, 0));
      const std::size_t c2 = static_cast<std::size_t> (z (i, 1));
      if (node_label[c1] != -1 && node_label[c1] == node_label[c2])
        {
          node_label[n + i] = node_label[c1];
        }
    }
  std::map<std::int64_t, std::size_t> cluster_root;
  for (std::size_t u = 0; u < 2 * n - 1; ++u)
    {
      if (node_label[u] != -1)
        {
          cluster_root[node_label[u]] = u;
        }
    }
  nd_tensor<std::size_t> L (shape_type {cluster_root.size ()});
  nd_tensor<std::int64_t> M (shape_type {cluster_root.size ()});
  std::size_t idx = 0;
  for (const auto &[cid, root_id] : cluster_root)
    {
      L[idx] = root_id;
      M[idx] = cid;
      ++idx;
    }
  return {std::move (L), std::move (M)};
}

} // namespace hyper_math::scientific::cluster

#endif /* HYPER_MATH_SCIENTIFIC_CLUSTER_HPP */
