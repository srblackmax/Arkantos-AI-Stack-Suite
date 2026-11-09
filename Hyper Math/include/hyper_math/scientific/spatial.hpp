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

#ifndef HYPER_MATH_SCIENTIFIC_SPATIAL_HPP
#define HYPER_MATH_SCIENTIFIC_SPATIAL_HPP

#include "hyper_math/scientific/linalg.hpp"
#include "hyper_math/tensor/nd_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace hyper_math::scientific::spatial
{

using tensor::nd_tensor;
using tensor::shape_type;

/* Pairwise metric distance evaluation between two D-dimensional vectors.  */
inline double
point_distance (const double *a,
                const double *b,
                std::size_t dim,
                const std::string &metric = "euclidean",
                double p = 2.0)
{
  if (metric == "euclidean")
    {
      double sum_sq = 0.0;
      for (std::size_t d = 0; d < dim; ++d)
        {
          const double diff = a[d] - b[d];
          sum_sq += diff * diff;
        }
      return std::sqrt (sum_sq);
    }
  else if (metric == "cityblock" || metric == "manhattan")
    {
      double sum_abs = 0.0;
      for (std::size_t d = 0; d < dim; ++d)
        {
          sum_abs += std::abs (a[d] - b[d]);
        }
      return sum_abs;
    }
  else if (metric == "chebyshev")
    {
      double best = 0.0;
      for (std::size_t d = 0; d < dim; ++d)
        {
          best = std::max (best, std::abs (a[d] - b[d]));
        }
      return best;
    }
  else if (metric == "cosine")
    {
      double dot_ab = 0.0;
      double na = 0.0;
      double nb = 0.0;
      for (std::size_t d = 0; d < dim; ++d)
        {
          dot_ab += a[d] * b[d];
          na += a[d] * a[d];
          nb += b[d] * b[d];
        }
      const double denom = std::sqrt (na * nb);
      return (denom > 0.0) ? (1.0 - dot_ab / denom) : 0.0;
    }
  else if (metric == "minkowski")
    {
      double sum_p = 0.0;
      for (std::size_t d = 0; d < dim; ++d)
        {
          sum_p += std::pow (std::abs (a[d] - b[d]), p);
        }
      return std::pow (sum_p, 1.0 / p);
    }
  else if (metric == "hamming")
    {
      std::size_t diff = 0;
      for (std::size_t d = 0; d < dim; ++d)
        {
          if (a[d] != b[d])
            {
              ++diff;
            }
        }
      return (dim > 0) ? (static_cast<double> (diff) / static_cast<double> (dim))
                       : 0.0;
    }
  else if (metric == "jaccard" || metric == "dice")
    {
      std::size_t c_tt = 0;
      std::size_t c_tf = 0;
      std::size_t c_ft = 0;
      for (std::size_t d = 0; d < dim; ++d)
        {
          const bool ba = (a[d] != 0.0);
          const bool bb = (b[d] != 0.0);
          if (ba && bb)
            {
              ++c_tt;
            }
          else if (ba && !bb)
            {
              ++c_tf;
            }
          else if (!ba && bb)
            {
              ++c_ft;
            }
        }
      if (metric == "jaccard")
        {
          const std::size_t denom = c_tt + c_tf + c_ft;
          return (denom > 0)
                 ? (static_cast<double> (c_tf + c_ft)
                    / static_cast<double> (denom))
                 : 0.0;
        }
      const std::size_t denom = 2 * c_tt + c_tf + c_ft;
      return (denom > 0)
             ? (static_cast<double> (c_tf + c_ft)
                / static_cast<double> (denom))
             : 0.0;
    }
  else if (metric == "canberra")
    {
      double s = 0.0;
      for (std::size_t d = 0; d < dim; ++d)
        {
          const double den = std::abs (a[d]) + std::abs (b[d]);
          if (den > 0.0)
            {
              s += std::abs (a[d] - b[d]) / den;
            }
        }
      return s;
    }
  else if (metric == "braycurtis")
    {
      double num = 0.0;
      double den = 0.0;
      for (std::size_t d = 0; d < dim; ++d)
        {
          num += std::abs (a[d] - b[d]);
          den += std::abs (a[d] + b[d]);
        }
      return (den > 0.0) ? (num / den) : 0.0;
    }
  else if (metric == "jensenshannon")
    {
      double sa = 0.0;
      double sb = 0.0;
      for (std::size_t d = 0; d < dim; ++d)
        {
          sa += a[d];
          sb += b[d];
        }
      double js = 0.0;
      for (std::size_t d = 0; d < dim; ++d)
        {
          const double pa = (sa > 0.0) ? (a[d] / sa) : 0.0;
          const double pb = (sb > 0.0) ? (b[d] / sb) : 0.0;
          const double m = 0.5 * (pa + pb);
          if (pa > 0.0 && m > 0.0)
            {
              js += 0.5 * pa * std::log (pa / m);
            }
          if (pb > 0.0 && m > 0.0)
            {
              js += 0.5 * pb * std::log (pb / m);
            }
        }
      return std::sqrt (std::max (0.0, js));
    }
  else if (metric == "sqeuclidean")
    {
      double sum_sq = 0.0;
      for (std::size_t d = 0; d < dim; ++d)
        {
          const double diff = a[d] - b[d];
          sum_sq += diff * diff;
        }
      return sum_sq;
    }
  else if (metric == "correlation")
    {
      double ma = 0.0;
      double mb = 0.0;
      for (std::size_t d = 0; d < dim; ++d)
        {
          ma += a[d];
          mb += b[d];
        }
      ma /= static_cast<double> (dim);
      mb /= static_cast<double> (dim);
      double num = 0.0;
      double da = 0.0;
      double db = 0.0;
      for (std::size_t d = 0; d < dim; ++d)
        {
          const double xa = a[d] - ma;
          const double xb = b[d] - mb;
          num += xa * xb;
          da += xa * xa;
          db += xb * xb;
        }
      const double denom = std::sqrt (da * db);
      return (denom > 0.0) ? (1.0 - num / denom) : 0.0;
    }
  else if (metric == "yule" || metric == "matching"
           || metric == "rogerstanimoto" || metric == "russellrao"
           || metric == "sokalmichener" || metric == "sokalsneath")
    {
      double c_tt = 0.0;
      double c_tf = 0.0;
      double c_ft = 0.0;
      double c_ff = 0.0;
      for (std::size_t d = 0; d < dim; ++d)
        {
          const bool ba = (a[d] != 0.0);
          const bool bb = (b[d] != 0.0);
          if (ba && bb)
            {
              c_tt += 1.0;
            }
          else if (ba && !bb)
            {
              c_tf += 1.0;
            }
          else if (!ba && bb)
            {
              c_ft += 1.0;
            }
          else
            {
              c_ff += 1.0;
            }
        }
      const double r = c_tf + c_ft;
      const double n = static_cast<double> (dim);
      if (metric == "matching")
        {
          return r / n;
        }
      if (metric == "rogerstanimoto" || metric == "sokalmichener")
        {
          return (2.0 * r) / (c_tt + c_ff + 2.0 * r);
        }
      if (metric == "russellrao")
        {
          return (n - c_tt) / n;
        }
      if (metric == "sokalsneath")
        {
          return (2.0 * r) / (c_tt + 2.0 * r);
        }
      const double denom = c_tt * c_ff + c_tf * c_ft;
      return (denom > 0.0) ? ((2.0 * c_tf * c_ft) / denom) : 0.0;
    }
  throw hyper_math_error ("unknown spatial distance metric: " + metric);
}

/* Cross-distance matrix between two 2D point collections X_A (m_a x dim)
   and X_B (m_b x dim) (SciPy spatial.distance.cdist equivalent).  */
inline nd_tensor<double>
cdist (const nd_tensor<double> &xa,
       const nd_tensor<double> &xb,
       const std::string &metric = "euclidean",
       double p = 2.0)
{
  if (xa.ndim () != 2 || xb.ndim () != 2 || xa.shape ()[1] != xb.shape ()[1])
    {
      throw shape_error ("cdist requires 2D tensors with equal column count");
    }
  const std::size_t ma = xa.shape ()[0];
  const std::size_t mb = xb.shape ()[0];
  const std::size_t dim = xa.shape ()[1];

  const nd_tensor<double> ca = xa.as_contiguous ();
  const nd_tensor<double> cb = xb.as_contiguous ();
  nd_tensor<double> out (shape_type {ma, mb});

  for (std::size_t i = 0; i < ma; ++i)
    {
      const double *pa = ca.raw_data () + i * dim;
      for (std::size_t j = 0; j < mb; ++j)
        {
          const double *pb = cb.raw_data () + j * dim;
          out (i, j) = point_distance (pa, pb, dim, metric, p);
        }
    }
  return out;
}

/* Condensed 1D pairwise distance vector of length m*(m-1)/2 (SciPy pdist).  */
inline nd_tensor<double>
pdist (const nd_tensor<double> &x,
       const std::string &metric = "euclidean",
       double p = 2.0)
{
  if (x.ndim () != 2)
    {
      throw shape_error ("pdist requires a 2D point tensor");
    }
  const std::size_t m = x.shape ()[0];
  const std::size_t dim = x.shape ()[1];
  const nd_tensor<double> cx = x.as_contiguous ();

  const std::size_t n_pairs = (m * (m - 1)) / 2;
  nd_tensor<double> out (shape_type {n_pairs});
  std::size_t k = 0;
  for (std::size_t i = 0; i < m; ++i)
    {
      const double *pi = cx.raw_data () + i * dim;
      for (std::size_t j = i + 1; j < m; ++j)
        {
          const double *pj = cx.raw_data () + j * dim;
          out[k++] = point_distance (pi, pj, dim, metric, p);
        }
    }
  return out;
}

/* Convert condensed 1D pdist vector to symmetric square matrix or vice versa
   (SciPy squareform).  */
inline nd_tensor<double>
squareform (const nd_tensor<double> &v)
{
  if (v.ndim () == 1)
    {
      const std::size_t len = v.size ();
      if (len == 0)
        {
          return nd_tensor<double> (shape_type {0, 0});
        }
      const std::size_t m = static_cast<std::size_t> (
        std::round (0.5 * (1.0 + std::sqrt (1.0 + 8.0 * static_cast<double> (len)))));
      if ((m * (m - 1)) / 2 != len)
        {
          throw shape_error (
            "invalid condensed distance vector length in squareform");
        }
      nd_tensor<double> mat (shape_type {m, m}, 0.0);
      std::size_t k = 0;
      for (std::size_t i = 0; i < m; ++i)
        {
          for (std::size_t j = i + 1; j < m; ++j)
            {
              const double d = v.eval_at (k++);
              mat (i, j) = d;
              mat (j, i) = d;
            }
        }
      return mat;
    }
  else if (v.ndim () == 2 && v.shape ()[0] == v.shape ()[1])
    {
      const std::size_t m = v.shape ()[0];
      nd_tensor<double> cond (shape_type {(m * (m - 1)) / 2});
      std::size_t k = 0;
      for (std::size_t i = 0; i < m; ++i)
        {
          for (std::size_t j = i + 1; j < m; ++j)
            {
              cond[k++] = v (i, j);
            }
        }
      return cond;
    }
  throw shape_error ("squareform requires 1D condensed vector or 2D square matrix");
}

/* Clean-room k-d Tree for fast O(log N) spatial nearest-neighbor and radius
   ball queries (SciPy spatial.KDTree equivalent).  */
class kd_tree
{
private:
  struct kd_node
  {
    std::size_t point_idx;
    std::size_t split_axis;
    double split_val;
    std::unique_ptr<kd_node> left;
    std::unique_ptr<kd_node> right;
  };

  nd_tensor<double> points_;
  std::size_t n_points_;
  std::size_t dim_;
  std::unique_ptr<kd_node> root_;

  std::unique_ptr<kd_node>
  build_recursive (std::vector<std::size_t> &indices,
                   std::size_t begin,
                   std::size_t end,
                   std::size_t depth)
  {
    if (begin >= end)
      {
        return nullptr;
      }
    const std::size_t axis = depth % dim_;
    const std::size_t mid = begin + (end - begin) / 2;

    std::nth_element (
      indices.begin () + static_cast<std::ptrdiff_t> (begin),
      indices.begin () + static_cast<std::ptrdiff_t> (mid),
      indices.begin () + static_cast<std::ptrdiff_t> (end),
      [this, axis] (std::size_t a, std::size_t b) {
        return points_ (a, axis) < points_ (b, axis);
      });

    auto node = std::make_unique<kd_node> ();
    node->point_idx = indices[mid];
    node->split_axis = axis;
    node->split_val = points_ (node->point_idx, axis);
    node->left = build_recursive (indices, begin, mid, depth + 1);
    node->right = build_recursive (indices, mid + 1, end, depth + 1);
    return node;
  }

  double
  dist_sq_to_point (std::size_t idx, const nd_tensor<double> &query_pt) const noexcept
  {
    double s = 0.0;
    for (std::size_t d = 0; d < dim_; ++d)
      {
        const double diff = points_ (idx, d) - query_pt.eval_at (d);
        s += diff * diff;
      }
    return s;
  }

  using heap_item = std::pair<double, std::size_t>;

  void
  knn_search (const kd_node *node,
              const nd_tensor<double> &query_pt,
              std::size_t k,
              std::priority_queue<heap_item> &max_heap) const
  {
    if (node == nullptr)
      {
        return;
      }
    const double d_sq = dist_sq_to_point (node->point_idx, query_pt);
    if (max_heap.size () < k)
      {
        max_heap.emplace (d_sq, node->point_idx);
      }
    else if (d_sq < max_heap.top ().first)
      {
        max_heap.pop ();
        max_heap.emplace (d_sq, node->point_idx);
      }

    const double q_val = query_pt.eval_at (node->split_axis);
    const double plane_diff = q_val - node->split_val;
    const kd_node *first =
      (plane_diff <= 0.0) ? node->left.get () : node->right.get ();
    const kd_node *second =
      (plane_diff <= 0.0) ? node->right.get () : node->left.get ();

    knn_search (first, query_pt, k, max_heap);
    if (max_heap.size () < k
        || (plane_diff * plane_diff) < max_heap.top ().first)
      {
        knn_search (second, query_pt, k, max_heap);
      }
  }

  void
  ball_search (const kd_node *node,
               const nd_tensor<double> &query_pt,
               double r_sq,
               std::vector<std::size_t> &matches) const
  {
    if (node == nullptr)
      {
        return;
      }
    if (dist_sq_to_point (node->point_idx, query_pt) <= r_sq)
      {
        matches.push_back (node->point_idx);
      }
    const double plane_diff =
      query_pt.eval_at (node->split_axis) - node->split_val;
    const kd_node *first =
      (plane_diff <= 0.0) ? node->left.get () : node->right.get ();
    const kd_node *second =
      (plane_diff <= 0.0) ? node->right.get () : node->left.get ();

    ball_search (first, query_pt, r_sq, matches);
    if ((plane_diff * plane_diff) <= r_sq)
      {
        ball_search (second, query_pt, r_sq, matches);
      }
  }

public:
  struct query_result
  {
    std::vector<double> distances;
    std::vector<std::size_t> indices;
  };

  explicit
  kd_tree (const nd_tensor<double> &points)
    : points_ (points.as_contiguous ()),
      n_points_ (0),
      dim_ (0),
      root_ (nullptr)
  {
    if (points_.ndim () != 2 || points_.size () == 0)
      {
        throw shape_error ("kd_tree requires non-empty 2D point tensor");
      }
    n_points_ = points_.shape ()[0];
    dim_ = points_.shape ()[1];
    std::vector<std::size_t> indices (n_points_);
    std::iota (indices.begin (), indices.end (), 0);
    root_ = build_recursive (indices, 0, n_points_, 0);
  }

  query_result
  query (const nd_tensor<double> &query_pt, std::size_t k = 1) const
  {
    if (query_pt.size () != dim_)
      {
        throw shape_error ("kd_tree::query point dimension mismatch");
      }
    if (k == 0)
      {
        return query_result {};
      }
    std::priority_queue<heap_item> max_heap;
    knn_search (root_.get (), query_pt, k, max_heap);

    const std::size_t found = max_heap.size ();
    query_result res;
    res.distances.resize (found);
    res.indices.resize (found);
    for (std::size_t i = found; i > 0; --i)
      {
        res.distances[i - 1] = std::sqrt (max_heap.top ().first);
        res.indices[i - 1] = max_heap.top ().second;
        max_heap.pop ();
      }
    return res;
  }

  std::vector<std::size_t>
  query_ball_point (const nd_tensor<double> &query_pt, double radius) const
  {
    if (query_pt.size () != dim_)
      {
        throw shape_error ("kd_tree::query_ball_point dimension mismatch");
      }
    if (radius < 0.0)
      {
        return {};
      }
    std::vector<std::size_t> matches;
    ball_search (root_.get (), query_pt, radius * radius, matches);
    std::sort (matches.begin (), matches.end ());
    return matches;
  }
};

/* 2-D Convex Hull via Andrew's Monotone Chain O(N log N) algorithm
   (SciPy spatial.ConvexHull 2D equivalent).
   Computes counter-clockwise hull vertex indices, perimeter, and area.  */
struct convex_hull_result
{
  std::vector<std::size_t> vertices;
  double perimeter;
  double area;
};

inline convex_hull_result
convex_hull_2d (const nd_tensor<double> &points)
{
  if (points.ndim () != 2 || points.shape ()[1] != 2 || points.shape ()[0] < 3)
    {
      throw shape_error ("convex_hull_2d requires (N, 2) tensor with N >= 3");
    }
  const std::size_t n = points.shape ()[0];
  std::vector<std::size_t> order (n);
  std::iota (order.begin (), order.end (), 0);
  std::sort (order.begin (), order.end (), [&points] (std::size_t a, std::size_t b) {
    if (points (a, 0) != points (b, 0))
      {
        return points (a, 0) < points (b, 0);
      }
    return points (a, 1) < points (b, 1);
  });

  auto cross = [&points] (std::size_t o, std::size_t a, std::size_t b) {
    return (points (a, 0) - points (o, 0)) * (points (b, 1) - points (o, 1))
           - (points (a, 1) - points (o, 1)) * (points (b, 0) - points (o, 0));
  };

  std::vector<std::size_t> hull;
  hull.reserve (2 * n);

  /* Lower hull.  */
  for (std::size_t idx : order)
    {
      while (hull.size () >= 2
             && cross (hull[hull.size () - 2], hull.back (), idx) <= 0.0)
        {
          hull.pop_back ();
        }
      hull.push_back (idx);
    }

  /* Upper hull.  */
  const std::size_t lower_size = hull.size ();
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t> (n) - 2; i >= 0; --i)
    {
      const std::size_t idx = order[static_cast<std::size_t> (i)];
      while (hull.size () > lower_size
             && cross (hull[hull.size () - 2], hull.back (), idx) <= 0.0)
        {
          hull.pop_back ();
        }
      hull.push_back (idx);
    }

  hull.pop_back ();

  /* Compute perimeter and Shoelace polygon area.  */
  const std::size_t h = hull.size ();
  double perim = 0.0;
  double twice_area = 0.0;
  for (std::size_t i = 0; i < h; ++i)
    {
      const std::size_t u = hull[i];
      const std::size_t v = hull[(i + 1) % h];
      perim += std::hypot (points (v, 0) - points (u, 0),
                           points (v, 1) - points (u, 1));
      twice_area += points (u, 0) * points (v, 1)
                    - points (v, 0) * points (u, 1);
    }

  return convex_hull_result {
    std::move (hull), perim, 0.5 * std::abs (twice_area)
  };
}

/* Directed and Symmetric Hausdorff Distance between two point clouds U and V
   (SciPy spatial.distance.directed_hausdorff).  */
inline double
directed_hausdorff (const nd_tensor<double> &u, const nd_tensor<double> &v)
{
  const nd_tensor<double> dists = cdist (u, v, "euclidean");
  const std::size_t nu = dists.shape ()[0];
  const std::size_t nv = dists.shape ()[1];
  double max_min = 0.0;
  for (std::size_t i = 0; i < nu; ++i)
    {
      double min_d = constants::inf;
      for (std::size_t j = 0; j < nv; ++j)
        {
          min_d = std::min (min_d, dists (i, j));
        }
      max_min = std::max (max_min, min_d);
    }
  return max_min;
}

inline double
hausdorff_distance (const nd_tensor<double> &u, const nd_tensor<double> &v)
{
  return std::max (directed_hausdorff (u, v), directed_hausdorff (v, u));
}

/* Orthogonal Procrustes Analysis: centers, normalizes to unit Frobenius norm,
   and finds optimal orthogonal rotation/reflection aligning mtx2 onto mtx1
   (SciPy spatial.procrustes equivalent).  */
struct procrustes_result
{
  nd_tensor<double> mtx1;
  nd_tensor<double> mtx2;
  double disparity;
};

inline procrustes_result
procrustes (const nd_tensor<double> &data1, const nd_tensor<double> &data2)
{
  if (data1.ndim () != 2 || data2.ndim () != 2
      || data1.shape () != data2.shape ())
    {
      throw shape_error ("procrustes requires 2D matrices of identical shape");
    }
  const std::size_t m = data1.shape ()[0];
  const std::size_t n = data1.shape ()[1];

  const nd_tensor<double> mu1 = tensor::mean (data1, 0);
  const nd_tensor<double> mu2 = tensor::mean (data2, 0);
  nd_tensor<double> a (shape_type {m, n});
  nd_tensor<double> b (shape_type {m, n});
  for (std::size_t i = 0; i < m; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          a (i, j) = data1 (i, j) - mu1[j];
          b (i, j) = data2 (i, j) - mu2[j];
        }
    }

  const double norm_a = linalg::norm (a);
  const double norm_b = linalg::norm (b);
  if (norm_a <= 0.0 || norm_b <= 0.0)
    {
      throw hyper_math_error ("procrustes inputs must contain >1 unique point");
    }
  a /= norm_a;
  b /= norm_b;

  /* Orthogonal Procrustes via SVD of A^T * B: R = V * U^T.  */
  const auto sv = linalg::svd (tensor::matmul (a.t (), b));
  const nd_tensor<double> r_mat = tensor::matmul (sv.vt.t (), sv.u.t ());
  const double scale = tensor::sum (sv.s);
  nd_tensor<double> b_aligned = scale * tensor::matmul (b, r_mat);

  double disp = 0.0;
  for (std::size_t i = 0; i < a.size (); ++i)
    {
      const double diff = a.eval_at (i) - b_aligned.eval_at (i);
      disp += diff * diff;
    }
  return procrustes_result {std::move (a), std::move (b_aligned), disp};
}

/* Bowyer-Watson 2-D Delaunay Triangulation (SciPy spatial.Delaunay equivalent).
   Returns an (M, 3) tensor of counter-clockwise triangle vertex indices
   satisfying the empty circumcircle property.  */
inline nd_tensor<std::size_t>
delaunay_2d (const nd_tensor<double> &points)
{
  if (points.ndim () != 2 || points.shape ()[1] != 2 || points.shape ()[0] < 3)
    {
      throw shape_error ("delaunay_2d requires (N, 2) tensor with N >= 3");
    }
  const std::size_t n = points.shape ()[0];

  /* Compute bounding box and construct a supertriangle enclosing all points. */
  double min_x = points (0, 0);
  double max_x = points (0, 0);
  double min_y = points (0, 1);
  double max_y = points (0, 1);
  for (std::size_t i = 1; i < n; ++i)
    {
      min_x = std::min (min_x, points (i, 0));
      max_x = std::max (max_x, points (i, 0));
      min_y = std::min (min_y, points (i, 1));
      max_y = std::max (max_y, points (i, 1));
    }
  const double dx = std::max (max_x - min_x, 1.0);
  const double dy = std::max (max_y - min_y, 1.0);
  const double dmax = std::max (dx, dy);
  const double mid_x = 0.5 * (min_x + max_x);
  const double mid_y = 0.5 * (min_y + max_y);

  std::vector<std::pair<double, double>> pts (n + 3);
  for (std::size_t i = 0; i < n; ++i)
    {
      pts[i] = {points (i, 0), points (i, 1)};
    }
  pts[n] = {mid_x - 20.0 * dmax, mid_y - dmax};
  pts[n + 1] = {mid_x, mid_y + 20.0 * dmax};
  pts[n + 2] = {mid_x + 20.0 * dmax, mid_y - dmax};

  struct tri_t
  {
    std::size_t a;
    std::size_t b;
    std::size_t c;
  };

  auto in_circumcircle = [&pts] (const tri_t &t, std::size_t p_idx) noexcept {
    const double ax = pts[t.a].first;
    const double ay = pts[t.a].second;
    const double bx = pts[t.b].first;
    const double by = pts[t.b].second;
    const double cx = pts[t.c].first;
    const double cy = pts[t.c].second;

    const double d =
      2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (std::abs (d) < 1e-15)
      {
        return false;
      }
    const double a2 = ax * ax + ay * ay;
    const double b2 = bx * bx + by * by;
    const double c2 = cx * cx + cy * cy;
    const double ux =
      (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    const double uy =
      (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    const double r2 = (ax - ux) * (ax - ux) + (ay - uy) * (ay - uy);
    const double px = pts[p_idx].first - ux;
    const double py = pts[p_idx].second - uy;
    return (px * px + py * py) <= r2 * (1.0 + 1e-12);
  };

  std::vector<tri_t> triangles {{n, n + 1, n + 2}};

  for (std::size_t i = 0; i < n; ++i)
    {
      std::vector<tri_t> bad_triangles;
      std::vector<tri_t> good_triangles;
      for (const auto &t : triangles)
        {
          if (in_circumcircle (t, i))
            {
              bad_triangles.push_back (t);
            }
          else
            {
              good_triangles.push_back (t);
            }
        }

      struct edge_t
      {
        std::size_t u;
        std::size_t v;
        bool
        same_undirected (const edge_t &other) const noexcept
        {
          return (u == other.u && v == other.v)
                 || (u == other.v && v == other.u);
        }
      };

      std::vector<edge_t> polygon;
      for (const auto &bt : bad_triangles)
        {
          const edge_t edges[3] = {{bt.a, bt.b}, {bt.b, bt.c}, {bt.c, bt.a}};
          for (const auto &e : edges)
            {
              bool shared = false;
              for (const auto &other : bad_triangles)
                {
                  if (&bt == &other)
                    {
                      continue;
                    }
                  const edge_t oedges[3] = {
                    {other.a, other.b}, {other.b, other.c}, {other.c, other.a}
                  };
                  if (e.same_undirected (oedges[0])
                      || e.same_undirected (oedges[1])
                      || e.same_undirected (oedges[2]))
                    {
                      shared = true;
                      break;
                    }
                }
              if (!shared)
                {
                  polygon.push_back (e);
                }
            }
        }

      triangles = std::move (good_triangles);
      for (const auto &e : polygon)
        {
          /* Ensure counter-clockwise orientation of (e.u, e.v, i).  */
          const double cross =
            (pts[e.v].first - pts[e.u].first) * (pts[i].second - pts[e.u].second)
            - (pts[e.v].second - pts[e.u].second) * (pts[i].first - pts[e.u].first);
          if (cross >= 0.0)
            {
              triangles.push_back (tri_t {e.u, e.v, i});
            }
          else
            {
              triangles.push_back (tri_t {e.v, e.u, i});
            }
        }
    }

  std::vector<tri_t> final_tris;
  for (const auto &t : triangles)
    {
      if (t.a < n && t.b < n && t.c < n)
        {
          final_tris.push_back (t);
        }
    }

  nd_tensor<std::size_t> simplices (shape_type {final_tris.size (), 3});
  for (std::size_t i = 0; i < final_tris.size (); ++i)
    {
      simplices (i, 0) = final_tris[i].a;
      simplices (i, 1) = final_tris[i].b;
      simplices (i, 2) = final_tris[i].c;
    }
  return simplices;
}

/* 3-D Spatial Rotation `rotation_3d` (`scipy.spatial.transform.Rotation`
   equivalent): unit quaternion (x, y, z, w) internal representation with
   rotation matrix, axis-angle (`rotvec`), ZYX Euler angles, vector application,
   group composition, inversion, and spherical linear interpolation `slerp`.  */
class rotation_3d
{
private:
  double x_ = 0.0;
  double y_ = 0.0;
  double z_ = 0.0;
  double w_ = 1.0;

public:
  rotation_3d () = default;

  rotation_3d (double x, double y, double z, double w)
    : x_ (x), y_ (y), z_ (z), w_ (w)
  {
    const double n = std::sqrt (x_ * x_ + y_ * y_ + z_ * z_ + w_ * w_);
    if (n <= 0.0)
      {
        throw hyper_math_error ("rotation_3d zero quaternion norm");
      }
    x_ /= n;
    y_ /= n;
    z_ /= n;
    w_ /= n;
  }

  static rotation_3d
  from_quat (const nd_tensor<double> &q)
  {
    if (q.size () != 4)
      {
        throw shape_error ("rotation_3d::from_quat requires 4-element (x, y, z, w)");
      }
    return rotation_3d (q[0], q[1], q[2], q[3]);
  }

  static rotation_3d
  from_rotvec (const nd_tensor<double> &rv)
  {
    if (rv.size () != 3)
      {
        throw shape_error ("rotation_3d::from_rotvec requires 3-element vector");
      }
    const double theta =
      std::sqrt (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
    if (theta < 1e-14)
      {
        return rotation_3d (0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2], 1.0);
      }
    const double s = std::sin (0.5 * theta) / theta;
    return rotation_3d (rv[0] * s, rv[1] * s, rv[2] * s,
                        std::cos (0.5 * theta));
  }

  static rotation_3d
  from_euler_zyx (double yaw_z, double pitch_y, double roll_x)
  {
    const double cy = std::cos (0.5 * yaw_z);
    const double sy = std::sin (0.5 * yaw_z);
    const double cp = std::cos (0.5 * pitch_y);
    const double sp = std::sin (0.5 * pitch_y);
    const double cr = std::cos (0.5 * roll_x);
    const double sr = std::sin (0.5 * roll_x);

    const double w = cr * cp * cy + sr * sp * sy;
    const double x = sr * cp * cy - cr * sp * sy;
    const double y = cr * sp * cy + sr * cp * sy;
    const double z = cr * cp * sy - sr * sp * cy;
    return rotation_3d (x, y, z, w);
  }

  static rotation_3d
  from_matrix (const nd_tensor<double> &r)
  {
    if (r.ndim () != 2 || r.shape ()[0] != 3 || r.shape ()[1] != 3)
      {
        throw shape_error ("rotation_3d::from_matrix requires a 3x3 matrix");
      }
    const double tr = r (0, 0) + r (1, 1) + r (2, 2);
    if (tr > 0.0)
      {
        const double s = 0.5 / std::sqrt (tr + 1.0);
        return rotation_3d ((r (2, 1) - r (1, 2)) * s,
                            (r (0, 2) - r (2, 0)) * s,
                            (r (1, 0) - r (0, 1)) * s,
                            0.25 / s);
      }
    if (r (0, 0) > r (1, 1) && r (0, 0) > r (2, 2))
      {
        const double s =
          2.0 * std::sqrt (1.0 + r (0, 0) - r (1, 1) - r (2, 2));
        return rotation_3d (0.25 * s,
                            (r (0, 1) + r (1, 0)) / s,
                            (r (0, 2) + r (2, 0)) / s,
                            (r (2, 1) - r (1, 2)) / s);
      }
    if (r (1, 1) > r (2, 2))
      {
        const double s =
          2.0 * std::sqrt (1.0 + r (1, 1) - r (0, 0) - r (2, 2));
        return rotation_3d ((r (0, 1) + r (1, 0)) / s,
                            0.25 * s,
                            (r (1, 2) + r (2, 1)) / s,
                            (r (0, 2) - r (2, 0)) / s);
      }
    const double s =
      2.0 * std::sqrt (1.0 + r (2, 2) - r (0, 0) - r (1, 1));
    return rotation_3d ((r (0, 2) + r (2, 0)) / s,
                        (r (1, 2) + r (2, 1)) / s,
                        0.25 * s,
                        (r (1, 0) - r (0, 1)) / s);
  }

  nd_tensor<double>
  as_quat () const
  {
    return nd_tensor<double> {x_, y_, z_, w_};
  }

  nd_tensor<double>
  as_matrix () const
  {
    const double xx = x_ * x_;
    const double yy = y_ * y_;
    const double zz = z_ * z_;
    const double xy = x_ * y_;
    const double xz = x_ * z_;
    const double yz = y_ * z_;
    const double wx = w_ * x_;
    const double wy = w_ * y_;
    const double wz = w_ * z_;

    return nd_tensor<double> {
      {1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)},
      {2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)},
      {2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)}
    };
  }

  nd_tensor<double>
  as_rotvec () const
  {
    const double sign = (w_ < 0.0) ? -1.0 : 1.0;
    const double vx = sign * x_;
    const double vy = sign * y_;
    const double vz = sign * z_;
    const double vw = std::clamp (sign * w_, -1.0, 1.0);
    const double sin_half = std::sqrt (vx * vx + vy * vy + vz * vz);
    if (sin_half < 1e-14)
      {
        return nd_tensor<double> {2.0 * vx, 2.0 * vy, 2.0 * vz};
      }
    const double theta = 2.0 * std::atan2 (sin_half, vw);
    const double scale = theta / sin_half;
    return nd_tensor<double> {vx * scale, vy * scale, vz * scale};
  }

  rotation_3d
  inv () const noexcept
  {
    return rotation_3d (-x_, -y_, -z_, w_);
  }

  rotation_3d
  operator* (const rotation_3d &rhs) const noexcept
  {
    return rotation_3d (
      w_ * rhs.x_ + x_ * rhs.w_ + y_ * rhs.z_ - z_ * rhs.y_,
      w_ * rhs.y_ - x_ * rhs.z_ + y_ * rhs.w_ + z_ * rhs.x_,
      w_ * rhs.z_ + x_ * rhs.y_ - y_ * rhs.x_ + z_ * rhs.w_,
      w_ * rhs.w_ - x_ * rhs.x_ - y_ * rhs.y_ - z_ * rhs.z_);
  }

  nd_tensor<double>
  apply (const nd_tensor<double> &v) const
  {
    if (v.size () != 3)
      {
        throw shape_error ("rotation_3d::apply requires a 3-element vector");
      }
    return tensor::matmul (as_matrix (), v);
  }
};

inline rotation_3d
slerp (const rotation_3d &r0, const rotation_3d &r1, double t)
{
  const nd_tensor<double> q0 = r0.as_quat ();
  nd_tensor<double> q1 = r1.as_quat ();
  double dot =
    q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
  if (dot < 0.0)
    {
      dot = -dot;
      q1 = -q1;
    }
  if (dot > 0.9995)
    {
      return rotation_3d (
        q0[0] + t * (q1[0] - q0[0]),
        q0[1] + t * (q1[1] - q0[1]),
        q0[2] + t * (q1[2] - q0[2]),
        q0[3] + t * (q1[3] - q0[3]));
    }
  const double theta_0 = std::acos (std::clamp (dot, -1.0, 1.0));
  const double theta = theta_0 * t;
  const double sin_theta = std::sin (theta);
  const double sin_theta_0 = std::sin (theta_0);
  const double s0 = std::cos (theta) - dot * sin_theta / sin_theta_0;
  const double s1 = sin_theta / sin_theta_0;
  return rotation_3d (
    s0 * q0[0] + s1 * q1[0],
    s0 * q0[1] + s1 * q1[1],
    s0 * q0[2] + s1 * q1[2],
    s0 * q0[3] + s1 * q1[3]);
}

/* 2-D Voronoi Diagram dual to Delaunay Triangulation `voronoi_2d`, computing
   triangle circumcenters (`vertices`) and shared-edge dual `ridge_points`
   (SciPy spatial.Voronoi equivalent).  */
struct voronoi_2d_result
{
  nd_tensor<double> vertices;
  std::vector<std::pair<std::size_t, std::size_t>> ridge_points;
};

inline voronoi_2d_result
voronoi_2d (const nd_tensor<double> &points)
{
  const nd_tensor<std::size_t> tris = delaunay_2d (points);
  const std::size_t n_tris = tris.shape ()[0];
  nd_tensor<double> verts (shape_type {n_tris, 2}, 0.0);
  std::vector<std::pair<std::size_t, std::size_t>> ridges;

  for (std::size_t t = 0; t < n_tris; ++t)
    {
      const std::size_t i0 = tris (t, 0);
      const std::size_t i1 = tris (t, 1);
      const std::size_t i2 = tris (t, 2);
      const double ax = points (i0, 0);
      const double ay = points (i0, 1);
      const double bx = points (i1, 0);
      const double by = points (i1, 1);
      const double cx = points (i2, 0);
      const double cy = points (i2, 1);
      const double d =
        2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
      const double a2 = ax * ax + ay * ay;
      const double b2 = bx * bx + by * by;
      const double c2 = cx * cx + cy * cy;
      verts (t, 0) =
        (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
      verts (t, 1) =
        (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;

      const std::size_t v_ids[3] = {i0, i1, i2};
      for (int e = 0; e < 3; ++e)
        {
          std::size_t u = v_ids[e];
          std::size_t v = v_ids[(e + 1) % 3];
          if (u > v)
            {
              std::swap (u, v);
            }
          ridges.emplace_back (u, v);
        }
    }
  std::sort (ridges.begin (), ridges.end ());
  ridges.erase (std::unique (ridges.begin (), ridges.end ()), ridges.end ());
  return voronoi_2d_result {std::move (verts), std::move (ridges)};
}

/* =========================================================================
   Mahalanobis & Standardized Euclidean Distances (`mahalanobis_distance`,
   `seuclidean_distance`), Spherical Voronoi & Geometric Slerp
   (`spherical_voronoi`, `geometric_slerp`), and 2D Halfspace Intersection
   (`halfspace_intersection_2d`).
   ========================================================================= */

inline double
mahalanobis_distance (const nd_tensor<double> &u,
                      const nd_tensor<double> &v,
                      const nd_tensor<double> &vi)
{
  if (u.size () != v.size ()
      || vi.ndim () != 2
      || vi.shape ()[0] != u.size ()
      || vi.shape ()[1] != u.size ())
    {
      throw shape_error ("mahalanobis_distance dimension mismatch");
    }
  const nd_tensor<double> diff = u - v;
  const nd_tensor<double> vi_d = tensor::matmul (vi, diff);
  return std::sqrt (std::max (0.0, tensor::dot (diff, vi_d)));
}

inline double
seuclidean_distance (const nd_tensor<double> &u,
                     const nd_tensor<double> &v,
                     const nd_tensor<double> &var_vec)
{
  if (u.size () != v.size () || u.size () != var_vec.size ())
    {
      throw shape_error ("seuclidean_distance dimension mismatch");
    }
  double sum = 0.0;
  for (std::size_t i = 0; i < u.size (); ++i)
    {
      const double d = u.eval_at (i) - v.eval_at (i);
      sum += (d * d) / std::max (1e-15, var_vec.eval_at (i));
    }
  return std::sqrt (sum);
}

inline nd_tensor<double>
geometric_slerp (const nd_tensor<double> &start,
                 const nd_tensor<double> &end,
                 const nd_tensor<double> &t)
{
  if (start.size () != end.size () || start.size () == 0)
    {
      throw shape_error ("geometric_slerp requires matching non-empty start and end vectors");
    }
  const std::size_t d = start.size ();
  const std::size_t m = t.size ();
  const double ns = linalg::norm (start);
  const double ne = linalg::norm (end);
  const nd_tensor<double> u0 = start / ns;
  const nd_tensor<double> u1 = end / ne;
  const double cos_omega = std::clamp (tensor::dot (u0, u1), -1.0, 1.0);
  const double omega = std::acos (cos_omega);
  const double sin_omega = std::sin (omega);
  nd_tensor<double> out (shape_type {m, d});
  for (std::size_t i = 0; i < m; ++i)
    {
      const double ti = t.eval_at (i);
      const double w0 =
        (sin_omega > 1e-12) ? (std::sin ((1.0 - ti) * omega) / sin_omega)
                            : (1.0 - ti);
      const double w1 =
        (sin_omega > 1e-12) ? (std::sin (ti * omega) / sin_omega) : ti;
      for (std::size_t j = 0; j < d; ++j)
        {
          out (i, j) = w0 * u0.eval_at (j) + w1 * u1.eval_at (j);
        }
    }
  return out;
}

inline nd_tensor<double>
spherical_voronoi (const nd_tensor<double> &points, double radius = 1.0)
{
  if (points.ndim () != 2 || points.shape ()[1] != 3 || points.shape ()[0] < 4)
    {
      throw shape_error ("spherical_voronoi requires (N, 3) points with N >= 4");
    }
  const std::size_t n = points.shape ()[0];
  std::vector<std::vector<double>> verts;
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = i + 1; j < n; ++j)
        {
          for (std::size_t k = j + 1; k < n; ++k)
            {
              const double ux = points (j, 0) - points (i, 0);
              const double uy = points (j, 1) - points (i, 1);
              const double uz = points (j, 2) - points (i, 2);
              const double vx = points (k, 0) - points (i, 0);
              const double vy = points (k, 1) - points (i, 1);
              const double vz = points (k, 2) - points (i, 2);
              double nx = uy * vz - uz * vy;
              double ny = uz * vx - ux * vz;
              double nz = ux * vy - uy * vx;
              const double nrm = std::sqrt (nx * nx + ny * ny + nz * nz);
              if (nrm < 1e-12)
                {
                  continue;
                }
              nx *= radius / nrm;
              ny *= radius / nrm;
              nz *= radius / nrm;
              for (double sgn : {1.0, -1.0})
                {
                  const double cx = sgn * nx;
                  const double cy = sgn * ny;
                  const double cz = sgn * nz;
                  const double dot_ijk =
                    cx * points (i, 0) + cy * points (i, 1) + cz * points (i, 2);
                  bool empty = true;
                  for (std::size_t m = 0; m < n; ++m)
                    {
                      if (m == i || m == j || m == k)
                        {
                          continue;
                        }
                      const double dot_m =
                        cx * points (m, 0) + cy * points (m, 1) + cz * points (m, 2);
                      if (dot_m > dot_ijk + 1e-9)
                        {
                          empty = false;
                          break;
                        }
                    }
                  if (empty)
                    {
                      verts.push_back ({cx, cy, cz});
                    }
                }
            }
        }
    }
  nd_tensor<double> out (shape_type {verts.size (), 3});
  for (std::size_t r = 0; r < verts.size (); ++r)
    {
      out (r, 0) = verts[r][0];
      out (r, 1) = verts[r][1];
      out (r, 2) = verts[r][2];
    }
  return out;
}

inline nd_tensor<double>
halfspace_intersection_2d (const nd_tensor<double> &halfspaces)
{
  /* Each row of `halfspaces` (M, 3) represents A_i x + B_i y + C_i <= 0.  */
  if (halfspaces.ndim () != 2 || halfspaces.shape ()[1] != 3
      || halfspaces.shape ()[0] < 3)
    {
      throw shape_error ("halfspace_intersection_2d requires (M, 3) matrix with M >= 3");
    }
  const std::size_t m = halfspaces.shape ()[0];
  std::vector<std::pair<double, double>> pts;
  for (std::size_t i = 0; i < m; ++i)
    {
      for (std::size_t j = i + 1; j < m; ++j)
        {
          const double a1 = halfspaces (i, 0);
          const double b1 = halfspaces (i, 1);
          const double c1 = halfspaces (i, 2);
          const double a2 = halfspaces (j, 0);
          const double b2 = halfspaces (j, 1);
          const double c2 = halfspaces (j, 2);
          const double det = a1 * b2 - a2 * b1;
          if (std::abs (det) < 1e-12)
            {
              continue;
            }
          const double x = (b1 * c2 - b2 * c1) / det;
          const double y = (c1 * a2 - c2 * a1) / det;
          bool feasible = true;
          for (std::size_t k = 0; k < m; ++k)
            {
              if (halfspaces (k, 0) * x + halfspaces (k, 1) * y
                  + halfspaces (k, 2) > 1e-9)
                {
                  feasible = false;
                  break;
                }
            }
          if (feasible)
            {
              pts.emplace_back (x, y);
            }
        }
    }
  nd_tensor<double> out (shape_type {pts.size (), 2});
  for (std::size_t i = 0; i < pts.size (); ++i)
    {
      out (i, 0) = pts[i].first;
      out (i, 1) = pts[i].second;
    }
  return out;
}

} // namespace hyper_math::scientific::spatial

#endif /* HYPER_MATH_SCIENTIFIC_SPATIAL_HPP */
