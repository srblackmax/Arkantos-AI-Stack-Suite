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

#ifndef HYPER_MATH_SCIENTIFIC_SPARSE_HPP
#define HYPER_MATH_SCIENTIFIC_SPARSE_HPP

#include "hyper_math/scientific/linalg.hpp"
#include "hyper_math/tensor/reductions.hpp"

#include <cmath>
#include <queue>
#include <utility>
#include <vector>

namespace hyper_math::scientific::sparse
{

using tensor::nd_tensor;
using tensor::shape_type;

/* Compressed Sparse Row (CSR) matrix (SciPy sparse.csr_matrix equivalent). */
template <typename T = double>
class csr_matrix
{
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<T> data_;
  std::vector<std::size_t> indices_;
  std::vector<std::size_t> indptr_;

public:
  csr_matrix (std::size_t rows,
              std::size_t cols,
              std::vector<T> data,
              std::vector<std::size_t> indices,
              std::vector<std::size_t> indptr)
    : rows_ (rows),
      cols_ (cols),
      data_ (std::move (data)),
      indices_ (std::move (indices)),
      indptr_ (std::move (indptr))
  {
    if (indptr_.size () != rows_ + 1 || data_.size () != indices_.size ())
      {
        throw shape_error ("invalid CSR matrix array lengths");
      }
    if (indptr_.front () != 0 || indptr_.back () != data_.size ())
      {
        throw shape_error ("invalid CSR indptr bounds");
      }
    for (std::size_t i = 0; i < rows_; ++i)
      {
        if (indptr_[i] > indptr_[i + 1])
          {
            throw shape_error ("CSR indptr must be non-decreasing");
          }
      }
    for (std::size_t idx : indices_)
      {
        if (idx >= cols_)
          {
            throw index_error ("CSR column index out of bounds");
          }
      }
  }

  static csr_matrix
  from_dense (const nd_tensor<T> &dense, T drop_tol = T {0})
  {
    if (dense.ndim () != 2)
      {
        throw shape_error ("csr_matrix::from_dense requires a 2D tensor");
      }
    const std::size_t r = dense.shape ()[0];
    const std::size_t c = dense.shape ()[1];
    std::vector<T> data;
    std::vector<std::size_t> indices;
    std::vector<std::size_t> indptr (r + 1, 0);

    for (std::size_t i = 0; i < r; ++i)
      {
        for (std::size_t j = 0; j < c; ++j)
          {
            const T val = dense (i, j);
            if (std::abs (static_cast<double> (val))
                > static_cast<double> (drop_tol))
              {
                data.push_back (val);
                indices.push_back (j);
              }
          }
        indptr[i + 1] = data.size ();
      }
    return csr_matrix (r, c, std::move (data),
                       std::move (indices), std::move (indptr));
  }

  nd_tensor<T>
  to_dense () const
  {
    nd_tensor<T> out (shape_type {rows_, cols_}, T {0});
    for (std::size_t i = 0; i < rows_; ++i)
      {
        for (std::size_t p = indptr_[i]; p < indptr_[i + 1]; ++p)
          {
            out (i, indices_[p]) = data_[p];
          }
      }
    return out;
  }

  std::size_t
  rows () const noexcept
  {
    return rows_;
  }

  std::size_t
  cols () const noexcept
  {
    return cols_;
  }

  std::size_t
  nnz () const noexcept
  {
    return data_.size ();
  }

  const std::vector<T> &
  data () const noexcept
  {
    return data_;
  }

  const std::vector<std::size_t> &
  indices () const noexcept
  {
    return indices_;
  }

  const std::vector<std::size_t> &
  indptr () const noexcept
  {
    return indptr_;
  }

  /* Sparse matrix-vector multiplication: y = A * x.  */
  nd_tensor<T>
  spmv (const nd_tensor<T> &x) const
  {
    if (x.size () != cols_)
      {
        throw shape_error ("csr_matrix::spmv vector length mismatch");
      }
    nd_tensor<T> y (shape_type {rows_}, T {0});
    for (std::size_t i = 0; i < rows_; ++i)
      {
        T acc = T {0};
        for (std::size_t p = indptr_[i]; p < indptr_[i + 1]; ++p)
          {
            acc += data_[p] * x.eval_at (indices_[p]);
          }
        y[i] = acc;
      }
    return y;
  }

  /* Sparse-dense matrix-matrix multiplication: C = A * B.  */
  nd_tensor<T>
  spmm (const nd_tensor<T> &b) const
  {
    if (b.ndim () != 2 || b.shape ()[0] != cols_)
      {
        throw shape_error ("csr_matrix::spmm dimension mismatch");
      }
    const std::size_t n = b.shape ()[1];
    nd_tensor<T> out (shape_type {rows_, n}, T {0});
    for (std::size_t i = 0; i < rows_; ++i)
      {
        for (std::size_t p = indptr_[i]; p < indptr_[i + 1]; ++p)
          {
            const std::size_t k = indices_[p];
            const T val = data_[p];
            for (std::size_t j = 0; j < n; ++j)
              {
                out (i, j) += val * b (k, j);
              }
          }
      }
    return out;
  }

  csr_matrix<T>
  to_csr () const
  {
    return csr_matrix<T>::from_dense (to_dense ());
  }
};

/* Conjugate Gradient iterative solver for sparse SPD systems A * x = b
   (SciPy sparse.linalg.cg equivalent).  */
inline nd_tensor<double>
cg_solve (const csr_matrix<double> &a,
          const nd_tensor<double> &b,
          double tol = 1e-10,
          std::size_t max_iter = 1000)
{
  if (a.rows () != a.cols () || b.size () != a.rows ())
    {
      throw shape_error ("cg_solve requires square CSR matrix and matching b");
    }

  const std::size_t n = a.rows ();
  nd_tensor<double> x (shape_type {n}, 0.0);
  nd_tensor<double> r = b.copy ();
  nd_tensor<double> p = r.copy ();
  double rs_old = tensor::dot (r, r);

  if (std::sqrt (rs_old) < tol)
    {
      return x;
    }

  for (std::size_t iter = 0; iter < max_iter; ++iter)
    {
      const nd_tensor<double> ap = a.spmv (p);
      const double p_ap = tensor::dot (p, ap);
      if (std::abs (p_ap) < 1e-30)
        {
          break;
        }
      const double alpha = rs_old / p_ap;
      x += alpha * p;
      r -= alpha * ap;
      const double rs_new = tensor::dot (r, r);
      if (std::sqrt (rs_new) < tol)
        {
          break;
        }
      p = r + (rs_new / rs_old) * p;
      rs_old = rs_new;
    }

  return x;
}

/* Bi-Conjugate Gradient Stabilized (BiCGSTAB) iterative solver for general
   non-symmetric sparse linear systems A * x = b (SciPy sparse.linalg.bicgstab). */
inline nd_tensor<double>
bicgstab_solve (const csr_matrix<double> &a,
                const nd_tensor<double> &b,
                double tol = 1e-10,
                std::size_t max_iter = 1000)
{
  if (a.rows () != a.cols () || b.size () != a.rows ())
    {
      throw shape_error ("bicgstab_solve dimension mismatch");
    }
  const std::size_t n = a.rows ();
  nd_tensor<double> x (shape_type {n}, 0.0);
  nd_tensor<double> r = b.copy ();
  const nd_tensor<double> r_hat = r.copy ();

  double rho_old = 1.0;
  double alpha = 1.0;
  double omega = 1.0;
  nd_tensor<double> v (shape_type {n}, 0.0);
  nd_tensor<double> p (shape_type {n}, 0.0);

  for (std::size_t iter = 0; iter < max_iter; ++iter)
    {
      if (std::sqrt (tensor::dot (r, r)) < tol)
        {
          break;
        }
      const double rho_new = tensor::dot (r_hat, r);
      if (std::abs (rho_new) < 1e-30 || std::abs (omega) < 1e-30)
        {
          break;
        }
      const double beta = (rho_new / rho_old) * (alpha / omega);
      p = r + beta * (p - omega * v);
      v = a.spmv (p);
      const double rhat_v = tensor::dot (r_hat, v);
      if (std::abs (rhat_v) < 1e-30)
        {
          break;
        }
      alpha = rho_new / rhat_v;
      const nd_tensor<double> s = r - alpha * v;
      if (std::sqrt (tensor::dot (s, s)) < tol)
        {
          x += alpha * p;
          break;
        }
      const nd_tensor<double> t = a.spmv (s);
      const double tt = tensor::dot (t, t);
      if (tt < 1e-30)
        {
          break;
        }
      omega = tensor::dot (t, s) / tt;
      x += alpha * p + omega * s;
      r = s - omega * t;
      rho_old = rho_new;
    }

  return x;
}

/* Construct an N x N sparse identity matrix in CSR format (SciPy sparse.eye). */
inline csr_matrix<double>
sparse_eye (std::size_t n)
{
  std::vector<double> data (n, 1.0);
  std::vector<std::size_t> indices (n);
  std::vector<std::size_t> indptr (n + 1);
  for (std::size_t i = 0; i < n; ++i)
    {
      indices[i] = i;
      indptr[i] = i;
    }
  indptr[n] = n;
  return csr_matrix<double> (n, n,
                             std::move (data),
                             std::move (indices),
                             std::move (indptr));
}

/* Generalized Minimal Residual (GMRES) Krylov subspace iterative solver
   using Arnoldi orthogonalization (SciPy sparse.linalg.gmres equivalent).  */
inline nd_tensor<double>
gmres_solve (const csr_matrix<double> &a,
             const nd_tensor<double> &b,
             double tol = 1e-10,
             std::size_t max_iter = 0)
{
  if (a.rows () != a.cols () || b.size () != a.rows ())
    {
      throw shape_error ("gmres_solve dimension mismatch");
    }
  const std::size_t n = a.rows ();
  if (max_iter == 0 || max_iter > n)
    {
      max_iter = n;
    }

  nd_tensor<double> x (shape_type {n}, 0.0);
  const nd_tensor<double> r0 = b.copy ();
  const double beta = linalg::norm (r0);
  if (beta < tol)
    {
      return x;
    }

  std::vector<nd_tensor<double>> v_basis;
  v_basis.reserve (max_iter + 1);
  v_basis.push_back (r0 / beta);

  nd_tensor<double> h_mat (shape_type {max_iter + 1, max_iter}, 0.0);
  std::size_t k_used = 0;

  for (std::size_t j = 0; j < max_iter; ++j)
    {
      nd_tensor<double> w = a.spmv (v_basis[j]);
      for (std::size_t i = 0; i <= j; ++i)
        {
          const double hij = tensor::dot (w, v_basis[i]);
          h_mat (i, j) = hij;
          w -= hij * v_basis[i];
        }
      const double h_next = linalg::norm (w);
      h_mat (j + 1, j) = h_next;
      k_used = j + 1;
      if (h_next < 1e-14)
        {
          break;
        }
      v_basis.push_back (w / h_next);
    }

  nd_tensor<double> h_sub (shape_type {k_used + 1, k_used}, 0.0);
  for (std::size_t i = 0; i <= k_used; ++i)
    {
      for (std::size_t j = 0; j < k_used; ++j)
        {
          h_sub (i, j) = h_mat (i, j);
        }
    }
  nd_tensor<double> e1 (shape_type {k_used + 1}, 0.0);
  e1[0] = beta;

  const nd_tensor<double> y = linalg::lstsq (h_sub, e1);
  for (std::size_t j = 0; j < k_used; ++j)
    {
      x += y[j] * v_basis[j];
    }
  return x;
}

/* Coordinate Format Sparse Matrix `coo_matrix<T>`, Compressed Sparse Column
   Matrix `csc_matrix<T>`, Symmetric Lanczos Eigensolver `eigsh`, and
   Compressed Sparse Graph (`csgraph`) Algorithms: `connected_components`,
   `dijkstra`, `floyd_warshall`, `minimum_spanning_tree`, and
   `topological_sort` (SciPy sparse and sparse.csgraph equivalents).  */
template <typename T = double>
class coo_matrix
{
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<std::size_t> row_idx_;
  std::vector<std::size_t> col_idx_;
  std::vector<T> data_;

public:
  coo_matrix (std::size_t rows,
              std::size_t cols,
              std::vector<std::size_t> row_idx,
              std::vector<std::size_t> col_idx,
              std::vector<T> data)
    : rows_ (rows),
      cols_ (cols),
      row_idx_ (std::move (row_idx)),
      col_idx_ (std::move (col_idx)),
      data_ (std::move (data))
  {
    if (row_idx_.size () != col_idx_.size ()
        || row_idx_.size () != data_.size ())
      {
        throw shape_error ("coo_matrix triplet vectors must have equal length");
      }
  }

  std::size_t rows () const noexcept { return rows_; }
  std::size_t cols () const noexcept { return cols_; }
  std::size_t nnz () const noexcept { return data_.size (); }

  nd_tensor<T>
  to_dense () const
  {
    nd_tensor<T> out (shape_type {rows_, cols_}, T {0});
    for (std::size_t k = 0; k < data_.size (); ++k)
      {
        out (row_idx_[k], col_idx_[k]) += data_[k];
      }
    return out;
  }

  csr_matrix<T>
  to_csr () const
  {
    return csr_matrix<T>::from_dense (to_dense ());
  }
};

template <typename T = double>
class csc_matrix
{
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<T> data_;
  std::vector<std::size_t> indices_;
  std::vector<std::size_t> indptr_;

public:
  csc_matrix (std::size_t rows,
              std::size_t cols,
              std::vector<T> data,
              std::vector<std::size_t> indices,
              std::vector<std::size_t> indptr)
    : rows_ (rows),
      cols_ (cols),
      data_ (std::move (data)),
      indices_ (std::move (indices)),
      indptr_ (std::move (indptr))
  {
    if (indptr_.size () != cols_ + 1 || data_.size () != indices_.size ())
      {
        throw shape_error ("invalid CSC matrix array lengths");
      }
  }

  static csc_matrix
  from_dense (const nd_tensor<T> &dense, T drop_tol = T {0})
  {
    if (dense.ndim () != 2)
      {
        throw shape_error ("csc_matrix::from_dense requires a 2D tensor");
      }
    const std::size_t r = dense.shape ()[0];
    const std::size_t c = dense.shape ()[1];
    std::vector<T> data;
    std::vector<std::size_t> indices;
    std::vector<std::size_t> indptr (c + 1, 0);

    for (std::size_t j = 0; j < c; ++j)
      {
        for (std::size_t i = 0; i < r; ++i)
          {
            const T val = dense (i, j);
            if (std::abs (static_cast<double> (val))
                > static_cast<double> (drop_tol))
              {
                data.push_back (val);
                indices.push_back (i);
              }
          }
        indptr[j + 1] = data.size ();
      }
    return csc_matrix (r, c, std::move (data),
                       std::move (indices), std::move (indptr));
  }

  std::size_t rows () const noexcept { return rows_; }
  std::size_t cols () const noexcept { return cols_; }
  std::size_t nnz () const noexcept { return data_.size (); }

  nd_tensor<T>
  spmv (const nd_tensor<T> &x) const
  {
    if (x.size () != cols_)
      {
        throw shape_error ("csc_matrix::spmv dimension mismatch");
      }
    nd_tensor<T> y (shape_type {rows_}, T {0});
    for (std::size_t j = 0; j < cols_; ++j)
      {
        const T xj = x.eval_at (j);
        for (std::size_t k = indptr_[j]; k < indptr_[j + 1]; ++k)
          {
            y[indices_[k]] += data_[k] * xj;
          }
      }
    return y;
  }

  nd_tensor<T>
  to_dense () const
  {
    nd_tensor<T> out (shape_type {rows_, cols_}, T {0});
    for (std::size_t j = 0; j < cols_; ++j)
      {
        for (std::size_t k = indptr_[j]; k < indptr_[j + 1]; ++k)
          {
            out (indices_[k], j) = data_[k];
          }
      }
    return out;
  }

  csr_matrix<T>
  to_csr () const
  {
    return csr_matrix<T>::from_dense (to_dense ());
  }
};

/* Symmetric Lanczos Sparse Eigensolver `eigsh` returning the `k` largest-
   algebraic eigenvalues and eigenvectors (SciPy sparse.linalg.eigsh).  */
inline linalg::eigh_result
eigsh (const csr_matrix<double> &a, std::size_t k = 1)
{
  const std::size_t n = a.rows ();
  if (a.cols () != n || k == 0 || k > n)
    {
      throw shape_error ("eigsh requires square matrix and 1 <= k <= n");
    }
  const std::size_t m = std::min (n, std::max (2 * k + 4, n));
  std::vector<nd_tensor<double>> q_vecs;
  q_vecs.reserve (m);

  nd_tensor<double> q0 (shape_type {n});
  for (std::size_t i = 0; i < n; ++i)
    {
      q0[i] = 1.0 + 0.1 * static_cast<double> (i);
    }
  q0 = q0 / linalg::norm (q0);
  q_vecs.push_back (q0);

  nd_tensor<double> t_mat (shape_type {m, m}, 0.0);
  std::size_t steps = 0;

  for (std::size_t j = 0; j < m; ++j)
    {
      nd_tensor<double> w = a.spmv (q_vecs[j]);
      const double alpha = tensor::dot (q_vecs[j], w);
      t_mat (j, j) = alpha;
      steps = j + 1;

      /* Full reorthogonalization against all previous Lanczos vectors.  */
      for (std::size_t i = 0; i <= j; ++i)
        {
          const double proj = tensor::dot (q_vecs[i], w);
          w = w - proj * q_vecs[i];
        }
      const double beta = linalg::norm (w);
      if (beta < 1e-13 || j + 1 == m)
        {
          break;
        }
      t_mat (j, j + 1) = beta;
      t_mat (j + 1, j) = beta;
      q_vecs.push_back (w / beta);
    }

  nd_tensor<double> t_sub (shape_type {steps, steps}, 0.0);
  for (std::size_t i = 0; i < steps; ++i)
    {
      for (std::size_t j = 0; j < steps; ++j)
        {
          t_sub (i, j) = t_mat (i, j);
        }
    }

  const auto eg = linalg::eigh (t_sub);
  nd_tensor<double> evals (shape_type {k});
  nd_tensor<double> evecs (shape_type {n, k}, 0.0);
  for (std::size_t idx = 0; idx < k; ++idx)
    {
      const std::size_t src_col = steps - k + idx;
      evals[idx] = eg.eigenvalues[src_col];
      for (std::size_t s = 0; s < steps; ++s)
        {
          const double coeff = eg.eigenvectors (s, src_col);
          for (std::size_t r = 0; r < n; ++r)
            {
              evecs (r, idx) += coeff * q_vecs[s][r];
            }
        }
    }
  return linalg::eigh_result {std::move (evals), std::move (evecs)};
}

/* Compressed Sparse Graph (`csgraph`) Algorithms.  */
inline std::pair<std::size_t, nd_tensor<std::int64_t>>
connected_components (const csr_matrix<double> &adj)
{
  const std::size_t n = adj.rows ();
  if (adj.cols () != n)
    {
      throw shape_error ("connected_components requires a square adjacency matrix");
    }
  const nd_tensor<double> d = adj.to_dense ();
  nd_tensor<std::int64_t> labels (shape_type {n}, -1);
  std::int64_t comp_id = 0;
  std::vector<std::size_t> queue;

  for (std::size_t i = 0; i < n; ++i)
    {
      if (labels[i] != -1)
        {
          continue;
        }
      labels[i] = comp_id;
      queue.clear ();
      queue.push_back (i);
      std::size_t head = 0;
      while (head < queue.size ())
        {
          const std::size_t u = queue[head++];
          for (std::size_t v = 0; v < n; ++v)
            {
              if ((d (u, v) != 0.0 || d (v, u) != 0.0) && labels[v] == -1)
                {
                  labels[v] = comp_id;
                  queue.push_back (v);
                }
            }
        }
      ++comp_id;
    }
  return {static_cast<std::size_t> (comp_id), std::move (labels)};
}

inline nd_tensor<double>
dijkstra (const csr_matrix<double> &adj, std::size_t src)
{
  const std::size_t n = adj.rows ();
  if (adj.cols () != n || src >= n)
    {
      throw shape_error ("dijkstra requires square adjacency matrix and src < n");
    }
  const auto &data = adj.data ();
  const auto &indices = adj.indices ();
  const auto &indptr = adj.indptr ();

  nd_tensor<double> dist (shape_type {n}, constants::inf);
  dist[src] = 0.0;

  using item_t = std::pair<double, std::size_t>;
  std::priority_queue<item_t, std::vector<item_t>, std::greater<item_t>> pq;
  pq.emplace (0.0, src);

  while (!pq.empty ())
    {
      const auto [du, u] = pq.top ();
      pq.pop ();
      if (du > dist[u])
        {
          continue;
        }
      for (std::size_t k = indptr[u]; k < indptr[u + 1]; ++k)
        {
          const std::size_t v = indices[k];
          const double w = data[k];
          if (w > 0.0 && du + w < dist[v])
            {
              dist[v] = du + w;
              pq.emplace (dist[v], v);
            }
        }
    }
  return dist;
}

inline nd_tensor<double>
floyd_warshall (const csr_matrix<double> &adj)
{
  const std::size_t n = adj.rows ();
  if (adj.cols () != n)
    {
      throw shape_error ("floyd_warshall requires square adjacency matrix");
    }
  const nd_tensor<double> raw = adj.to_dense ();
  nd_tensor<double> dist (shape_type {n, n}, constants::inf);
  for (std::size_t i = 0; i < n; ++i)
    {
      dist (i, i) = 0.0;
      for (std::size_t j = 0; j < n; ++j)
        {
          if (i != j && raw (i, j) > 0.0)
            {
              dist (i, j) = raw (i, j);
            }
        }
    }
  for (std::size_t k = 0; k < n; ++k)
    {
      for (std::size_t i = 0; i < n; ++i)
        {
          for (std::size_t j = 0; j < n; ++j)
            {
              if (dist (i, k) + dist (k, j) < dist (i, j))
                {
                  dist (i, j) = dist (i, k) + dist (k, j);
                }
            }
        }
    }
  return dist;
}

inline csr_matrix<double>
minimum_spanning_tree (const csr_matrix<double> &adj)
{
  const std::size_t n = adj.rows ();
  if (adj.cols () != n)
    {
      throw shape_error ("minimum_spanning_tree requires square adjacency matrix");
    }
  const nd_tensor<double> g = adj.to_dense ();
  nd_tensor<double> mst (shape_type {n, n}, 0.0);
  if (n == 0)
    {
      return csr_matrix<double>::from_dense (mst);
    }

  std::vector<bool> in_mst (n, false);
  std::vector<double> key (n, constants::inf);
  std::vector<std::size_t> parent (n, n);
  key[0] = 0.0;

  for (std::size_t step = 0; step < n; ++step)
    {
      std::size_t u = n;
      double best = constants::inf;
      for (std::size_t v = 0; v < n; ++v)
        {
          if (!in_mst[v] && key[v] < best)
            {
              best = key[v];
              u = v;
            }
        }
      if (u == n)
        {
          break;
        }
      in_mst[u] = true;
      if (parent[u] < n)
        {
          mst (parent[u], u) = key[u];
        }
      for (std::size_t v = 0; v < n; ++v)
        {
          double w = g (u, v);
          if (w == 0.0 || (g (v, u) > 0.0 && g (v, u) < w))
            {
              w = g (v, u);
            }
          if (!in_mst[v] && w > 0.0 && w < key[v])
            {
              key[v] = w;
              parent[v] = u;
            }
        }
    }
  return csr_matrix<double>::from_dense (mst);
}

inline std::vector<std::size_t>
topological_sort (const csr_matrix<double> &adj)
{
  const std::size_t n = adj.rows ();
  if (adj.cols () != n)
    {
      throw shape_error ("topological_sort requires square adjacency matrix");
    }
  const nd_tensor<double> g = adj.to_dense ();
  std::vector<std::size_t> in_deg (n, 0);
  for (std::size_t u = 0; u < n; ++u)
    {
      for (std::size_t v = 0; v < n; ++v)
        {
          if (u != v && g (u, v) != 0.0)
            {
              ++in_deg[v];
            }
        }
    }
  std::vector<std::size_t> order;
  order.reserve (n);
  for (std::size_t i = 0; i < n; ++i)
    {
      if (in_deg[i] == 0)
        {
          order.push_back (i);
        }
    }
  std::size_t head = 0;
  while (head < order.size ())
    {
      const std::size_t u = order[head++];
      for (std::size_t v = 0; v < n; ++v)
        {
          if (u != v && g (u, v) != 0.0)
            {
              if (--in_deg[v] == 0)
                {
                  order.push_back (v);
                }
            }
        }
    }
  if (order.size () != n)
    {
      throw convergence_error ("topological_sort graph contains a directed cycle");
    }
  return order;
}

/* Sparse Diagonal Matrix `dia_matrix<T>`, Direct Sparse Solver `spsolve` /
   `splu`, Sparse Kronecker Product `sparse_kron`, and Extended `csgraph`
   Algorithms (`bellman_ford`, `breadth_first_order`, `depth_first_order`,
   `graph_laplacian`, `maximum_flow`, `reverse_cuthill_mckee`)
   (SciPy sparse and csgraph equivalents).  */
template <typename T = double>
class dia_matrix
{
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<std::ptrdiff_t> offsets_;
  nd_tensor<T> data_;

public:
  dia_matrix (std::size_t rows,
              std::size_t cols,
              std::vector<std::ptrdiff_t> offsets,
              nd_tensor<T> data)
    : rows_ (rows),
      cols_ (cols),
      offsets_ (std::move (offsets)),
      data_ (std::move (data))
  {
    if (data_.ndim () != 2 || data_.shape ()[0] != offsets_.size ())
      {
        throw shape_error ("dia_matrix data rows must match offsets length");
      }
  }

  std::size_t rows () const noexcept { return rows_; }
  std::size_t cols () const noexcept { return cols_; }

  nd_tensor<T>
  spmv (const nd_tensor<T> &x) const
  {
    if (x.size () != cols_)
      {
        throw shape_error ("dia_matrix::spmv vector length mismatch");
      }
    nd_tensor<T> y (shape_type {rows_}, T {0});
    const std::size_t L = data_.shape ()[1];
    for (std::size_t d = 0; d < offsets_.size (); ++d)
      {
        const std::ptrdiff_t k = offsets_[d];
        for (std::size_t j = 0; j < std::min (cols_, L); ++j)
          {
            const std::ptrdiff_t i = static_cast<std::ptrdiff_t> (j) - k;
            if (i >= 0 && static_cast<std::size_t> (i) < rows_)
              {
                y[static_cast<std::size_t> (i)] +=
                  data_ (d, j) * x.eval_at (j);
              }
          }
      }
    return y;
  }

  nd_tensor<T>
  to_dense () const
  {
    nd_tensor<T> out (shape_type {rows_, cols_}, T {0});
    const std::size_t L = data_.shape ()[1];
    for (std::size_t d = 0; d < offsets_.size (); ++d)
      {
        const std::ptrdiff_t k = offsets_[d];
        for (std::size_t j = 0; j < std::min (cols_, L); ++j)
          {
            const std::ptrdiff_t i = static_cast<std::ptrdiff_t> (j) - k;
            if (i >= 0 && static_cast<std::size_t> (i) < rows_)
              {
                out (static_cast<std::size_t> (i), j) += data_ (d, j);
              }
          }
      }
    return out;
  }

  csr_matrix<T>
  to_csr () const
  {
    return csr_matrix<T>::from_dense (to_dense ());
  }
};

inline nd_tensor<double>
spsolve (const csr_matrix<double> &a, const nd_tensor<double> &b)
{
  return linalg::solve (a.to_dense (), b);
}

struct splu_factor
{
  linalg::lu_result lu;

  nd_tensor<double>
  solve (const nd_tensor<double> &b) const
  {
    return linalg::lu_solve (lu, b);
  }
};

inline splu_factor
splu (const csr_matrix<double> &a)
{
  return splu_factor {linalg::lu_factor (a.to_dense ())};
}

inline csr_matrix<double>
sparse_kron (const csr_matrix<double> &a, const csr_matrix<double> &b)
{
  return csr_matrix<double>::from_dense (
    tensor::kron (a.to_dense (), b.to_dense ()));
}

inline nd_tensor<double>
bellman_ford (const csr_matrix<double> &adj, std::size_t src)
{
  const std::size_t n = adj.rows ();
  if (adj.cols () != n || src >= n)
    {
      throw shape_error ("bellman_ford requires square matrix and valid src");
    }
  nd_tensor<double> dist (shape_type {n}, constants::inf);
  dist[src] = 0.0;
  const auto &indptr = adj.indptr ();
  const auto &indices = adj.indices ();
  const auto &data = adj.data ();

  for (std::size_t iter = 0; iter + 1 < n; ++iter)
    {
      bool updated = false;
      for (std::size_t u = 0; u < n; ++u)
        {
          if (!std::isfinite (dist[u]))
            {
              continue;
            }
          for (std::size_t k = indptr[u]; k < indptr[u + 1]; ++k)
            {
              const std::size_t v = indices[k];
              const double w = data[k];
              if (dist[u] + w < dist[v])
                {
                  dist[v] = dist[u] + w;
                  updated = true;
                }
            }
        }
      if (!updated)
        {
          break;
        }
    }

  for (std::size_t u = 0; u < n; ++u)
    {
      if (!std::isfinite (dist[u]))
        {
          continue;
        }
      for (std::size_t k = indptr[u]; k < indptr[u + 1]; ++k)
        {
          if (dist[u] + data[k] < dist[indices[k]] - 1e-12)
            {
              throw convergence_error ("bellman_ford detected a negative weight cycle");
            }
        }
    }
  return dist;
}

inline std::vector<std::size_t>
breadth_first_order (const csr_matrix<double> &adj, std::size_t i_start)
{
  const std::size_t n = adj.rows ();
  if (adj.cols () != n || i_start >= n)
    {
      throw shape_error ("breadth_first_order invalid matrix or start vertex");
    }
  const auto &indptr = adj.indptr ();
  const auto &indices = adj.indices ();
  std::vector<bool> vis (n, false);
  std::vector<std::size_t> order;
  vis[i_start] = true;
  order.push_back (i_start);
  std::size_t head = 0;
  while (head < order.size ())
    {
      const std::size_t u = order[head++];
      for (std::size_t k = indptr[u]; k < indptr[u + 1]; ++k)
        {
          const std::size_t v = indices[k];
          if (!vis[v])
            {
              vis[v] = true;
              order.push_back (v);
            }
        }
    }
  return order;
}

inline std::vector<std::size_t>
depth_first_order (const csr_matrix<double> &adj, std::size_t i_start)
{
  const std::size_t n = adj.rows ();
  if (adj.cols () != n || i_start >= n)
    {
      throw shape_error ("depth_first_order invalid matrix or start vertex");
    }
  const auto &indptr = adj.indptr ();
  const auto &indices = adj.indices ();
  std::vector<bool> vis (n, false);
  std::vector<std::size_t> order;
  auto dfs = [&] (auto &self, std::size_t u) -> void {
    vis[u] = true;
    order.push_back (u);
    for (std::size_t k = indptr[u]; k < indptr[u + 1]; ++k)
      {
        const std::size_t v = indices[k];
        if (!vis[v])
          {
            self (self, v);
          }
      }
  };
  dfs (dfs, i_start);
  return order;
}

inline nd_tensor<double>
graph_laplacian (const csr_matrix<double> &adj, bool normed = false)
{
  const std::size_t n = adj.rows ();
  if (adj.cols () != n)
    {
      throw shape_error ("graph_laplacian requires a square adjacency matrix");
    }
  const nd_tensor<double> a = adj.to_dense ();
  nd_tensor<double> lap (shape_type {n, n}, 0.0);
  std::vector<double> deg (n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          if (i != j)
            {
              deg[i] += a (i, j);
              lap (i, j) = -a (i, j);
            }
        }
      lap (i, i) = deg[i];
    }
  if (normed)
    {
      for (std::size_t i = 0; i < n; ++i)
        {
          for (std::size_t j = 0; j < n; ++j)
            {
              const double d = std::sqrt (deg[i] * deg[j]);
              lap (i, j) = (d > 0.0) ? (lap (i, j) / d) : 0.0;
            }
        }
    }
  return lap;
}

inline double
maximum_flow (const csr_matrix<double> &capacity,
              std::size_t source,
              std::size_t sink)
{
  const std::size_t n = capacity.rows ();
  if (capacity.cols () != n || source >= n || sink >= n || source == sink)
    {
      throw shape_error ("maximum_flow invalid graph or source/sink vertices");
    }
  nd_tensor<double> res_cap = capacity.to_dense ();
  double total_flow = 0.0;

  for (;;)
    {
      std::vector<std::ptrdiff_t> parent (n, -1);
      parent[source] = static_cast<std::ptrdiff_t> (source);
      std::vector<std::size_t> q = {source};
      std::size_t head = 0;
      while (head < q.size () && parent[sink] == -1)
        {
          const std::size_t u = q[head++];
          for (std::size_t v = 0; v < n; ++v)
            {
              if (parent[v] == -1 && res_cap (u, v) > 1e-12)
                {
                  parent[v] = static_cast<std::ptrdiff_t> (u);
                  q.push_back (v);
                }
            }
        }
      if (parent[sink] == -1)
        {
          break;
        }
      double path_flow = constants::inf;
      for (std::size_t v = sink; v != source;)
        {
          const std::size_t u = static_cast<std::size_t> (parent[v]);
          path_flow = std::min (path_flow, res_cap (u, v));
          v = u;
        }
      for (std::size_t v = sink; v != source;)
        {
          const std::size_t u = static_cast<std::size_t> (parent[v]);
          res_cap (u, v) -= path_flow;
          res_cap (v, u) += path_flow;
          v = u;
        }
      total_flow += path_flow;
    }
  return total_flow;
}

inline std::vector<std::size_t>
reverse_cuthill_mckee (const csr_matrix<double> &adj)
{
  const std::size_t n = adj.rows ();
  if (adj.cols () != n)
    {
      throw shape_error ("reverse_cuthill_mckee requires square matrix");
    }
  const auto &indptr = adj.indptr ();
  const auto &indices = adj.indices ();
  auto degree = [&] (std::size_t u) { return indptr[u + 1] - indptr[u]; };

  std::vector<bool> vis (n, false);
  std::vector<std::size_t> rcm;
  rcm.reserve (n);

  while (rcm.size () < n)
    {
      std::size_t start = 0;
      std::size_t min_deg = n + 1;
      for (std::size_t i = 0; i < n; ++i)
        {
          if (!vis[i] && degree (i) < min_deg)
            {
              min_deg = degree (i);
              start = i;
            }
        }
      vis[start] = true;
      std::size_t comp_head = rcm.size ();
      rcm.push_back (start);
      while (comp_head < rcm.size ())
        {
          const std::size_t u = rcm[comp_head++];
          std::vector<std::size_t> nbrs;
          for (std::size_t k = indptr[u]; k < indptr[u + 1]; ++k)
            {
              const std::size_t v = indices[k];
              if (!vis[v])
                {
                  vis[v] = true;
                  nbrs.push_back (v);
                }
            }
          std::sort (nbrs.begin (), nbrs.end (),
                     [&] (std::size_t a, std::size_t b) {
                       return degree (a) < degree (b);
                     });
          for (std::size_t v : nbrs)
            {
              rcm.push_back (v);
            }
        }
    }
  std::reverse (rcm.begin (), rcm.end ());
  return rcm;
}

/* =========================================================================
   Additional Sparse Formats (`dok_matrix`, `lil_matrix`, `bsr_matrix`),
   Sparse Linear Algebra (`svds`, `eigs`, `lsqr`, `minres`, `spilu`,
   `onenormest`, `expm_multiply`), and Extended Sparse Graph Algorithms
   (`shortest_path`, `johnson`, `maximum_bipartite_matching`,
   `min_weight_full_bipartite_matching`, `structural_rank`).
   ========================================================================= */

template <typename T = double>
class dok_matrix
{
private:
  std::size_t rows_;
  std::size_t cols_;
  std::map<std::pair<std::size_t, std::size_t>, T> entries_;

public:
  dok_matrix (std::size_t rows, std::size_t cols)
    : rows_ (rows), cols_ (cols), entries_ ()
  {
  }

  void
  set (std::size_t r, std::size_t c, T val)
  {
    if (r >= rows_ || c >= cols_)
      {
        throw index_error ("dok_matrix::set index out of bounds");
      }
    if (val == T {0})
      {
        entries_.erase ({r, c});
      }
    else
      {
        entries_[{r, c}] = val;
      }
  }

  T
  get (std::size_t r, std::size_t c) const
  {
    auto it = entries_.find ({r, c});
    return (it != entries_.end ()) ? it->second : T {0};
  }

  std::size_t nnz () const noexcept { return entries_.size (); }

  csr_matrix<T>
  to_csr () const
  {
    std::vector<std::size_t> r_idx;
    std::vector<std::size_t> c_idx;
    std::vector<T> vals;
    r_idx.reserve (entries_.size ());
    c_idx.reserve (entries_.size ());
    vals.reserve (entries_.size ());
    for (const auto &[rc, val] : entries_)
      {
        r_idx.push_back (rc.first);
        c_idx.push_back (rc.second);
        vals.push_back (val);
      }
    return coo_matrix<T> (rows_, cols_,
                          std::move (r_idx),
                          std::move (c_idx),
                          std::move (vals)).to_csr ();
  }
};

template <typename T = double>
class lil_matrix
{
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<std::map<std::size_t, T>> rows_data_;

public:
  lil_matrix (std::size_t rows, std::size_t cols)
    : rows_ (rows), cols_ (cols), rows_data_ (rows)
  {
  }

  void
  set (std::size_t r, std::size_t c, T val)
  {
    if (r >= rows_ || c >= cols_)
      {
        throw index_error ("lil_matrix::set index out of bounds");
      }
    if (val == T {0})
      {
        rows_data_[r].erase (c);
      }
    else
      {
        rows_data_[r][c] = val;
      }
  }

  T
  get (std::size_t r, std::size_t c) const
  {
    if (r >= rows_ || c >= cols_)
      {
        throw index_error ("lil_matrix::get index out of bounds");
      }
    auto it = rows_data_[r].find (c);
    return (it != rows_data_[r].end ()) ? it->second : T {0};
  }

  csr_matrix<T>
  to_csr () const
  {
    std::vector<std::size_t> r_idx;
    std::vector<std::size_t> c_idx;
    std::vector<T> vals;
    for (std::size_t r = 0; r < rows_; ++r)
      {
        for (const auto &[c, val] : rows_data_[r])
          {
            r_idx.push_back (r);
            c_idx.push_back (c);
            vals.push_back (val);
          }
      }
    return coo_matrix<T> (rows_, cols_,
                          std::move (r_idx),
                          std::move (c_idx),
                          std::move (vals)).to_csr ();
  }
};

template <typename T = double>
class bsr_matrix
{
private:
  csr_matrix<T> csr_;
  std::size_t block_r_;
  std::size_t block_c_;

public:
  bsr_matrix (const nd_tensor<T> &dense,
              std::size_t block_r = 2, std::size_t block_c = 2)
    : csr_ (csr_matrix<T>::from_dense (dense)),
      block_r_ (block_r),
      block_c_ (block_c)
  {
  }

  std::size_t block_rows () const noexcept { return block_r_; }
  std::size_t block_cols () const noexcept { return block_c_; }

  const csr_matrix<T> &to_csr () const noexcept { return csr_; }
  nd_tensor<T> to_dense () const { return csr_.to_dense (); }
};

struct svds_result
{
  nd_tensor<double> u;
  nd_tensor<double> s;
  nd_tensor<double> vt;
};

inline svds_result
svds (const csr_matrix<double> &a, std::size_t k = 2)
{
  const auto full = linalg::svd (a.to_dense ());
  const std::size_t r = std::min (k, full.s.size ());
  const std::size_t m = a.rows ();
  const std::size_t n = a.cols ();
  nd_tensor<double> u (shape_type {m, r});
  nd_tensor<double> s (shape_type {r});
  nd_tensor<double> vt (shape_type {r, n});
  for (std::size_t j = 0; j < r; ++j)
    {
      s[j] = full.s[j];
      for (std::size_t i = 0; i < m; ++i)
        {
          u (i, j) = full.u (i, j);
        }
      for (std::size_t c = 0; c < n; ++c)
        {
          vt (j, c) = full.vt (j, c);
        }
    }
  return {std::move (u), std::move (s), std::move (vt)};
}

inline linalg::eig_result
eigs (const csr_matrix<double> &a, std::size_t k = 2)
{
  const auto full = linalg::eig (a.to_dense ());
  const std::size_t n = a.rows ();
  const std::size_t r = std::min (k, n);
  std::vector<std::size_t> order (n);
  std::iota (order.begin (), order.end (), 0);
  std::sort (order.begin (), order.end (), [&] (std::size_t i, std::size_t j) {
    return std::abs (full.eigenvalues[i]) > std::abs (full.eigenvalues[j]);
  });
  nd_tensor<std::complex<double>> evals (shape_type {r});
  nd_tensor<std::complex<double>> evecs (shape_type {n, r});
  for (std::size_t j = 0; j < r; ++j)
    {
      const std::size_t idx = order[j];
      evals[j] = full.eigenvalues[idx];
      for (std::size_t i = 0; i < n; ++i)
        {
          evecs (i, j) = full.eigenvectors (i, idx);
        }
    }
  return {std::move (evals), std::move (evecs)};
}

inline nd_tensor<double>
lsqr (const csr_matrix<double> &a, const nd_tensor<double> &b,
      double damp = 0.0)
{
  const nd_tensor<double> ad = a.to_dense ();
  if (damp <= 0.0)
    {
      return linalg::lstsq (ad, b);
    }
  const std::size_t n = a.cols ();
  nd_tensor<double> ata = tensor::matmul (ad.t (), ad);
  for (std::size_t i = 0; i < n; ++i)
    {
      ata (i, i) += damp * damp;
    }
  const nd_tensor<double> atb = tensor::matmul (ad.t (), b);
  return linalg::solve (ata, atb);
}

inline nd_tensor<double>
minres (const csr_matrix<double> &a, const nd_tensor<double> &b,
        double tol = 1e-10, std::size_t max_iter = 200)
{
  return gmres_solve (a, b, tol, max_iter);
}

inline splu_factor
spilu (const csr_matrix<double> &a)
{
  return splu (a);
}

inline double
onenormest (const csr_matrix<double> &a)
{
  const std::size_t m = a.rows ();
  const std::size_t n = a.cols ();
  const nd_tensor<double> d = a.to_dense ();
  double best = 0.0;
  for (std::size_t j = 0; j < n; ++j)
    {
      double col_sum = 0.0;
      for (std::size_t i = 0; i < m; ++i)
        {
          col_sum += std::abs (d (i, j));
        }
      best = std::max (best, col_sum);
    }
  return best;
}

inline nd_tensor<double>
expm_multiply (const csr_matrix<double> &a, const nd_tensor<double> &b)
{
  return tensor::matmul (linalg::expm (a.to_dense ()), b);
}

inline nd_tensor<double>
shortest_path (const csr_matrix<double> &graph)
{
  return floyd_warshall (graph);
}

inline nd_tensor<double>
johnson (const csr_matrix<double> &graph)
{
  return floyd_warshall (graph);
}

inline nd_tensor<std::ptrdiff_t>
maximum_bipartite_matching (const csr_matrix<double> &graph)
{
  const std::size_t m = graph.rows ();
  const std::size_t n = graph.cols ();
  const auto &indptr = graph.indptr ();
  const auto &indices = graph.indices ();
  const auto &data = graph.data ();
  std::vector<std::ptrdiff_t> match_r (n, -1);
  std::vector<std::ptrdiff_t> match_l (m, -1);

  auto dfs = [&] (auto &self, std::size_t u,
                  std::vector<bool> &seen) -> bool {
    for (std::size_t k = indptr[u]; k < indptr[u + 1]; ++k)
      {
        if (std::abs (data[k]) <= 1e-15)
          {
            continue;
          }
        const std::size_t v = indices[k];
        if (!seen[v])
          {
            seen[v] = true;
            if (match_r[v] < 0
                || self (self, static_cast<std::size_t> (match_r[v]), seen))
              {
                match_r[v] = static_cast<std::ptrdiff_t> (u);
                match_l[u] = static_cast<std::ptrdiff_t> (v);
                return true;
              }
          }
      }
    return false;
  };

  for (std::size_t u = 0; u < m; ++u)
    {
      std::vector<bool> seen (n, false);
      dfs (dfs, u, seen);
    }
  return nd_tensor<std::ptrdiff_t> (shape_type {m}, std::move (match_l));
}

inline std::pair<nd_tensor<std::size_t>, nd_tensor<std::size_t>>
min_weight_full_bipartite_matching (const csr_matrix<double> &biadj)
{
  const std::size_t r = biadj.rows ();
  const std::size_t c = biadj.cols ();
  nd_tensor<double> cost (shape_type {r, c}, 1e12);
  const auto &indptr = biadj.indptr ();
  const auto &indices = biadj.indices ();
  const auto &data = biadj.data ();
  for (std::size_t i = 0; i < r; ++i)
    {
      for (std::size_t k = indptr[i]; k < indptr[i + 1]; ++k)
        {
          if (std::abs (data[k]) > 1e-15)
            {
              cost (i, indices[k]) = data[k];
            }
        }
    }
  const auto res = optimize::linear_sum_assignment (cost);
  return {res.row_ind, res.col_ind};
}

inline std::size_t
structural_rank (const csr_matrix<double> &graph)
{
  const auto m = maximum_bipartite_matching (graph);
  std::size_t cnt = 0;
  for (std::size_t i = 0; i < m.size (); ++i)
    {
      if (m.eval_at (i) >= 0)
        {
          ++cnt;
        }
    }
  return cnt;
}

} // namespace hyper_math::scientific::sparse

#endif /* HYPER_MATH_SCIENTIFIC_SPARSE_HPP */
