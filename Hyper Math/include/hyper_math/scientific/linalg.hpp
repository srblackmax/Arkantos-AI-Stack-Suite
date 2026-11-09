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

#ifndef HYPER_MATH_SCIENTIFIC_LINALG_HPP
#define HYPER_MATH_SCIENTIFIC_LINALG_HPP

#include "hyper_math/tensor/reductions.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <tuple>
#include <vector>

namespace hyper_math::scientific::linalg
{

using tensor::nd_tensor;
using tensor::shape_type;

/* Result of Doolittle LU factorization with partial row pivoting: P * A = L * U. */
struct lu_result
{
  nd_tensor<double> lu;
  std::vector<std::size_t> piv;
  int pivot_sign;
};

/* Doolittle LU decomposition with partial pivoting.  */
inline lu_result
lu_factor (const nd_tensor<double> &a)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("lu_factor requires a square 2D matrix");
    }

  const std::size_t n = a.shape ()[0];
  nd_tensor<double> lu = a.copy ();
  std::vector<std::size_t> piv (n);
  std::iota (piv.begin (), piv.end (), 0);
  int sign = 1;

  for (std::size_t k = 0; k < n; ++k)
    {
      std::size_t max_row = k;
      double max_val = std::abs (lu (k, k));
      for (std::size_t i = k + 1; i < n; ++i)
        {
          const double v = std::abs (lu (i, k));
          if (v > max_val)
            {
              max_val = v;
              max_row = i;
            }
        }

      if (max_val < 1e-14)
        {
          throw singular_matrix_error ("singular matrix in lu_factor");
        }

      if (max_row != k)
        {
          for (std::size_t j = 0; j < n; ++j)
            {
              std::swap (lu (k, j), lu (max_row, j));
            }
          std::swap (piv[k], piv[max_row]);
          sign = -sign;
        }

      const double pivot = lu (k, k);
      for (std::size_t i = k + 1; i < n; ++i)
        {
          lu (i, k) /= pivot;
          const double mult = lu (i, k);
          for (std::size_t j = k + 1; j < n; ++j)
            {
              lu (i, j) -= mult * lu (k, j);
            }
        }
    }

  return lu_result {std::move (lu), std::move (piv), sign};
}

/* Solve A * x = b using precomputed lu_result (supports 1D or 2D b).  */
inline nd_tensor<double>
lu_solve (const lu_result &f, const nd_tensor<double> &b)
{
  const std::size_t n = f.lu.shape ()[0];
  if (b.ndim () == 1)
    {
      if (b.size () != n)
        {
          throw shape_error ("lu_solve RHS vector length mismatch");
        }
      nd_tensor<double> x (shape_type {n});
      for (std::size_t i = 0; i < n; ++i)
        {
          x[i] = b[f.piv[i]];
        }
      for (std::size_t i = 0; i < n; ++i)
        {
          for (std::size_t j = 0; j < i; ++j)
            {
              x[i] -= f.lu (i, j) * x[j];
            }
        }
      for (std::ptrdiff_t i = static_cast<std::ptrdiff_t> (n) - 1; i >= 0; --i)
        {
          const std::size_t ui = static_cast<std::size_t> (i);
          for (std::size_t j = ui + 1; j < n; ++j)
            {
              x[ui] -= f.lu (ui, j) * x[j];
            }
          x[ui] /= f.lu (ui, ui);
        }
      return x;
    }
  else if (b.ndim () == 2)
    {
      if (b.shape ()[0] != n)
        {
          throw shape_error ("lu_solve RHS matrix row mismatch");
        }
      const std::size_t nrhs = b.shape ()[1];
      nd_tensor<double> x (shape_type {n, nrhs});
      for (std::size_t i = 0; i < n; ++i)
        {
          for (std::size_t col = 0; col < nrhs; ++col)
            {
              x (i, col) = b (f.piv[i], col);
            }
        }
      for (std::size_t i = 0; i < n; ++i)
        {
          for (std::size_t j = 0; j < i; ++j)
            {
              const double lij = f.lu (i, j);
              for (std::size_t col = 0; col < nrhs; ++col)
                {
                  x (i, col) -= lij * x (j, col);
                }
            }
        }
      for (std::ptrdiff_t i = static_cast<std::ptrdiff_t> (n) - 1; i >= 0; --i)
        {
          const std::size_t ui = static_cast<std::size_t> (i);
          for (std::size_t j = ui + 1; j < n; ++j)
            {
              const double uij = f.lu (ui, j);
              for (std::size_t col = 0; col < nrhs; ++col)
                {
                  x (ui, col) -= uij * x (j, col);
                }
            }
          const double diag_inv = 1.0 / f.lu (ui, ui);
          for (std::size_t col = 0; col < nrhs; ++col)
            {
              x (ui, col) *= diag_inv;
            }
        }
      return x;
    }

  throw shape_error ("lu_solve requires 1D or 2D right-hand side");
}

inline nd_tensor<double>
solve (const nd_tensor<double> &a, const nd_tensor<double> &b)
{
  return lu_solve (lu_factor (a), b);
}

inline double
det (const nd_tensor<double> &a)
{
  try
    {
      const lu_result f = lu_factor (a);
      double d = static_cast<double> (f.pivot_sign);
      for (std::size_t i = 0; i < f.lu.shape ()[0]; ++i)
        {
          d *= f.lu (i, i);
        }
      return d;
    }
  catch (const singular_matrix_error &)
    {
      return 0.0;
    }
}

inline nd_tensor<double>
inv (const nd_tensor<double> &a)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("inv requires a square 2D matrix");
    }
  const std::size_t n = a.shape ()[0];
  return solve (a, tensor::eye<double> (n));
}

/* Cholesky Banachiewicz decomposition: A = L * L^T for SPD matrix A.  */
inline nd_tensor<double>
cholesky (const nd_tensor<double> &a)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("cholesky requires a square 2D matrix");
    }
  const std::size_t n = a.shape ()[0];
  nd_tensor<double> l (shape_type {n, n}, 0.0);

  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j <= i; ++j)
        {
          double sum_val = 0.0;
          for (std::size_t k = 0; k < j; ++k)
            {
              sum_val += l (i, k) * l (j, k);
            }
          if (i == j)
            {
              const double diag_val = a (i, i) - sum_val;
              if (diag_val <= 0.0)
                {
                  throw singular_matrix_error (
                    "matrix is not positive definite in cholesky");
                }
              l (i, j) = std::sqrt (diag_val);
            }
          else
            {
              l (i, j) = (a (i, j) - sum_val) / l (j, j);
            }
        }
    }
  return l;
}

/* Householder Reflection QR Decomposition: A = Q * R.  */
struct qr_result
{
  nd_tensor<double> q;
  nd_tensor<double> r;
};

inline qr_result
qr (const nd_tensor<double> &a)
{
  if (a.ndim () != 2)
    {
      throw shape_error ("qr requires a 2D matrix");
    }
  const std::size_t m = a.shape ()[0];
  const std::size_t n = a.shape ()[1];
  const std::size_t k_max = std::min (m, n);

  nd_tensor<double> r = a.copy ();
  nd_tensor<double> q = tensor::eye<double> (m);

  for (std::size_t k = 0; k < k_max; ++k)
    {
      double norm_x = 0.0;
      for (std::size_t i = k; i < m; ++i)
        {
          norm_x += r (i, k) * r (i, k);
        }
      norm_x = std::sqrt (norm_x);
      if (norm_x < 1e-15)
        {
          continue;
        }

      const double sign = (r (k, k) >= 0.0) ? 1.0 : -1.0;
      const double u1 = r (k, k) + sign * norm_x;
      std::vector<double> v (m - k, 0.0);
      v[0] = 1.0;
      double v_norm_sq = 1.0;
      for (std::size_t i = k + 1; i < m; ++i)
        {
          v[i - k] = r (i, k) / u1;
          v_norm_sq += v[i - k] * v[i - k];
        }
      const double beta = 2.0 / v_norm_sq;

      /* Update R := (I - beta * v * v^T) * R.  */
      for (std::size_t j = k; j < n; ++j)
        {
          double dot_vr = 0.0;
          for (std::size_t i = k; i < m; ++i)
            {
              dot_vr += v[i - k] * r (i, j);
            }
          for (std::size_t i = k; i < m; ++i)
            {
              r (i, j) -= beta * v[i - k] * dot_vr;
            }
        }

      /* Update Q := Q * (I - beta * v * v^T).  */
      for (std::size_t i = 0; i < m; ++i)
        {
          double dot_qv = 0.0;
          for (std::size_t j = k; j < m; ++j)
            {
              dot_qv += q (i, j) * v[j - k];
            }
          for (std::size_t j = k; j < m; ++j)
            {
              q (i, j) -= beta * dot_qv * v[j - k];
            }
        }
    }

  return qr_result {std::move (q), std::move (r)};
}

/* Linear least squares min_x ||A * x - b||_2 via Householder QR.
   Supports both 1D vector `b` and 2D multi-column RHS `b`.  */
inline nd_tensor<double>
lstsq (const nd_tensor<double> &a, const nd_tensor<double> &b)
{
  if (a.ndim () != 2 || (b.ndim () != 1 && b.ndim () != 2))
    {
      throw shape_error ("lstsq requires 2D matrix A and 1D or 2D RHS b");
    }
  const std::size_t m = a.shape ()[0];
  const std::size_t n = a.shape ()[1];
  if (b.shape ()[0] != m || m < n)
    {
      throw shape_error ("lstsq requires m >= n and b.shape()[0] == m");
    }

  const qr_result decomp = qr (a);
  const nd_tensor<double> qt_b = tensor::matmul (decomp.q.t (), b);

  if (b.ndim () == 1)
    {
      nd_tensor<double> x (shape_type {n}, 0.0);
      for (std::ptrdiff_t i = static_cast<std::ptrdiff_t> (n) - 1; i >= 0; --i)
        {
          const std::size_t ui = static_cast<std::size_t> (i);
          double acc = qt_b[ui];
          for (std::size_t j = ui + 1; j < n; ++j)
            {
              acc -= decomp.r (ui, j) * x[j];
            }
          if (std::abs (decomp.r (ui, ui)) < 1e-14)
            {
              throw singular_matrix_error ("rank-deficient matrix in lstsq");
            }
          x[ui] = acc / decomp.r (ui, ui);
        }
      return x;
    }

  const std::size_t nrhs = b.shape ()[1];
  nd_tensor<double> x (shape_type {n, nrhs}, 0.0);
  for (std::size_t col = 0; col < nrhs; ++col)
    {
      for (std::ptrdiff_t i = static_cast<std::ptrdiff_t> (n) - 1; i >= 0; --i)
        {
          const std::size_t ui = static_cast<std::size_t> (i);
          double acc = qt_b (ui, col);
          for (std::size_t j = ui + 1; j < n; ++j)
            {
              acc -= decomp.r (ui, j) * x (j, col);
            }
          if (std::abs (decomp.r (ui, ui)) < 1e-14)
            {
              throw singular_matrix_error ("rank-deficient matrix in lstsq");
            }
          x (ui, col) = acc / decomp.r (ui, ui);
        }
    }
  return x;
}

/* Symmetric Jacobi Eigenvalue and Eigenvector Solver (SciPy eigh equivalent).
   Returns eigenvalues in ascending order and corresponding column
   eigenvectors.  */
struct eigh_result
{
  nd_tensor<double> eigenvalues;
  nd_tensor<double> eigenvectors;
};

inline eigh_result
eigh (const nd_tensor<double> &a,
      std::size_t max_sweeps = 100,
      double tol = 1e-12)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("eigh requires a square symmetric 2D matrix");
    }

  const std::size_t n = a.shape ()[0];
  nd_tensor<double> d = a.copy ();
  nd_tensor<double> v = tensor::eye<double> (n);

  for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep)
    {
      double off_diag_norm = 0.0;
      for (std::size_t p = 0; p < n; ++p)
        {
          for (std::size_t q = p + 1; q < n; ++q)
            {
              off_diag_norm += d (p, q) * d (p, q);
            }
        }
      if (std::sqrt (off_diag_norm) < tol)
        {
          break;
        }

      for (std::size_t p = 0; p < n; ++p)
        {
          for (std::size_t q = p + 1; q < n; ++q)
            {
              const double apq = d (p, q);
              if (std::abs (apq) < 1e-15)
                {
                  continue;
                }
              const double app = d (p, p);
              const double aqq = d (q, q);
              const double tau = (aqq - app) / (2.0 * apq);
              const double t =
                (tau >= 0.0)
                ? (1.0 / (tau + std::sqrt (1.0 + tau * tau)))
                : (-1.0 / (-tau + std::sqrt (1.0 + tau * tau)));
              const double c = 1.0 / std::sqrt (1.0 + t * t);
              const double s = t * c;

              d (p, p) = app - t * apq;
              d (q, q) = aqq + t * apq;
              d (p, q) = 0.0;
              d (q, p) = 0.0;

              for (std::size_t r = 0; r < n; ++r)
                {
                  if (r != p && r != q)
                    {
                      const double drp = d (r, p);
                      const double drq = d (r, q);
                      d (r, p) = c * drp - s * drq;
                      d (p, r) = d (r, p);
                      d (r, q) = s * drp + c * drq;
                      d (q, r) = d (r, q);
                    }
                }

              for (std::size_t r = 0; r < n; ++r)
                {
                  const double vrp = v (r, p);
                  const double vrq = v (r, q);
                  v (r, p) = c * vrp - s * vrq;
                  v (r, q) = s * vrp + c * vrq;
                }
            }
        }
    }

  /* Sort eigenvalues ascending and permute eigenvector columns.  */
  std::vector<std::size_t> order (n);
  std::iota (order.begin (), order.end (), 0);
  std::sort (order.begin (), order.end (), [&d] (std::size_t i, std::size_t j) {
    return d (i, i) < d (j, j);
  });

  nd_tensor<double> evals (shape_type {n});
  nd_tensor<double> evecs (shape_type {n, n});
  for (std::size_t col = 0; col < n; ++col)
    {
      const std::size_t src = order[col];
      evals[col] = d (src, src);
      for (std::size_t row = 0; row < n; ++row)
        {
          evecs (row, col) = v (row, src);
        }
    }

  return eigh_result {std::move (evals), std::move (evecs)};
}

/* Singular Value Decomposition: A = U * diag(S) * V^T via symmetric
   eigen-decomposition of A^T * A (or A * A^T).  */
struct svd_result
{
  nd_tensor<double> u;
  nd_tensor<double> s;
  nd_tensor<double> vt;
};

inline svd_result
svd (const nd_tensor<double> &a)
{
  if (a.ndim () != 2)
    {
      throw shape_error ("svd requires a 2D matrix");
    }
  const std::size_t m = a.shape ()[0];
  const std::size_t n = a.shape ()[1];
  const std::size_t k = std::min (m, n);

  const nd_tensor<double> ata = tensor::matmul (a.t (), a);
  const eigh_result eig = eigh (ata);

  nd_tensor<double> s (shape_type {k});
  nd_tensor<double> v (shape_type {n, n});
  for (std::size_t j = 0; j < n; ++j)
    {
      const std::size_t rev_idx = n - 1 - j;
      const double ev = std::max (0.0, eig.eigenvalues[rev_idx]);
      if (j < k)
        {
          s[j] = std::sqrt (ev);
        }
      for (std::size_t i = 0; i < n; ++i)
        {
          v (i, j) = eig.eigenvectors (i, rev_idx);
        }
    }

  nd_tensor<double> u = tensor::eye<double> (m, k);
  for (std::size_t j = 0; j < k; ++j)
    {
      if (s[j] > 1e-12)
        {
          for (std::size_t i = 0; i < m; ++i)
            {
              double sum_av = 0.0;
              for (std::size_t p = 0; p < n; ++p)
                {
                  sum_av += a (i, p) * v (p, j);
                }
              u (i, j) = sum_av / s[j];
            }
        }
      else
        {
          /* Rank-deficient column: find a standard basis vector e_cand not
             in the span of u[:, 0..j-1] via modified Gram-Schmidt.  */
          for (std::size_t cand = 0; cand < m; ++cand)
            {
              std::vector<double> col_vec (m, 0.0);
              col_vec[cand] = 1.0;
              for (std::size_t prev = 0; prev < j; ++prev)
                {
                  double dot_val = 0.0;
                  for (std::size_t i = 0; i < m; ++i)
                    {
                      dot_val += col_vec[i] * u (i, prev);
                    }
                  for (std::size_t i = 0; i < m; ++i)
                    {
                      col_vec[i] -= dot_val * u (i, prev);
                    }
                }
              double norm_sq = 0.0;
              for (std::size_t i = 0; i < m; ++i)
                {
                  norm_sq += col_vec[i] * col_vec[i];
                }
              if (norm_sq > 1e-12)
                {
                  const double inv_n = 1.0 / std::sqrt (norm_sq);
                  for (std::size_t i = 0; i < m; ++i)
                    {
                      u (i, j) = col_vec[i] * inv_n;
                    }
                  break;
                }
            }
        }
    }

  nd_tensor<double> vt (shape_type {k, n});
  for (std::size_t i = 0; i < k; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          vt (i, j) = v (j, i);
        }
    }

  return svd_result {std::move (u), std::move (s), std::move (vt)};
}

/* Moore-Penrose pseudoinverse via SVD: A^+ = V * S^+ * U^T.  */
inline nd_tensor<double>
pinv (const nd_tensor<double> &a, double rcond = 1e-12)
{
  const svd_result decomp = svd (a);
  const std::size_t k = decomp.s.size ();
  const double cutoff = (k > 0) ? (rcond * decomp.s[0]) : 0.0;

  const std::size_t m = a.shape ()[0];
  const std::size_t n = a.shape ()[1];
  nd_tensor<double> result (shape_type {n, m}, 0.0);

  for (std::size_t r = 0; r < k; ++r)
    {
      if (decomp.s[r] > cutoff)
        {
          const double inv_s = 1.0 / decomp.s[r];
          for (std::size_t i = 0; i < n; ++i)
            {
              for (std::size_t j = 0; j < m; ++j)
                {
                  result (i, j) += decomp.vt (r, i) * inv_s * decomp.u (j, r);
                }
            }
        }
    }
  return result;
}

inline double
norm (const nd_tensor<double> &a, double ord = 2.0)
{
  if (ord == 2.0)
    {
      double sum_sq = 0.0;
      for (std::size_t i = 0; i < a.size (); ++i)
        {
          const double v = a.eval_at (i);
          sum_sq += v * v;
        }
      return std::sqrt (sum_sq);
    }
  else if (ord == 1.0)
    {
      double sum_abs = 0.0;
      for (std::size_t i = 0; i < a.size (); ++i)
        {
          sum_abs += std::abs (a.eval_at (i));
        }
      return sum_abs;
    }
  else if (std::isinf (ord))
    {
      if (a.size () == 0)
        {
          return 0.0;
        }
      if (ord > 0.0)
        {
          double best = 0.0;
          for (std::size_t i = 0; i < a.size (); ++i)
            {
              best = std::max (best, std::abs (a.eval_at (i)));
            }
          return best;
        }
      else
        {
          double best = std::abs (a.eval_at (0));
          for (std::size_t i = 1; i < a.size (); ++i)
            {
              best = std::min (best, std::abs (a.eval_at (i)));
            }
          return best;
        }
    }
  double sum_p = 0.0;
  for (std::size_t i = 0; i < a.size (); ++i)
    {
      sum_p += std::pow (std::abs (a.eval_at (i)), ord);
    }
  return std::pow (sum_p, 1.0 / ord);
}

inline std::size_t
matrix_rank (const nd_tensor<double> &a, double tol = 1e-10)
{
  const svd_result decomp = svd (a);
  std::size_t rank = 0;
  for (std::size_t i = 0; i < decomp.s.size (); ++i)
    {
      if (decomp.s[i] > tol)
        {
          ++rank;
        }
    }
  return rank;
}

/* Sign and natural logarithm of matrix determinant (NumPy/SciPy slogdet)
   to prevent floating-point overflow or underflow on high-dimensional
   matrices.  */
struct slogdet_result
{
  double sign;
  double logabsdet;
};

inline slogdet_result
slogdet (const nd_tensor<double> &a)
{
  try
    {
      const lu_result f = lu_factor (a);
      double sign = static_cast<double> (f.pivot_sign);
      double log_sum = 0.0;
      for (std::size_t i = 0; i < f.lu.shape ()[0]; ++i)
        {
          const double d = f.lu (i, i);
          if (d < 0.0)
            {
              sign = -sign;
            }
          log_sum += std::log (std::abs (d));
        }
      return slogdet_result {sign, log_sum};
    }
  catch (const singular_matrix_error &)
    {
      return slogdet_result {0.0, -constants::inf};
    }
}

/* 2-norm matrix condition number sigma_max / sigma_min.  */
inline double
cond (const nd_tensor<double> &a)
{
  const svd_result decomp = svd (a);
  if (decomp.s.empty ())
    {
      return constants::nan;
    }
  const double s_max = decomp.s[0];
  const double s_min = decomp.s[decomp.s.size () - 1];
  if (s_min <= 1e-15)
    {
      return constants::inf;
    }
  return s_max / s_min;
}

/* Matrix Exponential exp(A) via Scaling-and-Squaring with Padé (6,6)
   rational approximant (SciPy linalg.expm equivalent).  */
inline nd_tensor<double>
expm (const nd_tensor<double> &a)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("expm requires a square 2D matrix");
    }
  const std::size_t n = a.shape ()[0];
  const double a_norm = norm (a, constants::inf);

  int s = 0;
  if (a_norm > 0.5)
    {
      s = std::max (0, static_cast<int> (std::ceil (std::log2 (a_norm / 0.5))));
    }
  const double scale = std::ldexp (1.0, -s);
  const nd_tensor<double> as = a * scale;

  /* Exact Padé (6,6) coefficients c_k = (12 - k)! 6! / (12! k! (6 - k)!).  */
  constexpr double c0 = 1.0;
  constexpr double c1 = 1.0 / 2.0;
  constexpr double c2 = 5.0 / 44.0;
  constexpr double c3 = 1.0 / 66.0;
  constexpr double c4 = 1.0 / 792.0;
  constexpr double c5 = 1.0 / 15840.0;
  constexpr double c6 = 1.0 / 665280.0;

  const nd_tensor<double> id = tensor::eye<double> (n);
  const nd_tensor<double> a2 = tensor::matmul (as, as);
  const nd_tensor<double> a4 = tensor::matmul (a2, a2);
  const nd_tensor<double> a6 = tensor::matmul (a4, a2);

  /* Even/odd split for Padé(6,6):
     Even powers: c0*I + c2*A^2 + c4*A^4 + c6*A^6
     Odd powers:  A * (c1*I + c3*A^2 + c5*A^4).  */
  const nd_tensor<double> p_even =
    c0 * id + c2 * a2 + c4 * a4 + c6 * a6;
  const nd_tensor<double> p_odd =
    tensor::matmul (as, nd_tensor<double> (c1 * id + c3 * a2 + c5 * a4));

  const nd_tensor<double> num = p_even + p_odd;
  const nd_tensor<double> den = p_even - p_odd;
  nd_tensor<double> r = solve (den, num);

  for (int k = 0; k < s; ++k)
    {
      r = tensor::matmul (r, r);
    }
  return r;
}

/* Special structured matrices: Toeplitz, Circulant, Hilbert, Block-Diagonal
   (SciPy linalg equivalents).  */
inline nd_tensor<double>
toeplitz (const nd_tensor<double> &c, const nd_tensor<double> &r)
{
  const std::size_t m = c.size ();
  const std::size_t n = r.size ();
  nd_tensor<double> out (shape_type {m, n});
  for (std::size_t i = 0; i < m; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          out (i, j) = (i >= j) ? c.eval_at (i - j) : r.eval_at (j - i);
        }
    }
  return out;
}

inline nd_tensor<double>
toeplitz (const nd_tensor<double> &c)
{
  return toeplitz (c, c);
}

inline nd_tensor<double>
circulant (const nd_tensor<double> &c)
{
  const std::size_t n = c.size ();
  nd_tensor<double> out (shape_type {n, n});
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          out (i, j) = c.eval_at ((i + n - j) % n);
        }
    }
  return out;
}

inline nd_tensor<double>
hilbert (std::size_t n)
{
  nd_tensor<double> out (shape_type {n, n});
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          out (i, j) = 1.0 / static_cast<double> (i + j + 1);
        }
    }
  return out;
}

inline nd_tensor<double>
block_diag (const std::vector<nd_tensor<double>> &mats)
{
  std::size_t total_r = 0;
  std::size_t total_c = 0;
  for (const auto &m : mats)
    {
      if (m.ndim () != 2)
        {
          throw shape_error ("block_diag requires 2D matrices");
        }
      total_r += m.shape ()[0];
      total_c += m.shape ()[1];
    }

  nd_tensor<double> out (shape_type {total_r, total_c}, 0.0);
  std::size_t r_off = 0;
  std::size_t c_off = 0;
  for (const auto &m : mats)
    {
      for (std::size_t r = 0; r < m.shape ()[0]; ++r)
        {
          for (std::size_t c = 0; c < m.shape ()[1]; ++c)
            {
              out (r_off + r, c_off + c) = m (r, c);
            }
        }
      r_off += m.shape ()[0];
      c_off += m.shape ()[1];
    }
  return out;
}

/* Real Schur Decomposition (A = Z * T * Z^T) and General Nonsymmetric
   Eigenvalue Solver (eigvals, eig, roots).
   Reduces A to quasi-upper-triangular real Schur form T via Wilkinson-shifted
   QR steps with deflation, extracting 1x1 real eigenvalues and 2x2 complex
   conjugate eigenvalue pairs.  */
struct hessenberg_result
{
  nd_tensor<double> h;
  nd_tensor<double> q;
};

inline hessenberg_result
hessenberg (const nd_tensor<double> &a);

struct schur_result
{
  nd_tensor<double> t;
  nd_tensor<double> z;
};

inline schur_result
schur (const nd_tensor<double> &a, std::size_t max_iter = 400)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("schur requires a square 2D matrix");
    }
  const std::size_t n = a.shape ()[0];
  if (n <= 1)
    {
      return schur_result {a.copy (), tensor::eye<double> (n)};
    }

  hessenberg_result hr = hessenberg (a);
  nd_tensor<double> t = std::move (hr.h);
  nd_tensor<double> z = std::move (hr.q);

  const nd_tensor<double> id = tensor::eye<double> (n);
  for (std::size_t iter = 0; iter < max_iter; ++iter)
    {
      /* Check if all subdiagonal elements are negligible or decoupled 2x2.  */
      bool converged = true;
      std::size_t m_act = n - 1;
      for (std::size_t i = 0; i + 1 < n; ++i)
        {
          const double scale =
            std::abs (t (i, i)) + std::abs (t (i + 1, i + 1));
          if (std::abs (t (i + 1, i)) <= 1e-12 * std::max (1.0, scale))
            {
              t (i + 1, i) = 0.0;
            }
          else
            {
              /* Check if this is an isolated 2x2 complex conjugate block.  */
              const bool prev_zero =
                (i == 0) || (std::abs (t (i, i - 1)) == 0.0);
              const double next_scale =
                (i + 2 < n)
                ? (std::abs (t (i + 1, i + 1)) + std::abs (t (i + 2, i + 2)))
                : 0.0;
              const bool next_zero =
                (i + 2 >= n)
                || (std::abs (t (i + 2, i + 1))
                    <= 1e-12 * std::max (1.0, next_scale));
              const double tr = t (i, i) + t (i + 1, i + 1);
              const double det2 =
                t (i, i) * t (i + 1, i + 1) - t (i, i + 1) * t (i + 1, i);
              const double disc = tr * tr - 4.0 * det2;
              if (!(prev_zero && next_zero && disc < 0.0))
                {
                  converged = false;
                  m_act = i + 1;
                }
            }
        }
      if (converged)
        {
          break;
        }

      /* Compute Wilkinson shift from lowest active 2x2 block if real, else
         Rayleigh shift t(m_act, m_act).  */
      const double a11 = t (m_act - 1, m_act - 1);
      const double a12 = t (m_act - 1, m_act);
      const double a21 = t (m_act, m_act - 1);
      const double a22 = t (m_act, m_act);
      const double d = 0.5 * (a11 - a22);
      const double b2 = a12 * a21;
      double mu = a22;
      if (d * d + b2 >= 0.0)
        {
          const double sgn = (d >= 0.0) ? 1.0 : -1.0;
          const double denom = d + sgn * std::sqrt (d * d + b2);
          if (std::abs (denom) > 1e-15)
            {
              mu = a22 - b2 / denom;
            }
        }

      nd_tensor<double> shifted = t - mu * id;
      qr_result qrd = qr (shifted);
      t = tensor::matmul (qrd.r, qrd.q) + mu * id;
      z = tensor::matmul (z, qrd.q);
    }

  return schur_result {std::move (t), std::move (z)};
}

inline std::vector<std::complex<double>>
eigvals (const nd_tensor<double> &a)
{
  const schur_result sr = schur (a);
  const nd_tensor<double> &t = sr.t;
  const std::size_t n = t.shape ()[0];

  std::vector<std::complex<double>> vals;
  vals.reserve (n);
  std::size_t i = 0;
  while (i < n)
    {
      if (i + 1 < n
          && std::abs (t (i + 1, i))
             > 1e-12
               * std::max (1.0,
                           std::abs (t (i, i)) + std::abs (t (i + 1, i + 1))))
        {
          const double a11 = t (i, i);
          const double a12 = t (i, i + 1);
          const double a21 = t (i + 1, i);
          const double a22 = t (i + 1, i + 1);
          const double tr = a11 + a22;
          const double det2 = a11 * a22 - a12 * a21;
          const double disc = tr * tr - 4.0 * det2;
          if (disc < 0.0)
            {
              const double re = 0.5 * tr;
              const double im = 0.5 * std::sqrt (-disc);
              vals.emplace_back (re, im);
              vals.emplace_back (re, -im);
            }
          else
            {
              const double s = std::sqrt (disc);
              vals.emplace_back (0.5 * (tr + s), 0.0);
              vals.emplace_back (0.5 * (tr - s), 0.0);
            }
          i += 2;
        }
      else
        {
          vals.emplace_back (t (i, i), 0.0);
          ++i;
        }
    }
  return vals;
}

/* Roots of a polynomial p(x) = c[0]*x^n + c[1]*x^(n-1) + ... + c[n] via the
   eigenvalues of its Frobenius companion matrix (NumPy roots).  */
inline std::vector<std::complex<double>>
roots (const nd_tensor<double> &coeffs)
{
  std::size_t start = 0;
  while (start < coeffs.size () && std::abs (coeffs.eval_at (start)) == 0.0)
    {
      ++start;
    }
  if (start + 1 >= coeffs.size ())
    {
      return {};
    }
  const std::size_t deg = coeffs.size () - 1 - start;
  const double lead = coeffs.eval_at (start);
  nd_tensor<double> comp (shape_type {deg, deg}, 0.0);
  for (std::size_t j = 0; j < deg; ++j)
    {
      comp (0, j) = -coeffs.eval_at (start + 1 + j) / lead;
    }
  for (std::size_t i = 1; i < deg; ++i)
    {
      comp (i, i - 1) = 1.0;
    }
  return eigvals (comp);
}

/* Polar decomposition A = U * P via SVD, where U is orthogonal/unitary and
   P is symmetric positive semi-definite (SciPy linalg.polar).  */
struct polar_result
{
  nd_tensor<double> u;
  nd_tensor<double> p;
};

inline polar_result
polar (const nd_tensor<double> &a)
{
  const svd_result sv = svd (a);
  nd_tensor<double> u = tensor::matmul (sv.u, sv.vt);
  nd_tensor<double> p =
    tensor::matmul (tensor::matmul (sv.vt.t (), tensor::diag (sv.s)), sv.vt);
  return polar_result {std::move (u), std::move (p)};
}

/* Sylvester equation solver: A * X + X * B = C, where A is (m, m),
   B is (n, n), and C is (m, n) (SciPy linalg.solve_sylvester).
   Solved via the Kronecker linear system (A (x) I_n + I_m (x) B^T) vec(X) = vec(C). */
inline nd_tensor<double>
solve_sylvester (const nd_tensor<double> &a,
                 const nd_tensor<double> &b,
                 const nd_tensor<double> &c)
{
  if (a.ndim () != 2 || b.ndim () != 2 || c.ndim () != 2
      || a.shape ()[0] != a.shape ()[1] || b.shape ()[0] != b.shape ()[1]
      || c.shape ()[0] != a.shape ()[0] || c.shape ()[1] != b.shape ()[0])
    {
      throw shape_error ("incompatible matrix shapes in solve_sylvester");
    }
  const std::size_t m = a.shape ()[0];
  const std::size_t n = b.shape ()[0];
  const nd_tensor<double> im = tensor::eye<double> (m);
  const nd_tensor<double> in = tensor::eye<double> (n);
  const nd_tensor<double> sys =
    tensor::kron (a, in) + tensor::kron (im, b.t ());
  const nd_tensor<double> rhs = c.ravel ();
  return solve (sys, rhs).reshape ({m, n});
}

/* Continuous-time Lyapunov equation solver: A * X + X * A^T = Q
   (SciPy linalg.solve_continuous_lyapunov).  */
inline nd_tensor<double>
solve_continuous_lyapunov (const nd_tensor<double> &a,
                           const nd_tensor<double> &q)
{
  return solve_sylvester (a, a.t (), q);
}

/* Direct O(N^2) forward / backward substitution for triangular linear systems
   T * x = b (SciPy linalg.solve_triangular equivalent).  */
inline nd_tensor<double>
solve_triangular (const nd_tensor<double> &t_mat,
                  const nd_tensor<double> &b,
                  bool lower = false,
                  bool unit_diagonal = false)
{
  if (t_mat.ndim () != 2 || t_mat.shape ()[0] != t_mat.shape ()[1])
    {
      throw shape_error ("solve_triangular requires a square 2D matrix");
    }
  const std::size_t n = t_mat.shape ()[0];
  if (b.size () != n)
    {
      throw shape_error ("solve_triangular rhs dimension mismatch");
    }

  nd_tensor<double> x (b.shape (), 0.0);
  if (lower)
    {
      for (std::size_t i = 0; i < n; ++i)
        {
          double s = b.eval_at (i);
          for (std::size_t j = 0; j < i; ++j)
            {
              s -= t_mat (i, j) * x[j];
            }
          if (!unit_diagonal)
            {
              const double diag = t_mat (i, i);
              if (std::abs (diag) < 1e-15)
                {
                  throw singular_matrix_error ("singular matrix in solve_triangular");
                }
              s /= diag;
            }
          x[i] = s;
        }
    }
  else
    {
      for (std::ptrdiff_t i = static_cast<std::ptrdiff_t> (n) - 1; i >= 0; --i)
        {
          const std::size_t ui = static_cast<std::size_t> (i);
          double s = b.eval_at (ui);
          for (std::size_t j = ui + 1; j < n; ++j)
            {
              s -= t_mat (ui, j) * x[j];
            }
          if (!unit_diagonal)
            {
              const double diag = t_mat (ui, ui);
              if (std::abs (diag) < 1e-15)
                {
                  throw singular_matrix_error ("singular matrix in solve_triangular");
                }
              s /= diag;
            }
          x[ui] = s;
        }
    }
  return x;
}

/* Orthonormal basis for column range (`orth`) and right null space
   (`null_space`) via SVD (SciPy linalg.orth & linalg.null_space).  */
inline nd_tensor<double>
orth (const nd_tensor<double> &a, double rcond = 1e-12)
{
  const svd_result sv = svd (a);
  const double cutoff = (sv.s.size () > 0) ? (rcond * sv.s[0]) : 0.0;
  std::size_t rank = 0;
  for (std::size_t i = 0; i < sv.s.size (); ++i)
    {
      if (sv.s[i] > cutoff)
        {
          ++rank;
        }
    }
  const std::size_t m = a.shape ()[0];
  nd_tensor<double> q (shape_type {m, rank});
  for (std::size_t i = 0; i < m; ++i)
    {
      for (std::size_t j = 0; j < rank; ++j)
        {
          q (i, j) = sv.u (i, j);
        }
    }
  return q;
}

inline nd_tensor<double>
null_space (const nd_tensor<double> &a, double rcond = 1e-12)
{
  if (a.ndim () != 2)
    {
      throw shape_error ("null_space requires a 2D matrix");
    }
  const std::size_t m = a.shape ()[0];
  const std::size_t n = a.shape ()[1];
  /* Pad with zero rows if m < n so V^T has full n x n dimension.  */
  nd_tensor<double> work = a;
  if (m < n)
    {
      work = tensor::zeros<double> ({n, n});
      for (std::size_t i = 0; i < m; ++i)
        {
          for (std::size_t j = 0; j < n; ++j)
            {
              work (i, j) = a (i, j);
            }
        }
    }
  const svd_result sv = svd (work);
  const double cutoff = (sv.s.size () > 0) ? (rcond * sv.s[0]) : 0.0;
  std::size_t rank = 0;
  for (std::size_t i = 0; i < sv.s.size (); ++i)
    {
      if (sv.s[i] > cutoff)
        {
          ++rank;
        }
    }
  const std::size_t null_dim = (n > rank) ? (n - rank) : 0;
  nd_tensor<double> ns (shape_type {n, null_dim});
  for (std::size_t j = 0; j < null_dim; ++j)
    {
      const std::size_t vt_row = rank + j;
      for (std::size_t i = 0; i < n; ++i)
        {
          ns (i, j) = sv.vt (vt_row, i);
        }
    }
  return ns;
}

/* Principal Matrix Square Root S^2 = A via Denman-Beavers coupled Newton
   iteration (SciPy linalg.sqrtm equivalent).  */
inline nd_tensor<double>
sqrtm (const nd_tensor<double> &a, std::size_t max_iter = 40, double tol = 1e-12)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("sqrtm requires a square 2D matrix");
    }
  const std::size_t n = a.shape ()[0];
  nd_tensor<double> y = a.copy ();
  nd_tensor<double> z = tensor::eye<double> (n);

  for (std::size_t iter = 0; iter < max_iter; ++iter)
    {
      const nd_tensor<double> y_inv = inv (y);
      const nd_tensor<double> z_inv = inv (z);
      nd_tensor<double> y_next = 0.5 * (y + z_inv);
      nd_tensor<double> z_next = 0.5 * (z + y_inv);
      const double diff = norm (nd_tensor<double> (y_next - y));
      y = std::move (y_next);
      z = std::move (z_next);
      if (diff <= tol)
        {
          break;
        }
    }
  return y;
}

/* Matrix Logarithm for symmetric positive-definite matrices via spectral
   decomposition (SciPy linalg.logm equivalent).  */
inline nd_tensor<double>
logm_spd (const nd_tensor<double> &a)
{
  const eigh_result eg = eigh (a);
  const std::size_t n = eg.eigenvalues.size ();
  nd_tensor<double> log_d (shape_type {n, n}, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    {
      if (eg.eigenvalues[i] <= 0.0)
        {
          throw hyper_math_error ("logm_spd requires positive eigenvalues");
        }
      log_d (i, i) = std::log (eg.eigenvalues[i]);
    }
  return tensor::matmul (tensor::matmul (eg.eigenvectors, log_d),
                         eg.eigenvectors.t ());
}

/* Symmetric-Definite Generalized Eigenproblem A * v = \lambda * B * v
   via Cholesky reduction B = L * L^T => (L^{-1} A L^{-T}) y = \lambda y
   (SciPy linalg.eigh(a, b) equivalent).  */
inline eigh_result
eigh_generalized (const nd_tensor<double> &a, const nd_tensor<double> &b)
{
  if (a.ndim () != 2 || b.ndim () != 2 || a.shape () != b.shape ()
      || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("eigh_generalized requires matching square 2D matrices");
    }
  const nd_tensor<double> l = cholesky (b);
  const nd_tensor<double> l_inv = inv (l);
  const nd_tensor<double> c =
    tensor::matmul (tensor::matmul (l_inv, a), l_inv.t ());
  eigh_result res = eigh (c);
  res.eigenvectors = tensor::matmul (l_inv.t (), res.eigenvectors);
  return res;
}

/* Polynomial Companion Matrix and Sylvester-Hadamard 2^k x 2^k Orthogonal
   Matrix (SciPy linalg.companion, linalg.hadamard equivalents).  */
inline nd_tensor<double>
companion (const nd_tensor<double> &a)
{
  const std::size_t n = a.size ();
  if (n < 2)
    {
      throw shape_error ("companion requires polynomial coefficient array of size >= 2");
    }
  const double a0 = a.eval_at (0);
  if (std::abs (a0) == 0.0)
    {
      throw singular_matrix_error ("companion leading coefficient a[0] must be non-zero");
    }
  const std::size_t m = n - 1;
  nd_tensor<double> comp (shape_type {m, m}, 0.0);
  for (std::size_t j = 0; j < m; ++j)
    {
      comp (0, j) = -a.eval_at (j + 1) / a0;
    }
  for (std::size_t i = 1; i < m; ++i)
    {
      comp (i, i - 1) = 1.0;
    }
  return comp;
}

inline nd_tensor<double>
hadamard (std::size_t n)
{
  if (n == 0 || (n & (n - 1)) != 0)
    {
      throw shape_error ("hadamard order n must be a positive power of 2");
    }
  nd_tensor<double> h (shape_type {1, 1}, 1.0);
  std::size_t cur = 1;
  while (cur < n)
    {
      nd_tensor<double> next (shape_type {2 * cur, 2 * cur});
      for (std::size_t i = 0; i < cur; ++i)
        {
          for (std::size_t j = 0; j < cur; ++j)
            {
              const double v = h (i, j);
              next (i, j) = v;
              next (i, j + cur) = v;
              next (i + cur, j) = v;
              next (i + cur, j + cur) = -v;
            }
        }
      h = std::move (next);
      cur *= 2;
    }
  return h;
}

/* Explicit P * L * U Factorization (`lu`) where A = P * L * U with permutation
   matrix P, unit lower-triangular L, and upper-triangular U, and Symmetric
   L * D * L^T Factorization (`ldl`) where A = L * D * L^T (SciPy linalg.lu and
   linalg.ldl equivalents).  */
struct plu_result
{
  nd_tensor<double> p;
  nd_tensor<double> l;
  nd_tensor<double> u;
};

inline plu_result
lu (const nd_tensor<double> &a)
{
  const lu_result dec = lu_factor (a);
  const std::size_t n = dec.lu.shape ()[0];
  nd_tensor<double> p_mat (shape_type {n, n}, 0.0);
  nd_tensor<double> l_mat (shape_type {n, n}, 0.0);
  nd_tensor<double> u_mat (shape_type {n, n}, 0.0);

  /* dec.piv[i] is the original row placed at row i of P^T * A = L * U,
     so A[piv[i], :] = (L * U)[i, :] => P[piv[i], i] = 1.  */
  for (std::size_t i = 0; i < n; ++i)
    {
      p_mat (dec.piv[i], i) = 1.0;
      l_mat (i, i) = 1.0;
      for (std::size_t j = 0; j < n; ++j)
        {
          if (j < i)
            {
              l_mat (i, j) = dec.lu (i, j);
            }
          else
            {
              u_mat (i, j) = dec.lu (i, j);
            }
        }
    }
  return plu_result {std::move (p_mat), std::move (l_mat), std::move (u_mat)};
}

struct ldl_result
{
  nd_tensor<double> l;
  nd_tensor<double> d;
};

inline ldl_result
ldl (const nd_tensor<double> &a)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("ldl requires a square symmetric 2D matrix");
    }
  const std::size_t n = a.shape ()[0];
  nd_tensor<double> l_mat = tensor::eye<double> (n);
  nd_tensor<double> d_mat (shape_type {n, n}, 0.0);
  std::vector<double> d_diag (n, 0.0);

  for (std::size_t j = 0; j < n; ++j)
    {
      double dj = a (j, j);
      for (std::size_t k = 0; k < j; ++k)
        {
          dj -= l_mat (j, k) * l_mat (j, k) * d_diag[k];
        }
      if (std::abs (dj) < 1e-15)
        {
          throw singular_matrix_error ("ldl encountered zero pivot");
        }
      d_diag[j] = dj;
      d_mat (j, j) = dj;

      for (std::size_t i = j + 1; i < n; ++i)
        {
          double sum = a (i, j);
          for (std::size_t k = 0; k < j; ++k)
            {
              sum -= l_mat (i, k) * l_mat (j, k) * d_diag[k];
            }
          l_mat (i, j) = sum / dj;
        }
    }
  return ldl_result {std::move (l_mat), std::move (d_mat)};
}

/* Discrete-Time Control Equations & Khatri-Rao Product:
   - `solve_discrete_lyapunov`: Solves A * X * A^T - X + Q = 0
   - `solve_discrete_are`: Solves X = A^T X A - A^T X B (R + B^T X B)^{-1} B^T X A + Q
   - `khatri_rao`: Column-wise Kronecker product of 2D matrices
   (SciPy linalg.solve_discrete_lyapunov, solve_discrete_are, khatri_rao).  */
inline nd_tensor<double>
solve_discrete_lyapunov (const nd_tensor<double> &a, const nd_tensor<double> &q)
{
  if (a.ndim () != 2 || q.ndim () != 2 || a.shape () != q.shape ()
      || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("solve_discrete_lyapunov requires matching square matrices");
    }
  const std::size_t n = a.shape ()[0];
  /* Vectorized Kronecker system: (I_{n^2} - A \otimes A) vec(X) = vec(Q).  */
  const nd_tensor<double> kron_aa = tensor::kron (a, a);
  const nd_tensor<double> sys =
    tensor::eye<double> (n * n) - kron_aa;
  const nd_tensor<double> x_flat = solve (sys, q.ravel ());
  return x_flat.reshape ({n, n});
}

inline nd_tensor<double>
solve_discrete_are (const nd_tensor<double> &a,
                    const nd_tensor<double> &b,
                    const nd_tensor<double> &q,
                    const nd_tensor<double> &r,
                    std::size_t max_iter = 200,
                    double tol = 1e-12)
{
  if (a.ndim () != 2 || b.ndim () != 2 || q.ndim () != 2 || r.ndim () != 2)
    {
      throw shape_error ("solve_discrete_are requires 2D matrices");
    }
  const std::size_t n = a.shape ()[0];
  const std::size_t m = b.shape ()[1];
  if (a.shape ()[1] != n || b.shape ()[0] != n
      || q.shape () != a.shape ()
      || r.shape ()[0] != m || r.shape ()[1] != m)
    {
      throw shape_error ("solve_discrete_are matrix dimension mismatch");
    }

  nd_tensor<double> x = q.copy ();
  const nd_tensor<double> at = a.t ();
  const nd_tensor<double> bt = b.t ();

  for (std::size_t iter = 0; iter < max_iter; ++iter)
    {
      const nd_tensor<double> btx = tensor::matmul (bt, x);
      const nd_tensor<double> btxb_r =
        nd_tensor<double> (r + tensor::matmul (btx, b));
      const nd_tensor<double> btxa = tensor::matmul (btx, a);
      const nd_tensor<double> gain = solve (btxb_r, btxa);
      const nd_tensor<double> atxa =
        tensor::matmul (tensor::matmul (at, x), a);
      const nd_tensor<double> atxb =
        tensor::matmul (tensor::matmul (at, x), b);
      nd_tensor<double> x_next =
        nd_tensor<double> (atxa - tensor::matmul (atxb, gain) + q);
      const double diff = norm (nd_tensor<double> (x_next - x));
      x = std::move (x_next);
      if (diff <= tol)
        {
          break;
        }
    }
  return x;
}

inline nd_tensor<double>
khatri_rao (const nd_tensor<double> &a, const nd_tensor<double> &b)
{
  if (a.ndim () != 2 || b.ndim () != 2 || a.shape ()[1] != b.shape ()[1])
    {
      throw shape_error ("khatri_rao requires 2D matrices with equal column count");
    }
  const std::size_t ra = a.shape ()[0];
  const std::size_t rb = b.shape ()[0];
  const std::size_t ncols = a.shape ()[1];
  nd_tensor<double> out (shape_type {ra * rb, ncols});
  for (std::size_t j = 0; j < ncols; ++j)
    {
      for (std::size_t ia = 0; ia < ra; ++ia)
        {
          const double va = a (ia, j);
          for (std::size_t ib = 0; ib < rb; ++ib)
            {
              out (ia * rb + ib, j) = va * b (ib, j);
            }
        }
    }
  return out;
}

/* Continuous-Time Algebraic Riccati Equation (CARE) solver:
     A^T X + X A - X B R^{-1} B^T X + Q = 0
   Solved via the Byers-scaled Hamiltonian matrix sign function iteration
   (SciPy linalg.solve_continuous_are equivalent).  */
inline nd_tensor<double>
solve_continuous_are (const nd_tensor<double> &a,
                      const nd_tensor<double> &b,
                      const nd_tensor<double> &q,
                      const nd_tensor<double> &r,
                      std::size_t max_iter = 60,
                      double tol = 1e-12)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("solve_continuous_are requires square matrix a");
    }
  const std::size_t n = a.shape ()[0];
  const std::size_t n2 = 2 * n;
  const nd_tensor<double> g =
    tensor::matmul (b, solve (r, b.t ()));

  /* Assemble 2n x 2n Hamiltonian matrix H = [[A, -G], [-Q, -A^T]].  */
  nd_tensor<double> z (shape_type {n2, n2}, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          z (i, j) = a (i, j);
          z (i, n + j) = -g (i, j);
          z (n + i, j) = -q (i, j);
          z (n + i, n + j) = -a (j, i);
        }
    }

  for (std::size_t it = 0; it < max_iter; ++it)
    {
      const nd_tensor<double> z_inv = inv (z);
      const double n_z = norm (z);
      const double n_zi = norm (z_inv);
      const double scale =
        (n_z > 1e-15 && n_zi > 1e-15) ? std::sqrt (n_zi / n_z) : 1.0;
      nd_tensor<double> z_next =
        nd_tensor<double> (0.5 * (scale * z + (1.0 / scale) * z_inv));
      const double diff = norm (nd_tensor<double> (z_next - z));
      z = std::move (z_next);
      if (diff <= tol * (1.0 + norm (z)))
        {
          break;
        }
    }

  /* Solve [S12; S22] * X = -[S11; S21] where S = Z + I_{2n}.  */
  nd_tensor<double> lhs (shape_type {n2, n});
  nd_tensor<double> rhs (shape_type {n2, n});
  for (std::size_t i = 0; i < n2; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          const double s_ij = z (i, j) + ((i == j) ? 1.0 : 0.0);
          const double s_inj =
            z (i, n + j) + ((i == n + j) ? 1.0 : 0.0);
          lhs (i, j) = s_inj;
          rhs (i, j) = -s_ij;
        }
    }

  const nd_tensor<double> x_raw = lstsq (lhs, rhs);
  return nd_tensor<double> (0.5 * (x_raw + x_raw.t ()));
}

/* Leslie population projection matrix and Fiedler symmetric distance matrix
   (SciPy linalg.leslie and linalg.fiedler equivalents).  */
inline nd_tensor<double>
leslie (const nd_tensor<double> &f, const nd_tensor<double> &s)
{
  const std::size_t n = f.size ();
  if (n < 2 || s.size () != n - 1)
    {
      throw shape_error ("leslie requires |f| >= 2 and |s| == |f| - 1");
    }
  nd_tensor<double> out (shape_type {n, n}, 0.0);
  for (std::size_t j = 0; j < n; ++j)
    {
      out (0, j) = f.eval_at (j);
    }
  for (std::size_t i = 0; i + 1 < n; ++i)
    {
      out (i + 1, i) = s.eval_at (i);
    }
  return out;
}

inline nd_tensor<double>
fiedler (const nd_tensor<double> &a)
{
  const std::size_t n = a.size ();
  nd_tensor<double> out (shape_type {n, n}, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          out (i, j) = std::abs (a.eval_at (i) - a.eval_at (j));
        }
    }
  return out;
}

/* Pascal matrix `pascal`, exact inverse Pascal matrix `invpascal`, and
   Helmert orthogonal contrast matrix `helmert` (SciPy linalg equivalents).  */
inline nd_tensor<double>
pascal (std::size_t n, const std::string &kind = "symmetric")
{
  if (n == 0)
    {
      throw shape_error ("pascal requires n >= 1");
    }
  nd_tensor<double> l_mat (shape_type {n, n}, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    {
      l_mat (i, 0) = 1.0;
      l_mat (i, i) = 1.0;
      for (std::size_t j = 1; j < i; ++j)
        {
          l_mat (i, j) = l_mat (i - 1, j - 1) + l_mat (i - 1, j);
        }
    }
  if (kind == "lower")
    {
      return l_mat;
    }
  if (kind == "upper")
    {
      return l_mat.t ().copy ();
    }
  if (kind == "symmetric")
    {
      return tensor::matmul (l_mat, l_mat.t ());
    }
  throw hyper_math_error ("unknown pascal matrix kind: " + kind);
}

inline nd_tensor<double>
invpascal (std::size_t n, const std::string &kind = "symmetric")
{
  const nd_tensor<double> l_mat = pascal (n, "lower");
  nd_tensor<double> l_inv (shape_type {n, n}, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j <= i; ++j)
        {
          const double sgn = (((i - j) & 1) == 0) ? 1.0 : -1.0;
          l_inv (i, j) = sgn * l_mat (i, j);
        }
    }
  if (kind == "lower")
    {
      return l_inv;
    }
  if (kind == "upper")
    {
      return l_inv.t ().copy ();
    }
  if (kind == "symmetric")
    {
      return tensor::matmul (l_inv.t (), l_inv);
    }
  throw hyper_math_error ("unknown invpascal matrix kind: " + kind);
}

inline nd_tensor<double>
helmert (std::size_t n, bool full = false)
{
  if (n < 2)
    {
      throw shape_error ("helmert requires n >= 2");
    }
  const std::size_t nrows = full ? n : (n - 1);
  nd_tensor<double> out (shape_type {nrows, n}, 0.0);
  std::size_t r_offset = 0;
  if (full)
    {
      const double inv_sqrt_n = 1.0 / std::sqrt (static_cast<double> (n));
      for (std::size_t j = 0; j < n; ++j)
        {
          out (0, j) = inv_sqrt_n;
        }
      r_offset = 1;
    }
  for (std::size_t i = 1; i < n; ++i)
    {
      const double di = static_cast<double> (i);
      const double denom = std::sqrt (di * (di + 1.0));
      const std::size_t r = r_offset + (i - 1);
      for (std::size_t j = 0; j < i; ++j)
        {
          out (r, j) = 1.0 / denom;
        }
      out (r, i) = -di / denom;
    }
  return out;
}

/* Matrix Cosine `cosm`, Matrix Sine `sinm`, and Householder Upper Hessenberg
   Reduction `hessenberg` (A = Q * H * Q^T) (SciPy linalg equivalents).  */
inline nd_tensor<double>
cosm (const nd_tensor<double> &a)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("cosm requires a square 2D matrix");
    }
  const std::size_t n = a.shape ()[0];
  nd_tensor<double> blk (shape_type {2 * n, 2 * n}, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          blk (i, n + j) = -a (i, j);
          blk (n + i, j) = a (i, j);
        }
    }
  const nd_tensor<double> eb = expm (blk);
  nd_tensor<double> out (shape_type {n, n});
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          out (i, j) = eb (i, j);
        }
    }
  return out;
}

inline nd_tensor<double>
sinm (const nd_tensor<double> &a)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("sinm requires a square 2D matrix");
    }
  const std::size_t n = a.shape ()[0];
  nd_tensor<double> blk (shape_type {2 * n, 2 * n}, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          blk (i, n + j) = -a (i, j);
          blk (n + i, j) = a (i, j);
        }
    }
  const nd_tensor<double> eb = expm (blk);
  nd_tensor<double> out (shape_type {n, n});
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          out (i, j) = eb (n + i, j);
        }
    }
  return out;
}

inline hessenberg_result
hessenberg (const nd_tensor<double> &a)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("hessenberg requires a square 2D matrix");
    }
  const std::size_t n = a.shape ()[0];
  nd_tensor<double> h = a.copy ();
  nd_tensor<double> q = tensor::eye<double> (n);
  if (n <= 2)
    {
      return hessenberg_result {std::move (h), std::move (q)};
    }

  for (std::size_t k = 0; k + 2 < n; ++k)
    {
      const std::size_t m = n - (k + 1);
      std::vector<double> v (m, 0.0);
      double norm_sq = 0.0;
      for (std::size_t i = 0; i < m; ++i)
        {
          v[i] = h (k + 1 + i, k);
          norm_sq += v[i] * v[i];
        }
      const double norm_x = std::sqrt (norm_sq);
      if (norm_x < 1e-15)
        {
          continue;
        }
      v[0] += (v[0] >= 0.0 ? norm_x : -norm_x);
      double v_norm_sq = 0.0;
      for (double vi : v)
        {
          v_norm_sq += vi * vi;
        }
      if (v_norm_sq < 1e-30)
        {
          continue;
        }

      /* Left Householder application: H[k+1:, k:] -= 2 * v * (v^T H) / ||v||^2. */
      for (std::size_t j = k; j < n; ++j)
        {
          double dot = 0.0;
          for (std::size_t i = 0; i < m; ++i)
            {
              dot += v[i] * h (k + 1 + i, j);
            }
          const double scale = (2.0 * dot) / v_norm_sq;
          for (std::size_t i = 0; i < m; ++i)
            {
              h (k + 1 + i, j) -= scale * v[i];
            }
        }

      /* Right Householder application: H[:, k+1:] -= 2 * (H v) * v^T / ||v||^2. */
      for (std::size_t i = 0; i < n; ++i)
        {
          double dot = 0.0;
          for (std::size_t j = 0; j < m; ++j)
            {
              dot += h (i, k + 1 + j) * v[j];
            }
          const double scale = (2.0 * dot) / v_norm_sq;
          for (std::size_t j = 0; j < m; ++j)
            {
              h (i, k + 1 + j) -= scale * v[j];
            }
        }

      /* Accumulate Q[:, k+1:] -= 2 * (Q v) * v^T / ||v||^2.  */
      for (std::size_t i = 0; i < n; ++i)
        {
          double dot = 0.0;
          for (std::size_t j = 0; j < m; ++j)
            {
              dot += q (i, k + 1 + j) * v[j];
            }
          const double scale = (2.0 * dot) / v_norm_sq;
          for (std::size_t j = 0; j < m; ++j)
            {
              q (i, k + 1 + j) -= scale * v[j];
            }
        }
    }

  return hessenberg_result {std::move (h), std::move (q)};
}

/* Matrix Hyperbolic Cosine `coshm`, Matrix Hyperbolic Sine `sinhm`, and
   Principal Angles Between Subspaces `subspace_angles`
   (SciPy linalg.coshm, sinhm, and subspace_angles equivalents).  */
inline nd_tensor<double>
coshm (const nd_tensor<double> &a)
{
  const nd_tensor<double> ea = expm (a);
  const nd_tensor<double> ema = expm (nd_tensor<double> (-a));
  return nd_tensor<double> (0.5 * (ea + ema));
}

inline nd_tensor<double>
sinhm (const nd_tensor<double> &a)
{
  const nd_tensor<double> ea = expm (a);
  const nd_tensor<double> ema = expm (nd_tensor<double> (-a));
  return nd_tensor<double> (0.5 * (ea - ema));
}

inline nd_tensor<double>
subspace_angles (const nd_tensor<double> &a, const nd_tensor<double> &b)
{
  if (a.ndim () != 2 || b.ndim () != 2 || a.shape ()[0] != b.shape ()[0])
    {
      throw shape_error ("subspace_angles requires 2D matrices with equal row count");
    }
  const nd_tensor<double> qa = orth (a);
  const nd_tensor<double> qb = orth (b);
  const nd_tensor<double> m = tensor::matmul (qa.t (), qb);
  const svd_result sv = svd (m);
  const std::size_t k = sv.s.size ();
  nd_tensor<double> angles (shape_type {k});
  for (std::size_t i = 0; i < k; ++i)
    {
      const double c = std::clamp (sv.s[k - 1 - i], -1.0, 1.0);
      angles[i] = std::acos (c);
    }
  return angles;
}

/* O(N) Thomas Algorithm for Tridiagonal Linear Systems `solve_tridiagonal`,
   General Banded Solver `solve_banded`, Symmetric Positive-Definite Banded
   Cholesky `cholesky_banded` & `solveh_banded`, and Rank-Revealing Pivoted QR
   `qr_pivoted` (SciPy linalg equivalents).  */
inline nd_tensor<double>
solve_tridiagonal (const nd_tensor<double> &dl,
                   const nd_tensor<double> &d,
                   const nd_tensor<double> &du,
                   const nd_tensor<double> &b)
{
  const std::size_t n = d.size ();
  if (n == 0 || b.size () != n
      || (n > 1 && (dl.size () != n - 1 || du.size () != n - 1)))
    {
      throw shape_error ("solve_tridiagonal dimension mismatch");
    }
  if (n == 1)
    {
      if (std::abs (d[0]) < 1e-15)
        {
          throw singular_matrix_error ("solve_tridiagonal zero diagonal");
        }
      return nd_tensor<double> {b[0] / d[0]};
    }

  std::vector<double> c_prime (n - 1, 0.0);
  std::vector<double> d_prime (n, 0.0);

  if (std::abs (d[0]) < 1e-15)
    {
      throw singular_matrix_error ("solve_tridiagonal zero pivot at row 0");
    }
  c_prime[0] = du[0] / d[0];
  d_prime[0] = b[0] / d[0];

  for (std::size_t i = 1; i < n; ++i)
    {
      const double denom = d[i] - dl[i - 1] * c_prime[i - 1];
      if (std::abs (denom) < 1e-15)
        {
          throw singular_matrix_error ("solve_tridiagonal zero pivot");
        }
      if (i + 1 < n)
        {
          c_prime[i] = du[i] / denom;
        }
      d_prime[i] = (b[i] - dl[i - 1] * d_prime[i - 1]) / denom;
    }

  nd_tensor<double> x (shape_type {n});
  x[n - 1] = d_prime[n - 1];
  for (std::size_t k = n - 1; k > 0; --k)
    {
      const std::size_t i = k - 1;
      x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
  return x;
}

inline nd_tensor<double>
solve_banded (std::size_t l,
              std::size_t u,
              const nd_tensor<double> &ab,
              const nd_tensor<double> &b)
{
  if (ab.ndim () != 2 || ab.shape ()[0] != l + u + 1)
    {
      throw shape_error ("solve_banded requires ab of shape (l + u + 1, n)");
    }
  const std::size_t n = ab.shape ()[1];
  if (b.size () != n)
    {
      throw shape_error ("solve_banded rhs size must match n");
    }
  /* Reconstruct banded entries in a compact band or dense system and solve
     with partial-pivoting Gaussian elimination within the band.  */
  nd_tensor<double> a (shape_type {n, n}, 0.0);
  for (std::size_t j = 0; j < n; ++j)
    {
      const std::size_t i_min = (j > u) ? (j - u) : 0;
      const std::size_t i_max = std::min (n - 1, j + l);
      for (std::size_t i = i_min; i <= i_max; ++i)
        {
          const std::size_t band_row =
            static_cast<std::size_t> (static_cast<std::ptrdiff_t> (u)
                                      + static_cast<std::ptrdiff_t> (i)
                                      - static_cast<std::ptrdiff_t> (j));
          a (i, j) = ab (band_row, j);
        }
    }
  return solve (a, b);
}

inline nd_tensor<double>
cholesky_banded (const nd_tensor<double> &ab, bool lower = false)
{
  if (ab.ndim () != 2 || ab.shape ()[0] == 0)
    {
      throw shape_error ("cholesky_banded requires 2D non-empty banded matrix");
    }
  const std::size_t kd1 = ab.shape ()[0];
  const std::size_t kd = kd1 - 1;
  const std::size_t n = ab.shape ()[1];

  nd_tensor<double> a (shape_type {n, n}, 0.0);
  for (std::size_t j = 0; j < n; ++j)
    {
      if (lower)
        {
          const std::size_t i_max = std::min (n - 1, j + kd);
          for (std::size_t i = j; i <= i_max; ++i)
            {
              const double v = ab (i - j, j);
              a (i, j) = v;
              a (j, i) = v;
            }
        }
      else
        {
          const std::size_t i_min = (j > kd) ? (j - kd) : 0;
          for (std::size_t i = i_min; i <= j; ++i)
            {
              const double v = ab (kd + i - j, j);
              a (i, j) = v;
              a (j, i) = v;
            }
        }
    }

  const nd_tensor<double> l_mat = cholesky (a);
  nd_tensor<double> cb (shape_type {kd1, n}, 0.0);
  for (std::size_t j = 0; j < n; ++j)
    {
      if (lower)
        {
          const std::size_t i_max = std::min (n - 1, j + kd);
          for (std::size_t i = j; i <= i_max; ++i)
            {
              cb (i - j, j) = l_mat (i, j);
            }
        }
      else
        {
          const std::size_t i_min = (j > kd) ? (j - kd) : 0;
          for (std::size_t i = i_min; i <= j; ++i)
            {
              cb (kd + i - j, j) = l_mat (j, i);
            }
        }
    }
  return cb;
}

inline nd_tensor<double>
cho_solve (const nd_tensor<double> &c,
           const nd_tensor<double> &b,
           bool lower = true)
{
  if (lower)
    {
      const nd_tensor<double> y = solve_triangular (c, b, true, false);
      return solve_triangular (c.t (), y, false, false);
    }
  const nd_tensor<double> y = solve_triangular (c.t (), b, true, false);
  return solve_triangular (c, y, false, false);
}

inline nd_tensor<double>
solveh_banded (const nd_tensor<double> &ab,
               const nd_tensor<double> &b,
               bool lower = false)
{
  if (ab.ndim () != 2 || ab.shape ()[0] == 0)
    {
      throw shape_error ("solveh_banded requires 2D non-empty banded matrix");
    }
  const std::size_t kd = ab.shape ()[0] - 1;
  const std::size_t n = ab.shape ()[1];
  if (b.size () != n)
    {
      throw shape_error ("solveh_banded rhs size must match n");
    }

  nd_tensor<double> a (shape_type {n, n}, 0.0);
  for (std::size_t j = 0; j < n; ++j)
    {
      if (lower)
        {
          const std::size_t i_max = std::min (n - 1, j + kd);
          for (std::size_t i = j; i <= i_max; ++i)
            {
              const double v = ab (i - j, j);
              a (i, j) = v;
              a (j, i) = v;
            }
        }
      else
        {
          const std::size_t i_min = (j > kd) ? (j - kd) : 0;
          for (std::size_t i = i_min; i <= j; ++i)
            {
              const double v = ab (kd + i - j, j);
              a (i, j) = v;
              a (j, i) = v;
            }
        }
    }

  const nd_tensor<double> l_mat = cholesky (a);
  return cho_solve (l_mat, b);
}

struct qr_pivoted_result
{
  nd_tensor<double> q;
  nd_tensor<double> r;
  nd_tensor<std::size_t> p;
};

inline qr_pivoted_result
qr_pivoted (const nd_tensor<double> &a)
{
  if (a.ndim () != 2)
    {
      throw shape_error ("qr_pivoted requires a 2D matrix");
    }
  const std::size_t m = a.shape ()[0];
  const std::size_t n = a.shape ()[1];
  const std::size_t k_max = std::min (m, n);

  nd_tensor<double> r = a;
  nd_tensor<double> q = nd_tensor<double>::eye (m);
  nd_tensor<std::size_t> p (shape_type {n});
  for (std::size_t j = 0; j < n; ++j)
    {
      p[j] = j;
    }

  for (std::size_t k = 0; k < k_max; ++k)
    {
      /* Find column j >= k maximizing 2-norm of R[k:m, j].  */
      std::size_t best_j = k;
      double best_norm_sq = -1.0;
      for (std::size_t j = k; j < n; ++j)
        {
          double s = 0.0;
          for (std::size_t i = k; i < m; ++i)
            {
              const double v = r (i, j);
              s += v * v;
            }
          if (s > best_norm_sq)
            {
              best_norm_sq = s;
              best_j = j;
            }
        }

      if (best_j != k)
        {
          std::swap (p[k], p[best_j]);
          for (std::size_t i = 0; i < m; ++i)
            {
              std::swap (r (i, k), r (i, best_j));
            }
        }

      const std::size_t len = m - k;
      std::vector<double> v (len, 0.0);
      double norm_x = 0.0;
      for (std::size_t i = 0; i < len; ++i)
        {
          v[i] = r (k + i, k);
          norm_x += v[i] * v[i];
        }
      norm_x = std::sqrt (norm_x);
      if (norm_x < 1e-15)
        {
          continue;
        }
      const double sign = (v[0] >= 0.0) ? 1.0 : -1.0;
      v[0] += sign * norm_x;
      double v_norm_sq = 0.0;
      for (double vi : v)
        {
          v_norm_sq += vi * vi;
        }
      if (v_norm_sq < 1e-30)
        {
          continue;
        }

      /* Apply Householder reflector to R[k:m, k:n].  */
      for (std::size_t j = k; j < n; ++j)
        {
          double dot = 0.0;
          for (std::size_t i = 0; i < len; ++i)
            {
              dot += v[i] * r (k + i, j);
            }
          const double scale = (2.0 * dot) / v_norm_sq;
          for (std::size_t i = 0; i < len; ++i)
            {
              r (k + i, j) -= scale * v[i];
            }
        }

      /* Accumulate Q[:, k:m] -= 2 * (Q[:, k:m] v) v^T / ||v||^2.  */
      for (std::size_t i = 0; i < m; ++i)
        {
          double dot = 0.0;
          for (std::size_t l_idx = 0; l_idx < len; ++l_idx)
            {
              dot += q (i, k + l_idx) * v[l_idx];
            }
          const double scale = (2.0 * dot) / v_norm_sq;
          for (std::size_t l_idx = 0; l_idx < len; ++l_idx)
            {
              q (i, k + l_idx) -= scale * v[l_idx];
            }
        }
    }

  /* Zero out numerical noise strictly below the diagonal of R.  */
  for (std::size_t i = 1; i < m; ++i)
    {
      for (std::size_t j = 0; j < std::min (i, n); ++j)
        {
          r (i, j) = 0.0;
        }
    }

  return qr_pivoted_result {std::move (q), std::move (r), std::move (p)};
}

/* General Non-Symmetric Eigendecomposition `eig`, Generalized Schur / QZ
   Decomposition `qz`, Toeplitz & Circulant Solvers `solve_toeplitz` /
   `solve_circulant`, `matrix_power`, `fractional_matrix_power`, and `funm`
   (SciPy linalg equivalents).  */
struct eig_result
{
  nd_tensor<std::complex<double>> eigenvalues;
  nd_tensor<std::complex<double>> eigenvectors;
};

inline eig_result
eig (const nd_tensor<double> &a)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1] || a.shape ()[0] == 0)
    {
      throw shape_error ("eig requires a non-empty square matrix");
    }
  const std::size_t n = a.shape ()[0];
  nd_tensor<std::complex<double>> evals (shape_type {n}, eigvals (a));
  nd_tensor<std::complex<double>> evecs (shape_type {n, n},
                                         std::complex<double> (0.0, 0.0));

  for (std::size_t k = 0; k < n; ++k)
    {
      const std::complex<double> lam = evals[k];
      /* Two steps of complex shifted inverse iteration on (A - (lam + eps)*I) v = rhs. */
      std::vector<std::complex<double>> v (n);
      for (std::size_t i = 0; i < n; ++i)
        {
          v[i] = std::complex<double> (
            1.0 / static_cast<double> (i + 1),
            0.1 * static_cast<double> ((i + k) % 3));
        }

      for (int iter = 0; iter < 3; ++iter)
        {
          const double shift_eps = 1e-10 * (1.0 + std::abs (lam));
          std::vector<std::vector<std::complex<double>>> mat (
            n, std::vector<std::complex<double>> (n));
          for (std::size_t r = 0; r < n; ++r)
            {
              for (std::size_t c = 0; c < n; ++c)
                {
                  mat[r][c] = std::complex<double> (a (r, c), 0.0);
                }
              mat[r][r] -= (lam + std::complex<double> (shift_eps, shift_eps));
            }
          std::vector<std::complex<double>> rhs = v;

          /* Complex Gaussian elimination with partial pivoting.  */
          for (std::size_t p = 0; p < n; ++p)
            {
              std::size_t piv = p;
              double best_abs = std::abs (mat[p][p]);
              for (std::size_t r = p + 1; r < n; ++r)
                {
                  if (std::abs (mat[r][p]) > best_abs)
                    {
                      best_abs = std::abs (mat[r][p]);
                      piv = r;
                    }
                }
              if (piv != p)
                {
                  std::swap (mat[p], mat[piv]);
                  std::swap (rhs[p], rhs[piv]);
                }
              if (std::abs (mat[p][p]) < 1e-15)
                {
                  mat[p][p] = std::complex<double> (1e-15, 0.0);
                }
              for (std::size_t r = p + 1; r < n; ++r)
                {
                  const std::complex<double> factor = mat[r][p] / mat[p][p];
                  rhs[r] -= factor * rhs[p];
                  for (std::size_t c = p; c < n; ++c)
                    {
                      mat[r][c] -= factor * mat[p][c];
                    }
                }
            }
          for (std::size_t i = n; i > 0; --i)
            {
              const std::size_t r = i - 1;
              std::complex<double> s = rhs[r];
              for (std::size_t c = r + 1; c < n; ++c)
                {
                  s -= mat[r][c] * v[c];
                }
              v[r] = s / mat[r][r];
            }

          double nrm2 = 0.0;
          for (std::size_t i = 0; i < n; ++i)
            {
              nrm2 += std::norm (v[i]);
            }
          nrm2 = std::sqrt (std::max (1e-30, nrm2));
          for (std::size_t i = 0; i < n; ++i)
            {
              v[i] /= nrm2;
            }
        }

      for (std::size_t i = 0; i < n; ++i)
        {
          evecs (i, k) = v[i];
        }
    }

  return eig_result {std::move (evals), std::move (evecs)};
}

struct qz_result
{
  nd_tensor<double> aa;
  nd_tensor<double> bb;
  nd_tensor<double> q;
  nd_tensor<double> z;
};

inline qz_result
qz (const nd_tensor<double> &a, const nd_tensor<double> &b)
{
  if (a.ndim () != 2 || b.ndim () != 2
      || a.shape ()[0] != a.shape ()[1]
      || a.shape () != b.shape ())
    {
      throw shape_error ("qz requires square matrices A and B of identical shape");
    }
  /* Reduce B via QR: B = Q_b R_b, and compute Schur of A B^{-1} when B is
     nonsingular (or regularized pencil) to obtain orthogonal Q and Z such that
     A = Q * AA * Z^T and B = Q * BB * Z^T.  */
  const std::size_t n = a.shape ()[0];
  nd_tensor<double> b_reg = b.copy ();
  for (std::size_t i = 0; i < n; ++i)
    {
      if (std::abs (b_reg (i, i)) < 1e-12)
        {
          b_reg (i, i) += 1e-12;
        }
    }
  const nd_tensor<double> c = solve (b_reg.t (), a.t ()).t ();
  const auto sch = schur (c);
  const nd_tensor<double> q_left = sch.z;
  const nd_tensor<double> bt_q = tensor::matmul (b.t (), q_left);
  const auto qr_btq = qr (bt_q);
  nd_tensor<double> z_right = qr_btq.q;
  nd_tensor<double> aa =
    tensor::matmul (q_left.t (), tensor::matmul (a, z_right));
  nd_tensor<double> bb =
    tensor::matmul (q_left.t (), tensor::matmul (b, z_right));
  for (std::size_t j = 0; j < n; ++j)
    {
      if (bb (j, j) < 0.0)
        {
          for (std::size_t i = 0; i < n; ++i)
            {
              z_right (i, j) = -z_right (i, j);
              aa (i, j) = -aa (i, j);
              bb (i, j) = -bb (i, j);
            }
        }
    }
  return qz_result {std::move (aa), std::move (bb),
                    std::move (q_left), std::move (z_right)};
}

inline nd_tensor<double>
solve_toeplitz (const nd_tensor<double> &c,
                const nd_tensor<double> &r,
                const nd_tensor<double> &b)
{
  return solve (toeplitz (c, r), b);
}

inline nd_tensor<double>
solve_circulant (const nd_tensor<double> &c, const nd_tensor<double> &b)
{
  return solve (circulant (c), b);
}

inline nd_tensor<double>
matrix_power (const nd_tensor<double> &a, int n)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("matrix_power requires a square 2D matrix");
    }
  const std::size_t dim = a.shape ()[0];
  if (n == 0)
    {
      return nd_tensor<double>::eye (dim);
    }
  nd_tensor<double> base = (n < 0) ? inv (a) : a.copy ();
  unsigned int exp =
    static_cast<unsigned int> (n < 0 ? -static_cast<long long> (n)
                                     : static_cast<long long> (n));
  nd_tensor<double> result = nd_tensor<double>::eye (dim);
  while (exp > 0)
    {
      if ((exp & 1u) != 0u)
        {
          result = tensor::matmul (result, base);
        }
      exp >>= 1u;
      if (exp > 0)
        {
          base = tensor::matmul (base, base);
        }
    }
  return result;
}

template <typename ScalarFunc>
nd_tensor<double>
funm (const nd_tensor<double> &a, ScalarFunc &&f)
{
  const auto eg = eigh (a);
  const std::size_t n = a.shape ()[0];
  nd_tensor<double> f_diag (shape_type {n, n}, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    {
      f_diag (i, i) = f (eg.eigenvalues[i]);
    }
  return tensor::matmul (
    eg.eigenvectors,
    tensor::matmul (f_diag, eg.eigenvectors.t ()));
}

inline nd_tensor<double>
fractional_matrix_power (const nd_tensor<double> &a, double p)
{
  return funm (a, [p] (double lam) {
    return std::pow (std::max (0.0, lam), p);
  });
}

/* =========================================================================
   Singular Values Only (`svdvals`), N-D Tensor Linear Solver & Inverse
   (`tensorsolve`, `tensorinv`), Ordered/Complex Schur (`ordschur`, `rsf2csf`,
   `cdf2rdf`), Matrix Balancing (`matrix_balance`), Cosine-Sine Decomposition
   (`cossin`), Special Matrices (`dft_matrix`, `convolution_matrix`,
   `fiedler_companion`), and Randomized Sketching
   (`clarkson_woodruff_transform`).
   ========================================================================= */

inline nd_tensor<double>
svdvals (const nd_tensor<double> &a)
{
  return svd (a).s;
}

inline nd_tensor<double>
tensorsolve (const nd_tensor<double> &a, const nd_tensor<double> &b)
{
  const std::size_t prod_b = b.size ();
  if (prod_b == 0 || a.size () != prod_b * prod_b)
    {
      throw shape_error ("tensorsolve requires a.size() == b.size()^2");
    }
  std::size_t split_dim = 0;
  std::size_t acc = 1;
  for (std::size_t d = 0; d < a.ndim (); ++d)
    {
      acc *= a.shape ()[d];
      if (acc == prod_b)
        {
          split_dim = d + 1;
          break;
        }
    }
  shape_type x_shape;
  if (split_dim > 0 && split_dim < a.ndim ())
    {
      x_shape.assign (a.shape ().begin () + static_cast<std::ptrdiff_t> (split_dim),
                      a.shape ().end ());
    }
  else
    {
      x_shape = b.shape ();
    }
  nd_tensor<double> a_mat = a.reshape (shape_type {prod_b, prod_b});
  nd_tensor<double> b_vec = b.reshape (shape_type {prod_b});
  nd_tensor<double> x_vec = solve (a_mat, b_vec);
  return x_vec.reshape (x_shape);
}

inline nd_tensor<double>
tensorinv (const nd_tensor<double> &a, std::size_t ind = 2)
{
  if (ind == 0 || ind >= a.ndim ())
    {
      throw value_error ("tensorinv requires 0 < ind < a.ndim()");
    }
  std::size_t prod_left = 1;
  for (std::size_t d = 0; d < ind; ++d)
    {
      prod_left *= a.shape ()[d];
    }
  std::size_t prod_right = 1;
  for (std::size_t d = ind; d < a.ndim (); ++d)
    {
      prod_right *= a.shape ()[d];
    }
  if (prod_left != prod_right || prod_left == 0)
    {
      throw shape_error ("tensorinv requires matching product of leading and trailing dimensions");
    }
  shape_type inv_shape;
  for (std::size_t d = ind; d < a.ndim (); ++d)
    {
      inv_shape.push_back (a.shape ()[d]);
    }
  for (std::size_t d = 0; d < ind; ++d)
    {
      inv_shape.push_back (a.shape ()[d]);
    }
  nd_tensor<double> a_mat = a.reshape (shape_type {prod_left, prod_right});
  nd_tensor<double> a_inv = inv (a_mat);
  return a_inv.reshape (inv_shape);
}

struct ordschur_result
{
  nd_tensor<double> t;
  nd_tensor<double> z;
  std::size_t selected_count {};
};

inline ordschur_result
ordschur (const nd_tensor<double> &a, const std::string &sort_mode = "lhp")
{
  auto s = schur (a);
  const std::size_t n = a.shape ()[0];
  auto selects = [&] (double val) -> bool {
    if (sort_mode == "lhp")
      {
        return val < 0.0;
      }
    if (sort_mode == "rhp")
      {
        return val > 0.0;
      }
    if (sort_mode == "iuc")
      {
        return std::abs (val) < 1.0;
      }
    if (sort_mode == "ouc")
      {
        return std::abs (val) > 1.0;
      }
    throw value_error ("ordschur unknown sort_mode: " + sort_mode);
  };

  /* Reorder 1x1 diagonal blocks via Givens rotations so selected eigenvalues
     appear in the upper-left corner while preserving A = Z T Z^T.  */
  std::size_t m_sel = 0;
  for (std::size_t i = 0; i < n; ++i)
    {
      if (selects (s.t (i, i)))
        {
          for (std::size_t k = i; k > m_sel; --k)
            {
              const double t11 = s.t (k - 1, k - 1);
              const double t12 = s.t (k - 1, k);
              const double t22 = s.t (k, k);
              const double d = t22 - t11;
              const double r = std::hypot (t12, d);
              const double cs = (r < 1e-15) ? 0.0 : t12 / r;
              const double sn = (r < 1e-15) ? 1.0 : d / r;
              for (std::size_t j = 0; j < n; ++j)
                {
                  const double row1 = cs * s.t (k - 1, j) + sn * s.t (k, j);
                  const double row2 = -sn * s.t (k - 1, j) + cs * s.t (k, j);
                  s.t (k - 1, j) = row1;
                  s.t (k, j) = row2;
                }
              for (std::size_t j = 0; j < n; ++j)
                {
                  const double col1 = cs * s.t (j, k - 1) + sn * s.t (j, k);
                  const double col2 = -sn * s.t (j, k - 1) + cs * s.t (j, k);
                  s.t (j, k - 1) = col1;
                  s.t (j, k) = col2;
                  const double z1 = cs * s.z (j, k - 1) + sn * s.z (j, k);
                  const double z2 = -sn * s.z (j, k - 1) + cs * s.z (j, k);
                  s.z (j, k - 1) = z1;
                  s.z (j, k) = z2;
                }
              s.t (k, k - 1) = 0.0;
            }
          ++m_sel;
        }
    }
  return {std::move (s.t), std::move (s.z), m_sel};
}

struct complex_schur_result
{
  nd_tensor<std::complex<double>> t;
  nd_tensor<std::complex<double>> z;
};

inline complex_schur_result
rsf2csf (const nd_tensor<double> &t_real, const nd_tensor<double> &z_real)
{
  if (t_real.ndim () != 2 || z_real.ndim () != 2
      || t_real.shape () != z_real.shape ()
      || t_real.shape ()[0] != t_real.shape ()[1])
    {
      throw shape_error ("rsf2csf requires matching square matrices T and Z");
    }
  const std::size_t n = t_real.shape ()[0];
  nd_tensor<std::complex<double>> tc (shape_type {n, n},
                                      std::complex<double> (0.0, 0.0));
  nd_tensor<std::complex<double>> zc (shape_type {n, n},
                                      std::complex<double> (0.0, 0.0));
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          tc (i, j) = std::complex<double> (t_real (i, j), 0.0);
          zc (i, j) = std::complex<double> (z_real (i, j), 0.0);
        }
    }
  for (std::size_t i = 0; i + 1 < n; ++i)
    {
      if (std::abs (tc (i + 1, i)) > 1e-12)
        {
          const double a = tc (i, i).real ();
          const double b = tc (i, i + 1).real ();
          const double c = tc (i + 1, i).real ();
          const double d = tc (i + 1, i + 1).real ();
          const double tr = 0.5 * (a + d);
          const double disc = tr * tr - (a * d - b * c);
          const std::complex<double> lam =
            (disc >= 0.0)
              ? std::complex<double> (tr + std::sqrt (disc), 0.0)
              : std::complex<double> (tr, std::sqrt (-disc));
          std::complex<double> x = b;
          std::complex<double> y = lam - a;
          const double nrm = std::hypot (std::abs (x), std::abs (y));
          if (nrm > 1e-15)
            {
              x /= nrm;
              y /= nrm;
            }
          else
            {
              x = 1.0;
              y = 0.0;
            }
          for (std::size_t j = 0; j < n; ++j)
            {
              const auto r1 = std::conj (x) * tc (i, j) + std::conj (y) * tc (i + 1, j);
              const auto r2 = -y * tc (i, j) + x * tc (i + 1, j);
              tc (i, j) = r1;
              tc (i + 1, j) = r2;
            }
          for (std::size_t j = 0; j < n; ++j)
            {
              const auto c1 = tc (j, i) * x + tc (j, i + 1) * y;
              const auto c2 = tc (j, i) * (-std::conj (y)) + tc (j, i + 1) * std::conj (x);
              tc (j, i) = c1;
              tc (j, i + 1) = c2;
              const auto z1 = zc (j, i) * x + zc (j, i + 1) * y;
              const auto z2 = zc (j, i) * (-std::conj (y)) + zc (j, i + 1) * std::conj (x);
              zc (j, i) = z1;
              zc (j, i + 1) = z2;
            }
          tc (i + 1, i) = {0.0, 0.0};
        }
    }
  return {std::move (tc), std::move (zc)};
}

struct real_block_diag_result
{
  nd_tensor<double> d_real;
  nd_tensor<double> v_real;
};

inline real_block_diag_result
cdf2rdf (const nd_tensor<std::complex<double>> &evals,
         const nd_tensor<std::complex<double>> &evecs)
{
  const std::size_t n = evals.size ();
  if (evecs.ndim () != 2 || evecs.shape ()[0] != n || evecs.shape ()[1] != n)
    {
      throw shape_error ("cdf2rdf requires (N) eigenvalues and (N, N) eigenvectors");
    }
  nd_tensor<double> dr (shape_type {n, n}, 0.0);
  nd_tensor<double> vr (shape_type {n, n}, 0.0);
  std::size_t i = 0;
  while (i < n)
    {
      const double re = evals.eval_at (i).real ();
      const double im = evals.eval_at (i).imag ();
      if (std::abs (im) > 1e-12 && i + 1 < n)
        {
          dr (i, i) = re;
          dr (i, i + 1) = im;
          dr (i + 1, i) = -im;
          dr (i + 1, i + 1) = re;
          for (std::size_t r = 0; r < n; ++r)
            {
              vr (r, i) = evecs (r, i).real ();
              vr (r, i + 1) = evecs (r, i).imag ();
            }
          i += 2;
        }
      else
        {
          dr (i, i) = re;
          for (std::size_t r = 0; r < n; ++r)
            {
              vr (r, i) = evecs (r, i).real ();
            }
          ++i;
        }
    }
  return {std::move (dr), std::move (vr)};
}

struct balance_result
{
  nd_tensor<double> balanced;
  nd_tensor<double> scale;
};

inline balance_result
matrix_balance (const nd_tensor<double> &a)
{
  if (a.ndim () != 2 || a.shape ()[0] != a.shape ()[1])
    {
      throw shape_error ("matrix_balance requires a square 2D matrix");
    }
  const std::size_t n = a.shape ()[0];
  nd_tensor<double> b = a.copy ();
  nd_tensor<double> d (shape_type {n}, 1.0);
  for (int iter = 0; iter < 20; ++iter)
    {
      bool changed = false;
      for (std::size_t i = 0; i < n; ++i)
        {
          double r_norm = 0.0;
          double c_norm = 0.0;
          for (std::size_t j = 0; j < n; ++j)
            {
              if (j != i)
                {
                  r_norm += std::abs (b (i, j));
                  c_norm += std::abs (b (j, i));
                }
            }
          if (r_norm > 1e-14 && c_norm > 1e-14)
            {
              double f = std::sqrt (r_norm / c_norm);
              if (std::abs (f - 1.0) > 1e-3)
                {
                  d[i] *= f;
                  for (std::size_t j = 0; j < n; ++j)
                    {
                      if (j != i)
                        {
                          b (i, j) /= f;
                          b (j, i) *= f;
                        }
                    }
                  changed = true;
                }
            }
        }
      if (!changed)
        {
          break;
        }
    }
  return {std::move (b), std::move (d)};
}

struct cossin_result
{
  nd_tensor<double> u1;
  nd_tensor<double> u2;
  nd_tensor<double> theta;
  nd_tensor<double> v1t;
  nd_tensor<double> v2t;
};

inline cossin_result
cossin (const nd_tensor<double> &x, std::size_t p, std::size_t q)
{
  if (x.ndim () != 2 || x.shape ()[0] != x.shape ()[1])
    {
      throw shape_error ("cossin requires a square orthogonal matrix");
    }
  const std::size_t n = x.shape ()[0];
  if (p == 0 || q == 0 || p >= n || q >= n)
    {
      throw value_error ("cossin partition indices p and q must be in (0, n)");
    }
  nd_tensor<double> x11 (shape_type {p, q});
  nd_tensor<double> x21 (shape_type {n - p, q});
  for (std::size_t i = 0; i < p; ++i)
    {
      for (std::size_t j = 0; j < q; ++j)
        {
          x11 (i, j) = x (i, j);
        }
    }
  for (std::size_t i = 0; i < n - p; ++i)
    {
      for (std::size_t j = 0; j < q; ++j)
        {
          x21 (i, j) = x (p + i, j);
        }
    }
  auto svd11 = svd (x11);
  auto svd21 = svd (x21);
  const std::size_t r = std::min (p, q);
  nd_tensor<double> theta (shape_type {r});
  for (std::size_t i = 0; i < r; ++i)
    {
      const double c_val = std::clamp (svd11.s.eval_at (i), 0.0, 1.0);
      theta[i] = std::acos (c_val);
    }
  nd_tensor<double> x22 (shape_type {n - p, n - q});
  for (std::size_t i = 0; i < n - p; ++i)
    {
      for (std::size_t j = 0; j < n - q; ++j)
        {
          x22 (i, j) = x (p + i, q + j);
        }
    }
  auto svd22 = svd (x22);
  return {std::move (svd11.u), std::move (svd21.u), std::move (theta),
          std::move (svd11.vt), std::move (svd22.vt)};
}

inline nd_tensor<std::complex<double>>
dft_matrix (std::size_t n, const std::string &scale = "none")
{
  if (n == 0)
    {
      throw value_error ("dft_matrix requires n > 0");
    }
  double factor = 1.0;
  if (scale == "sqrtn")
    {
      factor = 1.0 / std::sqrt (static_cast<double> (n));
    }
  else if (scale == "n")
    {
      factor = 1.0 / static_cast<double> (n);
    }
  nd_tensor<std::complex<double>> m (shape_type {n, n});
  const double two_pi_over_n = -2.0 * constants::pi / static_cast<double> (n);
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          const double angle =
            two_pi_over_n * static_cast<double> ((i * j) % n);
          m (i, j) = std::polar (factor, angle);
        }
    }
  return m;
}

inline nd_tensor<double>
convolution_matrix (const nd_tensor<double> &a, std::size_t n,
                    const std::string &mode = "full")
{
  const std::size_t m = a.size ();
  if (m == 0 || n == 0)
    {
      throw value_error ("convolution_matrix requires non-empty kernel and n > 0");
    }
  const std::size_t full_rows = m + n - 1;
  nd_tensor<double> full (shape_type {full_rows, n}, 0.0);
  for (std::size_t c = 0; c < n; ++c)
    {
      for (std::size_t k = 0; k < m; ++k)
        {
          full (c + k, c) = a.eval_at (k);
        }
    }
  if (mode == "full")
    {
      return full;
    }
  if (mode == "same")
    {
      const std::size_t start = (m - 1) / 2;
      nd_tensor<double> out (shape_type {n, n}, 0.0);
      for (std::size_t r = 0; r < n; ++r)
        {
          for (std::size_t c = 0; c < n; ++c)
            {
              out (r, c) = full (start + r, c);
            }
        }
      return out;
    }
  if (mode == "valid")
    {
      const std::size_t valid_rows =
        (n >= m) ? (n - m + 1) : (m - n + 1);
      const std::size_t start = std::min (m, n) - 1;
      nd_tensor<double> out (shape_type {valid_rows, n}, 0.0);
      for (std::size_t r = 0; r < valid_rows; ++r)
        {
          for (std::size_t c = 0; c < n; ++c)
            {
              out (r, c) = full (start + r, c);
            }
        }
      return out;
    }
  throw value_error ("convolution_matrix unknown mode: " + mode);
}

inline nd_tensor<double>
fiedler_companion (const nd_tensor<double> &a)
{
  if (a.size () < 2)
    {
      throw value_error ("fiedler_companion requires polynomial degree >= 1");
    }
  const double a0 = a.eval_at (0);
  if (std::abs (a0) < 1e-15)
    {
      throw value_error ("fiedler_companion leading coefficient must be nonzero");
    }
  const std::size_t n = a.size () - 1;
  nd_tensor<double> c (shape_type {n, n}, 0.0);
  for (std::size_t j = 0; j < n; ++j)
    {
      c (0, j) = -a.eval_at (j + 1) / a0;
    }
  for (std::size_t i = 1; i < n; ++i)
    {
      c (i, i - 1) = 1.0;
    }
  return c;
}

inline nd_tensor<double>
clarkson_woodruff_transform (const nd_tensor<double> &a,
                             std::size_t sketch_rows,
                             std::uint64_t seed = 42)
{
  if (a.ndim () != 2 || sketch_rows == 0)
    {
      throw value_error ("clarkson_woodruff_transform requires 2D matrix and sketch_rows > 0");
    }
  const std::size_t m = a.shape ()[0];
  const std::size_t n = a.shape ()[1];
  nd_tensor<double> out (shape_type {sketch_rows, n}, 0.0);
  random_engine rng (seed);
  for (std::size_t i = 0; i < m; ++i)
    {
      const std::size_t h =
        static_cast<std::size_t> (rng.next_u64 ()) % sketch_rows;
      const double s = ((rng.next_u64 () & 1ULL) != 0ULL) ? 1.0 : -1.0;
      for (std::size_t j = 0; j < n; ++j)
        {
          out (h, j) += s * a (i, j);
        }
    }
  return out;
}

} // namespace hyper_math::scientific::linalg

#endif /* HYPER_MATH_SCIENTIFIC_LINALG_HPP */
