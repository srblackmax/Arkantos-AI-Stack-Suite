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

#ifndef HYPER_MATH_SCIENTIFIC_FFT_HPP
#define HYPER_MATH_SCIENTIFIC_FFT_HPP

#include "hyper_math/scientific/linalg.hpp"
#include "hyper_math/tensor/nd_tensor.hpp"

#include <bit>
#include <cmath>
#include <complex>
#include <vector>

namespace hyper_math::scientific::fft
{

using complex_t = std::complex<double>;
using tensor::nd_tensor;
using tensor::shape_type;

namespace detail
{

  inline bool
  is_power_of_two (std::size_t n) noexcept
  {
    return n > 0 && (n & (n - 1)) == 0;
  }

  /* In-place Radix-2 Decimation-In-Time Cooley-Tukey FFT for power-of-2 N. */
  inline void
  fft_radix2_inplace (std::vector<complex_t> &buf, bool inverse)
  {
    const std::size_t n = buf.size ();
    for (std::size_t i = 1, j = 0; i < n; ++i)
      {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
          {
            j ^= bit;
          }
        j ^= bit;
        if (i < j)
          {
            std::swap (buf[i], buf[j]);
          }
      }

    const double sign = inverse ? 1.0 : -1.0;
    for (std::size_t len = 2; len <= n; len <<= 1)
      {
        const double angle = sign * constants::two_pi / static_cast<double> (len);
        const complex_t wlen (std::cos (angle), std::sin (angle));
        const std::size_t half = len >> 1;
        for (std::size_t i = 0; i < n; i += len)
          {
            complex_t w (1.0, 0.0);
            for (std::size_t j = 0; j < half; ++j)
              {
                const complex_t u = buf[i + j];
                const complex_t v = buf[i + j + half] * w;
                buf[i + j] = u + v;
                buf[i + j + half] = u - v;
                w *= wlen;
              }
          }
      }

    if (inverse)
      {
        const double inv_n = 1.0 / static_cast<double> (n);
        for (auto &val : buf)
          {
            val *= inv_n;
          }
      }
  }

  /* Bluestein Chirp-Z Transform for arbitrary length N in O(N log N).
     Uses the identity: k * n = -(n - k)^2 / 2 + n^2 / 2 + k^2 / 2
     to reduce an arbitrary-length DFT to a power-of-2 circular convolution. */
  inline std::vector<complex_t>
  fft_bluestein (const std::vector<complex_t> &input, bool inverse)
  {
    const std::size_t n = input.size ();
    if (n <= 1)
      {
        return input;
      }
    if (is_power_of_two (n))
      {
        std::vector<complex_t> out = input;
        fft_radix2_inplace (out, inverse);
        return out;
      }

    std::size_t m = 1;
    while (m < 2 * n - 1)
      {
        m <<= 1;
      }

    const double sign = inverse ? 1.0 : -1.0;
    std::vector<complex_t> chirp (n);
    std::vector<complex_t> a (m, complex_t (0.0, 0.0));
    std::vector<complex_t> b (m, complex_t (0.0, 0.0));

    for (std::size_t k = 0; k < n; ++k)
      {
        const std::uint64_t k_sq_mod =
          (static_cast<std::uint64_t> (k) * static_cast<std::uint64_t> (k))
          % (2 * n);
        const double angle =
          sign * constants::pi * static_cast<double> (k_sq_mod)
          / static_cast<double> (n);
        chirp[k] = complex_t (std::cos (angle), std::sin (angle));
        a[k] = input[k] * chirp[k];
      }

    b[0] = std::conj (chirp[0]);
    for (std::size_t k = 1; k < n; ++k)
      {
        b[k] = std::conj (chirp[k]);
        b[m - k] = std::conj (chirp[k]);
      }

    fft_radix2_inplace (a, false);
    fft_radix2_inplace (b, false);
    for (std::size_t i = 0; i < m; ++i)
      {
        a[i] *= b[i];
      }
    fft_radix2_inplace (a, true);

    std::vector<complex_t> out (n);
    const double scale = inverse ? (1.0 / static_cast<double> (n)) : 1.0;
    for (std::size_t k = 0; k < n; ++k)
      {
        out[k] = a[k] * chirp[k] * scale;
      }
    return out;
  }

} // namespace detail

/* 1-D Forward Fast Fourier Transform (complex input).  */
inline nd_tensor<complex_t>
fft (const nd_tensor<complex_t> &x)
{
  std::vector<complex_t> vec = x.to_vector ();
  std::vector<complex_t> res = detail::fft_bluestein (vec, false);
  return nd_tensor<complex_t> (shape_type {res.size ()}, res);
}

/* 1-D Forward Fast Fourier Transform (real input).  */
inline nd_tensor<complex_t>
fft (const nd_tensor<double> &x)
{
  std::vector<complex_t> vec (x.size ());
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      vec[i] = complex_t (x.eval_at (i), 0.0);
    }
  std::vector<complex_t> res = detail::fft_bluestein (vec, false);
  return nd_tensor<complex_t> (shape_type {res.size ()}, res);
}

/* 1-D Inverse Fast Fourier Transform.  */
inline nd_tensor<complex_t>
ifft (const nd_tensor<complex_t> &X)
{
  std::vector<complex_t> vec = X.to_vector ();
  std::vector<complex_t> res = detail::fft_bluestein (vec, true);
  return nd_tensor<complex_t> (shape_type {res.size ()}, res);
}

/* Real-input non-redundant half-spectrum FFT of length (N/2 + 1).  */
inline nd_tensor<complex_t>
rfft (const nd_tensor<double> &x)
{
  if (x.size () == 0)
    {
      return nd_tensor<complex_t> (shape_type {0});
    }
  nd_tensor<complex_t> full_spec = fft (x);
  const std::size_t n_out = x.size () / 2 + 1;
  nd_tensor<complex_t> half_spec (shape_type {n_out});
  for (std::size_t i = 0; i < n_out; ++i)
    {
      half_spec[i] = full_spec[i];
    }
  return half_spec;
}

/* 2-D Forward and Inverse FFT via row-column separable transform.  */
inline nd_tensor<complex_t>
fft2 (const nd_tensor<double> &x)
{
  if (x.ndim () != 2)
    {
      throw shape_error ("fft2 requires a 2D tensor");
    }
  const std::size_t rows = x.shape ()[0];
  const std::size_t cols = x.shape ()[1];
  nd_tensor<complex_t> out (shape_type {rows, cols});

  std::vector<complex_t> row_buf (cols);
  for (std::size_t r = 0; r < rows; ++r)
    {
      for (std::size_t c = 0; c < cols; ++c)
        {
          row_buf[c] = complex_t (x (r, c), 0.0);
        }
      const auto r_fft = detail::fft_bluestein (row_buf, false);
      for (std::size_t c = 0; c < cols; ++c)
        {
          out (r, c) = r_fft[c];
        }
    }

  std::vector<complex_t> col_buf (rows);
  for (std::size_t c = 0; c < cols; ++c)
    {
      for (std::size_t r = 0; r < rows; ++r)
        {
          col_buf[r] = out (r, c);
        }
      const auto c_fft = detail::fft_bluestein (col_buf, false);
      for (std::size_t r = 0; r < rows; ++r)
        {
          out (r, c) = c_fft[r];
        }
    }
  return out;
}

inline nd_tensor<complex_t>
ifft2 (const nd_tensor<complex_t> &X)
{
  if (X.ndim () != 2)
    {
      throw shape_error ("ifft2 requires a 2D tensor");
    }
  const std::size_t rows = X.shape ()[0];
  const std::size_t cols = X.shape ()[1];
  nd_tensor<complex_t> out (shape_type {rows, cols});

  std::vector<complex_t> row_buf (cols);
  for (std::size_t r = 0; r < rows; ++r)
    {
      for (std::size_t c = 0; c < cols; ++c)
        {
          row_buf[c] = X (r, c);
        }
      const auto r_ifft = detail::fft_bluestein (row_buf, true);
      for (std::size_t c = 0; c < cols; ++c)
        {
          out (r, c) = r_ifft[c];
        }
    }

  std::vector<complex_t> col_buf (rows);
  for (std::size_t c = 0; c < cols; ++c)
    {
      for (std::size_t r = 0; r < rows; ++r)
        {
          col_buf[r] = out (r, c);
        }
      const auto c_ifft = detail::fft_bluestein (col_buf, true);
      for (std::size_t r = 0; r < rows; ++r)
        {
          out (r, c) = c_ifft[r];
        }
    }
  return out;
}

/* Discrete Fourier Transform sample frequencies.  */
inline nd_tensor<double>
fftfreq (std::size_t n, double d = 1.0)
{
  if (n == 0)
    {
      throw shape_error ("fftfreq requires n > 0");
    }
  nd_tensor<double> freqs (shape_type {n});
  const double scale = 1.0 / (static_cast<double> (n) * d);
  const std::size_t half = (n - 1) / 2 + 1;
  for (std::size_t i = 0; i < half; ++i)
    {
      freqs[i] = static_cast<double> (i) * scale;
    }
  for (std::size_t i = half; i < n; ++i)
    {
      freqs[i] = -static_cast<double> (n - i) * scale;
    }
  return freqs;
}

/* Shift zero-frequency component to center of spectrum.  */
template <typename T>
nd_tensor<T>
fftshift (const nd_tensor<T> &x)
{
  const std::size_t n = x.size ();
  if (n == 0)
    {
      return x;
    }
  nd_tensor<T> out (x.shape ());
  const std::size_t pivot = (n + 1) / 2;
  for (std::size_t i = 0; i < n; ++i)
    {
      out[i] = x.eval_at ((i + pivot) % n);
    }
  return out;
}

/* Inverse of `fftshift`: restores zero-frequency component to index 0
   (NumPy/SciPy fft.ifftshift equivalent, exact for both even and odd N).  */
template <typename T>
nd_tensor<T>
ifftshift (const nd_tensor<T> &x)
{
  const std::size_t n = x.size ();
  if (n == 0)
    {
      return x;
    }
  nd_tensor<T> out (x.shape ());
  const std::size_t pivot = n / 2;
  for (std::size_t i = 0; i < n; ++i)
    {
      out[i] = x.eval_at ((i + pivot) % n);
    }
  return out;
}

/* Sample frequencies for `rfft` of length (n / 2 + 1)
   (NumPy/SciPy fft.rfftfreq equivalent).  */
inline nd_tensor<double>
rfftfreq (std::size_t n, double d = 1.0)
{
  if (n == 0)
    {
      throw shape_error ("rfftfreq requires n > 0");
    }
  const std::size_t n_out = n / 2 + 1;
  nd_tensor<double> freqs (shape_type {n_out});
  const double scale = 1.0 / (static_cast<double> (n) * d);
  for (std::size_t i = 0; i < n_out; ++i)
    {
      freqs[i] = static_cast<double> (i) * scale;
    }
  return freqs;
}

/* 1-D Inverse of `rfft` reconstructing a real-valued signal of length `n`
   (defaults to `2 * (X.size() - 1)` when `n == 0`, NumPy/SciPy irfft).  */
inline nd_tensor<double>
irfft (const nd_tensor<complex_t> &X, std::size_t n = 0)
{
  const std::size_t m = X.size ();
  if (m == 0)
    {
      throw shape_error ("irfft requires non-empty input");
    }
  if (n == 0)
    {
      if (m < 2)
        {
          throw shape_error ("irfft with n == 0 requires X.size() >= 2");
        }
      n = 2 * (m - 1);
    }

  std::vector<complex_t> full_spec (n, complex_t (0.0, 0.0));
  const std::size_t copy_len = std::min (m, n / 2 + 1);
  for (std::size_t k = 0; k < copy_len; ++k)
    {
      full_spec[k] = X.eval_at (k);
    }
  /* Enforce Hermitian symmetry: X[n - k] = conj(X[k]).  */
  for (std::size_t k = 1; k < (n + 1) / 2; ++k)
    {
      if (k < copy_len)
        {
          full_spec[n - k] = std::conj (full_spec[k]);
        }
    }

  const std::vector<complex_t> time_c = detail::fft_bluestein (full_spec, true);
  nd_tensor<double> out (shape_type {n});
  for (std::size_t i = 0; i < n; ++i)
    {
      out[i] = time_c[i].real ();
    }
  return out;
}

/* 2-D Real-input non-redundant half-spectrum FFT `rfft2` of shape
   (rows, cols / 2 + 1) and its inverse `irfft2` (NumPy/SciPy equivalents).  */
inline nd_tensor<complex_t>
rfft2 (const nd_tensor<double> &x)
{
  if (x.ndim () != 2)
    {
      throw shape_error ("rfft2 requires a 2D tensor");
    }
  const std::size_t rows = x.shape ()[0];
  const std::size_t cols = x.shape ()[1];
  const std::size_t cols_out = cols / 2 + 1;
  const nd_tensor<complex_t> full = fft2 (x);
  nd_tensor<complex_t> out (shape_type {rows, cols_out});
  for (std::size_t r = 0; r < rows; ++r)
    {
      for (std::size_t c = 0; c < cols_out; ++c)
        {
          out (r, c) = full (r, c);
        }
    }
  return out;
}

inline nd_tensor<double>
irfft2 (const nd_tensor<complex_t> &X, std::size_t cols = 0)
{
  if (X.ndim () != 2)
    {
      throw shape_error ("irfft2 requires a 2D tensor");
    }
  const std::size_t rows = X.shape ()[0];
  const std::size_t m_cols = X.shape ()[1];
  if (cols == 0)
    {
      if (m_cols < 2)
        {
          throw shape_error ("irfft2 with cols == 0 requires X.shape()[1] >= 2");
        }
      cols = 2 * (m_cols - 1);
    }

  /* First apply 1-D complex IFFT along columns (axis 0), then 1-D irfft
     along each row (axis 1).  */
  nd_tensor<complex_t> col_ifft (shape_type {rows, m_cols});
  std::vector<complex_t> col_buf (rows);
  for (std::size_t c = 0; c < m_cols; ++c)
    {
      for (std::size_t r = 0; r < rows; ++r)
        {
          col_buf[r] = X (r, c);
        }
      const auto ci = detail::fft_bluestein (col_buf, true);
      for (std::size_t r = 0; r < rows; ++r)
        {
          col_ifft (r, c) = ci[r];
        }
    }

  nd_tensor<double> out (shape_type {rows, cols});
  nd_tensor<complex_t> row_half (shape_type {m_cols});
  for (std::size_t r = 0; r < rows; ++r)
    {
      for (std::size_t c = 0; c < m_cols; ++c)
        {
          row_half[c] = col_ifft (r, c);
        }
      const nd_tensor<double> row_real = irfft (row_half, cols);
      for (std::size_t c = 0; c < cols; ++c)
        {
          out (r, c) = row_real[c];
        }
    }
  return out;
}

/* Type-II Discrete Cosine Transform (DCT-II) and Type-III Inverse (IDCT)
   with optional orthonormal scaling (SciPy fft.dct / fft.idct).  */
inline nd_tensor<double>
dct (const nd_tensor<double> &x, bool ortho = true)
{
  const std::size_t n = x.size ();
  if (n == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  nd_tensor<double> out (shape_type {n}, 0.0);
  const double dn = static_cast<double> (n);
  for (std::size_t k = 0; k < n; ++k)
    {
      double acc = 0.0;
      const double dk = static_cast<double> (k);
      for (std::size_t m = 0; m < n; ++m)
        {
          const double angle =
            constants::pi * (static_cast<double> (m) + 0.5) * dk / dn;
          acc += x.eval_at (m) * std::cos (angle);
        }
      if (ortho)
        {
          const double scale =
            (k == 0) ? std::sqrt (1.0 / dn) : std::sqrt (2.0 / dn);
          out[k] = acc * scale;
        }
      else
        {
          out[k] = 2.0 * acc;
        }
    }
  return out;
}

inline nd_tensor<double>
idct (const nd_tensor<double> &X, bool ortho = true)
{
  const std::size_t n = X.size ();
  if (n == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  nd_tensor<double> out (shape_type {n}, 0.0);
  const double dn = static_cast<double> (n);
  for (std::size_t m = 0; m < n; ++m)
    {
      double acc = ortho
                   ? (X.eval_at (0) * std::sqrt (1.0 / dn))
                   : (0.5 * X.eval_at (0));
      const double dm = static_cast<double> (m);
      for (std::size_t k = 1; k < n; ++k)
        {
          const double angle =
            constants::pi * (dm + 0.5) * static_cast<double> (k) / dn;
          const double coeff =
            ortho ? (X.eval_at (k) * std::sqrt (2.0 / dn)) : X.eval_at (k);
          acc += coeff * std::cos (angle);
        }
      out[m] = ortho ? acc : (acc / dn);
    }
  return out;
}

/* Fast O(N log N) 1-D linear convolution via FFT (SciPy signal.fftconvolve). */
inline nd_tensor<double>
convolve_fft (const nd_tensor<double> &a, const nd_tensor<double> &v)
{
  const std::size_t na = a.size ();
  const std::size_t nv = v.size ();
  if (na == 0 || nv == 0)
    {
      throw shape_error ("convolve_fft requires non-empty inputs");
    }
  const std::size_t out_len = na + nv - 1;
  std::size_t n_fft = 1;
  while (n_fft < out_len)
    {
      n_fft <<= 1;
    }

  std::vector<complex_t> fa (n_fft, complex_t (0.0, 0.0));
  std::vector<complex_t> fv (n_fft, complex_t (0.0, 0.0));
  for (std::size_t i = 0; i < na; ++i)
    {
      fa[i] = complex_t (a.eval_at (i), 0.0);
    }
  for (std::size_t i = 0; i < nv; ++i)
    {
      fv[i] = complex_t (v.eval_at (i), 0.0);
    }

  detail::fft_radix2_inplace (fa, false);
  detail::fft_radix2_inplace (fv, false);
  for (std::size_t i = 0; i < n_fft; ++i)
    {
      fa[i] *= fv[i];
    }
  detail::fft_radix2_inplace (fa, true);

  nd_tensor<double> out (shape_type {out_len});
  for (std::size_t i = 0; i < out_len; ++i)
    {
      out[i] = fa[i].real ();
    }
  return out;
}

/* 2-D Separable Discrete Cosine Transform `dct2` and Inverse `idct2`
   (SciPy fft.dctn / idctn 2-D equivalents).  */
inline nd_tensor<double>
dct2 (const nd_tensor<double> &x, bool ortho = true)
{
  if (x.ndim () != 2)
    {
      throw shape_error ("dct2 requires a 2D tensor");
    }
  const std::size_t rows = x.shape ()[0];
  const std::size_t cols = x.shape ()[1];
  nd_tensor<double> out (shape_type {rows, cols});

  nd_tensor<double> row_buf (shape_type {cols});
  for (std::size_t r = 0; r < rows; ++r)
    {
      for (std::size_t c = 0; c < cols; ++c)
        {
          row_buf[c] = x (r, c);
        }
      const nd_tensor<double> rd = dct (row_buf, ortho);
      for (std::size_t c = 0; c < cols; ++c)
        {
          out (r, c) = rd[c];
        }
    }

  nd_tensor<double> col_buf (shape_type {rows});
  for (std::size_t c = 0; c < cols; ++c)
    {
      for (std::size_t r = 0; r < rows; ++r)
        {
          col_buf[r] = out (r, c);
        }
      const nd_tensor<double> cd = dct (col_buf, ortho);
      for (std::size_t r = 0; r < rows; ++r)
        {
          out (r, c) = cd[r];
        }
    }
  return out;
}

inline nd_tensor<double>
idct2 (const nd_tensor<double> &X, bool ortho = true)
{
  if (X.ndim () != 2)
    {
      throw shape_error ("idct2 requires a 2D tensor");
    }
  const std::size_t rows = X.shape ()[0];
  const std::size_t cols = X.shape ()[1];
  nd_tensor<double> out (shape_type {rows, cols});

  nd_tensor<double> row_buf (shape_type {cols});
  for (std::size_t r = 0; r < rows; ++r)
    {
      for (std::size_t c = 0; c < cols; ++c)
        {
          row_buf[c] = X (r, c);
        }
      const nd_tensor<double> ri = idct (row_buf, ortho);
      for (std::size_t c = 0; c < cols; ++c)
        {
          out (r, c) = ri[c];
        }
    }

  nd_tensor<double> col_buf (shape_type {rows});
  for (std::size_t c = 0; c < cols; ++c)
    {
      for (std::size_t r = 0; r < rows; ++r)
        {
          col_buf[r] = out (r, c);
        }
      const nd_tensor<double> ci = idct (col_buf, ortho);
      for (std::size_t r = 0; r < rows; ++r)
        {
          out (r, c) = ci[r];
        }
    }
  return out;
}

/* 1-D Type-II Discrete Sine Transform `dst` and Type-III Inverse `idst`
   (SciPy fft.dst / idst equivalents).  */
inline nd_tensor<double>
dst (const nd_tensor<double> &x, bool ortho = true)
{
  const std::size_t n = x.size ();
  if (n == 0)
    {
      throw shape_error ("dst requires non-empty input");
    }
  nd_tensor<double> out (shape_type {n});
  const double dn = static_cast<double> (n);
  for (std::size_t k = 0; k < n; ++k)
    {
      double acc = 0.0;
      const double dk1 = static_cast<double> (k + 1);
      for (std::size_t m = 0; m < n; ++m)
        {
          const double angle =
            constants::pi * (static_cast<double> (m) + 0.5) * dk1 / dn;
          acc += x.eval_at (m) * std::sin (angle);
        }
      if (ortho)
        {
          const double scale =
            (k + 1 == n) ? std::sqrt (1.0 / dn) : std::sqrt (2.0 / dn);
          out[k] = acc * scale;
        }
      else
        {
          out[k] = 2.0 * acc;
        }
    }
  return out;
}

inline nd_tensor<double>
idst (const nd_tensor<double> &X, bool ortho = true)
{
  const std::size_t n = X.size ();
  if (n == 0)
    {
      throw shape_error ("idst requires non-empty input");
    }
  nd_tensor<double> out (shape_type {n});
  const double dn = static_cast<double> (n);
  for (std::size_t m = 0; m < n; ++m)
    {
      const double dm = static_cast<double> (m) + 0.5;
      double acc = 0.0;
      for (std::size_t k = 0; k < n; ++k)
        {
          const double angle =
            constants::pi * dm * static_cast<double> (k + 1) / dn;
          double coeff = X.eval_at (k);
          if (ortho)
            {
              coeff *=
                (k + 1 == n) ? std::sqrt (1.0 / dn) : std::sqrt (2.0 / dn);
            }
          else if (k + 1 == n)
            {
              coeff *= 0.5;
            }
          acc += coeff * std::sin (angle);
        }
      out[m] = ortho ? acc : (acc / dn);
    }
  return out;
}

/* 5-smooth FFT length helper `next_fast_len`, 2-D `fftshift2` / `ifftshift2`,
   and Fast Hankel Transform `fht` / `ifht` (SciPy fft equivalents).  */
inline std::size_t
next_fast_len (std::size_t target) noexcept
{
  if (target <= 1)
    {
      return target;
    }
  for (std::size_t n = target;; ++n)
    {
      std::size_t m = n;
      while ((m % 2) == 0)
        {
          m /= 2;
        }
      while ((m % 3) == 0)
        {
          m /= 3;
        }
      while ((m % 5) == 0)
        {
          m /= 5;
        }
      if (m == 1)
        {
          return n;
        }
    }
}

inline nd_tensor<double>
fftshift2 (const nd_tensor<double> &x)
{
  if (x.ndim () != 2)
    {
      throw shape_error ("fftshift2 requires a 2D tensor");
    }
  const std::size_t r = x.shape ()[0];
  const std::size_t c = x.shape ()[1];
  const std::size_t sr = r / 2;
  const std::size_t sc = c / 2;
  nd_tensor<double> out (shape_type {r, c});
  for (std::size_t i = 0; i < r; ++i)
    {
      for (std::size_t j = 0; j < c; ++j)
        {
          out ((i + sr) % r, (j + sc) % c) = x (i, j);
        }
    }
  return out;
}

inline nd_tensor<double>
ifftshift2 (const nd_tensor<double> &x)
{
  if (x.ndim () != 2)
    {
      throw shape_error ("ifftshift2 requires a 2D tensor");
    }
  const std::size_t r = x.shape ()[0];
  const std::size_t c = x.shape ()[1];
  const std::size_t sr = (r + 1) / 2;
  const std::size_t sc = (c + 1) / 2;
  nd_tensor<double> out (shape_type {r, c});
  for (std::size_t i = 0; i < r; ++i)
    {
      for (std::size_t j = 0; j < c; ++j)
        {
          out ((i + sr) % r, (j + sc) % c) = x (i, j);
        }
    }
  return out;
}

inline nd_tensor<double>
fht (const nd_tensor<double> &a,
     double dln,
     double mu,
     double offset = 0.0)
{
  const std::size_t n = a.size ();
  if (n == 0 || dln <= 0.0)
    {
      throw shape_error ("fht requires non-empty input and dln > 0");
    }
  /* Discrete Hankel Transform on logarithmically spaced grid via Bessel kernel
     summation: A_k = sum_j a_j J_mu(exp(offset + (k + j - n) * dln)) dln.  */
  nd_tensor<double> out (shape_type {n}, 0.0);
  const double mid = 0.5 * static_cast<double> (n - 1);
  for (std::size_t k = 0; k < n; ++k)
    {
      double acc = 0.0;
      for (std::size_t j = 0; j < n; ++j)
        {
          const double ln_kr =
            offset + (static_cast<double> (k + j) - 2.0 * mid) * dln;
          const double kr = std::exp (ln_kr);
          acc += a.eval_at (j) * std::cyl_bessel_j (mu, kr) * kr;
        }
      out[k] = acc * dln;
    }
  return out;
}

inline nd_tensor<double>
ifht (const nd_tensor<double> &A,
      double dln,
      double mu,
      double offset = 0.0)
{
  /* Hankel transform is self-reciprocal; solve linear kernel system for exact
     discrete round-trip inversion on finite grids.  */
  const std::size_t n = A.size ();
  if (n == 0 || dln <= 0.0)
    {
      throw shape_error ("ifht requires non-empty input and dln > 0");
    }
  const double mid = 0.5 * static_cast<double> (n - 1);
  nd_tensor<double> k_mat (shape_type {n, n}, 0.0);
  for (std::size_t k = 0; k < n; ++k)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          const double ln_kr =
            offset + (static_cast<double> (k + j) - 2.0 * mid) * dln;
          const double kr = std::exp (ln_kr);
          k_mat (k, j) = std::cyl_bessel_j (mu, kr) * kr * dln;
        }
      k_mat (k, k) += 1e-12;
    }
  return linalg::solve (k_mat, A);
}

} // namespace hyper_math::scientific::fft

#endif /* HYPER_MATH_SCIENTIFIC_FFT_HPP */
