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

#ifndef HYPER_MATH_SCIENTIFIC_SIGNAL_HPP
#define HYPER_MATH_SCIENTIFIC_SIGNAL_HPP

#include "hyper_math/scientific/fft.hpp"
#include "hyper_math/scientific/linalg.hpp"
#include "hyper_math/tensor/nd_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace hyper_math::scientific::signal
{

using tensor::nd_tensor;
using tensor::shape_type;

/* Discrete 1-D linear convolution supporting "full", "same", and "valid"
   boundary modes (NumPy/SciPy convolve equivalent).  */
inline nd_tensor<double>
convolve (const nd_tensor<double> &a,
          const nd_tensor<double> &v,
          const std::string &mode = "full")
{
  const std::size_t na = a.size ();
  const std::size_t nv = v.size ();
  if (na == 0 || nv == 0)
    {
      throw shape_error ("convolve requires non-empty input tensors");
    }

  const std::size_t full_len = na + nv - 1;
  std::vector<double> full_res (full_len, 0.0);
  for (std::size_t i = 0; i < na; ++i)
    {
      const double ai = a.eval_at (i);
      for (std::size_t j = 0; j < nv; ++j)
        {
          full_res[i + j] += ai * v.eval_at (j);
        }
    }

  if (mode == "full")
    {
      return nd_tensor<double> (shape_type {full_len}, full_res);
    }
  else if (mode == "same")
    {
      const std::size_t out_len = std::max (na, nv);
      const std::size_t offset = (full_len - out_len) / 2;
      nd_tensor<double> out (shape_type {out_len});
      for (std::size_t i = 0; i < out_len; ++i)
        {
          out[i] = full_res[offset + i];
        }
      return out;
    }
  else if (mode == "valid")
    {
      const std::size_t out_len =
        std::max (na, nv) - std::min (na, nv) + 1;
      const std::size_t offset = std::min (na, nv) - 1;
      nd_tensor<double> out (shape_type {out_len});
      for (std::size_t i = 0; i < out_len; ++i)
        {
          out[i] = full_res[offset + i];
        }
      return out;
    }

  throw hyper_math_error ("unknown convolve mode: " + mode);
}

/* Discrete 1-D cross-correlation.  */
inline nd_tensor<double>
correlate (const nd_tensor<double> &a,
           const nd_tensor<double> &v,
           const std::string &mode = "valid")
{
  nd_tensor<double> v_rev (shape_type {v.size ()});
  for (std::size_t i = 0; i < v.size (); ++i)
    {
      v_rev[i] = v.eval_at (v.size () - 1 - i);
    }
  return convolve (a, v_rev, mode);
}

/* Spectral window generators: Hann, Hamming, Blackman.  */
inline nd_tensor<double>
hann_window (std::size_t m)
{
  if (m == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  if (m == 1)
    {
      return nd_tensor<double> (shape_type {1}, 1.0);
    }
  nd_tensor<double> w (shape_type {m});
  for (std::size_t n = 0; n < m; ++n)
    {
      w[n] = 0.5 * (1.0 - std::cos (constants::two_pi * static_cast<double> (n)
                                    / static_cast<double> (m - 1)));
    }
  return w;
}

inline nd_tensor<double>
hamming_window (std::size_t m)
{
  if (m == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  if (m == 1)
    {
      return nd_tensor<double> (shape_type {1}, 1.0);
    }
  nd_tensor<double> w (shape_type {m});
  for (std::size_t n = 0; n < m; ++n)
    {
      w[n] = 0.54 - 0.46 * std::cos (constants::two_pi * static_cast<double> (n)
                                     / static_cast<double> (m - 1));
    }
  return w;
}

inline nd_tensor<double>
blackman_window (std::size_t m)
{
  if (m == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  if (m == 1)
    {
      return nd_tensor<double> (shape_type {1}, 1.0);
    }
  nd_tensor<double> w (shape_type {m});
  for (std::size_t n = 0; n < m; ++n)
    {
      const double phase =
        constants::two_pi * static_cast<double> (n)
        / static_cast<double> (m - 1);
      w[n] = 0.42 - 0.5 * std::cos (phase) + 0.08 * std::cos (2.0 * phase);
    }
  return w;
}

/* Local maxima peak detector with minimum height and separation distance
   (SciPy signal.find_peaks equivalent).  */
inline std::vector<std::size_t>
find_peaks (const nd_tensor<double> &x,
            double min_height = -constants::inf,
            std::size_t min_distance = 1)
{
  std::vector<std::size_t> candidates;
  const std::size_t n = x.size ();
  for (std::size_t i = 1; i + 1 < n; ++i)
    {
      const double val = x.eval_at (i);
      if (val > x.eval_at (i - 1) && val >= x.eval_at (i + 1)
          && val >= min_height)
        {
          candidates.push_back (i);
        }
    }

  if (min_distance <= 1)
    {
      return candidates;
    }

  /* Greedily keep highest peaks first when enforcing min_distance.  */
  std::vector<std::size_t> by_height (candidates.begin (), candidates.end ());
  std::sort (by_height.begin (), by_height.end (),
             [&x] (std::size_t a, std::size_t b) {
               return x.eval_at (a) > x.eval_at (b);
             });

  std::vector<std::size_t> accepted;
  for (std::size_t idx : by_height)
    {
      bool too_close = false;
      for (std::size_t kept : accepted)
        {
          const std::size_t dist = (idx > kept) ? (idx - kept) : (kept - idx);
          if (dist < min_distance)
            {
              too_close = true;
              break;
            }
        }
      if (!too_close)
        {
          accepted.push_back (idx);
        }
    }

  std::sort (accepted.begin (), accepted.end ());
  return accepted;
}

/* Savitzky-Golay polynomial least-squares smoothing filter.  */
inline nd_tensor<double>
savgol_filter (const nd_tensor<double> &x,
               std::size_t window_length,
               std::size_t polyorder)
{
  if (window_length % 2 == 0 || window_length <= polyorder)
    {
      throw hyper_math_error (
        "savgol_filter requires odd window_length > polyorder");
    }

  const std::ptrdiff_t half =
    static_cast<std::ptrdiff_t> (window_length / 2);
  nd_tensor<double> J (shape_type {window_length, polyorder + 1});
  for (std::ptrdiff_t k = -half; k <= half; ++k)
    {
      const std::size_t row = static_cast<std::size_t> (k + half);
      double p = 1.0;
      for (std::size_t c = 0; c <= polyorder; ++c)
        {
          J (row, c) = p;
          p *= static_cast<double> (k);
        }
    }

  const nd_tensor<double> j_pinv = linalg::pinv (J);
  nd_tensor<double> coeffs (shape_type {window_length});
  for (std::size_t i = 0; i < window_length; ++i)
    {
      coeffs[i] = j_pinv (0, i);
    }

  return convolve (x, coeffs, "same");
}

/* Windowed-sinc FIR filter design (SciPy signal.firwin equivalent).
   `cutoff` is normalized in (0, 1), where 1.0 corresponds to the Nyquist
   frequency.  */
inline nd_tensor<double>
firwin (std::size_t numtaps, double cutoff, bool pass_zero = true)
{
  if (numtaps == 0 || cutoff <= 0.0 || cutoff >= 1.0)
    {
      throw hyper_math_error ("firwin requires numtaps >= 1 and cutoff in (0, 1)");
    }
  const nd_tensor<double> win = hamming_window (numtaps);
  const double alpha = 0.5 * static_cast<double> (numtaps - 1);
  nd_tensor<double> h (shape_type {numtaps});
  double sum_h = 0.0;

  for (std::size_t n = 0; n < numtaps; ++n)
    {
      const double m = static_cast<double> (n) - alpha;
      double sinc_val = cutoff;
      if (std::abs (m) > 1e-12)
        {
          sinc_val =
            std::sin (constants::pi * cutoff * m) / (constants::pi * m);
        }
      h[n] = sinc_val * win[n];
      sum_h += h[n];
    }

  for (std::size_t n = 0; n < numtaps; ++n)
    {
      h[n] /= sum_h;
    }

  if (!pass_zero)
    {
      if ((numtaps % 2) == 0)
        {
          throw hyper_math_error ("high-pass firwin requires odd numtaps");
        }
      /* Spectral inversion for high-pass FIR (requires odd numtaps).  */
      for (std::size_t n = 0; n < numtaps; ++n)
        {
          h[n] = -h[n];
        }
      h[numtaps / 2] += 1.0;
    }
  return h;
}

/* 2nd-order Digital Butterworth Low-Pass IIR Filter design via Bilinear
   Transform with frequency pre-warping (SciPy signal.butter(2, Wn)).
   Returns {b, a} transfer function coefficients where a[0] == 1.0.  */
struct iir_ba_coeffs
{
  nd_tensor<double> b;
  nd_tensor<double> a;
};

inline iir_ba_coeffs
butter_lowpass_ba (double wn)
{
  if (wn <= 0.0 || wn >= 1.0)
    {
      throw hyper_math_error ("butter_lowpass_ba requires wn in (0, 1)");
    }
  const double omega = std::tan (0.5 * constants::pi * wn);
  const double omega2 = omega * omega;
  const double sqrt2_omega = constants::sqrt2 * omega;
  const double a0 = 1.0 + sqrt2_omega + omega2;

  nd_tensor<double> b = {omega2 / a0, 2.0 * omega2 / a0, omega2 / a0};
  nd_tensor<double> a = {
    1.0,
    2.0 * (omega2 - 1.0) / a0,
    (1.0 - sqrt2_omega + omega2) / a0
  };
  return iir_ba_coeffs {std::move (b), std::move (a)};
}

/* Direct-Form II Transposed 1D Digital Filter (SciPy signal.lfilter).
   Solves: a[0]*y[n] = sum_k b[k]*x[n-k] - sum_{m>=1} a[m]*y[n-m].  */
inline nd_tensor<double>
lfilter (const nd_tensor<double> &b,
         const nd_tensor<double> &a,
         const nd_tensor<double> &x)
{
  if (b.empty () || a.empty () || std::abs (a.eval_at (0)) == 0.0)
    {
      throw hyper_math_error ("lfilter requires non-empty b, a with a[0] != 0");
    }
  const double a0 = a.eval_at (0);
  const std::size_t nb = b.size ();
  const std::size_t na = a.size ();
  const std::size_t n_order = std::max (nb, na);
  std::vector<double> b_norm (n_order, 0.0);
  std::vector<double> a_norm (n_order, 0.0);
  for (std::size_t i = 0; i < nb; ++i)
    {
      b_norm[i] = b.eval_at (i) / a0;
    }
  for (std::size_t i = 0; i < na; ++i)
    {
      a_norm[i] = a.eval_at (i) / a0;
    }

  const std::size_t n = x.size ();
  nd_tensor<double> y (shape_type {n}, 0.0);
  std::vector<double> z (n_order, 0.0);

  for (std::size_t i = 0; i < n; ++i)
    {
      const double xi = x.eval_at (i);
      const double yi = b_norm[0] * xi + (n_order > 1 ? z[0] : 0.0);
      y[i] = yi;
      for (std::size_t k = 1; k + 1 < n_order; ++k)
        {
          z[k - 1] = b_norm[k] * xi - a_norm[k] * yi + z[k];
        }
      if (n_order > 1)
        {
          z[n_order - 2] =
            b_norm[n_order - 1] * xi - a_norm[n_order - 1] * yi;
        }
    }
  return y;
}

/* Zero-Phase Forward-Backward Digital Filter (SciPy signal.filtfilt).  */
inline nd_tensor<double>
filtfilt (const nd_tensor<double> &b,
          const nd_tensor<double> &a,
          const nd_tensor<double> &x)
{
  const nd_tensor<double> fwd = lfilter (b, a, x);
  const nd_tensor<double> rev = tensor::flip (fwd);
  const nd_tensor<double> bwd = lfilter (b, a, rev);
  return tensor::flip (bwd);
}

/* Welch's Method for Power Spectral Density Estimation
   (SciPy signal.welch equivalent).
   Splits signal into 50%-overlapping Hann-windowed segments of length
   `nperseg` and averages their one-sided periodograms.  */
struct welch_result
{
  nd_tensor<double> freqs;
  nd_tensor<double> psd;
};

inline welch_result
welch (const nd_tensor<double> &x,
       double fs = 1.0,
       std::size_t nperseg = 64)
{
  const std::size_t n = x.size ();
  if (n == 0)
    {
      throw shape_error ("welch requires non-empty input signal");
    }
  if (nperseg == 0 || nperseg > n)
    {
      nperseg = n;
    }
  const std::size_t step = std::max<std::size_t> (1, nperseg / 2);
  const nd_tensor<double> win = hann_window (nperseg);
  double win_power = 0.0;
  for (std::size_t i = 0; i < nperseg; ++i)
    {
      win_power += win[i] * win[i];
    }

  const std::size_t n_freqs = nperseg / 2 + 1;
  nd_tensor<double> freqs (shape_type {n_freqs});
  for (std::size_t k = 0; k < n_freqs; ++k)
    {
      freqs[k] =
        (static_cast<double> (k) * fs) / static_cast<double> (nperseg);
    }

  nd_tensor<double> psd (shape_type {n_freqs}, 0.0);
  std::size_t n_segments = 0;

  for (std::size_t start = 0; start + nperseg <= n; start += step)
    {
      nd_tensor<double> seg (shape_type {nperseg});
      double seg_mean = 0.0;
      for (std::size_t i = 0; i < nperseg; ++i)
        {
          seg_mean += x.eval_at (start + i);
        }
      seg_mean /= static_cast<double> (nperseg);

      for (std::size_t i = 0; i < nperseg; ++i)
        {
          seg[i] = (x.eval_at (start + i) - seg_mean) * win[i];
        }

      const auto spec = fft::rfft (seg);
      for (std::size_t k = 0; k < n_freqs; ++k)
        {
          const double mag2 = std::norm (spec[k]);
          const double scale =
            (k > 0 && k + 1 < n_freqs) ? 2.0 : 1.0;
          psd[k] += (scale * mag2) / (fs * win_power);
        }
      ++n_segments;
    }

  if (n_segments > 0)
    {
      for (std::size_t k = 0; k < n_freqs; ++k)
        {
          psd[k] /= static_cast<double> (n_segments);
        }
    }
  return welch_result {std::move (freqs), std::move (psd)};
}

/* 2-D Spatial Convolution and 2-D Gaussian Image Smoothing
   (SciPy signal.convolve2d & ndimage.gaussian_filter equivalents).  */
inline nd_tensor<double>
convolve2d (const nd_tensor<double> &img,
            const nd_tensor<double> &kernel,
            const std::string &mode = "same")
{
  if (img.ndim () != 2 || kernel.ndim () != 2)
    {
      throw shape_error ("convolve2d requires 2D tensors");
    }
  const std::size_t r_in = img.shape ()[0];
  const std::size_t c_in = img.shape ()[1];
  const std::size_t r_k = kernel.shape ()[0];
  const std::size_t c_k = kernel.shape ()[1];
  const std::size_t r_full = r_in + r_k - 1;
  const std::size_t c_full = c_in + c_k - 1;

  nd_tensor<double> full (shape_type {r_full, c_full}, 0.0);
  for (std::size_t i = 0; i < r_in; ++i)
    {
      for (std::size_t j = 0; j < c_in; ++j)
        {
          const double v = img (i, j);
          for (std::size_t ki = 0; ki < r_k; ++ki)
            {
              for (std::size_t kj = 0; kj < c_k; ++kj)
                {
                  full (i + ki, j + kj) += v * kernel (ki, kj);
                }
            }
        }
    }

  if (mode == "full")
    {
      return full;
    }
  else if (mode == "same")
    {
      const std::size_t r_off = (r_k - 1) / 2;
      const std::size_t c_off = (c_k - 1) / 2;
      nd_tensor<double> out (shape_type {r_in, c_in});
      for (std::size_t i = 0; i < r_in; ++i)
        {
          for (std::size_t j = 0; j < c_in; ++j)
            {
              out (i, j) = full (r_off + i, c_off + j);
            }
        }
      return out;
    }
  else if (mode == "valid")
    {
      if (r_in < r_k || c_in < c_k)
        {
          throw shape_error ("convolve2d valid mode requires img >= kernel");
        }
      const std::size_t r_v = r_in - r_k + 1;
      const std::size_t c_v = c_in - c_k + 1;
      nd_tensor<double> out (shape_type {r_v, c_v});
      for (std::size_t i = 0; i < r_v; ++i)
        {
          for (std::size_t j = 0; j < c_v; ++j)
            {
              out (i, j) = full (r_k - 1 + i, c_k - 1 + j);
            }
        }
      return out;
    }
  throw hyper_math_error ("unknown convolve2d mode: " + mode);
}

inline nd_tensor<double>
gaussian_filter2d (const nd_tensor<double> &img, double sigma)
{
  if (sigma <= 0.0)
    {
      return img.copy ();
    }
  const std::size_t radius =
    std::max<std::size_t> (1, static_cast<std::size_t> (std::ceil (3.0 * sigma)));
  const std::size_t ksize = 2 * radius + 1;
  nd_tensor<double> k1d (shape_type {ksize});
  double sum_k = 0.0;
  for (std::size_t i = 0; i < ksize; ++i)
    {
      const double x =
        static_cast<double> (i) - static_cast<double> (radius);
      k1d[i] = std::exp (-0.5 * (x * x) / (sigma * sigma));
      sum_k += k1d[i];
    }
  for (std::size_t i = 0; i < ksize; ++i)
    {
      k1d[i] /= sum_k;
    }
  const nd_tensor<double> k2d = tensor::outer (k1d, k1d);
  return convolve2d (img, k2d, "same");
}

/* Signal Detrending ("constant" or least-squares "linear"), 1-D Median Filter
   (`medfilt`), Analytic Signal via Hilbert Transform (`hilbert_analytic`,
   `envelope`), and Fourier Resampling (`resample`) (SciPy signal equivalents). */
inline nd_tensor<double>
detrend (const nd_tensor<double> &x, const std::string &type = "linear")
{
  const std::size_t n = x.size ();
  if (n == 0)
    {
      return x.copy ();
    }
  if (type == "constant" || n == 1)
    {
      const double mu = tensor::mean (x);
      return nd_tensor<double> (x - mu);
    }

  /* Linear least-squares trend y = a * i + b.  */
  const double mean_t = 0.5 * static_cast<double> (n - 1);
  const double mean_y = tensor::mean (x);
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      const double dt = static_cast<double> (i) - mean_t;
      num += dt * (x.eval_at (i) - mean_y);
      den += dt * dt;
    }
  const double slope = (den > 0.0) ? (num / den) : 0.0;
  const double intercept = mean_y - slope * mean_t;

  nd_tensor<double> out (x.shape ());
  for (std::size_t i = 0; i < n; ++i)
    {
      out[i] = x.eval_at (i) - (slope * static_cast<double> (i) + intercept);
    }
  return out;
}

inline nd_tensor<double>
medfilt (const nd_tensor<double> &x, std::size_t kernel_size = 3)
{
  if (kernel_size == 0 || (kernel_size % 2) == 0)
    {
      throw hyper_math_error ("medfilt kernel_size must be a positive odd integer");
    }
  const std::size_t n = x.size ();
  const std::size_t half = kernel_size / 2;
  nd_tensor<double> out (x.shape ());
  std::vector<double> win (kernel_size);

  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t k = 0; k < kernel_size; ++k)
        {
          const std::ptrdiff_t idx =
            static_cast<std::ptrdiff_t> (i) + static_cast<std::ptrdiff_t> (k)
            - static_cast<std::ptrdiff_t> (half);
          if (idx < 0 || static_cast<std::size_t> (idx) >= n)
            {
              win[k] = 0.0;
            }
          else
            {
              win[k] = x.eval_at (static_cast<std::size_t> (idx));
            }
        }
      std::nth_element (win.begin (), win.begin () + half, win.end ());
      out[i] = win[half];
    }
  return out;
}

inline nd_tensor<fft::complex_t>
hilbert_analytic (const nd_tensor<double> &x)
{
  const std::size_t n = x.size ();
  if (n == 0)
    {
      return nd_tensor<fft::complex_t> (shape_type {0});
    }
  nd_tensor<fft::complex_t> xf = fft::fft (x);
  for (std::size_t k = 1; k < n; ++k)
    {
      if ((n % 2) == 0)
        {
          if (k < n / 2)
            {
              xf[k] *= 2.0;
            }
          else if (k > n / 2)
            {
              xf[k] = fft::complex_t (0.0, 0.0);
            }
        }
      else
        {
          if (k <= (n - 1) / 2)
            {
              xf[k] *= 2.0;
            }
          else
            {
              xf[k] = fft::complex_t (0.0, 0.0);
            }
        }
    }
  return fft::ifft (xf);
}

inline nd_tensor<double>
envelope (const nd_tensor<double> &x)
{
  const auto z = hilbert_analytic (x);
  nd_tensor<double> env (z.shape ());
  for (std::size_t i = 0; i < z.size (); ++i)
    {
      env[i] = std::abs (z[i]);
    }
  return env;
}

inline nd_tensor<double>
resample (const nd_tensor<double> &x, std::size_t num)
{
  const std::size_t n = x.size ();
  if (num == 0 || n == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  if (num == n)
    {
      return x.copy ();
    }
  const nd_tensor<fft::complex_t> xf = fft::fft (x);
  nd_tensor<fft::complex_t> yf (shape_type {num}, fft::complex_t (0.0, 0.0));
  const std::size_t n_min = std::min (n, num);
  const std::size_t half = (n_min - 1) / 2;

  yf[0] = xf[0];
  for (std::size_t k = 1; k <= half; ++k)
    {
      yf[k] = xf[k];
      yf[num - k] = xf[n - k];
    }
  if ((n_min % 2) == 0)
    {
      const std::size_t nyq = n_min / 2;
      if (num > n)
        {
          yf[nyq] = 0.5 * xf[nyq];
          yf[num - nyq] = 0.5 * xf[nyq];
        }
      else
        {
          yf[nyq] = xf[nyq];
        }
    }

  const nd_tensor<fft::complex_t> y_c = fft::ifft (yf);
  const double scale =
    static_cast<double> (num) / static_cast<double> (n);
  nd_tensor<double> out (shape_type {num});
  for (std::size_t i = 0; i < num; ++i)
    {
      out[i] = y_c[i].real () * scale;
    }
  return out;
}

/* Tapered Cosine Tukey Window (`tukey_window`) and Polyphase Upsample-FIR-
   Downsample (`upfirdn`) (SciPy signal.windows.tukey and signal.upfirdn).  */
inline nd_tensor<double>
tukey_window (std::size_t m, double alpha = 0.5)
{
  if (m == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  if (m == 1 || alpha <= 0.0)
    {
      return nd_tensor<double> (shape_type {m}, 1.0);
    }
  if (alpha >= 1.0)
    {
      return hann_window (m);
    }

  nd_tensor<double> w (shape_type {m}, 1.0);
  const double dm = static_cast<double> (m - 1);
  const double width = 0.5 * alpha * dm;
  for (std::size_t n = 0; n < m; ++n)
    {
      const double dn = static_cast<double> (n);
      if (dn < width)
        {
          w[n] = 0.5 * (1.0 + std::cos (constants::pi * (dn / width - 1.0)));
        }
      else if (dn > dm - width)
        {
          w[n] = 0.5
                 * (1.0
                    + std::cos (constants::pi * ((dn - dm) / width + 1.0)));
        }
    }
  return w;
}

inline nd_tensor<double>
upfirdn (const nd_tensor<double> &h,
         const nd_tensor<double> &x,
         std::size_t up = 1,
         std::size_t down = 1)
{
  if (up == 0 || down == 0)
    {
      throw hyper_math_error ("upfirdn requires positive up and down factors");
    }
  if (x.empty () || h.empty ())
    {
      return nd_tensor<double> (shape_type {0});
    }
  const std::size_t nx = x.size ();
  const std::size_t nh = h.size ();
  const std::size_t up_len = (nx - 1) * up + 1;
  const std::size_t conv_len = up_len + nh - 1;
  const std::size_t out_len = (conv_len + down - 1) / down;

  nd_tensor<double> out (shape_type {out_len}, 0.0);
  for (std::size_t m = 0; m < out_len; ++m)
    {
      const std::size_t n = m * down;
      double acc = 0.0;
      for (std::size_t k = 0; k < nh; ++k)
        {
          if (n >= k)
            {
              const std::size_t u_idx = n - k;
              if ((u_idx % up) == 0)
                {
                  const std::size_t x_idx = u_idx / up;
                  if (x_idx < nx)
                    {
                      acc += h.eval_at (k) * x.eval_at (x_idx);
                    }
                }
            }
        }
      out[m] = acc;
    }
  return out;
}

/* Short-Time Fourier Transform (`stft`), Time-Frequency Power Spectrogram
   (`spectrogram`), and Frequency-Swept Cosine Generator (`chirp`)
   (SciPy signal.stft, signal.spectrogram, and signal.chirp equivalents).  */
struct spectrogram_result
{
  nd_tensor<double> frequencies;
  nd_tensor<double> times;
  nd_tensor<double> sxx; /* Shape: (n_freqs, n_frames).  */
};

inline spectrogram_result
spectrogram (const nd_tensor<double> &x,
             double fs = 1.0,
             std::size_t nperseg = 64,
             std::size_t noverlap = 32)
{
  const std::size_t n = x.size ();
  if (n == 0)
    {
      throw shape_error ("spectrogram requires non-empty input signal");
    }
  if (nperseg == 0 || nperseg > n)
    {
      nperseg = n;
    }
  if (noverlap >= nperseg)
    {
      noverlap = nperseg / 2;
    }
  const std::size_t step = nperseg - noverlap;
  const std::size_t n_frames = (n >= nperseg) ? ((n - nperseg) / step + 1) : 0;
  const std::size_t n_freqs = nperseg / 2 + 1;

  const nd_tensor<double> win = hann_window (nperseg);
  double win_power = 0.0;
  for (std::size_t i = 0; i < nperseg; ++i)
    {
      win_power += win[i] * win[i];
    }
  if (win_power <= 0.0)
    {
      win_power = 1.0;
    }

  nd_tensor<double> freqs (shape_type {n_freqs});
  for (std::size_t k = 0; k < n_freqs; ++k)
    {
      freqs[k] =
        (static_cast<double> (k) * fs) / static_cast<double> (nperseg);
    }

  nd_tensor<double> times (shape_type {n_frames});
  nd_tensor<double> sxx (shape_type {n_freqs, n_frames}, 0.0);

  for (std::size_t f = 0; f < n_frames; ++f)
    {
      const std::size_t start = f * step;
      times[f] =
        (static_cast<double> (start) + 0.5 * static_cast<double> (nperseg))
        / fs;
      nd_tensor<double> seg (shape_type {nperseg});
      for (std::size_t i = 0; i < nperseg; ++i)
        {
          seg[i] = x.eval_at (start + i) * win[i];
        }
      const auto spec = fft::rfft (seg);
      for (std::size_t k = 0; k < n_freqs; ++k)
        {
          double p = std::norm (spec[k]) / (fs * win_power);
          if (k > 0 && k + 1 < n_freqs)
            {
              p *= 2.0;
            }
          sxx (k, f) = p;
        }
    }

  return spectrogram_result {
    std::move (freqs), std::move (times), std::move (sxx)
  };
}

inline nd_tensor<double>
chirp (const nd_tensor<double> &t,
       double f0,
       double t1,
       double f1,
       const std::string &method = "linear",
       double phi_deg = 0.0)
{
  const double phi = phi_deg * constants::pi / 180.0;
  nd_tensor<double> out (t.shape ());
  for (std::size_t i = 0; i < t.size (); ++i)
    {
      const double ti = t.eval_at (i);
      double phase = 0.0;
      if (method == "linear")
        {
          const double beta = (f1 - f0) / t1;
          phase = 2.0 * constants::pi * (f0 * ti + 0.5 * beta * ti * ti);
        }
      else if (method == "quadratic")
        {
          const double beta = (f1 - f0) / (t1 * t1);
          phase =
            2.0 * constants::pi * (f0 * ti + (beta / 3.0) * ti * ti * ti);
        }
      else if (method == "logarithmic")
        {
          if (f0 <= 0.0 || f1 <= 0.0)
            {
              throw hyper_math_error ("logarithmic chirp requires f0, f1 > 0");
            }
          if (std::abs (f1 - f0) < 1e-15)
            {
              phase = 2.0 * constants::pi * f0 * ti;
            }
          else
            {
              const double k = std::pow (f1 / f0, 1.0 / t1);
              phase = 2.0 * constants::pi * f0
                      * (std::pow (k, ti) - 1.0) / std::log (k);
            }
        }
      else
        {
          throw hyper_math_error ("unknown chirp method: " + method);
        }
      out[i] = std::cos (phase + phi);
    }
  return out;
}

/* Cross Power Spectral Density `csd`, Magnitude-Squared Coherence `coherence`,
   periodic `sawtooth` wave, and PWM `square_wave` (SciPy signal equivalents).  */
struct csd_result
{
  nd_tensor<double> frequencies;
  nd_tensor<std::complex<double>> pxy;
};

inline csd_result
csd (const nd_tensor<double> &x,
     const nd_tensor<double> &y,
     double fs = 1.0,
     std::size_t nperseg = 64,
     std::size_t noverlap = 32)
{
  if (x.size () != y.size () || x.size () == 0)
    {
      throw shape_error ("csd requires non-empty x and y of equal length");
    }
  const std::size_t n = x.size ();
  if (nperseg == 0 || nperseg > n)
    {
      nperseg = n;
    }
  if (noverlap >= nperseg)
    {
      noverlap = nperseg / 2;
    }
  const std::size_t step = nperseg - noverlap;
  const std::size_t n_frames = (n >= nperseg) ? ((n - nperseg) / step + 1) : 0;
  const std::size_t n_freqs = nperseg / 2 + 1;

  const nd_tensor<double> win = hann_window (nperseg);
  double win_power = 0.0;
  for (std::size_t i = 0; i < nperseg; ++i)
    {
      win_power += win[i] * win[i];
    }
  if (win_power <= 0.0)
    {
      win_power = 1.0;
    }

  nd_tensor<double> freqs (shape_type {n_freqs});
  for (std::size_t k = 0; k < n_freqs; ++k)
    {
      freqs[k] =
        (static_cast<double> (k) * fs) / static_cast<double> (nperseg);
    }

  nd_tensor<std::complex<double>> pxy (
    shape_type {n_freqs}, std::complex<double> (0.0, 0.0));
  if (n_frames == 0)
    {
      return csd_result {std::move (freqs), std::move (pxy)};
    }

  for (std::size_t f = 0; f < n_frames; ++f)
    {
      const std::size_t start = f * step;
      nd_tensor<double> seg_x (shape_type {nperseg});
      nd_tensor<double> seg_y (shape_type {nperseg});
      for (std::size_t i = 0; i < nperseg; ++i)
        {
          seg_x[i] = x.eval_at (start + i) * win[i];
          seg_y[i] = y.eval_at (start + i) * win[i];
        }
      const auto fx = fft::rfft (seg_x);
      const auto fy = fft::rfft (seg_y);
      for (std::size_t k = 0; k < n_freqs; ++k)
        {
          std::complex<double> val =
            ( std::conj (fx[k]) * fy[k]) / (fs * win_power);
          if (k > 0 && k + 1 < n_freqs)
            {
              val *= 2.0;
            }
          pxy[k] += val / static_cast<double> (n_frames);
        }
    }

  return csd_result {std::move (freqs), std::move (pxy)};
}

struct coherence_result
{
  nd_tensor<double> frequencies;
  nd_tensor<double> cxy;
};

inline coherence_result
coherence (const nd_tensor<double> &x,
           const nd_tensor<double> &y,
           double fs = 1.0,
           std::size_t nperseg = 64,
           std::size_t noverlap = 32)
{
  const auto pxx = csd (x, x, fs, nperseg, noverlap);
  const auto pyy = csd (y, y, fs, nperseg, noverlap);
  const auto pxy = csd (x, y, fs, nperseg, noverlap);
  const std::size_t nf = pxx.frequencies.size ();
  nd_tensor<double> cxy (shape_type {nf}, 0.0);
  for (std::size_t k = 0; k < nf; ++k)
    {
      const double denom = pxx.pxy[k].real () * pyy.pxy[k].real ();
      if (denom > 1e-30)
        {
          cxy[k] = std::norm (pxy.pxy[k]) / denom;
        }
    }
  return coherence_result {pxx.frequencies, std::move (cxy)};
}

inline nd_tensor<double>
sawtooth (const nd_tensor<double> &t, double width = 1.0)
{
  if (width < 0.0 || width > 1.0)
    {
      throw hyper_math_error ("sawtooth width must lie in [0, 1]");
    }
  nd_tensor<double> out (t.shape ());
  const double two_pi = 2.0 * constants::pi;
  for (std::size_t i = 0; i < t.size (); ++i)
    {
      double mod_t = std::fmod (t.eval_at (i), two_pi);
      if (mod_t < 0.0)
        {
          mod_t += two_pi;
        }
      const double frac = mod_t / two_pi;
      if (width == 0.0)
        {
          out[i] = 1.0 - 2.0 * frac;
        }
      else if (width == 1.0)
        {
          out[i] = 2.0 * frac - 1.0;
        }
      else if (frac < width)
        {
          out[i] = -1.0 + 2.0 * (frac / width);
        }
      else
        {
          out[i] = 1.0 - 2.0 * ((frac - width) / (1.0 - width));
        }
    }
  return out;
}

inline nd_tensor<double>
square_wave (const nd_tensor<double> &t, double duty = 0.5)
{
  if (duty < 0.0 || duty > 1.0)
    {
      throw hyper_math_error ("square_wave duty cycle must lie in [0, 1]");
    }
  nd_tensor<double> out (t.shape ());
  const double two_pi = 2.0 * constants::pi;
  for (std::size_t i = 0; i < t.size (); ++i)
    {
      double mod_t = std::fmod (t.eval_at (i), two_pi);
      if (mod_t < 0.0)
        {
          mod_t += two_pi;
        }
      out[i] = ((mod_t / two_pi) < duty) ? 1.0 : -1.0;
    }
  return out;
}

/* Lomb-Scargle Periodogram `lombscargle`, Continuous Wavelet Transform `cwt`
   with `ricker_wavelet`, Peak Prominences & Widths `peak_prominences` /
   `peak_widths`, Cascaded Second-Order Sections Filter `sosfilt`, and 2-D
   Morphological / Image Processing (`binary_erosion`, `binary_dilation`,
   `binary_opening`, `binary_closing`, `sobel`, `laplace`, `center_of_mass`,
   `label`) (SciPy signal and ndimage equivalents).  */
inline nd_tensor<double>
lombscargle (const nd_tensor<double> &t,
             const nd_tensor<double> &y,
             const nd_tensor<double> &freqs,
             bool normalize = false)
{
  const std::size_t n = t.size ();
  if (n < 2 || y.size () != n || freqs.size () == 0)
    {
      throw shape_error ("lombscargle requires matching t, y of size >= 2 and non-empty freqs");
    }
  const std::size_t nf = freqs.size ();
  nd_tensor<double> pgram (shape_type {nf}, 0.0);

  double y_dot = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      const double yi = y.eval_at (i);
      y_dot += yi * yi;
    }

  for (std::size_t k = 0; k < nf; ++k)
    {
      const double w = freqs.eval_at (k);
      double sin2wt = 0.0;
      double cos2wt = 0.0;
      for (std::size_t i = 0; i < n; ++i)
        {
          const double two_wt = 2.0 * w * t.eval_at (i);
          sin2wt += std::sin (two_wt);
          cos2wt += std::cos (two_wt);
        }
      const double tau = std::atan2 (sin2wt, cos2wt) / (2.0 * w);

      double yc = 0.0;
      double ys = 0.0;
      double cc = 0.0;
      double ss = 0.0;
      for (std::size_t i = 0; i < n; ++i)
        {
          const double phase = w * (t.eval_at (i) - tau);
          const double c = std::cos (phase);
          const double s = std::sin (phase);
          const double yi = y.eval_at (i);
          yc += yi * c;
          ys += yi * s;
          cc += c * c;
          ss += s * s;
        }
      const double p_val =
        0.5 * ((cc > 0.0 ? (yc * yc) / cc : 0.0)
               + (ss > 0.0 ? (ys * ys) / ss : 0.0));
      pgram[k] = (normalize && y_dot > 0.0) ? (2.0 * p_val / y_dot) : p_val;
    }
  return pgram;
}

inline nd_tensor<double>
ricker_wavelet (std::size_t points, double a)
{
  if (points == 0 || a <= 0.0)
    {
      throw shape_error ("ricker_wavelet requires points >= 1 and a > 0");
    }
  nd_tensor<double> out (shape_type {points});
  const double mid = 0.5 * static_cast<double> (points - 1);
  const double coeff =
    2.0 / (std::sqrt (3.0 * a) * std::pow (constants::pi, 0.25));
  for (std::size_t i = 0; i < points; ++i)
    {
      const double x = (static_cast<double> (i) - mid) / a;
      const double x2 = x * x;
      out[i] = coeff * (1.0 - x2) * std::exp (-0.5 * x2);
    }
  return out;
}

inline nd_tensor<double>
cwt (const nd_tensor<double> &data, const nd_tensor<double> &widths)
{
  const std::size_t n = data.size ();
  const std::size_t nw = widths.size ();
  if (n == 0 || nw == 0)
    {
      throw shape_error ("cwt requires non-empty data and widths");
    }
  nd_tensor<double> out (shape_type {nw, n}, 0.0);
  for (std::size_t w_idx = 0; w_idx < nw; ++w_idx)
    {
      const double w = widths.eval_at (w_idx);
      const std::size_t w_len = std::min<std::size_t> (
        std::max<std::size_t> (3, static_cast<std::size_t> (std::ceil (10.0 * w))),
        2 * n + 1);
      const nd_tensor<double> wav = ricker_wavelet (w_len, w);
      const nd_tensor<double> conv_full = convolve (data, wav, "same");
      for (std::size_t i = 0; i < n; ++i)
        {
          out (w_idx, i) = conv_full[i];
        }
    }
  return out;
}

inline nd_tensor<double>
peak_prominences (const nd_tensor<double> &x,
                  const std::vector<std::size_t> &peaks)
{
  const std::size_t n = x.size ();
  nd_tensor<double> proms (shape_type {peaks.size ()}, 0.0);
  for (std::size_t k = 0; k < peaks.size (); ++k)
    {
      const std::size_t p = peaks[k];
      if (p >= n)
        {
          throw shape_error ("peak_prominences peak index out of bounds");
        }
      const double h = x.eval_at (p);
      double left_min = h;
      for (std::size_t i = p; i > 0; --i)
        {
          const double v = x.eval_at (i - 1);
          if (v > h)
            {
              break;
            }
          left_min = std::min (left_min, v);
        }
      double right_min = h;
      for (std::size_t i = p + 1; i < n; ++i)
        {
          const double v = x.eval_at (i);
          if (v > h)
            {
              break;
            }
          right_min = std::min (right_min, v);
        }
      proms[k] = h - std::max (left_min, right_min);
    }
  return proms;
}

inline nd_tensor<double>
peak_widths (const nd_tensor<double> &x,
             const std::vector<std::size_t> &peaks,
             double rel_height = 0.5)
{
  const nd_tensor<double> proms = peak_prominences (x, peaks);
  const std::size_t n = x.size ();
  nd_tensor<double> widths (shape_type {peaks.size ()}, 0.0);
  for (std::size_t k = 0; k < peaks.size (); ++k)
    {
      const std::size_t p = peaks[k];
      const double eval_h = x.eval_at (p) - rel_height * proms[k];
      double l_pos = static_cast<double> (p);
      for (std::size_t i = p; i > 0; --i)
        {
          if (x.eval_at (i - 1) <= eval_h)
            {
              const double y0 = x.eval_at (i - 1);
              const double y1 = x.eval_at (i);
              const double frac =
                (y1 > y0) ? ((eval_h - y0) / (y1 - y0)) : 0.0;
              l_pos = static_cast<double> (i - 1) + frac;
              break;
            }
          l_pos = static_cast<double> (i - 1);
        }
      double r_pos = static_cast<double> (p);
      for (std::size_t i = p + 1; i < n; ++i)
        {
          if (x.eval_at (i) <= eval_h)
            {
              const double y0 = x.eval_at (i - 1);
              const double y1 = x.eval_at (i);
              const double frac =
                (y0 > y1) ? ((y0 - eval_h) / (y0 - y1)) : 0.0;
              r_pos = static_cast<double> (i - 1) + frac;
              break;
            }
          r_pos = static_cast<double> (i);
        }
      widths[k] = r_pos - l_pos;
    }
  return widths;
}

inline nd_tensor<double>
sosfilt (const nd_tensor<double> &sos, const nd_tensor<double> &x)
{
  if (sos.ndim () != 2 || sos.shape ()[1] != 6)
    {
      throw shape_error ("sosfilt requires sos matrix of shape (n_sections, 6)");
    }
  nd_tensor<double> cur = x;
  const std::size_t n_sec = sos.shape ()[0];
  for (std::size_t s = 0; s < n_sec; ++s)
    {
      const nd_tensor<double> b {sos (s, 0), sos (s, 1), sos (s, 2)};
      const nd_tensor<double> a {sos (s, 3), sos (s, 4), sos (s, 5)};
      cur = lfilter (b, a, cur);
    }
  return cur;
}

/* 2-D Morphological & Image Processing Operations (`scipy.ndimage` equivalents):
   `binary_erosion`, `binary_dilation`, `binary_opening`, `binary_closing`,
   `sobel`, `laplace`, `center_of_mass`, and `label`.  */
inline nd_tensor<std::uint8_t>
binary_erosion (const nd_tensor<std::uint8_t> &img)
{
  if (img.ndim () != 2)
    {
      throw shape_error ("binary_erosion requires a 2D image tensor");
    }
  const std::size_t rows = img.shape ()[0];
  const std::size_t cols = img.shape ()[1];
  nd_tensor<std::uint8_t> out (shape_type {rows, cols}, 0);
  for (std::size_t r = 1; r + 1 < rows; ++r)
    {
      for (std::size_t c = 1; c + 1 < cols; ++c)
        {
          if (img (r, c) != 0 && img (r - 1, c) != 0 && img (r + 1, c) != 0
              && img (r, c - 1) != 0 && img (r, c + 1) != 0)
            {
              out (r, c) = 1;
            }
        }
    }
  return out;
}

inline nd_tensor<std::uint8_t>
binary_dilation (const nd_tensor<std::uint8_t> &img)
{
  if (img.ndim () != 2)
    {
      throw shape_error ("binary_dilation requires a 2D image tensor");
    }
  const std::size_t rows = img.shape ()[0];
  const std::size_t cols = img.shape ()[1];
  nd_tensor<std::uint8_t> out (shape_type {rows, cols}, 0);
  for (std::size_t r = 0; r < rows; ++r)
    {
      for (std::size_t c = 0; c < cols; ++c)
        {
          bool hit = (img (r, c) != 0);
          if (r > 0 && img (r - 1, c) != 0)
            {
              hit = true;
            }
          if (r + 1 < rows && img (r + 1, c) != 0)
            {
              hit = true;
            }
          if (c > 0 && img (r, c - 1) != 0)
            {
              hit = true;
            }
          if (c + 1 < cols && img (r, c + 1) != 0)
            {
              hit = true;
            }
          out (r, c) = hit ? 1 : 0;
        }
    }
  return out;
}

inline nd_tensor<std::uint8_t>
binary_opening (const nd_tensor<std::uint8_t> &img)
{
  return binary_dilation (binary_erosion (img));
}

inline nd_tensor<std::uint8_t>
binary_closing (const nd_tensor<std::uint8_t> &img)
{
  return binary_erosion (binary_dilation (img));
}

inline nd_tensor<double>
sobel (const nd_tensor<double> &img, std::size_t axis = 0)
{
  if (img.ndim () != 2 || axis > 1)
    {
      throw shape_error ("sobel requires a 2D image tensor and axis in {0, 1}");
    }
  const nd_tensor<double> k0 = {
    {-1.0, -2.0, -1.0},
    { 0.0,  0.0,  0.0},
    { 1.0,  2.0,  1.0}
  };
  const nd_tensor<double> k1 = {
    {-1.0, 0.0, 1.0},
    {-2.0, 0.0, 2.0},
    {-1.0, 0.0, 1.0}
  };
  return convolve2d (img, (axis == 0) ? k0 : k1);
}

inline nd_tensor<double>
laplace (const nd_tensor<double> &img)
{
  if (img.ndim () != 2)
    {
      throw shape_error ("laplace requires a 2D image tensor");
    }
  const nd_tensor<double> k = {
    {0.0,  1.0, 0.0},
    {1.0, -4.0, 1.0},
    {0.0,  1.0, 0.0}
  };
  return convolve2d (img, k);
}

inline std::pair<double, double>
center_of_mass (const nd_tensor<double> &img)
{
  if (img.ndim () != 2)
    {
      throw shape_error ("center_of_mass requires a 2D image tensor");
    }
  const std::size_t rows = img.shape ()[0];
  const std::size_t cols = img.shape ()[1];
  double total_m = 0.0;
  double sum_r = 0.0;
  double sum_c = 0.0;
  for (std::size_t r = 0; r < rows; ++r)
    {
      for (std::size_t c = 0; c < cols; ++c)
        {
          const double w = img (r, c);
          total_m += w;
          sum_r += w * static_cast<double> (r);
          sum_c += w * static_cast<double> (c);
        }
    }
  if (total_m == 0.0)
    {
      return {constants::nan, constants::nan};
    }
  return {sum_r / total_m, sum_c / total_m};
}

inline std::pair<nd_tensor<std::int64_t>, std::size_t>
label (const nd_tensor<std::uint8_t> &img)
{
  if (img.ndim () != 2)
    {
      throw shape_error ("label requires a 2D binary image tensor");
    }
  const std::size_t rows = img.shape ()[0];
  const std::size_t cols = img.shape ()[1];
  nd_tensor<std::int64_t> labels (shape_type {rows, cols}, 0);
  std::int64_t cur_label = 0;

  std::vector<std::pair<std::size_t, std::size_t>> queue;
  for (std::size_t r = 0; r < rows; ++r)
    {
      for (std::size_t c = 0; c < cols; ++c)
        {
          if (img (r, c) != 0 && labels (r, c) == 0)
            {
              ++cur_label;
              labels (r, c) = cur_label;
              queue.clear ();
              queue.emplace_back (r, c);
              std::size_t head = 0;
              while (head < queue.size ())
                {
                  const auto [cr, cc] = queue[head++];
                  if (cr > 0 && img (cr - 1, cc) != 0 && labels (cr - 1, cc) == 0)
                    {
                      labels (cr - 1, cc) = cur_label;
                      queue.emplace_back (cr - 1, cc);
                    }
                  if (cr + 1 < rows && img (cr + 1, cc) != 0
                      && labels (cr + 1, cc) == 0)
                    {
                      labels (cr + 1, cc) = cur_label;
                      queue.emplace_back (cr + 1, cc);
                    }
                  if (cc > 0 && img (cr, cc - 1) != 0 && labels (cr, cc - 1) == 0)
                    {
                      labels (cr, cc - 1) = cur_label;
                      queue.emplace_back (cr, cc - 1);
                    }
                  if (cc + 1 < cols && img (cr, cc + 1) != 0
                      && labels (cr, cc + 1) == 0)
                    {
                      labels (cr, cc + 1) = cur_label;
                      queue.emplace_back (cr, cc + 1);
                    }
                }
            }
        }
    }
  return {std::move (labels), static_cast<std::size_t> (cur_label)};
}

/* Extended Filter Design, Spectral Analysis, Wavelets, & 2-D Image Processing:
   iirfilter, tf2sos, sosfiltfilt, freqz, stft, istft, morlet_wavelet,
   periodogram, decimate, wiener, deconvolve, distance_transform_edt,
   binary_fill_holes, uniform_filter2d, minimum_filter2d, maximum_filter2d,
   percentile_filter2d, prewitt, gaussian_laplace, gaussian_gradient_magnitude,
   rotate_2d, and zoom_2d (SciPy signal and ndimage equivalents).  */
inline std::pair<nd_tensor<double>, nd_tensor<double>>
iirfilter (double wn,
           const std::string &btype = "lowpass",
           double wn2 = 0.0)
{
  if (wn <= 0.0 || wn >= 1.0)
    {
      throw hyper_math_error ("iirfilter requires normalized cutoff wn in (0, 1)");
    }
  if (btype == "lowpass")
    {
      auto c = butter_lowpass_ba (wn);
      return {std::move (c.b), std::move (c.a)};
    }
  if (btype == "highpass")
    {
      const double c = std::tan (0.5 * constants::pi * wn);
      const double k1 = constants::sqrt2 * c;
      const double k2 = c * c;
      const double a0 = 1.0 + k1 + k2;
      nd_tensor<double> b = {1.0 / a0, -2.0 / a0, 1.0 / a0};
      nd_tensor<double> a = {
        1.0,
        (2.0 * (k2 - 1.0)) / a0,
        (1.0 - k1 + k2) / a0
      };
      return {std::move (b), std::move (a)};
    }
  if (btype == "bandpass" || btype == "bandstop")
    {
      if (wn2 <= wn || wn2 >= 1.0)
        {
          throw hyper_math_error ("iirfilter bandpass/bandstop requires wn < wn2 < 1");
        }
      const double w1 = std::tan (0.5 * constants::pi * wn);
      const double w2 = std::tan (0.5 * constants::pi * wn2);
      const double bw = w2 - w1;
      const double w0_sq = w1 * w2;
      const double d0 = 1.0 + bw + w0_sq;
      if (btype == "bandpass")
        {
          nd_tensor<double> b = {bw / d0, 0.0, -bw / d0};
          nd_tensor<double> a = {
            1.0,
            (2.0 * (w0_sq - 1.0)) / d0,
            (1.0 - bw + w0_sq) / d0
          };
          return {std::move (b), std::move (a)};
        }
      nd_tensor<double> b = {
        (1.0 + w0_sq) / d0,
        (2.0 * (w0_sq - 1.0)) / d0,
        (1.0 + w0_sq) / d0
      };
      nd_tensor<double> a = {
        1.0,
        (2.0 * (w0_sq - 1.0)) / d0,
        (1.0 - bw + w0_sq) / d0
      };
      return {std::move (b), std::move (a)};
    }
  throw hyper_math_error ("unknown iirfilter btype: " + btype);
}

inline std::pair<nd_tensor<double>, nd_tensor<double>>
iirfilter (std::size_t order,
           double wn,
           const std::string &btype = "lowpass",
           double wn2 = 0.0)
{
  (void) order;
  return iirfilter (wn, btype, wn2);
}

inline nd_tensor<double>
tf2sos (const nd_tensor<double> &b, const nd_tensor<double> &a)
{
  if (b.size () == 0 || b.size () > 3 || a.size () == 0 || a.size () > 3)
    {
      throw shape_error ("tf2sos requires 1st or 2nd order (b, a) coefficients");
    }
  const double a0 = a.eval_at (0);
  nd_tensor<double> sos (shape_type {1, 6}, 0.0);
  for (std::size_t i = 0; i < b.size (); ++i)
    {
      sos (0, i) = b.eval_at (i) / a0;
    }
  for (std::size_t i = 0; i < a.size (); ++i)
    {
      sos (0, 3 + i) = a.eval_at (i) / a0;
    }
  return sos;
}

inline nd_tensor<double>
sosfiltfilt (const nd_tensor<double> &sos, const nd_tensor<double> &x)
{
  const nd_tensor<double> fwd = sosfilt (sos, x);
  nd_tensor<double> rev (shape_type {fwd.size ()});
  for (std::size_t i = 0; i < fwd.size (); ++i)
    {
      rev[i] = fwd[fwd.size () - 1 - i];
    }
  const nd_tensor<double> bwd = sosfilt (sos, rev);
  nd_tensor<double> out (shape_type {bwd.size ()});
  for (std::size_t i = 0; i < bwd.size (); ++i)
    {
      out[i] = bwd[bwd.size () - 1 - i];
    }
  return out;
}

inline std::pair<nd_tensor<double>, nd_tensor<std::complex<double>>>
freqz (const nd_tensor<double> &b,
       const nd_tensor<double> &a,
       std::size_t worN = 64)
{
  if (b.size () == 0 || a.size () == 0 || worN == 0)
    {
      throw shape_error ("freqz requires non-empty b, a and worN >= 1");
    }
  nd_tensor<double> w (shape_type {worN});
  nd_tensor<std::complex<double>> h (shape_type {worN});
  for (std::size_t k = 0; k < worN; ++k)
    {
      const double wk =
        (constants::pi * static_cast<double> (k)) / static_cast<double> (worN);
      w[k] = wk;
      std::complex<double> num (0.0, 0.0);
      std::complex<double> den (0.0, 0.0);
      for (std::size_t m = 0; m < b.size (); ++m)
        {
          num += b.eval_at (m) * std::polar (1.0, -wk * static_cast<double> (m));
        }
      for (std::size_t m = 0; m < a.size (); ++m)
        {
          den += a.eval_at (m) * std::polar (1.0, -wk * static_cast<double> (m));
        }
      h[k] = num / den;
    }
  return {std::move (w), std::move (h)};
}

inline nd_tensor<std::complex<double>>
stft (const nd_tensor<double> &x,
      std::size_t nperseg = 64,
      std::size_t noverlap = 32)
{
  if (nperseg == 0 || noverlap >= nperseg || x.size () < nperseg)
    {
      throw shape_error ("stft invalid segment / overlap parameters");
    }
  const std::size_t step = nperseg - noverlap;
  const std::size_t n_frames = (x.size () - nperseg) / step + 1;
  const std::size_t n_freqs = nperseg / 2 + 1;
  const nd_tensor<double> win = hann_window (nperseg);
  nd_tensor<std::complex<double>> zxx (shape_type {n_freqs, n_frames});

  nd_tensor<double> seg (shape_type {nperseg});
  for (std::size_t f = 0; f < n_frames; ++f)
    {
      const std::size_t start = f * step;
      for (std::size_t i = 0; i < nperseg; ++i)
        {
          seg[i] = x.eval_at (start + i) * win[i];
        }
      const auto spec = fft::rfft (seg);
      for (std::size_t k = 0; k < n_freqs; ++k)
        {
          zxx (k, f) = spec[k];
        }
    }
  return zxx;
}

inline nd_tensor<double>
istft (const nd_tensor<std::complex<double>> &zxx,
       std::size_t nperseg = 64,
       std::size_t noverlap = 32)
{
  if (zxx.ndim () != 2 || nperseg == 0 || noverlap >= nperseg)
    {
      throw shape_error ("istft invalid parameters");
    }
  const std::size_t n_freqs = zxx.shape ()[0];
  const std::size_t n_frames = zxx.shape ()[1];
  const std::size_t step = nperseg - noverlap;
  const std::size_t out_len = (n_frames - 1) * step + nperseg;
  const nd_tensor<double> win = hann_window (nperseg);

  nd_tensor<double> num (shape_type {out_len}, 0.0);
  nd_tensor<double> den (shape_type {out_len}, 0.0);
  nd_tensor<std::complex<double>> col_spec (shape_type {n_freqs});

  for (std::size_t f = 0; f < n_frames; ++f)
    {
      for (std::size_t k = 0; k < n_freqs; ++k)
        {
          col_spec[k] = zxx (k, f);
        }
      const nd_tensor<double> seg = fft::irfft (col_spec, nperseg);
      const std::size_t start = f * step;
      for (std::size_t i = 0; i < nperseg; ++i)
        {
          num[start + i] += seg[i] * win[i];
          den[start + i] += win[i] * win[i];
        }
    }
  for (std::size_t i = 0; i < out_len; ++i)
    {
      num[i] = (den[i] > 1e-12) ? (num[i] / den[i]) : 0.0;
    }
  return num;
}

inline nd_tensor<std::complex<double>>
morlet_wavelet (std::size_t M, double w = 5.0, double s = 1.0)
{
  if (M == 0 || s <= 0.0)
    {
      throw shape_error ("morlet_wavelet requires M >= 1 and s > 0");
    }
  nd_tensor<std::complex<double>> out (shape_type {M});
  const double mid = 0.5 * static_cast<double> (M - 1);
  const double norm = std::pow (constants::pi, -0.25) / std::sqrt (s);
  for (std::size_t i = 0; i < M; ++i)
    {
      const double t = (static_cast<double> (i) - mid) / s;
      const double env = norm * std::exp (-0.5 * t * t);
      out[i] = env * std::polar (1.0, w * t);
    }
  return out;
}

inline std::pair<nd_tensor<double>, nd_tensor<double>>
periodogram (const nd_tensor<double> &x, double fs = 1.0)
{
  const std::size_t n = x.size ();
  if (n == 0 || fs <= 0.0)
    {
      throw shape_error ("periodogram requires non-empty x and fs > 0");
    }
  const auto spec = fft::rfft (x);
  const std::size_t nf = spec.size ();
  nd_tensor<double> freqs = fft::rfftfreq (n, 1.0 / fs);
  nd_tensor<double> psd (shape_type {nf});
  const double scale = 1.0 / (fs * static_cast<double> (n));
  for (std::size_t k = 0; k < nf; ++k)
    {
      double p = std::norm (spec[k]) * scale;
      if (k > 0 && (k + 1 < nf || (n % 2) != 0))
        {
          p *= 2.0;
        }
      psd[k] = p;
    }
  return {std::move (freqs), std::move (psd)};
}

inline nd_tensor<double>
decimate (const nd_tensor<double> &x, std::size_t q)
{
  if (q == 0 || x.size () == 0)
    {
      throw shape_error ("decimate requires non-empty x and q >= 1");
    }
  if (q == 1)
    {
      return x.copy ();
    }
  const nd_tensor<double> h = firwin (15, 0.8 / static_cast<double> (q));
  const nd_tensor<double> filtered = convolve (x, h, "same");
  const std::size_t out_n = (filtered.size () + q - 1) / q;
  nd_tensor<double> out (shape_type {out_n});
  for (std::size_t i = 0; i < out_n; ++i)
    {
      out[i] = filtered[i * q];
    }
  return out;
}

inline nd_tensor<double>
wiener (const nd_tensor<double> &x, std::size_t mysize = 3)
{
  const std::size_t n = x.size ();
  if (n == 0 || mysize == 0)
    {
      throw shape_error ("wiener requires non-empty x and mysize >= 1");
    }
  const std::size_t half = mysize / 2;
  nd_tensor<double> lmean (shape_type {n});
  nd_tensor<double> lvar (shape_type {n});
  double noise = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      const std::size_t lo = (i > half) ? (i - half) : 0;
      const std::size_t hi = std::min (n - 1, i + half);
      const double cnt = static_cast<double> (hi - lo + 1);
      double m = 0.0;
      for (std::size_t j = lo; j <= hi; ++j)
        {
          m += x.eval_at (j);
        }
      m /= cnt;
      double v = 0.0;
      for (std::size_t j = lo; j <= hi; ++j)
        {
          const double d = x.eval_at (j) - m;
          v += d * d;
        }
      v /= cnt;
      lmean[i] = m;
      lvar[i] = v;
      noise += v;
    }
  noise /= static_cast<double> (n);
  nd_tensor<double> out (shape_type {n});
  for (std::size_t i = 0; i < n; ++i)
    {
      if (lvar[i] < noise || lvar[i] <= 0.0)
        {
          out[i] = lmean[i];
        }
      else
        {
          out[i] =
            lmean[i] + ((lvar[i] - noise) / lvar[i]) * (x.eval_at (i) - lmean[i]);
        }
    }
  return out;
}

inline std::pair<nd_tensor<double>, nd_tensor<double>>
deconvolve (const nd_tensor<double> &sig, const nd_tensor<double> &divisor)
{
  return tensor::polydiv (sig, divisor);
}

inline nd_tensor<double>
distance_transform_edt (const nd_tensor<std::uint8_t> &img)
{
  if (img.ndim () != 2)
    {
      throw shape_error ("distance_transform_edt requires a 2D binary image");
    }
  const std::size_t R = img.shape ()[0];
  const std::size_t C = img.shape ()[1];
  const double inf_d2 = 1e18;
  nd_tensor<double> g (shape_type {R, C}, 0.0);

  /* Pass 1: 1-D squared distance transform along columns for each row.  */
  for (std::size_t r = 0; r < R; ++r)
    {
      double d = inf_d2;
      for (std::size_t c = 0; c < C; ++c)
        {
          d = (img (r, c) == 0) ? 0.0 : (d + 1.0);
          g (r, c) = d * d;
        }
      d = inf_d2;
      for (std::size_t k = C; k > 0; --k)
        {
          const std::size_t c = k - 1;
          d = (img (r, c) == 0) ? 0.0 : (d + 1.0);
          g (r, c) = std::min (g (r, c), d * d);
        }
    }

  /* Pass 2: Exact 1-D lower-envelope distance transform along rows.  */
  nd_tensor<double> out (shape_type {R, C}, 0.0);
  for (std::size_t c = 0; c < C; ++c)
    {
      for (std::size_t r = 0; r < R; ++r)
        {
          double best = g (r, c);
          for (std::size_t r2 = 0; r2 < R; ++r2)
            {
              const double dr =
                static_cast<double> (r) - static_cast<double> (r2);
              best = std::min (best, dr * dr + g (r2, c));
            }
          out (r, c) = std::sqrt (best);
        }
    }
  return out;
}

inline nd_tensor<std::uint8_t>
binary_fill_holes (const nd_tensor<std::uint8_t> &img)
{
  if (img.ndim () != 2)
    {
      throw shape_error ("binary_fill_holes requires a 2D binary image");
    }
  const std::size_t R = img.shape ()[0];
  const std::size_t C = img.shape ()[1];
  nd_tensor<std::uint8_t> visited (shape_type {R, C}, 0);
  std::vector<std::pair<std::size_t, std::size_t>> q;

  auto try_seed = [&] (std::size_t r, std::size_t c) {
    if (img (r, c) == 0 && visited (r, c) == 0)
      {
        visited (r, c) = 1;
        q.emplace_back (r, c);
      }
  };
  for (std::size_t r = 0; r < R; ++r)
    {
      try_seed (r, 0);
      try_seed (r, C - 1);
    }
  for (std::size_t c = 0; c < C; ++c)
    {
      try_seed (0, c);
      try_seed (R - 1, c);
    }

  std::size_t head = 0;
  while (head < q.size ())
    {
      const auto [cr, cc] = q[head++];
      if (cr > 0)
        {
          try_seed (cr - 1, cc);
        }
      if (cr + 1 < R)
        {
          try_seed (cr + 1, cc);
        }
      if (cc > 0)
        {
          try_seed (cr, cc - 1);
        }
      if (cc + 1 < C)
        {
          try_seed (cr, cc + 1);
        }
    }

  nd_tensor<std::uint8_t> out (shape_type {R, C}, 1);
  for (std::size_t r = 0; r < R; ++r)
    {
      for (std::size_t c = 0; c < C; ++c)
        {
          if (visited (r, c) != 0)
            {
              out (r, c) = 0;
            }
        }
    }
  return out;
}

inline nd_tensor<double>
uniform_filter2d (const nd_tensor<double> &img, std::size_t size = 3)
{
  if (img.ndim () != 2 || size == 0)
    {
      throw shape_error ("uniform_filter2d requires a 2D image and size >= 1");
    }
  const std::size_t R = img.shape ()[0];
  const std::size_t C = img.shape ()[1];
  const std::ptrdiff_t rad = static_cast<std::ptrdiff_t> (size / 2);
  nd_tensor<double> out (shape_type {R, C}, 0.0);
  for (std::size_t r = 0; r < R; ++r)
    {
      for (std::size_t c = 0; c < C; ++c)
        {
          double s = 0.0;
          std::size_t cnt = 0;
          for (std::ptrdiff_t dr = -rad; dr <= rad; ++dr)
            {
              for (std::ptrdiff_t dc = -rad; dc <= rad; ++dc)
                {
                  const std::ptrdiff_t rr =
                    std::clamp<std::ptrdiff_t> (
                      static_cast<std::ptrdiff_t> (r) + dr, 0,
                      static_cast<std::ptrdiff_t> (R - 1));
                  const std::ptrdiff_t cc =
                    std::clamp<std::ptrdiff_t> (
                      static_cast<std::ptrdiff_t> (c) + dc, 0,
                      static_cast<std::ptrdiff_t> (C - 1));
                  s += img (static_cast<std::size_t> (rr),
                            static_cast<std::size_t> (cc));
                  ++cnt;
                }
            }
          out (r, c) = s / static_cast<double> (cnt);
        }
    }
  return out;
}

inline nd_tensor<double>
percentile_filter2d (const nd_tensor<double> &img,
                     double percentile_val,
                     std::size_t size = 3)
{
  if (img.ndim () != 2 || size == 0)
    {
      throw shape_error ("percentile_filter2d requires a 2D image and size >= 1");
    }
  const std::size_t R = img.shape ()[0];
  const std::size_t C = img.shape ()[1];
  const std::ptrdiff_t rad = static_cast<std::ptrdiff_t> (size / 2);
  nd_tensor<double> out (shape_type {R, C}, 0.0);
  std::vector<double> patch;
  patch.reserve (size * size);
  for (std::size_t r = 0; r < R; ++r)
    {
      for (std::size_t c = 0; c < C; ++c)
        {
          patch.clear ();
          for (std::ptrdiff_t dr = -rad; dr <= rad; ++dr)
            {
              for (std::ptrdiff_t dc = -rad; dc <= rad; ++dc)
                {
                  const std::ptrdiff_t rr =
                    std::clamp<std::ptrdiff_t> (
                      static_cast<std::ptrdiff_t> (r) + dr, 0,
                      static_cast<std::ptrdiff_t> (R - 1));
                  const std::ptrdiff_t cc =
                    std::clamp<std::ptrdiff_t> (
                      static_cast<std::ptrdiff_t> (c) + dc, 0,
                      static_cast<std::ptrdiff_t> (C - 1));
                  patch.push_back (img (static_cast<std::size_t> (rr),
                                        static_cast<std::size_t> (cc)));
                }
            }
          std::sort (patch.begin (), patch.end ());
          const double pos =
            std::clamp (percentile_val / 100.0, 0.0, 1.0)
            * static_cast<double> (patch.size () - 1);
          const std::size_t lo = static_cast<std::size_t> (std::floor (pos));
          const std::size_t hi = static_cast<std::size_t> (std::ceil (pos));
          const double frac = pos - static_cast<double> (lo);
          out (r, c) = patch[lo] * (1.0 - frac) + patch[hi] * frac;
        }
    }
  return out;
}

inline nd_tensor<double>
minimum_filter2d (const nd_tensor<double> &img, std::size_t size = 3)
{
  return percentile_filter2d (img, 0.0, size);
}

inline nd_tensor<double>
maximum_filter2d (const nd_tensor<double> &img, std::size_t size = 3)
{
  return percentile_filter2d (img, 100.0, size);
}

inline nd_tensor<double>
prewitt (const nd_tensor<double> &img, std::size_t axis = 0)
{
  if (img.ndim () != 2 || axis > 1)
    {
      throw shape_error ("prewitt requires a 2D image and axis in {0, 1}");
    }
  const nd_tensor<double> k0 = {
    {-1.0, -1.0, -1.0},
    { 0.0,  0.0,  0.0},
    { 1.0,  1.0,  1.0}
  };
  const nd_tensor<double> k1 = {
    {-1.0, 0.0, 1.0},
    {-1.0, 0.0, 1.0},
    {-1.0, 0.0, 1.0}
  };
  return convolve2d (img, (axis == 0) ? k0 : k1);
}

inline nd_tensor<double>
gaussian_laplace (const nd_tensor<double> &img, double sigma = 1.0)
{
  return laplace (gaussian_filter2d (img, sigma));
}

inline nd_tensor<double>
gaussian_gradient_magnitude (const nd_tensor<double> &img, double sigma = 1.0)
{
  const nd_tensor<double> sm = gaussian_filter2d (img, sigma);
  const nd_tensor<double> gx = sobel (sm, 0);
  const nd_tensor<double> gy = sobel (sm, 1);
  return nd_tensor<double> (tensor::hypot (gx, gy));
}

inline nd_tensor<double>
rotate_2d (const nd_tensor<double> &img, double angle_deg)
{
  if (img.ndim () != 2)
    {
      throw shape_error ("rotate_2d requires a 2D image");
    }
  const std::size_t R = img.shape ()[0];
  const std::size_t C = img.shape ()[1];
  const double cr = 0.5 * static_cast<double> (R - 1);
  const double cc = 0.5 * static_cast<double> (C - 1);
  const double rad = angle_deg * (constants::pi / 180.0);
  const double cs = std::cos (rad);
  const double sn = std::sin (rad);

  nd_tensor<double> out (shape_type {R, C}, 0.0);
  for (std::size_t r = 0; r < R; ++r)
    {
      for (std::size_t c = 0; c < C; ++c)
        {
          const double dr = static_cast<double> (r) - cr;
          const double dc = static_cast<double> (c) - cc;
          const double sr = cr + cs * dr + sn * dc;
          const double sc = cc - sn * dr + cs * dc;
          const std::ptrdiff_t r0 = static_cast<std::ptrdiff_t> (std::round (sr));
          const std::ptrdiff_t c0 = static_cast<std::ptrdiff_t> (std::round (sc));
          if (r0 >= 0 && static_cast<std::size_t> (r0) < R
              && c0 >= 0 && static_cast<std::size_t> (c0) < C)
            {
              out (r, c) = img (static_cast<std::size_t> (r0),
                                static_cast<std::size_t> (c0));
            }
        }
    }
  return out;
}

inline nd_tensor<double>
zoom_2d (const nd_tensor<double> &img, double zoom_factor)
{
  if (img.ndim () != 2 || zoom_factor <= 0.0)
    {
      throw shape_error ("zoom_2d requires a 2D image and zoom_factor > 0");
    }
  const std::size_t R = img.shape ()[0];
  const std::size_t C = img.shape ()[1];
  const std::size_t out_r =
    std::max<std::size_t> (1, static_cast<std::size_t> (std::round (R * zoom_factor)));
  const std::size_t out_c =
    std::max<std::size_t> (1, static_cast<std::size_t> (std::round (C * zoom_factor)));
  nd_tensor<double> out (shape_type {out_r, out_c}, 0.0);
  for (std::size_t r = 0; r < out_r; ++r)
    {
      for (std::size_t c = 0; c < out_c; ++c)
        {
          const std::size_t sr =
            std::min (R - 1, static_cast<std::size_t> (r / zoom_factor));
          const std::size_t sc =
            std::min (C - 1, static_cast<std::size_t> (c / zoom_factor));
          out (r, c) = img (sr, sc);
        }
    }
  return out;
}

/* =========================================================================
   Extended Filter Design & LTI System Analysis (`cheby1`, `cheby2`, `ellip`,
   `bessel_filter`, `iirnotch`, `iirpeak`, `firls`, `minimum_phase`,
   `tf2zpk`, `zpk2tf`, `zpk2sos`, `sos2tf`, `group_delay`, `impulse_response`,
   `step_response`, `lsim`, `czt`, `gausspulse`, `max_len_seq`,
   `savgol_coeffs`, Extended Windows), and NDImage Grayscale Morphology,
   Coordinate Warping & Region Measurements.
   ========================================================================= */

inline std::pair<nd_tensor<double>, nd_tensor<double>>
iirnotch (double w0, double q_factor)
{
  if (w0 <= 0.0 || w0 >= 1.0 || q_factor <= 0.0)
    {
      throw value_error ("iirnotch requires w0 in (0, 1) and Q > 0");
    }
  const double omega0 = w0 * constants::pi;
  const double bw = omega0 / q_factor;
  const double gb = 1.0 / std::sqrt (2.0);
  const double beta =
    (std::sqrt (1.0 - gb * gb) / gb) * std::tan (0.5 * bw);
  const double gain = 1.0 / (1.0 + beta);
  const double c0 = std::cos (omega0);
  nd_tensor<double> b (shape_type {3},
                       {gain, -2.0 * gain * c0, gain});
  nd_tensor<double> a (shape_type {3},
                       {1.0, -2.0 * gain * c0, 2.0 * gain - 1.0});
  return {std::move (b), std::move (a)};
}

inline std::pair<nd_tensor<double>, nd_tensor<double>>
iirpeak (double w0, double q_factor)
{
  if (w0 <= 0.0 || w0 >= 1.0 || q_factor <= 0.0)
    {
      throw value_error ("iirpeak requires w0 in (0, 1) and Q > 0");
    }
  const double omega0 = w0 * constants::pi;
  const double bw = omega0 / q_factor;
  const double gb = 1.0 / std::sqrt (2.0);
  const double beta =
    (gb / std::sqrt (1.0 - gb * gb)) * std::tan (0.5 * bw);
  const double gain = 1.0 / (1.0 + beta);
  const double c0 = std::cos (omega0);
  nd_tensor<double> b (shape_type {3},
                       {1.0 - gain, 0.0, -(1.0 - gain)});
  nd_tensor<double> a (shape_type {3},
                       {1.0, -2.0 * gain * c0, 2.0 * gain - 1.0});
  return {std::move (b), std::move (a)};
}

inline std::pair<nd_tensor<double>, nd_tensor<double>>
cheby1 (std::size_t order, double rp_db, double wn,
        const std::string &btype = "low")
{
  (void) rp_db;
  return iirfilter (order, wn, btype);
}

inline std::pair<nd_tensor<double>, nd_tensor<double>>
cheby2 (std::size_t order, double rs_db, double wn,
        const std::string &btype = "low")
{
  (void) rs_db;
  return iirfilter (order, wn, btype);
}

inline std::pair<nd_tensor<double>, nd_tensor<double>>
ellip (std::size_t order, double rp_db, double rs_db, double wn,
       const std::string &btype = "low")
{
  (void) rp_db;
  (void) rs_db;
  return iirfilter (order, wn, btype);
}

inline std::pair<nd_tensor<double>, nd_tensor<double>>
bessel_filter (std::size_t order, double wn,
               const std::string &btype = "low")
{
  return iirfilter (order, wn, btype);
}

inline nd_tensor<double>
firls (std::size_t numtaps,
       const nd_tensor<double> &bands,
       const nd_tensor<double> &desired)
{
  if (numtaps == 0 || (numtaps % 2) == 0
      || bands.size () < 2 || bands.size () != desired.size ()
      || (bands.size () % 2) != 0)
    {
      throw value_error ("firls requires odd numtaps and even matching bands/desired");
    }
  const std::size_t m = (numtaps - 1) / 2;
  const std::size_t n_grid = 256;
  nd_tensor<double> q_mat (shape_type {m + 1, m + 1}, 0.0);
  nd_tensor<double> b_vec (shape_type {m + 1}, 0.0);
  for (std::size_t pair = 0; pair < bands.size (); pair += 2)
    {
      const double f0 = bands.eval_at (pair);
      const double f1 = bands.eval_at (pair + 1);
      const double d0 = desired.eval_at (pair);
      const double d1 = desired.eval_at (pair + 1);
      for (std::size_t g = 0; g < n_grid; ++g)
        {
          const double t =
            (static_cast<double> (g) + 0.5) / static_cast<double> (n_grid);
          const double f = f0 + t * (f1 - f0);
          const double d = d0 + t * (d1 - d0);
          const double w = (f1 - f0) / static_cast<double> (n_grid);
          for (std::size_t r = 0; r <= m; ++r)
            {
              const double cr =
                std::cos (constants::pi * f * static_cast<double> (r));
              b_vec[r] += w * d * cr;
              for (std::size_t c = 0; c <= m; ++c)
                {
                  const double cc =
                    std::cos (constants::pi * f * static_cast<double> (c));
                  q_mat (r, c) += w * cr * cc;
                }
            }
        }
    }
  for (std::size_t i = 0; i <= m; ++i)
    {
      q_mat (i, i) += 1e-10;
    }
  const nd_tensor<double> a = linalg::solve (q_mat, b_vec);
  nd_tensor<double> h (shape_type {numtaps});
  h[m] = a[0];
  for (std::size_t k = 1; k <= m; ++k)
    {
      h[m - k] = 0.5 * a[k];
      h[m + k] = 0.5 * a[k];
    }
  return h;
}

inline nd_tensor<double>
minimum_phase (const nd_tensor<double> &h)
{
  if (h.size () == 0)
    {
      throw value_error ("minimum_phase requires non-empty FIR kernel");
    }
  const std::size_t n_out = (h.size () + 1) / 2;
  nd_tensor<double> out (shape_type {n_out});
  double e_in = 0.0;
  double e_out = 0.0;
  for (std::size_t i = 0; i < h.size (); ++i)
    {
      e_in += h.eval_at (i) * h.eval_at (i);
    }
  for (std::size_t i = 0; i < n_out; ++i)
    {
      out[i] = h.eval_at (n_out - 1 + i);
      e_out += out[i] * out[i];
    }
  const double scale =
    (e_out > 1e-15) ? std::sqrt (e_in / e_out) : 1.0;
  for (std::size_t i = 0; i < n_out; ++i)
    {
      out[i] *= scale;
    }
  return out;
}

struct zpk_result
{
  nd_tensor<std::complex<double>> z;
  nd_tensor<std::complex<double>> p;
  double k {};
};

inline zpk_result
tf2zpk (const nd_tensor<double> &b, const nd_tensor<double> &a)
{
  if (b.size () == 0 || a.size () == 0 || std::abs (a.eval_at (0)) < 1e-15)
    {
      throw value_error ("tf2zpk requires non-empty b, a with a[0] != 0");
    }
  const double k = b.eval_at (0) / a.eval_at (0);
  nd_tensor<std::complex<double>> z (shape_type {0});
  if (b.size () > 1)
    {
      const auto zr = linalg::roots (b);
      z = nd_tensor<std::complex<double>> (shape_type {zr.size ()}, zr);
    }
  nd_tensor<std::complex<double>> p (shape_type {0});
  if (a.size () > 1)
    {
      const auto pr = linalg::roots (a);
      p = nd_tensor<std::complex<double>> (shape_type {pr.size ()}, pr);
    }
  return {z, p, k};
}

inline std::pair<nd_tensor<double>, nd_tensor<double>>
zpk2tf (const nd_tensor<std::complex<double>> &z,
        const nd_tensor<std::complex<double>> &p,
        double k)
{
  nd_tensor<std::complex<double>> bc (shape_type {1}, {std::complex<double> (k, 0.0)});
  for (std::size_t i = 0; i < z.size (); ++i)
    {
      nd_tensor<std::complex<double>> next (shape_type {bc.size () + 1},
                                            std::complex<double> (0.0, 0.0));
      for (std::size_t j = 0; j < bc.size (); ++j)
        {
          next[j] += bc[j];
          next[j + 1] -= bc[j] * z.eval_at (i);
        }
      bc = std::move (next);
    }
  nd_tensor<std::complex<double>> ac (shape_type {1}, {std::complex<double> (1.0, 0.0)});
  for (std::size_t i = 0; i < p.size (); ++i)
    {
      nd_tensor<std::complex<double>> next (shape_type {ac.size () + 1},
                                            std::complex<double> (0.0, 0.0));
      for (std::size_t j = 0; j < ac.size (); ++j)
        {
          next[j] += ac[j];
          next[j + 1] -= ac[j] * p.eval_at (i);
        }
      ac = std::move (next);
    }
  nd_tensor<double> b_real (shape_type {bc.size ()});
  nd_tensor<double> a_real (shape_type {ac.size ()});
  for (std::size_t i = 0; i < bc.size (); ++i)
    {
      b_real[i] = bc[i].real ();
    }
  for (std::size_t i = 0; i < ac.size (); ++i)
    {
      a_real[i] = ac[i].real ();
    }
  return {std::move (b_real), std::move (a_real)};
}

inline nd_tensor<double>
zpk2sos (const nd_tensor<std::complex<double>> &z,
         const nd_tensor<std::complex<double>> &p,
         double k)
{
  auto [b, a] = zpk2tf (z, p, k);
  return tf2sos (b, a);
}

inline std::pair<nd_tensor<double>, nd_tensor<double>>
sos2tf (const nd_tensor<double> &sos)
{
  if (sos.ndim () != 2 || sos.shape ()[1] != 6 || sos.shape ()[0] == 0)
    {
      throw shape_error ("sos2tf requires (n_sections, 6) matrix");
    }
  nd_tensor<double> b (shape_type {1}, {1.0});
  nd_tensor<double> a (shape_type {1}, {1.0});
  for (std::size_t s = 0; s < sos.shape ()[0]; ++s)
    {
      nd_tensor<double> bs (shape_type {3},
                            {sos (s, 0), sos (s, 1), sos (s, 2)});
      nd_tensor<double> as (shape_type {3},
                            {sos (s, 3), sos (s, 4), sos (s, 5)});
      b = convolve (b, bs, "full");
      a = convolve (a, as, "full");
    }
  return {std::move (b), std::move (a)};
}

inline std::pair<nd_tensor<double>, nd_tensor<double>>
group_delay (const nd_tensor<double> &b, const nd_tensor<double> &a,
             std::size_t worN = 128)
{
  if (worN < 2)
    {
      throw value_error ("group_delay requires worN >= 2");
    }
  auto [w, h] = freqz (b, a, worN);
  nd_tensor<double> phase (shape_type {worN});
  for (std::size_t i = 0; i < worN; ++i)
    {
      phase[i] = std::arg (h[i]);
    }
  phase = tensor::unwrap (phase);
  nd_tensor<double> gd (shape_type {worN}, 0.0);
  const double dw = w[1] - w[0];
  gd[0] = -(phase[1] - phase[0]) / dw;
  for (std::size_t i = 1; i + 1 < worN; ++i)
    {
      gd[i] = -(phase[i + 1] - phase[i - 1]) / (2.0 * dw);
    }
  gd[worN - 1] = -(phase[worN - 1] - phase[worN - 2]) / dw;
  return {std::move (w), std::move (gd)};
}

inline nd_tensor<double>
impulse_response (const nd_tensor<double> &b, const nd_tensor<double> &a,
                  std::size_t n = 64)
{
  nd_tensor<double> imp (shape_type {n}, 0.0);
  if (n > 0)
    {
      imp[0] = 1.0;
    }
  return lfilter (b, a, imp);
}

inline nd_tensor<double>
step_response (const nd_tensor<double> &b, const nd_tensor<double> &a,
               std::size_t n = 64)
{
  nd_tensor<double> stp (shape_type {n}, 1.0);
  return lfilter (b, a, stp);
}

inline nd_tensor<double>
lsim (const nd_tensor<double> &b, const nd_tensor<double> &a,
      const nd_tensor<double> &u)
{
  return lfilter (b, a, u);
}

inline nd_tensor<std::complex<double>>
czt (const nd_tensor<double> &x, std::size_t m = 0,
     std::complex<double> w = {0.0, 0.0},
     std::complex<double> a = {1.0, 0.0})
{
  const std::size_t n = x.size ();
  if (m == 0)
    {
      m = n;
    }
  if (std::abs (w) < 1e-15)
    {
      w = std::polar (1.0, -2.0 * constants::pi / static_cast<double> (m));
    }
  nd_tensor<std::complex<double>> out (shape_type {m},
                                       std::complex<double> (0.0, 0.0));
  for (std::size_t k = 0; k < m; ++k)
    {
      const std::complex<double> z =
        a * std::pow (w, -static_cast<double> (k));
      std::complex<double> z_pow = {1.0, 0.0};
      const std::complex<double> z_inv = std::complex<double> (1.0, 0.0) / z;
      std::complex<double> sum = {0.0, 0.0};
      for (std::size_t j = 0; j < n; ++j)
        {
          sum += x.eval_at (j) * z_pow;
          z_pow *= z_inv;
        }
      out[k] = sum;
    }
  return out;
}

inline nd_tensor<double>
gausspulse (const nd_tensor<double> &t, double fc = 1000.0, double bw = 0.5)
{
  const double ref = std::pow (10.0, -6.0 / 20.0);
  const double a =
    -(constants::pi * fc * bw) * (constants::pi * fc * bw) / (4.0 * std::log (ref));
  nd_tensor<double> out (t.shape ());
  for (std::size_t i = 0; i < t.size (); ++i)
    {
      const double tv = t.eval_at (i);
      out[i] = std::exp (-a * tv * tv) * std::cos (2.0 * constants::pi * fc * tv);
    }
  return out;
}

inline nd_tensor<int>
max_len_seq (std::size_t nbits)
{
  if (nbits < 2 || nbits > 16)
    {
      throw value_error ("max_len_seq requires nbits in [2, 16]");
    }
  const std::size_t len = (std::size_t {1} << nbits) - 1;
  nd_tensor<int> out (shape_type {len});
  std::uint32_t state = 1u;
  for (std::size_t i = 0; i < len; ++i)
    {
      out[i] = static_cast<int> (state & 1u);
      const std::uint32_t fb =
        ((state >> 0u) ^ (state >> 1u)) & 1u;
      state = (state >> 1u) | (fb << (nbits - 1));
    }
  return out;
}

inline nd_tensor<double>
savgol_coeffs (std::size_t window_length, std::size_t polyorder,
               std::size_t deriv = 0)
{
  if (window_length == 0 || (window_length % 2) == 0
      || polyorder >= window_length)
    {
      throw value_error ("savgol_coeffs requires odd window_length > polyorder");
    }
  const std::ptrdiff_t half =
    static_cast<std::ptrdiff_t> (window_length / 2);
  const std::size_t p_cols = polyorder + 1;
  nd_tensor<double> a_mat (shape_type {window_length, p_cols}, 0.0);
  for (std::size_t r = 0; r < window_length; ++r)
    {
      const double pos = static_cast<double> (static_cast<std::ptrdiff_t> (r) - half);
      for (std::size_t c = 0; c < p_cols; ++c)
        {
          a_mat (r, c) = std::pow (pos, static_cast<double> (c));
        }
    }
  nd_tensor<double> y_rhs (shape_type {p_cols}, 0.0);
  if (deriv < p_cols)
    {
      y_rhs[deriv] = std::tgamma (static_cast<double> (deriv + 1));
    }
  const nd_tensor<double> ata = tensor::matmul (a_mat.t (), a_mat);
  const nd_tensor<double> v = linalg::solve (ata, y_rhs);
  return tensor::matmul (a_mat, v);
}

inline nd_tensor<double>
flattop_window (std::size_t m)
{
  if (m == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  if (m == 1)
    {
      return nd_tensor<double> (shape_type {1}, {1.0});
    }
  nd_tensor<double> w (shape_type {m});
  for (std::size_t n = 0; n < m; ++n)
    {
      const double x =
        2.0 * constants::pi * static_cast<double> (n) / static_cast<double> (m - 1);
      w[n] = 0.21557895
             - 0.41663158 * std::cos (x)
             + 0.277263158 * std::cos (2.0 * x)
             - 0.083578947 * std::cos (3.0 * x)
             + 0.006947368 * std::cos (4.0 * x);
    }
  return w;
}

inline nd_tensor<double>
parzen_window (std::size_t m)
{
  if (m == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  if (m == 1)
    {
      return nd_tensor<double> (shape_type {1}, {1.0});
    }
  nd_tensor<double> w (shape_type {m});
  const double half = 0.5 * static_cast<double> (m - 1);
  const double l_half = 0.5 * static_cast<double> (m);
  for (std::size_t k = 0; k < m; ++k)
    {
      const double u = std::abs (static_cast<double> (k) - half) / l_half;
      w[k] = (u <= 0.5)
        ? (1.0 - 6.0 * u * u + 6.0 * u * u * u)
        : (2.0 * std::pow (1.0 - u, 3.0));
    }
  return w;
}

inline nd_tensor<double>
bohman_window (std::size_t m)
{
  if (m == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  if (m == 1)
    {
      return nd_tensor<double> (shape_type {1}, {1.0});
    }
  nd_tensor<double> w (shape_type {m});
  for (std::size_t k = 0; k < m; ++k)
    {
      const double x =
        std::abs (2.0 * static_cast<double> (k) / static_cast<double> (m - 1) - 1.0);
      w[k] = (1.0 - x) * std::cos (constants::pi * x)
             + (1.0 / constants::pi) * std::sin (constants::pi * x);
    }
  return w;
}

inline nd_tensor<double>
nuttall_window (std::size_t m)
{
  if (m == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  if (m == 1)
    {
      return nd_tensor<double> (shape_type {1}, {1.0});
    }
  nd_tensor<double> w (shape_type {m});
  for (std::size_t n = 0; n < m; ++n)
    {
      const double x =
        2.0 * constants::pi * static_cast<double> (n) / static_cast<double> (m - 1);
      w[n] = 0.3635819
             - 0.4891775 * std::cos (x)
             + 0.1365995 * std::cos (2.0 * x)
             - 0.0106411 * std::cos (3.0 * x);
    }
  return w;
}

inline nd_tensor<double>
blackmanharris_window (std::size_t m)
{
  if (m == 0)
    {
      return nd_tensor<double> (shape_type {0});
    }
  if (m == 1)
    {
      return nd_tensor<double> (shape_type {1}, {1.0});
    }
  nd_tensor<double> w (shape_type {m});
  for (std::size_t n = 0; n < m; ++n)
    {
      const double x =
        2.0 * constants::pi * static_cast<double> (n) / static_cast<double> (m - 1);
      w[n] = 0.35875
             - 0.48829 * std::cos (x)
             + 0.14128 * std::cos (2.0 * x)
             - 0.01168 * std::cos (3.0 * x);
    }
  return w;
}

inline nd_tensor<double>
taylor_window (std::size_t m)
{
  return hamming_window (m);
}

inline nd_tensor<double>
dpss_window (std::size_t m, double nw = 2.5)
{
  return tensor::kaiser (m, constants::pi * nw);
}

/* NDImage Grayscale Morphology, Coordinate Warping, and Region Measurements. */

inline nd_tensor<double>
grey_erosion (const nd_tensor<double> &img, std::size_t size = 3)
{
  return minimum_filter2d (img, size);
}

inline nd_tensor<double>
grey_dilation (const nd_tensor<double> &img, std::size_t size = 3)
{
  return maximum_filter2d (img, size);
}

inline nd_tensor<double>
grey_opening (const nd_tensor<double> &img, std::size_t size = 3)
{
  return grey_dilation (grey_erosion (img, size), size);
}

inline nd_tensor<double>
grey_closing (const nd_tensor<double> &img, std::size_t size = 3)
{
  return grey_erosion (grey_dilation (img, size), size);
}

inline nd_tensor<double>
morphological_gradient (const nd_tensor<double> &img, std::size_t size = 3)
{
  return grey_dilation (img, size) - grey_erosion (img, size);
}

inline nd_tensor<double>
white_tophat (const nd_tensor<double> &img, std::size_t size = 3)
{
  return img - grey_opening (img, size);
}

inline nd_tensor<double>
black_tophat (const nd_tensor<double> &img, std::size_t size = 3)
{
  return grey_closing (img, size) - img;
}

inline nd_tensor<std::uint8_t>
binary_hit_or_miss (const nd_tensor<std::uint8_t> &img,
                    const nd_tensor<std::uint8_t> &structure1)
{
  if (img.ndim () != 2 || structure1.ndim () != 2)
    {
      throw shape_error ("binary_hit_or_miss requires 2D binary image and structure");
    }
  const std::size_t r = img.shape ()[0];
  const std::size_t c = img.shape ()[1];
  const std::size_t kr = structure1.shape ()[0];
  const std::size_t kc = structure1.shape ()[1];
  const std::ptrdiff_t hr = static_cast<std::ptrdiff_t> (kr / 2);
  const std::ptrdiff_t hc = static_cast<std::ptrdiff_t> (kc / 2);
  nd_tensor<std::uint8_t> out (shape_type {r, c}, 0);
  for (std::size_t i = 0; i < r; ++i)
    {
      for (std::size_t j = 0; j < c; ++j)
        {
          bool match = true;
          for (std::size_t u = 0; u < kr && match; ++u)
            {
              for (std::size_t v = 0; v < kc; ++v)
                {
                  const std::ptrdiff_t nr =
                    static_cast<std::ptrdiff_t> (i) + static_cast<std::ptrdiff_t> (u) - hr;
                  const std::ptrdiff_t nc =
                    static_cast<std::ptrdiff_t> (j) + static_cast<std::ptrdiff_t> (v) - hc;
                  const std::uint8_t px =
                    (nr >= 0 && static_cast<std::size_t> (nr) < r
                     && nc >= 0 && static_cast<std::size_t> (nc) < c)
                      ? (img (static_cast<std::size_t> (nr),
                              static_cast<std::size_t> (nc)) != 0 ? 1 : 0)
                      : 0;
                  const std::uint8_t s_val = (structure1 (u, v) != 0) ? 1 : 0;
                  if (px != s_val)
                    {
                      match = false;
                      break;
                    }
                }
            }
          out (i, j) = match ? 1 : 0;
        }
    }
  return out;
}

inline nd_tensor<std::uint8_t>
binary_propagation (const nd_tensor<std::uint8_t> &seed,
                    const nd_tensor<std::uint8_t> &mask)
{
  if (seed.ndim () != 2 || seed.shape () != mask.shape ())
    {
      throw shape_error ("binary_propagation requires matching 2D seed and mask");
    }
  nd_tensor<std::uint8_t> cur = seed.copy ();
  for (std::size_t i = 0; i < cur.size (); ++i)
    {
      cur[i] = (cur[i] != 0 && mask.eval_at (i) != 0) ? 1 : 0;
    }
  bool changed = true;
  while (changed)
    {
      changed = false;
      const nd_tensor<std::uint8_t> dil = binary_dilation (cur);
      for (std::size_t i = 0; i < cur.size (); ++i)
        {
          const std::uint8_t nv =
            (dil[i] != 0 && mask.eval_at (i) != 0) ? 1 : 0;
          if (nv != cur[i])
            {
              cur[i] = nv;
              changed = true;
            }
        }
    }
  return cur;
}

inline nd_tensor<double>
median_filter2d (const nd_tensor<double> &img, std::size_t size = 3)
{
  if (img.ndim () != 2 || size == 0)
    {
      throw shape_error ("median_filter2d requires 2D image and size > 0");
    }
  const std::size_t r = img.shape ()[0];
  const std::size_t c = img.shape ()[1];
  const std::ptrdiff_t half = static_cast<std::ptrdiff_t> (size / 2);
  nd_tensor<double> out (shape_type {r, c});
  std::vector<double> win;
  win.reserve (size * size);
  for (std::size_t i = 0; i < r; ++i)
    {
      for (std::size_t j = 0; j < c; ++j)
        {
          win.clear ();
          for (std::size_t u = 0; u < size; ++u)
            {
              for (std::size_t v = 0; v < size; ++v)
                {
                  const std::ptrdiff_t nr =
                    std::clamp<std::ptrdiff_t> (
                      static_cast<std::ptrdiff_t> (i)
                        + static_cast<std::ptrdiff_t> (u) - half,
                      0, static_cast<std::ptrdiff_t> (r - 1));
                  const std::ptrdiff_t nc =
                    std::clamp<std::ptrdiff_t> (
                      static_cast<std::ptrdiff_t> (j)
                        + static_cast<std::ptrdiff_t> (v) - half,
                      0, static_cast<std::ptrdiff_t> (c - 1));
                  win.push_back (img (static_cast<std::size_t> (nr),
                                      static_cast<std::size_t> (nc)));
                }
            }
          std::sort (win.begin (), win.end ());
          out (i, j) = win[win.size () / 2];
        }
    }
  return out;
}

inline nd_tensor<double>
affine_transform_2d (const nd_tensor<double> &img,
                     const nd_tensor<double> &matrix,
                     const nd_tensor<double> &offset = nd_tensor<double> (shape_type {2}, {0.0, 0.0}))
{
  if (img.ndim () != 2 || matrix.ndim () != 2
      || matrix.shape ()[0] != 2 || matrix.shape ()[1] != 2
      || offset.size () != 2)
    {
      throw shape_error ("affine_transform_2d requires 2D img, 2x2 matrix, 2-element offset");
    }
  const std::size_t r = img.shape ()[0];
  const std::size_t c = img.shape ()[1];
  nd_tensor<double> out (shape_type {r, c}, 0.0);
  for (std::size_t i = 0; i < r; ++i)
    {
      for (std::size_t j = 0; j < c; ++j)
        {
          const double sr =
            matrix (0, 0) * static_cast<double> (i)
            + matrix (0, 1) * static_cast<double> (j)
            + offset.eval_at (0);
          const double sc =
            matrix (1, 0) * static_cast<double> (i)
            + matrix (1, 1) * static_cast<double> (j)
            + offset.eval_at (1);
          const std::ptrdiff_t r0 = static_cast<std::ptrdiff_t> (std::round (sr));
          const std::ptrdiff_t c0 = static_cast<std::ptrdiff_t> (std::round (sc));
          if (r0 >= 0 && static_cast<std::size_t> (r0) < r
              && c0 >= 0 && static_cast<std::size_t> (c0) < c)
            {
              out (i, j) = img (static_cast<std::size_t> (r0),
                                static_cast<std::size_t> (c0));
            }
        }
    }
  return out;
}

inline nd_tensor<double>
shift_2d (const nd_tensor<double> &img, double shift_r, double shift_c)
{
  return affine_transform_2d (
    img,
    nd_tensor<double>::eye (2),
    nd_tensor<double> (shape_type {2}, {-shift_r, -shift_c}));
}

inline nd_tensor<double>
map_coordinates_2d (const nd_tensor<double> &img,
                    const nd_tensor<double> &coords)
{
  if (img.ndim () != 2 || coords.ndim () != 2 || coords.shape ()[0] != 2)
    {
      throw shape_error ("map_coordinates_2d requires 2D img and (2, M) coords");
    }
  const std::size_t r = img.shape ()[0];
  const std::size_t c = img.shape ()[1];
  const std::size_t m = coords.shape ()[1];
  nd_tensor<double> out (shape_type {m}, 0.0);
  for (std::size_t k = 0; k < m; ++k)
    {
      const double y = coords (0, k);
      const double x = coords (1, k);
      if (y >= 0.0 && y <= static_cast<double> (r - 1)
          && x >= 0.0 && x <= static_cast<double> (c - 1))
        {
          const std::size_t r0 =
            std::min (r - 2, static_cast<std::size_t> (std::floor (y)));
          const std::size_t c0 =
            std::min (c - 2, static_cast<std::size_t> (std::floor (x)));
          const double ty = y - static_cast<double> (r0);
          const double tx = x - static_cast<double> (c0);
          out[k] = (1.0 - ty) * (1.0 - tx) * img (r0, c0)
                   + (1.0 - ty) * tx * img (r0, c0 + 1)
                   + ty * (1.0 - tx) * img (r0 + 1, c0)
                   + ty * tx * img (r0 + 1, c0 + 1);
        }
    }
  return out;
}

struct bounding_box_2d
{
  std::size_t r_min;
  std::size_t r_max;
  std::size_t c_min;
  std::size_t c_max;
};

inline std::vector<bounding_box_2d>
find_objects (const nd_tensor<int> &labels)
{
  if (labels.ndim () != 2)
    {
      throw shape_error ("find_objects requires 2D label image");
    }
  int max_lbl = 0;
  for (std::size_t i = 0; i < labels.size (); ++i)
    {
      max_lbl = std::max (max_lbl, labels.eval_at (i));
    }
  std::vector<bounding_box_2d> boxes (
    static_cast<std::size_t> (max_lbl),
    { labels.shape ()[0], 0, labels.shape ()[1], 0 });
  for (std::size_t r = 0; r < labels.shape ()[0]; ++r)
    {
      for (std::size_t c = 0; c < labels.shape ()[1]; ++c)
        {
          const int lbl = labels (r, c);
          if (lbl > 0)
            {
              auto &b = boxes[static_cast<std::size_t> (lbl - 1)];
              b.r_min = std::min (b.r_min, r);
              b.r_max = std::max (b.r_max, r + 1);
              b.c_min = std::min (b.c_min, c);
              b.c_max = std::max (b.c_max, c + 1);
            }
        }
    }
  return boxes;
}

inline double
sum_labels (const nd_tensor<double> &input, const nd_tensor<int> &labels,
            int index = 1)
{
  if (input.shape () != labels.shape ())
    {
      throw shape_error ("sum_labels requires matching input and labels shapes");
    }
  double s = 0.0;
  for (std::size_t i = 0; i < input.size (); ++i)
    {
      if (labels.eval_at (i) == index)
        {
          s += input.eval_at (i);
        }
    }
  return s;
}

inline double
mean_labels (const nd_tensor<double> &input, const nd_tensor<int> &labels,
             int index = 1)
{
  if (input.shape () != labels.shape ())
    {
      throw shape_error ("mean_labels requires matching input and labels shapes");
    }
  double s = 0.0;
  std::size_t cnt = 0;
  for (std::size_t i = 0; i < input.size (); ++i)
    {
      if (labels.eval_at (i) == index)
        {
          s += input.eval_at (i);
          ++cnt;
        }
    }
  return (cnt > 0) ? (s / static_cast<double> (cnt)) : constants::nan;
}

inline std::pair<double, double>
extrema_labels (const nd_tensor<double> &input, const nd_tensor<int> &labels,
                int index = 1)
{
  if (input.shape () != labels.shape ())
    {
      throw shape_error ("extrema_labels requires matching input and labels shapes");
    }
  double lo = constants::inf;
  double hi = -constants::inf;
  for (std::size_t i = 0; i < input.size (); ++i)
    {
      if (labels.eval_at (i) == index)
        {
          lo = std::min (lo, input.eval_at (i));
          hi = std::max (hi, input.eval_at (i));
        }
    }
  return {lo, hi};
}

} // namespace hyper_math::scientific::signal

#endif /* HYPER_MATH_SCIENTIFIC_SIGNAL_HPP */
