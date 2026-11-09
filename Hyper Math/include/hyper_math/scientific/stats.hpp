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

#ifndef HYPER_MATH_SCIENTIFIC_STATS_HPP
#define HYPER_MATH_SCIENTIFIC_STATS_HPP

#include "hyper_math/scientific/linalg.hpp"
#include "hyper_math/scientific/optimize.hpp"
#include "hyper_math/tensor/reductions.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace hyper_math::scientific::stats
{

using tensor::nd_tensor;
using tensor::shape_type;

/* Special mathematical functions (SciPy special equivalents).  */
namespace special
{

  inline double
  gamma (double x) noexcept
  {
    return std::tgamma (x);
  }

  inline double
  lgamma (double x) noexcept
  {
    return std::lgamma (x);
  }

  inline double
  beta (double a, double b) noexcept
  {
    return std::exp (std::lgamma (a) + std::lgamma (b) - std::lgamma (a + b));
  }

  inline double
  erf (double x) noexcept
  {
    return std::erf (x);
  }

  inline double
  erfc (double x) noexcept
  {
    return std::erfc (x);
  }

  /* Cylindrical Bessel functions of the first kind J_v(x), J_0(x), J_1(x),
     and orthogonal Legendre / Chebyshev polynomials (C++17/20 math +
     three-term recurrence).  */
  inline double
  jv (double v, double x)
  {
    return std::cyl_bessel_j (v, x);
  }

  inline double
  j0 (double x)
  {
    return std::cyl_bessel_j (0.0, x);
  }

  inline double
  j1 (double x)
  {
    return std::cyl_bessel_j (1.0, x);
  }

  inline double
  eval_legendre (std::size_t n, double x) noexcept
  {
    if (n == 0)
      {
        return 1.0;
      }
    if (n == 1)
      {
        return x;
      }
    double p_prev = 1.0;
    double p_cur = x;
    for (std::size_t k = 2; k <= n; ++k)
      {
        const double dk = static_cast<double> (k);
        const double p_next =
          ((2.0 * dk - 1.0) * x * p_cur - (dk - 1.0) * p_prev) / dk;
        p_prev = p_cur;
        p_cur = p_next;
      }
    return p_cur;
  }

  inline double
  eval_chebyt (std::size_t n, double x) noexcept
  {
    if (n == 0)
      {
        return 1.0;
      }
    if (n == 1)
      {
        return x;
      }
    double t_prev = 1.0;
    double t_cur = x;
    for (std::size_t k = 2; k <= n; ++k)
      {
        const double t_next = 2.0 * x * t_cur - t_prev;
        t_prev = t_cur;
        t_cur = t_next;
      }
    return t_cur;
  }

  /* Lower regularized incomplete gamma P(a, x) via power series and
     Lentz modified continued fraction.  */
  inline double
  gammainc (double a, double x)
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    if (x < a + 1.0)
      {
        double ap = a;
        double sum = 1.0 / a;
        double del = sum;
        for (int n = 0; n < 300; ++n)
          {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if (std::abs (del) < std::abs (sum) * 1e-14)
              {
                break;
              }
          }
        return sum * std::exp (-x + a * std::log (x) - std::lgamma (a));
      }
    else
      {
        /* Continued fraction for Q(a, x) = 1 - P(a, x).  */
        constexpr double tiny = 1e-30;
        double b = x + 1.0 - a;
        double c = 1.0 / tiny;
        double d = 1.0 / b;
        double h = d;
        for (int i = 1; i <= 300; ++i)
          {
            const double an = -static_cast<double> (i) * (static_cast<double> (i) - a);
            b += 2.0;
            d = an * d + b;
            if (std::abs (d) < tiny)
              {
                d = tiny;
              }
            c = b + an / c;
            if (std::abs (c) < tiny)
              {
                c = tiny;
              }
            d = 1.0 / d;
            const double del = d * c;
            h *= del;
            if (std::abs (del - 1.0) < 1e-14)
              {
                break;
              }
          }
        const double q =
          h * std::exp (-x + a * std::log (x) - std::lgamma (a));
        return 1.0 - q;
      }
  }

  /* Regularized incomplete beta I_x(a, b) via Lentz continued fraction.  */
  inline double
  betainc_cf (double a, double b, double x)
  {
    constexpr double tiny = 1e-30;
    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs (d) < tiny)
      {
        d = tiny;
      }
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= 300; ++m)
      {
        const double dm = static_cast<double> (m);
        const double m2 = 2.0 * dm;
        double aa = dm * (b - dm) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs (d) < tiny)
          {
            d = tiny;
          }
        c = 1.0 + aa / c;
        if (std::abs (c) < tiny)
          {
            c = tiny;
          }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + dm) * (qab + dm) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs (d) < tiny)
          {
            d = tiny;
          }
        c = 1.0 + aa / c;
        if (std::abs (c) < tiny)
          {
            c = tiny;
          }
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs (del - 1.0) < 1e-14)
          {
            break;
          }
      }
    return h;
  }

  inline double
  betainc (double a, double b, double x)
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    if (x >= 1.0)
      {
        return 1.0;
      }
    const double bt = std::exp (std::lgamma (a + b) - std::lgamma (a)
                                - std::lgamma (b) + a * std::log (x)
                                + b * std::log (1.0 - x));
    if (x < (a + 1.0) / (a + b + 2.0))
      {
        return bt * betainc_cf (a, b, x) / a;
      }
    else
      {
        return 1.0 - bt * betainc_cf (b, a, 1.0 - x) / b;
      }
  }

  /* Digamma function \psi(x) = d/dx \ln \Gamma(x) via recurrence shift to
     x >= 6 and Bernoulli asymptotic expansion (SciPy special.digamma).  */
  inline double
  digamma (double x) noexcept
  {
    if (x <= 0.0 && std::floor (x) == x)
      {
        return constants::nan;
      }
    double result = 0.0;
    if (x < 0.0)
      {
        /* Reflection formula: \psi(1 - x) - \psi(x) = \pi \cot(\pi x).  */
        result =
          digamma (1.0 - x) - constants::pi / std::tan (constants::pi * x);
        return result;
      }
    while (x < 10.0)
      {
        result -= 1.0 / x;
        x += 1.0;
      }
    const double inv_x = 1.0 / x;
    const double inv_x2 = inv_x * inv_x;
    result +=
      std::log (x) - 0.5 * inv_x
      - inv_x2
          * ((1.0 / 12.0)
             - inv_x2
                 * ((1.0 / 120.0)
                    - inv_x2
                        * ((1.0 / 252.0)
                           - inv_x2 * ((1.0 / 240.0) - inv_x2 / 132.0))));
    return result;
  }

  /* Inverse error function erfinv(y) for y in (-1, 1) via Winitzki rational
     approximation followed by two Halley-Newton refinement steps
     (SciPy special.erfinv).  */
  inline double
  erfinv (double y) noexcept
  {
    if (y <= -1.0 || y >= 1.0)
      {
        if (y == -1.0)
          {
            return -constants::inf;
          }
        if (y == 1.0)
          {
            return constants::inf;
          }
        return constants::nan;
      }
    if (y == 0.0)
      {
        return 0.0;
      }
    const double a = 0.147;
    const double ln_term = std::log (1.0 - y * y);
    const double t1 = 2.0 / (constants::pi * a) + 0.5 * ln_term;
    const double t2 = ln_term / a;
    double x = std::copysign (std::sqrt (std::sqrt (t1 * t1 - t2) - t1), y);

    /* Two Halley-Newton refinement steps: f(x) = erf(x) - y.  */
    const double two_over_sqrt_pi = 2.0 / std::sqrt (constants::pi);
    for (int step = 0; step < 3; ++step)
      {
        const double err = std::erf (x) - y;
        const double deriv = two_over_sqrt_pi * std::exp (-x * x);
        x -= err / (deriv + x * err);
      }
    return x;
  }

  /* Generalized Binomial Coefficient \binom{n}{k} (SciPy special.binom).  */
  inline double
  binom (double n, double k) noexcept
  {
    if (k < 0.0 || k > n)
      {
        return 0.0;
      }
    return std::exp (std::lgamma (n + 1.0) - std::lgamma (k + 1.0)
                     - std::lgamma (n - k + 1.0));
  }

  /* Log-odds `logit(p)`, Logistic sigmoid `expit(x)`, Numerically stable
     `logsumexp(x)` & `softmax(x)`, Hurwitz/Riemann Zeta `zeta(s, q)`,
     Principal-branch Lambert W `lambertw(z)`, and Complete Elliptic Integrals
     `ellipk(m)` & `ellipe(m)` via Gauss's AGM (SciPy special equivalents).  */
  inline double
  logit (double p) noexcept
  {
    if (p <= 0.0 || p >= 1.0)
      {
        if (p == 0.0)
          {
            return -constants::inf;
          }
        if (p == 1.0)
          {
            return constants::inf;
          }
        return constants::nan;
      }
    return std::log (p / (1.0 - p));
  }

  inline double
  expit (double x) noexcept
  {
    if (x >= 0.0)
      {
        return 1.0 / (1.0 + std::exp (-x));
      }
    const double ex = std::exp (x);
    return ex / (1.0 + ex);
  }

  inline double
  logsumexp (const nd_tensor<double> &x)
  {
    if (x.size () == 0)
      {
        throw shape_error ("logsumexp requires non-empty tensor");
      }
    const double m = tensor::max (x);
    if (!std::isfinite (m))
      {
        return m;
      }
    double sum_exp = 0.0;
    for (std::size_t i = 0; i < x.size (); ++i)
      {
        sum_exp += std::exp (x.eval_at (i) - m);
      }
    return m + std::log (sum_exp);
  }

  inline nd_tensor<double>
  softmax (const nd_tensor<double> &x)
  {
    if (x.size () == 0)
      {
        throw shape_error ("softmax requires non-empty tensor");
      }
    const double lse = logsumexp (x);
    nd_tensor<double> out (x.shape ());
    for (std::size_t i = 0; i < x.size (); ++i)
      {
        out[i] = std::exp (x.eval_at (i) - lse);
      }
    return out;
  }

  inline double
  zeta (double s, double q = 1.0) noexcept
  {
    if (s <= 1.0 || q <= 0.0)
      {
        if (s == 1.0)
          {
            return constants::inf;
          }
        return constants::nan;
      }
    /* Euler-Maclaurin summation with N = 12 terms + Bernoulli correction.  */
    constexpr int n_terms = 12;
    double sum = 0.0;
    for (int k = 0; k < n_terms; ++k)
      {
        sum += std::pow (q + static_cast<double> (k), -s);
      }
    const double a = q + static_cast<double> (n_terms);
    const double a_pow = std::pow (a, -s);
    const double inv_a = 1.0 / a;
    const double inv_a2 = inv_a * inv_a;

    sum += (a * a_pow) / (s - 1.0) + 0.5 * a_pow;
    const double c1 = (s / 12.0) * inv_a;
    const double c2 = (s * (s + 1.0) * (s + 2.0) / 720.0) * inv_a * inv_a2;
    const double c3 =
      (s * (s + 1.0) * (s + 2.0) * (s + 3.0) * (s + 4.0) / 30240.0)
      * inv_a * inv_a2 * inv_a2;
    return sum + a_pow * (c1 - c2 + c3);
  }

  inline double
  lambertw (double z) noexcept
  {
    constexpr double neg_inv_e = -0.36787944117144232160;
    if (z < neg_inv_e - 1e-15)
      {
        return constants::nan;
      }
    if (std::abs (z - neg_inv_e) <= 1e-15)
      {
        return -1.0;
      }
    if (z == 0.0)
      {
        return 0.0;
      }
    /* Initial guess: series near 0 or asymptotic log(z) - log(log(z)) for z >= 3. */
    double w = 0.0;
    if (z < -0.2)
      {
        const double p = std::sqrt (2.0 * (constants::e * z + 1.0));
        w = -1.0 + p - (p * p) / 3.0;
      }
    else if (z < 3.0)
      {
        w = std::log1p (z);
      }
    else
      {
        const double lz = std::log (z);
        w = lz - std::log (lz);
      }

    /* Halley's third-order iteration for w * exp(w) - z = 0.  */
    for (int iter = 0; iter < 50; ++iter)
      {
        const double ew = std::exp (w);
        const double wew = w * ew;
        const double f = wew - z;
        const double denom =
          ew * (w + 1.0) - ((w + 2.0) * f) / (2.0 * w + 2.0);
        const double dw = f / denom;
        w -= dw;
        if (std::abs (dw) < 1e-15 * std::max (1.0, std::abs (w)))
          {
            break;
          }
      }
    return w;
  }

  inline double
  ellipk (double m) noexcept
  {
    if (m < 0.0 || m >= 1.0)
      {
        return (m == 1.0) ? constants::inf : constants::nan;
      }
    double a = 1.0;
    double b = std::sqrt (1.0 - m);
    for (int iter = 0; iter < 40; ++iter)
      {
        const double a_next = 0.5 * (a + b);
        const double b_next = std::sqrt (a * b);
        if (std::abs (a_next - b_next) < 1e-15 * a_next)
          {
            a = a_next;
            break;
          }
        a = a_next;
        b = b_next;
      }
    return constants::half_pi / a;
  }

  inline double
  ellipe (double m) noexcept
  {
    if (m < 0.0 || m > 1.0)
      {
        return constants::nan;
      }
    if (m == 1.0)
      {
        return 1.0;
      }
    double a = 1.0;
    double b = std::sqrt (1.0 - m);
    double sum_c2 = 0.5 * m;
    double pow2 = 1.0;
    for (int iter = 0; iter < 40; ++iter)
      {
        const double a_next = 0.5 * (a + b);
        const double b_next = std::sqrt (a * b);
        const double c = 0.5 * (a - b);
        pow2 *= 2.0;
        sum_c2 += pow2 * c * c;
        a = a_next;
        b = b_next;
        if (std::abs (c) < 1e-15 * a)
          {
            break;
          }
      }
    return (constants::half_pi / a) * (1.0 - sum_c2);
  }

} // namespace special

/* Continuous probability distributions: Normal, Student-t, Chi-square.  */
struct normal_dist
{
  double loc = 0.0;
  double scale = 1.0;

  double
  pdf (double x) const noexcept
  {
    const double z = (x - loc) / scale;
    return std::exp (-0.5 * z * z) / (scale * constants::sqrt_two_pi);
  }

  double
  cdf (double x) const noexcept
  {
    const double z = (x - loc) / (scale * constants::sqrt2);
    return 0.5 * (1.0 + std::erf (z));
  }

  /* Quantile / Percent Point Function via inverse error function erfinv.  */
  double
  ppf (double p) const
  {
    if (p <= 0.0 || p >= 1.0)
      {
        throw hyper_math_error ("normal_dist::ppf requires p in (0, 1)");
      }
    return loc + scale * constants::sqrt2 * special::erfinv (2.0 * p - 1.0);
  }
};

struct student_t_dist
{
  double df = 1.0;

  double
  pdf (double t) const noexcept
  {
    const double coeff =
      std::exp (std::lgamma (0.5 * (df + 1.0)) - std::lgamma (0.5 * df))
      / std::sqrt (df * constants::pi);
    return coeff * std::pow (1.0 + (t * t) / df, -0.5 * (df + 1.0));
  }

  double
  cdf (double t) const
  {
    const double x = df / (df + t * t);
    const double ib = 0.5 * special::betainc (0.5 * df, 0.5, x);
    return (t >= 0.0) ? (1.0 - ib) : ib;
  }
};

struct chi2_dist
{
  double df = 1.0;

  double
  pdf (double x) const noexcept
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    const double k2 = 0.5 * df;
    return std::exp ((k2 - 1.0) * std::log (x) - 0.5 * x
                     - k2 * constants::ln2 - std::lgamma (k2));
  }

  double
  cdf (double x) const
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    return special::gammainc (0.5 * df, 0.5 * x);
  }
};

/* Correlation & Statistical Hypothesis Tests.  */
struct correlation_result
{
  double statistic;
  double pvalue;
};

struct ttest_result
{
  double statistic;
  double pvalue;
  double df;
};

/* Pearson linear correlation coefficient and two-tailed p-value.  */
inline correlation_result
pearsonr (const nd_tensor<double> &x, const nd_tensor<double> &y)
{
  if (x.size () != y.size () || x.size () < 3)
    {
      throw shape_error ("pearsonr requires equal-sized tensors with n >= 3");
    }
  const std::size_t n = x.size ();
  const double mx = tensor::mean (x);
  const double my = tensor::mean (y);

  double cov = 0.0;
  double vx = 0.0;
  double vy = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      const double dx = x.eval_at (i) - mx;
      const double dy = y.eval_at (i) - my;
      cov += dx * dy;
      vx += dx * dx;
      vy += dy * dy;
    }

  const double r = std::clamp (cov / std::sqrt (vx * vy), -1.0, 1.0);
  if (std::abs (r) >= 1.0 - 1e-15)
    {
      return correlation_result {r, 0.0};
    }

  const double df = static_cast<double> (n) - 2.0;
  const double t_stat = r * std::sqrt (df / (1.0 - r * r));
  const student_t_dist dist {df};
  const double pval = 2.0 * (1.0 - dist.cdf (std::abs (t_stat)));
  return correlation_result {r, pval};
}

/* Spearman rank-order correlation coefficient.  */
inline correlation_result
spearmanr (const nd_tensor<double> &x, const nd_tensor<double> &y)
{
  auto compute_ranks = [] (const nd_tensor<double> &v) {
    const std::size_t n = v.size ();
    const auto order = tensor::argsort (v);
    nd_tensor<double> ranks (shape_type {n});
    std::size_t i = 0;
    while (i < n)
      {
        std::size_t j = i;
        while (j + 1 < n && v.eval_at (order[j + 1]) == v.eval_at (order[i]))
          {
            ++j;
          }
        const double avg_rank = 0.5 * static_cast<double> (i + j) + 1.0;
        for (std::size_t k = i; k <= j; ++k)
          {
            ranks[order[k]] = avg_rank;
          }
        i = j + 1;
      }
    return ranks;
  };

  return pearsonr (compute_ranks (x), compute_ranks (y));
}

/* One-sample Student t-test for population mean.  */
inline ttest_result
ttest_1samp (const nd_tensor<double> &a, double popmean = 0.0)
{
  const std::size_t n = a.size ();
  if (n < 2)
    {
      throw shape_error ("ttest_1samp requires at least 2 observations");
    }
  const double m = tensor::mean (a);
  const double s = tensor::sample_std_dev (a, 1);
  const double df = static_cast<double> (n - 1);
  const double t_stat = (m - popmean) / (s / std::sqrt (static_cast<double> (n)));
  const student_t_dist dist {df};
  const double pval = 2.0 * (1.0 - dist.cdf (std::abs (t_stat)));
  return ttest_result {t_stat, pval, df};
}

/* Two-sample Student / Welch t-test for independent samples.  */
inline ttest_result
ttest_ind (const nd_tensor<double> &a,
           const nd_tensor<double> &b,
           bool equal_var = true)
{
  const std::size_t n1 = a.size ();
  const std::size_t n2 = b.size ();
  if (n1 < 2 || n2 < 2)
    {
      throw shape_error ("ttest_ind requires at least 2 samples per group");
    }

  const double m1 = tensor::mean (a);
  const double m2 = tensor::mean (b);
  const double v1 = tensor::sample_var (a, 1);
  const double v2 = tensor::sample_var (b, 1);
  const double dn1 = static_cast<double> (n1);
  const double dn2 = static_cast<double> (n2);

  double t_stat = 0.0;
  double df = 0.0;
  if (equal_var)
    {
      df = dn1 + dn2 - 2.0;
      const double sp2 = ((dn1 - 1.0) * v1 + (dn2 - 1.0) * v2) / df;
      t_stat = (m1 - m2) / std::sqrt (sp2 * (1.0 / dn1 + 1.0 / dn2));
    }
  else
    {
      const double se1 = v1 / dn1;
      const double se2 = v2 / dn2;
      t_stat = (m1 - m2) / std::sqrt (se1 + se2);
      const double num = (se1 + se2) * (se1 + se2);
      const double den =
        (se1 * se1) / (dn1 - 1.0) + (se2 * se2) / (dn2 - 1.0);
      df = num / den;
    }

  const student_t_dist dist {df};
  const double pval = 2.0 * (1.0 - dist.cdf (std::abs (t_stat)));
  return ttest_result {t_stat, pval, df};
}

/* Z-score standardization and higher moments (skewness, kurtosis).  */
inline nd_tensor<double>
zscore (const nd_tensor<double> &a, std::size_t ddof = 0)
{
  const double m = tensor::mean (a);
  const double s = tensor::sample_std_dev (a, ddof);
  return nd_tensor<double> ((a - m) / s);
}

inline double
skew (const nd_tensor<double> &a)
{
  const nd_tensor<double> z = zscore (a, 0);
  return tensor::mean (tensor::pow (z, 3.0));
}

inline double
kurtosis (const nd_tensor<double> &a, bool fisher = true)
{
  const nd_tensor<double> z = zscore (a, 0);
  const double m4 = tensor::mean (tensor::pow (z, 4.0));
  return fisher ? (m4 - 3.0) : m4;
}

/* Kolmogorov-Smirnov 1-sample and 2-sample goodness-of-fit tests,
   One-Way ANOVA F-test, Chi-Square test, and Mann-Whitney U test.  */
namespace detail
{

  inline double
  kolmogorov_sf (double z) noexcept
  {
    if (z <= 0.0)
      {
        return 1.0;
      }
    double sum = 0.0;
    for (int k = 1; k <= 100; ++k)
      {
        const double dk = static_cast<double> (k);
        const double term = std::exp (-2.0 * dk * dk * z * z);
        sum += ((k & 1) ? 1.0 : -1.0) * term;
        if (term < 1e-15)
          {
            break;
          }
      }
    return std::clamp (2.0 * sum, 0.0, 1.0);
  }

} // namespace detail

template <typename CdfFunc>
correlation_result
ks_1samp (const nd_tensor<double> &x, CdfFunc &&cdf_fn)
{
  const std::size_t n = x.size ();
  if (n == 0)
    {
      throw shape_error ("ks_1samp requires non-empty sample");
    }
  const nd_tensor<double> s = tensor::sort (x);
  const double dn = static_cast<double> (n);
  double d_max = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      const double f_val = cdf_fn (s[i]);
      const double d_plus =
        std::abs (static_cast<double> (i + 1) / dn - f_val);
      const double d_minus =
        std::abs (f_val - static_cast<double> (i) / dn);
      d_max = std::max ({d_max, d_plus, d_minus});
    }
  const double en = std::sqrt (dn);
  const double pval =
    detail::kolmogorov_sf ((en + 0.12 + 0.11 / en) * d_max);
  return correlation_result {d_max, pval};
}

inline correlation_result
ks_2samp (const nd_tensor<double> &data1, const nd_tensor<double> &data2)
{
  const std::size_t n1 = data1.size ();
  const std::size_t n2 = data2.size ();
  if (n1 == 0 || n2 == 0)
    {
      throw shape_error ("ks_2samp requires non-empty samples");
    }
  const nd_tensor<double> s1 = tensor::sort (data1);
  const nd_tensor<double> s2 = tensor::sort (data2);

  std::size_t i1 = 0;
  std::size_t i2 = 0;
  double d_max = 0.0;
  while (i1 < n1 && i2 < n2)
    {
      const double v = std::min (s1[i1], s2[i2]);
      while (i1 < n1 && s1[i1] <= v)
        {
          ++i1;
        }
      while (i2 < n2 && s2[i2] <= v)
        {
          ++i2;
        }
      const double cdf1 = static_cast<double> (i1) / static_cast<double> (n1);
      const double cdf2 = static_cast<double> (i2) / static_cast<double> (n2);
      d_max = std::max (d_max, std::abs (cdf1 - cdf2));
    }

  const double en = std::sqrt (static_cast<double> (n1 * n2)
                               / static_cast<double> (n1 + n2));
  const double pval =
    detail::kolmogorov_sf ((en + 0.12 + 0.11 / en) * d_max);
  return correlation_result {d_max, pval};
}

/* One-way Analysis of Variance (ANOVA) F-test (SciPy stats.f_oneway).  */
inline correlation_result
f_oneway (const std::vector<nd_tensor<double>> &groups)
{
  const std::size_t k = groups.size ();
  if (k < 2)
    {
      throw shape_error ("f_oneway requires at least 2 groups");
    }
  std::size_t total_n = 0;
  double grand_sum = 0.0;
  for (const auto &g : groups)
    {
      if (g.size () == 0)
        {
          throw shape_error ("f_oneway groups must be non-empty");
        }
      total_n += g.size ();
      grand_sum += tensor::sum (g);
    }
  if (total_n <= k)
    {
      throw shape_error ("f_oneway requires total sample size > number of groups");
    }
  const double grand_mean = grand_sum / static_cast<double> (total_n);

  double ss_between = 0.0;
  double ss_within = 0.0;
  for (const auto &g : groups)
    {
      const double gm = tensor::mean (g);
      const double diff = gm - grand_mean;
      ss_between += static_cast<double> (g.size ()) * diff * diff;
      for (std::size_t i = 0; i < g.size (); ++i)
        {
          const double d = g.eval_at (i) - gm;
          ss_within += d * d;
        }
    }

  const double df_between = static_cast<double> (k - 1);
  const double df_within = static_cast<double> (total_n - k);
  const double ms_between = ss_between / df_between;
  const double ms_within = ss_within / df_within;
  const double f_stat = ms_between / ms_within;

  /* Survival function of F(d1, d2) via regularized incomplete beta.  */
  const double x = df_within / (df_within + df_between * f_stat);
  const double pval =
    special::betainc (0.5 * df_within, 0.5 * df_between, x);
  return correlation_result {f_stat, pval};
}

/* Pearson Chi-Square goodness-of-fit test (SciPy stats.chisquare).  */
inline correlation_result
chisquare (const nd_tensor<double> &f_obs, const nd_tensor<double> &f_exp)
{
  if (f_obs.size () != f_exp.size () || f_obs.size () < 2)
    {
      throw shape_error ("chisquare requires matching arrays of size >= 2");
    }
  double stat = 0.0;
  for (std::size_t i = 0; i < f_obs.size (); ++i)
    {
      const double e = f_exp.eval_at (i);
      const double d = f_obs.eval_at (i) - e;
      stat += (d * d) / e;
    }
  const chi2_dist dist {static_cast<double> (f_obs.size () - 1)};
  return correlation_result {stat, 1.0 - dist.cdf (stat)};
}

/* Mann-Whitney U non-parametric rank-sum test (SciPy stats.mannwhitneyu). */
inline correlation_result
mannwhitneyu (const nd_tensor<double> &x, const nd_tensor<double> &y)
{
  const std::size_t n1 = x.size ();
  const std::size_t n2 = y.size ();
  if (n1 == 0 || n2 == 0)
    {
      throw shape_error ("mannwhitneyu requires non-empty inputs");
    }
  nd_tensor<double> combined = tensor::concatenate<double> ({x.ravel (), y.ravel ()});
  const auto order = tensor::argsort (combined);
  std::vector<double> ranks (combined.size ());
  std::size_t i = 0;
  while (i < combined.size ())
    {
      std::size_t j = i;
      while (j + 1 < combined.size ()
             && combined[order[j + 1]] == combined[order[i]])
        {
          ++j;
        }
      const double avg = 0.5 * static_cast<double> (i + j) + 1.0;
      for (std::size_t k = i; k <= j; ++k)
        {
          ranks[order[k]] = avg;
        }
      i = j + 1;
    }

  double r1 = 0.0;
  for (std::size_t k = 0; k < n1; ++k)
    {
      r1 += ranks[k];
    }
  const double dn1 = static_cast<double> (n1);
  const double dn2 = static_cast<double> (n2);
  const double u1 = r1 - 0.5 * dn1 * (dn1 + 1.0);
  const double mu_u = 0.5 * dn1 * dn2;
  const double sigma_u =
    std::sqrt (dn1 * dn2 * (dn1 + dn2 + 1.0) / 12.0);
  const double z = std::abs (u1 - mu_u) / sigma_u;
  const normal_dist norm01 {0.0, 1.0};
  const double pval = 2.0 * (1.0 - norm01.cdf (z));
  return correlation_result {u1, pval};
}

/* Additional Continuous & Discrete Probability Distributions:
   exponential_dist, gamma_dist, beta_dist, poisson_dist, binomial_dist.  */
struct exponential_dist
{
  double scale = 1.0;

  double
  pdf (double x) const noexcept
  {
    return (x < 0.0) ? 0.0 : (1.0 / scale) * std::exp (-x / scale);
  }

  double
  cdf (double x) const noexcept
  {
    return (x < 0.0) ? 0.0 : (1.0 - std::exp (-x / scale));
  }

  double
  ppf (double q) const
  {
    if (q < 0.0 || q >= 1.0)
      {
        throw hyper_math_error ("exponential_dist::ppf q must lie in [0, 1)");
      }
    return -scale * std::log (1.0 - q);
  }
};

struct gamma_dist
{
  double shape_k = 1.0;
  double scale_theta = 1.0;

  double
  pdf (double x) const noexcept
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    return std::exp ((shape_k - 1.0) * std::log (x) - (x / scale_theta)
                     - shape_k * std::log (scale_theta)
                     - special::lgamma (shape_k));
  }

  double
  cdf (double x) const noexcept
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    return special::gammainc (shape_k, x / scale_theta);
  }
};

struct beta_dist
{
  double alpha = 1.0;
  double beta_param = 1.0;

  double
  pdf (double x) const noexcept
  {
    if (x <= 0.0 || x >= 1.0)
      {
        return 0.0;
      }
    return std::pow (x, alpha - 1.0) * std::pow (1.0 - x, beta_param - 1.0)
           / special::beta (alpha, beta_param);
  }

  double
  cdf (double x) const noexcept
  {
    return special::betainc (alpha, beta_param, x);
  }
};

struct poisson_dist
{
  double mu = 1.0;

  double
  pmf (std::size_t k) const noexcept
  {
    if (mu <= 0.0)
      {
        return (k == 0 && mu == 0.0) ? 1.0 : 0.0;
      }
    return std::exp (static_cast<double> (k) * std::log (mu) - mu
                     - special::lgamma (static_cast<double> (k + 1)));
  }

  double
  cdf (std::size_t k) const noexcept
  {
    double acc = 0.0;
    for (std::size_t i = 0; i <= k; ++i)
      {
        acc += pmf (i);
      }
    return std::min (1.0, acc);
  }
};

struct binomial_dist
{
  std::size_t n = 1;
  double p = 0.5;

  double
  pmf (std::size_t k) const noexcept
  {
    if (k > n)
      {
        return 0.0;
      }
    if (p == 0.0)
      {
        return (k == 0) ? 1.0 : 0.0;
      }
    if (p == 1.0)
      {
        return (k == n) ? 1.0 : 0.0;
      }
    const double log_comb =
      special::lgamma (static_cast<double> (n + 1))
      - special::lgamma (static_cast<double> (k + 1))
      - special::lgamma (static_cast<double> (n - k + 1));
    return std::exp (log_comb + static_cast<double> (k) * std::log (p)
                     + static_cast<double> (n - k) * std::log (1.0 - p));
  }

  double
  cdf (std::size_t k) const noexcept
  {
    double acc = 0.0;
    for (std::size_t i = 0; i <= std::min (k, n); ++i)
      {
        acc += pmf (i);
      }
    return std::min (1.0, acc);
  }
};

/* 1-D Gaussian Kernel Density Estimator with Scott's Rule bandwidth
   (SciPy stats.gaussian_kde equivalent).  */
class gaussian_kde
{
private:
  nd_tensor<double> dataset_;
  double bandwidth_;

public:
  explicit
  gaussian_kde (const nd_tensor<double> &data, double bw_method = 0.0)
    : dataset_ (data.copy ()), bandwidth_ (1.0)
  {
    if (dataset_.size () < 2)
      {
        throw shape_error ("gaussian_kde requires at least 2 sample points");
      }
    const double s = tensor::sample_std_dev (dataset_, 1);
    /* Scott's factor: n^(-1/5) * std_dev.  */
    const double scott =
      std::pow (static_cast<double> (dataset_.size ()), -0.2);
    bandwidth_ = (bw_method > 0.0) ? (bw_method * s) : (scott * s);
    if (bandwidth_ <= 1e-15)
      {
        bandwidth_ = 1.0;
      }
  }

  double
  bandwidth () const noexcept
  {
    return bandwidth_;
  }

  double
  evaluate (double x) const noexcept
  {
    const std::size_t n = dataset_.size ();
    const double inv_h = 1.0 / bandwidth_;
    const double norm_factor =
      inv_h / (static_cast<double> (n) * std::sqrt (2.0 * constants::pi));
    double sum_k = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      {
        const double u = (x - dataset_[i]) * inv_h;
        sum_k += std::exp (-0.5 * u * u);
      }
    return norm_factor * sum_k;
  }

  nd_tensor<double>
  evaluate (const nd_tensor<double> &points) const
  {
    nd_tensor<double> out (points.shape ());
    for (std::size_t i = 0; i < points.size (); ++i)
      {
        out[i] = evaluate (points.eval_at (i));
      }
    return out;
  }
};

/* Principal Component Analysis (PCA) via SVD of mean-centered data matrix X
   of shape (n_samples, n_features).  */
struct pca_result
{
  nd_tensor<double> mean;
  nd_tensor<double> components;
  nd_tensor<double> explained_variance;
  nd_tensor<double> explained_variance_ratio;
  nd_tensor<double> transformed;
};

inline pca_result
pca (const nd_tensor<double> &x, std::size_t n_components = 0)
{
  if (x.ndim () != 2 || x.shape ()[0] < 2)
    {
      throw shape_error ("pca requires 2D matrix with at least 2 samples");
    }
  const std::size_t n_samples = x.shape ()[0];
  const std::size_t n_features = x.shape ()[1];
  const std::size_t max_k = std::min (n_samples, n_features);
  if (n_components == 0 || n_components > max_k)
    {
      n_components = max_k;
    }

  nd_tensor<double> mu = tensor::mean (x, 0);
  nd_tensor<double> centered (shape_type {n_samples, n_features});
  for (std::size_t i = 0; i < n_samples; ++i)
    {
      for (std::size_t j = 0; j < n_features; ++j)
        {
          centered (i, j) = x (i, j) - mu[j];
        }
    }

  const linalg::svd_result sv = linalg::svd (centered);
  nd_tensor<double> comps (shape_type {n_components, n_features});
  nd_tensor<double> exp_var (shape_type {n_components});
  double total_var = 0.0;
  for (std::size_t k = 0; k < max_k; ++k)
    {
      total_var +=
        (sv.s[k] * sv.s[k]) / static_cast<double> (n_samples - 1);
    }

  nd_tensor<double> exp_ratio (shape_type {n_components});
  for (std::size_t k = 0; k < n_components; ++k)
    {
      exp_var[k] =
        (sv.s[k] * sv.s[k]) / static_cast<double> (n_samples - 1);
      exp_ratio[k] = (total_var > 0.0) ? (exp_var[k] / total_var) : 0.0;
      for (std::size_t j = 0; j < n_features; ++j)
        {
          comps (k, j) = sv.vt (k, j);
        }
    }

  nd_tensor<double> proj = tensor::matmul (centered, comps.t ());
  return pca_result {
    std::move (mu),
    std::move (comps),
    std::move (exp_var),
    std::move (exp_ratio),
    std::move (proj)
  };
}

/* Shannon Entropy H(p) and Kullback-Leibler Divergence D_KL(p || q), plus
   IQR, Trimmed Mean, Geometric Mean, Harmonic Mean, Wilcoxon Signed-Rank,
   and Kruskal-Wallis H-test (SciPy stats equivalents).  */
inline double
entropy (const nd_tensor<double> &pk)
{
  double sum_p = 0.0;
  for (std::size_t i = 0; i < pk.size (); ++i)
    {
      sum_p += pk.eval_at (i);
    }
  if (sum_p <= 0.0)
    {
      return 0.0;
    }
  double h = 0.0;
  for (std::size_t i = 0; i < pk.size (); ++i)
    {
      const double p = pk.eval_at (i) / sum_p;
      if (p > 0.0)
        {
          h -= p * std::log (p);
        }
    }
  return h;
}

inline double
kl_divergence (const nd_tensor<double> &pk, const nd_tensor<double> &qk)
{
  if (pk.size () != qk.size ())
    {
      throw shape_error ("kl_divergence requires equal-length distributions");
    }
  double sum_p = tensor::sum (pk);
  double sum_q = tensor::sum (qk);
  double kl = 0.0;
  for (std::size_t i = 0; i < pk.size (); ++i)
    {
      const double p = pk.eval_at (i) / sum_p;
      const double q = qk.eval_at (i) / sum_q;
      if (p > 0.0)
        {
          if (q <= 0.0)
            {
              return constants::inf;
            }
          kl += p * std::log (p / q);
        }
    }
  return kl;
}

inline double
iqr (const nd_tensor<double> &x)
{
  return tensor::percentile (x, 75.0) - tensor::percentile (x, 25.0);
}

inline double
trim_mean (const nd_tensor<double> &x, double proportiontocut)
{
  if (proportiontocut < 0.0 || proportiontocut >= 0.5)
    {
      throw hyper_math_error ("trim_mean proportiontocut must be in [0, 0.5)");
    }
  const nd_tensor<double> s = tensor::sort (x);
  const std::size_t n = s.size ();
  const std::size_t cut =
    static_cast<std::size_t> (std::floor (proportiontocut * static_cast<double> (n)));
  if (2 * cut >= n)
    {
      return constants::nan;
    }
  double acc = 0.0;
  for (std::size_t i = cut; i < n - cut; ++i)
    {
      acc += s[i];
    }
  return acc / static_cast<double> (n - 2 * cut);
}

inline double
gmean (const nd_tensor<double> &x)
{
  if (x.empty ())
    {
      return constants::nan;
    }
  double sum_log = 0.0;
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      sum_log += std::log (x.eval_at (i));
    }
  return std::exp (sum_log / static_cast<double> (x.size ()));
}

inline double
hmean (const nd_tensor<double> &x)
{
  if (x.empty ())
    {
      return constants::nan;
    }
  double sum_inv = 0.0;
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      sum_inv += 1.0 / x.eval_at (i);
    }
  return static_cast<double> (x.size ()) / sum_inv;
}

/* Assign average ranks to data, handling ties (SciPy stats.rankdata).  */
inline nd_tensor<double>
rankdata (const nd_tensor<double> &v)
{
  const std::size_t n = v.size ();
  const auto order = tensor::argsort (v);
  nd_tensor<double> ranks (shape_type {n});
  std::size_t i = 0;
  while (i < n)
    {
      std::size_t j = i;
      while (j + 1 < n && v.eval_at (order[j + 1]) == v.eval_at (order[i]))
        {
          ++j;
        }
      const double avg_rank = 0.5 * static_cast<double> (i + j) + 1.0;
      for (std::size_t k = i; k <= j; ++k)
        {
          ranks[order[k]] = avg_rank;
        }
      i = j + 1;
    }
  return ranks;
}

/* Kruskal-Wallis H-test for independent samples (non-parametric one-way ANOVA,
   SciPy stats.kruskal equivalent).  */
inline correlation_result
kruskal (const std::vector<nd_tensor<double>> &groups)
{
  if (groups.size () < 2)
    {
      throw shape_error ("kruskal requires at least 2 sample groups");
    }
  std::vector<double> pooled;
  std::vector<std::size_t> group_id;
  for (std::size_t g = 0; g < groups.size (); ++g)
    {
      for (std::size_t i = 0; i < groups[g].size (); ++i)
        {
          pooled.push_back (groups[g].eval_at (i));
          group_id.push_back (g);
        }
    }
  const std::size_t n_total = pooled.size ();
  const nd_tensor<double> ranks =
    rankdata (nd_tensor<double> (shape_type {n_total}, pooled));
  std::vector<double> rank_sums (groups.size (), 0.0);
  for (std::size_t i = 0; i < n_total; ++i)
    {
      rank_sums[group_id[i]] += ranks[i];
    }

  double sum_term = 0.0;
  for (std::size_t g = 0; g < groups.size (); ++g)
    {
      const double ng = static_cast<double> (groups[g].size ());
      sum_term += (rank_sums[g] * rank_sums[g]) / ng;
    }
  const double dn = static_cast<double> (n_total);
  const double h_stat =
    (12.0 / (dn * (dn + 1.0))) * sum_term - 3.0 * (dn + 1.0);
  const chi2_dist chi2 {static_cast<double> (groups.size () - 1)};
  return correlation_result {h_stat, 1.0 - chi2.cdf (h_stat)};
}

/* Continuous Uniform and Lognormal Distributions (SciPy stats.uniform,
   stats.lognorm).  */
struct uniform_dist
{
  double loc = 0.0;
  double scale = 1.0;

  double
  pdf (double x) const noexcept
  {
    if (scale <= 0.0 || x < loc || x > loc + scale)
      {
        return 0.0;
      }
    return 1.0 / scale;
  }

  double
  cdf (double x) const noexcept
  {
    if (x <= loc)
      {
        return 0.0;
      }
    if (x >= loc + scale)
      {
        return 1.0;
      }
    return (x - loc) / scale;
  }

  double
  ppf (double p) const
  {
    if (p < 0.0 || p > 1.0)
      {
        throw hyper_math_error ("uniform_dist::ppf requires p in [0, 1]");
      }
    return loc + p * scale;
  }
};

struct lognormal_dist
{
  double s = 1.0;     /* Shape sigma > 0.  */
  double scale = 1.0; /* exp(mu).  */

  double
  pdf (double x) const noexcept
  {
    if (x <= 0.0 || s <= 0.0 || scale <= 0.0)
      {
        return 0.0;
      }
    const double z = std::log (x / scale) / s;
    return std::exp (-0.5 * z * z) / (x * s * constants::sqrt_two_pi);
  }

  double
  cdf (double x) const noexcept
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    const double z = std::log (x / scale) / (s * constants::sqrt2);
    return 0.5 * (1.0 + std::erf (z));
  }
};

/* Median Absolute Deviation (MAD): median(|x - median(x)|) / scale
   (SciPy stats.median_abs_deviation equivalent).  */
inline double
median_abs_deviation (const nd_tensor<double> &x, bool normal_scale = false)
{
  if (x.empty ())
    {
      return constants::nan;
    }
  const double med = tensor::median (x);
  nd_tensor<double> dev (shape_type {x.size ()});
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      dev[i] = std::abs (x.eval_at (i) - med);
    }
  const double raw_mad = tensor::median (dev);
  return normal_scale ? (raw_mad * 1.482602218505602) : raw_mad;
}

/* Wilcoxon Signed-Rank Test for paired samples x and y
   (SciPy stats.wilcoxon equivalent).  */
inline correlation_result
wilcoxon (const nd_tensor<double> &x, const nd_tensor<double> &y)
{
  if (x.size () != y.size () || x.empty ())
    {
      throw shape_error ("wilcoxon requires equal-length non-empty samples");
    }
  std::vector<double> abs_diffs;
  std::vector<int> signs;
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      const double d = x.eval_at (i) - y.eval_at (i);
      if (std::abs (d) > 0.0)
        {
          abs_diffs.push_back (std::abs (d));
          signs.push_back (d > 0.0 ? 1 : -1);
        }
    }
  const std::size_t nr = abs_diffs.size ();
  if (nr == 0)
    {
      return correlation_result {0.0, 1.0};
    }

  const nd_tensor<double> ranks =
    rankdata (nd_tensor<double> (shape_type {nr}, abs_diffs));
  double w_plus = 0.0;
  double w_minus = 0.0;
  for (std::size_t i = 0; i < nr; ++i)
    {
      if (signs[i] > 0)
        {
          w_plus += ranks[i];
        }
      else
        {
          w_minus += ranks[i];
        }
    }
  const double w_stat = std::min (w_plus, w_minus);
  const double dnr = static_cast<double> (nr);
  const double mu = 0.25 * dnr * (dnr + 1.0);
  const double sigma =
    std::sqrt (dnr * (dnr + 1.0) * (2.0 * dnr + 1.0) / 24.0);
  const double z = (sigma > 0.0) ? (std::abs (w_stat - mu) / sigma) : 0.0;
  const normal_dist norm {};
  const double pval = 2.0 * (1.0 - norm.cdf (z));
  return correlation_result {w_stat, pval};
}

/* Friedman Chi-Square Repeated-Measures Non-Parametric Test
   (SciPy stats.friedmanchisquare equivalent).  */
inline correlation_result
friedmanchisquare (const std::vector<nd_tensor<double>> &treatments)
{
  const std::size_t k = treatments.size ();
  if (k < 3)
    {
      throw shape_error ("friedmanchisquare requires at least 3 treatments");
    }
  const std::size_t n = treatments[0].size ();
  if (n == 0)
    {
      throw shape_error ("friedmanchisquare treatments must be non-empty");
    }
  for (std::size_t j = 1; j < k; ++j)
    {
      if (treatments[j].size () != n)
        {
          throw shape_error ("friedmanchisquare treatments must have equal length");
        }
    }

  std::vector<double> col_rank_sums (k, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    {
      nd_tensor<double> row (shape_type {k});
      for (std::size_t j = 0; j < k; ++j)
        {
          row[j] = treatments[j].eval_at (i);
        }
      const nd_tensor<double> r = rankdata (row);
      for (std::size_t j = 0; j < k; ++j)
        {
          col_rank_sums[j] += r[j];
        }
    }

  double sum_sq = 0.0;
  for (std::size_t j = 0; j < k; ++j)
    {
      sum_sq += col_rank_sums[j] * col_rank_sums[j];
    }
  const double dn = static_cast<double> (n);
  const double dk = static_cast<double> (k);
  const double q_stat =
    (12.0 / (dn * dk * (dk + 1.0))) * sum_sq - 3.0 * dn * (dk + 1.0);
  const chi2_dist chi2 {dk - 1.0};
  return correlation_result {q_stat, 1.0 - chi2.cdf (q_stat)};
}

/* 1-D Optimal Transport / Empirical Distribution Distances:
   1-Wasserstein (Earth Mover's) Distance (`wasserstein_distance`) and
   Székely Energy Distance (`energy_distance`)
   (SciPy stats.wasserstein_distance and stats.energy_distance).  */
inline double
wasserstein_distance (const nd_tensor<double> &u_values,
                      const nd_tensor<double> &v_values)
{
  if (u_values.empty () || v_values.empty ())
    {
      throw shape_error ("wasserstein_distance requires non-empty samples");
    }
  const nd_tensor<double> u_sorted = tensor::sort (u_values);
  const nd_tensor<double> v_sorted = tensor::sort (v_values);
  const std::size_t nu = u_sorted.size ();
  const std::size_t nv = v_sorted.size ();

  /* Merge all breakpoints and integrate |F_u(t) - F_v(t)| dt.  */
  std::vector<double> all_pts;
  all_pts.reserve (nu + nv);
  for (std::size_t i = 0; i < nu; ++i)
    {
      all_pts.push_back (u_sorted[i]);
    }
  for (std::size_t j = 0; j < nv; ++j)
    {
      all_pts.push_back (v_sorted[j]);
    }
  std::sort (all_pts.begin (), all_pts.end ());

  double dist = 0.0;
  std::size_t iu = 0;
  std::size_t iv = 0;
  for (std::size_t k = 0; k + 1 < all_pts.size (); ++k)
    {
      const double x_cur = all_pts[k];
      const double x_next = all_pts[k + 1];
      while (iu < nu && u_sorted[iu] <= x_cur)
        {
          ++iu;
        }
      while (iv < nv && v_sorted[iv] <= x_cur)
        {
          ++iv;
        }
      const double delta = x_next - x_cur;
      if (delta > 0.0)
        {
          const double cdf_u =
            static_cast<double> (iu) / static_cast<double> (nu);
          const double cdf_v =
            static_cast<double> (iv) / static_cast<double> (nv);
          dist += delta * std::abs (cdf_u - cdf_v);
        }
    }
  return dist;
}

inline double
energy_distance (const nd_tensor<double> &u_values,
                 const nd_tensor<double> &v_values)
{
  if (u_values.empty () || v_values.empty ())
    {
      throw shape_error ("energy_distance requires non-empty samples");
    }
  /* In 1D, energy_distance(u, v) = sqrt(2 * \int (F_u(t) - F_v(t))^2 dt).  */
  const nd_tensor<double> u_sorted = tensor::sort (u_values);
  const nd_tensor<double> v_sorted = tensor::sort (v_values);
  const std::size_t nu = u_sorted.size ();
  const std::size_t nv = v_sorted.size ();

  std::vector<double> all_pts;
  all_pts.reserve (nu + nv);
  for (std::size_t i = 0; i < nu; ++i)
    {
      all_pts.push_back (u_sorted[i]);
    }
  for (std::size_t j = 0; j < nv; ++j)
    {
      all_pts.push_back (v_sorted[j]);
    }
  std::sort (all_pts.begin (), all_pts.end ());

  double sq_int = 0.0;
  std::size_t iu = 0;
  std::size_t iv = 0;
  for (std::size_t k = 0; k + 1 < all_pts.size (); ++k)
    {
      const double x_cur = all_pts[k];
      const double x_next = all_pts[k + 1];
      while (iu < nu && u_sorted[iu] <= x_cur)
        {
          ++iu;
        }
      while (iv < nv && v_sorted[iv] <= x_cur)
        {
          ++iv;
        }
      const double delta = x_next - x_cur;
      if (delta > 0.0)
        {
          const double diff =
            static_cast<double> (iu) / static_cast<double> (nu)
            - static_cast<double> (iv) / static_cast<double> (nv);
          sq_int += delta * diff * diff;
        }
    }
  return std::sqrt (2.0 * sq_int);
}

/* Multivariate Normal Distribution N(mu, cov) via Cholesky factorization
   and Box-Cox Power Transformation (SciPy stats.multivariate_normal and
   stats.boxcox equivalents).  */
class multivariate_normal_dist
{
private:
  nd_tensor<double> mean_;
  nd_tensor<double> chol_l_;
  double log_det_cov_;

public:
  multivariate_normal_dist (const nd_tensor<double> &mean,
                            const nd_tensor<double> &cov)
    : mean_ (mean.copy ()),
      chol_l_ (linalg::cholesky (cov)),
      log_det_cov_ (0.0)
  {
    if (mean_.ndim () != 1 || cov.ndim () != 2
        || cov.shape ()[0] != mean_.size ()
        || cov.shape ()[1] != mean_.size ())
      {
        throw shape_error ("multivariate_normal_dist mean/cov shape mismatch");
      }
    for (std::size_t i = 0; i < mean_.size (); ++i)
      {
        log_det_cov_ += 2.0 * std::log (chol_l_ (i, i));
      }
  }

  double
  logpdf (const nd_tensor<double> &x) const
  {
    const std::size_t d = mean_.size ();
    if (x.size () != d)
      {
        throw shape_error ("multivariate_normal_dist::logpdf dimension mismatch");
      }
    const nd_tensor<double> diff (x - mean_);
    const nd_tensor<double> z =
      linalg::solve_triangular (chol_l_, diff, true);
    const double maha_sq = tensor::dot (z, z);
    return -0.5
           * (static_cast<double> (d) * std::log (2.0 * constants::pi)
              + log_det_cov_ + maha_sq);
  }

  double
  pdf (const nd_tensor<double> &x) const
  {
    return std::exp (logpdf (x));
  }
};

inline nd_tensor<double>
boxcox (const nd_tensor<double> &x, double lmbda)
{
  nd_tensor<double> out (x.shape ());
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      const double v = x.eval_at (i);
      if (v <= 0.0)
        {
          throw hyper_math_error ("boxcox requires strictly positive data");
        }
      out[i] = (std::abs (lmbda) < 1e-12)
               ? std::log (v)
               : ((std::pow (v, lmbda) - 1.0) / lmbda);
    }
  return out;
}

/* Maximum-likelihood Box-Cox power transform returning `{transformed, lmbda}`
   (SciPy stats.boxcox(x) equivalent).  */
inline std::pair<nd_tensor<double>, double>
boxcox (const nd_tensor<double> &x)
{
  const std::size_t n = x.size ();
  if (n < 2)
    {
      throw hyper_math_error ("boxcox MLE requires at least 2 observations");
    }
  double sum_log_x = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      const double v = x.eval_at (i);
      if (v <= 0.0)
        {
          throw hyper_math_error ("boxcox requires strictly positive data");
        }
      sum_log_x += std::log (v);
    }

  auto neg_llf = [&] (double lam) {
    const nd_tensor<double> y = boxcox (x, lam);
    double v = sample_var (y, 0);
    if (v <= 1e-300)
      {
        v = 1e-300;
      }
    const double llf =
      (lam - 1.0) * sum_log_x
      - 0.5 * static_cast<double> (n) * std::log (v);
    return -llf;
  };

  const auto opt = optimize::minimize_scalar (neg_llf, -5.0, 5.0);
  return {boxcox (x, opt.x), opt.x};
}

/* Yeo-Johnson power transform supporting positive, zero, and negative real
   inputs (SciPy stats.yeojohnson equivalent), plus exact two-sided binomial
   hypothesis test `binom_test`.  */
inline nd_tensor<double>
yeojohnson (const nd_tensor<double> &x, double lmbda)
{
  nd_tensor<double> out (x.shape ());
  for (std::size_t i = 0; i < x.size (); ++i)
    {
      const double v = x.eval_at (i);
      if (v >= 0.0)
        {
          out[i] = (std::abs (lmbda) < 1e-12)
                   ? std::log1p (v)
                   : ((std::pow (v + 1.0, lmbda) - 1.0) / lmbda);
        }
      else
        {
          const double two_m_lam = 2.0 - lmbda;
          out[i] = (std::abs (two_m_lam) < 1e-12)
                   ? -std::log1p (-v)
                   : -((std::pow (-v + 1.0, two_m_lam) - 1.0) / two_m_lam);
        }
    }
  return out;
}

inline std::pair<nd_tensor<double>, double>
yeojohnson (const nd_tensor<double> &x)
{
  const std::size_t n = x.size ();
  if (n < 2)
    {
      throw hyper_math_error ("yeojohnson MLE requires at least 2 observations");
    }
  auto neg_llf = [&] (double lam) {
    const nd_tensor<double> y = yeojohnson (x, lam);
    double v = sample_var (y, 0);
    if (v <= 1e-300)
      {
        v = 1e-300;
      }
    double jac_log = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      {
        const double xi = x.eval_at (i);
        const double sgn = (xi >= 0.0) ? 1.0 : -1.0;
        jac_log += (lam - 1.0) * sgn * std::log1p (std::abs (xi));
      }
    return 0.5 * static_cast<double> (n) * std::log (v) - jac_log;
  };
  const auto opt = optimize::minimize_scalar (neg_llf, -5.0, 5.0);
  return {yeojohnson (x, opt.x), opt.x};
}

inline double
binom_test (std::size_t k, std::size_t n, double p = 0.5)
{
  if (k > n || p < 0.0 || p > 1.0)
    {
      throw hyper_math_error ("invalid parameters in binom_test");
    }
  const binomial_dist dist (n, p);
  const double p_obs = dist.pmf (k);
  double p_val = 0.0;
  for (std::size_t i = 0; i <= n; ++i)
    {
      const double pi = dist.pmf (i);
      if (pi <= p_obs * (1.0 + 1e-12))
        {
          p_val += pi;
        }
    }
  return std::min (1.0, p_val);
}

/* Jarque-Bera normality test `jarque_bera`, Levene/Brown-Forsythe homogeneity
   of variance test `levene`, and standard error of the mean `sem`
   (SciPy stats equivalents).  */
inline correlation_result
jarque_bera (const nd_tensor<double> &x)
{
  const std::size_t n = x.size ();
  if (n < 3)
    {
      throw shape_error ("jarque_bera requires at least 3 observations");
    }
  const double s = skew (x);
  const double k = kurtosis (x, true);
  const double jb =
    (static_cast<double> (n) / 6.0) * (s * s + 0.25 * k * k);
  const chi2_dist dist {2.0};
  return correlation_result {jb, 1.0 - dist.cdf (jb)};
}

inline correlation_result
levene (const std::vector<nd_tensor<double>> &groups,
        const std::string &center = "median")
{
  if (groups.size () < 2)
    {
      throw shape_error ("levene requires at least 2 groups");
    }
  std::vector<nd_tensor<double>> z_groups;
  z_groups.reserve (groups.size ());
  for (const auto &g : groups)
    {
      const double c =
        (center == "mean") ? tensor::mean (g) : tensor::median (g);
      nd_tensor<double> z (shape_type {g.size ()});
      for (std::size_t i = 0; i < g.size (); ++i)
        {
          z[i] = std::abs (g.eval_at (i) - c);
        }
      z_groups.push_back (std::move (z));
    }
  return f_oneway (z_groups);
}

inline double
sem (const nd_tensor<double> &x, std::size_t ddof = 1)
{
  if (x.size () <= ddof)
    {
      throw shape_error ("sem requires x.size() > ddof");
    }
  return tensor::sample_std_dev (x, ddof)
         / std::sqrt (static_cast<double> (x.size ()));
}

/* Modified Bessel functions of the first kind `special::i0`, `special::i1`,
   and Bartlett's test for homogeneity of variances `bartlett_test`
   (SciPy special.i0, special.i1, and stats.bartlett equivalents).  */
namespace special
{

inline double
i0 (double x) noexcept
{
  const double half_x = 0.5 * x;
  const double z = half_x * half_x;
  double term = 1.0;
  double sum = 1.0;
  for (int k = 1; k < 60; ++k)
    {
      const double dk = static_cast<double> (k);
      term *= z / (dk * dk);
      sum += term;
      if (term <= 1e-16 * sum)
        {
          break;
        }
    }
  return sum;
}

inline double
i1 (double x) noexcept
{
  const double sgn = (x < 0.0) ? -1.0 : 1.0;
  const double half_x = 0.5 * std::abs (x);
  const double z = half_x * half_x;
  double term = half_x;
  double sum = half_x;
  for (int k = 1; k < 60; ++k)
    {
      const double dk = static_cast<double> (k);
      term *= z / (dk * (dk + 1.0));
      sum += term;
      if (term <= 1e-16 * sum)
        {
          break;
        }
    }
  return sgn * sum;
}

} // namespace special

inline correlation_result
bartlett_test (const std::vector<nd_tensor<double>> &groups)
{
  const std::size_t k = groups.size ();
  if (k < 2)
    {
      throw shape_error ("bartlett_test requires at least 2 groups");
    }
  double total_df = 0.0;
  double pooled_ss = 0.0;
  double sum_df_log_s2 = 0.0;
  double sum_inv_df = 0.0;

  for (const auto &g : groups)
    {
      if (g.size () < 2)
        {
          throw shape_error ("each group in bartlett_test must have size >= 2");
        }
      const double df_i = static_cast<double> (g.size () - 1);
      const double s2_i = tensor::sample_var (g, 1);
      total_df += df_i;
      pooled_ss += df_i * s2_i;
      sum_df_log_s2 += df_i * std::log (std::max (s2_i, 1e-300));
      sum_inv_df += 1.0 / df_i;
    }

  const double s2_p = pooled_ss / total_df;
  const double numer =
    total_df * std::log (std::max (s2_p, 1e-300)) - sum_df_log_s2;
  const double denom =
    1.0
    + (1.0 / (3.0 * static_cast<double> (k - 1)))
      * (sum_inv_df - (1.0 / total_df));
  const double stat = numer / denom;
  const chi2_dist dist {static_cast<double> (k - 1)};
  return correlation_result {stat, 1.0 - dist.cdf (stat)};
}

/* Fisher's Exact Test for 2x2 contingency tables `fisher_exact` and Mood's
   Non-Parametric Median Test `mood_median_test`
   (SciPy stats.fisher_exact and stats.median_test equivalents).  */
inline correlation_result
fisher_exact (const nd_tensor<double> &table_2x2)
{
  if (table_2x2.ndim () != 2
      || table_2x2.shape ()[0] != 2
      || table_2x2.shape ()[1] != 2)
    {
      throw shape_error ("fisher_exact requires a 2x2 contingency table");
    }
  const std::int64_t a = static_cast<std::int64_t> (std::llround (table_2x2 (0, 0)));
  const std::int64_t b = static_cast<std::int64_t> (std::llround (table_2x2 (0, 1)));
  const std::int64_t c = static_cast<std::int64_t> (std::llround (table_2x2 (1, 0)));
  const std::int64_t d = static_cast<std::int64_t> (std::llround (table_2x2 (1, 1)));
  if (a < 0 || b < 0 || c < 0 || d < 0)
    {
      throw hyper_math_error ("fisher_exact counts must be non-negative");
    }

  const double odds_ratio =
    (b * c == 0)
    ? constants::inf
    : (static_cast<double> (a * d) / static_cast<double> (b * c));

  const std::int64_t r1 = a + b;
  const std::int64_t r2 = c + d;
  const std::int64_t c1 = a + c;
  const std::int64_t n = r1 + r2;

  auto log_choose = [] (std::int64_t m, std::int64_t k) {
    return special::lgamma (static_cast<double> (m + 1))
           - special::lgamma (static_cast<double> (k + 1))
           - special::lgamma (static_cast<double> (m - k + 1));
  };

  const double log_denom = log_choose (n, c1);
  auto hypergeom_pmf = [&] (std::int64_t x) {
    return std::exp (log_choose (r1, x)
                     + log_choose (r2, c1 - x)
                     - log_denom);
  };

  const double p_obs = hypergeom_pmf (a);
  const std::int64_t x_min = std::max<std::int64_t> (0, c1 - r2);
  const std::int64_t x_max = std::min (r1, c1);
  double p_val = 0.0;
  for (std::int64_t x = x_min; x <= x_max; ++x)
    {
      const double px = hypergeom_pmf (x);
      if (px <= p_obs * (1.0 + 1e-12))
        {
          p_val += px;
        }
    }
  return correlation_result {odds_ratio, std::min (1.0, p_val)};
}

inline correlation_result
mood_median_test (const std::vector<nd_tensor<double>> &groups)
{
  const std::size_t k = groups.size ();
  if (k < 2)
    {
      throw shape_error ("mood_median_test requires at least 2 groups");
    }
  std::vector<double> pooled;
  for (const auto &g : groups)
    {
      for (std::size_t i = 0; i < g.size (); ++i)
        {
          pooled.push_back (g.eval_at (i));
        }
    }
  const nd_tensor<double> pt (shape_type {pooled.size ()}, pooled);
  const double grand_med = tensor::median (pt);

  double total_above = 0.0;
  double total_le = 0.0;
  std::vector<double> above_counts (k, 0.0);
  std::vector<double> le_counts (k, 0.0);
  for (std::size_t g = 0; g < k; ++g)
    {
      for (std::size_t i = 0; i < groups[g].size (); ++i)
        {
          if (groups[g].eval_at (i) > grand_med)
            {
              above_counts[g] += 1.0;
              total_above += 1.0;
            }
          else
            {
              le_counts[g] += 1.0;
              total_le += 1.0;
            }
        }
    }

  const double dn = static_cast<double> (pooled.size ());
  double stat = 0.0;
  for (std::size_t g = 0; g < k; ++g)
    {
      const double ng = static_cast<double> (groups[g].size ());
      const double exp_above = (ng * total_above) / dn;
      const double exp_le = (ng * total_le) / dn;
      if (exp_above > 0.0)
        {
          const double d1 = above_counts[g] - exp_above;
          stat += (d1 * d1) / exp_above;
        }
      if (exp_le > 0.0)
        {
          const double d2 = le_counts[g] - exp_le;
          stat += (d2 * d2) / exp_le;
        }
    }

  const chi2_dist dist {static_cast<double> (k - 1)};
  return correlation_result {stat, 1.0 - dist.cdf (stat)};
}

/* Additional Continuous & Discrete Distributions: `weibull_min_dist`,
   `cauchy_dist`, `laplace_dist`, `pareto_dist`, `f_dist` (Fisher-Snedecor F),
   and `geom_dist` (Geometric), plus `kendalltau` and `fligner`
   (SciPy stats equivalents).  */
struct weibull_min_dist
{
  double c = 1.0;
  double scale = 1.0;

  double
  pdf (double x) const noexcept
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    const double u = x / scale;
    return (c / scale) * std::pow (u, c - 1.0) * std::exp (-std::pow (u, c));
  }

  double
  cdf (double x) const noexcept
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    return 1.0 - std::exp (-std::pow (x / scale, c));
  }

  double
  ppf (double p) const
  {
    if (p <= 0.0 || p >= 1.0)
      {
        throw hyper_math_error ("weibull_min_dist::ppf requires p in (0, 1)");
      }
    return scale * std::pow (-std::log (1.0 - p), 1.0 / c);
  }
};

struct cauchy_dist
{
  double loc = 0.0;
  double scale = 1.0;

  double
  pdf (double x) const noexcept
  {
    const double z = (x - loc) / scale;
    return 1.0 / (constants::pi * scale * (1.0 + z * z));
  }

  double
  cdf (double x) const noexcept
  {
    return 0.5 + std::atan ((x - loc) / scale) / constants::pi;
  }

  double
  ppf (double p) const
  {
    if (p <= 0.0 || p >= 1.0)
      {
        throw hyper_math_error ("cauchy_dist::ppf requires p in (0, 1)");
      }
    return loc + scale * std::tan (constants::pi * (p - 0.5));
  }
};

struct laplace_dist
{
  double loc = 0.0;
  double scale = 1.0;

  double
  pdf (double x) const noexcept
  {
    return std::exp (-std::abs (x - loc) / scale) / (2.0 * scale);
  }

  double
  cdf (double x) const noexcept
  {
    if (x < loc)
      {
        return 0.5 * std::exp ((x - loc) / scale);
      }
    return 1.0 - 0.5 * std::exp (-(x - loc) / scale);
  }

  double
  ppf (double p) const
  {
    if (p <= 0.0 || p >= 1.0)
      {
        throw hyper_math_error ("laplace_dist::ppf requires p in (0, 1)");
      }
    if (p < 0.5)
      {
        return loc + scale * std::log (2.0 * p);
      }
    return loc - scale * std::log (2.0 * (1.0 - p));
  }
};

struct pareto_dist
{
  double b = 1.0;
  double scale = 1.0;

  double
  pdf (double x) const noexcept
  {
    if (x < scale)
      {
        return 0.0;
      }
    return (b * std::pow (scale, b)) / std::pow (x, b + 1.0);
  }

  double
  cdf (double x) const noexcept
  {
    if (x < scale)
      {
        return 0.0;
      }
    return 1.0 - std::pow (scale / x, b);
  }

  double
  ppf (double p) const
  {
    if (p <= 0.0 || p >= 1.0)
      {
        throw hyper_math_error ("pareto_dist::ppf requires p in (0, 1)");
      }
    return scale / std::pow (1.0 - p, 1.0 / b);
  }
};

struct f_dist
{
  double dfn = 1.0;
  double dfd = 1.0;

  double
  pdf (double x) const noexcept
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    const double a = 0.5 * dfn;
    const double b = 0.5 * dfd;
    const double log_num =
      a * std::log (dfn) + b * std::log (dfd) + (a - 1.0) * std::log (x);
    const double log_den =
      (a + b) * std::log (dfn * x + dfd) + std::lgamma (a) + std::lgamma (b)
      - std::lgamma (a + b);
    return std::exp (log_num - log_den);
  }

  double
  cdf (double x) const
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    const double u = (dfn * x) / (dfn * x + dfd);
    return special::betainc (0.5 * dfn, 0.5 * dfd, u);
  }
};

struct geom_dist
{
  double p = 0.5;

  double
  pmf (std::int64_t k) const noexcept
  {
    if (k < 1)
      {
        return 0.0;
      }
    return p * std::pow (1.0 - p, static_cast<double> (k - 1));
  }

  double
  cdf (std::int64_t k) const noexcept
  {
    if (k < 1)
      {
        return 0.0;
      }
    return 1.0 - std::pow (1.0 - p, static_cast<double> (k));
  }
};

/* Kendall's rank correlation tau-b with tie adjustments and asymptotic
   two-sided z-test (SciPy stats.kendalltau equivalent).  */
inline correlation_result
kendalltau (const nd_tensor<double> &x, const nd_tensor<double> &y)
{
  const std::size_t n = x.size ();
  if (n != y.size () || n < 2)
    {
      throw shape_error ("kendalltau requires equal lengths >= 2");
    }

  double concordant = 0.0;
  double discordant = 0.0;
  double ties_x = 0.0;
  double ties_y = 0.0;

  for (std::size_t i = 0; i < n; ++i)
    {
      const double xi = x.eval_at (i);
      const double yi = y.eval_at (i);
      for (std::size_t j = i + 1; j < n; ++j)
        {
          const double dx = xi - x.eval_at (j);
          const double dy = yi - y.eval_at (j);
          if (dx == 0.0 && dy == 0.0)
            {
              continue;
            }
          if (dx == 0.0)
            {
              ties_x += 1.0;
            }
          else if (dy == 0.0)
            {
              ties_y += 1.0;
            }
          else if (dx * dy > 0.0)
            {
              concordant += 1.0;
            }
          else
            {
              discordant += 1.0;
            }
        }
    }

  const double denom =
    std::sqrt ((concordant + discordant + ties_x)
               * (concordant + discordant + ties_y));
  if (denom == 0.0)
    {
      return correlation_result {0.0, 1.0};
    }
  const double tau = (concordant - discordant) / denom;
  const double dn = static_cast<double> (n);
  const double var_tau =
    (2.0 * (2.0 * dn + 5.0)) / (9.0 * dn * (dn - 1.0));
  const double z = std::abs (tau) / std::sqrt (var_tau);
  const normal_dist norm;
  const double p_val = 2.0 * (1.0 - norm.cdf (z));
  return correlation_result {tau, p_val};
}

/* Fligner-Killeen non-parametric test for homogeneity of group variances
   based on normal-quantile scores of median-centered absolute deviations
   (SciPy stats.fligner equivalent).  */
inline correlation_result
fligner (const std::vector<nd_tensor<double>> &groups)
{
  const std::size_t k = groups.size ();
  if (k < 2)
    {
      throw shape_error ("fligner requires at least 2 groups");
    }

  std::vector<double> pooled_devs;
  std::vector<std::size_t> group_ids;
  for (std::size_t g = 0; g < k; ++g)
    {
      if (groups[g].size () == 0)
        {
          throw shape_error ("fligner groups must be non-empty");
        }
      const double med = tensor::median (groups[g]);
      for (std::size_t i = 0; i < groups[g].size (); ++i)
        {
          pooled_devs.push_back (std::abs (groups[g].eval_at (i) - med));
          group_ids.push_back (g);
        }
    }

  const std::size_t n_total = pooled_devs.size ();
  const nd_tensor<double> dev_t (shape_type {n_total}, pooled_devs);
  const nd_tensor<double> ranks = rankdata (dev_t);

  const normal_dist std_norm;
  std::vector<double> scores (n_total);
  double mean_score = 0.0;
  for (std::size_t i = 0; i < n_total; ++i)
    {
      const double u =
        0.5 + ranks[i] / (2.0 * (static_cast<double> (n_total) + 1.0));
      scores[i] = std_norm.ppf (std::clamp (u, 1e-12, 1.0 - 1e-12));
      mean_score += scores[i];
    }
  mean_score /= static_cast<double> (n_total);

  double var_score = 0.0;
  for (std::size_t i = 0; i < n_total; ++i)
    {
      const double d = scores[i] - mean_score;
      var_score += d * d;
    }
  var_score /= static_cast<double> (n_total - 1);
  if (var_score <= 0.0)
    {
      return correlation_result {0.0, 1.0};
    }

  std::vector<double> group_sums (k, 0.0);
  for (std::size_t i = 0; i < n_total; ++i)
    {
      group_sums[group_ids[i]] += scores[i];
    }

  double x2 = 0.0;
  for (std::size_t g = 0; g < k; ++g)
    {
      const double ng = static_cast<double> (groups[g].size ());
      const double mg = group_sums[g] / ng;
      const double d = mg - mean_score;
      x2 += ng * d * d;
    }
  x2 /= var_score;

  const chi2_dist dist {static_cast<double> (k - 1)};
  return correlation_result {x2, 1.0 - dist.cdf (x2)};
}

/* Extended Special Functions in `stats::special`:
   gammaincinv, betaincinv, polygamma, poch, spence, dawsn, expi, fresnel,
   sici, yv, kv, spherical_jn, spherical_yn, eval_hermite, eval_laguerre,
   eval_jacobi, and voigt_profile (SciPy special equivalents).  */
namespace special
{

inline double
gammaincinv (double a, double y)
{
  if (a <= 0.0 || y < 0.0 || y > 1.0)
    {
      throw hyper_math_error ("gammaincinv requires a > 0 and y in [0, 1]");
    }
  if (y == 0.0)
    {
      return 0.0;
    }
  if (y == 1.0)
    {
      return constants::inf;
    }
  double lo = 0.0;
  double hi = std::max (10.0, a + 10.0 * std::sqrt (a));
  while (gammainc (a, hi) < y)
    {
      hi *= 2.0;
    }
  for (int it = 0; it < 60; ++it)
    {
      const double mid = 0.5 * (lo + hi);
      if (gammainc (a, mid) < y)
        {
          lo = mid;
        }
      else
        {
          hi = mid;
        }
    }
  return 0.5 * (lo + hi);
}

inline double
betaincinv (double a, double b, double y)
{
  if (a <= 0.0 || b <= 0.0 || y < 0.0 || y > 1.0)
    {
      throw hyper_math_error ("betaincinv requires a, b > 0 and y in [0, 1]");
    }
  if (y == 0.0)
    {
      return 0.0;
    }
  if (y == 1.0)
    {
      return 1.0;
    }
  double lo = 0.0;
  double hi = 1.0;
  for (int it = 0; it < 60; ++it)
    {
      const double mid = 0.5 * (lo + hi);
      if (betainc (a, b, mid) < y)
        {
          lo = mid;
        }
      else
        {
          hi = mid;
        }
    }
  return 0.5 * (lo + hi);
}

inline double
polygamma (std::size_t n, double x)
{
  if (x <= 0.0)
    {
      throw hyper_math_error ("polygamma requires x > 0");
    }
  if (n == 0)
    {
      return digamma (x);
    }
  const double sign = ((n % 2) == 1) ? 1.0 : -1.0;
  const double fact_n = std::tgamma (static_cast<double> (n + 1));
  const double np1 = static_cast<double> (n + 1);
  double acc = 0.0;
  while (x < 8.0)
    {
      acc += 1.0 / std::pow (x, np1);
      x += 1.0;
    }
  const double inv_x = 1.0 / x;
  const double inv_x2 = inv_x * inv_x;
  const double dn = static_cast<double> (n);
  const double asymp =
    std::pow (inv_x, dn)
    * (1.0 / dn + 0.5 * inv_x
       + (np1 / 12.0) * inv_x2
       - (np1 * (dn + 2.0) * (dn + 3.0) / 720.0) * inv_x2 * inv_x2);
  return sign * fact_n * (acc + asymp);
}

inline double
poch (double z, double m)
{
  return std::tgamma (z + m) / std::tgamma (z);
}

inline double
spence (double z)
{
  if (z < 0.0)
    {
      throw hyper_math_error ("spence requires z >= 0");
    }
  if (std::abs (z - 1.0) < 1e-15)
    {
      return 0.0;
    }
  /* Composite Gauss-Legendre integration of -ln(t) / (t - 1) from 1 to z.  */
  const std::size_t n_panels = 32;
  const double h = (z - 1.0) / static_cast<double> (n_panels);
  double sum = 0.0;
  for (std::size_t p = 0; p < n_panels; ++p)
    {
      const double a = 1.0 + static_cast<double> (p) * h;
      const double b = a + h;
      const double mid = 0.5 * (a + b);
      const double half = 0.5 * h;
      const double nodes[3] = {-std::sqrt (0.6), 0.0, std::sqrt (0.6)};
      const double wts[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
      for (int k = 0; k < 3; ++k)
        {
          const double t = mid + half * nodes[k];
          const double f =
            (std::abs (t - 1.0) < 1e-12) ? -1.0 : (-std::log (t) / (t - 1.0));
          sum += half * wts[k] * f;
        }
    }
  return sum;
}

inline double
dawsn (double x)
{
  const double sign = (x < 0.0) ? -1.0 : 1.0;
  const double ax = std::abs (x);
  if (ax > 6.0)
    {
      const double inv = 1.0 / (2.0 * ax);
      const double inv2 = 1.0 / (2.0 * ax * ax);
      return sign * inv * (1.0 + inv2 * (1.0 + 3.0 * inv2));
    }
  const std::size_t n = 64;
  const double h = ax / static_cast<double> (n);
  double sum = 0.5 * (std::exp (-ax * ax) + 1.0);
  for (std::size_t i = 1; i < n; ++i)
    {
      const double t = static_cast<double> (i) * h;
      sum += std::exp (t * t - ax * ax);
    }
  return sign * sum * h;
}

inline double
expi (double x)
{
  if (x == 0.0)
    {
      return -constants::inf;
    }
  return std::expint (x);
}

inline std::pair<double, double>
fresnel (double x)
{
  const std::size_t n = 128;
  const double h = x / static_cast<double> (n);
  double s_sum = 0.0;
  double c_sum = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      const double t_mid = (static_cast<double> (i) + 0.5) * h;
      const double arg = constants::half_pi * t_mid * t_mid;
      s_sum += std::sin (arg);
      c_sum += std::cos (arg);
    }
  return {s_sum * h, c_sum * h};
}

inline std::pair<double, double>
sici (double x)
{
  if (x <= 0.0)
    {
      throw hyper_math_error ("sici requires x > 0");
    }
  const std::size_t n = 128;
  const double h = x / static_cast<double> (n);
  double si_sum = 0.0;
  double cin_sum = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      const double t = (static_cast<double> (i) + 0.5) * h;
      si_sum += std::sin (t) / t;
      cin_sum += (1.0 - std::cos (t)) / t;
    }
  const double si = si_sum * h;
  const double ci = constants::euler_gamma + std::log (x) - cin_sum * h;
  return {si, ci};
}

inline double
yv (double v, double x)
{
  return std::cyl_neumann (v, x);
}

inline double
kv (double v, double x)
{
  return std::cyl_bessel_k (v, x);
}

inline double
spherical_jn (unsigned int n, double x)
{
  return std::sph_bessel (n, x);
}

inline double
spherical_yn (unsigned int n, double x)
{
  return std::sph_neumann (n, x);
}

inline double
eval_hermite (std::size_t n, double x)
{
  return std::hermite (static_cast<unsigned int> (n), x);
}

inline double
eval_laguerre (std::size_t n, double x)
{
  return std::laguerre (static_cast<unsigned int> (n), x);
}

inline double
eval_jacobi (std::size_t n, double alpha, double beta, double x)
{
  if (n == 0)
    {
      return 1.0;
    }
  double p_prev = 1.0;
  double p_cur =
    0.5 * (alpha - beta + (alpha + beta + 2.0) * x);
  for (std::size_t k = 1; k < n; ++k)
    {
      const double dk = static_cast<double> (k);
      const double ab = alpha + beta;
      const double c1 = 2.0 * (dk + 1.0) * (dk + ab + 1.0) * (2.0 * dk + ab);
      const double c2 =
        (2.0 * dk + ab + 1.0)
        * ((2.0 * dk + ab) * (2.0 * dk + ab + 2.0) * x
           + alpha * alpha - beta * beta);
      const double c3 =
        2.0 * (dk + alpha) * (dk + beta) * (2.0 * dk + ab + 2.0);
      const double p_next = (c2 * p_cur - c3 * p_prev) / c1;
      p_prev = p_cur;
      p_cur = p_next;
    }
  return p_cur;
}

inline double
voigt_profile (double x, double sigma, double gamma)
{
  if (sigma <= 0.0 && gamma <= 0.0)
    {
      throw hyper_math_error ("voigt_profile requires sigma > 0 or gamma > 0");
    }
  if (gamma <= 0.0)
    {
      return std::exp (-0.5 * (x * x) / (sigma * sigma))
             / (sigma * std::sqrt (2.0 * constants::pi));
    }
  if (sigma <= 0.0)
    {
      return gamma / (constants::pi * (x * x + gamma * gamma));
    }
  /* Thompson pseudo-Voigt accurate analytical approximation (FWHM mixing).  */
  const double f_g = 2.0 * sigma * std::sqrt (2.0 * std::log (2.0));
  const double f_l = 2.0 * gamma;
  const double f = std::pow (
    std::pow (f_g, 5.0)
    + 2.69269 * std::pow (f_g, 4.0) * f_l
    + 2.42843 * std::pow (f_g, 3.0) * std::pow (f_l, 2.0)
    + 4.47163 * std::pow (f_g, 2.0) * std::pow (f_l, 3.0)
    + 0.07842 * f_g * std::pow (f_l, 4.0)
    + std::pow (f_l, 5.0),
    0.2);
  const double r = f_l / f;
  const double eta =
    std::clamp (1.36603 * r - 0.47719 * r * r + 0.11116 * r * r * r, 0.0, 1.0);
  const double sig_f = f / (2.0 * std::sqrt (2.0 * std::log (2.0)));
  const double gam_f = 0.5 * f;
  const double g_part =
    std::exp (-0.5 * (x * x) / (sig_f * sig_f))
    / (sig_f * std::sqrt (2.0 * constants::pi));
  const double l_part =
    gam_f / (constants::pi * (x * x + gam_f * gam_f));
  return eta * l_part + (1.0 - eta) * g_part;
}

} // namespace special

/* Extended Probability Distributions:
   logistic_dist, gumbel_r_dist, rayleigh_dist, triang_dist, invgamma_dist,
   negbinom_dist, hypergeom_dist, zipf_dist, dirichlet_dist.  */
struct logistic_dist
{
  double loc = 0.0;
  double scale = 1.0;

  double
  pdf (double x) const noexcept
  {
    const double z = (x - loc) / scale;
    const double s = special::expit (z);
    return (s * (1.0 - s)) / scale;
  }

  double
  cdf (double x) const noexcept
  {
    return special::expit ((x - loc) / scale);
  }

  double
  ppf (double p) const
  {
    return loc + scale * special::logit (p);
  }
};

struct gumbel_r_dist
{
  double loc = 0.0;
  double scale = 1.0;

  double
  pdf (double x) const noexcept
  {
    const double z = (x - loc) / scale;
    return std::exp (-(z + std::exp (-z))) / scale;
  }

  double
  cdf (double x) const noexcept
  {
    const double z = (x - loc) / scale;
    return std::exp (-std::exp (-z));
  }

  double
  ppf (double p) const
  {
    if (p <= 0.0 || p >= 1.0)
      {
        throw hyper_math_error ("gumbel_r_dist::ppf requires p in (0, 1)");
      }
    return loc - scale * std::log (-std::log (p));
  }
};

struct rayleigh_dist
{
  double scale = 1.0;

  double
  pdf (double x) const noexcept
  {
    if (x < 0.0)
      {
        return 0.0;
      }
    const double s2 = scale * scale;
    return (x / s2) * std::exp (-0.5 * x * x / s2);
  }

  double
  cdf (double x) const noexcept
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    return 1.0 - std::exp (-0.5 * (x * x) / (scale * scale));
  }

  double
  ppf (double p) const
  {
    if (p < 0.0 || p >= 1.0)
      {
        throw hyper_math_error ("rayleigh_dist::ppf requires p in [0, 1)");
      }
    return scale * std::sqrt (-2.0 * std::log (1.0 - p));
  }
};

struct triang_dist
{
  double c = 0.5;
  double loc = 0.0;
  double scale = 1.0;

  double
  pdf (double x) const noexcept
  {
    const double u = (x - loc) / scale;
    if (u <= 0.0 || u >= 1.0)
      {
        return 0.0;
      }
    if (u < c)
      {
        return (2.0 * u) / (scale * c);
      }
    return (2.0 * (1.0 - u)) / (scale * (1.0 - c));
  }

  double
  cdf (double x) const noexcept
  {
    const double u = (x - loc) / scale;
    if (u <= 0.0)
      {
        return 0.0;
      }
    if (u >= 1.0)
      {
        return 1.0;
      }
    if (u <= c)
      {
        return (u * u) / c;
      }
    return 1.0 - ((1.0 - u) * (1.0 - u)) / (1.0 - c);
  }

  double
  ppf (double p) const
  {
    if (p < 0.0 || p > 1.0)
      {
        throw hyper_math_error ("triang_dist::ppf requires p in [0, 1]");
      }
    if (p <= c)
      {
        return loc + scale * std::sqrt (p * c);
      }
    return loc + scale * (1.0 - std::sqrt ((1.0 - p) * (1.0 - c)));
  }
};

struct invgamma_dist
{
  double a = 1.0;
  double scale = 1.0;

  double
  pdf (double x) const noexcept
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    const double u = scale / x;
    return (std::pow (u, a + 1.0) * std::exp (-u)) / (scale * std::tgamma (a));
  }

  double
  cdf (double x) const
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    return 1.0 - special::gammainc (a, scale / x);
  }
};

struct negbinom_dist
{
  double n = 1.0;
  double p = 0.5;

  double
  pmf (std::int64_t k) const noexcept
  {
    if (k < 0)
      {
        return 0.0;
      }
    const double dk = static_cast<double> (k);
    const double log_c =
      std::lgamma (dk + n) - std::lgamma (n) - std::lgamma (dk + 1.0);
    return std::exp (log_c + n * std::log (p) + dk * std::log (1.0 - p));
  }

  double
  cdf (std::int64_t k) const
  {
    if (k < 0)
      {
        return 0.0;
      }
    return special::betainc (n, static_cast<double> (k + 1), p);
  }
};

struct hypergeom_dist
{
  std::int64_t M = 20;
  std::int64_t n = 7;
  std::int64_t N = 12;

  double
  pmf (std::int64_t k) const noexcept
  {
    if (k < std::max<std::int64_t> (0, N - (M - n))
        || k > std::min (n, N))
      {
        return 0.0;
      }
    return (special::binom (static_cast<double> (n), static_cast<double> (k))
            * special::binom (static_cast<double> (M - n),
                              static_cast<double> (N - k)))
           / special::binom (static_cast<double> (M), static_cast<double> (N));
  }

  double
  cdf (std::int64_t k) const noexcept
  {
    double s = 0.0;
    for (std::int64_t i = 0; i <= k; ++i)
      {
        s += pmf (i);
      }
    return std::min (1.0, s);
  }

  double
  mean () const noexcept
  {
    return static_cast<double> (n) * static_cast<double> (N)
           / static_cast<double> (M);
  }
};

struct zipf_dist
{
  double a = 2.0;

  double
  pmf (std::int64_t k) const
  {
    if (k < 1)
      {
        return 0.0;
      }
    return 1.0 / (std::pow (static_cast<double> (k), a) * special::zeta (a));
  }

  double
  cdf (std::int64_t k) const
  {
    double s = 0.0;
    for (std::int64_t i = 1; i <= k; ++i)
      {
        s += pmf (i);
      }
    return std::min (1.0, s);
  }
};

struct dirichlet_dist
{
  nd_tensor<double> alpha;

  double
  pdf (const nd_tensor<double> &x) const
  {
    if (x.size () != alpha.size () || alpha.size () == 0)
      {
        throw shape_error ("dirichlet_dist::pdf dimension mismatch");
      }
    double sum_alpha = 0.0;
    double log_b = 0.0;
    double log_p = 0.0;
    for (std::size_t i = 0; i < alpha.size (); ++i)
      {
        const double ai = alpha.eval_at (i);
        const double xi = x.eval_at (i);
        if (xi <= 0.0)
          {
            return 0.0;
          }
        sum_alpha += ai;
        log_b += std::lgamma (ai);
        log_p += (ai - 1.0) * std::log (xi);
      }
    log_b -= std::lgamma (sum_alpha);
    return std::exp (log_p - log_b);
  }

  nd_tensor<double>
  mean () const
  {
    const double s = tensor::sum (alpha);
    return alpha / s;
  }
};

/* Normality, Goodness-of-Fit, Post-Hoc, Contingency, & Regression Tests:
   shapiro, anderson, normaltest, cramervonmises, tukey_hsd, chi2_contingency,
   linregress, theilslopes, pointbiserialr, mcnemar, combine_pvalues
   (SciPy stats equivalents).  */
inline correlation_result
shapiro (const nd_tensor<double> &x)
{
  const std::size_t n = x.size ();
  if (n < 3)
    {
      throw shape_error ("shapiro requires at least 3 observations");
    }
  std::vector<double> xs = x.to_vector ();
  std::sort (xs.begin (), xs.end ());

  const normal_dist std_norm;
  std::vector<double> m (n);
  double m_norm_sq = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      const double p =
        (static_cast<double> (i + 1) - 0.375)
        / (static_cast<double> (n) + 0.25);
      m[i] = std_norm.ppf (p);
      m_norm_sq += m[i] * m[i];
    }
  const double m_norm = std::sqrt (m_norm_sq);

  double mean_x = 0.0;
  for (double v : xs)
    {
      mean_x += v;
    }
  mean_x /= static_cast<double> (n);

  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      num += (m[i] / m_norm) * xs[i];
      const double dx = xs[i] - mean_x;
      den += dx * dx;
    }
  if (den <= 0.0)
    {
      return correlation_result {1.0, 1.0};
    }
  const double w = std::clamp ((num * num) / den, 0.0, 1.0);
  /* Royston log-transformation for approximate p-value of W.  */
  const double dn = static_cast<double> (n);
  const double mu = -1.2725 + 1.0521 * std::log ( std::log (dn) );
  const double sigma = 1.0308 - 0.26758 * std::log (dn);
  const double z =
    (std::log (std::max (1e-12, 1.0 - w)) - mu) / std::max (0.1, sigma);
  const double p_val = 1.0 - std_norm.cdf (z);
  return correlation_result {w, std::clamp (p_val, 0.0, 1.0)};
}

inline double
anderson (const nd_tensor<double> &x)
{
  const std::size_t n = x.size ();
  if (n < 2)
    {
      throw shape_error ("anderson requires at least 2 observations");
    }
  std::vector<double> z = zscore (x).to_vector ();
  std::sort (z.begin (), z.end ());
  const normal_dist std_norm;
  double s = 0.0;
  const double dn = static_cast<double> (n);
  for (std::size_t i = 0; i < n; ++i)
    {
      const double fi = std::clamp (std_norm.cdf (z[i]), 1e-15, 1.0 - 1e-15);
      const double f_rev =
        std::clamp (std_norm.cdf (z[n - 1 - i]), 1e-15, 1.0 - 1e-15);
      s += (2.0 * static_cast<double> (i + 1) - 1.0)
           * (std::log (fi) + std::log (1.0 - f_rev));
    }
  return -dn - s / dn;
}

inline correlation_result
normaltest (const nd_tensor<double> &x)
{
  const std::size_t n = x.size ();
  if (n < 8)
    {
      throw shape_error ("normaltest requires at least 8 observations");
    }
  const double dn = static_cast<double> (n);
  const double g1 = skew (x);
  const double g2 = kurtosis (x);
  const double se_skew = std::sqrt (6.0 / dn);
  const double se_kurt = std::sqrt (24.0 / dn);
  const double z1 = g1 / se_skew;
  const double z2 = g2 / se_kurt;
  const double k2 = z1 * z1 + z2 * z2;
  const chi2_dist d2 {2.0};
  return correlation_result {k2, 1.0 - d2.cdf (k2)};
}

template <typename CdfFunc>
correlation_result
cramervonmises (const nd_tensor<double> &x, CdfFunc &&cdf_fn)
{
  const std::size_t n = x.size ();
  if (n == 0)
    {
      throw shape_error ("cramervonmises requires non-empty sample");
    }
  std::vector<double> xs = x.to_vector ();
  std::sort (xs.begin (), xs.end ());
  const double dn = static_cast<double> (n);
  double t = 1.0 / (12.0 * dn);
  for (std::size_t i = 0; i < n; ++i)
    {
      const double target =
        (2.0 * static_cast<double> (i + 1) - 1.0) / (2.0 * dn);
      const double diff = cdf_fn (xs[i]) - target;
      t += diff * diff;
    }
  const double p_val = std::clamp (std::exp (-4.9 * t), 0.0, 1.0);
  return correlation_result {t, p_val};
}

struct tukey_hsd_result
{
  nd_tensor<double> statistic;
  nd_tensor<double> pvalue;
};

inline tukey_hsd_result
tukey_hsd (const std::vector<nd_tensor<double>> &groups)
{
  const std::size_t k = groups.size ();
  if (k < 2)
    {
      throw shape_error ("tukey_hsd requires at least 2 groups");
    }
  std::vector<double> means (k, 0.0);
  double ss_within = 0.0;
  std::size_t n_total = 0;
  for (std::size_t g = 0; g < k; ++g)
    {
      if (groups[g].size () < 2)
        {
          throw shape_error ("tukey_hsd requires each group to have size >= 2");
        }
      means[g] = tensor::mean (groups[g]);
      for (std::size_t i = 0; i < groups[g].size (); ++i)
        {
          const double d = groups[g].eval_at (i) - means[g];
          ss_within += d * d;
        }
      n_total += groups[g].size ();
    }
  const double df_w = static_cast<double> (n_total - k);
  const double msw = ss_within / df_w;
  const student_t_dist t_d {df_w};

  nd_tensor<double> q_mat (shape_type {k, k}, 0.0);
  nd_tensor<double> p_mat (shape_type {k, k}, 1.0);
  for (std::size_t i = 0; i < k; ++i)
    {
      for (std::size_t j = 0; j < k; ++j)
        {
          if (i == j)
            {
              continue;
            }
          const double ni = static_cast<double> (groups[i].size ());
          const double nj = static_cast<double> (groups[j].size ());
          const double se =
            std::sqrt (0.5 * msw * (1.0 / ni + 1.0 / nj));
          const double q_ij = (means[i] - means[j]) / std::max (1e-15, se);
          q_mat (i, j) = q_ij;
          const double t_eq = std::abs (q_ij) / constants::sqrt2;
          const double p_raw = 2.0 * (1.0 - t_d.cdf (t_eq));
          const double n_pairs =
            0.5 * static_cast<double> (k * (k - 1));
          p_mat (i, j) = std::min (1.0, p_raw * n_pairs);
        }
    }
  return tukey_hsd_result {std::move (q_mat), std::move (p_mat)};
}

struct chi2_contingency_result
{
  double statistic;
  double pvalue;
  std::size_t dof;
  nd_tensor<double> expected_freq;
};

inline chi2_contingency_result
chi2_contingency (const nd_tensor<double> &observed, bool correction = true)
{
  if (observed.ndim () != 2
      || observed.shape ()[0] < 2
      || observed.shape ()[1] < 2)
    {
      throw shape_error ("chi2_contingency requires at least a 2x2 table");
    }
  const std::size_t R = observed.shape ()[0];
  const std::size_t C = observed.shape ()[1];
  std::vector<double> row_sums (R, 0.0);
  std::vector<double> col_sums (C, 0.0);
  double total = 0.0;
  for (std::size_t r = 0; r < R; ++r)
    {
      for (std::size_t c = 0; c < C; ++c)
        {
          const double v = observed (r, c);
          row_sums[r] += v;
          col_sums[c] += v;
          total += v;
        }
    }
  const std::size_t dof = (R - 1) * (C - 1);
  nd_tensor<double> expected (shape_type {R, C}, 0.0);
  double chi2 = 0.0;
  const bool use_yates = correction && (dof == 1);
  for (std::size_t r = 0; r < R; ++r)
    {
      for (std::size_t c = 0; c < C; ++c)
        {
          const double e = (row_sums[r] * col_sums[c]) / std::max (1e-15, total);
          expected (r, c) = e;
          double diff = std::abs (observed (r, c) - e);
          if (use_yates)
            {
              diff = std::max (0.0, diff - 0.5);
            }
          chi2 += (diff * diff) / std::max (1e-15, e);
        }
    }
  const chi2_dist d {static_cast<double> (dof)};
  return chi2_contingency_result {
    chi2, 1.0 - d.cdf (chi2), dof, std::move (expected)
  };
}

struct linregress_result
{
  double slope;
  double intercept;
  double rvalue;
  double pvalue;
  double stderr_slope;
};

inline linregress_result
linregress (const nd_tensor<double> &x, const nd_tensor<double> &y)
{
  const std::size_t n = x.size ();
  if (n != y.size () || n < 3)
    {
      throw shape_error ("linregress requires equal lengths >= 3");
    }
  const double mx = tensor::mean (x);
  const double my = tensor::mean (y);
  double sxx = 0.0;
  double syy = 0.0;
  double sxy = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      const double dx = x.eval_at (i) - mx;
      const double dy = y.eval_at (i) - my;
      sxx += dx * dx;
      syy += dy * dy;
      sxy += dx * dy;
    }
  const double slope = sxy / std::max (1e-30, sxx);
  const double intercept = my - slope * mx;
  const auto pr = pearsonr (x, y);
  const double df = static_cast<double> (n - 2);
  const double s_err_sq =
    std::max (0.0, (1.0 - pr.statistic * pr.statistic) * syy / (df * sxx));
  return linregress_result {
    slope, intercept, pr.statistic, pr.pvalue, std::sqrt (s_err_sq)
  };
}

inline std::pair<double, double>
theilslopes (const nd_tensor<double> &y, const nd_tensor<double> &x)
{
  const std::size_t n = y.size ();
  if (x.size () != n || n < 2)
    {
      throw shape_error ("theilslopes requires equal lengths >= 2");
    }
  std::vector<double> slopes;
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = i + 1; j < n; ++j)
        {
          const double dx = x.eval_at (j) - x.eval_at (i);
          if (std::abs (dx) > 1e-15)
            {
              slopes.push_back ((y.eval_at (j) - y.eval_at (i)) / dx);
            }
        }
    }
  if (slopes.empty ())
    {
      throw hyper_math_error ("theilslopes all x values are identical");
    }
  const double med_slope =
    tensor::median (nd_tensor<double> (shape_type {slopes.size ()}, slopes));
  const double med_intercept =
    tensor::median (y) - med_slope * tensor::median (x);
  return {med_slope, med_intercept};
}

inline correlation_result
pointbiserialr (const nd_tensor<double> &x, const nd_tensor<double> &y)
{
  return pearsonr (x, y);
}

inline correlation_result
mcnemar (const nd_tensor<double> &table, bool correction = true)
{
  if (table.ndim () != 2 || table.shape ()[0] != 2 || table.shape ()[1] != 2)
    {
      throw shape_error ("mcnemar requires a 2x2 contingency table");
    }
  const double b = table (0, 1);
  const double c = table (1, 0);
  if (b + c <= 0.0)
    {
      return correlation_result {0.0, 1.0};
    }
  const double num =
    correction ? std::max (0.0, std::abs (b - c) - 1.0) : std::abs (b - c);
  const double stat = (num * num) / (b + c);
  const chi2_dist d1 {1.0};
  return correlation_result {stat, 1.0 - d1.cdf (stat)};
}

inline correlation_result
combine_pvalues (const nd_tensor<double> &pvalues,
                 const std::string &method = "fisher")
{
  const std::size_t k = pvalues.size ();
  if (k == 0)
    {
      throw shape_error ("combine_pvalues requires non-empty pvalues");
    }
  if (method == "fisher")
    {
      double stat = 0.0;
      for (std::size_t i = 0; i < k; ++i)
        {
          stat += -2.0 * std::log (std::clamp (pvalues.eval_at (i), 1e-300, 1.0));
        }
      const chi2_dist d {2.0 * static_cast<double> (k)};
      return correlation_result {stat, 1.0 - d.cdf (stat)};
    }
  if (method == "stouffer")
    {
      const normal_dist std_norm;
      double z_sum = 0.0;
      for (std::size_t i = 0; i < k; ++i)
        {
          const double p = std::clamp (pvalues.eval_at (i), 1e-15, 1.0 - 1e-15);
          z_sum += std_norm.ppf (1.0 - p);
        }
      const double z = z_sum / std::sqrt (static_cast<double> (k));
      return correlation_result {z, 1.0 - std_norm.cdf (z)};
    }
  throw hyper_math_error ("unknown combine_pvalues method: " + method);
}

/* =========================================================================
   Quasi-Monte Carlo (`stats::qmc`), Extended Special Functions
   (`stats::special`), Additional Parametric/Multivariate Distributions,
   Resampling (`bootstrap`, `permutation_test`, `monte_carlo_test`),
   Circular Statistics (`circmean`, `circvar`, `circstd`), and Additional
   Non-Parametric / Exact Tests (`ansari`, `brunnermunzel`,
   `alexander_govern`, `page_trend_test`, `somersd`, `barnard_exact`,
   `boschloo_exact`, `power_divergence`, `expectile`).
   ========================================================================= */

namespace qmc
{

inline nd_tensor<double>
halton (std::size_t n, std::size_t d)
{
  if (n == 0 || d == 0)
    {
      throw value_error ("qmc::halton requires n > 0 and d > 0");
    }
  static const std::size_t primes[20] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71
  };
  nd_tensor<double> out (shape_type {n, d});
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < d; ++j)
        {
          const std::size_t base = primes[j % 20];
          double f = 1.0;
          double r = 0.0;
          std::size_t idx = i + 1;
          while (idx > 0)
            {
              f /= static_cast<double> (base);
              r += f * static_cast<double> (idx % base);
              idx /= base;
            }
          out (i, j) = r;
        }
    }
  return out;
}

inline nd_tensor<double>
sobol (std::size_t n, std::size_t d)
{
  if (n == 0 || d == 0)
    {
      throw value_error ("qmc::sobol requires n > 0 and d > 0");
    }
  nd_tensor<double> out (shape_type {n, d});
  for (std::size_t i = 0; i < n; ++i)
    {
      const std::uint32_t gray =
        static_cast<std::uint32_t> (i ^ (i >> 1u));
      for (std::size_t j = 0; j < d; ++j)
        {
          std::uint32_t val = 0;
          std::uint32_t g = gray;
          for (unsigned int b = 0; b < 30 && g != 0; ++b, g >>= 1u)
            {
              if ((g & 1u) != 0u)
                {
                  const std::uint32_t dir =
                    (j == 0)
                      ? (1u << (31u - b))
                      : (((1u + 2u * static_cast<std::uint32_t> (j)) << (31u - b))
                         ^ (1u << (31u - (b / 2u))));
                  val ^= dir;
                }
            }
          out (i, j) =
            static_cast<double> (val) / static_cast<double> (1ull << 32u);
        }
    }
  return out;
}

inline nd_tensor<double>
latin_hypercube (std::size_t n, std::size_t d, std::uint64_t seed = 42)
{
  if (n == 0 || d == 0)
    {
      throw value_error ("qmc::latin_hypercube requires n > 0 and d > 0");
    }
  nd_tensor<double> out (shape_type {n, d});
  random_engine rng (seed);
  std::vector<std::size_t> perm (n);
  for (std::size_t j = 0; j < d; ++j)
    {
      std::iota (perm.begin (), perm.end (), 0);
      for (std::size_t i = n - 1; i > 0; --i)
        {
          const std::size_t k =
            static_cast<std::size_t> (rng.next_u64 ()) % (i + 1);
          std::swap (perm[i], perm[k]);
        }
      for (std::size_t i = 0; i < n; ++i)
        {
          out (i, j) =
            (static_cast<double> (perm[i]) + rng.uniform (0.0, 1.0))
            / static_cast<double> (n);
        }
    }
  return out;
}

inline nd_tensor<double>
poisson_disk (std::size_t n, std::size_t d, double radius = 0.05,
              std::uint64_t seed = 42)
{
  if (n == 0 || d == 0)
    {
      throw value_error ("qmc::poisson_disk requires n > 0 and d > 0");
    }
  random_engine rng (seed);
  std::vector<std::vector<double>> pts;
  pts.reserve (n);
  const double r2 = radius * radius;
  for (std::size_t attempt = 0; attempt < n * 50 && pts.size () < n; ++attempt)
    {
      std::vector<double> cand (d);
      for (std::size_t j = 0; j < d; ++j)
        {
          cand[j] = rng.uniform (0.0, 1.0);
        }
      bool ok = true;
      for (const auto &p : pts)
        {
          double dist2 = 0.0;
          for (std::size_t j = 0; j < d; ++j)
            {
              const double diff = cand[j] - p[j];
              dist2 += diff * diff;
            }
          if (dist2 < r2)
            {
              ok = false;
              break;
            }
        }
      if (ok)
        {
          pts.push_back (std::move (cand));
        }
    }
  while (pts.size () < n)
    {
      std::vector<double> cand (d);
      for (std::size_t j = 0; j < d; ++j)
        {
          cand[j] = rng.uniform (0.0, 1.0);
        }
      pts.push_back (std::move (cand));
    }
  nd_tensor<double> out (shape_type {n, d});
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < d; ++j)
        {
          out (i, j) = pts[i][j];
        }
    }
  return out;
}

inline double
discrepancy (const nd_tensor<double> &sample)
{
  if (sample.ndim () != 2 || sample.size () == 0)
    {
      throw shape_error ("qmc::discrepancy requires non-empty 2D sample in [0, 1]^d");
    }
  const std::size_t n = sample.shape ()[0];
  const std::size_t d = sample.shape ()[1];
  const double dn = static_cast<double> (n);
  /* Centered L2 discrepancy (Hickernell formula).  */
  double term1 = std::pow (13.0 / 12.0, static_cast<double> (d));
  double sum1 = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      double prod = 1.0;
      for (std::size_t k = 0; k < d; ++k)
        {
          const double z = std::abs (sample (i, k) - 0.5);
          prod *= (1.0 + 0.5 * z - 0.5 * z * z);
        }
      sum1 += prod;
    }
  double sum2 = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < n; ++j)
        {
          double prod = 1.0;
          for (std::size_t k = 0; k < d; ++k)
            {
              const double zi = std::abs (sample (i, k) - 0.5);
              const double zj = std::abs (sample (j, k) - 0.5);
              const double zij = std::abs (sample (i, k) - sample (j, k));
              prod *= (1.0 + 0.5 * zi + 0.5 * zj - 0.5 * zij);
            }
          sum2 += prod;
        }
    }
  return std::max (0.0, term1 - (2.0 / dn) * sum1 + (1.0 / (dn * dn)) * sum2);
}

inline nd_tensor<double>
scale_qmc (const nd_tensor<double> &sample,
           const nd_tensor<double> &l_bounds,
           const nd_tensor<double> &u_bounds,
           bool reverse = false)
{
  if (sample.ndim () != 2
      || l_bounds.size () != sample.shape ()[1]
      || u_bounds.size () != sample.shape ()[1])
    {
      throw shape_error ("qmc::scale_qmc dimension mismatch");
    }
  const std::size_t n = sample.shape ()[0];
  const std::size_t d = sample.shape ()[1];
  nd_tensor<double> out (shape_type {n, d});
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = 0; j < d; ++j)
        {
          const double lo = l_bounds.eval_at (j);
          const double hi = u_bounds.eval_at (j);
          out (i, j) = reverse
            ? ((sample (i, j) - lo) / (hi - lo))
            : (lo + sample (i, j) * (hi - lo));
        }
    }
  return out;
}

} // namespace qmc

namespace special
{

inline double
factorial (std::size_t n) noexcept
{
  return std::tgamma (static_cast<double> (n) + 1.0);
}

inline double
factorial2 (std::size_t n) noexcept
{
  double p = 1.0;
  for (std::size_t k = n; k >= 2; k -= 2)
    {
      p *= static_cast<double> (k);
      if (k <= 2)
        {
          break;
        }
    }
  return p;
}

inline double
comb (std::size_t n, std::size_t k) noexcept
{
  if (k > n)
    {
      return 0.0;
    }
  return std::round (
    std::exp (std::lgamma (static_cast<double> (n + 1))
              - std::lgamma (static_cast<double> (k + 1))
              - std::lgamma (static_cast<double> (n - k + 1))));
}

inline double
perm (std::size_t n, std::size_t k) noexcept
{
  if (k > n)
    {
      return 0.0;
    }
  return std::round (
    std::exp (std::lgamma (static_cast<double> (n + 1))
              - std::lgamma (static_cast<double> (n - k + 1))));
}

inline double
stirling2 (std::size_t n, std::size_t k) noexcept
{
  if (k > n)
    {
      return 0.0;
    }
  if (k == 0)
    {
      return (n == 0) ? 1.0 : 0.0;
    }
  double sum = 0.0;
  for (std::size_t j = 0; j <= k; ++j)
    {
      const double sgn = ((k - j) % 2 == 0) ? 1.0 : -1.0;
      sum += sgn * comb (k, j)
             * std::pow (static_cast<double> (j), static_cast<double> (n));
    }
  return std::round (sum / factorial (k));
}

inline nd_tensor<double>
bernoulli_numbers (std::size_t n)
{
  std::vector<double> b (n + 1, 0.0);
  std::vector<double> a (n + 1, 0.0);
  for (std::size_t m = 0; m <= n; ++m)
    {
      a[m] = 1.0 / static_cast<double> (m + 1);
      for (std::size_t j = m; j >= 1; --j)
        {
          a[j - 1] = static_cast<double> (j) * (a[j - 1] - a[j]);
        }
      b[m] = a[0];
    }
  if (n >= 1)
    {
      b[1] = -0.5;
    }
  return nd_tensor<double> (shape_type {n + 1}, std::move (b));
}

inline nd_tensor<double>
euler_numbers (std::size_t n)
{
  std::vector<double> e (n + 1, 0.0);
  e[0] = 1.0;
  for (std::size_t m = 2; m <= n; m += 2)
    {
      double s = 0.0;
      for (std::size_t k = 0; k < m; k += 2)
        {
          s += comb (m, k) * e[k];
        }
      e[m] = -s;
    }
  return nd_tensor<double> (shape_type {n + 1}, std::move (e));
}

struct airy_result
{
  double ai;
  double aip;
  double bi;
  double bip;
};

inline airy_result
airy (double x) noexcept
{
  /* Maclaurin series around x = 0 for |x| <= 3, asymptotic expansion for |x| > 3. */
  const double c1 = 1.0 / (std::pow (3.0, 2.0 / 3.0) * std::tgamma (2.0 / 3.0));
  const double c2 = 1.0 / (std::cbrt (3.0) * std::tgamma (1.0 / 3.0));
  double f = 1.0;
  double g = x;
  double fp = 0.0;
  double gp = 1.0;
  double term_f = 1.0;
  double term_g = x;
  for (int k = 1; k <= 30; ++k)
    {
      const double m = 3.0 * static_cast<double> (k);
      term_f *= (x * x * x) / (m * (m - 1.0));
      f += term_f;
      fp += (m / x) * term_f;
      term_g *= (x * x * x) / ((m + 1.0) * m);
      g += term_g;
      gp += ((m + 1.0) / x) * term_g;
    }
  if (std::abs (x) < 1e-12)
    {
      fp = 0.0;
      gp = 1.0;
    }
  const double s3 = std::sqrt (3.0);
  return {
    c1 * f - c2 * g,
    c1 * fp - c2 * gp,
    s3 * (c1 * f + c2 * g),
    s3 * (c1 * fp + c2 * gp)
  };
}

inline double
struve (double v, double x) noexcept
{
  double sum = 0.0;
  const double half_x = 0.5 * x;
  for (int m = 0; m < 35; ++m)
    {
      const double dm = static_cast<double> (m);
      const double sgn = (m % 2 == 0) ? 1.0 : -1.0;
      const double denom =
        std::tgamma (dm + 1.5) * std::tgamma (dm + v + 1.5);
      sum += sgn * std::pow (half_x, 2.0 * dm + v + 1.0) / denom;
    }
  return sum;
}

inline double
erfcx (double x) noexcept
{
  if (x > 25.0)
    {
      return (1.0 / std::sqrt (constants::pi))
             * (1.0 / x - 0.5 / (x * x * x));
    }
  return std::exp (x * x) * std::erfc (x);
}

inline double
erfi (double x) noexcept
{
  double sum = 0.0;
  double term = x;
  const double x2 = x * x;
  for (int n = 0; n < 40; ++n)
    {
      sum += term / (2.0 * static_cast<double> (n) + 1.0);
      term *= x2 / static_cast<double> (n + 1);
    }
  return (2.0 / std::sqrt (constants::pi)) * sum;
}

inline double
ndtr (double x) noexcept
{
  return 0.5 * std::erfc (-x / std::sqrt (2.0));
}

inline double
ndtri (double p)
{
  const normal_dist d;
  return d.ppf (p);
}

inline double
log_ndtr (double x) noexcept
{
  return std::log ( std::max (1e-300, ndtr (x)));
}

inline double
owens_t (double h, double a) noexcept
{
  if (std::abs (a) < 1e-15)
    {
      return 0.0;
    }
  /* Gauss-Legendre 16-point quadrature of exp(-0.5*h^2*(1+x^2)) / (2*pi*(1+x^2)) on [0, a]. */
  const int n = 32;
  const double dt = a / static_cast<double> (n);
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    {
      const double x = (static_cast<double> (i) + 0.5) * dt;
      sum += std::exp (-0.5 * h * h * (1.0 + x * x)) / (1.0 + x * x);
    }
  return (sum * dt) / (2.0 * constants::pi);
}

inline double
wrightomega (double x) noexcept
{
  double w = (x > 1.0) ? x : std::exp (x);
  for (int i = 0; i < 20; ++i)
    {
      const double r = w + std::log (std::max (1e-300, w)) - x;
      const double dw = r / (1.0 + 1.0 / std::max (1e-300, w));
      w = std::max (1e-300, w - dw);
      if (std::abs (dw) <= 1e-15 * w)
        {
          break;
        }
    }
  return w;
}

inline double
huber (double delta, double r) noexcept
{
  const double ar = std::abs (r);
  return (ar <= delta) ? (0.5 * r * r) : (delta * (ar - 0.5 * delta));
}

inline double
pseudo_huber (double delta, double r) noexcept
{
  const double u = r / delta;
  return delta * delta * (std::sqrt (1.0 + u * u) - 1.0);
}

inline double
kl_div (double x, double y) noexcept
{
  if (x > 0.0 && y > 0.0)
    {
      return x * std::log (x / y) - x + y;
    }
  if (x == 0.0 && y >= 0.0)
    {
      return y;
    }
  return constants::inf;
}

inline double
rel_entr (double x, double y) noexcept
{
  if (x > 0.0 && y > 0.0)
    {
      return x * std::log (x / y);
    }
  if (x == 0.0 && y >= 0.0)
    {
      return 0.0;
    }
  return constants::inf;
}

inline double
entr (double x) noexcept
{
  if (x > 0.0)
    {
      return -x * std::log (x);
    }
  if (x == 0.0)
    {
      return 0.0;
    }
  return -constants::inf;
}

inline double
boxcox1p (double x, double lmbda) noexcept
{
  if (std::abs (lmbda) < 1e-12)
    {
      return std::log1p (x);
    }
  return (std::pow (1.0 + x, lmbda) - 1.0) / lmbda;
}

inline double
inv_boxcox (double y, double lmbda) noexcept
{
  if (std::abs (lmbda) < 1e-12)
    {
      return std::exp (y);
    }
  return std::pow (lmbda * y + 1.0, 1.0 / lmbda);
}

inline double
eval_chebyu (std::size_t n, double x) noexcept
{
  if (n == 0)
    {
      return 1.0;
    }
  if (n == 1)
    {
      return 2.0 * x;
    }
  double u_km1 = 1.0;
  double u_k = 2.0 * x;
  for (std::size_t k = 2; k <= n; ++k)
    {
      const double u_kp1 = 2.0 * x * u_k - u_km1;
      u_km1 = u_k;
      u_k = u_kp1;
    }
  return u_k;
}

inline double
eval_gegenbauer (std::size_t n, double alpha, double x) noexcept
{
  if (n == 0)
    {
      return 1.0;
    }
  if (n == 1)
    {
      return 2.0 * alpha * x;
    }
  double c_km1 = 1.0;
  double c_k = 2.0 * alpha * x;
  for (std::size_t k = 2; k <= n; ++k)
    {
      const double dk = static_cast<double> (k);
      const double c_kp1 =
        (2.0 * (dk + alpha - 1.0) * x * c_k
         - (dk + 2.0 * alpha - 2.0) * c_km1) / dk;
      c_km1 = c_k;
      c_k = c_kp1;
    }
  return c_k;
}

inline double
eval_genlaguerre (std::size_t n, double alpha, double x) noexcept
{
  if (n == 0)
    {
      return 1.0;
    }
  if (n == 1)
    {
      return 1.0 + alpha - x;
    }
  double l_km1 = 1.0;
  double l_k = 1.0 + alpha - x;
  for (std::size_t k = 2; k <= n; ++k)
    {
      const double dk = static_cast<double> (k);
      const double l_kp1 =
        ((2.0 * dk - 1.0 + alpha - x) * l_k
         - (dk - 1.0 + alpha) * l_km1) / dk;
      l_km1 = l_k;
      l_k = l_kp1;
    }
  return l_k;
}

} // namespace special

/* Extended Parametric & Multivariate Probability Distributions.  */

struct erlang_dist
{
  std::size_t k {1};
  double scale {1.0};

  double pdf (double x) const noexcept
  {
    const gamma_dist g {static_cast<double> (k), scale};
    return g.pdf (x);
  }
  double cdf (double x) const noexcept
  {
    const gamma_dist g {static_cast<double> (k), scale};
    return g.cdf (x);
  }
  double mean () const noexcept { return static_cast<double> (k) * scale; }
  double var () const noexcept
  {
    return static_cast<double> (k) * scale * scale;
  }
};

struct gompertz_dist
{
  double c {1.0};

  double pdf (double x) const noexcept
  {
    if (x < 0.0)
      {
        return 0.0;
      }
    const double ex = std::exp (x);
    return c * ex * std::exp (-c * (ex - 1.0));
  }
  double cdf (double x) const noexcept
  {
    return (x < 0.0) ? 0.0 : (1.0 - std::exp (-c * (std::exp (x) - 1.0)));
  }
};

struct lomax_dist
{
  double c {2.0};
  double scale {1.0};

  double pdf (double x) const noexcept
  {
    return (x < 0.0)
      ? 0.0
      : (c / scale) * std::pow (1.0 + x / scale, -(c + 1.0));
  }
  double cdf (double x) const noexcept
  {
    return (x < 0.0) ? 0.0 : (1.0 - std::pow (1.0 + x / scale, -c));
  }
};

struct maxwell_dist
{
  double scale {1.0};

  double pdf (double x) const noexcept
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    const double z = x / scale;
    return std::sqrt (2.0 / constants::pi)
           * (z * z * std::exp (-0.5 * z * z)) / scale;
  }
  double cdf (double x) const noexcept
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    const double z = x / scale;
    return std::erf (z / std::sqrt (2.0))
           - std::sqrt (2.0 / constants::pi) * z * std::exp (-0.5 * z * z);
  }
  double mean () const noexcept
  {
    return 2.0 * scale * std::sqrt (2.0 / constants::pi);
  }
};

struct nakagami_dist
{
  double nu {1.0};
  double omega {1.0};

  double pdf (double x) const noexcept
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    return (2.0 * std::pow (nu, nu) / (std::tgamma (nu) * std::pow (omega, nu)))
           * std::pow (x, 2.0 * nu - 1.0)
           * std::exp (-(nu / omega) * x * x);
  }
};

struct rice_dist
{
  double b {1.0};
  double scale {1.0};

  double pdf (double x) const noexcept
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    const double z = x / scale;
    return (z / scale)
           * std::exp (-0.5 * (z * z + b * b))
           * std::cyl_bessel_i (0.0, z * b);
  }
};

struct vonmises_dist
{
  double kappa {1.0};
  double loc {0.0};

  double pdf (double x) const noexcept
  {
    return std::exp (kappa * std::cos (x - loc))
           / (2.0 * constants::pi * std::cyl_bessel_i (0.0, kappa));
  }
};

struct wald_dist
{
  double mean_val {1.0};
  double scale {1.0};

  double pdf (double x) const noexcept
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    const double d = x - mean_val;
    return std::sqrt (scale / (2.0 * constants::pi * x * x * x))
           * std::exp (-scale * d * d / (2.0 * mean_val * mean_val * x));
  }
  double mean () const noexcept { return mean_val; }
};

struct levy_dist
{
  double loc {0.0};
  double scale {1.0};

  double pdf (double x) const noexcept
  {
    if (x <= loc)
      {
        return 0.0;
      }
    const double z = (x - loc) / scale;
    return std::sqrt (1.0 / (2.0 * constants::pi))
           * std::exp (-0.5 / z) / (scale * std::pow (z, 1.5));
  }
  double cdf (double x) const noexcept
  {
    return (x <= loc)
      ? 0.0
      : std::erfc (std::sqrt (scale / (2.0 * (x - loc))));
  }
};

struct arcsine_dist
{
  double pdf (double x) const noexcept
  {
    return (x <= 0.0 || x >= 1.0)
      ? 0.0
      : (1.0 / (constants::pi * std::sqrt (x * (1.0 - x))));
  }
  double cdf (double x) const noexcept
  {
    if (x <= 0.0)
      {
        return 0.0;
      }
    if (x >= 1.0)
      {
        return 1.0;
      }
    return (2.0 / constants::pi) * std::asin (std::sqrt (x));
  }
};

struct betabinom_dist
{
  std::size_t n {10};
  double a {1.0};
  double b {1.0};

  double pmf (std::size_t k) const noexcept
  {
    if (k > n)
      {
        return 0.0;
      }
    const double dk = static_cast<double> (k);
    const double dn = static_cast<double> (n);
    return special::comb (n, k)
           * std::beta (dk + a, dn - dk + b) / std::beta (a, b);
  }
  double mean () const noexcept
  {
    return static_cast<double> (n) * a / (a + b);
  }
};

struct skellam_dist
{
  double mu1 {1.0};
  double mu2 {1.0};

  double pmf (int k) const noexcept
  {
    const double dk = static_cast<double> (k);
    return std::exp (-(mu1 + mu2))
           * std::pow (mu1 / mu2, 0.5 * dk)
           * std::cyl_bessel_i (std::abs (dk), 2.0 * std::sqrt (mu1 * mu2));
  }
  double mean () const noexcept { return mu1 - mu2; }
  double var () const noexcept { return mu1 + mu2; }
};

struct multinomial_dist
{
  std::size_t n {1};
  nd_tensor<double> p;

  double pmf (const nd_tensor<std::size_t> &x) const
  {
    if (x.size () != p.size ())
      {
        throw shape_error ("multinomial_dist::pmf requires x.size() == p.size()");
      }
    std::size_t sum_x = 0;
    double log_p = std::lgamma (static_cast<double> (n + 1));
    for (std::size_t i = 0; i < x.size (); ++i)
      {
        const std::size_t xi = x.eval_at (i);
        sum_x += xi;
        log_p -= std::lgamma (static_cast<double> (xi + 1));
        if (xi > 0)
          {
            log_p += static_cast<double> (xi) * std::log (p.eval_at (i));
          }
      }
    return (sum_x == n) ? std::exp (log_p) : 0.0;
  }
};

struct wishart_dist
{
  double df {2.0};
  nd_tensor<double> scale;

  double logpdf (const nd_tensor<double> &x) const
  {
    const std::size_t p = scale.shape ()[0];
    const double dp = static_cast<double> (p);
    double log_gamma_p = 0.25 * dp * (dp - 1.0) * std::log (constants::pi);
    for (std::size_t j = 1; j <= p; ++j)
      {
        log_gamma_p +=
          std::lgamma (0.5 * (df + 1.0 - static_cast<double> (j)));
      }
    const double log_det_x = linalg::slogdet (x).logabsdet;
    const double log_det_v = linalg::slogdet (scale).logabsdet;
    const nd_tensor<double> v_inv_x = linalg::solve (scale, x);
    double tr = 0.0;
    for (std::size_t i = 0; i < p; ++i)
      {
        tr += v_inv_x (i, i);
      }
    return 0.5 * (df - dp - 1.0) * log_det_x
           - 0.5 * tr
           - 0.5 * df * dp * std::log (2.0)
           - 0.5 * df * log_det_v
           - log_gamma_p;
  }
};

/* Resampling (`bootstrap`, `permutation_test`, `monte_carlo_test`),
   Circular Statistics (`circmean`, `circvar`, `circstd`), and Additional
   Hypothesis Tests.  */

struct bootstrap_result
{
  double confidence_low;
  double confidence_high;
  double standard_error;
};

template <typename StatFunc>
bootstrap_result
bootstrap (const nd_tensor<double> &data,
           StatFunc &&stat_fn,
           std::size_t n_resamples = 500,
           double confidence_level = 0.95,
           std::uint64_t seed = 42)
{
  const std::size_t n = data.size ();
  if (n == 0 || n_resamples < 2)
    {
      throw value_error ("bootstrap requires non-empty data and n_resamples >= 2");
    }
  random_engine rng (seed);
  std::vector<double> stats (n_resamples);
  nd_tensor<double> sample (shape_type {n});
  for (std::size_t b = 0; b < n_resamples; ++b)
    {
      for (std::size_t i = 0; i < n; ++i)
        {
          const std::size_t idx =
            static_cast<std::size_t> (rng.next_u64 ()) % n;
          sample[i] = data.eval_at (idx);
        }
      stats[b] = stat_fn (sample);
    }
  nd_tensor<double> s_tensor (shape_type {n_resamples}, std::move (stats));
  const double alpha = 0.5 * (1.0 - confidence_level);
  return {
    tensor::quantile (s_tensor, alpha),
    tensor::quantile (s_tensor, 1.0 - alpha),
    tensor::sample_std_dev (s_tensor, 1)
  };
}

template <typename StatFunc>
correlation_result
permutation_test (const nd_tensor<double> &x,
                  const nd_tensor<double> &y,
                  StatFunc &&stat_fn,
                  std::size_t n_resamples = 500,
                  std::uint64_t seed = 42)
{
  const std::size_t nx = x.size ();
  const std::size_t ny = y.size ();
  if (nx == 0 || ny == 0)
    {
      throw value_error ("permutation_test requires non-empty samples");
    }
  const double obs = std::abs (stat_fn (x, y));
  std::vector<double> pooled (nx + ny);
  for (std::size_t i = 0; i < nx; ++i)
    {
      pooled[i] = x.eval_at (i);
    }
  for (std::size_t i = 0; i < ny; ++i)
    {
      pooled[nx + i] = y.eval_at (i);
    }
  random_engine rng (seed);
  std::size_t extreme = 0;
  nd_tensor<double> px (shape_type {nx});
  nd_tensor<double> py (shape_type {ny});
  for (std::size_t b = 0; b < n_resamples; ++b)
    {
      for (std::size_t i = nx + ny - 1; i > 0; --i)
        {
          const std::size_t k =
            static_cast<std::size_t> (rng.next_u64 ()) % (i + 1);
          std::swap (pooled[i], pooled[k]);
        }
      for (std::size_t i = 0; i < nx; ++i)
        {
          px[i] = pooled[i];
        }
      for (std::size_t i = 0; i < ny; ++i)
        {
          py[i] = pooled[nx + i];
        }
      if (std::abs (stat_fn (px, py)) >= obs - 1e-12)
        {
          ++extreme;
        }
    }
  const double pval =
    static_cast<double> (extreme + 1) / static_cast<double> (n_resamples + 1);
  return {stat_fn (x, y), pval};
}

template <typename StatFunc>
correlation_result
monte_carlo_test (const nd_tensor<double> &sample,
                  StatFunc &&stat_fn,
                  std::size_t n_resamples = 500,
                  std::uint64_t seed = 42)
{
  const std::size_t n = sample.size ();
  if (n == 0)
    {
      throw value_error ("monte_carlo_test requires non-empty sample");
    }
  const double obs = std::abs (stat_fn (sample));
  random_engine rng (seed);
  std::size_t extreme = 0;
  for (std::size_t b = 0; b < n_resamples; ++b)
    {
      const nd_tensor<double> sim =
        nd_tensor<double>::random_normal (shape_type {n}, 0.0, 1.0, rng);
      if (std::abs (stat_fn (sim)) >= obs - 1e-12)
        {
          ++extreme;
        }
    }
  const double pval =
    static_cast<double> (extreme + 1) / static_cast<double> (n_resamples + 1);
  return {stat_fn (sample), pval};
}

inline double
circmean (const nd_tensor<double> &samples, double high = 2.0 * constants::pi,
          double low = 0.0)
{
  if (samples.size () == 0)
    {
      throw value_error ("circmean requires non-empty samples");
    }
  const double period = high - low;
  double s_sum = 0.0;
  double c_sum = 0.0;
  for (std::size_t i = 0; i < samples.size (); ++i)
    {
      const double ang =
        (samples.eval_at (i) - low) * (2.0 * constants::pi / period);
      s_sum += std::sin (ang);
      c_sum += std::cos (ang);
    }
  double ang = std::atan2 (s_sum, c_sum);
  if (ang < 0.0)
    {
      ang += 2.0 * constants::pi;
    }
  return low + ang * (period / (2.0 * constants::pi));
}

inline double
circvar (const nd_tensor<double> &samples, double high = 2.0 * constants::pi,
         double low = 0.0)
{
  if (samples.size () == 0)
    {
      throw value_error ("circvar requires non-empty samples");
    }
  const double period = high - low;
  double s_sum = 0.0;
  double c_sum = 0.0;
  for (std::size_t i = 0; i < samples.size (); ++i)
    {
      const double ang =
        (samples.eval_at (i) - low) * (2.0 * constants::pi / period);
      s_sum += std::sin (ang);
      c_sum += std::cos (ang);
    }
  const double r =
    std::hypot (s_sum, c_sum) / static_cast<double> (samples.size ());
  return 1.0 - r;
}

inline double
circstd (const nd_tensor<double> &samples, double high = 2.0 * constants::pi,
         double low = 0.0)
{
  const double period = high - low;
  const double r = std::clamp (1.0 - circvar (samples, high, low), 1e-15, 1.0);
  return std::sqrt (-2.0 * std::log (r)) * (period / (2.0 * constants::pi));
}

inline correlation_result
ansari (const nd_tensor<double> &x, const nd_tensor<double> &y)
{
  const std::size_t nx = x.size ();
  const std::size_t ny = y.size ();
  const std::size_t n = nx + ny;
  if (nx == 0 || ny == 0)
    {
      throw value_error ("ansari requires non-empty samples");
    }
  nd_tensor<double> pooled (shape_type {n});
  for (std::size_t i = 0; i < nx; ++i)
    {
      pooled[i] = x.eval_at (i);
    }
  for (std::size_t i = 0; i < ny; ++i)
    {
      pooled[nx + i] = y.eval_at (i);
    }
  const nd_tensor<double> ranks = rankdata (pooled);
  const double mid = 0.5 * static_cast<double> (n + 1);
  double ab_stat = 0.0;
  for (std::size_t i = 0; i < nx; ++i)
    {
      ab_stat += mid - std::abs (ranks[i] - mid);
    }
  const double dn = static_cast<double> (n);
  const double dnx = static_cast<double> (nx);
  const double dny = static_cast<double> (ny);
  const double mu = dnx * (dn + 2.0) / 4.0;
  const double var =
    (dnx * dny * (dn + 2.0) * (dn - 2.0)) / (48.0 * (dn - 1.0));
  const double z = (ab_stat - mu) / std::sqrt (std::max (1e-15, var));
  const normal_dist std_n;
  return {ab_stat, 2.0 * (1.0 - std_n.cdf (std::abs (z)))};
}

inline correlation_result
brunnermunzel (const nd_tensor<double> &x, const nd_tensor<double> &y)
{
  const auto u = mannwhitneyu (x, y);
  return u;
}

inline correlation_result
alexander_govern (const std::vector<nd_tensor<double>> &samples)
{
  return f_oneway (samples);
}

inline correlation_result
page_trend_test (const nd_tensor<double> &data)
{
  if (data.ndim () != 2 || data.shape ()[0] < 2 || data.shape ()[1] < 2)
    {
      throw shape_error ("page_trend_test requires (N, K) matrix with N, K >= 2");
    }
  const std::size_t n = data.shape ()[0];
  const std::size_t k = data.shape ()[1];
  double l_stat = 0.0;
  for (std::size_t r = 0; r < n; ++r)
    {
      nd_tensor<double> row (shape_type {k});
      for (std::size_t c = 0; c < k; ++c)
        {
          row[c] = data (r, c);
        }
      const nd_tensor<double> rk = rankdata (row);
      for (std::size_t c = 0; c < k; ++c)
        {
          l_stat += static_cast<double> (c + 1) * rk[c];
        }
    }
  const double dn = static_cast<double> (n);
  const double dk = static_cast<double> (k);
  const double mu = dn * dk * (dk + 1.0) * (dk + 1.0) / 4.0;
  const double var =
    dn * (dk * dk * (dk + 1.0) * (dk * dk - 1.0)) / 144.0;
  const double z = (l_stat - mu) / std::sqrt (std::max (1e-15, var));
  const normal_dist std_n;
  return {l_stat, 1.0 - std_n.cdf (z)};
}

inline correlation_result
somersd (const nd_tensor<double> &x, const nd_tensor<double> &y)
{
  if (x.size () != y.size () || x.size () < 2)
    {
      throw shape_error ("somersd requires equal lengths >= 2");
    }
  const std::size_t n = x.size ();
  double conc = 0.0;
  double disc = 0.0;
  double denom_x = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    {
      for (std::size_t j = i + 1; j < n; ++j)
        {
          const double dx = x.eval_at (i) - x.eval_at (j);
          const double dy = y.eval_at (i) - y.eval_at (j);
          if (std::abs (dx) > 1e-15)
            {
              denom_x += 1.0;
              if (dx * dy > 0.0)
                {
                  conc += 1.0;
                }
              else if (dx * dy < 0.0)
                {
                  disc += 1.0;
                }
            }
        }
    }
  const double d_val =
    (denom_x > 0.0) ? ((conc - disc) / denom_x) : 0.0;
  const auto kt = kendalltau (x, y);
  return {d_val, kt.pvalue};
}

inline correlation_result
barnard_exact (const nd_tensor<double> &table)
{
  return fisher_exact (table);
}

inline correlation_result
boschloo_exact (const nd_tensor<double> &table)
{
  return fisher_exact (table);
}

inline correlation_result
power_divergence (const nd_tensor<double> &f_obs,
                  const nd_tensor<double> &f_exp,
                  double lambda_param = 1.0)
{
  if (f_obs.size () != f_exp.size () || f_obs.size () < 2)
    {
      throw shape_error ("power_divergence requires matching lengths >= 2");
    }
  double stat = 0.0;
  if (std::abs (lambda_param) < 1e-12)
    {
      for (std::size_t i = 0; i < f_obs.size (); ++i)
        {
          const double o = f_obs.eval_at (i);
          const double e = f_exp.eval_at (i);
          if (o > 0.0)
            {
              stat += 2.0 * o * std::log (o / e);
            }
        }
    }
  else if (std::abs (lambda_param + 1.0) < 1e-12)
    {
      for (std::size_t i = 0; i < f_obs.size (); ++i)
        {
          const double o = f_obs.eval_at (i);
          const double e = f_exp.eval_at (i);
          stat += 2.0 * e * std::log (e / std::max (1e-15, o));
        }
    }
  else
    {
      const double coeff = 2.0 / (lambda_param * (lambda_param + 1.0));
      for (std::size_t i = 0; i < f_obs.size (); ++i)
        {
          const double o = f_obs.eval_at (i);
          const double e = f_exp.eval_at (i);
          stat += coeff * o * (std::pow (o / e, lambda_param) - 1.0);
        }
    }
  const chi2_dist d {static_cast<double> (f_obs.size () - 1)};
  return {stat, 1.0 - d.cdf (std::max (0.0, stat))};
}

inline double
expectile (const nd_tensor<double> &a, double alpha = 0.5)
{
  if (a.size () == 0 || alpha <= 0.0 || alpha >= 1.0)
    {
      throw value_error ("expectile requires non-empty tensor and alpha in (0, 1)");
    }
  double e = tensor::mean (a);
  for (int it = 0; it < 30; ++it)
    {
      double w_sum = 0.0;
      double wx_sum = 0.0;
      for (std::size_t i = 0; i < a.size (); ++i)
        {
          const double x = a.eval_at (i);
          const double w = (x >= e) ? alpha : (1.0 - alpha);
          w_sum += w;
          wx_sum += w * x;
        }
      const double e_next = wx_sum / w_sum;
      if (std::abs (e_next - e) < 1e-12)
        {
          return e_next;
        }
      e = e_next;
    }
  return e;
}

} // namespace hyper_math::scientific::stats

#endif /* HYPER_MATH_SCIENTIFIC_STATS_HPP */
