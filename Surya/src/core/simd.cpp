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

#include <surya/core/simd.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#  include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#endif

namespace surya::inline v1::core
{

Simd_Level
detect_simd_level (void) noexcept
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#  if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init ();
  if (__builtin_cpu_supports ("avx512f")
      && __builtin_cpu_supports ("avx512bw"))
    {
      return Simd_Level::AVX512;
    }
  if (__builtin_cpu_supports ("avx2"))
    {
      return Simd_Level::AVX2;
    }
  if (__builtin_cpu_supports ("sse4.2"))
    {
      return Simd_Level::SSE4_2;
    }
#  endif
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  return Simd_Level::NEON;
#endif
  return Simd_Level::Scalar;
}

float
simd_dot_f32_scalar (const float *SURYA_RESTRICT a,
                     const float *SURYA_RESTRICT b,
                     size_t n) noexcept
{
  double sum = 0.0;
  for (size_t i = 0U; i < n; ++i)
    {
      sum += static_cast<double> (a[i]) * static_cast<double> (b[i]);
    }
  return static_cast<float> (sum);
}

namespace
{

#if (defined(__x86_64__) || defined(_M_X64)) \
    && (defined(__GNUC__) || defined(__clang__))

__attribute__ ((target ("avx512f"))) float
simd_dot_f32_avx512 (const float *SURYA_RESTRICT a,
                     const float *SURYA_RESTRICT b,
                     size_t n) noexcept
{
  alignas (64) float stage_a[16];
  alignas (64) float stage_b[16];
  __m512 acc = _mm512_setzero_ps ();
  size_t i = 0U;
  for (; i + 16U <= n; i += 16U)
    {
      std::memcpy (stage_a, a + i, 16U * sizeof (float));
      std::memcpy (stage_b, b + i, 16U * sizeof (float));
      const __m512 va = _mm512_load_ps (stage_a);
      const __m512 vb = _mm512_load_ps (stage_b);
      acc = _mm512_fmadd_ps (va, vb, acc);
    }
  float sum = _mm512_reduce_add_ps (acc);
  for (; i < n; ++i)
    {
      sum += a[i] * b[i];
    }
  return sum;
}

__attribute__ ((target ("avx2,fma"))) float
simd_dot_f32_avx2 (const float *SURYA_RESTRICT a,
                   const float *SURYA_RESTRICT b,
                   size_t n) noexcept
{
  alignas (64) float stage_a[16];
  alignas (64) float stage_b[16];
  __m256 acc0 = _mm256_setzero_ps ();
  __m256 acc1 = _mm256_setzero_ps ();
  size_t i = 0U;
  for (; i + 16U <= n; i += 16U)
    {
      std::memcpy (stage_a, a + i, 16U * sizeof (float));
      std::memcpy (stage_b, b + i, 16U * sizeof (float));
      const __m256 va0 = _mm256_load_ps (stage_a);
      const __m256 vb0 = _mm256_load_ps (stage_b);
      const __m256 va1 = _mm256_load_ps (stage_a + 8U);
      const __m256 vb1 = _mm256_load_ps (stage_b + 8U);
      acc0 = _mm256_fmadd_ps (va0, vb0, acc0);
      acc1 = _mm256_fmadd_ps (va1, vb1, acc1);
    }
  acc0 = _mm256_add_ps (acc0, acc1);
  alignas (64) float lanes[8];
  _mm256_store_ps (lanes, acc0);
  double sum = 0.0;
  for (size_t k = 0U; k < 8U; ++k)
    {
      sum += static_cast<double> (lanes[k]);
    }
  for (; i < n; ++i)
    {
      sum += static_cast<double> (a[i]) * static_cast<double> (b[i]);
    }
  return static_cast<float> (sum);
}

__attribute__ ((target ("avx512f"))) void
simd_conv1d_f32_avx512 (float *SURYA_RESTRICT dst,
                        const float *SURYA_RESTRICT src,
                        const float *SURYA_RESTRICT kernel,
                        size_t width,
                        size_t ksize) noexcept
{
  alignas (64) float stage_s[16];
  alignas (64) float stage_out[16];
  size_t x = 0U;
  for (; x + 16U <= width; x += 16U)
    {
      __m512 acc = _mm512_setzero_ps ();
      for (size_t k = 0U; k < ksize; ++k)
        {
          std::memcpy (stage_s, src + x + k, 16U * sizeof (float));
          const __m512 vs = _mm512_load_ps (stage_s);
          const __m512 vk = _mm512_set1_ps (kernel[k]);
          acc = _mm512_fmadd_ps (vs, vk, acc);
        }
      _mm512_store_ps (stage_out, acc);
      std::memcpy (dst + x, stage_out, 16U * sizeof (float));
    }
  for (; x < width; ++x)
    {
      double sum = 0.0;
      for (size_t k = 0U; k < ksize; ++k)
        {
          sum += static_cast<double> (src[x + k])
                 * static_cast<double> (kernel[k]);
        }
      dst[x] = static_cast<float> (sum);
    }
}

__attribute__ ((target ("avx2,fma"))) void
simd_conv1d_f32_avx2 (float *SURYA_RESTRICT dst,
                      const float *SURYA_RESTRICT src,
                      const float *SURYA_RESTRICT kernel,
                      size_t width,
                      size_t ksize) noexcept
{
  alignas (64) float stage_s[16];
  alignas (64) float stage_out[16];
  size_t x = 0U;
  for (; x + 16U <= width; x += 16U)
    {
      __m256 acc0 = _mm256_setzero_ps ();
      __m256 acc1 = _mm256_setzero_ps ();
      for (size_t k = 0U; k < ksize; ++k)
        {
          std::memcpy (stage_s, src + x + k, 16U * sizeof (float));
          const __m256 vs0 = _mm256_load_ps (stage_s);
          const __m256 vs1 = _mm256_load_ps (stage_s + 8U);
          const __m256 vk = _mm256_set1_ps (kernel[k]);
          acc0 = _mm256_fmadd_ps (vs0, vk, acc0);
          acc1 = _mm256_fmadd_ps (vs1, vk, acc1);
        }
      _mm256_store_ps (stage_out, acc0);
      _mm256_store_ps (stage_out + 8U, acc1);
      std::memcpy (dst + x, stage_out, 16U * sizeof (float));
    }
  for (; x < width; ++x)
    {
      double sum = 0.0;
      for (size_t k = 0U; k < ksize; ++k)
        {
          sum += static_cast<double> (src[x + k])
                 * static_cast<double> (kernel[k]);
        }
      dst[x] = static_cast<float> (sum);
    }
}

__attribute__ ((target ("avx512f"))) void
simd_ssim_moments_avx512 (const float *SURYA_RESTRICT a,
                          const float *SURYA_RESTRICT b,
                          size_t n,
                          double &sum_a,
                          double &sum_b,
                          double &sum_aa,
                          double &sum_bb,
                          double &sum_ab) noexcept
{
  alignas (64) float sa[16];
  alignas (64) float sb[16];
  __m512 va_sum = _mm512_setzero_ps ();
  __m512 vb_sum = _mm512_setzero_ps ();
  __m512 vaa_sum = _mm512_setzero_ps ();
  __m512 vbb_sum = _mm512_setzero_ps ();
  __m512 vab_sum = _mm512_setzero_ps ();

  size_t i = 0U;
  for (; i + 16U <= n; i += 16U)
    {
      std::memcpy (sa, a + i, 16U * sizeof (float));
      std::memcpy (sb, b + i, 16U * sizeof (float));
      const __m512 va = _mm512_load_ps (sa);
      const __m512 vb = _mm512_load_ps (sb);
      va_sum = _mm512_add_ps (va_sum, va);
      vb_sum = _mm512_add_ps (vb_sum, vb);
      vaa_sum = _mm512_fmadd_ps (va, va, vaa_sum);
      vbb_sum = _mm512_fmadd_ps (vb, vb, vbb_sum);
      vab_sum = _mm512_fmadd_ps (va, vb, vab_sum);
    }
  sum_a = static_cast<double> (_mm512_reduce_add_ps (va_sum));
  sum_b = static_cast<double> (_mm512_reduce_add_ps (vb_sum));
  sum_aa = static_cast<double> (_mm512_reduce_add_ps (vaa_sum));
  sum_bb = static_cast<double> (_mm512_reduce_add_ps (vbb_sum));
  sum_ab = static_cast<double> (_mm512_reduce_add_ps (vab_sum));

  for (; i < n; ++i)
    {
      const double da = static_cast<double> (a[i]);
      const double db = static_cast<double> (b[i]);
      sum_a += da;
      sum_b += db;
      sum_aa += da * da;
      sum_bb += db * db;
      sum_ab += da * db;
    }
}

__attribute__ ((target ("avx2,fma"))) void
simd_ssim_moments_avx2 (const float *SURYA_RESTRICT a,
                        const float *SURYA_RESTRICT b,
                        size_t n,
                        double &sum_a,
                        double &sum_b,
                        double &sum_aa,
                        double &sum_bb,
                        double &sum_ab) noexcept
{
  alignas (64) float sa[16];
  alignas (64) float sb[16];
  __m256 va_sum = _mm256_setzero_ps ();
  __m256 vb_sum = _mm256_setzero_ps ();
  __m256 vaa_sum = _mm256_setzero_ps ();
  __m256 vbb_sum = _mm256_setzero_ps ();
  __m256 vab_sum = _mm256_setzero_ps ();

  size_t i = 0U;
  for (; i + 8U <= n; i += 8U)
    {
      std::memcpy (sa, a + i, 8U * sizeof (float));
      std::memcpy (sb, b + i, 8U * sizeof (float));
      const __m256 va = _mm256_load_ps (sa);
      const __m256 vb = _mm256_load_ps (sb);
      va_sum = _mm256_add_ps (va_sum, va);
      vb_sum = _mm256_add_ps (vb_sum, vb);
      vaa_sum = _mm256_fmadd_ps (va, va, vaa_sum);
      vbb_sum = _mm256_fmadd_ps (vb, vb, vbb_sum);
      vab_sum = _mm256_fmadd_ps (va, vb, vab_sum);
    }

  alignas (64) float la[8];
  alignas (64) float lb[8];
  alignas (64) float laa[8];
  alignas (64) float lbb[8];
  alignas (64) float lab[8];
  _mm256_store_ps (la, va_sum);
  _mm256_store_ps (lb, vb_sum);
  _mm256_store_ps (laa, vaa_sum);
  _mm256_store_ps (lbb, vbb_sum);
  _mm256_store_ps (lab, vab_sum);

  sum_a = 0.0;
  sum_b = 0.0;
  sum_aa = 0.0;
  sum_bb = 0.0;
  sum_ab = 0.0;
  for (size_t k = 0U; k < 8U; ++k)
    {
      sum_a += static_cast<double> (la[k]);
      sum_b += static_cast<double> (lb[k]);
      sum_aa += static_cast<double> (laa[k]);
      sum_bb += static_cast<double> (lbb[k]);
      sum_ab += static_cast<double> (lab[k]);
    }
  for (; i < n; ++i)
    {
      const double da = static_cast<double> (a[i]);
      const double db = static_cast<double> (b[i]);
      sum_a += da;
      sum_b += db;
      sum_aa += da * da;
      sum_bb += db * db;
      sum_ab += da * db;
    }
}

#endif

float
compute_ssim_from_moments (double sum_a,
                           double sum_b,
                           double sum_aa,
                           double sum_bb,
                           double sum_ab,
                           size_t n,
                           float dynamic_range) noexcept
{
  const double inv_n = 1.0 / static_cast<double> (n);
  const double mu_a = sum_a * inv_n;
  const double mu_b = sum_b * inv_n;
  const double var_a = std::max (0.0, sum_aa * inv_n - mu_a * mu_a);
  const double var_b = std::max (0.0, sum_bb * inv_n - mu_b * mu_b);
  const double cov_ab = sum_ab * inv_n - mu_a * mu_b;

  const double c1 =
    (0.01 * static_cast<double> (dynamic_range))
    * (0.01 * static_cast<double> (dynamic_range));
  const double c2 =
    (0.03 * static_cast<double> (dynamic_range))
    * (0.03 * static_cast<double> (dynamic_range));

  const double num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2);
  const double den =
    (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2);
  return (den > 1e-12) ? static_cast<float> (num / den) : 1.0F;
}

} /* namespace */

float
simd_dot_f32 (const float *SURYA_RESTRICT a,
              const float *SURYA_RESTRICT b,
              size_t n) noexcept
{
#if (defined(__x86_64__) || defined(_M_X64)) \
    && (defined(__GNUC__) || defined(__clang__))
  const Simd_Level lvl = detect_simd_level ();
  if (lvl == Simd_Level::AVX512)
    {
      return simd_dot_f32_avx512 (a, b, n);
    }
  if (lvl == Simd_Level::AVX2)
    {
      return simd_dot_f32_avx2 (a, b, n);
    }
#endif
  return simd_dot_f32_scalar (a, b, n);
}

void
simd_conv1d_f32_scalar (float *SURYA_RESTRICT dst,
                        const float *SURYA_RESTRICT src,
                        const float *SURYA_RESTRICT kernel,
                        size_t width,
                        size_t ksize) noexcept
{
  for (size_t x = 0U; x < width; ++x)
    {
      double sum = 0.0;
      for (size_t k = 0U; k < ksize; ++k)
        {
          sum += static_cast<double> (src[x + k])
                 * static_cast<double> (kernel[k]);
        }
      dst[x] = static_cast<float> (sum);
    }
}

void
simd_conv1d_f32 (float *SURYA_RESTRICT dst,
                 const float *SURYA_RESTRICT src,
                 const float *SURYA_RESTRICT kernel,
                 size_t width,
                 size_t ksize) noexcept
{
#if (defined(__x86_64__) || defined(_M_X64)) \
    && (defined(__GNUC__) || defined(__clang__))
  const Simd_Level lvl = detect_simd_level ();
  if (lvl == Simd_Level::AVX512)
    {
      simd_conv1d_f32_avx512 (dst, src, kernel, width, ksize);
      return;
    }
  if (lvl == Simd_Level::AVX2)
    {
      simd_conv1d_f32_avx2 (dst, src, kernel, width, ksize);
      return;
    }
#endif
  simd_conv1d_f32_scalar (dst, src, kernel, width, ksize);
}

float
simd_ssim_f32_scalar (const float *SURYA_RESTRICT a,
                      const float *SURYA_RESTRICT b,
                      size_t n,
                      float dynamic_range) noexcept
{
  if (n == 0U)
    {
      return 1.0F;
    }
  double sum_a = 0.0;
  double sum_b = 0.0;
  double sum_aa = 0.0;
  double sum_bb = 0.0;
  double sum_ab = 0.0;
  for (size_t i = 0U; i < n; ++i)
    {
      const double da = static_cast<double> (a[i]);
      const double db = static_cast<double> (b[i]);
      sum_a += da;
      sum_b += db;
      sum_aa += da * da;
      sum_bb += db * db;
      sum_ab += da * db;
    }
  return compute_ssim_from_moments (
    sum_a, sum_b, sum_aa, sum_bb, sum_ab, n, dynamic_range);
}

float
simd_ssim_f32 (const float *SURYA_RESTRICT a,
               const float *SURYA_RESTRICT b,
               size_t n,
               float dynamic_range) noexcept
{
  if (n == 0U)
    {
      return 1.0F;
    }
#if (defined(__x86_64__) || defined(_M_X64)) \
    && (defined(__GNUC__) || defined(__clang__))
  const Simd_Level lvl = detect_simd_level ();
  double sum_a = 0.0;
  double sum_b = 0.0;
  double sum_aa = 0.0;
  double sum_bb = 0.0;
  double sum_ab = 0.0;
  if (lvl == Simd_Level::AVX512)
    {
      simd_ssim_moments_avx512 (
        a, b, n, sum_a, sum_b, sum_aa, sum_bb, sum_ab);
      return compute_ssim_from_moments (
        sum_a, sum_b, sum_aa, sum_bb, sum_ab, n, dynamic_range);
    }
  if (lvl == Simd_Level::AVX2)
    {
      simd_ssim_moments_avx2 (
        a, b, n, sum_a, sum_b, sum_aa, sum_bb, sum_ab);
      return compute_ssim_from_moments (
        sum_a, sum_b, sum_aa, sum_bb, sum_ab, n, dynamic_range);
    }
#endif
  return simd_ssim_f32_scalar (a, b, n, dynamic_range);
}

void
simd_abs_diff_u8 (uint8_t *SURYA_RESTRICT dst,
                  const uint8_t *SURYA_RESTRICT a,
                  const uint8_t *SURYA_RESTRICT b,
                  size_t n) noexcept
{
  size_t i = 0U;
#if defined(__AVX512F__) && defined(__AVX512BW__)
  for (; i + 64U <= n; i += 64U)
    {
      const __m512i va = _mm512_loadu_si512 (a + i);
      const __m512i vb = _mm512_loadu_si512 (b + i);
      const __m512i d1 = _mm512_subs_epu8 (va, vb);
      const __m512i d2 = _mm512_subs_epu8 (vb, va);
      const __m512i res = _mm512_or_si512 (d1, d2);
      _mm512_storeu_si512 (dst + i, res);
    }
#elif defined(__AVX2__)
  for (; i + 32U <= n; i += 32U)
    {
      const __m256i va =
        _mm256_loadu_si256 (reinterpret_cast<const __m256i *> (a + i));
      const __m256i vb =
        _mm256_loadu_si256 (reinterpret_cast<const __m256i *> (b + i));
      const __m256i d1 = _mm256_subs_epu8 (va, vb);
      const __m256i d2 = _mm256_subs_epu8 (vb, va);
      const __m256i res = _mm256_or_si256 (d1, d2);
      _mm256_storeu_si256 (reinterpret_cast<__m256i *> (dst + i), res);
    }
#endif
  for (; i < n; ++i)
    {
      const int diff = static_cast<int> (a[i]) - static_cast<int> (b[i]);
      dst[i] = static_cast<uint8_t> (diff < 0 ? -diff : diff);
    }
}

void
simd_add_sat_u8 (uint8_t *SURYA_RESTRICT dst,
                 const uint8_t *SURYA_RESTRICT a,
                 const uint8_t *SURYA_RESTRICT b,
                 size_t n) noexcept
{
  size_t i = 0U;
#if defined(__AVX512F__) && defined(__AVX512BW__)
  for (; i + 64U <= n; i += 64U)
    {
      const __m512i va = _mm512_loadu_si512 (a + i);
      const __m512i vb = _mm512_loadu_si512 (b + i);
      const __m512i res = _mm512_adds_epu8 (va, vb);
      _mm512_storeu_si512 (dst + i, res);
    }
#elif defined(__AVX2__)
  for (; i + 32U <= n; i += 32U)
    {
      const __m256i va =
        _mm256_loadu_si256 (reinterpret_cast<const __m256i *> (a + i));
      const __m256i vb =
        _mm256_loadu_si256 (reinterpret_cast<const __m256i *> (b + i));
      const __m256i res = _mm256_adds_epu8 (va, vb);
      _mm256_storeu_si256 (reinterpret_cast<__m256i *> (dst + i), res);
    }
#endif
  for (; i < n; ++i)
    {
      const unsigned int s =
        static_cast<unsigned int> (a[i]) + static_cast<unsigned int> (b[i]);
      dst[i] = static_cast<uint8_t> (s > 255U ? 255U : s);
    }
}

void
simd_scale_bias_u8_to_f32 (float *SURYA_RESTRICT dst,
                           const uint8_t *SURYA_RESTRICT x,
                           size_t n,
                           float scale,
                           float bias) noexcept
{
  size_t i = 0U;
#if defined(__AVX512F__)
  const __m512 vscale = _mm512_set1_ps (scale);
  const __m512 vbias = _mm512_set1_ps (bias);
  for (; i + 16U <= n; i += 16U)
    {
      const __m128i bytes =
        _mm_loadu_si128 (reinterpret_cast<const __m128i *> (x + i));
      const __m512i ints = _mm512_cvtepu8_epi32 (bytes);
      const __m512 flts = _mm512_cvtepi32_ps (ints);
      const __m512 out = _mm512_fmadd_ps (flts, vscale, vbias);
      _mm512_storeu_ps (dst + i, out);
    }
#elif defined(__AVX2__)
  const __m256 vscale = _mm256_set1_ps (scale);
  const __m256 vbias = _mm256_set1_ps (bias);
  for (; i + 8U <= n; i += 8U)
    {
      const __m128i bytes =
        _mm_loadl_epi64 (reinterpret_cast<const __m128i *> (x + i));
      const __m256i ints = _mm256_cvtepu8_epi32 (bytes);
      const __m256 flts = _mm256_cvtepi32_ps (ints);
#  if defined(__FMA__)
      const __m256 out = _mm256_fmadd_ps (flts, vscale, vbias);
#  else
      const __m256 out = _mm256_add_ps (_mm256_mul_ps (flts, vscale), vbias);
#  endif
      _mm256_storeu_ps (dst + i, out);
    }
#endif
  for (; i < n; ++i)
    {
      dst[i] = static_cast<float> (x[i]) * scale + bias;
    }
}

} /* namespace surya::inline v1::core */
