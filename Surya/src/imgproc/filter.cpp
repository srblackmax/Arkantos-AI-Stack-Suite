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

#include <surya/imgproc/filter.hpp>

#include <algorithm>
#include <array>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace surya::inline v1::imgproc
{

core::Result<std::vector<float>>
gaussian_kernel_1d (size_t ksize, float sigma)
{
  if (ksize == 0U || (ksize % 2U) == 0U)
    {
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "gaussian_kernel_1d: ksize must be a positive odd integer");
    }
  const float eff_sigma =
    sigma > 0.0F
      ? sigma
      : (0.3F * (static_cast<float> (ksize - 1U) * 0.5F - 1.0F) + 0.8F);
  const float inv_2s2 = 1.0F / (2.0F * eff_sigma * eff_sigma);
  const ptrdiff_t radius = static_cast<ptrdiff_t> (ksize / 2U);

  std::vector<float> kernel (ksize);
  float sum = 0.0F;
  for (ptrdiff_t i = -radius; i <= radius; ++i)
    {
      const float val =
        std::exp (-static_cast<float> (i * i) * inv_2s2);
      kernel[static_cast<size_t> (i + radius)] = val;
      sum += val;
    }
  if (sum > 0.0F)
    {
      const float inv_sum = 1.0F / sum;
      for (float &w : kernel)
        {
          w *= inv_sum;
        }
    }
  return kernel;
}

namespace
{

/* SIMD-accelerated vertical pass over float32 intermediate rows.  */
void
convolve_rows_vertical_f32 (float *SURYA_RESTRICT dst_row,
                            const float *const *row_ptrs,
                            const float *SURYA_RESTRICT ky,
                            size_t klen_y,
                            size_t num_elems) noexcept
{
  size_t x = 0U;
#if defined(__AVX2__)
  for (; x + 8U <= num_elems; x += 8U)
    {
      __m256 acc = _mm256_setzero_ps ();
      for (size_t k = 0U; k < klen_y; ++k)
        {
          const __m256 w = _mm256_set1_ps (ky[k]);
          const __m256 v = _mm256_loadu_ps (row_ptrs[k] + x);
#  if defined(__FMA__)
          acc = _mm256_fmadd_ps (v, w, acc);
#  else
          acc = _mm256_add_ps (acc, _mm256_mul_ps (v, w));
#  endif
        }
      _mm256_storeu_ps (dst_row + x, acc);
    }
#endif
  for (; x < num_elems; ++x)
    {
      float acc = 0.0F;
      for (size_t k = 0U; k < klen_y; ++k)
        {
          acc += row_ptrs[k][x] * ky[k];
        }
      dst_row[x] = acc;
    }
}

template <typename SrcT, size_t C>
void
convolve_horizontal_row (
  float *SURYA_RESTRICT dst_row,
  const SrcT *SURYA_RESTRICT src_row,
  size_t width,
  std::span<const float> kx,
  core::Border_Mode border) noexcept
{
  const ptrdiff_t rx = static_cast<ptrdiff_t> (kx.size () / 2U);
  const ptrdiff_t w = static_cast<ptrdiff_t> (width);
  const size_t klen = kx.size ();

  /* Interior range [rx, w - rx) requires no border checks.  */
  const ptrdiff_t x_left_end = std::min (rx, w);
  const ptrdiff_t x_right_start = std::max (x_left_end, w - rx);

  for (ptrdiff_t x = 0; x < x_left_end; ++x)
    {
      for (size_t ch = 0U; ch < C; ++ch)
        {
          float sum = 0.0F;
          for (ptrdiff_t k = -rx; k <= rx; ++k)
            {
              const ptrdiff_t sx =
                core::border_interpolate (x + k, w, border);
              const float val =
                sx >= 0
                  ? static_cast<float> (
                      src_row[static_cast<size_t> (sx) * C + ch])
                  : 0.0F;
              sum += val * kx[static_cast<size_t> (k + rx)];
            }
          dst_row[static_cast<size_t> (x) * C + ch] = sum;
        }
    }

  /* Vectorized/unrolled interior loop.  */
  for (ptrdiff_t x = x_left_end; x < x_right_start; ++x)
    {
      const size_t ux = static_cast<size_t> (x);
      for (size_t ch = 0U; ch < C; ++ch)
        {
          float sum = 0.0F;
          const SrcT *base =
            src_row + static_cast<size_t> (x - rx) * C + ch;
          for (size_t k = 0U; k < klen; ++k)
            {
              sum += static_cast<float> (base[k * C]) * kx[k];
            }
          dst_row[ux * C + ch] = sum;
        }
    }

  for (ptrdiff_t x = x_right_start; x < w; ++x)
    {
      for (size_t ch = 0U; ch < C; ++ch)
        {
          float sum = 0.0F;
          for (ptrdiff_t k = -rx; k <= rx; ++k)
            {
              const ptrdiff_t sx =
                core::border_interpolate (x + k, w, border);
              const float val =
                sx >= 0
                  ? static_cast<float> (
                      src_row[static_cast<size_t> (sx) * C + ch])
                  : 0.0F;
              sum += val * kx[static_cast<size_t> (k + rx)];
            }
          dst_row[static_cast<size_t> (x) * C + ch] = sum;
        }
    }
}

} /* namespace */

core::Result<core::Image_F32C1>
separable_filter_2d (core::Const_View_F32C1 src,
                     std::span<const float> kx,
                     std::span<const float> ky,
                     core::Border_Mode border)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "separable_filter_2d: input is empty");
    }
  if (kx.empty () || (kx.size () % 2U) == 0U || ky.empty ()
      || (ky.size () % 2U) == 0U)
    {
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "separable_filter_2d: kernels must have odd non-zero length");
    }

  auto tmp_res =
    core::Image_F32C1::create (src.width (), src.height (), false);
  auto dst_res =
    core::Image_F32C1::create (src.width (), src.height (), false);
  if (!tmp_res.has_value () || !dst_res.has_value ())
    {
      return core::make_error (core::Error_Code::Allocation_Failed,
                               "separable_filter_2d: allocation failed");
    }
  core::Image_F32C1 tmp = std::move (*tmp_res);
  core::Image_F32C1 dst = std::move (*dst_res);

  core::parallel_for_1d (
    0U, src.height (), 16U,
    [&] (size_t y0, size_t y1)
    {
      for (size_t y = y0; y < y1; ++y)
        {
          convolve_horizontal_row<float, 1U> (
            tmp.row_ptr (y), src.row_ptr (y), src.width (), kx, border);
        }
    });

  const ptrdiff_t ry = static_cast<ptrdiff_t> (ky.size () / 2U);
  const ptrdiff_t h = static_cast<ptrdiff_t> (src.height ());
  const std::vector<float> zero_row (src.width (), 0.0F);

  core::parallel_for_1d (
    0U, src.height (), 16U,
    [&] (size_t y0, size_t y1)
    {
      std::vector<const float *> row_ptrs (ky.size ());
      for (size_t y = y0; y < y1; ++y)
        {
          const ptrdiff_t py = static_cast<ptrdiff_t> (y);
          for (ptrdiff_t k = -ry; k <= ry; ++k)
            {
              const ptrdiff_t sy =
                core::border_interpolate (py + k, h, border);
              row_ptrs[static_cast<size_t> (k + ry)] =
                sy >= 0 ? tmp.row_ptr (static_cast<size_t> (sy))
                        : zero_row.data ();
            }
          convolve_rows_vertical_f32 (dst.row_ptr (y), row_ptrs.data (),
                                      ky.data (), ky.size (), src.width ());
        }
    });

  return dst;
}

core::Result<core::Image_U8C1>
separable_filter_2d (core::Const_View_U8C1 src,
                     std::span<const float> kx,
                     std::span<const float> ky,
                     core::Border_Mode border)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "separable_filter_2d: input is empty");
    }
  if (kx.empty () || (kx.size () % 2U) == 0U || ky.empty ()
      || (ky.size () % 2U) == 0U)
    {
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "separable_filter_2d: kernels must have odd non-zero length");
    }

  auto tmp_res =
    core::Image_F32C1::create (src.width (), src.height (), false);
  auto dst_res =
    core::Image_U8C1::create (src.width (), src.height (), false);
  if (!tmp_res.has_value () || !dst_res.has_value ())
    {
      return core::make_error (core::Error_Code::Allocation_Failed,
                               "separable_filter_2d: allocation failed");
    }
  core::Image_F32C1 tmp = std::move (*tmp_res);
  core::Image_U8C1 dst = std::move (*dst_res);

  core::parallel_for_1d (
    0U, src.height (), 16U,
    [&] (size_t y0, size_t y1)
    {
      for (size_t y = y0; y < y1; ++y)
        {
          convolve_horizontal_row<uint8_t, 1U> (
            tmp.row_ptr (y), src.row_ptr (y), src.width (), kx, border);
        }
    });

  const ptrdiff_t ry = static_cast<ptrdiff_t> (ky.size () / 2U);
  const ptrdiff_t h = static_cast<ptrdiff_t> (src.height ());
  const std::vector<float> zero_row (src.width (), 0.0F);

  core::parallel_for_1d (
    0U, src.height (), 16U,
    [&] (size_t y0, size_t y1)
    {
      std::vector<const float *> row_ptrs (ky.size ());
      std::vector<float> out_buf (src.width ());
      for (size_t y = y0; y < y1; ++y)
        {
          const ptrdiff_t py = static_cast<ptrdiff_t> (y);
          for (ptrdiff_t k = -ry; k <= ry; ++k)
            {
              const ptrdiff_t sy =
                core::border_interpolate (py + k, h, border);
              row_ptrs[static_cast<size_t> (k + ry)] =
                sy >= 0 ? tmp.row_ptr (static_cast<size_t> (sy))
                        : zero_row.data ();
            }
          convolve_rows_vertical_f32 (out_buf.data (), row_ptrs.data (),
                                      ky.data (), ky.size (), src.width ());
          uint8_t *d = dst.row_ptr (y);
          for (size_t x = 0U; x < src.width (); ++x)
            {
              d[x] = core::saturate_cast<uint8_t> (out_buf[x]);
            }
        }
    });

  return dst;
}

core::Result<core::Image_U8C3>
separable_filter_2d (core::Const_View_U8C3 src,
                     std::span<const float> kx,
                     std::span<const float> ky,
                     core::Border_Mode border)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "separable_filter_2d: input is empty");
    }
  if (kx.empty () || (kx.size () % 2U) == 0U || ky.empty ()
      || (ky.size () % 2U) == 0U)
    {
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "separable_filter_2d: kernels must have odd non-zero length");
    }

  auto tmp_res =
    core::Image_F32C3::create (src.width (), src.height (), false);
  auto dst_res =
    core::Image_U8C3::create (src.width (), src.height (), false);
  if (!tmp_res.has_value () || !dst_res.has_value ())
    {
      return core::make_error (core::Error_Code::Allocation_Failed,
                               "separable_filter_2d: allocation failed");
    }
  core::Image_F32C3 tmp = std::move (*tmp_res);
  core::Image_U8C3 dst = std::move (*dst_res);

  core::parallel_for_1d (
    0U, src.height (), 16U,
    [&] (size_t y0, size_t y1)
    {
      for (size_t y = y0; y < y1; ++y)
        {
          convolve_horizontal_row<uint8_t, 3U> (
            tmp.row_ptr (y), src.row_ptr (y), src.width (), kx, border);
        }
    });

  const size_t row_elems = src.width () * 3U;
  const ptrdiff_t ry = static_cast<ptrdiff_t> (ky.size () / 2U);
  const ptrdiff_t h = static_cast<ptrdiff_t> (src.height ());
  const std::vector<float> zero_row (row_elems, 0.0F);

  core::parallel_for_1d (
    0U, src.height (), 16U,
    [&] (size_t y0, size_t y1)
    {
      std::vector<const float *> row_ptrs (ky.size ());
      std::vector<float> out_buf (row_elems);
      for (size_t y = y0; y < y1; ++y)
        {
          const ptrdiff_t py = static_cast<ptrdiff_t> (y);
          for (ptrdiff_t k = -ry; k <= ry; ++k)
            {
              const ptrdiff_t sy =
                core::border_interpolate (py + k, h, border);
              row_ptrs[static_cast<size_t> (k + ry)] =
                sy >= 0 ? tmp.row_ptr (static_cast<size_t> (sy))
                        : zero_row.data ();
            }
          convolve_rows_vertical_f32 (out_buf.data (), row_ptrs.data (),
                                      ky.data (), ky.size (), row_elems);
          uint8_t *d = dst.row_ptr (y);
          for (size_t i = 0U; i < row_elems; ++i)
            {
              d[i] = core::saturate_cast<uint8_t> (out_buf[i]);
            }
        }
    });

  return dst;
}

core::Result<core::Image_U8C1>
gaussian_blur (core::Const_View_U8C1 src,
               size_t ksize,
               float sigma,
               core::Border_Mode border)
{
  auto k_res = gaussian_kernel_1d (ksize, sigma);
  if (!k_res.has_value ())
    {
      return std::unexpected (k_res.error ());
    }
  return separable_filter_2d (src, *k_res, *k_res, border);
}

core::Result<core::Image_U8C3>
gaussian_blur (core::Const_View_U8C3 src,
               size_t ksize,
               float sigma,
               core::Border_Mode border)
{
  auto k_res = gaussian_kernel_1d (ksize, sigma);
  if (!k_res.has_value ())
    {
      return std::unexpected (k_res.error ());
    }
  return separable_filter_2d (src, *k_res, *k_res, border);
}

core::Result<core::Image_F32C1>
gaussian_blur (core::Const_View_F32C1 src,
               size_t ksize,
               float sigma,
               core::Border_Mode border)
{
  auto k_res = gaussian_kernel_1d (ksize, sigma);
  if (!k_res.has_value ())
    {
      return std::unexpected (k_res.error ());
    }
  return separable_filter_2d (src, *k_res, *k_res, border);
}

core::Result<core::Image_I32C1>
integral_image (core::Const_View_U8C1 src)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "integral_image: input is empty");
    }
  auto sat_res =
    core::Image_I32C1::create (src.width () + 1U, src.height () + 1U, true);
  if (!sat_res.has_value ())
    {
      return std::unexpected (sat_res.error ());
    }
  core::Image_I32C1 sat = std::move (*sat_res);

  for (size_t y = 0U; y < src.height (); ++y)
    {
      const uint8_t *s_row = src.row_ptr (y);
      const int32_t *prev_sat = sat.row_ptr (y);
      int32_t *cur_sat = sat.row_ptr (y + 1U);
      int32_t row_sum = 0;
      cur_sat[0] = 0;
      for (size_t x = 0U; x < src.width (); ++x)
        {
          row_sum += static_cast<int32_t> (s_row[x]);
          cur_sat[x + 1U] = prev_sat[x + 1U] + row_sum;
        }
    }
  return sat;
}

core::Result<core::Image_U8C1>
box_filter (core::Const_View_U8C1 src, size_t kwidth, size_t kheight)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "box_filter: input is empty");
    }
  if (kwidth == 0U || (kwidth % 2U) == 0U || kheight == 0U
      || (kheight % 2U) == 0U)
    {
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "box_filter: kernel dimensions must be positive odd integers");
    }

  auto sat_res = integral_image (src);
  if (!sat_res.has_value ())
    {
      return std::unexpected (sat_res.error ());
    }
  const core::Image_I32C1 &sat = *sat_res;
  auto sat_v = sat.const_view ();

  auto dst_res = core::Image_U8C1::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C1 dst = std::move (*dst_res);

  const ptrdiff_t rx = static_cast<ptrdiff_t> (kwidth / 2U);
  const ptrdiff_t ry = static_cast<ptrdiff_t> (kheight / 2U);
  const ptrdiff_t w = static_cast<ptrdiff_t> (src.width ());
  const ptrdiff_t h = static_cast<ptrdiff_t> (src.height ());

  for (ptrdiff_t y = 0; y < h; ++y)
    {
      const size_t y0 = static_cast<size_t> (std::max<ptrdiff_t> (0, y - ry));
      const size_t y1 =
        static_cast<size_t> (std::min<ptrdiff_t> (h, y + ry + 1));
      uint8_t *d_row = dst.row_ptr (static_cast<size_t> (y));
      for (ptrdiff_t x = 0; x < w; ++x)
        {
          const size_t x0 =
            static_cast<size_t> (std::max<ptrdiff_t> (0, x - rx));
          const size_t x1 =
            static_cast<size_t> (std::min<ptrdiff_t> (w, x + rx + 1));
          const int32_t area =
            static_cast<int32_t> ((y1 - y0) * (x1 - x0));
          const int32_t sum = integral_box_sum (sat_v, x0, y0, x1, y1);
          d_row[static_cast<size_t> (x)] =
            static_cast<uint8_t> ((sum + area / 2) / area);
        }
    }
  return dst;
}

core::Result<core::Image_F32C1>
sobel_3x3 (core::Const_View_U8C1 src,
           int dx_order,
           int dy_order,
           core::Border_Mode border)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "sobel_3x3: input is empty");
    }
  if ((dx_order != 1 || dy_order != 0) && (dx_order != 0 || dy_order != 1))
    {
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "sobel_3x3: (dx_order, dy_order) must be (1, 0) or (0, 1)");
    }

  constexpr std::array<float, 3U> k_deriv = {-1.0F, 0.0F, 1.0F};
  constexpr std::array<float, 3U> k_smooth = {1.0F, 2.0F, 1.0F};

  auto f_res = core::Image_F32C1::create (src.width (), src.height (), false);
  if (!f_res.has_value ())
    {
      return std::unexpected (f_res.error ());
    }
  core::Image_F32C1 f_img = std::move (*f_res);
  for (size_t y = 0U; y < src.height (); ++y)
    {
      const uint8_t *s = src.row_ptr (y);
      float *d = f_img.row_ptr (y);
      for (size_t x = 0U; x < src.width (); ++x)
        {
          d[x] = static_cast<float> (s[x]);
        }
    }

  if (dx_order == 1)
    {
      return separable_filter_2d (f_img.const_view (), k_deriv, k_smooth,
                                  border);
    }
  return separable_filter_2d (f_img.const_view (), k_smooth, k_deriv, border);
}

core::Result<core::Image_F32C1>
scharr_3x3 (core::Const_View_U8C1 src,
            int dx_order,
            int dy_order,
            core::Border_Mode border)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "scharr_3x3: input is empty");
    }
  if ((dx_order != 1 || dy_order != 0) && (dx_order != 0 || dy_order != 1))
    {
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "scharr_3x3: (dx_order, dy_order) must be (1, 0) or (0, 1)");
    }

  constexpr std::array<float, 3U> k_deriv = {-1.0F, 0.0F, 1.0F};
  constexpr std::array<float, 3U> k_smooth = {3.0F, 10.0F, 3.0F};

  auto f_res = core::Image_F32C1::create (src.width (), src.height (), false);
  if (!f_res.has_value ())
    {
      return std::unexpected (f_res.error ());
    }
  core::Image_F32C1 f_img = std::move (*f_res);
  for (size_t y = 0U; y < src.height (); ++y)
    {
      const uint8_t *s = src.row_ptr (y);
      float *d = f_img.row_ptr (y);
      for (size_t x = 0U; x < src.width (); ++x)
        {
          d[x] = static_cast<float> (s[x]);
        }
    }

  if (dx_order == 1)
    {
      return separable_filter_2d (f_img.const_view (), k_deriv, k_smooth,
                                  border);
    }
  return separable_filter_2d (f_img.const_view (), k_smooth, k_deriv, border);
}

core::Result<core::Image_F32C1>
laplacian_3x3 (core::Const_View_U8C1 src, core::Border_Mode border)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "laplacian_3x3: input is empty");
    }
  auto dst_res =
    core::Image_F32C1::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_F32C1 dst = std::move (*dst_res);

  const ptrdiff_t w = static_cast<ptrdiff_t> (src.width ());
  const ptrdiff_t h = static_cast<ptrdiff_t> (src.height ());

  auto sample = [&] (ptrdiff_t y, ptrdiff_t x) noexcept -> float
  {
    const ptrdiff_t sy = core::border_interpolate (y, h, border);
    const ptrdiff_t sx = core::border_interpolate (x, w, border);
    if (sy < 0 || sx < 0)
      {
        return 0.0F;
      }
    return static_cast<float> (
      src.at (static_cast<size_t> (sy), static_cast<size_t> (sx)));
  };

  for (ptrdiff_t y = 0; y < h; ++y)
    {
      float *d_row = dst.row_ptr (static_cast<size_t> (y));
      for (ptrdiff_t x = 0; x < w; ++x)
        {
          const float c = sample (y, x);
          const float n = sample (y - 1, x);
          const float s = sample (y + 1, x);
          const float e = sample (y, x + 1);
          const float wst = sample (y, x - 1);
          d_row[static_cast<size_t> (x)] = (n + s + e + wst) - 4.0F * c;
        }
    }
  return dst;
}

core::Result<core::Image_U8C1>
median_filter_3x3 (core::Const_View_U8C1 src)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "median_filter_3x3: input is empty");
    }
  auto dst_res = core::Image_U8C1::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C1 dst = std::move (*dst_res);

  const ptrdiff_t w = static_cast<ptrdiff_t> (src.width ());
  const ptrdiff_t h = static_cast<ptrdiff_t> (src.height ());

  auto cswap = [] (uint8_t &a, uint8_t &b) noexcept
  {
    const uint8_t lo = std::min (a, b);
    const uint8_t hi = std::max (a, b);
    a = lo;
    b = hi;
  };

  for (ptrdiff_t y = 0; y < h; ++y)
    {
      const size_t y0 =
        static_cast<size_t> (core::border_interpolate (
          y - 1, h, core::Border_Mode::Clamp));
      const size_t y1 = static_cast<size_t> (y);
      const size_t y2 =
        static_cast<size_t> (core::border_interpolate (
          y + 1, h, core::Border_Mode::Clamp));

      const uint8_t *r0 = src.row_ptr (y0);
      const uint8_t *r1 = src.row_ptr (y1);
      const uint8_t *r2 = src.row_ptr (y2);
      uint8_t *d = dst.row_ptr (y1);

      for (ptrdiff_t x = 0; x < w; ++x)
        {
          const size_t x0 =
            static_cast<size_t> (core::border_interpolate (
              x - 1, w, core::Border_Mode::Clamp));
          const size_t x1 = static_cast<size_t> (x);
          const size_t x2 =
            static_cast<size_t> (core::border_interpolate (
              x + 1, w, core::Border_Mode::Clamp));

          uint8_t p[9] = {r0[x0], r0[x1], r0[x2],
                          r1[x0], r1[x1], r1[x2],
                          r2[x0], r2[x1], r2[x2]};

          /* Optimal 19-comparator sorting network for median of 9.  */
          cswap (p[1], p[2]);
          cswap (p[4], p[5]);
          cswap (p[7], p[8]);
          cswap (p[0], p[1]);
          cswap (p[3], p[4]);
          cswap (p[6], p[7]);
          cswap (p[1], p[2]);
          cswap (p[4], p[5]);
          cswap (p[7], p[8]);
          cswap (p[0], p[3]);
          cswap (p[5], p[8]);
          cswap (p[4], p[7]);
          cswap (p[3], p[6]);
          cswap (p[1], p[4]);
          cswap (p[2], p[5]);
          cswap (p[4], p[7]);
          cswap (p[4], p[2]);
          cswap (p[6], p[4]);
          cswap (p[4], p[2]);

          d[x1] = p[4];
        }
    }
  return dst;
}

core::Result<core::Image_U8C1>
bilateral_filter (core::Const_View_U8C1 src,
                  size_t diameter,
                  float sigma_color,
                  float sigma_space)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "bilateral_filter: input is empty");
    }
  if (diameter == 0U || (diameter % 2U) == 0U || sigma_color <= 0.0F
      || sigma_space <= 0.0F)
    {
      return core::make_error (
        core::Error_Code::Invalid_Argument,
        "bilateral_filter: invalid diameter or sigma parameters");
    }

  auto dst_res = core::Image_U8C1::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C1 dst = std::move (*dst_res);

  std::array<float, 256U> color_lut {};
  const float inv_2sc2 = -1.0F / (2.0F * sigma_color * sigma_color);
  for (size_t d = 0U; d < 256U; ++d)
    {
      const float fd = static_cast<float> (d);
      color_lut[d] = std::exp (fd * fd * inv_2sc2);
    }

  const ptrdiff_t r = static_cast<ptrdiff_t> (diameter / 2U);
  const float inv_2ss2 = -1.0F / (2.0F * sigma_space * sigma_space);
  std::vector<float> space_w (diameter * diameter);
  for (ptrdiff_t dy = -r; dy <= r; ++dy)
    {
      for (ptrdiff_t dx = -r; dx <= r; ++dx)
        {
          const float dist2 = static_cast<float> (dx * dx + dy * dy);
          const size_t idx =
            static_cast<size_t> (dy + r) * diameter
            + static_cast<size_t> (dx + r);
          space_w[idx] = std::exp (dist2 * inv_2ss2);
        }
    }

  const ptrdiff_t w = static_cast<ptrdiff_t> (src.width ());
  const ptrdiff_t h = static_cast<ptrdiff_t> (src.height ());

  core::parallel_for_1d (
    0U, src.height (), 8U,
    [&] (size_t y0, size_t y1)
    {
      for (size_t y = y0; y < y1; ++y)
        {
          const ptrdiff_t py = static_cast<ptrdiff_t> (y);
          uint8_t *d_row = dst.row_ptr (y);
          const uint8_t *center_row = src.row_ptr (y);

          for (ptrdiff_t px = 0; px < w; ++px)
            {
              const int center_val =
                static_cast<int> (center_row[static_cast<size_t> (px)]);
              float sum_val = 0.0F;
              float sum_w = 0.0F;

              for (ptrdiff_t dy = -r; dy <= r; ++dy)
                {
                  const ptrdiff_t sy = core::border_interpolate (
                    py + dy, h, core::Border_Mode::Reflect_101);
                  const uint8_t *s_row =
                    src.row_ptr (static_cast<size_t> (sy));
                  const size_t wy_off =
                    static_cast<size_t> (dy + r) * diameter;

                  for (ptrdiff_t dx = -r; dx <= r; ++dx)
                    {
                      const ptrdiff_t sx = core::border_interpolate (
                        px + dx, w, core::Border_Mode::Reflect_101);
                      const int v =
                        static_cast<int> (s_row[static_cast<size_t> (sx)]);
                      const size_t diff =
                        static_cast<size_t> (std::abs (v - center_val));
                      const float weight =
                        space_w[wy_off + static_cast<size_t> (dx + r)]
                        * color_lut[diff];
                      sum_val += static_cast<float> (v) * weight;
                      sum_w += weight;
                    }
                }
              d_row[static_cast<size_t> (px)] =
                core::saturate_cast<uint8_t> (sum_val / sum_w);
            }
        }
    });

  return dst;
}

} /* namespace surya::inline v1::imgproc */
