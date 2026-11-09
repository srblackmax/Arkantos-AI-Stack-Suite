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

#include <surya/imgproc/geometry.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace surya::inline v1::imgproc
{

Homography_3x3
Homography_3x3::rotation_scale_around (core::Point2f center,
                                       double angle_deg,
                                       double scale) noexcept
{
  const double rad = angle_deg * (std::numbers::pi / 180.0);
  const double alpha = scale * std::cos (rad);
  const double beta = scale * std::sin (rad);
  const double cx = static_cast<double> (center.x);
  const double cy = static_cast<double> (center.y);

  return Homography_3x3 (
    alpha, beta, (1.0 - alpha) * cx - beta * cy,
    -beta, alpha, beta * cx + (1.0 - alpha) * cy,
    0.0, 0.0, 1.0);
}

core::Result<Homography_3x3>
Homography_3x3::inverse (void) const noexcept
{
  const double c00 = m[4] * m[8] - m[5] * m[7];
  const double c01 = -(m[3] * m[8] - m[5] * m[6]);
  const double c02 = m[3] * m[7] - m[4] * m[6];

  const double det = m[0] * c00 + m[1] * c01 + m[2] * c02;
  if (std::abs (det) < 1e-14)
    {
      return core::make_error (
        core::Error_Code::Numerical_Failure,
        "Homography_3x3::inverse: singular transformation matrix");
    }

  const double inv_det = 1.0 / det;
  const double c10 = -(m[1] * m[8] - m[2] * m[7]);
  const double c11 = m[0] * m[8] - m[2] * m[6];
  const double c12 = -(m[0] * m[7] - m[1] * m[6]);

  const double c20 = m[1] * m[5] - m[2] * m[4];
  const double c21 = -(m[0] * m[5] - m[2] * m[3]);
  const double c22 = m[0] * m[4] - m[1] * m[3];

  return Homography_3x3 (
    c00 * inv_det, c10 * inv_det, c20 * inv_det,
    c01 * inv_det, c11 * inv_det, c21 * inv_det,
    c02 * inv_det, c12 * inv_det, c22 * inv_det);
}

namespace
{

struct Bilinear_Tab_Entry
{
  size_t i0;
  size_t i1;
  float w0;
  float w1;
};

std::vector<Bilinear_Tab_Entry>
build_bilinear_table (size_t src_len, size_t dst_len)
{
  std::vector<Bilinear_Tab_Entry> tab (dst_len);
  const float scale =
    static_cast<float> (src_len) / static_cast<float> (dst_len);
  const ptrdiff_t max_idx = static_cast<ptrdiff_t> (src_len) - 1;

  for (size_t d = 0U; d < dst_len; ++d)
    {
      float fx = (static_cast<float> (d) + 0.5F) * scale - 0.5F;
      ptrdiff_t sx = static_cast<ptrdiff_t> (std::floor (fx));
      float w1 = fx - static_cast<float> (sx);
      if (sx < 0)
        {
          sx = 0;
          w1 = 0.0F;
        }
      if (sx >= max_idx)
        {
          sx = max_idx;
          w1 = 0.0F;
        }
      const size_t i0 = static_cast<size_t> (sx);
      const size_t i1 =
        static_cast<size_t> (std::min (sx + 1, max_idx));
      tab[d] = Bilinear_Tab_Entry {i0, i1, 1.0F - w1, w1};
    }
  return tab;
}

inline void
cubic_weights (float f, float w[4]) noexcept
{
  /* Keys bicubic spline with a = -0.75.  */
  constexpr float a = -0.75F;
  w[0] = ((a * (f + 1.0F) - 5.0F * a) * (f + 1.0F) + 8.0F * a) * (f + 1.0F)
         - 4.0F * a;
  w[1] = ((a + 2.0F) * f - (a + 3.0F)) * f * f + 1.0F;
  w[2] = ((a + 2.0F) * (1.0F - f) - (a + 3.0F)) * (1.0F - f) * (1.0F - f)
         + 1.0F;
  w[3] = 1.0F - w[0] - w[1] - w[2];
}

template <typename T, size_t C>
core::Result<core::Image<T, C>>
resize_impl (core::Image_View<const T, C> src,
             size_t dst_w,
             size_t dst_h,
             Interpolation_Mode mode)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "resize: source image is empty");
    }
  if (dst_w == 0U || dst_h == 0U)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "resize: destination dimensions must be > 0");
    }

  auto dst_res = core::Image<T, C>::create (dst_w, dst_h, false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image<T, C> dst = std::move (*dst_res);

  const size_t src_w = src.width ();
  const size_t src_h = src.height ();

  if (mode == Interpolation_Mode::Nearest)
    {
      std::vector<size_t> x_tab (dst_w);
      for (size_t x = 0U; x < dst_w; ++x)
        {
          x_tab[x] = std::min ((x * src_w) / dst_w, src_w - 1U);
        }
      for (size_t y = 0U; y < dst_h; ++y)
        {
          const size_t sy = std::min ((y * src_h) / dst_h, src_h - 1U);
          const T *s_row = src.row_ptr (sy);
          T *d_row = dst.row_ptr (y);
          for (size_t x = 0U; x < dst_w; ++x)
            {
              const size_t sx = x_tab[x];
              for (size_t ch = 0U; ch < C; ++ch)
                {
                  d_row[x * C + ch] = s_row[sx * C + ch];
                }
            }
        }
      return dst;
    }

  if (mode == Interpolation_Mode::Bicubic)
    {
      const float scale_x =
        static_cast<float> (src_w) / static_cast<float> (dst_w);
      const float scale_y =
        static_cast<float> (src_h) / static_cast<float> (dst_h);
      const ptrdiff_t w = static_cast<ptrdiff_t> (src_w);
      const ptrdiff_t h = static_cast<ptrdiff_t> (src_h);

      for (size_t dy = 0U; dy < dst_h; ++dy)
        {
          const float fy = (static_cast<float> (dy) + 0.5F) * scale_y - 0.5F;
          const ptrdiff_t iy = static_cast<ptrdiff_t> (std::floor (fy));
          float wy[4];
          cubic_weights (fy - static_cast<float> (iy), wy);

          const T *rows[4];
          for (int m = -1; m <= 2; ++m)
            {
              const ptrdiff_t sy =
                core::border_interpolate (iy + m, h, core::Border_Mode::Clamp);
              rows[m + 1] = src.row_ptr (static_cast<size_t> (sy));
            }

          T *d_row = dst.row_ptr (dy);
          for (size_t dx = 0U; dx < dst_w; ++dx)
            {
              const float fx =
                (static_cast<float> (dx) + 0.5F) * scale_x - 0.5F;
              const ptrdiff_t ix = static_cast<ptrdiff_t> (std::floor (fx));
              float wx[4];
              cubic_weights (fx - static_cast<float> (ix), wx);

              size_t xs[4];
              for (int n = -1; n <= 2; ++n)
                {
                  xs[n + 1] = static_cast<size_t> (
                    core::border_interpolate (ix + n, w,
                                              core::Border_Mode::Clamp));
                }

              for (size_t ch = 0U; ch < C; ++ch)
                {
                  float sum = 0.0F;
                  for (int m = 0; m < 4; ++m)
                    {
                      const float row_val =
                        static_cast<float> (rows[m][xs[0] * C + ch]) * wx[0]
                        + static_cast<float> (rows[m][xs[1] * C + ch]) * wx[1]
                        + static_cast<float> (rows[m][xs[2] * C + ch]) * wx[2]
                        + static_cast<float> (rows[m][xs[3] * C + ch]) * wx[3];
                      sum += row_val * wy[m];
                    }
                  d_row[dx * C + ch] = core::saturate_cast<T> (sum);
                }
            }
        }
      return dst;
    }

  /* Bilinear and Area modes use precomputed horizontal/vertical tables.  */
  const auto x_tab = build_bilinear_table (src_w, dst_w);
  const auto y_tab = build_bilinear_table (src_h, dst_h);

  core::parallel_for_1d (
    0U, dst_h, 16U,
    [&] (size_t y0, size_t y1)
    {
      for (size_t dy = y0; dy < y1; ++dy)
        {
          const auto &yt = y_tab[dy];
          const T *r0 = src.row_ptr (yt.i0);
          const T *r1 = src.row_ptr (yt.i1);
          T *d_row = dst.row_ptr (dy);

          for (size_t dx = 0U; dx < dst_w; ++dx)
            {
              const auto &xt = x_tab[dx];
              const size_t x0 = xt.i0 * C;
              const size_t x1 = xt.i1 * C;
              for (size_t ch = 0U; ch < C; ++ch)
                {
                  const float v0 =
                    static_cast<float> (r0[x0 + ch]) * xt.w0
                    + static_cast<float> (r0[x1 + ch]) * xt.w1;
                  const float v1 =
                    static_cast<float> (r1[x0 + ch]) * xt.w0
                    + static_cast<float> (r1[x1 + ch]) * xt.w1;
                  d_row[dx * C + ch] =
                    core::saturate_cast<T> (v0 * yt.w0 + v1 * yt.w1);
                }
            }
        }
    });

  return dst;
}

template <size_t C>
core::Result<core::Image<uint8_t, C>>
warp_projective_impl (core::Image_View<const uint8_t, C> src,
                      const Homography_3x3 &forward_h,
                      size_t dst_w,
                      size_t dst_h,
                      Interpolation_Mode mode,
                      core::Border_Mode border,
                      core::Scalar<uint8_t, C> border_val,
                      bool is_affine)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "warp: source image is empty");
    }
  if (dst_w == 0U || dst_h == 0U)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "warp: destination dimensions must be > 0");
    }

  auto inv_res = forward_h.inverse ();
  if (!inv_res.has_value ())
    {
      return std::unexpected (inv_res.error ());
    }
  const Homography_3x3 inv = *inv_res;

  auto dst_res = core::Image<uint8_t, C>::create (dst_w, dst_h, false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image<uint8_t, C> dst = std::move (*dst_res);

  const ptrdiff_t sw = static_cast<ptrdiff_t> (src.width ());
  const ptrdiff_t sh = static_cast<ptrdiff_t> (src.height ());

  auto sample_pixel = [&] (ptrdiff_t y, ptrdiff_t x,
                           size_t ch) noexcept -> float
  {
    const ptrdiff_t sy = core::border_interpolate (y, sh, border);
    const ptrdiff_t sx = core::border_interpolate (x, sw, border);
    if (sy < 0 || sx < 0)
      {
        return static_cast<float> (border_val[ch]);
      }
    return static_cast<float> (
      src.at (static_cast<size_t> (sy), static_cast<size_t> (sx), ch));
  };

  core::parallel_for_1d (
    0U, dst_h, 16U,
    [&] (size_t y0, size_t y1)
    {
      for (size_t dy = y0; dy < y1; ++dy)
        {
          uint8_t *d_row = dst.row_ptr (dy);
          const double y_d = static_cast<double> (dy);

          for (size_t dx = 0U; dx < dst_w; ++dx)
            {
              const double x_d = static_cast<double> (dx);
              double sx = inv.m[0] * x_d + inv.m[1] * y_d + inv.m[2];
              double sy = inv.m[3] * x_d + inv.m[4] * y_d + inv.m[5];
              if (!is_affine)
                {
                  const double w =
                    inv.m[6] * x_d + inv.m[7] * y_d + inv.m[8];
                  const double inv_w =
                    std::abs (w) > 1e-12 ? (1.0 / w) : 1.0;
                  sx *= inv_w;
                  sy *= inv_w;
                }

              if (mode == Interpolation_Mode::Nearest)
                {
                  const ptrdiff_t ix =
                    static_cast<ptrdiff_t> (std::llround (sx));
                  const ptrdiff_t iy =
                    static_cast<ptrdiff_t> (std::llround (sy));
                  for (size_t ch = 0U; ch < C; ++ch)
                    {
                      const float samp = sample_pixel (iy, ix, ch);
                      d_row[dx * C + ch] =
                        core::saturate_cast<uint8_t> (samp);
                    }
                }
              else
                {
                  const ptrdiff_t x0 =
                    static_cast<ptrdiff_t> (std::floor (sx));
                  const ptrdiff_t y0_p =
                    static_cast<ptrdiff_t> (std::floor (sy));
                  const float fx =
                    static_cast<float> (sx - static_cast<double> (x0));
                  const float fy =
                    static_cast<float> (sy - static_cast<double> (y0_p));
                  const float wx0 = 1.0F - fx;
                  const float wy0 = 1.0F - fy;

                  /* Fast path when [x0, x0+1] x [y0_p, y0_p+1] is strictly
                     inside the source image.  */
                  if (x0 >= 0 && x0 + 1 < sw && y0_p >= 0 && y0_p + 1 < sh)
                    {
                      const uint8_t *r0 =
                        src.row_ptr (static_cast<size_t> (y0_p));
                      const uint8_t *r1 =
                        src.row_ptr (static_cast<size_t> (y0_p + 1));
                      const size_t ux0 = static_cast<size_t> (x0) * C;
                      const size_t ux1 = ux0 + C;
                      for (size_t ch = 0U; ch < C; ++ch)
                        {
                          const float v0 =
                            static_cast<float> (r0[ux0 + ch]) * wx0
                            + static_cast<float> (r0[ux1 + ch]) * fx;
                          const float v1 =
                            static_cast<float> (r1[ux0 + ch]) * wx0
                            + static_cast<float> (r1[ux1 + ch]) * fx;
                          d_row[dx * C + ch] =
                            core::saturate_cast<uint8_t> (v0 * wy0 + v1 * fy);
                        }
                    }
                  else
                    {
                      for (size_t ch = 0U; ch < C; ++ch)
                        {
                          const float p00 = sample_pixel (y0_p, x0, ch);
                          const float p01 = sample_pixel (y0_p, x0 + 1, ch);
                          const float p10 = sample_pixel (y0_p + 1, x0, ch);
                          const float p11 =
                            sample_pixel (y0_p + 1, x0 + 1, ch);
                          const float v0 = p00 * wx0 + p01 * fx;
                          const float v1 = p10 * wx0 + p11 * fx;
                          d_row[dx * C + ch] =
                            core::saturate_cast<uint8_t> (v0 * wy0 + v1 * fy);
                        }
                    }
                }
            }
        }
    });

  return dst;
}

} /* namespace */

core::Result<core::Image_U8C1>
resize (core::Const_View_U8C1 src,
        size_t dst_width,
        size_t dst_height,
        Interpolation_Mode mode)
{
  return resize_impl<uint8_t, 1U> (src, dst_width, dst_height, mode);
}

core::Result<core::Image_U8C3>
resize (core::Const_View_U8C3 src,
        size_t dst_width,
        size_t dst_height,
        Interpolation_Mode mode)
{
  return resize_impl<uint8_t, 3U> (src, dst_width, dst_height, mode);
}

core::Result<core::Image_F32C1>
resize (core::Const_View_F32C1 src,
        size_t dst_width,
        size_t dst_height,
        Interpolation_Mode mode)
{
  return resize_impl<float, 1U> (src, dst_width, dst_height, mode);
}

core::Result<core::Image_U8C1>
warp_affine (core::Const_View_U8C1 src,
             const Homography_3x3 &forward_affine,
             size_t dst_width,
             size_t dst_height,
             Interpolation_Mode mode,
             core::Border_Mode border,
             uint8_t border_value)
{
  return warp_projective_impl<1U> (
    src, forward_affine, dst_width, dst_height, mode, border,
    core::Scalar<uint8_t, 1U> (border_value), true);
}

core::Result<core::Image_U8C3>
warp_affine (core::Const_View_U8C3 src,
             const Homography_3x3 &forward_affine,
             size_t dst_width,
             size_t dst_height,
             Interpolation_Mode mode,
             core::Border_Mode border,
             core::Scalar<uint8_t, 3U> border_value)
{
  return warp_projective_impl<3U> (src, forward_affine, dst_width, dst_height,
                                   mode, border, border_value, true);
}

core::Result<core::Image_U8C1>
warp_perspective (core::Const_View_U8C1 src,
                  const Homography_3x3 &forward_homography,
                  size_t dst_width,
                  size_t dst_height,
                  Interpolation_Mode mode,
                  core::Border_Mode border,
                  uint8_t border_value)
{
  return warp_projective_impl<1U> (
    src, forward_homography, dst_width, dst_height, mode, border,
    core::Scalar<uint8_t, 1U> (border_value), false);
}

core::Result<core::Image_U8C3>
warp_perspective (core::Const_View_U8C3 src,
                  const Homography_3x3 &forward_homography,
                  size_t dst_width,
                  size_t dst_height,
                  Interpolation_Mode mode,
                  core::Border_Mode border,
                  core::Scalar<uint8_t, 3U> border_value)
{
  return warp_projective_impl<3U> (src, forward_homography, dst_width,
                                   dst_height, mode, border, border_value,
                                   false);
}

} /* namespace surya::inline v1::imgproc */
