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

#include <surya/imgproc/color.hpp>

#include <algorithm>
#include <cmath>

namespace surya::inline v1::imgproc
{

namespace
{

/* Q15 ITU-R BT.601 coefficients: 9798/32768 = 0.2990, 19235/32768 = 0.5870,
   3735/32768 = 0.1140.  Sum = 32768.  */
constexpr uint32_t kWeightR = 9798U;
constexpr uint32_t kWeightG = 19235U;
constexpr uint32_t kWeightB = 3735U;
constexpr uint32_t kRoundQ15 = 16384U;

void
convert_3ch_to_gray_row (uint8_t *SURYA_RESTRICT dst,
                         const uint8_t *SURYA_RESTRICT src,
                         size_t width,
                         uint32_t w0,
                         uint32_t w1,
                         uint32_t w2) noexcept
{
  size_t x = 0U;
  for (; x + 4U <= width; x += 4U)
    {
      const uint8_t *p = src + x * 3U;
      dst[x + 0U] = static_cast<uint8_t> (
        (w0 * p[0] + w1 * p[1] + w2 * p[2] + kRoundQ15) >> 15U);
      dst[x + 1U] = static_cast<uint8_t> (
        (w0 * p[3] + w1 * p[4] + w2 * p[5] + kRoundQ15) >> 15U);
      dst[x + 2U] = static_cast<uint8_t> (
        (w0 * p[6] + w1 * p[7] + w2 * p[8] + kRoundQ15) >> 15U);
      dst[x + 3U] = static_cast<uint8_t> (
        (w0 * p[9] + w1 * p[10] + w2 * p[11] + kRoundQ15) >> 15U);
    }
  for (; x < width; ++x)
    {
      const uint8_t *p = src + x * 3U;
      dst[x] = static_cast<uint8_t> (
        (w0 * p[0] + w1 * p[1] + w2 * p[2] + kRoundQ15) >> 15U);
    }
}

} /* namespace */

core::Result<core::Image_U8C1>
rgb_to_gray (core::Const_View_U8C3 src)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "rgb_to_gray: input image is empty");
    }
  auto dst_res = core::Image_U8C1::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C1 dst = std::move (*dst_res);

  core::parallel_for_1d (
    0U, src.height (), 32U,
    [&] (size_t y0, size_t y1)
    {
      for (size_t y = y0; y < y1; ++y)
        {
          convert_3ch_to_gray_row (dst.row_ptr (y), src.row_ptr (y),
                                   src.width (), kWeightR, kWeightG, kWeightB);
        }
    });
  return dst;
}

core::Result<core::Image_U8C1>
bgr_to_gray (core::Const_View_U8C3 src)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "bgr_to_gray: input image is empty");
    }
  auto dst_res = core::Image_U8C1::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C1 dst = std::move (*dst_res);

  core::parallel_for_1d (
    0U, src.height (), 32U,
    [&] (size_t y0, size_t y1)
    {
      for (size_t y = y0; y < y1; ++y)
        {
          convert_3ch_to_gray_row (dst.row_ptr (y), src.row_ptr (y),
                                   src.width (), kWeightB, kWeightG, kWeightR);
        }
    });
  return dst;
}

core::Result<core::Image_U8C3>
gray_to_rgb (core::Const_View_U8C1 src)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "gray_to_rgb: input image is empty");
    }
  auto dst_res = core::Image_U8C3::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C3 dst = std::move (*dst_res);

  for (size_t y = 0U; y < src.height (); ++y)
    {
      const uint8_t *s = src.row_ptr (y);
      uint8_t *d = dst.row_ptr (y);
      for (size_t x = 0U; x < src.width (); ++x)
        {
          const uint8_t v = s[x];
          d[x * 3U + 0U] = v;
          d[x * 3U + 1U] = v;
          d[x * 3U + 2U] = v;
        }
    }
  return dst;
}

core::Result<core::Image_U8C3>
rgb_to_bgr (core::Const_View_U8C3 src)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "rgb_to_bgr: input image is empty");
    }
  auto dst_res = core::Image_U8C3::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C3 dst = std::move (*dst_res);

  for (size_t y = 0U; y < src.height (); ++y)
    {
      const uint8_t *s = src.row_ptr (y);
      uint8_t *d = dst.row_ptr (y);
      for (size_t x = 0U; x < src.width (); ++x)
        {
          const uint8_t r = s[x * 3U + 0U];
          const uint8_t g = s[x * 3U + 1U];
          const uint8_t b = s[x * 3U + 2U];
          d[x * 3U + 0U] = b;
          d[x * 3U + 1U] = g;
          d[x * 3U + 2U] = r;
        }
    }
  return dst;
}

core::Result<core::Image_U8C4>
rgb_to_rgba (core::Const_View_U8C3 src, uint8_t alpha)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "rgb_to_rgba: input image is empty");
    }
  auto dst_res = core::Image_U8C4::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C4 dst = std::move (*dst_res);

  for (size_t y = 0U; y < src.height (); ++y)
    {
      const uint8_t *s = src.row_ptr (y);
      uint8_t *d = dst.row_ptr (y);
      for (size_t x = 0U; x < src.width (); ++x)
        {
          d[x * 4U + 0U] = s[x * 3U + 0U];
          d[x * 4U + 1U] = s[x * 3U + 1U];
          d[x * 4U + 2U] = s[x * 3U + 2U];
          d[x * 4U + 3U] = alpha;
        }
    }
  return dst;
}

core::Result<core::Image_U8C3>
rgba_to_rgb (core::Const_View_U8C4 src)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "rgba_to_rgb: input image is empty");
    }
  auto dst_res = core::Image_U8C3::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C3 dst = std::move (*dst_res);

  for (size_t y = 0U; y < src.height (); ++y)
    {
      const uint8_t *s = src.row_ptr (y);
      uint8_t *d = dst.row_ptr (y);
      for (size_t x = 0U; x < src.width (); ++x)
        {
          d[x * 3U + 0U] = s[x * 4U + 0U];
          d[x * 3U + 1U] = s[x * 4U + 1U];
          d[x * 3U + 2U] = s[x * 4U + 2U];
        }
    }
  return dst;
}

core::Result<core::Image_U8C3>
rgb_to_hsv (core::Const_View_U8C3 src)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "rgb_to_hsv: input image is empty");
    }
  auto dst_res = core::Image_U8C3::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C3 dst = std::move (*dst_res);

  for (size_t y = 0U; y < src.height (); ++y)
    {
      const uint8_t *s = src.row_ptr (y);
      uint8_t *d = dst.row_ptr (y);
      for (size_t x = 0U; x < src.width (); ++x)
        {
          const int r = static_cast<int> (s[x * 3U + 0U]);
          const int g = static_cast<int> (s[x * 3U + 1U]);
          const int b = static_cast<int> (s[x * 3U + 2U]);
          const int vmax = std::max ({r, g, b});
          const int vmin = std::min ({r, g, b});
          const int diff = vmax - vmin;

          const int v = vmax;
          const int sat =
            vmax > 0 ? ((255 * diff + (vmax / 2)) / vmax) : 0;

          int h = 0;
          if (diff > 0)
            {
              if (vmax == r)
                {
                  h = (30 * (g - b)) / diff;
                }
              else if (vmax == g)
                {
                  h = 60 + (30 * (b - r)) / diff;
                }
              else
                {
                  h = 120 + (30 * (r - g)) / diff;
                }
              if (h < 0)
                {
                  h += 180;
                }
            }
          d[x * 3U + 0U] = static_cast<uint8_t> (std::clamp (h, 0, 180));
          d[x * 3U + 1U] = static_cast<uint8_t> (std::clamp (sat, 0, 255));
          d[x * 3U + 2U] = static_cast<uint8_t> (v);
        }
    }
  return dst;
}

core::Result<core::Image_U8C3>
hsv_to_rgb (core::Const_View_U8C3 src)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "hsv_to_rgb: input image is empty");
    }
  auto dst_res = core::Image_U8C3::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C3 dst = std::move (*dst_res);

  for (size_t y = 0U; y < src.height (); ++y)
    {
      const uint8_t *s = src.row_ptr (y);
      uint8_t *d = dst.row_ptr (y);
      for (size_t x = 0U; x < src.width (); ++x)
        {
          const float h = static_cast<float> (s[x * 3U + 0U]) * 2.0F;
          const float sat = static_cast<float> (s[x * 3U + 1U]) / 255.0F;
          const float val = static_cast<float> (s[x * 3U + 2U]) / 255.0F;

          const float c = val * sat;
          const float h_sector = h / 60.0F;
          const float x_val =
            c * (1.0F - std::abs (std::fmod (h_sector, 2.0F) - 1.0F));
          const float m = val - c;

          float r1 = 0.0F;
          float g1 = 0.0F;
          float b1 = 0.0F;
          const int sec = static_cast<int> (h_sector) % 6;
          switch (sec)
            {
            case 0:
              r1 = c;
              g1 = x_val;
              break;
            case 1:
              r1 = x_val;
              g1 = c;
              break;
            case 2:
              g1 = c;
              b1 = x_val;
              break;
            case 3:
              g1 = x_val;
              b1 = c;
              break;
            case 4:
              r1 = x_val;
              b1 = c;
              break;
            default:
              r1 = c;
              b1 = x_val;
              break;
            }
          d[x * 3U + 0U] = core::saturate_cast<uint8_t> ((r1 + m) * 255.0F);
          d[x * 3U + 1U] = core::saturate_cast<uint8_t> ((g1 + m) * 255.0F);
          d[x * 3U + 2U] = core::saturate_cast<uint8_t> ((b1 + m) * 255.0F);
        }
    }
  return dst;
}

core::Result<core::Image_U8C3>
rgb_to_ycbcr (core::Const_View_U8C3 src)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "rgb_to_ycbcr: input image is empty");
    }
  auto dst_res = core::Image_U8C3::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C3 dst = std::move (*dst_res);

  for (size_t y = 0U; y < src.height (); ++y)
    {
      const uint8_t *s = src.row_ptr (y);
      uint8_t *d = dst.row_ptr (y);
      for (size_t x = 0U; x < src.width (); ++x)
        {
          const int r = static_cast<int> (s[x * 3U + 0U]);
          const int g = static_cast<int> (s[x * 3U + 1U]);
          const int b = static_cast<int> (s[x * 3U + 2U]);

          const int y_val = (9798 * r + 19235 * g + 3735 * b + 16384) >> 15;
          const int cb =
            128 + ((-5529 * r - 10855 * g + 16384 * b + 16384) >> 15);
          const int cr =
            128 + ((16384 * r - 13720 * g - 2664 * b + 16384) >> 15);

          d[x * 3U + 0U] = core::saturate_cast<uint8_t> (y_val);
          d[x * 3U + 1U] = core::saturate_cast<uint8_t> (cb);
          d[x * 3U + 2U] = core::saturate_cast<uint8_t> (cr);
        }
    }
  return dst;
}

core::Result<core::Image_U8C3>
ycbcr_to_rgb (core::Const_View_U8C3 src)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "ycbcr_to_rgb: input image is empty");
    }
  auto dst_res = core::Image_U8C3::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_U8C3 dst = std::move (*dst_res);

  for (size_t y = 0U; y < src.height (); ++y)
    {
      const uint8_t *s = src.row_ptr (y);
      uint8_t *d = dst.row_ptr (y);
      for (size_t x = 0U; x < src.width (); ++x)
        {
          const int y_val = static_cast<int> (s[x * 3U + 0U]);
          const int cb = static_cast<int> (s[x * 3U + 1U]) - 128;
          const int cr = static_cast<int> (s[x * 3U + 2U]) - 128;

          const int r = y_val + ((45941 * cr + 16384) >> 15);
          const int g = y_val - ((11277 * cb + 23401 * cr + 16384) >> 15);
          const int b = y_val + ((58065 * cb + 16384) >> 15);

          d[x * 3U + 0U] = core::saturate_cast<uint8_t> (r);
          d[x * 3U + 1U] = core::saturate_cast<uint8_t> (g);
          d[x * 3U + 2U] = core::saturate_cast<uint8_t> (b);
        }
    }
  return dst;
}

core::Result<core::Image_F32C3>
rgb_to_lab (core::Const_View_U8C3 src)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "rgb_to_lab: input image is empty");
    }
  auto dst_res =
    core::Image_F32C3::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_F32C3 dst = std::move (*dst_res);

  auto srgb_to_linear = [] (uint8_t u) noexcept -> float
  {
    const float v = static_cast<float> (u) / 255.0F;
    return v <= 0.04045F
             ? (v / 12.92F)
             : std::pow ((v + 0.055F) / 1.055F, 2.4F);
  };

  auto lab_f = [] (float t) noexcept -> float
  {
    constexpr float delta = 6.0F / 29.0F;
    constexpr float delta3 = delta * delta * delta;
    return t > delta3
             ? std::cbrt (t)
             : (t / (3.0F * delta * delta) + 4.0F / 29.0F);
  };

  for (size_t y = 0U; y < src.height (); ++y)
    {
      const uint8_t *s = src.row_ptr (y);
      float *d = dst.row_ptr (y);
      for (size_t x = 0U; x < src.width (); ++x)
        {
          const float r = srgb_to_linear (s[x * 3U + 0U]);
          const float g = srgb_to_linear (s[x * 3U + 1U]);
          const float b = srgb_to_linear (s[x * 3U + 2U]);

          const float X =
            (0.4124564F * r + 0.3575761F * g + 0.1804375F * b) / 0.95047F;
          const float Y = 0.2126729F * r + 0.7151522F * g + 0.0721750F * b;
          const float Z =
            (0.0193339F * r + 0.1191920F * g + 0.9503041F * b) / 1.08883F;

          const float fx = lab_f (X);
          const float fy = lab_f (Y);
          const float fz = lab_f (Z);

          d[x * 3U + 0U] = 116.0F * fy - 16.0F;
          d[x * 3U + 1U] = 500.0F * (fx - fy);
          d[x * 3U + 2U] = 200.0F * (fy - fz);
        }
    }
  return dst;
}

core::Result<core::Image_F32C3_Planar>
interleaved_to_planar (core::Const_View_F32C3 src)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "interleaved_to_planar: input image is empty");
    }
  auto dst_res =
    core::Image_F32C3_Planar::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_F32C3_Planar dst = std::move (*dst_res);

  for (size_t y = 0U; y < src.height (); ++y)
    {
      const float *s = src.row_ptr (y);
      float *p0 = dst.plane_row_ptr (0U, y);
      float *p1 = dst.plane_row_ptr (1U, y);
      float *p2 = dst.plane_row_ptr (2U, y);
      for (size_t x = 0U; x < src.width (); ++x)
        {
          p0[x] = s[x * 3U + 0U];
          p1[x] = s[x * 3U + 1U];
          p2[x] = s[x * 3U + 2U];
        }
    }
  return dst;
}

core::Result<core::Image_F32C3>
planar_to_interleaved (
  core::Image_View<const float, 3U, core::Channel_Layout::Planar> src)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "planar_to_interleaved: input image is empty");
    }
  auto dst_res =
    core::Image_F32C3::create (src.width (), src.height (), false);
  if (!dst_res.has_value ())
    {
      return std::unexpected (dst_res.error ());
    }
  core::Image_F32C3 dst = std::move (*dst_res);

  for (size_t y = 0U; y < src.height (); ++y)
    {
      const float *p0 = src.plane_row_ptr (0U, y);
      const float *p1 = src.plane_row_ptr (1U, y);
      const float *p2 = src.plane_row_ptr (2U, y);
      float *d = dst.row_ptr (y);
      for (size_t x = 0U; x < src.width (); ++x)
        {
          d[x * 3U + 0U] = p0[x];
          d[x * 3U + 1U] = p1[x];
          d[x * 3U + 2U] = p2[x];
        }
    }
  return dst;
}

} /* namespace surya::inline v1::imgproc */
