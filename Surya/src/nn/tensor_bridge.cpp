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

#include <surya/nn/tensor_bridge.hpp>

#include <cmath>

namespace surya::inline v1::nn
{

namespace
{

void
convert_hwc_u8_to_chw_f32_slice (float *SURYA_RESTRICT dst_chw,
                                 core::Const_View_U8C3 src,
                                 const Normalize_Params &norm,
                                 bool swap_rb) noexcept
{
  const size_t w = src.width ();
  const size_t h = src.height ();
  const size_t plane_elems = w * h;

  float *p0 = dst_chw + 0U * plane_elems;
  float *p1 = dst_chw + 1U * plane_elems;
  float *p2 = dst_chw + 2U * plane_elems;

  const float s0 = norm.scale / norm.std[0];
  const float s1 = norm.scale / norm.std[1];
  const float s2 = norm.scale / norm.std[2];
  const float b0 = -norm.mean[0] / norm.std[0];
  const float b1 = -norm.mean[1] / norm.std[1];
  const float b2 = -norm.mean[2] / norm.std[2];

  const size_t c0_idx = swap_rb ? 2U : 0U;
  const size_t c2_idx = swap_rb ? 0U : 2U;

  for (size_t y = 0U; y < h; ++y)
    {
      const uint8_t *s_row = src.row_ptr (y);
      float *d0 = p0 + y * w;
      float *d1 = p1 + y * w;
      float *d2 = p2 + y * w;

      for (size_t x = 0U; x < w; ++x)
        {
          const uint8_t v0 = s_row[x * 3U + c0_idx];
          const uint8_t v1 = s_row[x * 3U + 1U];
          const uint8_t v2 = s_row[x * 3U + c2_idx];
          d0[x] = static_cast<float> (v0) * s0 + b0;
          d1[x] = static_cast<float> (v1) * s1 + b1;
          d2[x] = static_cast<float> (v2) * s2 + b2;
        }
    }
}

} /* namespace */

core::Result<core::SuryaTensor>
image_to_tensor_nchw (core::Const_View_U8C3 src,
                      const Normalize_Params &norm,
                      bool swap_rb)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "image_to_tensor_nchw: input image is empty");
    }
  if (norm.std[0] == 0.0F || norm.std[1] == 0.0F || norm.std[2] == 0.0F)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "image_to_tensor_nchw: std must be non-zero");
    }

  const size_t w = src.width ();
  const size_t h = src.height ();
  core::SuryaTensor out ({1U, 3U, h, w}, 0.0F);
  convert_hwc_u8_to_chw_f32_slice (out.data (), src, norm, swap_rb);
  return out;
}

core::Result<core::SuryaTensor>
image_to_tensor_nchw (core::Const_View_U8C1 src,
                      float scale,
                      float mean,
                      float std_dev)
{
  if (src.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "image_to_tensor_nchw: input image is empty");
    }
  if (std_dev == 0.0F)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "image_to_tensor_nchw: std_dev cannot be 0");
    }

  const size_t w = src.width ();
  const size_t h = src.height ();
  core::SuryaTensor out ({1U, 1U, h, w}, 0.0F);
  float *dst = out.data ();
  const float eff_scale = scale / std_dev;
  const float eff_bias = -mean / std_dev;

  for (size_t y = 0U; y < h; ++y)
    {
      core::simd_scale_bias_u8_to_f32 (dst + y * w, src.row_ptr (y), w,
                                       eff_scale, eff_bias);
    }
  return out;
}

core::Result<core::SuryaTensor>
batch_images_to_tensor_nchw (std::span<const core::Const_View_U8C3> batch,
                             const Normalize_Params &norm,
                             bool swap_rb)
{
  if (batch.empty ())
    {
      return core::make_error (
        core::Error_Code::Empty_Input,
        "batch_images_to_tensor_nchw: batch span is empty");
    }
  const size_t w = batch[0].width ();
  const size_t h = batch[0].height ();
  if (w == 0U || h == 0U)
    {
      return core::make_error (
        core::Error_Code::Empty_Input,
        "batch_images_to_tensor_nchw: first image is empty");
    }
  for (size_t i = 1U; i < batch.size (); ++i)
    {
      if (batch[i].width () != w || batch[i].height () != h)
        {
          return core::make_error (
            core::Error_Code::Dimension_Mismatch,
            "batch_images_to_tensor_nchw: all images must share WxH");
        }
    }

  const size_t b_size = batch.size ();
  const size_t stride_per_img = 3U * h * w;
  core::SuryaTensor out ({b_size, 3U, h, w}, 0.0F);
  float *base = out.data ();

  core::parallel_for_1d (
    0U, b_size, 1U,
    [&] (size_t b0, size_t b1)
    {
      for (size_t i = b0; i < b1; ++i)
        {
          convert_hwc_u8_to_chw_f32_slice (
            base + i * stride_per_img, batch[i], norm, swap_rb);
        }
    });
  return out;
}

core::Result<core::Image_F32C3>
tensor_nchw_to_image_f32 (const core::SuryaTensor &tensor, size_t batch_idx)
{
  if (tensor.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "tensor_nchw_to_image_f32: empty tensor");
    }
  const auto &sh = tensor.shape ();

  size_t h = 0U;
  size_t w = 0U;
  const float *slice_ptr = nullptr;

  if (sh.size () == 4U && sh[1] == 3U)
    {
      if (batch_idx >= sh[0])
        {
          return core::make_error (
            core::Error_Code::Out_Of_Bounds,
            "tensor_nchw_to_image_f32: batch_idx out of range");
        }
      h = sh[2];
      w = sh[3];
      slice_ptr = tensor.data () + batch_idx * 3U * h * w;
    }
  else if (sh.size () == 3U && sh[0] == 3U)
    {
      h = sh[1];
      w = sh[2];
      slice_ptr = tensor.data ();
    }
  else
    {
      return core::make_error (
        core::Error_Code::Dimension_Mismatch,
        "tensor_nchw_to_image_f32: expected [B, 3, H, W] or [3, H, W]");
    }

  auto img_res = core::Image_F32C3::create (w, h, false);
  if (!img_res.has_value ())
    {
      return std::unexpected (img_res.error ());
    }
  core::Image_F32C3 img = std::move (*img_res);

  const size_t plane = h * w;
  const float *p0 = slice_ptr + 0U * plane;
  const float *p1 = slice_ptr + 1U * plane;
  const float *p2 = slice_ptr + 2U * plane;

  for (size_t y = 0U; y < h; ++y)
    {
      float *d = img.row_ptr (y);
      const float *r0 = p0 + y * w;
      const float *r1 = p1 + y * w;
      const float *r2 = p2 + y * w;
      for (size_t x = 0U; x < w; ++x)
        {
          d[x * 3U + 0U] = r0[x];
          d[x * 3U + 1U] = r1[x];
          d[x * 3U + 2U] = r2[x];
        }
    }
  return img;
}

core::Result<core::Image_U8C3>
tensor_nchw_to_image_u8 (const core::SuryaTensor &tensor,
                         size_t batch_idx,
                         float mul_scale)
{
  auto f_res = tensor_nchw_to_image_f32 (tensor, batch_idx);
  if (!f_res.has_value ())
    {
      return std::unexpected (f_res.error ());
    }
  const core::Image_F32C3 &f_img = *f_res;
  auto u_res =
    core::Image_U8C3::create (f_img.width (), f_img.height (), false);
  if (!u_res.has_value ())
    {
      return std::unexpected (u_res.error ());
    }
  core::Image_U8C3 u_img = std::move (*u_res);

  const size_t row_elems = f_img.width () * 3U;
  for (size_t y = 0U; y < f_img.height (); ++y)
    {
      const float *s = f_img.row_ptr (y);
      uint8_t *d = u_img.row_ptr (y);
      for (size_t i = 0U; i < row_elems; ++i)
        {
          d[i] = core::saturate_cast<uint8_t> (s[i] * mul_scale);
        }
    }
  return u_img;
}

} /* namespace surya::inline v1::nn */
