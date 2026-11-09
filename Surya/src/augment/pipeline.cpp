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

#include <surya/augment/pipeline.hpp>

#include <algorithm>
#include <cmath>

namespace surya::inline v1::augment
{

Augment_Fused_State::Augment_Fused_State (void) noexcept
{
  reset ();
}

void
Augment_Fused_State::reset (void) noexcept
{
  h_total_ = imgproc::Homography_3x3::identity ();
  for (size_t c = 0U; c < 3U; ++c)
    {
      for (size_t i = 0U; i < 256U; ++i)
        {
          lut_[c][i] = static_cast<uint8_t> (i);
        }
    }
  has_geo_ = false;
  has_lut_ = false;
}

void
Augment_Fused_State::append_homography (
  const imgproc::Homography_3x3 &h) noexcept
{
  h_total_ = h * h_total_;
  has_geo_ = true;
}

void
Augment_Fused_State::append_lut (
  const std::array<std::array<uint8_t, 256U>, 3U> &op_lut) noexcept
{
  for (size_t c = 0U; c < 3U; ++c)
    {
      for (size_t i = 0U; i < 256U; ++i)
        {
          lut_[c][i] = op_lut[c][lut_[c][i]];
        }
    }
  has_lut_ = true;
}

core::Result<void>
Augment_Fused_State::flush (Augment_Sample &sample, float min_box_visibility)
{
  if (sample.image.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "Augment_Fused_State::flush: empty image");
    }

  const size_t w = sample.image.width ();
  const size_t h = sample.image.height ();

  if (has_geo_)
    {
      auto w_img = imgproc::warp_perspective (
        sample.image.const_view (), h_total_, w, h,
        imgproc::Interpolation_Mode::Bilinear, core::Border_Mode::Constant);
      if (!w_img.has_value ())
        {
          return std::unexpected (w_img.error ());
        }
      sample.image = std::move (*w_img);

      if (!sample.mask.empty ())
        {
          auto w_msk = imgproc::warp_perspective (
            sample.mask.const_view (), h_total_, w, h,
            imgproc::Interpolation_Mode::Nearest,
            core::Border_Mode::Constant, 0U);
          if (!w_msk.has_value ())
            {
              return std::unexpected (w_msk.error ());
            }
          sample.mask = std::move (*w_msk);
        }

      /* Transform all 4 corners of each bounding box through H_total and
         recompute axis-aligned bounds clipped to [0, w] x [0, h].  */
      const float fw = static_cast<float> (w);
      const float fh = static_cast<float> (h);
      std::vector<Annotated_Box> kept_boxes;
      kept_boxes.reserve (sample.boxes.size ());

      for (const Annotated_Box &b : sample.boxes)
        {
          const float orig_area = b.rect.area ();
          if (orig_area <= 0.0F)
            {
              continue;
            }
          const core::Point2f corners[4] = {
            h_total_.transform_point (core::Point2f (b.rect.x, b.rect.y)),
            h_total_.transform_point (core::Point2f (b.rect.x2 (), b.rect.y)),
            h_total_.transform_point (core::Point2f (b.rect.x, b.rect.y2 ())),
            h_total_.transform_point (
              core::Point2f (b.rect.x2 (), b.rect.y2 ()))};

          float min_x = corners[0].x;
          float max_x = corners[0].x;
          float min_y = corners[0].y;
          float max_y = corners[0].y;
          for (int k = 1; k < 4; ++k)
            {
              min_x = std::min (min_x, corners[k].x);
              max_x = std::max (max_x, corners[k].x);
              min_y = std::min (min_y, corners[k].y);
              max_y = std::max (max_y, corners[k].y);
            }

          const float cx1 = std::clamp (min_x, 0.0F, fw);
          const float cy1 = std::clamp (min_y, 0.0F, fh);
          const float cx2 = std::clamp (max_x, 0.0F, fw);
          const float cy2 = std::clamp (max_y, 0.0F, fh);
          const core::Rectf clipped (cx1, cy1, cx2 - cx1, cy2 - cy1);
          if (clipped.area () >= orig_area * min_box_visibility)
            {
              Annotated_Box nb = b;
              nb.rect = clipped;
              kept_boxes.push_back (nb);
            }
        }
      sample.boxes = std::move (kept_boxes);

      for (Annotated_Keypoint &kp : sample.keypoints)
        {
          if (kp.visibility == 0U)
            {
              continue;
            }
          kp.pt = h_total_.transform_point (kp.pt);
          if (kp.pt.x < 0.0F || kp.pt.x >= fw || kp.pt.y < 0.0F
              || kp.pt.y >= fh)
            {
              kp.visibility = 0U;
            }
        }
    }

  if (has_lut_)
    {
      for (size_t y = 0U; y < h; ++y)
        {
          uint8_t *row = sample.image.row_ptr (y);
          for (size_t x = 0U; x < w; ++x)
            {
              row[x * 3U + 0U] = lut_[0][row[x * 3U + 0U]];
              row[x * 3U + 1U] = lut_[1][row[x * 3U + 1U]];
              row[x * 3U + 2U] = lut_[2][row[x * 3U + 2U]];
            }
        }
    }

  reset ();
  return {};
}

namespace
{

bool
coin_flip (float prob, std::mt19937_64 &rng) noexcept
{
  if (prob >= 1.0F)
    {
      return true;
    }
  if (prob <= 0.0F)
    {
      return false;
    }
  std::uniform_real_distribution<float> dist (0.0F, 1.0F);
  return dist (rng) < prob;
}

} /* namespace */

void
Horizontal_Flip::record (Augment_Fused_State &st,
                         size_t width,
                         size_t /*height*/,
                         std::mt19937_64 &rng) const noexcept
{
  if (!coin_flip (prob, rng) || width == 0U)
    {
      return;
    }
  const double max_x = static_cast<double> (width - 1U);
  st.append_homography (
    imgproc::Homography_3x3 (-1.0, 0.0, max_x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0));
}

void
Vertical_Flip::record (Augment_Fused_State &st,
                       size_t /*width*/,
                       size_t height,
                       std::mt19937_64 &rng) const noexcept
{
  if (!coin_flip (prob, rng) || height == 0U)
    {
      return;
    }
  const double max_y = static_cast<double> (height - 1U);
  st.append_homography (
    imgproc::Homography_3x3 (1.0, 0.0, 0.0, 0.0, -1.0, max_y, 0.0, 0.0, 1.0));
}

void
Shift_Scale_Rotate::record (Augment_Fused_State &st,
                            size_t width,
                            size_t height,
                            std::mt19937_64 &rng) const noexcept
{
  if (!coin_flip (prob, rng) || width == 0U || height == 0U)
    {
      return;
    }
  std::uniform_real_distribution<float> ang_d (-max_angle_deg, max_angle_deg);
  std::uniform_real_distribution<float> scl_d (min_scale, max_scale);
  std::uniform_real_distribution<float> shf_d (-max_shift_frac,
                                               max_shift_frac);

  const double angle = static_cast<double> (ang_d (rng));
  const double scale = static_cast<double> (scl_d (rng));
  const double dx =
    static_cast<double> (shf_d (rng)) * static_cast<double> (width);
  const double dy =
    static_cast<double> (shf_d (rng)) * static_cast<double> (height);

  const core::Point2f center (static_cast<float> (width) * 0.5F,
                              static_cast<float> (height) * 0.5F);
  imgproc::Homography_3x3 h =
    imgproc::Homography_3x3::rotation_scale_around (center, angle, scale);
  h.m[2] += dx;
  h.m[5] += dy;
  st.append_homography (h);
}

void
Brightness_Contrast::record (Augment_Fused_State &st,
                             size_t /*width*/,
                             size_t /*height*/,
                             std::mt19937_64 &rng) const noexcept
{
  if (!coin_flip (prob, rng))
    {
      return;
    }
  std::uniform_real_distribution<float> br_d (-max_brightness_delta,
                                              max_brightness_delta);
  std::uniform_real_distribution<float> ct_d (min_contrast, max_contrast);
  const float beta = br_d (rng);
  const float alpha = ct_d (rng);

  std::array<std::array<uint8_t, 256U>, 3U> op {};
  for (size_t i = 0U; i < 256U; ++i)
    {
      const float v =
        (static_cast<float> (i) - 127.5F) * alpha + 127.5F + beta;
      const uint8_t u = core::saturate_cast<uint8_t> (v);
      op[0][i] = u;
      op[1][i] = u;
      op[2][i] = u;
    }
  st.append_lut (op);
}

void
Gamma_Adjust::record (Augment_Fused_State &st,
                      size_t /*width*/,
                      size_t /*height*/,
                      std::mt19937_64 &rng) const noexcept
{
  if (!coin_flip (prob, rng))
    {
      return;
    }
  std::uniform_real_distribution<float> gm_d (min_gamma, max_gamma);
  const float gamma = std::max (0.05F, gm_d (rng));

  std::array<std::array<uint8_t, 256U>, 3U> op {};
  for (size_t i = 0U; i < 256U; ++i)
    {
      const float norm = static_cast<float> (i) / 255.0F;
      const uint8_t u =
        core::saturate_cast<uint8_t> (std::pow (norm, gamma) * 255.0F);
      op[0][i] = u;
      op[1][i] = u;
      op[2][i] = u;
    }
  st.append_lut (op);
}

void
Solarize::record (Augment_Fused_State &st,
                  size_t /*width*/,
                  size_t /*height*/,
                  std::mt19937_64 &rng) const noexcept
{
  if (!coin_flip (prob, rng))
    {
      return;
    }
  std::array<std::array<uint8_t, 256U>, 3U> op {};
  for (size_t i = 0U; i < 256U; ++i)
    {
      const uint8_t v = static_cast<uint8_t> (i);
      const uint8_t u = v >= threshold ? static_cast<uint8_t> (255U - v) : v;
      op[0][i] = u;
      op[1][i] = u;
      op[2][i] = u;
    }
  st.append_lut (op);
}

void
Posterize::record (Augment_Fused_State &st,
                   size_t /*width*/,
                   size_t /*height*/,
                   std::mt19937_64 &rng) const noexcept
{
  if (!coin_flip (prob, rng))
    {
      return;
    }
  const uint8_t shift = bits < 8U ? static_cast<uint8_t> (8U - bits) : 0U;
  const uint8_t mask = static_cast<uint8_t> (0xFFU << shift);
  std::array<std::array<uint8_t, 256U>, 3U> op {};
  for (size_t i = 0U; i < 256U; ++i)
    {
      const uint8_t u = static_cast<uint8_t> (i) & mask;
      op[0][i] = u;
      op[1][i] = u;
      op[2][i] = u;
    }
  st.append_lut (op);
}

core::Result<void>
Coarse_Dropout::apply_eager (Augment_Sample &sample,
                             std::mt19937_64 &rng) const
{
  if (sample.image.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "Coarse_Dropout: sample image is empty");
    }
  if (!coin_flip (prob, rng) || max_holes == 0U)
    {
      return {};
    }

  const size_t w = sample.image.width ();
  const size_t h = sample.image.height ();
  const size_t hw = std::clamp<size_t> (max_hole_width, 1U, w);
  const size_t hh = std::clamp<size_t> (max_hole_height, 1U, h);

  for (size_t k = 0U; k < max_holes; ++k)
    {
      const size_t x0 = (w > hw) ? (rng () % (w - hw)) : 0U;
      const size_t y0 = (h > hh) ? (rng () % (h - hh)) : 0U;
      for (size_t y = y0; y < y0 + hh; ++y)
        {
          uint8_t *row = sample.image.row_ptr (y);
          std::memset (row + x0 * 3U, fill_value, hw * 3U);
        }
    }
  return {};
}

core::Result<void>
Augment_Graph::apply (Augment_Sample &sample,
                      std::mt19937_64 &rng,
                      float min_box_visibility) const
{
  if (sample.image.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "Augment_Graph::apply: empty image");
    }

  Augment_Fused_State state;
  for (const auto &node : nodes_)
    {
      core::Result<void> step_res {};
      std::visit (
        [&] (const auto &stage)
        {
          using T = std::decay_t<decltype (stage)>;
          if constexpr (requires {
                          stage.record (state, sample.image.width (),
                                        sample.image.height (), rng);
                        })
            {
              stage.record (state, sample.image.width (),
                            sample.image.height (), rng);
            }
          else if constexpr (std::is_same_v<T, Coarse_Dropout>)
            {
              step_res = state.flush (sample, min_box_visibility);
              if (step_res.has_value ())
                {
                  step_res = stage.apply_eager (sample, rng);
                }
            }
        },
        node);
      if (!step_res.has_value ())
        {
          return step_res;
        }
    }
  return state.flush (sample, min_box_visibility);
}

core::Result<void>
Augment_Graph::apply_batch (std::span<Augment_Sample> batch,
                            uint64_t base_seed,
                            float min_box_visibility) const
{
  for (size_t i = 0U; i < batch.size (); ++i)
    {
      if (batch[i].image.empty ())
        {
          return core::make_error (
            core::Error_Code::Empty_Input,
            "Augment_Graph::apply_batch: sample image is empty");
        }
    }
  core::parallel_for_1d (
    0U, batch.size (), 1U,
    [&] (size_t b0, size_t b1)
    {
      for (size_t i = b0; i < b1; ++i)
        {
          std::mt19937_64 rng (base_seed + i * 0x9E3779B97F4A7C15ULL);
          auto r = apply (batch[i], rng, min_box_visibility);
          (void) r;
        }
    });
  return {};
}

core::Result<Augment_Sample>
mixup (const Augment_Sample &s1, const Augment_Sample &s2, float alpha)
{
  if (s1.image.empty () || s2.image.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "mixup: input sample is empty");
    }
  if (s1.image.width () != s2.image.width ()
      || s1.image.height () != s2.image.height ())
    {
      return core::make_error (core::Error_Code::Dimension_Mismatch,
                               "mixup: s1 and s2 dimensions must match");
    }

  const float w1 = std::clamp (alpha, 0.0F, 1.0F);
  const float w2 = 1.0F - w1;
  Augment_Sample out = s1.clone ();

  const size_t row_elems = out.image.width () * 3U;
  for (size_t y = 0U; y < out.image.height (); ++y)
    {
      const uint8_t *r1 = s1.image.row_ptr (y);
      const uint8_t *r2 = s2.image.row_ptr (y);
      uint8_t *d = out.image.row_ptr (y);
      for (size_t i = 0U; i < row_elems; ++i)
        {
          const float v =
            static_cast<float> (r1[i]) * w1 + static_cast<float> (r2[i]) * w2;
          d[i] = core::saturate_cast<uint8_t> (v);
        }
    }

  for (Annotated_Box &b : out.boxes)
    {
      b.weight *= w1;
    }
  for (Annotated_Box b : s2.boxes)
    {
      b.weight *= w2;
      out.boxes.push_back (b);
    }
  return out;
}

core::Result<Augment_Sample>
cutmix (const Augment_Sample &s1,
        const Augment_Sample &s2,
        core::Rectu patch_rect)
{
  if (s1.image.empty () || s2.image.empty ())
    {
      return core::make_error (core::Error_Code::Empty_Input,
                               "cutmix: input sample is empty");
    }
  if (s1.image.width () != s2.image.width ()
      || s1.image.height () != s2.image.height ())
    {
      return core::make_error (core::Error_Code::Dimension_Mismatch,
                               "cutmix: s1 and s2 dimensions must match");
    }
  const size_t w = s1.image.width ();
  const size_t h = s1.image.height ();
  if (patch_rect.empty () || patch_rect.x2 () > w || patch_rect.y2 () > h)
    {
      return core::make_error (core::Error_Code::Out_Of_Bounds,
                               "cutmix: patch_rect exceeds image bounds");
    }

  Augment_Sample out = s1.clone ();
  for (size_t y = patch_rect.y; y < patch_rect.y2 (); ++y)
    {
      const uint8_t *src_r = s2.image.row_ptr (y) + patch_rect.x * 3U;
      uint8_t *dst_r = out.image.row_ptr (y) + patch_rect.x * 3U;
      std::memcpy (dst_r, src_r, patch_rect.width * 3U);
    }

  const core::Rectf pf (static_cast<float> (patch_rect.x),
                        static_cast<float> (patch_rect.y),
                        static_cast<float> (patch_rect.width),
                        static_cast<float> (patch_rect.height));

  /* Append boxes from s2 that intersect the pasted patch.  */
  for (const Annotated_Box &b : s2.boxes)
    {
      const core::Rectf inter = b.rect.intersect (pf);
      if (inter.area () > 0.1F * b.rect.area ())
        {
          Annotated_Box nb = b;
          nb.rect = inter;
          out.boxes.push_back (nb);
        }
    }
  return out;
}

core::Result<Augment_Sample>
mosaic4 (const std::array<const Augment_Sample *, 4U> &samples,
         size_t out_width,
         size_t out_height,
         size_t div_x,
         size_t div_y)
{
  if (out_width == 0U || out_height == 0U || div_x == 0U
      || div_x >= out_width || div_y == 0U || div_y >= out_height)
    {
      return core::make_error (core::Error_Code::Invalid_Argument,
                               "mosaic4: invalid canvas or divider coords");
    }
  for (const Augment_Sample *s : samples)
    {
      if (s == nullptr || s->image.empty ())
        {
          return core::make_error (core::Error_Code::Empty_Input,
                                   "mosaic4: null or empty input sample");
        }
    }

  Augment_Sample out;
  auto img_res = core::Image_U8C3::create (out_width, out_height, true);
  if (!img_res.has_value ())
    {
      return std::unexpected (img_res.error ());
    }
  out.image = std::move (*img_res);

  const core::Rectu quadrants[4] = {
    core::Rectu (0U, 0U, div_x, div_y),
    core::Rectu (div_x, 0U, out_width - div_x, div_y),
    core::Rectu (0U, div_y, div_x, out_height - div_y),
    core::Rectu (div_x, div_y, out_width - div_x, out_height - div_y)};

  for (size_t q = 0U; q < 4U; ++q)
    {
      const Augment_Sample &src = *samples[q];
      const core::Rectu &qr = quadrants[q];

      auto resized = imgproc::resize (
        src.image.const_view (), qr.width, qr.height,
        imgproc::Interpolation_Mode::Bilinear);
      if (!resized.has_value ())
        {
          return std::unexpected (resized.error ());
        }

      for (size_t y = 0U; y < qr.height; ++y)
        {
          std::memcpy (out.image.row_ptr (qr.y + y) + qr.x * 3U,
                       resized->row_ptr (y), qr.width * 3U);
        }

      const float sx =
        static_cast<float> (qr.width)
        / static_cast<float> (src.image.width ());
      const float sy =
        static_cast<float> (qr.height)
        / static_cast<float> (src.image.height ());
      const float ox = static_cast<float> (qr.x);
      const float oy = static_cast<float> (qr.y);

      for (const Annotated_Box &b : src.boxes)
        {
          Annotated_Box nb = b;
          nb.rect = core::Rectf (ox + b.rect.x * sx, oy + b.rect.y * sy,
                                 b.rect.width * sx, b.rect.height * sy);
          out.boxes.push_back (nb);
        }

      for (const Annotated_Keypoint &kp : src.keypoints)
        {
          Annotated_Keypoint nkp = kp;
          nkp.pt = core::Point2f (ox + kp.pt.x * sx, oy + kp.pt.y * sy);
          out.keypoints.push_back (nkp);
        }
    }

  return out;
}

} /* namespace surya::inline v1::augment */
