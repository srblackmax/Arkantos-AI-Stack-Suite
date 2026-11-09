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

#include <surya/imgproc.hpp>

#include <cmath>
#include <iostream>

using namespace surya;
using namespace surya::core;
using namespace surya::imgproc;

namespace
{

int g_failures = 0;

void
check_cond (bool cond, const char *expr, int line)
{
  if (!cond)
    {
      std::cerr << "FAIL [test_imgproc.cpp:" << line << "]: " << expr << "\n";
      ++g_failures;
    }
}

#define CHECK(expr) check_cond ((expr), #expr, __LINE__)

void
test_color_conversions (void)
{
  /* Pure red (255, 0, 0) -> BT.601 gray = round(0.299 * 255) = 76.  */
  Image_U8C3 red (32U, 16U, Scalar<uint8_t, 3U> (255U, 0U, 0U));
  auto g_res = rgb_to_gray (red.const_view ());
  CHECK (g_res.has_value ());
  CHECK (g_res->at (4U, 4U) == 76U);

  auto bgr_res = rgb_to_bgr (red.const_view ());
  CHECK (bgr_res.has_value ());
  CHECK (bgr_res->at (0U, 0U, 0U) == 0U);
  CHECK (bgr_res->at (0U, 0U, 2U) == 255U);

  auto bgr_g = bgr_to_gray (bgr_res->const_view ());
  CHECK (bgr_g.has_value ());
  CHECK (bgr_g->at (4U, 4U) == 76U);

  /* HSV and YCbCr round-trips.  */
  Image_U8C3 sample (16U, 16U, Scalar<uint8_t, 3U> (60U, 180U, 120U));
  auto hsv = rgb_to_hsv (sample.const_view ());
  CHECK (hsv.has_value ());
  auto rgb_back = hsv_to_rgb (hsv->const_view ());
  CHECK (rgb_back.has_value ());
  CHECK (std::abs (static_cast<int> (rgb_back->at (0U, 0U, 0U)) - 60) <= 3);
  CHECK (std::abs (static_cast<int> (rgb_back->at (0U, 0U, 1U)) - 180) <= 3);
  CHECK (std::abs (static_cast<int> (rgb_back->at (0U, 0U, 2U)) - 120) <= 3);

  auto ycbcr = rgb_to_ycbcr (sample.const_view ());
  CHECK (ycbcr.has_value ());
  auto ycbcr_back = ycbcr_to_rgb (ycbcr->const_view ());
  CHECK (ycbcr_back.has_value ());
  CHECK (std::abs (static_cast<int> (ycbcr_back->at (0U, 0U, 0U)) - 60) <= 2);
  CHECK (std::abs (static_cast<int> (ycbcr_back->at (0U, 0U, 1U)) - 180) <= 2);
  CHECK (std::abs (static_cast<int> (ycbcr_back->at (0U, 0U, 2U)) - 120) <= 2);

  /* CIE Lab white (255, 255, 255) -> L ~ 100, a ~ 0, b ~ 0.  */
  Image_U8C3 white (8U, 8U, Scalar<uint8_t, 3U> (255U, 255U, 255U));
  auto lab = rgb_to_lab (white.const_view ());
  CHECK (lab.has_value ());
  CHECK (std::abs (lab->at (0U, 0U, 0U) - 100.0F) < 0.2F);
  CHECK (std::abs (lab->at (0U, 0U, 1U)) < 0.2F);
  CHECK (std::abs (lab->at (0U, 0U, 2U)) < 0.2F);

  /* Interleaved <-> Planar round-trip.  */
  auto pl = interleaved_to_planar (lab->const_view ());
  CHECK (pl.has_value ());
  auto il = planar_to_interleaved (pl->const_view ());
  CHECK (il.has_value ());
  CHECK (std::abs (il->at (2U, 3U, 0U) - lab->at (2U, 3U, 0U)) < 1e-6F);
}

void
test_filters_and_integral (void)
{
  Image_U8C1 img (64U, 48U, Scalar<uint8_t, 1U> (100U));
  auto g = gaussian_blur (img.const_view (), 5U, 1.2F);
  CHECK (g.has_value ());
  CHECK (g->at (24U, 32U) == 100U);

  auto b = box_filter (img.const_view (), 7U, 7U);
  CHECK (b.has_value ());
  CHECK (b->at (24U, 32U) == 100U);

  /* Salt-and-pepper impulse removed by 3x3 median filter.  */
  img.at (20U, 20U) = 255U;
  img.at (10U, 10U) = 0U;
  auto med = median_filter_3x3 (img.const_view ());
  CHECK (med.has_value ());
  CHECK (med->at (20U, 20U) == 100U);
  CHECK (med->at (10U, 10U) == 100U);

  auto bil = bilateral_filter (img.const_view (), 5U, 25.0F, 3.0F);
  CHECK (bil.has_value ());
  CHECK (bil->at (30U, 30U) == 100U);

  /* Integral image exact box sum test.  */
  Image_U8C1 ones (20U, 15U, Scalar<uint8_t, 1U> (3U));
  auto sat = integral_image (ones.const_view ());
  CHECK (sat.has_value ());
  const int32_t sum = integral_box_sum (sat->const_view (), 2U, 3U, 12U, 8U);
  CHECK (sum == 10 * 5 * 3);
}

void
test_geometry_and_morphology (void)
{
  Image_U8C1 src (64U, 64U, Scalar<uint8_t, 1U> (0U));
  for (size_t y = 16U; y < 48U; ++y)
    {
      for (size_t x = 16U; x < 48U; ++x)
        {
          src.at (y, x) = 200U;
        }
    }

  auto r_near =
    resize (src.const_view (), 128U, 128U, Interpolation_Mode::Nearest);
  auto r_bil =
    resize (src.const_view (), 128U, 128U, Interpolation_Mode::Bilinear);
  auto r_bic =
    resize (src.const_view (), 32U, 32U, Interpolation_Mode::Bicubic);
  CHECK (r_near.has_value () && r_near->at (64U, 64U) == 200U);
  CHECK (r_bil.has_value () && r_bil->at (64U, 64U) == 200U);
  CHECK (r_bic.has_value () && r_bic->at (16U, 16U) == 200U);

  /* Pure translation by (+4, +6) via warp_affine.  */
  const Homography_3x3 trans = Homography_3x3::translation (4.0, 6.0);
  auto warped = warp_affine (src.const_view (), trans, 64U, 64U);
  CHECK (warped.has_value ());
  CHECK (warped->at (22U, 20U) == 200U);
  CHECK (warped->at (18U, 18U) == 0U);

  /* Morphology: erode shrinks 32x32 square by 1px on each side (to 30x30),
     dilate expands by 1px on each side (to 34x34).  */
  auto er = erode (src.const_view (), 3U, 3U);
  auto dl = dilate (src.const_view (), 3U, 3U);
  CHECK (er.has_value () && dl.has_value ());
  CHECK (er->at (16U, 20U) == 0U);
  CHECK (er->at (17U, 20U) == 200U);
  CHECK (dl->at (15U, 20U) == 200U);
  CHECK (dl->at (14U, 20U) == 0U);

  auto grad =
    morphology_ex (src.const_view (), Morph_Op::Gradient, 3U, 3U);
  CHECK (grad.has_value ());
  CHECK (grad->at (16U, 20U) == 200U);
  CHECK (grad->at (32U, 32U) == 0U);
}

void
test_analysis_canny_and_components (void)
{
  /* Bimodal image with two separated foreground rectangles for Otsu, Canny,
     and connected components.  */
  Image_U8C1 img (80U, 60U, Scalar<uint8_t, 1U> (20U));
  for (size_t y = 10U; y < 25U; ++y)
    {
      for (size_t x = 10U; x < 30U; ++x)
        {
          img.at (y, x) = 230U;
        }
    }
  for (size_t y = 35U; y < 50U; ++y)
    {
      for (size_t x = 45U; x < 70U; ++x)
        {
          img.at (y, x) = 230U;
        }
    }

  auto otsu = otsu_threshold (img.const_view ());
  CHECK (otsu.has_value ());
  CHECK (otsu->first >= 20U && otsu->first < 230U);

  auto cc = connected_components (otsu->second.const_view (), 8);
  CHECK (cc.has_value ());
  CHECK (cc->components.size () == 2U);
  CHECK (cc->components[0].area == 15U * 20U);
  CHECK (cc->components[1].area == 15U * 25U);

  auto edges = canny (img.const_view (), 50.0F, 120.0F, 5U, 1.0F);
  CHECK (edges.has_value ());
  /* Interior of the constant background should have 0 edge response, while
     boundary around y=10, x=20 should have strong edge pixels.  */
  CHECK (edges->at (2U, 2U) == 0U);
  bool found_edge = false;
  for (size_t y = 8U; y <= 12U; ++y)
    {
      if (edges->at (y, 20U) == 255U)
        {
          found_edge = true;
        }
    }
  CHECK (found_edge);
}

} /* namespace */

int
main (void)
{
  test_color_conversions ();
  test_filters_and_integral ();
  test_geometry_and_morphology ();
  test_analysis_canny_and_components ();

  if (g_failures > 0)
    {
      std::cerr << "test_imgproc FAILED with " << g_failures << " error(s).\n";
      return 1;
    }
  std::cout << "test_imgproc PASSED.\n";
  return 0;
}
