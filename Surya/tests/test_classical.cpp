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

#include <surya/surya.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace surya;
using namespace surya::core;
using namespace surya::classical;

namespace
{

int g_failures = 0;

void
check_cond (bool cond, const char *expr, int line)
{
  if (!cond)
    {
      std::cerr << "FAIL [test_classical.cpp:" << line << "]: " << expr
                << "\n";
      ++g_failures;
    }
}

#define CHECK(expr) check_cond ((expr), #expr, __LINE__)

void
test_fast9_orb_and_hamming_matching (void)
{
  Image_U8C1 img1 (96U, 96U, Scalar<uint8_t, 1U> (30U));
  Image_U8C1 img2 (96U, 96U, Scalar<uint8_t, 1U> (30U));

  /* Draw bright distinct squares (shifted by +6px in img2).  */
  for (size_t y = 24U; y < 44U; ++y)
    {
      for (size_t x = 24U; x < 44U; ++x)
        {
          img1.at (y, x, 0U) = 230U;
          img2.at (y, x + 6U, 0U) = 230U;
        }
    }
  for (size_t y = 52U; y < 74U; ++y)
    {
      for (size_t x = 50U; x < 72U; ++x)
        {
          img1.at (y, x, 0U) = 200U;
          img2.at (y, x + 6U, 0U) = 200U;
        }
    }

  auto kps = detect_fast9_keypoints (img1.const_view (), 25U, true, 500U);
  CHECK (kps.has_value () && !kps->empty ());

  auto orb1 = extract_orb_features (img1.const_view (), 25U, 200U);
  auto orb2 = extract_orb_features (img2.const_view (), 25U, 200U);
  CHECK (orb1.has_value () && !orb1->empty ());
  CHECK (orb2.has_value () && !orb2->empty ());

  auto matches = match_descriptors_hamming (*orb1, *orb2, 0.9F, 64U);
  CHECK (!matches.empty ());
  CHECK (matches[0].distance == 0U);
}

void
test_ransac_homography_and_lk_flow (void)
{
  /* 1. RANSAC Homography with 8 inliers + 2 gross outliers.  */
  std::vector<Point2f> src_pts = {
    {10.0F, 10.0F}, {110.0F, 10.0F}, {110.0F, 90.0F}, {10.0F, 90.0F},
    {50.0F, 30.0F}, {80.0F, 60.0F}, {30.0F, 70.0F}, {95.0F, 25.0F},
    {40.0F, 40.0F}, {60.0F, 60.0F}
  };
  std::vector<Point2f> dst_pts;
  for (size_t i = 0U; i < 8U; ++i)
    {
      /* True homography: scale 1.5x + translate (+20, +15).  */
      dst_pts.push_back (Point2f (
        1.5F * src_pts[i].x + 20.0F, 1.5F * src_pts[i].y + 15.0F));
    }
  /* Add 2 gross outliers.  */
  dst_pts.push_back (Point2f (500.0F, -300.0F));
  dst_pts.push_back (Point2f (-250.0F, 420.0F));

  auto h_res =
    estimate_homography_ransac (src_pts, dst_pts, 2.0F, 300U, 42ULL);
  CHECK (h_res.has_value ());
  CHECK (h_res->num_inliers == 8U);

  const Point2f p_test = h_res->project_point (Point2f (20.0F, 40.0F));
  CHECK (std::fabs (p_test.x - 50.0F) < 0.5F);
  CHECK (std::fabs (p_test.y - 75.0F) < 0.5F);

  /* 2. Template matching (NCC, SAD, SSD) & Lucas-Kanade Optical Flow.  */
  Image_U8C1 prev (64U, 64U, Scalar<uint8_t, 1U> (40U));
  Image_U8C1 next (64U, 64U, Scalar<uint8_t, 1U> (40U));
  Image_U8C1 templ (12U, 12U, Scalar<uint8_t, 1U> (40U));

  for (size_t y = 0U; y < 12U; ++y)
    {
      for (size_t x = 0U; x < 12U; ++x)
        {
          const uint8_t val =
            static_cast<uint8_t> (80U + y * 10U + x * 5U);
          templ.at (y, x, 0U) = val;
          prev.at (20U + y, 22U + x, 0U) = val;
          next.at (21U + y, 24U + x, 0U) = val;
        }
    }

  auto tm_ncc = match_template (
    prev.const_view (), templ.const_view (), Template_Match_Mode::NCC);
  CHECK (tm_ncc.has_value ());
  CHECK (tm_ncc->best_x == 22U && tm_ncc->best_y == 20U);
  CHECK (tm_ncc->best_score > 0.99F);

  const std::vector<Point2f> lk_pts = {Point2f (26.0F, 26.0F)};
  auto flow_res = calc_optical_flow_lk (
    prev.const_view (), next.const_view (), lk_pts, 6U, 25U, 0.01F);
  CHECK (flow_res.has_value ());
  CHECK (flow_res->status[0] == 1U);
  CHECK (std::fabs (flow_res->next_points[0].x - 28.0F) < 0.6F);
  CHECK (std::fabs (flow_res->next_points[0].y - 27.0F) < 0.6F);
}

void
test_bellows_hog_svm_bovw_and_forest (void)
{
  /* Create synthetic horizontal-stripe (class 0) vs vertical-stripe (class 1)
     patches of size 16x16.  */
  std::vector<Image_U8C1> patch_imgs;
  std::vector<Const_View_U8C1> patch_views;
  std::vector<int> labels;

  for (int i = 0; i < 12; ++i)
    {
      Image_U8C1 p (16U, 16U, Scalar<uint8_t, 1U> (20U));
      const bool is_vert = (i >= 6);
      for (size_t y = 0U; y < 16U; ++y)
        {
          for (size_t x = 0U; x < 16U; ++x)
            {
              if (is_vert)
                {
                  p.at (y, x, 0U) = ((x / 2U) % 2U == 0U)
                                      ? static_cast<uint8_t> (220U - i * 2)
                                      : 20U;
                }
              else
                {
                  p.at (y, x, 0U) = ((y / 2U) % 2U == 0U)
                                      ? static_cast<uint8_t> (220U - i * 2)
                                      : 20U;
                }
            }
        }
      patch_imgs.push_back (std::move (p));
      labels.push_back (is_vert ? 1 : 0);
    }

  for (const auto &img : patch_imgs)
    {
      patch_views.push_back (img.const_view ());
    }

  /* 1. HOG + Bellows Max-Margin Classifier (SVM).  */
  HOG_SVM_Classifier hog_svm (HOG_Extractor (8U, 2U, 9U), 0.005, 0.1, 300U);
  auto fit_svm = hog_svm.fit (patch_views, labels);
  CHECK (fit_svm.has_value ());

  auto pred0 = hog_svm.predict (patch_views[0]);
  auto pred1 = hog_svm.predict (patch_views[11]);
  CHECK (pred0.has_value () && *pred0 == 0);
  CHECK (pred1.has_value () && *pred1 == 1);

  auto probs1 = hog_svm.predict_probabilities (patch_views[11]);
  CHECK (probs1.has_value () && probs1->size () == 2U);
  CHECK ((*probs1)[1] > (*probs1)[0]);

  /* 2. HOG + Bellows Random Forest (Subspace Tree Ensemble).  */
  Patch_Forest_Classifier forest (HOG_Extractor (8U, 2U, 9U), 25U, 5U, 42ULL);
  auto fit_rf = forest.fit (patch_views, labels);
  CHECK (fit_rf.has_value ());
  auto rf_p0 = forest.predict (patch_views[0]);
  auto rf_p1 = forest.predict (patch_views[11]);
  CHECK (rf_p0.has_value () && *rf_p0 == 0);
  CHECK (rf_p1.has_value () && *rf_p1 == 1);

  /* 3. Bag-of-Visual-Words via Bellows Centroid Partitioner (K-Means).  */
  SuryaTensor desc_tab ({16U, 4U}, 0.0F);
  for (size_t r = 0U; r < 16U; ++r)
    {
      const float base = (r < 8U) ? 10.0F : 100.0F;
      for (size_t c = 0U; c < 4U; ++c)
        {
          desc_tab.at_2d (r, c) = base + static_cast<float> (r % 3U);
        }
    }

  Visual_Vocabulary_BoVW bovw (2U, 100U, 42ULL);
  auto fit_bovw = bovw.fit (desc_tab);
  CHECK (fit_bovw.has_value ());

  auto hist = bovw.encode_histogram (desc_tab);
  CHECK (hist.has_value () && hist->size () == 2U);
  CHECK (std::fabs ((*hist)[0] - 0.5) < 1e-6);
  CHECK (std::fabs ((*hist)[1] - 0.5) < 1e-6);
}

} /* namespace */

int
main (void)
{
  test_fast9_orb_and_hamming_matching ();
  test_ransac_homography_and_lk_flow ();
  test_bellows_hog_svm_bovw_and_forest ();

  if (g_failures != 0)
    {
      std::cerr << "test_classical FAILED with " << g_failures
                << " error(s).\n";
      return EXIT_FAILURE;
    }
  std::cout << "test_classical PASSED all assertions.\n";
  return EXIT_SUCCESS;
}
