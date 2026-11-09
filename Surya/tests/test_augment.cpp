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

#include <surya/augment.hpp>

#include <cmath>
#include <iostream>

using namespace surya;
using namespace surya::core;
using namespace surya::augment;

namespace
{

int g_failures = 0;

void
check_cond (bool cond, const char *expr, int line)
{
  if (!cond)
    {
      std::cerr << "FAIL [test_augment.cpp:" << line << "]: " << expr << "\n";
      ++g_failures;
    }
}

#define CHECK(expr) check_cond ((expr), #expr, __LINE__)

void
test_homography_and_lut_fusion (void)
{
  /* Double horizontal flip (p=1.0) fused into H_total must be exact identity
     on pixels, bounding boxes, and keypoints!  */
  Augment_Sample samp;
  samp.image = Image_U8C3 (100U, 80U, Scalar<uint8_t, 3U> (50U, 100U, 150U));
  samp.image.at (10U, 20U, 0U) = 250U;
  samp.boxes.push_back (
    Annotated_Box {Rectf (10.0F, 15.0F, 30.0F, 20.0F), 1, 1.0F});
  samp.keypoints.push_back (
    Annotated_Keypoint {Point2f (20.0F, 10.0F), 2U, 0});

  const Augment_Pipeline pipe (
    Horizontal_Flip {1.0F},
    Horizontal_Flip {1.0F},
    Solarize {0U, 1.0F},
    Solarize {0U, 1.0F});

  std::mt19937_64 rng (1234ULL);
  auto res = pipe.apply (samp, rng);
  CHECK (res.has_value ());
  CHECK (samp.image.at (10U, 20U, 0U) == 250U);
  CHECK (samp.boxes.size () == 1U);
  CHECK (std::abs (samp.boxes[0].rect.x - 10.0F) < 1e-4F);
  CHECK (std::abs (samp.keypoints[0].pt.x - 20.0F) < 1e-4F);

  /* Single horizontal flip on width=100 maps x=20 -> x=79.  */
  const Augment_Pipeline single_flip (Horizontal_Flip {1.0F});
  auto r2 = single_flip.apply (samp, rng);
  CHECK (r2.has_value ());
  CHECK (samp.image.at (10U, 79U, 0U) == 250U);
  CHECK (std::abs (samp.keypoints[0].pt.x - 79.0F) < 1e-4F);
}

void
test_runtime_graph_and_batch_ops (void)
{
  Augment_Graph graph;
  graph.add_stage (Horizontal_Flip {0.8F});
  graph.add_stage (Shift_Scale_Rotate {10.0F, 0.9F, 1.1F, 0.05F, 0.8F});
  graph.add_stage (Brightness_Contrast {20.0F, 0.9F, 1.1F, 0.8F});
  graph.add_stage (Gamma_Adjust {0.9F, 1.1F, 0.8F});
  graph.add_stage (Coarse_Dropout {2U, 8U, 8U, 0U, 0.8F});

  std::vector<Augment_Sample> batch (4U);
  for (auto &s : batch)
    {
      s.image = Image_U8C3 (64U, 64U, Scalar<uint8_t, 3U> (120U, 130U, 140U));
      s.boxes.push_back (
        Annotated_Box {Rectf (16.0F, 16.0F, 32.0F, 32.0F), 2, 1.0F});
    }

  auto b_res = graph.apply_batch (batch, 999ULL);
  CHECK (b_res.has_value ());
  for (const auto &s : batch)
    {
      CHECK (!s.image.empty ());
      CHECK (s.boxes.size () == 1U);
    }

  /* MixUp, CutMix, and Mosaic4 verification.  */
  Augment_Sample s1;
  s1.image = Image_U8C3 (64U, 64U, Scalar<uint8_t, 3U> (100U, 100U, 100U));
  s1.boxes.push_back (
    Annotated_Box {Rectf (10.0F, 10.0F, 20.0F, 20.0F), 0, 1.0F});

  Augment_Sample s2;
  s2.image = Image_U8C3 (64U, 64U, Scalar<uint8_t, 3U> (200U, 200U, 200U));
  s2.boxes.push_back (
    Annotated_Box {Rectf (20.0F, 20.0F, 20.0F, 20.0F), 1, 1.0F});

  auto mixed = mixup (s1, s2, 0.5F);
  CHECK (mixed.has_value ());
  CHECK (mixed->image.at (0U, 0U, 0U) == 150U);
  CHECK (mixed->boxes.size () == 2U);

  auto cm = cutmix (s1, s2, Rectu (16U, 16U, 32U, 32U));
  CHECK (cm.has_value ());
  CHECK (cm->image.at (24U, 24U, 0U) == 200U);
  CHECK (cm->image.at (4U, 4U, 0U) == 100U);

  const std::array<const Augment_Sample *, 4U> ptrs = {&s1, &s2, &s1, &s2};
  auto mos = mosaic4 (ptrs, 128U, 128U, 64U, 64U);
  CHECK (mos.has_value ());
  CHECK (mos->image.width () == 128U);
  CHECK (mos->image.height () == 128U);
  CHECK (mos->boxes.size () == 4U);
}

void
test_property_based_invariants (void)
{
  const Augment_Pipeline prop_pipe (
    Horizontal_Flip {0.5F},
    Vertical_Flip {0.5F},
    Shift_Scale_Rotate {15.0F, 0.85F, 1.15F, 0.08F, 0.9F},
    Brightness_Contrast {30.0F, 0.8F, 1.2F, 0.9F},
    Gamma_Adjust {0.8F, 1.2F, 0.9F});

  std::mt19937_64 rng (20261031ULL);
  for (size_t trial = 0U; trial < 40U; ++trial)
    {
      const size_t w = 32U + (trial * 7U) % 96U;
      const size_t h = 24U + (trial * 11U) % 80U;
      Augment_Sample s;
      s.image = Image_U8C3 (w, h, Scalar<uint8_t, 3U> (90U, 120U, 160U));
      s.boxes.push_back (Annotated_Box {
        Rectf (4.0F, 4.0F, static_cast<float> (w / 2U),
               static_cast<float> (h / 2U)),
        static_cast<int> (trial % 5U), 1.0F});

      auto r = prop_pipe.apply (s, rng);
      CHECK (r.has_value ());
      CHECK (s.image.width () == w);
      CHECK (s.image.height () == h);
      CHECK (s.image.is_row_aligned (64U));

      for (const auto &b : s.boxes)
        {
          CHECK (b.rect.x >= -1e-4F);
          CHECK (b.rect.y >= -1e-4F);
          CHECK (b.rect.x + b.rect.width <= static_cast<float> (w) + 1e-3F);
          CHECK (b.rect.y + b.rect.height <= static_cast<float> (h) + 1e-3F);
        }
    }

  /* Empty input must return a clean SuryaError, never crash.  */
  Augment_Sample empty_s;
  auto bad_res = prop_pipe.apply (empty_s, rng);
  CHECK (!bad_res.has_value ());
}

} /* namespace */

int
main (void)
{
  test_homography_and_lut_fusion ();
  test_runtime_graph_and_batch_ops ();
  test_property_based_invariants ();

  if (g_failures > 0)
    {
      std::cerr << "test_augment FAILED with " << g_failures
                << " error(s).\n";
      return 1;
    }
  std::cout << "test_augment PASSED.\n";
  return 0;
}
