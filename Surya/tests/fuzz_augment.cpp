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

#include <iostream>
#include <random>

using namespace surya;
using namespace surya::core;
using namespace surya::augment;

int
main (void)
{
  std::mt19937_64 rng (0xA064E47ULL);
  for (size_t iter = 0U; iter < 300U; ++iter)
    {
      const size_t w = 1U + (rng () % 97U);
      const size_t h = 1U + (rng () % 89U);

      Augment_Sample samp;
      samp.image = Image_U8C3 (w, h, Scalar<uint8_t, 3U> (90U, 120U, 150U));
      samp.mask = Image_U8C1 (w, h, Scalar<uint8_t, 1U> (1U));
      samp.boxes.push_back (
        Annotated_Box {Rectf (0.0F, 0.0F, static_cast<float> (w),
                              static_cast<float> (h)),
                       1, 1.0F});
      samp.keypoints.push_back (
        Annotated_Keypoint {Point2f (static_cast<float> (w) * 0.5F,
                                     static_cast<float> (h) * 0.5F),
                            2U, 0});

      Augment_Graph g;
      g.add_stage (Horizontal_Flip {0.7F});
      g.add_stage (Vertical_Flip {0.7F});
      g.add_stage (Shift_Scale_Rotate {45.0F, 0.5F, 1.5F, 0.25F, 0.9F});
      g.add_stage (Brightness_Contrast {64.0F, 0.5F, 1.8F, 0.9F});
      g.add_stage (Gamma_Adjust {0.4F, 2.5F, 0.9F});
      g.add_stage (Solarize {static_cast<uint8_t> (rng () & 0xFFU), 0.8F});
      const uint8_t p_bits = static_cast<uint8_t> (1U + (rng () % 8U));
      g.add_stage (Posterize {p_bits, 0.8F});
      g.add_stage (Coarse_Dropout {3U, 16U, 16U, 0U, 0.8F});

      auto r = g.apply (samp, rng, 0.05F);
      (void) r;
    }

  std::cout << "fuzz_augment PASSED (300 randomized DAG executions).\n";
  return 0;
}
