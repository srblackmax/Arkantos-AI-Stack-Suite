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

#include <darshana/darshana.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{

/* Asserts a boolean condition and aborts with location info on failure.  */
void
require (bool cond, const char *expr, int line)
{
  if (!cond)
    {
      std::cerr << "test_property FAILED at line " << line << ": " << expr
                << "\n";
      std::abort ();
    }
}

#define DARSHANA_REQUIRE(cond) require ((cond), #cond, __LINE__)

/* Property 1: For any valid Spec, JSON round-trip produces equal Spec.  */
void
test_property_spec_json_round_trip ()
{
  using namespace darshana;

  std::mt19937 rng (42U);
  std::uniform_int_distribution<uint32_t> dim_dist (120U, 1920U);
  std::uniform_int_distribution<int> byte_dist (0, 255);
  std::uniform_int_distribution<int> mark_dist (0, 12);
  std::uniform_int_distribution<int> shape_dist (0, 4);
  std::uniform_int_distribution<int> dash_dist (0, 3);
  std::uniform_real_distribution<float> f_dist (-50.0F, 50.0F);
  std::uniform_real_distribution<float> unit_dist (0.05F, 0.95F);

  for (int iter = 0; iter < 200; ++iter)
    {
      Spec s;
      s.title = "Plot_" + std::to_string (iter);
      s.subtitle = "Sub_" + std::to_string (iter * 3);
      s.x_label = "x_" + std::to_string (iter);
      s.y_label = "y_" + std::to_string (iter);
      s.width = dim_dist (rng);
      s.height = dim_dist (rng);
      s.dpi = 96.0F + static_cast<float> (iter);
      s.background = ColorRgba{
        static_cast<uint8_t> (byte_dist (rng)),
        static_cast<uint8_t> (byte_dist (rng)),
        static_cast<uint8_t> (byte_dist (rng)),
        255U,
      };
      s.show_grid = (iter % 2 == 0);
      s.show_legend = (iter % 3 == 0);

      s.x_scale.type = (iter % 3 == 1) ? ScaleType::Log : ScaleType::Linear;
      s.x_scale.domain_min = (s.x_scale.type == ScaleType::Log) ? 1.0 : -20.0;
      s.x_scale.domain_max = 100.0 + static_cast<double> (iter);
      s.y_scale.domain_min = -10.0;
      s.y_scale.domain_max = 50.0 + static_cast<double> (iter);

      s.overlay.crosshair_enabled = (iter % 2 == 1);
      s.overlay.crosshair_x = unit_dist (rng);
      s.overlay.crosshair_y = unit_dist (rng);
      s.overlay.tooltip_enabled = (iter % 4 == 0);
      s.overlay.tooltip_text = "Tip_" + std::to_string (iter);

      const int num_layers = 1 + (iter % 3);
      for (int l = 0; l < num_layers; ++l)
        {
          LayerSpec layer;
          layer.mark = static_cast<MarkType> (mark_dist (rng));
          layer.encoding.label = "Series_" + std::to_string (l);
          layer.aesthetics.marker_shape
              = static_cast<MarkerShape> (shape_dist (rng));
          layer.aesthetics.dash_pattern
              = static_cast<DashPattern> (dash_dist (rng));
          layer.aesthetics.stroke_width = 1.0F + unit_dist (rng);
          layer.aesthetics.marker_size = 2.0F + unit_dist (rng) * 4.0F;
          layer.aesthetics.opacity = unit_dist (rng);
          for (int k = 0; k < 5; ++k)
            {
              layer.inline_x.push_back (f_dist (rng));
              layer.inline_y.push_back (f_dist (rng));
            }
          s.layers.push_back (std::move (layer));
        }

      auto json_res = s.to_json ();
      DARSHANA_REQUIRE (json_res.has_value ());
      auto parsed_res = Spec::from_json (*json_res);
      DARSHANA_REQUIRE (parsed_res.has_value ());
      DARSHANA_REQUIRE (*parsed_res == s);
    }
}

/* Property 2: Scale mapping is monotonic and bounded within target range.  */
void
test_property_scale_monotonicity_and_bounds ()
{
  using namespace darshana;

  std::mt19937 rng (1337U);
  std::uniform_real_distribution<double> pos_dist (0.5, 500.0);

  for (int iter = 0; iter < 300; ++iter)
    {
      ScaleSpec scale;
      scale.clamp = true;
      scale.type
          = (iter % 3 == 0)
                ? ScaleType::Linear
                : ((iter % 3 == 1) ? ScaleType::Log : ScaleType::Categorical);

      const double d0 = pos_dist (rng);
      const double d1 = d0 + pos_dist (rng) + 1.0;
      scale.domain_min = d0;
      scale.domain_max = d1;

      const double r0 = pos_dist (rng);
      const double r1 = r0 + pos_dist (rng) + 10.0;
      const bool inverted = (iter % 2 == 1);
      scale.range_min = inverted ? r1 : r0;
      scale.range_max = inverted ? r0 : r1;

      const double r_lo = std::min (scale.range_min, scale.range_max);
      const double r_hi = std::max (scale.range_min, scale.range_max);

      double prev_mapped = 0.0;
      for (int s = 0; s <= 20; ++s)
        {
          const double frac = static_cast<double> (s) / 20.0;
          const double x = d0 + frac * (d1 - d0);
          auto m = scale.map (x);
          DARSHANA_REQUIRE (m.has_value ());
          DARSHANA_REQUIRE (*m >= r_lo - 1e-9 && *m <= r_hi + 1e-9);
          if (s > 0)
            {
              if (!inverted)
                {
                  DARSHANA_REQUIRE (*m >= prev_mapped - 1e-9);
                }
              else
                {
                  DARSHANA_REQUIRE (*m <= prev_mapped + 1e-9);
                }
            }
          prev_mapped = *m;
        }
    }
}

/* Property 3: Rendering or mutating Figure B never mutates Figure A.  */
void
test_property_figure_memory_isolation ()
{
  using namespace darshana;

  const std::vector<float> xa = { 1.0F, 2.0F, 3.0F, 4.0F };
  const std::vector<float> ya = { 2.0F, 4.0F, 6.0F, 8.0F };
  auto spec_a = stat::regplot (xa, ya, 0.95F, "Figure A");
  DARSHANA_REQUIRE (spec_a.has_value ());

  auto fig_a = Figure::create (*spec_a, BackendKind::Auto, 512U);
  DARSHANA_REQUIRE (fig_a.has_value ());
  DARSHANA_REQUIRE (fig_a->render_frame ().has_value ());
  const Spec saved_spec_a = fig_a->spec ();
  auto img_a_before = fig_a->readback_image ();
  DARSHANA_REQUIRE (img_a_before.has_value ());
  const uint64_t hash_a_before = img_a_before->fnv1a_hash ();

  for (int i = 0; i < 25; ++i)
    {
      const std::vector<float> xb = { 10.0F, 20.0F, 30.0F, 40.0F };
      auto spec_b = stat::histplot (xb, 5U, "Figure B " + std::to_string (i));
      DARSHANA_REQUIRE (spec_b.has_value ());
      auto fig_b = Figure::create (*spec_b, BackendKind::Auto, 512U);
      DARSHANA_REQUIRE (fig_b.has_value ());
      DARSHANA_REQUIRE (
          fig_b->set_viewport (0.0, 50.0, 0.0, 10.0).has_value ());
      DARSHANA_REQUIRE (fig_b->render_frame ().has_value ());
    }

  DARSHANA_REQUIRE (fig_a->spec () == saved_spec_a);
  auto img_a_after = fig_a->readback_image ();
  DARSHANA_REQUIRE (img_a_after.has_value ());
  DARSHANA_REQUIRE (img_a_after->fnv1a_hash () == hash_a_before);
}

} /* namespace */

auto
main () -> int
{
  test_property_spec_json_round_trip ();
  test_property_scale_monotonicity_and_bounds ();
  test_property_figure_memory_isolation ();
  std::cout << "All Darshana property-based tests PASSED.\n";
  return 0;
}
