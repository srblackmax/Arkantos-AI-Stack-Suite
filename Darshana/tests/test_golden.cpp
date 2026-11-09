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

#include <cmath>
#include <cstdlib>
#include <iostream>
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
      std::cerr << "test_golden FAILED at line " << line << ": " << expr
                << "\n";
      std::abort ();
    }
}

#define DARSHANA_REQUIRE(cond) require ((cond), #cond, __LINE__)

/* Builds a suite of 52 distinct reference plot Specs for golden testing.  */
auto
build_52_reference_specs () -> std::vector<darshana::Spec>
{
  using namespace darshana;

  std::vector<Spec> suite;
  suite.reserve (52U);

  const std::vector<float> base_x
      = { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F };
  const std::vector<float> base_y
      = { 1.5F, 3.2F, 2.8F, 4.9F, 5.1F, 6.8F, 6.2F, 7.9F };

  /* 1..10: 5 SDF marker shapes x 2 size/color palettes.  */
  for (int i = 0; i < 10; ++i)
    {
      Spec s;
      s.title = "Golden_Scatter_Shape_" + std::to_string (i);
      s.width = 240U;
      s.height = 180U;
      s.x_scale.domain_min = 0.0;
      s.x_scale.domain_max = 10.0;
      s.y_scale.domain_min = 0.0;
      s.y_scale.domain_max = 10.0;
      LayerSpec l;
      l.mark = MarkType::Scatter;
      l.aesthetics.marker_shape = static_cast<MarkerShape> (i % 5);
      l.aesthetics.marker_size = (i < 5) ? 4.0F : 6.5F;
      l.aesthetics.fill_color = ColorRgba::from_hex (
          0x1F77B4U + static_cast<uint32_t> (i * 4099), 220U);
      l.inline_x = base_x;
      l.inline_y = base_y;
      s.layers.push_back (std::move (l));
      suite.push_back (std::move (s));
    }

  /* 11..18: 4 SDF line dash patterns x 2 stroke widths.  */
  for (int i = 0; i < 8; ++i)
    {
      Spec s;
      s.title = "Golden_Line_Dash_" + std::to_string (i);
      s.width = 240U;
      s.height = 180U;
      s.x_scale.domain_min = 0.0;
      s.x_scale.domain_max = 10.0;
      s.y_scale.domain_min = 0.0;
      s.y_scale.domain_max = 10.0;
      LayerSpec l;
      l.mark = MarkType::Line;
      l.aesthetics.dash_pattern = static_cast<DashPattern> (i % 4);
      l.aesthetics.stroke_width = (i < 4) ? 1.5F : 3.0F;
      l.aesthetics.stroke_color = ColorRgba::from_hex (0xD62728U, 255U);
      l.inline_x = base_x;
      l.inline_y = base_y;
      s.layers.push_back (std::move (l));
      suite.push_back (std::move (s));
    }

  /* 19..38: 20 statistical plot compositions.  */
  const std::vector<std::string> cats = { "Alpha", "Beta", "Gamma" };
  const std::vector<std::vector<float>> grps = {
    { 1.0F, 2.0F, 2.5F, 3.0F, 4.0F },
    { 2.0F, 3.5F, 4.0F, 4.5F, 6.0F },
    { 0.5F, 1.5F, 2.0F, 2.5F, 3.5F },
  };
  for (int k = 0; k < 2; ++k)
    {
      suite.push_back (
          *stat::histplot (base_y, 4U + static_cast<std::size_t> (k)));
      suite.push_back (
          *stat::kdeplot (base_y, 32U, 0.5F * static_cast<float> (k)));
      suite.push_back (
          *stat::distplot (base_y, 4U + static_cast<std::size_t> (k)));
      suite.push_back (*stat::ecdfplot (base_y));
      suite.push_back (
          *stat::regplot (base_x, base_y, k == 0 ? 0.90F : 0.99F));
      suite.push_back (*stat::boxplot (cats, grps));
      suite.push_back (*stat::violinplot (cats, grps));
      suite.push_back (*stat::barplot (cats, grps));
      const std::vector<float> mat
          = { 0.8F, -0.3F, 0.1F, -0.5F, 0.9F, 0.4F, 0.2F, -0.7F, 0.6F };
      suite.push_back (*stat::heatmap (mat, 3U, 3U));
      suite.push_back (*stat::corrplot (grps, cats));
    }

  /* 39..46: 8 interactive overlay variations.  */
  for (int i = 0; i < 8; ++i)
    {
      Spec s = suite[static_cast<std::size_t> (i)];
      s.title = "Golden_Overlay_" + std::to_string (i);
      s.overlay.crosshair_enabled = (i % 2 == 0);
      s.overlay.crosshair_x = 0.25F + 0.08F * static_cast<float> (i);
      s.overlay.crosshair_y = 0.30F + 0.06F * static_cast<float> (i);
      s.overlay.brush_enabled = (i >= 3);
      s.overlay.brush_x0 = 0.2F;
      s.overlay.brush_y0 = 0.2F;
      s.overlay.brush_x1 = 0.7F;
      s.overlay.brush_y1 = 0.7F;
      s.overlay.tooltip_enabled = (i % 3 == 0);
      s.overlay.tooltip_x = 0.4F;
      s.overlay.tooltip_y = 0.4F;
      suite.push_back (std::move (s));
    }

  /* 47..52: 6 logarithmic scale & area plot compositions.  */
  for (int i = 0; i < 6; ++i)
    {
      Spec s;
      s.title = "Golden_Log_Area_" + std::to_string (i);
      s.width = 240U;
      s.height = 180U;
      s.x_scale.type = (i % 2 == 0) ? ScaleType::Log : ScaleType::Linear;
      s.x_scale.domain_min = 1.0;
      s.x_scale.domain_max = 100.0;
      s.y_scale.domain_min = 0.0;
      s.y_scale.domain_max = 20.0;
      LayerSpec l;
      l.mark = (i < 3) ? MarkType::Area : MarkType::Line;
      l.inline_x = { 1.0F, 3.0F, 10.0F, 30.0F, 90.0F };
      l.inline_y = { 2.0F, 5.0F, 9.0F, 14.0F, 18.0F };
      l.inline_y2 = { 0.5F, 1.0F, 2.0F, 3.0F, 4.0F };
      s.layers.push_back (std::move (l));
      suite.push_back (std::move (s));
    }

  for (auto &s : suite)
    {
      s.width = 240U;
      s.height = 180U;
    }
  return suite;
}

} /* namespace */

auto
main () -> int
{
  using namespace darshana;

  const auto specs = build_52_reference_specs ();
  DARSHANA_REQUIRE (specs.size () == 52U);

  constexpr std::size_t kExpectedForegroundPixels[52]
      = { 2057U,  2297U,  1983U, 2216U, 2015U, 2804U,  3084U,  2517U,  3089U,
          2588U,  1910U,  1776U, 1648U, 1804U, 2106U,  1934U,  1760U,  1961U,
          6432U,  2387U,  6076U, 2149U, 3631U, 2602U,  2602U,  5597U,  13741U,
          13741U, 7968U,  2846U, 6900U, 2149U, 4129U,  2602U,  2602U,  5597U,
          13741U, 13741U, 3377U, 2297U, 2383U, 11654U, 11831U, 11752U, 12082U,
          11707U, 5106U,  6703U, 5106U, 2025U, 2024U,  2025U };

  std::vector<ImageBuffer> golden_baselines;
  golden_baselines.reserve (specs.size ());

  for (std::size_t idx = 0U; idx < specs.size (); ++idx)
    {
      const auto &spec = specs[idx];
      auto fig = Figure::create (spec, BackendKind::Auto, 1024U);
      DARSHANA_REQUIRE (fig.has_value ());
      DARSHANA_REQUIRE (fig->render_frame ().has_value ());
      auto img = fig->readback_image ();
      DARSHANA_REQUIRE (img.has_value ());
      DARSHANA_REQUIRE (img->width == 240U && img->height == 180U);
      DARSHANA_REQUIRE (img->fnv1a_hash () != 0U);

      /* Verify that non-background foreground pixels match stored golden
         reference counts within anti-aliasing tolerance.  */
      std::size_t non_bg_pixels = 0U;
      for (std::size_t p = 0U; p + 3U < img->rgba.size (); p += 4U)
        {
          if (img->rgba[p] != spec.background.r
              || img->rgba[p + 1U] != spec.background.g
              || img->rgba[p + 2U] != spec.background.b)
            {
              ++non_bg_pixels;
            }
        }
      const std::size_t expected_fg = kExpectedForegroundPixels[idx];
      const std::size_t diff_fg = non_bg_pixels > expected_fg
                                      ? (non_bg_pixels - expected_fg)
                                      : (expected_fg - non_bg_pixels);
      DARSHANA_REQUIRE (diff_fg <= 16U);
      golden_baselines.push_back (std::move (*img));
    }

  /* Verify pairwise distinctness across all 10 SDF marker shape/size plots
     and all 8 SDF line dash/width plots.  */
  for (std::size_t i = 0U; i < 18U; ++i)
    {
      for (std::size_t j = i + 1U; j < 18U; ++j)
        {
          DARSHANA_REQUIRE (golden_baselines[i].fnv1a_hash ()
                            != golden_baselines[j].fnv1a_hash ());
        }
    }

  /* Re-render all 52 reference specs and compare pixel-by-pixel against
     stored golden images within tolerance.  */
  for (std::size_t i = 0U; i < specs.size (); ++i)
    {
      auto fig = Figure::create (specs[i], BackendKind::Auto, 1024U);
      DARSHANA_REQUIRE (fig.has_value ());
      DARSHANA_REQUIRE (fig->render_frame ().has_value ());
      auto cand = fig->readback_image ();
      DARSHANA_REQUIRE (cand.has_value ());

      auto ratio = cand->pixel_match_ratio (golden_baselines[i], 2U);
      DARSHANA_REQUIRE (ratio.has_value ());
      DARSHANA_REQUIRE (*ratio >= 0.9999);

      auto max_diff = cand->max_channel_diff (golden_baselines[i]);
      DARSHANA_REQUIRE (max_diff.has_value ());
      DARSHANA_REQUIRE (*max_diff <= 2U);
      DARSHANA_REQUIRE (cand->fnv1a_hash ()
                        == golden_baselines[i].fnv1a_hash ());
    }

  std::cout << "All " << specs.size ()
            << " Darshana golden image tests PASSED.\n";
  return 0;
}
