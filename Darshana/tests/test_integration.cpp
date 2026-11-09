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

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <thread>
#include <vector>

namespace
{

/* Asserts a boolean condition and aborts with location info on failure.  */
void
require (bool cond, const char *expr, int line)
{
  if (!cond)
    {
      std::cerr << "test_integration FAILED at line " << line << ": " << expr
                << "\n";
      std::abort ();
    }
}

#define DARSHANA_REQUIRE(cond) require ((cond), #cond, __LINE__)

/* Verifies full plot lifecycle and zero base-layer redraw on interaction.  */
void
test_lifecycle_and_layered_overlays ()
{
  using namespace darshana;

  Spec spec;
  spec.title = "Layered Interactive Scatter";
  spec.width = 640U;
  spec.height = 480U;
  spec.x_scale.domain_min = 0.0;
  spec.x_scale.domain_max = 100.0;
  spec.y_scale.domain_min = 0.0;
  spec.y_scale.domain_max = 100.0;
  LayerSpec l;
  l.mark = MarkType::Scatter;
  spec.layers.push_back (l);

  auto fig_res = Figure::create (spec, BackendKind::Auto, 4096U);
  DARSHANA_REQUIRE (fig_res.has_value ());
  auto &fig = *fig_res;

  std::vector<float> xs (1000U);
  std::vector<float> ys (1000U);
  for (std::size_t i = 0U; i < 1000U; ++i)
    {
      xs[i] = static_cast<float> (i) * 0.1F;
      ys[i] = 50.0F + 25.0F * static_cast<float> (i % 4U) * 0.2F;
    }
  DARSHANA_REQUIRE (fig.bind_xy (xs, ys).has_value ());

  auto f0 = fig.render_frame ();
  DARSHANA_REQUIRE (f0.has_value ());
  DARSHANA_REQUIRE (fig.base_redraw_count () == 1U);
  DARSHANA_REQUIRE (fig.overlay_redraw_count () == 1U);

  for (int step = 0; step < 50; ++step)
    {
      OverlaySpec ov;
      ov.crosshair_enabled = true;
      ov.crosshair_x = 0.1F + 0.015F * static_cast<float> (step);
      ov.crosshair_y = 0.2F + 0.012F * static_cast<float> (step);
      ov.brush_enabled = (step % 2 == 0);
      ov.brush_x0 = 0.2F;
      ov.brush_y0 = 0.2F;
      ov.brush_x1 = 0.6F;
      ov.brush_y1 = 0.7F;
      DARSHANA_REQUIRE (fig.update_overlay (ov).has_value ());
      auto fn = fig.render_frame ();
      DARSHANA_REQUIRE (fn.has_value ());
      DARSHANA_REQUIRE (fn->gpu_allocations_in_frame == 0U);
    }

  DARSHANA_REQUIRE (fig.base_redraw_count () == 1U);
  DARSHANA_REQUIRE (fig.overlay_redraw_count () == 51U);

  auto img = fig.readback_image ();
  DARSHANA_REQUIRE (img.has_value ());
  DARSHANA_REQUIRE (img->width == 640U && img->height == 480U);

  const std::string out_svg = "/tmp/darshana_integration.svg";
  auto fut = fig.export_async (out_svg, ExportFormat::Svg);
  DARSHANA_REQUIRE (fut.get ().has_value ());
  (void)std::filesystem::remove (out_svg);
}

/* Verifies concurrent rendering of multiple Figures across threads.  */
void
test_concurrent_multi_figure_rendering ()
{
  using namespace darshana;

  constexpr std::size_t kThreads = 4U;
  std::vector<std::jthread> workers;
  workers.reserve (kThreads);

  for (std::size_t t = 0U; t < kThreads; ++t)
    {
      workers.emplace_back ([t] () {
        Spec spec;
        spec.title = "Thread Figure " + std::to_string (t);
        spec.width = 320U;
        spec.height = 240U;
        spec.x_scale.domain_min = 0.0;
        spec.x_scale.domain_max = 10.0;
        spec.y_scale.domain_min = 0.0;
        spec.y_scale.domain_max = 10.0;
        LayerSpec l;
        l.mark = (t % 2U == 0U) ? MarkType::Scatter : MarkType::Line;
        spec.layers.push_back (l);

        auto fig_res = Figure::create (spec, BackendKind::Auto, 1024U);
        DARSHANA_REQUIRE (fig_res.has_value ());
        for (int f = 0; f < 20; ++f)
          {
            DARSHANA_REQUIRE (
                fig_res
                    ->append_stream (static_cast<float> (f) * 0.4F,
                                     static_cast<float> ((f + t) % 9U))
                    .has_value ());
            DARSHANA_REQUIRE (fig_res->render_frame ().has_value ());
          }
      });
    }
}

/* Verifies dedicated RenderLoop thread and lock-free command/completion.  */
void
test_render_loop_streaming ()
{
  using namespace darshana;

  Spec spec;
  spec.title = "RenderLoop 60Hz Stream";
  spec.width = 400U;
  spec.height = 300U;
  spec.x_scale.domain_min = 0.0;
  spec.x_scale.domain_max = 100.0;
  spec.y_scale.domain_min = -10.0;
  spec.y_scale.domain_max = 10.0;
  LayerSpec l;
  l.mark = MarkType::Line;
  spec.layers.push_back (l);

  auto fig_res = Figure::create (spec, BackendKind::Auto, 2048U);
  DARSHANA_REQUIRE (fig_res.has_value ());

  auto loop_res = RenderLoop::create (*fig_res, 60U);
  DARSHANA_REQUIRE (loop_res.has_value ());
  auto &loop = *loop_res;

  uint64_t last_fid = 0U;
  for (int i = 0; i < 30; ++i)
    {
      auto r = loop.submit_stream_point (static_cast<float> (i),
                                         static_cast<float> (i % 5));
      DARSHANA_REQUIRE (r.has_value ());
      last_fid = *r;
    }

  OverlaySpec ov;
  ov.crosshair_enabled = true;
  ov.crosshair_x = 0.5F;
  ov.crosshair_y = 0.5F;
  auto ov_fid = loop.submit_overlay (ov);
  DARSHANA_REQUIRE (ov_fid.has_value ());
  last_fid = *ov_fid;

  auto done_res = loop.wait_for_frame (last_fid, 2000U);
  DARSHANA_REQUIRE (done_res.has_value ());
  DARSHANA_REQUIRE (done_res->frame_id >= last_fid);
  loop.stop ();
}

} /* namespace */

auto
main () -> int
{
  test_lifecycle_and_layered_overlays ();
  test_concurrent_multi_figure_rendering ();
  test_render_loop_streaming ();
  std::cout << "All Darshana integration tests PASSED.\n";
  return 0;
}
