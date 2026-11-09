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

#include <atomic>
#include <cstdlib>
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
      std::cerr << "test_stress FAILED at line " << line << ": " << expr
                << "\n";
      std::abort ();
    }
}

#define DARSHANA_REQUIRE(cond) require ((cond), #cond, __LINE__)

/* Section 8.5.1: 10,000 sequential figure creations and destructions with
   zero GPU memory leak delta.  */
void
test_10000_sequential_figure_lifecycle ()
{
  using namespace darshana;

  Spec spec;
  spec.title = "Sequential Stress";
  spec.width = 160U;
  spec.height = 120U;
  spec.x_scale.domain_min = 0.0;
  spec.x_scale.domain_max = 10.0;
  spec.y_scale.domain_min = 0.0;
  spec.y_scale.domain_max = 10.0;
  LayerSpec l;
  l.mark = MarkType::Scatter;
  l.inline_x = { 1.0F, 5.0F, 9.0F };
  l.inline_y = { 2.0F, 6.0F, 8.0F };
  spec.layers.push_back (l);

  GpuDevice dev;
  bool has_gpu = false;
  if (is_vulkan_available ())
    {
      auto d_res = GpuDevice::create ();
      if (d_res.has_value ())
        {
          dev = std::move (*d_res);
          has_gpu = true;
        }
    }

  const std::size_t baseline_gpu_bytes = has_gpu ? dev.allocated_bytes () : 0U;
  const BackendKind kind
      = has_gpu ? BackendKind::VulkanGpu : BackendKind::Auto;

  for (std::size_t i = 0U; i < 10000U; ++i)
    {
      const BackendKind iter_kind
          = (i % 100U == 0U) ? kind : BackendKind::Auto;
      auto fig_res = Figure::create (spec, iter_kind, 256U);
      DARSHANA_REQUIRE (fig_res.has_value ());
      if (i % 100U == 0U)
        {
          DARSHANA_REQUIRE (fig_res->render_frame ().has_value ());
        }
    }

  if (has_gpu)
    {
      DARSHANA_REQUIRE (dev.allocated_bytes () == baseline_gpu_bytes);
    }
}

/* Section 8.5.2: 100 concurrent figure renderings across 8 threads with no
   deadlocks, no data races (TSan), and no GPU memory leaks.  */
void
test_100_concurrent_figure_renderings_8_threads ()
{
  using namespace darshana;

  constexpr std::size_t kTotalFigures = 100U;
  constexpr std::size_t kThreads = 8U;
  std::atomic<std::size_t> next_fig{ 0U };
  std::atomic<std::size_t> completed{ 0U };

  GpuDevice dev;
  bool has_gpu = false;
  if (is_vulkan_available ())
    {
      auto d_res = GpuDevice::create ();
      if (d_res.has_value ())
        {
          dev = std::move (*d_res);
          has_gpu = true;
        }
    }
  const std::size_t before_bytes = has_gpu ? dev.allocated_bytes () : 0U;
  const BackendKind kind
      = has_gpu ? BackendKind::VulkanGpu : BackendKind::Auto;

  {
    std::vector<std::jthread> pool;
    pool.reserve (kThreads);
    for (std::size_t t = 0U; t < kThreads; ++t)
      {
        pool.emplace_back ([&] () {
          while (true)
            {
              const std::size_t idx
                  = next_fig.fetch_add (1U, std::memory_order_relaxed);
              if (idx >= kTotalFigures)
                {
                  break;
                }
              Spec spec;
              spec.title = "Concurrent_" + std::to_string (idx);
              spec.width = 160U;
              spec.height = 120U;
              spec.x_scale.domain_min = 0.0;
              spec.x_scale.domain_max = 10.0;
              spec.y_scale.domain_min = 0.0;
              spec.y_scale.domain_max = 10.0;
              LayerSpec l;
              l.mark = (idx % 2U == 0U) ? MarkType::Scatter : MarkType::Line;
              l.inline_x = { 1.0F, 3.0F, 5.0F, 7.0F };
              l.inline_y = { 2.0F, 4.0F, 6.0F, 8.0F };
              spec.layers.push_back (l);

              auto fig = Figure::create (spec, kind, 512U);
              DARSHANA_REQUIRE (fig.has_value ());
              DARSHANA_REQUIRE (fig->render_frame ().has_value ());
              completed.fetch_add (1U, std::memory_order_relaxed);
            }
        });
      }
  }

  DARSHANA_REQUIRE (completed.load () == kTotalFigures);
  if (has_gpu)
    {
      DARSHANA_REQUIRE (dev.allocated_bytes () == before_bytes);
    }
}

/* Section 8.5.3: Sustained 60 Hz ring-buffer streaming stability test
   (216,000 updates = 1 hour of 60 Hz telemetry).  */
void
test_streaming_ring_buffer_stability ()
{
  using namespace darshana;

  Spec spec;
  spec.title = "Streaming Stability";
  spec.width = 320U;
  spec.height = 240U;
  spec.x_scale.domain_min = 0.0;
  spec.x_scale.domain_max = 1000.0;
  spec.y_scale.domain_min = -10.0;
  spec.y_scale.domain_max = 10.0;
  LayerSpec l;
  l.mark = MarkType::Line;
  spec.layers.push_back (l);

  auto fig = Figure::create (spec, BackendKind::Auto, 1024U);
  DARSHANA_REQUIRE (fig.has_value ());

  constexpr int kOneHourAt60Hz = 3600 * 60;
  for (int tick = 0; tick < kOneHourAt60Hz; ++tick)
    {
      DARSHANA_REQUIRE (
          fig->append_stream (static_cast<float> (tick % 1000),
                              static_cast<float> ((tick % 20) - 10))
              .has_value ());
      if (tick % 3600 == 0)
        {
          auto st = fig->render_frame ();
          DARSHANA_REQUIRE (st.has_value ());
          DARSHANA_REQUIRE (st->gpu_allocations_in_frame == 0U);
        }
    }
  auto final_st = fig->render_frame ();
  DARSHANA_REQUIRE (final_st.has_value ());
  DARSHANA_REQUIRE (final_st->gpu_allocations_in_frame == 0U);
}

} /* namespace */

auto
main () -> int
{
  test_10000_sequential_figure_lifecycle ();
  test_100_concurrent_figure_renderings_8_threads ();
  test_streaming_ring_buffer_stability ();
  std::cout << "All Darshana stress & leak tests PASSED.\n";
  return 0;
}
