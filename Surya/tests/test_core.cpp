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

#include <surya/core.hpp>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

using namespace surya::core;

namespace
{

int g_failures = 0;

void
check_cond (bool cond, const char *expr, int line)
{
  if (!cond)
    {
      std::cerr << "FAIL [test_core.cpp:" << line << "]: " << expr << "\n";
      ++g_failures;
    }
}

#define CHECK(expr) check_cond ((expr), #expr, __LINE__)

void
test_types_and_geometry (void)
{
  CHECK (saturate_cast<uint8_t> (-15) == 0U);
  CHECK (saturate_cast<uint8_t> (300) == 255U);
  CHECK (saturate_cast<uint8_t> (127.6F) == 128U);
  CHECK (saturate_cast<int16_t> (-40000) == -32768);

  CHECK (border_interpolate (-1, 10, Border_Mode::Clamp) == 0);
  CHECK (border_interpolate (12, 10, Border_Mode::Clamp) == 9);
  CHECK (border_interpolate (-1, 10, Border_Mode::Reflect_101) == 1);
  CHECK (border_interpolate (-2, 10, Border_Mode::Reflect_101) == 2);
  CHECK (border_interpolate (10, 10, Border_Mode::Reflect_101) == 8);
  CHECK (border_interpolate (-1, 10, Border_Mode::Constant) == -1);
  CHECK (border_interpolate (-1, 10, Border_Mode::Wrap) == 9);

  const Rectf r1 (10.0F, 10.0F, 20.0F, 20.0F);
  const Rectf r2 (20.0F, 20.0F, 20.0F, 20.0F);
  const Rectf inter = r1.intersect (r2);
  CHECK (std::abs (inter.area () - 100.0F) < 1e-5F);
  /* Union = 400 + 400 - 100 = 700, IoU = 100 / 700 = 1/7.  */
  CHECK (std::abs (r1.iou (r2) - (1.0 / 7.0)) < 1e-6);
}

void
test_arena_and_pool (void)
{
  Arena_Allocator arena (8192U);
  CHECK (arena.used_bytes () == 0U);
  float *p1 = arena.allocate_array<float> (32U);
  CHECK (p1 != nullptr);
  CHECK (reinterpret_cast<uintptr_t> (p1) % 64U == 0U);
  {
    Scoped_Arena_Mark mark (arena);
    uint8_t *p2 = arena.allocate_array<uint8_t> (256U);
    CHECK (p2 != nullptr);
    CHECK (reinterpret_cast<uintptr_t> (p2) % 64U == 0U);
    CHECK (arena.used_bytes () > 128U);
  }
  CHECK (arena.used_bytes () == 128U);
  arena.reset ();
  CHECK (arena.used_bytes () == 0U);

  Aligned_Buffer_Pool pool (8U);
  auto a1 = pool.acquire (1000U);
  CHECK (a1.ptr != nullptr);
  CHECK (reinterpret_cast<uintptr_t> (a1.ptr) % 64U == 0U);
  CHECK (a1.capacity_bytes == 1024U);
  void *orig_ptr = a1.ptr;
  pool.release (a1);

  auto a2 = pool.acquire (900U);
  CHECK (a2.ptr == orig_ptr);
  CHECK (pool.cache_hits () == 1U);
  pool.release (a2);

  /* Multi-threaded stress test on Aligned_Buffer_Pool.  */
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t)
    {
      threads.emplace_back (
        [&pool] (void)
        {
          for (int iter = 0; iter < 50; ++iter)
            {
              auto blk = pool.acquire (4096U);
              CHECK (blk.ptr != nullptr);
              pool.release (blk);
            }
        });
    }
  for (auto &th : threads)
    {
      th.join ();
    }
}

void
test_image_and_views (void)
{
  /* Odd width (13 pixels * 3 channels = 39 bytes per row) must still be
     padded to 64 bytes per scanline.  */
  Image_U8C3 img (13U, 7U, Scalar<uint8_t, 3U> (10U, 20U, 30U));
  CHECK (!img.empty ());
  CHECK (img.width () == 13U);
  CHECK (img.height () == 7U);
  CHECK (img.row_stride_bytes () == 64U);
  CHECK (img.is_row_aligned (64U));

  for (size_t y = 0U; y < img.height (); ++y)
    {
      const uintptr_t row_addr =
        reinterpret_cast<uintptr_t> (img.row_ptr (y));
      CHECK (row_addr % 64U == 0U);
    }

  CHECK (img.at (3U, 5U, 0U) == 10U);
  CHECK (img.at (3U, 5U, 1U) == 20U);
  CHECK (img.at (3U, 5U, 2U) == 30U);

  auto roi_res = img.subview (Rectu (2U, 1U, 5U, 4U));
  CHECK (roi_res.has_value ());
  View_U8C3 roi = *roi_res;
  CHECK (roi.width () == 5U);
  CHECK (roi.height () == 4U);
  roi.at (0U, 0U, 0U) = 99U;
  CHECK (img.at (1U, 2U, 0U) == 99U);

  /* Out-of-bounds ROI must fail cleanly with Error_Code::Out_Of_Bounds.  */
  auto bad_roi = img.subview (Rectu (10U, 1U, 5U, 4U));
  CHECK (!bad_roi.has_value ());
  CHECK (bad_roi.error ().code () == Error_Code::Out_Of_Bounds);

  /* Planar CHW image test.  */
  Image_F32C3_Planar planar (17U, 5U, Scalar<float, 3U> (1.0F, 2.0F, 3.0F));
  CHECK (planar.is_row_aligned (64U));
  CHECK (planar.row_stride_bytes () == 128U);
  CHECK (std::abs (planar.at (2U, 8U, 0U) - 1.0F) < 1e-6F);
  CHECK (std::abs (planar.at (2U, 8U, 1U) - 2.0F) < 1e-6F);
  CHECK (std::abs (planar.at (2U, 8U, 2U) - 3.0F) < 1e-6F);

  /* Dyn_Image round-trip test.  */
  Dyn_Image dyn = Dyn_Image::from_view (img.const_view ());
  CHECK (!dyn.empty ());
  CHECK (dyn.width () == 13U);
  CHECK (dyn.height () == 7U);
  CHECK (dyn.channels () == 3U);
  CHECK (dyn.depth () == Pixel_Depth::U8);

  auto typed_view = dyn.try_view<uint8_t, 3U> ();
  CHECK (typed_view.has_value ());
  CHECK (typed_view->at (1U, 2U, 0U) == 99U);

  auto wrong_view = dyn.try_view<float, 3U> ();
  CHECK (!wrong_view.has_value ());
  CHECK (wrong_view.error ().code () == Error_Code::Format_Mismatch);
}

void
test_simd_and_parallel (void)
{
  const Simd_Level lvl = detect_simd_level ();
  CHECK (!simd_level_name (lvl).empty ());

  std::vector<float> a (129U, 2.0F);
  std::vector<float> b (129U, 3.5F);
  const float dot = simd_dot_f32 (a.data (), b.data (), a.size ());
  CHECK (std::abs (dot - 129.0F * 7.0F) < 1e-3F);

  std::vector<uint8_t> u1 (77U, 200U);
  std::vector<uint8_t> u2 (77U, 100U);
  std::vector<uint8_t> diff (77U, 0U);
  std::vector<uint8_t> sat (77U, 0U);
  simd_abs_diff_u8 (diff.data (), u2.data (), u1.data (), u1.size ());
  simd_add_sat_u8 (sat.data (), u1.data (), u2.data (), u1.size ());
  for (size_t i = 0U; i < u1.size (); ++i)
    {
      CHECK (diff[i] == 100U);
      CHECK (sat[i] == 255U);
    }

  std::vector<float> scaled (77U, 0.0F);
  simd_scale_bias_u8_to_f32 (scaled.data (), u2.data (), u2.size (), 0.5F,
                             -10.0F);
  for (size_t i = 0U; i < scaled.size (); ++i)
    {
      CHECK (std::abs (scaled[i] - 40.0F) < 1e-5F);
    }

  const uint64_t desc_a[4] = {0xFFFFFFFFFFFFFFFFULL, 0ULL, 0xF0F0ULL, 1ULL};
  const uint64_t desc_b[4] = {0ULL, 0ULL, 0ULL, 0ULL};
  CHECK (hamming_distance_256 (desc_a, desc_b) == 64U + 8U + 1U);

  Image_I32C1 grid (200U, 150U, true);
  parallel_for_2d (
    grid.height (), grid.width (), 32U, 32U,
    [&] (size_t y0, size_t y1, size_t x0, size_t x1)
    {
      for (size_t y = y0; y < y1; ++y)
        {
          int32_t *row = grid.row_ptr (y);
          for (size_t x = x0; x < x1; ++x)
            {
              row[x] = static_cast<int32_t> (y * 1000U + x);
            }
        }
    });

  for (size_t y = 0U; y < grid.height (); ++y)
    {
      for (size_t x = 0U; x < grid.width (); ++x)
        {
          CHECK (grid.at (y, x) == static_cast<int32_t> (y * 1000U + x));
        }
    }
}

void
test_avx512_numerical_regression_and_misaligned (void)
{
  /* Compare AVX-512/AVX2 runtime-dispatched kernels against scalar reference
     within <= 1e-6F, including deliberately misaligned pointers (+1, +3).  */
  std::vector<float> buf_a (263U, 0.0F);
  std::vector<float> buf_b (263U, 0.0F);
  for (size_t i = 0U; i < buf_a.size (); ++i)
    {
      buf_a[i] = 0.01F * static_cast<float> ((i * 17U) % 97U) + 0.1F;
      buf_b[i] = 0.01F * static_cast<float> ((i * 31U) % 89U) + 0.2F;
    }

  for (size_t off = 0U; off < 4U; ++off)
    {
      const size_t n = 251U;
      const float *pa = buf_a.data () + off;
      const float *pb = buf_b.data () + off;

      const float d_simd = simd_dot_f32 (pa, pb, n);
      const float d_ref = simd_dot_f32_scalar (pa, pb, n);
      const float rel_dot =
        std::abs (d_simd - d_ref) / std::max (1.0F, std::abs (d_ref));
      CHECK (rel_dot <= 1e-6F);

      const float ssim_simd = simd_ssim_f32 (pa, pb, n);
      const float ssim_ref = simd_ssim_f32_scalar (pa, pb, n);
      CHECK (std::abs (ssim_simd - ssim_ref) <= 1e-6F);
    }

  const float k5[5] = {0.0625F, 0.25F, 0.375F, 0.25F, 0.0625F};
  std::vector<float> conv_simd (240U, 0.0F);
  std::vector<float> conv_ref (240U, 0.0F);
  simd_conv1d_f32 (conv_simd.data () + 1, buf_a.data () + 1, k5, 235U, 5U);
  simd_conv1d_f32_scalar (conv_ref.data () + 1, buf_a.data () + 1, k5, 235U,
                          5U);
  for (size_t i = 0U; i < conv_simd.size (); ++i)
    {
      CHECK (std::abs (conv_simd[i] - conv_ref[i]) <= 1e-6F);
    }
}

void
test_pmr_lockfree_and_observability (void)
{
  Arena_Allocator backing_arena (4096U);
  Pmr_Arena_Resource pmr_res (backing_arena);
  {
    std::pmr::vector<int> vec (&pmr_res);
    for (int i = 0; i < 64; ++i)
      {
        vec.push_back (i * 3);
      }
    CHECK (vec[10] == 30);
    CHECK (backing_arena.used_bytes () > 0U);
  }
  backing_arena.reset ();
  CHECK (backing_arena.used_bytes () == 0U);

  Lock_Free_Queue<uint64_t, 64U> lfq;
  CHECK (lfq.approx_size () == 0U);
  for (uint64_t v = 1ULL; v <= 32ULL; ++v)
    {
      CHECK (lfq.try_enqueue (v));
    }
  uint64_t sum = 0ULL;
  while (auto out_v = lfq.try_dequeue ())
    {
      sum += *out_v;
    }
  CHECK (sum == (32ULL * 33ULL) / 2ULL);

  SuryaStats &st = SuryaStats::global_stats ();
  st.reset ();
  st.add_preprocess_ns (1200ULL);
  st.inc_graph_launch ();
  const SuryaStatsSnapshot snap = st.snapshot ();
  CHECK (snap.preprocess_ns == 1200ULL);
  CHECK (snap.cuda_graph_launches == 1ULL);

  ScopedProfileRange prof ("unit_test_scope");
}

} /* namespace */

int
main (void)
{
  test_types_and_geometry ();
  test_arena_and_pool ();
  test_image_and_views ();
  test_simd_and_parallel ();
  test_avx512_numerical_regression_and_misaligned ();
  test_pmr_lockfree_and_observability ();

  if (g_failures > 0)
    {
      std::cerr << "test_core FAILED with " << g_failures << " error(s).\n";
      return 1;
    }
  std::cout << "test_core PASSED (SIMD level: "
            << simd_level_name (detect_simd_level ()) << ").\n";
  return 0;
}
