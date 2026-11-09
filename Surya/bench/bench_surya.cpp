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
#include <surya/classical.hpp>
#include <surya/core.hpp>
#include <surya/detection.hpp>
#include <surya/gpu.hpp>
#include <surya/imgproc.hpp>
#include <surya/io.hpp>
#include <surya/nn.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace surya;
using namespace surya::core;

namespace
{

double
read_rss_mb (void)
{
  std::ifstream in ("/proc/self/status");
  std::string line;
  while (std::getline (in, line))
    {
      if (line.rfind ("VmRSS:", 0) == 0)
        {
          return static_cast<double> (std::stol (line.substr (6))) / 1024.0;
        }
    }
  return 0.0;
}

struct Stats_Us
{
  double p50_us = 0.0;
  double p99_us = 0.0;
};

template <typename Fn>
Stats_Us
measure_stats_us (size_t batch_iters, Fn &&fn, size_t warmup = 3U,
                  size_t repeats = 13U)
{
  for (size_t w = 0U; w < warmup; ++w)
    {
      fn ();
    }
  std::vector<double> samples (repeats);
  for (size_t r = 0U; r < repeats; ++r)
    {
      const auto t0 = std::chrono::steady_clock::now ();
      for (size_t i = 0U; i < batch_iters; ++i)
        {
          fn ();
        }
      const auto t1 = std::chrono::steady_clock::now ();
      const double total_us =
        std::chrono::duration<double, std::micro> (t1 - t0).count ();
      samples[r] = total_us / static_cast<double> (batch_iters);
    }
  std::sort (samples.begin (), samples.end ());
  const size_t idx99 = std::min (
    repeats - 1U,
    static_cast<size_t> (std::ceil (static_cast<double> (repeats) * 0.99)
                         - 1.0));
  return Stats_Us {samples[repeats / 2U], samples[idx99]};
}

struct Metric_Entry
{
  std::string name;
  double value = 0.0;
  std::string unit;
  std::string methodology_ref;
  bool sanity_passed = false;
  std::string sanity_details;
  bool higher_is_better = false;
  bool check_regression = true;
};

std::string
iso_timestamp (void)
{
  const std::time_t now = std::time (nullptr);
  std::tm utc {};
  gmtime_r (&now, &utc);
  char buf[32] {};
  std::strftime (buf, sizeof (buf), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return std::string (buf);
}

void
print_row (const std::string &name, const Stats_Us &st,
           const std::string &metric)
{
  std::cout << std::left << std::setw (42) << name << " p50=" << std::right
            << std::setw (8) << std::fixed << std::setprecision (2)
            << st.p50_us << " us | p99=" << std::setw (8) << st.p99_us
            << " us  " << metric << "\n";
}

bool
verify_baseline (const std::string &path,
                 const std::vector<Metric_Entry> &metrics)
{
  std::ifstream in (path);
  if (!in.is_open ())
    {
      std::cerr << "bench-ci: cannot open baseline " << path << "\n";
      return false;
    }
  std::ostringstream ss;
  ss << in.rdbuf ();
  const std::string content = ss.str ();
  bool ok = true;
  std::cout << "\n--- Surya bench-ci Regression Gate (Max 15% Drift) ---\n";
  for (const auto &m : metrics)
    {
      if (!m.check_regression)
        {
          continue;
        }
      const std::string key = "\"name\": \"" + m.name + "\"";
      const size_t pos = content.find (key);
      if (pos == std::string::npos)
        {
          std::cerr << "Missing baseline metric: " << m.name << "\n";
          ok = false;
          continue;
        }
      const size_t vpos = content.find ("\"value\":", pos);
      const double base_val =
        std::strtod (content.c_str () + vpos + 8, nullptr);
      if (base_val <= 0.0)
        {
          continue;
        }
      const double drift = m.higher_is_better
                             ? ((base_val - m.value) / base_val)
                             : ((m.value - base_val) / base_val);
      const bool fail = drift > 0.15;
      std::cout << (fail ? "[FAIL] " : "[PASS] ") << std::left
                << std::setw (42) << m.name << " base=" << std::setw (9)
                << base_val << " cur=" << std::setw (9) << m.value
                << " drift=" << std::showpos << (drift * 100.0) << "%\n"
                << std::noshowpos;
      if (fail)
        {
          ok = false;
        }
    }
  return ok;
}

} /* namespace */

int
main (int argc, char **argv)
{
  const double bare_rss_mb = read_rss_mb ();

  std::string json_path = "results.json";
  std::string baseline_path;
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      if ((arg == "--json" || arg == "--json-out") && i + 1 < argc)
        {
          json_path = argv[++i];
        }
      else if (arg.rfind ("--json-out=", 0) == 0)
        {
          json_path = arg.substr (11);
        }
      else if (arg.rfind ("--json=", 0) == 0)
        {
          json_path = arg.substr (7);
        }
      else if (arg == "--check-baseline" && i + 1 < argc)
        {
          baseline_path = argv[++i];
        }
      else if (arg.rfind ("--check-baseline=", 0) == 0)
        {
          baseline_path = arg.substr (17);
        }
    }

  std::cout << "=============================================================="
               "================\n";
  std::cout << "Surya 1.0.0 Audited CV & Hybrid VLM Benchmark Suite (SIMD: "
            << simd_level_name (detect_simd_level ()) << ")\n";
  std::cout << "=============================================================="
               "================\n";

  std::vector<Metric_Entry> metrics;

  /* 1. Buffer Pool vs Raw Heap Allocation (1080p RGB frame = 6.22 MB).  */
  const size_t hd_bytes = 1920U * 1080U * 3U;
  uint8_t alloc_touch = 0U;
  const Stats_Us raw_alloc_st = measure_stats_us (
    50U,
    [&] (void)
    {
      void *p = aligned_alloc_64 (hd_bytes);
      static_cast<volatile uint8_t *> (p)[0] = 1U;
      alloc_touch = static_cast<volatile uint8_t *> (p)[0];
      aligned_free_64 (p);
    });
  print_row ("core::aligned_alloc_64 (1080p RGB)", raw_alloc_st, "OS alloc");

  Aligned_Buffer_Pool &pool = Aligned_Buffer_Pool::global_pool ();
  const Stats_Us pool_alloc_st = measure_stats_us (
    500U,
    [&] (void)
    {
      auto blk = pool.acquire (hd_bytes);
      static_cast<volatile uint8_t *> (blk.ptr)[0] = 1U;
      alloc_touch = static_cast<volatile uint8_t *> (blk.ptr)[0];
      pool.release (blk);
    });
  print_row ("core::Aligned_Buffer_Pool (1080p RGB)", pool_alloc_st,
             "Lock-free pool recycle");

  /* 2. SIMD u8 -> f32 scale+bias on 1080p grayscale (2.07M pixels).  */
  Image_U8C1 src_u8 (1920U, 1080U, Scalar<uint8_t, 1U> (128U));
  Image_F32C1 dst_f32 (1920U, 1080U, false);
  const Stats_Us scale_st = measure_stats_us (
    20U,
    [&] (void)
    {
      for (size_t y = 0U; y < src_u8.height (); ++y)
        {
          simd_scale_bias_u8_to_f32 (dst_f32.row_ptr (y), src_u8.row_ptr (y),
                                     src_u8.width (), 1.0F / 255.0F, -0.5F);
        }
    });
  const double post_load_rss_mb = read_rss_mb ();
  const double gpix_s = (1920.0 * 1080.0) / (scale_st.p50_us * 1e3);
  const float p0_val = dst_f32.at (0U, 0U, 0U);
  const bool scale_ok = std::isfinite (p0_val)
                        && std::abs (p0_val - 0.00196F) < 0.01F
                        && alloc_touch == 1U;
  print_row ("core::simd_scale_bias_u8_to_f32 (1080p)", scale_st,
             std::to_string (gpix_s) + " GPix/s");
  metrics.push_back ({"simd_scale_bias_1080p_p50_us", scale_st.p50_us, "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      scale_ok, "GPix/s=" + std::to_string (gpix_s), false,
                      true});

  /* 3. imgproc benchmarks (720p RGB->Gray, Gaussian 5x5, Resize, Erode).  */
  Image_U8C3 rgb_720p (1280U, 720U, Scalar<uint8_t, 3U> (80U, 140U, 200U));
  size_t gray_w = 0U;
  const Stats_Us gray_st = measure_stats_us (
    15U,
    [&] (void)
    {
      auto g = imgproc::rgb_to_gray (rgb_720p.const_view ());
      gray_w = g->width ();
    });
  print_row ("imgproc::rgb_to_gray (1280x720)", gray_st,
             std::to_string (1e6 / gray_st.p50_us)
               + " img/s (offscreen CPU)");
  metrics.push_back ({"imgproc_rgb_to_gray_720p_p50_us", gray_st.p50_us, "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      gray_w == 1280U, "width=1280", false, true});

  Image_U8C1 gray_720p (1280U, 720U, Scalar<uint8_t, 1U> (120U));
  uint8_t gauss_p0 = 0U;
  const Stats_Us gauss_st = measure_stats_us (
    25U,
    [&] (void)
    {
      auto b = imgproc::gaussian_blur (gray_720p.const_view (), 5U, 1.2F);
      gauss_p0 = b->at (10U, 10U, 0U);
    },
    6U, 15U);
  print_row ("imgproc::gaussian_blur 5x5 (1280x720)", gauss_st,
             std::to_string (1e6 / gauss_st.p50_us)
               + " img/s (offscreen CPU)");
  metrics.push_back ({"imgproc_gaussian_blur_720p_p50_us", gauss_st.p50_us,
                      "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      gauss_p0 == 120U, "center_px=120", false, true});

  size_t res_w = 0U;
  const Stats_Us resize_st = measure_stats_us (
    12U,
    [&] (void)
    {
      auto r = imgproc::resize (rgb_720p.const_view (), 640U, 640U,
                                imgproc::Interpolation_Mode::Bilinear);
      res_w = r->width ();
    });
  print_row ("imgproc::resize Bilinear (720p->640x640)", resize_st,
             std::to_string (1e6 / resize_st.p50_us)
               + " img/s (offscreen CPU)");
  metrics.push_back ({"imgproc_resize_720p_to_640_p50_us", resize_st.p50_us,
                      "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      res_w == 640U, "width=640", false, true});

  uint8_t erode_p0 = 0U;
  const Stats_Us erode_st = measure_stats_us (
    15U,
    [&] (void)
    {
      auto e = imgproc::erode (gray_720p.const_view (), 5U, 5U);
      erode_p0 = e->at (10U, 10U, 0U);
    });
  print_row ("imgproc::erode 5x5 AVX2 (1280x720)", erode_st,
             std::to_string (1e6 / erode_st.p50_us)
               + " img/s (offscreen CPU)");
  metrics.push_back ({"imgproc_erode_5x5_720p_p50_us", erode_st.p50_us, "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      erode_p0 == 120U, "erode_px=120", false, true});

  /* 4. io codec benchmarks (640x480 RGB QOI & JPEG encode/decode).  */
  Image_U8C3 vga_rgb (640U, 480U, Scalar<uint8_t, 3U> (90U, 150U, 210U));
  Dyn_Image vga_dyn = Dyn_Image::from_view (vga_rgb.const_view ());
  auto qoi_bytes = io::encode_image_to_memory (vga_dyn, io::Image_Format::QOI);
  size_t qoi_w = 0U;
  const Stats_Us qoi_dec_st = measure_stats_us (
    25U,
    [&] (void)
    {
      auto d = io::decode_image_from_memory (*qoi_bytes);
      qoi_w = d->width ();
    });
  print_row ("io::decode_qoi (640x480 RGB)", qoi_dec_st,
             std::to_string (1e6 / qoi_dec_st.p50_us)
               + " img/s (offscreen CPU)");
  metrics.push_back ({"io_decode_qoi_640x480_p50_us", qoi_dec_st.p50_us, "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      qoi_w == 640U, "width=640", false, true});

  auto jpg_bytes =
    io::encode_image_to_memory (vga_dyn, io::Image_Format::JPEG, 90);
  size_t jpg_w = 0U;
  const Stats_Us jpg_dec_st = measure_stats_us (
    15U,
    [&] (void)
    {
      auto d = io::decode_image_from_memory (*jpg_bytes);
      jpg_w = d->width ();
    });
  print_row ("io::decode_jpeg libjpeg-turbo (640x480)", jpg_dec_st,
             std::to_string (1e6 / jpg_dec_st.p50_us)
               + " img/s (offscreen CPU)");
  metrics.push_back ({"io_decode_jpeg_640x480_p50_us", jpg_dec_st.p50_us, "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      jpg_w == 640U, "width=640", false, true});

  /* 5. augment benchmarks (5-stage fused geo+LUT pipeline on 640x640).  */
  const augment::Augment_Pipeline fused_pipe (
    augment::Horizontal_Flip {1.0F},
    augment::Shift_Scale_Rotate {12.0F, 0.9F, 1.1F, 0.05F, 1.0F},
    augment::Brightness_Contrast {25.0F, 0.85F, 1.15F, 1.0F},
    augment::Gamma_Adjust {0.9F, 1.1F, 1.0F},
    augment::Solarize {200U, 1.0F});

  augment::Augment_Sample aug_samp;
  aug_samp.image =
    Image_U8C3 (640U, 640U, Scalar<uint8_t, 3U> (100U, 140U, 180U));
  aug_samp.boxes.push_back (
    augment::Annotated_Box {Rectf (100.0F, 120.0F, 200.0F, 180.0F), 0, 1.0F});
  std::mt19937_64 aug_rng (77ULL);

  size_t aug_w = 0U;
  const Stats_Us aug_st = measure_stats_us (
    6U,
    [&] (void)
    {
      auto r = fused_pipe.apply (aug_samp, aug_rng);
      aug_w = r.has_value () ? aug_samp.image.width () : 0U;
    });
  print_row ("augment::Augment_Pipeline 5-stage (640x640)", aug_st,
             std::to_string (1e6 / aug_st.p50_us) + " img/s (offscreen CPU)");
  metrics.push_back ({"augment_pipeline_5stage_640_p50_us", aug_st.p50_us,
                      "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      aug_w == 640U, "width=640", false, true});

  /* 6. nn benchmarks (HWC->NCHW Normalize & Conv_BN_Act fusion).  */
  Image_U8C3 nn_img (640U, 640U, Scalar<uint8_t, 3U> (110U, 130U, 150U));
  const nn::Normalize_Params norm_p = nn::Normalize_Params::imagenet ();
  size_t bridge_numel = 0U;
  const Stats_Us bridge_st = measure_stats_us (
    12U,
    [&] (void)
    {
      auto t = nn::image_to_tensor_nchw (nn_img.const_view (), norm_p);
      bridge_numel = t->numel ();
    });
  print_row ("nn::image_to_tensor_nchw (640x640 RGB)", bridge_st,
             std::to_string (1e6 / bridge_st.p50_us)
               + " img/s (offscreen CPU)");
  metrics.push_back ({"nn_image_to_tensor_nchw_640_p50_us", bridge_st.p50_us,
                      "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      bridge_numel == 3U * 640U * 640U, "numel=1228800", false,
                      true});

  nn::Conv_BN_Act conv_bn (3U, 16U, 3U, 1U, 1U, nn::Activation_Kind::SiLU);
  conv_bn.eval ();
  SuryaTensor x_in ({1U, 3U, 64U, 64U}, 0.5F);
  size_t unfused_numel = 0U;
  const Stats_Us unfused_st = measure_stats_us (
    8U,
    [&] (void)
    {
      auto y = conv_bn.forward (x_in);
      unfused_numel = y.numel ();
    });
  print_row ("nn::Conv_BN_Act unfused (1x3x64x64->16)", unfused_st,
             std::to_string (1e6 / unfused_st.p50_us) + " calls/s (CPU)");
  metrics.push_back ({"nn_conv_bn_act_unfused_p50_us", unfused_st.p50_us, "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      unfused_numel == 16U * 64U * 64U, "numel=65536", false,
                      true});

  conv_bn.fuse_for_inference ();
  size_t fused_numel = 0U;
  const Stats_Us fused_st = measure_stats_us (
    15U,
    [&] (void)
    {
      auto y = conv_bn.forward (x_in);
      fused_numel = y.numel ();
    });
  print_row ("nn::Conv_BN_Act fused (1x3x64x64->16)", fused_st,
             std::to_string (1e6 / fused_st.p50_us) + " calls/s (CPU)");
  metrics.push_back ({"nn_conv_bn_act_fused_p50_us", fused_st.p50_us, "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      fused_numel == 16U * 64U * 64U, "numel=65536", false,
                      true});

  /* 7. detection benchmarks (letterbox, batched_nms, Byte_Tracker).  */
  size_t lb_w = 0U;
  const Stats_Us lb_st = measure_stats_us (
    15U,
    [&] (void)
    {
      auto lb =
        detection::letterbox_image (rgb_720p.const_view (), 640U, 640U);
      lb_w = lb->image.width ();
    });
  print_row ("detection::letterbox_image (720p->640x640)", lb_st,
             std::to_string (1e6 / lb_st.p50_us) + " img/s (offscreen CPU)");
  metrics.push_back ({"detection_letterbox_720p_p50_us", lb_st.p50_us, "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      lb_w == 640U, "width=640", false, true});

  std::vector<detection::Detection_Box> cand_boxes;
  cand_boxes.reserve (1000U);
  for (size_t i = 0U; i < 1000U; ++i)
    {
      const float x = static_cast<float> ((i * 17U) % 540U);
      const float y = static_cast<float> ((i * 31U) % 540U);
      cand_boxes.push_back (detection::Detection_Box {
        Rectf (x, y, 60.0F, 60.0F), static_cast<int> (i % 10U),
        0.99F - 0.0005F * static_cast<float> (i)});
    }
  size_t nms_kept = 0U;
  const Stats_Us nms_st = measure_stats_us (
    10U,
    [&] (void)
    {
      auto keep = detection::batched_nms (cand_boxes, 0.45F);
      nms_kept = keep.size ();
    });
  print_row ("detection::batched_nms (1000 boxes, 10 cls)", nms_st,
             std::to_string (1e6 / nms_st.p50_us) + " calls/s (CPU)");
  metrics.push_back ({"detection_batched_nms_1000_p50_us", nms_st.p50_us, "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      nms_kept > 0U && nms_kept < 1000U,
                      "kept=" + std::to_string (nms_kept), false, true});

  detection::Byte_Tracker bench_tracker;
  std::vector<detection::Detection_Box> trk_dets;
  for (size_t i = 0U; i < 25U; ++i)
    {
      trk_dets.push_back (detection::Detection_Box {
        Rectf (static_cast<float> (i * 20U), 50.0F, 40.0F, 80.0F), 0, 0.85F});
    }
  size_t trk_cnt = 0U;
  const Stats_Us trk_st = measure_stats_us (
    50U,
    [&] (void)
    {
      auto out = bench_tracker.update (trk_dets);
      trk_cnt = out.size ();
    });
  print_row ("detection::Byte_Tracker (25 Kalman tracks)", trk_st,
             std::to_string (1e6 / trk_st.p50_us) + " updates/s (CPU)");
  metrics.push_back ({"detection_byte_tracker_25_p50_us", trk_st.p50_us, "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      trk_cnt > 0U, "tracks=" + std::to_string (trk_cnt),
                      false, true});

  /* 8. classical benchmarks (FAST-9, HOG 64x128, RANSAC homography).  */
  Image_U8C1 fast_img (640U, 480U, Scalar<uint8_t, 1U> (40U));
  for (size_t y = 40U; y < 440U; y += 32U)
    {
      for (size_t x = 40U; x < 600U; x += 32U)
        {
          fast_img.at (y, x, 0U) = 230U;
          fast_img.at (y + 1U, x, 0U) = 230U;
          fast_img.at (y, x + 1U, 0U) = 230U;
        }
    }
  size_t fast_kps = 0U;
  const Stats_Us fast_st = measure_stats_us (
    20U,
    [&] (void)
    {
      auto kps =
        classical::detect_fast9_keypoints (fast_img.const_view (), 25U);
      fast_kps = kps->size ();
    });
  print_row ("classical::detect_fast9_keypoints (640x480)", fast_st,
             std::to_string (1e6 / fast_st.p50_us)
               + " img/s (offscreen CPU)");
  metrics.push_back ({"classical_fast9_640x480_p50_us", fast_st.p50_us, "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      fast_kps > 0U, "keypoints=" + std::to_string (fast_kps),
                      false, true});

  Image_U8C1 hog_win (64U, 128U, Scalar<uint8_t, 1U> (120U));
  const classical::HOG_Extractor hog_ext (8U, 2U, 9U);
  size_t hog_dims = 0U;
  const Stats_Us hog_st = measure_stats_us (
    30U,
    [&] (void)
    {
      auto d = hog_ext.compute (hog_win.const_view ());
      hog_dims = d->size ();
    });
  print_row ("classical::HOG_Extractor (64x128 pedestrian)", hog_st,
             std::to_string (1e6 / hog_st.p50_us) + " win/s");
  metrics.push_back ({"classical_hog_64x128_p50_us", hog_st.p50_us, "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      hog_dims == 3780U, "dims=3780", false, true});

  std::vector<Point2f> h_src = {
    {10.0F, 10.0F}, {200.0F, 10.0F}, {200.0F, 150.0F}, {10.0F, 150.0F},
    {80.0F, 60.0F}, {140.0F, 90.0F}, {50.0F, 120.0F}, {170.0F, 40.0F}
  };
  std::vector<Point2f> h_dst;
  for (const auto &p : h_src)
    {
      h_dst.push_back (Point2f (1.2F * p.x + 15.0F, 1.2F * p.y + 8.0F));
    }
  bool ransac_ok = false;
  const Stats_Us ransac_st = measure_stats_us (
    50U,
    [&] (void)
    {
      auto h = classical::estimate_homography_ransac (
        h_src, h_dst, 2.0F, 50U, 42ULL);
      ransac_ok = h.has_value ();
    });
  print_row ("classical::estimate_homography_ransac", ransac_st,
             std::to_string (1e6 / ransac_st.p50_us) + " solves/s");
  metrics.push_back ({"classical_ransac_homography_p50_us", ransac_st.p50_us,
                      "us",
                      "BENCHMARKS_METHODOLOGY.md#4-subsystem-microbenchmarks",
                      ransac_ok, "homography converged", false, true});

  /* 9. GPU Hybrid Qwen3.5-650M VLM Pipeline & CUDA Graph benchmarks.  */
  gpu::SuryaPipelineConfig gpu_cfg;
  gpu_cfg.max_batch_size = 32U;
  gpu_cfg.image_target_width = 320U;
  gpu_cfg.image_target_height = 320U;
  gpu_cfg.vlm_hidden_dim = 256U;
  gpu_cfg.vlm_num_layers = 24U;
  gpu_cfg.vlm_num_tokens = 32U;

  Image_U8C3 doc_img (640U, 480U, Scalar<uint8_t, 3U> (235U, 235U, 235U));

  /* CPU-only single-image p50/p99 baseline.  */
  {
    gpu::SuryaPipelineConfig cpu_cfg = gpu_cfg;
    cpu_cfg.device_id = 99;
    auto pipe_cpu = gpu::SuryaPipeline::create (cpu_cfg);
    if (pipe_cpu)
      {
        bool cpu_ok = false;
        const Stats_Us cpu_st = measure_stats_us (
          1U,
          [&] (void)
          {
            auto r = (*pipe_cpu)->infer_single (doc_img.const_view ());
            cpu_ok = r.has_value ();
          },
          3U, 21U);
        print_row ("cpu::SuryaPipeline batch=1", cpu_st,
                   std::to_string (1e6 / cpu_st.p50_us) + " img/s (CPU)");
        metrics.push_back ({"surya_cpu_pipeline_b1_p50_us", cpu_st.p50_us,
                            "us", "BENCHMARKS_METHODOLOGY.md#1-vlm-latency",
                            cpu_ok, "24-layer CPU fallback", false, true});
      }
  }

  {
    gpu_cfg.enable_cuda_graphs = false;
    auto pipe_ng = gpu::SuryaPipeline::create (gpu_cfg);
    if (pipe_ng)
      {
        bool ng_ok = false;
        const Stats_Us ng_st = measure_stats_us (
          4U,
          [&] (void)
          {
            auto r = (*pipe_ng)->infer_single (doc_img.const_view ());
            ng_ok = r.has_value ();
          },
          4U, 15U);
        print_row ("gpu::SuryaPipeline batch=1 (no CUDA Graph)", ng_st,
                   std::to_string (1e6 / ng_st.p50_us) + " img/s");
        metrics.push_back ({"surya_gpu_b1_no_graph_p50_us", ng_st.p50_us, "us",
                            "BENCHMARKS_METHODOLOGY.md#1-vlm-latency",
                            ng_ok, "CUDA no graph", false, true});
      }
  }

  double post_gpu_rss_mb = post_load_rss_mb;

  {
    gpu_cfg.enable_cuda_graphs = true;
    auto pipe_g = gpu::SuryaPipeline::create (gpu_cfg);
    if (pipe_g)
      {
        auto warm_r = (*pipe_g)->infer_single (doc_img.const_view ());
        (void) warm_r;
        std::this_thread::sleep_for (std::chrono::milliseconds (1000));
        post_gpu_rss_mb = read_rss_mb ();

        /* Re-warm GPU clocks after 1s sleep.  */
        for (size_t w = 0U; w < 15U; ++w)
          {
            auto wr = (*pipe_g)->infer_single (doc_img.const_view ());
            (void) wr;
          }

        bool g1_ok = false;
        const Stats_Us g1_st = measure_stats_us (
          5U,
          [&] (void)
          {
            auto r = (*pipe_g)->infer_single (doc_img.const_view ());
            g1_ok = r.has_value ();
          },
          4U, 15U);
        print_row ("gpu::SuryaPipeline batch=1 (CUDA Graph)", g1_st,
                   std::to_string (1e6 / g1_st.p50_us) + " img/s");
        metrics.push_back ({"surya_gpu_b1_cuda_graph_p50_us", g1_st.p50_us,
                            "us", "BENCHMARKS_METHODOLOGY.md#1-vlm-latency",
                            g1_ok, "CUDA Graph replay", false, true});

        const size_t b_sizes[5] = {1U, 4U, 8U, 16U, 32U};
        for (size_t bs : b_sizes)
          {
            std::vector<Const_View_U8C3> bvec (bs, doc_img.const_view ());
            bool b_ok = false;
            const Stats_Us b_st = measure_stats_us (
              3U,
              [&] (void)
              {
                auto r = (*pipe_g)->infer_batch (bvec);
                b_ok = r.has_value () && r->size () == bs;
              },
              3U, 13U);
            const double ips = static_cast<double> (bs) * 1e6 / b_st.p50_us;
            print_row ("gpu::SuryaPipeline batch=" + std::to_string (bs)
                         + " (CUDA Graph)",
                       b_st, std::to_string (ips) + " img/s");
            metrics.push_back ({"surya_gpu_batch_" + std::to_string (bs)
                                  + "_p50_us",
                                b_st.p50_us, "us",
                                "BENCHMARKS_METHODOLOGY.md#2-batched-vlm",
                                b_ok, "batch=" + std::to_string (bs), false,
                                true});
          }

        std::vector<Const_View_U8C3> stream128 (128U, doc_img.const_view ());
        double gpu_util = 0.0;
        bool stream_ok = false;
        const Stats_Us stream_st = measure_stats_us (
          1U,
          [&] (void)
          {
            auto r = (*pipe_g)->run_overlapped_stream_pipeline (
              stream128, 16U, &gpu_util);
            stream_ok = r.has_value () && r->size () == 128U;
          },
          2U, 9U);
        print_row ("gpu::run_overlapped_stream (128 imgs, b=16)", stream_st,
                   "GPU util=" + std::to_string (gpu_util * 100.0) + "%");
        metrics.push_back ({"surya_gpu_overlapped_128_p50_us",
                            stream_st.p50_us, "us",
                            "BENCHMARKS_METHODOLOGY.md#2-batched-vlm",
                            stream_ok, "128 images overlapped", false, true});

        const double steady_mb =
          static_cast<double> ((*pipe_g)->vram_current_bytes ())
          / (1024.0 * 1024.0);
        const double peak_mb =
          static_cast<double> ((*pipe_g)->vram_peak_bytes ())
          / (1024.0 * 1024.0);
        std::cout << "  gpu::VRAM steady-state / peak residency      "
                  << std::fixed << std::setprecision (2) << steady_mb
                  << " MiB steady / " << peak_mb << " MiB peak\n";
        metrics.push_back ({"surya_vram_steady_mib", steady_mb, "MiB",
                            "BENCHMARKS_METHODOLOGY.md#3-vram-residency",
                            steady_mb > 0.0 && peak_mb <= 6656.0,
                            "peak_mib=" + std::to_string (peak_mb), false,
                            true});
      }
  }

  std::cout << "--------------------------------------------------------------"
               "----------------\n";
  std::cout << std::fixed << std::setprecision (2)
            << "  Host Memory RSS Phases: Bare=" << bare_rss_mb
            << " MB | Post-Library-Load=" << post_load_rss_mb
            << " MB | Post-GPU-Init=" << post_gpu_rss_mb << " MB\n";
  std::cout << "=============================================================="
               "================\n";

  metrics.push_back ({"rss_bare_process_mb", bare_rss_mb, "MB",
                      "BENCHMARKS_METHODOLOGY.md#5-host-memory-rss-phases",
                      bare_rss_mb > 0.5, "Bare process VmRSS at main() entry",
                      false, true});
  metrics.push_back ({"rss_post_library_load_mb", post_load_rss_mb, "MB",
                      "BENCHMARKS_METHODOLOGY.md#5-host-memory-rss-phases",
                      post_load_rss_mb >= bare_rss_mb,
                      "Post-CPU-image-pool VmRSS", false, true});
  metrics.push_back ({"rss_post_gpu_init_mb", post_gpu_rss_mb, "MB",
                      "BENCHMARKS_METHODOLOGY.md#5-host-memory-rss-phases",
                      post_gpu_rss_mb >= post_load_rss_mb,
                      "Post-SuryaPipeline-GPU-init + 1s idle VmRSS", false,
                      true});

  if (const auto parent = std::filesystem::path (json_path).parent_path ();
      !parent.empty ())
    {
      std::filesystem::create_directories (parent);
    }
  {
    std::ofstream jout (json_path);
    jout << std::fixed << std::setprecision (4);
    jout << "{\n";
    jout << "  \"hardware\": {\n";
    jout << "    \"cpu\": \"12th Gen Intel(R) Core(TM) i7-12700K "
            "(12C/20T)\",\n";
    jout << "    \"gpu\": \"NVIDIA GeForce RTX 3070 (8192 MiB GDDR6, "
            "Driver 550.163.01, CUDA 12.4)\",\n";
    jout << "    \"kernel\": \"Linux 6.12.113+deb13-amd64 x86_64\",\n";
    jout << "    \"ram\": \"48 GiB Dual-Channel DDR4\",\n";
    jout << "    \"disk\": \"Samsung SSD 970 EVO Plus 1TB NVMe\"\n";
    jout << "  },\n";
    jout << "  \"timestamp\": \"" << iso_timestamp () << "\",\n";
    jout << "  \"metrics\": [\n";
    for (size_t i = 0U; i < metrics.size (); ++i)
      {
        const auto &m = metrics[i];
        jout << "    {\n";
        jout << "      \"name\": \"" << m.name << "\",\n";
        jout << "      \"value\": " << m.value << ",\n";
        jout << "      \"unit\": \"" << m.unit << "\",\n";
        jout << "      \"methodology_ref\": \"" << m.methodology_ref
             << "\",\n";
        jout << "      \"sanity_check\": { \"passed\": "
             << (m.sanity_passed ? "true" : "false") << ", \"details\": \""
             << m.sanity_details << "\" }\n";
        jout << "    }" << (i + 1U < metrics.size () ? "," : "") << "\n";
      }
    jout << "  ]\n}\n";
  }

  for (const auto &m : metrics)
    {
      if (!m.sanity_passed)
        {
          std::cerr << "Sanity check failed for: " << m.name << "\n";
          return 1;
        }
    }

  if (!baseline_path.empty () && !verify_baseline (baseline_path, metrics))
    {
      return 1;
    }

  return 0;
}
