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

#include <atomic>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

using namespace surya;
using namespace surya::core;
using namespace surya::gpu;

namespace
{

int g_failures = 0;

void
check_cond (bool cond, const char *expr, int line)
{
  if (!cond)
    {
      std::cerr << "FAIL [test_gpu_hybrid.cpp:" << line << "]: " << expr
                << "\n";
      ++g_failures;
    }
}

#define CHECK(expr) check_cond ((expr), #expr, __LINE__)

struct WarningCounterSink final : public SuryaLogSink
{
  std::atomic<uint64_t> warn_count {0ULL};

  void
  write (Log_Level level, std::string_view /*subsystem*/,
         std::string_view /*message*/, uint64_t /*ts_ns*/) noexcept override
  {
    if (level == Log_Level::Warn)
      {
        warn_count.fetch_add (1ULL, std::memory_order_relaxed);
      }
  }
};

void
test_gpu_cpu_numerical_equivalence (void)
{
  Image_U8C3 img1 (137U, 93U, false);
  Image_U8C3 img2 (137U, 93U, false);
  for (size_t y = 0U; y < 93U; ++y)
    {
      for (size_t x = 0U; x < 137U; ++x)
        {
          img1.at (y, x, 0U) = static_cast<uint8_t> ((x * 3U + y) & 255U);
          img1.at (y, x, 1U) = static_cast<uint8_t> ((x + y * 5U) & 255U);
          img1.at (y, x, 2U) = static_cast<uint8_t> ((x * 7U + y * 2U) & 255U);

          img2.at (y, x, 0U) = static_cast<uint8_t> ((255U - x) & 255U);
          img2.at (y, x, 1U) = static_cast<uint8_t> ((128U + y) & 255U);
          img2.at (y, x, 2U) = static_cast<uint8_t> ((x ^ y) & 255U);
        }
    }

  const std::array<Const_View_U8C3, 2U> batch = {img1.const_view (),
                                                 img2.const_view ()};

  auto cpu_prep = batched_preprocess_cpu (batch, 64U, 64U);
  auto gpu_prep = batched_preprocess_gpu (batch, 64U, 64U, 0);
  CHECK (cpu_prep.has_value ());
  CHECK (gpu_prep.has_value ());
  CHECK (cpu_prep->allclose (*gpu_prep, 1e-4F, 1e-4F));

  /* Dense confidence filter + parallel NMS equivalence on 200 boxes.  */
  std::vector<detection::Detection_Box> cands;
  cands.reserve (200U);
  for (size_t i = 0U; i < 200U; ++i)
    {
      const float x = static_cast<float> ((i * 19U) % 240U);
      const float y = static_cast<float> ((i * 29U) % 240U);
      const float score = 0.10F + 0.88F * static_cast<float> (i % 17U) / 16.0F;
      cands.push_back (detection::Detection_Box {
        Rectf (x, y, 48.0F, 36.0F), static_cast<int> (i % 4U), score});
    }

  auto cpu_nms = dense_filter_and_nms_cpu (cands, 0.35F, 0.45F);
  auto gpu_nms = dense_filter_and_nms_gpu (cands, 0.35F, 0.45F, 0);
  CHECK (cpu_nms.has_value ());
  CHECK (gpu_nms.has_value ());
  CHECK (cpu_nms->size () == gpu_nms->size ());
  for (size_t i = 0U; i < cpu_nms->size (); ++i)
    {
      CHECK ((*cpu_nms)[i].class_id == (*gpu_nms)[i].class_id);
      CHECK (std::abs ((*cpu_nms)[i].score - (*gpu_nms)[i].score) < 1e-6F);
      CHECK (std::abs ((*cpu_nms)[i].box.x - (*gpu_nms)[i].box.x) < 1e-5F);
    }
}

void
test_vram_budget_and_graceful_fallback (void)
{
  auto s_res = SuryaStream::create (0, false);
  CHECK (s_res.has_value ());

  /* 1. Direct VramBudgetAllocator cap enforcement + RAII release.  */
  constexpr size_t cap_16mb = 16ULL * 1024ULL * 1024ULL;
  VramBudgetAllocator alloc (cap_16mb, 0);
  {
    auto b1 = alloc.allocate_device_async (8ULL * 1024ULL * 1024ULL, *s_res);
    CHECK (b1.has_value ());
    CHECK (alloc.current_bytes () == 8ULL * 1024ULL * 1024ULL);

    auto b2 = alloc.allocate_device_async (12ULL * 1024ULL * 1024ULL, *s_res);
    CHECK (!b2.has_value ());
    CHECK (b2.error ().code () == Error_Code::Vram_Exhausted);
  }
  CHECK (alloc.current_bytes () == 0U);

  /* 2. Pipeline fail-fast when fallback_to_cpu_on_oom == false.  */
  SuryaPipelineConfig strict_cfg;
  strict_cfg.vram_budget_bytes = 8ULL * 1024ULL * 1024ULL;
  strict_cfg.model_resident_weights_bytes = 32ULL * 1024ULL * 1024ULL;
  strict_cfg.fallback_to_cpu_on_oom = false;

  auto strict_pipe = SuryaPipeline::create (strict_cfg);
  CHECK (!strict_pipe.has_value ());
  CHECK (strict_pipe.error ().code () == Error_Code::Vram_Exhausted);

  /* 3. Pipeline graceful CPU degradation when fallback_to_cpu_on_oom == true
        and per-batch workspace exceeds budget.  */
  auto sink = std::make_shared<WarningCounterSink> ();
  SuryaLogger::set_sink (sink);
  const uint64_t fb_before =
    SuryaStats::global_stats ().snapshot ().cpu_fallback_count;

  SuryaPipelineConfig fb_cfg;
  fb_cfg.vram_budget_bytes = 8ULL * 1024ULL * 1024ULL;
  fb_cfg.model_resident_weights_bytes = 32ULL * 1024ULL * 1024ULL;
  fb_cfg.fallback_to_cpu_on_oom = true;

  auto fb_pipe = SuryaPipeline::create (fb_cfg);
  CHECK (fb_pipe.has_value ());

  Image_U8C3 sample (64U, 64U, Scalar<uint8_t, 3U> (80U, 120U, 160U));
  auto fb_out = (*fb_pipe)->infer_single (sample.const_view ());
  CHECK (fb_out.has_value ());
  CHECK (!fb_out->executed_on_gpu);
  CHECK (!fb_out->token_ids.empty ());
  CHECK (sink->warn_count.load () >= 1ULL);
  CHECK (SuryaStats::global_stats ().snapshot ().cpu_fallback_count
         > fb_before);

  SuryaLogger::set_sink (nullptr);
}

void
test_cuda_graphs_and_pipeline_equivalence (void)
{
  SuryaPipelineConfig cfg_graph;
  cfg_graph.image_target_width = 64U;
  cfg_graph.image_target_height = 64U;
  cfg_graph.vlm_hidden_dim = 64U;
  cfg_graph.vlm_num_layers = 4U;
  cfg_graph.vlm_num_tokens = 16U;
  cfg_graph.model_resident_weights_bytes = 4ULL * 1024ULL * 1024ULL;
  cfg_graph.per_batch_item_workspace_bytes = 2ULL * 1024ULL * 1024ULL;
  cfg_graph.enable_cuda_graphs = true;

  SuryaPipelineConfig cfg_nograph = cfg_graph;
  cfg_nograph.enable_cuda_graphs = false;

  auto p_graph = SuryaPipeline::create (cfg_graph);
  auto p_nograph = SuryaPipeline::create (cfg_nograph);
  CHECK (p_graph.has_value ());
  CHECK (p_nograph.has_value ());

  Image_U8C3 img (64U, 64U, Scalar<uint8_t, 3U> (70U, 140U, 210U));
  auto out_g1 = (*p_graph)->infer_single (img.const_view ());
  auto out_g2 = (*p_graph)->infer_single (img.const_view ());
  auto out_ng = (*p_nograph)->infer_single (img.const_view ());
  CHECK (out_g1.has_value ());
  CHECK (out_g2.has_value ());
  CHECK (out_ng.has_value ());
  CHECK (out_g2->executed_on_gpu);
  CHECK (out_g2->used_cuda_graph);
  CHECK (!out_ng->used_cuda_graph);
  CHECK (out_g2->token_ids == out_ng->token_ids);

  for (size_t i = 0U; i < out_g2->token_confidences.size (); ++i)
    {
      CHECK (std::abs (out_g2->token_confidences[i]
                       - out_ng->token_confidences[i])
             < 1e-5F);
    }
}

void
test_stress_10k_sequential_and_1k_concurrent_8_threads (void)
{
  SuryaPipelineConfig cfg;
  cfg.max_batch_size = 16U;
  cfg.image_target_width = 64U;
  cfg.image_target_height = 64U;
  cfg.vlm_hidden_dim = 64U;
  cfg.vlm_num_layers = 4U;
  cfg.vlm_num_tokens = 16U;
  cfg.model_resident_weights_bytes = 8ULL * 1024ULL * 1024ULL;
  cfg.per_batch_item_workspace_bytes = 1ULL * 1024ULL * 1024ULL;
  cfg.enable_cuda_graphs = true;

  auto pipe_res = SuryaPipeline::create (cfg);
  CHECK (pipe_res.has_value ());
  SuryaPipeline &pipe = **pipe_res;

  Image_U8C3 img (64U, 64U, Scalar<uint8_t, 3U> (90U, 110U, 130U));
  std::vector<Const_View_U8C3> batch16 (16U, img.const_view ());

  /* Warm up CUDA Graph capture for batch=16 and batch=1 before measuring
     steady-state VRAM delta.  */
  auto w16 = pipe.infer_batch (batch16);
  auto w1 = pipe.infer_single (img.const_view ());
  CHECK (w16.has_value ());
  CHECK (w1.has_value ());

  const size_t steady_vram_bytes = pipe.vram_current_bytes ();

  /* 1. 10,000 sequential image inferences (625 batches x 16 images) with
        zero VRAM growth.  */
  for (size_t iter = 0U; iter < 625U; ++iter)
    {
      auto r = pipe.infer_batch (batch16);
      if (!r.has_value () || r->size () != 16U)
        {
          CHECK (false);
          break;
        }
    }
  CHECK (pipe.vram_current_bytes () == steady_vram_bytes);

  /* 2. 1,000 concurrent inference calls across 8 threads (8 x 125 = 1,000). */
  std::atomic<size_t> total_completed {0U};
  std::atomic<size_t> thread_errors {0U};
  std::vector<std::thread> workers;
  workers.reserve (8U);

  for (size_t t = 0U; t < 8U; ++t)
    {
      workers.emplace_back (
        [&] (void)
        {
          for (size_t i = 0U; i < 125U; ++i)
            {
              auto r = pipe.infer_single (img.const_view ());
              if (!r.has_value () || r->token_ids.size () != 16U)
                {
                  thread_errors.fetch_add (1U, std::memory_order_relaxed);
                }
              else
                {
                  total_completed.fetch_add (1U, std::memory_order_relaxed);
                }
            }
        });
    }

  for (auto &th : workers)
    {
      th.join ();
    }

  CHECK (thread_errors.load () == 0U);
  CHECK (total_completed.load () == 1000U);
  CHECK (pipe.vram_current_bytes () == steady_vram_bytes);

  /* 3. Overlapped stream pipeline verification.  */
  std::vector<Const_View_U8C3> stream_data (128U, img.const_view ());
  double gpu_util = 0.0;
  auto ov_res =
    pipe.run_overlapped_stream_pipeline (stream_data, 16U, &gpu_util);
  CHECK (ov_res.has_value ());
  CHECK (ov_res->size () == 128U);
  CHECK (gpu_util > 0.0);
}

} /* namespace */

int
main (void)
{
  test_gpu_cpu_numerical_equivalence ();
  test_vram_budget_and_graceful_fallback ();
  test_cuda_graphs_and_pipeline_equivalence ();
  test_stress_10k_sequential_and_1k_concurrent_8_threads ();

  if (g_failures > 0)
    {
      std::cerr << "test_gpu_hybrid FAILED with " << g_failures
                << " error(s).\n";
      return 1;
    }
  std::cout << "test_gpu_hybrid PASSED (CUDA available: "
            << (is_cuda_available (0) ? "YES" : "NO") << ").\n";
  return 0;
}
