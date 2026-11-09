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
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
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
      std::cerr << "test_unit FAILED at line " << line << ": " << expr << "\n";
      std::abort ();
    }
}

#define DARSHANA_REQUIRE(cond) require ((cond), #cond, __LINE__)

/* Custom in-memory LogSink for verifying structured observability.  */
class CapturingLogSink final : public darshana::LogSink
{
public:
  void
  write (darshana::LogLevel level, std::string_view subsystem,
         std::string_view message, uint64_t timestamp_ns) noexcept override
  {
    last_level = level;
    last_subsystem = std::string (subsystem);
    last_message = std::string (message);
    last_ts = timestamp_ns;
    ++count;
  }

  darshana::LogLevel last_level = darshana::LogLevel::Trace;
  std::string last_subsystem;
  std::string last_message;
  uint64_t last_ts = 0U;
  std::size_t count = 0U;
};

/* Custom ProfilerHook for verifying zone, counter, and frame callbacks.  */
class CapturingProfilerHook final : public darshana::ProfilerHook
{
public:
  void
  on_zone_begin (std::string_view zone_name) noexcept override
  {
    last_zone = std::string (zone_name);
    ++zones_begun;
  }

  void
  on_zone_end (std::string_view zone_name,
               uint64_t elapsed_ns) noexcept override
  {
    last_zone = std::string (zone_name);
    total_ns += elapsed_ns;
    ++zones_ended;
  }

  void
  on_counter (std::string_view counter_name, int64_t value) noexcept override
  {
    last_counter = std::string (counter_name);
    last_value = value;
  }

  void
  on_frame_mark (uint64_t frame_index) noexcept override
  {
    last_frame = frame_index;
  }

  std::string last_zone;
  std::string last_counter;
  int64_t last_value = 0;
  uint64_t total_ns = 0U;
  uint64_t last_frame = 0U;
  std::size_t zones_begun = 0U;
  std::size_t zones_ended = 0U;
};

/* Tests Error taxonomy, domain factories, and VkResult string lookup.  */
void
test_error_taxonomy ()
{
  using namespace darshana;

  const auto e_gpu
      = Error::gpu (GpuErrorCode::DeviceLost, -4, "Simulated GPU device loss");
  DARSHANA_REQUIRE (e_gpu.is_gpu_error ());
  DARSHANA_REQUIRE (!e_gpu.is_data_error ());
  DARSHANA_REQUIRE (e_gpu.category_name () == "GPU");
  DARSHANA_REQUIRE (vk_result_to_string (-4) == "VK_ERROR_DEVICE_LOST");
  DARSHANA_REQUIRE (vk_result_to_string (0) == "VK_SUCCESS");

  const auto e_data
      = Error::data (DataErrorCode::EmptyDataset, "x", "Dataset is empty");
  DARSHANA_REQUIRE (e_data.is_data_error ());
  DARSHANA_REQUIRE (e_data.category_name () == "Data");

  const auto e_exp = Error::export_err (ExportErrorCode::IoFailure, "/bad",
                                        "Failed to open export file");
  DARSHANA_REQUIRE (e_exp.is_export_error ());
  DARSHANA_REQUIRE (e_exp.category_name () == "Export");

  const auto e_fw = Error::framework (FrameworkErrorCode::ResourceExhaustion,
                                      12, "Resource exhausted");
  DARSHANA_REQUIRE (e_fw.is_framework_error ());
  DARSHANA_REQUIRE (e_fw.category_name () == "Framework");
}

/* Tests Logger, FrameStats, Profiler, and ScopedFrameTimer.  */
void
test_observability ()
{
  using namespace darshana;

  auto sink = std::make_shared<CapturingLogSink> ();
  Logger::set_sink (sink);
  Logger::set_level (LogLevel::Info);

  Logger::log (LogLevel::Debug, "test", "Filtered debug message");
  DARSHANA_REQUIRE (sink->count == 0U);

  Logger::log (LogLevel::Warn, "gpu", "Fallback warning message");
  DARSHANA_REQUIRE (sink->count == 1U);
  DARSHANA_REQUIRE (sink->last_subsystem == "gpu");
  Logger::set_sink (nullptr);

  CapturingProfilerHook hook;
  Profiler::set_hook (&hook);
  FrameStats stats;
  {
    ScopedFrameTimer timer (stats, FrameStage::CpuPrep, "unit_prep");
  }
  Profiler::emit_counter ("points", 1000);
  stats.record_frame (5000U);
  stats.record_base_redraw ();
  stats.record_overlay_redraw ();

  const auto snap = stats.snapshot ();
  DARSHANA_REQUIRE (snap.frame_count == 1U);
  DARSHANA_REQUIRE (snap.base_layer_redraws == 1U);
  DARSHANA_REQUIRE (snap.overlay_layer_redraws == 1U);
  DARSHANA_REQUIRE (hook.zones_begun == 1U);
  DARSHANA_REQUIRE (hook.zones_ended == 1U);
  DARSHANA_REQUIRE (hook.last_value == 1000);
  DARSHANA_REQUIRE (hook.last_frame == 1U);

  stats.reset ();
  DARSHANA_REQUIRE (stats.snapshot ().frame_count == 0U);
  Profiler::set_hook (nullptr);
}

/* Tests ScratchArena, StreamingBuffer, and SnapshotData.  */
void
test_memory_and_data_paths ()
{
  using namespace darshana;

  ScratchArena arena (1024U);
  DARSHANA_REQUIRE (arena.capacity_bytes () == 1024U);
  auto p1 = arena.allocate (256U, 16U);
  DARSHANA_REQUIRE (p1.has_value ());
  DARSHANA_REQUIRE (arena.used_bytes () >= 256U);
  auto p_fail = arena.allocate (2048U, 16U);
  DARSHANA_REQUIRE (!p_fail.has_value ());
  arena.reset ();
  DARSHANA_REQUIRE (arena.used_bytes () == 0U);
  DARSHANA_REQUIRE (arena.peak_bytes () >= 256U);

  DARSHANA_REQUIRE (!StreamingBuffer::create (0U).has_value ());
  auto ring_res = StreamingBuffer::create (4U);
  DARSHANA_REQUIRE (ring_res.has_value ());
  auto &ring = *ring_res;
  for (int i = 1; i <= 6; ++i)
    {
      DARSHANA_REQUIRE (
          ring.push (static_cast<float> (i), static_cast<float> (i * 10))
              .has_value ());
    }
  DARSHANA_REQUIRE (ring.size () == 4U);
  std::vector<DataPoint2D> ordered (4U);
  DARSHANA_REQUIRE (ring.copy_ordered (ordered) == 4U);
  DARSHANA_REQUIRE (ordered[0].x == 3.0F && ordered[3].x == 6.0F);
  DARSHANA_REQUIRE (!ring.push (std::numeric_limits<float>::quiet_NaN (), 1.0F)
                         .has_value ());

  const std::vector<float> empty_v;
  const std::vector<float> x_v = { 1.0F, 2.0F, 3.0F };
  const std::vector<float> y_v = { 4.0F, 5.0F, 6.0F };
  const std::vector<float> short_v = { 1.0F };
  DARSHANA_REQUIRE (!SnapshotData::from_xy (empty_v, empty_v).has_value ());
  DARSHANA_REQUIRE (!SnapshotData::from_xy (x_v, short_v).has_value ());

  auto snap_res = SnapshotData::from_xy (x_v, y_v);
  DARSHANA_REQUIRE (snap_res.has_value ());
  DARSHANA_REQUIRE (snap_res->point_count () == 3U);
  DARSHANA_REQUIRE (snap_res->set_column ("weight", x_v).has_value ());
  DARSHANA_REQUIRE (snap_res->column ("weight").has_value ());
  DARSHANA_REQUIRE (!snap_res->column ("missing").has_value ());
}

/* Tests ScaleSpec mapping, Spec validation, and statistical compositions.  */
void
test_spec_and_statistics ()
{
  using namespace darshana;

  ScaleSpec lin;
  lin.type = ScaleType::Linear;
  lin.domain_min = 0.0;
  lin.domain_max = 10.0;
  lin.range_min = 0.0;
  lin.range_max = 100.0;
  auto m_lin = lin.map (5.0);
  DARSHANA_REQUIRE (m_lin.has_value () && std::abs (*m_lin - 50.0) < 1e-6);

  ScaleSpec log_s;
  log_s.type = ScaleType::Log;
  log_s.domain_min = 1.0;
  log_s.domain_max = 100.0;
  log_s.range_min = 0.0;
  log_s.range_max = 1.0;
  auto m_log = log_s.map (10.0);
  DARSHANA_REQUIRE (m_log.has_value () && std::abs (*m_log - 0.5) < 1e-6);
  DARSHANA_REQUIRE (!log_s.map (-1.0).has_value ());

  ScaleSpec cat_s;
  cat_s.type = ScaleType::Categorical;
  cat_s.categories = { "A", "B" };
  DARSHANA_REQUIRE (cat_s.map_category ("A").has_value ());
  DARSHANA_REQUIRE (!cat_s.map_category ("Z").has_value ());

  Spec invalid_spec;
  DARSHANA_REQUIRE (!invalid_spec.validate ().has_value ());
  DARSHANA_REQUIRE (!Spec::from_json ("{broken_json").has_value ());

  const std::vector<float> xs = { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
  const std::vector<float> ys = { 2.1F, 3.9F, 6.2F, 8.0F, 9.9F, 12.1F };

  DARSHANA_REQUIRE (stat::histplot (xs, 4U).has_value ());
  DARSHANA_REQUIRE (!stat::histplot (xs, 0U).has_value ());
  DARSHANA_REQUIRE (stat::kdeplot (xs, 32U).has_value ());
  DARSHANA_REQUIRE (stat::distplot (xs, 4U).has_value ());
  DARSHANA_REQUIRE (stat::ecdfplot (xs).has_value ());
  DARSHANA_REQUIRE (stat::regplot (xs, ys, 0.95F).has_value ());
  DARSHANA_REQUIRE (!stat::regplot (xs, ys, 1.5F).has_value ());

  const std::vector<std::string> cats = { "G1", "G2" };
  const std::vector<std::vector<float>> grps = { xs, ys };
  DARSHANA_REQUIRE (stat::boxplot (cats, grps).has_value ());
  DARSHANA_REQUIRE (stat::violinplot (cats, grps).has_value ());
  DARSHANA_REQUIRE (stat::barplot (cats, grps).has_value ());
  DARSHANA_REQUIRE (stat::corrplot (grps, cats).has_value ());

  const std::vector<float> mat = { 0.1F, 0.5F, -0.4F, 0.9F };
  DARSHANA_REQUIRE (stat::heatmap (mat, 2U, 2U).has_value ());
  DARSHANA_REQUIRE (!stat::heatmap (mat, 3U, 2U).has_value ());
}

/* Tests GPU device loss, forced failure graceful fallback, and export.  */
void
test_gpu_and_export_error_paths ()
{
  using namespace darshana;

  GpuDeviceConfig fail_cfg;
  fail_cfg.force_device_failure = true;
  auto fail_dev = GpuDevice::create (fail_cfg);
  DARSHANA_REQUIRE (!fail_dev.has_value ());
  DARSHANA_REQUIRE (fail_dev.error ().is_gpu_error ());

  const std::vector<float> xs = { 0.0F, 1.0F, 2.0F, 3.0F };
  auto spec_res = stat::ecdfplot (xs, "Fallback Test");
  DARSHANA_REQUIRE (spec_res.has_value ());

  GpuDevice invalid_dev;
  auto fig_fallback = Figure::create_with_device (*spec_res, invalid_dev, 64U);
  DARSHANA_REQUIRE (fig_fallback.has_value ());
  DARSHANA_REQUIRE (fig_fallback->active_backend () == BackendKind::CpuVector);
  DARSHANA_REQUIRE (fig_fallback->render_frame ().has_value ());

  if (is_vulkan_available ())
    {
      auto dev_res = GpuDevice::create ();
      DARSHANA_REQUIRE (dev_res.has_value ());
      dev_res->simulate_device_loss ();
      DARSHANA_REQUIRE (!dev_res->is_valid ());
      DARSHANA_REQUIRE (!dev_res->wait_idle ().has_value ());
      DARSHANA_REQUIRE (!GpuBuffer::allocate (*dev_res, 1024U).has_value ());
    }

  DARSHANA_REQUIRE (
      !export_file_sync (*spec_res, "", ExportFormat::Svg).has_value ());

  const std::string svg_path = "/tmp/darshana_unit_test.svg";
  const std::string pdf_path = "/tmp/darshana_unit_test.pdf";
  bool callback_called = false;
  auto fut = fig_fallback->export_async (svg_path, ExportFormat::Svg,
                                         [&] (const Result<ExportResult> &r) {
                                           callback_called = r.has_value ();
                                         });
  auto exp_res = fut.get ();
  DARSHANA_REQUIRE (exp_res.has_value ());
  DARSHANA_REQUIRE (callback_called);
  DARSHANA_REQUIRE (exp_res->bytes_written > 50U);

  auto pdf_res = export_file_sync (*spec_res, pdf_path, ExportFormat::Pdf);
  DARSHANA_REQUIRE (pdf_res.has_value ());
  DARSHANA_REQUIRE (pdf_res->bytes_written > 50U);

  auto pdf_str = render_pdf_string (*spec_res);
  DARSHANA_REQUIRE (pdf_str.has_value ());
  DARSHANA_REQUIRE (pdf_str->find ("%PDF-1.4") != std::string::npos);
  DARSHANA_REQUIRE (pdf_str->find ("xref") != std::string::npos);
  DARSHANA_REQUIRE (pdf_str->find ("startxref") != std::string::npos);

  const std::vector<float> rx = { 1.0F, 2.0F, 3.0F, 4.0F };
  const std::vector<float> ry = { 2.0F, 4.1F, 5.9F, 8.0F };
  auto reg_spec = stat::regplot (rx, ry, 0.95F, "Regression SVG Test");
  DARSHANA_REQUIRE (reg_spec.has_value ());
  auto reg_svg = render_svg_string (*reg_spec);
  DARSHANA_REQUIRE (reg_svg.has_value ());
  DARSHANA_REQUIRE (reg_svg->find ("<polygon") != std::string::npos);
  DARSHANA_REQUIRE (reg_svg->find ("<path") != std::string::npos);
  DARSHANA_REQUIRE (reg_svg->find ("<rect") != std::string::npos);

  (void)std::filesystem::remove (svg_path);
  (void)std::filesystem::remove (pdf_path);
}

} /* namespace */

auto
main () -> int
{
  test_error_taxonomy ();
  test_observability ();
  test_memory_and_data_paths ();
  test_spec_and_statistics ();
  test_gpu_and_export_error_paths ();
  std::cout << "All Darshana unit tests PASSED.\n";
  return 0;
}
