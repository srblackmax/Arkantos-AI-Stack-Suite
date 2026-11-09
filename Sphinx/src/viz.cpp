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

#include <sphinx/viz.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace sphinx::inline v1
{

namespace
{

constexpr uint32_t kDefaultPlotWidth = 1100U;
constexpr uint32_t kDefaultPlotHeight = 680U;
constexpr float kDefaultDpi = 96.0F;
constexpr float kChatBudgetMs = 700.0F;
constexpr float kTaskBudgetMs = 1500.0F;
constexpr float kRouterBudgetMs = 100.0F;

/* Builds the main session overview Spec (<session_id>.svg):
   Panel layers combining turn timeline stages, TTFS bars, and tok/s line.  */
[[nodiscard]] auto
build_main_session_spec (std::string_view session_id,
                         std::span<const TurnTelemetry> turns)
    -> darshana::Spec
{
  darshana::Spec spec;
  spec.title = "Sphinx Session Overview - " + std::string (session_id);
  spec.subtitle
      = "Turn Timeline (STT / Router / LLM+Tools / TTS) & Throughput";
  spec.x_label = "Turn Index";
  spec.y_label = "Latency (ms) / Throughput (tok/s)";
  spec.width = kDefaultPlotWidth;
  spec.height = kDefaultPlotHeight;
  spec.dpi = kDefaultDpi;
  spec.show_grid = true;
  spec.show_legend = true;

  std::vector<float> xs;
  std::vector<float> ttfs_vals;
  std::vector<float> stt_vals;
  std::vector<float> llm_vals;
  std::vector<float> tts_vals;
  std::vector<float> tok_s_vals;
  std::vector<float> chat_budget_line;
  std::vector<float> task_budget_line;

  const std::size_t n = std::max<std::size_t> (1U, turns.size ());
  xs.reserve (n);
  ttfs_vals.reserve (n);
  stt_vals.reserve (n);
  llm_vals.reserve (n);
  tts_vals.reserve (n);
  tok_s_vals.reserve (n);
  chat_budget_line.reserve (n);
  task_budget_line.reserve (n);

  if (turns.empty ())
    {
      xs.push_back (0.0F);
      ttfs_vals.push_back (0.0F);
      stt_vals.push_back (0.0F);
      llm_vals.push_back (0.0F);
      tts_vals.push_back (0.0F);
      tok_s_vals.push_back (0.0F);
      chat_budget_line.push_back (kChatBudgetMs);
      task_budget_line.push_back (kTaskBudgetMs);
    }
  else
    {
      for (std::size_t i = 0U; i < turns.size (); ++i)
        {
          const TurnTelemetry &t = turns[i];
          xs.push_back (static_cast<float> (i + 1U));
          ttfs_vals.push_back (static_cast<float> (t.ttfs_ms));
          stt_vals.push_back (static_cast<float> (t.vad_eos_ms + t.stt_ms));
          llm_vals.push_back (static_cast<float> (t.llm_total_ms + t.tool_ms));
          tts_vals.push_back (static_cast<float> (t.tts_first_sample_ms));
          tok_s_vals.push_back (static_cast<float> (t.tok_per_sec));
          chat_budget_line.push_back (kChatBudgetMs);
          task_budget_line.push_back (kTaskBudgetMs);
        }
    }

  darshana::LayerSpec ttfs_bar;
  ttfs_bar.mark = darshana::MarkType::Bar;
  ttfs_bar.encoding.label = "Total TTFS (ms)";
  ttfs_bar.aesthetics.fill_color = darshana::ColorRgba::from_hex (0x1F77B4,
                                                                  185U);
  ttfs_bar.aesthetics.stroke_color = darshana::ColorRgba::from_hex (0x1F77B4,
                                                                    255U);
  ttfs_bar.inline_x = xs;
  ttfs_bar.inline_y = ttfs_vals;
  spec.layers.push_back (std::move (ttfs_bar));

  darshana::LayerSpec llm_line;
  llm_line.mark = darshana::MarkType::Line;
  llm_line.encoding.label = "LLM + Tool Stage (ms)";
  llm_line.aesthetics.stroke_color = darshana::ColorRgba::from_hex (0xFF7F0E,
                                                                    255U);
  llm_line.aesthetics.stroke_width = 2.0F;
  llm_line.inline_x = xs;
  llm_line.inline_y = llm_vals;
  spec.layers.push_back (std::move (llm_line));

  darshana::LayerSpec tts_scatter;
  tts_scatter.mark = darshana::MarkType::Scatter;
  tts_scatter.encoding.label = "TTS First Sample (ms)";
  tts_scatter.aesthetics.fill_color = darshana::ColorRgba::from_hex (0x2CA02C,
                                                                     230U);
  tts_scatter.aesthetics.stroke_color
      = darshana::ColorRgba::from_hex (0x2CA02C, 255U);
  tts_scatter.aesthetics.marker_size = 5.5F;
  tts_scatter.inline_x = xs;
  tts_scatter.inline_y = tts_vals;
  spec.layers.push_back (std::move (tts_scatter));

  darshana::LayerSpec tok_line;
  tok_line.mark = darshana::MarkType::Line;
  tok_line.encoding.label = "Decode Speed (tok/s)";
  tok_line.aesthetics.stroke_color = darshana::ColorRgba::from_hex (0x9467BD,
                                                                    255U);
  tok_line.aesthetics.stroke_width = 2.0F;
  tok_line.inline_x = xs;
  tok_line.inline_y = tok_s_vals;
  spec.layers.push_back (std::move (tok_line));

  darshana::LayerSpec budget_chat;
  budget_chat.mark = darshana::MarkType::Line;
  budget_chat.encoding.label = "CHAT TTFS Target (700 ms)";
  budget_chat.aesthetics.stroke_color
      = darshana::ColorRgba::from_hex (0xD62728, 220U);
  budget_chat.aesthetics.dash_pattern = darshana::DashPattern::Dashed;
  budget_chat.inline_x = xs;
  budget_chat.inline_y = chat_budget_line;
  spec.layers.push_back (std::move (budget_chat));

  darshana::LayerSpec budget_task;
  budget_task.mark = darshana::MarkType::Line;
  budget_task.encoding.label = "TASK TTFS Target (1500 ms)";
  budget_task.aesthetics.stroke_color
      = darshana::ColorRgba::from_hex (0x8C564B, 220U);
  budget_task.aesthetics.dash_pattern = darshana::DashPattern::Dotted;
  budget_task.inline_x = xs;
  budget_task.inline_y = task_budget_line;
  spec.layers.push_back (std::move (budget_task));

  return spec;
}

/* Builds the per-turn latency breakdown Spec (<session_id>-latency.svg).  */
[[nodiscard]] auto
build_latency_spec (std::string_view session_id,
                    std::span<const TurnTelemetry> turns) -> darshana::Spec
{
  darshana::Spec spec;
  spec.title = "Sphinx Per-Turn Latency Breakdown - "
               + std::string (session_id);
  spec.subtitle = "Stage Latencies (VAD+STT, Router, LLM, Tool, TTS) vs "
                  "700 ms / 1500 ms Targets";
  spec.x_label = "Turn Index";
  spec.y_label = "Latency (ms)";
  spec.width = kDefaultPlotWidth;
  spec.height = kDefaultPlotHeight;
  spec.dpi = kDefaultDpi;
  spec.show_grid = true;
  spec.show_legend = true;

  std::vector<float> xs;
  std::vector<float> stt_y;
  std::vector<float> router_y;
  std::vector<float> llm_y;
  std::vector<float> tts_y;
  std::vector<float> ttfs_y;
  std::vector<float> target_chat_y;
  std::vector<float> target_task_y;

  if (turns.empty ())
    {
      xs.push_back (1.0F);
      stt_y.push_back (0.0F);
      router_y.push_back (0.0F);
      llm_y.push_back (0.0F);
      tts_y.push_back (0.0F);
      ttfs_y.push_back (0.0F);
      target_chat_y.push_back (kChatBudgetMs);
      target_task_y.push_back (kTaskBudgetMs);
    }
  else
    {
      for (std::size_t i = 0U; i < turns.size (); ++i)
        {
          const TurnTelemetry &t = turns[i];
          xs.push_back (static_cast<float> (i + 1U));
          stt_y.push_back (static_cast<float> (t.vad_eos_ms + t.stt_ms));
          router_y.push_back (static_cast<float> (t.router_ms));
          llm_y.push_back (static_cast<float> (t.llm_ttft_ms + t.tool_ms));
          tts_y.push_back (static_cast<float> (t.tts_first_sample_ms));
          ttfs_y.push_back (static_cast<float> (t.ttfs_ms));
          target_chat_y.push_back (kChatBudgetMs);
          target_task_y.push_back (kTaskBudgetMs);
        }
    }

  darshana::LayerSpec ttfs_layer;
  ttfs_layer.mark = darshana::MarkType::Bar;
  ttfs_layer.encoding.label = "End-to-End TTFS (ms)";
  ttfs_layer.aesthetics.fill_color
      = darshana::ColorRgba::from_hex (0x2B5C8F, 180U);
  ttfs_layer.aesthetics.stroke_color
      = darshana::ColorRgba::from_hex (0x2B5C8F, 255U);
  ttfs_layer.inline_x = xs;
  ttfs_layer.inline_y = ttfs_y;
  spec.layers.push_back (std::move (ttfs_layer));

  darshana::LayerSpec stt_layer;
  stt_layer.mark = darshana::MarkType::Line;
  stt_layer.encoding.label = "VAD + Whisper STT (ms)";
  stt_layer.aesthetics.stroke_color
      = darshana::ColorRgba::from_hex (0x17BECF, 255U);
  stt_layer.aesthetics.stroke_width = 2.0F;
  stt_layer.inline_x = xs;
  stt_layer.inline_y = stt_y;
  spec.layers.push_back (std::move (stt_layer));

  darshana::LayerSpec router_layer;
  router_layer.mark = darshana::MarkType::Line;
  router_layer.encoding.label = "1.5B Router (ms)";
  router_layer.aesthetics.stroke_color
      = darshana::ColorRgba::from_hex (0xBCBD22, 255U);
  router_layer.aesthetics.stroke_width = 2.0F;
  router_layer.inline_x = xs;
  router_layer.inline_y = router_y;
  spec.layers.push_back (std::move (router_layer));

  darshana::LayerSpec llm_layer;
  llm_layer.mark = darshana::MarkType::Line;
  llm_layer.encoding.label = "LLM TTFT + Tool Dispatch (ms)";
  llm_layer.aesthetics.stroke_color
      = darshana::ColorRgba::from_hex (0xFF7F0E, 255U);
  llm_layer.aesthetics.stroke_width = 2.0F;
  llm_layer.inline_x = xs;
  llm_layer.inline_y = llm_y;
  spec.layers.push_back (std::move (llm_layer));

  darshana::LayerSpec tts_layer;
  tts_layer.mark = darshana::MarkType::Line;
  tts_layer.encoding.label = "Piper TTS First Sample (ms)";
  tts_layer.aesthetics.stroke_color
      = darshana::ColorRgba::from_hex (0x2CA02C, 255U);
  tts_layer.aesthetics.stroke_width = 2.0F;
  tts_layer.inline_x = xs;
  tts_layer.inline_y = tts_y;
  spec.layers.push_back (std::move (tts_layer));

  darshana::LayerSpec chat_thresh;
  chat_thresh.mark = darshana::MarkType::Line;
  chat_thresh.encoding.label = "700 ms CHAT Budget";
  chat_thresh.aesthetics.stroke_color
      = darshana::ColorRgba::from_hex (0xD62728, 220U);
  chat_thresh.aesthetics.dash_pattern = darshana::DashPattern::Dashed;
  chat_thresh.inline_x = xs;
  chat_thresh.inline_y = target_chat_y;
  spec.layers.push_back (std::move (chat_thresh));

  darshana::LayerSpec task_thresh;
  task_thresh.mark = darshana::MarkType::Line;
  task_thresh.encoding.label = "1500 ms TASK Budget";
  task_thresh.aesthetics.stroke_color
      = darshana::ColorRgba::from_hex (0x9467BD, 220U);
  task_thresh.aesthetics.dash_pattern = darshana::DashPattern::Dotted;
  task_thresh.inline_x = xs;
  task_thresh.inline_y = target_task_y;
  spec.layers.push_back (std::move (task_thresh));

  return spec;
}

/* Builds the router decision distribution Spec (<session_id>-router.svg).  */
[[nodiscard]] auto
build_router_spec (std::string_view session_id,
                   std::span<const TurnTelemetry> turns) -> darshana::Spec
{
  darshana::Spec spec;
  spec.title = "Sphinx TurnRouter Classification Distribution - "
               + std::string (session_id);
  spec.subtitle
      = "CHAT vs TASK Decisions, Router Latency (ms), and Confidence";
  spec.x_label = "Turn Index";
  spec.y_label = "Router Latency (ms) / Confidence (%)";
  spec.width = kDefaultPlotWidth;
  spec.height = kDefaultPlotHeight;
  spec.dpi = kDefaultDpi;
  spec.show_grid = true;
  spec.show_legend = true;

  std::vector<float> chat_x;
  std::vector<float> chat_lat;
  std::vector<float> task_x;
  std::vector<float> task_lat;
  std::vector<float> all_x;
  std::vector<float> conf_pct;
  std::vector<float> budget_y;

  if (turns.empty ())
    {
      chat_x.push_back (1.0F);
      chat_lat.push_back (0.0F);
      all_x.push_back (1.0F);
      conf_pct.push_back (100.0F);
      budget_y.push_back (kRouterBudgetMs);
    }
  else
    {
      for (std::size_t i = 0U; i < turns.size (); ++i)
        {
          const TurnTelemetry &t = turns[i];
          const float idx = static_cast<float> (i + 1U);
          all_x.push_back (idx);
          conf_pct.push_back (t.router_confidence * 100.0F);
          budget_y.push_back (kRouterBudgetMs);
          if (t.route == Route::TASK)
            {
              task_x.push_back (idx);
              task_lat.push_back (static_cast<float> (t.router_ms));
            }
          else
            {
              chat_x.push_back (idx);
              chat_lat.push_back (static_cast<float> (t.router_ms));
            }
        }
    }

  if (!chat_x.empty ())
    {
      darshana::LayerSpec chat_pts;
      chat_pts.mark = darshana::MarkType::Scatter;
      chat_pts.encoding.label = "Route::CHAT Latency (ms)";
      chat_pts.aesthetics.fill_color
          = darshana::ColorRgba::from_hex (0x1F77B4, 230U);
      chat_pts.aesthetics.stroke_color
          = darshana::ColorRgba::from_hex (0x1F77B4, 255U);
      chat_pts.aesthetics.marker_size = 6.0F;
      chat_pts.aesthetics.marker_shape = darshana::MarkerShape::Circle;
      chat_pts.inline_x = std::move (chat_x);
      chat_pts.inline_y = std::move (chat_lat);
      spec.layers.push_back (std::move (chat_pts));
    }

  if (!task_x.empty ())
    {
      darshana::LayerSpec task_pts;
      task_pts.mark = darshana::MarkType::Scatter;
      task_pts.encoding.label = "Route::TASK Latency (ms)";
      task_pts.aesthetics.fill_color
          = darshana::ColorRgba::from_hex (0xFF7F0E, 230U);
      task_pts.aesthetics.stroke_color
          = darshana::ColorRgba::from_hex (0xFF7F0E, 255U);
      task_pts.aesthetics.marker_size = 6.5F;
      task_pts.aesthetics.marker_shape = darshana::MarkerShape::Diamond;
      task_pts.inline_x = std::move (task_x);
      task_pts.inline_y = std::move (task_lat);
      spec.layers.push_back (std::move (task_pts));
    }

  darshana::LayerSpec conf_line;
  conf_line.mark = darshana::MarkType::Line;
  conf_line.encoding.label = "Router Confidence (%)";
  conf_line.aesthetics.stroke_color
      = darshana::ColorRgba::from_hex (0x2CA02C, 220U);
  conf_line.aesthetics.stroke_width = 1.8F;
  conf_line.inline_x = all_x;
  conf_line.inline_y = std::move (conf_pct);
  spec.layers.push_back (std::move (conf_line));

  darshana::LayerSpec p99_budget;
  p99_budget.mark = darshana::MarkType::Line;
  p99_budget.encoding.label = "100 ms Router p99 Ceiling";
  p99_budget.aesthetics.stroke_color
      = darshana::ColorRgba::from_hex (0xD62728, 220U);
  p99_budget.aesthetics.dash_pattern = darshana::DashPattern::Dashed;
  p99_budget.inline_x = std::move (all_x);
  p99_budget.inline_y = std::move (budget_y);
  spec.layers.push_back (std::move (p99_budget));

  return spec;
}

} /* namespace */

auto
render_session_svgs (std::string_view session_id,
                     const std::filesystem::path &output_dir,
                     std::span<const TurnTelemetry> turns) noexcept
    -> sutra::Result<SessionPlotsReport>
{
  try
    {
      std::error_code ec;
      std::filesystem::create_directories (output_dir, ec);
      if (ec)
        {
          return std::unexpected (sutra::AgentError::framework (
              sutra::FrameworkErrorCode::InvalidConfiguration, ec.value (),
              "Failed to create session SVG output directory"));
        }

      const std::string sid = session_id.empty () ? "session"
                                                  : std::string (session_id);
      const std::filesystem::path main_path = output_dir / (sid + ".svg");
      const std::filesystem::path lat_path
          = output_dir / (sid + "-latency.svg");
      const std::filesystem::path rot_path
          = output_dir / (sid + "-router.svg");

      SessionPlotsReport report;
      report.main_svg_path = main_path.string ();
      report.latency_svg_path = lat_path.string ();
      report.router_svg_path = rot_path.string ();

      const auto t0 = std::chrono::steady_clock::now ();
      const darshana::Spec main_spec = build_main_session_spec (sid, turns);
      auto r1 = darshana::export_file_sync (main_spec, report.main_svg_path,
                                            darshana::ExportFormat::Svg);
      const auto t1 = std::chrono::steady_clock::now ();
      if (!r1)
        {
          return std::unexpected (sutra::AgentError::framework (
              sutra::FrameworkErrorCode::InvalidConfiguration, 0,
              "Darshana failed to export main session SVG"));
        }
      report.main_render_ms
          = std::chrono::duration<double, std::milli> (t1 - t0).count ();

      const darshana::Spec lat_spec = build_latency_spec (sid, turns);
      auto r2 = darshana::export_file_sync (lat_spec, report.latency_svg_path,
                                            darshana::ExportFormat::Svg);
      const auto t2 = std::chrono::steady_clock::now ();
      if (!r2)
        {
          return std::unexpected (sutra::AgentError::framework (
              sutra::FrameworkErrorCode::InvalidConfiguration, 0,
              "Darshana failed to export latency session SVG"));
        }
      report.latency_render_ms
          = std::chrono::duration<double, std::milli> (t2 - t1).count ();

      const darshana::Spec rot_spec = build_router_spec (sid, turns);
      auto r3 = darshana::export_file_sync (rot_spec, report.router_svg_path,
                                            darshana::ExportFormat::Svg);
      const auto t3 = std::chrono::steady_clock::now ();
      if (!r3)
        {
          return std::unexpected (sutra::AgentError::framework (
              sutra::FrameworkErrorCode::InvalidConfiguration, 0,
              "Darshana failed to export router session SVG"));
        }
      report.router_render_ms
          = std::chrono::duration<double, std::milli> (t3 - t2).count ();

      return report;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in render_session_svgs"));
    }
}

auto
validate_svg_xml (const std::filesystem::path &svg_path) noexcept
    -> sutra::Result<void>
{
  try
    {
      std::ifstream in (svg_path, std::ios::binary);
      if (!in.is_open ())
        {
          return std::unexpected (sutra::AgentError::framework (
              sutra::FrameworkErrorCode::InvalidConfiguration, 0,
              "Cannot open SVG file for validation: " + svg_path.string ()));
        }
      std::string content ((std::istreambuf_iterator<char> (in)),
                           std::istreambuf_iterator<char> ());
      if (content.empty ())
        {
          return std::unexpected (sutra::AgentError::framework (
              sutra::FrameworkErrorCode::InvalidConfiguration, 0,
              "SVG file is empty: " + svg_path.string ()));
        }
      if (content.find ("<svg") == std::string::npos
          || content.find ("</svg>") == std::string::npos)
        {
          return std::unexpected (sutra::AgentError::framework (
              sutra::FrameworkErrorCode::InvalidConfiguration, 0,
              "SVG file missing <svg> root element: " + svg_path.string ()));
        }
      if (content.find ("<rect") == std::string::npos
          && content.find ("<circle") == std::string::npos
          && content.find ("<path") == std::string::npos
          && content.find ("<polyline") == std::string::npos)
        {
          return std::unexpected (sutra::AgentError::framework (
              sutra::FrameworkErrorCode::InvalidConfiguration, 0,
              "SVG file contains no geometric mark elements: "
                  + svg_path.string ()));
        }
      return {};
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in validate_svg_xml"));
    }
}

} /* namespace sphinx::inline v1 */
