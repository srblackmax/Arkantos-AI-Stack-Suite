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

#ifndef SPHINX_VIZ_HPP
#define SPHINX_VIZ_HPP

/**
 * @file viz.hpp
 * @brief Darshana declarative Spec session visualization for Sphinx.
 */

#include <darshana/darshana.hpp>
#include <sphinx/router.hpp>
#include <sutra/error.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sphinx::inline v1
{

/**
 * @brief Per-turn telemetry record captured by SphinxSession and logged to
 *        Yantra Run and Darshana visualizations.
 */
struct TurnTelemetry
{
  uint64_t turn_index = 0U;
  std::string user_text;
  std::string assistant_text;
  Route route = Route::CHAT;
  float router_confidence = 1.0F;
  double vad_eos_ms = 0.0;
  double stt_ms = 0.0;
  double router_ms = 0.0;
  double llm_ttft_ms = 0.0;
  double llm_total_ms = 0.0;
  double tool_ms = 0.0;
  double tts_first_sample_ms = 0.0;
  double tts_total_ms = 0.0;
  double ttfs_ms = 0.0;
  double barge_in_ms = 0.0;
  double backchannel_ms = 0.0;
  uint64_t tokens_generated = 0U;
  double tok_per_sec = 0.0;
  std::size_t sentences_spoken = 0U;
  double max_inter_sentence_gap_ms = 0.0;
  std::size_t dag_nodes_executed = 0U;
  std::size_t dag_retries = 0U;
  bool barge_in_triggered = false;
  bool backchannel_emitted = false;
  bool thinking_enabled = false;
  std::vector<std::string> tools_invoked;
};

/**
 * @brief Summary of the 3 Darshana session SVG plots rendered at session end.
 */
struct SessionPlotsReport
{
  std::string main_svg_path;
  std::string latency_svg_path;
  std::string router_svg_path;
  double main_render_ms = 0.0;
  double latency_render_ms = 0.0;
  double router_render_ms = 0.0;
};

/**
 * @brief Builds and renders the 3 Darshana session plots (<session_id>.svg,
 *        <session_id>-latency.svg, <session_id>-router.svg) using
 *        darshana::Spec in <= 500 ms per SVG.
 */
[[nodiscard]] auto
render_session_svgs (std::string_view session_id,
                     const std::filesystem::path &output_dir,
                     std::span<const TurnTelemetry> turns) noexcept
    -> sutra::Result<SessionPlotsReport>;

/**
 * @brief Validates that an exported SVG file on disk is non-empty, well-formed
 *        XML containing <svg> root, <rect> or <circle>/<path> marks, and
 *        balanced tags.
 */
[[nodiscard]] auto
validate_svg_xml (const std::filesystem::path &svg_path) noexcept
    -> sutra::Result<void>;

} /* namespace sphinx::inline v1 */

#endif /* SPHINX_VIZ_HPP */
