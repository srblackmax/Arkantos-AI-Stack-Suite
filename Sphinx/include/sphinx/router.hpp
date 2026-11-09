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

#ifndef SPHINX_ROUTER_HPP
#define SPHINX_ROUTER_HPP

/**
 * @file router.hpp
 * @brief LLM-backed TurnRouter classifying user utterances into CHAT or TASK.
 * @note Compiles with only the C++23 standard library and public headers of
 *       Sutra and Yantra.
 */

#include <sutra/error.hpp>
#include <sutra/llm_backends.hpp>
#include <yantra/track.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sphinx::inline v1
{

/**
 * @brief Routing destination for a single user voice turn.
 */
enum class Route : uint8_t
{
  CHAT = 0,
  TASK = 1,
  BIG_CHAT = 2
};

/**
 * @brief Structured classification decision produced by TurnRouter::classify.
 */
struct RouterDecision
{
  Route route = Route::CHAT;
  float confidence = 1.0F;
  std::string reasoning;
  double latency_ms = 0.0;
  bool used_fallback = false;
};

/**
 * @brief Single labeled evaluation turn in the 20-turn router benchmark.
 */
struct RouterBenchmarkEntry
{
  std::string utterance;
  Route expected = Route::CHAT;
  Route predicted = Route::CHAT;
  float confidence = 1.0F;
  double latency_ms = 0.0;
  bool correct = false;
};

/**
 * @brief Summary and confusion matrix for the 20-turn router benchmark.
 */
struct RouterBenchmarkReport
{
  std::vector<RouterBenchmarkEntry> entries;
  std::size_t true_chat = 0U;
  std::size_t false_task = 0U;
  std::size_t false_chat = 0U;
  std::size_t true_task = 0U;
  std::size_t correct_count = 0U;
  double accuracy_pct = 0.0;
  double p50_ms = 0.0;
  double p99_ms = 0.0;
};

/**
 * @brief Returns default path for Qwen2.5-1.5B-Instruct-Q4_K_M.gguf.
 */
[[nodiscard]] inline auto
default_router_model_path () -> std::string
{
  const char *home = std::getenv ("HOME");
  const std::string base = (home != nullptr && home[0] != '\0') ? home : ".";
  return base + "/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf";
}

/**
 * @brief Configuration for sphinx::TurnRouter.
 */
struct TurnRouterConfig
{
  std::string model_path = default_router_model_path ();
  uint32_t context_len = 512U;
  int32_t n_gpu_layers = -1;
  uint32_t n_threads = 8U;
};

/**
 * @brief LLM-backed turn router using resident Qwen2.5-1.5B-Instruct-Q4_K_M
 *        with keyword heuristic fallback on malformed output.
 */
class TurnRouter
{
  struct PrivateTag
  {
    explicit PrivateTag () = default;
  };
  struct Impl;

public:
  TurnRouter (PrivateTag tag, std::unique_ptr<Impl> impl) noexcept;

  /**
   * @brief Creates a TurnRouter loading the resident 1.5B router model (or
   *        using an injected LlmClient).
   */
  [[nodiscard]] static auto create (
      const TurnRouterConfig &config = TurnRouterConfig{},
      std::shared_ptr<sutra::LlmClient> client_override = nullptr) noexcept
      -> sutra::Result<std::unique_ptr<TurnRouter>>;

  TurnRouter (const TurnRouter &) = delete;
  auto operator= (const TurnRouter &) -> TurnRouter & = delete;
  TurnRouter (TurnRouter &&) noexcept;
  auto operator= (TurnRouter &&) noexcept -> TurnRouter &;
  ~TurnRouter () noexcept;

  /**
   * @brief Classifies a user utterance as Route::CHAT or Route::TASK via the
   *        resident 1.5B LLM (max_tokens=2, temperature=0.0f) and logs
   *        router_latency_ms and route to active_run if non-null.
   */
  [[nodiscard]] auto classify (std::string_view user_text,
                               yantra::Run *active_run = nullptr,
                               uint64_t step = 0U) noexcept
      -> sutra::Result<RouterDecision>;

  /**
   * @brief Evaluates the 38-keyword fallback classifier directly.
   */
  [[nodiscard]] static auto
  classify_keyword_fallback (std::string_view user_text) noexcept -> Route;

  /**
   * @brief Sets the tool name invoked on the immediately preceding turn
   *        (empty if the previous turn was CHAT or NoPlan) for stateful
   *        verification routing.
   */
  void set_last_tool_invoked (std::string_view tool_name) noexcept;

  /**
   * @brief Returns the tool name invoked on the immediately preceding turn.
   */
  [[nodiscard]] auto last_tool_invoked () const noexcept -> std::string;

  /**
   * @brief Runs the 20-turn labeled router accuracy benchmark (10 CHAT + 10
   *        TASK including 4 ambiguous phrasing cases).
   */
  [[nodiscard]] auto
  run_20_turn_benchmark (yantra::Run *active_run = nullptr) noexcept
      -> sutra::Result<RouterBenchmarkReport>;

  /**
   * @brief Runs 100 consecutive classification calls and returns (p50_ms,
   *        p99_ms) latency quantiles.
   */
  [[nodiscard]] auto run_100_call_latency_benchmark () noexcept
      -> sutra::Result<std::pair<double, double>>;

private:
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Converts Route enum to string ("CHAT", "TASK", or "BIG_CHAT").
 */
[[nodiscard]] constexpr auto
route_to_string (Route r) noexcept -> std::string_view
{
  if (r == Route::TASK)
    {
      return "TASK";
    }
  if (r == Route::BIG_CHAT)
    {
      return "BIG_CHAT";
    }
  return "CHAT";
}

} /* namespace sphinx::inline v1 */

#endif /* SPHINX_ROUTER_HPP */
