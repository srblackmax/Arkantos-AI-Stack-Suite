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

#ifndef SUTRA_PLANNER_HPP
#define SUTRA_PLANNER_HPP

#include <sutra/error.hpp>
#include <sutra/memory.hpp>
#include <sutra/scheduler.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sutra::inline v1
{

inline constexpr std::size_t kDefaultModelContextLen = 4096U;

/**
 * @brief Zero-copy streamed token piece emitted during live LLM generation.
 * @note text is valid only for the duration of the TokenStreamFn callback.
 */
struct TokenChunk
{
  std::string_view text;
  std::optional<uint32_t> token_id;
  float logprob = -std::numeric_limits<float>::infinity ();
  bool is_first_token = false;
};

/**
 * @brief Lightweight zero-copy tool schema view passed in CompletionRequest.
 */
struct ToolSchema
{
  std::string_view name;
  std::string_view description;
  std::string_view schema_json;
};

/**
 * @brief Identifies the model-family tool-call formatting dialect.
 */
enum class ToolCallStyle : uint8_t
{
  Llama3 = 0,
  Qwen = 1,
  Mistral = 2,
  Hermes = 3,
  DeepSeek = 4,
  Phi = 5,
  Fallback = 6,
  Gemma4 = 7
};

/**
 * @brief Describes context window and tool-calling capabilities of a model.
 */
struct ModelCapabilities
{
  std::size_t context_len = kDefaultModelContextLen;
  ToolCallStyle tool_style = ToolCallStyle::Fallback;
  std::size_t vocab_size = 0U;
  std::string model_family;
};

/**
 * @brief Extracted tool call emitted by a model-family ToolCallParser.
 */
struct ToolCall
{
  std::string id;
  std::string name;
  std::string arguments_json = "{}";
};

/**
 * @brief Final outcome of an asynchronous streaming completion request.
 */
struct CompletionResult
{
  std::string text;
  std::vector<ToolCall> tool_calls;
  std::size_t prompt_tokens = 0U;
  std::size_t generated_tokens = 0U;
  uint64_t ttft_ns = 0U;
  uint64_t total_latency_ns = 0U;
};

/**
 * @brief Parameters for a non-blocking streaming LLM completion request.
 */
struct CompletionRequest
{
  std::span<const Message> messages;
  std::span<const ToolSchema> tools;
  std::string_view model_id;
  float temperature = 0.0F;
  std::optional<uint32_t> max_tokens;
  std::span<const std::string_view> stop_sequences;
  std::optional<uint64_t> seed;
};

using TokenStreamFn = std::function<void (const TokenChunk &)>;
using CompletionDone = std::function<void (Result<CompletionResult>)>;

/**
 * @brief Callback invoked by streaming LLM backends as soon as a tool call
 *        is decoded, enabling tool execution to overlap with token generation.
 */
using StreamingToolDispatchFn = std::function<void (ToolCallRequest)>;

/**
 * @brief Output produced by a single LLM inference step.
 * @note Thread-safety: Value type.
 */
struct LlmStepOutput
{
  std::string reasoning_thought;
  std::string final_answer;
  std::vector<ToolCallRequest> tool_calls;
  uint64_t generated_tokens = 0U;
  bool is_final = false;
};

/**
 * @brief Pluggable interface for native C++ LLM inference backends.
 * @pre Implementations must be safe for concurrent generate() calls across
 *      distinct agent runs.
 * @post Never throws across ABI boundaries; returns Result<LlmStepOutput>.
 * @note Thread-safety: Implementations must be stateless or internally synced.
 */
class LlmClient
{
public:
  LlmClient () noexcept = default;
  LlmClient (const LlmClient &) noexcept = default;
  auto operator= (const LlmClient &) noexcept -> LlmClient & = default;
  LlmClient (LlmClient &&) noexcept = default;
  auto operator= (LlmClient &&) noexcept -> LlmClient & = default;
  virtual ~LlmClient () noexcept = default;

  /**
   * @brief Generates the next reasoning step, optionally streaming tool calls.
   * @param history Zero-copy view of short-term conversation messages.
   * @param step_index Zero-based planning step index within the run.
   * @param on_early_tool Callback invoked mid-inference to overlap tools.
   * @param scratch_arena Per-turn PMR monotonic memory resource.
   * @pre None.
   * @post Returns LlmStepOutput or LlmError on failure.
   * @returns Result<LlmStepOutput> or AgentError (LlmErrorCode).
   */
  [[nodiscard]] virtual auto
  generate (const ConversationSpanView &history, std::size_t step_index,
            const StreamingToolDispatchFn &on_early_tool,
            std::pmr::memory_resource *scratch_arena) noexcept
      -> Result<LlmStepOutput> = 0;

  /**
   * @brief Dispatches a non-blocking streaming completion to the executor.
   * @param req CompletionRequest holding messages, tools, stop sequences,
   * seed.
   * @param on_token Callback invoked synchronously on each decoded token
   * piece.
   * @param on_done Callback invoked exactly once when generation finishes or
   * aborts.
   * @pre None.
   * @post Returns immediately on the caller thread; no callback fires after
   * on_done.
   * @returns Result<void> indicating whether dispatch succeeded.
   */
  [[nodiscard]] virtual auto
  stream_completion (const CompletionRequest &req, TokenStreamFn on_token,
                     CompletionDone on_done) noexcept -> Result<void>;

  /**
   * @brief Cooperatively cancels an active streaming generation within 50 ms.
   * @pre None.
   * @post Active stream_completion invokes on_done with AgentError::Cancelled.
   * @returns Result<void>.
   */
  [[nodiscard]] virtual auto cancel () noexcept -> Result<void>;

  /**
   * @brief Queries the model's context length and tool-call style.
   * @returns ModelCapabilities descriptor.
   */
  [[nodiscard]] virtual auto
  capabilities () const noexcept -> ModelCapabilities;
};

/**
 * @brief Decision produced by a Planner for a single orchestration step.
 * @note Thread-safety: Value type.
 */
struct PlanDecision
{
  std::string thought;
  std::string final_answer;
  std::vector<ToolCallRequest> tool_calls;
  uint64_t generated_tokens = 0U;
  bool completed = false;
};

/**
 * @brief Swappable planning strategy interface (ReAct vs Plan-and-Execute).
 * @note Thread-safety: Implementations must be safe for concurrent calls.
 */
class Planner
{
public:
  Planner () noexcept = default;
  Planner (const Planner &) noexcept = default;
  auto operator= (const Planner &) noexcept -> Planner & = default;
  Planner (Planner &&) noexcept = default;
  auto operator= (Planner &&) noexcept -> Planner & = default;
  virtual ~Planner () noexcept = default;

  /**
   * @brief Returns the human-readable strategy name.
   * @returns Strategy identifier ("ReAct" or "PlanAndExecute").
   */
  [[nodiscard]] virtual auto name () const noexcept -> std::string_view = 0;

  /**
   * @brief Computes the next PlanDecision given conversation history.
   * @param llm Active LlmClient backend.
   * @param history Zero-copy conversation history view.
   * @param step_index Current zero-based step number.
   * @param on_early_tool Streaming tool dispatch callback for overlap.
   * @param scratch_arena Per-turn PMR monotonic arena.
   * @pre step_index < max_steps.
   * @post Returns PlanDecision with either tool_calls or completed == true.
   * @returns Result<PlanDecision> or AgentError.
   */
  [[nodiscard]] virtual auto
  plan_step (LlmClient &llm, const ConversationSpanView &history,
             std::size_t step_index,
             const StreamingToolDispatchFn &on_early_tool,
             std::pmr::memory_resource *scratch_arena) noexcept
      -> Result<PlanDecision> = 0;
};

/**
 * @brief Baseline Reason + Act (ReAct) planner with tool-error recovery.
 * @note Thread-safety: Stateless; safe for concurrent agent runs.
 */
class ReActPlanner final : public Planner
{
public:
  ReActPlanner () noexcept = default;

  [[nodiscard]] auto name () const noexcept -> std::string_view override;

  [[nodiscard]] auto
  plan_step (LlmClient &llm, const ConversationSpanView &history,
             std::size_t step_index,
             const StreamingToolDispatchFn &on_early_tool,
             std::pmr::memory_resource *scratch_arena) noexcept
      -> Result<PlanDecision> override;
};

/**
 * @brief Plan-and-Execute planner that validates multi-tool DAGs before
 * dispatch.
 * @note Thread-safety: Stateless; safe for concurrent agent runs.
 */
class PlanAndExecutePlanner final : public Planner
{
public:
  PlanAndExecutePlanner () noexcept = default;

  [[nodiscard]] auto name () const noexcept -> std::string_view override;

  [[nodiscard]] auto
  plan_step (LlmClient &llm, const ConversationSpanView &history,
             std::size_t step_index,
             const StreamingToolDispatchFn &on_early_tool,
             std::pmr::memory_resource *scratch_arena) noexcept
      -> Result<PlanDecision> override;
};

} /* namespace sutra::inline v1 */

#endif /* SUTRA_PLANNER_HPP */
