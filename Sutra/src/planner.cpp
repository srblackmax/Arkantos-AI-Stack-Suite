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

#include <sutra/observability.hpp>
#include <sutra/planner.hpp>

#include <utility>

namespace sutra::inline v1
{

/* Default stream_completion adapter for synchronous mock LlmClient subclasses.
 */
auto
LlmClient::stream_completion (const CompletionRequest &req,
                              TokenStreamFn on_token,
                              CompletionDone on_done) noexcept -> Result<void>
{
  ConversationSpanView view;
  view.first_segment = req.messages;
  view.total_turns = req.messages.size ();

  const uint64_t t0 = monotonic_time_ns ();
  auto res = generate (view, 0U, StreamingToolDispatchFn{},
                       std::pmr::new_delete_resource ());
  const uint64_t t1 = monotonic_time_ns ();

  if (!res)
    {
      if (on_done)
        {
          on_done (std::unexpected (res.error ()));
        }
      return {};
    }

  const std::string text
      = res->is_final ? res->final_answer : res->reasoning_thought;
  if (on_token && !text.empty ())
    {
      TokenChunk chunk;
      chunk.text = text;
      chunk.token_id = 0U;
      chunk.is_first_token = true;
      on_token (chunk);
    }

  if (on_done)
    {
      CompletionResult comp;
      comp.text = text;
      comp.generated_tokens = static_cast<std::size_t> (res->generated_tokens);
      comp.ttft_ns = (t1 >= t0) ? (t1 - t0) : 0U;
      comp.total_latency_ns = comp.ttft_ns;
      for (const ToolCallRequest &tc : res->tool_calls)
        {
          comp.tool_calls.push_back (
              ToolCall{ tc.call_id, tc.tool_name, tc.arguments_json });
        }
      on_done (std::move (comp));
    }
  return {};
}

/* Default cooperative cancel no-op for synchronous mock LlmClient subclasses.
 */
auto
LlmClient::cancel () noexcept -> Result<void>
{
  return {};
}

/* Default capabilities descriptor for mock LlmClient subclasses.  */
auto
LlmClient::capabilities () const noexcept -> ModelCapabilities
{
  return ModelCapabilities{};
}

/* Returns the strategy identifier for ReActPlanner.  */
auto
ReActPlanner::name () const noexcept -> std::string_view
{
  return "ReAct";
}

/* Executes one ReAct planning step and validates that either a final answer
   or at least one tool call was emitted.  */
auto
ReActPlanner::plan_step (
    LlmClient &llm, const ConversationSpanView &history,
    std::size_t step_index, const StreamingToolDispatchFn &on_early_tool,
    std::pmr::memory_resource *scratch_arena) noexcept -> Result<PlanDecision>
{
  const ScopedStageTimer plan_timer (AgentStats::global (),
                                     StageKind::Planning, "react_plan_step");

  if (history.size () == 0U)
    {
      return std::unexpected (AgentError::planning (
          PlanningErrorCode::NoValidPlan, step_index,
          "Cannot plan a ReAct step with an empty conversation history"));
    }

  Result<LlmStepOutput> llm_res;
  {
    const ScopedStageTimer llm_timer (
        AgentStats::global (), StageKind::LlmInference, "llm_inference");
    llm_res = llm.generate (history, step_index, on_early_tool, scratch_arena);
  }

  if (!llm_res)
    {
      return std::unexpected (llm_res.error ());
    }

  if (!llm_res->is_final && llm_res->tool_calls.empty ())
    {
      return std::unexpected (AgentError::planning (
          PlanningErrorCode::NoValidPlan, step_index,
          "ReAct step produced neither a final answer nor any tool calls"));
    }

  PlanDecision decision;
  decision.thought = std::move (llm_res->reasoning_thought);
  decision.final_answer = std::move (llm_res->final_answer);
  decision.tool_calls = std::move (llm_res->tool_calls);
  decision.generated_tokens = llm_res->generated_tokens;
  decision.completed = llm_res->is_final;
  return decision;
}

/* Returns the strategy identifier for PlanAndExecutePlanner.  */
auto
PlanAndExecutePlanner::name () const noexcept -> std::string_view
{
  return "PlanAndExecute";
}

/* Executes a Plan-and-Execute planning step and statically verifies the
   tool dependency graph before returning the PlanDecision.  */
auto
PlanAndExecutePlanner::plan_step (
    LlmClient &llm, const ConversationSpanView &history,
    std::size_t step_index, const StreamingToolDispatchFn &on_early_tool,
    std::pmr::memory_resource *scratch_arena) noexcept -> Result<PlanDecision>
{
  const ScopedStageTimer plan_timer (
      AgentStats::global (), StageKind::Planning, "plan_and_execute_step");

  if (history.size () == 0U)
    {
      return std::unexpected (AgentError::planning (
          PlanningErrorCode::NoValidPlan, step_index,
          "Cannot run PlanAndExecutePlanner with an empty conversation "
          "history"));
    }

  Result<LlmStepOutput> llm_res;
  {
    const ScopedStageTimer llm_timer (
        AgentStats::global (), StageKind::LlmInference, "llm_inference");
    llm_res = llm.generate (history, step_index, on_early_tool, scratch_arena);
  }

  if (!llm_res)
    {
      return std::unexpected (llm_res.error ());
    }

  if (!llm_res->is_final && llm_res->tool_calls.empty ())
    {
      return std::unexpected (AgentError::planning (
          PlanningErrorCode::NoValidPlan, step_index,
          "PlanAndExecute step produced an empty execution plan"));
    }

  if (!llm_res->tool_calls.empty ())
    {
      const auto dep_res
          = ToolScheduler::analyze_dependencies (llm_res->tool_calls);
      if (!dep_res)
        {
          return std::unexpected (dep_res.error ());
        }
    }

  PlanDecision decision;
  decision.thought = std::move (llm_res->reasoning_thought);
  decision.final_answer = std::move (llm_res->final_answer);
  decision.tool_calls = std::move (llm_res->tool_calls);
  decision.generated_tokens = llm_res->generated_tokens;
  decision.completed = llm_res->is_final;
  return decision;
}

} /* namespace sutra::inline v1 */
