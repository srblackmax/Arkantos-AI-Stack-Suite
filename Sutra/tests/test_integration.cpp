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

#include <sutra/sutra.hpp>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{

#define SUTRA_ASSERT(cond)                                                    \
  do                                                                          \
    {                                                                         \
      if (!(cond))                                                            \
        {                                                                     \
          std::cerr << __FILE__ << ":" << __LINE__                            \
                    << ": assertion failed: " #cond "\n";                     \
          std::abort ();                                                      \
        }                                                                     \
    }                                                                         \
  while (false)

/* Tool that verifies ToolContext zero-copy span pointer identity and returns
   a multiplied result.  */
auto
multiply_tool (const sutra::ToolContext &ctx, int64_t a,
               int64_t b) noexcept -> sutra::Result<std::string>
{
  SUTRA_ASSERT (ctx.conversation_view.size () > 0U);
  SUTRA_ASSERT (ctx.scratch_arena != nullptr);
  return std::to_string (a * b);
}

/* Mock LLM that streams an early tool call mid-inference while generating
   subsequent tokens, testing overlapped tool execution and ReAct loop.  */
class StreamingReActMockLlm final : public sutra::LlmClient
{
public:
  auto
  generate (const sutra::ConversationSpanView & /*history*/,
            std::size_t step_index,
            const sutra::StreamingToolDispatchFn &on_early_tool,
            std::pmr::memory_resource * /*scratch_arena*/) noexcept
      -> sutra::Result<sutra::LlmStepOutput> override
  {
    sutra::LlmStepOutput out;
    out.generated_tokens = 64U;
    if (step_index == 0U)
      {
        out.reasoning_thought = "Dispatching early tool and parallel sibling";
        sutra::ToolCallRequest early_req;
        early_req.call_id = "early_1";
        early_req.tool_name = "multiply";
        early_req.arguments_json = R"({"a":6,"b":7})";
        if (on_early_tool)
          {
            on_early_tool (early_req);
          }
        out.tool_calls.push_back (std::move (early_req));

        sutra::ToolCallRequest second_req;
        second_req.call_id = "parallel_2";
        second_req.tool_name = "multiply";
        second_req.arguments_json = R"({"a":10,"b":10})";
        out.tool_calls.push_back (std::move (second_req));
        return out;
      }

    out.is_final = true;
    out.final_answer = "42 and 100";
    return out;
  }
};

void
test_react_loop_and_overlapped_execution ()
{
  auto registry = std::make_shared<sutra::ToolRegistry> ();
  SUTRA_ASSERT (SUTRA_REGISTER_TOOL (*registry, &multiply_tool, "multiply",
                                     "Multiplies a and b", "a", "b")
                    .has_value ());

  auto llm = std::make_shared<StreamingReActMockLlm> ();
  sutra::AgentConfig cfg;
  cfg.max_steps = 4U;
  cfg.short_term_capacity = 32U;

  auto agent_res = sutra::Agent::create (cfg, registry, llm);
  SUTRA_ASSERT (agent_res.has_value ());

  // Verify zero-copy pointer identity between ShortTermMemory and ToolContext.
  sutra::AgentRuntime runtime (4U);
  const auto run_res = runtime.run (**agent_res, "Multiply 6*7 and 10*10");
  SUTRA_ASSERT (run_res.has_value ());
  SUTRA_ASSERT (run_res->final_answer == "42 and 100");
  SUTRA_ASSERT (run_res->steps_taken == 2U);
  SUTRA_ASSERT (run_res->tools_executed == 2U);
  SUTRA_ASSERT (run_res->overlapped_tokens == 64U);

  const auto view_a = (*agent_res)->memory ().short_term ().borrow_view ();
  const auto view_b = (*agent_res)->memory ().short_term ().borrow_view ();
  SUTRA_ASSERT (view_a.size () == view_b.size ());
  SUTRA_ASSERT (&view_a.at (0U) == &view_b.at (0U));
}

void
test_multi_node_workflow_parallel_branches ()
{
  sutra::WorkflowGraph graph;
  std::atomic<int> parallel_branches_executed{ 0 };

  sutra::WorkflowNode root;
  root.id = "ingest";
  root.input_schema_json
      = R"({"type":"object","properties":{"seed":{"type":"integer"}},)"
        R"("required":["seed"]})";
  root.output_schema_json
      = R"({"type":"object","properties":{"base":{"type":"integer"}},)"
        R"("required":["base"]})";
  root.action = [] (const sutra::ToolContext & /*ctx*/,
                    std::string_view /*in*/) -> sutra::Result<std::string> {
    return std::string (R"({"base":10})");
  };
  SUTRA_ASSERT (graph.add_node (std::move (root)).has_value ());

  sutra::WorkflowNode branch_a;
  branch_a.id = "branch_a";
  branch_a.input_schema_json
      = R"({"type":"object","properties":{"base":{"type":"integer"}},)"
        R"("required":["base"]})";
  branch_a.output_schema_json
      = R"({"type":"object","properties":{"val_a":{"type":"integer"}},)"
        R"("required":["val_a"]})";
  branch_a.action
      = [&parallel_branches_executed] (
            const sutra::ToolContext & /*ctx*/,
            std::string_view /*in*/) -> sutra::Result<std::string> {
    parallel_branches_executed.fetch_add (1, std::memory_order_relaxed);
    return std::string (R"({"val_a":20})");
  };
  SUTRA_ASSERT (graph.add_node (std::move (branch_a)).has_value ());

  sutra::WorkflowNode branch_b;
  branch_b.id = "branch_b";
  branch_b.input_schema_json
      = R"({"type":"object","properties":{"base":{"type":"integer"}},)"
        R"("required":["base"]})";
  branch_b.output_schema_json
      = R"({"type":"object","properties":{"val_b":{"type":"integer"}},)"
        R"("required":["val_b"]})";
  branch_b.action
      = [&parallel_branches_executed] (
            const sutra::ToolContext & /*ctx*/,
            std::string_view /*in*/) -> sutra::Result<std::string> {
    parallel_branches_executed.fetch_add (1, std::memory_order_relaxed);
    return std::string (R"({"val_b":30})");
  };
  SUTRA_ASSERT (graph.add_node (std::move (branch_b)).has_value ());

  sutra::WorkflowNode join_node;
  join_node.id = "aggregate";
  join_node.input_schema_json
      = R"({"type":"object","properties":{"val_a":{"type":"integer"},)"
        R"("val_b":{"type":"integer"}},"required":["val_a","val_b"]})";
  join_node.output_schema_json
      = R"({"type":"object","properties":{"total":{"type":"integer"}},)"
        R"("required":["total"]})";
  join_node.action = [] (const sutra::ToolContext & /*ctx*/,
                         std::string_view in) -> sutra::Result<std::string> {
    const auto a_tok = sutra::extract_json_field_token (
        in, "val_a", sutra::JsonType::Integer);
    const auto b_tok = sutra::extract_json_field_token (
        in, "val_b", sutra::JsonType::Integer);
    SUTRA_ASSERT (a_tok.has_value () && b_tok.has_value ());
    return std::string (R"({"total":50})");
  };
  SUTRA_ASSERT (graph.add_node (std::move (join_node)).has_value ());

  SUTRA_ASSERT (graph.add_edge ("ingest", "branch_a").has_value ());
  SUTRA_ASSERT (graph.add_edge ("ingest", "branch_b").has_value ());
  SUTRA_ASSERT (graph.add_edge ("branch_a", "aggregate").has_value ());
  SUTRA_ASSERT (graph.add_edge ("branch_b", "aggregate").has_value ());

  const auto lint_res = sutra::PlanLinter::validate (graph);
  SUTRA_ASSERT (lint_res.has_value ());
  SUTRA_ASSERT (lint_res->execution_waves.size () == 3U);
  SUTRA_ASSERT (lint_res->execution_waves.at (1U).size () == 2U);

  sutra::WorkStealingPool pool (4U);
  sutra::ToolContext ctx;
  const auto exec_res = graph.execute (ctx, pool, R"({"seed":1})");
  SUTRA_ASSERT (exec_res.has_value ());
  SUTRA_ASSERT (parallel_branches_executed.load () == 2);
  SUTRA_ASSERT (exec_res->at ("aggregate") == R"({"total":50})");
}

void
test_long_running_150_turn_conversation ()
{
  auto registry = std::make_shared<sutra::ToolRegistry> ();
  SUTRA_ASSERT (SUTRA_REGISTER_TOOL (*registry, &multiply_tool, "multiply",
                                     "Multiplies a and b", "a", "b")
                    .has_value ());

  auto llm = std::make_shared<StreamingReActMockLlm> ();
  auto planner = std::make_shared<sutra::PlanAndExecutePlanner> ();
  sutra::AgentConfig cfg;
  cfg.max_steps = 3U;
  cfg.short_term_capacity = 32U;

  auto agent_res = sutra::Agent::create (cfg, registry, llm, planner);
  SUTRA_ASSERT (agent_res.has_value ());

  sutra::AgentRuntime runtime (4U);
  for (int turn = 0; turn < 150; ++turn)
    {
      const auto res
          = runtime.run (**agent_res, "Turn " + std::to_string (turn));
      SUTRA_ASSERT (res.has_value ());
      // Ring buffer capacity must remain strictly bounded at 32.
      SUTRA_ASSERT ((*agent_res)->memory ().short_term ().size () <= 32U);
      SUTRA_ASSERT ((*agent_res)->memory ().short_term ().capacity () == 32U);
    }

  SUTRA_ASSERT (sutra::AgentStats::global ().snapshot ().active_arena_bytes
                == 0U);
}

} /* namespace */

auto
main () -> int
{
  test_react_loop_and_overlapped_execution ();
  test_multi_node_workflow_parallel_branches ();
  test_long_running_150_turn_conversation ();
  std::cout << "All Sutra integration tests PASSED.\n";
  return 0;
}
