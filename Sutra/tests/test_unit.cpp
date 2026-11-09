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

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
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

/* Reflected tool function computing the sum of two 64-bit integers.  */
auto
add_numbers_tool (const sutra::ToolContext &ctx, int64_t lhs,
                  int64_t rhs) noexcept -> sutra::Result<std::string>
{
  if (ctx.scratch_arena == nullptr)
    {
      return std::unexpected (sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "add_numbers",
          "ToolContext scratch_arena must not be null"));
    }
  return std::to_string (lhs + rhs);
}

/* Reflected tool function that sleeps to trigger timeout testing.  */
auto
slow_tool (int64_t sleep_ms) noexcept -> sutra::Result<std::string>
{
  std::this_thread::sleep_for (std::chrono::milliseconds (sleep_ms));
  return std::string ("completed");
}

/* Reflected tool function that returns an execution failure error.  */
auto
failing_tool (std::string_view reason) noexcept -> sutra::Result<std::string>
{
  return std::unexpected (
      sutra::AgentError::tool (sutra::ToolErrorCode::ExecutionFailure,
                               "failing_tool", std::string (reason)));
}

/* Custom in-memory LogSink for unit testing observability.  */
class TestLogSink final : public sutra::LogSink
{
public:
  void
  write (sutra::LogLevel /*level*/, std::string_view /*subsystem*/,
         std::string_view message, uint64_t /*timestamp_ns*/) noexcept override
  {
    const std::scoped_lock lock (mtx_);
    messages.emplace_back (message);
  }

  std::mutex mtx_;
  std::vector<std::string> messages;
};

/* Custom ProfilerHook for unit testing external profiler integration.  */
class TestProfilerHook final : public sutra::ProfilerHook
{
public:
  void
  on_zone_begin (std::string_view /*zone_name*/) noexcept override
  {
    zones_begun.fetch_add (1U, std::memory_order_relaxed);
  }

  void
  on_zone_end (std::string_view /*zone_name*/,
               uint64_t /*elapsed_ns*/) noexcept override
  {
    zones_ended.fetch_add (1U, std::memory_order_relaxed);
  }

  void
  on_counter (std::string_view /*counter_name*/,
              int64_t value) noexcept override
  {
    last_counter.store (value, std::memory_order_relaxed);
  }

  std::atomic<uint64_t> zones_begun{ 0U };
  std::atomic<uint64_t> zones_ended{ 0U };
  std::atomic<int64_t> last_counter{ 0 };
};

/* Mock LLM that recovers after observing a tool error, or fails if max_steps
   is too small.  */
class RecoveryMockLlm final : public sutra::LlmClient
{
public:
  auto
  generate (const sutra::ConversationSpanView &history, std::size_t step_index,
            const sutra::StreamingToolDispatchFn & /*on_early_tool*/,
            std::pmr::memory_resource * /*scratch_arena*/) noexcept
      -> sutra::Result<sutra::LlmStepOutput> override
  {
    sutra::LlmStepOutput out;
    out.generated_tokens = 12U;
    if (step_index == 0U)
      {
        out.reasoning_thought = "Attempting failing tool first";
        sutra::ToolCallRequest req;
        req.call_id = "call_fail_1";
        req.tool_name = "failing_tool";
        req.arguments_json = R"({"reason":"Simulated database disconnect"})";
        out.tool_calls.push_back (std::move (req));
        return out;
      }

    // Verify the error context was fed back into ShortTermMemory.
    bool saw_error_observation = false;
    for (std::size_t i = 0U; i < history.size (); ++i)
      {
        if (history.at (i).role == sutra::MessageRole::Tool
            && history.at (i).content.find ("ERROR:") != std::string::npos)
          {
            saw_error_observation = true;
          }
      }
    SUTRA_ASSERT (saw_error_observation);

    if (step_index == 1U)
      {
        out.reasoning_thought = "Recovering with add_numbers";
        sutra::ToolCallRequest req;
        req.call_id = "call_ok_2";
        req.tool_name = "add_numbers";
        req.arguments_json = R"({"lhs":20,"rhs":22})";
        out.tool_calls.push_back (std::move (req));
        return out;
      }

    out.is_final = true;
    out.final_answer = "Recovered answer: 42";
    return out;
  }
};

void
test_errors_and_observability ()
{
  SUTRA_ASSERT (sutra::version_string () == "1.0.0");
  SUTRA_ASSERT (sutra::v1::version_string () == "1.0.0");

  const auto e_llm = sutra::AgentError::llm (
      sutra::LlmErrorCode::RateLimited, "Rate limit exceeded on endpoint");
  SUTRA_ASSERT (e_llm.is_llm_error ());
  SUTRA_ASSERT (e_llm.category_name () == "LLM");

  const auto e_tool = sutra::AgentError::tool (
      sutra::ToolErrorCode::Timeout, "search", "Tool timed out after 50 ms");
  SUTRA_ASSERT (e_tool.is_tool_error ());
  SUTRA_ASSERT (e_tool.category_name () == "Tool");

  const auto e_plan = sutra::AgentError::planning (
      sutra::PlanningErrorCode::CycleDetected, 2U, "Cycle found in graph");
  SUTRA_ASSERT (e_plan.is_planning_error ());
  SUTRA_ASSERT (e_plan.category_name () == "Planning");

  const auto e_fw = sutra::AgentError::framework (
      sutra::FrameworkErrorCode::ResourceExhaustion, 12, "Out of memory");
  SUTRA_ASSERT (e_fw.is_framework_error ());
  SUTRA_ASSERT (e_fw.category_name () == "Framework");

  auto sink = std::make_shared<TestLogSink> ();
  sutra::Logger::set_sink (sink);
  sutra::Logger::set_level (sutra::LogLevel::Info);
  sutra::Logger::log (sutra::LogLevel::Debug, "unit", "Should be filtered");
  sutra::Logger::log (sutra::LogLevel::Info, "unit", "Visible log event");
  SUTRA_ASSERT (sink->messages.size () == 1U);
  SUTRA_ASSERT (sink->messages.at (0U) == "Visible log event");
  sutra::Logger::set_sink (nullptr);
  sutra::Logger::set_level (sutra::LogLevel::Warn);

  TestProfilerHook prof_hook;
  sutra::AgentProfiler::set_hook (&prof_hook);
  sutra::AgentStats::global ().reset ();
  {
    const sutra::ScopedStageTimer timer (
        sutra::AgentStats::global (), sutra::StageKind::Planning, "test_zone");
    sutra::AgentProfiler::emit_counter ("tokens", 99);
  }
  SUTRA_ASSERT (prof_hook.zones_begun.load () == 1U);
  SUTRA_ASSERT (prof_hook.zones_ended.load () == 1U);
  SUTRA_ASSERT (prof_hook.last_counter.load () == 99);
  SUTRA_ASSERT (sutra::AgentStats::global ().snapshot ().planning_calls == 1U);
  sutra::AgentProfiler::set_hook (nullptr);
}

void
test_memory_tiers_and_boundaries ()
{
  // Boundary condition: zero capacity ShortTermMemory must fail cleanly.
  const auto bad_st = sutra::ShortTermMemory::create (0U);
  SUTRA_ASSERT (!bad_st.has_value ());
  SUTRA_ASSERT (bad_st.error ().is_framework_error ());

  // Ring buffer wrap-around test.
  auto st_res = sutra::ShortTermMemory::create (3U);
  SUTRA_ASSERT (st_res.has_value ());
  for (int i = 1; i <= 5; ++i)
    {
      sutra::Message msg;
      msg.role = sutra::MessageRole::User;
      msg.content = "msg_" + std::to_string (i);
      const auto seq_res = st_res->append (std::move (msg));
      SUTRA_ASSERT (seq_res.has_value ());
      SUTRA_ASSERT (*seq_res == static_cast<uint64_t> (i));
    }
  SUTRA_ASSERT (st_res->size () == 3U);
  SUTRA_ASSERT (st_res->total_appended () == 5U);
  const auto view = st_res->borrow_view ();
  SUTRA_ASSERT (view.size () == 3U);
  SUTRA_ASSERT (view.at (0U).content == "msg_3");
  SUTRA_ASSERT (view.at (1U).content == "msg_4");
  SUTRA_ASSERT (view.at (2U).content == "msg_5");

  // LongTermMemory persistence & corruption check.
  sutra::LongTermMemory ltm;
  SUTRA_ASSERT (!ltm.put ("", "invalid").has_value ());
  SUTRA_ASSERT (ltm.put ("user_pref", "dark_mode").has_value ());
  SUTRA_ASSERT (ltm.put ("arch", "x86_64").has_value ());
  SUTRA_ASSERT (ltm.size () == 2U);
  SUTRA_ASSERT (*ltm.get ("user_pref") == "dark_mode");
  SUTRA_ASSERT (!ltm.get ("nonexistent").has_value ());

  const std::string ltm_path = "/tmp/sutra_unit_ltm.bin";
  SUTRA_ASSERT (ltm.save_to_file (ltm_path).has_value ());

  sutra::LongTermMemory ltm_loaded;
  SUTRA_ASSERT (ltm_loaded.load_from_file (ltm_path).has_value ());
  SUTRA_ASSERT (ltm_loaded.size () == 2U);
  SUTRA_ASSERT (*ltm_loaded.get ("arch") == "x86_64");

  // Corrupt the magic header and verify CheckpointCorrupted error.
  {
    std::ofstream bad_out (ltm_path, std::ios::binary | std::ios::trunc);
    bad_out << "BADMAGIC";
  }
  SUTRA_ASSERT (!ltm_loaded.load_from_file (ltm_path).has_value ());
  std::filesystem::remove (ltm_path);

  // SemanticMemory boundary conditions.
  SUTRA_ASSERT (!sutra::SemanticMemory::create (0U).has_value ());
  auto sem_res = sutra::SemanticMemory::create (4U);
  SUTRA_ASSERT (sem_res.has_value ());
  const std::array<float, 4U> vec1{ 1.0F, 0.0F, 0.0F, 0.0F };
  const std::array<float, 3U> bad_vec{ 1.0F, 0.0F, 0.0F };
  SUTRA_ASSERT (!sem_res->add_document ("", "empty id", vec1).has_value ());
  SUTRA_ASSERT (
      !sem_res->add_document ("doc1", "wrong dim", bad_vec).has_value ());
  SUTRA_ASSERT (sem_res->add_document ("doc1", "hello", vec1).has_value ());
  SUTRA_ASSERT (!sem_res->search (vec1, 0U).has_value ());
  const auto matches = sem_res->search (vec1, 1U);
  SUTRA_ASSERT (matches.has_value () && matches->size () == 1U);
  SUTRA_ASSERT (matches->at (0U).id == "doc1");
}

void
test_tools_and_error_recovery ()
{
  auto registry = std::make_shared<sutra::ToolRegistry> ();
  SUTRA_ASSERT (SUTRA_REGISTER_TOOL (*registry, &add_numbers_tool,
                                     "add_numbers", "Adds two integers", "lhs",
                                     "rhs")
                    .has_value ());
  SUTRA_ASSERT (SUTRA_REGISTER_TOOL (*registry, &slow_tool, "slow_tool",
                                     "Sleeps for sleep_ms", "sleep_ms")
                    .has_value ());
  SUTRA_ASSERT (SUTRA_REGISTER_TOOL (*registry, &failing_tool, "failing_tool",
                                     "Always returns ExecutionFailure",
                                     "reason")
                    .has_value ());

  // Duplicate tool registration must fail.
  SUTRA_ASSERT (!SUTRA_REGISTER_TOOL (*registry, &slow_tool, "slow_tool",
                                      "Duplicate", "sleep_ms")
                     .has_value ());

  sutra::InferenceArena arena (16384U);
  sutra::ToolContext ctx;
  ctx.scratch_arena = arena.resource ();

  // 1. Valid invocation.
  const auto ok_res
      = registry->invoke ("add_numbers", ctx, R"({"lhs":15,"rhs":27})");
  SUTRA_ASSERT (ok_res.has_value () && *ok_res == "42");

  // 2. Invalid tool name.
  const auto missing_res = registry->invoke ("unknown_tool", ctx, "{}");
  SUTRA_ASSERT (!missing_res.has_value ());
  SUTRA_ASSERT (
      std::get<sutra::ToolError> (missing_res.error ().variant ()).code
      == sutra::ToolErrorCode::ToolNotFound);

  // 3. Malformed JSON.
  const auto malformed_res
      = registry->invoke ("add_numbers", ctx, R"({"lhs":15,"rhs":)");
  SUTRA_ASSERT (!malformed_res.has_value ());
  SUTRA_ASSERT (
      std::get<sutra::ToolError> (malformed_res.error ().variant ()).code
      == sutra::ToolErrorCode::InvalidInput);

  // 4. Schema violation (wrong type / missing field).
  const auto schema_err_res
      = registry->invoke ("add_numbers", ctx, R"({"lhs":"not_an_int"})");
  SUTRA_ASSERT (!schema_err_res.has_value ());
  SUTRA_ASSERT (
      std::get<sutra::ToolError> (schema_err_res.error ().variant ()).code
      == sutra::ToolErrorCode::SchemaViolation);

  // 5. Tool timeout in ToolScheduler.
  sutra::WorkStealingPool pool (4U);
  sutra::ToolScheduler scheduler (pool, 4U);
  std::vector<sutra::ToolCallRequest> timeout_calls (1U);
  timeout_calls.at (0U).call_id = "c1";
  timeout_calls.at (0U).tool_name = "slow_tool";
  timeout_calls.at (0U).arguments_json = R"({"sleep_ms":25})";
  timeout_calls.at (0U).timeout_ms = 5U;

  const auto batch_res
      = scheduler.execute_batch (*registry, ctx, timeout_calls);
  SUTRA_ASSERT (batch_res.has_value () && batch_res->size () == 1U);
  SUTRA_ASSERT (!batch_res->at (0U).output.has_value ());
  SUTRA_ASSERT (std::get<sutra::ToolError> (
                    batch_res->at (0U).output.error ().variant ())
                    .code
                == sutra::ToolErrorCode::Timeout);

  // 6. MCP export verification.
  const std::string mcp_json = registry->export_mcp_tools_json ();
  SUTRA_ASSERT (mcp_json.find ("add_numbers") != std::string::npos);
  SUTRA_ASSERT (mcp_json.find ("inputSchema") != std::string::npos);

  // 7. Agent recovery from tool error within max_steps.
  sutra::AgentRuntime runtime (4U);
  auto llm = std::make_shared<RecoveryMockLlm> ();
  sutra::AgentConfig cfg;
  cfg.max_steps = 4U;
  auto agent_res = sutra::Agent::create (cfg, registry, llm);
  SUTRA_ASSERT (agent_res.has_value ());

  // Empty prompt boundary check.
  SUTRA_ASSERT (!runtime.run (**agent_res, "").has_value ());

  // Max-length context overflow boundary check.
  const std::string huge_prompt (5U * 1024U * 1024U, 'A');
  const auto overflow_res = runtime.run (**agent_res, huge_prompt);
  SUTRA_ASSERT (!overflow_res.has_value ()
                && overflow_res.error ().is_llm_error ());

  // Successful recovery run (step 0 fails tool, step 1 succeeds, step 2
  // finishes).
  const auto rec_res = runtime.run (**agent_res, "Compute with recovery");
  SUTRA_ASSERT (rec_res.has_value ());
  SUTRA_ASSERT (rec_res->final_answer == "Recovered answer: 42");
  SUTRA_ASSERT (rec_res->steps_taken == 3U);

  // Unrecoverable tool error when max_steps == 1 returns the last tool error.
  cfg.max_steps = 1U;
  auto agent_1step = sutra::Agent::create (cfg, registry, llm);
  SUTRA_ASSERT (agent_1step.has_value ());
  const auto unrec_res = runtime.run (**agent_1step, "Fail on step 0");
  SUTRA_ASSERT (!unrec_res.has_value ());
  SUTRA_ASSERT (unrec_res.error ().is_tool_error ());
}

} /* namespace */

auto
main () -> int
{
  test_errors_and_observability ();
  test_memory_tiers_and_boundaries ();
  test_tools_and_error_recovery ();
  std::cout << "All Sutra unit tests PASSED.\n";
  return 0;
}
