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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
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

/* Reads current process RSS in kilobytes from /proc/self/statm on Linux.  */
auto
read_rss_kb () -> std::size_t
{
  std::ifstream in ("/proc/self/statm");
  std::size_t virt_pages = 0U;
  std::size_t rss_pages = 0U;
  if (in >> virt_pages >> rss_pages)
    {
      return rss_pages * 4U;
    }
  return 0U;
}

auto
fast_echo_tool (const sutra::ToolContext & /*ctx*/,
                int64_t val) noexcept -> sutra::Result<std::string>
{
  return std::to_string (val + 1);
}

class FastMockLlm final : public sutra::LlmClient
{
public:
  auto
  generate (const sutra::ConversationSpanView & /*history*/,
            std::size_t step_index,
            const sutra::StreamingToolDispatchFn & /*on_early_tool*/,
            std::pmr::memory_resource * /*scratch_arena*/) noexcept
      -> sutra::Result<sutra::LlmStepOutput> override
  {
    sutra::LlmStepOutput out;
    out.generated_tokens = 8U;
    if (step_index == 0U)
      {
        out.reasoning_thought = "Calling fast_echo";
        sutra::ToolCallRequest req;
        req.call_id = "c1";
        req.tool_name = "fast_echo";
        req.arguments_json = R"({"val":41})";
        out.tool_calls.push_back (std::move (req));
        return out;
      }
    out.is_final = true;
    out.final_answer = "42";
    return out;
  }
};

/* 7.4.1: 10,000 sequential agent runs with zero memory growth.  */
void
test_10000_sequential_agent_runs ()
{
  auto registry = std::make_shared<sutra::ToolRegistry> ();
  SUTRA_ASSERT (SUTRA_REGISTER_TOOL (*registry, &fast_echo_tool, "fast_echo",
                                     "Increments val", "val")
                    .has_value ());
  auto llm = std::make_shared<FastMockLlm> ();

  sutra::AgentConfig cfg;
  cfg.max_steps = 3U;
  cfg.short_term_capacity = 16U;
  cfg.scratch_arena_bytes = 8192U;

  auto agent_res = sutra::Agent::create (cfg, registry, llm);
  SUTRA_ASSERT (agent_res.has_value ());

  sutra::AgentRuntime runtime (4U);

  // Warm up 200 runs to stabilize allocator pages.
  for (int i = 0; i < 200; ++i)
    {
      const auto r = runtime.run (**agent_res, "warmup");
      SUTRA_ASSERT (r.has_value ());
    }

  const std::size_t rss_before_kb = read_rss_kb ();

  for (int i = 0; i < 10000; ++i)
    {
      const auto r = runtime.run (**agent_res, "seq_run");
      SUTRA_ASSERT (r.has_value () && r->final_answer == "42");
    }

  const std::size_t rss_after_kb = read_rss_kb ();
  SUTRA_ASSERT (sutra::AgentStats::global ().snapshot ().active_arena_bytes
                == 0U);
  // RSS growth across 10,000 runs must be negligible (< 4 MB).
  if (rss_after_kb > rss_before_kb)
    {
      SUTRA_ASSERT ((rss_after_kb - rss_before_kb) < 4096U);
    }
}

/* 7.4.2: 1,000 concurrent agent runs across 8 threads with zero races/leaks.
 */
void
test_1000_concurrent_agent_runs ()
{
  auto registry = std::make_shared<sutra::ToolRegistry> ();
  SUTRA_ASSERT (SUTRA_REGISTER_TOOL (*registry, &fast_echo_tool, "fast_echo",
                                     "Increments val", "val")
                    .has_value ());
  auto llm = std::make_shared<FastMockLlm> ();

  sutra::AgentRuntime runtime (8U);
  constexpr int kTotalRuns = 1000;
  constexpr int kThreads = 8;
  std::atomic<int> next_run{ 0 };
  std::atomic<int> completed_runs{ 0 };

  std::vector<std::jthread> callers;
  callers.reserve (kThreads);
  for (int t = 0; t < kThreads; ++t)
    {
      callers.emplace_back ([&] {
        sutra::AgentConfig cfg;
        cfg.max_steps = 3U;
        cfg.short_term_capacity = 16U;
        cfg.scratch_arena_bytes = 8192U;
        auto local_agent = sutra::Agent::create (cfg, registry, llm);
        SUTRA_ASSERT (local_agent.has_value ());

        for (;;)
          {
            const int idx = next_run.fetch_add (1, std::memory_order_relaxed);
            if (idx >= kTotalRuns)
              {
                break;
              }
            const auto res = runtime.run (**local_agent, "concurrent_query");
            SUTRA_ASSERT (res.has_value () && res->final_answer == "42");
            completed_runs.fetch_add (1, std::memory_order_relaxed);
          }
      });
    }
  callers.clear ();
  SUTRA_ASSERT (completed_runs.load () == kTotalRuns);
}

/* 7.4.3: 500-turn conversation with incremental binary checkpointing enabled;
   verify flat per-turn latency and lossless binary replay.  */
void
test_500_turn_checkpoint_latency_stability ()
{
  const std::string cpk_path = "/tmp/sutra_500_turns_checkpoint.bin";
  std::filesystem::remove (cpk_path);

  auto registry = std::make_shared<sutra::ToolRegistry> ();
  SUTRA_ASSERT (SUTRA_REGISTER_TOOL (*registry, &fast_echo_tool, "fast_echo",
                                     "Increments val", "val")
                    .has_value ());
  auto llm = std::make_shared<FastMockLlm> ();

  sutra::AgentConfig cfg;
  cfg.max_steps = 3U;
  cfg.short_term_capacity = 64U;
  cfg.enable_checkpointing = true;
  cfg.checkpoint_path = cpk_path;

  auto agent_res = sutra::Agent::create (cfg, registry, llm);
  SUTRA_ASSERT (agent_res.has_value ());

  sutra::AgentRuntime runtime (4U);
  constexpr std::size_t kTurns = 500U;
  std::vector<uint64_t> turn_latencies_ns (kTurns, 0U);

  for (std::size_t turn = 0U; turn < kTurns; ++turn)
    {
      const auto res
          = runtime.run (**agent_res, "turn_" + std::to_string (turn));
      SUTRA_ASSERT (res.has_value ());
      turn_latencies_ns.at (turn) = res->total_latency_ns;
    }

  SUTRA_ASSERT ((*agent_res)->flush_checkpoints ().has_value ());

  // Replay all 500 turns (each turn appends 3 messages: User, Assistant, Tool,
  // plus final Assistant = 4 messages/turn -> 2000 messages total).
  const auto replayed = sutra::CheckpointManager::replay (cpk_path);
  SUTRA_ASSERT (replayed.has_value ());
  SUTRA_ASSERT (replayed->size () == kTurns * 4U);

  // Compare average latency of early turns [20..70) vs late turns [450..500).
  double early_sum = 0.0;
  double late_sum = 0.0;
  for (std::size_t i = 20U; i < 70U; ++i)
    {
      early_sum += static_cast<double> (turn_latencies_ns.at (i));
    }
  for (std::size_t i = 450U; i < 500U; ++i)
    {
      late_sum += static_cast<double> (turn_latencies_ns.at (i));
    }
  const double early_avg_us = (early_sum / 50.0) / 1000.0;
  const double late_avg_us = (late_sum / 50.0) / 1000.0;

  std::cout << "500-turn checkpoint stability: early_avg=" << early_avg_us
            << " us, late_avg=" << late_avg_us << " us\n";

  // Verify late_avg does not exhibit O(N) doubling (allow noise margin for
  // microsecond-scale jitter).
  SUTRA_ASSERT (late_avg_us < early_avg_us * 1.75 + 150.0);

  std::filesystem::remove (cpk_path);
}

} /* namespace */

auto
main () -> int
{
  test_10000_sequential_agent_runs ();
  test_1000_concurrent_agent_runs ();
  test_500_turn_checkpoint_latency_stability ();
  std::cout << "All Sutra stress and leak tests PASSED.\n";
  return 0;
}
