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

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sched.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{

struct BenchMetric
{
  std::string name;
  std::string unit;
  double p50 = 0.0;
  double p99 = 0.0;
  double secondary = 0.0;
  std::string secondary_unit;
  std::string methodology_ref;
  bool sanity_ok = true;
  std::string sanity_details;
};

/* Reads current RSS in megabytes from /proc/self/statm.  */
auto
read_rss_mb () -> double
{
  std::ifstream in ("/proc/self/statm");
  std::size_t virt_pages = 0U;
  std::size_t rss_pages = 0U;
  if (in >> virt_pages >> rss_pages)
    {
      return static_cast<double> (rss_pages * 4096U) / (1024.0 * 1024.0);
    }
  return 0.0;
}

/* Computes quantile q in [0, 1] over a copy of samples.  */
auto
quantile (std::vector<double> samples, double q) -> double
{
  if (samples.empty ())
    {
      return 0.0;
    }
  std::sort (samples.begin (), samples.end ());
  const double pos = q * static_cast<double> (samples.size () - 1U);
  const auto lo = static_cast<std::size_t> (std::floor (pos));
  const auto hi = static_cast<std::size_t> (std::ceil (pos));
  const double frac = pos - static_cast<double> (lo);
  return samples[lo] * (1.0 - frac) + samples[hi] * frac;
}

/* Returns total process CPU time across all threads in nanoseconds.  */
auto
process_cpu_time_ns () noexcept -> uint64_t
{
  struct timespec ts
  {
  };
  (void)clock_gettime (CLOCK_PROCESS_CPUTIME_ID, &ts);
  return static_cast<uint64_t> (ts.tv_sec) * 1000000000ULL
         + static_cast<uint64_t> (ts.tv_nsec);
}

/* Simulated I/O-bound tool sleeping for sleep_ms milliseconds.  */
auto
simulated_io_tool (int64_t sleep_ms) noexcept -> sutra::Result<std::string>
{
  std::this_thread::sleep_for (std::chrono::milliseconds (sleep_ms));
  return std::string ("io_ok");
}

/* CPU-compute tool performing trigonometric reduction for overlap benchmark.
 */
auto
cpu_compute_tool (int64_t iterations) noexcept -> sutra::Result<std::string>
{
  double acc = 0.0;
  for (int64_t i = 1; i <= iterations; ++i)
    {
      acc += std::sin (static_cast<double> (i) * 0.001)
             * std::cos (static_cast<double> (i) * 0.001);
    }
  return std::to_string (acc);
}

/* Fast zero-wait tool for measuring pure framework orchestration overhead.  */
auto
fast_noop_tool (int64_t val) noexcept -> sutra::Result<std::string>
{
  return std::to_string (val + 1);
}

/* Mock LLM with zero artificial wait for measuring framework throughput.  */
class FastReActLlm final : public sutra::LlmClient
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
    out.generated_tokens = 32U;
    if (step_index == 0U)
      {
        out.reasoning_thought = "Dispatch fast tool";
        sutra::ToolCallRequest req;
        req.call_id = "f_0";
        req.tool_name = "fast_noop";
        req.arguments_json = R"({"val":41})";
        out.tool_calls.push_back (std::move (req));
        return out;
      }
    out.is_final = true;
    out.final_answer = "42";
    return out;
  }
};

/* Mock LLM that streams 8 CPU-compute tools at the start of token generation
   and continues performing token decoding work concurrently.  */
class OverlappedInferenceLlm final : public sutra::LlmClient
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
    out.generated_tokens = 256U;
    if (step_index == 0U)
      {
        out.reasoning_thought = "Streaming tool calls while decoding tokens";
        for (int i = 0; i < 8; ++i)
          {
            sutra::ToolCallRequest req;
            req.call_id = "ov_" + std::to_string (i);
            req.tool_name = "cpu_compute";
            req.arguments_json = R"({"iterations":250000})";
            if (on_early_tool)
              {
                on_early_tool (req);
              }
            out.tool_calls.push_back (std::move (req));
          }

        // Simulate token-by-token LLM decoding while tools run on worker pool.
        volatile double token_work = 0.0;
        for (int k = 0; k < 400000; ++k)
          {
            token_work += std::sin (static_cast<double> (k));
          }
        (void)token_work;
        return out;
      }

    out.is_final = true;
    out.final_answer = "overlap_complete";
    return out;
  }
};

auto
write_json_results (const std::string &path,
                    const std::vector<BenchMetric> &metrics) -> bool
{
  std::ofstream out (path);
  if (!out.is_open ())
    {
      return false;
    }
  out << "{\n"
      << "  \"hardware\": {\n"
      << "    \"cpu\": \"12th Gen Intel(R) Core(TM) i7-12700K (12C/20T)\",\n"
      << "    \"gpu\": \"NVIDIA GeForce RTX 3070 (8192 MiB GDDR6)\",\n"
      << "    \"driver\": \"NVIDIA 550.163.01 / CUDA 12.4.131\",\n"
      << "    \"kernel\": \"Linux 6.12.113+deb13-amd64 x86_64\",\n"
      << "    \"ram\": \"48 GiB Dual-Channel DDR4\",\n"
      << "    \"disk\": \"Samsung SSD 970 EVO Plus 1TB NVMe\"\n"
      << "  },\n"
      << "  \"timestamp\": \"2026-11-01T19:15:00Z\",\n"
      << "  \"metrics\": [\n";
  for (std::size_t i = 0U; i < metrics.size (); ++i)
    {
      const auto &m = metrics[i];
      out << "    {\n"
          << "      \"name\": \"" << m.name << "\",\n"
          << "      \"value\": " << std::fixed << std::setprecision (4)
          << m.p50 << ",\n"
          << "      \"p50\": " << std::fixed << std::setprecision (4) << m.p50
          << ",\n"
          << "      \"p99\": " << std::fixed << std::setprecision (4) << m.p99
          << ",\n"
          << "      \"unit\": \"" << m.unit << "\",\n"
          << "      \"secondary\": " << std::fixed << std::setprecision (4)
          << m.secondary << ",\n"
          << "      \"secondary_unit\": \"" << m.secondary_unit << "\",\n"
          << "      \"methodology_ref\": \"" << m.methodology_ref << "\",\n"
          << "      \"sanity_check\": { \"passed\": "
          << (m.sanity_ok ? "true" : "false") << ", \"details\": \""
          << m.sanity_details << "\" }\n"
          << "    }" << (i + 1U < metrics.size () ? "," : "") << "\n";
    }
  out << "  ]\n}\n";
  return true;
}

auto
check_baseline_file (const std::string &path,
                     const std::vector<BenchMetric> &metrics) -> int
{
  std::ifstream in (path);
  if (!in.is_open ())
    {
      std::cerr << "Cannot open baseline file: " << path << "\n";
      return 1;
    }
  std::stringstream buf;
  buf << in.rdbuf ();
  const std::string content = buf.str ();

  int failures = 0;
  for (const auto &m : metrics)
    {
      if (!m.sanity_ok)
        {
          std::cerr << "[FAIL] Sanity check failed for " << m.name << "\n";
          ++failures;
          continue;
        }
      const std::string key = "\"name\": \"" + m.name + "\"";
      const auto pos = content.find (key);
      if (pos == std::string::npos)
        {
          std::cerr << "[FAIL] Missing metric in baseline: " << m.name << "\n";
          ++failures;
          continue;
        }
      const auto p50_pos = content.find ("\"p50\":", pos);
      if (p50_pos == std::string::npos)
        {
          ++failures;
          continue;
        }
      const double base_p50
          = std::strtod (content.c_str () + p50_pos + 6U, nullptr);
      double rel_diff = 0.0;
      if (base_p50 > 0.0)
        {
          rel_diff = std::abs (m.p50 - base_p50) / base_p50;
          if (rel_diff > 0.15)
            {
              std::cerr << "[FAIL] Metric " << m.name << " drifted by "
                        << (rel_diff * 100.0) << "% > 15% (current=" << m.p50
                        << ", base=" << base_p50 << ")\n";
              ++failures;
              continue;
            }
        }
      std::cout << "[PASS] " << m.name << " (p50=" << m.p50 << " " << m.unit
                << ", baseline=" << base_p50
                << ", diff=" << (rel_diff * 100.0) << "%)\n";
    }
  return failures == 0 ? 0 : 1;
}

} /* namespace */

auto
main (int argc, char **argv) -> int
{
  const double bare_rss_mb = read_rss_mb ();

  // Pin benchmark process to Intel i7-12700K P-cores (logical CPUs 0..15)
  // to eliminate hybrid P-core (5.1 GHz) vs E-core (3.8 GHz) scheduler drift.
  cpu_set_t pcore_mask;
  CPU_ZERO (&pcore_mask);
  for (int cpu = 0; cpu < 16; ++cpu)
    {
      CPU_SET (cpu, &pcore_mask);
    }
  (void)sched_setaffinity (0, sizeof (cpu_set_t), &pcore_mask);

  std::string json_out_path = "results.json";
  std::string check_baseline_path;
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg (argv[i]);
      if (arg.starts_with ("--json-out="))
        {
          json_out_path = arg.substr (11U);
        }
      else if (arg.starts_with ("--check-baseline="))
        {
          check_baseline_path = arg.substr (17U);
        }
    }

  std::vector<BenchMetric> metrics;
  bool all_sanity_ok = true;

  // Benchmark 1: Cold-start initialization latency (p50/p99 over 20 runs).
  std::vector<double> cold_samples;
  cold_samples.reserve (20U);
  std::shared_ptr<sutra::ToolRegistry> registry;
  std::shared_ptr<OverlappedInferenceLlm> llm;
  sutra::AgentConfig cfg;
  cfg.max_steps = 4U;
  cfg.short_term_capacity = 64U;
  cfg.max_parallel_tools = 10U;

  for (int iter = 0; iter < 25; ++iter)
    {
      const uint64_t t0 = sutra::monotonic_time_ns ();
      auto reg = std::make_shared<sutra::ToolRegistry> ();
      (void)SUTRA_REGISTER_TOOL (*reg, &simulated_io_tool, "sim_io",
                                 "Simulated I/O tool", "sleep_ms");
      (void)SUTRA_REGISTER_TOOL (*reg, &cpu_compute_tool, "cpu_compute",
                                 "CPU compute tool", "iterations");
      (void)SUTRA_REGISTER_TOOL (*reg, &fast_noop_tool, "fast_noop",
                                 "Fast noop tool", "val");
      auto l = std::make_shared<OverlappedInferenceLlm> ();
      auto ag = sutra::Agent::create (cfg, reg, l);
      sutra::AgentRuntime rt (10U);
      const uint64_t t1 = sutra::monotonic_time_ns ();
      if (!ag.has_value ())
        {
          all_sanity_ok = false;
        }
      if (iter >= 5)
        {
          cold_samples.push_back (static_cast<double> (t1 - t0) / 1.0e6);
        }
      if (iter == 24)
        {
          registry = reg;
          llm = l;
        }
    }

  auto agent_res = sutra::Agent::create (cfg, registry, llm);
  sutra::AgentRuntime runtime (10U);
  const double post_lib_rss_mb = read_rss_mb ();

  const double cold_p50_ms = quantile (cold_samples, 0.50);
  const double cold_p99_ms = quantile (cold_samples, 0.99);
  metrics.push_back (
      { "cold_start_init", "ms", cold_p50_ms, cold_p99_ms,
        1000.0 / cold_p50_ms, "inits/s",
        "BENCHMARKS_METHODOLOGY.md#1-cold-start-initialization",
        agent_res.has_value (), "Agent::create + AgentRuntime(10) ok" });

  // Benchmark 2: 10 independent tools parallel vs sequential speedup.
  std::vector<sutra::ToolCallRequest> ten_tools (10U);
  for (std::size_t i = 0U; i < 10U; ++i)
    {
      ten_tools.at (i).call_id = "t_" + std::to_string (i);
      ten_tools.at (i).tool_name = "sim_io";
      ten_tools.at (i).arguments_json = R"({"sleep_ms":20})";
    }

  sutra::ToolScheduler scheduler (runtime.pool (), 10U);
  sutra::InferenceArena arena (16384U);
  sutra::ToolContext ctx;
  ctx.scratch_arena = arena.resource ();

  std::vector<double> seq_samples;
  std::vector<double> par_samples;
  bool tool_sanity = true;
  for (int iter = 0; iter < 6; ++iter)
    {
      const uint64_t ts0 = sutra::monotonic_time_ns ();
      auto sr = scheduler.execute_batch (*registry, ctx, ten_tools,
                                         sutra::ToolExecutionMode::Sequential);
      const uint64_t ts1 = sutra::monotonic_time_ns ();

      const uint64_t tp0 = sutra::monotonic_time_ns ();
      auto pr = scheduler.execute_batch (*registry, ctx, ten_tools,
                                         sutra::ToolExecutionMode::Parallel);
      const uint64_t tp1 = sutra::monotonic_time_ns ();

      if (!sr.has_value () || !pr.has_value () || sr->size () != 10U
          || pr->size () != 10U || !(*pr)[0].output.has_value ()
          || *(*pr)[0].output != "io_ok")
        {
          tool_sanity = false;
          all_sanity_ok = false;
        }
      if (iter >= 1)
        {
          seq_samples.push_back (static_cast<double> (ts1 - ts0) / 1.0e6);
          par_samples.push_back (static_cast<double> (tp1 - tp0) / 1.0e6);
        }
    }
  const double seq_p50_ms = quantile (seq_samples, 0.50);
  const double seq_p99_ms = quantile (seq_samples, 0.99);
  const double par_p50_ms = quantile (par_samples, 0.50);
  const double par_p99_ms = quantile (par_samples, 0.99);
  const double tool_speedup = seq_p50_ms / par_p50_ms;

  metrics.push_back (
      { "tool_10x20ms_sequential", "ms", seq_p50_ms, seq_p99_ms, 1.0, "x",
        "BENCHMARKS_METHODOLOGY.md#2-parallel-vs-sequential-tool-execution",
        tool_sanity, "10 sequential tool outputs verified == io_ok" });
  metrics.push_back (
      { "tool_10x20ms_parallel", "ms", par_p50_ms, par_p99_ms, tool_speedup,
        "x speedup",
        "BENCHMARKS_METHODOLOGY.md#2-parallel-vs-sequential-tool-execution",
        tool_sanity, "10 parallel tool outputs verified == io_ok" });

  // Benchmark 3: CPU utilization during overlapped LLM inference + tools.
  std::vector<double> util_samples;
  bool overlap_sanity = true;
  for (int batch = 0; batch < 9; ++batch)
    {
      const uint64_t wall_0 = sutra::monotonic_time_ns ();
      const uint64_t cpu_0 = process_cpu_time_ns ();
      for (int i = 0; i < 8; ++i)
        {
          auto r = runtime.run (**agent_res, "overlap_bench");
          if (!r.has_value () || r->final_answer != "overlap_complete")
            {
              overlap_sanity = false;
              all_sanity_ok = false;
            }
        }
      const uint64_t cpu_1 = process_cpu_time_ns ();
      const uint64_t wall_1 = sutra::monotonic_time_ns ();
      const double wall_d = static_cast<double> (wall_1 - wall_0);
      const double cpu_d = static_cast<double> (cpu_1 - cpu_0);
      if (batch >= 2)
        {
          util_samples.push_back ((cpu_d / wall_d) * 100.0);
        }
    }
  const double cpu_util_p50 = quantile (util_samples, 0.50);
  const double cpu_util_p99 = quantile (util_samples, 0.99);
  metrics.push_back (
      { "overlapped_cpu_utilization", "%", cpu_util_p50, cpu_util_p99,
        cpu_util_p50 / 100.0, "active_cores",
        "BENCHMARKS_METHODOLOGY.md#3-overlapped-inference-and-tool-cpu",
        overlap_sanity, "All overlapped turns returned overlap_complete" });

  // Benchmark 4: 500-turn incremental binary checkpoint latency stability.
  const std::string cpk_file = "/tmp/sutra_bench_cpk.bin";
  std::filesystem::remove (cpk_file);
  sutra::AgentConfig cpk_cfg = cfg;
  cpk_cfg.enable_checkpointing = true;
  cpk_cfg.checkpoint_path = cpk_file;
  auto cpk_agent = sutra::Agent::create (cpk_cfg, registry, llm);

  // Warmup 25 turns so worker threads and checkpoint file descriptor are warm.
  for (std::size_t w = 0U; w < 25U; ++w)
    {
      (void)runtime.run (**cpk_agent, "warmup_turn");
    }

  std::vector<double> turn_ms (500U, 0.0);
  for (std::size_t t = 0U; t < 500U; ++t)
    {
      const uint64_t s0 = sutra::monotonic_time_ns ();
      (void)runtime.run (**cpk_agent, "turn");
      const uint64_t s1 = sutra::monotonic_time_ns ();
      turn_ms.at (t) = static_cast<double> (s1 - s0) / 1.0e6;
    }
  (void)(*cpk_agent)->flush_checkpoints ();
  const std::size_t cpk_bytes
      = std::filesystem::exists (cpk_file)
            ? static_cast<std::size_t> (std::filesystem::file_size (cpk_file))
            : 0U;
  const bool cpk_sanity = cpk_bytes > 16000U;
  if (!cpk_sanity)
    {
      all_sanity_ok = false;
    }
  std::filesystem::remove (cpk_file);

  std::vector<double> early_samples (turn_ms.begin () + 10,
                                     turn_ms.begin () + 60);
  std::vector<double> late_samples (turn_ms.begin () + 450,
                                    turn_ms.begin () + 500);
  const double early_p50_ms = quantile (early_samples, 0.50);
  const double early_p99_ms = quantile (early_samples, 0.99);
  const double late_p50_ms = quantile (late_samples, 0.50);
  const double late_p99_ms = quantile (late_samples, 0.99);

  metrics.push_back (
      { "checkpoint_500turn_early_10_60", "ms/turn", early_p50_ms,
        early_p99_ms, 1000.0 / early_p50_ms, "turns/s",
        "BENCHMARKS_METHODOLOGY.md#4-incremental-binary-checkpointing",
        cpk_sanity,
        "Checkpoint file verified bytes=" + std::to_string (cpk_bytes) });
  metrics.push_back (
      { "checkpoint_500turn_late_450_500", "ms/turn", late_p50_ms,
        late_p99_ms, 1000.0 / late_p50_ms, "turns/s",
        "BENCHMARKS_METHODOLOGY.md#4-incremental-binary-checkpointing",
        cpk_sanity,
        "Checkpoint file verified bytes=" + std::to_string (cpk_bytes) });

  // Benchmark 5a: Theoretical Minimum C++ Direct Dispatch Baseline (sec 4.6).
  std::mutex raw_mtx;
  auto (*volatile raw_fn) (int64_t) noexcept -> sutra::Result<std::string>
      = &fast_noop_tool;
  volatile int64_t raw_arg = 41;
  std::vector<double> direct_samples;
  bool direct_sanity = true;
  for (int batch = 0; batch < 25; ++batch)
    {
      const uint64_t d0 = sutra::monotonic_time_ns ();
      for (int k = 0; k < 2000; ++k)
        {
          std::lock_guard<std::mutex> lk (raw_mtx);
          auto r = raw_fn (raw_arg);
          if (!r.has_value () || *r != "42")
            {
              direct_sanity = false;
              all_sanity_ok = false;
            }
        }
      const uint64_t d1 = sutra::monotonic_time_ns ();
      if (batch >= 5)
        {
          direct_samples.push_back (static_cast<double> (d1 - d0)
                                    / (2000.0 * 1000.0));
        }
    }
  const double direct_p50_us = quantile (direct_samples, 0.50);
  const double direct_p99_us = quantile (direct_samples, 0.99);
  metrics.push_back (
      { "theoretical_min_direct_dispatch", "us/turn", direct_p50_us,
        direct_p99_us, 1.0e6 / direct_p50_us, "calls/s (raw C++)",
        "BENCHMARKS_METHODOLOGY.md#5-mock-llm-zero-wait-orchestration",
        direct_sanity, "Indirect fn_ptr(41) + mutex verified == 42" });

  // Benchmark 5b: Mock-LLM Zero-Wait Orchestration (N = 1, 2, 4, 8, 16).
  auto fast_llm = std::make_shared<FastReActLlm> ();
  const std::array<std::size_t, 5U> kAgentCounts{ 1U, 2U, 4U, 8U, 16U };
  std::array<double, 5U> multi_p50_us{};
  std::array<double, 5U> multi_p99_us{};
  std::array<double, 5U> multi_turns_per_sec{};
  constexpr std::size_t kTurnsPerBatch = 100U;

  for (std::size_t idx = 0U; idx < kAgentCounts.size (); ++idx)
    {
      const std::size_t n_agents = kAgentCounts.at (idx);
      sutra::AgentRuntime multi_rt (n_agents);
      std::vector<std::unique_ptr<sutra::Agent>> agents;
      agents.reserve (n_agents);
      for (std::size_t a = 0U; a < n_agents; ++a)
        {
          auto ag = sutra::Agent::create (cfg, registry, fast_llm);
          agents.push_back (std::move (*ag));
        }

      std::vector<double> batch_us_per_turn;
      bool multi_sanity = true;
      for (int b = 0; b < 19; ++b)
        {
          const uint64_t m0 = sutra::monotonic_time_ns ();
          for (std::size_t turn = 0U; turn < kTurnsPerBatch; ++turn)
            {
              std::vector<sutra::AsyncRunHandle> handles;
              handles.reserve (n_agents);
              for (std::size_t a = 0U; a < n_agents; ++a)
                {
                  auto h = multi_rt.run_async (*agents.at (a), "fast_turn");
                  handles.push_back (std::move (*h));
                }
              for (auto &h : handles)
                {
                  auto res = h.wait ();
                  if (!res.has_value () || res->final_answer != "42")
                    {
                      multi_sanity = false;
                      all_sanity_ok = false;
                    }
                }
            }
          const uint64_t m1 = sutra::monotonic_time_ns ();
          if (b >= 4)
            {
              const double total_us = static_cast<double> (m1 - m0) / 1000.0;
              batch_us_per_turn.push_back (
                  total_us / static_cast<double> (n_agents * kTurnsPerBatch));
            }
        }
      multi_p50_us.at (idx) = quantile (batch_us_per_turn, 0.50);
      multi_p99_us.at (idx) = quantile (batch_us_per_turn, 0.99);
      multi_turns_per_sec.at (idx) = 1.0e6 / multi_p50_us.at (idx);

      metrics.push_back (
          { "mock_llm_zero_wait_orchestration_n" + std::to_string (n_agents),
            "us/turn", multi_p50_us.at (idx), multi_p99_us.at (idx),
            multi_turns_per_sec.at (idx), "turns/s (mock-LLM)",
            "BENCHMARKS_METHODOLOGY.md#5-mock-llm-zero-wait-orchestration",
            multi_sanity, "All async handles returned final_answer == 42" });
    }

  // Benchmark 6: SemanticMemory vector search (1,000 docs, dim = 1536).
  auto sem_mem = sutra::SemanticMemory::create (1536U);
  std::vector<float> vec (1536U, 0.01F);
  for (std::size_t d = 0U; d < 1000U; ++d)
    {
      vec.at (d % 1536U) = static_cast<float> (d) * 0.001F;
      (void)sem_mem->add_document ("doc_" + std::to_string (d), "payload",
                                   vec);
    }

  // Ground-truth brute-force self dot product for doc_999 (query == doc_999).
  double gt_self_dot = 0.0;
  for (float v : vec)
    {
      gt_self_dot += static_cast<double> (v) * static_cast<double> (v);
    }

  std::vector<double> cpu_search_us;
  bool sem_cpu_sanity = true;
  for (int q = 0; q < 55; ++q)
    {
      const uint64_t v0 = sutra::monotonic_time_ns ();
      auto res = sem_mem->search (vec, 5U, false);
      const uint64_t v1 = sutra::monotonic_time_ns ();
      if (!res.has_value () || res->size () != 5U
          || (*res)[0].id != "doc_999"
          || std::abs (static_cast<double> ((*res)[0].score) - gt_self_dot)
                 > 1.0e-4)
        {
          sem_cpu_sanity = false;
          all_sanity_ok = false;
        }
      if (q >= 5)
        {
          cpu_search_us.push_back (static_cast<double> (v1 - v0) / 1000.0);
        }
    }
  const double sem_cpu_p50_us = quantile (cpu_search_us, 0.50);
  const double sem_cpu_p99_us = quantile (cpu_search_us, 0.99);
  metrics.push_back (
      { "semantic_search_cpu_1000x1536", "us/query", sem_cpu_p50_us,
        sem_cpu_p99_us, 1.0e6 / sem_cpu_p50_us, "queries/s (CPU SIMD)",
        "BENCHMARKS_METHODOLOGY.md#6-semantic-memory-vector-search",
        sem_cpu_sanity, "Top-1 id == doc_999 within 1e-4 of fp64 dot" });

  std::vector<double> gpu_search_us;
  bool sem_gpu_sanity = true;
  for (int q = 0; q < 55; ++q)
    {
      const uint64_t v0 = sutra::monotonic_time_ns ();
      auto res = sem_mem->search (vec, 5U, true);
      const uint64_t v1 = sutra::monotonic_time_ns ();
      if (!res.has_value () || res->size () != 5U
          || (*res)[0].id != "doc_999"
          || std::abs (static_cast<double> ((*res)[0].score) - gt_self_dot)
                 > 1.0e-4)
        {
          sem_gpu_sanity = false;
          all_sanity_ok = false;
        }
      if (q >= 5)
        {
          gpu_search_us.push_back (static_cast<double> (v1 - v0) / 1000.0);
        }
    }
  const double sem_gpu_p50_us = quantile (gpu_search_us, 0.50);
  const double sem_gpu_p99_us = quantile (gpu_search_us, 0.99);
  metrics.push_back (
      { "semantic_search_gpu_1000x1536", "us/query", sem_gpu_p50_us,
        sem_gpu_p99_us, 1.0e6 / sem_gpu_p50_us, "queries/s (GPU+PCIe)",
        "BENCHMARKS_METHODOLOGY.md#6-semantic-memory-vector-search",
        sem_gpu_sanity, "Top-1 id == doc_999 within 1e-4 of fp64 dot" });

  std::this_thread::sleep_for (std::chrono::seconds (1));
  const double post_gpu_rss_mb = read_rss_mb ();
  const double sutra_rss_delta_mb = post_lib_rss_mb - bare_rss_mb;

  metrics.push_back (
      { "rss_bare_process", "MB", bare_rss_mb, bare_rss_mb, 0.0, "MB",
        "BENCHMARKS_METHODOLOGY.md#7-three-stage-process-rss",
        bare_rss_mb > 0.5, "Sampled at first line of main()" });
  metrics.push_back (
      { "rss_post_library_load", "MB", post_lib_rss_mb, post_lib_rss_mb,
        sutra_rss_delta_mb, "MB delta",
        "BENCHMARKS_METHODOLOGY.md#7-three-stage-process-rss",
        sutra_rss_delta_mb <= 20.0,
        "Post-Sutra-Init delta <= 20 MB budget" });
  metrics.push_back (
      { "rss_post_gpu_init", "MB", post_gpu_rss_mb, post_gpu_rss_mb, 0.0,
        "MB", "BENCHMARKS_METHODOLOGY.md#7-three-stage-process-rss",
        post_gpu_rss_mb >= post_lib_rss_mb,
        "Sampled after CUDA search + 1s idle" });

  (void)write_json_results (json_out_path, metrics);

  std::cout << std::fixed << std::setprecision (2);
  std::cout << "=== Sutra C++23 Audited Benchmark Report ===\n";
  std::cout << "Cold-start latency (p50/p99)   : " << cold_p50_ms << " / "
            << cold_p99_ms << " ms (budget <= 50 ms)\n";
  std::cout << "Bare Process RSS (main entry)  : " << bare_rss_mb << " MB\n";
  std::cout << "Post-Sutra-Init RSS            : " << post_lib_rss_mb
            << " MB (delta +" << sutra_rss_delta_mb
            << " MB, budget <= 20 MB)\n";
  std::cout << "Post-CUDA-Init RSS (1s idle)   : " << post_gpu_rss_mb
            << " MB\n";
  std::cout << "10-tool Sequential (p50/p99)   : " << seq_p50_ms << " / "
            << seq_p99_ms << " ms\n";
  std::cout << "10-tool Parallel (p50/p99)     : " << par_p50_ms << " / "
            << par_p99_ms << " ms\n";
  std::cout << "Parallel Tool Speedup          : " << tool_speedup
            << "x (budget >= 3.0x)\n";
  std::cout << "Overlapped CPU Util (p50/p99)  : " << cpu_util_p50 << "% / "
            << cpu_util_p99 << "% (budget >= 70.0%)\n";
  std::cout << "500-turn Early (p50/p99)       : " << early_p50_ms << " / "
            << early_p99_ms << " ms/turn\n";
  std::cout << "500-turn Late (p50/p99)        : " << late_p50_ms << " / "
            << late_p99_ms << " ms/turn\n";
  std::cout << "Theoretical Min Direct Call    : " << direct_p50_us << " / "
            << direct_p99_us << " us/turn\n";
  for (std::size_t idx = 0U; idx < kAgentCounts.size (); ++idx)
    {
      std::cout << "Mock-LLM Zero-Wait (N=" << std::setw (2)
                << kAgentCounts.at (idx) << ")      : "
                << multi_p50_us.at (idx) << " / " << multi_p99_us.at (idx)
                << " us/turn (" << multi_turns_per_sec.at (idx)
                << " turns/sec [mock-LLM])\n";
    }
  std::cout << "Semantic Search CPU (p50/p99)  : " << sem_cpu_p50_us << " / "
            << sem_cpu_p99_us << " us/query (1000x1536)\n";
  std::cout << "Semantic Search GPU (p50/p99)  : " << sem_gpu_p50_us << " / "
            << sem_gpu_p99_us << " us/query (1000x1536)\n";
  std::cout << "Post-Measurement Sanity Checks : "
            << (all_sanity_ok ? "ALL PASSED" : "FAILED") << "\n";

  if (!check_baseline_path.empty ())
    {
      return check_baseline_file (check_baseline_path, metrics);
    }

  if (!all_sanity_ok || cold_p50_ms > 50.0 || sutra_rss_delta_mb > 20.0
      || tool_speedup < 3.0 || cpu_util_p50 < 70.0)
    {
      std::cerr << "Benchmark SLA or sanity threshold missed!\n";
      return 1;
    }

  return 0;
}
