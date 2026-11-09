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
#include <sutra/scheduler.hpp>

#include <algorithm>
#include <chrono>
#include <unordered_map>

namespace sutra::inline v1
{

namespace
{

constexpr std::size_t kDefaultWorkerThreads = 8U;
constexpr uint64_t kNsPerMillisecond = 1000000ULL;

/* Executes a single ToolCallRequest and enforces timeout_ms if non-zero.  */
[[nodiscard]] auto
run_single_tool_call (const ToolRegistry &registry, const ToolContext &ctx,
                      const ToolCallRequest &req) noexcept -> ToolCallResult
{
  const ScopedStageTimer timer (AgentStats::global (),
                                StageKind::ToolExecution, "tool_execution");
  const uint64_t start_ns = monotonic_time_ns ();
  Result<std::string> out
      = registry.invoke (req.tool_name, ctx, req.arguments_json);
  const uint64_t end_ns = monotonic_time_ns ();
  const uint64_t elapsed_ns = (end_ns >= start_ns) ? (end_ns - start_ns) : 0U;

  if (req.timeout_ms > 0U
      && elapsed_ns
             > static_cast<uint64_t> (req.timeout_ms) * kNsPerMillisecond)
    {
      out = std::unexpected (AgentError::tool (
          ToolErrorCode::Timeout, req.tool_name,
          "Tool execution exceeded configured timeout_ms budget"));
    }

  return ToolCallResult{ req.call_id, req.tool_name, std::move (out),
                         elapsed_ns };
}

} /* namespace */

/* Initializes the work-stealing thread pool and per-worker lock-free queues.
 */
WorkStealingPool::WorkStealingPool (std::size_t num_threads,
                                    std::size_t queue_capacity)
    : num_threads_ (num_threads > 0U
                        ? num_threads
                        : std::max<std::size_t> (
                              1U, std::thread::hardware_concurrency ()))
{
  if (num_threads_ == 0U)
    {
      num_threads_ = kDefaultWorkerThreads;
    }

  queues_.reserve (num_threads_);
  for (std::size_t i = 0U; i < num_threads_; ++i)
    {
      queues_.push_back (
          std::make_unique<LockFreeQueue<TaskFn>> (queue_capacity));
    }

  workers_.reserve (num_threads_);
  for (std::size_t i = 0U; i < num_threads_; ++i)
    {
      workers_.emplace_back ([this, i] { worker_loop (i); });
    }
}

/* Signals stop_ and wakes all worker threads for clean RAII join.  */
WorkStealingPool::~WorkStealingPool () noexcept
{
  stop_.store (true, std::memory_order_release);
  work_signal_.fetch_add (1U, std::memory_order_acq_rel);
  work_signal_.notify_all ();
}

/* Enqueues a task onto a worker queue and notifies sleeping workers.  */
void
WorkStealingPool::submit (TaskFn task) noexcept
{
  const std::size_t start_idx
      = submit_cursor_.fetch_add (1U, std::memory_order_relaxed)
        % num_threads_;

  for (std::size_t attempt = 0U; attempt < num_threads_; ++attempt)
    {
      const std::size_t idx = (start_idx + attempt) % num_threads_;
      if (queues_.at (idx)->try_push (std::move (task)))
        {
          work_signal_.fetch_add (1U, std::memory_order_release);
          work_signal_.notify_one ();
          return;
        }
    }

  // Fallback if all bounded queues are momentarily saturated: run inline.
  if (task)
    {
      task ();
    }
}

/* Steals and runs one available task from any worker queue.  */
auto
WorkStealingPool::try_run_one () noexcept -> bool
{
  const std::size_t start_idx
      = submit_cursor_.load (std::memory_order_relaxed) % num_threads_;
  TaskFn task;
  for (std::size_t offset = 0U; offset < num_threads_; ++offset)
    {
      const std::size_t idx = (start_idx + offset) % num_threads_;
      if (queues_.at (idx)->try_pop (task))
        {
          if (task)
            {
              task ();
            }
          return true;
        }
    }
  return false;
}

/* Returns the number of active worker threads.  */
auto
WorkStealingPool::thread_count () const noexcept -> std::size_t
{
  return num_threads_;
}

/* Main worker loop popping local tasks or stealing from sibling queues.  */
void
WorkStealingPool::worker_loop (std::size_t worker_id) noexcept
{
  TaskFn task;
  while (!stop_.load (std::memory_order_acquire))
    {
      bool found = queues_.at (worker_id)->try_pop (task);
      if (!found)
        {
          for (std::size_t offset = 1U; offset < num_threads_; ++offset)
            {
              const std::size_t victim = (worker_id + offset) % num_threads_;
              if (queues_.at (victim)->try_pop (task))
                {
                  found = true;
                  break;
                }
            }
        }

      if (found)
        {
          if (task)
            {
              task ();
            }
        }
      else
        {
          const uint32_t observed
              = work_signal_.load (std::memory_order_acquire);
          if (!stop_.load (std::memory_order_acquire))
            {
              work_signal_.wait (observed, std::memory_order_relaxed);
            }
        }
    }

  // Drain any remaining local tasks before exiting.
  while (queues_.at (worker_id)->try_pop (task))
    {
      if (task)
        {
          task ();
        }
    }
}

/* Constructs a ToolScheduler with a reference to the WorkStealingPool.  */
ToolScheduler::ToolScheduler (WorkStealingPool &pool,
                              std::size_t max_concurrency) noexcept
    : pool_ (&pool),
      max_concurrency_ (max_concurrency > 0U ? max_concurrency : 1U)
{
}

/* Performs Kahn topological level partitioning on tool call dependencies.  */
auto
ToolScheduler::analyze_dependencies (
    std::span<const ToolCallRequest> calls) noexcept
    -> Result<std::vector<std::vector<std::size_t>>>
{
  if (calls.empty ())
    {
      return std::vector<std::vector<std::size_t>>{};
    }

  try
    {
      const std::size_t n = calls.size ();
      std::unordered_map<std::string, std::size_t> id_to_idx;
      id_to_idx.reserve (n);

      for (std::size_t i = 0U; i < n; ++i)
        {
          if (calls[i].call_id.empty ())
            {
              return std::unexpected (AgentError::planning (
                  PlanningErrorCode::NoValidPlan, i,
                  "ToolCallRequest call_id must not be empty"));
            }
          if (id_to_idx.contains (calls[i].call_id))
            {
              return std::unexpected (AgentError::planning (
                  PlanningErrorCode::NoValidPlan, i,
                  "Duplicate call_id in ToolCallRequest batch"));
            }
          id_to_idx.emplace (calls[i].call_id, i);
        }

      std::vector<std::vector<std::size_t>> adj (n);
      std::vector<std::size_t> in_degree (n, 0U);

      for (std::size_t i = 0U; i < n; ++i)
        {
          for (const std::string &dep_id : calls[i].depends_on_call_ids)
            {
              const auto it = id_to_idx.find (dep_id);
              if (it == id_to_idx.end ())
                {
                  return std::unexpected (AgentError::planning (
                      PlanningErrorCode::DanglingDependency, i,
                      "Tool call depends on unknown call_id"));
                }
              adj.at (it->second).push_back (i);
              ++in_degree.at (i);
            }
        }

      std::vector<std::size_t> current_wave;
      for (std::size_t i = 0U; i < n; ++i)
        {
          if (in_degree.at (i) == 0U)
            {
              current_wave.push_back (i);
            }
        }

      std::vector<std::vector<std::size_t>> waves;
      std::size_t visited = 0U;

      while (!current_wave.empty ())
        {
          std::vector<std::size_t> next_wave;
          for (const std::size_t u : current_wave)
            {
              ++visited;
              for (const std::size_t v : adj.at (u))
                {
                  --in_degree.at (v);
                  if (in_degree.at (v) == 0U)
                    {
                      next_wave.push_back (v);
                    }
                }
            }
          waves.push_back (std::move (current_wave));
          current_wave = std::move (next_wave);
        }

      if (visited != n)
        {
          return std::unexpected (AgentError::planning (
              PlanningErrorCode::CycleDetected, 0U,
              "Cyclic dependency detected among tool calls in planning step"));
        }

      return waves;
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory while analyzing tool dependencies"));
    }
}

/* Executes independent tool waves in parallel by default.  */
auto
ToolScheduler::execute_batch (
    const ToolRegistry &registry, const ToolContext &ctx,
    std::span<const ToolCallRequest> calls,
    ToolExecutionMode mode) noexcept -> Result<std::vector<ToolCallResult>>
{
  const ScopedStageTimer dispatch_timer (
      AgentStats::global (), StageKind::ToolDispatch, "tool_dispatch");

  auto waves_res = analyze_dependencies (calls);
  if (!waves_res)
    {
      return std::unexpected (waves_res.error ());
    }

  try
    {
      std::vector<ToolCallResult> results (calls.size ());

      for (const std::vector<std::size_t> &wave : *waves_res)
        {
          if (mode == ToolExecutionMode::Sequential || wave.size () == 1U
              || max_concurrency_ == 1U || pool_ == nullptr)
            {
              for (const std::size_t idx : wave)
                {
                  results.at (idx)
                      = run_single_tool_call (registry, ctx, calls[idx]);
                }
              continue;
            }

          // Parallel execution across WorkStealingPool with concurrency cap.
          std::atomic<std::size_t> next_item{ 0U };
          std::atomic<std::size_t> tasks_done{ 0U };
          const std::size_t wave_size = wave.size ();
          const std::size_t num_tasks = std::min (wave_size, max_concurrency_);

          for (std::size_t worker = 0U; worker < num_tasks; ++worker)
            {
              pool_->submit ([&registry, &ctx, calls, &wave, &results,
                              &next_item, &tasks_done, wave_size] () noexcept {
                for (;;)
                  {
                    const std::size_t pos
                        = next_item.fetch_add (1U, std::memory_order_relaxed);
                    if (pos >= wave_size)
                      {
                        break;
                      }
                    const std::size_t call_idx = wave.at (pos);
                    results.at (call_idx) = run_single_tool_call (
                        registry, ctx, calls[call_idx]);
                  }
                tasks_done.fetch_add (1U, std::memory_order_release);
              });
            }

          for (;;)
            {
              const std::size_t pos
                  = next_item.fetch_add (1U, std::memory_order_relaxed);
              if (pos >= wave_size)
                {
                  break;
                }
              const std::size_t call_idx = wave.at (pos);
              results.at (call_idx)
                  = run_single_tool_call (registry, ctx, calls[call_idx]);
            }

          while (tasks_done.load (std::memory_order_acquire) < num_tasks)
            {
              if (!pool_->try_run_one ())
                {
                  std::this_thread::yield ();
                }
            }
        }

      return results;
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory while executing tool call batch"));
    }
}

/* Returns the maximum number of concurrent tool executions per wave.  */
auto
ToolScheduler::max_concurrency () const noexcept -> std::size_t
{
  return max_concurrency_;
}

} /* namespace sutra::inline v1 */
