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
#include <sutra/runtime.hpp>

#include <unordered_set>
#include <utility>
#include <vector>

namespace sutra::inline v1
{

namespace
{

constexpr std::size_t kEarlyCompletionQueueSize = 128U;
constexpr std::size_t kMaxPromptLengthBytes = 4U * 1024U * 1024U;

} /* namespace */

/* Constructs an Agent with pre-validated components.  */
Agent::Agent (PrivateTag /*tag*/, AgentConfig config, MemorySystem memory,
              std::shared_ptr<ToolRegistry> tools,
              std::shared_ptr<LlmClient> llm, std::shared_ptr<Planner> planner,
              std::unique_ptr<CheckpointManager> checkpointer)
    : config_ (std::move (config)), memory_ (std::move (memory)),
      tools_ (std::move (tools)), llm_ (std::move (llm)),
      planner_ (std::move (planner)), checkpointer_ (std::move (checkpointer))
{
}

/* Validates configuration and creates an Agent instance.  */
auto
Agent::create (AgentConfig config, std::shared_ptr<ToolRegistry> tools,
               std::shared_ptr<LlmClient> llm,
               std::shared_ptr<Planner> planner) noexcept
    -> Result<std::unique_ptr<Agent>>
{
  if (config.max_steps == 0U || config.short_term_capacity == 0U
      || config.semantic_dimension == 0U || config.max_parallel_tools == 0U
      || config.scratch_arena_bytes == 0U)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "AgentConfig capacities and max_steps must all be greater than "
          "zero"));
    }
  if (tools == nullptr || llm == nullptr)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "Agent::create requires non-null ToolRegistry and LlmClient"));
    }

  auto mem_res = MemorySystem::create (config.short_term_capacity,
                                       config.semantic_dimension);
  if (!mem_res)
    {
      return std::unexpected (mem_res.error ());
    }

  std::unique_ptr<CheckpointManager> cpk;
  if (config.enable_checkpointing)
    {
      auto cpk_res = CheckpointManager::create (config.checkpoint_path);
      if (!cpk_res)
        {
          return std::unexpected (cpk_res.error ());
        }
      cpk = std::move (*cpk_res);
    }

  try
    {
      if (planner == nullptr)
        {
          planner = std::make_shared<ReActPlanner> ();
        }
      return std::make_unique<Agent> (PrivateTag{}, std::move (config),
                                      std::move (*mem_res), std::move (tools),
                                      std::move (llm), std::move (planner),
                                      std::move (cpk));
    }
  catch (...)
    {
      return std::unexpected (
          AgentError::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                                 "Out of memory while constructing Agent"));
    }
}

/* Returns mutable reference to the agent's MemorySystem.  */
auto
Agent::memory () noexcept -> MemorySystem &
{
  return memory_;
}

/* Returns const reference to the agent's MemorySystem.  */
auto
Agent::memory () const noexcept -> const MemorySystem &
{
  return memory_;
}

/* Returns const reference to the agent's configuration.  */
auto
Agent::config () const noexcept -> const AgentConfig &
{
  return config_;
}

/* Returns const reference to the bound ToolRegistry.  */
auto
Agent::tools () const noexcept -> const ToolRegistry &
{
  return *tools_;
}

/* Returns mutable reference to the bound LlmClient.  */
auto
Agent::llm () noexcept -> LlmClient &
{
  return *llm_;
}

/* Returns mutable reference to the bound Planner.  */
auto
Agent::planner () noexcept -> Planner &
{
  return *planner_;
}

/* Flushes any queued asynchronous checkpoints to disk.  */
auto
Agent::flush_checkpoints () noexcept -> Result<void>
{
  if (checkpointer_ == nullptr)
    {
      return {};
    }
  return checkpointer_->flush ();
}

/* Constructs an AsyncRunHandle backed by a lock-free completion queue.  */
AsyncRunHandle::AsyncRunHandle (std::shared_ptr<SharedState> state,
                                WorkStealingPool *pool) noexcept
    : state_ (std::move (state)), pool_ (pool)
{
}

/* Checks whether the async run has posted its completion result.  */
auto
AsyncRunHandle::is_ready () const noexcept -> bool
{
  if (state_ == nullptr)
    {
      return false;
    }
  return state_->ready_flag.load (std::memory_order_acquire) != 0U;
}

/* Cooperatively drains worker tasks while waiting for async run completion. */
auto
AsyncRunHandle::wait () noexcept -> Result<AgentRunResult>
{
  if (state_ == nullptr)
    {
      return std::unexpected (
          AgentError::framework (FrameworkErrorCode::InvalidConfiguration, 0,
                                 "Cannot wait on an empty AsyncRunHandle"));
    }

  while (state_->ready_flag.load (std::memory_order_acquire) == 0U)
    {
      if (pool_ == nullptr || !pool_->try_run_one ())
        {
          std::this_thread::yield ();
        }
    }

  Result<AgentRunResult> res = std::unexpected (
      AgentError::framework (FrameworkErrorCode::InvalidConfiguration, 0,
                             "AsyncRunHandle result was already consumed"));
  (void)state_->completion_queue.try_pop (res);
  return res;
}

/* Initializes the AgentRuntime and its WorkStealingPool.  */
AgentRuntime::AgentRuntime (std::size_t num_threads,
                            std::size_t queue_capacity)
    : pool_ (num_threads, queue_capacity)
{
}

/* Executes an agent run with zero-copy ToolContext, overlapped streaming tool
   execution, parallel ToolScheduler waves, and async binary checkpointing.  */
auto
AgentRuntime::run (Agent &agent, std::string_view user_prompt) noexcept
    -> Result<AgentRunResult>
{
  if (user_prompt.empty ())
    {
      return std::unexpected (
          AgentError::framework (FrameworkErrorCode::InvalidConfiguration, 0,
                                 "User prompt must not be empty"));
    }
  if (user_prompt.size () > kMaxPromptLengthBytes)
    {
      return std::unexpected (AgentError::llm (
          LlmErrorCode::ContextOverflow,
          "User prompt exceeds maximum supported context byte length"));
    }

  // Enforce explicit concurrency policy on the target Agent instance.
  const uint32_t prev_callers
      = agent.active_callers_.fetch_add (1U, std::memory_order_acq_rel);
  struct CallerGuard
  {
    std::atomic<uint32_t> *counter = nullptr;
    explicit CallerGuard (std::atomic<uint32_t> *c) noexcept : counter (c) {}
    CallerGuard (const CallerGuard &) = delete;
    auto operator= (const CallerGuard &) -> CallerGuard & = delete;
    CallerGuard (CallerGuard &&) = delete;
    auto operator= (CallerGuard &&) -> CallerGuard & = delete;
    ~CallerGuard () noexcept
    {
      if (counter != nullptr)
        {
          counter->fetch_sub (1U, std::memory_order_acq_rel);
        }
    }
  } const caller_guard (&agent.active_callers_);

  if (prev_callers > 0U && agent.config_.reject_concurrent_same_agent)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ConcurrencyViolation, 0,
          "Concurrent calls on the same Agent instance are disabled by "
          "reject_concurrent_same_agent"));
    }

  const std::scoped_lock turn_lock (agent.turn_mtx_);
  const uint64_t run_start_ns = monotonic_time_ns ();
  const uint64_t run_id
      = agent.run_counter_.fetch_add (1U, std::memory_order_relaxed) + 1U;

  try
    {
      InferenceArena scratch_arena (agent.config_.scratch_arena_bytes);
      ToolScheduler scheduler (pool_, agent.config_.max_parallel_tools);

      Message user_msg;
      user_msg.role = MessageRole::User;
      user_msg.content = std::string (user_prompt);
      user_msg.timestamp_ns = run_start_ns;
      const auto append_res
          = agent.memory_.short_term ().append (std::move (user_msg));
      if (!append_res)
        {
          return std::unexpected (append_res.error ());
        }

      AgentRunResult run_result;
      AgentError last_tool_error;
      bool saw_tool_error = false;

      for (std::size_t step = 0U; step < agent.config_.max_steps; ++step)
        {
          scratch_arena.reset ();
          const ConversationSpanView history_view
              = agent.memory_.short_term ().borrow_view ();

          ToolContext tool_ctx;
          tool_ctx.conversation_view = history_view;
          tool_ctx.scratch_arena = scratch_arena.resource ();
          tool_ctx.run_id = run_id;
          tool_ctx.step_index = step;

          // Lock-free completion queue for tools dispatched mid-LLM-inference.
          LockFreeQueue<ToolCallResult> early_completions (
              kEarlyCompletionQueueSize);
          std::atomic<std::size_t> early_dispatched_count{ 0U };
          std::atomic<std::size_t> early_completed_count{ 0U };
          std::unordered_set<std::string> early_call_ids;

          const StreamingToolDispatchFn on_early_tool
              = [this, &agent, &tool_ctx, &early_completions,
                 &early_dispatched_count, &early_completed_count,
                 &early_call_ids] (ToolCallRequest req) noexcept {
                  if (req.call_id.empty ())
                    {
                      return;
                    }
                  early_call_ids.insert (req.call_id);
                  early_dispatched_count.fetch_add (1U,
                                                    std::memory_order_release);

                  pool_.submit ([&agent, &tool_ctx, &early_completions,
                                 &early_completed_count,
                                 req = std::move (req)] () noexcept {
                    const ScopedStageTimer timer (AgentStats::global (),
                                                  StageKind::ToolExecution,
                                                  "overlapped_tool_execution");
                    const uint64_t t0 = monotonic_time_ns ();
                    Result<std::string> out = agent.tools_->invoke (
                        req.tool_name, tool_ctx, req.arguments_json);
                    const uint64_t t1 = monotonic_time_ns ();
                    ToolCallResult res{ req.call_id, req.tool_name,
                                        std::move (out),
                                        (t1 >= t0) ? (t1 - t0) : 0U };
                    (void)early_completions.try_push (std::move (res));
                    early_completed_count.fetch_add (
                        1U, std::memory_order_release);
                  });
                };

          auto decision_res = agent.planner_->plan_step (
              *agent.llm_, history_view, step, on_early_tool,
              scratch_arena.resource ());
          if (!decision_res)
            {
              // Wait for any in-flight early tools before returning error.
              while (early_completed_count.load (std::memory_order_acquire)
                     < early_dispatched_count.load (std::memory_order_acquire))
                {
                  if (!pool_.try_run_one ())
                    {
                      std::this_thread::yield ();
                    }
                }
              return std::unexpected (decision_res.error ());
            }

          ++run_result.steps_taken;
          run_result.generated_tokens += decision_res->generated_tokens;
          const std::size_t num_early
              = early_dispatched_count.load (std::memory_order_acquire);
          if (num_early > 0U && decision_res->generated_tokens > 0U)
            {
              AgentStats::global ().record_overlapped_tokens (
                  decision_res->generated_tokens);
              run_result.overlapped_tokens += decision_res->generated_tokens;
            }

          // Wait for overlapped early-dispatched tools to finish.
          while (early_completed_count.load (std::memory_order_acquire)
                 < num_early)
            {
              if (!pool_.try_run_one ())
                {
                  std::this_thread::yield ();
                }
            }

          std::vector<ToolCallResult> step_tool_results;
          ToolCallResult early_res;
          while (early_completions.try_pop (early_res))
            {
              step_tool_results.push_back (std::move (early_res));
            }

          // Collect remaining tool calls not already dispatched during stream.
          std::vector<ToolCallRequest> remaining_calls;
          for (const ToolCallRequest &req : decision_res->tool_calls)
            {
              if (!early_call_ids.contains (req.call_id))
                {
                  remaining_calls.push_back (req);
                }
            }

          if (!remaining_calls.empty ())
            {
              auto batch_res = scheduler.execute_batch (
                  *agent.tools_, tool_ctx, remaining_calls,
                  agent.config_.tool_execution_mode);
              if (!batch_res)
                {
                  return std::unexpected (batch_res.error ());
                }
              for (ToolCallResult &r : *batch_res)
                {
                  step_tool_results.push_back (std::move (r));
                }
            }

          run_result.tools_executed += step_tool_results.size ();

          if (!decision_res->thought.empty () || decision_res->completed)
            {
              Message assistant_msg;
              assistant_msg.role = MessageRole::Assistant;
              assistant_msg.content = decision_res->completed
                                          ? decision_res->final_answer
                                          : decision_res->thought;
              (void)agent.memory_.short_term ().append (
                  std::move (assistant_msg));
            }

          for (const ToolCallResult &tr : step_tool_results)
            {
              Message obs_msg;
              obs_msg.role = MessageRole::Tool;
              obs_msg.name = tr.tool_name;
              obs_msg.tool_call_id = tr.call_id;
              if (tr.output.has_value ())
                {
                  obs_msg.content = *tr.output;
                }
              else
                {
                  saw_tool_error = true;
                  last_tool_error = tr.output.error ();
                  obs_msg.content = "ERROR: ";
                  obs_msg.content.append (tr.output.error ().message ());
                }
              (void)agent.memory_.short_term ().append (std::move (obs_msg));
            }

          // Enqueue incremental binary checkpoint asynchronously without
          // blocking the agent loop.
          if (agent.checkpointer_ != nullptr)
            {
              const uint64_t prev_seq = agent.last_checkpoint_seq_.load (
                  std::memory_order_relaxed);
              std::vector<Message> delta
                  = agent.memory_.short_term ().delta_since (prev_seq);
              if (!delta.empty ())
                {
                  const uint64_t high_seq = delta.back ().sequence_id;
                  agent.last_checkpoint_seq_.store (high_seq,
                                                    std::memory_order_relaxed);
                  CheckpointDelta cpk_delta{ run_id, prev_seq, high_seq,
                                             std::move (delta) };
                  const auto cpk_res = agent.checkpointer_->enqueue_delta (
                      std::move (cpk_delta));
                  if (!cpk_res)
                    {
                      return std::unexpected (cpk_res.error ());
                    }
                }
            }

          if (decision_res->completed)
            {
              run_result.final_answer = std::move (decision_res->final_answer);
              const uint64_t end_ns = monotonic_time_ns ();
              run_result.total_latency_ns
                  = (end_ns >= run_start_ns) ? (end_ns - run_start_ns) : 0U;
              return run_result;
            }
        }

      if (saw_tool_error)
        {
          return std::unexpected (last_tool_error);
        }

      return std::unexpected (AgentError::planning (
          PlanningErrorCode::MaxStepsExceeded, agent.config_.max_steps,
          "Agent exceeded maximum configured planning steps without reaching "
          "a final answer"));
    }
  catch (...)
    {
      return std::unexpected (
          AgentError::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                                 "Out of memory during AgentRuntime::run"));
    }
}

/* Submits an asynchronous agent run to the work-stealing pool and returns
   a lock-free AsyncRunHandle.  */
auto
AgentRuntime::run_async (Agent &agent, std::string user_prompt) noexcept
    -> Result<AsyncRunHandle>
{
  try
    {
      auto state = std::make_shared<AsyncRunHandle::SharedState> ();
      pool_.submit (
          [this, &agent, prompt = std::move (user_prompt), state] () noexcept {
            Result<AgentRunResult> res = run (agent, prompt);
            (void)state->completion_queue.try_push (std::move (res));
            state->ready_flag.store (1U, std::memory_order_release);
          });
      return AsyncRunHandle (std::move (state), &pool_);
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Failed to allocate state for AgentRuntime::run_async"));
    }
}

/* Returns mutable reference to the runtime's WorkStealingPool.  */
auto
AgentRuntime::pool () noexcept -> WorkStealingPool &
{
  return pool_;
}

/* Returns the worker thread count of the runtime.  */
auto
AgentRuntime::thread_count () const noexcept -> std::size_t
{
  return pool_.thread_count ();
}

} /* namespace sutra::inline v1 */
