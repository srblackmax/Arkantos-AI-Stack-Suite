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

#ifndef SUTRA_RUNTIME_HPP
#define SUTRA_RUNTIME_HPP

#include <sutra/checkpoint.hpp>
#include <sutra/error.hpp>
#include <sutra/memory.hpp>
#include <sutra/planner.hpp>
#include <sutra/scheduler.hpp>
#include <sutra/tool.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace sutra::inline v1
{

/**
 * @brief Default maximum planning steps per agent run.
 */
inline constexpr std::size_t kDefaultMaxSteps = 8U;

/**
 * @brief Default maximum parallel tools dispatched per planning step.
 */
inline constexpr std::size_t kDefaultMaxParallelTools = 8U;

/**
 * @brief Configuration parameters for an Agent instance.
 * @note Thread-safety: Value type.
 */
struct AgentConfig
{
  std::size_t max_steps = kDefaultMaxSteps;
  std::size_t short_term_capacity = kDefaultShortTermCapacity;
  std::size_t semantic_dimension = kDefaultSemanticDimension;
  std::size_t max_parallel_tools = kDefaultMaxParallelTools;
  std::size_t scratch_arena_bytes = kDefaultArenaCapacityBytes;
  ToolExecutionMode tool_execution_mode = ToolExecutionMode::Parallel;
  bool enable_checkpointing = false;
  bool reject_concurrent_same_agent = false;
  std::string checkpoint_path;
};

/**
 * @brief Telemetry and final output returned by a completed agent run.
 * @note Thread-safety: Value type.
 */
struct AgentRunResult
{
  std::string final_answer;
  std::size_t steps_taken = 0U;
  std::size_t tools_executed = 0U;
  uint64_t total_latency_ns = 0U;
  uint64_t overlapped_tokens = 0U;
  uint64_t generated_tokens = 0U;
};

/**
 * @brief Stateful agent binding a MemorySystem, ToolRegistry, LlmClient,
 *        and Planner strategy.
 * @pre llm, planner, and tools must be non-null.
 * @post Provides deterministic concurrency guarantees via reader-writer memory
 *       locks and per-turn serialization or explicit ConcurrencyViolation.
 * @note Thread-safety: Safe for concurrent calls across threads.
 */
class Agent
{
private:
  struct PrivateTag
  {
    explicit PrivateTag () = default;
  };

public:
  Agent (PrivateTag tag, AgentConfig config, MemorySystem memory,
         std::shared_ptr<ToolRegistry> tools, std::shared_ptr<LlmClient> llm,
         std::shared_ptr<Planner> planner,
         std::unique_ptr<CheckpointManager> checkpointer);

  /**
   * @brief Creates a configured Agent instance.
   * @param config Agent configuration parameters.
   * @param tools Shared ToolRegistry.
   * @param llm Shared LlmClient backend.
   * @param planner Shared Planner strategy (defaults to ReActPlanner if null).
   * @pre config.max_steps > 0, tools != nullptr, and llm != nullptr.
   * @post Agent is initialized with pre-allocated memory tiers.
   * @returns Result<std::unique_ptr<Agent>> or AgentError.
   */
  [[nodiscard]] static auto
  create (AgentConfig config, std::shared_ptr<ToolRegistry> tools,
          std::shared_ptr<LlmClient> llm,
          std::shared_ptr<Planner> planner
          = nullptr) noexcept -> Result<std::unique_ptr<Agent>>;

  Agent (const Agent &) = delete;
  auto operator= (const Agent &) -> Agent & = delete;
  Agent (Agent &&) = delete;
  auto operator= (Agent &&) -> Agent & = delete;
  ~Agent () noexcept = default;

  /**
   * @brief Accesses the agent's Three-Tier MemorySystem.
   */
  [[nodiscard]] auto memory () noexcept -> MemorySystem &;
  [[nodiscard]] auto memory () const noexcept -> const MemorySystem &;

  /**
   * @brief Accesses the agent's configuration.
   */
  [[nodiscard]] auto config () const noexcept -> const AgentConfig &;

  /**
   * @brief Accesses the bound ToolRegistry.
   */
  [[nodiscard]] auto tools () const noexcept -> const ToolRegistry &;

  /**
   * @brief Accesses the bound LlmClient.
   */
  [[nodiscard]] auto llm () noexcept -> LlmClient &;

  /**
   * @brief Accesses the bound Planner.
   */
  [[nodiscard]] auto planner () noexcept -> Planner &;

  /**
   * @brief Flushes any pending asynchronous checkpoints to disk.
   * @returns Result<void> or FrameworkError on I/O failure.
   */
  [[nodiscard]] auto flush_checkpoints () noexcept -> Result<void>;

private:
  friend class AgentRuntime;

  AgentConfig config_;
  MemorySystem memory_;
  std::shared_ptr<ToolRegistry> tools_;
  std::shared_ptr<LlmClient> llm_;
  std::shared_ptr<Planner> planner_;
  std::unique_ptr<CheckpointManager> checkpointer_;
  std::atomic<uint32_t> active_callers_{ 0U };
  std::atomic<uint64_t> last_checkpoint_seq_{ 0U };
  std::atomic<uint64_t> run_counter_{ 0U };
  std::mutex turn_mtx_;
};

/**
 * @brief Handle to an asynchronous agent run executing on AgentRuntime.
 * @note Thread-safety: Move-only handle; wait() is safe to call once.
 */
class AsyncRunHandle
{
public:
  struct SharedState
  {
    LockFreeQueue<Result<AgentRunResult>> completion_queue{ 2U };
    std::atomic<uint32_t> ready_flag{ 0U };
  };

  AsyncRunHandle () noexcept = default;
  explicit AsyncRunHandle (std::shared_ptr<SharedState> state,
                           WorkStealingPool *pool) noexcept;

  AsyncRunHandle (const AsyncRunHandle &) = delete;
  auto operator= (const AsyncRunHandle &) -> AsyncRunHandle & = delete;
  AsyncRunHandle (AsyncRunHandle &&) noexcept = default;
  auto operator= (AsyncRunHandle &&) noexcept -> AsyncRunHandle & = default;
  ~AsyncRunHandle () noexcept = default;

  /**
   * @brief Checks whether the asynchronous run has completed without blocking.
   * @returns True if result is ready in the lock-free completion queue.
   */
  [[nodiscard]] auto is_ready () const noexcept -> bool;

  /**
   * @brief Waits for completion and retrieves the AgentRunResult.
   * @pre Handle must be valid.
   * @post Consumes the completion result from the lock-free queue.
   * @returns Result<AgentRunResult> or AgentError.
   */
  [[nodiscard]] auto wait () noexcept -> Result<AgentRunResult>;

private:
  std::shared_ptr<SharedState> state_;
  WorkStealingPool *pool_ = nullptr;
};

/**
 * @brief High-performance execution runtime managing worker threads,
 *        work-stealing queues, and lock-free completion queues.
 * @pre num_threads >= 1 (0 selects hardware concurrency).
 * @post Supports N concurrent agent runs and overlaps LLM inference with
 * tools.
 * @note Thread-safety: Fully thread-safe across all public methods.
 */
class AgentRuntime
{
public:
  /**
   * @brief Constructs an AgentRuntime with a work-stealing thread pool.
   * @param num_threads Worker thread count (0 auto-detects CPU threads).
   * @param queue_capacity Per-worker lock-free queue capacity.
   * @pre None.
   * @post Worker pool is running and ready for run() and run_async().
   */
  explicit AgentRuntime (std::size_t num_threads = 0U,
                         std::size_t queue_capacity
                         = kDefaultWorkerQueueCapacity);

  AgentRuntime (const AgentRuntime &) = delete;
  auto operator= (const AgentRuntime &) -> AgentRuntime & = delete;
  AgentRuntime (AgentRuntime &&) = delete;
  auto operator= (AgentRuntime &&) -> AgentRuntime & = delete;
  ~AgentRuntime () noexcept = default;

  /**
   * @brief Executes a synchronous agent orchestration loop for user_prompt.
   * @param agent Target Agent instance.
   * @param user_prompt Input query from the user.
   * @pre !user_prompt.empty().
   * @post Updates agent short-term memory, dispatches tools in
   * parallel/overlap with LLM token streaming, and enqueues incremental
   * checkpoints.
   * @returns Result<AgentRunResult> or AgentError.
   * @note Thread-safety: Safe for N concurrent calls across agents.
   */
  [[nodiscard]] auto run (Agent &agent, std::string_view user_prompt) noexcept
      -> Result<AgentRunResult>;

  /**
   * @brief Dispatches an asynchronous agent run onto the work-stealing pool.
   * @param agent Target Agent instance (must outlive the AsyncRunHandle).
   * @param user_prompt Input query from the user.
   * @pre !user_prompt.empty().
   * @post Run executes in background and posts result to completion queue.
   * @returns Result<AsyncRunHandle> or AgentError.
   * @note Thread-safety: Lock-free submission to WorkStealingPool.
   */
  [[nodiscard]] auto run_async (Agent &agent, std::string user_prompt) noexcept
      -> Result<AsyncRunHandle>;

  /**
   * @brief Accesses the underlying WorkStealingPool.
   * @returns Reference to WorkStealingPool.
   */
  [[nodiscard]] auto pool () noexcept -> WorkStealingPool &;

  /**
   * @brief Returns the number of worker threads in the runtime.
   * @returns Worker count.
   */
  [[nodiscard]] auto thread_count () const noexcept -> std::size_t;

private:
  WorkStealingPool pool_;
};

} /* namespace sutra::inline v1 */

#endif /* SUTRA_RUNTIME_HPP */
