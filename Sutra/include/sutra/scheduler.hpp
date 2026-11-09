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

#ifndef SUTRA_SCHEDULER_HPP
#define SUTRA_SCHEDULER_HPP

#include <sutra/error.hpp>
#include <sutra/tool.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sutra::inline v1
{

inline constexpr std::size_t kDefaultLockFreeQueueCapacity = 256U;
inline constexpr std::size_t kDefaultLockFreeQueueMask = 255U;
inline constexpr std::size_t kCacheLineAlignBytes = 64U;
inline constexpr std::size_t kDefaultWorkerQueueCapacity = 512U;
inline constexpr std::size_t kDefaultToolConcurrency = 8U;

/**
 * @brief Bounded lock-free MPMC/MPSC ring queue using atomic sequence numbers.
 * @tparam T Move-assignable and default-constructible element type.
 * @pre Capacity is rounded up to a power of two >= 2.
 * @post Zero std::mutex usage; safe for concurrent producers and consumers.
 * @note Thread-safety: Lock-free via acquire/release atomic operations.
 */
template <typename T> class LockFreeQueue
{
public:
  /**
   * @brief Constructs a lock-free ring buffer of at least min_capacity slots.
   * @param min_capacity Minimum number of elements (>= 2).
   * @pre min_capacity >= 2.
   * @post All slot sequence numbers are initialized to their index.
   */
  explicit LockFreeQueue (std::size_t min_capacity
                          = kDefaultLockFreeQueueCapacity)
      : capacity_ (next_power_of_two (min_capacity)), mask_ (capacity_ - 1U),
        cells_ (capacity_)
  {
    for (std::size_t i = 0U; i < capacity_; ++i)
      {
        cells_.at (i).sequence.store (i, std::memory_order_relaxed);
      }
  }

  LockFreeQueue (const LockFreeQueue &) = delete;
  auto operator= (const LockFreeQueue &) -> LockFreeQueue & = delete;
  LockFreeQueue (LockFreeQueue &&) = delete;
  auto operator= (LockFreeQueue &&) -> LockFreeQueue & = delete;
  ~LockFreeQueue () noexcept = default;

  /**
   * @brief Attempts to enqueue an item without blocking.
   * @param item Value to move into the queue.
   * @pre None.
   * @post Returns true if enqueued, false if the ring is currently full.
   * @returns True on success, false if full.
   * @note Thread-safety: Lock-free CAS loop with acquire/release ordering.
   */
  [[nodiscard]] auto
  try_push (T item) noexcept -> bool
  {
    Cell *cell = nullptr;
    std::size_t pos = enqueue_pos_.load (std::memory_order_relaxed);
    for (;;)
      {
        cell = &cells_.at (pos & mask_);
        const std::size_t seq
            = cell->sequence.load (std::memory_order_acquire);
        const auto diff = static_cast<std::ptrdiff_t> (seq)
                          - static_cast<std::ptrdiff_t> (pos);
        if (diff == 0)
          {
            if (enqueue_pos_.compare_exchange_weak (pos, pos + 1U,
                                                    std::memory_order_relaxed))
              {
                break;
              }
          }
        else if (diff < 0)
          {
            return false;
          }
        else
          {
            pos = enqueue_pos_.load (std::memory_order_relaxed);
          }
      }
    cell->data = std::move (item);
    cell->sequence.store (pos + 1U, std::memory_order_release);
    return true;
  }

  /**
   * @brief Attempts to dequeue an item without blocking.
   * @param out Destination receiving the dequeued value.
   * @pre None.
   * @post Returns true and populates out if non-empty, false if empty.
   * @returns True on success, false if empty.
   * @note Thread-safety: Lock-free CAS loop with acquire/release ordering.
   */
  [[nodiscard]] auto
  try_pop (T &out) noexcept -> bool
  {
    Cell *cell = nullptr;
    std::size_t pos = dequeue_pos_.load (std::memory_order_relaxed);
    for (;;)
      {
        cell = &cells_.at (pos & mask_);
        const std::size_t seq
            = cell->sequence.load (std::memory_order_acquire);
        const auto diff = static_cast<std::ptrdiff_t> (seq)
                          - static_cast<std::ptrdiff_t> (pos + 1U);
        if (diff == 0)
          {
            if (dequeue_pos_.compare_exchange_weak (pos, pos + 1U,
                                                    std::memory_order_relaxed))
              {
                break;
              }
          }
        else if (diff < 0)
          {
            return false;
          }
        else
          {
            pos = dequeue_pos_.load (std::memory_order_relaxed);
          }
      }
    out = std::move (cell->data);
    cell->sequence.store (pos + mask_ + 1U, std::memory_order_release);
    return true;
  }

  /**
   * @brief Returns the power-of-two capacity of the queue.
   * @returns Slot capacity.
   */
  [[nodiscard]] auto
  capacity () const noexcept -> std::size_t
  {
    return capacity_;
  }

private:
  struct Cell
  {
    std::atomic<std::size_t> sequence{ 0U };
    T data{};
  };

  [[nodiscard]] static constexpr auto
  next_power_of_two (std::size_t v) noexcept -> std::size_t
  {
    std::size_t p = 2U;
    while (p < v)
      {
        p <<= 1U;
      }
    return p;
  }

  std::size_t capacity_ = kDefaultLockFreeQueueCapacity;
  std::size_t mask_ = kDefaultLockFreeQueueMask;
  std::vector<Cell> cells_;
  alignas (kCacheLineAlignBytes) std::atomic<std::size_t> enqueue_pos_{ 0U };
  alignas (kCacheLineAlignBytes) std::atomic<std::size_t> dequeue_pos_{ 0U };
};

/**
 * @brief Lock-free work-stealing thread pool for parallel tool execution.
 * @pre num_threads >= 1.
 * @post All worker threads join cleanly via RAII on destruction.
 * @note Thread-safety: Hot-path task submission and stealing use zero mutexes.
 */
class WorkStealingPool
{
public:
  using TaskFn = std::function<void ()>;

  /**
   * @brief Constructs a work-stealing pool with num_threads workers.
   * @param num_threads Number of worker threads (0 selects hardware threads).
   * @param queue_capacity Per-worker lock-free queue capacity.
   * @pre None.
   * @post Worker threads are spawned and ready to execute tasks.
   */
  explicit WorkStealingPool (std::size_t num_threads = 0U,
                             std::size_t queue_capacity
                             = kDefaultWorkerQueueCapacity);

  WorkStealingPool (const WorkStealingPool &) = delete;
  auto operator= (const WorkStealingPool &) -> WorkStealingPool & = delete;
  WorkStealingPool (WorkStealingPool &&) = delete;
  auto operator= (WorkStealingPool &&) -> WorkStealingPool & = delete;

  /**
   * @brief Signals shutdown and joins all worker threads.
   */
  ~WorkStealingPool () noexcept;

  /**
   * @brief Submits a task to a worker queue in round-robin order.
   * @param task Callable to execute asynchronously.
   * @pre task is valid.
   * @post Task will be executed by the target worker or stolen by an idle
   * peer.
   * @note Thread-safety: Lock-free atomic submission.
   */
  void submit (TaskFn task) noexcept;

  /**
   * @brief Executes a single pending task on the calling thread if available.
   * @pre None.
   * @post Helps drain worker queues during cooperative waits.
   * @returns True if a task was stolen and executed, false otherwise.
   */
  auto try_run_one () noexcept -> bool;

  /**
   * @brief Returns the number of worker threads in the pool.
   * @returns Worker count.
   */
  [[nodiscard]] auto thread_count () const noexcept -> std::size_t;

private:
  void worker_loop (std::size_t worker_id) noexcept;

  std::size_t num_threads_ = 1U;
  std::vector<std::unique_ptr<LockFreeQueue<TaskFn>>> queues_;
  std::atomic<std::size_t> submit_cursor_{ 0U };
  std::atomic<uint32_t> work_signal_{ 0U };
  std::atomic<bool> stop_{ false };
  std::vector<std::jthread> workers_;
};

/**
 * @brief Controls whether independent tools run concurrently or sequentially.
 * @note Thread-safety: Immutable enum type.
 */
enum class ToolExecutionMode : uint8_t
{
  Parallel = 0,
  Sequential = 1
};

/**
 * @brief Represents a single tool call planned by the LLM or workflow.
 * @note Thread-safety: Value type.
 */
struct ToolCallRequest
{
  std::string call_id;
  std::string tool_name;
  std::string arguments_json = "{}";
  std::vector<std::string> depends_on_call_ids;
  uint32_t timeout_ms = 0U;
};

/**
 * @brief Represents the completed outcome of a tool invocation.
 * @note Thread-safety: Value type.
 */
struct ToolCallResult
{
  std::string call_id;
  std::string tool_name;
  Result<std::string> output{ std::string{} };
  uint64_t duration_ns = 0U;
};

/**
 * @brief Dependency-analyzing parallel tool scheduler with concurrency limits.
 * @pre max_concurrency >= 1.
 * @post Defaults to parallel execution of independent calls; never copies
 *       conversation history into tool invocations.
 * @note Thread-safety: Safe for concurrent execute_batch() calls across
 * agents.
 */
class ToolScheduler
{
public:
  /**
   * @brief Constructs a ToolScheduler backed by a WorkStealingPool.
   * @param pool Reference to the shared work-stealing thread pool.
   * @param max_concurrency Maximum number of simultaneously executing tools.
   * @pre max_concurrency >= 1.
   * @post Scheduler is ready to analyze and execute tool call batches.
   */
  explicit ToolScheduler (WorkStealingPool &pool,
                          std::size_t max_concurrency
                          = kDefaultToolConcurrency) noexcept;

  /**
   * @brief Analyzes dependencies among tool calls and builds parallel waves.
   * @param calls Span of tool call requests within a planning step.
   * @pre Call IDs must be unique and non-empty.
   * @post Partitioned indices in each wave have no mutual dependencies.
   * @returns Result<std::vector<std::vector<std::size_t>>> or PlanningError.
   * @note Thread-safety: Pure static dependency analyzer; thread-safe.
   */
  [[nodiscard]] static auto
  analyze_dependencies (std::span<const ToolCallRequest> calls) noexcept
      -> Result<std::vector<std::vector<std::size_t>>>;

  /**
   * @brief Executes a batch of tool calls, defaulting to parallel waves.
   * @param registry ToolRegistry containing registered tools.
   * @param ctx Zero-copy ToolContext borrowing conversation history.
   * @param calls Span of ToolCallRequest items to execute.
   * @param mode Parallel (default) or explicit Sequential opt-in.
   * @pre None.
   * @post Returns results in the exact order of calls; tool failures are
   *       captured inside each ToolCallResult::output without crashing.
   * @returns Result<std::vector<ToolCallResult>> or PlanningError on cycle.
   * @note Thread-safety: Lock-free dispatch and completion tracking.
   */
  [[nodiscard]] auto
  execute_batch (const ToolRegistry &registry, const ToolContext &ctx,
                 std::span<const ToolCallRequest> calls,
                 ToolExecutionMode mode = ToolExecutionMode::Parallel) noexcept
      -> Result<std::vector<ToolCallResult>>;

  /**
   * @brief Returns the configured concurrency limit.
   * @returns Maximum concurrent tools per batch.
   */
  [[nodiscard]] auto max_concurrency () const noexcept -> std::size_t;

private:
  WorkStealingPool *pool_ = nullptr;
  std::size_t max_concurrency_ = kDefaultToolConcurrency;
};

} /* namespace sutra::inline v1 */

#endif /* SUTRA_SCHEDULER_HPP */
