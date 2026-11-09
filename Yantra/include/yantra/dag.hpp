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

#ifndef YANTRA_DAG_HPP
#define YANTRA_DAG_HPP

#include <yantra/error.hpp>
#include <yantra/observability.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <memory_resource>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace yantra::inline v1
{

/**
 * @brief Hardware resource requirements declared by a DAG Task node.
 */
struct ResourceSpec
{
  uint32_t cpu_cores = 1U;
  uint64_t gpu_vram_bytes = 0U;
};

/**
 * @brief Exponential backoff retry policy for transient task failures.
 */
struct RetryPolicy
{
  uint32_t max_retries = 0U;
  uint64_t initial_backoff_ns = 1000000ULL; /* 1 ms */
  double backoff_multiplier = 2.0;
};

/**
 * @brief Terminal or active status of a DAG Task node.
 */
enum class TaskStatus : uint8_t
{
  Pending = 0,
  Running = 1,
  Completed = 2,
  Failed = 3,
  Skipped = 4,
  TimedOut = 5
};

/**
 * @brief Execution telemetry and output artifact references produced by a
 * single DAG task execution.
 */
struct TaskResult
{
  std::string task_id;
  TaskStatus status = TaskStatus::Pending;
  uint64_t start_timestamp_ns = 0U;
  uint64_t end_timestamp_ns = 0U;
  uint64_t duration_ns = 0U;
  uint32_t attempts = 0U;
  ResourceSpec resource_usage;
  std::vector<std::string> output_artifact_hashes;
  std::string error_message;
};

/**
 * @brief Pre-allocated lock-free task execution metadata record indexed by
 * integer task slot.
 */
struct TaskMetadata
{
  std::atomic<uint32_t> pending_dependencies{ 0U };
  std::atomic<TaskStatus> status{ TaskStatus::Pending };
  TaskResult result;
};

/**
 * @brief Flat pre-indexed task result container avoiding per-execute
 * hash-table node allocations on the DAG execution hot path.
 */
class TaskResultMap
{
public:
  using value_type = std::pair<std::string, TaskResult>;
  using container_type = std::vector<value_type>;
  using iterator = container_type::iterator;
  using const_iterator = container_type::const_iterator;
  using index_map_type = std::unordered_map<std::string, std::size_t>;

  TaskResultMap () noexcept = default;

  TaskResultMap (const index_map_type *idx_map,
                 container_type *borrowed_entries,
                 std::atomic<bool> *release_flag) noexcept
      : id_to_idx_ (idx_map),
        entries_ (borrowed_entries),
        release_flag_ (release_flag)
  {
  }

  TaskResultMap (const index_map_type *idx_map,
                 std::unique_ptr<container_type> owned_entries) noexcept
      : id_to_idx_ (idx_map),
        entries_ (owned_entries.get ()),
        owned_entries_ (std::move (owned_entries))
  {
  }

  ~TaskResultMap () noexcept
  {
    if (release_flag_ != nullptr)
      {
        release_flag_->store (false, std::memory_order_release);
      }
  }

  TaskResultMap (TaskResultMap &&other) noexcept
      : id_to_idx_ (other.id_to_idx_),
        entries_ (other.entries_),
        owned_entries_ (std::move (other.owned_entries_)),
        release_flag_ (other.release_flag_)
  {
    other.id_to_idx_ = nullptr;
    other.entries_ = nullptr;
    other.release_flag_ = nullptr;
  }

  auto
  operator= (TaskResultMap &&other) noexcept -> TaskResultMap &
  {
    if (this != &other)
      {
        if (release_flag_ != nullptr)
          {
            release_flag_->store (false, std::memory_order_release);
          }
        id_to_idx_ = other.id_to_idx_;
        entries_ = other.entries_;
        owned_entries_ = std::move (other.owned_entries_);
        release_flag_ = other.release_flag_;
        other.id_to_idx_ = nullptr;
        other.entries_ = nullptr;
        other.release_flag_ = nullptr;
      }
    return *this;
  }

  TaskResultMap (const TaskResultMap &other) : id_to_idx_ (other.id_to_idx_)
  {
    if (other.entries_ != nullptr)
      {
        owned_entries_ = std::make_unique<container_type> (*other.entries_);
        entries_ = owned_entries_.get ();
      }
  }

  auto
  operator= (const TaskResultMap &other) -> TaskResultMap &
  {
    if (this != &other)
      {
        if (release_flag_ != nullptr)
          {
            release_flag_->store (false, std::memory_order_release);
            release_flag_ = nullptr;
          }
        id_to_idx_ = other.id_to_idx_;
        if (other.entries_ != nullptr)
          {
            owned_entries_
                = std::make_unique<container_type> (*other.entries_);
            entries_ = owned_entries_.get ();
          }
        else
          {
            owned_entries_.reset ();
            entries_ = nullptr;
          }
      }
    return *this;
  }

  [[nodiscard]] auto
  data () noexcept -> value_type *
  {
    ensure_entries ();
    return entries_->data ();
  }

  [[nodiscard]] auto
  size () const noexcept -> std::size_t
  {
    return entries_ ? entries_->size () : 0U;
  }

  [[nodiscard]] auto
  empty () const noexcept -> bool
  {
    return !entries_ || entries_->empty ();
  }

  [[nodiscard]] auto
  begin () noexcept -> iterator
  {
    ensure_entries ();
    return entries_->begin ();
  }

  [[nodiscard]] auto
  end () noexcept -> iterator
  {
    ensure_entries ();
    return entries_->end ();
  }

  [[nodiscard]] auto
  begin () const noexcept -> const_iterator
  {
    return entries_ ? entries_->cbegin () : empty_vec ().cbegin ();
  }

  [[nodiscard]] auto
  end () const noexcept -> const_iterator
  {
    return entries_ ? entries_->cend () : empty_vec ().cend ();
  }

  [[nodiscard]] auto
  find (const std::string &key) const noexcept -> const_iterator
  {
    if (!entries_)
      {
        return empty_vec ().cend ();
      }
    if (id_to_idx_)
      {
        const auto it = id_to_idx_->find (key);
        if (it != id_to_idx_->end () && it->second < entries_->size ())
          {
            return entries_->cbegin ()
                   + static_cast<std::ptrdiff_t> (it->second);
          }
        return entries_->cend ();
      }
    for (auto it = entries_->cbegin (); it != entries_->cend (); ++it)
      {
        if (it->first == key)
          {
            return it;
          }
      }
    return entries_->cend ();
  }

  [[nodiscard]] auto
  find (const std::string &key) noexcept -> iterator
  {
    ensure_entries ();
    if (id_to_idx_)
      {
        const auto it = id_to_idx_->find (key);
        if (it != id_to_idx_->end () && it->second < entries_->size ())
          {
            return entries_->begin ()
                   + static_cast<std::ptrdiff_t> (it->second);
          }
        return entries_->end ();
      }
    for (auto it = entries_->begin (); it != entries_->end (); ++it)
      {
        if (it->first == key)
          {
            return it;
          }
      }
    return entries_->end ();
  }

  [[nodiscard]] auto
  contains (const std::string &key) const noexcept -> bool
  {
    return find (key) != end ();
  }

  [[nodiscard]] auto
  at (const std::string &key) const -> const TaskResult &
  {
    const auto it = find (key);
    if (it == end ())
      {
        throw std::out_of_range ("TaskResultMap::at: key not found");
      }
    return it->second;
  }

  [[nodiscard]] auto
  at (const std::string &key) -> TaskResult &
  {
    const auto it = find (key);
    if (it == end ())
      {
        throw std::out_of_range ("TaskResultMap::at: key not found");
      }
    return it->second;
  }

  auto
  operator[] (const std::string &key) -> TaskResult &
  {
    const auto it = find (key);
    if (it != end ())
      {
        return it->second;
      }
    ensure_entries ();
    entries_->emplace_back (key, TaskResult{});
    return entries_->back ().second;
  }

  void
  emplace (std::string key, TaskResult val)
  {
    const auto it = find (key);
    if (it != end ())
      {
        it->second = std::move (val);
        return;
      }
    ensure_entries ();
    entries_->emplace_back (std::move (key), std::move (val));
  }

private:
  void
  ensure_entries ()
  {
    if (entries_ == nullptr)
      {
        owned_entries_ = std::make_unique<container_type> ();
        entries_ = owned_entries_.get ();
      }
  }

  [[nodiscard]] static auto
  empty_vec () noexcept -> const container_type &
  {
    static const container_type kEmpty{};
    return kEmpty;
  }

  const index_map_type *id_to_idx_ = nullptr;
  container_type *entries_ = nullptr;
  std::unique_ptr<container_type> owned_entries_;
  std::atomic<bool> *release_flag_ = nullptr;
};

/**
 * @brief Per-task execution context providing a std::pmr scratch arena and
 * read-only access to upstream task results.
 */
struct TaskContext
{
  std::string_view task_id;
  uint32_t attempt = 1U;
  std::pmr::memory_resource *scratch_arena = nullptr;
  const TaskResultMap *upstream_results = nullptr;
};

using TaskRunFn
    = std::function<Result<std::vector<std::string>> (TaskContext &)>;
using TaskConditionFn = std::function<bool (const TaskContext &)>;

/**
 * @brief Declarative specification of a single node in a Yantra workflow DAG.
 */
struct Task
{
  std::string id;
  std::vector<std::string> dependencies;
  ResourceSpec resources;
  RetryPolicy retry_policy;
  uint64_t timeout_ns = 0U; /* 0 = no timeout */
  TaskConditionFn condition;
  TaskRunFn run;
};

/**
 * @brief Data-driven DAG pipeline definition supporting both C++ builder API
 * and JSON serialization/deserialization for hot-swapping and multi-tenancy.
 */
class PipelineDefinition
{
public:
  PipelineDefinition () noexcept = default;

  /**
   * @brief Adds a task node to the pipeline definition.
   * @param task Task specification to add.
   * @returns Result<void> or DagErrorCode::InvalidDefinition if duplicate ID.
   */
  [[nodiscard]] auto add_task (Task task) noexcept -> Result<void>;

  /**
   * @brief Parses a PipelineDefinition from a declarative JSON string.
   * @param json_text JSON string specifying tasks, dependencies, resources.
   * @param task_registry Lookup table mapping task_type/id to TaskRunFn.
   * @returns Result<PipelineDefinition> or DagErrorCode::InvalidDefinition.
   */
  [[nodiscard]] static auto from_json (
      std::string_view json_text,
      const std::unordered_map<std::string, TaskRunFn> &task_registry
      = {}) noexcept -> Result<PipelineDefinition>;

  /**
   * @brief Serializes the pipeline topology and resource specs to JSON.
   * @returns JSON representation string.
   */
  [[nodiscard]] auto to_json () const noexcept -> std::string;

  /**
   * @brief Accesses the list of tasks in the pipeline.
   */
  [[nodiscard]] auto tasks () const noexcept -> const std::vector<Task> &;

private:
  std::vector<Task> tasks_;
  mutable const void *compiled_cache_ = nullptr;
  friend class DagScheduler;
};

/**
 * @brief Complete execution report for a workflow DAG run.
 */
struct DagExecutionReport
{
  TaskResultMap results;
  std::span<const std::string> topological_order;
  uint64_t scheduling_latency_ns = 0U;
  uint64_t total_wall_ns = 0U;
  bool all_succeeded = true;
};

/**
 * @brief Compact node timing record for low-overhead indexed DAG execution.
 */
struct IndexedNodeTiming
{
  uint64_t start_ns = 0U;
  uint64_t end_ns = 0U;
  uint32_t executed_count = 0U;
};

/**
 * @brief Work-stealing DAG workflow orchestrator with per-worker Chase-Lev
 * deques, topological cycle detection, resource constraint admission control,
 * conditional edges, retries, and timeouts.
 * @note Thread-safety: Safe for concurrent execute() invocations across
 * multiple threads (multi-tenant execution).
 */
class DagScheduler
{
public:
  /**
   * @brief Constructs a work-stealing DAG scheduler pool.
   * @param num_workers Number of worker threads (0 = hardware concurrency).
   * @param cluster_capacity Total CPU cores and GPU VRAM available.
   */
  explicit DagScheduler (
      std::size_t num_workers = 8U,
      ResourceSpec cluster_capacity
      = { 20U, 8ULL * 1024ULL * 1024ULL * 1024ULL }) noexcept;

  DagScheduler (const DagScheduler &) = delete;
  auto operator= (const DagScheduler &) -> DagScheduler & = delete;
  DagScheduler (DagScheduler &&other) noexcept;
  auto operator= (DagScheduler &&other) noexcept -> DagScheduler &;
  ~DagScheduler () noexcept;

  /**
   * @brief Validates a pipeline for cycles, missing dependencies, and
   * unsatisfiable resource requests without executing tasks.
   * @param pipeline Pipeline definition to validate.
   * @returns Result<std::vector<std::string>> containing a valid topological
   * ordering, or DagError (CycleDetected, MissingDependency,
   * ResourceDeadlock).
   */
  [[nodiscard]] auto
  validate (const PipelineDefinition &pipeline) const noexcept
      -> Result<std::vector<std::string>>;

  /**
   * @brief Schedules and executes all tasks in topological order across the
   * work-stealing thread pool, enforcing resource constraints, retries,
   * timeouts, and conditional edges.
   * @param pipeline Declarative pipeline definition to execute.
   * @returns Result<DagExecutionReport> or DagError on structural error.
   */
  [[nodiscard]] auto
  execute (const PipelineDefinition &pipeline) noexcept
      -> Result<DagExecutionReport>;

  /**
   * @brief Executes an indexed static DAG on the work-stealing pool with
   * atomic in-degree resolution and records per-node monotonic start/end
   * timestamps for topological invariant verification.
   * @param num_nodes Total number of nodes (0 .. num_nodes - 1).
   * @param children_csr Flattened adjacency list of child indices per node.
   * @param row_offsets CSR row offsets of size num_nodes + 1.
   * @param initial_indegree Initial parent count for each node.
   * @param node_fn Function pointer or callback invoked as node_fn(node_id).
   * @param out_timings Output span of size num_nodes receiving timestamps.
   * @returns Result<uint64_t> total execution time in nanoseconds.
   */
  [[nodiscard]] auto execute_indexed_dag (
      std::size_t num_nodes, std::span<const uint32_t> children_csr,
      std::span<const uint32_t> row_offsets,
      std::span<const uint32_t> initial_indegree,
      void (*node_fn) (std::size_t),
      std::span<IndexedNodeTiming> out_timings) noexcept -> Result<uint64_t>;

  /**
   * @brief Queries the number of worker threads in the work-stealing pool.
   */
  [[nodiscard]] auto worker_count () const noexcept -> std::size_t;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

using DagExecutor = DagScheduler;

namespace dag
{
using yantra::v1::DagExecutionReport;
using yantra::v1::DagExecutor;
using yantra::v1::DagScheduler;
using yantra::v1::IndexedNodeTiming;
using yantra::v1::PipelineDefinition;
using yantra::v1::ResourceSpec;
using yantra::v1::RetryPolicy;
using yantra::v1::Task;
using yantra::v1::TaskConditionFn;
using yantra::v1::TaskContext;
using yantra::v1::TaskResult;
using yantra::v1::TaskRunFn;
using yantra::v1::TaskStatus;
} /* namespace dag */

} /* namespace yantra::inline v1 */

#endif /* YANTRA_DAG_HPP */
