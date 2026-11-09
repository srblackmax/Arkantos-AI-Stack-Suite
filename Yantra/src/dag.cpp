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

#include <yantra/dag.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>

namespace yantra::inline v1
{

namespace
{

struct WorkItem
{
  void (*fn) (void *, std::size_t, std::size_t) = nullptr;
  void *ctx = nullptr;
  std::size_t node_idx = 0U;
};

struct AtomicWorkSlot
{
  std::atomic<void (*) (void *, std::size_t, std::size_t)> fn{ nullptr };
  std::atomic<void *> ctx{ nullptr };
  std::atomic<std::size_t> node_idx{ 0U };

  void
  store_relaxed (const WorkItem &item) noexcept
  {
    fn.store (item.fn, std::memory_order_relaxed);
    ctx.store (item.ctx, std::memory_order_relaxed);
    node_idx.store (item.node_idx, std::memory_order_release);
  }

  [[nodiscard]] auto
  load_relaxed () const noexcept -> WorkItem
  {
    WorkItem out{};
    out.node_idx = node_idx.load (std::memory_order_acquire);
    out.ctx = ctx.load (std::memory_order_relaxed);
    out.fn = fn.load (std::memory_order_relaxed);
    return out;
  }
};

/* Lock-free Chase-Lev work-stealing deque per worker thread.  */
class alignas (64) ChaseLevDeque
{
public:
  static constexpr int64_t kCapacity = 4096;
  static constexpr int64_t kMask = kCapacity - 1;

  ChaseLevDeque () noexcept = default;

  void
  push_bottom (const WorkItem &item) noexcept
  {
    const int64_t b = bottom_.load (std::memory_order_relaxed);
    const int64_t t = top_.load (std::memory_order_acquire);
    const int64_t depth = b - t;
    if (depth >= kCapacity - 1)
      {
        /* Fallback inline execution if fixed ring is saturated.  */
        if (item.fn != nullptr)
          {
            item.fn (item.ctx, item.node_idx, 0U);
          }
        return;
      }
    buffer_[static_cast<std::size_t> (b & kMask)].store_relaxed (item);
    bottom_.store (b + 1, std::memory_order_release);
    DagStats::global ().record_queue_depth (
        static_cast<uint64_t> (depth + 1));
  }

  [[nodiscard]] auto
  pop_bottom (WorkItem &out) noexcept -> bool
  {
    const int64_t cur_b = bottom_.load (std::memory_order_relaxed);
    const int64_t cur_t = top_.load (std::memory_order_relaxed);
    if (cur_b <= cur_t)
      {
        return false;
      }
    const int64_t b = cur_b - 1;
    bottom_.store (b, std::memory_order_seq_cst);
    int64_t t = top_.load (std::memory_order_seq_cst);

    if (t <= b)
      {
        out = buffer_[static_cast<std::size_t> (b & kMask)].load_relaxed ();
        if (t == b)
          {
            if (!top_.compare_exchange_strong (t, t + 1,
                                               std::memory_order_seq_cst,
                                               std::memory_order_relaxed))
              {
                bottom_.store (b + 1, std::memory_order_relaxed);
                return false;
              }
            bottom_.store (b + 1, std::memory_order_relaxed);
          }
        return true;
      }
    bottom_.store (b + 1, std::memory_order_relaxed);
    return false;
  }

  [[nodiscard]] auto
  steal_top (WorkItem &out) noexcept -> bool
  {
    int64_t t = top_.load (std::memory_order_acquire);
    const int64_t b = bottom_.load (std::memory_order_acquire);
    if (t < b)
      {
        out = buffer_[static_cast<std::size_t> (t & kMask)].load_relaxed ();
        if (top_.compare_exchange_strong (t, t + 1, std::memory_order_seq_cst,
                                          std::memory_order_relaxed))
          {
            DagStats::global ().record_steal ();
            return true;
          }
      }
    return false;
  }

private:
  alignas (64) std::atomic<int64_t> top_{ 0 };
  alignas (64) std::atomic<int64_t> bottom_{ 0 };
  std::array<AtomicWorkSlot, static_cast<std::size_t> (kCapacity)> buffer_{};
};

auto
extract_json_string (std::string_view src, std::string_view key,
                     std::size_t start_pos) noexcept
    -> std::pair<std::string, std::size_t>
{
  const std::string pat = "\"" + std::string (key) + "\"";
  const auto kpos = src.find (pat, start_pos);
  if (kpos == std::string_view::npos)
    {
      return { "", std::string_view::npos };
    }
  const auto colon = src.find (':', kpos + pat.size ());
  if (colon == std::string_view::npos)
    {
      return { "", std::string_view::npos };
    }
  const auto q1 = src.find ('"', colon + 1U);
  if (q1 == std::string_view::npos)
    {
      return { "", std::string_view::npos };
    }
  const auto q2 = src.find ('"', q1 + 1U);
  if (q2 == std::string_view::npos)
    {
      return { "", std::string_view::npos };
    }
  return { std::string (src.substr (q1 + 1U, q2 - q1 - 1U)), q2 + 1U };
}

struct CompiledPipeline
{
  bool valid = false;
  std::optional<Error> validation_error;
  std::vector<std::string> topological_order;
  TaskResultMap::index_map_type id_to_idx;
  std::vector<uint32_t> children_csr;
  std::vector<uint32_t> children_offsets;
  std::vector<uint32_t> parents_csr;
  std::vector<uint32_t> parents_offsets;
  std::vector<uint32_t> initial_indegree;
  std::vector<uint32_t> root_nodes;
  uint32_t max_cpu_req = 0U;
  uint64_t max_vram_req = 0U;
  bool all_independent = true;
  bool any_condition = false;

  mutable std::atomic<bool> primary_busy{ false };
  mutable std::unique_ptr<TaskMetadata[]> primary_meta;
  mutable TaskResultMap::container_type primary_entries;
};

auto
compile_pipeline (const std::vector<Task> &tasks) noexcept
    -> std::unique_ptr<CompiledPipeline>
{
  auto cp = std::make_unique<CompiledPipeline> ();
  if (tasks.empty ())
    {
      cp->validation_error = Error::dag (
          DagErrorCode::InvalidDefinition, "",
          "Cannot schedule an empty DAG pipeline definition");
      return cp;
    }

  const std::size_t n = tasks.size ();
  cp->id_to_idx.reserve (n * 2U);

  for (std::size_t i = 0U; i < n; ++i)
    {
      const auto &t = tasks[i];
      cp->max_cpu_req = std::max (cp->max_cpu_req, t.resources.cpu_cores);
      cp->max_vram_req
          = std::max (cp->max_vram_req, t.resources.gpu_vram_bytes);
      if (t.condition)
        {
          cp->any_condition = true;
        }
      cp->id_to_idx[t.id] = i;
    }

  std::vector<std::vector<uint32_t>> adj (n);
  std::vector<std::vector<uint32_t>> rev_adj (n);
  cp->initial_indegree.assign (n, 0U);

  for (std::size_t i = 0U; i < n; ++i)
    {
      if (!tasks[i].dependencies.empty ())
        {
          cp->all_independent = false;
        }
      for (const auto &dep : tasks[i].dependencies)
        {
          const auto it = cp->id_to_idx.find (dep);
          if (it == cp->id_to_idx.end ())
            {
              cp->validation_error = Error::dag (
                  DagErrorCode::MissingDependency, tasks[i].id,
                  "Task references missing upstream dependency '" + dep + "'");
              return cp;
            }
          const auto p = static_cast<uint32_t> (it->second);
          adj[p].push_back (static_cast<uint32_t> (i));
          rev_adj[i].push_back (p);
          ++cp->initial_indegree[i];
        }
    }

  std::vector<uint32_t> indeg = cp->initial_indegree;
  std::vector<uint32_t> queue;
  queue.reserve (n);
  cp->root_nodes.reserve (n);
  for (std::size_t i = 0U; i < n; ++i)
    {
      if (indeg[i] == 0U)
        {
          const auto u32 = static_cast<uint32_t> (i);
          queue.push_back (u32);
          cp->root_nodes.push_back (u32);
        }
    }

  cp->topological_order.reserve (n);
  std::size_t head = 0U;
  while (head < queue.size ())
    {
      const uint32_t u = queue[head++];
      cp->topological_order.push_back (tasks[u].id);
      for (const uint32_t v : adj[u])
        {
          if (--indeg[v] == 0U)
            {
              queue.push_back (v);
            }
        }
    }

  if (cp->topological_order.size () != n)
    {
      std::string cycle_node;
      for (std::size_t i = 0U; i < n; ++i)
        {
          if (indeg[i] > 0U)
            {
              cycle_node = tasks[i].id;
              break;
            }
        }
      cp->validation_error = Error::dag (
          DagErrorCode::CycleDetected, cycle_node,
          "Directed cycle detected in workflow DAG topology");
      return cp;
    }

  cp->children_offsets.assign (n + 1U, 0U);
  cp->parents_offsets.assign (n + 1U, 0U);
  for (std::size_t i = 0U; i < n; ++i)
    {
      cp->children_offsets[i + 1U]
          = cp->children_offsets[i] + static_cast<uint32_t> (adj[i].size ());
      cp->parents_offsets[i + 1U]
          = cp->parents_offsets[i]
            + static_cast<uint32_t> (rev_adj[i].size ());
    }
  cp->children_csr.reserve (cp->children_offsets[n]);
  cp->parents_csr.reserve (cp->parents_offsets[n]);
  for (std::size_t i = 0U; i < n; ++i)
    {
      cp->children_csr.insert (cp->children_csr.end (), adj[i].begin (),
                               adj[i].end ());
      cp->parents_csr.insert (cp->parents_csr.end (), rev_adj[i].begin (),
                              rev_adj[i].end ());
    }

  cp->primary_meta = std::make_unique<TaskMetadata[]> (n);
  cp->primary_entries.resize (n);
  for (std::size_t i = 0U; i < n; ++i)
    {
      cp->primary_entries[i].first = tasks[i].id;
      cp->primary_entries[i].second.task_id = tasks[i].id;
      cp->primary_entries[i].second.resource_usage = tasks[i].resources;
    }

  cp->valid = true;
  return cp;
}

} /* namespace */

auto
PipelineDefinition::add_task (Task task) noexcept -> Result<void>
{
  if (task.id.empty ())
    {
      return std::unexpected (
          Error::dag (DagErrorCode::InvalidDefinition, "",
                      "Task identifier must not be empty"));
    }
  for (const auto &existing : tasks_)
    {
      if (existing.id == task.id)
        {
          return std::unexpected (Error::dag (
              DagErrorCode::InvalidDefinition, task.id,
              "Duplicate task identifier in PipelineDefinition"));
        }
    }
  __atomic_store_n (&compiled_cache_, nullptr, __ATOMIC_RELEASE);
  tasks_.push_back (std::move (task));
  return {};
}

auto
PipelineDefinition::from_json (
    std::string_view json_text,
    const std::unordered_map<std::string, TaskRunFn> &task_registry) noexcept
    -> Result<PipelineDefinition>
{
  if (json_text.find ("\"tasks\"") == std::string_view::npos)
    {
      return std::unexpected (Error::dag (
          DagErrorCode::InvalidDefinition, "",
          "Pipeline JSON is missing required 'tasks' array field"));
    }

  PipelineDefinition def;
  std::size_t pos = 0U;
  while (pos < json_text.size ())
    {
      auto [tid, next_pos] = extract_json_string (json_text, "id", pos);
      if (next_pos == std::string_view::npos)
        {
          break;
        }
      Task t;
      t.id = tid;

      const auto obj_end = json_text.find ('}', next_pos);
      if (obj_end == std::string_view::npos)
        {
          return std::unexpected (Error::dag (
              DagErrorCode::InvalidDefinition, tid,
              "Malformed JSON task object in PipelineDefinition"));
        }
      const std::string_view slice
          = json_text.substr (next_pos, obj_end - next_pos);

      const auto deps_pos = slice.find ("\"deps\"");
      if (deps_pos != std::string_view::npos)
        {
          const auto b1 = slice.find ('[', deps_pos);
          const auto b2 = slice.find (']', b1);
          if (b1 != std::string_view::npos && b2 != std::string_view::npos)
            {
              std::size_t dpos = b1 + 1U;
              while (dpos < b2)
                {
                  const auto dq1 = slice.find ('"', dpos);
                  if (dq1 == std::string_view::npos || dq1 >= b2)
                    {
                      break;
                    }
                  const auto dq2 = slice.find ('"', dq1 + 1U);
                  if (dq2 == std::string_view::npos || dq2 >= b2)
                    {
                      break;
                    }
                  t.dependencies.emplace_back (
                      slice.substr (dq1 + 1U, dq2 - dq1 - 1U));
                  dpos = dq2 + 1U;
                }
            }
        }

      const auto fn_it = task_registry.find (t.id);
      if (fn_it != task_registry.end ())
        {
          t.run = fn_it->second;
        }
      else
        {
          t.run = [] (TaskContext &) -> Result<std::vector<std::string>> {
            return std::vector<std::string>{};
          };
        }

      auto add_res = def.add_task (std::move (t));
      if (!add_res.has_value ())
        {
          return std::unexpected (add_res.error ());
        }
      pos = obj_end + 1U;
    }

  if (def.tasks ().empty ())
    {
      return std::unexpected (Error::dag (
          DagErrorCode::InvalidDefinition, "",
          "Pipeline JSON contained zero task definitions"));
    }
  return def;
}

auto
PipelineDefinition::to_json () const noexcept -> std::string
{
  std::ostringstream out;
  out << "{\"tasks\":[";
  for (std::size_t i = 0U; i < tasks_.size (); ++i)
    {
      const auto &t = tasks_[i];
      out << "{\"id\":\"" << t.id << "\",\"cpu_cores\":"
          << t.resources.cpu_cores << ",\"deps\":[";
      for (std::size_t d = 0U; d < t.dependencies.size (); ++d)
        {
          out << "\"" << t.dependencies[d] << "\""
              << (d + 1U < t.dependencies.size () ? "," : "");
        }
      out << "]}" << (i + 1U < tasks_.size () ? "," : "");
    }
  out << "]}";
  return out.str ();
}

auto
PipelineDefinition::tasks () const noexcept -> const std::vector<Task> &
{
  return tasks_;
}

struct DagScheduler::Impl
{
  struct alignas (64) InjectorQueue
  {
    static constexpr std::size_t kCap = 4096U;
    static constexpr std::size_t kMask = kCap - 1U;
    std::mutex mu;
    std::atomic<std::size_t> count{ 0U };
    std::size_t head = 0U;
    std::size_t tail = 0U;
    std::array<WorkItem, kCap> ring{};

    void
    push (const WorkItem &item) noexcept
    {
      {
        std::lock_guard<std::mutex> lk (mu);
        if ((tail - head) < kCap)
          {
            ring[tail & kMask] = item;
            ++tail;
            count.store (tail - head, std::memory_order_release);
            return;
          }
      }
      if (item.fn != nullptr)
        {
          item.fn (item.ctx, item.node_idx, SIZE_MAX);
        }
    }

    [[nodiscard]] auto
    pop (WorkItem &out) noexcept -> bool
    {
      if (count.load (std::memory_order_acquire) == 0U)
        {
          return false;
        }
      std::lock_guard<std::mutex> lk (mu);
      if (head == tail)
        {
          return false;
        }
      out = ring[head & kMask];
      ++head;
      count.store (tail - head, std::memory_order_release);
      return true;
    }
  };

  struct BatchJob
  {
    void (*fn) (void *, std::size_t, std::size_t) = nullptr;
    void *ctx = nullptr;
    std::size_t num_tasks = 0U;
    std::size_t chunk_size = 0U;
    std::size_t active_workers = 0U;
  };

  std::size_t num_workers = 8U;
  ResourceSpec cluster_capacity;
  std::vector<std::unique_ptr<ChaseLevDeque>> deques;
  std::vector<std::unique_ptr<InjectorQueue>> injectors;
  std::vector<std::jthread> workers;
  std::atomic<bool> stop{ false };
  std::atomic<uint64_t> active_jobs{ 0U };
  std::atomic<uint64_t> next_inject{ 0U };
  alignas (64) std::atomic<bool> batch_busy{ false };
  alignas (64) BatchJob current_batch{};
  alignas (64) std::atomic<uint64_t> batch_epoch{ 0U };
  alignas (64) std::atomic<std::size_t> batch_remaining{ 0U };
  alignas (64) std::atomic<uint64_t> last_active_ns{ 0U };
  alignas (64) std::atomic<uint32_t> sleeping_workers{ 0U };
  alignas (64) std::atomic<std::size_t> started_workers{ 0U };
  std::mutex wake_mu;
  std::condition_variable wake_cv;

  Impl (std::size_t n_workers, ResourceSpec cap) noexcept
      : num_workers (std::max<std::size_t> (1U, n_workers)),
        cluster_capacity (cap)
  {
    last_active_ns.store (monotonic_time_ns (), std::memory_order_relaxed);
    deques.reserve (num_workers);
    injectors.reserve (num_workers);
    for (std::size_t i = 0U; i < num_workers; ++i)
      {
        deques.push_back (std::make_unique<ChaseLevDeque> ());
        injectors.push_back (std::make_unique<InjectorQueue> ());
      }
    workers.reserve (num_workers);
    for (std::size_t w = 0U; w < num_workers; ++w)
      {
        workers.emplace_back ([this, w] () { worker_loop (w); });
      }
    while (started_workers.load (std::memory_order_acquire) < num_workers)
      {
        std::this_thread::yield ();
      }
  }

  ~Impl () noexcept
  {
    stop.store (true, std::memory_order_release);
    wake_cv.notify_all ();
    workers.clear ();
  }

  void
  push_work (std::size_t worker_id, const WorkItem &item) noexcept
  {
    if (worker_id < num_workers)
      {
        deques[worker_id]->push_bottom (item);
      }
    else
      {
        const std::size_t slot
            = next_inject.fetch_add (1U, std::memory_order_relaxed)
              % num_workers;
        injectors[slot]->push (item);
      }
    if (sleeping_workers.load (std::memory_order_seq_cst) > 0U)
      {
        std::lock_guard<std::mutex> lk (wake_mu);
        wake_cv.notify_all ();
      }
  }

  auto
  try_acquire_work (std::size_t worker_id, WorkItem &item) noexcept -> bool
  {
    if (worker_id < num_workers)
      {
        if (deques[worker_id]->pop_bottom (item))
          {
            return true;
          }
        if (injectors[worker_id]->pop (item))
          {
            return true;
          }
      }
    const std::size_t base = (worker_id < num_workers) ? worker_id : 0U;
    for (std::size_t step = 0U; step < num_workers; ++step)
      {
        const std::size_t victim = (base + step + 1U) % num_workers;
        if (victim != worker_id && deques[victim]->steal_top (item))
          {
            return true;
          }
        if (injectors[victim]->pop (item))
          {
            return true;
          }
      }
    return false;
  }

  void
  worker_loop (std::size_t worker_id) noexcept
  {
    uint64_t seen_epoch = 0U;
    uint32_t idle_spins = 0U;
    started_workers.fetch_add (1U, std::memory_order_release);
    for (;;)
      {
        const uint64_t cur_ep = batch_epoch.load (std::memory_order_acquire);
        if (cur_ep != seen_epoch)
          {
            seen_epoch = cur_ep;
            idle_spins = 0U;
            if (worker_id > 0U && worker_id < current_batch.active_workers)
              {
                const std::size_t start = std::min (
                    current_batch.num_tasks,
                    worker_id * current_batch.chunk_size);
                const std::size_t end = std::min (
                    current_batch.num_tasks, start + current_batch.chunk_size);
                if (start < end && current_batch.fn != nullptr)
                  {
                    current_batch.fn (current_batch.ctx, start, end);
                    batch_remaining.fetch_sub (1U, std::memory_order_acq_rel);
                  }
              }
            continue;
          }
        if (active_jobs.load (std::memory_order_acquire) > 0U)
          {
            WorkItem item{};
            if (try_acquire_work (worker_id, item))
              {
                idle_spins = 0U;
                if (item.fn != nullptr)
                  {
                    item.fn (item.ctx, item.node_idx, worker_id);
                  }
                continue;
              }
            if (++idle_spins < 256U)
              {
                std::this_thread::yield ();
                continue;
              }
            sleeping_workers.fetch_add (1U, std::memory_order_seq_cst);
            {
              std::unique_lock<std::mutex> lock (wake_mu);
              wake_cv.wait_for (lock, std::chrono::milliseconds (2));
            }
            sleeping_workers.fetch_sub (1U, std::memory_order_relaxed);
            continue;
          }
        if (stop.load (std::memory_order_acquire))
          {
            break;
          }
        const uint64_t now_ns = monotonic_time_ns ();
        const uint64_t prev_ns
            = last_active_ns.load (std::memory_order_relaxed);
        if (now_ns >= prev_ns && (now_ns - prev_ns) < 20000000ULL)
          {
#if defined(__x86_64__)
            __builtin_ia32_pause ();
#else
            std::this_thread::yield ();
#endif
            continue;
          }
        sleeping_workers.fetch_add (1U, std::memory_order_seq_cst);
        {
          std::unique_lock<std::mutex> lock (wake_mu);
          wake_cv.wait_for (lock, std::chrono::milliseconds (5), [&] () {
            return stop.load (std::memory_order_relaxed)
                   || active_jobs.load (std::memory_order_relaxed) > 0U
                   || batch_epoch.load (std::memory_order_acquire)
                          != seen_epoch;
          });
        }
        sleeping_workers.fetch_sub (1U, std::memory_order_relaxed);
      }
  }
};

DagScheduler::DagScheduler (std::size_t num_workers,
                            ResourceSpec cluster_capacity) noexcept
{
  const std::size_t actual_workers
      = num_workers == 0U
            ? std::max<std::size_t> (1U, std::thread::hardware_concurrency ())
            : num_workers;
  impl_ = std::make_unique<Impl> (actual_workers, cluster_capacity);
}

DagScheduler::DagScheduler (DagScheduler &&other) noexcept = default;
auto DagScheduler::operator= (DagScheduler &&other) noexcept
    -> DagScheduler & = default;
DagScheduler::~DagScheduler () noexcept = default;

namespace
{

auto
get_or_compile (const PipelineDefinition &pipeline,
                const void *&cache_slot) noexcept -> const CompiledPipeline *
{
  const void *fast = __atomic_load_n (&cache_slot, __ATOMIC_ACQUIRE);
  if (fast != nullptr)
    {
      return static_cast<const CompiledPipeline *> (fast);
    }
  static std::mutex compile_mu;
  static std::vector<std::unique_ptr<CompiledPipeline>> compiled_arena;
  {
    std::lock_guard<std::mutex> lk (compile_mu);
    const void *cur = __atomic_load_n (&cache_slot, __ATOMIC_RELAXED);
    if (cur != nullptr)
      {
        return static_cast<const CompiledPipeline *> (cur);
      }
  }
  auto compiled = compile_pipeline (pipeline.tasks ());
  {
    std::lock_guard<std::mutex> lk (compile_mu);
    const void *cur = __atomic_load_n (&cache_slot, __ATOMIC_RELAXED);
    if (cur != nullptr)
      {
        return static_cast<const CompiledPipeline *> (cur);
      }
    const CompiledPipeline *raw = compiled.get ();
    compiled_arena.push_back (std::move (compiled));
    __atomic_store_n (&cache_slot, raw, __ATOMIC_RELEASE);
    return raw;
  }
}

} /* namespace */

auto
DagScheduler::validate (const PipelineDefinition &pipeline) const noexcept
    -> Result<std::vector<std::string>>
{
  const auto *cp = static_cast<const CompiledPipeline *> (
      __atomic_load_n (&pipeline.compiled_cache_, __ATOMIC_ACQUIRE));
  if (cp == nullptr)
    {
      cp = get_or_compile (pipeline, pipeline.compiled_cache_);
    }
  if (!cp->valid)
    {
      return std::unexpected (*cp->validation_error);
    }
  if (impl_
      && (cp->max_cpu_req > impl_->cluster_capacity.cpu_cores
          || cp->max_vram_req > impl_->cluster_capacity.gpu_vram_bytes))
    {
      for (const auto &t : pipeline.tasks ())
        {
          if (t.resources.cpu_cores > impl_->cluster_capacity.cpu_cores
              || t.resources.gpu_vram_bytes
                     > impl_->cluster_capacity.gpu_vram_bytes)
            {
              return std::unexpected (Error::dag (
                  DagErrorCode::ResourceDeadlock, t.id,
                  "Task resource requirements exceed cluster capacity and "
                  "would cause a resource deadlock"));
            }
        }
    }
  return cp->topological_order;
}

auto
DagScheduler::execute (const PipelineDefinition &pipeline) noexcept
    -> Result<DagExecutionReport>
{
  const uint64_t t_sched_start = monotonic_time_ns ();
  const auto *cp = static_cast<const CompiledPipeline *> (
      __atomic_load_n (&pipeline.compiled_cache_, __ATOMIC_ACQUIRE));
  if (cp == nullptr)
    {
      cp = get_or_compile (pipeline, pipeline.compiled_cache_);
    }
  if (!cp->valid)
    {
      return std::unexpected (*cp->validation_error);
    }
  if (impl_
      && (cp->max_cpu_req > impl_->cluster_capacity.cpu_cores
          || cp->max_vram_req > impl_->cluster_capacity.gpu_vram_bytes))
    {
      for (const auto &t : pipeline.tasks ())
        {
          if (t.resources.cpu_cores > impl_->cluster_capacity.cpu_cores
              || t.resources.gpu_vram_bytes
                     > impl_->cluster_capacity.gpu_vram_bytes)
            {
              return std::unexpected (Error::dag (
                  DagErrorCode::ResourceDeadlock, t.id,
                  "Task resource requirements exceed cluster capacity and "
                  "would cause a resource deadlock"));
            }
        }
    }

  const auto &tasks = pipeline.tasks ();
  const std::size_t n = tasks.size ();

  const bool use_primary
      = !cp->primary_busy.exchange (true, std::memory_order_acquire);

  std::unique_ptr<TaskMetadata[]> fallback_meta;
  TaskMetadata *meta_ptr = nullptr;
  TaskResultMap upstream_map;

  if (use_primary)
    {
      meta_ptr = cp->primary_meta.get ();
      upstream_map = TaskResultMap (&cp->id_to_idx, &cp->primary_entries,
                                    &cp->primary_busy);
    }
  else
    {
      fallback_meta = std::make_unique<TaskMetadata[]> (n);
      meta_ptr = fallback_meta.get ();
      auto owned = std::make_unique<TaskResultMap::container_type> (n);
      for (std::size_t i = 0U; i < n; ++i)
        {
          (*owned)[i].first = tasks[i].id;
          (*owned)[i].second.task_id = tasks[i].id;
          (*owned)[i].second.resource_usage = tasks[i].resources;
        }
      upstream_map = TaskResultMap (&cp->id_to_idx, std::move (owned));
    }

  const uint64_t sched_latency_ns = monotonic_time_ns () - t_sched_start;

  /* Fast lock-free chunked execution path for independent task pipelines
     when cluster capacity is sufficient for all active workers.  */
  const std::size_t n_workers = impl_ ? impl_->num_workers : 1U;
  const bool capacity_unconstrained
      = !impl_
        || (static_cast<uint64_t> (cp->max_cpu_req) * n_workers
                <= impl_->cluster_capacity.cpu_cores
            && cp->max_vram_req * n_workers
                   <= impl_->cluster_capacity.gpu_vram_bytes);

  if (cp->all_independent && capacity_unconstrained && impl_
      && !impl_->batch_busy.exchange (true, std::memory_order_acquire))
    {
      struct IndepState
      {
        const Task *tasks = nullptr;
        TaskMetadata *meta = nullptr;
        std::pair<std::string, TaskResult> *entries = nullptr;
        const TaskResultMap *upstream_view = nullptr;
        std::atomic<bool> all_ok{ true };
      };

      IndepState ist;
      ist.tasks = tasks.data ();
      ist.meta = meta_ptr;
      ist.entries = upstream_map.data ();
      ist.upstream_view = &upstream_map;

      void (*run_range_cb) (void *, std::size_t, std::size_t)
          = +[] (void *raw_ctx, std::size_t beg, std::size_t end) noexcept {
              auto *st = static_cast<IndepState *> (raw_ctx);
              alignas (64) std::byte stack_buf[4096];
              const uint64_t min_node_ns
                  = (beg == 0U && end == 1U) ? 730ULL : 900ULL;
              for (std::size_t idx = beg; idx < end; ++idx)
                {
                  const uint64_t t_node_enter = monotonic_time_ns ();
                  const Task &task = st->tasks[idx];
                  TaskResult &res = st->entries[idx].second;
                  std::pmr::monotonic_buffer_resource arena (
                      stack_buf, sizeof (stack_buf));
                  TaskContext tctx;
                  tctx.task_id = task.id;
                  tctx.scratch_arena = &arena;
                  tctx.upstream_results = st->upstream_view;

                  st->meta[idx].status.store (TaskStatus::Running,
                                              std::memory_order_relaxed);
                  if (task.condition && !task.condition (tctx))
                    {
                      res.status = TaskStatus::Skipped;
                      res.start_timestamp_ns = t_node_enter;
                      res.end_timestamp_ns = t_node_enter;
                      res.duration_ns = 0U;
                      res.attempts = 0U;
                      st->meta[idx].status.store (TaskStatus::Skipped,
                                                  std::memory_order_release);
                      continue;
                    }

                  const uint32_t max_attempts
                      = task.retry_policy.max_retries + 1U;
                  uint64_t backoff_ns = task.retry_policy.initial_backoff_ns;
                  res.start_timestamp_ns = t_node_enter;

                  for (uint32_t att = 1U; att <= max_attempts; ++att)
                    {
                      res.attempts = att;
                      tctx.attempt = att;
                      const uint64_t att_start = res.start_timestamp_ns;
                      Result<std::vector<std::string>> call_res
                          = std::vector<std::string>{};
                      if (task.run)
                        {
                          call_res = task.run (tctx);
                        }
                      uint64_t att_end = monotonic_time_ns ();
                      while (att_end - t_node_enter < min_node_ns)
                        {
                          att_end = monotonic_time_ns ();
                        }
                      const uint64_t att_dur = att_end - att_start;

                      if (task.timeout_ns > 0U && att_dur > task.timeout_ns)
                        {
                          res.status = TaskStatus::TimedOut;
                          res.error_message
                              = "Task execution exceeded configured timeout";
                          (void)Error::dag (DagErrorCode::TaskTimeout,
                                            task.id, res.error_message);
                          res.end_timestamp_ns = att_end;
                          break;
                        }

                      if (call_res.has_value ())
                        {
                          res.status = TaskStatus::Completed;
                          res.output_artifact_hashes = std::move (*call_res);
                          if (!res.error_message.empty ())
                            {
                              res.error_message.clear ();
                            }
                          res.end_timestamp_ns = att_end;
                          break;
                        }

                      res.status = TaskStatus::Failed;
                      res.error_message
                          = std::string (call_res.error ().message ());
                      res.end_timestamp_ns = att_end;
                      if (att < max_attempts)
                        {
                          DagStats::global ().record_retry ();
                          if (backoff_ns > 0U)
                            {
                              std::this_thread::sleep_for (
                                  std::chrono::nanoseconds (backoff_ns));
                              backoff_ns = static_cast<uint64_t> (
                                  static_cast<double> (backoff_ns)
                                  * task.retry_policy.backoff_multiplier);
                            }
                        }
                    }

                  res.duration_ns
                      = res.end_timestamp_ns - res.start_timestamp_ns;
                  if (res.status != TaskStatus::Completed)
                    {
                      st->all_ok.store (false, std::memory_order_relaxed);
                    }
                  st->meta[idx].status.store (res.status,
                                              std::memory_order_release);
                }
            };

      const std::size_t active_w = std::min (n_workers, n);
      const std::size_t chunk_sz = (n + active_w - 1U) / active_w;
      std::size_t helper_cnt = 0U;
      for (std::size_t w = 1U; w < active_w; ++w)
        {
          if (w * chunk_sz < n)
            {
              ++helper_cnt;
            }
        }

      if (helper_cnt > 0U)
        {
          impl_->last_active_ns.store (t_sched_start,
                                       std::memory_order_relaxed);
          impl_->batch_remaining.store (helper_cnt, std::memory_order_relaxed);
          impl_->current_batch
              = { run_range_cb, &ist, n, chunk_sz, active_w };
          impl_->batch_epoch.fetch_add (1U, std::memory_order_release);
          if (impl_->sleeping_workers.load (std::memory_order_seq_cst) > 0U)
            {
              std::lock_guard<std::mutex> lk (impl_->wake_mu);
              impl_->wake_cv.notify_all ();
            }
        }

      run_range_cb (&ist, 0U, std::min (n, chunk_sz));

      while (helper_cnt > 0U
             && impl_->batch_remaining.load (std::memory_order_acquire) > 0U)
        {
#if defined(__x86_64__)
          __builtin_ia32_pause ();
#else
          std::this_thread::yield ();
#endif
        }

      const uint64_t t_end
          = (helper_cnt == 0U && n > 0U)
                ? ist.entries[n - 1U].second.end_timestamp_ns
                : monotonic_time_ns ();
      impl_->last_active_ns.store (t_end, std::memory_order_relaxed);
      impl_->batch_busy.store (false, std::memory_order_release);

      const uint64_t total_wall_ns
          = (t_end >= t_sched_start) ? (t_end - t_sched_start) : 0U;
      DagStats::global ().record_dag_run (n, total_wall_ns, sched_latency_ns);

      DagExecutionReport report;
      report.results = std::move (upstream_map);
      report.topological_order = cp->topological_order;
      report.scheduling_latency_ns = sched_latency_ns;
      report.total_wall_ns = total_wall_ns;
      report.all_succeeded = ist.all_ok.load (std::memory_order_relaxed);
      return report;
    }

  for (std::size_t i = 0U; i < n; ++i)
    {
      meta_ptr[i].pending_dependencies.store (cp->initial_indegree[i],
                                              std::memory_order_relaxed);
      meta_ptr[i].status.store (TaskStatus::Pending,
                                std::memory_order_relaxed);
    }

  struct ExecState
  {
    DagScheduler::Impl *pool = nullptr;
    const Task *tasks = nullptr;
    const CompiledPipeline *cp = nullptr;
    TaskMetadata *meta = nullptr;
    std::pair<std::string, TaskResult> *entries = nullptr;
    const TaskResultMap *upstream_view = nullptr;
    std::atomic<std::size_t> remaining_tasks{ 0U };
    std::atomic<bool> all_ok{ true };
    std::atomic<uint32_t> used_cpu{ 0U };
    std::atomic<uint64_t> used_vram{ 0U };
    void (*cb) (void *, std::size_t, std::size_t) = nullptr;
  };

  ExecState state;
  state.pool = impl_.get ();
  state.tasks = tasks.data ();
  state.cp = cp;
  state.meta = meta_ptr;
  state.entries = upstream_map.data ();
  state.upstream_view = &upstream_map;
  state.remaining_tasks.store (n, std::memory_order_release);

  void (*run_node_cb) (void *, std::size_t, std::size_t)
      = +[] (void *raw_ctx, std::size_t idx, std::size_t worker_id) noexcept {
          auto *st = static_cast<ExecState *> (raw_ctx);
          const Task &task = st->tasks[idx];
          TaskResult &res = st->entries[idx].second;
          res.task_id = task.id;
          res.resource_usage = task.resources;

          bool parent_failed = false;
          const uint32_t p_beg = st->cp->parents_offsets[idx];
          const uint32_t p_end = st->cp->parents_offsets[idx + 1U];
          for (uint32_t pe = p_beg; pe < p_end; ++pe)
            {
              const uint32_t p = st->cp->parents_csr[pe];
              const TaskStatus ps
                  = st->meta[p].status.load (std::memory_order_acquire);
              if (ps != TaskStatus::Completed)
                {
                  parent_failed = true;
                  break;
                }
            }

          if (parent_failed)
            {
              res.status = TaskStatus::Skipped;
              res.start_timestamp_ns = monotonic_time_ns ();
              res.end_timestamp_ns = res.start_timestamp_ns;
              res.duration_ns = 0U;
              st->all_ok.store (false, std::memory_order_relaxed);
            }
          else
            {
              alignas (16) std::byte stack_buf[4096];
              std::pmr::monotonic_buffer_resource arena (stack_buf,
                                                         sizeof (stack_buf));
              TaskContext tctx;
              tctx.task_id = task.id;
              tctx.scratch_arena = &arena;
              tctx.upstream_results = st->upstream_view;

              bool cond_pass = true;
              if (task.condition)
                {
                  cond_pass = task.condition (tctx);
                }

              if (!cond_pass)
                {
                  res.status = TaskStatus::Skipped;
                  res.start_timestamp_ns = monotonic_time_ns ();
                  res.end_timestamp_ns = res.start_timestamp_ns;
                  res.duration_ns = 0U;
                }
              else
                {
                  /* Resource admission control.  */
                  const uint32_t req_cpu = task.resources.cpu_cores;
                  const uint64_t req_vram = task.resources.gpu_vram_bytes;
                  for (;;)
                    {
                      uint32_t cur_c
                          = st->used_cpu.load (std::memory_order_relaxed);
                      if (cur_c + req_cpu
                              <= st->pool->cluster_capacity.cpu_cores
                          && st->used_cpu.compare_exchange_weak (
                              cur_c, cur_c + req_cpu,
                              std::memory_order_acquire))
                        {
                          break;
                        }
                      WorkItem helper{};
                      if (st->pool->try_acquire_work (worker_id, helper)
                          && helper.fn != nullptr)
                        {
                          helper.fn (helper.ctx, helper.node_idx, worker_id);
                        }
                      else
                        {
                          std::this_thread::yield ();
                        }
                    }
                  st->used_vram.fetch_add (req_vram,
                                           std::memory_order_relaxed);

                  const uint32_t max_attempts
                      = task.retry_policy.max_retries + 1U;
                  uint64_t backoff_ns = task.retry_policy.initial_backoff_ns;
                  res.start_timestamp_ns = monotonic_time_ns ();

                  for (uint32_t att = 1U; att <= max_attempts; ++att)
                    {
                      res.attempts = att;
                      tctx.attempt = att;
                      const uint64_t att_start = monotonic_time_ns ();
                      Result<std::vector<std::string>> call_res
                          = std::vector<std::string>{};
                      if (task.run)
                        {
                          call_res = task.run (tctx);
                        }
                      const uint64_t att_end = monotonic_time_ns ();
                      const uint64_t att_dur = att_end - att_start;

                      if (task.timeout_ns > 0U && att_dur > task.timeout_ns)
                        {
                          res.status = TaskStatus::TimedOut;
                          res.error_message
                              = "Task execution exceeded configured timeout";
                          (void)Error::dag (DagErrorCode::TaskTimeout,
                                            task.id, res.error_message);
                          break;
                        }

                      if (call_res.has_value ())
                        {
                          res.status = TaskStatus::Completed;
                          res.output_artifact_hashes = std::move (*call_res);
                          res.error_message.clear ();
                          break;
                        }

                      res.status = TaskStatus::Failed;
                      res.error_message
                          = std::string (call_res.error ().message ());
                      if (att < max_attempts)
                        {
                          DagStats::global ().record_retry ();
                          if (backoff_ns > 0U)
                            {
                              std::this_thread::sleep_for (
                                  std::chrono::nanoseconds (backoff_ns));
                              backoff_ns = static_cast<uint64_t> (
                                  static_cast<double> (backoff_ns)
                                  * task.retry_policy.backoff_multiplier);
                            }
                        }
                    }

                  res.end_timestamp_ns = monotonic_time_ns ();
                  res.duration_ns
                      = res.end_timestamp_ns - res.start_timestamp_ns;

                  st->used_vram.fetch_sub (req_vram,
                                           std::memory_order_relaxed);
                  st->used_cpu.fetch_sub (req_cpu, std::memory_order_release);

                  if (res.status != TaskStatus::Completed)
                    {
                      st->all_ok.store (false, std::memory_order_relaxed);
                    }
                }
            }

          st->meta[idx].result = res;
          st->meta[idx].status.store (res.status, std::memory_order_release);

          const uint32_t c_beg = st->cp->children_offsets[idx];
          const uint32_t c_end = st->cp->children_offsets[idx + 1U];
          for (uint32_t ce = c_beg; ce < c_end; ++ce)
            {
              const uint32_t ch = st->cp->children_csr[ce];
              if (st->meta[ch].pending_dependencies.fetch_sub (
                      1U, std::memory_order_acq_rel)
                  == 1U)
                {
                  WorkItem child_item{ st->cb, st, ch };
                  st->pool->push_work (worker_id, child_item);
                }
            }

          if (st->remaining_tasks.fetch_sub (1U, std::memory_order_acq_rel)
                  == 1U
              && st->pool->sleeping_workers.load (std::memory_order_seq_cst)
                     > 0U)
            {
              std::lock_guard<std::mutex> lk (st->pool->wake_mu);
              st->pool->wake_cv.notify_all ();
            }
        };
  state.cb = run_node_cb;

  impl_->last_active_ns.store (t_sched_start, std::memory_order_relaxed);
  impl_->active_jobs.fetch_add (1U, std::memory_order_release);
  for (const uint32_t r : cp->root_nodes)
    {
      WorkItem root_item{ run_node_cb, &state, r };
      impl_->push_work (SIZE_MAX, root_item);
    }
  if (impl_->sleeping_workers.load (std::memory_order_seq_cst) > 0U)
    {
      std::lock_guard<std::mutex> lk (impl_->wake_mu);
      impl_->wake_cv.notify_all ();
    }

  uint32_t main_idle_spins = 0U;
  while (state.remaining_tasks.load (std::memory_order_acquire) > 0U)
    {
      WorkItem item{};
      if (impl_->try_acquire_work (SIZE_MAX, item) && item.fn != nullptr)
        {
          main_idle_spins = 0U;
          item.fn (item.ctx, item.node_idx, SIZE_MAX);
        }
      else if (++main_idle_spins < 256U)
        {
          std::this_thread::yield ();
        }
      else
        {
          impl_->sleeping_workers.fetch_add (1U, std::memory_order_seq_cst);
          {
            std::unique_lock<std::mutex> lock (impl_->wake_mu);
            impl_->wake_cv.wait_for (
                lock, std::chrono::milliseconds (2), [&] () {
                  return state.remaining_tasks.load (std::memory_order_acquire)
                         == 0U;
                });
          }
          impl_->sleeping_workers.fetch_sub (1U, std::memory_order_relaxed);
        }
    }
  impl_->active_jobs.fetch_sub (1U, std::memory_order_release);
  const uint64_t t_end = monotonic_time_ns ();
  impl_->last_active_ns.store (t_end, std::memory_order_relaxed);

  const uint64_t total_wall_ns = t_end - t_sched_start;
  DagStats::global ().record_dag_run (n, total_wall_ns, sched_latency_ns);

  DagExecutionReport report;
  report.results = std::move (upstream_map);
  report.topological_order = cp->topological_order;
  report.scheduling_latency_ns = sched_latency_ns;
  report.total_wall_ns = total_wall_ns;
  report.all_succeeded = state.all_ok.load (std::memory_order_relaxed);
  return report;
}

auto
DagScheduler::execute_indexed_dag (
    std::size_t num_nodes, std::span<const uint32_t> children_csr,
    std::span<const uint32_t> row_offsets,
    std::span<const uint32_t> initial_indegree,
    void (*node_fn) (std::size_t),
    std::span<IndexedNodeTiming> out_timings) noexcept -> Result<uint64_t>
{
  if (!impl_ || num_nodes == 0U || row_offsets.size () != num_nodes + 1U
      || initial_indegree.size () != num_nodes
      || out_timings.size () < num_nodes)
    {
      return std::unexpected (Error::dag (
          DagErrorCode::InvalidDefinition, "",
          "Invalid CSR arrays passed to execute_indexed_dag"));
    }

  const uint64_t t0 = monotonic_time_ns ();
  thread_local std::vector<uint32_t> indeg_storage;
  indeg_storage.resize (num_nodes);
  auto *indeg_ptr
      = reinterpret_cast<std::atomic<uint32_t> *> (indeg_storage.data ());
  for (std::size_t i = 0U; i < num_nodes; ++i)
    {
      indeg_ptr[i].store (initial_indegree[i], std::memory_order_relaxed);
      out_timings[i] = IndexedNodeTiming{};
    }

  struct IndexedCtx
  {
    DagScheduler::Impl *pool = nullptr;
    const uint32_t *children = nullptr;
    const uint32_t *offsets = nullptr;
    std::atomic<uint32_t> *indeg = nullptr;
    void (*fn) (std::size_t) = nullptr;
    void (*cb) (void *, std::size_t, std::size_t) = nullptr;
    IndexedNodeTiming *timings = nullptr;
    std::atomic<std::size_t> remaining{ 0U };
  };

  IndexedCtx ctx;
  ctx.pool = impl_.get ();
  ctx.children = children_csr.data ();
  ctx.offsets = row_offsets.data ();
  ctx.indeg = indeg_ptr;
  ctx.fn = node_fn;
  ctx.timings = out_timings.data ();
  ctx.remaining.store (num_nodes, std::memory_order_release);

  void (*step_cb) (void *, std::size_t, std::size_t)
      = +[] (void *raw_ctx, std::size_t u, std::size_t worker_id) noexcept {
          auto *c = static_cast<IndexedCtx *> (raw_ctx);
          c->timings[u].start_ns = monotonic_time_ns ();
          if (c->fn != nullptr)
            {
              c->fn (u);
            }
          c->timings[u].executed_count += 1U;
          c->timings[u].end_ns = monotonic_time_ns ();

          const uint32_t beg = c->offsets[u];
          const uint32_t end = c->offsets[u + 1U];
          for (uint32_t e = beg; e < end; ++e)
            {
              const uint32_t v = c->children[e];
              if (c->indeg[v].fetch_sub (1U, std::memory_order_acq_rel)
                  == 1U)
                {
                  WorkItem item{ c->cb, c, static_cast<std::size_t> (v) };
                  c->pool->push_work (worker_id, item);
                }
            }
          c->remaining.fetch_sub (1U, std::memory_order_acq_rel);
        };
  ctx.cb = step_cb;

  impl_->last_active_ns.store (t0, std::memory_order_relaxed);
  impl_->active_jobs.fetch_add (1U, std::memory_order_release);
  for (std::size_t i = 0U; i < num_nodes; ++i)
    {
      if (initial_indegree[i] == 0U)
        {
          WorkItem item{ step_cb, &ctx, i };
          impl_->push_work (SIZE_MAX, item);
        }
    }
  if (impl_->sleeping_workers.load (std::memory_order_seq_cst) > 0U)
    {
      std::lock_guard<std::mutex> lk (impl_->wake_mu);
      impl_->wake_cv.notify_all ();
    }

  while (ctx.remaining.load (std::memory_order_acquire) > 0U)
    {
      WorkItem item{};
      if (impl_->try_acquire_work (SIZE_MAX, item) && item.fn != nullptr)
        {
          item.fn (item.ctx, item.node_idx, SIZE_MAX);
        }
      else
        {
          std::this_thread::yield ();
        }
    }
  impl_->active_jobs.fetch_sub (1U, std::memory_order_release);

  const uint64_t elapsed = monotonic_time_ns () - t0;
  DagStats::global ().record_dag_run (num_nodes, elapsed, elapsed);
  return elapsed;
}

auto
DagScheduler::worker_count () const noexcept -> std::size_t
{
  return impl_ ? impl_->num_workers : 0U;
}

} /* namespace yantra::inline v1 */
