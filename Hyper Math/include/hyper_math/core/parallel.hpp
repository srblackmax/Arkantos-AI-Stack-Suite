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

#ifndef HYPER_MATH_CORE_PARALLEL_HPP
#define HYPER_MATH_CORE_PARALLEL_HPP

#include "hyper_math/core/config.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace hyper_math
{

namespace detail
{

inline bool &
in_pool_worker_flag () noexcept
{
  thread_local bool flag = false;
  return flag;
}

} // namespace detail

/* Clean-room thread pool for parallelizing large tensor contractions,
   elementwise reductions, and columnar query pipelines.  */
class thread_pool
{
private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void ()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable cv_;
  bool stop_;

public:
  explicit
  thread_pool (std::size_t num_threads = 0)
    : workers_ (), tasks_ (), queue_mutex_ (), cv_ (), stop_ (false)
  {
    if (num_threads == 0)
      {
        const unsigned int hw = std::thread::hardware_concurrency ();
        num_threads = (hw > 0)
                      ? std::min<std::size_t> (8, static_cast<std::size_t> (hw))
                      : 2;
      }

    workers_.reserve (num_threads);
    for (std::size_t t = 0; t < num_threads; ++t)
      {
        workers_.emplace_back ([this] () {
          detail::in_pool_worker_flag () = true;
          while (true)
            {
              std::function<void ()> task;
              {
                std::unique_lock<std::mutex> lock (this->queue_mutex_);
                this->cv_.wait (lock, [this] () {
                  return this->stop_ || !this->tasks_.empty ();
                });
                if (this->stop_ && this->tasks_.empty ())
                  {
                    return;
                  }
                task = std::move (this->tasks_.front ());
                this->tasks_.pop ();
              }
              task ();
            }
        });
      }
  }

  ~thread_pool () noexcept
  {
    {
      std::unique_lock<std::mutex> lock (queue_mutex_);
      stop_ = true;
    }
    cv_.notify_all ();
    for (std::thread &worker : workers_)
      {
        if (worker.joinable ())
          {
            worker.join ();
          }
      }
  }

  thread_pool (const thread_pool &) = delete;
  thread_pool &operator= (const thread_pool &) = delete;

  std::size_t
  num_workers () const noexcept
  {
    return workers_.size ();
  }

  template <typename Fn>
  std::future<std::invoke_result_t<Fn>>
  submit (Fn &&fn)
  {
    using return_type = std::invoke_result_t<Fn>;
    auto task = std::make_shared<std::packaged_task<return_type ()>> (
      std::forward<Fn> (fn));
    std::future<return_type> res = task->get_future ();
    {
      std::unique_lock<std::mutex> lock (queue_mutex_);
      if (stop_)
        {
          throw hyper_math_error ("cannot submit task to stopped thread_pool");
        }
      tasks_.emplace ([task] () { (*task) (); });
    }
    cv_.notify_one ();
    return res;
  }
};

inline thread_pool &
global_thread_pool ()
{
  static thread_pool pool;
  return pool;
}

/* Parallel 1D index range execution: splits [begin, end) into chunks of at
   least `grain_size` and dispatches across `global_thread_pool()`.  */
template <typename IndexFn>
void
parallel_for (std::size_t begin,
              std::size_t end,
              std::size_t grain_size,
              IndexFn &&fn)
{
  if (end <= begin)
    {
      return;
    }
  const std::size_t total = end - begin;
  const std::size_t eff_grain = std::max<std::size_t> (1, grain_size);
#if defined(_OPENMP) && !defined(__SANITIZE_THREAD__)
  if (total > eff_grain && total >= 8 && !omp_in_parallel ()
      && !detail::in_pool_worker_flag ())
    {
      const std::size_t n_chunks
        = std::min<std::size_t> (8, (total + eff_grain - 1) / eff_grain);
      if (n_chunks > 1)
        {
          int num_t = static_cast<int> (n_chunks);
          std::exception_ptr first_exc = nullptr;
          std::mutex exc_mtx;
          #pragma omp parallel for schedule(static) num_threads(num_t)
          for (std::size_t i = begin; i < end; ++i)
            {
              try
                {
                  fn (i);
                }
              catch (...)
                {
                  std::lock_guard<std::mutex> lock (exc_mtx);
                  if (first_exc == nullptr)
                    {
                      first_exc = std::current_exception ();
                    }
                }
            }
          if (first_exc != nullptr)
            {
              std::rethrow_exception (first_exc);
            }
          return;
        }
    }
#endif
  if (total <= eff_grain
      || detail::in_pool_worker_flag ()
      || global_thread_pool ().num_workers () <= 1)
    {
      for (std::size_t i = begin; i < end; ++i)
        {
          fn (i);
        }
      return;
    }

  const std::size_t n_workers = global_thread_pool ().num_workers ();
  const std::size_t n_chunks =
    std::min (n_workers, (total + eff_grain - 1) / eff_grain);
  const std::size_t chunk_len = (total + n_chunks - 1) / n_chunks;

  std::vector<std::future<void>> futures;
  futures.reserve (n_chunks > 0 ? n_chunks - 1 : 0);

  for (std::size_t c = 1; c < n_chunks; ++c)
    {
      const std::size_t c_start = begin + c * chunk_len;
      const std::size_t c_end = std::min (end, c_start + chunk_len);
      if (c_start < c_end)
        {
          futures.push_back (
            global_thread_pool ().submit ([c_start, c_end, &fn] () {
              for (std::size_t i = c_start; i < c_end; ++i)
                {
                  fn (i);
                }
            }));
        }
    }

  /* Execute chunk 0 on the calling thread and ensure all worker futures
     complete before stack unwinding if an exception is thrown.  */
  const std::size_t c0_end = std::min (end, begin + chunk_len);
  try
    {
      for (std::size_t i = begin; i < c0_end; ++i)
        {
          fn (i);
        }
      for (auto &f : futures)
        {
          f.get ();
        }
    }
  catch (...)
    {
      for (auto &f : futures)
        {
          if (f.valid ())
            {
              f.wait ();
            }
        }
      throw;
    }
}

/* Parallel map-reduce over [begin, end).  */
template <typename AccT, typename MapFn, typename ReduceFn>
AccT
parallel_reduce (std::size_t begin,
                 std::size_t end,
                 std::size_t grain_size,
                 AccT identity_val,
                 MapFn &&map_fn,
                 ReduceFn &&reduce_fn)
{
  if (end <= begin)
    {
      return identity_val;
    }
  const std::size_t total = end - begin;
  const std::size_t eff_grain = std::max<std::size_t> (1, grain_size);
  if (total <= eff_grain
      || detail::in_pool_worker_flag ()
      || global_thread_pool ().num_workers () <= 1)
    {
      AccT acc = identity_val;
      for (std::size_t i = begin; i < end; ++i)
        {
          acc = reduce_fn (acc, map_fn (i));
        }
      return acc;
    }

  const std::size_t n_workers = global_thread_pool ().num_workers ();
  const std::size_t n_chunks =
    std::min (n_workers, (total + eff_grain - 1) / eff_grain);
  const std::size_t chunk_len = (total + n_chunks - 1) / n_chunks;

  std::vector<std::future<AccT>> futures;
  futures.reserve (n_chunks > 0 ? n_chunks - 1 : 0);

  for (std::size_t c = 1; c < n_chunks; ++c)
    {
      const std::size_t c_start = begin + c * chunk_len;
      const std::size_t c_end = std::min (end, c_start + chunk_len);
      if (c_start < c_end)
        {
          futures.push_back (global_thread_pool ().submit (
            [c_start, c_end, identity_val, &map_fn, &reduce_fn] () {
              AccT local_acc = identity_val;
              for (std::size_t i = c_start; i < c_end; ++i)
                {
                  local_acc = reduce_fn (local_acc, map_fn (i));
                }
              return local_acc;
            }));
        }
    }

  AccT total_acc = identity_val;
  const std::size_t c0_end = std::min (end, begin + chunk_len);
  try
    {
      for (std::size_t i = begin; i < c0_end; ++i)
        {
          total_acc = reduce_fn (total_acc, map_fn (i));
        }
      for (auto &f : futures)
        {
          total_acc = reduce_fn (total_acc, f.get ());
        }
    }
  catch (...)
    {
      for (auto &f : futures)
        {
          if (f.valid ())
            {
              f.wait ();
            }
        }
      throw;
    }
  return total_acc;
}

} // namespace hyper_math

#endif /* HYPER_MATH_CORE_PARALLEL_HPP */
