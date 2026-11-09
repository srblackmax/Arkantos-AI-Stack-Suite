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

#include <surya/core/parallel.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <thread>
#include <vector>

namespace surya::inline v1::core
{

Thread_Pool::Thread_Pool (size_t num_threads) noexcept
{
  if (num_threads == 0U)
    {
      const unsigned int hw = std::thread::hardware_concurrency ();
      num_threads_ = hw > 1U ? static_cast<size_t> (hw) : 2U;
    }
  else
    {
      num_threads_ = num_threads;
    }
}

Thread_Pool::~Thread_Pool (void) noexcept = default;

void
Thread_Pool::parallel_for (
  size_t begin,
  size_t end,
  size_t grain,
  const std::function<void (size_t, size_t)> &body)
{
  if (end <= begin)
    {
      return;
    }
  const size_t total = end - begin;
  const size_t safe_grain = grain > 0U ? grain : 1U;
  if (total <= safe_grain)
    {
      body (begin, end);
      return;
    }

#if defined(__SANITIZE_THREAD__)
  /* Under ThreadSanitizer, avoid calling into the uninstrumented system
     libtbb.so.12 binary and partition across TSan-instrumented std::threads
     so all happens-before edges are visible to TSan.  */
  const size_t max_workers = std::min<size_t> (num_threads_, 4U);
  const size_t num_chunks =
    std::min (max_workers, (total + safe_grain - 1U) / safe_grain);
  if (num_chunks <= 1U)
    {
      body (begin, end);
      return;
    }
  const size_t chunk_size = (total + num_chunks - 1U) / num_chunks;
  std::vector<std::thread> workers;
  workers.reserve (num_chunks - 1U);
  for (size_t c = 1U; c < num_chunks; ++c)
    {
      const size_t c_begin = begin + c * chunk_size;
      if (c_begin >= end)
        {
          break;
        }
      const size_t c_end = std::min (end, c_begin + chunk_size);
      workers.emplace_back (
        [&body, c_begin, c_end] (void)
        {
          body (c_begin, c_end);
        });
    }
  body (begin, std::min (end, begin + chunk_size));
  for (auto &w : workers)
    {
      w.join ();
    }
#else
  tbb::parallel_for (
    tbb::blocked_range<size_t> (begin, end, safe_grain),
    [&body] (const tbb::blocked_range<size_t> &r)
    {
      body (r.begin (), r.end ());
    });
#endif
}

Thread_Pool &
Thread_Pool::global_pool (void) noexcept
{
  static Thread_Pool instance (0U);
  return instance;
}

} /* namespace surya::inline v1::core */
