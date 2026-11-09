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

#ifndef SURYA_CORE_PARALLEL_HPP
#define SURYA_CORE_PARALLEL_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <surya/core/config.hpp>

namespace surya::inline v1::core
{

/**
 * @brief Work-stealing parallel execution engine backed by Intel oneTBB.
 * @note Thread-safety: Safe to invoke concurrently from any thread, including
 *       nested parallel regions.
 */
class SURYA_API Thread_Pool
{
public:
  explicit Thread_Pool (size_t num_threads = 0U) noexcept;

  ~Thread_Pool (void) noexcept;

  Thread_Pool (const Thread_Pool &) = delete;
  Thread_Pool &operator= (const Thread_Pool &) = delete;

  /**
   * @brief Return the configured hardware concurrency thread count.
   * @pre None.
   * @post Returns value >= 1.
   * @returns Number of worker threads.
   * @note Thread-safety: Safe to call concurrently.
   */
  size_t
  num_threads (void) const noexcept
  {
    return num_threads_;
  }

  /**
   * @brief Execute a 1D index range [BEGIN, END) in chunks of at least GRAIN
   *        using Intel oneTBB work-stealing parallel_for.
   * @pre BEGIN <= END.
   * @post BODY has been invoked across a partition of [BEGIN, END).
   * @note Thread-safety: Lock-free work-stealing; safe for concurrent and
   *       nested calls.
   */
  void
  parallel_for (size_t begin,
                size_t end,
                size_t grain,
                const std::function<void (size_t, size_t)> &body);

  /**
   * @brief Return the global default thread pool singleton.
   */
  static Thread_Pool &
  global_pool (void) noexcept;

private:
  size_t num_threads_ = 1U;
};

/**
 * @brief Execute FN(r_begin, r_end) in parallel across [BEGIN, END).
 * @pre BEGIN <= END.
 * @post All sub-ranges of [BEGIN, END) have completed.
 * @note Thread-safety: Safe to call concurrently.
 */
template <typename Func>
void
parallel_for_1d (size_t begin, size_t end, size_t grain, Func &&fn)
{
  if (end <= begin)
    {
      return;
    }
  const size_t safe_grain = grain > 0U ? grain : 1U;
  if (end - begin <= safe_grain)
    {
      fn (begin, end);
      return;
    }
  Thread_Pool::global_pool ().parallel_for (
    begin, end, safe_grain,
    [&fn] (size_t b, size_t e)
    {
      fn (b, e);
    });
}

/**
 * @brief Execute FN(y0, y1, x0, x1) over cache-blocked 2D tiles of a
 *        HEIGHT x WIDTH image domain.
 * @pre None.
 * @post All 2D tiles have completed.
 * @note Thread-safety: Safe to call concurrently.
 */
template <typename Func>
void
parallel_for_2d (size_t height,
                 size_t width,
                 size_t tile_h,
                 size_t tile_w,
                 Func &&fn)
{
  if (height == 0U || width == 0U)
    {
      return;
    }
  const size_t th = tile_h > 0U ? tile_h : 32U;
  const size_t tw = tile_w > 0U ? tile_w : 64U;
  const size_t num_tiles_y = (height + th - 1U) / th;
  const size_t num_tiles_x = (width + tw - 1U) / tw;
  const size_t total_tiles = num_tiles_y * num_tiles_x;

  parallel_for_1d (
    0U, total_tiles, 1U,
    [&] (size_t t_begin, size_t t_end)
    {
      for (size_t t = t_begin; t < t_end; ++t)
        {
          const size_t ty = t / num_tiles_x;
          const size_t tx = t % num_tiles_x;
          const size_t y0 = ty * th;
          const size_t y1 = std::min (y0 + th, height);
          const size_t x0 = tx * tw;
          const size_t x1 = std::min (x0 + tw, width);
          fn (y0, y1, x0, x1);
        }
    });
}

} /* namespace surya::inline v1::core */

#endif /* SURYA_CORE_PARALLEL_HPP */
