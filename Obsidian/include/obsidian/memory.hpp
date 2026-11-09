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

#ifndef OBSIDIAN_MEMORY_HPP
#define OBSIDIAN_MEMORY_HPP

/**
 * @file memory.hpp
 * @brief High-performance PMR monotonic arena allocators and aligned memory.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <memory_resource>
#include <span>
#include <string>
#include <vector>
#include <obsidian/config.hpp>
#include <obsidian/error.hpp>

namespace obsidian
{

/**
 * @brief Polymorphic vector alias backed by std::pmr::polymorphic_allocator.
 */
template <typename T>
using Pmr_Vector = std::pmr::vector<T>;

/**
 * @brief Polymorphic string alias backed by std::pmr::polymorphic_allocator.
 */
using Pmr_String = std::pmr::string;

/**
 * @brief Fast stack-backed and pooled monotonic memory arena.
 *
 * Provides O(1) bump-pointer allocation for tokenization, Viterbi lattices,
 * and temporary NLP buffers without hitting the global heap in the hot path.
 */
class OBSIDIAN_API Arena
{
public:
  static constexpr size_t k_default_inline_bytes = 16384;

  /**
   * @brief Construct an Arena with an inline buffer and optional upstream.
   * @pre UPSTREAM must not be null.
   * @complexity O(1) time and O(1) heap allocations.
   */
  explicit Arena (std::pmr::memory_resource *upstream
                  = std::pmr::new_delete_resource ()) noexcept
    : upstream_ (upstream),
      resource_ (inline_buf_.data (), inline_buf_.size (), upstream_)
  {
  }

  /**
   * @brief Construct an Arena over a caller-provided byte span.
   * @pre EXT_BUF must remain valid for the lifetime of the Arena.
   * @complexity O(1) time and O(1) space.
   */
  explicit Arena (std::span<std::byte> ext_buf,
                  std::pmr::memory_resource *upstream
                  = std::pmr::new_delete_resource ()) noexcept
    : upstream_ (upstream),
      resource_ (ext_buf.data (), ext_buf.size (), upstream_)
  {
  }

  Arena (const Arena &) = delete;
  Arena &operator= (const Arena &) = delete;

  /**
   * @brief Return the underlying std::pmr::memory_resource pointer.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  std::pmr::memory_resource *
  resource (void) noexcept
  {
    return &resource_;
  }

  /**
   * @brief Reclaim all memory allocated from this arena in a single step.
   * @pre All containers using this arena must no longer access their memory.
   * @complexity O(K) where K is the number of overflow chunks (0 in hot path).
   */
  void
  reset (void) noexcept
  {
    resource_.release ();
  }

  /**
   * @brief Allocate an aligned array of COUNT elements of type T.
   * @pre COUNT > 0.
   * @complexity O(1) bump-pointer time.
   */
  template <typename T>
  std::span<T>
  allocate_span (size_t count)
  {
    if (count == 0)
      {
        return {};
      }
    void *raw = resource_.allocate (count * sizeof (T), alignof (T));
    return std::span<T> (static_cast<T *> (raw), count);
  }

private:
  alignas (k_cache_line_bytes)
    std::array<std::byte, k_default_inline_bytes> inline_buf_;
  std::pmr::memory_resource *upstream_;
  std::pmr::monotonic_buffer_resource resource_;
};

/**
 * @brief Return a thread-local scratch Arena for zero-allocation hot paths.
 * @pre Caller must call reset() before or after top-level use.
 * @complexity O(1) time and O(1) space.
 */
OBSIDIAN_API Arena &
thread_local_arena (void) noexcept;

} /* namespace obsidian */

#endif /* OBSIDIAN_MEMORY_HPP */
