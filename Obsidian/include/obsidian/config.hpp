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

#ifndef OBSIDIAN_CONFIG_HPP
#define OBSIDIAN_CONFIG_HPP

/**
 * @file config.hpp
 * @brief Compiler attributes, symbol visibility, and SIMD feature detection.
 */

#include <cassert>
#include <cstddef>
#include <cstdint>

#if defined(_WIN32) || defined(__CYGWIN__)
#  define OBSIDIAN_API __declspec (dllexport)
#elif defined(__GNUC__) && __GNUC__ >= 4
#  define OBSIDIAN_API __attribute__ ((visibility ("default")))
#else
#  define OBSIDIAN_API
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define OBSIDIAN_LIKELY(x) __builtin_expect (!!(x), 1)
#  define OBSIDIAN_UNLIKELY(x) __builtin_expect (!!(x), 0)
#  define OBSIDIAN_ALWAYS_INLINE inline __attribute__ ((always_inline))
#  define OBSIDIAN_RESTRICT __restrict__
#else
#  define OBSIDIAN_LIKELY(x) (x)
#  define OBSIDIAN_UNLIKELY(x) (x)
#  define OBSIDIAN_ALWAYS_INLINE inline
#  define OBSIDIAN_RESTRICT
#endif

#if defined(__AVX2__)
#  define OBSIDIAN_HAS_AVX2 1
#else
#  define OBSIDIAN_HAS_AVX2 0
#endif

#if defined(__SSE4_2__)
#  define OBSIDIAN_HAS_SSE42 1
#else
#  define OBSIDIAN_HAS_SSE42 0
#endif

/**
 * @brief Internal invariant assertion macro wrapping standard assert.
 * @pre Expression COND must be free of side effects.
 * @complexity O(1) time and O(1) space.
 */
#define OBSIDIAN_ASSERT(cond) assert (cond)

namespace obsidian
{

/**
 * @brief Cache line alignment constant (64 bytes on modern x86_64/ARM64).
 */
inline constexpr size_t k_cache_line_bytes = 64;

} /* namespace obsidian */

#endif /* OBSIDIAN_CONFIG_HPP */
