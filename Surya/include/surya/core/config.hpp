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

#ifndef SURYA_CORE_CONFIG_HPP
#define SURYA_CORE_CONFIG_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>

#if defined(_WIN32) || defined(__CYGWIN__)
#  define SURYA_API __declspec(dllexport)
#elif defined(__GNUC__) && __GNUC__ >= 4
#  define SURYA_API __attribute__ ((visibility ("default")))
#else
#  define SURYA_API
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define SURYA_INLINE inline __attribute__ ((always_inline))
#  define SURYA_RESTRICT __restrict__
#else
#  define SURYA_INLINE inline
#  define SURYA_RESTRICT
#endif

#define SURYA_ABI_VERSION 1
#define SURYA_VERSION_MAJOR 1
#define SURYA_VERSION_MINOR 0
#define SURYA_VERSION_PATCH 0
#define SURYA_VERSION_STRING "1.0.0"

/* Default 6.5 GB VRAM ceiling on 8 GB RTX 3070 (leaves ~1.5 GB headroom).  */
#define SURYA_DEFAULT_VRAM_BUDGET_BYTES (6656ULL * 1024ULL * 1024ULL)

/* Hardware cache line and AVX-512 alignment boundary in bytes.  */
#define SURYA_CACHELINE_BYTES 64U

/* Internal invariant assertion macro.  Never use for user input.  */
#define SURYA_ASSERT(expr) assert (expr)

namespace surya::inline v1::core
{

/**
 * @brief Return the semantic version string of the Surya library.
 * @pre None.
 * @post Returns a non-null static NUL-terminated string.
 * @returns Pointer to static version string "1.0.0".
 * @note Thread-safety: Safe to call concurrently from any thread.
 */
constexpr const char *
version_string (void) noexcept
{
  return SURYA_VERSION_STRING;
}

/**
 * @brief Return the active inline ABI version number (1 for surya::v1).
 * @pre None.
 * @post Returns a positive integer ABI version.
 * @returns Integer ABI version.
 * @note Thread-safety: Safe to call concurrently from any thread.
 */
constexpr int
abi_version (void) noexcept
{
  return SURYA_ABI_VERSION;
}

} /* namespace surya::inline v1::core */

#endif /* SURYA_CORE_CONFIG_HPP */
