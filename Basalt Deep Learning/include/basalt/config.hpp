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

#ifndef BASALT_CONFIG_HPP
#define BASALT_CONFIG_HPP

/* Core configuration macros and portable compiler annotations.
   Every public macro uses the BASALT_ prefix, and internal helper macros
   use BASALT_INTERNAL_ so user translation units are never polluted.  */

#include <cstddef>
#include <cstdint>
#include <basalt/version.h>

#if defined(__GNUC__) || defined(__clang__)
#  define BASALT_API __attribute__ ((visibility ("default")))
#  define BASALT_INTERNAL_LIKELY(x) __builtin_expect (!!(x), 1)
#  define BASALT_INTERNAL_UNLIKELY(x) __builtin_expect (!!(x), 0)
#  define BASALT_INTERNAL_ALWAYS_INLINE inline __attribute__ ((always_inline))
#else
#  define BASALT_API
#  define BASALT_INTERNAL_LIKELY(x) (x)
#  define BASALT_INTERNAL_UNLIKELY(x) (x)
#  define BASALT_INTERNAL_ALWAYS_INLINE inline
#endif

#endif /* BASALT_CONFIG_HPP */
