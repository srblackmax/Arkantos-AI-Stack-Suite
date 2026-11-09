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

#ifndef OBSIDIAN_SIMD_HPP
#define OBSIDIAN_SIMD_HPP

/**
 * @file simd.hpp
 * @brief SIMD-accelerated string scanning, UTF-8 routines, and vector math.
 */

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <obsidian/config.hpp>
#include <obsidian/error.hpp>
#include <obsidian/memory.hpp>

namespace obsidian::simd
{

/**
 * @brief Return the name of the active SIMD instruction set ("AVX2", etc.).
 * @pre None.
 * @complexity O(1) time and O(1) space.
 */
OBSIDIAN_API const char *
active_isa_name (void) noexcept;

/**
 * @brief Find the first ASCII whitespace byte in TEXT using SIMD scanning.
 * @pre TEXT may be empty or arbitrary length.
 * @complexity O(N / W) time where W is the SIMD register width (32 on AVX2).
 */
OBSIDIAN_API size_t
find_first_whitespace (std::string_view text) noexcept;

/**
 * @brief Find the first non-whitespace byte in TEXT using SIMD scanning.
 * @pre TEXT may be empty or arbitrary length.
 * @complexity O(N / W) time where W is the SIMD register width.
 */
OBSIDIAN_API size_t
find_first_non_whitespace (std::string_view text) noexcept;

/**
 * @brief Scan TEXT and append zero-copy word/punctuation spans to OUT_SPANS.
 *
 * Splits on ASCII whitespace and isolates ASCII punctuation characters into
 * distinct token views without heap-allocating std::string objects.
 *
 * @pre OUT_SPANS is backed by a valid std::pmr::memory_resource.
 * @complexity O(N) time and O(T) arena space for T pre-tokens.
 */
OBSIDIAN_API void
scan_pretoken_spans (std::string_view text,
                     Pmr_Vector<std::string_view> &out_spans,
                     bool split_punctuation = true);

/**
 * @brief Validate whether TEXT is well-formed UTF-8 and count codepoints.
 * @pre None.
 * @complexity O(N) time and O(1) space in byte length N.
 */
OBSIDIAN_API Result<size_t>
validate_and_count_utf8 (std::string_view text) noexcept;

/**
 * @brief Convert ASCII uppercase bytes in SRC into lowercase in DST.
 * @pre DST.size() >= SRC.size().
 * @complexity O(N / W) SIMD time and O(1) extra space.
 */
OBSIDIAN_API void
ascii_to_lower (std::string_view src, char *dst) noexcept;

/**
 * @brief Compute the inner product of two float spans using SIMD FMA.
 * @pre A.size() == B.size().
 * @complexity O(D / 8) AVX2 vector operations and O(1) space.
 */
OBSIDIAN_API float
dot_f32 (std::span<const float> a, std::span<const float> b) noexcept;

/**
 * @brief Compute the squared L2 norm of a float span using SIMD FMA.
 * @pre None.
 * @complexity O(D / 8) AVX2 vector operations and O(1) space.
 */
OBSIDIAN_API float
norm_sq_f32 (std::span<const float> x) noexcept;

/**
 * @brief Compute the L2 norm of a float span using SIMD FMA.
 * @pre None.
 * @complexity O(D / 8) AVX2 vector operations and O(1) space.
 */
OBSIDIAN_API float
norm_f32 (std::span<const float> x) noexcept;

/**
 * @brief Compute the cosine similarity between two float spans using SIMD.
 * @pre A.size() == B.size().
 * @complexity O(D / 8) single-pass AVX2 reductions and O(1) space.
 */
OBSIDIAN_API float
cosine_similarity_f32 (std::span<const float> a,
                       std::span<const float> b,
                       float eps = 1e-8f) noexcept;

/**
 * @brief Compute Y[i] += ALPHA * X[i] in-place using SIMD FMA.
 * @pre Y.size() == X.size().
 * @complexity O(D / 8) AVX2 vector operations and O(1) space.
 */
OBSIDIAN_API void
axpy_f32 (float alpha, std::span<const float> x,
          std::span<float> y) noexcept;

/**
 * @brief Normalize a float vector in-place to unit L2 length.
 * @pre None.
 * @complexity O(D / 8) AVX2 vector operations and O(1) space.
 */
OBSIDIAN_API void
normalize_inplace_f32 (std::span<float> x, float eps = 1e-8f) noexcept;

} /* namespace obsidian::simd */

#endif /* OBSIDIAN_SIMD_HPP */
