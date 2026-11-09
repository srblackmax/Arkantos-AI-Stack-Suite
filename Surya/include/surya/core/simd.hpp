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

#ifndef SURYA_CORE_SIMD_HPP
#define SURYA_CORE_SIMD_HPP

#include <bit>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <surya/core/config.hpp>

namespace surya::inline v1::core
{

/**
 * @brief Supported hardware SIMD instruction set levels.
 * @note Thread-safety: Value type.
 */
enum class Simd_Level : uint8_t
{
  Scalar = 0,
  SSE4_2 = 1,
  AVX2 = 2,
  AVX512 = 3,
  NEON = 4
};

/**
 * @brief Detect highest supported SIMD instruction set on the host CPU
 *        at runtime via __builtin_cpu_supports.
 * @pre None.
 * @post Returns detected Simd_Level.
 * @returns Highest supported Simd_Level on current CPU.
 * @note Thread-safety: Safe to call concurrently from any thread.
 */
SURYA_API Simd_Level
detect_simd_level (void) noexcept;

/**
 * @brief Return human-readable name for a Simd_Level enum value.
 * @pre None.
 * @post Returns static string_view.
 * @returns Name of LEVEL.
 * @note Thread-safety: Pure constexpr function.
 */
constexpr std::string_view
simd_level_name (Simd_Level level) noexcept
{
  switch (level)
    {
    case Simd_Level::Scalar:
      return "Scalar";
    case Simd_Level::SSE4_2:
      return "SSE4.2";
    case Simd_Level::AVX2:
      return "AVX2";
    case Simd_Level::AVX512:
      return "AVX-512";
    case Simd_Level::NEON:
      return "NEON";
    }
  return "Unknown";
}

/**
 * @brief Compute 256-bit Hamming distance between two 32-byte binary
 *        descriptors (e.g., ORB / BRIEF) using hardware POPCNT.
 * @pre A and B point to at least 4 uint64_t words (32 bytes).
 * @post Returns bit count in [0, 256].
 * @returns Hamming distance between A and B.
 * @note Thread-safety: Pure function; safe across all threads.
 */
SURYA_INLINE uint32_t
hamming_distance_256 (const uint64_t *SURYA_RESTRICT a,
                      const uint64_t *SURYA_RESTRICT b) noexcept
{
  const uint32_t d0 = static_cast<uint32_t> (std::popcount (a[0] ^ b[0]));
  const uint32_t d1 = static_cast<uint32_t> (std::popcount (a[1] ^ b[1]));
  const uint32_t d2 = static_cast<uint32_t> (std::popcount (a[2] ^ b[2]));
  const uint32_t d3 = static_cast<uint32_t> (std::popcount (a[3] ^ b[3]));
  return d0 + d1 + d2 + d3;
}

/**
 * @brief Reference scalar dot product of two float32 arrays.
 * @pre A and B contain at least N elements.
 * @post Returns inner product sum_{i=0}^{N-1} A[i] * B[i].
 * @returns Scalar inner product.
 * @note Thread-safety: Safe to call concurrently.
 */
SURYA_API float
simd_dot_f32_scalar (const float *SURYA_RESTRICT a,
                     const float *SURYA_RESTRICT b,
                     size_t n) noexcept;

/**
 * @brief Runtime-dispatched (AVX-512 / AVX2 / Scalar) dot product of two
 *        float32 arrays using 64-byte aligned staging buffers.
 * @pre A and B contain at least N elements.
 * @post Returns inner product within 1e-6 relative tolerance of scalar path.
 * @returns SIMD inner product.
 * @note Thread-safety: Safe to call concurrently.
 */
SURYA_API float
simd_dot_f32 (const float *SURYA_RESTRICT a,
              const float *SURYA_RESTRICT b,
              size_t n) noexcept;

/**
 * @brief Reference scalar 1D dense convolution over a scanline.
 * @pre SRC has length WIDTH + KSIZE - 1; KERNEL has length KSIZE;
 *      DST has length WIDTH.
 * @post Populates DST[0..WIDTH-1].
 * @note Thread-safety: Safe to call concurrently on distinct DST buffers.
 */
SURYA_API void
simd_conv1d_f32_scalar (float *SURYA_RESTRICT dst,
                        const float *SURYA_RESTRICT src,
                        const float *SURYA_RESTRICT kernel,
                        size_t width,
                        size_t ksize) noexcept;

/**
 * @brief Runtime-dispatched (AVX-512 / AVX2 / Scalar) 1D dense convolution
 *        using alignas(64) staging buffers and 512-bit / 256-bit vector loads.
 * @pre SRC has length WIDTH + KSIZE - 1; KERNEL has length KSIZE;
 *      DST has length WIDTH.
 * @post Populates DST[0..WIDTH-1] within 1e-6 of simd_conv1d_f32_scalar.
 * @note Thread-safety: Safe to call concurrently on distinct DST buffers.
 */
SURYA_API void
simd_conv1d_f32 (float *SURYA_RESTRICT dst,
                 const float *SURYA_RESTRICT src,
                 const float *SURYA_RESTRICT kernel,
                 size_t width,
                 size_t ksize) noexcept;

/**
 * @brief Reference scalar Structural Similarity (SSIM) over two float32
 *        windows in [0, 255] dynamic range.
 * @pre A and B contain at least N elements (N > 0).
 * @post Returns SSIM index in [-1.0, 1.0].
 * @returns Scalar SSIM score.
 * @note Thread-safety: Safe to call concurrently.
 */
SURYA_API float
simd_ssim_f32_scalar (const float *SURYA_RESTRICT a,
                      const float *SURYA_RESTRICT b,
                      size_t n,
                      float dynamic_range = 255.0F) noexcept;

/**
 * @brief Runtime-dispatched (AVX-512 / AVX2 / Scalar) Structural Similarity
 *        (SSIM) using alignas(64) staging buffers and 512-bit / 256-bit loads.
 * @pre A and B contain at least N elements (N > 0).
 * @post Returns SSIM index within 1e-6 of simd_ssim_f32_scalar.
 * @returns SIMD SSIM score.
 * @note Thread-safety: Safe to call concurrently.
 */
SURYA_API float
simd_ssim_f32 (const float *SURYA_RESTRICT a,
               const float *SURYA_RESTRICT b,
               size_t n,
               float dynamic_range = 255.0F) noexcept;

/**
 * @brief Compute SIMD-accelerated absolute difference |A[i] - B[i]| for
 *        uint8_t scanlines.
 * @pre DST, A, and B contain at least N elements.
 * @post Populates DST[0..N-1].
 * @note Thread-safety: Safe to call concurrently on distinct DST buffers.
 */
SURYA_API void
simd_abs_diff_u8 (uint8_t *SURYA_RESTRICT dst,
                  const uint8_t *SURYA_RESTRICT a,
                  const uint8_t *SURYA_RESTRICT b,
                  size_t n) noexcept;

/**
 * @brief Compute SIMD-accelerated saturating addition A[i] + B[i] for
 *        uint8_t scanlines.
 * @pre DST, A, and B contain at least N elements.
 * @post Populates DST[0..N-1].
 * @note Thread-safety: Safe to call concurrently on distinct DST buffers.
 */
SURYA_API void
simd_add_sat_u8 (uint8_t *SURYA_RESTRICT dst,
                 const uint8_t *SURYA_RESTRICT a,
                 const uint8_t *SURYA_RESTRICT b,
                 size_t n) noexcept;

/**
 * @brief Compute SIMD-accelerated affine scaling DST[i] = X[i] * SCALE + BIAS.
 * @pre DST and X contain at least N elements.
 * @post Populates DST[0..N-1].
 * @note Thread-safety: Safe to call concurrently on distinct DST buffers.
 */
SURYA_API void
simd_scale_bias_u8_to_f32 (float *SURYA_RESTRICT dst,
                           const uint8_t *SURYA_RESTRICT x,
                           size_t n,
                           float scale,
                           float bias) noexcept;

} /* namespace surya::inline v1::core */

#endif /* SURYA_CORE_SIMD_HPP */
