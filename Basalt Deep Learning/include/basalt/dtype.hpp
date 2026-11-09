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

#ifndef BASALT_DTYPE_HPP
#define BASALT_DTYPE_HPP

/* Scalar data types, mixed-precision 16-bit floating-point representations
   (IEEE-754 binary16 and Brain Floating Point bf16), and type promotion
   rules used across tensors and automatic mixed precision.  */

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <basalt/config.hpp>

namespace basalt
{

/* Enumeration of all scalar data types supported by basalt::Tensor.  */
enum class Dtype : uint8_t
{
  Float32 = 0,
  Float64 = 1,
  Float16 = 2,
  BFloat16 = 3,
  Int32 = 4,
  Int64 = 5,
  UInt8 = 6,
  Bool = 7
};

/* IEEE-754 half-precision (1-5-10) floating-point type for mixed-precision
   storage and compute.  */
struct BASALT_API Float16
{
  uint16_t bits = 0;

  constexpr Float16 (void) noexcept = default;

  explicit Float16 (float value) noexcept
    : bits (from_float32 (value))
  {
  }

  explicit Float16 (double value) noexcept
    : bits (from_float32 (static_cast<float> (value)))
  {
  }

  static constexpr Float16
  from_bits (uint16_t raw) noexcept
  {
    Float16 out;
    out.bits = raw;
    return out;
  }

  operator float (void) const noexcept
  {
    return to_float32 (bits);
  }

  static uint16_t
  from_float32 (float value) noexcept;

  static float
  to_float32 (uint16_t raw) noexcept;
};

/* Brain Floating Point (1-8-7) 16-bit type preserving the dynamic range of
   IEEE-754 single precision with round-to-nearest-even conversion.  */
struct BASALT_API BFloat16
{
  uint16_t bits = 0;

  constexpr BFloat16 (void) noexcept = default;

  explicit BFloat16 (float value) noexcept
    : bits (from_float32 (value))
  {
  }

  explicit BFloat16 (double value) noexcept
    : bits (from_float32 (static_cast<float> (value)))
  {
  }

  static constexpr BFloat16
  from_bits (uint16_t raw) noexcept
  {
    BFloat16 out;
    out.bits = raw;
    return out;
  }

  operator float (void) const noexcept
  {
    return to_float32 (bits);
  }

  static uint16_t
  from_float32 (float value) noexcept
  {
    uint32_t u = std::bit_cast<uint32_t> (value);
    if ((u & 0x7FFFFFFFU) > 0x7F800000U)
      {
        return static_cast<uint16_t> ((u >> 16) | 0x0040U);
      }
    uint32_t lsb = (u >> 16) & 1U;
    u += 0x7FFFU + lsb;
    return static_cast<uint16_t> (u >> 16);
  }

  static float
  to_float32 (uint16_t raw) noexcept
  {
    uint32_t u = static_cast<uint32_t> (raw) << 16;
    return std::bit_cast<float> (u);
  }
};

/* Return the byte width of a scalar element of type DTYPE.  */
BASALT_API size_t
dtype_size (Dtype dtype) noexcept;

/* Return the short canonical name of DTYPE (e.g. "float32", "bfloat16").  */
BASALT_API const char *
dtype_name (Dtype dtype) noexcept;

/* Parse a canonical or alias string into a Dtype.  */
BASALT_API Dtype
dtype_from_name (std::string_view name);

/* Return true if DTYPE is Float16, BFloat16, Float32, or Float64.  */
BASALT_API bool
is_floating_point (Dtype dtype) noexcept;

/* Return true if DTYPE is Int32, Int64, or UInt8.  */
BASALT_API bool
is_integral (Dtype dtype) noexcept;

/* Compute the smallest common Dtype capable of representing operands of
   types A and B without narrowing.  */
BASALT_API Dtype
promote_dtypes (Dtype a, Dtype b) noexcept;

/* Map a C++ scalar type T to its corresponding Dtype enum constant.  */
template <typename T>
constexpr Dtype
dtype_of (void) noexcept
{
  using Clean = std::remove_cv_t<T>;
  if constexpr (std::is_same_v<Clean, float>)
    {
      return Dtype::Float32;
    }
  else if constexpr (std::is_same_v<Clean, double>)
    {
      return Dtype::Float64;
    }
  else if constexpr (std::is_same_v<Clean, Float16>)
    {
      return Dtype::Float16;
    }
  else if constexpr (std::is_same_v<Clean, BFloat16>)
    {
      return Dtype::BFloat16;
    }
  else if constexpr (std::is_same_v<Clean, int32_t>)
    {
      return Dtype::Int32;
    }
  else if constexpr (std::is_same_v<Clean, int64_t>)
    {
      return Dtype::Int64;
    }
  else if constexpr (std::is_same_v<Clean, uint8_t>)
    {
      return Dtype::UInt8;
    }
  else if constexpr (std::is_same_v<Clean, bool>)
    {
      return Dtype::Bool;
    }
  else
    {
      static_assert (sizeof (T) == 0, "Unsupported C++ scalar type for Dtype");
      return Dtype::Float32;
    }
}

} /* namespace basalt */

#endif /* BASALT_DTYPE_HPP */
