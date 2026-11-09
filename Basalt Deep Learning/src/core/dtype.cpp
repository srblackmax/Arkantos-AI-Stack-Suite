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

#include "config.h"
#include <basalt/dtype.hpp>
#include <basalt/error.hpp>
#include <string>

namespace basalt
{

uint16_t
Float16::from_float32 (float value) noexcept
{
  uint32_t f = std::bit_cast<uint32_t> (value);
  uint32_t sign = (f >> 16) & 0x8000U;
  int32_t exp = static_cast<int32_t> ((f >> 23) & 0xFFU) - 127 + 15;
  uint32_t mant = f & 0x007FFFFFU;

  if (exp <= 0)
    {
      if (exp < -10)
        {
          return static_cast<uint16_t> (sign);
        }
      mant |= 0x00800000U;
      uint32_t shift = static_cast<uint32_t> (14 - exp);
      uint32_t round_bit = 1U << (shift - 1U);
      uint32_t mask = round_bit - 1U;
      uint32_t half_mant = mant >> shift;
      if ((mant & round_bit) != 0U
          && ((mant & mask) != 0U || (half_mant & 1U) != 0U))
        {
          ++half_mant;
        }
      return static_cast<uint16_t> (sign | half_mant);
    }
  else if (exp == 0xFF - 127 + 15)
    {
      if (mant == 0U)
        {
          return static_cast<uint16_t> (sign | 0x7C00U);
        }
      mant >>= 13;
      return static_cast<uint16_t> (sign | 0x7C00U | mant | (mant == 0U));
    }

  uint32_t half_mant = mant >> 13;
  uint32_t rem = mant & 0x1FFFU;
  if (rem > 0x1000U || (rem == 0x1000U && (half_mant & 1U) != 0U))
    {
      ++half_mant;
      if (half_mant == 0x0400U)
        {
          half_mant = 0U;
          ++exp;
        }
    }
  if (exp >= 31)
    {
      return static_cast<uint16_t> (sign | 0x7C00U);
    }
  return static_cast<uint16_t> (
    sign | (static_cast<uint32_t> (exp) << 10) | half_mant);
}

float
Float16::to_float32 (uint16_t raw) noexcept
{
  uint32_t sign = (static_cast<uint32_t> (raw & 0x8000U)) << 16;
  uint32_t exp = (raw >> 10) & 0x1FU;
  uint32_t mant = raw & 0x03FFU;

  if (exp == 0U)
    {
      if (mant == 0U)
        {
          return std::bit_cast<float> (sign);
        }
      exp = 1U;
      while ((mant & 0x0400U) == 0U)
        {
          mant <<= 1;
          --exp;
        }
      mant &= 0x03FFU;
      uint32_t f_exp = (exp + (127U - 15U)) << 23;
      uint32_t f_mant = mant << 13;
      return std::bit_cast<float> (sign | f_exp | f_mant);
    }
  else if (exp == 31U)
    {
      uint32_t f_mant = mant << 13;
      return std::bit_cast<float> (sign | 0x7F800000U | f_mant);
    }

  uint32_t f_exp = (exp + (127U - 15U)) << 23;
  uint32_t f_mant = mant << 13;
  return std::bit_cast<float> (sign | f_exp | f_mant);
}

size_t
dtype_size (Dtype dtype) noexcept
{
  switch (dtype)
    {
    case Dtype::Float32:
    case Dtype::Int32:
      return 4;
    case Dtype::Float64:
    case Dtype::Int64:
      return 8;
    case Dtype::Float16:
    case Dtype::BFloat16:
      return 2;
    case Dtype::UInt8:
    case Dtype::Bool:
      return 1;
    }
  return 4;
}

const char *
dtype_name (Dtype dtype) noexcept
{
  switch (dtype)
    {
    case Dtype::Float32:
      return "float32";
    case Dtype::Float64:
      return "float64";
    case Dtype::Float16:
      return "float16";
    case Dtype::BFloat16:
      return "bfloat16";
    case Dtype::Int32:
      return "int32";
    case Dtype::Int64:
      return "int64";
    case Dtype::UInt8:
      return "uint8";
    case Dtype::Bool:
      return "bool";
    }
  return "unknown";
}

Dtype
dtype_from_name (std::string_view name)
{
  if (name == "float32" || name == "f32" || name == "F32")
    {
      return Dtype::Float32;
    }
  if (name == "float64" || name == "f64" || name == "F64")
    {
      return Dtype::Float64;
    }
  if (name == "float16" || name == "f16" || name == "F16")
    {
      return Dtype::Float16;
    }
  if (name == "bfloat16" || name == "bf16" || name == "BF16")
    {
      return Dtype::BFloat16;
    }
  if (name == "int32" || name == "i32" || name == "I32")
    {
      return Dtype::Int32;
    }
  if (name == "int64" || name == "i64" || name == "I64")
    {
      return Dtype::Int64;
    }
  if (name == "uint8" || name == "u8" || name == "U8")
    {
      return Dtype::UInt8;
    }
  if (name == "bool" || name == "BOOL")
    {
      return Dtype::Bool;
    }
  BASALT_THROW (Dtype_Error,
                "unrecognized dtype name: " + std::string (name));
}

bool
is_floating_point (Dtype dtype) noexcept
{
  return dtype == Dtype::Float32 || dtype == Dtype::Float64
         || dtype == Dtype::Float16 || dtype == Dtype::BFloat16;
}

bool
is_integral (Dtype dtype) noexcept
{
  return dtype == Dtype::Int32 || dtype == Dtype::Int64
         || dtype == Dtype::UInt8;
}

Dtype
promote_dtypes (Dtype a, Dtype b) noexcept
{
  if (a == b)
    {
      return a;
    }
  if (a == Dtype::Float64 || b == Dtype::Float64)
    {
      return Dtype::Float64;
    }
  if (a == Dtype::Float32 || b == Dtype::Float32)
    {
      return Dtype::Float32;
    }
  if ((a == Dtype::Float16 && b == Dtype::BFloat16)
      || (a == Dtype::BFloat16 && b == Dtype::Float16))
    {
      return Dtype::Float32;
    }
  if (a == Dtype::Float16 || b == Dtype::Float16)
    {
      return Dtype::Float16;
    }
  if (a == Dtype::BFloat16 || b == Dtype::BFloat16)
    {
      return Dtype::BFloat16;
    }
  if (a == Dtype::Int64 || b == Dtype::Int64)
    {
      return Dtype::Int64;
    }
  if (a == Dtype::Int32 || b == Dtype::Int32)
    {
      return Dtype::Int32;
    }
  if (a == Dtype::UInt8 || b == Dtype::UInt8)
    {
      return Dtype::UInt8;
    }
  return Dtype::Bool;
}

} /* namespace basalt */
