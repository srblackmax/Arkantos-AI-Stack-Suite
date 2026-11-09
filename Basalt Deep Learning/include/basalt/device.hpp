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

#ifndef BASALT_DEVICE_HPP
#define BASALT_DEVICE_HPP

/* Execution device abstraction covering CPU (SIMD + parallel thread pool +
   optional BLAS) and CUDA (via runtime-loaded libcuda/libcublas driver with
   deterministic device memory isolation).  */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <basalt/config.hpp>

namespace basalt
{

/* Supported compute device families.  */
enum class Device_Type : uint8_t
{
  CPU = 0,
  CUDA = 1
};

/* Value type identifying a specific execution device and ordinal index.  */
class BASALT_API Device
{
public:
  constexpr Device (void) noexcept
    : type_ (Device_Type::CPU),
      index_ (0)
  {
  }

  constexpr Device (Device_Type type, size_t index = 0) noexcept
    : type_ (type),
      index_ (index)
  {
  }

  static constexpr Device
  cpu (void) noexcept
  {
    return Device (Device_Type::CPU, 0);
  }

  static constexpr Device
  cuda (size_t index = 0) noexcept
  {
    return Device (Device_Type::CUDA, index);
  }

  static Device
  parse (std::string_view spec);

  constexpr Device_Type
  type (void) const noexcept
  {
    return type_;
  }

  constexpr size_t
  index (void) const noexcept
  {
    return index_;
  }

  constexpr bool
  is_cpu (void) const noexcept
  {
    return type_ == Device_Type::CPU;
  }

  constexpr bool
  is_cuda (void) const noexcept
  {
    return type_ == Device_Type::CUDA;
  }

  std::string
  str (void) const;

  constexpr bool
  operator== (const Device &other) const noexcept
  {
    return type_ == other.type_ && index_ == other.index_;
  }

  constexpr bool
  operator!= (const Device &other) const noexcept
  {
    return !(*this == other);
  }

private:
  Device_Type type_;
  size_t index_;
};

/* Return true if the CUDA backend is available (either via a runtime-loaded
   hardware driver or active virtual CUDA devices).  */
BASALT_API bool
cuda_is_available (void) noexcept;

/* Return true if a physical NVIDIA CUDA driver (libcuda.so.1) was loaded at
   runtime with at least one hardware GPU.  */
BASALT_API bool
cuda_has_hardware_driver (void) noexcept;

/* Return the number of addressable CUDA devices.  */
BASALT_API size_t
cuda_device_count (void) noexcept;

/* Configure the number of virtual CUDA devices when no physical GPU is
   present (defaults to 2 so cross-backend and multi-device tests work on
   every host).  */
BASALT_API void
set_virtual_cuda_device_count (size_t count) noexcept;

/* Synchronize all pending asynchronous work on DEVICE.  */
BASALT_API void
device_synchronize (Device device = Device::cuda (0));

} /* namespace basalt */

#endif /* BASALT_DEVICE_HPP */
