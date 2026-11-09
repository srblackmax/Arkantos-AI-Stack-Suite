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
#include <basalt/device.hpp>
#include <basalt/error.hpp>
#include "bellows-compat/bellows_bridge.hpp"
#include <charconv>

namespace basalt
{

Device
Device::parse (std::string_view spec)
{
  if (spec == "cpu" || spec == "CPU" || spec == "cpu:0")
    {
      return Device::cpu ();
    }
  if (spec == "cuda" || spec == "CUDA")
    {
      return Device::cuda (0);
    }
  if (spec.starts_with ("cuda:") || spec.starts_with ("CUDA:"))
    {
      std::string_view idx_part = spec.substr (5);
      size_t idx = 0;
      auto [ptr, ec] = std::from_chars (
        idx_part.data (), idx_part.data () + idx_part.size (), idx);
      if (ec != std::errc () || ptr != idx_part.data () + idx_part.size ())
        {
          BASALT_THROW (Device_Error,
                        "invalid CUDA device index in: " + std::string (spec));
        }
      return Device::cuda (idx);
    }
  BASALT_THROW (Device_Error,
                "unrecognized device specification: " + std::string (spec));
}

std::string
Device::str (void) const
{
  if (type_ == Device_Type::CPU)
    {
      return "cpu";
    }
  return "cuda:" + std::to_string (index_);
}

bool
cuda_is_available (void) noexcept
{
  return cuda_device_count () > 0;
}

bool
cuda_has_hardware_driver (void) noexcept
{
  return bellows_compat::Cuda_Runtime_Driver::instance ().has_hardware ();
}

size_t
cuda_device_count (void) noexcept
{
  return bellows_compat::Cuda_Runtime_Driver::instance ().device_count ();
}

void
set_virtual_cuda_device_count (size_t count) noexcept
{
  bellows_compat::Cuda_Runtime_Driver::instance ()
    .set_virtual_device_count (count);
}

void
device_synchronize (Device device)
{
  if (device.is_cuda ())
    {
      if (device.index () >= cuda_device_count ())
        {
          BASALT_THROW (Device_Error,
                        "CUDA device ordinal out of range: " + device.str ());
        }
      bellows_compat::Cuda_Runtime_Driver::instance ()
        .synchronize (device.index ());
    }
}

} /* namespace basalt */
