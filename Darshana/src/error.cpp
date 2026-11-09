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

#include <darshana/error.hpp>

#include <utility>

namespace darshana::inline v1
{

/* Maps a raw Vulkan VkResult integer to a human-readable constant name.  */
auto
vk_result_to_string (int32_t vk_result) noexcept -> std::string_view
{
  switch (vk_result)
    {
    case 0:
      return "VK_SUCCESS";
    case 1:
      return "VK_NOT_READY";
    case 2:
      return "VK_TIMEOUT";
    case 3:
      return "VK_EVENT_SET";
    case 4:
      return "VK_EVENT_RESET";
    case 5:
      return "VK_INCOMPLETE";
    case -1:
      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case -2:
      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case -3:
      return "VK_ERROR_INITIALIZATION_FAILED";
    case -4:
      return "VK_ERROR_DEVICE_LOST";
    case -5:
      return "VK_ERROR_MEMORY_MAP_FAILED";
    case -6:
      return "VK_ERROR_LAYER_NOT_PRESENT";
    case -7:
      return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case -8:
      return "VK_ERROR_FEATURE_NOT_PRESENT";
    case -9:
      return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case -10:
      return "VK_ERROR_TOO_MANY_OBJECTS";
    case -11:
      return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case -12:
      return "VK_ERROR_FRAGMENTED_POOL";
    default:
      return "VK_ERROR_UNKNOWN";
    }
}

/* Initializes an Error from a typed domain variant and diagnostic text.  */
Error::Error (ErrorVariant domain, std::string message) noexcept
    : domain_ (std::move (domain)), message_ (std::move (message))
{
}

/* Constructs a GPU-domain Error.  */
auto
Error::gpu (GpuErrorCode code, int32_t vk_result,
            std::string message) noexcept -> Error
{
  return Error (GpuError{ code, vk_result }, std::move (message));
}

/* Constructs a Data-domain Error.  */
auto
Error::data (DataErrorCode code, std::string field_name,
             std::string message) noexcept -> Error
{
  return Error (DataError{ code, std::move (field_name) },
                std::move (message));
}

/* Constructs an Export-domain Error.  */
auto
Error::export_err (ExportErrorCode code, std::string path,
                   std::string message) noexcept -> Error
{
  return Error (ExportError{ code, std::move (path) }, std::move (message));
}

/* Constructs a Framework-domain Error.  */
auto
Error::framework (FrameworkErrorCode code, int32_t platform_code,
                  std::string message) noexcept -> Error
{
  return Error (FrameworkError{ code, platform_code }, std::move (message));
}

/* Returns the underlying discriminated error variant.  */
auto
Error::variant () const noexcept -> const ErrorVariant &
{
  return domain_;
}

/* Returns the human-readable diagnostic string.  */
auto
Error::message () const noexcept -> std::string_view
{
  return message_;
}

/* Returns the top-level category identifier string.  */
auto
Error::category_name () const noexcept -> std::string_view
{
  if (std::holds_alternative<GpuError> (domain_))
    {
      return "GPU";
    }
  if (std::holds_alternative<DataError> (domain_))
    {
      return "Data";
    }
  if (std::holds_alternative<ExportError> (domain_))
    {
      return "Export";
    }
  return "Framework";
}

/* Checks whether the error belongs to the GPU domain.  */
auto
Error::is_gpu_error () const noexcept -> bool
{
  return std::holds_alternative<GpuError> (domain_);
}

/* Checks whether the error belongs to the Data domain.  */
auto
Error::is_data_error () const noexcept -> bool
{
  return std::holds_alternative<DataError> (domain_);
}

/* Checks whether the error belongs to the Export domain.  */
auto
Error::is_export_error () const noexcept -> bool
{
  return std::holds_alternative<ExportError> (domain_);
}

/* Checks whether the error belongs to the Framework domain.  */
auto
Error::is_framework_error () const noexcept -> bool
{
  return std::holds_alternative<FrameworkError> (domain_);
}

} /* namespace darshana::inline v1 */
