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

#ifndef DARSHANA_ERROR_HPP
#define DARSHANA_ERROR_HPP

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <variant>

namespace darshana::inline v1
{

/**
 * @brief Enumerates failure modes originating from the Vulkan GPU backend.
 * @pre None.
 * @post Immutable value type.
 * @note Thread-safety: Safe for concurrent reads.
 */
enum class GpuErrorCode : uint8_t
{
  DeviceLost = 0,
  OutOfMemory = 1,
  PipelineCreationFailed = 2,
  InitializationFailed = 3,
  TransferFailed = 4,
  SurfaceFailed = 5
};

/**
 * @brief Enumerates failure modes originating from plot specification or data.
 * @pre None.
 * @post Immutable value type.
 * @note Thread-safety: Safe for concurrent reads.
 */
enum class DataErrorCode : uint8_t
{
  InvalidSpec = 0,
  UnsupportedDataType = 1,
  EmptyDataset = 2,
  DimensionMismatch = 3,
  InvalidScaleDomain = 4,
  JsonParseError = 5
};

/**
 * @brief Enumerates failure modes originating from static vector export.
 * @pre None.
 * @post Immutable value type.
 * @note Thread-safety: Safe for concurrent reads.
 */
enum class ExportErrorCode : uint8_t
{
  UnsupportedFormat = 0,
  IoFailure = 1,
  BufferOverflow = 2,
  Cancelled = 3
};

/**
 * @brief Enumerates failure modes originating from the framework runtime.
 * @pre None.
 * @post Immutable value type.
 * @note Thread-safety: Safe for concurrent reads.
 */
enum class FrameworkErrorCode : uint8_t
{
  InvalidState = 0,
  ResourceExhaustion = 1,
  ConcurrencyViolation = 2,
  NotInitialized = 3
};

/**
 * @brief Describes a GPU error payload within darshana::Error.
 * @pre None.
 * @post Holds typed GPU error code and raw VkResult integer.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct GpuError
{
  GpuErrorCode code = GpuErrorCode::InitializationFailed;
  int32_t vk_result = 0;
};

/**
 * @brief Describes a data or specification error payload within Error.
 * @pre None.
 * @post Holds typed DataErrorCode and offending field identifier.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct DataError
{
  DataErrorCode code = DataErrorCode::InvalidSpec;
  std::string field_name;
};

/**
 * @brief Describes an export error payload within darshana::Error.
 * @pre None.
 * @post Holds typed ExportErrorCode and target output path.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct ExportError
{
  ExportErrorCode code = ExportErrorCode::IoFailure;
  std::string path;
};

/**
 * @brief Describes an internal framework error payload within darshana::Error.
 * @pre None.
 * @post Holds typed FrameworkErrorCode and platform status code.
 * @note Thread-safety: Value type; safe when not mutated concurrently.
 */
struct FrameworkError
{
  FrameworkErrorCode code = FrameworkErrorCode::InvalidState;
  int32_t platform_code = 0;
};

/**
 * @brief Discriminated union holding the specific domain of a Darshana Error.
 */
using ErrorVariant
    = std::variant<GpuError, DataError, ExportError, FrameworkError>;

/**
 * @brief Converts a raw Vulkan VkResult integer into a diagnostic string.
 * @param vk_result Signed 32-bit VkResult value returned by Vulkan.
 * @pre None.
 * @post State is unchanged.
 * @returns Static string_view naming the VkResult constant.
 * @note Thread-safety: Pure function; safe for concurrent calls.
 */
[[nodiscard]] auto
vk_result_to_string (int32_t vk_result) noexcept -> std::string_view;

/**
 * @brief Represents a structured, non-throwing error across Darshana APIs.
 * @pre Error message must follow GNU diagnostic formatting (capitalized,
 *      complete sentence, no trailing period).
 * @post Holds a valid ErrorVariant and human-readable diagnostic message.
 * @note Thread-safety: Const methods are safe for concurrent reads.
 */
class Error
{
public:
  /**
   * @brief Constructs a default framework error.
   * @pre None.
   * @post Initializes variant to FrameworkError.
   */
  Error () noexcept = default;

  /**
   * @brief Constructs an Error from a domain variant and diagnostic message.
   * @param domain Typed error domain payload.
   * @param message Human-readable diagnostic sentence without trailing period.
   * @pre None.
   * @post Stores domain and message in the Error instance.
   */
  Error (ErrorVariant domain, std::string message) noexcept;

  /**
   * @brief Creates a GPU-domain Error.
   * @param code Specific GpuErrorCode value.
   * @param vk_result Raw Vulkan VkResult code (0 if not applicable).
   * @param message Diagnostic sentence suggesting corrective action.
   * @pre None.
   * @post is_gpu_error() returns true on the returned object.
   * @returns Constructed Error for the GPU domain.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto gpu (GpuErrorCode code, int32_t vk_result,
                                 std::string message) noexcept -> Error;

  /**
   * @brief Creates a Data-domain Error.
   * @param code Specific DataErrorCode value.
   * @param field_name Name of the spec field or column that failed validation.
   * @param message Diagnostic sentence suggesting corrective action.
   * @pre None.
   * @post is_data_error() returns true on the returned object.
   * @returns Constructed Error for the Data domain.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto data (DataErrorCode code, std::string field_name,
                                  std::string message) noexcept -> Error;

  /**
   * @brief Creates an Export-domain Error.
   * @param code Specific ExportErrorCode value.
   * @param path Target file path or export identifier.
   * @param message Diagnostic sentence suggesting corrective action.
   * @pre None.
   * @post is_export_error() returns true on the returned object.
   * @returns Constructed Error for the Export domain.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto export_err (ExportErrorCode code, std::string path,
                                        std::string message) noexcept -> Error;

  /**
   * @brief Creates a Framework-domain Error.
   * @param code Specific FrameworkErrorCode value.
   * @param platform_code OS error code if applicable.
   * @param message Diagnostic sentence suggesting corrective action.
   * @pre None.
   * @post is_framework_error() returns true on the returned object.
   * @returns Constructed Error for the Framework domain.
   * @note Thread-safety: Pure factory function; thread-safe.
   */
  [[nodiscard]] static auto framework (FrameworkErrorCode code,
                                       int32_t platform_code,
                                       std::string message) noexcept -> Error;

  /**
   * @brief Retrieves the underlying error variant.
   * @pre None.
   * @post State is unchanged.
   * @returns Reference to the stored ErrorVariant.
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto variant () const noexcept -> const ErrorVariant &;

  /**
   * @brief Retrieves the human-readable error message.
   * @pre None.
   * @post State is unchanged.
   * @returns View of the diagnostic string.
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto message () const noexcept -> std::string_view;

  /**
   * @brief Retrieves the top-level category name for structured logging.
   * @pre None.
   * @post State is unchanged.
   * @returns One of "GPU", "Data", "Export", or "Framework".
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto category_name () const noexcept -> std::string_view;

  /**
   * @brief Checks whether the error originated from the GPU backend.
   * @pre None.
   * @post State is unchanged.
   * @returns True if variant holds GpuError.
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto is_gpu_error () const noexcept -> bool;

  /**
   * @brief Checks whether the error originated from data or spec validation.
   * @pre None.
   * @post State is unchanged.
   * @returns True if variant holds DataError.
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto is_data_error () const noexcept -> bool;

  /**
   * @brief Checks whether the error originated from plot export.
   * @pre None.
   * @post State is unchanged.
   * @returns True if variant holds ExportError.
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto is_export_error () const noexcept -> bool;

  /**
   * @brief Checks whether the error originated from the framework runtime.
   * @pre None.
   * @post State is unchanged.
   * @returns True if variant holds FrameworkError.
   * @note Thread-safety: Safe for concurrent reads.
   */
  [[nodiscard]] auto is_framework_error () const noexcept -> bool;

private:
  ErrorVariant domain_{ FrameworkError{} };
  std::string message_;
};

/**
 * @brief Standard return type for all fallible Darshana public APIs.
 */
template <typename T> using Result = std::expected<T, Error>;

/**
 * @brief Wraps a Vulkan API call and returns a typed GpuError on failure.
 */
#define DARSHANA_VK_CHECK(expr, err_code, context_msg)                        \
  do                                                                          \
    {                                                                         \
      const int32_t darshana_vk_res_ = static_cast<int32_t> (expr);           \
      if (darshana_vk_res_ != 0)                                              \
        {                                                                     \
          std::string full_msg_ (context_msg);                                \
          full_msg_ += " (VkResult=";                                         \
          full_msg_ += ::darshana::vk_result_to_string (darshana_vk_res_);    \
          full_msg_ += ")";                                                   \
          return std::unexpected (::darshana::Error::gpu (                    \
              (err_code), darshana_vk_res_, std::move (full_msg_)));          \
        }                                                                     \
    }                                                                         \
  while (false)

} /* namespace darshana::inline v1 */

#endif /* DARSHANA_ERROR_HPP */
