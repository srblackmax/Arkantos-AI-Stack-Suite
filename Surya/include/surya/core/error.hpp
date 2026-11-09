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

#ifndef SURYA_CORE_ERROR_HPP
#define SURYA_CORE_ERROR_HPP

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <utility>
#include <surya/core/config.hpp>

namespace surya::inline v1::core
{

/**
 * @brief Categorized status codes for recoverable runtime operations.
 * @note Thread-safety: Value type; safe across all threads.
 */
enum class Error_Code : uint32_t
{
  Ok = 0,
  Invalid_Argument = 1,
  Out_Of_Bounds = 2,
  Allocation_Failed = 3,
  Dimension_Mismatch = 4,
  Format_Mismatch = 5,
  Io_Error = 6,
  Codec_Error = 7,
  Unsupported_Operation = 8,
  Numerical_Failure = 9,
  Empty_Input = 10,
  Vram_Exhausted = 11,
  Cuda_Error = 12,
  Graph_Capture_Error = 13,
  Queue_Full = 14,
  Device_Unavailable = 15
};

/**
 * @brief Return a human-readable static string for an Error_Code.
 * @pre None.
 * @post Returns a valid string_view naming CODE.
 * @returns Static string_view of the error enumerator.
 * @note Thread-safety: Safe to call concurrently from any thread.
 */
constexpr std::string_view
error_code_name (Error_Code code) noexcept
{
  switch (code)
    {
    case Error_Code::Ok:
      return "Ok";
    case Error_Code::Invalid_Argument:
      return "Invalid_Argument";
    case Error_Code::Out_Of_Bounds:
      return "Out_Of_Bounds";
    case Error_Code::Allocation_Failed:
      return "Allocation_Failed";
    case Error_Code::Dimension_Mismatch:
      return "Dimension_Mismatch";
    case Error_Code::Format_Mismatch:
      return "Format_Mismatch";
    case Error_Code::Io_Error:
      return "Io_Error";
    case Error_Code::Codec_Error:
      return "Codec_Error";
    case Error_Code::Unsupported_Operation:
      return "Unsupported_Operation";
    case Error_Code::Numerical_Failure:
      return "Numerical_Failure";
    case Error_Code::Empty_Input:
      return "Empty_Input";
    case Error_Code::Vram_Exhausted:
      return "Vram_Exhausted";
    case Error_Code::Cuda_Error:
      return "Cuda_Error";
    case Error_Code::Graph_Capture_Error:
      return "Graph_Capture_Error";
    case Error_Code::Queue_Full:
      return "Queue_Full";
    case Error_Code::Device_Unavailable:
      return "Device_Unavailable";
    }
  return "Unknown_Error";
}

/**
 * @brief Value-semantic error descriptor carrying an Error_Code and message.
 * @note Thread-safety: Distinct instances may be read/mutated concurrently.
 */
class SURYA_API Error
{
public:
  Error (void) noexcept = default;

  Error (Error_Code code, std::string message) noexcept
    : code_ (code), message_ (std::move (message))
  {
  }

  Error_Code
  code (void) const noexcept
  {
    return code_;
  }

  const std::string &
  message (void) const noexcept
  {
    return message_;
  }

  std::string
  to_string (void) const
  {
    std::string out (error_code_name (code_));
    if (!message_.empty ())
      {
        out += ": ";
        out += message_;
      }
    return out;
  }

private:
  Error_Code code_ = Error_Code::Ok;
  std::string message_;
};

/**
 * @brief Canonical alias for Surya error payloads.
 */
using SuryaError = Error;

/**
 * @brief Monadic result alias for all fallible Surya routines.
 */
template <typename T>
using Result = std::expected<T, SuryaError>;

/**
 * @brief Helper to construct an unexpected SuryaError value without throwing.
 * @pre None.
 * @post Wraps CODE and MESSAGE into std::unexpected<SuryaError>.
 * @returns std::unexpected<SuryaError> ready for Result<T> return.
 * @note Thread-safety: Safe to call concurrently from any thread.
 */
inline std::unexpected<SuryaError>
make_error (Error_Code code, std::string message) noexcept
{
  return std::unexpected<SuryaError> (SuryaError (code, std::move (message)));
}

} /* namespace surya::inline v1::core */

namespace surya::inline v1
{
using SuryaError = core::SuryaError;
using Error_Code = core::Error_Code;
template <typename T>
using Result = core::Result<T>;
} /* namespace surya::inline v1 */

#endif /* SURYA_CORE_ERROR_HPP */
