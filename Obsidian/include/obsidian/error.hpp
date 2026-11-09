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

#ifndef OBSIDIAN_ERROR_HPP
#define OBSIDIAN_ERROR_HPP

/**
 * @file error.hpp
 * @brief Recoverable error representation and std::expected Result alias.
 */

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <utility>
#include <obsidian/config.hpp>

namespace obsidian
{

/**
 * @brief Categorized error codes for recoverable runtime failures.
 */
enum class Error_Code : uint8_t
{
  Ok = 0,
  Invalid_Argument = 1,
  Out_Of_Range = 2,
  Shape_Mismatch = 3,
  Dtype_Mismatch = 4,
  Device_Mismatch = 5,
  IO_Failure = 6,
  Corrupted_Format = 7,
  Invalid_UTF8 = 8,
  Vocab_Not_Found = 9,
  Not_Initialized = 10,
  Convergence_Failure = 11,
  Backend_Failure = 12
};

/**
 * @brief Return a static human-readable name for CODE.
 * @pre CODE must be a valid Error_Code enumerator.
 * @complexity O(1) time and O(1) space.
 */
OBSIDIAN_API const char *
error_code_name (Error_Code code) noexcept;

/**
 * @brief Structured value type describing a recoverable error condition.
 *
 * Used as the error channel of obsidian::Result<T> (std::expected<T, Error>)
 * so user input errors never abort the process.
 */
class OBSIDIAN_API Error
{
public:
  /**
   * @brief Construct an Error with code, message, and optional source info.
   * @pre MESSAGE may be any valid UTF-8 string view.
   * @complexity O(N) time and space in message length N.
   */
  Error (Error_Code code, std::string message,
         const char *file = "", size_t line = 0)
    : code_ (code),
      message_ (std::move (message)),
      file_ (file != nullptr ? file : ""),
      line_ (line)
  {
  }

  /**
   * @brief Return the categorical error code.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  Error_Code
  code (void) const noexcept
  {
    return code_;
  }

  /**
   * @brief Return the descriptive error message.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  const std::string &
  message (void) const noexcept
  {
    return message_;
  }

  /**
   * @brief Return the source file where the error was produced.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  const char *
  file (void) const noexcept
  {
    return file_;
  }

  /**
   * @brief Return the line number where the error was produced.
   * @pre None.
   * @complexity O(1) time and O(1) space.
   */
  size_t
  line (void) const noexcept
  {
    return line_;
  }

  /**
   * @brief Format the error in GNU standard diagnostic format.
   * @pre None.
   * @complexity O(N) time and space in formatted message length.
   */
  std::string
  format (void) const;

private:
  Error_Code code_;
  std::string message_;
  const char *file_;
  size_t line_;
};

/**
 * @brief Standard result alias wrapping std::expected<T, Error>.
 */
template <typename T>
using Result = std::expected<T, Error>;

/**
 * @brief Construct an unexpected Error value for returning from Result<T>.
 * @pre MSG must be a valid string view.
 * @complexity O(N) time in message length N.
 */
inline std::unexpected<Error>
make_error (Error_Code code, std::string_view msg,
            const char *file = "", size_t line = 0)
{
  return std::unexpected<Error> (
    Error (code, std::string (msg), file, line));
}

} /* namespace obsidian */

#define OBSIDIAN_ERROR(code, msg) \
  ::obsidian::make_error ((code), (msg), __FILE__, \
                          static_cast<size_t> (__LINE__))

#endif /* OBSIDIAN_ERROR_HPP */
