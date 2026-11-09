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

#ifndef BASALT_ERROR_HPP
#define BASALT_ERROR_HPP

/* Exception hierarchy and GNU-standard diagnostic reporting.
   Library diagnostics follow the format "basalt: file: message" on stderr,
   while command-line tools emit "<tool>: message" on stderr.  */

#include <stdexcept>
#include <string>
#include <basalt/config.hpp>

namespace basalt
{

/* Base exception class for all recoverable errors in Basalt Deep Learning.
   Inherits from std::runtime_error to provide strong exception safety across
   all public entry points.  */
class BASALT_API Basalt_Error : public std::runtime_error
{
public:
  explicit Basalt_Error (const std::string &message);
  Basalt_Error (const char *file, size_t line, const std::string &message);
  ~Basalt_Error (void) noexcept override;

  const std::string &
  file_name (void) const noexcept
  {
    return file_;
  }

  size_t
  line_number (void) const noexcept
  {
    return line_;
  }

private:
  std::string file_;
  size_t line_;
};

/* Raised when an argument violates preconditions.  */
class BASALT_API Invalid_Argument_Error : public Basalt_Error
{
public:
  using Basalt_Error::Basalt_Error;
};

/* Raised when tensor shapes or strides are incompatible.  */
class BASALT_API Shape_Error : public Basalt_Error
{
public:
  using Basalt_Error::Basalt_Error;
};

/* Raised when tensor data types do not match or are unsupported.  */
class BASALT_API Dtype_Error : public Basalt_Error
{
public:
  using Basalt_Error::Basalt_Error;
};

/* Raised when tensors reside on different devices or a backend fails.  */
class BASALT_API Device_Error : public Basalt_Error
{
public:
  using Basalt_Error::Basalt_Error;
};

/* Raised when file serialization or deserialization fails.  */
class BASALT_API IO_Error : public Basalt_Error
{
public:
  using Basalt_Error::Basalt_Error;
};

/* Raised when reverse-mode automatic differentiation encounters an invalid
   graph state (such as in-place mutation of a needed saved tensor).  */
class BASALT_API Autograd_Error : public Basalt_Error
{
public:
  using Basalt_Error::Basalt_Error;
};

/* Format and write "basalt: <file>: <message>\n" to stderr.  */
BASALT_API void
report_diagnostic (const char *file, const std::string &message) noexcept;

/* Format and write "<program_name>: <message>\n" to stderr.  */
BASALT_API void
report_tool_diagnostic (const char *program_name,
                        const std::string &message) noexcept;

/* Control whether throwing a Basalt_Error also logs to stderr.
   Disabled by default during silent unit test exception checks.  */
BASALT_API void
set_diagnostic_logging (bool enabled) noexcept;

BASALT_API bool
diagnostic_logging_enabled (void) noexcept;

} /* namespace basalt */

#define BASALT_THROW(exc_type, msg) \
  do \
    { \
      throw exc_type (__FILE__, static_cast<size_t> (__LINE__), (msg)); \
    } \
  while (0)

#define BASALT_ASSERT(cond, msg) \
  do \
    { \
      if (BASALT_INTERNAL_UNLIKELY (!(cond))) \
        { \
          throw ::basalt::Basalt_Error ( \
            __FILE__, static_cast<size_t> (__LINE__), (msg)); \
        } \
    } \
  while (0)

#define BASALT_ASSERT_ARG(cond, msg) \
  do \
    { \
      if (BASALT_INTERNAL_UNLIKELY (!(cond))) \
        { \
          throw ::basalt::Invalid_Argument_Error ( \
            __FILE__, static_cast<size_t> (__LINE__), (msg)); \
        } \
    } \
  while (0)

#define BASALT_ASSERT_SHAPE(cond, msg) \
  do \
    { \
      if (BASALT_INTERNAL_UNLIKELY (!(cond))) \
        { \
          throw ::basalt::Shape_Error ( \
            __FILE__, static_cast<size_t> (__LINE__), (msg)); \
        } \
    } \
  while (0)

#define BASALT_ASSERT_DTYPE(cond, msg) \
  do \
    { \
      if (BASALT_INTERNAL_UNLIKELY (!(cond))) \
        { \
          throw ::basalt::Dtype_Error ( \
            __FILE__, static_cast<size_t> (__LINE__), (msg)); \
        } \
    } \
  while (0)

#define BASALT_ASSERT_DEVICE(cond, msg) \
  do \
    { \
      if (BASALT_INTERNAL_UNLIKELY (!(cond))) \
        { \
          throw ::basalt::Device_Error ( \
            __FILE__, static_cast<size_t> (__LINE__), (msg)); \
        } \
    } \
  while (0)

#endif /* BASALT_ERROR_HPP */
