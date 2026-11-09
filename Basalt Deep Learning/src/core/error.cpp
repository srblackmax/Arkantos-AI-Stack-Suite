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
#include <basalt/error.hpp>
#include <atomic>
#include <cstdio>
#include <cstring>

namespace basalt
{

namespace
{

/* Strip leading directory components so diagnostics stay concise.  */
const char *
basename_only (const char *path) noexcept
{
  if (path == nullptr)
    {
      return "unknown";
    }
  const char *slash = std::strrchr (path, '/');
  return (slash != nullptr) ? (slash + 1) : path;
}

std::atomic<bool> &
log_flag (void) noexcept
{
  static std::atomic<bool> enabled{false};
  return enabled;
}

} /* namespace */

Basalt_Error::Basalt_Error (const std::string &message)
  : std::runtime_error ("basalt: " + message),
    file_ ("unknown"),
    line_ (0)
{
  if (diagnostic_logging_enabled ())
    {
      report_diagnostic ("unknown", message);
    }
}

Basalt_Error::Basalt_Error (const char *file, size_t line,
                            const std::string &message)
  : std::runtime_error (std::string ("basalt: ")
                        + basename_only (file) + ": " + message),
    file_ (basename_only (file)),
    line_ (line)
{
  if (diagnostic_logging_enabled ())
    {
      report_diagnostic (file_.c_str (), message);
    }
}

Basalt_Error::~Basalt_Error (void) noexcept = default;

void
report_diagnostic (const char *file, const std::string &message) noexcept
{
  const char *short_file = basename_only (file);
  std::fprintf (stderr, "basalt: %s: %s\n", short_file, message.c_str ());
}

void
report_tool_diagnostic (const char *program_name,
                        const std::string &message) noexcept
{
  const char *name = (program_name != nullptr) ? program_name : "basalt";
  std::fprintf (stderr, "%s: %s\n", name, message.c_str ());
}

void
set_diagnostic_logging (bool enabled) noexcept
{
  log_flag ().store (enabled, std::memory_order_relaxed);
}

bool
diagnostic_logging_enabled (void) noexcept
{
  return log_flag ().load (std::memory_order_relaxed);
}

} /* namespace basalt */
