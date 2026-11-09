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

#ifndef SPHINX_TOOLS_HPP
#define SPHINX_TOOLS_HPP

/**
 * @file tools.hpp
 * @brief Sandboxed code, file, shell, and SSH tools registered via Sutra
 *        SUTRA_REGISTER_TOOL and tracked via Yantra Run.
 * @note Exposes zero libssh2_* symbols in public headers.
 */

#include <sutra/error.hpp>
#include <sutra/tool.hpp>
#include <yantra/track.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace sphinx::inline v1
{

/**
 * @brief Default sandbox directory for Sphinx file and shell tools.
 */
inline constexpr std::string_view kDefaultSandboxDir
    = "~/sphinx-workspace";

/**
 * @brief Configures the active sandbox directory and optional Yantra Run
 *        pointer used by all Sphinx tool invocations on the current thread or
 *        process.
 */
struct ToolEnvironment
{
  /**
   * @brief Sets the active sandbox directory (creates it if missing).
   */
  static auto set_sandbox_dir (const std::filesystem::path &dir) noexcept
      -> sutra::Result<std::filesystem::path>;

  /**
   * @brief Returns the canonical active sandbox directory.
   */
  [[nodiscard]] static auto sandbox_dir () noexcept -> std::filesystem::path;

  /**
   * @brief Binds an active yantra::Run and step counter for tool telemetry.
   */
  static void bind_yantra_run (yantra::Run *run, uint64_t step = 0U) noexcept;

  /**
   * @brief Sets the subprocess execution timeout in milliseconds (default
   *        30000 ms; configurable for fast unit-test timeout verification).
   */
  static void set_command_timeout_ms (uint32_t timeout_ms) noexcept;

  /**
   * @brief Returns the configured subprocess execution timeout in ms.
   */
  [[nodiscard]] static auto command_timeout_ms () noexcept -> uint32_t;

  /**
   * @brief Sets the run_readonly execution timeout in milliseconds (default
   *        15000 ms; configurable for fast unit-test timeout verification).
   */
  static void set_readonly_timeout_ms (uint32_t timeout_ms) noexcept;

  /**
   * @brief Returns the configured run_readonly execution timeout in ms.
   */
  [[nodiscard]] static auto readonly_timeout_ms () noexcept -> uint32_t;
};

/**
 * @brief Validates that a candidate path resolves inside the active sandbox.
 * @returns Canonical path or AgentError::ToolSandboxViolation.
 */
[[nodiscard]] auto
resolve_sandboxed_path (std::string_view user_path) noexcept
    -> sutra::Result<std::filesystem::path>;

/**
 * @brief Validates that a shell command matches the Sphinx command allowlist
 *        and contains no shell metacharacter injection.
 */
[[nodiscard]] auto
validate_allowed_command (std::string_view cmd) noexcept
    -> sutra::Result<void>;

/**
 * @brief Validates that an SSH command does not match the dangerous command
 *        denylist (rm -rf, dd, mkfs, shutdown, reboot, > /dev/).
 */
[[nodiscard]] auto
validate_ssh_command (std::string_view cmd) noexcept -> sutra::Result<void>;

/**
 * @brief Create a directory inside the Sphinx sandbox.
 *
 * The path is interpreted relative to SPHINX_SANDBOX (or
 * ~/sphinx-workspace if unset). Absolute paths and paths containing
 * ".." are rejected. Parent directories are created recursively
 * (mkdir -p semantics). If the target already exists as a directory,
 * returns success (idempotent). If it exists as a file, returns
 * ToolSandboxViolation.
 *
 * @param path Relative path to the directory to create.
 * @returns The absolute path of the created (or existing) directory.
 */
[[nodiscard]] auto
make_directory (std::string_view path) noexcept
    -> sutra::Result<std::filesystem::path>;

/**
 * @brief Result of a read-only diagnostic command invocation.
 */
struct CommandResult
{
  int exit_code = 0;
  std::string stdout_str;
  std::string stderr_str;
  std::size_t truncated_stdout_bytes = 0U;
  std::size_t truncated_stderr_bytes = 0U;
};

/**
 * @brief Execute a read-only diagnostic command.
 *
 * The command must be in the diagnostic allowlist (section 3.2). Arguments
 * are validated against per-command subcommand restrictions. The
 * process is spawned via execvp with an argv array - no shell is
 * ever invoked, so shell metacharacters in arguments are inert.
 *
 * @param command  Bare command name (no path separators).
 * @param args     Argument vector.
 * @returns        {exit_code, stdout, stderr} with output capped at
 *                 4 KB per stream.
 */
[[nodiscard]] auto
run_readonly (std::string_view command,
              std::span<const std::string_view> args) noexcept
    -> sutra::Result<CommandResult>;

/**
 * @brief Read a system file from an allowlisted prefix.
 *
 * Paths inside /etc, /var/log, /proc, /sys, ~/.config, and the
 * specific dotfiles ~/.bashrc, ~/.zshrc, ~/.profile, ~/.ssh/config
 * are readable. Secret material is refused even inside those
 * prefixes (see denylist). Files larger than 256 KB are truncated
 * with a marker. Binary files are refused.
 *
 * @param path Absolute path.
 * @returns File contents, or an error naming the path and the
 *          reason it was refused.
 */
[[nodiscard]] auto
read_file_system (std::string_view path) noexcept
    -> sutra::Result<std::string>;

/**
 * @brief Tool 1: read_file(path, max_lines, start_line) -> string.
 */
[[nodiscard]] auto
tool_read_file (const sutra::ToolContext &ctx, std::string path,
                int64_t max_lines, int64_t start_line = 1) noexcept
    -> sutra::Result<std::string>;

/**
 * @brief Tool 2: write_file(path, content) -> string.
 */
[[nodiscard]] auto
tool_write_file (const sutra::ToolContext &ctx, std::string path,
                 std::string content) noexcept -> sutra::Result<std::string>;

/**
 * @brief Tool 2b: edit_file(path, target, replacement) -> string.
 */
[[nodiscard]] auto
tool_edit_file (const sutra::ToolContext &ctx, std::string path,
                std::string target, std::string replacement) noexcept
    -> sutra::Result<std::string>;

/**
 * @brief Tool 3: list_dir(path) -> string.
 */
[[nodiscard]] auto
tool_list_dir (const sutra::ToolContext &ctx,
               std::string path) noexcept -> sutra::Result<std::string>;

/**
 * @brief Tool 4: run_command(cmd, cwd) -> string.
 */
[[nodiscard]] auto
tool_run_command (const sutra::ToolContext &ctx, std::string cmd,
                  std::string cwd) noexcept -> sutra::Result<std::string>;

/**
 * @brief Tool 5: compile_check(file, flags) -> string.
 */
[[nodiscard]] auto
tool_compile_check (const sutra::ToolContext &ctx, std::string file,
                    std::string flags) noexcept -> sutra::Result<std::string>;

/**
 * @brief Tool 6: ssh_exec(host, cmd) -> string (via libssh2 dlopen).
 */
[[nodiscard]] auto
tool_ssh_exec (const sutra::ToolContext &ctx, std::string host,
               std::string cmd) noexcept -> sutra::Result<std::string>;

/**
 * @brief Tool 7: ssh_test(host) -> string ("ok" or error via libssh2).
 */
[[nodiscard]] auto
tool_ssh_test (const sutra::ToolContext &ctx,
               std::string host) noexcept -> sutra::Result<std::string>;

/**
 * @brief Tool 8: echo(text) -> string.
 */
[[nodiscard]] auto
tool_echo (const sutra::ToolContext &ctx,
           std::string text) noexcept -> sutra::Result<std::string>;

/**
 * @brief Tool 9: make_directory(path) -> string (absolute path).
 */
[[nodiscard]] auto
tool_make_directory (const sutra::ToolContext &ctx,
                     std::string path) noexcept -> sutra::Result<std::string>;

/**
 * @brief Tool 10: run_readonly(command, args) -> string.
 */
[[nodiscard]] auto
tool_run_readonly (const sutra::ToolContext &ctx, std::string command,
                   std::string args) noexcept -> sutra::Result<std::string>;

/**
 * @brief Tool 11: read_file_system(path) -> string.
 */
[[nodiscard]] auto
tool_read_file_system (const sutra::ToolContext &ctx,
                       std::string path) noexcept
    -> sutra::Result<std::string>;

/**
 * @brief Tool 12: remove_file(path) -> string.
 *
 * Removes a file or directory inside SPHINX_SANDBOX, or clears all
 * non-memory workspace files when path is "*" or "all" or "workspace".
 */
[[nodiscard]] auto
tool_remove_file (const sutra::ToolContext &ctx,
                  std::string path) noexcept -> sutra::Result<std::string>;

/**
 * @brief Registers all 12 Sphinx tools into the given Sutra ToolRegistry.
 */
[[nodiscard]] auto
register_all_sphinx_tools (sutra::ToolRegistry &registry) noexcept
    -> sutra::Result<void>;

} /* namespace sphinx::inline v1 */

#endif /* SPHINX_TOOLS_HPP */
