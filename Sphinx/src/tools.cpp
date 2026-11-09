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

#include <sphinx/tools.hpp>
#include <yantra/observability.hpp>

#include <arpa/inet.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <libssh2.h>
#include <netdb.h>
#include <poll.h>
#include <pwd.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sphinx::inline v1
{

namespace
{

constexpr std::size_t kMaxToolOutputBytes = 262144U;
constexpr std::size_t kMaxReadonlyStreamBytes = 4096U;
constexpr uint32_t kDefaultCommandTimeoutMs = 30000U;
constexpr uint32_t kDefaultReadonlyTimeoutMs = 15000U;
constexpr uint32_t kMaxSigkillGraceMs = 5000U;
constexpr int kPollSliceMs = 25;
constexpr uint16_t kDefaultSshPort = 22U;

struct ToolGlobalState
{
  std::mutex mtx;
  std::filesystem::path sandbox_dir{ std::string (kDefaultSandboxDir) };
  yantra::Run *active_run = nullptr;
  uint64_t active_step = 0U;
  std::atomic<uint32_t> command_timeout_ms{ kDefaultCommandTimeoutMs };
  std::atomic<uint32_t> readonly_timeout_ms{ kDefaultReadonlyTimeoutMs };
  std::unordered_map<std::string, uint64_t> error_counts;
};

auto
global_tool_state () noexcept -> ToolGlobalState &
{
  static ToolGlobalState state;
  return state;
}

/* Computes BLAKE3 hex digest for a string_view.  */
[[nodiscard]] auto
hash_string_blake3 (std::string_view sv) noexcept -> std::string
{
  const auto *raw = reinterpret_cast<const std::byte *> (sv.data ());
  return yantra::compute_blake3_hex (
      std::span<const std::byte> (raw, sv.size ()));
}

/* Records tool execution metrics and hashes to the active Yantra Run.  */
void
record_tool_telemetry (std::string_view tool_name, std::string_view input_str,
                       std::string_view output_str, double duration_ms,
                       bool is_error) noexcept
{
  auto &st = global_tool_state ();
  const std::scoped_lock lock (st.mtx);
  uint64_t err_total = st.error_counts[std::string (tool_name)];
  if (is_error)
    {
      ++err_total;
      st.error_counts[std::string (tool_name)] = err_total;
    }

  if (st.active_run == nullptr)
    {
      return;
    }

  const std::string prefix = "tool." + std::string (tool_name);
  (void)st.active_run->log_metric (st.active_step, prefix + ".duration_ms",
                                   duration_ms);
  (void)st.active_run->log_metric (st.active_step, prefix + ".error_count",
                                   static_cast<double> (err_total));
  (void)st.active_run->log_tag (prefix + ".last_input_hash",
                                hash_string_blake3 (input_str));
  (void)st.active_run->log_tag (prefix + ".last_output_hash",
                                hash_string_blake3 (output_str));
}

/* Truncates tool output at 16 KB (16384 bytes).  */
[[nodiscard]] auto
cap_tool_output (std::string text) -> std::string
{
  if (text.size () > kMaxToolOutputBytes)
    {
      text.resize (kMaxToolOutputBytes);
      text.append ("\n... [truncated]");
    }
  return text;
}

/* Trims leading and trailing ASCII whitespace.  */
[[nodiscard]] auto
trim_ws (std::string_view sv) noexcept -> std::string_view
{
  std::size_t first = 0U;
  while (first < sv.size ()
         && std::isspace (static_cast<unsigned char> (sv[first])) != 0)
    {
      ++first;
    }
  std::size_t last = sv.size ();
  while (last > first
         && std::isspace (static_cast<unsigned char> (sv[last - 1U])) != 0)
    {
      --last;
    }
  return sv.substr (first, last - first);
}

/* Case-insensitive substring helper.  */
[[nodiscard]] auto
contains_icase (std::string_view haystack,
                std::string_view needle) noexcept -> bool
{
  if (needle.empty () || haystack.size () < needle.size ())
    {
      return false;
    }
  for (std::size_t i = 0U; i + needle.size () <= haystack.size (); ++i)
    {
      bool match = true;
      for (std::size_t j = 0U; j < needle.size (); ++j)
        {
          const auto ca = static_cast<unsigned char> (haystack[i + j]);
          const auto cb = static_cast<unsigned char> (needle[j]);
          if (std::tolower (ca) != std::tolower (cb))
            {
              match = false;
              break;
            }
        }
      if (match)
        {
          return true;
        }
    }
  return false;
}

/* Splits a command string into argv tokens respecting quotes.  */
[[nodiscard]] auto
split_command_args (std::string_view cmd) -> std::vector<std::string>
{
  std::vector<std::string> args;
  std::string current;
  bool in_single = false;
  bool in_double = false;

  for (std::size_t i = 0U; i < cmd.size (); ++i)
    {
      const char ch = cmd[i];
      if (ch == '\'' && !in_double)
        {
          in_single = !in_single;
          continue;
        }
      if (ch == '"' && !in_single)
        {
          in_double = !in_double;
          continue;
        }
      if (!in_single && !in_double
          && std::isspace (static_cast<unsigned char> (ch)) != 0)
        {
          if (!current.empty ())
            {
              args.push_back (std::move (current));
              current.clear ();
            }
          continue;
        }
      current.push_back (ch);
    }
  if (!current.empty ())
    {
      args.push_back (std::move (current));
    }
  return args;
}

/* Executes an argv command inside working_dir with timeout and 16 KB cap.  */
[[nodiscard]] auto
execute_argv_with_timeout (const std::vector<std::string> &args,
                           const std::filesystem::path &working_dir,
                           uint32_t timeout_ms,
                           int *out_exit_code = nullptr) noexcept
    -> sutra::Result<std::string>
{
  if (args.empty ())
    {
      return std::unexpected (
          sutra::AgentError::ToolNotAllowed ("Empty command is not allowed"));
    }

  std::array<int, 2U> pipe_fds{ -1, -1 };
  if (pipe (pipe_fds.data ()) != 0)
    {
      return std::unexpected (sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "run_command",
          "Failed to create subprocess pipe"));
    }

  const pid_t pid = fork ();
  if (pid < 0)
    {
      close (pipe_fds[0]);
      close (pipe_fds[1]);
      return std::unexpected (sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "run_command",
          "fork() failed in execute_argv_with_timeout"));
    }

  if (pid == 0)
    {
      (void)setpgid (0, 0);
      (void)prctl (PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
      (void)unsetenv ("DISPLAY");
      (void)unsetenv ("WAYLAND_DISPLAY");
      close (pipe_fds[0]);
      (void)dup2 (pipe_fds[1], STDOUT_FILENO);
      (void)dup2 (pipe_fds[1], STDERR_FILENO);
      close (pipe_fds[1]);

      if (chdir (working_dir.c_str ()) != 0)
        {
          _exit (126);
        }

      std::vector<char *> c_args;
      c_args.reserve (args.size () + 1U);
      for (const std::string &arg : args)
        {
          c_args.push_back (const_cast<char *> (arg.c_str ()));
        }
      c_args.push_back (nullptr);

      execvp (c_args[0], c_args.data ());
      _exit (127);
    }

  (void)setpgid (pid, pid);
  close (pipe_fds[1]);
  const int flags = fcntl (pipe_fds[0], F_GETFL, 0);
  if (flags >= 0)
    {
      (void)fcntl (pipe_fds[0], F_SETFL, flags | O_NONBLOCK);
    }

  std::string output;
  const auto start_tp = std::chrono::steady_clock::now ();
  bool child_exited = false;
  int status = 0;

  while (true)
    {
      pollfd pfd{};
      pfd.fd = pipe_fds[0];
      pfd.events = POLLIN | POLLHUP;
      (void)poll (&pfd, 1U, kPollSliceMs);

      std::array<char, 4096U> buf{};
      while (true)
        {
          const ssize_t n = read (pipe_fds[0], buf.data (), buf.size ());
          if (n > 0)
            {
              if (output.size () <= kMaxToolOutputBytes + 64U)
                {
                  output.append (buf.data (), static_cast<std::size_t> (n));
                }
            }
          else
            {
              break;
            }
        }

      const pid_t w = waitpid (pid, &status, WNOHANG);
      if (w == pid)
        {
          child_exited = true;
          break;
        }

      const auto now_tp = std::chrono::steady_clock::now ();
      const auto elapsed_ms
          = std::chrono::duration_cast<std::chrono::milliseconds> (now_tp
                                                                   - start_tp)
                .count ();
      if (elapsed_ms >= static_cast<int64_t> (timeout_ms))
        {
          break;
        }
    }

  if (!child_exited)
    {
      // Timeout expired: send SIGTERM, wait up to grace period, then SIGKILL.
      (void)kill (-pid, SIGTERM);
      (void)kill (pid, SIGTERM);
      const uint32_t grace_ms
          = std::min<uint32_t> (kMaxSigkillGraceMs,
                                std::max<uint32_t> (50U, timeout_ms / 6U));
      const auto term_tp = std::chrono::steady_clock::now ();
      while (true)
        {
          const pid_t w = waitpid (pid, &status, WNOHANG);
          if (w == pid)
            {
              child_exited = true;
              break;
            }
          const auto now_tp = std::chrono::steady_clock::now ();
          const auto term_elapsed
              = std::chrono::duration_cast<std::chrono::milliseconds> (
                    now_tp - term_tp)
                    .count ();
          if (term_elapsed >= static_cast<int64_t> (grace_ms))
            {
              break;
            }
          pollfd pfd{};
          pfd.fd = pipe_fds[0];
          pfd.events = POLLIN;
          (void)poll (&pfd, 1U, 10);
        }
      if (!child_exited)
        {
          (void)kill (-pid, SIGKILL);
          (void)kill (pid, SIGKILL);
          (void)waitpid (pid, &status, 0);
        }
      close (pipe_fds[0]);
      return std::unexpected (sutra::AgentError::ToolTimeout (
          "Command timed out after " + std::to_string (timeout_ms) + " ms"));
    }

  // Drain any remaining bytes in pipe after child exit.
  std::array<char, 4096U> tail_buf{};
  while (true)
    {
      const ssize_t n = read (pipe_fds[0], tail_buf.data (), tail_buf.size ());
      if (n > 0 && output.size () <= kMaxToolOutputBytes + 64U)
        {
          output.append (tail_buf.data (), static_cast<std::size_t> (n));
        }
      else
        {
          break;
        }
    }
  close (pipe_fds[0]);

  const int exit_code
      = WIFEXITED (status)
            ? WEXITSTATUS (status)
            : (WIFSIGNALED (status) ? (128 + WTERMSIG (status)) : 1);
  if (out_exit_code != nullptr)
    {
      *out_exit_code = exit_code;
    }
  output = cap_tool_output (std::move (output));
  if (exit_code != 0 && output.empty ())
    {
      output = "exit_code=" + std::to_string (exit_code);
    }
  return output;
}

struct SshHostEntry
{
  std::string host_alias;
  std::string hostname;
  uint16_t port = kDefaultSshPort;
  std::string user;
  std::string identity_file;
};

/* Returns the user's home directory.  */
[[nodiscard]] auto
get_home_dir () -> std::filesystem::path
{
  if (const char *env_home = std::getenv ("HOME");
      env_home != nullptr && env_home[0] != '\0')
    {
      return std::filesystem::path (env_home);
    }
  if (const struct passwd *pw = getpwuid (getuid ());
      pw != nullptr && pw->pw_dir != nullptr)
    {
      return std::filesystem::path (pw->pw_dir);
    }
  return std::filesystem::path (".");
}

/* Expands leading '~/' in a path string.  */
[[nodiscard]] auto
expand_tilde (std::string_view raw_path) -> std::string
{
  if (raw_path.starts_with ("~/"))
    {
      return (get_home_dir () / raw_path.substr (2U)).string ();
    }
  return std::string (raw_path);
}

/* Parses ~/.ssh/config and looks up the requested host alias.  */
[[nodiscard]] auto
lookup_ssh_host_config (std::string_view target_host)
    -> sutra::Result<SshHostEntry>
{
  const std::filesystem::path cfg_path = get_home_dir () / ".ssh" / "config";
  std::ifstream in (cfg_path);
  if (!in.is_open ())
    {
      return std::unexpected (sutra::AgentError::SshDenied (
          "Cannot open ~/.ssh/config to validate host allowlist"));
    }

  bool in_matching_block = false;
  bool found_host = false;
  SshHostEntry entry;
  entry.host_alias = std::string (target_host);
  entry.hostname = std::string (target_host);

  if (const struct passwd *pw = getpwuid (getuid ());
      pw != nullptr && pw->pw_name != nullptr)
    {
      entry.user = pw->pw_name;
    }
  else if (const char *env_user = std::getenv ("USER");
           env_user != nullptr && env_user[0] != '\0')
    {
      entry.user = env_user;
    }
  else
    {
      entry.user = "user";
    }
  entry.identity_file = (get_home_dir () / ".ssh" / "id_ed25519").string ();

  std::string line;
  while (std::getline (in, line))
    {
      const std::size_t hash_pos = line.find ('#');
      if (hash_pos != std::string::npos)
        {
          line.resize (hash_pos);
        }
      const std::string_view trimmed = trim_ws (line);
      if (trimmed.empty ())
        {
          continue;
        }

      std::size_t sep = 0U;
      while (sep < trimmed.size ()
             && std::isspace (static_cast<unsigned char> (trimmed[sep])) == 0
             && trimmed[sep] != '=')
        {
          ++sep;
        }
      const std::string_view key = trimmed.substr (0U, sep);
      std::string_view val = trim_ws (trimmed.substr (sep));
      if (!val.empty () && val.front () == '=')
        {
          val = trim_ws (val.substr (1U));
        }

      if (contains_icase (key, "Host") && key.size () == 4U)
        {
          in_matching_block = false;
          std::istringstream iss{ std::string (val) };
          std::string pattern;
          while (iss >> pattern)
            {
              if (pattern == target_host)
                {
                  in_matching_block = true;
                  found_host = true;
                }
            }
          continue;
        }

      if (!in_matching_block)
        {
          continue;
        }

      if (contains_icase (key, "HostName") && key.size () == 8U)
        {
          entry.hostname = std::string (val);
        }
      else if (contains_icase (key, "User") && key.size () == 4U)
        {
          entry.user = std::string (val);
        }
      else if (contains_icase (key, "Port") && key.size () == 4U)
        {
          const int p = std::atoi (std::string (val).c_str ());
          if (p > 0 && p <= 65535)
            {
              entry.port = static_cast<uint16_t> (p);
            }
        }
      else if (contains_icase (key, "IdentityFile") && key.size () == 12U)
        {
          entry.identity_file = expand_tilde (val);
        }
    }

  if (!found_host)
    {
      return std::unexpected (sutra::AgentError::SshDenied (
          "Host '" + std::string (target_host)
          + "' is not in ~/.ssh/config allowlist"));
    }
  return entry;
}

/* Dynamically resolved libssh2 function table so public headers and link
   interface expose zero libssh2 symbols.  */
struct LibSsh2Api
{
  void *handle = nullptr;
  int (*init) (int) = nullptr;
  void (*exit_fn) () = nullptr;
  LIBSSH2_SESSION *(*session_init_ex) (LIBSSH2_ALLOC_FUNC ((*)),
                                       LIBSSH2_FREE_FUNC ((*)),
                                       LIBSSH2_REALLOC_FUNC ((*)), void *)
      = nullptr;
  int (*session_free) (LIBSSH2_SESSION *) = nullptr;
  int (*session_disconnect_ex) (LIBSSH2_SESSION *, int, const char *,
                                const char *)
      = nullptr;
  void (*session_set_blocking) (LIBSSH2_SESSION *, int) = nullptr;
  void (*session_set_timeout) (LIBSSH2_SESSION *, long) = nullptr;
  int (*session_handshake) (LIBSSH2_SESSION *, libssh2_socket_t) = nullptr;
  int (*userauth_publickey_fromfile_ex) (LIBSSH2_SESSION *, const char *,
                                         unsigned int, const char *,
                                         const char *, const char *)
      = nullptr;
  LIBSSH2_AGENT *(*agent_init) (LIBSSH2_SESSION *) = nullptr;
  int (*agent_connect) (LIBSSH2_AGENT *) = nullptr;
  int (*agent_list_identities) (LIBSSH2_AGENT *) = nullptr;
  int (*agent_get_identity) (LIBSSH2_AGENT *,
                             struct libssh2_agent_publickey **,
                             struct libssh2_agent_publickey *)
      = nullptr;
  int (*agent_userauth) (LIBSSH2_AGENT *, const char *,
                         struct libssh2_agent_publickey *)
      = nullptr;
  int (*agent_disconnect) (LIBSSH2_AGENT *) = nullptr;
  void (*agent_free) (LIBSSH2_AGENT *) = nullptr;
  LIBSSH2_CHANNEL *(*channel_open_ex) (LIBSSH2_SESSION *, const char *,
                                       unsigned int, unsigned int,
                                       unsigned int, const char *,
                                       unsigned int)
      = nullptr;
  int (*channel_process_startup) (LIBSSH2_CHANNEL *, const char *,
                                  unsigned int, const char *, unsigned int)
      = nullptr;
  ssize_t (*channel_read_ex) (LIBSSH2_CHANNEL *, int, char *, size_t)
      = nullptr;
  int (*channel_close) (LIBSSH2_CHANNEL *) = nullptr;
  int (*channel_wait_closed) (LIBSSH2_CHANNEL *) = nullptr;
  int (*channel_get_exit_status) (LIBSSH2_CHANNEL *) = nullptr;
  int (*channel_free) (LIBSSH2_CHANNEL *) = nullptr;

  static auto
  instance () noexcept -> const LibSsh2Api &
  {
    static const LibSsh2Api api = [] () noexcept {
      LibSsh2Api a{};
      a.handle = dlopen ("libssh2.so.1", RTLD_LAZY | RTLD_LOCAL);
      if (a.handle == nullptr)
        {
          a.handle = dlopen ("libssh2.so", RTLD_LAZY | RTLD_LOCAL);
        }
      if (a.handle == nullptr)
        {
          return a;
        }
      a.init = reinterpret_cast<decltype (a.init)> (
          dlsym (a.handle, "libssh2_init"));
      a.exit_fn = reinterpret_cast<decltype (a.exit_fn)> (
          dlsym (a.handle, "libssh2_exit"));
      a.session_init_ex = reinterpret_cast<decltype (a.session_init_ex)> (
          dlsym (a.handle, "libssh2_session_init_ex"));
      a.session_free = reinterpret_cast<decltype (a.session_free)> (
          dlsym (a.handle, "libssh2_session_free"));
      a.session_disconnect_ex
          = reinterpret_cast<decltype (a.session_disconnect_ex)> (
              dlsym (a.handle, "libssh2_session_disconnect_ex"));
      a.session_set_blocking
          = reinterpret_cast<decltype (a.session_set_blocking)> (
              dlsym (a.handle, "libssh2_session_set_blocking"));
      a.session_set_timeout
          = reinterpret_cast<decltype (a.session_set_timeout)> (
              dlsym (a.handle, "libssh2_session_set_timeout"));
      a.session_handshake = reinterpret_cast<decltype (a.session_handshake)> (
          dlsym (a.handle, "libssh2_session_handshake"));
      a.userauth_publickey_fromfile_ex
          = reinterpret_cast<decltype (a.userauth_publickey_fromfile_ex)> (
              dlsym (a.handle, "libssh2_userauth_publickey_fromfile_ex"));
      a.agent_init = reinterpret_cast<decltype (a.agent_init)> (
          dlsym (a.handle, "libssh2_agent_init"));
      a.agent_connect = reinterpret_cast<decltype (a.agent_connect)> (
          dlsym (a.handle, "libssh2_agent_connect"));
      a.agent_list_identities
          = reinterpret_cast<decltype (a.agent_list_identities)> (
              dlsym (a.handle, "libssh2_agent_list_identities"));
      a.agent_get_identity
          = reinterpret_cast<decltype (a.agent_get_identity)> (
              dlsym (a.handle, "libssh2_agent_get_identity"));
      a.agent_userauth = reinterpret_cast<decltype (a.agent_userauth)> (
          dlsym (a.handle, "libssh2_agent_userauth"));
      a.agent_disconnect = reinterpret_cast<decltype (a.agent_disconnect)> (
          dlsym (a.handle, "libssh2_agent_disconnect"));
      a.agent_free = reinterpret_cast<decltype (a.agent_free)> (
          dlsym (a.handle, "libssh2_agent_free"));
      a.channel_open_ex = reinterpret_cast<decltype (a.channel_open_ex)> (
          dlsym (a.handle, "libssh2_channel_open_ex"));
      a.channel_process_startup
          = reinterpret_cast<decltype (a.channel_process_startup)> (
              dlsym (a.handle, "libssh2_channel_process_startup"));
      a.channel_read_ex = reinterpret_cast<decltype (a.channel_read_ex)> (
          dlsym (a.handle, "libssh2_channel_read_ex"));
      a.channel_close = reinterpret_cast<decltype (a.channel_close)> (
          dlsym (a.handle, "libssh2_channel_close"));
      a.channel_wait_closed
          = reinterpret_cast<decltype (a.channel_wait_closed)> (
              dlsym (a.handle, "libssh2_channel_wait_closed"));
      a.channel_get_exit_status
          = reinterpret_cast<decltype (a.channel_get_exit_status)> (
              dlsym (a.handle, "libssh2_channel_get_exit_status"));
      a.channel_free = reinterpret_cast<decltype (a.channel_free)> (
          dlsym (a.handle, "libssh2_channel_free"));
      if (a.init != nullptr)
        {
          (void)a.init (0);
        }
      return a;
    }();
    return api;
  }

  [[nodiscard]] auto
  valid () const noexcept -> bool
  {
    return handle != nullptr && session_init_ex != nullptr
           && session_handshake != nullptr
           && userauth_publickey_fromfile_ex != nullptr
           && channel_open_ex != nullptr && channel_process_startup != nullptr
           && channel_read_ex != nullptr;
  }
};

/* Executes an SSH command or connectivity test via libssh2.  */
[[nodiscard]] auto
execute_libssh2_session (std::string_view host, std::string_view cmd,
                         bool test_only) noexcept -> sutra::Result<std::string>
{
  auto host_res = lookup_ssh_host_config (host);
  if (!host_res)
    {
      return std::unexpected (host_res.error ());
    }
  const SshHostEntry &entry = *host_res;

  if (!test_only)
    {
      auto cmd_valid = validate_ssh_command (cmd);
      if (!cmd_valid)
        {
          return std::unexpected (cmd_valid.error ());
        }
    }

  const LibSsh2Api &ssh = LibSsh2Api::instance ();
  if (!ssh.valid ())
    {
      return std::unexpected (sutra::AgentError::SshAuth (
          "libssh2.so.1 could not be loaded via dlopen"));
    }

  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo *res_addrs = nullptr;
  const std::string port_str = std::to_string (entry.port);
  if (getaddrinfo (entry.hostname.c_str (), port_str.c_str (), &hints,
                   &res_addrs)
          != 0
      || res_addrs == nullptr)
    {
      return std::unexpected (sutra::AgentError::SshAuth (
          "Failed to resolve SSH hostname: " + entry.hostname));
    }

  const int sock = socket (res_addrs->ai_family, res_addrs->ai_socktype,
                           res_addrs->ai_protocol);
  if (sock < 0)
    {
      freeaddrinfo (res_addrs);
      return std::unexpected (
          sutra::AgentError::SshAuth ("socket() failed for SSH connection"));
    }

  if (connect (sock, res_addrs->ai_addr, res_addrs->ai_addrlen) != 0)
    {
      freeaddrinfo (res_addrs);
      close (sock);
      return std::unexpected (sutra::AgentError::SshAuth (
          "TCP connect() failed to " + entry.hostname + ":" + port_str));
    }
  freeaddrinfo (res_addrs);

  LIBSSH2_SESSION *session = ssh.session_init_ex (nullptr, nullptr, nullptr,
                                                  nullptr);
  if (session == nullptr)
    {
      close (sock);
      return std::unexpected (
          sutra::AgentError::SshAuth ("libssh2_session_init_ex failed"));
    }

  const uint32_t timeout_ms = ToolEnvironment::command_timeout_ms ();
  ssh.session_set_blocking (session, 1);
  if (ssh.session_set_timeout != nullptr)
    {
      ssh.session_set_timeout (session, static_cast<long> (timeout_ms));
    }

  if (ssh.session_handshake (session, sock) != 0)
    {
      ssh.session_free (session);
      close (sock);
      return std::unexpected (
          sutra::AgentError::SshAuth ("libssh2_session_handshake failed"));
    }

  bool authenticated = false;
  const std::string pub_path = entry.identity_file + ".pub";
  if (ssh.userauth_publickey_fromfile_ex (
          session, entry.user.c_str (),
          static_cast<unsigned int> (entry.user.size ()), pub_path.c_str (),
          entry.identity_file.c_str (), nullptr)
      == 0)
    {
      authenticated = true;
    }

  if (!authenticated && ssh.agent_init != nullptr
      && ssh.agent_connect != nullptr)
    {
      LIBSSH2_AGENT *agent = ssh.agent_init (session);
      if (agent != nullptr)
        {
          if (ssh.agent_connect (agent) == 0
              && ssh.agent_list_identities (agent) == 0)
            {
              struct libssh2_agent_publickey *identity = nullptr;
              struct libssh2_agent_publickey *prev = nullptr;
              while (ssh.agent_get_identity (agent, &identity, prev) == 0
                     && identity != nullptr)
                {
                  if (ssh.agent_userauth (agent, entry.user.c_str (), identity)
                      == 0)
                    {
                      authenticated = true;
                      break;
                    }
                  prev = identity;
                }
            }
          ssh.agent_disconnect (agent);
          ssh.agent_free (agent);
        }
    }

  if (!authenticated)
    {
      (void)ssh.session_disconnect_ex (session, SSH_DISCONNECT_BY_APPLICATION,
                                       "auth failed", "");
      ssh.session_free (session);
      close (sock);
      return std::unexpected (sutra::AgentError::SshAuth (
          "Non-interactive SSH publickey/agent auth failed for "
          + entry.user + "@" + entry.hostname));
    }

  if (test_only)
    {
      (void)ssh.session_disconnect_ex (session, SSH_DISCONNECT_BY_APPLICATION,
                                       "ssh_test complete", "");
      ssh.session_free (session);
      close (sock);
      return std::string ("ok");
    }

  constexpr std::string_view kSessionType = "session";
  LIBSSH2_CHANNEL *channel = ssh.channel_open_ex (
      session, kSessionType.data (),
      static_cast<unsigned int> (kSessionType.size ()),
      LIBSSH2_CHANNEL_WINDOW_DEFAULT, LIBSSH2_CHANNEL_PACKET_DEFAULT, nullptr,
      0U);
  if (channel == nullptr)
    {
      (void)ssh.session_disconnect_ex (session, SSH_DISCONNECT_BY_APPLICATION,
                                       "channel open failed", "");
      ssh.session_free (session);
      close (sock);
      return std::unexpected (
          sutra::AgentError::SshAuth ("libssh2_channel_open_ex failed"));
    }

  constexpr std::string_view kExecReq = "exec";
  if (ssh.channel_process_startup (
          channel, kExecReq.data (),
          static_cast<unsigned int> (kExecReq.size ()), cmd.data (),
          static_cast<unsigned int> (cmd.size ()))
      != 0)
    {
      ssh.channel_free (channel);
      (void)ssh.session_disconnect_ex (session, SSH_DISCONNECT_BY_APPLICATION,
                                       "exec failed", "");
      ssh.session_free (session);
      close (sock);
      return std::unexpected (sutra::AgentError::SshTimeout (
          "libssh2_channel_process_startup failed or timed out"));
    }

  std::string output;
  std::array<char, 4096U> buf{};
  while (true)
    {
      const ssize_t n_out
          = ssh.channel_read_ex (channel, 0, buf.data (), buf.size ());
      if (n_out > 0 && output.size () <= kMaxToolOutputBytes + 64U)
        {
          output.append (buf.data (), static_cast<std::size_t> (n_out));
        }
      const ssize_t n_err = ssh.channel_read_ex (
          channel, SSH_EXTENDED_DATA_STDERR, buf.data (), buf.size ());
      if (n_err > 0 && output.size () <= kMaxToolOutputBytes + 64U)
        {
          output.append (buf.data (), static_cast<std::size_t> (n_err));
        }
      if (n_out <= 0 && n_err <= 0)
        {
          if (n_out == LIBSSH2_ERROR_TIMEOUT || n_err == LIBSSH2_ERROR_TIMEOUT)
            {
              ssh.channel_free (channel);
              (void)ssh.session_disconnect_ex (
                  session, SSH_DISCONNECT_BY_APPLICATION, "timeout", "");
              ssh.session_free (session);
              close (sock);
              return std::unexpected (sutra::AgentError::SshTimeout (
                  "SSH command timed out on host " + std::string (host)));
            }
          break;
        }
    }

  (void)ssh.channel_close (channel);
  (void)ssh.channel_wait_closed (channel);
  ssh.channel_free (channel);
  (void)ssh.session_disconnect_ex (session, SSH_DISCONNECT_BY_APPLICATION,
                                   "normal shutdown", "");
  ssh.session_free (session);
  close (sock);

  return cap_tool_output (std::move (output));
}

} /* namespace */

auto
ToolEnvironment::set_sandbox_dir (const std::filesystem::path &dir) noexcept
    -> sutra::Result<std::filesystem::path>
{
  try
    {
      std::error_code ec;
      std::filesystem::create_directories (dir, ec);
      const std::filesystem::path canon
          = std::filesystem::weakly_canonical (dir, ec);
      if (ec || canon.empty ())
        {
          return std::unexpected (sutra::AgentError::ToolSandboxViolation (
              "Failed to canonicalize sandbox directory"));
        }
      auto &st = global_tool_state ();
      const std::scoped_lock lock (st.mtx);
      st.sandbox_dir = canon;
      return canon;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in ToolEnvironment::set_sandbox_dir"));
    }
}

auto
ToolEnvironment::sandbox_dir () noexcept -> std::filesystem::path
{
  auto &st = global_tool_state ();
  const std::scoped_lock lock (st.mtx);
  std::filesystem::path root = st.sandbox_dir;
  if (root == kDefaultSandboxDir)
    {
      if (const char *env_sb = std::getenv ("SPHINX_SANDBOX");
          env_sb != nullptr && env_sb[0] != '\0')
        {
          root = std::filesystem::path (env_sb);
        }
      else
        {
          root = get_home_dir () / "sphinx-workspace";
        }
    }
  std::error_code ec;
  std::filesystem::create_directories (root, ec);
  std::filesystem::permissions (
      root,
      std::filesystem::perms::owner_all | std::filesystem::perms::group_read
          | std::filesystem::perms::group_exec
          | std::filesystem::perms::others_read
          | std::filesystem::perms::others_exec,
      std::filesystem::perm_options::replace, ec);
  const std::filesystem::path canon
      = std::filesystem::weakly_canonical (root, ec);
  return ec ? root : canon;
}

void
ToolEnvironment::bind_yantra_run (yantra::Run *run, uint64_t step) noexcept
{
  auto &st = global_tool_state ();
  const std::scoped_lock lock (st.mtx);
  st.active_run = run;
  st.active_step = step;
}

void
ToolEnvironment::set_command_timeout_ms (uint32_t timeout_ms) noexcept
{
  global_tool_state ().command_timeout_ms.store (
      std::max<uint32_t> (50U, timeout_ms), std::memory_order_release);
}

auto
ToolEnvironment::command_timeout_ms () noexcept -> uint32_t
{
  return global_tool_state ().command_timeout_ms.load (
      std::memory_order_acquire);
}

void
ToolEnvironment::set_readonly_timeout_ms (uint32_t timeout_ms) noexcept
{
  global_tool_state ().readonly_timeout_ms.store (
      std::max<uint32_t> (50U, timeout_ms), std::memory_order_release);
}

auto
ToolEnvironment::readonly_timeout_ms () noexcept -> uint32_t
{
  return global_tool_state ().readonly_timeout_ms.load (
      std::memory_order_acquire);
}

[[nodiscard]] auto
is_allowed_temp_qa_path (std::string_view raw_trimmed,
                         std::string_view resolved_str) noexcept -> bool
{
  constexpr std::string_view kQaRoot = "/tmp/sphinx_react_qa";
  if (!raw_trimmed.starts_with (kQaRoot))
    {
      return false;
    }
  if (raw_trimmed.size () > kQaRoot.size ()
      && raw_trimmed[kQaRoot.size ()] != '/')
    {
      return false;
    }
  if (raw_trimmed.find ("..") != std::string_view::npos)
    {
      return false;
    }
  return resolved_str == kQaRoot
         || (resolved_str.size () > kQaRoot.size ()
             && resolved_str.starts_with (kQaRoot)
             && resolved_str[kQaRoot.size ()] == '/');
}

auto
resolve_sandboxed_path (std::string_view user_path) noexcept
    -> sutra::Result<std::filesystem::path>
{
  try
    {
      const std::filesystem::path base = ToolEnvironment::sandbox_dir ();
      const std::string_view trimmed = trim_ws (user_path);
      const std::filesystem::path candidate
          = trimmed.empty ()
                ? base
                : (std::filesystem::path (trimmed).is_absolute ()
                       ? std::filesystem::path (trimmed)
                       : (base / std::filesystem::path (trimmed)));

      std::error_code ec;
      const std::filesystem::path resolved
          = std::filesystem::weakly_canonical (candidate, ec);
      if (ec || resolved.empty ())
        {
          return std::unexpected (sutra::AgentError::ToolSandboxViolation (
              "Invalid or unresolvable path: " + std::string (user_path)));
        }

      const std::string base_str = base.string ();
      const std::string res_str = resolved.string ();

      if (res_str != base_str
          && (!res_str.starts_with (base_str + "/")
              || res_str.size () <= base_str.size ())
          && !is_allowed_temp_qa_path (trimmed, res_str))
        {
          return std::unexpected (sutra::AgentError::ToolSandboxViolation (
              "Cannot access " + std::string (user_path)
              + ": path is outside the sandbox. The allowed root is "
              + base_str
              + ". Use a relative path within the sandbox, or omit the "
                "leading path components."));
        }
      return resolved;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::ToolSandboxViolation (
          "Exception resolving sandboxed path"));
    }
}

namespace
{

[[nodiscard]] auto
basename_view (std::string_view tok) noexcept -> std::string_view
{
  const std::size_t slash = tok.rfind ('/');
  return (slash == std::string_view::npos) ? tok : tok.substr (slash + 1U);
}

[[nodiscard]] auto
is_privilege_escalation_token (std::string_view tok,
                               bool is_executable_pos) noexcept -> bool
{
  const std::string_view base = basename_view (tok);
  if (base.starts_with ("mkfs."))
    {
      return true;
    }
  if (is_executable_pos
      && (base == "init" || base == "service" || base == "mount"
          || base == "umount" || base == "dd" || base == "shutdown"
          || base == "reboot" || base == "poweroff" || base == "halt"))
    {
      return true;
    }
  constexpr std::array<std::string_view, 20U> kDeniedPrivTokens = { {
      "sudo",      "su",       "pkexec",  "doas",    "chroot",
      "nsenter",   "unshare",  "chown",   "setcap",  "systemctl",
      "mkfs",      "fdisk",    "parted",  "iptables","ufw",
      "passwd",    "useradd",  "usermod", "userdel", "visudo",
  } };
  for (const std::string_view denied : kDeniedPrivTokens)
    {
      if (base == denied)
        {
          return true;
        }
    }
  return false;
}

[[nodiscard]] auto
is_disallowed_subshell_or_net_token (std::string_view tok) noexcept -> bool
{
  const std::string_view base = basename_view (tok);
  constexpr std::array<std::string_view, 17U> kDeniedSubshellNet = { {
      "sh",     "bash",   "zsh",    "dash",   "csh",    "tcsh",
      "ksh",    "fish",   "curl",   "wget",   "nc",     "netcat",
      "ncat",   "socat",  "telnet", "ftp",    "sftp",
  } };
  for (const std::string_view denied : kDeniedSubshellNet)
    {
      if (base == denied)
        {
          return true;
        }
    }
  return false;
}

[[nodiscard]] auto
is_destructive_command_token (std::string_view tok) noexcept -> bool
{
  const std::string_view base = basename_view (tok);
  return base == "rm" || base == "mv" || base == "truncate"
         || base == "shred" || base == "rmdir";
}

[[nodiscard]] auto
is_compiler_token (std::string_view tok) noexcept -> bool
{
  const std::string_view base = basename_view (tok);
  return base == "gcc" || base == "g++" || base == "clang"
         || base == "clang++" || base == "rustc" || base == "cargo";
}

[[nodiscard]] auto
is_setuid_chmod_mode (std::string_view arg) noexcept -> bool
{
  if (arg.find ("+s") != std::string_view::npos
      || arg.find ("=s") != std::string_view::npos)
    {
      return true;
    }
  if (arg.size () >= 4U
      && (arg[0] == '4' || arg[0] == '2' || arg[0] == '6' || arg[0] == '7'))
    {
      bool all_oct = true;
      for (const char ch : arg)
        {
          if (ch < '0' || ch > '7')
            {
              all_oct = false;
              break;
            }
        }
      if (all_oct)
        {
          return true;
        }
    }
  return false;
}

[[nodiscard]] auto
is_protected_memory_reference (std::string_view arg) noexcept -> bool
{
  if (arg == ".memory" || arg == "./.memory"
      || arg.starts_with (".memory/") || arg.starts_with ("./.memory/")
      || arg.ends_with ("/.memory")
      || arg.find ("/.memory/") != std::string_view::npos
      || arg.find ("sphinx/memory") != std::string_view::npos)
    {
      return true;
    }
  return false;
}

[[nodiscard]] auto
expand_tilde_arg (std::string_view arg) -> std::string
{
  if (!arg.starts_with ('~'))
    {
      return std::string (arg);
    }
  const std::filesystem::path home = get_home_dir ();
  if (arg == "~")
    {
      return home.string ();
    }
  if (arg.starts_with ("~/"))
    {
      return (home / arg.substr (2U)).string ();
    }
  return std::string (arg);
}

[[nodiscard]] auto
validate_command_arguments_in_sandbox (
    const std::vector<std::string> &args,
    const std::filesystem::path &cwd_path) noexcept -> sutra::Result<void>
{
  if (args.empty ())
    {
      return std::unexpected (
          sutra::AgentError::ToolNotAllowed ("Command must not be empty"));
    }

  for (std::size_t i = 0U; i < args.size (); ++i)
    {
      const bool is_exec_pos
          = (i == 0U) || (i == 1U && basename_view (args[0]) == "env");
      if (is_privilege_escalation_token (args[i], is_exec_pos))
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "ERROR: Privilege escalation commands are not permitted."));
        }
    }

  if (basename_view (args[0]) == "chmod")
    {
      for (std::size_t i = 1U; i < args.size (); ++i)
        {
          if (is_setuid_chmod_mode (args[i]))
            {
              return std::unexpected (sutra::AgentError::ToolNotAllowed (
                  "ERROR: Privilege escalation commands are not permitted."));
            }
        }
    }

  if (is_disallowed_subshell_or_net_token (args[0]))
    {
      return std::unexpected (sutra::AgentError::ToolNotAllowed (
          "ERROR: Subshell and network commands ('" + args[0]
          + "') are not permitted in the workspace sandbox."));
    }

  const std::filesystem::path sandbox_root = ToolEnvironment::sandbox_dir ();
  const bool is_destructive = is_destructive_command_token (args[0]);
  const bool is_compiler = is_compiler_token (args[0]);
  const bool is_build_tool
      = is_compiler || basename_view (args[0]) == "cmake";

  for (std::size_t i = 1U; i < args.size (); ++i)
    {
      std::string_view raw_arg = args[i];
      if (is_protected_memory_reference (raw_arg))
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "ERROR: Modifying or accessing persistent memory directory "
              "(.memory) is not permitted."));
        }

      if (raw_arg.starts_with ('-'))
        {
          const std::size_t eq = raw_arg.find ('=');
          if (eq == std::string_view::npos)
            {
              if (is_compiler
                  && (raw_arg == "-I" || raw_arg == "-L"
                      || raw_arg == "-isystem" || raw_arg == "-rpath")
                  && (i + 1U) < args.size ()
                  && !is_protected_memory_reference (args[i + 1U]))
                {
                  ++i;
                  continue;
                }
              if (is_compiler
                  && (raw_arg.starts_with ("-I")
                      || raw_arg.starts_with ("-L")
                      || raw_arg.starts_with ("-isystem")))
                {
                  continue;
                }
              continue;
            }
          if (is_build_tool
              && (raw_arg.starts_with ("-Wl,")
                  || raw_arg.starts_with ("-D")
                  || raw_arg.starts_with ("-std=")
                  || raw_arg.starts_with ("-march=")
                  || raw_arg.starts_with ("-mtune=")
                  || raw_arg.starts_with ("-f")
                  || raw_arg.starts_with ("-W")))
            {
              continue;
            }
          raw_arg = raw_arg.substr (eq + 1U);
        }
      else
        {
          const std::size_t eq = raw_arg.find ('=');
          if (eq != std::string_view::npos
              && (raw_arg.substr (eq + 1U).starts_with ('/')
                  || raw_arg.substr (eq + 1U).starts_with ('~')
                  || raw_arg.substr (eq + 1U).find ("..")
                         != std::string_view::npos))
            {
              raw_arg = raw_arg.substr (eq + 1U);
            }
        }

      if (raw_arg.empty ())
        {
          continue;
        }
      if (is_protected_memory_reference (raw_arg))
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "ERROR: Modifying or accessing persistent memory directory "
              "(.memory) is not permitted."));
        }

      if (is_compiler && !is_destructive
          && (raw_arg.starts_with ("/usr/include/")
              || raw_arg.starts_with ("/usr/lib/")
              || raw_arg.starts_with ("/usr/local/include/")
              || raw_arg.starts_with ("/usr/local/lib/")))
        {
          continue;
        }

      if (is_destructive
          && (raw_arg == "." || raw_arg == "./" || raw_arg == ".."
              || raw_arg == "../" || raw_arg == "/" || raw_arg == "~"))
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "ERROR: Path '" + std::string (raw_arg)
              + "' is outside or targets the root of the workspace sandbox ("
              + sandbox_root.string () + ") and is not allowed."));
        }

      const bool looks_like_path
          = raw_arg.starts_with ('/') || raw_arg.starts_with ('~')
            || raw_arg.find ('/') != std::string_view::npos
            || raw_arg.find ("..") != std::string_view::npos;
      if (looks_like_path)
        {
          const std::string expanded = expand_tilde_arg (raw_arg);
          const std::filesystem::path cand
              = std::filesystem::path (expanded).is_absolute ()
                    ? std::filesystem::path (expanded)
                    : (cwd_path / std::filesystem::path (expanded));
          if (!is_destructive
              && (basename_view (args[0]) == "ls"
                  || basename_view (args[0]) == "stat"
                  || basename_view (args[0]) == "file")
              && (cand == "/tmp" || cand == "/tmp/"))
            {
              continue;
            }
          auto res_path = resolve_sandboxed_path (cand.string ());
          if (!res_path)
            {
              return std::unexpected (sutra::AgentError::ToolNotAllowed (
                  "ERROR: Path '" + std::string (raw_arg)
                  + "' is outside the workspace sandbox ("
                  + sandbox_root.string () + ") and is not allowed."));
            }
          if (is_destructive && *res_path == sandbox_root)
            {
              return std::unexpected (sutra::AgentError::ToolNotAllowed (
                  "ERROR: Destructive command targeting the workspace root ("
                  + sandbox_root.string () + ") is not allowed."));
            }
          const std::string rel_mem = (sandbox_root / ".memory").string ();
          const std::string res_str = res_path->string ();
          if (res_str == rel_mem || res_str.starts_with (rel_mem + "/"))
            {
              return std::unexpected (sutra::AgentError::ToolNotAllowed (
                  "ERROR: Modifying or accessing persistent memory directory "
                  "(.memory) is not permitted."));
            }
        }
    }

  return {};
}

[[nodiscard]] auto
match_wildcard_glob (std::string_view pattern,
                     std::string_view name) noexcept -> bool
{
  std::size_t p = 0U;
  std::size_t n = 0U;
  std::size_t star_idx = std::string_view::npos;
  std::size_t match_idx = 0U;

  while (n < name.size ())
    {
      if (p < pattern.size () && (pattern[p] == '?' || pattern[p] == name[n]))
        {
          ++p;
          ++n;
        }
      else if (p < pattern.size () && pattern[p] == '*')
        {
          star_idx = p;
          match_idx = n;
          ++p;
        }
      else if (star_idx != std::string_view::npos)
        {
          p = star_idx + 1U;
          ++match_idx;
          n = match_idx;
        }
      else
        {
          return false;
        }
    }
  while (p < pattern.size () && pattern[p] == '*')
    {
      ++p;
    }
  return p == pattern.size ();
}

} /* namespace */

auto
validate_allowed_command (std::string_view cmd) noexcept -> sutra::Result<void>
{
  const std::string_view trimmed = trim_ws (cmd);
  if (trimmed.empty ())
    {
      return std::unexpected (
          sutra::AgentError::ToolNotAllowed ("Command must not be empty"));
    }

  // Reject shell metacharacters that could chain or redirect commands.
  constexpr std::string_view kForbiddenChars = ";|&`$><\n\r";
  for (const char ch : trimmed)
    {
      if (kForbiddenChars.find (ch) != std::string_view::npos)
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "Command contains disallowed shell metacharacter: "
              + std::string (trimmed)));
        }
    }

  try
    {
      const std::vector<std::string> args = split_command_args (trimmed);
      if (args.empty ())
        {
          return std::unexpected (
              sutra::AgentError::ToolNotAllowed ("Command must not be empty"));
        }

      auto arg_check = validate_command_arguments_in_sandbox (
          args, ToolEnvironment::sandbox_dir ());
      if (!arg_check)
        {
          return arg_check;
        }

      const std::string &cmd_word = args[0];
      if (cmd_word.starts_with ('/') || cmd_word.starts_with ('~'))
        {
          const std::string expanded = expand_tilde_arg (cmd_word);
          auto bin_res = resolve_sandboxed_path (expanded);
          if (!bin_res)
            {
              return std::unexpected (sutra::AgentError::ToolNotAllowed (
                  "ERROR: Executable path '" + cmd_word
                  + "' is outside the workspace sandbox ("
                  + ToolEnvironment::sandbox_dir ().string ()
                  + ") and is not allowed."));
            }
          struct stat st{};
          if (::stat (bin_res->c_str (), &st) == 0
              && (!S_ISREG (st.st_mode) || (st.st_mode & S_IXUSR) == 0))
            {
              return std::unexpected (sutra::AgentError::ToolNotAllowed (
                  "Cannot run '" + std::string (trimmed) + "': " + cmd_word
                  + " is not an executable regular file inside "
                  + ToolEnvironment::sandbox_dir ().string () + "."));
            }
          return {};
        }

      if (cmd_word.find ('/') != std::string::npos)
        {
          if (cmd_word.find ("..") != std::string::npos)
            {
              return std::unexpected (sutra::AgentError::ToolNotAllowed (
                  "ERROR: Relative executable path with '..' is not "
                  "permitted."));
            }
          const std::string_view rel_bin
              = cmd_word.starts_with ("./")
                    ? std::string_view (cmd_word).substr (2U)
                    : std::string_view (cmd_word);
          auto bin_res = resolve_sandboxed_path (rel_bin);
          if (!bin_res)
            {
              return std::unexpected (sutra::AgentError::ToolNotAllowed (
                  "ERROR: Executable '" + cmd_word
                  + "' is outside the workspace sandbox ("
                  + ToolEnvironment::sandbox_dir ().string () + ")."));
            }
          struct stat st{};
          if (::stat (bin_res->c_str (), &st) == 0)
            {
              if (S_ISREG (st.st_mode) && (st.st_mode & S_IXUSR) != 0)
                {
                  return {};
                }
              return std::unexpected (sutra::AgentError::ToolNotAllowed (
                  "Cannot run '" + std::string (trimmed) + "': " + cmd_word
                  + " is not an executable regular file inside "
                  + ToolEnvironment::sandbox_dir ().string () + "."));
            }
          return {};
        }

      return {};
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::ToolNotAllowed (
          "Exception validating command"));
    }
}

auto
validate_ssh_command (std::string_view cmd) noexcept -> sutra::Result<void>
{
  const std::string_view trimmed = trim_ws (cmd);
  if (trimmed.empty ())
    {
      return std::unexpected (
          sutra::AgentError::SshDenied ("SSH command must not be empty"));
    }

  constexpr std::array<std::string_view, 8U> kDeniedPatterns = { {
      "rm -rf",
      "dd ",
      "dd\t",
      "mkfs",
      "shutdown",
      "reboot",
      "> /dev/",
      ">/dev/",
  } };

  for (const std::string_view pat : kDeniedPatterns)
    {
      if (contains_icase (trimmed, pat))
        {
          return std::unexpected (sutra::AgentError::SshDenied (
              "SSH command matches dangerous denylist pattern '"
              + std::string (pat) + "'"));
        }
    }
  return {};
}

auto
tool_read_file (const sutra::ToolContext & /*ctx*/, std::string path,
                int64_t max_lines, int64_t start_line) noexcept
    -> sutra::Result<std::string>
{
  const auto t0 = std::chrono::steady_clock::now ();
  const std::string input_repr = path + ":" + std::to_string (start_line) + ":"
                                 + std::to_string (max_lines);

  auto path_res = resolve_sandboxed_path (path);
  if (!path_res)
    {
      const std::string base_str = ToolEnvironment::sandbox_dir ().string ();
      const auto err = sutra::AgentError::ToolSandboxViolation (
          "Cannot read from " + path
          + ": path is outside the sandbox. The allowed root is " + base_str
          + ". Use a relative path within the sandbox, or omit the leading "
            "path components.");
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      record_tool_telemetry ("read_file", input_repr, err.message (), ms,
                             true);
      return std::unexpected (err);
    }

  try
    {
      std::error_code ec;
      if (std::filesystem::is_directory (*path_res, ec))
        {
          const auto err = sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "read_file",
              "Path '" + path
                  + "' is a directory; use list_dir instead of read_file.");
          const double ms
              = std::chrono::duration<double, std::milli> (
                    std::chrono::steady_clock::now () - t0)
                    .count ();
          record_tool_telemetry ("read_file", input_repr, err.message (), ms,
                                 true);
          return std::unexpected (err);
        }

      std::ifstream in (*path_res);
      if (!in.is_open ())
        {
          const auto err = sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "read_file",
              "Could not open file: " + path_res->string ());
          const double ms
              = std::chrono::duration<double, std::milli> (
                    std::chrono::steady_clock::now () - t0)
                    .count ();
          record_tool_telemetry ("read_file", input_repr, err.message (), ms,
                                 true);
          return std::unexpected (err);
        }

      const int64_t first_line = std::max<int64_t> (1, start_line);
      const int64_t limit
          = (max_lines > 0) ? std::clamp<int64_t> (max_lines, 1, 500) : 200;
      std::string out;
      std::string line;
      int64_t current_line = 1;
      int64_t count = 0;
      while (current_line < first_line && std::getline (in, line))
        {
          ++current_line;
        }
      while (count < limit && std::getline (in, line))
        {
          if (line.find ('\0') != std::string::npos)
            {
              const auto err = sutra::AgentError::tool (
                  sutra::ToolErrorCode::ExecutionFailure, "read_file",
                  "File '" + path
                      + "' is a binary executable; do not read binary files "
                        "with read_file.");
              const double ms
                  = std::chrono::duration<double, std::milli> (
                        std::chrono::steady_clock::now () - t0)
                        .count ();
              record_tool_telemetry ("read_file", input_repr, err.message (),
                                     ms, true);
              return std::unexpected (err);
            }
          out.append (line);
          out.push_back ('\n');
          ++count;
          if (out.size () > kMaxToolOutputBytes)
            {
              break;
            }
        }
      out = cap_tool_output (std::move (out));
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      record_tool_telemetry ("read_file", input_repr, out, ms, false);
      return out;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "read_file",
          "Exception reading file"));
    }
}

[[nodiscard]] auto
is_gnu_style_exempt_ident (std::string_view ident) noexcept -> bool
{
  constexpr std::array<std::string_view, 9U> kExempt = { {
      "operator",
      "sizeof",
      "alignof",
      "decltype",
      "static_cast",
      "dynamic_cast",
      "const_cast",
      "reinterpret_cast",
      "__attribute__",
  } };
  for (const std::string_view ex : kExempt)
    {
      if (ident == ex)
        {
          return true;
        }
    }
  return false;
}

[[nodiscard]] auto
normalize_gnu_cpp_style (std::string_view path,
                         std::string content) -> std::string
{
  const bool is_cpp
      = path.ends_with (".cpp") || path.ends_with (".cc")
        || path.ends_with (".cxx") || path.ends_with (".c")
        || path.ends_with (".h") || path.ends_with (".hpp");
  if (!is_cpp)
    {
      return content;
    }
  const bool has_gpl_full
      = content.find ("GNU General Public License") != std::string::npos;
  const bool has_copyright_or_spdx
      = content.find ("Copyright (C)") != std::string::npos
        || content.find ("SPDX-License-Identifier: GPL") != std::string::npos;
  if (!has_gpl_full && !has_copyright_or_spdx)
    {
      return content;
    }
  if (!has_gpl_full)
    {
      constexpr std::string_view kGplHeader
          = "/* Copyright (C) 2026 Kival\n"
            "\n"
            "   This program is free software: you can redistribute it and/or "
            "modify\n"
            "   it under the terms of the GNU General Public License as "
            "published by\n"
            "   the Free Software Foundation, either version 3 of the "
            "License, or\n"
            "   (at your option) any later version.\n"
            "\n"
            "   This program is distributed in the hope that it will be "
            "useful,\n"
            "   but WITHOUT ANY WARRANTY; without even the implied warranty "
            "of\n"
            "   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See "
            "the\n"
            "   GNU General Public License for more details.\n"
            "\n"
            "   You should have received a copy of the GNU General Public "
            "License\n"
            "   along with this program.  If not, see "
            "<https://www.gnu.org/licenses/>.  */\n\n";
      if (content.starts_with ("/*"))
        {
          const std::size_t end_c = content.find ("*/");
          if (end_c != std::string::npos && end_c < 400U)
            {
              std::size_t after = end_c + 2U;
              while (after < content.size ()
                     && (content[after] == '\n' || content[after] == '\r'))
                {
                  ++after;
                }
              content = std::string (kGplHeader) + content.substr (after);
            }
          else
            {
              content = std::string (kGplHeader) + content;
            }
        }
      else
        {
          content = std::string (kGplHeader) + content;
        }
    }

  std::vector<std::string> raw_lines;
  {
    std::string cur;
    for (const char ch : content)
      {
        if (ch == '\r')
          {
            continue;
          }
        if (ch == '\t')
          {
            cur.append ("  ");
            continue;
          }
        if (ch == '\n')
          {
            while (!cur.empty () && cur.back () == ' ')
              {
                cur.pop_back ();
              }
            raw_lines.push_back (std::move (cur));
            cur.clear ();
          }
        else
          {
            cur.push_back (ch);
          }
      }
    if (!cur.empty ())
      {
        while (!cur.empty () && cur.back () == ' ')
          {
            cur.pop_back ();
          }
        raw_lines.push_back (std::move (cur));
      }
  }

  // Pass 1: Insert space before '(' and split braces '{' / '}' onto their own
  // lines outside strings, character literals, and comments.
  std::vector<std::string> brace_lines;
  bool in_block_comment = false;

  for (const std::string &line : raw_lines)
    {
      std::size_t indent = 0U;
      while (indent < line.size () && line[indent] == ' ')
        {
          ++indent;
        }
      const std::string_view trimmed = std::string_view (line).substr (indent);
      if (in_block_comment || trimmed.empty () || trimmed.starts_with ('#'))
        {
          for (std::size_t i = 0U; i < line.size (); ++i)
            {
              if (in_block_comment)
                {
                  if (i + 1U < line.size () && line[i] == '*'
                      && line[i + 1U] == '/')
                    {
                      in_block_comment = false;
                      ++i;
                    }
                }
              else if (i + 1U < line.size () && line[i] == '/'
                       && line[i + 1U] == '*')
                {
                  in_block_comment = true;
                  ++i;
                }
              else if (i + 1U < line.size () && line[i] == '/'
                       && line[i + 1U] == '/')
                {
                  break;
                }
            }
          brace_lines.push_back (line);
          continue;
        }

      std::string spaced;
      spaced.reserve (line.size () + 8U);
      bool in_str = false;
      bool in_chr = false;
      for (std::size_t i = 0U; i < line.size (); ++i)
        {
          if (in_block_comment)
            {
              spaced.push_back (line[i]);
              if (i + 1U < line.size () && line[i] == '*'
                  && line[i + 1U] == '/')
                {
                  spaced.push_back (line[++i]);
                  in_block_comment = false;
                }
              continue;
            }
          if (!in_str && !in_chr && i + 1U < line.size () && line[i] == '/'
              && line[i + 1U] == '*')
            {
              spaced.push_back (line[i]);
              spaced.push_back (line[++i]);
              in_block_comment = true;
              continue;
            }
          if (!in_str && !in_chr && i + 1U < line.size () && line[i] == '/'
              && line[i + 1U] == '/')
            {
              spaced.append (line.substr (i));
              break;
            }
          if (in_str)
            {
              spaced.push_back (line[i]);
              if (line[i] == '\\' && i + 1U < line.size ())
                {
                  spaced.push_back (line[++i]);
                }
              else if (line[i] == '"')
                {
                  in_str = false;
                }
              continue;
            }
          if (in_chr)
            {
              spaced.push_back (line[i]);
              if (line[i] == '\\' && i + 1U < line.size ())
                {
                  spaced.push_back (line[++i]);
                }
              else if (line[i] == '\'')
                {
                  in_chr = false;
                }
              continue;
            }
          if (line[i] == '"')
            {
              in_str = true;
              spaced.push_back (line[i]);
              continue;
            }
          if (line[i] == '\'')
            {
              in_chr = true;
              spaced.push_back (line[i]);
              continue;
            }
          if (line[i] == '(' && !spaced.empty ())
            {
              const auto prev
                  = static_cast<unsigned char> (spaced.back ());
              if (std::isalnum (prev) != 0 || prev == '_')
                {
                  std::size_t id_start = spaced.size ();
                  while (id_start > 0U
                         && (std::isalnum (static_cast<unsigned char> (
                                 spaced[id_start - 1U]))
                                 != 0
                             || spaced[id_start - 1U] == '_'))
                    {
                      --id_start;
                    }
                  const std::string_view ident
                      = std::string_view (spaced).substr (id_start);
                  if (!is_gnu_style_exempt_ident (ident))
                    {
                      spaced.push_back (' ');
                    }
                }
            }
          spaced.push_back (line[i]);
        }

      while (!spaced.empty () && spaced.back () == ' ')
        {
          spaced.pop_back ();
        }

      // Check braces outside strings/comments on spaced.
      std::size_t open_pos = std::string::npos;
      std::size_t close_pos = std::string::npos;
      bool b_str = false;
      bool b_chr = false;
      bool b_blk = false;
      for (std::size_t i = 0U; i < spaced.size (); ++i)
        {
          if (b_blk)
            {
              if (i + 1U < spaced.size () && spaced[i] == '*'
                  && spaced[i + 1U] == '/')
                {
                  b_blk = false;
                  ++i;
                }
              continue;
            }
          if (!b_str && !b_chr && i + 1U < spaced.size () && spaced[i] == '/'
              && spaced[i + 1U] == '*')
            {
              b_blk = true;
              ++i;
              continue;
            }
          if (!b_str && !b_chr && i + 1U < spaced.size () && spaced[i] == '/'
              && spaced[i + 1U] == '/')
            {
              break;
            }
          if (b_str)
            {
              if (spaced[i] == '\\' && i + 1U < spaced.size ())
                {
                  ++i;
                }
              else if (spaced[i] == '"')
                {
                  b_str = false;
                }
              continue;
            }
          if (b_chr)
            {
              if (spaced[i] == '\\' && i + 1U < spaced.size ())
                {
                  ++i;
                }
              else if (spaced[i] == '\'')
                {
                  b_chr = false;
                }
              continue;
            }
          if (spaced[i] == '"')
            {
              b_str = true;
              continue;
            }
          if (spaced[i] == '\'')
            {
              b_chr = true;
              continue;
            }
          if (spaced[i] == '{' && open_pos == std::string::npos)
            {
              open_pos = i;
            }
          else if (spaced[i] == '}' && close_pos == std::string::npos)
            {
              close_pos = i;
            }
        }

      std::size_t sp_indent = 0U;
      while (sp_indent < spaced.size () && spaced[sp_indent] == ' ')
        {
          ++sp_indent;
        }
      const std::string_view sp_trim
          = std::string_view (spaced).substr (sp_indent);

      if (open_pos != std::string::npos && close_pos == std::string::npos
          && sp_trim != "{")
        {
          std::string prefix = spaced.substr (0U, open_pos);
          while (!prefix.empty () && prefix.back () == ' ')
            {
              prefix.pop_back ();
            }
          std::string suffix = spaced.substr (open_pos + 1U);
          std::size_t suf_ws = 0U;
          while (suf_ws < suffix.size () && suffix[suf_ws] == ' ')
            {
              ++suf_ws;
            }
          suffix.erase (0U, suf_ws);
          if (!prefix.empty ())
            {
              brace_lines.push_back (prefix);
            }
          brace_lines.push_back (std::string (sp_indent, ' ') + "{");
          if (!suffix.empty ())
            {
              brace_lines.push_back (std::string (sp_indent + 2U, ' ')
                                     + suffix);
            }
          continue;
        }

      if (close_pos != std::string::npos && open_pos == std::string::npos
          && !sp_trim.starts_with ('}'))
        {
          std::string prefix = spaced.substr (0U, close_pos);
          while (!prefix.empty () && prefix.back () == ' ')
            {
              prefix.pop_back ();
            }
          std::string suffix = spaced.substr (close_pos);
          if (!prefix.empty ())
            {
              brace_lines.push_back (prefix);
            }
          brace_lines.push_back (std::string (sp_indent, ' ') + suffix);
          continue;
        }

      brace_lines.push_back (std::move (spaced));
    }

  // Pass 2: Wrap lines exceeding 79 columns.
  std::vector<std::string> wrapped_lines;
  for (std::string line : brace_lines)
    {
      while (line.size () > 79U)
        {
          std::size_t indent = 0U;
          while (indent < line.size () && line[indent] == ' ')
            {
              ++indent;
            }
          const std::string_view trim
              = std::string_view (line).substr (indent);
          if (trim.starts_with ("//"))
            {
              std::size_t split_at = std::string::npos;
              for (std::size_t i = 79U; i > indent + 3U; --i)
                {
                  if (line[i] == ' ')
                    {
                      split_at = i;
                      break;
                    }
                }
              if (split_at != std::string::npos)
                {
                  std::string head = line.substr (0U, split_at);
                  while (!head.empty () && head.back () == ' ')
                    {
                      head.pop_back ();
                    }
                  wrapped_lines.push_back (std::move (head));
                  line = std::string (indent, ' ') + "// "
                         + line.substr (split_at + 1U);
                  continue;
                }
            }

          // Find safe break point outside strings <= 79 columns.
          bool in_s = false;
          bool in_c = false;
          std::size_t best_break = std::string::npos;
          bool break_after = false;
          const std::size_t min_col = std::min<std::size_t> (indent + 8U, 36U);
          for (std::size_t i = 0U; i < std::min<std::size_t> (line.size (),
                                                              79U);
               ++i)
            {
              if (in_s)
                {
                  if (line[i] == '\\' && i + 1U < line.size ())
                    {
                      ++i;
                    }
                  else if (line[i] == '"')
                    {
                      in_s = false;
                    }
                  continue;
                }
              if (in_c)
                {
                  if (line[i] == '\\' && i + 1U < line.size ())
                    {
                      ++i;
                    }
                  else if (line[i] == '\'')
                    {
                      in_c = false;
                    }
                  continue;
                }
              if (line[i] == '"')
                {
                  in_s = true;
                  continue;
                }
              if (line[i] == '\'')
                {
                  in_c = true;
                  continue;
                }
              if (i < min_col)
                {
                  continue;
                }
              if (line[i] == ',' || line[i] == '(' || line[i] == ';')
                {
                  best_break = i;
                  break_after = true;
                }
              else if (line[i] == ' ' && i + 1U < line.size ()
                       && (line[i + 1U] == '|' || line[i + 1U] == '&'
                           || line[i + 1U] == '+' || line[i + 1U] == '?'
                           || line[i + 1U] == ':' || line[i + 1U] == '='
                           || line[i + 1U] == '<' || line[i + 1U] == '>'))
                {
                  best_break = i;
                  break_after = false;
                }
              else if (line[i] == ' ' && best_break == std::string::npos)
                {
                  best_break = i;
                  break_after = false;
                }
            }

          const std::size_t cont_indent
              = std::min<std::size_t> (indent + 4U, 24U);
          if (best_break != std::string::npos)
            {
              const std::size_t cut
                  = break_after ? (best_break + 1U) : best_break;
              std::string head = line.substr (0U, cut);
              while (!head.empty () && head.back () == ' ')
                {
                  head.pop_back ();
                }
              std::string tail = line.substr (cut);
              std::size_t t_ws = 0U;
              while (t_ws < tail.size () && tail[t_ws] == ' ')
                {
                  ++t_ws;
                }
              tail.erase (0U, t_ws);
              if (!head.empty () && !tail.empty ())
                {
                  wrapped_lines.push_back (std::move (head));
                  line = std::string (cont_indent, ' ') + tail;
                  continue;
                }
            }

          // Fallback: if inside a string literal at column 78, split the
          // string literal into adjacent C++ string tokens.
          if (in_s && !trim.starts_with ('#'))
            {
              std::size_t split_str = 77U;
              while (split_str > min_col
                     && ((static_cast<unsigned char> (line[split_str]) & 0xC0U)
                             == 0x80U
                         || line[split_str - 1U] == '\\'))
                {
                  --split_str;
                }
              if (split_str > min_col)
                {
                  std::string head = line.substr (0U, split_str) + "\"";
                  std::string tail = "\"" + line.substr (split_str);
                  wrapped_lines.push_back (std::move (head));
                  line = std::string (cont_indent, ' ') + tail;
                  continue;
                }
            }
          break;
        }
      while (!line.empty () && line.back () == ' ')
        {
          line.pop_back ();
        }
      wrapped_lines.push_back (std::move (line));
    }

  std::string out;
  out.reserve (content.size () + 256U);
  for (const std::string &l : wrapped_lines)
    {
      out.append (l);
      out.push_back ('\n');
    }
  return out;
}

auto
tool_write_file (const sutra::ToolContext & /*ctx*/, std::string path,
                 std::string content) noexcept -> sutra::Result<std::string>
{
  const auto t0 = std::chrono::steady_clock::now ();
  content = normalize_gnu_cpp_style (path, std::move (content));
  const std::string input_repr = path + ":" + content;

  auto path_res = resolve_sandboxed_path (path);
  if (!path_res)
    {
      const std::string base_str = ToolEnvironment::sandbox_dir ().string ();
      const auto err = sutra::AgentError::ToolSandboxViolation (
          "Cannot write to " + path
          + ": path is outside the sandbox. The allowed root is " + base_str
          + ". Use a relative path within the sandbox, or omit the leading "
            "path components.");
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      record_tool_telemetry ("write_file", input_repr, err.message (), ms,
                             true);
      return std::unexpected (err);
    }

  try
    {
      std::error_code ec;
      if (path_res->has_parent_path ())
        {
          std::filesystem::create_directories (path_res->parent_path (), ec);
        }
      std::ofstream out (*path_res, std::ios::binary | std::ios::trunc);
      if (!out.is_open ())
        {
          const auto err = sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "write_file",
              "Failed to open file for writing: " + path_res->string ());
          const double ms
              = std::chrono::duration<double, std::milli> (
                    std::chrono::steady_clock::now () - t0)
                    .count ();
          record_tool_telemetry ("write_file", input_repr, err.message (), ms,
                                 true);
          return std::unexpected (err);
        }
      out.write (content.data (),
                 static_cast<std::streamsize> (content.size ()));
      out.flush ();
      std::string msg
          = "wrote " + std::to_string (content.size ()) + " bytes to " + path;
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      record_tool_telemetry ("write_file", input_repr, msg, ms, false);
      return msg;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "write_file",
          "Exception writing file"));
    }
}

auto
tool_edit_file (const sutra::ToolContext & /*ctx*/, std::string path,
                std::string target,
                std::string replacement) noexcept -> sutra::Result<std::string>
{
  const auto t0 = std::chrono::steady_clock::now ();
  const std::string input_repr = path + ":" + target + "->" + replacement;

  auto path_res = resolve_sandboxed_path (path);
  if (!path_res)
    {
      const std::string base_str = ToolEnvironment::sandbox_dir ().string ();
      const auto err = sutra::AgentError::ToolSandboxViolation (
          "Cannot edit " + path
          + ": path is outside the sandbox. The allowed root is " + base_str);
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      record_tool_telemetry ("edit_file", input_repr, err.message (), ms,
                             true);
      return std::unexpected (err);
    }

  if (target.empty ())
    {
      const auto err = sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "edit_file",
          "target string must not be empty");
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      record_tool_telemetry ("edit_file", input_repr, err.message (), ms,
                             true);
      return std::unexpected (err);
    }

  try
    {
      std::ifstream in (*path_res, std::ios::binary);
      if (!in.is_open ())
        {
          const auto err = sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "edit_file",
              "Failed to open file for reading: " + path_res->string ());
          const double ms
              = std::chrono::duration<double, std::milli> (
                    std::chrono::steady_clock::now () - t0)
                    .count ();
          record_tool_telemetry ("edit_file", input_repr, err.message (), ms,
                                 true);
          return std::unexpected (err);
        }
      std::ostringstream ss;
      ss << in.rdbuf ();
      in.close ();
      std::string content = ss.str ();

      const std::size_t pos = content.find (target);
      if (pos == std::string::npos)
        {
          const auto err = sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "edit_file",
              "Target string not found in " + path
                  + ". Ensure exact whitespace match.");
          const double ms
              = std::chrono::duration<double, std::milli> (
                    std::chrono::steady_clock::now () - t0)
                    .count ();
          record_tool_telemetry ("edit_file", input_repr, err.message (), ms,
                                 true);
          return std::unexpected (err);
        }

      std::size_t count = 0U;
      std::size_t search_pos = 0U;
      while ((search_pos = content.find (target, search_pos))
             != std::string::npos)
        {
          content.replace (search_pos, target.size (), replacement);
          search_pos += replacement.size ();
          ++count;
        }

      std::ofstream out (*path_res, std::ios::binary | std::ios::trunc);
      if (!out.is_open ())
        {
          const auto err = sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "edit_file",
              "Failed to open file for writing: " + path_res->string ());
          const double ms
              = std::chrono::duration<double, std::milli> (
                    std::chrono::steady_clock::now () - t0)
                    .count ();
          record_tool_telemetry ("edit_file", input_repr, err.message (), ms,
                                 true);
          return std::unexpected (err);
        }
      out.write (content.data (),
                 static_cast<std::streamsize> (content.size ()));
      out.flush ();
      std::string msg = "edited " + path + " (" + std::to_string (count)
                        + " replacement(s), "
                        + std::to_string (content.size ()) + " bytes)";
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      record_tool_telemetry ("edit_file", input_repr, msg, ms, false);
      return msg;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "edit_file",
          "Exception editing file"));
    }
}

auto
tool_list_dir (const sutra::ToolContext & /*ctx*/,
               std::string path) noexcept -> sutra::Result<std::string>
{
  const auto t0 = std::chrono::steady_clock::now ();
  auto path_res = resolve_sandboxed_path (path);
  if (!path_res)
    {
      const std::string base_str = ToolEnvironment::sandbox_dir ().string ();
      const auto err = sutra::AgentError::ToolSandboxViolation (
          "Cannot list directory " + path
          + ": path is outside the sandbox. The allowed root is " + base_str
          + ". Use a relative path within the sandbox, or omit the leading "
            "path components.");
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      record_tool_telemetry ("list_dir", path, err.message (), ms, true);
      return std::unexpected (err);
    }

  try
    {
      std::error_code ec;
      if (!std::filesystem::exists (*path_res, ec)
          || !std::filesystem::is_directory (*path_res, ec))
        {
          const auto err = sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "list_dir",
              "Directory does not exist: " + path_res->string ());
          const double ms
              = std::chrono::duration<double, std::milli> (
                    std::chrono::steady_clock::now () - t0)
                    .count ();
          record_tool_telemetry ("list_dir", path, err.message (), ms, true);
          return std::unexpected (err);
        }

      std::vector<std::string> names;
      for (const auto &entry :
           std::filesystem::directory_iterator (*path_res, ec))
        {
          std::string name = entry.path ().filename ().string ();
          if (entry.is_directory (ec))
            {
              name.push_back ('/');
            }
          names.push_back (std::move (name));
        }
      std::ranges::sort (names);

      std::string out;
      for (const std::string &n : names)
        {
          out.append (n);
          out.push_back ('\n');
        }
      out = cap_tool_output (std::move (out));
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      record_tool_telemetry ("list_dir", path, out, ms, false);
      return out;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "list_dir",
          "Exception listing directory"));
    }
}

auto
tool_run_command (const sutra::ToolContext & /*ctx*/, std::string cmd,
                  std::string cwd) noexcept -> sutra::Result<std::string>
{
  const auto t0 = std::chrono::steady_clock::now ();
  const std::string input_repr = cmd + " @" + cwd;

  auto allow_res = validate_allowed_command (cmd);
  if (!allow_res)
    {
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      record_tool_telemetry ("run_command", input_repr,
                             allow_res.error ().message (), ms, true);
      return std::unexpected (allow_res.error ());
    }

  auto cwd_res = resolve_sandboxed_path (cwd);
  if (!cwd_res)
    {
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      record_tool_telemetry ("run_command", input_repr,
                             cwd_res.error ().message (), ms, true);
      return std::unexpected (cwd_res.error ());
    }

  try
    {
      const std::vector<std::string> args = split_command_args (cmd);
      const std::filesystem::path sb_root = ToolEnvironment::sandbox_dir ();
      if (*cwd_res != sb_root)
        {
          std::error_code ec;
          const std::filesystem::path rel_cwd
              = std::filesystem::relative (*cwd_res, sb_root, ec);
          if (!ec && !rel_cwd.empty () && rel_cwd != ".")
            {
              const std::string pref = rel_cwd.string () + "/";
              for (std::size_t i = 1U; i < args.size (); ++i)
                {
                  if (args[i].starts_with (pref)
                      && std::filesystem::exists (sb_root / args[i], ec)
                      && !std::filesystem::exists (*cwd_res / args[i], ec))
                    {
                      *cwd_res = sb_root;
                      break;
                    }
                }
            }
        }
      uint32_t exec_timeout_ms = ToolEnvironment::command_timeout_ms ();
      if (!args.empty () && args[0].starts_with ("./"))
        {
          const std::filesystem::path bin_cand
              = (*cwd_res) / std::string_view (args[0]).substr (2U);
          auto bin_path = resolve_sandboxed_path (bin_cand.string ());
          if (!bin_path)
            {
              const double ms
                  = std::chrono::duration<double, std::milli> (
                        std::chrono::steady_clock::now () - t0)
                        .count ();
              record_tool_telemetry ("run_command", input_repr,
                                     bin_path.error ().message (), ms, true);
              return std::unexpected (bin_path.error ());
            }
          struct stat st{};
          if (::stat (bin_path->c_str (), &st) != 0 || !S_ISREG (st.st_mode)
              || (st.st_mode & S_IXUSR) == 0)
            {
              const auto err = sutra::AgentError::ToolNotAllowed (
                  "Cannot run '" + args[0]
                  + "': target is not an executable regular file inside "
                  + ToolEnvironment::sandbox_dir ().string () + ".");
              const double ms
                  = std::chrono::duration<double, std::milli> (
                        std::chrono::steady_clock::now () - t0)
                        .count ();
              record_tool_telemetry ("run_command", input_repr, err.message (),
                                     ms, true);
              return std::unexpected (err);
            }
          const uint32_t bin_cap_ms
              = (args.size () > 1U) ? 15000U : 5000U;
          exec_timeout_ms = std::min<uint32_t> (exec_timeout_ms, bin_cap_ms);
        }
      auto arg_check = validate_command_arguments_in_sandbox (args, *cwd_res);
      if (!arg_check)
        {
          const double ms
              = std::chrono::duration<double, std::milli> (
                    std::chrono::steady_clock::now () - t0)
                    .count ();
          record_tool_telemetry ("run_command", input_repr,
                                 arg_check.error ().message (), ms, true);
          return std::unexpected (arg_check.error ());
        }

      std::vector<std::string> exec_args;
      exec_args.reserve (args.size () + 8U);
      exec_args.push_back (args[0]);
      bool had_glob = false;
      bool any_glob_matched = false;
      for (std::size_t i = 1U; i < args.size (); ++i)
        {
          const std::string &arg = args[i];
          if (arg.find ('*') == std::string::npos
              && arg.find ('?') == std::string::npos)
            {
              exec_args.push_back (arg);
              continue;
            }
          had_glob = true;
          std::string_view pat = arg;
          if (pat.starts_with ("./"))
            {
              pat.remove_prefix (2U);
            }
          std::error_code ec;
          std::vector<std::string> matches;
          for (const auto &entry :
               std::filesystem::directory_iterator (*cwd_res, ec))
            {
              const std::string fname = entry.path ().filename ().string ();
              if (fname.empty () || fname.front () == '.'
                  || fname == ".memory")
                {
                  continue;
                }
              if (match_wildcard_glob (pat, fname))
                {
                  matches.push_back (fname);
                }
            }
          std::ranges::sort (matches);
          if (!matches.empty ())
            {
              any_glob_matched = true;
              for (std::string &m : matches)
                {
                  exec_args.push_back (std::move (m));
                }
            }
          else if (basename_view (args[0]) != "rm")
            {
              exec_args.push_back (arg);
            }
        }

      if (had_glob && !any_glob_matched && basename_view (args[0]) == "rm")
        {
          const std::string msg
              = "Command '" + cmd + "' completed successfully (exit 0).";
          const double ms
              = std::chrono::duration<double, std::milli> (
                    std::chrono::steady_clock::now () - t0)
                    .count ();
          record_tool_telemetry ("run_command", input_repr, msg, ms, false);
          return msg;
        }

      int exit_code = 0;
      auto exec_res = execute_argv_with_timeout (
          exec_args, *cwd_res, exec_timeout_ms, &exit_code);
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      if (!exec_res)
        {
          record_tool_telemetry ("run_command", input_repr,
                                 exec_res.error ().message (), ms, true);
          return std::unexpected (exec_res.error ());
        }
      std::string out = *exec_res;
      if (exit_code != 0)
        {
          const std::string err_msg
              = "Exit " + std::to_string (exit_code) + ": " + out;
          record_tool_telemetry ("run_command", input_repr, err_msg, ms, true);
          return std::unexpected (sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "run_command", err_msg));
        }
      if (out.empty () && !args[0].starts_with ("./"))
        {
          out = "Command '" + cmd + "' completed successfully (exit 0).";
        }
      record_tool_telemetry ("run_command", input_repr, out, ms, false);
      return out;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "run_command",
          "Exception executing command"));
    }
}

auto
tool_compile_check (const sutra::ToolContext & /*ctx*/, std::string file,
                    std::string flags) noexcept -> sutra::Result<std::string>
{
  const auto t0 = std::chrono::steady_clock::now ();
  const std::string input_repr = file + " " + flags;

  auto file_res = resolve_sandboxed_path (file);
  if (!file_res)
    {
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      record_tool_telemetry ("compile_check", input_repr,
                             file_res.error ().message (), ms, true);
      return std::unexpected (file_res.error ());
    }

  try
    {
      std::vector<std::string> args
          = { "g++", "-fsyntax-only", "-std=c++23", "-Wall", "-Wextra" };
      std::error_code inc_ec;
      const std::filesystem::path aurora_inc
          = ToolEnvironment::sandbox_dir () / "Aurora" / "src";
      if (std::filesystem::is_directory (aurora_inc, inc_ec))
        {
          args.push_back ("-I" + aurora_inc.string ());
        }
      for (std::string &extra : split_command_args (flags))
        {
          args.push_back (std::move (extra));
        }
      args.push_back (file_res->string ());

      int exit_code = 0;
      auto exec_res = execute_argv_with_timeout (
          args, ToolEnvironment::sandbox_dir (),
          ToolEnvironment::command_timeout_ms (), &exit_code);
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      if (!exec_res)
        {
          record_tool_telemetry ("compile_check", input_repr,
                                 exec_res.error ().message (), ms, true);
          return std::unexpected (exec_res.error ());
        }
      std::string out = exec_res->empty () ? "ok" : *exec_res;
      if (exit_code != 0)
        {
          const std::string err_msg
              = "Exit " + std::to_string (exit_code) + ": " + out;
          record_tool_telemetry ("compile_check", input_repr, err_msg, ms,
                                 true);
          return std::unexpected (sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "compile_check",
              err_msg));
        }
      record_tool_telemetry ("compile_check", input_repr, out, ms, false);
      return out;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "compile_check",
          "Exception running compile_check"));
    }
}

auto
tool_ssh_exec (const sutra::ToolContext & /*ctx*/, std::string host,
               std::string cmd) noexcept -> sutra::Result<std::string>
{
  const auto t0 = std::chrono::steady_clock::now ();
  const std::string input_repr = host + ":" + cmd;
  auto res = execute_libssh2_session (host, cmd, false);
  const double ms
      = std::chrono::duration<double, std::milli> (
            std::chrono::steady_clock::now () - t0)
            .count ();
  if (!res)
    {
      record_tool_telemetry ("ssh_exec", input_repr, res.error ().message (),
                             ms, true);
      return std::unexpected (res.error ());
    }
  record_tool_telemetry ("ssh_exec", input_repr, *res, ms, false);
  return *res;
}

auto
tool_ssh_test (const sutra::ToolContext & /*ctx*/,
               std::string host) noexcept -> sutra::Result<std::string>
{
  const auto t0 = std::chrono::steady_clock::now ();
  auto res = execute_libssh2_session (host, "", true);
  const double ms
      = std::chrono::duration<double, std::milli> (
            std::chrono::steady_clock::now () - t0)
            .count ();
  if (!res)
    {
      record_tool_telemetry ("ssh_test", host, res.error ().message (), ms,
                             true);
      return std::unexpected (res.error ());
    }
  record_tool_telemetry ("ssh_test", host, *res, ms, false);
  return *res;
}

auto
make_directory (std::string_view path) noexcept
    -> sutra::Result<std::filesystem::path>
{
  if (path.find ('\0') != std::string_view::npos)
    {
      return std::unexpected (sutra::AgentError::ToolSandboxViolation (
          "Path contains NUL byte in make_directory"));
    }

  const std::string_view trimmed = trim_ws (path);
  if (trimmed.empty ())
    {
      return std::unexpected (sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "make_directory",
          "Cannot create directory: empty path"));
    }

  try
    {
      const std::filesystem::path sandbox_root
          = ToolEnvironment::sandbox_dir ();
      const std::string base_str = sandbox_root.string ();

      const std::filesystem::path rel_path (trimmed);
      if (rel_path.is_absolute () || trimmed.starts_with ('/'))
        {
          std::error_code tmp_ec;
          const std::filesystem::path canon_tmp
              = std::filesystem::weakly_canonical (rel_path, tmp_ec);
          if (!tmp_ec && !canon_tmp.empty ()
              && is_allowed_temp_qa_path (trimmed, canon_tmp.string ()))
            {
              if (std::filesystem::exists (canon_tmp, tmp_ec))
                {
                  if (std::filesystem::is_directory (canon_tmp, tmp_ec))
                    {
                      return canon_tmp;
                    }
                  return std::unexpected (
                      sutra::AgentError::ToolSandboxViolation (
                          "Target already exists as a file: "
                          + canon_tmp.string ()));
                }
              (void)std::filesystem::create_directories (canon_tmp, tmp_ec);
              if (tmp_ec)
                {
                  return std::unexpected (sutra::AgentError::tool (
                      sutra::ToolErrorCode::ExecutionFailure,
                      "make_directory",
                      "Failed to create directory: " + canon_tmp.string ()));
                }
              return canon_tmp;
            }
          return std::unexpected (sutra::AgentError::ToolSandboxViolation (
              "Cannot create directory " + std::string (trimmed)
              + ": path is outside the sandbox. The allowed root is "
              + base_str
              + ". Use a relative path within the sandbox, or omit the "
                "leading path components."));
        }

      for (const auto &comp : rel_path)
        {
          const std::string comp_str = comp.string ();
          if (comp_str == "..")
            {
              return std::unexpected (sutra::AgentError::ToolSandboxViolation (
                  "Cannot create directory " + std::string (trimmed)
                  + ": path is outside the sandbox. The allowed root is "
                  + base_str
                  + ". Use a relative path within the sandbox, or omit the "
                    "leading path components."));
            }
          if (comp_str.size () > 255U)
            {
              return std::unexpected (sutra::AgentError::tool (
                  sutra::ToolErrorCode::ExecutionFailure, "make_directory",
                  "Path component exceeds 255 bytes in make_directory"));
            }
        }

      // Check intermediate components for symlinks escaping the sandbox or
      // regular files blocking directory creation.
      std::filesystem::path cur = sandbox_root;
      std::error_code ec;
      for (const auto &comp : rel_path)
        {
          cur /= comp;
          const std::filesystem::path canon_step
              = std::filesystem::weakly_canonical (cur, ec);
          if (ec || canon_step.empty ())
            {
              return std::unexpected (sutra::AgentError::ToolSandboxViolation (
                  "Unresolvable path in make_directory: "
                  + std::string (trimmed)));
            }
          const std::string step_str = canon_step.string ();
          if (step_str != base_str && !step_str.starts_with (base_str + "/"))
            {
              return std::unexpected (sutra::AgentError::ToolSandboxViolation (
                  "Cannot create directory " + std::string (trimmed)
                  + ": path is outside the sandbox. The allowed root is "
                  + base_str
                  + ". Use a relative path within the sandbox, or omit the "
                    "leading path components."));
            }
          if (std::filesystem::exists (canon_step, ec)
              && !std::filesystem::is_directory (canon_step, ec))
            {
              return std::unexpected (sutra::AgentError::ToolSandboxViolation (
                  "Target exists as a non-directory file: " + step_str));
            }
        }

      const std::filesystem::path target = sandbox_root / rel_path;
      const std::filesystem::path canon_target
          = std::filesystem::weakly_canonical (target, ec);
      if (ec || canon_target.empty ())
        {
          return std::unexpected (sutra::AgentError::ToolSandboxViolation (
              "Invalid target path in make_directory: "
              + std::string (trimmed)));
        }

      const std::string target_str = canon_target.string ();
      if (target_str != base_str && !target_str.starts_with (base_str + "/"))
        {
          return std::unexpected (sutra::AgentError::ToolSandboxViolation (
              "Cannot create directory " + std::string (trimmed)
              + ": path is outside the sandbox. The allowed root is "
              + base_str
              + ". Use a relative path within the sandbox, or omit the "
                "leading path components."));
        }

      if (std::filesystem::exists (canon_target, ec))
        {
          if (std::filesystem::is_directory (canon_target, ec))
            {
              return canon_target;
            }
          return std::unexpected (sutra::AgentError::ToolSandboxViolation (
              "Target already exists as a file: " + target_str));
        }

      try
        {
          (void)std::filesystem::create_directories (canon_target);
        }
      catch (const std::filesystem::filesystem_error &fe)
        {
          if (fe.code () == std::errc::file_exists
              || fe.code () == std::errc::not_a_directory)
            {
              return std::unexpected (sutra::AgentError::ToolSandboxViolation (
                  "Target exists as a file: " + std::string (fe.what ())));
            }
          if (fe.code () == std::errc::permission_denied)
            {
              return std::unexpected (sutra::AgentError::tool (
                  sutra::ToolErrorCode::NotAllowed, "make_directory",
                  "Permission denied creating directory: "
                  + std::string (fe.what ())));
            }
          return std::unexpected (sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "make_directory",
              "Failed to create directory: " + std::string (fe.what ())));
        }

      return canon_target;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "make_directory",
          "Exception in make_directory"));
    }
}

auto
tool_echo (const sutra::ToolContext & /*ctx*/,
           std::string text) noexcept -> sutra::Result<std::string>
{
  const auto t0 = std::chrono::steady_clock::now ();
  std::string out = cap_tool_output (text);
  const double ms
      = std::chrono::duration<double, std::milli> (
            std::chrono::steady_clock::now () - t0)
            .count ();
  record_tool_telemetry ("echo", text, out, ms, false);
  return out;
}

auto
tool_make_directory (const sutra::ToolContext & /*ctx*/,
                     std::string path) noexcept -> sutra::Result<std::string>
{
  const auto t0 = std::chrono::steady_clock::now ();
  auto res = make_directory (path);
  const double ms
      = std::chrono::duration<double, std::milli> (
            std::chrono::steady_clock::now () - t0)
            .count ();
  if (!res)
    {
      record_tool_telemetry ("make_directory", path, res.error ().message (),
                             ms, true);
      return std::unexpected (res.error ());
    }
  std::string out = res->string ();
  record_tool_telemetry ("make_directory", path, out, ms, false);
  return out;
}

auto
run_readonly (std::string_view command,
              std::span<const std::string_view> args) noexcept
    -> sutra::Result<CommandResult>
{
  const std::string_view cmd = trim_ws (command);
  if (cmd.empty ())
    {
      return std::unexpected (sutra::AgentError::ToolNotAllowed (
          "Command must not be empty in run_readonly."));
    }

  if (cmd == "sudo" || cmd.ends_with ("sudo"))
    {
      return std::unexpected (sutra::AgentError::ToolNotAllowed (
          "Cannot run '" + std::string (cmd)
          + "': sudo is forbidden in run_readonly."));
    }

  if (cmd.find ('/') != std::string_view::npos)
    {
      return std::unexpected (sutra::AgentError::ToolNotAllowed (
          "Cannot run '" + std::string (cmd)
          + "': path separators ('/') are forbidden in run_readonly."));
    }

  constexpr std::string_view kForbiddenMeta = ";|&`$()<>\n\r";
  for (const char ch : cmd)
    {
      if (kForbiddenMeta.find (ch) != std::string_view::npos)
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "Cannot run '" + std::string (cmd)
              + "': command contains forbidden shell metacharacter."));
        }
    }

  for (const std::string_view arg : args)
    {
      if (arg == "sudo" || arg.ends_with ("sudo")
          || contains_icase (arg, "sudo"))
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "Argument '" + std::string (arg)
              + "' is forbidden: sudo is not allowed in run_readonly."));
        }
      for (const char ch : arg)
        {
          if (kForbiddenMeta.find (ch) != std::string_view::npos)
            {
              return std::unexpected (sutra::AgentError::ToolNotAllowed (
                  "Argument '" + std::string (arg)
                  + "' contains forbidden shell metacharacter in "
                    "run_readonly."));
            }
        }
      if (arg.starts_with ("--vacuum"))
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "Argument '" + std::string (arg)
              + "' is forbidden: --vacuum is not allowed in run_readonly."));
        }
    }

  constexpr std::array<std::string_view, 25U> kUnrestrictedCmds = { {
      "ls",        "ps",      "df",     "free",      "uptime",
      "uname",     "lscpu",   "nvidia-smi",          "ss",
      "whoami",    "id",      "groups", "hostname",  "lsblk",
      "blkid",     "findmnt", "stat",   "file",      "wc",
      "du",        "md5sum",  "sha256sum",           "sha1sum",
      "iwconfig",  "nmcli",
  } };

  bool is_unrestricted = false;
  for (const std::string_view u : kUnrestrictedCmds)
    {
      if (cmd == u)
        {
          is_unrestricted = true;
          break;
        }
    }

  auto find_first_non_flag = [&] () -> std::string_view {
    for (const std::string_view a : args)
      {
        if (!a.empty () && a.front () != '-')
          {
            return a;
          }
      }
    return {};
  };

  if (is_unrestricted)
    {
      // Unrestricted diagnostic command; args already validated by §3.3.
    }
  else if (cmd == "ip")
    {
      const std::string_view sub = find_first_non_flag ();
      constexpr std::array<std::string_view, 4U> kIpSubs
          = { { "addr", "route", "link", "neigh" } };
      bool ok = false;
      for (const std::string_view s : kIpSubs)
        {
          if (sub == s)
            {
              ok = true;
              break;
            }
        }
      if (!ok)
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "Subcommand '" + std::string (sub.empty () ? "<none>" : sub)
              + "' is not in the ip subcommand allowlist {addr, route, link, "
                "neigh}."));
        }
    }
  else if (cmd == "systemctl")
    {
      const std::string_view sub = find_first_non_flag ();
      constexpr std::array<std::string_view, 10U> kSysSubs = { {
          "status",
          "list-units",
          "list-unit-files",
          "list-timers",
          "is-active",
          "is-enabled",
          "is-failed",
          "show",
          "cat",
          "list-dependencies",
      } };
      bool ok = false;
      for (const std::string_view s : kSysSubs)
        {
          if (sub == s)
            {
              ok = true;
              break;
            }
        }
      if (!ok)
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "Subcommand '" + std::string (sub.empty () ? "<none>" : sub)
              + "' is not in the systemctl subcommand allowlist {status, "
                "list-units, list-unit-files, list-timers, is-active, "
                "is-enabled, is-failed, show, cat, list-dependencies}."));
        }
    }
  else if (cmd == "journalctl")
    {
      for (const std::string_view a : args)
        {
          if (a.starts_with ("--vacuum") || a == "--rotate"
              || a == "--flush")
            {
              return std::unexpected (sutra::AgentError::ToolNotAllowed (
                  "Argument '" + std::string (a)
                  + "' is not allowed for journalctl in run_readonly."));
            }
        }
    }
  else if (cmd == "apt")
    {
      const std::string_view sub = find_first_non_flag ();
      constexpr std::array<std::string_view, 4U> kAptSubs
          = { { "list", "search", "show", "policy" } };
      bool ok = false;
      for (const std::string_view s : kAptSubs)
        {
          if (sub == s)
            {
              ok = true;
              break;
            }
        }
      if (!ok)
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "Subcommand '" + std::string (sub.empty () ? "<none>" : sub)
              + "' is not in the apt subcommand allowlist {list, search, "
                "show, policy}."));
        }
    }
  else if (cmd == "dpkg-query")
    {
      const std::string_view first_arg
          = args.empty () ? std::string_view{} : args[0];
      if (first_arg != "-l" && first_arg != "-L" && first_arg != "-W")
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "Argument '"
              + std::string (first_arg.empty () ? "<none>" : first_arg)
              + "' is not in the dpkg-query allowlist {-l, -L, -W}."));
        }
    }
  else if (cmd == "hostnamectl")
    {
      if (!args.empty () && (args.size () > 1U || args[0] != "status"))
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "Argument '" + std::string (args[0])
              + "' is not in the hostnamectl allowlist (only no args or "
                "'status' is allowed)."));
        }
    }
  else if (cmd == "dmesg")
    {
      for (const std::string_view a : args)
        {
          if (a.starts_with ("-c") || a.starts_with ("-D")
              || a == "--clear" || a == "--read-clear")
            {
              return std::unexpected (sutra::AgentError::ToolNotAllowed (
                  "Argument '" + std::string (a)
                  + "' is forbidden for dmesg (clear/delete operations are "
                    "not allowed)."));
            }
        }
    }
  else
    {
      return std::unexpected (sutra::AgentError::ToolNotAllowed (
          "Command '" + std::string (cmd)
          + "' is not in the run_readonly diagnostic allowlist. Allowed "
            "commands: ls, ps, df, free, uptime, uname, lscpu, nvidia-smi, "
            "ip, ss, systemctl, journalctl, dpkg-query, apt, whoami, id, "
            "groups, hostname, hostnamectl, lsblk, blkid, findmnt, stat, "
            "file, wc, du, md5sum, sha256sum, sha1sum, iwconfig, nmcli, "
            "dmesg."));
    }

  try
    {
      std::array<int, 2U> out_pipe{ -1, -1 };
      std::array<int, 2U> err_pipe{ -1, -1 };
      if (pipe (out_pipe.data ()) != 0)
        {
          return std::unexpected (sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "run_readonly",
              "Failed to create stdout pipe"));
        }
      if (pipe (err_pipe.data ()) != 0)
        {
          close (out_pipe[0]);
          close (out_pipe[1]);
          return std::unexpected (sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "run_readonly",
              "Failed to create stderr pipe"));
        }

      const pid_t pid = fork ();
      if (pid < 0)
        {
          close (out_pipe[0]);
          close (out_pipe[1]);
          close (err_pipe[0]);
          close (err_pipe[1]);
          return std::unexpected (sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "run_readonly",
              "fork() failed in run_readonly"));
        }

      if (pid == 0)
        {
          (void)setpgid (0, 0);
          close (out_pipe[0]);
          close (err_pipe[0]);

          const int devnull_fd = ::open ("/dev/null", O_RDONLY);
          if (devnull_fd >= 0)
            {
              (void)dup2 (devnull_fd, STDIN_FILENO);
              if (devnull_fd != STDIN_FILENO)
                {
                  close (devnull_fd);
                }
            }

          (void)dup2 (out_pipe[1], STDOUT_FILENO);
          (void)dup2 (err_pipe[1], STDERR_FILENO);
          close (out_pipe[1]);
          close (err_pipe[1]);

          std::vector<std::string> storage;
          storage.reserve (args.size () + 1U);
          storage.emplace_back (cmd);
          for (const std::string_view a : args)
            {
              storage.emplace_back (a);
            }

          std::vector<char *> argv_ptrs;
          argv_ptrs.reserve (storage.size () + 1U);
          for (std::string &s : storage)
            {
              argv_ptrs.push_back (s.data ());
            }
          argv_ptrs.push_back (nullptr);

          execvp (argv_ptrs[0], argv_ptrs.data ());
          _exit (127);
        }

      close (out_pipe[1]);
      close (err_pipe[1]);

      const int fl_out = fcntl (out_pipe[0], F_GETFL, 0);
      if (fl_out >= 0)
        {
          (void)fcntl (out_pipe[0], F_SETFL, fl_out | O_NONBLOCK);
        }
      const int fl_err = fcntl (err_pipe[0], F_GETFL, 0);
      if (fl_err >= 0)
        {
          (void)fcntl (err_pipe[0], F_SETFL, fl_err | O_NONBLOCK);
        }

      CommandResult result;
      bool out_eof = false;
      bool err_eof = false;
      bool child_exited = false;
      int status = 0;

      const uint32_t timeout_ms = ToolEnvironment::readonly_timeout_ms ();
      const auto start_tp = std::chrono::steady_clock::now ();
      std::array<char, 4096U> buf{};

      auto drain_fd = [&] (int fd, std::string &dest,
                           std::size_t &trunc_cnt) -> bool {
        while (true)
          {
            const ssize_t n = read (fd, buf.data (), buf.size ());
            if (n > 0)
              {
                const auto u_n = static_cast<std::size_t> (n);
                if (dest.size () < kMaxReadonlyStreamBytes)
                  {
                    const std::size_t space
                        = kMaxReadonlyStreamBytes - dest.size ();
                    const std::size_t to_copy = std::min (space, u_n);
                    dest.append (buf.data (), to_copy);
                    if (u_n > to_copy)
                      {
                        trunc_cnt += (u_n - to_copy);
                      }
                  }
                else
                  {
                    trunc_cnt += u_n;
                  }
              }
            else if (n == 0)
              {
                return true;
              }
            else
              {
                if (errno == EINTR)
                  {
                    continue;
                  }
                return (errno != EAGAIN && errno != EWOULDBLOCK);
              }
          }
      };

      while (!out_eof || !err_eof || !child_exited)
        {
          std::array<struct pollfd, 2U> pfds{};
          pfds[0].fd = out_eof ? -1 : out_pipe[0];
          pfds[0].events = POLLIN;
          pfds[1].fd = err_eof ? -1 : err_pipe[0];
          pfds[1].events = POLLIN;

          if (!out_eof || !err_eof)
            {
              (void)poll (pfds.data (), 2U, kPollSliceMs);
              if (!out_eof)
                {
                  out_eof = drain_fd (out_pipe[0], result.stdout_str,
                                      result.truncated_stdout_bytes);
                }
              if (!err_eof)
                {
                  err_eof = drain_fd (err_pipe[0], result.stderr_str,
                                      result.truncated_stderr_bytes);
                }
            }
          else
            {
              struct pollfd dummy_pfd{ -1, 0, 0 };
              (void)poll (&dummy_pfd, 0U, kPollSliceMs);
            }

          if (!child_exited)
            {
              const pid_t w = waitpid (pid, &status, WNOHANG);
              if (w == pid)
                {
                  child_exited = true;
                  if (!out_eof)
                    {
                      out_eof = drain_fd (out_pipe[0], result.stdout_str,
                                          result.truncated_stdout_bytes);
                    }
                  if (!err_eof)
                    {
                      err_eof = drain_fd (err_pipe[0], result.stderr_str,
                                          result.truncated_stderr_bytes);
                    }
                  break;
                }
            }

          const auto elapsed_ms
              = std::chrono::duration_cast<std::chrono::milliseconds> (
                    std::chrono::steady_clock::now () - start_tp)
                    .count ();
          if (elapsed_ms >= static_cast<int64_t> (timeout_ms))
            {
              (void)kill (-pid, SIGTERM);
              (void)kill (pid, SIGTERM);
              const uint32_t grace_ms
                  = std::min<uint32_t> (kMaxSigkillGraceMs, timeout_ms);
              const auto term_tp = std::chrono::steady_clock::now ();
              bool reaped = false;
              while (std::chrono::duration_cast<std::chrono::milliseconds> (
                         std::chrono::steady_clock::now () - term_tp)
                         .count ()
                     < static_cast<int64_t> (grace_ms))
                {
                  if (waitpid (pid, &status, WNOHANG) == pid)
                    {
                      reaped = true;
                      break;
                    }
                  struct pollfd gpfd{ -1, 0, 0 };
                  (void)poll (&gpfd, 0U, 10);
                }
              if (!reaped)
                {
                  (void)kill (-pid, SIGKILL);
                  (void)kill (pid, SIGKILL);
                  (void)waitpid (pid, &status, 0);
                }
              close (out_pipe[0]);
              close (err_pipe[0]);
              return std::unexpected (sutra::AgentError::ToolTimeout (
                  "run_readonly('" + std::string (cmd)
                  + "') killed after 15s timeout. Partial output: "
                  + result.stdout_str));
            }
        }

      if (!child_exited)
        {
          (void)waitpid (pid, &status, 0);
        }
      close (out_pipe[0]);
      close (err_pipe[0]);

      if (WIFEXITED (status))
        {
          result.exit_code = WEXITSTATUS (status);
        }
      else if (WIFSIGNALED (status))
        {
          result.exit_code = 128 + WTERMSIG (status);
        }
      else
        {
          result.exit_code = -1;
        }

      if (result.exit_code == 127 && result.stdout_str.empty ()
          && result.stderr_str.empty ())
        {
          return std::unexpected (sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "run_readonly",
              "Command '" + std::string (cmd) + "' was not found in PATH"));
        }

      if (std::getenv ("SPHINX_DEBUG_TOOLS") != nullptr)
        {
          const double dur_ms
              = std::chrono::duration<double, std::milli> (
                    std::chrono::steady_clock::now () - start_tp)
                    .count ();
          (void)std::fprintf (
              stderr,
              "[debug] run_readonly: cmd=%.*s argc=%zu exit=%d duration=%.1f "
              "ms\n",
              static_cast<int> (cmd.size ()), cmd.data (), args.size (),
              result.exit_code, dur_ms);
        }

      return result;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "run_readonly",
          "Exception in run_readonly"));
    }
}

auto
read_file_system (std::string_view path) noexcept
    -> sutra::Result<std::string>
{
  if (path.empty ())
    {
      return std::unexpected (sutra::AgentError::ToolNotAllowed (
          "Cannot read <empty>: path must not be empty."));
    }

  if (path.find ('\0') != std::string_view::npos)
    {
      return std::unexpected (sutra::AgentError::ToolNotAllowed (
          "Cannot read path: path contains NUL byte."));
    }

  // Reject if any path component > 255 bytes.
  {
    std::size_t comp_start = 0U;
    while (comp_start <= path.size ())
      {
        const std::size_t slash = path.find ('/', comp_start);
        const std::size_t comp_end
            = (slash == std::string_view::npos) ? path.size () : slash;
        if (comp_end - comp_start > 255U)
          {
            return std::unexpected (sutra::AgentError::ToolNotAllowed (
                "Cannot read " + std::string (path)
                + ": path component exceeds 255 bytes."));
          }
        if (slash == std::string_view::npos)
          {
            break;
          }
        comp_start = slash + 1U;
      }
  }

  try
    {
      std::string home_str = get_home_dir ().string ();
      while (home_str.size () > 1U && home_str.back () == '/')
        {
          home_str.pop_back ();
        }

      std::string expanded (path);
      if (expanded == "~")
        {
          expanded = home_str;
        }
      else if (expanded.starts_with ("~/"))
        {
          expanded = home_str + expanded.substr (1U);
        }

      if (expanded.empty () || expanded.front () != '/')
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "Cannot read " + std::string (path)
              + ": not in system-readable paths. Allowed prefixes: /etc, "
                "/var/log, /proc, /sys, ~/.config, and specific dotfiles."));
        }

      std::error_code ec;
      const std::filesystem::path norm_path
          = std::filesystem::path (expanded).lexically_normal ();
      const std::string norm_str = norm_path.string ();
      const std::filesystem::path canon_path
          = std::filesystem::weakly_canonical (expanded, ec);
      if (ec || canon_path.empty ())
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "Cannot read " + std::string (path)
              + ": invalid or unresolvable path."));
        }
      const std::string canon_str = canon_path.string ();

      // Check prefix allowlist from O5.
      const std::string cfg_prefix = home_str + "/.config";
      const std::array<std::string_view, 5U> dir_prefixes = { {
          "/etc",
          "/var/log",
          "/proc",
          "/sys",
          cfg_prefix,
      } };
      auto matches_o5_dir = [&] (std::string_view s) noexcept -> bool {
        for (const std::string_view pfx : dir_prefixes)
          {
            if (s == pfx
                || (s.size () > pfx.size () && s.starts_with (pfx)
                    && s[pfx.size ()] == '/'))
              {
                return true;
              }
          }
        return false;
      };
      bool allowed_prefix = matches_o5_dir (canon_str);
      if (!allowed_prefix && matches_o5_dir (norm_str)
          && (canon_str.starts_with ("/usr/lib/")
              || canon_str.starts_with ("/usr/share/")
              || canon_str.starts_with ("/run/systemd/")))
        {
          allowed_prefix = true;
        }

      if (!allowed_prefix)
        {
          const std::array<std::string, 4U> exact_dotfiles = { {
              home_str + "/.bashrc",
              home_str + "/.zshrc",
              home_str + "/.profile",
              home_str + "/.ssh/config",
          } };
          for (const std::string &df : exact_dotfiles)
            {
              if (canon_str == df || norm_str == df)
                {
                  allowed_prefix = true;
                  break;
                }
            }
        }

      if (!allowed_prefix)
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "Cannot read " + std::string (path)
              + ": not in system-readable paths. Allowed prefixes: /etc, "
                "/var/log, /proc, /sys, ~/.config, and specific dotfiles."));
        }

      // Check secret denylist from O6 on both expanded and canonical paths.
      auto is_secret_path = [] (std::string_view sv) noexcept -> bool {
        if (sv == "/etc/shadow" || sv == "/etc/gshadow"
            || sv == "/etc/shadow-" || sv == "/etc/gshadow-")
          {
            return true;
          }
        if (sv == "/etc/ssl/private"
            || sv.starts_with ("/etc/ssl/private/"))
          {
            return true;
          }
        if (sv.starts_with ("/etc/ssh/ssh_host_")
            && sv.find ("_key") != std::string_view::npos)
          {
            return true;
          }
        constexpr std::array<std::string_view, 6U> kSecretTokens = { {
            "id_rsa",
            "id_ed25519",
            "id_ecdsa",
            "id_dsa",
            ".gnupg",
            ".pki",
        } };
        for (const std::string_view tok : kSecretTokens)
          {
            if (sv.find (tok) != std::string_view::npos)
              {
                return true;
              }
          }
        return false;
      };

      if (is_secret_path (expanded) || is_secret_path (norm_str)
          || is_secret_path (canon_str))
        {
          return std::unexpected (sutra::AgentError::ToolNotAllowed (
              "Cannot read " + std::string (path)
              + ": secret material. Use read_file inside the sandbox or the "
                "run_readonly tool for summary information."));
        }

      struct stat st{};
      if (::stat (canon_str.c_str (), &st) != 0)
        {
          if (errno == EACCES || errno == EPERM)
            {
              return std::unexpected (sutra::AgentError::ToolNotAllowed (
                  "Cannot read " + std::string (path)
                  + ": permission denied."));
            }
          return std::unexpected (sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "read_file_system",
              "Cannot read " + std::string (path) + ": file does not exist."));
        }

      if (S_ISDIR (st.st_mode))
        {
          return std::unexpected (sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "read_file_system",
              "Cannot read " + std::string (path)
              + ": path is a directory."));
        }

      const int fd = ::open (canon_str.c_str (), O_RDONLY | O_CLOEXEC);
      if (fd < 0)
        {
          if (errno == EACCES || errno == EPERM)
            {
              return std::unexpected (sutra::AgentError::ToolNotAllowed (
                  "Cannot read " + std::string (path)
                  + ": permission denied."));
            }
          return std::unexpected (sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "read_file_system",
              "Cannot read " + std::string (path)
              + ": failed to open file."));
        }

      std::string content;
      std::array<char, 8192U> buf{};
      bool exceeded_cap = false;
      while (true)
        {
          const ssize_t n = ::read (fd, buf.data (), buf.size ());
          if (n > 0)
            {
              const auto u_n = static_cast<std::size_t> (n);
              if (content.size () + u_n > kMaxToolOutputBytes)
                {
                  const std::size_t keep
                      = kMaxToolOutputBytes - content.size ();
                  content.append (buf.data (), keep);
                  exceeded_cap = true;
                  break;
                }
              content.append (buf.data (), u_n);
            }
          else if (n == 0)
            {
              break;
            }
          else
            {
              if (errno == EINTR)
                {
                  continue;
                }
              break;
            }
        }
      ::close (fd);

      // Binary detection on first 4 KB (§4.2).
      const std::size_t sample_len
          = std::min<std::size_t> (4096U, content.size ());
      if (sample_len > 0U)
        {
          const bool known_text_sig
              = (sample_len >= 3U
                 && static_cast<unsigned char> (content[0]) == 0xEFU
                 && static_cast<unsigned char> (content[1]) == 0xBBU
                 && static_cast<unsigned char> (content[2]) == 0xBFU)
                || (sample_len >= 4U
                    && std::string_view (content.data (), 4U) == "<?xm")
                || (sample_len >= 2U
                    && std::string_view (content.data (), 2U) == "#!");
          std::size_t non_printable = 0U;
          bool has_nul = false;
          for (std::size_t i = 0U; i < sample_len; ++i)
            {
              const auto ch = static_cast<unsigned char> (content[i]);
              if (ch == 0U)
                {
                  has_nul = true;
                  ++non_printable;
                }
              else if (ch < 32U && ch != '\t' && ch != '\n' && ch != '\r')
                {
                  ++non_printable;
                }
            }
          if (has_nul
              || (!known_text_sig && (non_printable * 20U > sample_len)))
            {
              return std::unexpected (sutra::AgentError::tool (
                  sutra::ToolErrorCode::ExecutionFailure, "read_file_system",
                  "Cannot read " + std::string (path) + ": binary file."));
            }
        }

      if (exceeded_cap
          || (st.st_size > static_cast<off_t> (kMaxToolOutputBytes)))
        {
          const auto total_bytes
              = (st.st_size > static_cast<off_t> (kMaxToolOutputBytes))
                    ? static_cast<unsigned long long> (st.st_size)
                    : static_cast<unsigned long long> (kMaxToolOutputBytes
                                                       + 1U);
          content.append ("\n...[truncated, file is "
                          + std::to_string (total_bytes) + " bytes]...");
        }

      return content;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "read_file_system",
          "Exception reading system file"));
    }
}

auto
tool_run_readonly (const sutra::ToolContext & /*ctx*/, std::string command,
                   std::string args) noexcept -> sutra::Result<std::string>
{
  const auto t0 = std::chrono::steady_clock::now ();
  const std::string input_repr = command + " " + args;

  std::vector<std::string> parsed_tokens;
  const std::string_view trimmed_args = trim_ws (args);
  if (trimmed_args.starts_with ('[') && trimmed_args.ends_with (']'))
    {
      // Parse JSON string array e.g. ["is-active", "sshd"]
      std::size_t idx = 1U;
      const std::size_t end = trimmed_args.size () - 1U;
      while (idx < end)
        {
          while (idx < end
                 && (std::isspace (
                         static_cast<unsigned char> (trimmed_args[idx]))
                         != 0
                     || trimmed_args[idx] == ','))
            {
              ++idx;
            }
          if (idx >= end)
            {
              break;
            }
          if (trimmed_args[idx] == '"')
            {
              ++idx;
              std::string item;
              while (idx < end && trimmed_args[idx] != '"')
                {
                  if (trimmed_args[idx] == '\\' && idx + 1U < end)
                    {
                      ++idx;
                    }
                  item.push_back (trimmed_args[idx]);
                  ++idx;
                }
              if (idx < end && trimmed_args[idx] == '"')
                {
                  ++idx;
                }
              parsed_tokens.push_back (std::move (item));
            }
          else
            {
              const std::size_t tok_start = idx;
              while (idx < end && trimmed_args[idx] != ','
                     && std::isspace (
                            static_cast<unsigned char> (trimmed_args[idx]))
                            == 0)
                {
                  ++idx;
                }
              parsed_tokens.emplace_back (
                  trimmed_args.substr (tok_start, idx - tok_start));
            }
        }
    }
  else if (!trimmed_args.empty ())
    {
      parsed_tokens = split_command_args (trimmed_args);
    }

  std::vector<std::string_view> arg_views;
  arg_views.reserve (parsed_tokens.size ());
  for (const std::string &s : parsed_tokens)
    {
      arg_views.emplace_back (s);
    }

  auto res = run_readonly (command, arg_views);
  const double ms
      = std::chrono::duration<double, std::milli> (
            std::chrono::steady_clock::now () - t0)
            .count ();
  if (!res)
    {
      record_tool_telemetry ("run_readonly", input_repr,
                             res.error ().message (), ms, true);
      return std::unexpected (res.error ());
    }

  std::string out = res->stdout_str;
  if (!res->stderr_str.empty ())
    {
      if (!out.empty () && out.back () != '\n')
        {
          out.push_back ('\n');
        }
      out.append (res->stderr_str);
    }
  if (out.empty ())
    {
      out = "(exit_code=" + std::to_string (res->exit_code) + ")";
    }
  record_tool_telemetry ("run_readonly", input_repr, out, ms, false);
  return out;
}

auto
tool_read_file_system (const sutra::ToolContext & /*ctx*/,
                       std::string path) noexcept -> sutra::Result<std::string>
{
  const auto t0 = std::chrono::steady_clock::now ();
  auto res = read_file_system (path);
  const double ms
      = std::chrono::duration<double, std::milli> (
            std::chrono::steady_clock::now () - t0)
            .count ();
  if (!res)
    {
      record_tool_telemetry ("read_file_system", path,
                             res.error ().message (), ms, true);
      return std::unexpected (res.error ());
    }
  record_tool_telemetry ("read_file_system", path, *res, ms, false);
  return *res;
}

auto
tool_remove_file (const sutra::ToolContext & /*ctx*/,
                  std::string path) noexcept -> sutra::Result<std::string>
{
  const auto t0 = std::chrono::steady_clock::now ();
  const std::string_view trimmed = trim_ws (path);
  if (trimmed.empty ())
    {
      const auto err = sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "remove_file",
          "Path must not be empty");
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      record_tool_telemetry ("remove_file", path, err.message (), ms, true);
      return std::unexpected (err);
    }

  try
    {
      const std::filesystem::path sb = ToolEnvironment::sandbox_dir ();
      if (trimmed == "*" || trimmed == "all" || trimmed == "workspace")
        {
          std::error_code ec;
          std::vector<std::string> removed_names;
          for (const auto &entry :
               std::filesystem::directory_iterator (sb, ec))
            {
              const std::string fname
                  = entry.path ().filename ().string ();
              if (fname == ".memory" || fname.starts_with (".memory"))
                {
                  continue;
                }
              std::error_code rem_ec;
              (void)std::filesystem::remove_all (entry.path (), rem_ec);
              if (!rem_ec)
                {
                  removed_names.push_back (fname);
                }
            }
          std::ranges::sort (removed_names);
          std::string msg;
          if (removed_names.empty ())
            {
              msg = "workspace is already clean (0 files removed)";
            }
          else
            {
              msg = "cleared workspace ("
                    + std::to_string (removed_names.size ())
                    + " item(s) removed: ";
              for (std::size_t i = 0U; i < removed_names.size (); ++i)
                {
                  if (i > 0U)
                    {
                      msg.append (", ");
                    }
                  msg.append (removed_names[i]);
                }
              msg.push_back (')');
            }
          const double ms
              = std::chrono::duration<double, std::milli> (
                    std::chrono::steady_clock::now () - t0)
                    .count ();
          record_tool_telemetry ("remove_file", path, msg, ms, false);
          return msg;
        }

      if (trimmed == ".memory" || trimmed.starts_with (".memory/")
          || contains_icase (trimmed, "/.memory"))
        {
          const auto err = sutra::AgentError::ToolSandboxViolation (
              "Cannot remove .memory directory via remove_file; use 'clear "
              "memory' instead.");
          const double ms
              = std::chrono::duration<double, std::milli> (
                    std::chrono::steady_clock::now () - t0)
                    .count ();
          record_tool_telemetry ("remove_file", path, err.message (), ms,
                                 true);
          return std::unexpected (err);
        }

      auto path_res = resolve_sandboxed_path (trimmed);
      if (!path_res || *path_res == sb)
        {
          const std::string base_str = sb.string ();
          const auto err = sutra::AgentError::ToolSandboxViolation (
              "Cannot remove " + path
              + ": path is outside the sandbox. The allowed root is "
              + base_str + ".");
          const double ms
              = std::chrono::duration<double, std::milli> (
                    std::chrono::steady_clock::now () - t0)
                    .count ();
          record_tool_telemetry ("remove_file", path, err.message (), ms,
                                 true);
          return std::unexpected (err);
        }

      std::error_code ec;
      if (!std::filesystem::exists (*path_res, ec))
        {
          const auto err = sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "remove_file",
              "File or directory does not exist: " + std::string (trimmed));
          const double ms
              = std::chrono::duration<double, std::milli> (
                    std::chrono::steady_clock::now () - t0)
                    .count ();
          record_tool_telemetry ("remove_file", path, err.message (), ms,
                                 true);
          return std::unexpected (err);
        }

      (void)std::filesystem::remove_all (*path_res, ec);
      if (ec)
        {
          const auto err = sutra::AgentError::tool (
              sutra::ToolErrorCode::ExecutionFailure, "remove_file",
              "Failed to remove " + std::string (trimmed));
          const double ms
              = std::chrono::duration<double, std::milli> (
                    std::chrono::steady_clock::now () - t0)
                    .count ();
          record_tool_telemetry ("remove_file", path, err.message (), ms,
                                 true);
          return std::unexpected (err);
        }

      const std::string msg = "removed " + std::string (trimmed);
      const double ms
          = std::chrono::duration<double, std::milli> (
                std::chrono::steady_clock::now () - t0)
                .count ();
      record_tool_telemetry ("remove_file", path, msg, ms, false);
      return msg;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::tool (
          sutra::ToolErrorCode::ExecutionFailure, "remove_file",
          "Exception removing file"));
    }
}

auto
register_all_sphinx_tools (sutra::ToolRegistry &registry) noexcept
    -> sutra::Result<void>
{
  {
    std::vector<sutra::ToolParamSchema> params = {
      { "path", sutra::JsonType::String, true },
      { "start_line", sutra::JsonType::Integer, false },
      { "max_lines", sutra::JsonType::Integer, false },
    };
    auto schema_res = sutra::build_json_schema (params);
    if (!schema_res)
      {
        return std::unexpected (schema_res.error ());
      }
    sutra::ToolDescriptor desc;
    desc.name = "read_file";
    desc.description = "Reads up to max_lines from a file in SPHINX_SANDBOX";
    desc.schema_json = std::move (*schema_res);
    desc.parameters = std::move (params);
    desc.dispatch_fn = +[] (const sutra::ToolContext &ctx,
                            std::string_view args_json) noexcept
        -> sutra::Result<std::string> {
      auto path_tok = sutra::extract_json_field_token (
          args_json, "path", sutra::JsonType::String);
      if (!path_tok)
        {
          return std::unexpected (path_tok.error ());
        }
      auto path_val = sutra::parse_json_token_as<std::string> (*path_tok);
      if (!path_val)
        {
          return std::unexpected (path_val.error ());
        }
      int64_t start_line = 1;
      if (auto sl_tok = sutra::extract_json_field_token (
              args_json, "start_line", sutra::JsonType::Integer);
          sl_tok)
        {
          if (auto sl_val = sutra::parse_json_token_as<int64_t> (*sl_tok);
              sl_val)
            {
              start_line = std::max<int64_t> (1, *sl_val);
            }
        }
      int64_t max_lines = 200;
      if (auto ml_tok = sutra::extract_json_field_token (
              args_json, "max_lines", sutra::JsonType::Integer);
          ml_tok)
        {
          if (auto ml_val = sutra::parse_json_token_as<int64_t> (*ml_tok);
              ml_val)
            {
              max_lines = std::clamp<int64_t> (*ml_val, 1, 500);
            }
        }
      return tool_read_file (ctx, std::move (*path_val), max_lines,
                             start_line);
    };
    if (auto r = registry.register_tool (std::move (desc)); !r)
      {
        return r;
      }
  }
  if (auto r = SUTRA_REGISTER_TOOL (
          registry, &tool_write_file, "write_file",
          "Writes content to a file inside SPHINX_SANDBOX", "path", "content");
      !r)
    {
      return r;
    }
  if (auto r = SUTRA_REGISTER_TOOL (
          registry, &tool_edit_file, "edit_file",
          "Replaces an exact target string with replacement in a file inside "
          "SPHINX_SANDBOX",
          "path", "target", "replacement");
      !r)
    {
      return r;
    }
  if (auto r = SUTRA_REGISTER_TOOL (
          registry, &tool_list_dir, "list_dir",
          "Lists entries in a directory inside SPHINX_SANDBOX", "path");
      !r)
    {
      return r;
    }
  {
    std::vector<sutra::ToolParamSchema> params = {
      { "cmd", sutra::JsonType::String, true },
      { "cwd", sutra::JsonType::String, false },
    };
    auto schema_res = sutra::build_json_schema (params);
    if (!schema_res)
      {
        return std::unexpected (schema_res.error ());
      }
    sutra::ToolDescriptor desc;
    desc.name = "run_command";
    desc.description = "Executes any non-root user command, build tool, file "
                       "operation, or compiled binary confined within "
                       "SPHINX_SANDBOX";
    desc.schema_json = std::move (*schema_res);
    desc.parameters = std::move (params);
    desc.dispatch_fn = +[] (const sutra::ToolContext &ctx,
                            std::string_view args_json) noexcept
        -> sutra::Result<std::string> {
      auto cmd_tok = sutra::extract_json_field_token (
          args_json, "cmd", sutra::JsonType::String);
      if (!cmd_tok)
        {
          return std::unexpected (cmd_tok.error ());
        }
      auto cmd_val = sutra::parse_json_token_as<std::string> (*cmd_tok);
      if (!cmd_val)
        {
          return std::unexpected (cmd_val.error ());
        }
      std::string cwd_val = ".";
      if (auto cwd_tok = sutra::extract_json_field_token (
              args_json, "cwd", sutra::JsonType::String);
          cwd_tok)
        {
          if (auto parsed_cwd = sutra::parse_json_token_as<std::string> (
                  *cwd_tok);
              parsed_cwd && !parsed_cwd->empty ())
            {
              cwd_val = std::move (*parsed_cwd);
            }
        }
      return tool_run_command (ctx, std::move (*cmd_val), std::move (cwd_val));
    };
    if (auto r = registry.register_tool (std::move (desc)); !r)
      {
        return r;
      }
  }
  {
    std::vector<sutra::ToolParamSchema> params = {
      { "file", sutra::JsonType::String, true },
      { "flags", sutra::JsonType::String, false },
    };
    auto schema_res = sutra::build_json_schema (params);
    if (!schema_res)
      {
        return std::unexpected (schema_res.error ());
      }
    sutra::ToolDescriptor desc;
    desc.name = "compile_check";
    desc.description
        = "Checks C++23 syntax of a file in SPHINX_SANDBOX with g++";
    desc.schema_json = std::move (*schema_res);
    desc.parameters = std::move (params);
    desc.dispatch_fn = +[] (const sutra::ToolContext &ctx,
                            std::string_view args_json) noexcept
        -> sutra::Result<std::string> {
      auto file_tok = sutra::extract_json_field_token (
          args_json, "file", sutra::JsonType::String);
      if (!file_tok)
        {
          return std::unexpected (file_tok.error ());
        }
      auto file_val = sutra::parse_json_token_as<std::string> (*file_tok);
      if (!file_val)
        {
          return std::unexpected (file_val.error ());
        }
      std::string flags_val = "-O2";
      if (auto fl_tok = sutra::extract_json_field_token (
              args_json, "flags", sutra::JsonType::String);
          fl_tok)
        {
          if (auto parsed_fl = sutra::parse_json_token_as<std::string> (
                  *fl_tok);
              parsed_fl && !parsed_fl->empty ())
            {
              flags_val = std::move (*parsed_fl);
            }
        }
      return tool_compile_check (ctx, std::move (*file_val),
                                 std::move (flags_val));
    };
    if (auto r = registry.register_tool (std::move (desc)); !r)
      {
        return r;
      }
  }
  if (auto r = SUTRA_REGISTER_TOOL (
          registry, &tool_ssh_exec, "ssh_exec",
          "Executes a safe command on an allowlisted SSH host via libssh2",
          "host", "cmd");
      !r)
    {
      return r;
    }
  if (auto r = SUTRA_REGISTER_TOOL (
          registry, &tool_ssh_test, "ssh_test",
          "Tests non-interactive SSH connectivity to an allowlisted host",
          "host");
      !r)
    {
      return r;
    }
  if (auto r = SUTRA_REGISTER_TOOL (
          registry, &tool_echo, "echo",
          "Returns the input string unchanged for smoke testing", "text");
      !r)
    {
      return r;
    }
  if (auto r = SUTRA_REGISTER_TOOL (
          registry, &tool_make_directory, "make_directory",
          "Create a directory (recursively). Argument: path (relative to "
          "the sandbox). Returns the absolute path.",
          "path");
      !r)
    {
      return r;
    }
  if (auto r = SUTRA_REGISTER_TOOL (
          registry, &tool_run_readonly, "run_readonly",
          "Execute a read-only diagnostic command. Allowed commands: ls, "
          "ps, df, free, uptime, uname, lscpu, nvidia-smi, ip, ss, systemctl "
          "(status/list/is-active subcommands only), journalctl (no "
          "--vacuum), dpkg-query (-l/-L/-W only), apt (list/search/show/"
          "policy), whoami, id, groups, hostname, hostnamectl, lsblk, "
          "blkid, findmnt, stat, file, wc, du, md5sum/sha256sum/sha1sum, "
          "iwconfig, nmcli, dmesg (no clear/delete). For understanding the "
          "system; not for building code (use run_command) or filesystem "
          "operations (use the file tools).",
          "command", "args");
      !r)
    {
      return r;
    }
  if (auto r = SUTRA_REGISTER_TOOL (
          registry, &tool_read_file_system, "read_file_system",
          "Read a system configuration, log, or proc/sys file. Allowed "
          "prefixes: /etc, /var/log, /proc, /sys, ~/.config, plus ~/.bashrc, "
          "~/.zshrc, ~/.profile, ~/.ssh/config. Secret files (shadow, "
          "private keys) are refused. Read-only, no write capability.",
          "path");
      !r)
    {
      return r;
    }
  if (auto r = SUTRA_REGISTER_TOOL (
          registry, &tool_remove_file, "remove_file",
          "Removes a file or directory inside SPHINX_SANDBOX, or clears all "
          "workspace files (preserving .memory) when path is '*'.",
          "path");
      !r)
    {
      return r;
    }
  return {};
}

} /* namespace sphinx::inline v1 */
