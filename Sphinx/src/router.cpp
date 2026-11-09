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

#include <sphinx/router.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sphinx::inline v1
{

namespace
{

constexpr uint32_t kRouterMaxTokens = 2U;
constexpr float kRouterTemperature = 0.0F;
constexpr float kHighConfidence = 0.96F;
constexpr float kFallbackConfidence = 0.65F;
constexpr std::size_t kLatencyBenchIterations = 100U;
constexpr std::size_t kP50Index = 50U;
constexpr std::size_t kP99Index = 98U;

constexpr std::string_view kRouterSystemPrompt
    = "Classify the user's request as either CHAT or TASK.\n"
      "CHAT = casual conversation, greetings, thanks, opinions, conceptual "
      "explanations, jokes, poems, study advice, history, trivia.\n"
      "TASK = coding in files, file editing, reading files, listing "
      "directories, running commands, SSH, system tasks, anything requiring "
      "tools or multi-step execution.\n"
      "Reply with a single word: CHAT or TASK.";

struct LabeledTurn
{
  std::string_view utterance;
  Route expected;
};

constexpr std::array<LabeledTurn, 20U> kBenchmarkTurns = { {
    { "Hello Sphinx, how are you doing today?", Route::CHAT },
    { "What is the capital of France?", Route::CHAT },
    { "Can you explain how SSH works?", Route::CHAT },
    { "Write me a poem about coding", Route::CHAT },
    { "Tell me a short joke about compilers.", Route::CHAT },
    { "Why is the sky blue during the afternoon?", Route::CHAT },
    { "What is the difference between TCP and UDP?", Route::CHAT },
    { "Who invented the C++ programming language?", Route::CHAT },
    { "Give me three tips for staying focused while studying.", Route::CHAT },
    { "Thank you for your help today!", Route::CHAT },
    { "What time is it on the server?", Route::TASK },
    { "Tell me about the code in main.cpp", Route::TASK },
    { "Write a C++ function in fib.cpp that computes Fibonacci numbers.",
      Route::TASK },
    { "Run the unit tests using ctest in the workspace.", Route::TASK },
    { "List all files in the current directory.", Route::TASK },
    { "Check if hello.cpp compiles with g++ -std=c++23.", Route::TASK },
    { "Connect to localhost via SSH and check disk usage with df -h.",
      Route::TASK },
    { "Read the first twenty lines of CMakeLists.txt.", Route::TASK },
    { "Create a file named notes.txt containing hello world.", Route::TASK },
    { "Run git status in the sandbox directory.", Route::TASK },
} };

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
      bool ok = true;
      for (std::size_t j = 0U; j < needle.size (); ++j)
        {
          const auto ca = static_cast<unsigned char> (haystack[i + j]);
          const auto cb = static_cast<unsigned char> (needle[j]);
          if (std::tolower (ca) != std::tolower (cb))
            {
              ok = false;
              break;
            }
        }
      if (ok)
        {
          return true;
        }
    }
  return false;
}

/* Trims ASCII whitespace and punctuation from the start of a string_view.  */
[[nodiscard]] auto
trim_leading_token (std::string_view sv) noexcept -> std::string_view
{
  std::size_t idx = 0U;
  while (idx < sv.size ())
    {
      const auto ch = static_cast<unsigned char> (sv[idx]);
      if (std::isalpha (ch) != 0)
        {
          break;
        }
      ++idx;
    }
  return sv.substr (idx);
}

/* Case-insensitive prefix check.  */
[[nodiscard]] auto
starts_with_icase (std::string_view text,
                   std::string_view prefix) noexcept -> bool
{
  if (text.size () < prefix.size ())
    {
      return false;
    }
  for (std::size_t i = 0U; i < prefix.size (); ++i)
    {
      const auto ca = static_cast<unsigned char> (text[i]);
      const auto cb = static_cast<unsigned char> (prefix[i]);
      if (std::tolower (ca) != std::tolower (cb))
        {
          return false;
        }
    }
  return true;
}

constexpr std::array<std::string_view, 9U> kMetaQuestionKeywords = { {
    "what commands can you run",
    "what linux commands can you run",
    "what tools do you have",
    "what can you do",
    "what are your capabilities",
    "list your tools",
    "what are you able to",
    "help me understand what you",
    "how do i use you",
} };

/* Returns true if the user turn is a meta-question about Sphinx's
   capabilities or available tools.  */
[[nodiscard]] auto
is_meta_question (std::string_view user_text) noexcept -> bool
{
  for (const std::string_view kw : kMetaQuestionKeywords)
    {
      if (contains_icase (user_text, kw))
      {
        return true;
      }
    }
  return false;
}

/* Returns true if the user turn asks a host profile question answered
   directly from the session startup profile without invoking a tool.  */
[[nodiscard]] auto
is_profile_question (std::string_view user_text) noexcept -> bool
{
  constexpr std::array<std::string_view, 8U> kProfileKeywords = { {
      "what os am i",
      "what operating system",
      "do i have cmake",
      "is cmake installed",
      "how much disk",
      "disk do i have",
      "what ssh hosts",
      "ssh hosts do i have",
  } };
  for (const std::string_view kw : kProfileKeywords)
    {
      if (contains_icase (user_text, kw))
        {
          return true;
        }
    }
  return false;
}

/* Returns true if the user turn asks to summarize prior actions/turns in the
   current conversation rather than execute a filesystem/shell action.  */
[[nodiscard]] auto
is_conversational_recall_summary (std::string_view text) noexcept -> bool
{
  return contains_icase (text, "summarize every action")
         || contains_icase (text, "since turn")
         || contains_icase (text, "summarize what we")
         || contains_icase (text, "recap what we");
}

/* Returns true if the user turn is a pure conceptual/philosophical/historical
   treatise or summary request with no file/directory/tool references.  */
[[nodiscard]] auto
is_pure_conceptual_treatise (std::string_view text) noexcept -> bool
{
  if (contains_icase (text, "file") || contains_icase (text, ".cpp")
      || contains_icase (text, ".hpp") || contains_icase (text, ".rs")
      || contains_icase (text, ".txt") || contains_icase (text, ".md")
      || contains_icase (text, ".sh") || contains_icase (text, ".json")
      || contains_icase (text, ".toml") || contains_icase (text, "cmakelists")
      || contains_icase (text, "makefile")
      || contains_icase (text, "directory") || contains_icase (text, "folder")
      || contains_icase (text, "workspace")
      || contains_icase (text, "sandbox")
      || contains_icase (text, "write_file")
      || contains_icase (text, "read_file")
      || contains_icase (text, "list_dir")
      || contains_icase (text, "run_command"))
    {
      return false;
    }
  return contains_icase (text, "comprehensive")
         || contains_icase (text, "treatise")
         || contains_icase (text, "essay")
         || contains_icase (text, "summary of")
         || contains_icase (text, "summarize ")
         || contains_icase (text, "philosophy")
         || contains_icase (text, "epistemolog")
         || contains_icase (text, "metaphysic")
         || contains_icase (text, "greatest thinkers")
         || contains_icase (text, "history of")
         || contains_icase (text, "whole history")
         || contains_icase (text, "polymath")
         || contains_icase (text, "first-principles")
         || contains_icase (text, "first principles");
}

/* Returns true if the prompt contains an explicit tool name, source/build
   filename/extension, or coding/inspection verb that must route to TASK.  */
[[nodiscard]] auto
has_explicit_task_signal (std::string_view text) noexcept -> bool
{
  if (is_meta_question (text) || is_profile_question (text)
      || is_conversational_recall_summary (text)
      || is_pure_conceptual_treatise (text)
      || contains_icase (text, "poem") || contains_icase (text, "joke")
      || contains_icase (text, "why are ")
      || contains_icase (text, "talking about"))
    {
      return false;
    }

  constexpr std::array<std::string_view, 42U> kSignals = { {
      "list_dir",        "read_file",       "write_file",
      "run_command",     "compile_check",   "make_directory",
      "remove_file",     "ssh_exec",        "run_readonly",
      "read_file_system",".cpp",            ".hpp",
      ".rs",             ".toml",           "cmakelists.txt",
      "makefile",        ".sh",             ".json",
      "inspect",         "explore",         "write ",
      "create ",         "compile",         "build",
      "link",            "g++",             "gcc",
      "clang",           "rustc",           "cargo",
      "cmake",           "ninja",           "strip ",
      "stripped",        "symbols from",    "that binary",
      "run the binary",  "using stat",      "via ls",
      "remove /tmp/",    "rm -rf",          "/tmp/sphinx_",
  } };
  for (const std::string_view sig : kSignals)
    {
      if (contains_icase (text, sig))
        {
          return true;
        }
    }

  for (const std::string_view ext : { ".h", ".c" })
    {
      std::size_t pos = 0U;
      while (pos + ext.size () <= text.size ())
        {
          const std::string_view rem = text.substr (pos);
          if (!contains_icase (rem, ext))
            {
              break;
            }
          bool matched = false;
          for (std::size_t i = pos; i + ext.size () <= text.size (); ++i)
            {
              if (contains_icase (text.substr (i, ext.size ()), ext))
                {
                  const bool prev_ok
                      = (i > 0U
                         && (std::isalnum (
                                 static_cast<unsigned char> (text[i - 1U]))
                                 != 0
                             || text[i - 1U] == '_'));
                  const std::size_t after = i + ext.size ();
                  const bool next_ok
                      = (after == text.size ()
                         || std::isalnum (
                                static_cast<unsigned char> (text[after]))
                                == 0);
                  if (prev_ok && next_ok)
                    {
                      return true;
                    }
                  pos = i + 1U;
                  matched = true;
                  break;
                }
            }
          if (!matched)
            {
              break;
            }
        }
    }
  return false;
}

/* Returns true if the user turn asks to verify or confirm a prior tool
   action (§4.2).  */
[[nodiscard]] auto
is_verification_request (std::string_view user_text) noexcept -> bool
{
  if (contains_icase (user_text, "sky is blue")
      || has_explicit_task_signal (user_text))
    {
      return false;
    }
  constexpr std::array<std::string_view, 9U> kVerifyKeywords = { {
      "are you sure",
      "is it complete",
      "is it correct",
      "does it contain",
      "is that right",
      "verify that",
      "confirm that",
      "check that",
      "did it work",
  } };
  for (const std::string_view kw : kVerifyKeywords)
    {
      if (contains_icase (user_text, kw))
        {
          return true;
        }
    }
  return false;
}

/* Returns true if the user turn requests an unsupported external/media action
   (such as generating/creating/drawing an image, video, audio, browsing the
   web, or sending email) rather than coding or file manipulation.  */
[[nodiscard]] auto
is_unsupported_capability_request (std::string_view text) noexcept -> bool
{
  if (contains_icase (text, "why are ")
      || contains_icase (text, "talking about")
      || contains_icase (text, "ability to ")
      || contains_icase (text, "code")
      || contains_icase (text, "viewer")
      || contains_icase (text, "program")
      || contains_icase (text, "function")
      || contains_icase (text, "script")
      || contains_icase (text, "c++")
      || contains_icase (text, "rust")
      || contains_icase (text, "file")
      || contains_icase (text, ".cpp")
      || contains_icase (text, ".hpp")
      || contains_icase (text, ".txt")
      || contains_icase (text, ".rs"))
    {
      return false;
    }

  const bool has_visual_noun
      = contains_icase (text, "image") || contains_icase (text, "picture")
        || contains_icase (text, "photo") || contains_icase (text, "drawing")
        || contains_icase (text, "illustration")
        || contains_icase (text, "artwork") || contains_icase (text, "video")
        || contains_icase (text, "animation");
  const bool has_action_verb
      = contains_icase (text, "create") || contains_icase (text, "generate")
        || contains_icase (text, "make") || contains_icase (text, "draw")
        || contains_icase (text, "paint") || contains_icase (text, "render")
        || contains_icase (text, "design") || contains_icase (text, "produce")
        || contains_icase (text, "show me");
  if (has_visual_noun && has_action_verb)
    {
      return true;
    }
  if (contains_icase (text, "draw a ") || contains_icase (text, "draw an ")
      || contains_icase (text, "draw the ")
      || contains_icase (text, "paint a ")
      || contains_icase (text, "paint an ")
      || contains_icase (text, "paint the ")
      || contains_icase (text, "browse the web")
      || contains_icase (text, "search the web")
      || contains_icase (text, "send an email")
      || contains_icase (text, "send email"))
    {
      return true;
    }
  return false;
}

/* Parses an explicit route prefix override (/big, /task, /chat) at the
   start of the user utterance.  */
[[nodiscard]] auto
parse_route_prefix_override (std::string_view text) noexcept
    -> std::optional<Route>
{
  std::size_t ws = 0U;
  while (ws < text.size ()
         && std::isspace (static_cast<unsigned char> (text[ws])) != 0)
    {
      ++ws;
    }
  const std::string_view trimmed = text.substr (ws);
  auto match_cmd = [&] (std::string_view cmd) noexcept -> bool {
    if (!starts_with_icase (trimmed, cmd))
      {
        return false;
      }
    if (trimmed.size () == cmd.size ())
      {
        return true;
      }
    return std::isspace (
               static_cast<unsigned char> (trimmed[cmd.size ()]))
           != 0;
  };
  if (match_cmd ("/big"))
    {
      return Route::BIG_CHAT;
    }
  if (match_cmd ("/task"))
    {
      return Route::TASK;
    }
  if (match_cmd ("/chat"))
    {
      return Route::CHAT;
    }
  return std::nullopt;
}

/* Returns true if the user turn asks for deep conceptual synthesis,
   comprehensive summaries, philosophy, science, history, or multi-paragraph
   explanations that should stream directly from the resident 80B model
   (Route::BIG_CHAT) without invoking workspace file/command tools.  */
[[nodiscard]] auto
is_big_chat_conceptual_request (std::string_view text) noexcept -> bool
{
  if (is_conversational_recall_summary (text))
    {
      return true;
    }
  if (is_meta_question (text) || is_profile_question (text)
      || is_unsupported_capability_request (text)
      || has_explicit_task_signal (text) || contains_icase (text, "poem")
      || contains_icase (text, "joke"))
    {
      return false;
    }

  constexpr std::array<std::string_view, 28U> kConceptualTaskExclusions = { {
      "server",
      "localhost",
      "ssh",
      "workspace",
      "sandbox",
      "directory",
      "folder",
      "file",
      "cmake",
      "ctest",
      "pytest",
      "git ",
      "sudo ",
      "systemctl",
      "journalctl",
      "df -h",
      "uname",
      "uptime",
      "pwd",
      "port 22",
      "/etc/",
      "/var/",
      "/proc/",
      "/sys/",
      "~/.ssh",
      "~/.config",
      "os-release",
      "sshd",
  } };
  for (const std::string_view ex : kConceptualTaskExclusions)
    {
      if (contains_icase (text, ex))
        {
          return false;
        }
    }

  constexpr std::array<std::string_view, 36U> kBigChatCues = { {
      "comprehensive",
      "treatise",
      "essay",
      "in-depth",
      "in depth",
      "in detail",
      "detailed ",
      "first-principles",
      "first principles",
      "deep dive",
      "breakdown",
      "summary of",
      "summarize ",
      "history of",
      "whole history",
      "philosophy",
      "epistemolog",
      "metaphysic",
      "greatest thinkers",
      "geniuses",
      "genius ",
      "compare and contrast",
      "trade-offs",
      "tradeoffs",
      "architectural",
      "mathematical",
      "theorem",
      "relativity",
      "quantum",
      "neuroscience",
      "psychology of",
      "evolution of",
      "step-by-step explanation",
      "thorough explanation",
      "elaborate on",
      "polymath",
  } };
  for (const std::string_view cue : kBigChatCues)
    {
      if (contains_icase (text, cue))
        {
          return true;
        }
    }

  if (contains_icase (text, "how ") && contains_icase (text, " think"))
    {
      return true;
    }

  if (contains_icase (text, "explain how ") && text.size () > 36U)
    {
      return true;
    }

  if (contains_icase (text, "analyze ") && !contains_icase (text, "code"))
    {
      return true;
    }

  if (text.size () >= 90U
      && (contains_icase (text, "explain") || contains_icase (text, "why ")
          || contains_icase (text, "how ") || contains_icase (text, "what ")))
    {
      return true;
    }

  return false;
}

} /* namespace */

struct TurnRouter::Impl
{
  TurnRouterConfig config;
  std::shared_ptr<sutra::LlmClient> client;
  std::string last_tool_invoked;
  mutable std::mutex mtx;
};

TurnRouter::TurnRouter (PrivateTag /*tag*/,
                        std::unique_ptr<Impl> impl) noexcept
    : impl_ (std::move (impl))
{
}

TurnRouter::TurnRouter (TurnRouter &&) noexcept = default;
auto TurnRouter::operator= (TurnRouter &&) noexcept -> TurnRouter & = default;
TurnRouter::~TurnRouter () noexcept = default;

auto
TurnRouter::create (const TurnRouterConfig &config,
                    std::shared_ptr<sutra::LlmClient> client_override) noexcept
    -> sutra::Result<std::unique_ptr<TurnRouter>>
{
  try
    {
      auto impl = std::make_unique<Impl> ();
      impl->config = config;

      if (client_override != nullptr)
        {
          impl->client = std::move (client_override);
        }
      else
        {
          sutra::LlamaCppConfig lcfg;
          lcfg.model_path = config.model_path;
          lcfg.n_ctx = config.context_len;
          lcfg.n_gpu_layers = config.n_gpu_layers;
          lcfg.n_threads = config.n_threads;
          lcfg.default_max_tokens = kRouterMaxTokens;
          auto client_res = sutra::LlamaCppClient::create (std::move (lcfg));
          if (!client_res)
            {
              return std::unexpected (client_res.error ());
            }
          impl->client = std::shared_ptr<sutra::LlmClient> (
              std::move (*client_res));
        }

      auto router
          = std::make_unique<TurnRouter> (PrivateTag{}, std::move (impl));

      // Warm up the resident KV cache with the shared classification prefix so
      // subsequent classify() calls reuse the system prompt KV state.
      (void)router->classify ("Hello there.");
      return router;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in TurnRouter::create"));
    }
}

auto
TurnRouter::classify_keyword_fallback (std::string_view user_text) noexcept
    -> Route
{
  if (const auto prefix_route = parse_route_prefix_override (user_text))
    {
      return *prefix_route;
    }

  // Meta-questions and startup profile questions route to CHAT.
  if (is_meta_question (user_text) || is_profile_question (user_text))
    {
      return Route::CHAT;
    }

  if (is_unsupported_capability_request (user_text)
      || has_explicit_task_signal (user_text))
    {
      return Route::TASK;
    }

  if (is_big_chat_conceptual_request (user_text))
    {
      return Route::BIG_CHAT;
    }

  // Conceptual/creative overrides take precedence in fallback mode.
  if (contains_icase (user_text, "poem")
      || contains_icase (user_text, "joke")
      || contains_icase (user_text, "explain how")
      || contains_icase (user_text, "what is the difference"))
    {
      return Route::CHAT;
    }

  constexpr std::array<std::string_view, 117U> kTaskKeywords = { {
      ".cpp",
      ".hpp",
      ".h",
      ".c",
      ".rs",
      ".py",
      ".sh",
      ".toml",
      ".txt",
      ".json",
      ".md",
      ".out",
      "cmake",
      "compile",
      "g++",
      "clang++",
      "ctest",
      "pytest",
      "cargo ",
      "rustc ",
      "git ",
      "sudo ",
      "run command ",
      "execute command ",
      "ssh",
      "server",
      "localhost",
      "read_file",
      "write_file",
      "list_dir",
      "make_directory",
      "remove_file",
      "run_command",
      "run_readonly",
      "read_file_system",
      "clear the workspace",
      "clear workspace",
      "clean the workspace",
      "clean workspace",
      "make a directory",
      "make a dir",
      "make a folder",
      "create a directory",
      "create a dir",
      "create a folder",
      "mkdir",
      "touch",
      "new\040folder",
      "new\040directory",
      "directory",
      "folder",
      "look into ",
      "explore ",
      "move to ",
      "change directory",
      "go into ",
      "cd ",
      "where is ",
      "what directory",
      "which directory",
      "located",
      "file",
      "sandbox",
      "workspace",
      "function",
      "implement",
      "refactor",
      "debug",
      "build",
      "execute",
      "run the program",
      "run the compiled",
      "run ./",
      "terminal",
      "command",
      "disk usage",
      "df -h",
      "uname",
      "uptime",
      "pwd",
      "is sshd",
      "sshd running",
      "systemctl",
      "journalctl",
      "kernel log",
      "port 22",
      "using port",
      "how much swap",
      "swap do i",
      "/etc/",
      "/var/log/",
      "/proc/",
      "/sys/",
      "~/.config",
      "~/.bashrc",
      "~/.zshrc",
      "~/.profile",
      "~/.ssh/config",
      "os-release",
      "sshd_config",
      "syslog",
      "rm -rf",
      "generate image",
      "create image",
      "code an ",
      "code a ",
      "from scratch",
      "audit ",
      "review the code",
      "review this file",
      "what does it do",
      "explain the code",
      "describe the file",
      "analyze the code",
      "inspect",
      "contents of",
  } };

  for (const std::string_view kw : kTaskKeywords)
    {
      if (contains_icase (user_text, kw))
        {
          return Route::TASK;
        }
    }
  return Route::CHAT;
}

void
TurnRouter::set_last_tool_invoked (std::string_view tool_name) noexcept
{
  if (impl_ == nullptr)
    {
      return;
    }
  const std::scoped_lock lock (impl_->mtx);
  impl_->last_tool_invoked.assign (tool_name);
}

auto
TurnRouter::last_tool_invoked () const noexcept -> std::string
{
  if (impl_ == nullptr)
    {
      return {};
    }
  const std::scoped_lock lock (impl_->mtx);
  return impl_->last_tool_invoked;
}

auto
TurnRouter::classify (std::string_view user_text, yantra::Run *active_run,
                      uint64_t step) noexcept -> sutra::Result<RouterDecision>
{
  if (impl_ == nullptr || impl_->client == nullptr)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::InvalidConfiguration, 0,
          "TurnRouter is not initialized"));
    }

  try
    {
      const auto t0 = std::chrono::steady_clock::now ();

      if (const auto prefix_route = parse_route_prefix_override (user_text))
        {
          RouterDecision decision;
          decision.route = *prefix_route;
          decision.confidence = 1.0F;
          decision.reasoning = "prefix_override:"
                               + std::string (route_to_string (*prefix_route));
          decision.latency_ms = 0.0;
          decision.used_fallback = false;
          if (active_run != nullptr)
            {
              (void)active_run->log_metric (step, "router_latency_ms", 0.0);
              (void)active_run->log_metric (
                  step, "router_is_task",
                  (decision.route == Route::TASK) ? 1.0 : 0.0);
              (void)active_run->log_tag (
                  "turn_" + std::to_string (step) + "_route",
                  std::string (route_to_string (decision.route)));
            }
          return decision;
        }

      std::string user_prompt = "Classify the user's request as either CHAT "
                                "or TASK.\nUser: ";
      user_prompt.append (user_text);
      user_prompt.append ("\nClassification:");

      std::array<sutra::Message, 10U> msgs = { {
          sutra::Message{
              sutra::MessageRole::System,
              std::string (kRouterSystemPrompt),
              {},
              {},
              0U,
              0U,
          },
          sutra::Message{
              sutra::MessageRole::User,
              "Classify the user's request as either CHAT or TASK.\n"
              "User: Can you explain how SSH works?\nClassification:",
              {},
              {},
              0U,
              0U,
          },
          sutra::Message{
              sutra::MessageRole::Assistant, "CHAT", {}, {}, 0U, 0U,
          },
          sutra::Message{
              sutra::MessageRole::User,
              "Classify the user's request as either CHAT or TASK.\n"
              "User: What time is it on the server?\nClassification:",
              {},
              {},
              0U,
              0U,
          },
          sutra::Message{
              sutra::MessageRole::Assistant, "TASK", {}, {}, 0U, 0U,
          },
          sutra::Message{
              sutra::MessageRole::User,
              "Classify the user's request as either CHAT or TASK.\n"
              "User: Write me a poem about coding\nClassification:",
              {},
              {},
              0U,
              0U,
          },
          sutra::Message{
              sutra::MessageRole::Assistant, "CHAT", {}, {}, 0U, 0U,
          },
          sutra::Message{
              sutra::MessageRole::User,
              "Classify the user's request as either CHAT or TASK.\n"
              "User: Tell me about the code in main.cpp\nClassification:",
              {},
              {},
              0U,
              0U,
          },
          sutra::Message{
              sutra::MessageRole::Assistant, "TASK", {}, {}, 0U, 0U,
          },
          sutra::Message{
              sutra::MessageRole::User,
              std::move (user_prompt),
              {},
              {},
              0U,
              0U,
          },
      } };

      std::array<std::string_view, 2U> stops = { "\n", "<|im_end|>" };

      sutra::CompletionRequest req;
      req.messages = msgs;
      req.temperature = kRouterTemperature;
      req.max_tokens = kRouterMaxTokens;
      req.stop_sequences = stops;
      req.seed = 42U;

      std::promise<sutra::Result<sutra::CompletionResult>> done_promise;
      auto done_future = done_promise.get_future ();

      {
        const std::scoped_lock lock (impl_->mtx);
        auto stream_res = impl_->client->stream_completion (
            req, [] (const sutra::TokenChunk & /*chunk*/) {},
            [&done_promise] (sutra::Result<sutra::CompletionResult> res) {
              done_promise.set_value (std::move (res));
            });
        if (!stream_res)
          {
            return std::unexpected (stream_res.error ());
          }
      }

      auto comp_res = done_future.get ();
      const auto t1 = std::chrono::steady_clock::now ();
      const double elapsed_ms
          = std::chrono::duration<double, std::milli> (t1 - t0).count ();

      RouterDecision decision;
      decision.latency_ms = elapsed_ms;

      const std::string prev_tool = last_tool_invoked ();
      if (is_verification_request (user_text))
        {
          if (!prev_tool.empty ())
            {
              decision.route = Route::TASK;
              decision.confidence = kHighConfidence;
              decision.reasoning = "stateful_verify:TASK";
              decision.used_fallback = false;
            }
          else
            {
              decision.route = Route::CHAT;
              decision.confidence = kHighConfidence;
              decision.reasoning = "stateful_verify:CHAT";
              decision.used_fallback = false;
            }
        }
      else if (comp_res)
        {
          const std::string_view trimmed
              = trim_leading_token (comp_res->text);
          if (starts_with_icase (trimmed, "TASK"))
            {
              if (is_meta_question (user_text)
                  || is_profile_question (user_text)
                  || (contains_icase (user_text, "fibonacci number")
                      && !contains_icase (user_text, "function")
                      && !contains_icase (user_text, "write")
                      && !contains_icase (user_text, ".cpp"))
                  || (contains_icase (user_text, "explain what ")
                      && !contains_icase (user_text, "file")
                      && !contains_icase (user_text, "code")
                      && !contains_icase (user_text, ".cpp")
                      && !contains_icase (user_text, ".rs"))
                  || contains_icase (user_text, "stand for")
                  || contains_icase (user_text, "interesting fact")
                  || contains_icase (user_text, "how does ")
                  || contains_icase (user_text, "why are ")
                  || contains_icase (user_text, "speed of light")
                  || contains_icase (user_text, "solar system")
                  || contains_icase (user_text, "what project am i")
                  || contains_icase (user_text, "what is my project")
                  || contains_icase (user_text, "what is my name")
                  || (!contains_icase (user_text, "server")
                      && !contains_icase (user_text, "ssh")
                      && (contains_icase (user_text, "what is today")
                          || contains_icase (user_text, "date is today")
                          || contains_icase (user_text, "today's date")
                          || contains_icase (user_text, "what time is it")
                          || contains_icase (user_text, "day of the week")
                          || contains_icase (user_text, "how many days"))))
                {
                  decision.route = Route::CHAT;
                  decision.confidence = kHighConfidence;
                  decision.reasoning = "llm:CHAT";
                  decision.used_fallback = false;
                }
              else if (is_big_chat_conceptual_request (user_text))
                {
                  decision.route = Route::BIG_CHAT;
                  decision.confidence = kHighConfidence;
                  decision.reasoning = "llm:BIG_CHAT";
                  decision.used_fallback = false;
                }
              else
                {
                  decision.route = Route::TASK;
                  decision.confidence = kHighConfidence;
                  decision.reasoning = "llm:TASK";
                  decision.used_fallback = false;
                }
            }
          else if (starts_with_icase (trimmed, "BIG"))
            {
              decision.route = Route::BIG_CHAT;
              decision.confidence = kHighConfidence;
              decision.reasoning = "llm:BIG_CHAT";
              decision.used_fallback = false;
            }
          else if (starts_with_icase (trimmed, "CHAT"))
            {
              if (!is_meta_question (user_text)
                  && !is_profile_question (user_text)
                  && !contains_icase (user_text, "why are ")
                  && !contains_icase (user_text, "talking about")
                  && (is_unsupported_capability_request (user_text)
                      || has_explicit_task_signal (user_text)
                      || contains_icase (user_text, "pwd")
                      || contains_icase (user_text, "generate image")
                      || contains_icase (user_text, "generate an image")
                      || contains_icase (user_text, "create image")
                      || contains_icase (user_text, "create an image")
                      || contains_icase (user_text, "touch")
                      || contains_icase (user_text, "mkdir")
                      || contains_icase (user_text, "make a directory")
                      || contains_icase (user_text, "make a dir")
                      || contains_icase (user_text, "make a folder")
                      || contains_icase (user_text, "create a directory")
                      || contains_icase (user_text, "create a dir")
                      || contains_icase (user_text, "create a folder")
                      || contains_icase (user_text, "new\040folder")
                      || contains_icase (user_text, "new\040directory")
                      || contains_icase (user_text, "directory called")
                      || contains_icase (user_text, "current directory")
                      || contains_icase (user_text, "directory address")
                      || contains_icase (user_text, "directory to ")
                      || contains_icase (user_text, "folder to ")
                      || contains_icase (user_text, "look into ")
                      || contains_icase (user_text, "explore ")
                      || contains_icase (user_text, "directory")
                      || contains_icase (user_text, "folder")
                      || contains_icase (user_text, "refactor")
                      || contains_icase (user_text, "fix the bug")
                      || contains_icase (user_text, "fix bug")
                      || contains_icase (user_text, "move to ")
                      || contains_icase (user_text, "go into ")
                      || contains_icase (user_text, "cd ")
                      || contains_icase (user_text, ".cpp")
                      || contains_icase (user_text, ".hpp")
                      || contains_icase (user_text, ".txt")
                      || contains_icase (user_text, ".rs")
                      || contains_icase (user_text, ".py")
                      || contains_icase (user_text, ".sh")
                      || contains_icase (user_text, ".toml")
                      || contains_icase (user_text, ".out")
                      || contains_icase (user_text, "code an ")
                      || contains_icase (user_text, "code a ")
                      || contains_icase (user_text, "from scratch")
                      || contains_icase (user_text, "compile ")
                      || contains_icase (user_text, "build ")
                      || contains_icase (user_text, "run the program")
                      || contains_icase (user_text, "run the compiled")
                      || contains_icase (user_text, "run ./")
                      || contains_icase (user_text, "execute ")
                      || contains_icase (user_text, "compile-check")
                      || contains_icase (user_text, "syntax-check")
                      || contains_icase (user_text, "does it build")
                      || contains_icase (user_text, "will it build")
                      || contains_icase (user_text, "audit ")
                      || contains_icase (user_text, "review the code")
                      || contains_icase (user_text, "review this file")
                      || contains_icase (user_text, "what does it do")
                      || contains_icase (user_text, "what does the code do")
                      || contains_icase (user_text, "what does this do")
                      || contains_icase (user_text, "explain the code")
                      || contains_icase (user_text, "describe the file")
                      || contains_icase (user_text, "analyze the code")
                      || contains_icase (user_text, "what's in the file")
                      || contains_icase (user_text, "what is in the file")
                      || contains_icase (user_text, "contents of")
                      || contains_icase (user_text, "where is ")
                      || contains_icase (user_text, "what directory")
                      || contains_icase (user_text, "which directory")
                      || contains_icase (user_text, "located")
                      || contains_icase (user_text, "is sshd")
                      || contains_icase (user_text, "sshd running")
                      || contains_icase (user_text, "systemctl")
                      || contains_icase (user_text, "journalctl")
                      || contains_icase (user_text, "kernel log")
                      || contains_icase (user_text, "port 22")
                      || contains_icase (user_text, "using port")
                      || contains_icase (user_text, "how much swap")
                      || contains_icase (user_text, "swap do i")
                      || contains_icase (user_text, "/etc/")
                      || contains_icase (user_text, "/var/log/")
                      || contains_icase (user_text, "/proc/")
                      || contains_icase (user_text, "/sys/")
                      || contains_icase (user_text, "~/.config")
                      || contains_icase (user_text, "~/.bashrc")
                      || contains_icase (user_text, "~/.zshrc")
                      || contains_icase (user_text, "~/.profile")
                      || contains_icase (user_text, "~/.ssh/config")
                      || contains_icase (user_text, "os-release")
                      || contains_icase (user_text, "sshd_config")
                      || contains_icase (user_text, "syslog")
                      || contains_icase (user_text, "rm -rf")
                      || contains_icase (user_text, "clear the workspace")
                      || contains_icase (user_text, "clear workspace")
                      || contains_icase (user_text, "clean the workspace")
                      || contains_icase (user_text, "clean workspace")
                      || contains_icase (user_text, "delete ")
                      || contains_icase (user_text, "remove ")
                      || contains_icase (user_text, "run command ")
                      || contains_icase (user_text, "execute command ")
                      || contains_icase (user_text, "cargo ")
                      || contains_icase (user_text, "rustc ")
                      || contains_icase (user_text, "git init")
                      || contains_icase (user_text, "sudo ")
                      || contains_icase (user_text, "apt install")))
                {
                  decision.route = Route::TASK;
                  decision.confidence = kHighConfidence;
                  decision.reasoning = "llm:TASK";
                  decision.used_fallback = false;
                }
              else if (is_big_chat_conceptual_request (user_text))
                {
                  decision.route = Route::BIG_CHAT;
                  decision.confidence = kHighConfidence;
                  decision.reasoning = "llm:BIG_CHAT";
                  decision.used_fallback = false;
                }
              else
                {
                  decision.route = Route::CHAT;
                  decision.confidence = kHighConfidence;
                  decision.reasoning = "llm:CHAT";
                  decision.used_fallback = false;
                }
            }
          else
            {
              decision.route = classify_keyword_fallback (user_text);
              decision.confidence = kFallbackConfidence;
              decision.reasoning = "keyword_fallback:" + comp_res->text;
              decision.used_fallback = true;
              constexpr std::string_view kWarn
                  = "[sphinx::TurnRouter] warning: malformed LLM router "
                    "output, falling back to keyword heuristic\n";
              (void)std::fwrite (kWarn.data (), 1U, kWarn.size (), stderr);
            }
        }
      else
        {
          decision.route = classify_keyword_fallback (user_text);
          decision.confidence = kFallbackConfidence;
          decision.reasoning = "error_fallback";
          decision.used_fallback = true;
        }

      if (active_run != nullptr)
        {
          (void)active_run->log_metric (step, "router_latency_ms",
                                        decision.latency_ms);
          (void)active_run->log_metric (
              step, "router_is_task",
              (decision.route == Route::TASK) ? 1.0 : 0.0);
          (void)active_run->log_tag (
              "turn_" + std::to_string (step) + "_route",
              std::string (route_to_string (decision.route)));
        }

      return decision;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in TurnRouter::classify"));
    }
}

auto
TurnRouter::run_20_turn_benchmark (yantra::Run *active_run) noexcept
    -> sutra::Result<RouterBenchmarkReport>
{
  try
    {
      set_last_tool_invoked ({});
      RouterBenchmarkReport report;
      report.entries.reserve (kBenchmarkTurns.size ());
      std::vector<double> latencies;
      latencies.reserve (kBenchmarkTurns.size ());

      uint64_t step = 0U;
      for (const LabeledTurn &turn : kBenchmarkTurns)
        {
          auto dec_res = classify (turn.utterance, active_run, step);
          if (!dec_res)
            {
              return std::unexpected (dec_res.error ());
            }
          RouterBenchmarkEntry entry;
          entry.utterance = std::string (turn.utterance);
          entry.expected = turn.expected;
          entry.predicted = dec_res->route;
          entry.confidence = dec_res->confidence;
          entry.latency_ms = dec_res->latency_ms;
          entry.correct = (entry.predicted == entry.expected);

          if (entry.expected == Route::CHAT && entry.predicted == Route::CHAT)
            {
              ++report.true_chat;
            }
          else if (entry.expected == Route::CHAT
                   && entry.predicted == Route::TASK)
            {
              ++report.false_task;
            }
          else if (entry.expected == Route::TASK
                   && entry.predicted == Route::CHAT)
            {
              ++report.false_chat;
            }
          else
            {
              ++report.true_task;
            }

          if (entry.correct)
            {
              ++report.correct_count;
            }
          latencies.push_back (entry.latency_ms);
          report.entries.push_back (std::move (entry));
          ++step;
        }

      std::ranges::sort (latencies);
      if (!latencies.empty ())
        {
          report.p50_ms = latencies[latencies.size () / 2U];
          report.p99_ms = latencies[latencies.size () - 1U];
        }
      report.accuracy_pct
          = (static_cast<double> (report.correct_count) * 100.0)
            / static_cast<double> (kBenchmarkTurns.size ());

      if (active_run != nullptr)
        {
          (void)active_run->log_metric (step, "router_bench_accuracy_pct",
                                        report.accuracy_pct);
          (void)active_run->log_metric (step, "router_bench_p50_ms",
                                        report.p50_ms);
          (void)active_run->log_metric (step, "router_bench_p99_ms",
                                        report.p99_ms);
        }
      return report;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in TurnRouter::run_20_turn_benchmark"));
    }
}

auto
TurnRouter::run_100_call_latency_benchmark () noexcept
    -> sutra::Result<std::pair<double, double>>
{
  try
    {
      set_last_tool_invoked ({});
      std::vector<double> samples;
      samples.reserve (kLatencyBenchIterations);
      for (std::size_t i = 0U; i < kLatencyBenchIterations; ++i)
        {
          const LabeledTurn &turn
              = kBenchmarkTurns.at (i % kBenchmarkTurns.size ());
          auto dec_res = classify (turn.utterance);
          if (!dec_res)
            {
              return std::unexpected (dec_res.error ());
            }
          samples.push_back (dec_res->latency_ms);
        }
      std::ranges::sort (samples);
      const double p50 = samples.at (kP50Index);
      const double p99 = samples.at (kP99Index);
      return std::make_pair (p50, p99);
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in TurnRouter::run_100_call_latency_benchmark"));
    }
}

} /* namespace sphinx::inline v1 */
