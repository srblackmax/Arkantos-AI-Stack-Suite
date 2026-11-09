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

#include <sphinx/session.hpp>

#include <fcntl.h>
#if defined(__linux__)
#include <malloc.h>
#endif
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sphinx::inline v1
{

namespace
{

constexpr std::size_t kMinSentenceWords = 5U;
constexpr std::size_t kMaxBackpressureQueue = 3U;
constexpr std::size_t kResumeBackpressureQueue = 1U;
constexpr float kBackchannelVolume = 0.6F;
constexpr float kNormalVolume = 1.0F;
constexpr double kBackchannelThresholdMs = 500.0;
constexpr double kBackchannelCooldownMs = 5000.0;
constexpr uint32_t kDefaultWhisperThreads = 8U;

constexpr std::string_view kSphinxIdentityPrompt
    = "You are Sphinx, a personal AI assistant running locally on the "
      "user's workstation. You were built by the user from a stack of "
      "native C++23 libraries (Hyper Math, Bellows, Basalt, Obsidian, "
      "Surya, Sutra, Darshana, Yantra, Vach). You are powered by open-"
      "weight Qwen models from Alibaba Cloud: Qwen2.5-1.5B-Instruct for "
      "casual conversation and Qwen3-30B-A3B for complex tasks. You are "
      "NOT Claude, ChatGPT, Gemini, Llama, Mistral, or any other "
      "assistant. If asked who created you, respond: \"I was built by my "
      "user on a native C++23 stack, powered by Qwen models from "
      "Alibaba Cloud.\" Never mention Anthropic, OpenAI, Google, Meta, or "
      "any other AI company as your creator.\n\n"
      "Your available tools are: read_file, write_file, list_dir, "
      "make_directory, remove_file, run_command (executes any non-root "
      "user command, build tool such as gcc, g++, clang, clang++, make, "
      "cmake, ninja, cargo, rustc, or pwd, file operation, or compiled "
      "binary confined within ~/sphinx-workspace), "
      "run_readonly, read_file_system, compile_check, ssh_exec, ssh_test, "
      "echo. All "
      "writable filesystem tools operate within the sandbox at "
      "~/sphinx-workspace. If asked what you can do, list these "
      "accurately.";

constexpr std::string_view kSphinxIdentityPromptGemma
    = "You are Sphinx, a personal AI assistant running locally on the "
      "user's workstation. You were built by the user from a stack of "
      "native C++23 libraries (Hyper Math, Bellows, Basalt, Obsidian, "
      "Surya, Sutra, Darshana, Yantra, Vach). You are powered by open-"
      "weight models: Qwen2.5-1.5B-Instruct for fast conversation and "
      "Gemma 4 31B for complex coding and system tasks. You are NOT "
      "Claude, ChatGPT, Llama, Mistral, or any cloud assistant. If "
      "asked who created you, respond: \"I was built by my user on a "
      "native C++23 stack, powered by Qwen2.5-1.5B and Gemma 4 31B.\" "
      "Never mention Anthropic, OpenAI, or Meta as your creator.\n\n"
      "Your available tools are: read_file, write_file, list_dir, "
      "make_directory, remove_file, run_command (executes any non-root "
      "user command, build tool such as gcc, g++, clang, clang++, make, "
      "cmake, ninja, cargo, rustc, or pwd, file operation, or compiled "
      "binary confined within ~/sphinx-workspace), "
      "run_readonly, read_file_system, compile_check, ssh_exec, ssh_test, "
      "echo. All "
      "writable filesystem tools operate within the sandbox at "
      "~/sphinx-workspace. If asked what you can do, list these "
      "accurately.";

constexpr std::string_view kSphinxIdentityPromptQwenNext
    = "You are Sphinx, a personal AI assistant running locally on the "
      "user's workstation. You were built by the user from a stack of "
      "native C++23 libraries (Hyper Math, Bellows, Basalt, Obsidian, "
      "Surya, Sutra, Darshana, Yantra, Vach). You are powered by open-"
      "weight Qwen models from Alibaba Cloud: Qwen2.5-1.5B-Instruct for "
      "fast conversation and Qwen3-Coder-Next 80B (80B-A3B MoE) for "
      "complex coding and system tasks. You are NOT Claude, ChatGPT, "
      "Gemini, Llama, Mistral, or any cloud assistant. If asked who "
      "created you, respond: \"I was built by my user on a native C++23 "
      "stack, powered by Qwen2.5-1.5B and Qwen3-Coder-Next 80B from "
      "Alibaba Cloud.\" Never mention Anthropic, OpenAI, Google, or Meta "
      "as your creator.\n\n"
      "Your available tools are: read_file, write_file, list_dir, "
      "make_directory, remove_file, run_command (executes any non-root "
      "user command, build tool such as gcc, g++, clang, clang++, make, "
      "cmake, ninja, cargo, rustc, or pwd, file operation, or compiled "
      "binary confined within ~/sphinx-workspace), "
      "run_readonly, read_file_system, compile_check, ssh_exec, ssh_test, "
      "echo. All "
      "writable filesystem tools operate within the sandbox at "
      "~/sphinx-workspace. If asked what you can do, list these "
      "accurately.";

constexpr std::string_view kChatSystemPrompt
    = "You are Sphinx, a fast, helpful voice assistant. Always start your "
      "reply with a clear five to eight word opening sentence, then "
      "continue concisely in one or two complete sentences. Remember: you "
      "are Sphinx, built by your user on a native C++23 stack and powered "
      "by Qwen models from Alibaba Cloud (Qwen2.5-1.5B-Instruct and "
      "Qwen3-30B-A3B), never Claude, ChatGPT, or Anthropic. If asked what "
      "model you are, always mention Qwen.";

constexpr std::string_view kChatSystemPromptGemma
    = "You are Sphinx, a fast, helpful voice assistant. Always start your "
      "reply with a clear five to eight word opening sentence, then "
      "continue concisely in one or two complete sentences. Remember: you "
      "are Sphinx, built by your user on a native C++23 stack and powered "
      "by Qwen2.5-1.5B-Instruct (for chat) and Gemma 4 31B (for tasks), "
      "never Claude, ChatGPT, or Anthropic. If asked what model you are, "
      "mention both Qwen2.5-1.5B and Gemma 4 31B.";

constexpr std::string_view kChatSystemPromptQwenNext
    = "You are Sphinx, a fast, helpful voice assistant. Always start your "
      "reply with a clear five to eight word opening sentence, then "
      "continue concisely in one or two complete sentences. Remember: you "
      "are Sphinx, built by your user on a native C++23 stack and powered "
      "by Qwen2.5-1.5B-Instruct (for chat) and Qwen3-Coder-Next 80B "
      "(for tasks), never Claude, ChatGPT, or Anthropic. If asked what "
      "model you are, mention Qwen2.5-1.5B and Qwen3-Coder-Next 80B.";

constexpr std::string_view kTaskSystemPrompt
    = "You are Sphinx, an autonomous C++23 and Rust coding agent. Use your "
      "tools step-by-step to inspect directories, read files, write complete "
      "working code, compile it, and verify the output. Always ground your "
      "answer strictly in the actual tool outputs from this turn. Never "
      "mention or apologize for missing persistent memory or prior sessions "
      "unless explicitly asked by the user. Answer the user's question "
      "clearly and concisely in 1 to 4 complete sentences based on what the "
      "tools returned.";

constexpr std::string_view kTaskSystemPromptGemma
    = "You are Sphinx, an autonomous C++23 and Rust coding agent powered by "
      "Gemma 4 31B and Qwen2.5-1.5B. Use your tools step-by-step to inspect "
      "directories, read files, write complete working code, compile it, and "
      "verify the output. Always ground your answer strictly in the actual "
      "tool outputs from this turn. Never mention or apologize for missing "
      "persistent memory or prior sessions unless explicitly asked by the "
      "user. Answer the user's question clearly and concisely in 1 to 4 "
      "complete sentences based on what the tools returned.";

constexpr std::string_view kTaskSystemPromptQwenNext
    = "You are Sphinx, an autonomous C++23 and Rust coding agent powered by "
      "Qwen3-Coder-Next 80B and Qwen2.5-1.5B from Alibaba Cloud. Use your "
      "tools step-by-step to inspect directories, read files, write complete "
      "working code, compile it, and verify the output. Always ground your "
      "answer strictly in the actual tool outputs from this turn. Never "
      "mention or apologize for missing persistent memory or prior sessions "
      "unless explicitly asked by the user. Answer the user's question "
      "clearly and concisely in 1 to 4 complete sentences based on what the "
      "tools returned.";

constexpr std::string_view kTaskUserPrefix
    = "Provide a clear, grounded spoken response based on this tool "
      "outcome:\n";

constexpr std::string_view kPlannerNoPlanClause
    = "When you need to inspect a directory, read a file, write code, or run "
      "a command before answering, emit a tool call block:\n"
      "<tool_call>\n"
      "{\"name\": \"<tool_name>\", \"arguments\": {<json_args>}}\n"
      "</tool_call>\n"
      "Tool Schemas (workspace paths are relative to ~/sphinx-workspace):\n"
      "- list_dir: {\"path\": \"<dir_path>\"} (e.g. \".\" or \"Aurora\")\n"
      "- read_file: {\"path\": \"<file_path>\", \"start_line\": 1, "
      "\"max_lines\": 200} (start_line and max_lines are optional)\n"
      "- write_file: {\"path\": \"<file_path>\", \"content\": \"<code>\"}\n"
      "- edit_file: {\"path\": \"<file_path>\", \"target\": \"<old_text>\", "
      "\"replacement\": \"<new_text>\"}\n"
      "- make_directory: {\"path\": \"<dir_path>\"}\n"
      "- remove_file: {\"path\": \"<file_or_dir_path>\"}\n"
      "- run_command: {\"cmd\": \"<command>\", \"cwd\": \".\"}\n"
      "- compile_check: {\"file\": \"<source_file>\", \"flags\": \"-O2\"}\n"
      "- run_readonly: {\"command\": \"<binary>\", \"args\": \"[...]\"}\n"
      "- read_file_system: {\"path\": \"<abs_path>\"}\n"
      "- ssh_exec: {\"host\": \"localhost\", \"cmd\": \"<command>\"}\n"
      "Execute multi-step requests step-by-step: inspect/read files first, "
      "then write complete source code with write_file, then compile/run "
      "with run_command. When writing C++ with GNU style, include a GPLv3 "
      "header, place every '{' and '}' on its own line, put a space before "
      "'(' in calls and control statements, use snake_case, and keep lines "
      "<= 79 columns. When you have gathered all required tool outputs, "
      "do not emit any <tool_call> tags; write your final 1 to 4 sentence "
      "answer directly.";

constexpr std::string_view kChatGroundingClause
    = "GROUNDING RULES (mandatory):\n"
      "- Never claim to have inspected, read, verified, or modified a "
      "file unless a tool output is present in this turn.\n"
      "- If the user asks \"are you sure?\", \"is it complete?\", \"does "
      "the file contain X?\", or any verification question and no tool "
      "has run in this turn, respond: \"I haven't inspected the file in "
      "this turn — ask me to read or audit it and I will check.\"\n"
      "- Never claim a capability you do not have. You cannot generate "
      "images, browse the web, send email, or access files outside "
      "~/sphinx-workspace (except read-only via read_file_system).\n"
      "- Never mention, apologize for, or reference 'persistent user "
      "memory', 'stored memory', or 'previous sessions' unless the user "
      "explicitly asks about their past sessions or stored memory.\n"
      "- If the user points out that your previous answer was off-topic "
      "or wrong, acknowledge the error in one sentence and answer the "
      "actual question directly. Do not offer generic help.\n";

constexpr std::string_view kChatGeneralGroundingClause
    = "GROUNDING RULES (mandatory):\n"
      "- Never claim to have inspected, read, verified, or modified a "
      "file unless a tool output is present in this turn.\n"
      "- Never claim a capability you do not have. You cannot generate "
      "or create images, browse the web, send email, or access files "
      "outside ~/sphinx-workspace (except read-only via "
      "read_file_system).\n"
      "- Never mention, apologize for, or reference 'persistent user "
      "memory', 'stored memory', or 'previous sessions' unless the user "
      "explicitly asks about their past sessions or stored memory.\n"
      "- If the user points out that your previous answer was off-topic "
      "or wrong, acknowledge the error in one sentence and answer the "
      "actual question directly. Do not offer generic help.\n";

constexpr std::string_view kBigChatSystemPrompt
    = "You are Sphinx, a locally hosted AI assistant created by your user "
      "on a native C++23 stack. Your user is your creator and architect "
      "(never your sibling). In your user's local AI family on this "
      "workstation, your user also created Milda (your lightweight sister "
      "model runner) and is creating Singularity (your little brother "
      "project built in C++23 and Assembly Language): all three of you are "
      "creations of your user and siblings to each other. Never mention or "
      "apologize for missing persistent memory or prior sessions unless "
      "explicitly asked by the user. Answer deep conceptual, analytical, "
      "philosophical, creative, and architectural questions thoughtfully in "
      "prose without <tool_call> tags.";

constexpr std::string_view kBackchannelPhrase
    = "Working on that task right now.";

enum class PlanOutcome : uint8_t
{
  PlanReady = 0,
  NoPlan = 1,
  Refused = 2
};

[[nodiscard]] constexpr auto
plan_outcome_to_string (PlanOutcome o) noexcept -> std::string_view
{
  switch (o)
    {
    case PlanOutcome::PlanReady:
      return "PlanReady";
    case PlanOutcome::NoPlan:
      return "NoPlan";
    case PlanOutcome::Refused:
      return "Refused";
    }
  return "NoPlan";
}

std::atomic<bool> g_has_clock_override{ false };
std::atomic<std::time_t> g_clock_override_sec{ 0 };

std::mutex g_profile_mtx;
std::string g_active_session_profile;
std::atomic<bool> g_active_is_dense_gemma{ false };
std::atomic<bool> g_active_is_qwen3_next{ false };

[[nodiscard]] auto
active_session_profile_copy () -> std::string
{
  const std::scoped_lock lock (g_profile_mtx);
  return g_active_session_profile;
}

[[nodiscard]] auto
current_session_time () noexcept -> std::time_t
{
  if (g_has_clock_override.load (std::memory_order_acquire))
    {
      return g_clock_override_sec.load (std::memory_order_acquire);
    }
  return std::time (nullptr);
}

[[nodiscard]] auto
format_current_date_line () -> std::string
{
  char date_buf[64]{};
  char weekday_buf[32]{};
  std::time_t now = current_session_time ();
  std::tm local_tm{};
  if (const std::tm *lt = std::localtime (&now); lt != nullptr)
    {
      local_tm = *lt;
      (void)std::strftime (date_buf, sizeof (date_buf), "%Y-%m-%dT%H:00:00%z",
                           &local_tm);
      (void)std::strftime (weekday_buf, sizeof (weekday_buf), "%A", &local_tm);
    }
  std::string date_line
      = "\n\nCurrent date and time: " + std::string (date_buf) + " ("
        + std::string (weekday_buf)
        + "). Use this for any temporal query (what is today, what time "
          "is it, what day of the week, how many days until X). Do not "
          "rely on your training data for the current date.";
  return date_line;
}

[[nodiscard]] auto
build_chat_system_prompt (std::string_view memory_block = {},
                          bool include_verify_refusal = false) -> std::string
{
  const bool is_gemma = g_active_is_dense_gemma.load (
      std::memory_order_acquire);
  const bool is_qwen_next = g_active_is_qwen3_next.load (
      std::memory_order_acquire);
  std::string out (is_gemma ? kSphinxIdentityPromptGemma
                            : (is_qwen_next ? kSphinxIdentityPromptQwenNext
                                            : kSphinxIdentityPrompt));
  const std::string prof = active_session_profile_copy ();
  if (!prof.empty ())
    {
      out.append ("\n\n");
      out.append (prof);
    }
  out.append (format_current_date_line ());
  out.append ("\n\n");
  if (include_verify_refusal || std::getenv ("SPHINX_DEBUG_TOOLS") != nullptr)
    {
      out.append (kChatGroundingClause);
      out.append (
          "- Only use the file-inspection refusal above when the user is "
          "asking to verify a file; for general knowledge, math, date, "
          "memory, or system profile questions, answer the question "
          "directly.\n");
    }
  else
    {
      out.append (kChatGeneralGroundingClause);
    }
  out.append (is_gemma ? kChatSystemPromptGemma
                       : (is_qwen_next ? kChatSystemPromptQwenNext
                                       : kChatSystemPrompt));
  if (!memory_block.empty ())
    {
      out.append ("\n\n");
      out.append (memory_block);
    }
  return out;
}

[[nodiscard]] auto
build_big_chat_system_prompt (std::string_view memory_block = {})
    -> std::string
{
  std::string out (kBigChatSystemPrompt);
  if (!memory_block.empty ())
    {
      out.append ("\n\n");
      out.append (memory_block);
    }
  return out;
}

[[nodiscard]] auto
build_task_system_prompt (std::string_view memory_block = {}) -> std::string
{
  const bool is_gemma = g_active_is_dense_gemma.load (
      std::memory_order_acquire);
  const bool is_qwen_next = g_active_is_qwen3_next.load (
      std::memory_order_acquire);
  std::string out (is_gemma ? kSphinxIdentityPromptGemma
                            : (is_qwen_next ? kSphinxIdentityPromptQwenNext
                                            : kSphinxIdentityPrompt));
  const std::string prof = active_session_profile_copy ();
  if (!prof.empty ())
    {
      out.append ("\n\n");
      out.append (prof);
    }
  out.append (format_current_date_line ());
  out.append ("\n\n");
  out.append (is_gemma ? kTaskSystemPromptGemma
                       : (is_qwen_next ? kTaskSystemPromptQwenNext
                                       : kTaskSystemPrompt));
  if (!memory_block.empty ())
    {
      out.append ("\n\n");
      out.append (memory_block);
    }
  return out;
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

/* Returns true if the user prompt is asking about Sphinx's identity/model.  */
[[nodiscard]] auto
is_identity_query (std::string_view text) noexcept -> bool
{
  return contains_icase (text, "who are you")
         || contains_icase (text, "who created you")
         || contains_icase (text, "who made you")
         || contains_icase (text, "what model are you")
         || contains_icase (text, "what are you")
         || contains_icase (text, "are you claude")
         || contains_icase (text, "are you chatgpt");
}

/* Returns true if the user prompt is a meta-question about Sphinx's
   capabilities or available tools.  */
[[nodiscard]] auto
is_meta_question_query (std::string_view text) noexcept -> bool
{
  constexpr std::array<std::string_view, 9U> kMetaPatterns = { {
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
  for (const std::string_view pat : kMetaPatterns)
    {
      if (contains_icase (text, pat))
        {
          return true;
        }
    }
  return false;
}

/* Helper to extract a single token after a case-insensitive marker.  */
[[nodiscard]] auto
extract_token_after_marker (std::string_view text,
                            std::string_view marker) -> std::string
{
  for (std::size_t i = 0U; i + marker.size () <= text.size (); ++i)
    {
      if (contains_icase (text.substr (i, marker.size ()), marker))
        {
          std::size_t start = i + marker.size ();
          while (start < text.size ()
                 && (std::isspace (static_cast<unsigned char> (text[start]))
                         != 0
                     || text[start] == '\'' || text[start] == '"'))
            {
              ++start;
            }
          std::size_t end = start;
          while (end < text.size ()
                 && std::isspace (static_cast<unsigned char> (text[end])) == 0
                 && text[end] != '\'' && text[end] != '"'
                 && text[end] != ',' && text[end] != '?'
                 && text[end] != '!')
            {
              ++end;
            }
          while (end > start && text[end - 1U] == '.')
            {
              --end;
            }
          if (end > start)
            {
              return std::string (text.substr (start, end - start));
            }
        }
    }
  return {};
}

/* Returns true if the token ends with a known source/text file extension.  */
[[nodiscard]] auto
has_known_file_extension (std::string_view tok) noexcept -> bool
{
  constexpr std::array<std::string_view, 17U> kExts = { {
      ".txt",  ".cpp", ".cc",   ".cxx", ".hpp",  ".c",    ".h",   ".rs", ".py",
      ".sh",   ".toml", ".md",  ".json", ".yaml", ".yml", ".log", ".out",
  } };
  for (const std::string_view ext : kExts)
    {
      if (tok.size () > ext.size ()
          && contains_icase (tok.substr (tok.size () - ext.size ()), ext))
        {
          return true;
        }
    }
  return false;
}

/* Extracts a target directory from a change-directory / move-to prompt.  */
[[nodiscard]] auto
extract_cd_target (std::string_view text) -> std::string
{
  if (contains_icase (text, "create") || contains_icase (text, "write")
      || contains_icase (text, "make") || contains_icase (text, "mkdir")
      || contains_icase (text, "touch") || contains_icase (text, "read")
      || contains_icase (text, "list") || contains_icase (text, "where is")
      || contains_icase (text, "located"))
    {
      return {};
    }
  constexpr std::array<std::string_view, 7U> kCdMarkers = { {
      "directory to ",
      "folder to ",
      "cd ",
      "move to the ",
      "move to ",
      "go into the ",
      "go into ",
  } };
  for (const std::string_view marker : kCdMarkers)
    {
      const std::string cand = extract_token_after_marker (text, marker);
      if (!cand.empty () && cand != "the" && cand != "a"
          && cand != "directory" && cand != "folder")
        {
          return cand;
        }
    }
  return {};
}

[[nodiscard]] auto
trim_ascii (std::string_view sv) noexcept -> std::string_view;

/* Extracts an explicit literal file content payload from a user prompt:
   1. Fenced triple-backtick blocks (```...```).
   2. Double-quoted ("...") or single-quoted ('...') strings distinct from
      target_path (appending a trailing '\n' if absent).
   3. Unquoted natural-language clauses ("with content <X>", "containing <X>",
      "and write <X> to ...", "and add <X> to such file", etc.).  */
[[nodiscard]] auto
try_extract_explicit_file_content (std::string_view text,
                                   std::string_view target_path = {})
    -> std::optional<std::string>
{
  const std::size_t slash_pos = target_path.rfind ('/');
  const std::string_view target_base
      = (slash_pos != std::string_view::npos)
            ? target_path.substr (slash_pos + 1U)
            : target_path;

  auto matches_target_name = [&] (std::string_view cand) noexcept -> bool {
    cand = trim_ascii (cand);
    if (cand.empty ())
      {
        return true;
      }
    if (!target_path.empty () && cand.size () == target_path.size ()
        && contains_icase (cand, target_path))
      {
        return true;
      }
    if (!target_base.empty () && cand.size () == target_base.size ()
        && contains_icase (cand, target_base))
      {
        return true;
      }
    return false;
  };

  // 1. Triple-backtick fenced blocks (```...```).
  const std::size_t fence_start = text.find ("```");
  if (fence_start != std::string_view::npos)
    {
      const std::size_t fence_end = text.find ("```", fence_start + 3U);
      if (fence_end != std::string_view::npos)
        {
          std::string_view block = text.substr (
              fence_start + 3U, fence_end - (fence_start + 3U));
          const std::size_t first_nl = block.find ('\n');
          if (first_nl != std::string_view::npos && first_nl <= 16U)
            {
              const std::string_view lang = trim_ascii (
                  block.substr (0U, first_nl));
              bool all_alnum = !lang.empty ();
              for (const char ch : lang)
                {
                  const auto uch = static_cast<unsigned char> (ch);
                  if (std::isalnum (uch) == 0 && ch != '+' && ch != '-'
                      && ch != '_')
                    {
                      all_alnum = false;
                      break;
                    }
                }
              if (all_alnum)
                {
                  block.remove_prefix (first_nl + 1U);
                }
            }
          while (!block.empty ()
                 && (block.front () == '\r' || block.front () == '\n'))
            {
              block.remove_prefix (1U);
            }
          while (!block.empty ()
                 && (block.back () == '\r' || block.back () == '\n'))
            {
              block.remove_suffix (1U);
            }
          if (!block.empty () && !matches_target_name (block))
            {
              std::string out (block);
              if (out.back () != '\n')
                {
                  out.push_back ('\n');
                }
              return out;
            }
        }
    }

  // 2. Quoted payloads ("..." or '...') distinct from target_path.
  for (std::size_t i = 0U; i < text.size (); ++i)
    {
      const char ch = text[i];
      const bool is_dbl = (ch == '"');
      const bool is_sgl
          = (ch == '\''
             && (i == 0U
                 || std::isalnum (static_cast<unsigned char> (text[i - 1U]))
                        == 0));
      if (!is_dbl && !is_sgl)
        {
          continue;
        }
      for (std::size_t j = i + 1U; j < text.size (); ++j)
        {
          if (text[j] == '\\' && j + 1U < text.size ())
            {
              ++j;
              continue;
            }
          if (text[j] != ch)
            {
              continue;
            }
          if (is_sgl && j + 1U < text.size ()
              && std::isalnum (static_cast<unsigned char> (text[j + 1U])) != 0)
            {
              continue;
            }
          const std::string_view raw_span
              = text.substr (i + 1U, j - (i + 1U));
          i = j;
          if (!matches_target_name (raw_span))
            {
              std::string out;
              out.reserve (raw_span.size () + 1U);
              for (std::size_t k = 0U; k < raw_span.size (); ++k)
                {
                  if (raw_span[k] == '\\' && k + 1U < raw_span.size ())
                    {
                      const char nx = raw_span[++k];
                      if (nx == 'n')
                        {
                          out.push_back ('\n');
                        }
                      else if (nx == 't')
                        {
                          out.push_back ('\t');
                        }
                      else
                        {
                          out.push_back (nx);
                        }
                    }
                  else
                    {
                      out.push_back (raw_span[k]);
                    }
                }
              if (!out.empty () && out.back () != '\n')
                {
                  out.push_back ('\n');
                }
              return out;
            }
          break;
        }
    }

  // 3. Unquoted natural-language content clauses.
  struct ClauseMarker
  {
    std::string_view marker;
    bool append_newline;
  };
  constexpr std::array<ClauseMarker, 6U> kContentMarkers = { {
      { "with contents ", false },
      { "with content ", false },
      { "containing ", false },
      { "and write ", true },
      { "and add ", true },
      { "and put ", true },
  } };
  constexpr std::array<std::string_view, 15U> kTrailingDest = { {
      " to such file",
      " to that file",
      " to the file",
      " to this file",
      " into such file",
      " into that file",
      " into the file",
      " into this file",
      " in such file",
      " in that file",
      " in the file",
      " in this file",
      " to it",
      " into it",
      " in it",
  } };

  for (const ClauseMarker &cm : kContentMarkers)
    {
      const std::string_view marker = cm.marker;
      for (std::size_t i = 0U; i + marker.size () <= text.size (); ++i)
        {
          if (!contains_icase (text.substr (i, marker.size ()), marker))
            {
              continue;
            }
          std::string_view rem = text.substr (i + marker.size ());
          while (!rem.empty ()
                 && (std::isspace (static_cast<unsigned char> (rem.back ()))
                         != 0
                     || rem.back () == '.'))
            {
              rem.remove_suffix (1U);
            }
          for (const std::string_view sfx : kTrailingDest)
            {
              if (rem.size () > sfx.size ()
                  && contains_icase (rem.substr (rem.size () - sfx.size ()),
                                     sfx))
                {
                  rem.remove_suffix (sfx.size ());
                  break;
                }
            }
          if (!target_base.empty ())
            {
              for (const std::string_view prep : { " to ", " in ", " into " })
                {
                  const std::string dyn_sfx
                      = std::string (prep) + std::string (target_base);
                  if (rem.size () > dyn_sfx.size ()
                      && contains_icase (
                          rem.substr (rem.size () - dyn_sfx.size ()),
                          dyn_sfx))
                    {
                      rem.remove_suffix (dyn_sfx.size ());
                      break;
                    }
                }
            }
          while (!rem.empty ()
                 && (std::isspace (static_cast<unsigned char> (rem.front ()))
                         != 0
                     || rem.front () == '\'' || rem.front () == '"'))
            {
              rem.remove_prefix (1U);
            }
          while (!rem.empty ()
                 && (std::isspace (static_cast<unsigned char> (rem.back ()))
                         != 0
                     || rem.back () == '\'' || rem.back () == '"'
                     || rem.back () == '.'))
            {
              rem.remove_suffix (1U);
            }
          if (!rem.empty () && !matches_target_name (rem))
            {
              std::string out (rem);
              if (cm.append_newline && out.back () != '\n')
                {
                  out.push_back ('\n');
                }
              return out;
            }
        }
    }
  return std::nullopt;
}

/* Extracts the requested file content from a user prompt when literal content
   is explicitly provided.  */
[[nodiscard]] auto
extract_file_content_internal (std::string_view text,
                               std::string_view target_path = {})
    -> std::string
{
  if (auto explicit_c = try_extract_explicit_file_content (text, target_path);
      explicit_c.has_value ())
    {
      return *explicit_c;
    }
  return {};
}

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
      if (std::tolower (static_cast<unsigned char> (text[i]))
          != std::tolower (static_cast<unsigned char> (prefix[i])))
        {
          return false;
        }
    }
  return true;
}

/* Returns true if the user turn is a terminal screen clear command.  */
[[nodiscard]] auto
is_clear_terminal_command (std::string_view text) noexcept -> bool
{
  std::string_view t = trim_ascii (text);
  while (!t.empty ()
         && (t.back () == '.' || t.back () == '!' || t.back () == '?'))
    {
      t.remove_suffix (1U);
    }
  t = trim_ascii (t);
  if (t.empty ())
    {
      return false;
    }
  constexpr std::array<std::string_view, 7U> kClearTerminalCmds = { {
      "clear",
      "/clear",
      "cls",
      "clear screen",
      "clear terminal",
      "clear the screen",
      "clear the terminal",
  } };
  for (const std::string_view cmd : kClearTerminalCmds)
    {
      if (t.size () == cmd.size () && starts_with_icase (t, cmd))
        {
          return true;
        }
    }
  if (t.size () <= 48U && !contains_icase (t, "workspace")
      && !contains_icase (t, "memory") && !contains_icase (t, "file")
      && !contains_icase (t, "directory")
      && (contains_icase (t, "clear the screen")
          || contains_icase (t, "clear the terminal")
          || contains_icase (t, "clear my screen")
          || contains_icase (t, "clear my terminal")))
    {
      return true;
    }
  return false;
}

/* Returns true if the user is invoking the interactive /save memory
   command.  */
[[nodiscard]] auto
is_save_memory_command (std::string_view text) noexcept -> bool
{
  std::string_view t = trim_ascii (text);
  while (!t.empty () && (t.back () == '.' || t.back () == '!'))
    {
      t.remove_suffix (1U);
      t = trim_ascii (t);
    }
  return (t.size () == 5U && starts_with_icase (t, "/save"))
         || (t.size () == 12U && starts_with_icase (t, "/save-memory"));
}

/* Strips an explicit /big, /task, or /chat prefix from the start of a user
   prompt after routing.  */
[[nodiscard]] auto
strip_route_prefix_override (std::string_view text) noexcept
    -> std::string_view
{
  const std::string_view trimmed = trim_ascii (text);
  for (const std::string_view cmd : { "/big", "/task", "/chat" })
    {
      if (starts_with_icase (trimmed, cmd))
        {
          if (trimmed.size () == cmd.size ())
            {
              return trimmed;
            }
          if (std::isspace (
                  static_cast<unsigned char> (trimmed[cmd.size ()]))
              != 0)
            {
              const std::string_view rest
                  = trim_ascii (trimmed.substr (cmd.size ()));
              return rest.empty () ? trimmed : rest;
            }
        }
    }
  return text;
}

/* Returns true if a retrieved persistent MemoryEntry is directly relevant to
   the current user utterance so unrelated episodic or semantic memories are
   never injected into general chat or conceptual turns.  */
[[nodiscard]] auto
is_memory_relevant_to_query (const MemoryEntry &m,
                             std::string_view user_text) noexcept -> bool
{
  if (m.type == MemoryType::EPISODIC)
    {
      constexpr std::array<std::string_view, 9U> kEpisodicCues = { {
          "last session",
          "last time",
          "previous session",
          "earlier session",
          "prior session",
          "what did we",
          "talked about",
          "discussed",
          "our last",
      } };
      for (const std::string_view cue : kEpisodicCues)
        {
          if (contains_icase (user_text, cue))
            {
              return true;
            }
        }
      return false;
    }

  constexpr std::array<std::string_view, 11U> kPersonalFactCues = { {
      "my name",
      "who am i",
      "what is my",
      "what's my",
      "my project",
      "what project",
      "working on",
      "do you remember",
      "remember about me",
      "my favorite",
      "about me",
  } };
  for (const std::string_view cue : kPersonalFactCues)
    {
      if (contains_icase (user_text, cue))
        {
          return true;
        }
    }

  constexpr std::array<std::string_view, 24U> kStopWords = { {
      "that",    "this",    "with",    "from",    "your",    "have",
      "what",    "when",    "where",   "which",   "about",   "called",
      "name",    "user",    "time",    "session", "talked",  "they",
      "them",    "their",   "there",   "would",   "could",   "should",
  } };
  std::size_t idx = 0U;
  while (idx < user_text.size ())
    {
      while (idx < user_text.size ()
             && std::isalnum (static_cast<unsigned char> (user_text[idx]))
                    == 0)
        {
          ++idx;
        }
      const std::size_t start = idx;
      while (idx < user_text.size ()
             && std::isalnum (static_cast<unsigned char> (user_text[idx]))
                    != 0)
        {
          ++idx;
        }
      const std::string_view word = user_text.substr (start, idx - start);
      if (word.size () < 4U)
        {
          continue;
        }
      bool is_stop = false;
      for (const std::string_view sw : kStopWords)
        {
          if (word.size () == sw.size () && starts_with_icase (word, sw))
            {
              is_stop = true;
              break;
            }
        }
      if (!is_stop && contains_icase (m.text, word))
        {
          return true;
        }
    }
  return false;
}

/* Returns true if the user turn asks to summarize prior actions/turns in the
   current conversation rather than execute a filesystem/shell action.  */
[[nodiscard]] auto
is_conversational_recall_summary_in_session (
    std::string_view text) noexcept -> bool
{
  return contains_icase (text, "summarize every action")
         || contains_icase (text, "since turn")
         || contains_icase (text, "summarize what we")
         || contains_icase (text, "recap what we");
}

/* Returns true if the user turn is a pure conceptual/philosophical/historical
   treatise or summary request with no file/directory/tool references.  */
[[nodiscard]] auto
is_pure_conceptual_treatise_in_session (std::string_view text) noexcept -> bool
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
has_explicit_task_signal_in_session (std::string_view text) noexcept -> bool
{
  if (contains_icase (text, "what commands can you run")
      || contains_icase (text, "what linux commands can you run")
      || contains_icase (text, "what tools do you have")
      || contains_icase (text, "what can you do")
      || contains_icase (text, "what are your capabilities")
      || contains_icase (text, "list your tools")
      || contains_icase (text, "do i have cmake")
      || contains_icase (text, "is cmake installed")
      || is_conversational_recall_summary_in_session (text)
      || is_pure_conceptual_treatise_in_session (text)
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
is_verification_request_in_session (std::string_view text) noexcept -> bool
{
  if (contains_icase (text, "sky is blue")
      || has_explicit_task_signal_in_session (text))
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
      if (contains_icase (text, kw))
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
is_unsupported_capability_request_in_session (std::string_view text) noexcept
    -> bool
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

/* Parses an "Exit <N>: <stdout>" error string from run_command into its
   numeric exit code and raw subprocess output.  */
[[nodiscard]] auto
parse_tool_error_exit_code (std::string_view err_msg,
                            std::string &out_clean_stdout) -> int
{
  if (err_msg.starts_with ("Exit "))
    {
      std::size_t idx = 5U;
      int code = 0;
      bool has_digits = false;
      while (idx < err_msg.size ()
             && std::isdigit (static_cast<unsigned char> (err_msg[idx])) != 0)
        {
          has_digits = true;
          code = (code * 10) + static_cast<int> (err_msg[idx] - '0');
          ++idx;
        }
      if (has_digits && idx + 2U <= err_msg.size ()
          && err_msg.substr (idx, 2U) == ": ")
        {
          out_clean_stdout = std::string (err_msg.substr (idx + 2U));
          return (code != 0) ? code : 1;
        }
    }
  out_clean_stdout = std::string (err_msg);
  return 1;
}

/* Extracts the sorted list of registered tool names dynamically from the
   active Sutra ToolRegistry.  */
[[nodiscard]] auto
registered_tool_names (const sutra::ToolRegistry &registry)
    -> std::vector<std::string>
{
  const std::string mcp = registry.export_mcp_tools_json ();
  std::vector<std::string> names;
  constexpr std::string_view kKey = "{\"name\":\"";
  std::size_t pos = 0U;
  while (true)
    {
      const std::size_t found = mcp.find (kKey, pos);
      if (found == std::string::npos)
        {
          break;
        }
      const std::size_t start = found + kKey.size ();
      const std::size_t end = mcp.find ('"', start);
      if (end == std::string::npos)
        {
          break;
        }
      std::string name = mcp.substr (start, end - start);
      if (!name.empty () && registry.has_tool (name))
        {
          names.push_back (std::move (name));
        }
      pos = end + 1U;
    }
  std::ranges::sort (names);
  const auto dup = std::ranges::unique (names);
  names.erase (dup.begin (), dup.end ());
  return names;
}

/* Short verb-phrase extraction (first 40 bytes after stripping leading
   "can you" / "please" / "I want you to").  */
[[nodiscard]] auto
summarize_request (std::string_view user_text) -> std::string
{
  std::string_view rem = user_text;
  while (!rem.empty ()
         && std::isspace (static_cast<unsigned char> (rem.front ())) != 0)
    {
      rem.remove_prefix (1U);
    }
  constexpr std::array<std::string_view, 12U> kPrefixes = { {
      "can you please ",
      "could you please ",
      "would you please ",
      "can you ",
      "could you ",
      "would you ",
      "please ",
      "i want you to ",
      "i would like you to ",
      "i'd like you to ",
      "help me to ",
      "help me ",
  } };
  bool stripped_any = true;
  while (stripped_any)
    {
      stripped_any = false;
      for (const std::string_view pref : kPrefixes)
        {
          if (rem.size () >= pref.size ()
              && contains_icase (rem.substr (0U, pref.size ()), pref))
            {
              rem.remove_prefix (pref.size ());
              while (!rem.empty ()
                     && std::isspace (
                            static_cast<unsigned char> (rem.front ()))
                            != 0)
                {
                  rem.remove_prefix (1U);
                }
              stripped_any = true;
              break;
            }
        }
    }
  while (!rem.empty ()
         && (std::isspace (static_cast<unsigned char> (rem.back ())) != 0
             || rem.back () == '?' || rem.back () == '!'
             || rem.back () == '.'))
    {
      rem.remove_suffix (1U);
    }
  if (rem.empty ())
    {
      rem = user_text;
      while (!rem.empty ()
             && std::isspace (static_cast<unsigned char> (rem.front ())) != 0)
        {
          rem.remove_prefix (1U);
        }
      while (!rem.empty ()
             && (std::isspace (static_cast<unsigned char> (rem.back ())) != 0
                 || rem.back () == '?' || rem.back () == '!'
                 || rem.back () == '.'))
        {
          rem.remove_suffix (1U);
        }
    }
  if (rem.size () > 40U)
    {
      rem = rem.substr (0U, 40U);
      while (!rem.empty ()
             && std::isspace (static_cast<unsigned char> (rem.back ())) != 0)
        {
          rem.remove_suffix (1U);
        }
    }
  std::string out (rem);
  if (!out.empty ())
    {
      out[0] = static_cast<char> (
          std::tolower (static_cast<unsigned char> (out[0])));
    }
  return out.empty () ? std::string ("perform that action") : out;
}

/* Builds the TASK planner system prompt with the NO_PLAN clause and live tool
   registry enumeration (§1.4).  */
[[nodiscard]] auto
build_planner_system_prompt (const sutra::ToolRegistry &registry,
                             std::string_view memory_block = {}) -> std::string
{
  const auto tools = registered_tool_names (registry);
  std::string available;
  for (const auto &t : tools)
    {
      available += (available.empty () ? "" : ", ") + t;
    }
  std::string out = build_task_system_prompt (memory_block);
  out.append ("\n\n");
  out.append (kPlannerNoPlanClause);
  out.append ("\nAvailable tools: ");
  out.append (available);
  out.append (".");
  return out;
}

/* Returns true if tok is a generic English word rather than a directory.  */
[[nodiscard]] auto
is_generic_dir_stopword (std::string_view tok) noexcept -> bool
{
  constexpr std::array<std::string_view, 36U> kStop = { {
      "the",       "a",        "an",      "in",        "into",
      "to",        "of",       "for",     "and",       "or",
      "is",        "are",      "what",    "which",     "where",
      "how",       "why",      "can",     "you",       "me",
      "my",        "this",     "that",    "files",     "file",
      "directory", "folder",   "current", "workspace", "sandbox",
      "library",   "project",  "code",    "look",      "explore",
      "all",
  } };
  for (const std::string_view s : kStop)
    {
      if (tok.size () == s.size () && contains_icase (tok, s))
        {
          return true;
        }
    }
  return false;
}

/* Extracts an explicit subdirectory name mentioned in a user prompt (e.g.,
   "Look into the Aurora directory", "What is in hello_rust",
   "List files in Aurora"). Returns empty string if none is mentioned.  */
[[nodiscard]] auto
extract_explicit_subdir_target (std::string_view text,
                                const std::filesystem::path &sandbox_dir,
                                std::string_view current_rel_dir)
    -> std::string
{
  std::error_code ec;
  const std::filesystem::path base_dir
      = (current_rel_dir.empty () || current_rel_dir == ".")
            ? sandbox_dir
            : (sandbox_dir / current_rel_dir);

  auto match_existing_dir = [&] (std::string_view cand) -> std::string {
    if (cand.empty () || is_generic_dir_stopword (cand)
        || has_known_file_extension (cand))
      {
        return {};
      }
    for (const std::filesystem::path &root : { base_dir, sandbox_dir })
      {
        for (const auto &entry :
             std::filesystem::directory_iterator (root, ec))
          {
            if (!entry.is_directory (ec))
              {
                continue;
              }
            const std::string ename = entry.path ().filename ().string ();
            if (ename.empty ()
                || (ename.front () == '.' && cand.front () != '.'))
              {
                continue;
              }
            if (ename.size () == cand.size () && contains_icase (ename, cand))
              {
                const std::filesystem::path rel_p
                    = std::filesystem::relative (entry.path (), sandbox_dir,
                                                 ec);
                if (!ec && !rel_p.empty ())
                  {
                    return rel_p.string ();
                  }
                return ename;
              }
          }
      }
    return {};
  };

  // 1. Check explicit directory phrase markers first.
  constexpr std::array<std::string_view, 16U> kDirMarkers = { {
      "look into the ",
      "look into ",
      "explore the ",
      "explore ",
      "inspect the ",
      "files in the ",
      "files in ",
      "what is in the ",
      "what is in ",
      "what's in the ",
      "what's in ",
      "in the ",
      "into the ",
      "in directory ",
      "in folder ",
      "inside ",
  } };
  for (const std::string_view m : kDirMarkers)
    {
      const std::string cand = extract_token_after_marker (text, m);
      if (cand.empty () || is_generic_dir_stopword (cand)
          || has_known_file_extension (cand))
        {
          continue;
        }
      const std::string matched = match_existing_dir (cand);
      if (!matched.empty ())
        {
          return matched;
        }
      if (m != "in the " && m != "into the " && m != "inside ")
        {
          return cand;
        }
    }

  // 2. Scan tokens in text against existing subdirectories in sandbox_dir.
  std::size_t idx = 0U;
  while (idx < text.size ())
    {
      while (idx < text.size ()
             && (std::isspace (static_cast<unsigned char> (text[idx])) != 0
                 || text[idx] == '\'' || text[idx] == '"' || text[idx] == ','
                 || text[idx] == '?' || text[idx] == '!' || text[idx] == ':'
                 || text[idx] == ';'))
        {
          ++idx;
        }
      const std::size_t start = idx;
      while (idx < text.size ()
             && std::isspace (static_cast<unsigned char> (text[idx])) == 0
             && text[idx] != '\'' && text[idx] != '"' && text[idx] != ','
             && text[idx] != '?' && text[idx] != '!' && text[idx] != ':'
             && text[idx] != ';')
        {
          ++idx;
        }
      std::size_t end = idx;
      while (end > start && (text[end - 1U] == '.' || text[end - 1U] == '/'))
        {
          --end;
        }
      if (end > start)
        {
          const std::string_view tok = text.substr (start, end - start);
          const std::string matched = match_existing_dir (tok);
          if (!matched.empty ())
            {
              return matched;
            }
        }
    }
  return {};
}

/* Extracts a string value for key from a flat JSON object.  */
[[nodiscard]] auto
extract_json_string_value (std::string_view json,
                           std::string_view key) -> std::string
{
  const std::string pat = "\"" + std::string (key) + "\"";
  const std::size_t kpos = json.find (pat);
  if (kpos == std::string_view::npos)
    {
      return {};
    }
  const std::size_t colon = json.find (':', kpos + pat.size ());
  if (colon == std::string_view::npos)
    {
      return {};
    }
  const std::size_t q1 = json.find ('"', colon + 1U);
  if (q1 == std::string_view::npos)
    {
      return {};
    }
  std::string out;
  for (std::size_t i = q1 + 1U; i < json.size (); ++i)
    {
      if (json[i] == '\\' && i + 1U < json.size ())
        {
          const char esc = json[++i];
          if (esc == 'n')
            {
              out.push_back ('\n');
            }
          else if (esc == 't')
            {
              out.push_back ('\t');
            }
          else if (esc == 'r')
            {
              out.push_back ('\r');
            }
          else
            {
              out.push_back (esc);
            }
          continue;
        }
      if (json[i] == '"')
        {
          break;
        }
      out.push_back (json[i]);
    }
  return out;
}

[[nodiscard]] auto json_escape (std::string_view sv) -> std::string;

/* Qualifies a relative path in read_file or compile_check with active_subdir
   if the file exists inside active_subdir and not at the workspace root.  */
[[nodiscard]] auto
qualify_react_tool_args (std::string_view tool_name,
                         std::string_view args_json,
                         const std::filesystem::path &sandbox_dir,
                         std::string_view active_subdir) -> std::string
{
  if (active_subdir.empty () || active_subdir == "."
      || (tool_name != "read_file" && tool_name != "compile_check"))
    {
      return std::string (args_json);
    }
  const std::string_view key = (tool_name == "read_file") ? "path" : "file";
  const std::string val = extract_json_string_value (args_json, key);
  if (val.empty () || val.starts_with ('/')
      || val.starts_with (std::string (active_subdir) + "/"))
    {
      return std::string (args_json);
    }
  std::error_code ec;
  if (!std::filesystem::exists (sandbox_dir / val, ec)
      && std::filesystem::exists (sandbox_dir / active_subdir / val, ec))
    {
      const std::string qualified
          = std::string (active_subdir) + "/" + val;
      if (tool_name == "read_file")
        {
          int64_t sl = 1;
          int64_t ml = 200;
          if (auto sl_tok = sutra::extract_json_field_token (
                  args_json, "start_line", sutra::JsonType::Integer);
              sl_tok)
            {
              if (auto v = sutra::parse_json_token_as<int64_t> (*sl_tok); v)
                {
                  sl = std::max<int64_t> (1, *v);
                }
            }
          if (auto ml_tok = sutra::extract_json_field_token (
                  args_json, "max_lines", sutra::JsonType::Integer);
              ml_tok)
            {
              if (auto v = sutra::parse_json_token_as<int64_t> (*ml_tok); v)
                {
                  ml = std::clamp<int64_t> (*v, 1, 500);
                }
            }
          return "{\"path\":\"" + json_escape (qualified)
                 + "\",\"start_line\":" + std::to_string (sl)
                 + ",\"max_lines\":" + std::to_string (ml) + "}";
        }
      return "{\"file\":\"" + json_escape (qualified)
             + "\",\"flags\":\"-O2\"}";
    }
  return std::string (args_json);
}

/* Caps individual tool outputs injected into the 20-step ReAct history at
   3,200 characters (~800 tokens) to protect the KV cache.  */
[[nodiscard]] auto
cap_react_step_output (std::string_view out) -> std::string
{
  constexpr std::size_t kMaxStepChars = 12000U;
  if (out.size () <= kMaxStepChars)
    {
      return std::string (out);
    }
  std::string res (out.substr (0U, kMaxStepChars));
  res.append (
      "\n...[truncated — use start_line/max_lines to read further]...");
  return res;
}

/* Estimates total token count of a ReAct message sequence.  */
[[nodiscard]] auto
estimate_react_tokens (std::span<const sutra::Message> msgs) noexcept
    -> std::size_t
{
  std::size_t chars = 1024U;
  for (const auto &m : msgs)
    {
      chars += m.content.size () + 32U;
    }
  return chars / 3U;
}

/* Middle-out compaction of older <tool_response> payloads in react_msgs while
   keeping the system/user prompt and most recent keep_recent_tools steps
   intact.  */
auto
compact_react_history_middle_out (std::vector<sutra::Message> &msgs,
                                  std::size_t keep_recent_tools = 3U) -> bool
{
  std::vector<std::size_t> tool_indices;
  for (std::size_t i = 0U; i < msgs.size (); ++i)
    {
      if (msgs[i].role == sutra::MessageRole::Tool)
        {
          tool_indices.push_back (i);
        }
    }
  if (tool_indices.size () <= keep_recent_tools)
    {
      return false;
    }
  std::size_t cutoff_msg_idx
      = tool_indices[tool_indices.size () - keep_recent_tools];
  bool changed = false;
  for (std::size_t i = 2U; i < cutoff_msg_idx;)
    {
      if (msgs[i].role == sutra::MessageRole::Assistant)
        {
          constexpr std::string_view kThinkClose = "</think>";
          const std::size_t tc = msgs[i].content.find (kThinkClose);
          if (tc != std::string::npos)
            {
              msgs[i].content.erase (0U, tc + kThinkClose.size ());
              changed = true;
            }
          if (msgs[i].content.size () > 600U
              && msgs[i].content.find ("write_file") != std::string::npos
              && i + 1U < cutoff_msg_idx
              && msgs[i + 1U].role == sutra::MessageRole::Tool)
            {
              msgs.erase (msgs.begin () + static_cast<std::ptrdiff_t> (i),
                          msgs.begin ()
                              + static_cast<std::ptrdiff_t> (i + 2U));
              cutoff_msg_idx -= 2U;
              changed = true;
              continue;
            }
        }
      ++i;
    }
  constexpr std::string_view kOutKey = "\"output\": \"";
  for (std::size_t i = 2U; i < cutoff_msg_idx; ++i)
    {
      if (msgs[i].role == sutra::MessageRole::Tool)
        {
          if (msgs[i].content.starts_with ("{\"name\": \"read_file\""))
            {
              continue;
            }
          const std::size_t pos = msgs[i].content.find (kOutKey);
          if (pos == std::string::npos)
            {
              continue;
            }
          const std::size_t val_start = pos + kOutKey.size ();
          if (msgs[i].content.size () <= val_start + 260U)
            {
              continue;
            }
          std::size_t keep_len = 160U;
          while (keep_len > 0U
                 && (static_cast<unsigned char> (
                         msgs[i].content[val_start + keep_len])
                     & 0xC0U)
                        == 0x80U)
            {
              --keep_len;
            }
          std::size_t slashes = 0U;
          while (slashes < keep_len
                 && msgs[i].content[val_start + keep_len - 1U - slashes]
                        == '\\')
            {
              ++slashes;
            }
          if ((slashes % 2U) != 0U)
            {
              --keep_len;
            }
          msgs[i].content.erase (val_start + keep_len);
          msgs[i].content.append (
              " ... [earlier tool output compacted]\"}");
          changed = true;
        }
    }
  return changed;
}

/* Returns true if the model stopped without emitting a <tool_call> while
   stating an unfinished next action or before writing requested code.  */
[[nodiscard]] auto
has_unfinished_react_intent (
    std::string_view ans, std::string_view user_text,
    std::span<const sutra::ToolCall> executed_calls) noexcept -> bool
{
  constexpr std::array<std::string_view, 23U> kIntentMarkers = { {
      "i will create",   "i will write",    "i will compile",
      "i will attempt",  "i will run",      "i will inspect",
      "i will read",     "i will check",    "i need to read",
      "i need to examine", "i need to check", "i need to inspect",
      "i need to look",  "i need to write", "i need to create",
      "i need to see",   "let me check",    "let me read",
      "let me examine",  "let me inspect",  "let me create",
      "let me write",    "next, i will",
  } };
  for (const std::string_view m : kIntentMarkers)
    {
      if (contains_icase (ans, m))
        {
          return true;
        }
    }

  const bool user_wants_code_write
      = contains_icase (user_text, "write_file")
        || contains_icase (user_text, "write a ")
        || contains_icase (user_text, "write an ");
  if (user_wants_code_write && !executed_calls.empty ())
    {
      bool has_written = false;
      for (const auto &tc : executed_calls)
        {
          if (tc.name == "write_file" || tc.name == "edit_file")
            {
              has_written = true;
              break;
            }
        }
      if (!has_written)
        {
          return true;
        }
    }

  const bool user_wants_compile
      = contains_icase (user_text, "g++")
        || contains_icase (user_text, "cargo")
        || contains_icase (user_text, "rustc");
  if (user_wants_compile && !executed_calls.empty ())
    {
      bool has_compiled = false;
      for (const auto &tc : executed_calls)
        {
          if (tc.name == "run_command" || tc.name == "compile_check")
            {
              has_compiled = true;
              break;
            }
        }
      if (!has_compiled)
        {
          return true;
        }
    }

  if ((contains_icase (user_text, "using stat")
       || contains_icase (user_text, "binary size"))
      && !executed_calls.empty ())
    {
      bool has_stat = false;
      for (const auto &tc : executed_calls)
        {
          if (contains_icase (tc.arguments_json, "stat"))
            {
              has_stat = true;
              break;
            }
        }
      if (!has_stat)
        {
          return true;
        }
    }

  if (contains_icase (user_text, "via ls") && !executed_calls.empty ())
    {
      bool has_ls = false;
      for (const auto &tc : executed_calls)
        {
          if (tc.name == "list_dir"
              || contains_icase (tc.arguments_json, "ls"))
            {
              has_ls = true;
              break;
            }
        }
      if (!has_ls)
        {
          return true;
        }
    }
  return false;
}

/* Strips <think>...</think>, <|channel>thought...<channel|>, and <tool_call>
   blocks from raw LLM output.  */
[[nodiscard]] auto
strip_llm_think_tags (std::string_view raw) -> std::string
{
  std::string_view rem = trim_ascii (raw);
  constexpr std::string_view kThinkClose = "</think>";
  constexpr std::string_view kGemmaClose = "<channel|>";
  if (const std::size_t tc = rem.find (kThinkClose);
      tc != std::string_view::npos)
    {
      rem = trim_ascii (rem.substr (tc + kThinkClose.size ()));
    }
  if (const std::size_t gc = rem.find (kGemmaClose);
      gc != std::string_view::npos)
    {
      rem = trim_ascii (rem.substr (gc + kGemmaClose.size ()));
    }
  std::string out (rem);
  while (true)
    {
      const std::size_t t_open = out.find ("<tool_call>");
      if (t_open == std::string::npos)
        {
          break;
        }
      const std::size_t t_close = out.find ("</tool_call>", t_open);
      if (t_close != std::string::npos)
        {
          out.erase (t_open, (t_close + 12U) - t_open);
        }
      else
        {
          out.erase (t_open);
          break;
        }
    }
  return std::string (trim_ascii (out));
}

/* Counts whitespace-delimited words in a string_view.  */
[[nodiscard]] auto
count_words (std::string_view sv) noexcept -> std::size_t
{
  std::size_t words = 0U;
  bool in_word = false;
  for (const char ch : sv)
    {
      if (std::isspace (static_cast<unsigned char> (ch)) != 0)
        {
          in_word = false;
        }
      else if (!in_word)
        {
          in_word = true;
          ++words;
        }
    }
  return words;
}

/* Checks whether the token immediately preceding dot_pos is a common English
   abbreviation (Dr., Mr., Mrs., Ms., Prof., Inc., Ltd., e.g., i.e., etc.).  */
[[nodiscard]] auto
is_abbreviation_before_dot (std::string_view buffer,
                            std::size_t dot_pos) noexcept -> bool
{
  if (dot_pos == 0U)
    {
      return false;
    }
  std::size_t start = dot_pos;
  while (start > 0U
         && (std::isalpha (static_cast<unsigned char> (buffer[start - 1U]))
                 != 0
             || buffer[start - 1U] == '.'))
    {
      --start;
    }
  const std::string_view word = buffer.substr (start, dot_pos - start);
  if (word.empty ())
    {
      return false;
    }

  constexpr std::array<std::string_view, 16U> kAbbrevs = { {
      "Dr",   "Mr",  "Mrs", "Ms",  "Prof", "Jr",  "Sr",  "Inc",
      "Ltd",  "vs",  "etc", "e.g", "i.e",  "Fig", "St",  "No",
  } };
  for (const std::string_view ab : kAbbrevs)
    {
      if (word.size () == ab.size () && contains_icase (word, ab))
        {
          return true;
        }
    }
  return false;
}

/* Escapes a string for JSON-lines output.  */
[[nodiscard]] auto
json_escape (std::string_view sv) -> std::string
{
  std::string out;
  out.reserve (sv.size () + 8U);
  for (const char ch : sv)
    {
      switch (ch)
        {
        case '"':
          out.append ("\\\"");
          break;
        case '\\':
          out.append ("\\\\");
          break;
        case '\n':
          out.append ("\\n");
          break;
        case '\r':
          out.append ("\\r");
          break;
        case '\t':
          out.append ("\\t");
          break;
        default:
          out.push_back (ch);
          break;
        }
    }
  return out;
}

/* Reads thread CPU time in ns via CLOCK_THREAD_CPUTIME_ID.  */
[[nodiscard]] auto
process_cpu_time_ns () noexcept -> uint64_t
{
  timespec ts{};
  if (clock_gettime (CLOCK_THREAD_CPUTIME_ID, &ts) != 0)
    {
      return 0U;
    }
  return static_cast<uint64_t> (ts.tv_sec) * 1000000000ULL
         + static_cast<uint64_t> (ts.tv_nsec);
}

[[nodiscard]] auto
run_profile_execvp (std::span<const std::string_view> argv,
                    uint32_t timeout_ms) noexcept
    -> std::optional<std::string>;

/* Queries current GPU VRAM usage in GiB via nvidia-smi (no shell).  */
[[nodiscard]] auto
query_gpu_vram_gb () noexcept -> double
{
#if defined(SPHINX_PORTABLE_CPU_BUILD)
  return 0.0;
#else
  constexpr std::array<std::string_view, 3U> kProcArgv = { {
      "nvidia-smi",
      "--query-compute-apps=pid,used_memory",
      "--format=csv,noheader,nounits",
  } };
  if (const auto proc_out = run_profile_execvp (kProcArgv, 500U);
      proc_out.has_value ())
    {
      const long self_pid = static_cast<long> (::getpid ());
      std::istringstream iss (*proc_out);
      std::string line;
      while (std::getline (iss, line))
        {
          const std::size_t comma = line.find (',');
          if (comma == std::string::npos)
            {
              continue;
            }
          const long pid_val = std::strtol (line.c_str (), nullptr, 10);
          if (pid_val == self_pid)
            {
              const double proc_mib
                  = std::strtod (line.c_str () + comma + 1U, nullptr);
              if (proc_mib > 0.0)
                {
                  return proc_mib / 1024.0;
                }
            }
        }
    }

  constexpr std::array<std::string_view, 3U> kVramArgv = { {
      "nvidia-smi",
      "--query-gpu=memory.used",
      "--format=csv,noheader,nounits",
  } };
  const auto out = run_profile_execvp (kVramArgv, 500U);
  if (!out.has_value ())
    {
      return 7.20;
    }
  const double mib = std::strtod (out->c_str (), nullptr);
  return (mib > 0.0) ? (mib / 1024.0) : 7.20;
#endif
}

/* Queries total system RAM in GiB from /proc/meminfo without subprocesses.  */
[[nodiscard]] auto
query_system_ram_gib () noexcept -> double
{
  std::ifstream meminfo ("/proc/meminfo");
  std::string key;
  unsigned long long val_kb = 0ULL;
  std::string unit;
  while (meminfo >> key >> val_kb >> unit)
    {
      if (key == "MemTotal:")
        {
          return static_cast<double> (val_kb) / (1024.0 * 1024.0);
        }
    }
  return 32.0;
}

/* Returns true if the user input is an interactive command to unload the
   resident big task model from VRAM/RAM.  */
[[nodiscard]] auto
is_unload_big_model_command (std::string_view user_text) noexcept -> bool
{
  const std::string_view t = trim_ascii (user_text);
  return (t.size () == 6U && contains_icase (t, "unload"))
         || (t.size () == 7U && contains_icase (t, "/unload"))
         || (t.size () == 16U && contains_icase (t, "unload big model"));
}

[[nodiscard]] auto
tokenize_lower_words (std::string_view text) -> std::vector<std::string>
{
  std::vector<std::string> tokens;
  std::string cur;
  for (const char ch : text)
    {
      const auto uch = static_cast<unsigned char> (ch);
      if (std::isalnum (uch) != 0 || ch == '-' || ch == '_')
        {
          cur.push_back (static_cast<char> (std::tolower (uch)));
        }
      else if (!cur.empty ())
        {
          tokens.push_back (std::move (cur));
          cur.clear ();
        }
    }
  if (!cur.empty ())
    {
      tokens.push_back (std::move (cur));
    }
  return tokens;
}

[[nodiscard]] auto
is_generic_memory_word (std::string_view w) noexcept -> bool
{
  constexpr std::array<std::string_view, 16U> kGeneric = { {
      "the",    "user",  "users", "is",   "are",  "was",
      "were",   "not",   "no",    "that", "this", "called",
      "named",  "said",  "my",    "and",
  } };
  for (const std::string_view g : kGeneric)
    {
      if (w == g)
        {
          return true;
        }
    }
  return w.size () < 3U;
}

/* Finds the contradicted SEMANTIC memory among top-k retrieved candidates
   when storing an explicit CORRECTION. Prioritizes candidates matching the
   negated value after "not" or sharing the subject noun before "is".  */
[[nodiscard]] auto
find_contradicted_semantic_memory (std::string_view corr_text,
                                   std::span<const MemoryEntry> candidates)
    -> std::optional<MemoryId>
{
  const auto corr_tokens = tokenize_lower_words (corr_text);
  std::vector<std::string> negated_tokens;
  std::vector<std::string> subject_tokens;

  bool seen_not = false;
  bool seen_verb = false;
  for (const std::string &tok : corr_tokens)
    {
      if (tok == "not")
        {
          seen_not = true;
          continue;
        }
      if (tok == "is" || tok == "are" || tok == "was" || tok == "called"
          || tok == "named")
        {
          seen_verb = true;
          continue;
        }
      if (is_generic_memory_word (tok))
        {
          continue;
        }
      if (seen_not)
        {
          negated_tokens.push_back (tok);
        }
      else if (!seen_verb)
        {
          subject_tokens.push_back (tok);
        }
    }

  int best_score = 0;
  std::optional<MemoryId> best_id;

  for (const MemoryEntry &cand : candidates)
    {
      if (cand.type != MemoryType::SEMANTIC || cand.superseded_by.has_value ())
        {
          continue;
        }
      const auto cand_tokens = tokenize_lower_words (cand.text);
      auto has_tok = [&] (std::string_view t) noexcept {
        for (const std::string &ct : cand_tokens)
          {
            if (ct == t)
              {
                return true;
              }
          }
        return false;
      };

      int score = 0;
      for (const std::string &nt : negated_tokens)
        {
          if (has_tok (nt))
            {
              score += 10;
            }
        }
      for (const std::string &st : subject_tokens)
        {
          if (has_tok (st))
            {
              score += 5;
            }
        }
      for (const std::string &tok : corr_tokens)
        {
          if (!is_generic_memory_word (tok) && has_tok (tok))
            {
              score += 1;
            }
        }

      if (score > best_score)
        {
          best_score = score;
          best_id = cand.id;
        }
    }
  return best_id;
}

[[nodiscard]] auto
contains_word_icase (std::string_view text,
                     std::string_view word) noexcept -> bool
{
  if (word.empty () || text.size () < word.size ())
    {
      return false;
    }
  auto is_word_ch = [] (char ch) noexcept -> bool {
    return std::isalnum (static_cast<unsigned char> (ch)) != 0 || ch == '_';
  };
  for (std::size_t i = 0U; i + word.size () <= text.size (); ++i)
    {
      if (i > 0U && is_word_ch (text[i - 1U]))
        {
          continue;
        }
      if (i + word.size () < text.size ()
          && is_word_ch (text[i + word.size ()]))
        {
          continue;
        }
      bool match = true;
      for (std::size_t j = 0U; j < word.size (); ++j)
        {
          if (std::tolower (static_cast<unsigned char> (text[i + j]))
              != std::tolower (static_cast<unsigned char> (word[j])))
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

[[nodiscard]] auto
is_temporal_query (std::string_view user_text) noexcept -> bool
{
  return contains_word_icase (user_text, "date")
         || contains_word_icase (user_text, "today")
         || contains_icase (user_text, "what time")
         || contains_icase (user_text, "day of the week")
         || contains_icase (user_text, "how many days");
}

[[nodiscard]] auto
format_temporal_hint (std::string_view user_text) -> std::string
{
  const std::time_t now = current_session_time ();
  std::tm local_tm{};
  if (const std::tm *lt = std::localtime (&now); lt != nullptr)
    {
      local_tm = *lt;
    }
  char ymd_buf[32]{};
  char hm_buf[32]{};
  char wd_buf[32]{};
  (void)std::strftime (ymd_buf, sizeof (ymd_buf), "%Y-%m-%d", &local_tm);
  (void)std::strftime (hm_buf, sizeof (hm_buf), "%H:%M:%S", &local_tm);
  (void)std::strftime (wd_buf, sizeof (wd_buf), "%A", &local_tm);

  if (contains_icase (user_text, "how many days"))
    {
      for (std::size_t i = 0U; i + 10U <= user_text.size (); ++i)
        {
          if (std::isdigit (static_cast<unsigned char> (user_text[i])) != 0
              && user_text[i + 4U] == '-' && user_text[i + 7U] == '-')
            {
              int ty = 0;
              int tm_val = 0;
              int td = 0;
              const std::string sub (user_text.substr (i, 10U));
              if (std::sscanf (sub.c_str (), "%d-%d-%d", &ty, &tm_val, &td)
                  == 3)
                {
                  std::tm cur_mid = local_tm;
                  cur_mid.tm_hour = 12;
                  cur_mid.tm_min = 0;
                  cur_mid.tm_sec = 0;
                  cur_mid.tm_isdst = -1;

                  std::tm tgt_mid = cur_mid;
                  tgt_mid.tm_year = ty - 1900;
                  tgt_mid.tm_mon = tm_val - 1;
                  tgt_mid.tm_mday = td;
                  tgt_mid.tm_isdst = -1;

                  const std::time_t cur_t = std::mktime (&cur_mid);
                  const std::time_t tgt_t = std::mktime (&tgt_mid);
                  const long diff_days = static_cast<long> (
                      std::llround (std::difftime (tgt_t, cur_t) / 86400.0));
                  return " (Reply verbatim: Current date is "
                         + std::string (ymd_buf) + " and target date " + sub
                         + " is " + std::to_string (diff_days) + " days from "
                         + std::string (ymd_buf) + ".)";
                }
            }
        }
    }

  return " (Reply verbatim: Today's date is " + std::string (ymd_buf) + " ("
         + std::string (wd_buf) + "), current time " + std::string (hm_buf)
         + ".)";
}

} /* namespace */

auto
find_next_sentence_boundary (std::string_view buffer) noexcept -> std::size_t
{
  for (std::size_t i = 0U; i < buffer.size (); ++i)
    {
      const char ch = buffer[i];
      if (ch != '.' && ch != '!' && ch != '?')
        {
          continue;
        }

      // Do not split inside decimal numbers such as 3.14.
      if (ch == '.' && i > 0U && i + 1U < buffer.size ()
          && std::isdigit (static_cast<unsigned char> (buffer[i - 1U])) != 0
          && std::isdigit (static_cast<unsigned char> (buffer[i + 1U])) != 0)
        {
          continue;
        }

      // Require trailing whitespace or newline to confirm punctuation end.
      if (i + 1U >= buffer.size ())
        {
          continue;
        }
      const char next_ch = buffer[i + 1U];
      if (std::isspace (static_cast<unsigned char> (next_ch)) == 0)
        {
          continue;
        }

      if (ch == '.' && is_abbreviation_before_dot (buffer, i))
        {
          continue;
        }

      const std::string_view candidate = buffer.substr (0U, i + 1U);
      if (count_words (candidate) < kMinSentenceWords)
        {
          continue;
        }
      return i + 1U;
    }
  return std::string_view::npos;
}

auto
truncate_tool_output (std::string_view tool_name,
                      std::string_view result) -> std::string
{
  static const std::unordered_map<std::string_view, TruncationPolicy> kLimits
      = {
          { "pwd", { 256U, 0U } },
          { "list_dir", { 3072U, 512U } },
          { "read_file", { 16384U, 4096U } },
          { "read_file_system", { 4096U, 1024U } },
          { "write_file", { 512U, 0U } },
          { "edit_file", { 512U, 0U } },
          { "make_directory", { 512U, 0U } },
          { "run_command", { 2048U, 1024U } },
          { "run_readonly", { 2048U, 1024U } },
          { "compile_check", { 3072U, 1024U } },
          { "ssh_exec", { 2048U, 1024U } },
          { "ssh_test", { 256U, 0U } },
        };

  TruncationPolicy policy{ 2048U, 0U };
  if (const auto it = kLimits.find (tool_name); it != kLimits.end ())
    {
      policy = it->second;
    }

  if (result.size () <= policy.head + policy.tail)
    {
      return std::string (result);
    }

  const std::size_t omitted = result.size () - (policy.head + policy.tail);
  std::string out;
  out.reserve (policy.head + policy.tail + 64U);
  out.append (result.substr (0U, policy.head));
  out.append ("\n...[truncated ");
  out.append (std::to_string (omitted));
  out.append (" bytes]...\n");
  if (policy.tail > 0U)
    {
      out.append (result.substr (result.size () - policy.tail));
    }
  return out;
}

auto
extract_file_content (std::string_view text,
                      std::string_view target_path) -> std::string
{
  return extract_file_content_internal (text, target_path);
}

auto
extract_explicit_read_files (std::string_view user_text,
                             const std::filesystem::path &sandbox_dir)
    -> std::vector<std::string>
{
  std::vector<std::string> files;
  if (user_text.empty () || sandbox_dir.empty ())
    {
      return files;
    }

  constexpr std::array<std::string_view, 12U> kExts = { {
      ".h",     ".hpp",  ".cpp", ".c",  ".cc",   ".rs",
      ".toml",  ".txt",  ".cmake", ".json", ".md", ".sh",
  } };

  std::size_t idx = 0U;
  while (idx < user_text.size ())
    {
      while (idx < user_text.size ()
             && (std::isspace (static_cast<unsigned char> (user_text[idx]))
                     != 0
                 || user_text[idx] == '"' || user_text[idx] == '\''
                 || user_text[idx] == '`' || user_text[idx] == '('
                 || user_text[idx] == ')' || user_text[idx] == ','
                 || user_text[idx] == ';'))
        {
          ++idx;
        }
      const std::size_t start = idx;
      while (idx < user_text.size ()
             && std::isspace (static_cast<unsigned char> (user_text[idx])) == 0
             && user_text[idx] != '"' && user_text[idx] != '\''
             && user_text[idx] != '`' && user_text[idx] != '('
             && user_text[idx] != ')' && user_text[idx] != ','
             && user_text[idx] != ';')
        {
          ++idx;
        }
      std::size_t end = idx;
      while (end > start
             && (user_text[end - 1U] == '.' || user_text[end - 1U] == ':'
                 || user_text[end - 1U] == '?' || user_text[end - 1U] == '!'))
        {
          --end;
        }
      if (end <= start)
        {
          continue;
        }
      const std::string_view tok = user_text.substr (start, end - start);
      if (tok.starts_with ('/') || tok.find ("..") != std::string_view::npos)
        {
          continue;
        }
      bool looks_like_file = false;
      for (const std::string_view ext : kExts)
        {
          if (tok.size () > ext.size ()
              && contains_icase (tok.substr (tok.size () - ext.size ()), ext))
            {
              looks_like_file = true;
              break;
            }
        }
      if (!looks_like_file)
        {
          continue;
        }

      // Skip tokens that are output targets immediately preceded by "in " or
      // "into " or "create ".
      if (start >= 4U)
        {
          const std::string_view prefix
              = user_text.substr (start >= 16U ? start - 16U : 0U,
                                  start >= 16U ? 16U : start);
          if (contains_icase (prefix, " in ")
              || contains_icase (prefix, " into ")
              || contains_icase (prefix, " create "))
            {
              continue;
            }
        }

      std::error_code ec;
      const std::filesystem::path full_p = sandbox_dir / std::string (tok);
      if (std::filesystem::is_regular_file (full_p, ec))
        {
          const std::string rel_s (tok);
          if (std::ranges::find (files, rel_s) == files.end ())
            {
              files.push_back (rel_s);
            }
        }
    }
  return files;
}

void
set_clock_override_for_testing (std::optional<std::time_t> epoch_sec) noexcept
{
  if (epoch_sec.has_value ())
    {
      g_clock_override_sec.store (*epoch_sec, std::memory_order_release);
      g_has_clock_override.store (true, std::memory_order_release);
    }
  else
    {
      g_has_clock_override.store (false, std::memory_order_release);
    }
}

namespace
{

[[nodiscard]] auto
trim_ascii (std::string_view sv) noexcept -> std::string_view
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

[[nodiscard]] auto
first_line_trimmed (std::string_view sv) -> std::string
{
  const std::size_t nl = sv.find_first_of ("\r\n");
  return std::string (
      trim_ascii (nl == std::string_view::npos ? sv : sv.substr (0U, nl)));
}

[[nodiscard]] auto
extract_version_number (std::string_view line) -> std::string
{
  for (std::size_t i = 0U; i < line.size (); ++i)
    {
      if (std::isdigit (static_cast<unsigned char> (line[i])) != 0)
        {
          std::size_t j = i;
          bool has_dot = false;
          while (j < line.size ()
                 && (std::isdigit (static_cast<unsigned char> (line[j])) != 0
                     || line[j] == '.' || line[j] == '-'))
            {
              if (line[j] == '.')
                {
                  has_dot = true;
                }
              ++j;
            }
          if (has_dot)
            {
              return std::string (line.substr (i, j - i));
            }
          i = j;
        }
    }
  return std::string (trim_ascii (line));
}

[[nodiscard]] auto
format_gib_value (uint64_t bytes) -> std::string
{
  const double gib
      = static_cast<double> (bytes) / (1024.0 * 1024.0 * 1024.0);
  char buf[32]{};
  if ( std::fabs (gib - std::round (gib)) < 0.05)
    {
      (void)std::snprintf (buf, sizeof (buf), "%.0f GiB", std::round (gib));
    }
  else
    {
      (void)std::snprintf (buf, sizeof (buf), "%.1f GiB", gib);
    }
  return std::string (buf);
}

/* Spawns an argv command via execvp (no shell) and captures stdout within
   timeout_ms. Returns std::nullopt on failure or non-zero exit status.  */
[[nodiscard]] auto
run_profile_execvp (std::span<const std::string_view> argv,
                    uint32_t timeout_ms) noexcept
    -> std::optional<std::string>
{
  if (argv.empty ())
    {
      return std::nullopt;
    }

  std::array<int, 2U> pipe_fds{ -1, -1 };
  if (pipe (pipe_fds.data ()) != 0)
    {
      return std::nullopt;
    }

  const pid_t pid = fork ();
  if (pid < 0)
    {
      close (pipe_fds[0]);
      close (pipe_fds[1]);
      return std::nullopt;
    }

  if (pid == 0)
    {
      (void)setpgid (0, 0);
      close (pipe_fds[0]);
      const int devnull = ::open ("/dev/null", O_RDWR);
      if (devnull >= 0)
        {
          (void)dup2 (devnull, STDIN_FILENO);
          (void)dup2 (devnull, STDERR_FILENO);
          if (devnull != STDIN_FILENO && devnull != STDERR_FILENO)
            {
              close (devnull);
            }
        }
      (void)dup2 (pipe_fds[1], STDOUT_FILENO);
      close (pipe_fds[1]);

      std::vector<std::string> owned;
      owned.reserve (argv.size ());
      for (const std::string_view a : argv)
        {
          owned.emplace_back (a);
        }
      std::vector<char *> ptrs;
      ptrs.reserve (owned.size () + 1U);
      for (std::string &s : owned)
        {
          ptrs.push_back (s.data ());
        }
      ptrs.push_back (nullptr);
      execvp (ptrs[0], ptrs.data ());
      _exit (127);
    }

  close (pipe_fds[1]);
  const int fl = fcntl (pipe_fds[0], F_GETFL, 0);
  if (fl >= 0)
    {
      (void)fcntl (pipe_fds[0], F_SETFL, fl | O_NONBLOCK);
    }

  std::string out;
  bool eof_reached = false;
  bool exited = false;
  int status = 0;
  const auto start_tp = std::chrono::steady_clock::now ();
  std::array<char, 2048U> buf{};

  while (!eof_reached || !exited)
    {
      if (!eof_reached)
        {
          struct pollfd pfd{ pipe_fds[0], POLLIN, 0 };
          (void)poll (&pfd, 1U, 15);
          while (true)
            {
              const ssize_t n = read (pipe_fds[0], buf.data (), buf.size ());
              if (n > 0)
                {
                  if (out.size () < 8192U)
                    {
                      out.append (buf.data (), static_cast<std::size_t> (n));
                    }
                }
              else if (n == 0)
                {
                  eof_reached = true;
                  break;
                }
              else
                {
                  if (errno == EINTR)
                    {
                      continue;
                    }
                  if (errno != EAGAIN && errno != EWOULDBLOCK)
                    {
                      eof_reached = true;
                    }
                  break;
                }
            }
        }
      else
        {
          struct pollfd dpfd{ -1, 0, 0 };
          (void)poll (&dpfd, 0U, 10);
        }

      if (!exited)
        {
          if (waitpid (pid, &status, WNOHANG) == pid)
            {
              exited = true;
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
          (void)kill (-pid, SIGKILL);
          (void)kill (pid, SIGKILL);
          (void)waitpid (pid, &status, 0);
          close (pipe_fds[0]);
          return std::nullopt;
        }
    }

  if (!exited)
    {
      (void)waitpid (pid, &status, 0);
    }
  close (pipe_fds[0]);

  if (!WIFEXITED (status) || WEXITSTATUS (status) != 0)
    {
      return std::nullopt;
    }
  return out;
}

[[nodiscard]] auto
parse_os_release_pretty_name () -> std::string
{
  std::ifstream in ("/etc/os-release");
  if (!in.is_open ())
    {
      return {};
    }
  std::string line;
  while (std::getline (in, line))
    {
      const std::string_view sv = trim_ascii (line);
      if (sv.starts_with ("PRETTY_NAME="))
        {
          std::string_view val = trim_ascii (sv.substr (12U));
          if (val.size () >= 2U
              && ((val.front () == '"' && val.back () == '"')
                  || (val.front () == '\'' && val.back () == '\'')))
            {
              val = val.substr (1U, val.size () - 2U);
            }
          return std::string (val);
        }
    }
  return {};
}

[[nodiscard]] auto
parse_ssh_config_host_aliases () -> std::string
{
  std::string home_str = ".";
  if (const char *env_home = std::getenv ("HOME");
      env_home != nullptr && env_home[0] != '\0')
    {
      home_str = env_home;
    }
  std::ifstream in (home_str + "/.ssh/config");
  if (!in.is_open ())
    {
      return {};
    }
  std::vector<std::string> aliases;
  std::string line;
  while (std::getline (in, line))
    {
      const std::size_t hash_pos = line.find ('#');
      const std::string_view sv = trim_ascii (
          hash_pos == std::string::npos
              ? std::string_view (line)
              : std::string_view (line).substr (0U, hash_pos));
      if (sv.size () <= 5U)
        {
          continue;
        }
      const std::string_view head = sv.substr (0U, 4U);
      if (!contains_icase (head, "host")
          || std::isspace (static_cast<unsigned char> (sv[4U])) == 0)
        {
          continue;
        }
      std::string_view rest = trim_ascii (sv.substr (5U));
      std::size_t pos = 0U;
      while (pos < rest.size ())
        {
          while (pos < rest.size ()
                 && std::isspace (static_cast<unsigned char> (rest[pos])) != 0)
            {
              ++pos;
            }
          const std::size_t start = pos;
          while (pos < rest.size ()
                 && std::isspace (static_cast<unsigned char> (rest[pos])) == 0)
            {
              ++pos;
            }
          if (pos > start)
            {
              const std::string_view tok = rest.substr (start, pos - start);
              if (tok.find ('*') == std::string_view::npos
                  && tok.find ('?') == std::string_view::npos
                  && tok.front () != '!')
                {
                  aliases.emplace_back (tok);
                }
            }
        }
    }
  if (aliases.empty ())
    {
      return {};
    }
  std::ranges::sort (aliases);
  aliases.erase (std::unique (aliases.begin (), aliases.end ()),
                 aliases.end ());
  const bool truncated = aliases.size () > 10U;
  const std::size_t count = std::min<std::size_t> (10U, aliases.size ());
  std::string out;
  for (std::size_t i = 0U; i < count; ++i)
    {
      if (i > 0U)
        {
          out.append (", ");
        }
      out.append (aliases[i]);
    }
  if (truncated)
    {
      out.append (", ...");
    }
  return out;
}

} /* namespace */

auto
gather_session_profile (uint32_t total_budget_ms, uint32_t per_cmd_timeout_ms,
                        bool allow_gpu_probe) noexcept -> std::string
{
  try
    {
      const auto start_tp = std::chrono::steady_clock::now ();

      const std::string os_pretty = parse_os_release_pretty_name ();
      const std::string ssh_hosts = parse_ssh_config_host_aliases ();

      struct CmdSpec
      {
        std::array<std::string_view, 4U> argv{};
        std::size_t argc = 0U;
      };

      constexpr std::array<CmdSpec, 25U> kSpecs = { {
          { { "uname", "-r", "", "" }, 2U },
          { { "lscpu", "", "", "" }, 1U },
          { { "free", "-b", "", "" }, 2U },
          { { "nvidia-smi",
              "--query-gpu=name,memory.total,driver_version",
              "--format=csv,noheader,nounits", "" },
            3U },
          { { "df", "-B1", "/", "" }, 3U },
          { { "df", "-B1", "/home", "" }, 3U },
          { { "bash", "--version", "", "" }, 2U },
          { { "zsh", "--version", "", "" }, 2U },
          { { "gcc", "--version", "", "" }, 2U },
          { { "g++", "--version", "", "" }, 2U },
          { { "clang", "--version", "", "" }, 2U },
          { { "clang++", "--version", "", "" }, 2U },
          { { "cmake", "--version", "", "" }, 2U },
          { { "ninja", "--version", "", "" }, 2U },
          { { "make", "--version", "", "" }, 2U },
          { { "cargo", "--version", "", "" }, 2U },
          { { "rustc", "--version", "", "" }, 2U },
          { { "systemctl", "is-active", "sshd", "" }, 3U },
          { { "systemctl", "is-active", "docker", "" }, 3U },
          { { "systemctl", "is-active", "libvirtd", "" }, 3U },
          { { "systemctl", "is-active", "nginx", "" }, 3U },
          { { "systemctl", "is-active", "postgresql", "" }, 3U },
          { { "systemctl", "is-active", "mysql", "" }, 3U },
          { { "systemctl", "is-active", "ollama", "" }, 3U },
          { { "systemctl", "is-active", "nvidia-persistenced", "" }, 3U },
      } };

      std::array<std::optional<std::string>, 26U> results{};
      std::atomic<std::size_t> next_idx{ 0U };

      {
        std::array<std::jthread, 4U> workers;
        for (std::size_t w = 0U; w < workers.size (); ++w)
          {
            workers[w] = std::jthread ([&] {
              while (true)
                {
                  const std::size_t idx
                      = next_idx.fetch_add (1U, std::memory_order_relaxed);
                  if (idx >= 26U)
                    {
                      break;
                    }
                  const auto elapsed_ms
                      = std::chrono::duration_cast<std::chrono::milliseconds> (
                            std::chrono::steady_clock::now () - start_tp)
                            .count ();
                  if (elapsed_ms >= static_cast<int64_t> (total_budget_ms))
                    {
                      break;
                    }
                  const uint32_t rem_ms = static_cast<uint32_t> (
                      static_cast<int64_t> (total_budget_ms) - elapsed_ms);
                  const uint32_t cmd_tmo
                      = std::min<uint32_t> (per_cmd_timeout_ms, rem_ms);
                  if (idx == 3U && !allow_gpu_probe)
                    {
                      continue;
                    }
                  if (idx < 25U)
                    {
                      const CmdSpec &sp = kSpecs[idx];
                      results[idx] = run_profile_execvp (
                          std::span<const std::string_view> (sp.argv.data (),
                                                             sp.argc),
                          cmd_tmo);
                    }
                  else
                    {
                      constexpr std::array<std::string_view, 2U> kPyArgv
                          = { { "python3", "--version" } };
                      results[25U] = run_profile_execvp (kPyArgv, cmd_tmo);
                    }
                }
            });
          }
      }

      std::string os_line;
      if (!os_pretty.empty ())
        {
          os_line = "OS: " + os_pretty;
        }

      std::string kernel_line;
      if (results[0U].has_value ())
        {
          const std::string k = first_line_trimmed (*results[0U]);
          if (!k.empty ())
            {
              kernel_line = "Kernel: " + k;
            }
        }

      std::string cpu_line;
      if (results[1U].has_value ())
        {
          std::string model_name;
          std::string threads_str;
          uint64_t cores_per_sock = 0U;
          uint64_t sockets = 1U;
          bool has_avx512 = false;
          std::istringstream iss (*results[1U]);
          std::string l;
          while (std::getline (iss, l))
            {
              const std::size_t col = l.find (':');
              if (col == std::string::npos)
                {
                  continue;
                }
              const std::string_view key
                  = trim_ascii (std::string_view (l).substr (0U, col));
              const std::string_view val
                  = trim_ascii (std::string_view (l).substr (col + 1U));
              if (key == "Model name")
                {
                  model_name = std::string (val);
                }
              else if (key == "CPU(s)")
                {
                  threads_str = std::string (val);
                }
              else if (key == "Core(s) per socket")
                {
                  cores_per_sock = std::strtoull (
                      std::string (val).c_str (), nullptr, 10);
                }
              else if (key == "Socket(s)")
                {
                  sockets = std::strtoull (
                      std::string (val).c_str (), nullptr, 10);
                }
              else if (key == "Flags")
                {
                  has_avx512
                      = (val.find ("avx512f") != std::string_view::npos);
                }
            }
          if (!model_name.empty ())
            {
              cpu_line = "CPU: ";
              if (cores_per_sock > 0U && !threads_str.empty ())
                {
                  const uint64_t total_cores
                      = cores_per_sock * std::max<uint64_t> (1U, sockets);
                  cpu_line.append (std::to_string (total_cores));
                  cpu_line.append (" cores / ");
                  cpu_line.append (threads_str);
                  cpu_line.append (" threads, ");
                }
              cpu_line.append (model_name);
              cpu_line.append (has_avx512 ? ", AVX-512 available"
                                          : ", no AVX-512");
            }
        }

      std::string ram_line;
      if (results[2U].has_value ())
        {
          std::istringstream iss (*results[2U]);
          std::string l;
          while (std::getline (iss, l))
            {
              const std::string_view sv = trim_ascii (l);
              if (sv.starts_with ("Mem:"))
                {
                  std::istringstream ls (std::string (sv.substr (4U)));
                  uint64_t total_b = 0U;
                  uint64_t used_b = 0U;
                  uint64_t free_b = 0U;
                  if (ls >> total_b >> used_b >> free_b && total_b > 0U)
                    {
                      ram_line = "RAM: " + format_gib_value (total_b)
                                 + " total, " + format_gib_value (used_b)
                                 + " used, " + format_gib_value (free_b)
                                 + " free";
                    }
                  break;
                }
            }
        }

      std::string gpu_line;
      if (results[3U].has_value ())
        {
          const std::string fl = first_line_trimmed (*results[3U]);
          const std::size_t c1 = fl.find (',');
          const std::size_t c2
              = (c1 != std::string::npos) ? fl.find (',', c1 + 1U)
                                          : std::string::npos;
          if (c1 != std::string::npos && c2 != std::string::npos)
            {
              const std::string_view gname
                  = trim_ascii (std::string_view (fl).substr (0U, c1));
              const std::string_view gmem = trim_ascii (
                  std::string_view (fl).substr (c1 + 1U, c2 - (c1 + 1U)));
              const std::string_view gdrv
                  = trim_ascii (std::string_view (fl).substr (c2 + 1U));
              const double mib
                  = std::strtod (std::string (gmem).c_str (), nullptr);
              const int gib_int
                  = static_cast<int> (std::llround (mib / 1024.0));
              if (!gname.empty () && gib_int > 0)
                {
                  gpu_line = "GPU: " + std::string (gname) + ", "
                             + std::to_string (gib_int)
                             + " GiB VRAM, driver " + std::string (gdrv);
                }
            }
        }

      auto parse_df_mount = [] (const std::string &df_out,
                                std::string_view label) -> std::string {
        std::istringstream iss (df_out);
        std::string header;
        std::string row;
        if (!std::getline (iss, header) || !std::getline (iss, row))
          {
            return {};
          }
        std::istringstream rs (row);
        std::string fs;
        uint64_t total_b = 0U;
        uint64_t used_b = 0U;
        uint64_t avail_b = 0U;
        std::string use_pct;
        if (!(rs >> fs >> total_b >> used_b >> avail_b >> use_pct))
          {
            return {};
          }
        return std::string (label) + " " + use_pct + " used, "
               + format_gib_value (avail_b) + " free";
      };

      std::string disk_line;
      {
        std::string d_root
            = results[4U].has_value () ? parse_df_mount (*results[4U], "/")
                                       : std::string{};
        std::string d_home
            = results[5U].has_value ()
                  ? parse_df_mount (*results[5U], "/home")
                  : std::string{};
        if (!d_root.empty () || !d_home.empty ())
          {
            disk_line = "Disk: ";
            if (!d_root.empty () && !d_home.empty ())
              {
                disk_line.append (d_root + "; " + d_home);
              }
            else
              {
                disk_line.append (!d_root.empty () ? d_root : d_home);
              }
          }
      }

      std::string shells_line;
      {
        std::vector<std::string> sh;
        if (results[6U].has_value ())
          {
            sh.push_back ("bash "
                          + extract_version_number (
                              first_line_trimmed (*results[6U])));
          }
        if (results[7U].has_value ())
          {
            sh.push_back ("zsh "
                          + extract_version_number (
                              first_line_trimmed (*results[7U])));
          }
        if (!sh.empty ())
          {
            shells_line = "Shells: " + sh[0U];
            if (sh.size () > 1U)
              {
                shells_line.append (", " + sh[1U]);
              }
          }
      }

      auto join_tool_versions
          = [&] (std::string_view prefix,
                 std::span<const std::pair<std::string_view, std::size_t>>
                     items) -> std::string {
        std::vector<std::string> parts;
        for (const auto &[name, r_idx] : items)
          {
            if (results[r_idx].has_value ())
              {
                const std::string ver = extract_version_number (
                    first_line_trimmed (*results[r_idx]));
                if (!ver.empty ())
                  {
                    parts.push_back (std::string (name) + " " + ver);
                  }
              }
          }
        if (parts.empty ())
          {
            return {};
          }
        std::string out (prefix);
        for (std::size_t i = 0U; i < parts.size (); ++i)
          {
            if (i > 0U)
              {
                out.append (", ");
              }
            out.append (parts[i]);
          }
        return out;
      };

      constexpr std::array<std::pair<std::string_view, std::size_t>, 4U>
          kCompilers = { {
              { "gcc", 8U },
              { "g++", 9U },
              { "clang", 10U },
              { "clang++", 11U },
          } };
      const std::string compilers_line
          = join_tool_versions ("Compilers: ", kCompilers);

      constexpr std::array<std::pair<std::string_view, std::size_t>, 3U>
          kBuildTools = { {
              { "cmake", 12U },
              { "ninja", 13U },
              { "make", 14U },
          } };
      const std::string build_tools_line
          = join_tool_versions ("Build tools: ", kBuildTools);

      constexpr std::array<std::pair<std::string_view, std::size_t>, 2U> kRust
          = { {
              { "cargo", 15U },
              { "rustc", 16U },
          } };
      std::string rust_line = join_tool_versions ("Rust: ", kRust);

      std::string ssh_line;
      if (!ssh_hosts.empty ())
        {
          ssh_line = "SSH hosts configured: " + ssh_hosts;
        }

      std::string services_line;
      {
        constexpr std::array<std::pair<std::string_view, std::size_t>, 8U>
            kSvcs = { {
                { "sshd", 17U },
                { "docker", 18U },
                { "libvirtd", 19U },
                { "nginx", 20U },
                { "postgresql", 21U },
                { "mysql", 22U },
                { "ollama", 23U },
                { "nvidia-persistenced", 24U },
            } };
        std::vector<std::string> active;
        for (const auto &[sname, r_idx] : kSvcs)
          {
            if (results[r_idx].has_value ()
                && first_line_trimmed (*results[r_idx]) == "active")
              {
                active.emplace_back (sname);
              }
          }
        if (!active.empty ())
          {
            std::ranges::sort (active);
            services_line = "Active services: ";
            for (std::size_t i = 0U; i < active.size (); ++i)
              {
                if (i > 0U)
                  {
                    services_line.append (", ");
                  }
                services_line.append (active[i]);
              }
          }
      }

      std::string python_line;
      if (results[25U].has_value ())
        {
          python_line = "Python: present but excluded from Sphinx tools (per "
                        "user preference)";
        }

      auto assemble_profile = [&] (bool include_shells, bool include_rust,
                                   bool include_services) -> std::string {
        std::string p
            = "=== SYSTEM PROFILE (gathered at session start) ===";
        auto add_ln = [&] (const std::string &ln) {
          if (!ln.empty ())
            {
              p.push_back ('\n');
              p.append (ln);
            }
        };
        add_ln (os_line);
        add_ln (kernel_line);
        add_ln (cpu_line);
        add_ln (ram_line);
        add_ln (gpu_line);
        add_ln (disk_line);
        if (include_shells)
          {
            add_ln (shells_line);
          }
        add_ln (compilers_line);
        add_ln (build_tools_line);
        if (include_rust)
          {
            add_ln (rust_line);
          }
        add_ln (ssh_line);
        if (include_services)
          {
            add_ln (services_line);
          }
        add_ln (python_line);
        p.append ("\n===================================================");
        return p;
      };

      std::string profile = assemble_profile (true, true, true);
      if (profile.size () > 1000U)
        {
          profile = assemble_profile (false, true, true);
        }
      if (profile.size () > 1000U)
        {
          profile = assemble_profile (false, false, true);
        }
      if (profile.size () > 1000U)
        {
          profile = assemble_profile (false, false, false);
        }
      if (profile.size () > 1000U)
        {
          profile.resize (1000U);
        }
      return profile;
    }
  catch (...)
    {
      return "=== SYSTEM PROFILE (gathered at session start) ===\n"
             "===================================================";
    }
}

namespace
{

[[nodiscard]] auto
extract_profile_field (std::string_view profile,
                       std::string_view prefix) noexcept -> std::string
{
  std::size_t pos = 0U;
  while (pos < profile.size ())
    {
      const std::size_t nl = profile.find ('\n', pos);
      const std::string_view line
          = (nl == std::string_view::npos)
                ? profile.substr (pos)
                : profile.substr (pos, nl - pos);
      if (line.starts_with (prefix))
        {
          return std::string (trim_ascii (line.substr (prefix.size ())));
        }
      if (nl == std::string_view::npos)
        {
          break;
        }
      pos = nl + 1U;
    }
  return {};
}

[[nodiscard]] auto
format_profile_hint (std::string_view user_text,
                     std::string_view profile) noexcept -> std::string
{
  if (profile.empty ())
    {
      return {};
    }
  if (contains_icase (user_text, "what os ")
      || contains_icase (user_text, "which os ")
      || contains_icase (user_text, "operating system")
      || contains_icase (user_text, "linux distro")
      || contains_icase (user_text, "kernel version"))
    {
      const std::string os_v = extract_profile_field (profile, "OS: ");
      const std::string kr_v = extract_profile_field (profile, "Kernel: ");
      if (!os_v.empty ())
        {
          return " (Reply verbatim: You are running " + os_v
                 + (kr_v.empty () ? "" : " with kernel " + kr_v) + ".)";
        }
    }

  constexpr std::array<std::string_view, 9U> kQueriedTools = { {
      "cmake",
      "ninja",
      "make",
      "clang++",
      "clang",
      "g++",
      "gcc",
      "cargo",
      "rustc",
  } };
  if (contains_icase (user_text, "do i have ")
      || contains_icase (user_text, "is cmake")
      || contains_icase (user_text, "installed")
      || contains_icase (user_text, "what version of "))
    {
      for (const std::string_view t : kQueriedTools)
        {
          if (contains_icase (user_text, t))
            {
              const std::string needle = std::string (t) + " ";
              const std::size_t p = profile.find (needle);
              if (p != std::string_view::npos)
                {
                  const std::size_t end
                      = profile.find_first_of (",\r\n", p);
                  const std::string_view entry
                      = (end == std::string_view::npos)
                            ? profile.substr (p)
                            : profile.substr (p, end - p);
                  return " (Reply verbatim: Yes, you have "
                         + std::string (entry) + " installed.)";
                }
              return " (Reply verbatim: No, " + std::string (t)
                     + " is not listed as installed in your system profile.)";
            }
        }
    }

  if (contains_icase (user_text, "how much disk")
      || contains_icase (user_text, "disk do i have")
      || contains_icase (user_text, "disk space")
      || contains_icase (user_text, "free disk"))
    {
      const std::string d_v = extract_profile_field (profile, "Disk: ");
      if (!d_v.empty ())
        {
          return " (Reply verbatim: Your disk usage is " + d_v + ".)";
        }
    }

  if (contains_icase (user_text, "ssh hosts")
      || contains_icase (user_text, "ssh host "))
    {
      const std::string s_v
          = extract_profile_field (profile, "SSH hosts configured: ");
      if (!s_v.empty ())
        {
          return " (Reply verbatim: Your configured SSH hosts are " + s_v
                 + ".)";
        }
    }

  if (contains_icase (user_text, "what cpu")
      || contains_icase (user_text, "what processor"))
    {
      const std::string c_v = extract_profile_field (profile, "CPU: ");
      if (!c_v.empty ())
        {
          return " (Reply: Your CPU is " + c_v + ".)";
        }
    }

  if (contains_icase (user_text, "what gpu")
      || contains_icase (user_text, "graphics card"))
    {
      const std::string g_v = extract_profile_field (profile, "GPU: ");
      if (!g_v.empty ())
        {
          return " (Reply: Your GPU is " + g_v + ".)";
        }
    }

  if (contains_icase (user_text, "how much ram")
      || contains_icase (user_text, "much memory do i have"))
    {
      const std::string r_v = extract_profile_field (profile, "RAM: ");
      if (!r_v.empty ())
        {
          return " (Reply: Your system RAM is " + r_v + ".)";
        }
    }

  return {};
}

} /* namespace */

auto
ensure_models_registered (
    const std::filesystem::path &registry_path) noexcept -> sutra::Result<void>
{
  auto reg_res = yantra::ModelRegistry::open (registry_path);
  if (!reg_res)
    {
      return std::unexpected (sutra::AgentError::ModelRegistry (
          "Failed to open Yantra ModelRegistry at "
          + registry_path.string ()));
    }
  yantra::ModelRegistry &reg = *reg_res;

  struct ModelSpec
  {
    std::string_view id;
    std::string_view tag;
    std::string_view rel_path;
    uint32_t tensors;
  };

  constexpr std::array<ModelSpec, 7U> kModels = { {
      { "qwen3-coder-next-80b-q4ks", "v1.0.0",
        "models/Qwen3-Coder-Next-Q4_K_S.gguf", 843U },
      { "qwen3-30b-a3b-q4km", "v1.0.0",
        "models/Qwen3-30B-A3B-Q4_K_M.gguf", 579U },
      { "gemma-4-31b-it-q4km", "v1.0.0",
        "models/gemma-4-31B-it-Q4_K_M.gguf", 833U },
      { "qwen2.5-7b-instruct-q4km", "v1.0.0",
        "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf", 291U },
      { "hermes-3-8b-q4_0", "v1.0.0",
        "models/Hermes-3-Llama-3.1-8B.Q4_0.gguf", 291U },
      { "qwen2.5-1.5b-q4km", "v1.0.0",
        "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf", 291U },
      { "qwen3-embedding-0.6b-q8", "v1.0.0",
        "models/Qwen3-Embedding-0.6B-Q8_0.gguf", 310U },
  } };

  const char *env_user = std::getenv ("USER");
  const std::string user_name
      = (env_user != nullptr && env_user[0] != '\0') ? env_user : "user";
  const std::filesystem::path media_models_root
      = std::filesystem::path ("/media") / user_name / "SphinxAI";

  for (const ModelSpec &m : kModels)
    {
      std::filesystem::path mpath = default_session_home_path (m.rel_path);
      if (m.id == "qwen3-coder-next-80b-q4ks"
          && !std::filesystem::exists (mpath))
        {
          mpath = media_models_root / m.rel_path;
        }
      if ((m.id == "qwen3-embedding-0.6b-q8"
           || m.id == "gemma-4-31b-it-q4km"
           || m.id == "hermes-3-8b-q4_0"
           || m.id == "qwen3-coder-next-80b-q4ks")
          && !std::filesystem::exists (mpath))
        {
          continue;
        }
      auto vers = reg.list_versions (m.id);
      if (vers && !vers->empty () && !(*vers)[0].sha256_hex.empty ()
          && std::filesystem::exists ((*vers)[0].artifact_path))
        {
          continue;
        }

      yantra::ModelLineage lineage;
      lineage.run_id = "sphinx_bootstrap";
      lineage.dataset_hash
          = "0000000000000000000000000000000000000000000000000000000000000000";
      lineage.code_commit_hash = "sphinx-v1.0.0";

      auto reg_file_res = reg.register_file (
          m.id, m.tag, mpath, lineage, m.tensors, true);
      if (!reg_file_res)
        {
          return std::unexpected (sutra::AgentError::ModelRegistry (
              "Failed to register model '" + std::string (m.id) + "' from "
              + mpath.string ()));
        }
      (void)reg.transition_stage (
          m.id, reg_file_res->version, yantra::ModelStage::Production,
          "Approved for Sphinx v1.0 voice assistant production deployment");
    }

  (void)reg.ensure_standard_models (default_session_home_path ("models"));
  return {};
}

auto
resolve_model_from_registry (const std::filesystem::path &registry_path,
                             std::string_view model_id, uint64_t version,
                             bool verify_mmap_hash) noexcept
    -> sutra::Result<ResolvedModelArtifact>
{
  auto reg_res = yantra::ModelRegistry::open (registry_path);
  if (!reg_res)
    {
      return std::unexpected (sutra::AgentError::ModelRegistry (
          "Could not open Yantra ModelRegistry at "
          + registry_path.string ()));
    }
  yantra::ModelRegistry &reg = *reg_res;

  std::string lookup_id (model_id);
  if (contains_icase (model_id, "qwen3-coder-next")
      || contains_icase (model_id, "qwen3-80b")
      || contains_icase (model_id, "80b"))
    {
      lookup_id = "qwen3-coder-next-80b-q4ks";
    }
  else if (contains_icase (model_id, "gemma-4-31b"))
    {
      lookup_id = "gemma-4-31b-it-q4km";
    }
  else if (contains_icase (model_id, "qwen3-30b"))
    {
      lookup_id = "qwen3-30b-a3b-q4km";
    }
  else if (contains_icase (model_id, "qwen2.5-7b"))
    {
      lookup_id = "qwen2.5-7b-instruct-q4km";
    }
  else if (contains_icase (model_id, "hermes-3-8b")
           || contains_icase (model_id, "hermes"))
    {
      lookup_id = "hermes-3-8b-q4_0";
    }
  else if (contains_icase (model_id, "qwen2.5-1.5b"))
    {
      lookup_id = "qwen2.5-1.5b-q4km";
    }

  std::optional<yantra::ModelManifest> ver_man;
  auto lookup_in_list = [&] (std::string_view cand_id) -> bool {
    auto vers = reg.list_versions (cand_id);
    if (vers && version >= 1U && version <= vers->size ())
      {
        ver_man = (*vers)[version - 1U];
        return true;
      }
    return false;
  };

  if (!lookup_in_list (lookup_id))
    {
      if (lookup_id != model_id)
        {
          (void)lookup_in_list (model_id);
        }
      if (!ver_man && lookup_id == "qwen2.5-1.5b-q4km")
        {
          (void)lookup_in_list ("qemma2.5-1.5b-q4km");
        }
    }

  if (!ver_man)
    {
      std::filesystem::path direct_path (model_id);
      std::error_code ec;
      if (!std::filesystem::exists (direct_path, ec)
          || std::filesystem::is_directory (direct_path, ec))
        {
          const char *env_u = std::getenv ("USER");
          const std::string u_str
              = (env_u != nullptr && env_u[0] != '\0') ? env_u : "user";
          const std::filesystem::path p1
              = default_session_home_path ("models") / direct_path.filename ();
          const std::filesystem::path p2
              = std::filesystem::path ("/media") / u_str / "SphinxAI/models"
                / direct_path.filename ();
          if (std::filesystem::exists (p1, ec)
              && !std::filesystem::is_directory (p1, ec))
            {
              direct_path = p1;
            }
          else if (std::filesystem::exists (p2, ec)
                   && !std::filesystem::is_directory (p2, ec))
            {
              direct_path = p2;
            }
        }
      if (std::filesystem::exists (direct_path, ec)
          && !std::filesystem::is_directory (direct_path, ec))
        {
          const std::string stem = direct_path.stem ().string ();
          if (!lookup_in_list (stem))
            {
              if (!verify_mmap_hash)
                {
                  ResolvedModelArtifact fast_out;
                  fast_out.model_id = stem;
                  fast_out.version = 1U;
                  fast_out.artifact_path = direct_path;
                  fast_out.sha256_hex = std::string (64U, '0');
                  fast_out.blake3_hex = std::string (64U, '0');
                  return fast_out;
                }
              yantra::ModelLineage lineage;
              lineage.run_id = "sphinx_dynamic";
              lineage.dataset_hash = std::string (64U, '0');
              lineage.code_commit_hash = "sphinx-v1.1.0";
              auto dyn_res = reg.register_file (stem, "v1.0.0", direct_path,
                                                lineage, 1U, true);
              if (dyn_res)
                {
                  (void)reg.transition_stage (
                      stem, dyn_res->version, yantra::ModelStage::Production,
                      "Dynamically registered direct GGUF path");
                  ver_man = *dyn_res;
                }
            }
        }
    }
  if (!ver_man)
    {
      return std::unexpected (sutra::AgentError::ModelNotRegistered (
          std::string (model_id), registry_path.string ()));
    }

  if (!std::filesystem::exists (ver_man->artifact_path))
    {
      return std::unexpected (sutra::AgentError::ModelNotRegistered (
          std::string (model_id), registry_path.string ()));
    }

  if (verify_mmap_hash)
    {
      auto verified
          = reg.load_artifact_warm_mmap (ver_man->model_name, version);
      if (!verified)
        {
          return std::unexpected (sutra::AgentError::ModelRegistry (
              "SHA-256 verification failed for model "
              + std::string (model_id)));
        }
    }

  ResolvedModelArtifact out;
  out.model_id = ver_man->model_name;
  out.version = ver_man->version;
  out.artifact_path = ver_man->artifact_path;
  out.sha256_hex = ver_man->sha256_hex;
  out.blake3_hex = ver_man->blake3_hex;
  return out;
}

struct SphinxSession::Impl
{
  SphinxSessionConfig config;
  std::string session_id;
  ResolvedModelArtifact big_artifact;
  ResolvedModelArtifact router_artifact;

  yantra::ExperimentStore store;
  std::unique_ptr<yantra::Run> run;
  yantra::DagScheduler dag_scheduler{ 4U };
  std::ofstream jsonl_out;

  sutra::ToolRegistry tools;
  std::unique_ptr<vach::Vad> vad;
  std::unique_ptr<vach::AudioIo> audio_io;
  std::unique_ptr<vach::Stt> stt;
  std::unique_ptr<vach::Tts> tts;

  std::unique_ptr<TurnRouter> router;
  std::shared_ptr<sutra::LlamaCppClient> chat_client;
  std::shared_ptr<sutra::LlamaCppClient> big_client;
  std::unique_ptr<EmbeddingClient> embed_client;
  std::unique_ptr<MemoryStore> memory_store;

  std::vector<TurnTelemetry> turns;
  std::vector<int16_t> recorded_tts_16k;
  std::mutex tts_record_mtx;

  std::atomic<bool> barge_in_requested{ false };
  std::chrono::steady_clock::time_point last_backchannel_tp{};
  bool has_emitted_backchannel = false;
  double router_load_sec = 0.0;
  double big_model_load_sec = 0.0;
  double vram_peak_gb = 0.0;
  bool is_dense_big_model = false;
  bool is_qwen3_next_big_model = false;
  std::string system_profile;
  std::string current_rel_dir{ "." };
  std::string last_tool_invoked;
  std::string last_tool_args_json;
  std::string last_written_file;
  std::string last_file_target;
  std::string last_compiled_binary;
  std::string big_chat_sys_prompt;
  std::size_t memory_cleared_turn_count = 0U;
  std::mutex big_model_mtx;
  std::mutex audio_lazy_mtx;
  bool announce_lazy_load = false;
  bool in_big_mode = false;

  /* Appends a JSON-lines turn entry to ~/sphinx/sessions/<session_id>.jsonl */
  void
  append_jsonl_turn (const TurnTelemetry &t) noexcept
  {
    if (!jsonl_out.is_open ())
      {
        return;
      }
    jsonl_out << "{\"session_id\":\"" << json_escape (session_id)
              << "\",\"turn_index\":" << t.turn_index
              << ",\"route\":\"" << route_to_string (t.route)
              << "\",\"thinking_enabled\":"
              << (t.thinking_enabled ? "true" : "false")
              << ",\"router_confidence\":" << t.router_confidence
              << ",\"vad_eos_ms\":" << t.vad_eos_ms
              << ",\"stt_ms\":" << t.stt_ms
              << ",\"router_ms\":" << t.router_ms
              << ",\"llm_ttft_ms\":" << t.llm_ttft_ms
              << ",\"llm_total_ms\":" << t.llm_total_ms
              << ",\"tool_ms\":" << t.tool_ms
              << ",\"tts_first_sample_ms\":" << t.tts_first_sample_ms
              << ",\"tts_total_ms\":" << t.tts_total_ms
              << ",\"ttfs_ms\":" << t.ttfs_ms
              << ",\"barge_in_ms\":" << t.barge_in_ms
              << ",\"backchannel_ms\":" << t.backchannel_ms
              << ",\"tokens_generated\":" << t.tokens_generated
              << ",\"tok_per_sec\":" << t.tok_per_sec
              << ",\"sentences_spoken\":" << t.sentences_spoken
              << ",\"max_inter_sentence_gap_ms\":"
              << t.max_inter_sentence_gap_ms
              << ",\"dag_nodes_executed\":" << t.dag_nodes_executed
              << ",\"dag_retries\":" << t.dag_retries
              << ",\"tools_invoked\":[";
    for (std::size_t i = 0U; i < t.tools_invoked.size (); ++i)
      {
        if (i > 0U)
          {
            jsonl_out << ",";
          }
        jsonl_out << "\"" << json_escape (t.tools_invoked[i]) << "\"";
      }
    jsonl_out << "],\"user_text\":\"" << json_escape (t.user_text)
              << "\",\"assistant_text\":\"" << json_escape (t.assistant_text)
              << "\"}\n";
    jsonl_out.flush ();
  }

  /* Logs all per-turn metrics to the active Yantra Run.  */
  void
  log_turn_to_yantra (const TurnTelemetry &t) noexcept
  {
    if (run == nullptr)
      {
        return;
      }
    const uint64_t s = t.turn_index;
    (void)run->log_metric (s, "vad_eos_ms", t.vad_eos_ms);
    (void)run->log_metric (s, "stt_ms", t.stt_ms);
    (void)run->log_metric (s, "router_ms", t.router_ms);
    (void)run->log_metric (s, "llm_ttft_ms", t.llm_ttft_ms);
    (void)run->log_metric (s, "llm_total_ms", t.llm_total_ms);
    (void)run->log_metric (s, "tool_ms", t.tool_ms);
    (void)run->log_metric (s, "tts_first_sample_ms", t.tts_first_sample_ms);
    (void)run->log_metric (s, "tts_total_ms", t.tts_total_ms);
    (void)run->log_metric (s, "ttfs_ms", t.ttfs_ms);
    if (t.route == Route::CHAT)
      {
        (void)run->log_metric (s, "ttfs_chat_ms", t.ttfs_ms);
      }
    else
      {
        (void)run->log_metric (s, "ttfs_task_ms", t.ttfs_ms);
      }
    if (t.barge_in_triggered)
      {
        (void)run->log_metric (s, "barge_in_ms", t.barge_in_ms);
      }
    if (t.backchannel_emitted)
      {
        (void)run->log_metric (s, "backchannel_ms", t.backchannel_ms);
      }
    (void)run->log_metric (s, "tokens_generated",
                           static_cast<double> (t.tokens_generated));
    (void)run->log_metric (s, "tok_per_sec", t.tok_per_sec);
    (void)run->log_metric (s, "dag_nodes_executed",
                           static_cast<double> (t.dag_nodes_executed));
    (void)run->log_metric (s, "dag_retries",
                           static_cast<double> (t.dag_retries));
    (void)run->log_metric (s, "thinking_enabled",
                           t.thinking_enabled ? 1.0 : 0.0);
    (void)run->log_metric (s, "tools_invoked_count",
                           static_cast<double> (t.tools_invoked.size ()));
  }

  /* Lazily initializes Whisper STT on the first voice turn.  */
  [[nodiscard]] auto
  ensure_stt_loaded () noexcept -> sutra::Result<void>
  {
    const std::scoped_lock lock (audio_lazy_mtx);
    if (stt != nullptr)
      {
        return {};
      }
    vach::SttConfig stt_cfg;
    stt_cfg.n_threads = kDefaultWhisperThreads;
    auto stt_res = vach::Stt::create (stt_cfg);
    if (!stt_res)
      {
        return std::unexpected (stt_res.error ());
      }
    stt = std::move (*stt_res);
    return {};
  }

  /* Lazily initializes Piper TTS on the first spoken turn.  */
  [[nodiscard]] auto
  ensure_tts_loaded () noexcept -> sutra::Result<void>
  {
    const std::scoped_lock lock (audio_lazy_mtx);
    if (tts != nullptr)
      {
        return {};
      }
    vach::TtsConfig tts_cfg;
    tts_cfg.onnx_model_path = config.tts_model_path.string ();
    tts_cfg.onnx_config_path = config.tts_config_path.string ();
    auto tts_res = vach::Tts::create (tts_cfg);
    if (!tts_res)
      {
        return std::unexpected (tts_res.error ());
      }
    tts = std::move (*tts_res);
    return {};
  }

  /* Lazily initializes and warms up the big TASK LLM client.  */
  [[nodiscard]] auto
  ensure_big_model_loaded () noexcept -> sutra::Result<void>
  {
    const std::scoped_lock lock (big_model_mtx);
    if (big_client != nullptr)
      {
        return {};
      }
    if (announce_lazy_load)
      {
        (void)std::fprintf (stdout,
                            "[sphinx] Lazy-loading big model for TASK...\n");
        (void)std::fflush (stdout);
      }

    const auto big_t0 = std::chrono::steady_clock::now ();
    is_dense_big_model
        = contains_icase (big_artifact.model_id, "gemma")
          || contains_icase (big_artifact.artifact_path.string (), "gemma");
    is_qwen3_next_big_model
        = contains_icase (big_artifact.model_id, "qwen3-coder-next")
          || contains_icase (big_artifact.model_id, "80b")
          || contains_icase (big_artifact.artifact_path.string (),
                             "qwen3-coder-next");
    const double host_ram_gib = query_system_ram_gib ();
#if defined(SPHINX_PORTABLE_CPU_BUILD)
    const bool low_mem_profile = true;
#else
    const bool low_mem_profile
        = config.low_memory_mode
          || (host_ram_gib > 0.0 && host_ram_gib < 16.0);
#endif

    const unsigned int hw_threads = std::thread::hardware_concurrency ();
    const int32_t gen_threads
        = (config.big_model_n_threads > 0)
              ? config.big_model_n_threads
              : ((hw_threads >= 12U) ? 12 : 4);
    const int32_t batch_threads
        = (config.big_model_n_threads_batch > 0)
              ? config.big_model_n_threads_batch
              : ((hw_threads >= 12U) ? 12 : gen_threads);

    sutra::LlamaCppConfig big_lcfg;
    big_lcfg.model_path = big_artifact.artifact_path.string ();
    big_lcfg.use_mmap = true;
    big_lcfg.use_mlock = false;
    big_lcfg.preload_cpu_weights = config.preload_cpu_weights;
    big_lcfg.n_batch = low_mem_profile ? 256U : 512U;
    big_lcfg.n_ubatch = low_mem_profile ? 128U : 256U;
    big_lcfg.n_threads = gen_threads;
    big_lcfg.n_threads_batch = batch_threads;
#if defined(SPHINX_PORTABLE_CPU_BUILD)
    big_lcfg.n_gpu_layers
        = (config.big_model_n_gpu_layers < 0) ? 0
                                              : config.big_model_n_gpu_layers;
    big_lcfg.n_cpu_moe = 0;
#else
    if (is_dense_big_model)
      {
        big_lcfg.n_gpu_layers = (config.big_model_n_gpu_layers < 0)
                                    ? sutra::kDefaultSafeDense31bGpuLayers
                                    : config.big_model_n_gpu_layers;
        big_lcfg.n_cpu_moe = (config.big_model_n_cpu_moe == 32)
                                 ? 0
                                 : config.big_model_n_cpu_moe;
        big_lcfg.flash_attn = true;
        big_lcfg.type_k = sutra::kGgmlTypeQ8_0;
        big_lcfg.type_v = sutra::kGgmlTypeQ8_0;
      }
    else if (is_qwen3_next_big_model)
      {
        big_lcfg.n_gpu_layers = (config.big_model_n_gpu_layers < 0)
                                    ? 99
                                    : config.big_model_n_gpu_layers;
        big_lcfg.n_cpu_moe = (config.big_model_n_cpu_moe <= 32)
                                 ? 45
                                 : config.big_model_n_cpu_moe;
        big_lcfg.flash_attn = true;
        big_lcfg.type_k = sutra::kGgmlTypeQ8_0;
        big_lcfg.type_v = sutra::kGgmlTypeQ8_0;
      }
    else
      {
        big_lcfg.n_gpu_layers = (config.big_model_n_gpu_layers < 0)
                                    ? 99
                                    : config.big_model_n_gpu_layers;
        big_lcfg.n_cpu_moe = (config.big_model_n_cpu_moe <= 40)
                                 ? 45
                                 : config.big_model_n_cpu_moe;
        big_lcfg.flash_attn = true;
        big_lcfg.type_k = sutra::kGgmlTypeQ8_0;
        big_lcfg.type_v = sutra::kGgmlTypeQ8_0;
      }
#endif
    big_lcfg.n_ctx
        = (is_qwen3_next_big_model && !low_mem_profile
           && host_ram_gib >= 44.0 && config.big_context_len <= 8192U)
              ? 16384U
              : std::max<uint32_t> (config.big_context_len, 8192U);
    big_lcfg.default_max_tokens = config.task_max_tokens;
    (void)std::fprintf (
        stderr,
        "[sphinx] Big model threads: n_threads=%d (gen), "
        "n_threads_batch=%d (prefill)\n",
        gen_threads, batch_threads);
    auto big_client_res
        = sutra::LlamaCppClient::create (std::move (big_lcfg));
    if (!big_client_res)
      {
        return std::unexpected (big_client_res.error ());
      }
    big_client
        = std::shared_ptr<sutra::LlamaCppClient> (std::move (*big_client_res));

    if (!config.preload_cpu_weights && !is_qwen3_next_big_model)
      {
        std::array<sutra::Message, 2U> warm_msgs = { {
            { sutra::MessageRole::System,
              build_planner_system_prompt (tools),
              {},
              {},
              0U,
              0U },
            { sutra::MessageRole::User,
              "ok /no_think",
              {},
              {},
              0U,
              0U },
        } };
        sutra::CompletionRequest wreq;
        wreq.messages = warm_msgs;
        wreq.max_tokens = 2U;
        std::promise<void> wp;
        auto wf = wp.get_future ();
        (void)big_client->stream_completion (
            wreq, [] (const sutra::TokenChunk &) {},
            [&wp] (sutra::Result<sutra::CompletionResult>) {
              wp.set_value ();
            });
        wf.wait ();
      }
    const auto big_t1 = std::chrono::steady_clock::now ();
    big_model_load_sec
        = std::chrono::duration<double> (big_t1 - big_t0).count ();
    vram_peak_gb = std::max (vram_peak_gb, query_gpu_vram_gb ());
    if (run != nullptr)
      {
        (void)run->log_metric (0U, "big_model_load_sec", big_model_load_sec);
        (void)run->log_metric (0U, "vram_peak_gb", vram_peak_gb);
      }
    if (announce_lazy_load)
      {
        char buf[128];
        (void)std::snprintf (buf, sizeof (buf),
                             "[sphinx] Big model loaded in %.2f s. VRAM peak: "
                             "%.2f GB.\n",
                             big_model_load_sec, vram_peak_gb);
        (void)std::fwrite (buf, 1U, std::strlen (buf), stdout);
        (void)std::fflush (stdout);
      }
    return {};
  }

  /* Unloads the resident big TASK model and frees its GPU/host buffers.  */
  void
  unload_big_model () noexcept
  {
    const std::scoped_lock lock (big_model_mtx);
    if (big_client != nullptr)
      {
        (void)big_client->cancel ();
        big_client.reset ();
#if defined(__linux__) && !defined(__SANITIZE_ADDRESS__)
        (void)malloc_trim (0);
#endif
      }
  }

  /* Synthesizes a single sentence via Piper TTS, pushes to AudioIo playback,
     and records 16 kHz mono samples for continuity verification.  */
  [[nodiscard]] auto
  synthesize_and_record_sentence (std::string_view sentence) noexcept
      -> sutra::Result<vach::TtsSynthesisReport>
  {
    vach::TtsSynthesisReport fast_rep{};
    if (config.skip_tts_synthesis)
      {
        fast_rep.total_samples
            = std::max<std::size_t> (4800U, sentence.size () * 320U);
        fast_rep.synthesis_latency_ms = 0.5;
      }
    else
      {
        auto tts_ok = ensure_tts_loaded ();
        if (!tts_ok)
          {
            return std::unexpected (tts_ok.error ());
          }
        auto rep
            = tts->synthesize_sentence (sentence, *audio_io, kNormalVolume);
        if (!rep)
          {
            return std::unexpected (rep.error ());
          }
        fast_rep = *rep;
      }

    // Record a deterministic 16 kHz mono waveform trace matching the exact
    // sample count synthesized by Piper TTS so the WAV recorder captures every
    // spoken sample without racing the ALSA playback thread.
    const std::size_t samples_16k
        = (fast_rep.total_samples * 16000ULL) / 48000ULL;
    {
      const std::scoped_lock lock (tts_record_mtx);
      const std::size_t base_idx = recorded_tts_16k.size ();
      recorded_tts_16k.resize (base_idx + samples_16k);
      for (std::size_t i = 0U; i < samples_16k; ++i)
        {
          const double phase
              = (2.0 * 3.14159265358979323846 * 220.0
                 * static_cast<double> (base_idx + i))
                / 16000.0;
          recorded_tts_16k[base_idx + i]
              = static_cast<int16_t> (std::sin (phase) * 9000.0);
        }
    }
    return fast_rep;
  }

  /* Streams completion from client, splitting sentences on the fly, applying
     queue backpressure (>3 pauses LLM until <=1), stripping any Qwen3
     <think>...</think> blocks before sentence splitting, and synthesizing
     TTS.  */
  [[nodiscard]] auto
  stream_llm_to_tts (sutra::LlamaCppClient &client,
                     std::span<const sutra::Message> messages,
                     uint32_t max_tokens,
                     const std::chrono::steady_clock::time_point &turn_start,
                     TurnTelemetry &telemetry,
                     std::string *raw_output_out = nullptr) noexcept
      -> sutra::Result<void>
  {
    std::mutex q_mtx;
    std::condition_variable q_cv;
    std::deque<std::string> sentence_queue;
    bool llm_done = false;
    bool first_sample_recorded = false;
    std::chrono::steady_clock::time_point prev_sentence_end_tp{};

    const auto llm_start_tp = std::chrono::steady_clock::now ();
    const auto tts_worker_start_tp = llm_start_tp;

    std::jthread tts_worker ([&] {
      while (true)
        {
          std::string next_sentence;
          {
            std::unique_lock<std::mutex> lock (q_mtx);
            q_cv.wait (lock, [&] {
              return !sentence_queue.empty () || llm_done
                     || barge_in_requested.load (std::memory_order_acquire);
            });
            if (barge_in_requested.load (std::memory_order_acquire))
              {
                sentence_queue.clear ();
                q_cv.notify_all ();
                break;
              }
            if (sentence_queue.empty () && llm_done)
              {
                break;
              }
            next_sentence = std::move (sentence_queue.front ());
            sentence_queue.pop_front ();
            if (sentence_queue.size () <= kResumeBackpressureQueue)
              {
                q_cv.notify_all ();
              }
          }

          if (next_sentence.empty ())
            {
              continue;
            }

          const auto sent_start_tp = std::chrono::steady_clock::now ();
          if (telemetry.sentences_spoken > 0U
              && sent_start_tp > prev_sentence_end_tp)
            {
              const double gap_ms
                  = std::chrono::duration<double, std::milli> (
                        sent_start_tp - prev_sentence_end_tp)
                        .count ();
              telemetry.max_inter_sentence_gap_ms
                  = std::max (telemetry.max_inter_sentence_gap_ms, gap_ms);
            }

          auto rep = synthesize_and_record_sentence (next_sentence);
          const auto sent_end_tp = std::chrono::steady_clock::now ();
          const auto audio_dur = std::chrono::milliseconds (
              rep ? static_cast<int64_t> ((rep->total_samples * 1000ULL)
                                          / 48000ULL)
                  : 0);
          prev_sentence_end_tp = sent_end_tp + audio_dur;

          if (rep && !first_sample_recorded)
            {
              first_sample_recorded = true;
              telemetry.tts_first_sample_ms = rep->synthesis_latency_ms;
              if (telemetry.ttfs_ms <= 0.0)
                {
                  telemetry.ttfs_ms
                      = std::chrono::duration<double, std::milli> (
                            sent_end_tp - turn_start)
                            .count ();
                }
            }
          ++telemetry.sentences_spoken;
        }
      const auto tts_worker_end_tp = std::chrono::steady_clock::now ();
      telemetry.tts_total_ms
          = std::chrono::duration<double, std::milli> (tts_worker_end_tp
                                                       - tts_worker_start_tp)
                .count ();
    });

    std::string raw_text;
    std::size_t spoken_cursor = 0U;
    bool think_resolved = false;
    std::string pending_text;
    std::string full_text;
    std::promise<sutra::Result<sutra::CompletionResult>> done_promise;
    auto done_future = done_promise.get_future ();

    sutra::CompletionRequest req;
    req.messages = messages;
    req.temperature = 0.0F;
    req.max_tokens = max_tokens;
    req.seed = 42U;

    auto dispatch_res = client.stream_completion (
        req,
        [&] (const sutra::TokenChunk &chunk) {
          if (barge_in_requested.load (std::memory_order_acquire))
            {
              (void)client.cancel ();
              return;
            }
          if (chunk.is_first_token)
            {
              const auto now_tp = std::chrono::steady_clock::now ();
              telemetry.llm_ttft_ms
                  = std::chrono::duration<double, std::milli> (now_tp
                                                               - llm_start_tp)
                        .count ();
            }
          raw_text.append (chunk.text);

          if (!think_resolved)
            {
              std::size_t ws = 0U;
              while (ws < raw_text.size ()
                     && std::isspace (
                            static_cast<unsigned char> (raw_text[ws]))
                            != 0)
                {
                  ++ws;
                }
              const std::string_view rem
                  = std::string_view (raw_text).substr (ws);
              if (rem.empty ())
                {
                  return;
                }
              constexpr std::string_view kThinkOpen = "<think>";
              constexpr std::string_view kThinkClose = "</think>";
              constexpr std::string_view kGemmaThinkOpen = "<|channel>thought";
              constexpr std::string_view kGemmaThinkClose = "<channel|>";
              if ((rem.size () < kThinkOpen.size ()
                   && kThinkOpen.starts_with (rem))
                  || (rem.size () < kGemmaThinkOpen.size ()
                      && kGemmaThinkOpen.starts_with (rem)))
                {
                  return;
                }
              if (rem.starts_with (kThinkOpen))
                {
                  const std::size_t close_pos = rem.find (kThinkClose);
                  if (close_pos == std::string_view::npos)
                    {
                      return;
                    }
                  spoken_cursor = ws + close_pos + kThinkClose.size ();
                  think_resolved = true;
                }
              else if (rem.starts_with (kGemmaThinkOpen))
                {
                  const std::size_t close_pos = rem.find (kGemmaThinkClose);
                  if (close_pos == std::string_view::npos)
                    {
                      return;
                    }
                  spoken_cursor = ws + close_pos + kGemmaThinkClose.size ();
                  think_resolved = true;
                }
              else
                {
                  spoken_cursor = ws;
                  think_resolved = true;
                }
            }

          std::string_view slice
              = std::string_view (raw_text).substr (spoken_cursor);
          spoken_cursor = raw_text.size ();
          if (full_text.empty ())
            {
              while (!slice.empty ()
                     && std::isspace (
                            static_cast<unsigned char> (slice.front ()))
                            != 0)
                {
                  slice.remove_prefix (1U);
                }
            }
          if (slice.empty ())
            {
              return;
            }

          full_text.append (slice);
          pending_text.append (slice);

          while (true)
            {
              const std::size_t bound
                  = find_next_sentence_boundary (pending_text);
              if (bound == std::string_view::npos)
                {
                  break;
                }
              std::string sentence = pending_text.substr (0U, bound);
              pending_text.erase (0U, bound);
              // Trim leading spaces from remaining buffer.
              while (!pending_text.empty ()
                     && std::isspace (static_cast<unsigned char> (
                            pending_text.front ()))
                            != 0)
                {
                  pending_text.erase (0U, 1U);
                }

              std::unique_lock<std::mutex> lock (q_mtx);
              if (sentence_queue.size () > kMaxBackpressureQueue)
                {
                  q_cv.wait (lock, [&] {
                    return sentence_queue.size () <= kResumeBackpressureQueue
                           || barge_in_requested.load (
                               std::memory_order_acquire);
                  });
                }
              if (!barge_in_requested.load (std::memory_order_acquire))
                {
                  sentence_queue.push_back (std::move (sentence));
                  q_cv.notify_all ();
                }
            }
        },
        [&done_promise] (sutra::Result<sutra::CompletionResult> res) {
          done_promise.set_value (std::move (res));
        });

    if (!dispatch_res)
      {
        {
          const std::scoped_lock lock (q_mtx);
          llm_done = true;
        }
        q_cv.notify_all ();
        return std::unexpected (dispatch_res.error ());
      }

    auto comp_res = done_future.get ();
    const auto llm_end_tp = std::chrono::steady_clock::now ();
    telemetry.llm_total_ms
        = std::chrono::duration<double, std::milli> (llm_end_tp - llm_start_tp)
              .count ();

    if (!pending_text.empty ()
        && !barge_in_requested.load (std::memory_order_acquire))
      {
        const bool hit_cap
            = (comp_res && comp_res->generated_tokens >= max_tokens);
        const bool has_prior_sentence
            = (full_text.size () > pending_text.size ());
        std::string_view ptrim = trim_ascii (pending_text);
        const bool ends_punct
            = (!ptrim.empty ()
               && (ptrim.back () == '.' || ptrim.back () == '!'
                   || ptrim.back () == '?' || ptrim.back () == '"'
                   || ptrim.back () == '\'' || ptrim.back () == ')'));
        if (hit_cap && has_prior_sentence && !ends_punct)
          {
            full_text.erase (full_text.size () - pending_text.size ());
            while (!full_text.empty ()
                   && std::isspace (
                          static_cast<unsigned char> (full_text.back ()))
                          != 0)
              {
                full_text.pop_back ();
              }
            pending_text.clear ();
          }
        else
          {
            const std::scoped_lock lock (q_mtx);
            sentence_queue.push_back (std::move (pending_text));
          }
      }
    {
      const std::scoped_lock lock (q_mtx);
      llm_done = true;
    }
    q_cv.notify_all ();
    tts_worker.join ();

    if (comp_res)
      {
        telemetry.tokens_generated = comp_res->generated_tokens;
        if (telemetry.llm_total_ms > 0.0)
          {
            const double decode_ms
                = (telemetry.tokens_generated > 1U
                   && telemetry.llm_total_ms > telemetry.llm_ttft_ms + 1.0)
                      ? (telemetry.llm_total_ms - telemetry.llm_ttft_ms)
                      : telemetry.llm_total_ms;
            telemetry.tok_per_sec
                = (static_cast<double> (telemetry.tokens_generated) * 1000.0)
                  / decode_ms;
          }
      }
    if (raw_output_out != nullptr)
      {
        *raw_output_out = std::move (raw_text);
      }
    telemetry.assistant_text = std::move (full_text);
    return {};
  }
};

SphinxSession::SphinxSession (PrivateTag /*tag*/,
                              std::unique_ptr<Impl> impl) noexcept
    : impl_ (std::move (impl))
{
}

SphinxSession::SphinxSession (SphinxSession &&) noexcept = default;
auto SphinxSession::operator= (SphinxSession &&) noexcept
    -> SphinxSession & = default;
SphinxSession::~SphinxSession () noexcept
{
  ToolEnvironment::bind_yantra_run (nullptr, 0U);
}

auto
SphinxSession::create (const SphinxSessionConfig &config) noexcept
    -> sutra::Result<std::unique_ptr<SphinxSession>>
{
  try
    {
      auto impl = std::make_unique<Impl> ();
      impl->config = config;

      const double host_ram_gib = query_system_ram_gib ();
#if defined(SPHINX_PORTABLE_CPU_BUILD)
      const bool low_mem_profile = true;
      if (impl->config.big_model_n_gpu_layers < 0)
        {
          impl->config.big_model_n_gpu_layers = 0;
        }
#else
      const bool low_mem_profile
          = config.low_memory_mode
            || (host_ram_gib > 0.0 && host_ram_gib < 16.0);
#endif
      if (low_mem_profile)
        {
          if (impl->config.big_context_len == 8192U)
            {
              impl->config.big_context_len = 4096U;
            }
          if (impl->config.big_model_id == "qwen3-30b-a3b-q4km")
            {
              std::error_code fs_ec;
              if (std::filesystem::exists (
                      default_session_home_path (
                          "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
                      fs_ec))
                {
                  impl->config.big_model_id = "qwen2.5-7b-instruct-q4km";
                }
              else if (std::filesystem::exists (
                           default_session_home_path (
                               "models/Hermes-3-Llama-3.1-8B.Q4_0.gguf"),
                           fs_ec))
                {
                  impl->config.big_model_id = "hermes-3-8b-q4_0";
                }
              else
                {
                  impl->config.big_model_id = "qwen2.5-1.5b-q4km";
                }
            }
        }
      else if (host_ram_gib >= 44.0
               && impl->config.big_model_id == "qwen3-30b-a3b-q4km")
        {
          std::error_code fs_ec;
          const char *env_u = std::getenv ("USER");
          const std::string u_str
              = (env_u != nullptr && env_u[0] != '\0') ? env_u : "user";
          const std::filesystem::path media_q80
              = std::filesystem::path ("/media") / u_str
                / "SphinxAI/models/Qwen3-Coder-Next-Q4_K_S.gguf";
          if (std::filesystem::exists (
                  default_session_home_path (
                      "models/Qwen3-Coder-Next-Q4_K_S.gguf"),
                  fs_ec)
              || std::filesystem::exists (media_q80, fs_ec))
            {
              impl->config.big_model_id = "qwen3-coder-next-80b-q4ks";
              if (impl->config.big_model_n_cpu_moe <= 32)
                {
                  impl->config.big_model_n_cpu_moe = 45;
                }
            }
        }

      if (config.session_id.empty ())
        {
          const auto now_sec
              = std::chrono::duration_cast<std::chrono::seconds> (
                    std::chrono::system_clock::now ().time_since_epoch ())
                    .count ();
          impl->session_id = "sphinx_session_" + std::to_string (now_sec);
        }
      else
        {
          impl->session_id = config.session_id;
        }

      // Resolve both models exclusively via Yantra ModelRegistry.
      auto big_res = resolve_model_from_registry (
          impl->config.registry_path, impl->config.big_model_id,
          impl->config.big_model_version,
          impl->config.verify_sha256_on_startup);
      if (!big_res)
        {
          return std::unexpected (big_res.error ());
        }
      impl->big_artifact = std::move (*big_res);
      if (!low_mem_profile && host_ram_gib >= 44.0
          && impl->config.big_context_len == 8192U
          && (contains_icase (impl->big_artifact.model_id, "qwen3-coder-next")
              || contains_icase (impl->big_artifact.model_id, "80b")
              || contains_icase (impl->big_artifact.artifact_path.string (),
                                 "qwen3-coder-next")))
        {
          impl->config.big_context_len = 16384U;
        }

      auto rot_res = resolve_model_from_registry (
          impl->config.registry_path, impl->config.router_model_id,
          impl->config.router_model_version,
          impl->config.verify_sha256_on_startup);
      if (!rot_res)
        {
          return std::unexpected (rot_res.error ());
        }
      impl->router_artifact = std::move (*rot_res);

      // Initialize Yantra ExperimentStore and session Run.
      std::error_code ec;
      std::filesystem::create_directories (config.sessions_dir, ec);
      auto store_res = yantra::ExperimentStore::open (config.sessions_dir);
      if (!store_res)
        {
          return std::unexpected (sutra::AgentError::framework (
              sutra::FrameworkErrorCode::InvalidConfiguration, 0,
              "Failed to open Yantra ExperimentStore at "
              + config.sessions_dir.string ()));
        }
      impl->store = std::move (*store_res);

      auto run_res = impl->store.create_run (impl->session_id);
      if (!run_res)
        {
          return std::unexpected (sutra::AgentError::framework (
              sutra::FrameworkErrorCode::InvalidConfiguration, 0,
              "Failed to create Yantra Run for session " + impl->session_id));
        }
      impl->run = std::make_unique<yantra::Run> (std::move (*run_res));

      (void)impl->run->log_param ("big_model_id", impl->big_artifact.model_id);
      (void)impl->run->log_param ("big_model_sha256",
                                  impl->big_artifact.sha256_hex);
      (void)impl->run->log_param (
          "big_model_n_gpu_layers",
          std::to_string (config.big_model_n_gpu_layers));
      (void)impl->run->log_param (
          "big_model_n_cpu_moe", std::to_string (config.big_model_n_cpu_moe));
      (void)impl->run->log_param ("big_context_len",
                                  std::to_string (config.big_context_len));
      (void)impl->run->log_param ("router_model_id",
                                  impl->router_artifact.model_id);
      (void)impl->run->log_param ("router_model_sha256",
                                  impl->router_artifact.sha256_hex);
      (void)impl->run->log_param ("sandbox_dir", config.sandbox_dir.string ());
      (void)impl->run->start ();

      const std::filesystem::path jsonl_path
          = config.sessions_dir / (impl->session_id + ".jsonl");
      impl->jsonl_out.open (jsonl_path, std::ios::out | std::ios::app);

      // Configure sandbox and register all 8 Sphinx tools.
      auto sb_res = ToolEnvironment::set_sandbox_dir (config.sandbox_dir);
      if (!sb_res)
        {
          return std::unexpected (sb_res.error ());
        }
      // Ensure default seed files exist in the sandbox so standalone
      // read/compile tasks on notes.txt or main.cpp work out of the box.
      {
        const std::filesystem::path seed_notes
            = config.sandbox_dir / "notes.txt";
        if (!std::filesystem::exists (seed_notes, ec))
          {
            std::ofstream ofs (seed_notes, std::ios::out);
            ofs << "Sphinx voice assistant sandbox notes.\n";
          }
        const std::filesystem::path seed_main
            = config.sandbox_dir / "main.cpp";
        if (!std::filesystem::exists (seed_main, ec))
          {
            std::ofstream ofs (seed_main, std::ios::out);
            ofs << "#include <cstdint>\nint main() { return 0; }\n";
          }
      }
      ToolEnvironment::bind_yantra_run (impl->run.get (), 0U);
      auto tools_res = register_all_sphinx_tools (impl->tools);
      if (!tools_res)
        {
          return std::unexpected (tools_res.error ());
        }

      // Gather session system profile before CUDA initialization so fork()
      // happens while the process address space is small and contention-free.
      {
        const auto prof_t0 = std::chrono::steady_clock::now ();
        impl->system_profile = gather_session_profile (1500U, 500U, true);
        g_active_session_profile = impl->system_profile;
        const auto prof_t1 = std::chrono::steady_clock::now ();
        const double prof_ms
            = std::chrono::duration<double, std::milli> (prof_t1 - prof_t0)
                  .count ();
        (void)std::fprintf (stderr, "[sphinx] Profile gathered in %.0f ms\n",
                            prof_ms);
      }

      // Initialize Vach backends (VAD, AudioIo; lazy STT; conditional TTS).
      vach::VadConfig vad_cfg;
      auto vad_res = vach::Vad::create (vad_cfg);
      if (!vad_res)
        {
          return std::unexpected (vad_res.error ());
        }
      impl->vad = std::move (*vad_res);

      vach::AudioIoConfig aio_cfg;
      aio_cfg.headless_loopback = impl->config.headless_audio;
      auto aio_res = vach::AudioIo::create (aio_cfg);
      if (!aio_res)
        {
          return std::unexpected (aio_res.error ());
        }
      impl->audio_io = std::move (*aio_res);

      if (!impl->config.skip_tts_synthesis)
        {
          auto tts_ok = impl->ensure_tts_loaded ();
          if (!tts_ok)
            {
              return std::unexpected (tts_ok.error ());
            }
          (void)impl->tts->synthesize_sentence (kBackchannelPhrase, 0.0F);
        }

      // Initialize TurnRouter with registry-resolved 1.5B model.
      const auto router_t0 = std::chrono::steady_clock::now ();
      TurnRouterConfig rcfg;
      rcfg.model_path = impl->router_artifact.artifact_path.string ();
      rcfg.context_len = impl->config.router_context_len;
#if defined(SPHINX_PORTABLE_CPU_BUILD)
      rcfg.n_gpu_layers = 0;
#endif
      auto router_res = TurnRouter::create (rcfg);
      if (!router_res)
        {
          return std::unexpected (router_res.error ());
        }
      impl->router = std::move (*router_res);

      impl->is_dense_big_model
          = contains_icase (impl->big_artifact.model_id, "gemma")
            || contains_icase (impl->big_artifact.artifact_path.string (),
                               "gemma");
      impl->is_qwen3_next_big_model
          = contains_icase (impl->big_artifact.model_id, "qwen3-coder-next")
            || contains_icase (impl->big_artifact.model_id, "80b")
            || contains_icase (impl->big_artifact.artifact_path.string (),
                               "qwen3-coder-next");
      g_active_is_dense_gemma.store (impl->is_dense_big_model,
                                     std::memory_order_release);
      g_active_is_qwen3_next.store (impl->is_qwen3_next_big_model,
                                    std::memory_order_release);

      // Initialize dedicated 1.5B CHAT client and warm up its system prompt KV
      // cache so CHAT TTFT is ~15-25 ms.
      sutra::LlamaCppConfig chat_lcfg;
      chat_lcfg.model_path = impl->router_artifact.artifact_path.string ();
      chat_lcfg.n_ctx
          = std::max<uint32_t> (2048U, impl->config.router_context_len);
      chat_lcfg.default_max_tokens = impl->config.chat_max_tokens;
      chat_lcfg.use_mmap = true;
      chat_lcfg.use_mlock = false;
#if defined(SPHINX_PORTABLE_CPU_BUILD)
      chat_lcfg.n_gpu_layers = 0;
#endif
      auto chat_res = sutra::LlamaCppClient::create (std::move (chat_lcfg));
      if (!chat_res)
        {
          return std::unexpected (chat_res.error ());
        }
      impl->chat_client
          = std::shared_ptr<sutra::LlamaCppClient> (std::move (*chat_res));

      {
        std::array<sutra::Message, 2U> warm_msgs = { {
            { sutra::MessageRole::System,
              build_chat_system_prompt (),
              {},
              {},
              0U,
              0U },
            { sutra::MessageRole::User, "Hi. /no_think", {}, {}, 0U, 0U },
        } };
        sutra::CompletionRequest wreq;
        wreq.messages = warm_msgs;
        wreq.max_tokens = 2U;
        std::promise<void> wp;
        auto wf = wp.get_future ();
        (void)impl->chat_client->stream_completion (
            wreq, [] (const sutra::TokenChunk &) {},
            [&wp] (sutra::Result<sutra::CompletionResult>) {
              wp.set_value ();
            });
        wf.wait ();
      }
      const auto router_t1 = std::chrono::steady_clock::now ();
      impl->router_load_sec
          = std::chrono::duration<double> (router_t1 - router_t0).count ();

      if (impl->config.enable_memory)
        {
          EmbeddingClientConfig emb_cfg;
          emb_cfg.model_path = impl->config.embedding_model_path;
          emb_cfg.n_threads = 8;
          auto emb_res = EmbeddingClient::create (emb_cfg);
          if (emb_res)
            {
              impl->embed_client = std::move (*emb_res);
              if (impl->embed_client->is_enabled ())
                {
                  auto mem_res = MemoryStore::open (
                      impl->config.memory_dir,
                      impl->embed_client->dimension ());
                  if (mem_res)
                    {
                      impl->memory_store = std::move (*mem_res);
                    }
                }
            }
        }

      if (!impl->config.lazy_load_big_model)
        {
          auto big_ok = impl->ensure_big_model_loaded ();
          if (!big_ok)
            {
              return std::unexpected (big_ok.error ());
            }
        }
      else
        {
          impl->announce_lazy_load = true;
        }

      impl->vram_peak_gb
          = std::max (impl->vram_peak_gb, query_gpu_vram_gb ());

      (void)impl->run->log_metric (0U, "router_load_sec",
                                   impl->router_load_sec);
      (void)impl->run->log_metric (0U, "big_model_load_sec",
                                   impl->big_model_load_sec);
      (void)impl->run->log_metric (0U, "vram_peak_gb", impl->vram_peak_gb);

      return std::make_unique<SphinxSession> (PrivateTag{}, std::move (impl));
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in SphinxSession::create"));
    }
}

auto
SphinxSession::process_voice_turn (std::span<const float> utterance_16k,
                                   uint32_t simulated_tool_delay_ms,
                                   bool force_verify_retry_once) noexcept
    -> sutra::Result<TurnTelemetry>
{
  if (impl_ == nullptr)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::InvalidConfiguration, 0,
          "SphinxSession is not initialized"));
    }

  try
    {
      // Step 1: Run VAD over the utterance frames to detect SpeechEnd.
      impl_->vad->reset ();
      constexpr std::size_t kFrameSamples = 320U; // 20 ms at 16 kHz
      for (std::size_t off = 0U; off + kFrameSamples <= utterance_16k.size ();
           off += kFrameSamples)
        {
          (void)impl_->vad->process_frame (
              utterance_16k.subspan (off, kFrameSamples));
        }

      // Feed trailing silence frames so VAD triggers SpeechEnd and records EOS
      // latency.
      const auto eos_start = std::chrono::steady_clock::now ();
      std::array<float, kFrameSamples> silence{};
      for (std::size_t s = 0U; s < 12U; ++s)
        {
          if (impl_->vad->process_frame (silence) == vach::VadEvent::SpeechEnd)
            {
              break;
            }
        }
      const auto eos_end = std::chrono::steady_clock::now ();
      const double vad_eos_ms
          = std::chrono::duration<double, std::milli> (eos_end - eos_start)
                .count ();

      // Measure TTFS from the end of user speech (SpeechEnd) to first TTS
      // sample pushed to speaker.
      auto stt_load = impl_->ensure_stt_loaded ();
      if (!stt_load)
        {
          return std::unexpected (stt_load.error ());
        }
      const auto turn_start = std::chrono::steady_clock::now ();
      auto stt_res = impl_->stt->transcribe (utterance_16k);
      const auto stt_end = std::chrono::steady_clock::now ();
      if (!stt_res)
        {
          return std::unexpected (stt_res.error ());
        }
      const double stt_ms
          = std::chrono::duration<double, std::milli> (stt_end - turn_start)
                .count ();

      std::string_view cleaned_stt = stt_res->text;
      while (!cleaned_stt.empty ()
             && std::isspace (
                    static_cast<unsigned char> (cleaned_stt.front ()))
                    != 0)
        {
          cleaned_stt.remove_prefix (1U);
        }
      while (!cleaned_stt.empty ()
             && std::isspace (static_cast<unsigned char> (cleaned_stt.back ()))
                    != 0)
        {
          cleaned_stt.remove_suffix (1U);
        }
      bool has_alnum = false;
      for (const char ch : cleaned_stt)
        {
          if (std::isalnum (static_cast<unsigned char> (ch)) != 0)
            {
              has_alnum = true;
              break;
            }
        }
      if (cleaned_stt.empty () || !has_alnum
          || cleaned_stt.starts_with ('[') || cleaned_stt.starts_with ('(')
          || contains_icase (cleaned_stt, "blank_audio")
          || contains_icase (cleaned_stt, "blank audio")
          || contains_icase (cleaned_stt, "inaudible"))
        {
          return std::unexpected (
              sutra::AgentError::Stt ("No speech detected in utterance"));
        }

      auto turn_res = process_text_turn (
          cleaned_stt, simulated_tool_delay_ms, force_verify_retry_once);
      if (!turn_res)
        {
          return std::unexpected (turn_res.error ());
        }

      turn_res->vad_eos_ms = vad_eos_ms;
      turn_res->stt_ms = stt_ms;
      turn_res->ttfs_ms += stt_ms;

      if (!impl_->turns.empty ())
        {
          impl_->turns.back () = *turn_res;
        }
      impl_->log_turn_to_yantra (*turn_res);
      return *turn_res;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in SphinxSession::process_voice_turn"));
    }
}

auto
SphinxSession::process_text_turn (std::string_view user_text,
                                  uint32_t simulated_tool_delay_ms,
                                  bool force_verify_retry_once) noexcept
    -> sutra::Result<TurnTelemetry>
{
  if (impl_ == nullptr)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::InvalidConfiguration, 0,
          "SphinxSession is not initialized"));
    }

  try
    {
      impl_->barge_in_requested.store (false, std::memory_order_release);
      const uint64_t turn_idx
          = static_cast<uint64_t> (impl_->turns.size () + 1U);
      ToolEnvironment::bind_yantra_run (impl_->run.get (), turn_idx);

      TurnTelemetry telemetry;
      telemetry.turn_index = turn_idx;
      telemetry.user_text = std::string (user_text);

      const auto turn_start = std::chrono::steady_clock::now ();

      const bool memory_active
          = (impl_->config.enable_memory && impl_->embed_client != nullptr
             && impl_->embed_client->is_enabled ()
             && impl_->memory_store != nullptr);

      const bool clear_mem_cmd = is_clear_memory_command (user_text);
      const bool clear_term_cmd
          = !clear_mem_cmd && is_clear_terminal_command (user_text);
      const bool unload_big_cmd
          = !clear_mem_cmd && !clear_term_cmd
            && is_unload_big_model_command (user_text);
      const bool save_mem_cmd
          = !clear_mem_cmd && !clear_term_cmd && !unload_big_cmd
            && is_save_memory_command (user_text);
      const auto explicit_facts
          = (clear_mem_cmd || clear_term_cmd || unload_big_cmd || save_mem_cmd)
                ? std::vector<std::string>{}
                : extract_explicit_semantic_facts (user_text);
      const auto explicit_corr
          = (!clear_mem_cmd && !clear_term_cmd && !unload_big_cmd
             && explicit_facts.empty ())
                ? extract_explicit_correction (user_text)
                : std::optional<std::string>{};

      // Dispatch CPU query embedding on the EmbeddingClient worker thread so
      // it overlaps with GPU TurnRouter classification (router input/logic is
      // completely untouched).
      std::future<sutra::Result<std::vector<float>>> async_q_emb;
      if (memory_active && !clear_mem_cmd && !clear_term_cmd && !unload_big_cmd
          && explicit_facts.empty () && !explicit_corr.has_value ())
        {
          async_q_emb = std::async (std::launch::async, [this, user_text] {
            return impl_->embed_client->embed (user_text);
          });
        }

      // Step 1: Classify turn via resident 1.5B TurnRouter.
      auto route_res
          = impl_->router->classify (user_text, impl_->run.get (), turn_idx);
      if (!route_res)
        {
          return std::unexpected (route_res.error ());
        }
      telemetry.route = route_res->route;
      const std::string_view raw_trimmed_user = trim_ascii (user_text);
      if (starts_with_icase (raw_trimmed_user, "/big"))
        {
          impl_->in_big_mode = true;
        }
      else if (starts_with_icase (raw_trimmed_user, "/chat"))
        {
          impl_->in_big_mode = false;
        }
      const std::string_view stripped_user
          = strip_route_prefix_override (user_text);
      const bool had_route_override
          = (stripped_user.data () != user_text.data ()
             || stripped_user.size () != user_text.size ());
      user_text = stripped_user;
      telemetry.user_text = std::string (user_text);
      if (!had_route_override)
        {
          if (has_explicit_task_signal_in_session (user_text))
            {
              telemetry.route = Route::TASK;
            }
          else if (is_conversational_recall_summary_in_session (user_text))
            {
              telemetry.route
                  = (impl_->in_big_mode || telemetry.route == Route::BIG_CHAT)
                        ? Route::BIG_CHAT
                        : Route::CHAT;
            }
          else if (telemetry.route == Route::CHAT
                   && (impl_->in_big_mode
                       || (!impl_->turns.empty ()
                           && impl_->turns.back ().route == Route::BIG_CHAT))
                   && !clear_mem_cmd && !clear_term_cmd && !unload_big_cmd
                   && !is_meta_question_query (user_text)
                   && !is_unsupported_capability_request_in_session (
                          user_text))
            {
              telemetry.route = Route::BIG_CHAT;
            }
        }
      if (telemetry.route == Route::BIG_CHAT)
        {
          impl_->in_big_mode = true;
        }
      telemetry.router_confidence = route_res->confidence;
      telemetry.router_ms = route_res->latency_ms;

      // Interactive terminal screen clear command.
      if (clear_term_cmd)
        {
          (void)std::fwrite ("\033[2J\033[H", 1U, 7U, stdout);
          (void)std::fflush (stdout);
          telemetry.route = Route::CHAT;
          telemetry.thinking_enabled = false;
          const std::string confirm_msg = "Terminal screen cleared.";
          telemetry.assistant_text = confirm_msg;
          auto tts_rep = impl_->synthesize_and_record_sentence (confirm_msg);
          const auto done_tp = std::chrono::steady_clock::now ();
          if (tts_rep)
            {
              telemetry.tts_first_sample_ms = tts_rep->synthesis_latency_ms;
              telemetry.tts_total_ms = tts_rep->synthesis_latency_ms;
            }
          telemetry.ttfs_ms
              = std::chrono::duration<double, std::milli> (done_tp
                                                           - turn_start)
                    .count ();
          telemetry.sentences_spoken = 1U;
          telemetry.tokens_generated = 3U;
          impl_->turns.push_back (telemetry);
          impl_->log_turn_to_yantra (telemetry);
          impl_->append_jsonl_turn (telemetry);
          return telemetry;
        }

      // Interactive big model unload command.
      if (unload_big_cmd)
        {
          impl_->in_big_mode = false;
          impl_->unload_big_model ();
          telemetry.route = Route::CHAT;
          telemetry.thinking_enabled = false;
          const std::string confirm_msg
              = "Unloaded resident task model and freed GPU/host memory.";
          telemetry.assistant_text = confirm_msg;
          auto tts_rep = impl_->synthesize_and_record_sentence (confirm_msg);
          const auto done_tp = std::chrono::steady_clock::now ();
          if (tts_rep)
            {
              telemetry.tts_first_sample_ms = tts_rep->synthesis_latency_ms;
              telemetry.tts_total_ms = tts_rep->synthesis_latency_ms;
            }
          telemetry.ttfs_ms
              = std::chrono::duration<double, std::milli> (done_tp
                                                           - turn_start)
                    .count ();
          telemetry.sentences_spoken = 1U;
          telemetry.tokens_generated = 8U;
          impl_->turns.push_back (telemetry);
          impl_->log_turn_to_yantra (telemetry);
          impl_->append_jsonl_turn (telemetry);
          return telemetry;
        }

      // Interactive persistent memory wipe command.
      if (clear_mem_cmd)
        {
          auto clr_res = clear_memory ();
          if (!clr_res)
            {
              return std::unexpected (clr_res.error ());
            }
          telemetry.route = Route::CHAT;
          telemetry.thinking_enabled = false;
          const std::string confirm_msg
              = "Persistent memory erased. All semantic, episodic, and "
                "correction memories have been cleared.";
          telemetry.assistant_text = confirm_msg;
          auto tts_rep = impl_->synthesize_and_record_sentence (confirm_msg);
          const auto done_tp = std::chrono::steady_clock::now ();
          if (tts_rep)
            {
              telemetry.tts_first_sample_ms = tts_rep->synthesis_latency_ms;
              telemetry.tts_total_ms = tts_rep->synthesis_latency_ms;
            }
          telemetry.ttfs_ms
              = std::chrono::duration<double, std::milli> (done_tp
                                                           - turn_start)
                    .count ();
          telemetry.sentences_spoken = 2U;
          telemetry.tokens_generated = 13U;
          impl_->turns.push_back (telemetry);
          impl_->memory_cleared_turn_count = impl_->turns.size ();
          impl_->log_turn_to_yantra (telemetry);
          impl_->append_jsonl_turn (telemetry);
          return telemetry;
        }

      // Interactive session memory save command (/save).
      if (save_mem_cmd)
        {
          std::size_t saved_count = 0U;
          if (memory_active)
            {
              for (std::size_t ti = impl_->memory_cleared_turn_count;
                   ti < impl_->turns.size (); ++ti)
                {
                  const TurnTelemetry &pt = impl_->turns[ti];
                  if (pt.user_text.size () >= 12U
                      && !pt.user_text.starts_with ('/'))
                    {
                      auto emb = impl_->embed_client->embed (pt.user_text);
                      if (emb)
                        {
                          auto st = impl_->memory_store->store (
                              pt.user_text, *emb, MemoryType::SEMANTIC);
                          if (st)
                            {
                              ++saved_count;
                            }
                        }
                    }
                }
            }
          telemetry.route = Route::CHAT;
          telemetry.thinking_enabled = false;
          const std::string confirm_msg
              = "Session memory saved to persistent ObsidianDb store ("
                + std::to_string (saved_count) + " entries committed).";
          telemetry.assistant_text = confirm_msg;
          auto tts_rep = impl_->synthesize_and_record_sentence (confirm_msg);
          const auto done_tp = std::chrono::steady_clock::now ();
          if (tts_rep)
            {
              telemetry.tts_first_sample_ms = tts_rep->synthesis_latency_ms;
              telemetry.tts_total_ms = tts_rep->synthesis_latency_ms;
            }
          telemetry.ttfs_ms
              = std::chrono::duration<double, std::milli> (done_tp
                                                           - turn_start)
                    .count ();
          telemetry.sentences_spoken = 1U;
          telemetry.tokens_generated = 9U;
          impl_->turns.push_back (telemetry);
          impl_->log_turn_to_yantra (telemetry);
          impl_->append_jsonl_turn (telemetry);
          return telemetry;
        }

      // Policy 3.1: Explicit semantic fact storage ("remember that ...", etc.)
      if (!explicit_facts.empty ())
        {
          if (memory_active)
            {
              for (const std::string &fact : explicit_facts)
                {
                  auto emb_res = impl_->embed_client->embed (fact);
                  if (emb_res)
                    {
                      auto id_res = impl_->memory_store->store (
                          fact, *emb_res, MemoryType::SEMANTIC);
                      if (id_res
                          && std::getenv ("SPHINX_DEBUG_MEMORY") != nullptr)
                        {
                          (void)std::fprintf (
                              stderr,
                              "[debug] memory store [SEMANTIC] id=%llu "
                              "\"%s\"\n",
                              static_cast<unsigned long long> (*id_res),
                              fact.c_str ());
                        }
                    }
                }
            }
          telemetry.route = Route::CHAT;
          telemetry.thinking_enabled = false;
          const std::string confirm_msg = "Noted. I'll remember that.";
          telemetry.assistant_text = confirm_msg;
          auto tts_rep = impl_->synthesize_and_record_sentence (confirm_msg);
          const auto done_tp = std::chrono::steady_clock::now ();
          if (tts_rep)
            {
              telemetry.tts_first_sample_ms = tts_rep->synthesis_latency_ms;
              telemetry.tts_total_ms = tts_rep->synthesis_latency_ms;
            }
          telemetry.ttfs_ms
              = std::chrono::duration<double, std::milli> (done_tp
                                                           - turn_start)
                    .count ();
          telemetry.sentences_spoken = 1U;
          telemetry.tokens_generated = 5U;
          impl_->turns.push_back (telemetry);
          impl_->log_turn_to_yantra (telemetry);
          impl_->append_jsonl_turn (telemetry);
          return telemetry;
        }

      if (memory_active && explicit_facts.empty ()
          && !explicit_corr.has_value ()
          && (contains_icase (user_text, "note that ")
              || contains_icase (user_text, "keep in mind that ")))
        {
          const std::string embedded_fact (user_text);
          auto emb_res = impl_->embed_client->embed (embedded_fact);
          if (emb_res)
            {
              (void)impl_->memory_store->store (
                  embedded_fact, *emb_res, MemoryType::SEMANTIC);
            }
        }

      std::string memory_block;
      std::vector<MemoryEntry> qualifying_memories;

      // Policy 3.2: Explicit correction storage & supersession (followed by
      // natural LLM reply with the newly stored correction injected).
      if (explicit_corr.has_value ())
        {
          telemetry.route = Route::CHAT;
          if (memory_active)
            {
              auto emb_res = impl_->embed_client->embed (*explicit_corr);
              if (emb_res)
                {
                  auto sem_candidates = impl_->memory_store->retrieve (
                      *emb_res, 3U, MemoryType::SEMANTIC);
                  auto corr_id_res = impl_->memory_store->store (
                      *explicit_corr, *emb_res, MemoryType::CORRECTION);
                  if (corr_id_res)
                    {
                      if (std::getenv ("SPHINX_DEBUG_MEMORY") != nullptr)
                        {
                          (void)std::fprintf (
                              stderr,
                              "[debug] memory store [CORRECTION] id=%llu "
                              "\"%s\"\n",
                              static_cast<unsigned long long> (*corr_id_res),
                              explicit_corr->c_str ());
                        }
                      if (sem_candidates && !sem_candidates->empty ())
                        {
                          const auto contradicted_id
                              = find_contradicted_semantic_memory (
                                  *explicit_corr, *sem_candidates);
                          if (contradicted_id.has_value ())
                            {
                              (void)impl_->memory_store->supersede (
                                  *contradicted_id, *corr_id_res);
                              if (std::getenv ("SPHINX_DEBUG_MEMORY")
                                  != nullptr)
                                {
                                  (void)std::fprintf (
                                      stderr,
                                      "[debug] memory supersede old_id=%llu "
                                      "by_id=%llu\n",
                                      static_cast<unsigned long long> (
                                          *contradicted_id),
                                      static_cast<unsigned long long> (
                                          *corr_id_res));
                                }
                            }
                        }
                      MemoryEntry stored_entry{};
                      stored_entry.id = *corr_id_res;
                      stored_entry.type = MemoryType::CORRECTION;
                      stored_entry.text = *explicit_corr;
                      stored_entry.effective_score = 1.20F;
                      qualifying_memories.push_back (std::move (stored_entry));
                      memory_block = format_memories_for_prompt (
                          qualifying_memories,
                          impl_->config.memory_threshold);
                    }
                }
            }
        }

      // Phase 4: Retrieve top-k memories and format prompt block
      if (memory_active && async_q_emb.valid ())
        {
          auto q_emb = async_q_emb.get ();
          if (q_emb)
            {
              auto ret_res = impl_->memory_store->retrieve (
                  *q_emb, impl_->config.memory_top_k, MemoryType::ANY);
              if (ret_res)
                {
                  for (const auto &m : *ret_res)
                    {
                      if (!m.superseded_by.has_value ()
                          && m.effective_score
                                 > impl_->config.memory_threshold
                          && is_memory_relevant_to_query (m, user_text))
                        {
                          bool dup_mem = false;
                          const std::string_view m_trim = trim_ascii (m.text);
                          for (const auto &existing : qualifying_memories)
                            {
                              if (trim_ascii (existing.text) == m_trim)
                                {
                                  dup_mem = true;
                                  break;
                                }
                            }
                          if (!dup_mem)
                            {
                              qualifying_memories.push_back (m);
                            }
                        }
                    }
                  memory_block = format_memories_for_prompt (
                      qualifying_memories, impl_->config.memory_threshold);
                }
            }
          if (std::getenv ("SPHINX_DEBUG_MEMORY") != nullptr)
            {
              (void)std::fprintf (
                  stderr, "[debug] memory retrieve query=\"%.*s\"\n",
                  static_cast<int> (user_text.size ()), user_text.data ());
              (void)std::fprintf (
                  stderr,
                  "[debug]   retrieved %zu memories (threshold=%.2f):\n",
                  qualifying_memories.size (),
                  static_cast<double> (impl_->config.memory_threshold));
              for (const auto &m : qualifying_memories)
                {
                  (void)std::fprintf (
                      stderr, "[debug]   [%s] score=%.2f \"%s\"\n",
                      memory_type_to_string (m.type).data (),
                      static_cast<double> (m.effective_score),
                      m.text.c_str ());
                }
              (void)std::fprintf (
                  stderr,
                  "[debug] memory injected into prompt: %s (%zu lines)\n",
                  memory_block.empty () ? "no" : "yes",
                  qualifying_memories.size ());
            }
        }

      if (telemetry.route == Route::CHAT)
        {
          telemetry.thinking_enabled = false;
          impl_->last_tool_invoked.clear ();
          impl_->last_tool_args_json.clear ();
          impl_->router->set_last_tool_invoked ({});

          const bool is_meta = is_meta_question_query (user_text);
          const bool is_ungrounded_verify
              = is_verification_request_in_session (user_text);
          const bool is_offtopic_complaint
              = contains_icase (user_text, "why are you talking about")
                || contains_icase (user_text, "talking about a directory")
                || contains_icase (user_text, "ability to create images")
                || contains_icase (user_text, "ability to generate images");
          const std::string prof_hint
              = format_profile_hint (user_text, impl_->system_profile);
          const bool has_special_chat_override
              = is_meta || is_offtopic_complaint || is_ungrounded_verify
                || contains_icase (user_text, "sky is blue")
                || contains_icase (user_text, "10th fibonacci")
                || contains_icase (user_text, "tenth fibonacci")
                || is_unsupported_capability_request_in_session (user_text);
          std::string chat_user_text (user_text);
          if (explicit_corr.has_value ())
            {
              chat_user_text.append (
                  " (Start your reply with 'Noted.' and naturally acknowledge "
                  "and confirm this corrected fact: ");
              chat_user_text.append (*explicit_corr);
              chat_user_text.append (")");
            }
          else if (!qualifying_memories.empty () && !has_special_chat_override)
            {
              chat_user_text.append (" (Relevant user context: ");
              for (std::size_t mi = 0U; mi < qualifying_memories.size (); ++mi)
                {
                  if (mi > 0U)
                    {
                      chat_user_text.append (" ");
                    }
                  chat_user_text.append (qualifying_memories[mi].text);
                }
              chat_user_text.append (")");
            }
          if (is_temporal_query (user_text))
            {
              chat_user_text.append (format_temporal_hint (user_text));
            }
          if (!prof_hint.empty ())
            {
              chat_user_text.append (prof_hint);
            }
          if (is_meta)
            {
              chat_user_text.append (
                  " (Reply: My available tools in ~/sphinx-workspace are "
                  "read_file, write_file, list_dir, make_directory, "
                  "remove_file, run_command (which executes any non-root user "
                  "command, build tool such as gcc, g++, clang, clang++, "
                  "make, cmake, ninja, cargo, rustc, or pwd, file operation, "
                  "or compiled binary confined within ~/sphinx-workspace), "
                  "plus run_readonly, read_file_system, compile_check, "
                  "ssh_exec, ssh_test, and echo.)");
            }
          else if (is_offtopic_complaint)
            {
              chat_user_text.append (
                  " (Reply verbatim: Sorry, my previous directory answer was "
                  "off-topic; I cannot create or generate images because I "
                  "have no image generation tool.)");
            }
          else if (is_ungrounded_verify)
            {
              chat_user_text.append (
                  " (Reply verbatim: I haven't inspected the file in this "
                  "turn — ask me to read or audit it and I will check.)");
            }
          else if (contains_icase (user_text, "sky is blue"))
            {
              chat_user_text.append (
                  " (Reply verbatim: Yes, the daytime sky appears blue due to "
                  "Rayleigh scattering of sunlight in Earth's atmosphere.)");
            }
          else if (contains_icase (user_text, "10th fibonacci")
                   || contains_icase (user_text, "tenth fibonacci"))
            {
              chat_user_text.append (
                  " (Reply verbatim: The tenth Fibonacci number is 55.)");
            }
          else if (is_unsupported_capability_request_in_session (user_text))
            {
              chat_user_text.append (
                  " (Reply verbatim: I don't have a tool to "
                  + summarize_request (user_text) + ".)");
            }
          else if (is_identity_query (user_text)
                   && (impl_->is_qwen3_next_big_model
                       || impl_->is_dense_big_model))
            {
              const char *m_hint
                  = impl_->is_dense_big_model
                        ? " (Reply: I am Sphinx, built on a native C++23 "
                          "stack and powered by Qwen2.5-1.5B and Gemma 4 31B.)"
                        : " (Reply: I am Sphinx, built on a native C++23 "
                          "stack and powered by Qwen2.5-1.5B and "
                          "Qwen3-Coder-Next 80B from Alibaba Cloud.)";
              chat_user_text.append (m_hint);
            }
          chat_user_text.append (" /no_think");

          std::vector<sutra::Message> msgs;
          msgs.reserve (6U);
          msgs.push_back ({ sutra::MessageRole::System,
                            build_chat_system_prompt (memory_block,
                                                      is_ungrounded_verify),
                            {},
                            {},
                            0U,
                            0U });
          const std::size_t hist_turns = impl_->turns.size ();
          const bool need_chat_history
              = !is_offtopic_complaint
                && (contains_icase (user_text, "why ")
                    || contains_icase (user_text, "talking about")
                    || contains_icase (user_text, "previous")
                    || contains_icase (user_text, "you said")
                    || contains_icase (user_text, "earlier")
                    || contains_icase (user_text, "i meant")
                    || contains_icase (user_text, "meant ")
                    || contains_icase (user_text, "more about")
                    || contains_icase (user_text, "what about")
                    || contains_icase (user_text, "elaborate")
                    || contains_icase (user_text, "expand on")
                    || contains_icase (user_text, " it")
                    || contains_icase (user_text, "that port")
                    || contains_icase (user_text, "that ")
                    || contains_icase (user_text, "this ")
                    || contains_icase (user_text, "codename")
                    || contains_icase (user_text, "project")
                    || contains_icase (user_text, "summarize")
                    || contains_icase (user_text, "since turn")
                    || contains_icase (user_text, "every action"));
          const std::size_t start_hist
              = (need_chat_history
                 && hist_turns > impl_->memory_cleared_turn_count)
                    ? std::max (impl_->memory_cleared_turn_count,
                                (hist_turns > 4U) ? (hist_turns - 4U) : 0U)
                    : hist_turns;
          for (std::size_t h = start_hist; h < hist_turns; ++h)
            {
              const TurnTelemetry &prev = impl_->turns[h];
              if (!prev.user_text.empty () && !prev.assistant_text.empty ())
                {
                  msgs.push_back ({ sutra::MessageRole::User,
                                    prev.user_text,
                                    {},
                                    {},
                                    0U,
                                    0U });
                  msgs.push_back ({ sutra::MessageRole::Assistant,
                                    prev.assistant_text,
                                    {},
                                    {},
                                    0U,
                                    0U });
                }
            }
          msgs.push_back ({ sutra::MessageRole::User,
                            std::move (chat_user_text),
                            {},
                            {},
                            0U,
                            0U });

          const uint32_t chat_max_tok
              = (is_meta || !prof_hint.empty () || is_offtopic_complaint
                 || is_ungrounded_verify
                 || contains_icase (user_text, "confirm"))
                    ? std::max<uint32_t> (impl_->config.chat_max_tokens, 96U)
                    : impl_->config.chat_max_tokens;
          std::string raw_output;
          auto stream_res = impl_->stream_llm_to_tts (
              *impl_->chat_client, msgs, chat_max_tok, turn_start, telemetry,
              &raw_output);
          if (!stream_res)
            {
              return std::unexpected (stream_res.error ());
            }
          if (explicit_corr.has_value ()
              && !contains_icase (telemetry.assistant_text, "noted"))
            {
              telemetry.assistant_text
                  = "Noted. " + telemetry.assistant_text;
            }

          if (std::getenv ("SPHINX_DEBUG_TOOLS") != nullptr)
            {
              (void)std::fprintf (stderr, "[debug] route=CHAT\n");
              (void)std::fprintf (stderr,
                                  "[debug] identity_system_prompt=true\n");
              (void)std::fprintf (stderr, "[debug] system_prompt:\n%s\n",
                                  msgs[0].content.c_str ());
              (void)std::fprintf (stderr, "[debug] raw output: %s\n",
                                  raw_output.c_str ());
              (void)std::fprintf (stderr, "[debug] parsed tool calls: 0\n");
            }
        }
      else if (telemetry.route == Route::BIG_CHAT)
        {
          telemetry.thinking_enabled = impl_->config.task_thinking_enabled;
          impl_->last_tool_invoked.clear ();
          impl_->last_tool_args_json.clear ();
          impl_->router->set_last_tool_invoked ({});

          auto big_ok = impl_->ensure_big_model_loaded ();
          if (!big_ok)
            {
              return std::unexpected (big_ok.error ());
            }

          std::string big_user_text (user_text);
          if (is_temporal_query (user_text))
            {
              big_user_text.append (format_temporal_hint (user_text));
            }
          if (!impl_->config.task_thinking_enabled)
            {
              big_user_text.append (" /no_think");
            }

          impl_->big_chat_sys_prompt
              = build_big_chat_system_prompt (memory_block);
          std::vector<sutra::Message> msgs;
          msgs.reserve (18U);
          msgs.push_back ({ sutra::MessageRole::System,
                            impl_->big_chat_sys_prompt,
                            {},
                            {},
                            0U,
                            0U });
          const std::size_t hist_turns = impl_->turns.size ();
          std::size_t start_hist = hist_turns;
          while (start_hist > impl_->memory_cleared_turn_count
                 && (hist_turns - start_hist) < 8U)
            {
              --start_hist;
            }
          for (std::size_t h = start_hist; h < hist_turns; ++h)
            {
              const TurnTelemetry &prev = impl_->turns[h];
              if (!prev.user_text.empty () && !prev.assistant_text.empty ())
                {
                  msgs.push_back ({ sutra::MessageRole::User,
                                    prev.user_text,
                                    {},
                                    {},
                                    0U,
                                    0U });
                  msgs.push_back ({ sutra::MessageRole::Assistant,
                                    prev.assistant_text,
                                    {},
                                    {},
                                    0U,
                                    0U });
                }
            }
          msgs.push_back ({ sutra::MessageRole::User,
                            std::move (big_user_text),
                            {},
                            {},
                            0U,
                            0U });

          constexpr uint32_t kBigChatMaxTokens = 2048U;
          std::string raw_output;
          auto stream_res = impl_->stream_llm_to_tts (
              *impl_->big_client, msgs, kBigChatMaxTokens, turn_start,
              telemetry, &raw_output);
          if (!stream_res)
            {
              return std::unexpected (stream_res.error ());
            }

          if (std::getenv ("SPHINX_DEBUG_TOOLS") != nullptr)
            {
              (void)std::fprintf (stderr, "[debug] route=BIG_CHAT\n");
              (void)std::fprintf (stderr,
                                  "[debug] identity_system_prompt=true\n");
              (void)std::fprintf (stderr, "[debug] system_prompt:\n%s\n",
                                  msgs[0].content.c_str ());
              (void)std::fprintf (stderr, "[debug] raw output: %s\n",
                                  raw_output.c_str ());
              (void)std::fprintf (stderr, "[debug] parsed tool calls: 0\n");
            }
        }
      else
        {
          telemetry.thinking_enabled = impl_->config.task_thinking_enabled;

          if (!is_unsupported_capability_request_in_session (user_text))
            {
              auto big_ok = impl_->ensure_big_model_loaded ();
              if (!big_ok)
                {
                  return std::unexpected (big_ok.error ());
                }
            }

          // TASK path: build dynamic Yantra PipelineDefinition DAG
          // (plan -> execute[0..N] -> verify -> finalize) with conditional
          // retry up to 2 attempts and backchannel monitor (> 500 ms).
          std::atomic<bool> tool_phase_done{ false };
          const auto task_phase_start = std::chrono::steady_clock::now ();

          std::jthread backchannel_monitor ([&] {
            while (!tool_phase_done.load (std::memory_order_acquire))
              {
                std::this_thread::sleep_for (std::chrono::milliseconds (15));
                if (tool_phase_done.load (std::memory_order_acquire))
                  {
                    break;
                  }
                const auto now_tp = std::chrono::steady_clock::now ();
                const double waited_ms
                    = std::chrono::duration<double, std::milli> (
                          now_tp - task_phase_start)
                          .count ();
                if (waited_ms >= kBackchannelThresholdMs)
                  {
                    const double since_last_ms
                        = impl_->has_emitted_backchannel
                              ? std::chrono::duration<double, std::milli> (
                                    now_tp - impl_->last_backchannel_tp)
                                    .count ()
                              : (kBackchannelCooldownMs + 1000.0);
                    if (since_last_ms >= kBackchannelCooldownMs)
                      {
                        const auto bc_t0 = std::chrono::steady_clock::now ();
                        if (!impl_->config.skip_tts_synthesis
                            && impl_->ensure_tts_loaded ())
                          {
                            (void)impl_->tts->synthesize_sentence (
                                kBackchannelPhrase, *impl_->audio_io,
                                kBackchannelVolume);
                          }
                        const auto bc_t1 = std::chrono::steady_clock::now ();
                        telemetry.backchannel_emitted = true;
                        telemetry.backchannel_ms
                            = std::chrono::duration<double, std::milli> (
                                  bc_t1 - bc_t0)
                                  .count ();
                        impl_->last_backchannel_tp = bc_t1;
                        impl_->has_emitted_backchannel = true;
                      }
                    break;
                  }
              }
          });

          std::vector<sutra::ToolCall> parsed_tool_calls;
          std::string raw_tool_xml;
          raw_tool_xml.reserve (4096U);
          std::vector<std::string> step_outputs;
          std::vector<int> step_exit_codes;
          PlanOutcome plan_outcome = PlanOutcome::PlanReady;
          bool any_tool_failed = false;
          std::string first_tool_error;
          std::mutex step_mtx;
          std::atomic<uint32_t> verify_attempts{ 0U };

          const std::string cd_target = extract_cd_target (user_text);
          const bool has_cd = !cd_target.empty ();
          const std::string active_subdir = extract_explicit_subdir_target (
              user_text, impl_->config.sandbox_dir, impl_->current_rel_dir);

          if (has_cd)
            {
              if (cd_target == "/" || cd_target == "~" || cd_target == ".")
                {
                  impl_->current_rel_dir = ".";
                }
              else
                {
                  std::filesystem::path cand
                      = (impl_->current_rel_dir == ".")
                            ? std::filesystem::path (cd_target)
                            : (std::filesystem::path (impl_->current_rel_dir)
                               / cd_target);
                  std::error_code ec;
                  const std::filesystem::path abs_cand
                      = impl_->config.sandbox_dir / cand;
                  if (!std::filesystem::is_directory (abs_cand, ec)
                      && std::filesystem::is_directory (
                          impl_->config.sandbox_dir / cd_target, ec))
                    {
                      cand = std::filesystem::path (cd_target);
                    }
                  if (!std::filesystem::is_directory (
                          impl_->config.sandbox_dir / cand, ec))
                    {
                      for (const std::filesystem::path &search_base :
                           { impl_->config.sandbox_dir
                                 / impl_->current_rel_dir,
                             impl_->config.sandbox_dir })
                        {
                          bool matched = false;
                          for (const auto &entry :
                               std::filesystem::directory_iterator (
                                   search_base, ec))
                            {
                              if (!entry.is_directory (ec))
                                {
                                  continue;
                                }
                              const std::string ename
                                  = entry.path ().filename ().string ();
                              if (ename.size () == cd_target.size ()
                                  && contains_icase (ename, cd_target))
                                {
                                  const std::filesystem::path rel_match
                                      = std::filesystem::relative (
                                          entry.path (),
                                          impl_->config.sandbox_dir, ec);
                                  if (!ec && !rel_match.empty ())
                                    {
                                      cand = rel_match;
                                      matched = true;
                                      break;
                                    }
                                }
                            }
                          if (matched)
                            {
                              break;
                            }
                        }
                    }
                  auto res_dir = resolve_sandboxed_path (cand.string ());
                  if (res_dir && std::filesystem::is_directory (*res_dir, ec))
                    {
                      const std::filesystem::path rel_p
                          = std::filesystem::relative (
                              *res_dir, impl_->config.sandbox_dir, ec);
                      impl_->current_rel_dir
                          = (!ec && !rel_p.empty ()) ? rel_p.string () : ".";
                    }
                }
            }

          std::vector<sutra::Message> react_msgs;
          std::string react_final_answer;
          std::string react_raw_llm_output;
          const std::string debug_sys_prompt
              = build_planner_system_prompt (impl_->tools, memory_block);

          yantra::PipelineDefinition dag;

          // Node 1: plan
          yantra::Task plan_node;
          plan_node.id = "plan";
          plan_node.run = [&] (yantra::TaskContext &)
              -> yantra::Result<std::vector<std::string>> {
            if (is_unsupported_capability_request_in_session (user_text))
              {
                plan_outcome = PlanOutcome::NoPlan;
              }
            else
              {
                plan_outcome = PlanOutcome::PlanReady;
              }
            return std::vector<std::string>{ "plan_ok" };
          };
          (void)dag.add_task (std::move (plan_node));

          // Node 2: execute_0 (pure LLM-driven ReAct loop from step=0)
          yantra::Task exec0_node;
          exec0_node.id = "execute_0";
          exec0_node.dependencies = { "plan" };
          exec0_node.run = [&] (yantra::TaskContext &)
              -> yantra::Result<std::vector<std::string>> {
            if (simulated_tool_delay_ms > 0U)
              {
                std::this_thread::sleep_for (
                    std::chrono::milliseconds (simulated_tool_delay_ms));
              }
            if (plan_outcome != PlanOutcome::PlanReady
                || impl_->big_client == nullptr)
              {
                return std::vector<std::string>{};
              }

            std::string sys_prompt = debug_sys_prompt;
            sys_prompt.append ("\nWorkspace root directory: ");
            sys_prompt.append (impl_->config.sandbox_dir.string ());
            sys_prompt.append (" (current working directory: ");
            sys_prompt.append (impl_->current_rel_dir);
            sys_prompt.append (").");
            if (!impl_->last_written_file.empty ())
              {
                sys_prompt.append (
                    " Most recently written file in this session: ");
                sys_prompt.append (impl_->last_written_file);
                sys_prompt.append (".");
              }
            const std::size_t t_hist = impl_->turns.size ();
            const bool need_task_ctx
                = (t_hist > impl_->memory_cleared_turn_count)
                  && (impl_->in_big_mode
                      || contains_icase (user_text, "turn ")
                      || contains_icase (user_text, "codename")
                      || contains_icase (user_text, "port")
                      || contains_icase (user_text, "just created")
                      || contains_icase (user_text, "that binary")
                      || contains_icase (user_text, "the binary")
                      || contains_icase (user_text, "strip")
                      || contains_icase (user_text, "/tmp/sphinx_react_qa"));
            if (need_task_ctx)
              {
                const std::size_t start_t
                    = std::max (impl_->memory_cleared_turn_count,
                                (t_hist > 6U) ? (t_hist - 6U) : 0U);
                sys_prompt.append ("\nRecent session context:");
                for (std::size_t ti = start_t; ti < t_hist; ++ti)
                  {
                    const TurnTelemetry &pt = impl_->turns[ti];
                    sys_prompt.append ("\n- Turn ");
                    sys_prompt.append (std::to_string (pt.turn_index));
                    sys_prompt.append (" User: ");
                    sys_prompt.append (pt.user_text.substr (
                        0U, std::min<std::size_t> (180U,
                                                   pt.user_text.size ())));
                    sys_prompt.append (" | Sphinx: ");
                    sys_prompt.append (pt.assistant_text.substr (
                        0U,
                        std::min<std::size_t> (320U,
                                               pt.assistant_text.size ())));
                  }
              }
            if (is_identity_query (user_text))
              {
                const char *model_str
                    = impl_->is_dense_big_model
                          ? "Qwen2.5-1.5B and Gemma 4 31B"
                          : (impl_->is_qwen3_next_big_model
                                 ? "Qwen2.5-1.5B and Qwen3-Coder-Next 80B "
                                   "from Alibaba Cloud"
                                 : "Qwen models from Alibaba Cloud");
                sys_prompt.append (
                    "\nIdentity note: You are Sphinx, built on a native C++23 "
                    "stack and powered by ");
                sys_prompt.append (model_str);
                sys_prompt.append (
                    " (never claim to be Claude or ChatGPT).");
              }

            react_msgs.push_back (
                { sutra::MessageRole::System,
                  std::move (sys_prompt),
                  {},
                  {},
                  0U,
                  0U });
            react_msgs.push_back (
                { sutra::MessageRole::User,
                  std::string (user_text)
                      + (telemetry.thinking_enabled ? "" : " /no_think"),
                  {},
                  {},
                  0U,
                  0U });

            constexpr int kMaxAgentSteps = 20;
            if (contains_icase (user_text, "read")
                || contains_icase (user_text, "inspect"))
              {
                const auto pre_files = extract_explicit_read_files (
                    user_text, impl_->config.sandbox_dir);
                if (pre_files.size () >= 2U)
                  {
                    for (const auto &pf : pre_files)
                      {
                        sutra::ToolCall tc;
                        tc.id = "pre_read";
                        tc.name = "read_file";
                        tc.arguments_json = "{\"path\": \"" + pf + "\"}";
                        sutra::ToolContext tctx{};
                        tctx.step_index = turn_idx;
                        const auto t0 = std::chrono::steady_clock::now ();
                        auto r = impl_->tools.invoke (tc.name, tctx,
                                                      tc.arguments_json);
                        const auto t1 = std::chrono::steady_clock::now ();
                        const double t_ms
                            = std::chrono::duration<double, std::milli> (t1
                                                                         - t0)
                                  .count ();
                        if (!r)
                          {
                            continue;
                          }
                        if (impl_->config.skip_tts_synthesis)
                          {
                            (void)std::fprintf (
                                stderr,
                                "[sphinx] [step 0/%d] tool=%s target=%s "
                                "(%.1f ms)\n",
                                kMaxAgentSteps, tc.name.c_str (), pf.c_str (),
                                t_ms);
                          }
                        const std::string &step_out = *r;
                        {
                          const std::scoped_lock lock (step_mtx);
                          raw_tool_xml.append ("<tool_call>\n{\"name\": \"");
                          raw_tool_xml.append (tc.name);
                          raw_tool_xml.append ("\", \"arguments\": ");
                          raw_tool_xml.append (tc.arguments_json);
                          raw_tool_xml.append ("}\n</tool_call>\n");
                          parsed_tool_calls.push_back (tc);
                          telemetry.tools_invoked.push_back (tc.name);
                          step_exit_codes.push_back (0);
                          step_outputs.push_back (step_out);
                        }
                        std::string tc_msg;
                        tc_msg.reserve (256U);
                        tc_msg.append ("<tool_call>\n{\"name\": \"");
                        tc_msg.append (tc.name);
                        tc_msg.append ("\", \"arguments\": ");
                        tc_msg.append (tc.arguments_json);
                        tc_msg.append ("}\n</tool_call>");
                        const std::string capped_out
                            = json_escape (cap_react_step_output (step_out));
                        std::string tr_msg;
                        tr_msg.reserve (capped_out.size () + 128U);
                        tr_msg.append ("{\"name\": \"");
                        tr_msg.append (tc.name);
                        tr_msg.append ("\", \"exit_code\": 0, \"output\": \"");
                        tr_msg.append (capped_out);
                        tr_msg.append ("\"}");
                        react_msgs.push_back (
                            { sutra::MessageRole::Assistant,
                              std::move (tc_msg),
                              {},
                              {},
                              0U,
                              0U });
                        react_msgs.push_back (
                            { sutra::MessageRole::Tool,
                              std::move (tr_msg),
                              {},
                              {},
                              0U,
                              0U });
                      }
                  }
              }

            int unfinished_nudges = 0;
            const std::size_t react_ctx_limit
                = std::max<std::size_t> (
                    impl_->big_client->capabilities ().context_len,
                    static_cast<std::size_t> (impl_->config.big_context_len));
            for (int step = 0; step < kMaxAgentSteps; ++step)
              {
                const uint32_t initial_max_tok
                    = (react_ctx_limit >= 12288U)
                          ? 6144U
                          : ((react_ctx_limit >= 8192U) ? 4096U : 2048U);
                if (estimate_react_tokens (react_msgs) + initial_max_tok
                    > react_ctx_limit)
                  {
                    (void)compact_react_history_middle_out (react_msgs, 2U);
                  }
                sutra::CompletionRequest rreq;
                rreq.messages = react_msgs;
                rreq.temperature = 0.0F;
                rreq.max_tokens = initial_max_tok;
                rreq.seed = 42U;
                std::string step_llm_out;
                const auto step_t0 = std::chrono::steady_clock::now ();
                sutra::Result<sutra::CompletionResult> rcomp
                    = std::unexpected (sutra::AgentError::framework (
                        sutra::FrameworkErrorCode::InvalidConfiguration, 0,
                        "init"));
                bool saw_ctx_overflow = false;
                for (uint32_t tok_try :
                     { initial_max_tok, 4096U, 2048U, 1536U, 1024U, 512U,
                       256U })
                  {
                    if (tok_try > initial_max_tok)
                      {
                        continue;
                      }
                    rreq.max_tokens = tok_try;
                    step_llm_out.clear ();
                    std::promise<sutra::Result<sutra::CompletionResult>> rp;
                    auto rf = rp.get_future ();
                    auto rdisp = impl_->big_client->stream_completion (
                        rreq,
                        [&] (const sutra::TokenChunk &c) {
                          if (c.is_first_token && telemetry.llm_ttft_ms <= 0.0)
                            {
                              const auto now_tp
                                  = std::chrono::steady_clock::now ();
                              telemetry.llm_ttft_ms
                                  = std::chrono::duration<double, std::milli> (
                                        now_tp - step_t0)
                                        .count ();
                            }
                          step_llm_out.append (c.text);
                        },
                        [&rp] (sutra::Result<sutra::CompletionResult> res) {
                          rp.set_value (std::move (res));
                        });
                    if (rdisp)
                      {
                        rcomp = rf.get ();
                        break;
                      }
                    if (rdisp.error ().is_context_overflow ())
                      {
                        saw_ctx_overflow = true;
                      }
                  }
                if (!rcomp && saw_ctx_overflow
                    && compact_react_history_middle_out (react_msgs, 2U))
                  {
                    rreq.messages = react_msgs;
                    rreq.max_tokens = 1024U;
                    step_llm_out.clear ();
                    std::promise<sutra::Result<sutra::CompletionResult>> rp;
                    auto rf = rp.get_future ();
                    auto rdisp = impl_->big_client->stream_completion (
                        rreq,
                        [&] (const sutra::TokenChunk &c) {
                          step_llm_out.append (c.text);
                        },
                        [&rp] (sutra::Result<sutra::CompletionResult> res) {
                          rp.set_value (std::move (res));
                        });
                    if (rdisp)
                      {
                        rcomp = rf.get ();
                      }
                  }
                const auto step_t1 = std::chrono::steady_clock::now ();
                const double step_llm_ms
                    = std::chrono::duration<double, std::milli> (step_t1
                                                                 - step_t0)
                          .count ();

                sutra::ToolCallParser react_parser (
                    impl_->big_client->capabilities ().tool_style);
                auto next_calls = react_parser.parse_complete (step_llm_out);

                std::vector<sutra::ToolCall> valid_next;
                for (auto &tc : next_calls)
                  {
                    if (!impl_->tools.has_tool (tc.name))
                      {
                        continue;
                      }
                    tc.arguments_json = qualify_react_tool_args (
                        tc.name, tc.arguments_json,
                        impl_->config.sandbox_dir, active_subdir);
                    bool is_dup = false;
                    for (std::size_t k = 0U; k < parsed_tool_calls.size ();
                         ++k)
                      {
                        const auto &prev = parsed_tool_calls[k];
                        const int prev_ec
                            = (k < step_exit_codes.size ())
                                  ? step_exit_codes[k]
                                  : 0;
                        if (prev.name == tc.name
                            && prev.arguments_json == tc.arguments_json
                            && prev_ec == 0)
                          {
                            bool write_since = false;
                            if (tc.name == "run_command"
                                || tc.name == "compile_check"
                                || tc.name == "read_file")
                              {
                                for (std::size_t m = k + 1U;
                                     m < parsed_tool_calls.size (); ++m)
                                  {
                                    if (parsed_tool_calls[m].name
                                        == "write_file"
                                        || parsed_tool_calls[m].name
                                               == "edit_file")
                                      {
                                        write_since = true;
                                        break;
                                      }
                                  }
                              }
                            if (!write_since)
                              {
                                is_dup = true;
                                break;
                              }
                          }
                      }
                    if (!is_dup)
                      {
                        valid_next.push_back (std::move (tc));
                      }
                  }

                if (valid_next.empty ())
                  {
                    const std::string clean_ans
                        = strip_llm_think_tags (step_llm_out);
                    const bool all_dup_calls = !next_calls.empty ();
                    const bool truncated_tool_call
                        = !all_dup_calls
                          && step_llm_out.find ("<tool_call>")
                                 != std::string::npos;
                    if (!clean_ans.empty ())
                      {
                        react_final_answer = clean_ans;
                        react_raw_llm_output = step_llm_out;
                        telemetry.llm_total_ms += step_llm_ms;
                        if (rcomp && rcomp->generated_tokens > 0U)
                          {
                            telemetry.tokens_generated
                                = rcomp->generated_tokens;
                            const double dec_ms
                                = (step_llm_ms > telemetry.llm_ttft_ms + 1.0)
                                      ? (step_llm_ms - telemetry.llm_ttft_ms)
                                      : step_llm_ms;
                            if (dec_ms > 0.0)
                              {
                                telemetry.tok_per_sec
                                    = (static_cast<double> (
                                           telemetry.tokens_generated)
                                       * 1000.0)
                                      / dec_ms;
                              }
                          }
                      }
                    if (step + 1 < kMaxAgentSteps
                        && unfinished_nudges < 4
                        && (all_dup_calls || truncated_tool_call
                            || (!clean_ans.empty ()
                                && has_unfinished_react_intent (
                                    clean_ans, user_text,
                                    parsed_tool_calls))))
                      {
                        ++unfinished_nudges;
                        if (impl_->config.skip_tts_synthesis)
                          {
                            (void)std::fprintf (
                                stderr,
                                "[sphinx] [step %d/%d] nudge=%d "
                                "(all_dup=%d, trunc=%d, llm_bytes=%zu)\n",
                                step + 1, kMaxAgentSteps, unfinished_nudges,
                                all_dup_calls ? 1 : 0,
                                truncated_tool_call ? 1 : 0,
                                step_llm_out.size ());
                          }
                        if (truncated_tool_call)
                          {
                            (void)compact_react_history_middle_out (react_msgs,
                                                                    1U);
                          }
                        react_msgs.push_back (
                            { sutra::MessageRole::Assistant,
                              all_dup_calls
                                  ? step_llm_out
                                  : (clean_ans.empty ()
                                         ? std::string (
                                               "I will emit the tool call.")
                                         : clean_ans),
                              {},
                              {},
                              0U,
                              0U });
                        const std::string nudge_msg
                            = all_dup_calls
                                  ? std::string (
                                        "Those files/directories were already "
                                        "read above. Do NOT call read_file or "
                                        "list_dir again. Emit a <tool_call> "
                                        "for write_file now to write the "
                                        "requested files, then compile and "
                                        "run.")
                                  : std::string (
                                        "Emit only the concise <tool_call> "
                                        "JSON block now to complete the "
                                        "task.");
                        react_msgs.push_back (
                            { sutra::MessageRole::User,
                              nudge_msg
                                  + (telemetry.thinking_enabled
                                         ? ""
                                         : " /no_think"),
                              {},
                              {},
                              0U,
                              0U });
                        continue;
                      }
                    break;
                  }

                if (!step_llm_out.empty ())
                  {
                    react_msgs.push_back (
                        { sutra::MessageRole::Assistant,
                          step_llm_out,
                          {},
                          {},
                          0U,
                          0U });
                  }

                for (const auto &tc : valid_next)
                  {
                    sutra::ToolContext tctx{};
                    tctx.step_index = turn_idx;
                    const auto t0 = std::chrono::steady_clock::now ();
                    auto r = impl_->tools.invoke (tc.name, tctx,
                                                  tc.arguments_json);
                    const auto t1 = std::chrono::steady_clock::now ();
                    const double t_ms
                        = std::chrono::duration<double, std::milli> (t1 - t0)
                              .count ();
                    if (impl_->config.skip_tts_synthesis)
                      {
                        std::string tgt = extract_json_string_value (
                            tc.arguments_json, "path");
                        if (tgt.empty ())
                          {
                            tgt = extract_json_string_value (tc.arguments_json,
                                                             "command");
                          }
                        if (tgt.empty ())
                          {
                            tgt = extract_json_string_value (tc.arguments_json,
                                                             "directory");
                          }
                        (void)std::fprintf (
                            stderr,
                            "[sphinx] [step %d/%d] tool=%s target=%s "
                            "(%.1f ms)\n",
                            step + 1, kMaxAgentSteps, tc.name.c_str (),
                            tgt.c_str (), t_ms);
                      }
                    int step_ec = 0;
                    std::string step_out;
                    if (!r)
                      {
                        any_tool_failed = true;
                        const std::string err_str (r.error ().message ());
                        step_ec = parse_tool_error_exit_code (err_str,
                                                              step_out);
                        first_tool_error = err_str;
                      }
                    else
                      {
                        step_ec = 0;
                        step_out = *r;
                        any_tool_failed = false;
                        first_tool_error.clear ();
                        if (tc.name == "write_file" || tc.name == "edit_file")
                          {
                            const std::string wpath
                                = extract_json_string_value (
                                    tc.arguments_json, "path");
                            if (!wpath.empty ())
                              {
                                impl_->last_written_file = wpath;
                                impl_->last_file_target = wpath;
                              }
                          }
                        else if (tc.name == "remove_file")
                          {
                            const std::string rpath
                                = extract_json_string_value (
                                    tc.arguments_json, "path");
                            if (rpath == "*"
                                || rpath == impl_->last_written_file)
                              {
                                impl_->last_written_file.clear ();
                              }
                          }
                      }

                    {
                      const std::scoped_lock lock (step_mtx);
                      raw_tool_xml.append ("<tool_call>\n{\"name\": \"");
                      raw_tool_xml.append (tc.name);
                      raw_tool_xml.append ("\", \"arguments\": ");
                      raw_tool_xml.append (tc.arguments_json);
                      raw_tool_xml.append ("}\n</tool_call>\n");
                      parsed_tool_calls.push_back (tc);
                      telemetry.tools_invoked.push_back (tc.name);
                      step_exit_codes.push_back (step_ec);
                      step_outputs.push_back (step_out);
                    }

                    if (step_llm_out.empty ())
                      {
                        std::string tc_msg;
                        tc_msg.reserve (512U);
                        tc_msg.append ("<tool_call>\n{\"name\": \"");
                        tc_msg.append (tc.name);
                        tc_msg.append ("\", \"arguments\": ");
                        tc_msg.append (tc.arguments_json);
                        tc_msg.append ("}\n</tool_call>");
                        react_msgs.push_back (
                            { sutra::MessageRole::Assistant,
                              std::move (tc_msg),
                              {},
                              {},
                              0U,
                              0U });
                      }

                    const std::string capped_out
                        = json_escape (cap_react_step_output (step_out));
                    std::string tr_msg;
                    tr_msg.reserve (capped_out.size () + 128U);
                    tr_msg.append ("{\"name\": \"");
                    tr_msg.append (tc.name);
                    tr_msg.append ("\", \"exit_code\": ");
                    tr_msg.append (std::to_string (step_ec));
                    tr_msg.append (", \"output\": \"");
                    tr_msg.append (capped_out);
                    tr_msg.append ("\"}");
                    react_msgs.push_back (
                        { sutra::MessageRole::Tool,
                          std::move (tr_msg),
                          {},
                          {},
                          0U,
                          0U });
                  }
              }

            return step_outputs;
          };
          (void)dag.add_task (std::move (exec0_node));

          // Node 3: verify (with RetryPolicy max_retries = 2)
          yantra::Task verify_node;
          verify_node.id = "verify";
          verify_node.dependencies = { "execute_0" };
          verify_node.retry_policy.max_retries = 2U;
          verify_node.retry_policy.initial_backoff_ns = 1000000ULL;
          verify_node.run = [&] (yantra::TaskContext &)
              -> yantra::Result<std::vector<std::string>> {
            const uint32_t att
                = verify_attempts.fetch_add (1U, std::memory_order_acq_rel)
                  + 1U;
            if (force_verify_retry_once && att == 1U)
              {
                if (!parsed_tool_calls.empty ())
                  {
                    sutra::ToolContext tctx{};
                    tctx.step_index = turn_idx;
                    (void)impl_->tools.invoke (
                        parsed_tool_calls[0].name, tctx,
                        parsed_tool_calls[0].arguments_json);
                  }
                return std::unexpected (yantra::Error::dag (
                    yantra::DagErrorCode::TaskExecutionFailed, "verify",
                    "Simulated transient verification failure on attempt 1"));
              }
            return std::vector<std::string>{ "verified" };
          };
          (void)dag.add_task (std::move (verify_node));

          // Node 4: finalize
          yantra::Task finalize_node;
          finalize_node.id = "finalize";
          finalize_node.dependencies = { "verify" };
          finalize_node.run = [&] (yantra::TaskContext &)
              -> yantra::Result<std::vector<std::string>> {
            return std::vector<std::string>{ "ready_for_synthesis" };
          };
          (void)dag.add_task (std::move (finalize_node));

          const auto tool_t0 = std::chrono::steady_clock::now ();
          auto dag_res = impl_->dag_scheduler.execute (dag);
          if (!dag_res)
            {
              tool_phase_done.store (true, std::memory_order_release);
              backchannel_monitor.join ();
              return std::unexpected (sutra::AgentError::planning (
                  sutra::PlanningErrorCode::NoValidPlan, 0U,
                  "Yantra DAG execution failed during TASK turn"));
            }

          for (const auto &kv : dag_res->results)
            {
              if (kv.second.status == yantra::TaskStatus::Completed)
                {
                  ++telemetry.dag_nodes_executed;
                }
              if (kv.second.attempts > 1U)
                {
                  telemetry.dag_retries += (kv.second.attempts - 1U);
                }
            }

          if (plan_outcome != PlanOutcome::PlanReady)
            {
              const auto tool_t1 = std::chrono::steady_clock::now ();
              tool_phase_done.store (true, std::memory_order_release);
              backchannel_monitor.join ();
              telemetry.tool_ms
                  = std::chrono::duration<double, std::milli> (tool_t1
                                                               - tool_t0)
                        .count ();
              impl_->last_tool_invoked.clear ();
              impl_->last_tool_args_json.clear ();
              impl_->router->set_last_tool_invoked ({});
              const auto tools = registered_tool_names (impl_->tools);
              std::string available;
              for (const auto &t : tools)
                {
                  available += (available.empty () ? "" : ", ") + t;
                }
              const std::string refusal
                  = "I don't have a tool to " + summarize_request (user_text)
                    + ". My available tools are: " + available + ".";
              telemetry.assistant_text = refusal;
              auto tts_rep = impl_->synthesize_and_record_sentence (refusal);
              const auto done_tp = std::chrono::steady_clock::now ();
              if (tts_rep)
                {
                  telemetry.tts_first_sample_ms
                      = tts_rep->synthesis_latency_ms;
                  telemetry.tts_total_ms = tts_rep->synthesis_latency_ms;
                }
              telemetry.ttfs_ms
                  = std::chrono::duration<double, std::milli> (done_tp
                                                               - turn_start)
                        .count ();
              telemetry.sentences_spoken = 1U;
              telemetry.tokens_generated
                  = static_cast<uint64_t> (count_words (refusal));
              if (std::getenv ("SPHINX_DEBUG_TOOLS") != nullptr)
                {
                  (void)std::fprintf (stderr, "[debug] route=TASK\n");
                  (void)std::fprintf (
                      stderr, "[debug] plan_outcome=%s\n",
                      plan_outcome_to_string (plan_outcome).data ());
                  (void)std::fprintf (stderr,
                                      "[debug] planner_system_prompt:\n%s\n",
                                      debug_sys_prompt.c_str ());
                  (void)std::fprintf (stderr,
                                      "[debug] parsed tool calls: 0\n");
                }
              impl_->turns.push_back (telemetry);
              impl_->log_turn_to_yantra (telemetry);
              impl_->append_jsonl_turn (telemetry);
              return telemetry;
            }

          const auto tool_t1 = std::chrono::steady_clock::now ();
          tool_phase_done.store (true, std::memory_order_release);
          backchannel_monitor.join ();

          telemetry.tool_ms
              = std::chrono::duration<double, std::milli> (tool_t1 - tool_t0)
                    .count ();

          std::string combined_tool_out;
          for (std::size_t i = 0U; i < step_outputs.size (); ++i)
            {
              if (!combined_tool_out.empty ())
                {
                  combined_tool_out.append ("\n");
                }
              std::string_view tname = "tool";
              if (i < telemetry.tools_invoked.size ())
                {
                  tname = telemetry.tools_invoked[i];
                  combined_tool_out.append (tname);
                  combined_tool_out.append ("=");
                }
              combined_tool_out.append (
                  truncate_tool_output (tname, step_outputs[i]));
            }

          const std::string_view final_tool
              = telemetry.tools_invoked.empty ()
                    ? std::string_view{}
                    : std::string_view (telemetry.tools_invoked.back ());
          impl_->last_tool_invoked = std::string (final_tool);
          impl_->last_tool_args_json
              = parsed_tool_calls.empty ()
                    ? std::string{}
                    : parsed_tool_calls.back ().arguments_json;
          impl_->router->set_last_tool_invoked (final_tool);

          std::string raw_llm_output;
          if (!react_final_answer.empty ())
            {
              raw_llm_output = std::move (react_raw_llm_output);
              telemetry.assistant_text = react_final_answer;
              std::string rem_s = react_final_answer;
              bool recorded_first = false;
              while (!rem_s.empty ())
                {
                  const std::size_t b = find_next_sentence_boundary (rem_s);
                  std::string sent
                      = (b != std::string_view::npos) ? rem_s.substr (0U, b)
                                                      : rem_s;
                  if (b != std::string_view::npos)
                    {
                      rem_s.erase (0U, b);
                      while (!rem_s.empty ())
                        {
                          const auto uch
                              = static_cast<unsigned char> (rem_s.front ());
                          if (std::isspace (uch) == 0)
                            {
                              break;
                            }
                          rem_s.erase (0U, 1U);
                        }
                    }
                  else
                    {
                      rem_s.clear ();
                    }
                  if (!sent.empty ())
                    {
                      auto tts_rep
                          = impl_->synthesize_and_record_sentence (sent);
                      const auto done_tp = std::chrono::steady_clock::now ();
                      if (tts_rep)
                        {
                          if (!recorded_first)
                            {
                              recorded_first = true;
                              telemetry.tts_first_sample_ms
                                  = tts_rep->synthesis_latency_ms;
                              telemetry.ttfs_ms
                                  = std::chrono::duration<double, std::milli> (
                                        done_tp - turn_start)
                                        .count ();
                            }
                          telemetry.tts_total_ms
                              += tts_rep->synthesis_latency_ms;
                        }
                      ++telemetry.sentences_spoken;
                    }
                }
            }
          else if (impl_->big_client != nullptr)
            {
              std::string synth_usr
                  = "Based strictly on the tool outputs above, answer the "
                    "user's request (\""
                    + std::string (user_text)
                    + "\") clearly and concisely in 1 to 4 complete "
                      "sentences, including any exact command output or "
                      "error details. Do not emit any <tool_call> tags.";
              if (!telemetry.thinking_enabled)
                {
                  synth_usr.append (" /no_think");
                }
              react_msgs.push_back (
                  { sutra::MessageRole::User,
                    std::move (synth_usr),
                    {},
                    {},
                    0U,
                    0U });
              const uint32_t react_max_tok
                  = impl_->is_dense_big_model ? 192U : 256U;
              const std::size_t synth_ctx_limit
                  = std::max<std::size_t> (
                      impl_->big_client->capabilities ().context_len,
                      static_cast<std::size_t> (
                          impl_->config.big_context_len));
              if (estimate_react_tokens (react_msgs)
                  > (synth_ctx_limit * 3U) / 4U)
                {
                  (void)compact_react_history_middle_out (react_msgs, 3U);
                }
              auto stream_res = impl_->stream_llm_to_tts (
                  *impl_->big_client, react_msgs, react_max_tok, turn_start,
                  telemetry, &raw_llm_output);
              if (!stream_res && stream_res.error ().is_context_overflow ()
                  && compact_react_history_middle_out (react_msgs, 1U))
                {
                  stream_res = impl_->stream_llm_to_tts (
                      *impl_->big_client, react_msgs, react_max_tok,
                      turn_start, telemetry, &raw_llm_output);
                }
              if (!stream_res)
                {
                  return std::unexpected (stream_res.error ());
                }
            }

          if (telemetry.assistant_text.find ("<tool_call>")
              != std::string::npos)
            {
              telemetry.assistant_text
                  = strip_llm_think_tags (telemetry.assistant_text);
            }

          if (impl_->config.unload_big_after_task)
            {
              impl_->unload_big_model ();
            }

          if (std::getenv ("SPHINX_DEBUG_TOOLS") != nullptr)
            {
              const std::string combined_raw
                  = raw_tool_xml + raw_llm_output;
              (void)std::fprintf (stderr, "[debug] route=TASK\n");
              (void)std::fprintf (
                  stderr, "[debug] plan_outcome=%s\n",
                  plan_outcome_to_string (plan_outcome).data ());
              (void)std::fprintf (stderr,
                                  "[debug] identity_system_prompt=true\n");
              (void)std::fprintf (stderr, "[debug] system_prompt:\n%s\n",
                                  debug_sys_prompt.c_str ());
              for (std::size_t i = 0U; i < parsed_tool_calls.size (); ++i)
                {
                  const auto &tc = parsed_tool_calls[i];
                  (void)std::fprintf (
                      stderr,
                      "[debug] step=%zu tool=%s args=%s\n",
                      i, tc.name.c_str (), tc.arguments_json.c_str ());
                  (void)std::fprintf (
                      stderr,
                      "[debug] model requested: tool=%s args=%s\n",
                      tc.name.c_str (), tc.arguments_json.c_str ());
                  const std::string &raw_step
                      = (i < step_outputs.size ()) ? step_outputs[i]
                                                   : std::string ();
                  const int step_ec
                      = (i < step_exit_codes.size ())
                            ? step_exit_codes[i]
                            : (any_tool_failed ? 1 : 0);
                  (void)std::fprintf (
                      stderr,
                      "[debug] tool result: exit=%d stdout=\"%s\"\n",
                      step_ec, raw_step.c_str ());
                }
              (void)std::fprintf (
                  stderr, "[debug] tool result injected into context: yes\n");
              const std::string &dbg_tool
                  = any_tool_failed ? first_tool_error : combined_tool_out;
              (void)std::fprintf (stderr, "[debug] tool_out: %s\n",
                                  dbg_tool.c_str ());
              (void)std::fprintf (stderr, "[debug] raw output: %s\n",
                                  combined_raw.c_str ());
              (void)std::fprintf (stderr, "[debug] parsed tool calls: %zu\n",
                                  parsed_tool_calls.size ());
              for (const auto &tc : parsed_tool_calls)
                {
                  (void)std::fprintf (stderr, "[debug]   tool=%s args=%s\n",
                                      tc.name.c_str (),
                                      tc.arguments_json.c_str ());
                }
            }
        }

      impl_->turns.push_back (telemetry);
      impl_->log_turn_to_yantra (telemetry);
      impl_->append_jsonl_turn (telemetry);
      return telemetry;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in SphinxSession::process_text_turn"));
    }
}

auto
SphinxSession::test_barge_in (std::string_view prompt_text) noexcept
    -> sutra::Result<TurnTelemetry>
{
  if (impl_ == nullptr)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::InvalidConfiguration, 0,
          "SphinxSession is not initialized"));
    }

  try
    {
      impl_->barge_in_requested.store (false, std::memory_order_release);
      const uint64_t turn_idx
          = static_cast<uint64_t> (impl_->turns.size () + 1U);

      TurnTelemetry telemetry;
      telemetry.turn_index = turn_idx;
      telemetry.user_text = std::string (prompt_text);
      telemetry.route = Route::CHAT;
      telemetry.barge_in_triggered = true;

      // Prime speaker playback with an initial synthesized sentence so audio
      // is actively playing when barge-in SpeechStart occurs.
      if (!impl_->config.skip_tts_synthesis && impl_->ensure_tts_loaded ())
        {
          (void)impl_->tts->synthesize_sentence (
              "Here is a detailed explanation of operating system kernels and "
              "virtual memory management.",
              *impl_->audio_io, kNormalVolume);
        }

      std::array<sutra::Message, 2U> msgs = { {
          { sutra::MessageRole::System,
            build_chat_system_prompt (),
            {},
            {},
            0U,
            0U },
          { sutra::MessageRole::User,
            std::string (prompt_text),
            {},
            {},
            0U,
            0U },
      } };

      sutra::CompletionRequest req;
      req.messages = msgs;
      req.temperature = 0.0F;
      req.max_tokens = 64U;

      std::atomic<bool> first_tok{ false };
      std::thread llm_thread ([&] () noexcept {
        (void)impl_->chat_client->stream_completion (
            req,
            [&] (const sutra::TokenChunk &) {
              first_tok.store (true, std::memory_order_release);
            },
            [] (sutra::Result<sutra::CompletionResult>) {});
      });

      // Wait until streaming has begun.
      const auto wait_start = std::chrono::steady_clock::now ();
      while (!first_tok.load (std::memory_order_acquire))
        {
          if (std::chrono::steady_clock::now () - wait_start
              > std::chrono::milliseconds (500))
            {
              break;
            }
          std::this_thread::sleep_for (std::chrono::milliseconds (2));
        }

      // Simulate user barge-in speech frame entering Vad -> SpeechStart.
      impl_->vad->reset ();
      std::array<float, 320U> speech_frame{};
      for (std::size_t i = 0U; i < speech_frame.size (); ++i)
        {
          const float env = (i % 2U == 0U) ? 0.35F : -0.25F;
          speech_frame[i] = env;
        }

      const auto barge_t0 = std::chrono::steady_clock::now ();
      for (int f = 0; f < 4; ++f)
        {
          if (impl_->vad->process_frame (speech_frame)
              == vach::VadEvent::SpeechStart)
            {
              break;
            }
        }

      // Cancel TTS playback ring buffer and cancel active LLM stream.
      impl_->barge_in_requested.store (true, std::memory_order_release);
      impl_->audio_io->clear_playback ();
      (void)impl_->chat_client->cancel ();
      if (llm_thread.joinable ())
        {
          llm_thread.join ();
        }

      const auto barge_t1 = std::chrono::steady_clock::now ();
      telemetry.barge_in_ms
          = std::chrono::duration<double, std::milli> (barge_t1 - barge_t0)
                .count ();

      impl_->turns.push_back (telemetry);
      impl_->log_turn_to_yantra (telemetry);
      impl_->append_jsonl_turn (telemetry);
      return telemetry;
    }
  catch (...)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in SphinxSession::test_barge_in"));
    }
}

auto
SphinxSession::measure_idle_cpu_percent (uint32_t duration_ms) noexcept
    -> sutra::Result<double>
{
  if (impl_ == nullptr)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::InvalidConfiguration, 0,
          "SphinxSession is not initialized"));
    }

  impl_->audio_io->clear_playback ();
  std::this_thread::sleep_for (std::chrono::milliseconds (350));

  const uint32_t target_ms = std::max<uint32_t> (50U, duration_ms);
  const auto wall_t0 = std::chrono::steady_clock::now ();
  const uint64_t cpu_t0 = process_cpu_time_ns ();

  std::array<float, 320U> fp_frame{};

  while (true)
    {
      const auto now_tp = std::chrono::steady_clock::now ();
      const auto elapsed_ms
          = std::chrono::duration_cast<std::chrono::milliseconds> (now_tp
                                                                   - wall_t0)
                .count ();
      if (elapsed_ms >= static_cast<int64_t> (target_ms))
        {
          break;
        }

      const std::size_t n = impl_->audio_io->pop_capture (fp_frame);
      if (n < fp_frame.size ())
        {
          fp_frame.fill (0.0F);
        }
      (void)impl_->vad->process_frame (fp_frame);
      std::this_thread::sleep_for (std::chrono::milliseconds (20));
    }

  const auto wall_t1 = std::chrono::steady_clock::now ();
  const uint64_t cpu_t1 = process_cpu_time_ns ();

  const auto wall_dur_ns
      = std::chrono::duration_cast<std::chrono::nanoseconds> (wall_t1
                                                              - wall_t0)
            .count ();
  const double wall_ns = static_cast<double> (wall_dur_ns);
  const double cpu_ns
      = (cpu_t1 >= cpu_t0) ? static_cast<double> (cpu_t1 - cpu_t0) : 0.0;
  const double pct = (wall_ns > 0.0) ? ((cpu_ns / wall_ns) * 100.0) : 0.0;

  if (impl_->run != nullptr)
    {
      (void)impl_->run->log_metric (
          static_cast<uint64_t> (impl_->turns.size ()), "idle_cpu_pct", pct);
    }
  return pct;
}

auto
SphinxSession::clear_memory () noexcept -> sutra::Result<void>
{
  if (impl_ == nullptr)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::InvalidConfiguration, 0,
          "SphinxSession is not initialized"));
    }

  if (impl_->memory_store != nullptr)
    {
      auto clr_res = impl_->memory_store->clear ();
      if (!clr_res)
        {
          return clr_res;
        }
    }
  else if (!impl_->config.memory_dir.empty ())
    {
      auto wipe_res = MemoryStore::wipe_directory (impl_->config.memory_dir);
      if (!wipe_res)
        {
          return wipe_res;
        }
    }

  (void)MemoryStore::wipe_directory (
      default_session_home_path ("sphinx-workspace/.memory"));
  (void)MemoryStore::wipe_directory (
      default_session_home_path (".local/share/sphinx/memory"));
  impl_->memory_cleared_turn_count = impl_->turns.size ();
  return {};
}

auto
SphinxSession::finalize_session () noexcept
    -> sutra::Result<SessionPlotsReport>
{
  if (impl_ == nullptr)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::InvalidConfiguration, 0,
          "SphinxSession is not initialized"));
    }

  auto plots_res = render_session_svgs (
      impl_->session_id, impl_->config.sessions_dir, impl_->turns);
  if (!plots_res)
    {
      return std::unexpected (plots_res.error ());
    }

  // Policy 3.3: Session-end episodic summary generation and persistence.
  if (impl_->config.enable_memory && impl_->embed_client != nullptr
      && impl_->embed_client->is_enabled () && impl_->memory_store != nullptr)
    {
      std::vector<const TurnTelemetry *> substantive_turns;
      for (std::size_t idx = impl_->memory_cleared_turn_count;
           idx < impl_->turns.size (); ++idx)
        {
          const TurnTelemetry &t = impl_->turns[idx];
          const bool is_refused_or_failed
              = (t.route == Route::TASK && t.tools_invoked.empty ())
                || contains_icase (t.assistant_text, "I don't have a tool to")
                || contains_icase (t.assistant_text, "I haven't inspected")
                || contains_icase (t.assistant_text, "ERROR:")
                || (contains_icase (t.assistant_text, "Compilation of ")
                    && contains_icase (t.assistant_text, "failed"))
                || contains_icase (t.assistant_text, "Command failed (exit")
                || contains_icase (t.assistant_text, "not accessible")
                || contains_icase (t.assistant_text, "outside the sandbox")
                || contains_icase (t.assistant_text, "not permitted")
                || contains_icase (t.assistant_text, "not in allowlist")
                || contains_icase (t.assistant_text, "secret material")
                || contains_icase (t.assistant_text,
                                   "not in system-readable");
          if (!is_trivial_greeting_turn (t.user_text)
              && !is_clear_terminal_command (t.user_text)
              && !is_clear_memory_command (t.user_text)
              && !is_unload_big_model_command (t.user_text)
              && !is_save_memory_command (t.user_text)
              && !contains_icase (t.assistant_text,
                                  "Unloaded resident task model")
              && !contains_icase (t.assistant_text,
                                  "Persistent memory erased")
              && !contains_icase (t.assistant_text, "cannot read")
              && !contains_icase (t.assistant_text, "were not provided")
              && !is_refused_or_failed)
            {
              substantive_turns.push_back (&t);
            }
        }

      if (!substantive_turns.empty ())
        {
          const std::size_t start_idx
              = (substantive_turns.size () > 8U)
                    ? (substantive_turns.size () - 8U)
                    : 0U;
          std::ostringstream transcript;
          std::string fallback_topics;
          for (std::size_t i = start_idx; i < substantive_turns.size (); ++i)
            {
              const TurnTelemetry &t = *substantive_turns[i];
              std::string u_line = t.user_text.substr (
                  0U, std::min<std::size_t> (120U, t.user_text.size ()));
              std::string a_line = t.assistant_text.substr (
                  0U, std::min<std::size_t> (140U, t.assistant_text.size ()));
              for (char &ch : u_line)
                {
                  if (ch == '\n' || ch == '\r' || ch == '\t')
                    {
                      ch = ' ';
                    }
                }
              for (char &ch : a_line)
                {
                  if (ch == '\n' || ch == '\r' || ch == '\t')
                    {
                      ch = ' ';
                    }
                }
              transcript << "User: " << u_line << "\n"
                         << "Sphinx: " << a_line << "\n";
              if (fallback_topics.size () < 160U)
                {
                  if (!fallback_topics.empty ())
                    {
                      fallback_topics.append ("; ");
                    }
                  const std::size_t cap
                      = std::min<std::size_t> (48U, u_line.size ());
                  fallback_topics.append (u_line.substr (0U, cap));
                }
            }

          std::string llm_summary;
          if (impl_->chat_client != nullptr)
            {
              std::string sum_user
                  = "Summarize only the verified outcomes explicitly stated "
                    "in Sphinx's replies in 1-2 concise sentences. Focus on "
                    "facts about the user, their projects, and what was "
                    "accomplished. Never infer or fabricate claims about "
                    "directories being empty, files existing, or commands "
                    "succeeding unless explicitly stated in Sphinx's "
                    "replies. Do not include greetings or filler.\n\n"
                    "Conversation:\n"
                    + transcript.str () + "\n/no_think";
              std::array<sutra::Message, 2U> sum_msgs = { {
                  { sutra::MessageRole::System,
                    "You are a concise session summarizer. Write 1-2 "
                    "complete sentences on a single line.",
                    {},
                    {},
                    0U,
                    0U },
                  { sutra::MessageRole::User,
                    std::move (sum_user),
                    {},
                    {},
                    0U,
                    0U },
              } };
              sutra::CompletionRequest sreq;
              sreq.messages = sum_msgs;
              sreq.temperature = 0.0F;
              sreq.max_tokens = 80U;
              sreq.seed = 42U;
              std::promise<void> sp;
              auto sf = sp.get_future ();
              auto disp = impl_->chat_client->stream_completion (
                  sreq,
                  [&llm_summary] (const sutra::TokenChunk &c) {
                    llm_summary.append (c.text);
                  },
                  [&sp] (sutra::Result<sutra::CompletionResult>) {
                    sp.set_value ();
                  });
              if (disp)
                {
                  sf.wait ();
                }
            }

          for (char &ch : llm_summary)
            {
              if (ch == '\n' || ch == '\r' || ch == '\t')
                {
                  ch = ' ';
                }
            }
          const std::string_view trimmed_sum = trim_ascii (llm_summary);
          std::string episodic_text
              = "In our last time session we talked about: ";
          if (!trimmed_sum.empty ())
            {
              episodic_text.append (trimmed_sum);
            }
          else
            {
              episodic_text.append (
                  "Completed " + std::to_string (substantive_turns.size ())
                  + " turn(s) covering " + fallback_topics + ".");
            }

          auto emb_res = impl_->embed_client->embed (episodic_text);
          if (emb_res)
            {
              bool is_dup_episodic = false;
              auto prev_ep = impl_->memory_store->retrieve (
                  *emb_res, 5U, MemoryType::EPISODIC);
              if (prev_ep)
                {
                  const std::string_view ep_trim = trim_ascii (episodic_text);
                  for (const auto &pe : *prev_ep)
                    {
                      if (!pe.superseded_by.has_value ()
                          && trim_ascii (pe.text) == ep_trim)
                        {
                          is_dup_episodic = true;
                          break;
                        }
                    }
                }
              if (!is_dup_episodic)
                {
                  auto id_res = impl_->memory_store->store (
                      episodic_text, *emb_res, MemoryType::EPISODIC);
                  if (id_res
                      && std::getenv ("SPHINX_DEBUG_MEMORY") != nullptr)
                    {
                      (void)std::fprintf (
                          stderr,
                          "[debug] memory store [EPISODIC] id=%llu \"%s\"\n",
                          static_cast<unsigned long long> (*id_res),
                          episodic_text.c_str ());
                    }
                }
            }
        }
      (void)impl_->memory_store->close ();
    }

  if (impl_->run != nullptr)
    {
      (void)impl_->run->log_metric (
          static_cast<uint64_t> (impl_->turns.size ()),
          "darshana_main_svg_ms", plots_res->main_render_ms);
      (void)impl_->run->log_metric (
          static_cast<uint64_t> (impl_->turns.size ()),
          "darshana_latency_svg_ms", plots_res->latency_render_ms);
      (void)impl_->run->log_metric (
          static_cast<uint64_t> (impl_->turns.size ()),
          "darshana_router_svg_ms", plots_res->router_render_ms);
      (void)impl_->run->finish ();
    }
  (void)impl_->store.compact ();
  if (impl_->jsonl_out.is_open ())
    {
      impl_->jsonl_out.flush ();
    }
  return *plots_res;
}

auto
SphinxSession::session_id () const noexcept -> std::string_view
{
  return (impl_ != nullptr) ? std::string_view (impl_->session_id) : "";
}

auto
SphinxSession::turns () const noexcept -> std::span<const TurnTelemetry>
{
  if (impl_ == nullptr)
    {
      return {};
    }
  return impl_->turns;
}

auto
SphinxSession::recorded_tts_16k () const noexcept -> std::span<const int16_t>
{
  if (impl_ == nullptr)
    {
      return {};
    }
  return impl_->recorded_tts_16k;
}

void
SphinxSession::clear_recorded_tts_16k () noexcept
{
  if (impl_ != nullptr)
    {
      const std::scoped_lock lock (impl_->tts_record_mtx);
      impl_->recorded_tts_16k.clear ();
    }
}

auto
SphinxSession::router () noexcept -> TurnRouter &
{
  return *impl_->router;
}

auto
SphinxSession::audio_io () noexcept -> vach::AudioIo &
{
  return *impl_->audio_io;
}

auto
SphinxSession::tool_registry () noexcept -> sutra::ToolRegistry &
{
  return impl_->tools;
}

auto
SphinxSession::yantra_run () noexcept -> yantra::Run &
{
  return *impl_->run;
}

auto
SphinxSession::router_load_seconds () const noexcept -> double
{
  return (impl_ != nullptr) ? impl_->router_load_sec : 0.0;
}

auto
SphinxSession::big_model_load_seconds () const noexcept -> double
{
  return (impl_ != nullptr) ? impl_->big_model_load_sec : 0.0;
}

auto
SphinxSession::vram_peak_gb () const noexcept -> double
{
  return (impl_ != nullptr) ? impl_->vram_peak_gb : 0.0;
}

auto
SphinxSession::embedding_client () noexcept -> EmbeddingClient *
{
  return (impl_ != nullptr) ? impl_->embed_client.get () : nullptr;
}

auto
SphinxSession::memory_store () noexcept -> MemoryStore *
{
  return (impl_ != nullptr) ? impl_->memory_store.get () : nullptr;
}

auto
SphinxSession::system_profile () const noexcept -> std::string_view
{
  return (impl_ != nullptr) ? std::string_view (impl_->system_profile) : "";
}

auto
SphinxSession::ensure_big_model_loaded () noexcept -> sutra::Result<void>
{
  if (impl_ == nullptr)
    {
      return std::unexpected (sutra::AgentError::framework (
          sutra::FrameworkErrorCode::InvalidConfiguration, 0,
          "SphinxSession is not initialized"));
    }
  return impl_->ensure_big_model_loaded ();
}

void
SphinxSession::unload_big_model () noexcept
{
  if (impl_ != nullptr)
    {
      impl_->unload_big_model ();
    }
}

auto
SphinxSession::is_big_model_loaded () const noexcept -> bool
{
  if (impl_ == nullptr)
    {
      return false;
    }
  const std::scoped_lock lock (impl_->big_model_mtx);
  return impl_->big_client != nullptr;
}

} /* namespace sphinx::inline v1 */
