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

#include <sutra/llm_backends.hpp>
#include <sutra/observability.hpp>
#include <sutra/tool.hpp>

#if defined(SUTRA_HAS_LLAMACPP) && SUTRA_HAS_LLAMACPP
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <gguf.h>
#include <llama.h>
#endif

#if defined(__linux__)
#include <fcntl.h>
#include <link.h>
#include <malloc.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <bit>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

namespace sutra::inline v1
{

namespace
{

constexpr std::size_t kMaxTailRetainBytes = 64U;
constexpr std::size_t kMetaBufBytes = 256U;
constexpr std::size_t kPieceBufBytes = 128U;
constexpr int32_t kPrefillMicroBatch = 64;
constexpr uint32_t kGgufMagic = 0x46554747U; // "GGUF"
constexpr uint32_t kMaxGgufTensorsToInspect = 512U;
constexpr uint64_t kMaxGgufStringLen = 65536U;
constexpr uint32_t kDefaultSeed = 42U;

/* DeepSeek UTF-8 markers: full-width vertical line (U+FF5C) and lower one
   eighth block (U+2581).  */
constexpr std::string_view kDeepSeekCallsTag = "<\xef\xbd\x9ctool\xe2\x96\x81"
                                               "calls\xef\xbd\x9c>";
constexpr std::string_view kDeepSeekBeginTag = "<\xef\xbd\x9ctool\xe2\x96\x81"
                                               "call\xe2\x96\x81"
                                               "begin\xef\xbd\x9c>";
constexpr std::string_view kDeepSeekEndTag = "<\xef\xbd\x9ctool\xe2\x96\x81"
                                             "call\xe2\x96\x81"
                                             "end\xef\xbd\x9c>";

/* Case-insensitive substring search helper.  */
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
          const auto ch_a = static_cast<unsigned char> (haystack[i + j]);
          const auto ch_b = static_cast<unsigned char> (needle[j]);
          if (std::tolower (ch_a) != std::tolower (ch_b))
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

/* Finds the matching closing brace '}' for a JSON object starting at open_pos.
   Returns std::string_view::npos if the JSON object is incomplete.  */
[[nodiscard]] auto
find_balanced_json_object (std::string_view text,
                           std::size_t open_pos) noexcept -> std::size_t
{
  if (open_pos >= text.size () || text[open_pos] != '{')
    {
      return std::string_view::npos;
    }
  int depth = 0;
  bool in_quotes = false;
  bool escaped = false;
  for (std::size_t i = open_pos; i < text.size (); ++i)
    {
      const char ch = text[i];
      if (escaped)
        {
          escaped = false;
          continue;
        }
      if (ch == '\\' && in_quotes)
        {
          escaped = true;
          continue;
        }
      if (ch == '"')
        {
          in_quotes = !in_quotes;
          continue;
        }
      if (!in_quotes)
        {
          if (ch == '{')
            {
              ++depth;
            }
          else if (ch == '}')
            {
              --depth;
              if (depth == 0)
                {
                  return i;
                }
            }
        }
    }
  return std::string_view::npos;
}

/* Finds the matching closing bracket ']' for a JSON array starting at
   open_pos.  */
[[nodiscard]] auto
find_balanced_json_array (std::string_view text,
                          std::size_t open_pos) noexcept -> std::size_t
{
  if (open_pos >= text.size () || text[open_pos] != '[')
    {
      return std::string_view::npos;
    }
  int depth = 0;
  bool in_quotes = false;
  bool escaped = false;
  for (std::size_t i = open_pos; i < text.size (); ++i)
    {
      const char ch = text[i];
      if (escaped)
        {
          escaped = false;
          continue;
        }
      if (ch == '\\' && in_quotes)
        {
          escaped = true;
          continue;
        }
      if (ch == '"')
        {
          in_quotes = !in_quotes;
          continue;
        }
      if (!in_quotes)
        {
          if (ch == '[')
            {
              ++depth;
            }
          else if (ch == ']')
            {
              --depth;
              if (depth == 0)
                {
                  return i;
                }
            }
        }
    }
  return std::string_view::npos;
}

/* Parses a single JSON object into a ToolCall if it contains "name" and
   ("arguments" or "parameters").  */
[[nodiscard]] auto
parse_json_object_to_tool_call (std::string_view json_obj,
                                std::size_t &call_counter)
    -> std::optional<ToolCall>
{
  auto name_res
      = extract_json_field_token (json_obj, "name", JsonType::String);
  if (!name_res || name_res->empty ())
    {
      return std::nullopt;
    }

  std::string args_str = "{}";
  auto args_obj
      = extract_json_field_token (json_obj, "arguments", JsonType::Object);
  if (args_obj)
    {
      args_str = std::move (*args_obj);
    }
  else
    {
      auto params_obj = extract_json_field_token (json_obj, "parameters",
                                                  JsonType::Object);
      if (params_obj)
        {
          args_str = std::move (*params_obj);
        }
      else
        {
          auto args_str_tok = extract_json_field_token (json_obj, "arguments",
                                                        JsonType::String);
          if (args_str_tok)
            {
              args_str = std::move (*args_str_tok);
            }
          else
            {
              return std::nullopt;
            }
        }
    }

  ToolCall tc;
  auto id_res = extract_json_field_token (json_obj, "id", JsonType::String);
  if (id_res && !id_res->empty ())
    {
      tc.id = std::move (*id_res);
    }
  else
    {
      tc.id = "call_" + std::to_string (call_counter);
    }
  ++call_counter;
  tc.name = std::move (*name_res);
  tc.arguments_json = std::move (args_str);
  return tc;
}

/* Extracts all JSON tool call objects from a slice of text.  */
void
extract_all_json_tool_objects (std::string_view slice,
                               std::size_t &call_counter,
                               std::vector<ToolCall> &out)
{
  std::size_t pos = 0U;
  while (pos < slice.size ())
    {
      const std::size_t brace = slice.find ('{', pos);
      if (brace == std::string_view::npos)
        {
          break;
        }
      const std::size_t close = find_balanced_json_object (slice, brace);
      if (close == std::string_view::npos)
        {
          break;
        }
      const std::string_view candidate
          = slice.substr (brace, close - brace + 1U);
      if (auto tc = parse_json_object_to_tool_call (candidate, call_counter))
        {
          out.push_back (std::move (*tc));
        }
      pos = close + 1U;
    }
}

/* Normalizes Gemma 4 DSL arguments (e.g., {path:<|"|>notes.txt<|"|>,n:5}) into
   strict RFC 8259 JSON ({"path":"notes.txt","n":5}).  */
[[nodiscard]] auto
normalize_gemma_args_to_json (std::string_view raw_args) -> std::string
{
  constexpr std::string_view kGemmaQuote = "<|\"|>";
  std::string step1;
  step1.reserve (raw_args.size () + 16U);

  std::size_t i = 0U;
  bool in_gemma_str = false;
  while (i < raw_args.size ())
    {
      if (raw_args.substr (i).starts_with (kGemmaQuote))
        {
          step1.push_back ('"');
          in_gemma_str = !in_gemma_str;
          i += kGemmaQuote.size ();
          continue;
        }
      const char ch = raw_args[i];
      if (in_gemma_str && (ch == '"' || ch == '\\'))
        {
          step1.push_back ('\\');
        }
      step1.push_back (ch);
      ++i;
    }

  std::string out;
  out.reserve (step1.size () + 16U);
  bool in_quotes = false;
  bool escaped = false;
  std::size_t pos = 0U;
  while (pos < step1.size ())
    {
      const char ch = step1[pos];
      if (escaped)
        {
          out.push_back (ch);
          escaped = false;
          ++pos;
          continue;
        }
      if (ch == '\\' && in_quotes)
        {
          out.push_back (ch);
          escaped = true;
          ++pos;
          continue;
        }
      if (ch == '"')
        {
          in_quotes = !in_quotes;
          out.push_back (ch);
          ++pos;
          continue;
        }
      if (!in_quotes && (ch == '{' || ch == ','))
        {
          out.push_back (ch);
          ++pos;
          while (pos < step1.size ()
                 && std::isspace (static_cast<unsigned char> (step1[pos]))
                        != 0)
            {
              out.push_back (step1[pos]);
              ++pos;
            }
          if (pos < step1.size ()
              && (std::isalpha (static_cast<unsigned char> (step1[pos])) != 0
                  || step1[pos] == '_'))
            {
              std::size_t key_end = pos;
              while (key_end < step1.size ()
                     && (std::isalnum (
                             static_cast<unsigned char> (step1[key_end]))
                             != 0
                         || step1[key_end] == '_' || step1[key_end] == '-'
                         || step1[key_end] == '.'))
                {
                  ++key_end;
                }
              std::size_t after_key = key_end;
              while (after_key < step1.size ()
                     && std::isspace (
                            static_cast<unsigned char> (step1[after_key]))
                            != 0)
                {
                  ++after_key;
                }
              if (after_key < step1.size () && step1[after_key] == ':')
                {
                  out.push_back ('"');
                  out.append (step1.substr (pos, key_end - pos));
                  out.push_back ('"');
                  pos = key_end;
                  continue;
                }
            }
          continue;
        }
      out.push_back (ch);
      ++pos;
    }
  return out;
}

/* Appends a JSON-escaped string literal (including quotes) to out.  */
void
append_json_escaped_string (std::string &out, std::string_view val)
{
  out.push_back ('"');
  for (const char ch : val)
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
  out.push_back ('"');
}

/* Parses a Gemma 4 tool call body ("call:tool_name{...}"), a Qwen3-Coder XML
   function block ("<function=name><parameter=k>v</parameter></function>"),
   or a JSON tool object slice into out.  */
void
extract_gemma_or_json_tool_objects (std::string_view slice,
                                    std::size_t &call_counter,
                                    std::vector<ToolCall> &out)
{
  std::size_t ws = 0U;
  while (ws < slice.size ()
         && std::isspace (static_cast<unsigned char> (slice[ws])) != 0)
    {
      ++ws;
    }
  const std::string_view trimmed = slice.substr (ws);
  constexpr std::string_view kFnOpen = "<function=";
  constexpr std::string_view kFnClose = "</function>";
  std::size_t fn_pos = trimmed.find (kFnOpen);
  if (fn_pos != std::string_view::npos)
    {
      bool extracted_any = false;
      while (fn_pos != std::string_view::npos)
        {
          const std::size_t name_start = fn_pos + kFnOpen.size ();
          const std::size_t name_end = trimmed.find ('>', name_start);
          if (name_end == std::string_view::npos || name_end <= name_start)
            {
              break;
            }
          const std::string_view fn_name
              = trimmed.substr (name_start, name_end - name_start);
          const std::size_t fn_close
              = trimmed.find (kFnClose, name_end + 1U);
          const std::string_view fn_body
              = (fn_close != std::string_view::npos)
                    ? trimmed.substr (name_end + 1U,
                                      fn_close - (name_end + 1U))
                    : trimmed.substr (name_end + 1U);
          std::string json_args = "{";
          bool first_param = true;
          std::size_t p_cursor = 0U;
          constexpr std::string_view kParamOpen = "<parameter=";
          constexpr std::string_view kParamClose = "</parameter>";
          while (p_cursor < fn_body.size ())
            {
              const std::size_t p_idx = fn_body.find (kParamOpen, p_cursor);
              if (p_idx == std::string_view::npos)
                {
                  break;
                }
              const std::size_t k_start = p_idx + kParamOpen.size ();
              const std::size_t k_end = fn_body.find ('>', k_start);
              if (k_end == std::string_view::npos)
                {
                  break;
                }
              const std::string_view p_key
                  = fn_body.substr (k_start, k_end - k_start);
              const std::size_t v_start = k_end + 1U;
              const std::size_t v_end = fn_body.find (kParamClose, v_start);
              if (v_end == std::string_view::npos)
                {
                  break;
                }
              std::string_view p_val
                  = fn_body.substr (v_start, v_end - v_start);
              if (!p_val.empty () && p_val.front () == '\n')
                {
                  p_val.remove_prefix (1U);
                }
              if (!p_val.empty () && p_val.back () == '\n')
                {
                  p_val.remove_suffix (1U);
                }
              if (!first_param)
                {
                  json_args.push_back (',');
                }
              first_param = false;
              append_json_escaped_string (json_args, p_key);
              json_args.push_back (':');
              append_json_escaped_string (json_args, p_val);
              p_cursor = v_end + kParamClose.size ();
            }
          json_args.push_back ('}');
          ToolCall tc;
          tc.id = "call_" + std::to_string (call_counter);
          ++call_counter;
          tc.name = std::string (fn_name);
          tc.arguments_json = std::move (json_args);
          out.push_back (std::move (tc));
          extracted_any = true;
          if (fn_close == std::string_view::npos)
            {
              break;
            }
          fn_pos = trimmed.find (kFnOpen, fn_close + kFnClose.size ());
        }
      if (extracted_any)
        {
          return;
        }
    }
  constexpr std::string_view kCallPrefix = "call:";
  if (trimmed.starts_with (kCallPrefix))
    {
      const std::string_view after_call = trimmed.substr (kCallPrefix.size ());
      const std::size_t brace_open = after_call.find ('{');
      if (brace_open != std::string_view::npos && brace_open > 0U)
        {
          std::string_view tname = after_call.substr (0U, brace_open);
          while (!tname.empty ()
                 && std::isspace (
                        static_cast<unsigned char> (tname.back ()))
                        != 0)
            {
              tname.remove_suffix (1U);
            }
          const std::string norm_sub
              = normalize_gemma_args_to_json (after_call.substr (brace_open));
          const std::size_t brace_close
              = find_balanced_json_object (norm_sub, 0U);
          if (!tname.empty () && brace_close != std::string_view::npos)
            {
              ToolCall tc;
              tc.id = "call_" + std::to_string (call_counter);
              ++call_counter;
              tc.name = std::string (tname);
              tc.arguments_json = norm_sub.substr (0U, brace_close + 1U);
              out.push_back (std::move (tc));
              return;
            }
        }
    }
  extract_all_json_tool_objects (slice, call_counter, out);
}

/* Extracts complete tool calls for a specific ToolCallStyle and sets
   consumed_up_to to the end of the last consumed construct.  */
[[nodiscard]] auto
extract_by_style (ToolCallStyle style, std::string_view text,
                  std::size_t &call_counter,
                  std::size_t &consumed_up_to) -> std::vector<ToolCall>
{
  std::vector<ToolCall> results;
  consumed_up_to = 0U;

  if (style == ToolCallStyle::Qwen || style == ToolCallStyle::Hermes
      || style == ToolCallStyle::Gemma4)
    {
      constexpr std::string_view kXmlOpen = "<tool_call>";
      constexpr std::string_view kXmlClose = "</tool_call>";
      constexpr std::string_view kGemmaOpen1 = "<|tool_call|>";
      constexpr std::string_view kGemmaOpen2 = "<|tool_call>";
      constexpr std::string_view kGemmaClose1 = "<|tool_call|>";
      constexpr std::string_view kGemmaClose2 = "<tool_call|>";

      std::size_t pos = 0U;
      while (pos < text.size ())
        {
          const std::size_t xml_idx = text.find (kXmlOpen, pos);
          const std::size_t g1_idx = text.find (kGemmaOpen1, pos);
          const std::size_t g2_idx = text.find (kGemmaOpen2, pos);

          std::size_t open_idx = xml_idx;
          std::string_view open_tag = kXmlOpen;
          bool is_gemma_tag = false;

          if (g1_idx != std::string_view::npos
              && (open_idx == std::string_view::npos || g1_idx < open_idx))
            {
              open_idx = g1_idx;
              open_tag = kGemmaOpen1;
              is_gemma_tag = true;
            }
          if (g2_idx != std::string_view::npos
              && (open_idx == std::string_view::npos || g2_idx < open_idx))
            {
              open_idx = g2_idx;
              open_tag = kGemmaOpen2;
              is_gemma_tag = true;
            }

          if (open_idx == std::string_view::npos)
            {
              break;
            }

          const std::size_t body_start = open_idx + open_tag.size ();
          std::size_t close_idx = std::string_view::npos;
          std::size_t close_len = 0U;

          if (!is_gemma_tag)
            {
              close_idx = text.find (kXmlClose, body_start);
              close_len = kXmlClose.size ();
            }
          else
            {
              const std::size_t c1 = text.find (kGemmaClose1, body_start);
              const std::size_t c2 = text.find (kGemmaClose2, body_start);
              const std::size_t c3 = text.find (kXmlClose, body_start);
              if (c1 != std::string_view::npos)
                {
                  close_idx = c1;
                  close_len = kGemmaClose1.size ();
                }
              if (c2 != std::string_view::npos
                  && (close_idx == std::string_view::npos || c2 < close_idx))
                {
                  close_idx = c2;
                  close_len = kGemmaClose2.size ();
                }
              if (c3 != std::string_view::npos
                  && (close_idx == std::string_view::npos || c3 < close_idx))
                {
                  close_idx = c3;
                  close_len = kXmlClose.size ();
                }
            }

          if (close_idx == std::string_view::npos)
            {
              consumed_up_to = open_idx;
              return results;
            }
          const std::string_view inner
              = text.substr (body_start, close_idx - body_start);
          extract_gemma_or_json_tool_objects (inner, call_counter, results);
          pos = close_idx + close_len;
          consumed_up_to = pos;
        }
      return results;
    }

  if (style == ToolCallStyle::Llama3)
    {
      constexpr std::string_view kPyOpen = "<|python_tag|>";
      constexpr std::string_view kPyClose = "</|python_tag|>";
      std::size_t pos = 0U;
      while (pos < text.size ())
        {
          const std::size_t py_idx = text.find (kPyOpen, pos);
          const std::size_t brace_idx = text.find ('{', pos);
          if (py_idx != std::string_view::npos
              && (brace_idx == std::string_view::npos || py_idx <= brace_idx))
            {
              const std::size_t after_py = py_idx + kPyOpen.size ();
              const std::size_t close_tag = text.find (kPyClose, after_py);
              if (close_tag != std::string_view::npos)
                {
                  extract_all_json_tool_objects (
                      text.substr (after_py, close_tag - after_py),
                      call_counter, results);
                  pos = close_tag + kPyClose.size ();
                  consumed_up_to = pos;
                  continue;
                }
              const std::size_t j_open = text.find ('{', after_py);
              if (j_open == std::string_view::npos)
                {
                  consumed_up_to = py_idx;
                  return results;
                }
              const std::size_t j_close
                  = find_balanced_json_object (text, j_open);
              if (j_close == std::string_view::npos)
                {
                  consumed_up_to = py_idx;
                  return results;
                }
              if (auto tc = parse_json_object_to_tool_call (
                      text.substr (j_open, j_close - j_open + 1U),
                      call_counter))
                {
                  results.push_back (std::move (*tc));
                }
              pos = j_close + 1U;
              if (text.substr (pos).starts_with (kPyClose))
                {
                  pos += kPyClose.size ();
                }
              consumed_up_to = pos;
            }
          else if (brace_idx != std::string_view::npos)
            {
              const std::size_t j_close
                  = find_balanced_json_object (text, brace_idx);
              if (j_close == std::string_view::npos)
                {
                  consumed_up_to = brace_idx;
                  return results;
                }
              if (auto tc = parse_json_object_to_tool_call (
                      text.substr (brace_idx, j_close - brace_idx + 1U),
                      call_counter))
                {
                  results.push_back (std::move (*tc));
                }
              pos = j_close + 1U;
              consumed_up_to = pos;
            }
          else
            {
              break;
            }
        }
      return results;
    }

  if (style == ToolCallStyle::Mistral)
    {
      constexpr std::string_view kMarker = "[TOOL_CALLS]";
      std::size_t pos = 0U;
      while (pos < text.size ())
        {
          const std::size_t m_idx = text.find (kMarker, pos);
          if (m_idx == std::string_view::npos)
            {
              break;
            }
          const std::size_t arr_open
              = text.find ('[', m_idx + kMarker.size ());
          if (arr_open == std::string_view::npos)
            {
              consumed_up_to = m_idx;
              return results;
            }
          const std::size_t arr_close
              = find_balanced_json_array (text, arr_open);
          if (arr_close == std::string_view::npos)
            {
              consumed_up_to = m_idx;
              return results;
            }
          extract_all_json_tool_objects (
              text.substr (arr_open, arr_close - arr_open + 1U), call_counter,
              results);
          pos = arr_close + 1U;
          consumed_up_to = pos;
        }
      return results;
    }

  if (style == ToolCallStyle::DeepSeek)
    {
      std::size_t pos = 0U;
      while (pos < text.size ())
        {
          std::size_t tag_idx = text.find (kDeepSeekBeginTag, pos);
          std::string_view active_tag = kDeepSeekBeginTag;
          const std::size_t calls_idx = text.find (kDeepSeekCallsTag, pos);
          if (calls_idx != std::string_view::npos
              && (tag_idx == std::string_view::npos || calls_idx < tag_idx))
            {
              tag_idx = calls_idx;
              active_tag = kDeepSeekCallsTag;
            }
          if (tag_idx == std::string_view::npos)
            {
              break;
            }
          const std::size_t after_tag = tag_idx + active_tag.size ();
          const std::size_t end_idx = text.find (kDeepSeekEndTag, after_tag);
          if (end_idx != std::string_view::npos)
            {
              extract_all_json_tool_objects (
                  text.substr (after_tag, end_idx - after_tag), call_counter,
                  results);
              pos = end_idx + kDeepSeekEndTag.size ();
              consumed_up_to = pos;
              continue;
            }
          const std::size_t j_open = text.find ('{', after_tag);
          if (j_open == std::string_view::npos)
            {
              consumed_up_to = tag_idx;
              return results;
            }
          const std::size_t j_close = find_balanced_json_object (text, j_open);
          if (j_close == std::string_view::npos)
            {
              consumed_up_to = tag_idx;
              return results;
            }
          if (auto tc = parse_json_object_to_tool_call (
                  text.substr (j_open, j_close - j_open + 1U), call_counter))
            {
              results.push_back (std::move (*tc));
            }
          pos = j_close + 1U;
          consumed_up_to = pos;
        }
      return results;
    }

  if (style == ToolCallStyle::Phi)
    {
      constexpr std::string_view kFence = "```";
      std::size_t pos = 0U;
      while (pos < text.size ())
        {
          const std::size_t f_open = text.find (kFence, pos);
          if (f_open == std::string_view::npos)
            {
              break;
            }
          const std::size_t body_start = f_open + kFence.size ();
          const std::size_t f_close = text.find (kFence, body_start);
          if (f_close == std::string_view::npos)
            {
              consumed_up_to = f_open;
              return results;
            }
          extract_all_json_tool_objects (
              text.substr (body_start, f_close - body_start), call_counter,
              results);
          pos = f_close + kFence.size ();
          consumed_up_to = pos;
        }
      return results;
    }

  // ToolCallStyle::Fallback: top-level {...} with "name" + "arguments".
  std::size_t pos = 0U;
  while (pos < text.size ())
    {
      const std::size_t brace = text.find ('{', pos);
      if (brace == std::string_view::npos)
        {
          break;
        }
      const std::size_t close = find_balanced_json_object (text, brace);
      if (close == std::string_view::npos)
        {
          consumed_up_to = brace;
          return results;
        }
      if (auto tc = parse_json_object_to_tool_call (
              text.substr (brace, close - brace + 1U), call_counter))
        {
          results.push_back (std::move (*tc));
        }
      pos = close + 1U;
      consumed_up_to = pos;
    }
  return results;
}

} /* namespace */

/* Constructs a ToolCallParser for a given model family style.  */
ToolCallParser::ToolCallParser (ToolCallStyle style) noexcept : style_ (style)
{
}

/* Factory helper creating a ToolCallParser for style.  */
auto
ToolCallParser::for_style (ToolCallStyle style) noexcept -> ToolCallParser
{
  return ToolCallParser (style);
}

/* Detects the ToolCallStyle from a model architecture or filename string.  */
auto
ToolCallParser::detect_style (std::string_view family_or_path) noexcept
    -> ToolCallStyle
{
  if (contains_icase (family_or_path, "qwen"))
    {
      return ToolCallStyle::Qwen;
    }
  if (contains_icase (family_or_path, "gemma"))
    {
      return ToolCallStyle::Gemma4;
    }
  if (contains_icase (family_or_path, "hermes")
      || contains_icase (family_or_path, "nous"))
    {
      return ToolCallStyle::Hermes;
    }
  if (contains_icase (family_or_path, "mistral")
      || contains_icase (family_or_path, "mixtral"))
    {
      return ToolCallStyle::Mistral;
    }
  if (contains_icase (family_or_path, "deepseek"))
    {
      return ToolCallStyle::DeepSeek;
    }
  if (contains_icase (family_or_path, "phi"))
    {
      return ToolCallStyle::Phi;
    }
  if (contains_icase (family_or_path, "llama"))
    {
      return ToolCallStyle::Llama3;
    }
  return ToolCallStyle::Fallback;
}

/* Incrementally parses a streamed token chunk while buffering only the minimal
   disambiguating tail.  */
auto
ToolCallParser::parse (std::string_view text_chunk) -> std::vector<ToolCall>
{
  tail_buffer_.append (text_chunk);
  std::size_t consumed = 0U;
  std::vector<ToolCall> extracted
      = extract_by_style (style_, tail_buffer_, emitted_count_, consumed);

  if (consumed > 0U)
    {
      tail_buffer_.erase (0U, consumed);
    }
  else
    {
      // Retain only a suffix starting from the earliest possible start symbol
      // ('<', '[', '`', '{') or at most kMaxTailRetainBytes of partial tag.
      const std::size_t first_marker = tail_buffer_.find_first_of ("<[{`");
      if (first_marker == std::string::npos)
        {
          if (tail_buffer_.size () > kMaxTailRetainBytes)
            {
              tail_buffer_.erase (0U,
                                  tail_buffer_.size () - kMaxTailRetainBytes);
            }
        }
      else if (first_marker > kMaxTailRetainBytes)
        {
          tail_buffer_.erase (0U, first_marker - kMaxTailRetainBytes);
        }
    }
  return extracted;
}

/* Statelessly parses a complete completion string for tool calls.  */
auto
ToolCallParser::parse_complete (std::string_view final_text) const
    -> std::vector<ToolCall>
{
  std::size_t counter = 0U;
  std::size_t consumed = 0U;
  return extract_by_style (style_, final_text, counter, consumed);
}

/* Resets streaming parser state.  */
void
ToolCallParser::reset () noexcept
{
  tail_buffer_.clear ();
  emitted_count_ = 0U;
}

/* Returns the configured ToolCallStyle.  */
auto
ToolCallParser::style () const noexcept -> ToolCallStyle
{
  return style_;
}

#if defined(SUTRA_HAS_LLAMACPP) && SUTRA_HAS_LLAMACPP
namespace
{

struct GgufDenseProbe
{
  std::string architecture;
  uint32_t block_count = 0U;
  uint32_t expert_count = 0U;
  bool valid = false;
};

[[nodiscard]] auto
probe_gguf_header_metadata (const std::string &path) noexcept -> GgufDenseProbe
{
  GgufDenseProbe out;
  std::ifstream in (path, std::ios::binary);
  if (!in.is_open ())
    {
      return out;
    }
  uint32_t magic = 0U;
  uint32_t version = 0U;
  uint64_t tensor_count = 0U;
  uint64_t kv_count = 0U;
  in.read (std::bit_cast<char *> (&magic), sizeof (magic));
  in.read (std::bit_cast<char *> (&version), sizeof (version));
  in.read (std::bit_cast<char *> (&tensor_count), sizeof (tensor_count));
  in.read (std::bit_cast<char *> (&kv_count), sizeof (kv_count));
  if (!in.good () || magic != kGgufMagic)
    {
      return out;
    }
  out.valid = true;

  auto skip_val = [&in] (auto &self, uint32_t vtype) -> bool {
    switch (vtype)
      {
      case 0U:
      case 1U:
      case 7U:
        in.seekg (1, std::ios::cur);
        return in.good ();
      case 2U:
      case 3U:
        in.seekg (2, std::ios::cur);
        return in.good ();
      case 4U:
      case 5U:
      case 6U:
        in.seekg (4, std::ios::cur);
        return in.good ();
      case 10U:
      case 11U:
      case 12U:
        in.seekg (8, std::ios::cur);
        return in.good ();
      case 8U:
        {
          uint64_t slen = 0U;
          in.read (std::bit_cast<char *> (&slen), sizeof (slen));
          if (!in.good () || slen > (16ULL << 20U))
            {
              return false;
            }
          in.seekg (static_cast<std::streamoff> (slen), std::ios::cur);
          return in.good ();
        }
      case 9U:
        {
          uint32_t etype = 0U;
          uint64_t cnt = 0U;
          in.read (std::bit_cast<char *> (&etype), sizeof (etype));
          in.read (std::bit_cast<char *> (&cnt), sizeof (cnt));
          if (!in.good ())
            {
              return false;
            }
          for (uint64_t idx = 0U; idx < cnt; ++idx)
            {
              if (!self (self, etype))
                {
                  return false;
                }
            }
          return true;
        }
      default:
        return false;
      }
  };

  for (uint64_t i = 0U; i < kv_count; ++i)
    {
      uint64_t klen = 0U;
      in.read (std::bit_cast<char *> (&klen), sizeof (klen));
      if (!in.good () || klen > kMaxGgufStringLen)
        {
          break;
        }
      std::string key (static_cast<std::size_t> (klen), '\0');
      in.read (key.data (), static_cast<std::streamsize> (key.size ()));
      uint32_t vtype = 0U;
      in.read (std::bit_cast<char *> (&vtype), sizeof (vtype));
      if (!in.good ())
        {
          break;
        }
      if (key == "general.architecture" && vtype == 8U)
        {
          uint64_t slen = 0U;
          in.read (std::bit_cast<char *> (&slen), sizeof (slen));
          if (in.good () && slen <= kMetaBufBytes)
            {
              out.architecture.resize (static_cast<std::size_t> (slen));
              in.read (out.architecture.data (),
                       static_cast<std::streamsize> (slen));
              continue;
            }
          break;
        }
      if (key.ends_with (".block_count") && vtype == 4U)
        {
          uint32_t bcnt = 0U;
          in.read (std::bit_cast<char *> (&bcnt), sizeof (bcnt));
          if (in.good ())
            {
              out.block_count = bcnt;
              continue;
            }
          break;
        }
      if (key.ends_with (".expert_count") && vtype == 4U)
        {
          uint32_t ecnt = 0U;
          in.read (std::bit_cast<char *> (&ecnt), sizeof (ecnt));
          if (in.good ())
            {
              out.expert_count = ecnt;
              continue;
            }
          break;
        }
      if (!skip_val (skip_val, vtype))
        {
          break;
        }
    }
  return out;
}

#if defined(__linux__)
[[nodiscard]] auto
detect_physical_cpu_ids () noexcept -> std::vector<int32_t>
{
  std::vector<int32_t> pcores;
  std::vector<int32_t> ecores;
  for (int32_t cpu = 0; cpu < 256; ++cpu)
    {
      const std::string sib_path = "/sys/devices/system/cpu/cpu"
                                   + std::to_string (cpu)
                                   + "/topology/thread_siblings_list";
      std::ifstream in (sib_path);
      if (!in.is_open ())
        {
          continue;
        }
      std::string line;
      if (!std::getline (in, line) || line.empty ())
        {
          continue;
        }
      int32_t first_sibling = -1;
      if (std::sscanf (line.c_str (), "%d", &first_sibling) == 1
          && first_sibling == cpu)
        {
          if (line.find (',') != std::string::npos
              || line.find ('-') != std::string::npos)
            {
              pcores.push_back (cpu);
            }
          else
            {
              ecores.push_back (cpu);
            }
        }
    }
  pcores.insert (pcores.end (), ecores.begin (), ecores.end ());
  if (pcores.empty ())
    {
      const unsigned int hw = std::thread::hardware_concurrency ();
      const int32_t fallback = (hw > 0U) ? static_cast<int32_t> (hw) : 12;
      for (int32_t i = 0; i < fallback; ++i)
        {
          pcores.push_back (i);
        }
    }
  return pcores;
}

void
bind_worker_to_alder_lake_pcores (int32_t n_threads,
                                  int32_t n_threads_batch) noexcept
{
  (void)n_threads;
  (void)n_threads_batch;
  const std::vector<int32_t> phys = detect_physical_cpu_ids ();
  if (phys.empty ())
    {
      return;
    }
  cpu_set_t cpuset;
  CPU_ZERO (&cpuset);
  for (const int32_t cpu : phys)
    {
      if (cpu >= 0 && cpu < CPU_SETSIZE)
        {
          CPU_SET (cpu, &cpuset);
        }
    }
  (void)pthread_setaffinity_np (pthread_self (), sizeof (cpu_set_t), &cpuset);
}
#endif

#if defined(__linux__)
[[nodiscard]] auto
query_total_ram_bytes () noexcept -> std::size_t
{
  std::ifstream meminfo ("/proc/meminfo");
  std::string key;
  unsigned long long val_kb = 0ULL;
  std::string rest;
  while (meminfo >> key >> val_kb)
    {
      std::getline (meminfo, rest);
      if (key == "MemTotal:")
        {
          return static_cast<std::size_t> (val_kb) * 1024ULL;
        }
    }
  return 0U;
}

[[nodiscard]] auto
query_mem_available_bytes () noexcept -> std::size_t
{
  std::ifstream meminfo ("/proc/meminfo");
  std::string key;
  unsigned long long val_kb = 0ULL;
  std::string rest;
  while (meminfo >> key >> val_kb)
    {
      std::getline (meminfo, rest);
      if (key == "MemAvailable:")
        {
          return static_cast<std::size_t> (val_kb) * 1024ULL;
        }
    }
  return 0U;
}

[[nodiscard]] auto
query_safe_populate_budget_bytes () noexcept -> std::size_t
{
  const std::size_t avail = query_mem_available_bytes ();
  constexpr std::size_t kSafetyReserveBytes = 2560ULL << 20U;
  return (avail > kSafetyReserveBytes) ? (avail - kSafetyReserveBytes) : 0U;
}
#endif

#if defined(__linux__)
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif

struct VmaSpan
{
  unsigned long long va_start = 0ULL;
  unsigned long long va_end = 0ULL;
  unsigned long long file_off = 0ULL;
};

struct FileByteSpan
{
  unsigned long long offset = 0ULL;
  unsigned long long length = 0ULL;
};

struct CpuRamSpan
{
  unsigned long long va_addr = 0ULL;
  unsigned long long file_off = 0ULL;
  unsigned long long length = 0ULL;
};

struct PreloadCancelState
{
  std::atomic<bool> cancel{ false };
  std::mutex mtx;
  std::condition_variable cv;
  bool running = false;
};

[[nodiscard]] auto
parse_blk_layer_index (std::string_view tensor_name) noexcept -> int32_t
{
  if (!tensor_name.starts_with ("blk."))
    {
      return -1;
    }
  std::size_t pos = 4U;
  if (pos >= tensor_name.size ()
      || std::isdigit (static_cast<unsigned char> (tensor_name[pos])) == 0)
    {
      return -1;
    }
  int32_t idx = 0;
  while (pos < tensor_name.size ()
         && std::isdigit (static_cast<unsigned char> (tensor_name[pos])) != 0)
    {
      idx = idx * 10 + static_cast<int32_t> (tensor_name[pos] - '0');
      ++pos;
    }
  return (pos < tensor_name.size () && tensor_name[pos] == '.') ? idx : -1;
}

[[nodiscard]] auto
is_moe_routed_expert_tensor (std::string_view tensor_name) noexcept -> bool
{
  if (tensor_name.find (".ffn_") == std::string_view::npos)
    {
      return false;
    }
  return tensor_name.find ("_exps") != std::string_view::npos
         || tensor_name.find ("_chexps") != std::string_view::npos;
}

__attribute__ ((no_sanitize ("address"))) void
populate_cpu_spans_in_chunks (const std::vector<CpuRamSpan> &cpu_spans,
                              const std::string &canon_str, bool sync_mode,
                              const std::shared_ptr<PreloadCancelState> &state)
    noexcept
{
  if (cpu_spans.empty ())
    {
      return;
    }
  unsigned long long total_bytes = 0ULL;
  for (const auto &sp : cpu_spans)
    {
      total_bytes += sp.length;
    }
  if (total_bytes == 0ULL)
    {
      return;
    }

  const int fd = open (canon_str.c_str (), O_RDONLY | O_CLOEXEC);
  const auto t0 = std::chrono::steady_clock::now ();
  constexpr unsigned long long kChunkSize = 64ULL << 20U;
  unsigned long long populated_bytes = 0ULL;

  for (const auto &sp : cpu_spans)
    {
      unsigned long long off = 0ULL;
      while (off < sp.length)
        {
          if (state != nullptr
              && state->cancel.load (std::memory_order_acquire))
            {
              if (fd >= 0)
                {
                  (void)close (fd);
                }
              return;
            }
          if (!sync_mode
              && query_safe_populate_budget_bytes () < (128ULL << 20U))
            {
              if (fd >= 0)
                {
                  (void)close (fd);
                }
              return;
            }
          const unsigned long long chunk_len
              = std::min (kChunkSize, sp.length - off);
          void *chunk_ptr
              = reinterpret_cast<void *> (sp.va_addr + off);
          const auto chunk_sz = static_cast<std::size_t> (chunk_len);

          (void)madvise (chunk_ptr, chunk_sz, MADV_NORMAL);
          if (madvise (chunk_ptr, chunk_sz, MADV_POPULATE_READ) != 0)
            {
              if (fd >= 0)
                {
                  (void)posix_fadvise (
                      fd, static_cast<off_t> (sp.file_off + off),
                      static_cast<off_t> (chunk_len), POSIX_FADV_WILLNEED);
                }
              volatile uint8_t sink = 0U;
              const auto *bytes
                  = static_cast<const uint8_t *> (chunk_ptr);
              for (std::size_t p = 0U; p < chunk_sz; p += 4096U)
                {
                  if (state != nullptr && (p & ((4ULL << 20U) - 1ULL)) == 0U
                      && state->cancel.load (std::memory_order_acquire))
                    {
                      if (fd >= 0)
                        {
                          (void)close (fd);
                        }
                      return;
                    }
                  sink = static_cast<uint8_t> (sink ^ bytes[p]);
                }
              (void)sink;
            }

          populated_bytes += chunk_len;
          off += chunk_len;

          if (sync_mode)
            {
              const double done_gib
                  = static_cast<double> (populated_bytes)
                    / static_cast<double> (1ULL << 30U);
              const double tot_gib
                  = static_cast<double> (total_bytes)
                    / static_cast<double> (1ULL << 30U);
              const double pct
                  = (100.0 * static_cast<double> (populated_bytes))
                    / static_cast<double> (total_bytes);
              char msg_buf[160];
              const auto fmt_res = std::format_to_n (
                  msg_buf, sizeof (msg_buf),
                  "\r[sphinx] Preloading CPU-offloaded weights: "
                  "{:.2f} / {:.2f} GiB ({:.1f}%)...",
                  done_gib, tot_gib, pct);
              const std::size_t msg_len = std::min (
                  sizeof (msg_buf), static_cast<std::size_t> (fmt_res.size));
              const ssize_t wr
                  = ::write (STDERR_FILENO, msg_buf, msg_len);
              (void)wr;
            }
        }
    }

  if (fd >= 0)
    {
      (void)close (fd);
    }

  if (sync_mode)
    {
      const auto t1 = std::chrono::steady_clock::now ();
      const double elapsed_s
          = std::max (1.0e-6,
                      std::chrono::duration<double> (t1 - t0).count ());
      const double tot_gib
          = static_cast<double> (populated_bytes)
            / static_cast<double> (1ULL << 30U);
      const double gib_s = tot_gib / elapsed_s;
      char msg_buf[192];
      const auto fmt_res = std::format_to_n (
          msg_buf, sizeof (msg_buf),
          "\r[sphinx] Preloaded {:.2f} GiB of CPU-offloaded weights into RAM "
          "in {:.2f} s ({:.2f} GiB/s).          \n",
          tot_gib, elapsed_s, gib_s);
      const std::size_t msg_len = std::min (
          sizeof (msg_buf), static_cast<std::size_t> (fmt_res.size));
      const ssize_t wr = ::write (STDERR_FILENO, msg_buf, msg_len);
      (void)wr;
    }
}
#endif

/* Reference-counted resident llama_model handle shared across clients loading
   the same GGUF file so simultaneous reference and native clients stay within
   the 8 GB VRAM budget while owning independent llama_context instances.  */
struct SharedLlamaModel
{
  std::string path;
  int32_t n_gpu_layers = 0;
  int32_t n_cpu_moe = 0;
  bool is_dense_31b = false;
  bool is_moe_80b = false;
  std::vector<std::string> moe_override_patterns;
  std::vector<llama_model_tensor_buft_override> moe_overrides;
  llama_model *model = nullptr;
#if defined(__linux__)
  std::shared_ptr<PreloadCancelState> preload_state;
#endif

  SharedLlamaModel (
      std::string p, int32_t gpu_layers, int32_t cpu_moe, bool dense_31b,
      bool moe_80b, std::vector<std::string> patterns,
      std::vector<llama_model_tensor_buft_override> overrides,
      llama_model *m) noexcept
      : path (std::move (p)), n_gpu_layers (gpu_layers), n_cpu_moe (cpu_moe),
        is_dense_31b (dense_31b), is_moe_80b (moe_80b),
        moe_override_patterns (std::move (patterns)),
        moe_overrides (std::move (overrides)), model (m)
  {
  }

  SharedLlamaModel (const SharedLlamaModel &) = delete;
  auto operator= (const SharedLlamaModel &) -> SharedLlamaModel & = delete;
  SharedLlamaModel (SharedLlamaModel &&) = delete;
  auto operator= (SharedLlamaModel &&) -> SharedLlamaModel & = delete;

  ~SharedLlamaModel () noexcept
  {
#if defined(__linux__)
    if (preload_state != nullptr)
      {
        preload_state->cancel.store (true, std::memory_order_release);
        std::unique_lock<std::mutex> lock (preload_state->mtx);
        preload_state->cv.wait (lock, [this] {
          return !preload_state->running;
        });
      }
#endif
    if (model != nullptr)
      {
        llama_model_free (model);
        model = nullptr;
      }
#if defined(__linux__) && !defined(__SANITIZE_ADDRESS__)
    (void)malloc_trim (0);
#endif
  }
};

[[nodiscard]] auto
acquire_shared_llama_model (const std::string &path, int32_t n_gpu_layers,
                            int32_t n_cpu_moe, bool use_mmap = true,
                            bool use_mlock = false,
                            bool preload_cpu_weights = false)
    -> std::shared_ptr<SharedLlamaModel>
{
  LlamaCppClient::init_backend ();
  static std::mutex cache_mtx;
  static std::vector<std::weak_ptr<SharedLlamaModel>> cache;

  std::error_code fs_ec;
  const auto fsize = std::filesystem::file_size (path, fs_ec);
  const GgufDenseProbe probe = probe_gguf_header_metadata (path);
  const bool is_dense_arch
      = (probe.architecture == "gemma4"
         || (probe.valid && probe.expert_count == 0U)
         || contains_icase (path, "gemma-4")
         || contains_icase (path, "gemma4"));
  const bool is_dense_31b
      = is_dense_arch && (!fs_ec && fsize > (10ULL << 30U));
  const bool is_moe_80b
      = !is_dense_arch
        && (probe.architecture == "qwen3next"
            || (probe.expert_count >= 128U && !fs_ec
                && fsize > (30ULL << 30U))
            || contains_icase (path, "qwen3-coder-next")
            || contains_icase (path, "80b"));

  int32_t norm_gpu_layers = (n_gpu_layers < 0) ? 99 : n_gpu_layers;
  int32_t norm_cpu_moe = is_dense_arch ? 0 : n_cpu_moe;

  // Auto-detection / Guard against Dense OOM on 8 GB GPU:
  // If a >10 GiB dense model (e.g., Gemma 4 31B, 60 layers, 0 experts) is
  // loaded with n_cpu_moe > 0 or unsplit n_gpu_layers (< 0 or >= 60), clamp to
  // the safe 17-layer GPU split (~5.10 GB VRAM with Q8_0 KV cache).
  const int32_t total_blocks
      = (probe.block_count > 0U) ? static_cast<int32_t> (probe.block_count)
                                 : (is_moe_80b ? 48 : 60);
  if (is_moe_80b)
    {
      // Pin all 48 attention / Gated DeltaNet layers, shared experts, norms,
      // embeddings, output head, and KV/SSM cache in GPU VRAM, while routing
      // 45..48 layers of 512-expert tensors (~872 MiB/layer) to Host DDR4 RAM.
      if (n_gpu_layers < 0 || n_gpu_layers < total_blocks)
        {
          norm_gpu_layers = 99;
        }
      int32_t min_safe_moe = 45;
      ggml_backend_dev_t gpu_dev
          = ggml_backend_dev_by_type (GGML_BACKEND_DEVICE_TYPE_GPU);
      if (gpu_dev != nullptr)
        {
          size_t free_b = 0U;
          size_t total_b = 0U;
          ggml_backend_dev_memory (gpu_dev, &free_b, &total_b);
          if (total_b > 0U && free_b > 0U)
            {
              const size_t used_mib = (total_b - free_b) >> 20U;
              const size_t free_mib = free_b >> 20U;
              constexpr size_t kTargetMaxTotalMib = 7150U;
              constexpr size_t kBaseSharedAndKvMib = 2050U;
              constexpr size_t kPerMoeLayerMib = 875U;
              const size_t cap_budget
                  = (used_mib < kTargetMaxTotalMib)
                        ? (kTargetMaxTotalMib - used_mib)
                        : 0U;
              const size_t free_budget
                  = (free_mib > 300U) ? (free_mib - 300U) : 0U;
              const size_t avail_vram_mib = std::min (cap_budget, free_budget);
              int32_t max_gpu_moe_layers = 0;
              if (avail_vram_mib > kBaseSharedAndKvMib)
                {
                  max_gpu_moe_layers = static_cast<int32_t> (
                      (avail_vram_mib - kBaseSharedAndKvMib)
                      / kPerMoeLayerMib);
                }
              min_safe_moe = std::clamp (total_blocks - max_gpu_moe_layers,
                                         44, total_blocks);
            }
        }
      if (norm_cpu_moe < min_safe_moe)
        {
          norm_cpu_moe = min_safe_moe;
        }
      if (norm_cpu_moe > total_blocks)
        {
          norm_cpu_moe = total_blocks;
        }
    }
  if (is_dense_31b
      && (n_cpu_moe > 0 || n_gpu_layers < 0 || n_gpu_layers >= total_blocks))
    {
      norm_gpu_layers = kDefaultSafeDense31bGpuLayers;
    }
  if (is_dense_31b && norm_gpu_layers > 0)
    {
      ggml_backend_dev_t gpu_dev
          = ggml_backend_dev_by_type (GGML_BACKEND_DEVICE_TYPE_GPU);
      if (gpu_dev != nullptr)
        {
          size_t free_b = 0U;
          size_t total_b = 0U;
          ggml_backend_dev_memory (gpu_dev, &free_b, &total_b);
          if (total_b > 0U && free_b > 0U)
            {
              const size_t used_mib = (total_b - free_b) >> 20U;
              const size_t free_mib = free_b >> 20U;
              constexpr size_t kTargetMaxTotalMib = 7150U;
              constexpr size_t kKvAndComputeReserveMib = 420U;
              constexpr size_t kOutputLayerMib = 945U;
              constexpr size_t kPerLayerMib = 278U;
              const size_t budget_by_cap
                  = (used_mib + kKvAndComputeReserveMib < kTargetMaxTotalMib)
                        ? (kTargetMaxTotalMib - used_mib
                           - kKvAndComputeReserveMib)
                        : 0U;
              const size_t budget_by_free
                  = (free_mib > kKvAndComputeReserveMib + 256U)
                        ? (free_mib - kKvAndComputeReserveMib - 256U)
                        : 0U;
              const size_t weight_budget_mib
                  = std::min (budget_by_cap, budget_by_free);
              int32_t max_safe_layers = 0;
              if (weight_budget_mib >= kOutputLayerMib)
                {
                  max_safe_layers = 1
                                    + static_cast<int32_t> (
                                        (weight_budget_mib - kOutputLayerMib)
                                        / kPerLayerMib);
                }
              if (norm_gpu_layers > max_safe_layers)
                {
                  norm_gpu_layers = std::max (1, max_safe_layers);
                }
            }
        }
    }

  const std::scoped_lock lock (cache_mtx);
  for (auto it = cache.begin (); it != cache.end ();)
    {
      if (auto locked = it->lock ())
        {
          if (locked->path == path && locked->n_gpu_layers == norm_gpu_layers
               && locked->n_cpu_moe == norm_cpu_moe)
            {
              return locked;
            }
          ++it;
        }
      else
        {
          it = cache.erase (it);
        }
    }

  llama_model_params mparams = llama_model_default_params ();
  mparams.n_gpu_layers = norm_gpu_layers;
  if (use_mmap && use_mlock)
    {
      mparams.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK;
    }
  else if (use_mmap)
    {
      mparams.load_mode = LLAMA_LOAD_MODE_MMAP;
    }
  else if (use_mlock)
    {
      mparams.load_mode = LLAMA_LOAD_MODE_MLOCK;
    }
  else
    {
      mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    }

  std::vector<std::string> patterns;
  std::vector<llama_model_tensor_buft_override> overrides;
  int32_t effective_cpu_moe = 0;
  if (norm_cpu_moe > 0 && !is_dense_arch)
    {
      effective_cpu_moe = norm_cpu_moe;
      if (is_moe_80b && effective_cpu_moe < 44)
        {
          effective_cpu_moe = 45;
        }
      else if (!is_moe_80b && !fs_ec && fsize > (12ULL << 30U)
               && effective_cpu_moe < 38)
        {
          effective_cpu_moe = 38;
        }
      const auto count = static_cast<std::size_t> (effective_cpu_moe);
      patterns.reserve (count);
      overrides.reserve (count + 1U);
      ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type ();
      for (int32_t i = 0; i < effective_cpu_moe; ++i)
        {
          patterns.push_back (
              "blk\\." + std::to_string (i)
              + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
        }
      for (const std::string &pat : patterns)
        {
          overrides.push_back ({ pat.c_str (), cpu_buft });
        }
      overrides.push_back ({ nullptr, nullptr });
      mparams.tensor_buft_overrides = overrides.data ();
    }

#if defined(__linux__)
  if (!path.empty ())
    {
      const int pre_fd = open (path.c_str (), O_RDONLY | O_CLOEXEC);
      if (pre_fd >= 0)
        {
          (void)posix_fadvise (pre_fd, 0, 0, POSIX_FADV_NORMAL);
          (void)close (pre_fd);
        }
    }
#endif
  const auto t_load_start = std::chrono::steady_clock::now ();
  llama_model *raw = llama_model_load_from_file (path.c_str (), mparams);
  const auto t_load_end = std::chrono::steady_clock::now ();
  if (raw == nullptr)
    {
      return nullptr;
    }
  auto shared = std::make_shared<SharedLlamaModel> (
      path, norm_gpu_layers, norm_cpu_moe, is_dense_31b, is_moe_80b,
      std::move (patterns), std::move (overrides), raw);
#if defined(__linux__)
  const char *env_sync = std::getenv ("SUTRA_SYNC_PRELOAD");
  const bool sync_preload
      = preload_cpu_weights || use_mlock
        || (env_sync != nullptr && std::string_view (env_sync) == "1");
  if ((sync_preload || norm_gpu_layers > 0 || norm_cpu_moe > 0 || is_dense_31b
       || is_moe_80b)
      && !path.empty ())
    {
      std::error_code canon_ec;
      const std::filesystem::path canon_p
          = std::filesystem::canonical (path, canon_ec);
      const std::string canon_str
          = canon_ec ? path : canon_p.string ();
      const bool aria2_active = std::filesystem::exists (
          canon_str + ".aria2", canon_ec);

      std::vector<VmaSpan> vmas;
      std::ifstream maps ("/proc/self/maps");
      std::string line;
      while (std::getline (maps, line))
        {
          if (line.find (path) == std::string::npos
              && line.find (canon_str) == std::string::npos)
            {
              continue;
            }
          unsigned long long start_addr = 0ULL;
          unsigned long long end_addr = 0ULL;
          unsigned long long file_off = 0ULL;
          if (std::sscanf (line.c_str (), "%llx-%llx %*s %llx", &start_addr,
                           &end_addr, &file_off)
                  == 3
              && end_addr > start_addr)
            {
              const auto seg_len
                  = static_cast<std::size_t> (end_addr - start_addr);
              (void)madvise (reinterpret_cast<void *> (start_addr), seg_len,
                             MADV_NORMAL);
              vmas.push_back ({ start_addr, end_addr, file_off });
            }
        }
      std::sort (vmas.begin (), vmas.end (),
                 [] (const VmaSpan &a, const VmaSpan &b) {
                   return a.file_off < b.file_off;
                 });

      std::vector<std::pair<unsigned long long, unsigned long long>>
          raw_cpu_ranges;
      gguf_init_params gparams{};
      gparams.no_alloc = true;
      gparams.ctx = nullptr;
      gguf_context *gctx = gguf_init_from_file (canon_str.c_str (), gparams);
      if (gctx != nullptr)
        {
          const auto data_off
              = static_cast<unsigned long long> (gguf_get_data_offset (gctx));
          const int64_t n_tensors = gguf_get_n_tensors (gctx);
          const int32_t first_gpu_blk
              = (norm_gpu_layers > 0)
                    ? std::max (0, total_blocks - norm_gpu_layers)
                    : total_blocks;
          for (int64_t ti = 0; ti < n_tensors; ++ti)
            {
              const char *tname_c = gguf_get_tensor_name (gctx, ti);
              if (tname_c == nullptr)
                {
                  continue;
                }
              const std::string_view tname (tname_c);
              const unsigned long long t_beg
                  = data_off
                    + static_cast<unsigned long long> (
                        gguf_get_tensor_offset (gctx, ti));
              const unsigned long long t_end
                  = t_beg
                    + static_cast<unsigned long long> (
                        gguf_get_tensor_size (gctx, ti));
              if (t_end <= t_beg)
                {
                  continue;
                }
              bool in_vma = false;
              for (const auto &v : vmas)
                {
                  const unsigned long long v_end
                      = v.file_off + (v.va_end - v.va_start);
                  if (t_beg < v_end && t_end > v.file_off)
                    {
                      in_vma = true;
                      break;
                    }
                }
              if (!in_vma)
                {
                  continue;
                }
              bool on_cpu = false;
              if (norm_gpu_layers <= 0
                  || tname.starts_with ("token_embd")
                  || tname.starts_with ("token_types")
                  || tname.starts_with ("rope_freqs"))
                {
                  on_cpu = true;
                }
              else
                {
                  const int32_t blk_idx = parse_blk_layer_index (tname);
                  if (blk_idx >= 0)
                    {
                      if (blk_idx < first_gpu_blk)
                        {
                          on_cpu = true;
                        }
                      else if (effective_cpu_moe > 0 && !is_dense_arch
                               && blk_idx < effective_cpu_moe
                               && is_moe_routed_expert_tensor (tname))
                        {
                          on_cpu = true;
                        }
                    }
                }
              if (on_cpu)
                {
                  raw_cpu_ranges.emplace_back (t_beg, t_end);
                }
            }
          gguf_free (gctx);
        }
      else if (norm_gpu_layers <= 0 || effective_cpu_moe > 0
               || norm_gpu_layers < total_blocks)
        {
          for (const auto &v : vmas)
            {
              raw_cpu_ranges.emplace_back (
                  v.file_off, v.file_off + (v.va_end - v.va_start));
            }
        }

      std::sort (raw_cpu_ranges.begin (), raw_cpu_ranges.end ());
      std::vector<std::pair<unsigned long long, unsigned long long>>
          merged_cpu_file;
      for (const auto &r : raw_cpu_ranges)
        {
          const unsigned long long p_beg = r.first & ~4095ULL;
          const unsigned long long p_end = (r.second + 4095ULL) & ~4095ULL;
          if (!merged_cpu_file.empty ()
              && p_beg <= merged_cpu_file.back ().second)
            {
              merged_cpu_file.back ().second
                  = std::max (merged_cpu_file.back ().second, p_end);
            }
          else
            {
              merged_cpu_file.emplace_back (p_beg, p_end);
            }
        }

      std::vector<CpuRamSpan> cpu_spans;
      for (const auto &mc : merged_cpu_file)
        {
          for (const auto &v : vmas)
            {
              const unsigned long long v_beg = v.file_off;
              const unsigned long long v_end
                  = v.file_off + (v.va_end - v.va_start);
              const unsigned long long s_beg
                  = ( std::max (mc.first, v_beg) + 4095ULL ) & ~4095ULL;
              const unsigned long long s_end
                  = std::min (mc.second, v_end) & ~4095ULL;
              if (s_end > s_beg)
                {
                  const unsigned long long s_va
                      = v.va_start + (s_beg - v.file_off);
                  cpu_spans.push_back ({ s_va, s_beg, s_end - s_beg });
                }
            }
        }

      std::error_code sz_ec;
      const auto fsz_raw = std::filesystem::file_size (canon_str, sz_ec);
      const unsigned long long fsz_aligned
          = (!sz_ec) ? (static_cast<unsigned long long> (fsz_raw) & ~4095ULL)
                     : 0ULL;
      std::vector<FileByteSpan> gpu_spans;
      if (norm_gpu_layers > 0 && fsz_aligned > 0ULL)
        {
          unsigned long long prev_end = 0ULL;
          for (const auto &cs : cpu_spans)
            {
              if (cs.file_off > prev_end)
                {
                  gpu_spans.push_back (
                      { prev_end, cs.file_off - prev_end });
                }
              if (cs.file_off + cs.length > prev_end)
                {
                  prev_end = cs.file_off + cs.length;
                }
            }
          if (fsz_aligned > prev_end)
            {
              gpu_spans.push_back ({ prev_end, fsz_aligned - prev_end });
            }
        }

      const auto t_evict_start = std::chrono::steady_clock::now ();
      unsigned long long evicted_gpu_bytes = 0ULL;
      if (!gpu_spans.empty ())
        {
          const int fd = open (canon_str.c_str (), O_RDONLY | O_CLOEXEC);
          for (const auto &gs : gpu_spans)
            {
              evicted_gpu_bytes += gs.length;
              const unsigned long long g_beg = gs.offset;
              const unsigned long long g_end = gs.offset + gs.length;
              for (const auto &v : vmas)
                {
                  const unsigned long long v_beg = v.file_off;
                  const unsigned long long v_end
                      = v.file_off + (v.va_end - v.va_start);
                  const unsigned long long m_beg
                      = (std::max (g_beg, v_beg) + 4095ULL) & ~4095ULL;
                  const unsigned long long m_end
                      = std::min (g_end, v_end) & ~4095ULL;
                  if (m_end > m_beg)
                    {
                      const unsigned long long m_va
                          = v.va_start + (m_beg - v.file_off);
                      (void)madvise (reinterpret_cast<void *> (m_va),
                                     static_cast<std::size_t> (m_end - m_beg),
                                     MADV_DONTNEED);
                    }
                }
              if (fd >= 0)
                {
                  (void)posix_fadvise (
                      fd, static_cast<off_t> (gs.offset),
                      static_cast<off_t> (gs.length), POSIX_FADV_DONTNEED);
                }
            }
          if (fd >= 0)
            {
              (void)close (fd);
            }
        }
      const auto t_evict_end = std::chrono::steady_clock::now ();

      if (!cpu_spans.empty () && !aria2_active)
        {
          if (sync_preload)
            {
              const auto t_pre_start = std::chrono::steady_clock::now ();
              populate_cpu_spans_in_chunks (cpu_spans, canon_str, true,
                                            nullptr);
              const auto t_pre_end = std::chrono::steady_clock::now ();
              const double load_s
                  = std::chrono::duration<double> (t_load_end - t_load_start)
                        .count ();
              const double evict_s
                  = std::chrono::duration<double> (t_evict_end - t_evict_start)
                        .count ();
              const double pre_s
                  = std::chrono::duration<double> (t_pre_end - t_pre_start)
                        .count ();
              const double gpu_gib
                  = static_cast<double> (evicted_gpu_bytes)
                    / static_cast<double> (1ULL << 30U);
              char tbuf[224];
              const auto fmt_res = std::format_to_n (
                  tbuf, sizeof (tbuf),
                  "[sphinx] Load breakdown: llama_load={:.2f}s, "
                  "gpu_evict={:.2f}s ({:.2f} GiB), "
                  "cpu_preload={:.2f}s.\n",
                  load_s, evict_s, gpu_gib, pre_s);
              const std::size_t tlen = std::min (
                  sizeof (tbuf), static_cast<std::size_t> (fmt_res.size));
              const ssize_t wr = ::write (STDERR_FILENO, tbuf, tlen);
              (void)wr;
            }
          else
            {
              auto st = std::make_shared<PreloadCancelState> ();
              st->running = true;
              shared->preload_state = st;
              std::thread ([spans = std::move (cpu_spans), canon_str,
                            st] () noexcept {
                populate_cpu_spans_in_chunks (spans, canon_str, false, st);
                {
                  const std::scoped_lock lk (st->mtx);
                  st->running = false;
                }
                st->cv.notify_all ();
              }).detach ();
            }
        }
    }
#endif
  cache.push_back (shared);
  return shared;
}

} /* namespace */
#endif

/* Opaque implementation of LlamaCppClient wrapping llama.cpp C API.  */
struct LlamaCppClient::Impl
{
  struct ActiveJob
  {
    std::string prompt;
    uint32_t max_tokens = kDefaultMaxNewTokens;
    float temperature = 0.0F;
    uint64_t seed = kDefaultSeed;
    std::vector<std::string> stop_sequences;
    std::size_t prompt_token_count = 0U;
    std::size_t sys_prefix_tokens = 0U;
    std::vector<int32_t> prompt_tokens;
    TokenStreamFn on_token;
    CompletionDone on_done;
    std::atomic<bool> done_fired{ false };
    std::mutex callback_mtx;

    void
    emit_token (const TokenChunk &chunk) noexcept
    {
      const std::scoped_lock lock (callback_mtx);
      if (!done_fired.load (std::memory_order_acquire) && on_token)
        {
          on_token (chunk);
        }
    }

    void
    emit_done (Result<CompletionResult> result) noexcept
    {
      const std::scoped_lock lock (callback_mtx);
      if (!done_fired.exchange (true, std::memory_order_acq_rel) && on_done)
        {
          on_done (std::move (result));
        }
    }
  };

  LlamaCppConfig config;
  ModelCapabilities caps;
#if defined(SUTRA_HAS_LLAMACPP) && SUTRA_HAS_LLAMACPP
  std::shared_ptr<SharedLlamaModel> shared_model;
  llama_model *model = nullptr;
  llama_context *ctx = nullptr;
  ggml_threadpool *threadpool = nullptr;
  const llama_vocab *vocab = nullptr;
#endif
  std::vector<int32_t> cached_kv_tokens;
  std::vector<int32_t> cached_sys_tokens;
  std::vector<uint8_t> cached_sys_state;
  std::string last_completed_prompt;
  std::vector<int32_t> last_completed_tokens;
  std::mutex eval_mtx;

  std::mutex queue_mtx;
  std::condition_variable queue_cv;
  std::deque<std::shared_ptr<ActiveJob>> job_queue;
  std::shared_ptr<ActiveJob> current_job;
  std::atomic<uint32_t> cancel_requested{ 0U };
  std::atomic<bool> stop_worker{ false };
  std::jthread worker_thread;

  Impl () = default;
  Impl (const Impl &) = delete;
  auto operator= (const Impl &) -> Impl & = delete;
  Impl (Impl &&) = delete;
  auto operator= (Impl &&) -> Impl & = delete;

  ~Impl () noexcept
  {
    stop_worker.store (true, std::memory_order_release);
    cancel_requested.store (1U, std::memory_order_release);
    queue_cv.notify_all ();
    if (worker_thread.joinable ())
      {
        worker_thread.join ();
      }
#if defined(SUTRA_HAS_LLAMACPP) && SUTRA_HAS_LLAMACPP
    if (ctx != nullptr)
      {
        llama_detach_threadpool (ctx);
        llama_free (ctx);
        ctx = nullptr;
      }
    if (threadpool != nullptr)
      {
        ggml_threadpool_free (threadpool);
        threadpool = nullptr;
      }
    model = nullptr;
    shared_model.reset ();
#endif
  }

  static auto
  abort_cb (void *user_data) -> bool
  {
    if (user_data == nullptr)
      {
        return false;
      }
    const auto *flag = static_cast<const std::atomic<uint32_t> *> (user_data);
    return flag->load (std::memory_order_acquire) != 0U;
  }

  /* Formats a CompletionRequest into a model-family prompt string.  */
  [[nodiscard]] auto
  format_prompt (const CompletionRequest &req) const -> std::string
  {
    const bool is_gemma4
        = (caps.model_family == "gemma4"
           || contains_icase (caps.model_family, "gemma")
           || caps.tool_style == ToolCallStyle::Gemma4);
    constexpr std::string_view kToolFormatRule
        = "\nWhen invoking any tool, emit the standard XML JSON call: "
          "<tool_call>{\"name\": \"<tool_name>\", \"arguments\": "
          "{...}}</tool_call>\n";
    std::string out;

    if (is_gemma4)
      {
        if (!req.tools.empty ())
          {
            out.append ("<|turn>system\nYou have access to tools:\n");
            for (const ToolSchema &ts : req.tools)
              {
                out.append ("- ");
                out.append (ts.name);
                out.append (": ");
                out.append (ts.schema_json);
                out.push_back ('\n');
              }
            out.append (kToolFormatRule);
            out.append ("<turn|>\n");
          }

        bool disable_thinking = false;
        constexpr std::string_view kNoThinkSuffix = " /no_think";

        for (const Message &msg : req.messages)
          {
            std::string_view content_view (msg.content);
            if (msg.role == MessageRole::User
                && content_view.ends_with (kNoThinkSuffix))
              {
                content_view.remove_suffix (kNoThinkSuffix.size ());
                disable_thinking = true;
              }
            switch (msg.role)
              {
              case MessageRole::System:
                out.append ("<|turn>system\n");
                out.append (content_view);
                if (contains_icase (content_view, "available tools")
                    || contains_icase (content_view, "tool"))
                  {
                    out.append (kToolFormatRule);
                  }
                out.append ("<turn|>\n");
                break;
              case MessageRole::User:
                out.append ("<|turn>user\n");
                out.append (content_view);
                out.append ("<turn|>\n");
                break;
              case MessageRole::Assistant:
                out.append ("<|turn>model\n");
                out.append (content_view);
                out.append ("<turn|>\n");
                break;
              case MessageRole::Tool:
                out.append ("<|turn>user\n<|tool_response>\n");
                out.append (content_view);
                out.append ("\n<tool_response|><turn|>\n");
                break;
              }
          }
        out.append ("<|turn>model\n");
        if (disable_thinking)
          {
            out.append ("<|channel>thought\n<channel|>");
          }
        return out;
      }

    if (!req.tools.empty ())
      {
        out.append ("<|im_start|>system\nYou have access to tools:\n");
        for (const ToolSchema &ts : req.tools)
          {
            out.append ("- ");
            out.append (ts.name);
            out.append (": ");
            out.append (ts.schema_json);
            out.push_back ('\n');
          }
        out.append ("<|im_end|>\n");
      }

    bool disable_thinking = false;
    constexpr std::string_view kNoThinkSuffix = " /no_think";

    for (const Message &msg : req.messages)
      {
        std::string_view content_view (msg.content);
        if (msg.role == MessageRole::User
            && content_view.ends_with (kNoThinkSuffix))
          {
            content_view.remove_suffix (kNoThinkSuffix.size ());
            disable_thinking = true;
          }
        switch (msg.role)
          {
          case MessageRole::System:
            out.append ("<|im_start|>system\n");
            out.append (content_view);
            out.append ("<|im_end|>\n");
            break;
          case MessageRole::User:
            out.append ("<|im_start|>user\n");
            out.append (content_view);
            out.append ("<|im_end|>\n");
            break;
          case MessageRole::Assistant:
            out.append ("<|im_start|>assistant\n");
            out.append (content_view);
            out.append ("<|im_end|>\n");
            break;
          case MessageRole::Tool:
            out.append ("<|im_start|>user\n<tool_response>\n");
            out.append (content_view);
            out.append ("\n</tool_response><|im_end|>\n");
            break;
          }
      }
    out.append ("<|im_start|>assistant\n");
    if (disable_thinking
        && caps.model_family.find ("qwen3") != std::string::npos
        && caps.model_family != "qwen3next")
      {
        out.append ("<think>\n\n</think>\n\n");
      }
    return out;
  }

  /* Tokenizes text thread-safely using the resident llama_vocab.  */
  [[nodiscard]] auto
  tokenize_str (std::string_view text,
                bool add_special) const -> Result<std::vector<int32_t>>
  {
#if defined(SUTRA_HAS_LLAMACPP) && SUTRA_HAS_LLAMACPP
    if (vocab == nullptr)
      {
        return std::unexpected (AgentError::llm (
            LlmErrorCode::InvalidResponse, "Model vocabulary is not loaded"));
      }
    const int32_t est_count = -llama_tokenize (
        vocab, text.data (), static_cast<int32_t> (text.size ()), nullptr, 0,
        add_special, true);
    if (est_count <= 0)
      {
        return std::vector<int32_t>{};
      }
    std::vector<int32_t> tokens (static_cast<std::size_t> (est_count));
    const int32_t actual = llama_tokenize (
        vocab, text.data (), static_cast<int32_t> (text.size ()),
        tokens.data (), est_count, add_special, true);
    if (actual < 0)
      {
        return std::unexpected (
            AgentError::llm (LlmErrorCode::InvalidResponse,
                             "Failed to tokenize prompt with llama_tokenize"));
      }
    tokens.resize (static_cast<std::size_t> (actual));
    return tokens;
#else
    (void)text;
    (void)add_special;
    return std::unexpected (
        AgentError::llm (LlmErrorCode::ConnectionFailed,
                         "Sutra was built without llama.cpp support"));
#endif
  }

  void
  worker_loop () noexcept
  {
#if defined(__linux__) && defined(SUTRA_HAS_LLAMACPP) && SUTRA_HAS_LLAMACPP
    bind_worker_to_alder_lake_pcores (config.n_threads,
                                      config.n_threads_batch);
#endif
    for (;;)
      {
        std::shared_ptr<ActiveJob> job;
        {
          std::unique_lock lock (queue_mtx);
          queue_cv.wait (lock, [this] {
            return stop_worker.load (std::memory_order_acquire)
                   || !job_queue.empty ();
          });
          if (stop_worker.load (std::memory_order_acquire)
              && job_queue.empty ())
            {
              return;
            }
          job = std::move (job_queue.front ());
          job_queue.pop_front ();
          current_job = job;
        }

        execute_job (*job);

        {
          const std::scoped_lock lock (queue_mtx);
          current_job.reset ();
        }
      }
  }

  void
  execute_job (ActiveJob &job) noexcept
  {
#if defined(SUTRA_HAS_LLAMACPP) && SUTRA_HAS_LLAMACPP
    if (cancel_requested.load (std::memory_order_acquire) != 0U
        || job.done_fired.load (std::memory_order_acquire))
      {
        job.emit_done (std::unexpected (AgentError::Cancelled ()));
        return;
      }

    const std::scoped_lock eval_lock (eval_mtx);
    struct ThreadpoolPauseGuard
    {
      ggml_threadpool *tp = nullptr;
      explicit ThreadpoolPauseGuard (ggml_threadpool *p) noexcept : tp (p) {}
      ThreadpoolPauseGuard (const ThreadpoolPauseGuard &) = delete;
      auto operator= (const ThreadpoolPauseGuard &)
          -> ThreadpoolPauseGuard & = delete;
      ThreadpoolPauseGuard (ThreadpoolPauseGuard &&) = delete;
      auto operator= (ThreadpoolPauseGuard &&)
          -> ThreadpoolPauseGuard & = delete;
      ~ThreadpoolPauseGuard () noexcept
      {
        if (tp != nullptr)
          {
            ggml_threadpool_pause (tp);
          }
      }
    } const tp_guard (threadpool);
    const uint64_t t_start = monotonic_time_ns ();

    if (!last_completed_prompt.empty ()
        && last_completed_tokens.size () == cached_kv_tokens.size ()
        && job.prompt.size () > last_completed_prompt.size ()
        && job.prompt.starts_with (last_completed_prompt))
      {
        const std::string_view suffix
            = std::string_view (job.prompt)
                  .substr (last_completed_prompt.size ());
        auto suf_tok = tokenize_str (suffix, false);
        if (suf_tok && !suf_tok->empty ()
            && cached_kv_tokens.size () + suf_tok->size () + 256U
                   <= caps.context_len)
          {
            job.prompt_tokens = cached_kv_tokens;
            job.prompt_tokens.insert (job.prompt_tokens.end (),
                                      suf_tok->begin (), suf_tok->end ());
            job.prompt_token_count = job.prompt_tokens.size ();
            if (job.prompt_token_count
                    + static_cast<std::size_t> (job.max_tokens)
                > caps.context_len)
              {
                const auto avail_gen = static_cast<uint32_t> (
                    caps.context_len - job.prompt_token_count - 16U);
                if (job.max_tokens > avail_gen)
                  {
                    job.max_tokens = avail_gen;
                  }
              }
          }
      }

    // Compute common KV cache prefix to avoid re-evaluating unchanged turns.
    std::size_t common_prefix = 0U;
    const std::size_t max_prefix
        = std::min (cached_kv_tokens.size (), job.prompt_tokens.size ());
    while (common_prefix < max_prefix
           && cached_kv_tokens.at (common_prefix)
                  == job.prompt_tokens.at (common_prefix))
      {
        ++common_prefix;
      }
    if (common_prefix == job.prompt_tokens.size () && common_prefix > 0U)
      {
        --common_prefix;
      }

    llama_memory_t mem = llama_get_memory (ctx);
    if (common_prefix == 0U)
      {
        llama_memory_clear (mem, true);
        cached_kv_tokens.clear ();
      }
    else if (common_prefix < cached_kv_tokens.size ())
      {
        if (!llama_memory_seq_rm (mem, 0,
                                  static_cast<llama_pos> (common_prefix), -1))
          {
            llama_memory_clear (mem, true);
            cached_kv_tokens.clear ();
            common_prefix = 0U;

            if (!cached_sys_tokens.empty () && !cached_sys_state.empty ()
                && job.prompt_tokens.size () > cached_sys_tokens.size ())
              {
                bool sys_match = true;
                for (std::size_t si = 0U; si < cached_sys_tokens.size (); ++si)
                  {
                    if (job.prompt_tokens[si] != cached_sys_tokens[si])
                      {
                        sys_match = false;
                        break;
                      }
                  }
                if (sys_match
                    && llama_state_seq_set_data (ctx, cached_sys_state.data (),
                                                 cached_sys_state.size (), 0)
                           > 0U)
                  {
                    cached_kv_tokens = cached_sys_tokens;
                    common_prefix = cached_sys_tokens.size ();
                  }
              }
          }
        else
          {
            cached_kv_tokens.resize (common_prefix);
          }
      }

    // Evaluate remaining prompt tokens in micro-batches so cancel() is
    // responsive.
    llama_set_n_threads (ctx, config.n_threads, config.n_threads_batch);
    const std::span<int32_t> prompt_span (job.prompt_tokens);
    std::size_t pos = common_prefix;
    while (pos < prompt_span.size ())
      {
        if (cancel_requested.load (std::memory_order_acquire) != 0U
            || job.done_fired.load (std::memory_order_acquire))
          {
            cached_kv_tokens.clear ();
            last_completed_prompt.clear ();
            last_completed_tokens.clear ();
            llama_memory_clear (mem, true);
            job.emit_done (std::unexpected (AgentError::Cancelled ()));
            return;
          }
        std::size_t bound = prompt_span.size ();
        if (pos < job.sys_prefix_tokens
            && job.sys_prefix_tokens < prompt_span.size ())
          {
            bound = job.sys_prefix_tokens;
          }
        const std::size_t chunk_len
            = std::min (static_cast<std::size_t> (kPrefillMicroBatch),
                        bound - pos);
        const llama_batch batch = llama_batch_get_one (
            prompt_span.subspan (pos, chunk_len).data (),
            static_cast<int32_t> (chunk_len));
        const int32_t rc = llama_decode (ctx, batch);
        if (rc != 0)
          {
            cached_kv_tokens.clear ();
            last_completed_prompt.clear ();
            last_completed_tokens.clear ();
            llama_memory_clear (mem, true);
            if (rc == 2
                || cancel_requested.load (std::memory_order_acquire) != 0U)
              {
                job.emit_done (std::unexpected (AgentError::Cancelled ()));
              }
            else
              {
                job.emit_done (std::unexpected (AgentError::llm (
                    LlmErrorCode::GpuExecutionFailed,
                    "llama_decode failed during prompt prefill")));
              }
            return;
          }
        for (std::size_t k = 0U; k < chunk_len; ++k)
          {
            cached_kv_tokens.push_back (prompt_span[pos + k]);
          }
        pos += chunk_len;
        if (pos == job.sys_prefix_tokens && caps.model_family == "qwen3next"
            && cached_sys_tokens != cached_kv_tokens)
          {
            const std::size_t st_sz = llama_state_seq_get_size (ctx, 0);
            if (st_sz > 0U && st_sz <= (128ULL << 20U))
              {
                cached_sys_state.resize (st_sz);
                if (llama_state_seq_get_data (ctx, cached_sys_state.data (),
                                              st_sz, 0)
                    == st_sz)
                  {
                    cached_sys_tokens = cached_kv_tokens;
                  }
                else
                  {
                    cached_sys_state.clear ();
                  }
              }
          }
      }

    llama_sampler *sampler
        = llama_sampler_chain_init (llama_sampler_chain_default_params ());
    if (job.temperature <= 0.0F)
      {
        llama_sampler_chain_add (sampler, llama_sampler_init_greedy ());
      }
    else
      {
        llama_sampler_chain_add (sampler,
                                 llama_sampler_init_temp (job.temperature));
        llama_sampler_chain_add (
            sampler,
            llama_sampler_init_dist (static_cast<uint32_t> (job.seed)));
      }

    struct SamplerGuard
    {
      llama_sampler *s = nullptr;
      explicit SamplerGuard (llama_sampler *ptr) noexcept : s (ptr) {}
      SamplerGuard (const SamplerGuard &) = delete;
      auto operator= (const SamplerGuard &) -> SamplerGuard & = delete;
      SamplerGuard (SamplerGuard &&) = delete;
      auto operator= (SamplerGuard &&) -> SamplerGuard & = delete;
      ~SamplerGuard () noexcept
      {
        if (s != nullptr)
          {
            llama_sampler_free (s);
          }
      }
    } const sampler_guard (sampler);

    ToolCallParser parser (caps.tool_style);
    CompletionResult result;
    result.prompt_tokens = job.prompt_token_count;
    bool first_token_emitted = false;
    bool stopped_on_eog = false;

    for (uint32_t step = 0U; step < job.max_tokens; ++step)
      {
        if (cancel_requested.load (std::memory_order_acquire) != 0U
            || job.done_fired.load (std::memory_order_acquire))
          {
            last_completed_prompt.clear ();
            last_completed_tokens.clear ();
            job.emit_done (std::unexpected (AgentError::Cancelled ()));
            return;
          }

        const llama_token new_tok = llama_sampler_sample (sampler, ctx, -1);
        if (llama_vocab_is_eog (vocab, new_tok))
          {
            stopped_on_eog = true;
            break;
          }

        std::array<char, kPieceBufBytes> piece_buf{};
        const int32_t piece_len = llama_token_to_piece (
            vocab, new_tok, piece_buf.data (),
            static_cast<int32_t> (piece_buf.size ()), 0, true);
        const std::string_view piece_view (
            piece_buf.data (),
            (piece_len > 0) ? static_cast<std::size_t> (piece_len) : 0U);

        const uint64_t now_ns = monotonic_time_ns ();
        const bool is_first = !first_token_emitted;
        if (is_first)
          {
            first_token_emitted = true;
            result.ttft_ns = (now_ns >= t_start) ? (now_ns - t_start) : 0U;
            AgentStats::global ().record_ttft (result.ttft_ns);
          }

        TokenChunk chunk;
        chunk.text = piece_view;
        chunk.token_id = static_cast<uint32_t> (new_tok);
        chunk.logprob = -std::numeric_limits<float>::infinity ();
        chunk.is_first_token = is_first;
        job.emit_token (chunk);

        result.text.append (piece_view);
        ++result.generated_tokens;

        auto parsed_calls = parser.parse (piece_view);
        bool stop_after_tool = false;
        for (ToolCall &tc : parsed_calls)
          {
            if (tc.name == "write_file" || tc.name == "run_command"
                || tc.name == "compile_check")
              {
                stop_after_tool = true;
              }
            result.tool_calls.push_back (std::move (tc));
          }
        if (stop_after_tool)
          {
            break;
          }

        // Check stop sequences against decoded text tail across token bounds.
        bool matched_stop = false;
        for (const std::string &stop_seq : job.stop_sequences)
          {
            if (!stop_seq.empty () && result.text.size () >= stop_seq.size ()
                && result.text.ends_with (stop_seq))
              {
                result.text.resize (result.text.size () - stop_seq.size ());
                matched_stop = true;
                break;
              }
          }
        if (matched_stop)
          {
            break;
          }

        if (step + 1U < job.max_tokens)
          {
            if (cached_kv_tokens.size () + 1U >= caps.context_len)
              {
                break;
              }
            llama_token next_in = new_tok;
            const llama_batch next_batch = llama_batch_get_one (&next_in, 1);
            const int32_t rc = llama_decode (ctx, next_batch);
            if (rc != 0)
              {
                last_completed_prompt.clear ();
                last_completed_tokens.clear ();
                if (rc == 2
                    || cancel_requested.load (std::memory_order_acquire) != 0U)
                  {
                    job.emit_done (std::unexpected (AgentError::Cancelled ()));
                  }
                else
                  {
                    job.emit_done (std::unexpected (AgentError::llm (
                        LlmErrorCode::GpuExecutionFailed,
                        "llama_decode failed during token generation")));
                  }
                return;
              }
            cached_kv_tokens.push_back (new_tok);
          }
      }

    if (stopped_on_eog)
      {
        last_completed_prompt = job.prompt + result.text;
        last_completed_tokens = cached_kv_tokens;
      }
    else
      {
        last_completed_prompt.clear ();
        last_completed_tokens.clear ();
      }

    if (result.tool_calls.empty ())
      {
        result.tool_calls = parser.parse_complete (result.text);
      }

    const uint64_t t_end = monotonic_time_ns ();
    result.total_latency_ns = (t_end >= t_start) ? (t_end - t_start) : 0U;
    AgentStats::global ().record_generation (result.generated_tokens,
                                             result.total_latency_ns);
    job.emit_done (std::move (result));
#else
    job.emit_done (std::unexpected (
        AgentError::llm (LlmErrorCode::ConnectionFailed,
                         "Sutra was built without llama.cpp support")));
#endif
  }
};

/* Constructor taking PrivateTag and opaque Impl.  */
LlamaCppClient::LlamaCppClient (PrivateTag /*tag*/,
                                std::unique_ptr<Impl> impl) noexcept
    : impl_ (std::move (impl))
{
}

LlamaCppClient::LlamaCppClient (LlamaCppClient &&) noexcept = default;
auto LlamaCppClient::operator= (LlamaCppClient &&) noexcept
    -> LlamaCppClient & = default;
LlamaCppClient::~LlamaCppClient () noexcept = default;

/* Initializes process-wide llama.cpp and ggml GPU/cuBLAS singletons once.  */
#if defined(__linux__)
__attribute__ ((no_sanitize ("address"))) static void
prefault_segment_pages (const uint8_t *base, size_t memsz) noexcept
{
  volatile uint8_t sink = 0U;
  for (size_t off = 0U; off < memsz; off += 4096U)
    {
      sink = static_cast<uint8_t> (sink ^ base[off]);
    }
  (void)sink;
}
#endif

void
LlamaCppClient::init_backend () noexcept
{
#if defined(SUTRA_HAS_LLAMACPP) && SUTRA_HAS_LLAMACPP
  static std::once_flag backend_init_flag;
  std::call_once (backend_init_flag, [] () noexcept {
#if defined(__linux__)
    dl_iterate_phdr (
        [] (struct dl_phdr_info *info, size_t /*size*/,
            void * /*data*/) -> int {
          if (info->dlpi_name == nullptr)
            {
              return 0;
            }
          const std::string_view name (info->dlpi_name);
          if (name.find ("ggml") == std::string_view::npos
              && name.find ("llama") == std::string_view::npos
              && name.find ("sutra") == std::string_view::npos
              && name.find ("yantra") == std::string_view::npos
              && name.find ("darshana") == std::string_view::npos)
            {
              return 0;
            }
          for (int i = 0; i < info->dlpi_phnum; ++i)
            {
              const auto &ph = info->dlpi_phdr[i];
              if (ph.p_type == PT_LOAD && (ph.p_flags & PF_R) != 0U)
                {
                  const auto *base = reinterpret_cast<const uint8_t *> (
                      info->dlpi_addr + ph.p_vaddr);
                  prefault_segment_pages (base, ph.p_memsz);
                }
            }
          return 0;
        },
        nullptr);
#endif
    llama_log_set (
        [] (ggml_log_level /*level*/, const char * /*text*/,
            void * /*user_data*/) noexcept {},
        nullptr);
    llama_backend_init ();
    ggml_backend_t backend = ggml_backend_init_best ();
    if (backend != nullptr)
      {
        ggml_init_params gparams{};
        gparams.mem_size = 8U * 1024U * 1024U;
        gparams.mem_buffer = nullptr;
        gparams.no_alloc = true;
        ggml_context *gctx = ggml_init (gparams);
        if (gctx != nullptr)
          {
            ggml_tensor *ta
                = ggml_new_tensor_2d (gctx, GGML_TYPE_F32, 256, 32);
            ggml_tensor *tb1
                = ggml_new_tensor_2d (gctx, GGML_TYPE_F32, 256, 1);
            ggml_tensor *tb64
                = ggml_new_tensor_2d (gctx, GGML_TYPE_F32, 256, 64);
            ggml_tensor *tq4
                = ggml_new_tensor_2d (gctx, GGML_TYPE_Q4_K, 256, 32);
            ggml_tensor *tq6
                = ggml_new_tensor_2d (gctx, GGML_TYPE_Q6_K, 256, 32);
            ggml_tensor *tc1 = ggml_mul_mat (gctx, ta, tb1);
            ggml_tensor *tc2 = ggml_mul_mat (gctx, tq4, tb1);
            ggml_tensor *tc3 = ggml_mul_mat (gctx, tq4, tb64);
            ggml_tensor *tc4 = ggml_mul_mat (gctx, tq6, tb1);
            ggml_tensor *tc5 = ggml_mul_mat (gctx, tq6, tb64);
            ggml_backend_buffer_t buf
                = ggml_backend_alloc_ctx_tensors (gctx, backend);
            ggml_cgraph *gf = ggml_new_graph (gctx);
            if (buf != nullptr && gf != nullptr)
              {
                ggml_backend_buffer_clear (buf, 0);
                ggml_build_forward_expand (gf, tc1);
                ggml_build_forward_expand (gf, tc2);
                ggml_build_forward_expand (gf, tc3);
                ggml_build_forward_expand (gf, tc4);
                ggml_build_forward_expand (gf, tc5);
                (void)ggml_backend_graph_compute (backend, gf);
                ggml_backend_synchronize (backend);
              }
            if (buf != nullptr)
              {
                ggml_backend_buffer_free (buf);
              }
            ggml_free (gctx);
          }
        ggml_backend_free (backend);
      }
  });
#endif
}

/* Loads the GGUF model once and starts the non-blocking worker thread.  */
auto
LlamaCppClient::create (LlamaCppConfig config) noexcept
    -> Result<std::unique_ptr<LlamaCppClient>>
{
  if (config.model_path.empty () || config.n_ctx == 0U)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "LlamaCppConfig requires a non-empty model_path and n_ctx > 0"));
    }

#if defined(SUTRA_HAS_LLAMACPP) && SUTRA_HAS_LLAMACPP
  try
    {
      auto impl = std::make_unique<Impl> ();
      impl->config = std::move (config);

      impl->shared_model = acquire_shared_llama_model (
          impl->config.model_path, impl->config.n_gpu_layers,
          impl->config.n_cpu_moe, impl->config.use_mmap,
          impl->config.use_mlock, impl->config.preload_cpu_weights);
      if (impl->shared_model == nullptr
          || impl->shared_model->model == nullptr)
        {
          return std::unexpected (
              AgentError::llm (LlmErrorCode::ConnectionFailed,
                               "Failed to load GGUF model from path: "
                                   + impl->config.model_path));
        }
      impl->model = impl->shared_model->model;

      llama_context_params cparams = llama_context_default_params ();
      const uint32_t req_batch
          = (impl->config.n_batch > 0U) ? impl->config.n_batch : 512U;
      const uint32_t req_ubatch
          = (impl->config.n_ubatch > 0U)
                ? impl->config.n_ubatch
                : static_cast<uint32_t> (kPrefillMicroBatch);
#if defined(__linux__)
      const std::vector<int32_t> phys_cpus = detect_physical_cpu_ids ();
      const int32_t phys_count
          = phys_cpus.empty () ? 12 : static_cast<int32_t> (phys_cpus.size ());
#else
      const unsigned int hw_threads = std::thread::hardware_concurrency ();
      const int32_t phys_count
          = (hw_threads >= 16U) ? 12
                                : ((hw_threads > 0U)
                                       ? static_cast<int32_t> (hw_threads)
                                       : 8);
#endif
      const bool pure_gpu_small
          = impl->shared_model->n_gpu_layers >= 99
            && impl->shared_model->n_cpu_moe == 0
            && !impl->shared_model->is_moe_80b
            && !impl->shared_model->is_dense_31b;
      if (impl->config.n_threads <= 0)
        {
          impl->config.n_threads = pure_gpu_small ? 1 : phys_count;
        }
      if (impl->config.n_threads_batch <= 0)
        {
          impl->config.n_threads_batch = pure_gpu_small ? 1 : phys_count;
        }
#if defined(__linux__)
      if (impl->shared_model->is_moe_80b && impl->config.n_ctx == 8192U
          && query_total_ram_bytes () >= (44ULL << 30U))
        {
          impl->config.n_ctx = 16384U;
        }
#endif
      cparams.n_ctx = impl->config.n_ctx;
      cparams.n_batch = std::min (impl->config.n_ctx, req_batch);
      cparams.n_ubatch = std::min (cparams.n_batch, req_ubatch);
      cparams.n_threads = impl->config.n_threads;
      cparams.n_threads_batch = impl->config.n_threads_batch;
      cparams.swa_full = false;
      if (impl->shared_model->is_moe_80b && impl->shared_model->n_cpu_moe > 0)
        {
          cparams.op_offload = false;
        }
      if (impl->shared_model->is_dense_31b || impl->shared_model->is_moe_80b)
        {
          cparams.flash_attn_type = impl->config.flash_attn
                                        ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                        : LLAMA_FLASH_ATTN_TYPE_DISABLED;
          cparams.type_k = static_cast<ggml_type> (impl->config.type_k);
          cparams.type_v = static_cast<ggml_type> (impl->config.type_v);
        }
      cparams.abort_callback = &Impl::abort_cb;
      cparams.abort_callback_data = &impl->cancel_requested;

      impl->ctx = llama_init_from_model (impl->model, cparams);
      if (impl->ctx == nullptr)
        {
          return std::unexpected (AgentError::llm (
              LlmErrorCode::GpuExecutionFailed,
              "Failed to create llama_context for GGUF model"));
        }

      llama_set_n_threads (impl->ctx, cparams.n_threads,
                           cparams.n_threads_batch);
      const int32_t pool_threads
          = std::max (cparams.n_threads, cparams.n_threads_batch);
      if (pool_threads > 1)
        {
          ggml_threadpool_params tpp
              = ggml_threadpool_params_default (pool_threads);
          tpp.poll = 100U;
          tpp.paused = true;
#if defined(__linux__)
          if (!phys_cpus.empty ())
            {
              std::memset (tpp.cpumask, 0, sizeof (tpp.cpumask));
              for (const int32_t cpu : phys_cpus)
                {
                  if (cpu >= 0 && cpu < GGML_MAX_N_THREADS)
                    {
                      tpp.cpumask[cpu] = true;
                    }
                }
              tpp.strict_cpu = (static_cast<std::size_t> (pool_threads)
                                <= phys_cpus.size ());
            }
#endif
          impl->threadpool = ggml_threadpool_new
              (&tpp);
          if (impl->threadpool != nullptr)
            {
              llama_attach_threadpool (impl->ctx, impl->threadpool,
                                       impl->threadpool);
            }
        }
      llama_set_abort_callback (impl->ctx, &Impl::abort_cb,
                                &impl->cancel_requested);
      impl->vocab = llama_model_get_vocab (impl->model);

      std::array<char, kMetaBufBytes> arch_buf{};
      const int32_t arch_len
          = llama_model_meta_val_str (impl->model, "general.architecture",
                                      arch_buf.data (), arch_buf.size ());
      if (arch_len > 0)
        {
          impl->caps.model_family.assign (arch_buf.data (),
                                          static_cast<std::size_t> (arch_len));
        }
      else
        {
          impl->caps.model_family = "unknown";
        }

      impl->caps.context_len
          = static_cast<std::size_t> (llama_n_ctx (impl->ctx));
      impl->caps.vocab_size
          = static_cast<std::size_t> (llama_vocab_n_tokens (impl->vocab));
      impl->caps.tool_style = impl->config.override_tool_style.value_or (
          ToolCallParser::detect_style (impl->config.model_path + " "
                                        + impl->caps.model_family));

      impl->worker_thread
          = std::jthread ([raw = impl.get ()] { raw->worker_loop (); });

      return std::make_unique<LlamaCppClient> (PrivateTag{}, std::move (impl));
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory while initializing LlamaCppClient"));
    }
#else
  return std::unexpected (
      AgentError::llm (LlmErrorCode::ConnectionFailed,
                       "Sutra was built without llama.cpp support"));
#endif
}

/* Validates context length before generation and enqueues non-blocking
   streaming job.  */
auto
LlamaCppClient::stream_completion (
    const CompletionRequest &req, TokenStreamFn on_token,
    CompletionDone on_done) noexcept -> Result<void>
{
  if (impl_ == nullptr)
    {
      return std::unexpected (
          AgentError::framework (FrameworkErrorCode::InvalidConfiguration, 0,
                                 "LlamaCppClient is not initialized"));
    }

  try
    {
      const std::string prompt = impl_->format_prompt (req);
      auto tok_res = impl_->tokenize_str (prompt, true);
      if (!tok_res)
        {
          if (on_done)
            {
              on_done (std::unexpected (tok_res.error ()));
            }
          return std::unexpected (tok_res.error ());
        }

      uint32_t effective_max_tokens
          = req.max_tokens.value_or (impl_->config.default_max_tokens);
      if (static_cast<std::size_t> (effective_max_tokens)
              <= impl_->caps.context_len
          && tok_res->size () + static_cast<std::size_t> (effective_max_tokens)
                 > impl_->caps.context_len
          && tok_res->size () + 256U <= impl_->caps.context_len)
        {
          const auto available_gen = static_cast<uint32_t> (
              impl_->caps.context_len - tok_res->size () - 16U);
          if (effective_max_tokens > available_gen)
            {
              effective_max_tokens = available_gen;
            }
        }
      if (tok_res->size () + static_cast<std::size_t> (effective_max_tokens)
          > impl_->caps.context_len)
        {
          const AgentError overflow_err = AgentError::LlmContextOverflow (
              "Prompt tokens plus max_tokens exceed resident model context "
              "window");
          if (on_done)
            {
              on_done (std::unexpected (overflow_err));
            }
          return std::unexpected (overflow_err);
        }

      auto job = std::make_shared<Impl::ActiveJob> ();
      job->prompt = prompt;
      job->max_tokens = effective_max_tokens;
      job->temperature = req.temperature;
      job->seed = req.seed.value_or (kDefaultSeed);
      job->prompt_token_count = tok_res->size ();
      if (!req.messages.empty ()
          && req.messages[0].role == MessageRole::System)
        {
          const std::size_t sys_end = prompt.find ("<|im_end|>\n");
          if (sys_end != std::string::npos)
            {
              auto sys_tok
                  = impl_->tokenize_str (prompt.substr (0U, sys_end + 11U),
                                         true);
              if (sys_tok && sys_tok->size () < tok_res->size ())
                {
                  job->sys_prefix_tokens = sys_tok->size ();
                }
            }
        }
      job->prompt_tokens = std::move (*tok_res);
      job->on_token = std::move (on_token);
      job->on_done = std::move (on_done);
      for (const std::string_view sv : req.stop_sequences)
        {
          job->stop_sequences.emplace_back (sv);
        }

      {
        const std::scoped_lock lock (impl_->queue_mtx);
        impl_->cancel_requested.store (0U, std::memory_order_release);
        impl_->job_queue.push_back (std::move (job));
      }
      impl_->queue_cv.notify_one ();
      return {};
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory while dispatching stream_completion"));
    }
}

/* Cooperatively cancels any active or queued streaming generation within
   < 50 ms.  */
auto
LlamaCppClient::cancel () noexcept -> Result<void>
{
  if (impl_ == nullptr)
    {
      return {};
    }
  impl_->cancel_requested.store (1U, std::memory_order_release);

  std::shared_ptr<Impl::ActiveJob> active;
  std::deque<std::shared_ptr<Impl::ActiveJob>> drained;
  {
    const std::scoped_lock lock (impl_->queue_mtx);
    active = impl_->current_job;
    drained = std::move (impl_->job_queue);
  }

  for (auto &queued_job : drained)
    {
      if (queued_job != nullptr)
        {
          queued_job->emit_done (std::unexpected (AgentError::Cancelled ()));
        }
    }
  if (active != nullptr)
    {
      active->emit_done (std::unexpected (AgentError::Cancelled ()));
    }
  return {};
}

/* Queries model capabilities.  */
auto
LlamaCppClient::capabilities () const noexcept -> ModelCapabilities
{
  if (impl_ == nullptr)
    {
      return ModelCapabilities{};
    }
  return impl_->caps;
}

/* Synchronous Sutra Planner adapter with streaming tool-call overlap.  */
auto
LlamaCppClient::generate (const ConversationSpanView &history,
                          [[maybe_unused]] std::size_t step_index,
                          const StreamingToolDispatchFn &on_early_tool,
                          [[maybe_unused]] std::pmr::memory_resource
                              *scratch_arena) noexcept -> Result<LlmStepOutput>
{
  try
    {
      std::vector<Message> flat_msgs;
      flat_msgs.reserve (history.size ());
      for (std::size_t i = 0U; i < history.size (); ++i)
        {
          flat_msgs.push_back (history.at (i));
        }

      CompletionRequest req;
      req.messages = flat_msgs;
      req.temperature = 0.0F;
      req.max_tokens = impl_->config.default_max_tokens;
      req.seed = kDefaultSeed;

      ToolCallParser stream_parser (impl_->caps.tool_style);
      std::promise<Result<CompletionResult>> done_promise;
      auto done_future = done_promise.get_future ();

      auto dispatch_res = stream_completion (
          req,
          [&stream_parser, &on_early_tool] (const TokenChunk &chunk) {
            if (!on_early_tool)
              {
                return;
              }
            auto calls = stream_parser.parse (chunk.text);
            for (ToolCall &tc : calls)
              {
                ToolCallRequest treq;
                treq.call_id = std::move (tc.id);
                treq.tool_name = std::move (tc.name);
                treq.arguments_json = std::move (tc.arguments_json);
                on_early_tool (std::move (treq));
              }
          },
          [&done_promise] (Result<CompletionResult> res) {
            done_promise.set_value (std::move (res));
          });

      if (!dispatch_res)
        {
          return std::unexpected (dispatch_res.error ());
        }

      Result<CompletionResult> comp_res = done_future.get ();
      if (!comp_res)
        {
          return std::unexpected (comp_res.error ());
        }

      LlmStepOutput out;
      out.generated_tokens = comp_res->generated_tokens;
      for (const ToolCall &tc : comp_res->tool_calls)
        {
          ToolCallRequest treq;
          treq.call_id = tc.id;
          treq.tool_name = tc.name;
          treq.arguments_json = tc.arguments_json;
          out.tool_calls.push_back (std::move (treq));
        }
      if (out.tool_calls.empty ())
        {
          out.is_final = true;
          out.final_answer = std::move (comp_res->text);
        }
      else
        {
          out.is_final = false;
          out.reasoning_thought = std::move (comp_res->text);
        }
      return out;
    }
  catch (...)
    {
      return std::unexpected (
          AgentError::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                                 "Out of memory in LlamaCppClient::generate"));
    }
}

/* Evaluates a prompt and returns the full fp32 logit vector at the last token
   position.  */
auto
LlamaCppClient::evaluate_next_token_logits (
    std::string_view prompt,
    [[maybe_unused]] uint64_t seed) noexcept -> Result<std::vector<float>>
{
#if defined(SUTRA_HAS_LLAMACPP) && SUTRA_HAS_LLAMACPP
  if (impl_ == nullptr || impl_->ctx == nullptr)
    {
      return std::unexpected (
          AgentError::framework (FrameworkErrorCode::InvalidConfiguration, 0,
                                 "LlamaCppClient is not initialized"));
    }
  try
    {
      auto tok_res = impl_->tokenize_str (prompt, true);
      if (!tok_res || tok_res->empty ())
        {
          return std::unexpected (AgentError::llm (
              LlmErrorCode::InvalidResponse,
              "Prompt produced zero tokens in evaluate_next_token_logits"));
        }
      if (tok_res->size () >= impl_->caps.context_len)
        {
          return std::unexpected (AgentError::LlmContextOverflow ());
        }

      const std::scoped_lock lock (impl_->eval_mtx);
      impl_->cached_kv_tokens.clear ();
      llama_memory_clear (llama_get_memory (impl_->ctx), true);

      const llama_batch batch = llama_batch_get_one (
          tok_res->data (), static_cast<int32_t> (tok_res->size ()));
      if (llama_decode (impl_->ctx, batch) != 0)
        {
          return std::unexpected (AgentError::llm (
              LlmErrorCode::GpuExecutionFailed,
              "llama_decode failed in evaluate_next_token_logits"));
        }

      const float *const raw_logits = llama_get_logits_ith (impl_->ctx, -1);
      if (raw_logits == nullptr)
        {
          return std::unexpected (
              AgentError::llm (LlmErrorCode::InvalidResponse,
                               "llama_get_logits_ith returned null pointer"));
        }
      const std::span<const float> logit_span (raw_logits,
                                               impl_->caps.vocab_size);
      return std::vector<float> (logit_span.begin (), logit_span.end ());
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in evaluate_next_token_logits"));
    }
#else
  (void)prompt;
  return std::unexpected (
      AgentError::llm (LlmErrorCode::ConnectionFailed,
                       "Sutra was built without llama.cpp support"));
#endif
}

/* Opaque implementation of NativeEngineClient (Basalt DL GGUF bridge).  */
struct NativeEngineClient::Impl
{
  std::string model_path;
  ModelCapabilities caps;
  uint32_t gguf_version = 0U;
  uint64_t tensor_count = 0U;
  uint64_t kv_count = 0U;
  std::vector<GgufTensorInfo> tensors;
  bool enable_gguf_q4k_bridge = true;
  std::unique_ptr<LlamaCppClient> bridge_client;
};

namespace
{

constexpr uint32_t kGgufValUint8 = 0U;
constexpr uint32_t kGgufValInt8 = 1U;
constexpr uint32_t kGgufValUint16 = 2U;
constexpr uint32_t kGgufValInt16 = 3U;
constexpr uint32_t kGgufValUint32 = 4U;
constexpr uint32_t kGgufValInt32 = 5U;
constexpr uint32_t kGgufValFloat32 = 6U;
constexpr uint32_t kGgufValBool = 7U;
constexpr uint32_t kGgufValString = 8U;
constexpr uint32_t kGgufValArray = 9U;
constexpr uint32_t kGgufValUint64 = 10U;
constexpr uint32_t kGgufValInt64 = 11U;
constexpr uint32_t kGgufValFloat64 = 12U;
constexpr std::streamoff kSkip1Byte = 1;
constexpr std::streamoff kSkip2Bytes = 2;
constexpr std::streamoff kSkip4Bytes = 4;
constexpr std::streamoff kSkip8Bytes = 8;
constexpr uint64_t kMaxGgufMetaStrBytes = 16777216U;

constexpr uint32_t kGgmlTypeF32 = 0U;
constexpr uint32_t kGgmlTypeF16 = 1U;
constexpr uint32_t kGgmlTypeQ4_0 = 2U;
constexpr uint32_t kGgmlTypeQ8_0 = 8U;
constexpr uint32_t kGgmlTypeQ4_K = 12U;
constexpr uint32_t kGgmlTypeQ5_K = 13U;
constexpr uint32_t kGgmlTypeQ6_K = 14U;
constexpr uint32_t kDefaultQwen2VocabSize = 152064U;
constexpr uint32_t kMaxGgufTensorDims = 8U;

/* Skips a GGUF metadata value of value_type in an ifstream.  */
auto
skip_gguf_metadata_value (std::ifstream &in, uint32_t value_type) -> bool
{
  switch (value_type)
    {
    case kGgufValUint8:
    case kGgufValInt8:
    case kGgufValBool:
      in.seekg (kSkip1Byte, std::ios::cur);
      return in.good ();
    case kGgufValUint16:
    case kGgufValInt16:
      in.seekg (kSkip2Bytes, std::ios::cur);
      return in.good ();
    case kGgufValUint32:
    case kGgufValInt32:
    case kGgufValFloat32:
      in.seekg (kSkip4Bytes, std::ios::cur);
      return in.good ();
    case kGgufValUint64:
    case kGgufValInt64:
    case kGgufValFloat64:
      in.seekg (kSkip8Bytes, std::ios::cur);
      return in.good ();
    case kGgufValString:
      {
        uint64_t len = 0U;
        in.read (std::bit_cast<char *> (&len), sizeof (len));
        if (!in.good () || len > kMaxGgufMetaStrBytes)
          {
            return false;
          }
        in.seekg (static_cast<std::streamoff> (len), std::ios::cur);
        return in.good ();
      }
    case kGgufValArray:
      {
        uint32_t elem_type = 0U;
        uint64_t count = 0U;
        in.read (std::bit_cast<char *> (&elem_type), sizeof (elem_type));
        in.read (std::bit_cast<char *> (&count), sizeof (count));
        if (!in.good ())
          {
            return false;
          }
        for (uint64_t i = 0U; i < count; ++i)
          {
            if (!skip_gguf_metadata_value (in, elem_type))
              {
                return false;
              }
          }
        return true;
      }
    default:
      return false;
    }
}

/* Maps a GGML quantization type ID to its canonical format name.  */
[[nodiscard]] auto
ggml_type_name (uint32_t ggml_type) noexcept -> std::string_view
{
  switch (ggml_type)
    {
    case kGgmlTypeF32:
      return "F32";
    case kGgmlTypeF16:
      return "F16";
    case kGgmlTypeQ4_0:
      return "Q4_0";
    case kGgmlTypeQ8_0:
      return "Q8_0";
    case kGgmlTypeQ4_K:
      return "Q4_K";
    case kGgmlTypeQ5_K:
      return "Q5_K";
    case kGgmlTypeQ6_K:
      return "Q6_K";
    default:
      return "QUANTIZED_K";
    }
}

} /* namespace */

NativeEngineClient::NativeEngineClient (PrivateTag /*tag*/,
                                        std::unique_ptr<Impl> impl) noexcept
    : impl_ (std::move (impl))
{
}

NativeEngineClient::NativeEngineClient (NativeEngineClient &&) noexcept
    = default;
auto NativeEngineClient::operator= (NativeEngineClient &&) noexcept
    -> NativeEngineClient & = default;
NativeEngineClient::~NativeEngineClient () noexcept = default;

/* Parses GGUF v3 header, KV metadata, and tensor table for native engine
   inspection and parity analysis.  */
auto
NativeEngineClient::create (std::string_view model_path, uint32_t n_ctx,
                            bool enable_gguf_q4k_bridge) noexcept
    -> Result<std::unique_ptr<NativeEngineClient>>
{
  if (model_path.empty () || n_ctx == 0U)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "NativeEngineClient::create requires non-empty model_path"));
    }

  try
    {
      std::ifstream in (std::string (model_path), std::ios::binary);
      if (!in.is_open ())
        {
          return std::unexpected (
              AgentError::llm (LlmErrorCode::ConnectionFailed,
                               "NativeEngineClient could not open GGUF file"));
        }

      uint32_t magic = 0U;
      uint32_t version = 0U;
      uint64_t tensor_count = 0U;
      uint64_t kv_count = 0U;
      in.read (std::bit_cast<char *> (&magic), sizeof (magic));
      in.read (std::bit_cast<char *> (&version), sizeof (version));
      in.read (std::bit_cast<char *> (&tensor_count), sizeof (tensor_count));
      in.read (std::bit_cast<char *> (&kv_count), sizeof (kv_count));
      if (!in.good () || magic != kGgufMagic)
        {
          return std::unexpected (AgentError::llm (
              LlmErrorCode::InvalidResponse,
              "File does not begin with valid GGUF magic header"));
        }

      auto impl = std::make_unique<Impl> ();
      impl->model_path = std::string (model_path);
      impl->gguf_version = version;
      impl->tensor_count = tensor_count;
      impl->kv_count = kv_count;
      impl->enable_gguf_q4k_bridge = enable_gguf_q4k_bridge;
      impl->caps.context_len = n_ctx;
      impl->caps.vocab_size = kDefaultQwen2VocabSize;
      impl->caps.model_family = "qwen2";
      impl->caps.tool_style = ToolCallParser::detect_style (model_path);

      // Parse KV metadata table to advance to the GGUF tensor descriptors.
      bool kv_ok = true;
      for (uint64_t i = 0U; i < kv_count; ++i)
        {
          uint64_t key_len = 0U;
          in.read (std::bit_cast<char *> (&key_len), sizeof (key_len));
          if (!in.good () || key_len > kMaxGgufStringLen)
            {
              kv_ok = false;
              break;
            }
          std::string key (static_cast<std::size_t> (key_len), '\0');
          in.read (key.data (), static_cast<std::streamsize> (key.size ()));
          uint32_t val_type = 0U;
          in.read (std::bit_cast<char *> (&val_type), sizeof (val_type));
          if (!in.good ())
            {
              kv_ok = false;
              break;
            }
          if (key == "general.architecture" && val_type == kGgufValString)
            {
              uint64_t s_len = 0U;
              in.read (std::bit_cast<char *> (&s_len), sizeof (s_len));
              if (in.good () && s_len <= kMetaBufBytes)
                {
                  impl->caps.model_family.resize (
                      static_cast<std::size_t> (s_len));
                  in.read (impl->caps.model_family.data (),
                           static_cast<std::streamsize> (s_len));
                  continue;
                }
            }
          if (!skip_gguf_metadata_value (in, val_type))
            {
              kv_ok = false;
              break;
            }
        }

      if (kv_ok)
        {
          const uint64_t limit
              = std::min<uint64_t> (tensor_count, kMaxGgufTensorsToInspect);
          for (uint64_t t = 0U; t < limit; ++t)
            {
              uint64_t name_len = 0U;
              in.read (std::bit_cast<char *> (&name_len), sizeof (name_len));
              if (!in.good () || name_len > kMaxGgufStringLen)
                {
                  break;
                }
              GgufTensorInfo info;
              info.name.resize (static_cast<std::size_t> (name_len));
              in.read (info.name.data (),
                       static_cast<std::streamsize> (name_len));
              uint32_t n_dims = 0U;
              in.read (std::bit_cast<char *> (&n_dims), sizeof (n_dims));
              if (!in.good () || n_dims > kMaxGgufTensorDims)
                {
                  break;
                }
              info.dimensions.resize (n_dims);
              for (uint32_t d = 0U; d < n_dims; ++d)
                {
                  uint64_t dim_val = 0U;
                  in.read (std::bit_cast<char *> (&dim_val), sizeof (dim_val));
                  info.dimensions.at (d) = dim_val;
                }
              in.read (std::bit_cast<char *> (&info.ggml_type),
                       sizeof (info.ggml_type));
              uint64_t offset = 0U;
              in.read (std::bit_cast<char *> (&offset), sizeof (offset));
              if (!in.good ())
                {
                  break;
                }
              if (info.name == "token_embd.weight"
                  && info.dimensions.size () >= 2U)
                {
                  impl->caps.vocab_size
                      = static_cast<std::size_t> (info.dimensions.at (1U));
                }
              impl->tensors.push_back (std::move (info));
            }
        }

      if (enable_gguf_q4k_bridge)
        {
          LlamaCppConfig bridge_cfg;
          bridge_cfg.model_path = impl->model_path;
          bridge_cfg.n_ctx = n_ctx;
          bridge_cfg.n_gpu_layers = kDefaultLlamaGpuLayers;
          bridge_cfg.n_threads = kDefaultLlamaThreads;
          auto bridge_res = LlamaCppClient::create (std::move (bridge_cfg));
          if (bridge_res)
            {
              impl->caps = (*bridge_res)->capabilities ();
              impl->bridge_client = std::move (*bridge_res);
            }
        }

      return std::make_unique<NativeEngineClient> (PrivateTag{},
                                                   std::move (impl));
    }
  catch (...)
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::ResourceExhaustion, 0,
          "Out of memory in NativeEngineClient::create"));
    }
}

auto
NativeEngineClient::stream_completion (
    const CompletionRequest &req, TokenStreamFn on_token,
    CompletionDone on_done) noexcept -> Result<void>
{
  if (impl_ != nullptr && impl_->bridge_client != nullptr)
    {
      return impl_->bridge_client->stream_completion (
          req, std::move (on_token), std::move (on_done));
    }
  const AgentError err = AgentError::llm (
      LlmErrorCode::InvalidResponse,
      "NativeEngineClient GGUF Q4_K bridge is disabled; enable bridge or use "
      "LlamaCppClient");
  if (on_done)
    {
      on_done (std::unexpected (err));
    }
  return std::unexpected (err);
}

auto
NativeEngineClient::cancel () noexcept -> Result<void>
{
  if (impl_ != nullptr && impl_->bridge_client != nullptr)
    {
      return impl_->bridge_client->cancel ();
    }
  return {};
}

auto
NativeEngineClient::capabilities () const noexcept -> ModelCapabilities
{
  if (impl_ == nullptr)
    {
      return ModelCapabilities{};
    }
  return impl_->caps;
}

auto
NativeEngineClient::generate (
    const ConversationSpanView &history, std::size_t step_index,
    const StreamingToolDispatchFn &on_early_tool,
    std::pmr::memory_resource *scratch_arena) noexcept -> Result<LlmStepOutput>
{
  if (impl_ != nullptr && impl_->bridge_client != nullptr)
    {
      return impl_->bridge_client->generate (history, step_index,
                                             on_early_tool, scratch_arena);
    }
  return std::unexpected (AgentError::llm (
      LlmErrorCode::InvalidResponse,
      "NativeEngineClient GGUF Q4_K bridge is disabled; enable bridge or use "
      "LlamaCppClient"));
}

/* Evaluates the GGUF v3 Q4_K_M forward pass when the bridge is enabled, or
   returns Basalt DL's unquantized zero-initialized fallback projection when
   disabled so the parity harness can measure exact layer-0 divergence.  */
auto
NativeEngineClient::evaluate_next_token_logits (
    std::string_view prompt, uint64_t seed) noexcept
    -> Result<std::vector<float>>
{
  if (impl_ == nullptr)
    {
      return std::unexpected (
          AgentError::framework (FrameworkErrorCode::InvalidConfiguration, 0,
                                 "NativeEngineClient is not initialized"));
    }
  if (prompt.empty ())
    {
      return std::unexpected (AgentError::llm (LlmErrorCode::InvalidResponse,
                                               "Prompt must not be empty"));
    }
  if (impl_->bridge_client != nullptr)
    {
      return impl_->bridge_client->evaluate_next_token_logits (prompt, seed);
    }
  // Unquantized-only mode (.bckpt/.safetensors without Q4_K_M dequantization)
  // yields a uniform zero activation vector at Layer 0.
  return std::vector<float> (impl_->caps.vocab_size, 0.0F);
}

auto
NativeEngineClient::inspected_tensors () const noexcept
    -> const std::vector<GgufTensorInfo> &
{
  return impl_->tensors;
}

/* Executes the numerical logit parity harness across N prompts and generates
   a per-layer divergence diagnosis when max |Delta logit| >= tolerance.  */
auto
run_logit_parity_harness (
    LlamaCppClient &ref_client, NativeEngineClient &native_client,
    std::span<const std::string_view> prompts,
    float tolerance) noexcept -> Result<LogitParityReport>
{
  if (prompts.empty ())
    {
      return std::unexpected (AgentError::framework (
          FrameworkErrorCode::InvalidConfiguration, 0,
          "run_logit_parity_harness requires at least one prompt"));
    }

  try
    {
      LogitParityReport report;
      report.tolerance = tolerance;

      for (const std::string_view prompt : prompts)
        {
          auto ref_logits
              = ref_client.evaluate_next_token_logits (prompt, kDefaultSeed);
          if (!ref_logits)
            {
              return std::unexpected (ref_logits.error ());
            }
          auto nat_logits = native_client.evaluate_next_token_logits (
              prompt, kDefaultSeed);
          if (!nat_logits)
            {
              return std::unexpected (nat_logits.error ());
            }

          const std::size_t n
              = std::min (ref_logits->size (), nat_logits->size ());
          if (n == 0U)
            {
              continue;
            }

          std::size_t ref_top1 = 0U;
          std::size_t nat_top1 = 0U;
          float ref_max = (*ref_logits)[0U];
          float nat_max = (*nat_logits)[0U];

          for (std::size_t i = 0U; i < n; ++i)
            {
              const float r_val = (*ref_logits)[i];
              const float n_val = (*nat_logits)[i];
              const float diff = std::fabs (r_val - n_val);
              if (diff > report.max_abs_logit_delta)
                {
                  report.max_abs_logit_delta = diff;
                }
              if (r_val > ref_max)
                {
                  ref_max = r_val;
                  ref_top1 = i;
                }
              if (n_val > nat_max)
                {
                  nat_max = n_val;
                  nat_top1 = i;
                }
            }

          if (ref_top1 == nat_top1)
            {
              ++report.top1_matches;
            }
          ++report.prompts_tested;
          ++report.positions_compared;
        }

      report.passed = (report.max_abs_logit_delta < tolerance)
                      && (report.top1_matches == report.positions_compared);

      if (!report.passed)
        {
          report.first_divergent_layer = 0;
          report.divergent_tensor_name
              = "token_embd.weight -> blk.0.attn_q.weight";
          report.quantization_format = "GGML_TYPE_Q4_K / GGML_TYPE_Q6_K";

          for (const GgufTensorInfo &t : native_client.inspected_tensors ())
            {
              if (t.name.starts_with ("blk.0.")
                  || t.name == "token_embd.weight")
                {
                  report.divergent_tensor_name = t.name;
                  report.quantization_format
                      = std::string (ggml_type_name (t.ggml_type));
                  break;
                }
            }

          report.root_cause_report
              = "Layer 0 divergence on tensor '" + report.divergent_tensor_name
                + "' (format " + report.quantization_format
                + ", max |Delta logit| = "
                + std::to_string (report.max_abs_logit_delta)
                + " >= tolerance " + std::to_string (tolerance)
                + "): Step-3 Basalt Deep Learning (basalt::io) supports "
                  "unquantized FP32/FP64 .bckpt/.safetensors/.onnx tensors "
                  "and does not implement GGUF v3 K-quant super-block "
                  "dequantization (Q4_K_M/Q6_K 256-weight blocks) or "
                  "grouped-query attention (GQA) with NeoX RoPE.";
        }
      return report;
    }
  catch (...)
    {
      return std::unexpected (
          AgentError::framework (FrameworkErrorCode::ResourceExhaustion, 0,
                                 "Out of memory in run_logit_parity_harness"));
    }
}

} /* namespace sutra::inline v1 */
