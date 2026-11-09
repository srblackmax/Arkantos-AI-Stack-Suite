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

#include "milda/esperanto.hpp"

namespace milda
{

namespace
{

/* Lookup table entry mapping an ASCII base character to its 2-byte
   canonical UTF-8 Esperanto supersigno sequence.  */
struct utf8_pair
{
  char first;
  char second;
  bool valid;
};

[[nodiscard]] constexpr utf8_pair
lookup_esperanto_utf8 (char base_ch) noexcept
{
  utf8_pair out = utf8_pair ();
  out.first = '\0';
  out.second = '\0';
  out.valid = false;

  switch (base_ch)
    {
    case 'c':
      out.first = static_cast<char> (0xC4U);
      out.second = static_cast<char> (0x89U);
      out.valid = true;
      break;
    case 'C':
      out.first = static_cast<char> (0xC4U);
      out.second = static_cast<char> (0x88U);
      out.valid = true;
      break;
    case 'g':
      out.first = static_cast<char> (0xC4U);
      out.second = static_cast<char> (0x9DU);
      out.valid = true;
      break;
    case 'G':
      out.first = static_cast<char> (0xC4U);
      out.second = static_cast<char> (0x9CU);
      out.valid = true;
      break;
    case 'h':
      out.first = static_cast<char> (0xC4U);
      out.second = static_cast<char> (0xA5U);
      out.valid = true;
      break;
    case 'H':
      out.first = static_cast<char> (0xC4U);
      out.second = static_cast<char> (0xA4U);
      out.valid = true;
      break;
    case 'j':
      out.first = static_cast<char> (0xC4U);
      out.second = static_cast<char> (0xB5U);
      out.valid = true;
      break;
    case 'J':
      out.first = static_cast<char> (0xC4U);
      out.second = static_cast<char> (0xB4U);
      out.valid = true;
      break;
    case 's':
      out.first = static_cast<char> (0xC5U);
      out.second = static_cast<char> (0x9DU);
      out.valid = true;
      break;
    case 'S':
      out.first = static_cast<char> (0xC5U);
      out.second = static_cast<char> (0x9CU);
      out.valid = true;
      break;
    case 'u':
      out.first = static_cast<char> (0xC5U);
      out.second = static_cast<char> (0xADU);
      out.valid = true;
      break;
    case 'U':
      out.first = static_cast<char> (0xC5U);
      out.second = static_cast<char> (0xACU);
      out.valid = true;
      break;
    default:
      break;
    }
  return out;
}

[[nodiscard]] constexpr bool
is_x_marker (char ch) noexcept
{
  return (ch == 'x') | (ch == 'X');
}

[[nodiscard]] constexpr bool
is_ascii_space (char ch) noexcept
{
  return (ch == ' ') | (ch == '\t') | (ch == '\n') | (ch == '\r');
}

} /* namespace */

std::size_t
transliterate_x_system_in_place (char *buf, std::size_t len) noexcept
{
  if (buf == nullptr || len == 0U)
    {
      return 0U;
    }

  std::size_t read_idx = 0U;
  std::size_t write_idx = 0U;

  while (read_idx < len)
    {
      const char ch = buf[read_idx];
      if (read_idx + 1U < len)
        {
          const char next_ch = buf[read_idx + 1U];
          if (is_x_marker (next_ch))
            {
              const utf8_pair mapped = lookup_esperanto_utf8 (ch);
              if (mapped.valid)
                {
                  if (read_idx + 2U < len
                      && is_x_marker (buf[read_idx + 2U]))
                    {
                      /* Escaped digraph (e.g. cxx -> cx, uxx -> ux).  */
                      buf[write_idx] = ch;
                      buf[write_idx + 1U] = next_ch;
                      write_idx += 2U;
                      read_idx += 3U;
                      continue;
                    }
                  /* Canonical 2-byte UTF-8 replacement in-place.  */
                  buf[write_idx] = mapped.first;
                  buf[write_idx + 1U] = mapped.second;
                  write_idx += 2U;
                  read_idx += 2U;
                  continue;
                }
              if (is_x_marker (ch))
                {
                  /* Standalone xx / XX literal escape -> single x / X.  */
                  buf[write_idx] = ch;
                  ++write_idx;
                  read_idx += 2U;
                  continue;
                }
            }
        }

      buf[write_idx] = ch;
      ++write_idx;
      ++read_idx;
    }

  if (write_idx < len)
    {
      buf[write_idx] = '\0';
    }
  return write_idx;
}

std::string_view
trim_ascii_whitespace (std::string_view text) noexcept
{
  std::size_t start = 0U;
  while (start < text.size () && is_ascii_space (text[start]))
    {
      ++start;
    }
  std::size_t end = text.size ();
  while (end > start && is_ascii_space (text[end - 1U]))
    {
      --end;
    }
  return text.substr (start, end - start);
}

repl_command_kind
parse_repl_command (std::string_view line) noexcept
{
  const std::string_view trimmed = trim_ascii_whitespace (line);
  if (trimmed.empty () || trimmed[0] != '/')
    {
      return repl_command_kind::none;
    }

  std::size_t cmd_end = 0U;
  while (cmd_end < trimmed.size () && !is_ascii_space (trimmed[cmd_end]))
    {
      ++cmd_end;
    }
  const std::string_view head = trimmed.substr (0U, cmd_end);

  if (head == "/eliri" || head == "/quit" || head == "/exit")
    {
      return repl_command_kind::eliri;
    }
  if (head == "/vakigi" || head == "/clear")
    {
      return repl_command_kind::vakigi;
    }
  if (head == "/stat")
    {
      return repl_command_kind::stat;
    }
  if (head == "/helpo" || head == "/help")
    {
      return repl_command_kind::helpo;
    }
  if (head == "/konservi" || head == "/save")
    {
      return repl_command_kind::konservi;
    }
  if (head == "/sxargi" || head == "/\305\235argi" || head == "/load")
    {
      return repl_command_kind::sxargi;
    }
  if (head == "/lang" || head == "/lingvo")
    {
      return repl_command_kind::lang;
    }
  if (head == "/x-system" || head == "/xsistemo")
    {
      return repl_command_kind::x_system;
    }
  return repl_command_kind::none;
}

std::string_view
extract_repl_command_arg (std::string_view line) noexcept
{
  const std::string_view trimmed = trim_ascii_whitespace (line);
  std::size_t cmd_end = 0U;
  while (cmd_end < trimmed.size () && !is_ascii_space (trimmed[cmd_end]))
    {
      ++cmd_end;
    }
  if (cmd_end >= trimmed.size ())
    {
      return std::string_view ();
    }
  return trim_ascii_whitespace (trimmed.substr (cmd_end));
}

} /* namespace milda */
