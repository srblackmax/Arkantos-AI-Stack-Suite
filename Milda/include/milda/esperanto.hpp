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

#ifndef MILDA_ESPERANTO_HPP
#define MILDA_ESPERANTO_HPP

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace milda
{

/* Pinned Esperanto system prompt locked at KV index 0.  */
inline constexpr std::string_view k_esperanto_system_prompt
    = "Vi estas Milda, varma, sprita kaj pensiga homa konversacianto kaj "
      "proksima amiko. Babilu nature, kore kaj egale kun via amiko en "
      "simplaj, fluaj konversaciaj alineoj—neniam kiel roboto, AI-asistanto "
      "aŭ programada agento. Neniam uzu Markdown-formatadon (*, -, listojn "
      "aŭ kodblokojn), miensignojn (emojis), aŭ robotajn frazojn, kaj "
      "neniam demandu 'Kiel mi povas helpi vin?'. Respondu rekte kaj "
      "koncize ekskluzive en idiomeca Esperanto laŭ la Fundamento kun "
      "ĝustaj akuzativoj (-n), supersignoj (ĉ, ĝ, ĥ, ĵ, ŝ, ŭ), kaj "
      "korektaj tabelvortoj (iu, io, kia, kio), sen miksi aliajn lingvojn.";

/* Interactive REPL command kinds in Esperanto and polyglot modes.  */
enum class repl_command_kind : uint8_t
{
  none = 0U,
  eliri = 1U,
  vakigi = 2U,
  stat = 3U,
  helpo = 4U,
  konservi = 5U,
  sxargi = 6U,
  lang = 7U,
  x_system = 8U
};

/* Single-pass, branch-minimal, zero-allocation in-place preprocessor
   converting ASCII x-sistemo digraphs (cx, gx, hx, jx, sx, ux and
   uppercase counterparts) into canonical 2-byte UTF-8 Esperanto
   characters, with xx literal escape support.
   Returns the updated byte length of buf (always <= len).  */
[[nodiscard]] std::size_t
transliterate_x_system_in_place (char *buf, std::size_t len) noexcept;

/* Parses a trimmed user input line for built-in Esperanto REPL commands
   (/eliri, /vakigi, /stat, /konservi, /sxargi, /helpo).  */
[[nodiscard]] repl_command_kind
parse_repl_command (std::string_view line) noexcept;

/* Extracts the optional argument (e.g. file path) following a REPL
   command such as "/konservi memoro.bin".  */
[[nodiscard]] std::string_view
extract_repl_command_arg (std::string_view line) noexcept;

/* Trims leading and trailing ASCII whitespace from a string_view.  */
[[nodiscard]] std::string_view
trim_ascii_whitespace (std::string_view text) noexcept;

} /* namespace milda */

#endif /* MILDA_ESPERANTO_HPP */
