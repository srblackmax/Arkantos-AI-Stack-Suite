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

#include <sphinx/sphinx.hpp>

#include <array>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <string_view>

auto
main () -> int
{
  const std::filesystem::path sb = "/tmp/sphinx_prop_sandbox";
  std::error_code ec;
  std::filesystem::remove_all (sb, ec);
  auto sb_res = sphinx::ToolEnvironment::set_sandbox_dir (sb);
  if (!sb_res)
    {
      return 1;
    }

  std::mt19937_64 rng (0x535048494E58ULL);
  constexpr std::string_view kChars
      = "abcdefghijklmnopqrstuvwxyz0123456789/._-;|$`>< ";

  for (std::size_t iter = 0U; iter < 500U; ++iter)
    {
      std::string rand_str;
      const std::size_t len = (rng () % 48U) + 1U;
      rand_str.reserve (len);
      for (std::size_t i = 0U; i < len; ++i)
        {
          rand_str.push_back (kChars[rng () % kChars.size ()]);
        }

      // Property 1: Any resolved path must start with canonical sandbox_dir.
      auto p_res = sphinx::resolve_sandboxed_path (rand_str);
      if (p_res)
        {
          const std::string resolved = p_res->string ();
          const std::string base = sb_res->string ();
          if (resolved != base && !resolved.starts_with (base + "/"))
            {
              (void)std::fprintf (stderr, "Sandbox invariant violated: %s\n",
                                  resolved.c_str ());
              return 1;
            }
        }

      // Property 2: Any command containing shell metacharacters is rejected.
      const std::string injected = "ls ; " + rand_str;
      if (sphinx::validate_allowed_command (injected))
        {
          (void)std::fprintf (stderr, "Command injection not rejected\n");
          return 1;
        }

      // Property 3: Any SSH command containing dangerous patterns is rejected.
      const std::string ssh_bad = "echo hi && rm -rf " + rand_str;
      if (sphinx::validate_ssh_command (ssh_bad))
        {
          (void)std::fprintf (stderr, "SSH dangerous pattern not rejected\n");
          return 1;
        }

      // Property 4: Sentence splitter index is either npos or <= size().
      const std::size_t bound = sphinx::find_next_sentence_boundary (rand_str);
      if (bound != std::string_view::npos && bound > rand_str.size ())
        {
          (void)std::fprintf (stderr, "Sentence boundary out of range\n");
          return 1;
        }
    }

  return 0;
}
