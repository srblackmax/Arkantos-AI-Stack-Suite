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

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{

constexpr std::size_t kMaxLineColumns = 79U;

[[nodiscard]] auto
check_file (const std::filesystem::path &path, bool is_public_header,
            bool is_lib_or_src) -> int
{
  std::ifstream in (path);
  if (!in.is_open ())
    {
      (void)std::fprintf (stderr, "Failed to open %s\n",
                          path.string ().c_str ());
      return 1;
    }

  int errors = 0;
  std::string line;
  std::size_t line_no = 0U;
  bool saw_gpl = false;

  while (std::getline (in, line))
    {
      ++line_no;
      if (!line.empty () && line.back () == '\r')
        {
          line.pop_back ();
        }
      if (line.find ("GNU General Public License") != std::string::npos)
        {
          saw_gpl = true;
        }
      if (line.size () > kMaxLineColumns)
        {
          (void)std::fprintf (stderr,
                              "%s:%zu: line exceeds 79 columns (%zu)\n",
                              path.string ().c_str (), line_no, line.size ());
          ++errors;
        }
      if (line.find ('\t') != std::string::npos)
        {
          (void)std::fprintf (stderr, "%s:%zu: tab character found\n",
                              path.string ().c_str (), line_no);
          ++errors;
        }
      if (!line.empty () && (line.back () == ' ' || line.back () == '\t'))
        {
          (void)std::fprintf (stderr, "%s:%zu: trailing whitespace\n",
                              path.string ().c_str (), line_no);
          ++errors;
        }
      if (is_public_header
          && line.find ("#pragma once") != std::string::npos)
        {
          (void)std::fprintf (stderr, "%s:%zu: #pragma once forbidden\n",
                              path.string ().c_str (), line_no);
          ++errors;
        }
      if (is_public_header)
        {
          constexpr std::array<std::string_view, 6U> kForbiddenHeaderSyms
              = { { "whisper_", "piper_", "onnxruntime_", "espeak_",
                    "snd_pcm_", "libssh2_" } };
          for (const std::string_view sym : kForbiddenHeaderSyms)
            {
              // Allow comments mentioning zero symbols, reject code usages.
              if (line.find (sym) != std::string::npos
                  && line.find ("*") == std::string::npos
                  && line.find ("//") == std::string::npos)
                {
                  (void)std::fprintf (
                      stderr, "%s:%zu: leaked backend symbol '%.*s'\n",
                      path.string ().c_str (), line_no,
                      static_cast<int> (sym.size ()), sym.data ());
                  ++errors;
                }
            }
        }
      if (is_lib_or_src)
        {
          if (line.find ("std::cout") != std::string::npos
              || line.find ("std::endl") != std::string::npos
              || line.find ("std::any") != std::string::npos
              || line.find ("new ") != std::string::npos)
            {
              (void)std::fprintf (
                  stderr, "%s:%zu: forbidden construct in library source\n",
                  path.string ().c_str (), line_no);
              ++errors;
            }
        }
    }

  if (!saw_gpl)
    {
      (void)std::fprintf (stderr, "%s: missing GPLv3 license header\n",
                          path.string ().c_str ());
      ++errors;
    }
  return errors;
}

} /* namespace */

auto
main (int argc, char **argv) -> int
{
  const std::filesystem::path root
      = (argc > 1) ? std::filesystem::path (argv[1])
                   : std::filesystem::current_path ();
  int total_errors = 0;

  for (const std::string_view sub : { "include", "src", "tests", "bench" })
    {
      const std::filesystem::path dir = root / sub;
      if (!std::filesystem::exists (dir))
        {
          continue;
        }
      for (const auto &entry :
           std::filesystem::recursive_directory_iterator (dir))
        {
          if (!entry.is_regular_file ())
            {
              continue;
            }
          const auto ext = entry.path ().extension ().string ();
          if (ext != ".hpp" && ext != ".cpp" && ext != ".h")
            {
              continue;
            }
          const bool is_hdr = (sub == "include");
          const bool is_src = (sub == "include" || sub == "src");
          total_errors += check_file (entry.path (), is_hdr, is_src);
        }
    }

  return (total_errors == 0) ? 0 : 1;
}
