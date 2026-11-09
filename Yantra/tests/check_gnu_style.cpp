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

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

constexpr std::size_t kMaxLineLength = 79U;
constexpr std::size_t kHeaderScanLines = 20U;

auto
is_checked_extension (const fs::path &p) -> bool
{
  const std::string ext = p.extension ().string ();
  return ext == ".hpp" || ext == ".h" || ext == ".cpp" || ext == ".cc"
         || ext == ".cu";
}

auto
verify_file (const fs::path &filepath, const std::string &subdir) -> int
{
  std::ifstream in (filepath);
  if (!in.is_open ())
    {
      std::cerr << "check_gnu_style: cannot open " << filepath << "\n";
      return 1;
    }

  const bool is_lib_code = (subdir == "include" || subdir == "src");
  const bool is_pub_hdr = (subdir == "include");

  int violations = 0;
  std::size_t line_no = 0U;
  bool found_gpl = false;
  std::string line;

  while (std::getline (in, line))
    {
      ++line_no;
      if (!line.empty () && line.back () == '\r')
        {
          line.pop_back ();
        }

      if (line_no <= kHeaderScanLines
          && line.find ("GNU General Public License") != std::string::npos)
        {
          found_gpl = true;
        }

      if (line.find ('\t') != std::string::npos)
        {
          std::cerr << filepath.string () << ":" << line_no
                    << ": error: tab character found\n";
          ++violations;
        }

      if (line.size () > kMaxLineLength)
        {
          std::cerr << filepath.string () << ":" << line_no
                    << ": error: line length " << line.size ()
                    << " exceeds 79 columns\n";
          ++violations;
        }

      if (!line.empty () && line.back () == ' ')
        {
          std::cerr << filepath.string () << ":" << line_no
                    << ": error: trailing whitespace\n";
          ++violations;
        }

      if (is_lib_code)
        {
          if (line.find ("std::cout") != std::string::npos
              || line.find ("std::endl") != std::string::npos
              || line.find ("printf (") != std::string::npos)
            {
              std::cerr << filepath.string () << ":" << line_no
                        << ": error: forbidden console I/O in library code\n";
              ++violations;
            }

          if (line.find ("std::any") != std::string::npos)
            {
              std::cerr << filepath.string () << ":" << line_no
                        << ": error: forbidden std::any type erasure in "
                           "library code\n";
              ++violations;
            }

          if (line.find ("new ") != std::string::npos
              && line.find ("//") > line.find ("new "))
            {
              std::cerr << filepath.string () << ":" << line_no
                        << ": error: forbidden raw new in library code\n";
              ++violations;
            }
        }

      if (is_pub_hdr && line.find ("#include") != std::string::npos)
        {
          if (line.find ("cuda") != std::string::npos
              || line.find ("surya") != std::string::npos
              || line.find ("immintrin.h") != std::string::npos)
            {
              std::cerr << filepath.string () << ":" << line_no
                        << ": error: forbidden non-std include in public "
                           "header\n";
              ++violations;
            }
        }
    }

  if (!found_gpl)
    {
      std::cerr << filepath.string ()
                << ":1: error: missing GPLv3 header banner\n";
      ++violations;
    }

  return violations;
}

} /* namespace */

auto
main (int argc, char **argv) -> int
{
  fs::path root = fs::current_path ();
  if (argc > 1)
    {
      root = fs::path (argv[1]);
    }

  const std::vector<std::string> subdirs
      = { "include", "src", "tests", "bench" };
  int total_violations = 0;
  std::size_t files_checked = 0U;

  for (const std::string &sub : subdirs)
    {
      const fs::path dir = root / sub;
      if (!fs::exists (dir))
        {
          continue;
        }
      for (const auto &entry : fs::recursive_directory_iterator (dir))
        {
          if (entry.is_regular_file ())
            {
              if (entry.path ().extension () == ".py")
                {
                  std::cerr << entry.path ().string ()
                            << ": error: forbidden Python file in Yantra\n";
                  ++total_violations;
                }
              else if (is_checked_extension (entry.path ()))
                {
                  ++files_checked;
                  total_violations += verify_file (entry.path (), sub);
                }
            }
        }
    }

  if (total_violations > 0)
    {
      std::cerr << "GNU style check FAILED with " << total_violations
                << " violation(s) across " << files_checked << " files.\n";
      return 1;
    }

  std::cout << "GNU style check PASSED (" << files_checked
            << " C++/CUDA files verified).\n";
  return 0;
}
