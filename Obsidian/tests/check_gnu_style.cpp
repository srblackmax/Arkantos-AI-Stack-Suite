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

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace
{

const std::string_view k_required_gpl_notice =
  "/*\n"
  " * Copyright (C) 2026 Kival\n"
  " * This file is part of the Arkantos AI Stack Suite.\n"
  " *\n"
  " * This program is free software: you can redistribute it and/or modify\n"
  " * it under the terms of the GNU General Public License as published by\n"
  " * the Free Software Foundation, either version 3 of the License, or\n"
  " * (at your option) any later version.\n"
  " *\n"
  " * This program is distributed in the hope that it will be useful,\n"
  " * but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
  " * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the\n"
  " * GNU General Public License for more details.\n"
  " *\n"
  " * You should have received a copy of the GNU General Public License\n"
  " * along with this program. If not, see <https://www.gnu.org/licenses/>.\n"
  " */";

bool
has_checked_extension (const fs::path &p)
{
  std::string ext = p.extension ().string ();
  return ext == ".h" || ext == ".hpp" || ext == ".cpp" || ext == ".in";
}

void
check_file (const fs::path &path, bool is_public_header,
            std::vector<std::string> &errors)
{
  std::ifstream ifs (path);
  if (!ifs)
    {
      errors.push_back (path.string () + ": failed to open file");
      return;
    }

  std::ostringstream buffer;
  buffer << ifs.rdbuf ();
  std::string text = buffer.str ();

  if (text.find (k_required_gpl_notice) == std::string::npos)
    {
      errors.push_back (path.string () + ": missing GPLv3 license header");
    }

  if (is_public_header && path.filename () != "obsidian.hpp")
    {
      if (text.find ("@file") == std::string::npos
          || text.find ("@brief") == std::string::npos)
        {
          errors.push_back (
            path.string () + ": missing Doxygen @file/@brief tags");
        }
    }

  std::istringstream lines (text);
  std::string line;
  size_t lineno = 0;
  while (std::getline (lines, line))
    {
      ++lineno;
      if (!line.empty () && line.back () == '\r')
        {
          line.pop_back ();
        }
      if (line.find ('\t') != std::string::npos)
        {
          errors.push_back (path.string () + ":" + std::to_string (lineno)
                            + ": tab character found");
        }
      if (!line.empty () && (line.back () == ' ' || line.back () == '\t'))
        {
          errors.push_back (path.string () + ":" + std::to_string (lineno)
                            + ": trailing whitespace");
        }
      if (line.size () > 79)
        {
          errors.push_back (path.string () + ":" + std::to_string (lineno)
                            + ": line length " + std::to_string (line.size ())
                            + " exceeds 79 columns");
        }
    }
}

} /* namespace */

int
main (int argc, char **argv)
{
  fs::path root = ".";
  if (argc > 1)
    {
      root = argv[1];
    }
  else if (const char *srcdir = std::getenv ("srcdir"))
    {
      root = srcdir;
    }

  const std::vector<std::string> dirs = {
    "include", "src", "tests", "examples", "bench"
  };

  std::vector<std::string> errors;
  size_t files_checked = 0;

  for (const auto &d : dirs)
    {
      fs::path base = root / d;
      if (!fs::exists (base))
        {
          continue;
        }
      std::vector<fs::path> found;
      for (const auto &entry : fs::recursive_directory_iterator (base))
        {
          if (entry.is_regular_file ()
              && has_checked_extension (entry.path ()))
            {
              found.push_back (entry.path ());
            }
        }
      std::sort (found.begin (), found.end ());
      for (const auto &p : found)
        {
          ++files_checked;
          check_file (p, d == "include", errors);
        }
    }

  if (files_checked == 0)
    {
      std::cerr << "check_gnu_style: no source files found in "
                << root.string () << "\n";
      return EXIT_FAILURE;
    }

  if (!errors.empty ())
    {
      for (const auto &err : errors)
        {
          std::cerr << err << "\n";
        }
      return EXIT_FAILURE;
    }

  std::cout << "check_gnu_style: verified " << files_checked
            << " C++ files with 0 violations.\n";
  return EXIT_SUCCESS;
}
