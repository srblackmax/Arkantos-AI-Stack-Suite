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

#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <string>
#include <basalt/basalt.hpp>

namespace
{

const char *const k_program_name = "basalt-inspect";

void
print_version (void)
{
  std::cout
    << "Basalt Deep Learning " << basalt_version_string () << "\n"
    << "Copyright (C) 2026 Kival\n"
    << "License GPLv3+: GNU GPL version 3 or later "
       "<https://gnu.org/licenses/gpl.html>\n"
    << "This is free software: you are free to change and redistribute it.\n"
    << "There is NO WARRANTY, to the extent permitted by law.\n";
}

void
print_help (void)
{
  std::cout
    << "Usage: " << k_program_name << " [OPTION]... FILE...\n"
    << "Inspect checkpoints (.bckpt), Safetensors (.safetensors), or ONNX "
       "(.onnx) files.\n\n"
    << "  -n, --dry-run          verify arguments without reading files\n"
    << "  -v, --verbose          print extended tensor metadata\n"
    << "  -q, --quiet            suppress stdout, only check validity\n"
    << "  -h, --help             display this help and exit\n"
    << "  -V, --version          output version information and exit\n\n"
    << "Report bugs to <Kival@basalt-dl.org>.\n";
}

} /* anonymous namespace */

int
main (int argc, char **argv)
{
  bool dry_run = false;
  bool verbose = false;
  bool quiet = false;

  static const struct option long_options[] = {
    {"dry-run", no_argument, nullptr, 'n'},
    {"verbose", no_argument, nullptr, 'v'},
    {"quiet", no_argument, nullptr, 'q'},
    {"help", no_argument, nullptr, 'h'},
    {"version", no_argument, nullptr, 'V'},
    {nullptr, 0, nullptr, 0}
  };

  opterr = 0;
  int opt = 0;
  while ((opt = getopt_long (
            argc, argv, "nvqhV", long_options, nullptr)) != -1)
    {
      switch (opt)
        {
        case 'n':
          dry_run = true;
          break;
        case 'v':
          verbose = true;
          break;
        case 'q':
          quiet = true;
          break;
        case 'h':
          print_help ();
          return EXIT_SUCCESS;
        case 'V':
          print_version ();
          return EXIT_SUCCESS;
        default:
          basalt::report_tool_diagnostic (
            k_program_name, "unrecognized option; try '--help'");
          return EXIT_FAILURE;
        }
    }

  if (optind >= argc)
    {
      basalt::report_tool_diagnostic (
        k_program_name, "missing file operand; try '--help'");
      return EXIT_FAILURE;
    }

  (void) verbose;
  try
    {
      for (int i = optind; i < argc; ++i)
        {
          std::string path = argv[i];
          if (dry_run)
            {
              if (!quiet)
                std::cout << k_program_name << ": would inspect '"
                          << path << "'\n";
              continue;
            }
          std::string report = basalt::io::inspect_artifact (path);
          if (!quiet)
            std::cout << report;
        }
      return EXIT_SUCCESS;
    }
  catch (const std::exception &e)
    {
      basalt::report_tool_diagnostic (k_program_name, e.what ());
      return EXIT_FAILURE;
    }
}
