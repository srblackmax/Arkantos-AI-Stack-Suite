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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "cli_common.hpp"

namespace
{

void
print_help ()
{
  std::cout
    << "Usage: bellows [OPTION]...\n"
    << "Inspect Bellows model artefacts and library capabilities.\n\n"
    << "Options:\n"
    << "  --inspect FILE   Verify and summarise a binary .blws model file\n"
    << "  --info           Display available concepts and boosting policies\n"
    << "  --help           Display this help and exit\n"
    << "  --version        Output version information and exit\n";
}

int
inspect_artifact (const std::string &path)
{
  std::ifstream in (path, std::ios::binary);
  if (!in)
    {
      bellows::emit_cli_error ("bellows",
                                "cannot open '" + path + "' for inspection");
      return 1;
    }

  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint32_t tag = 0;
  std::uint64_t length = 0;
  std::uint64_t checksum = 0;

  in.read (reinterpret_cast<char *> (&magic), sizeof (magic));
  in.read (reinterpret_cast<char *> (&version), sizeof (version));
  in.read (reinterpret_cast<char *> (&tag), sizeof (tag));
  in.read (reinterpret_cast<char *> (&length), sizeof (length));
  in.read (reinterpret_cast<char *> (&checksum), sizeof (checksum));

  if (!in || magic != bellows::artifact_magic)
    {
      bellows::emit_cli_error ("bellows",
                                path + ": invalid Bellows artifact header");
      return 1;
    }
  if (version != bellows::artifact_format_version)
    {
      bellows::emit_cli_error (
        "bellows", path + ": unsupported artifact format version");
      return 1;
    }

  std::string payload (static_cast<std::size_t> (length), '\0');
  in.read (payload.data (), static_cast<std::streamsize> (length));
  if (!in)
    {
      bellows::emit_cli_error ("bellows",
                                path + ": truncated artifact payload");
      return 1;
    }

  std::uint64_t actual =
    bellows::compute_fnv1a_checksum (payload.data (), payload.size ());
  if (actual != checksum)
    {
      bellows::emit_cli_error ("bellows",
                                path + ": artifact checksum mismatch");
      return 1;
    }

  std::cout << "Artifact: " << path << "\n"
            << "Format version: " << version << "\n"
            << "Model tag: " << tag << "\n"
            << "Payload bytes: " << length << "\n"
            << "Checksum (FNV-1a): 0x" << std::hex << checksum << std::dec
            << " (verified)\n";
  return 0;
}

} // namespace

int
main (int argc, char **argv)
{
  for (int i = 1; i < argc; ++i)
    {
      std::string arg = argv[i];
      if (arg == "--help" || arg == "-h")
        {
          print_help ();
          return 0;
        }
      if (arg == "--version" || arg == "-v")
        {
          std::cout << bellows::package_version_banner ("bellows");
          return 0;
        }
      if (arg == "--info")
        {
          std::cout << "Bellows " << bellows::version_string ()
                    << " — C++20 Concept Lattice ML Library\n"
                    << "Cut policies: exact_cut_policy, quantile_bin_policy\n"
                    << "Growth policies: depth_synchronous_growth, "
                        "best_gain_frontier_growth, "
                        "oblivious_symmetric_growth\n"
                    << "Sampling policies: uniform_full_sample, "
                        "gradient_magnitude_filter\n";
          return 0;
        }
      if (arg == "--inspect")
        {
          if (i + 1 >= argc)
            {
              bellows::emit_cli_error ("bellows",
                                        "option '--inspect' requires an argument");
              return 1;
            }
          return inspect_artifact (argv[i + 1]);
        }
      bellows::emit_cli_error ("bellows", "unrecognised option '" + arg + "'");
      return 1;
    }

  print_help ();
  return 0;
}
