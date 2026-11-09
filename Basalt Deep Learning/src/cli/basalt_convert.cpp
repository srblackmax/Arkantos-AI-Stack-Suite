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
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <map>
#include <string>
#include <basalt/basalt.hpp>

namespace
{

const char *const k_program_name = "basalt-convert";

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
    << "Usage: " << k_program_name << " [OPTION]... -i INPUT -o OUTPUT\n"
    << "Convert model weights between Basalt (.bckpt), Safetensors "
       "(.safetensors),\nand ONNX (.onnx) formats.\n\n"
    << "  -i, --input=FILE       source artifact file\n"
    << "  -o, --output=FILE      destination artifact file\n"
    << "  -n, --dry-run          verify conversion plan without writing\n"
    << "  -v, --verbose          print conversion details\n"
    << "  -q, --quiet            suppress standard output\n"
    << "  -h, --help             display this help and exit\n"
    << "  -V, --version          output version information and exit\n\n"
    << "Report bugs to <Kival@basalt-dl.org>.\n";
}

bool
ends_with (const std::string &s, const std::string &suffix)
{
  return s.size () >= suffix.size ()
         && s.compare (s.size () - suffix.size (),
                       suffix.size (), suffix) == 0;
}

class Dict_Module : public basalt::nn::Module
{
public:
  explicit Dict_Module (const std::map<std::string, basalt::Tensor> &tensors)
  {
    for (const auto &kv : tensors)
      register_parameter (kv.first, kv.second, false);
  }
};

std::map<std::string, basalt::Tensor>
load_any_tensors (const std::string &path)
{
  std::ifstream in (path, std::ios::binary);
  if (!in)
    BASALT_THROW (basalt::IO_Error, "cannot open '" + path + "'");
  uint32_t magic = 0;
  in.read (reinterpret_cast<char *> (&magic), sizeof (magic));
  in.close ();

  if (magic == basalt::io::k_checkpoint_magic || ends_with (path, ".bckpt"))
    {
      basalt::io::Checkpoint ckpt = basalt::io::load_checkpoint (path);
      return ckpt.model_state;
    }
  if (ends_with (path, ".onnx"))
    return basalt::io::import_onnx_weights (path);
  return basalt::io::load_safetensors (path);
}

void
save_any_tensors (const std::string &path,
                  const std::map<std::string, basalt::Tensor> &tensors)
{
  if (ends_with (path, ".bckpt"))
    {
      basalt::io::Checkpoint ckpt;
      ckpt.model_state = tensors;
      basalt::io::save_checkpoint (path, ckpt);
    }
  else if (ends_with (path, ".onnx"))
    {
      Dict_Module mod (tensors);
      basalt::io::export_onnx (path, mod);
    }
  else
    {
      basalt::io::save_safetensors (path, tensors);
    }
}

} /* anonymous namespace */

int
main (int argc, char **argv)
{
  std::string input_path;
  std::string output_path;
  bool dry_run = false;
  bool verbose = false;
  bool quiet = false;

  static const struct option long_options[] = {
    {"input", required_argument, nullptr, 'i'},
    {"output", required_argument, nullptr, 'o'},
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
            argc, argv, "i:o:nvqhV", long_options, nullptr)) != -1)
    {
      switch (opt)
        {
        case 'i':
          input_path = optarg;
          break;
        case 'o':
          output_path = optarg;
          break;
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

  if (input_path.empty () && optind < argc)
    input_path = argv[optind++];
  if (output_path.empty () && optind < argc)
    output_path = argv[optind++];

  if (input_path.empty () || output_path.empty ())
    {
      basalt::report_tool_diagnostic (
        k_program_name, "both input and output files are required");
      return EXIT_FAILURE;
    }

  try
    {
      if (dry_run)
        {
          if (!quiet)
            std::cout << k_program_name << ": would convert '" << input_path
                      << "' -> '" << output_path << "'\n";
          return EXIT_SUCCESS;
        }

      auto tensors = load_any_tensors (input_path);
      save_any_tensors (output_path, tensors);
      if (verbose && !quiet)
        std::cout << "Converted " << tensors.size () << " tensors from '"
                  << input_path << "' to '" << output_path << "'\n";
      return EXIT_SUCCESS;
    }
  catch (const std::exception &e)
    {
      basalt::report_tool_diagnostic (k_program_name, e.what ());
      return EXIT_FAILURE;
    }
}
