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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cli_common.hpp"

namespace
{

void
print_help ()
{
  std::cout
    << "Usage: bellows-infer [OPTION]...\n"
    << "Load a .blws model artifact and generate predictions.\n\n"
    << "Options:\n"
    << "  --model FILE     Path to a trained .blws model artifact\n"
    << "  --input FILE     Path to an unlabeled CSV feature table\n"
    << "  --row VALUES     Comma-separated single observation for low-latency infer\n"
    << "  --help           Display this help and exit\n"
    << "  --version        Output version information and exit\n";
}

} // namespace

int
main (int argc, char **argv)
{
  std::string model_path;
  std::string input_path;
  std::string single_row_str;

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
          std::cout << bellows::package_version_banner ("bellows-infer");
          return 0;
        }
      if (arg == "--model" && i + 1 < argc)
        {
          model_path = argv[++i];
        }
      else if (arg == "--input" && i + 1 < argc)
        {
          input_path = argv[++i];
        }
      else if (arg == "--row" && i + 1 < argc)
        {
          single_row_str = argv[++i];
        }
      else
        {
          bellows::emit_cli_error ("bellows-infer",
                                    "unrecognised or incomplete option '" + arg
                                      + "'");
          return 1;
        }
    }

  if (model_path.empty () || (input_path.empty () && single_row_str.empty ()))
    {
      bellows::emit_cli_error (
        "bellows-infer",
        "'--model FILE' and either '--input FILE' or '--row VALUES' are required");
      return 1;
    }

  try
    {
      std::ifstream peek_in (model_path, std::ios::binary);
      if (!peek_in)
        {
          bellows::emit_cli_error ("bellows-infer",
                                    "cannot open model '" + model_path + "'");
          return 1;
        }
      std::uint32_t magic = 0;
      std::uint32_t version = 0;
      std::uint32_t tag = 0;
      peek_in.read (reinterpret_cast<char *> (&magic), sizeof (magic));
      peek_in.read (reinterpret_cast<char *> (&version), sizeof (version));
      peek_in.read (reinterpret_cast<char *> (&tag), sizeof (tag));
      peek_in.close ();

      auto run_with_model = [&] (auto &booster)
      {
        bellows::load_artifact_file (booster, model_path);
        if (!single_row_str.empty ())
          {
            std::vector<double> row = bellows::parse_csv_line (single_row_str);
            std::cout << bellows::infer_one (booster, row) << "\n";
          }
        else
          {
            bellows::data_table table = bellows::load_unlabeled_csv (input_path);
            auto preds = bellows::predict (booster, table);
            for (double p : preds)
              {
                std::cout << p << "\n";
              }
          }
      };

      if (tag == 2U)
        {
          bellows::stagewise_booster<bellows::logistic_bernoulli_loss> booster;
          run_with_model (booster);
        }
      else if (tag == 3U)
        {
          bellows::stagewise_booster<bellows::multinomial_deviance_loss>
            booster;
          run_with_model (booster);
        }
      else
        {
          bellows::stagewise_booster<bellows::squared_residual_loss> booster;
          run_with_model (booster);
        }
    }
  catch (const std::exception &ex)
    {
      bellows::emit_cli_error ("bellows-infer", ex.what ());
      return 1;
    }

  return 0;
}
