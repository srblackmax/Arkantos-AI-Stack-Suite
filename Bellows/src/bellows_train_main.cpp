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
#include <iostream>
#include <string>
#include <vector>

#include "cli_common.hpp"

namespace
{

void
print_help ()
{
  std::cout
    << "Usage: bellows-train [OPTION]...\n"
    << "Calibrate a Bellows stagewise booster and save a .blws artifact.\n\n"
    << "Options:\n"
    << "  --input FILE     Input CSV file (last column is target)\n"
    << "  --output FILE    Destination .blws model artifact path\n"
    << "  --task MODE      Task mode: 'regression' (default) or 'binary'\n"
    << "  --stages COUNT   Number of boosting stages (default: 40)\n"
    << "  --rate VALUE     Step shrinkage rate (default: 0.1)\n"
    << "  --help           Display this help and exit\n"
    << "  --version        Output version information and exit\n";
}

} // namespace

int
main (int argc, char **argv)
{
  std::string input_path;
  std::string output_path;
  std::string task_mode = "regression";
  std::size_t stages = 40;
  double rate = 0.1;

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
          std::cout << bellows::package_version_banner ("bellows-train");
          return 0;
        }
      if (arg == "--input" && i + 1 < argc)
        {
          input_path = argv[++i];
        }
      else if (arg == "--output" && i + 1 < argc)
        {
          output_path = argv[++i];
        }
      else if (arg == "--task" && i + 1 < argc)
        {
          task_mode = argv[++i];
        }
      else if (arg == "--stages" && i + 1 < argc)
        {
          stages = static_cast<std::size_t> (std::stoul (argv[++i]));
        }
      else if (arg == "--rate" && i + 1 < argc)
        {
          rate = std::stod (argv[++i]);
        }
      else
        {
          bellows::emit_cli_error ("bellows-train",
                                    "unrecognised or incomplete option '" + arg
                                      + "'");
          return 1;
        }
    }

  if (input_path.empty () || output_path.empty ())
    {
      bellows::emit_cli_error (
        "bellows-train", "both '--input FILE' and '--output FILE' are required");
      return 1;
    }
  if (stages == 0 || rate <= 0.0)
    {
      bellows::emit_cli_error (
        "bellows-train", "'--stages' and '--rate' must be strictly positive");
      return 1;
    }

  try
    {
      auto [table, targets] = bellows::load_labeled_csv (input_path);
      if (table.row_count () == 0)
        {
          bellows::emit_cli_error ("bellows-train",
                                    input_path + ": dataset contains no rows");
          return 1;
        }
      if (task_mode == "binary")
        {
          bellows::stagewise_booster<bellows::logistic_bernoulli_loss> booster (
            bellows::with_stage_count{stages}, bellows::with_step_rate{rate});
          bellows::fit (booster, table, targets);
          bellows::save_artifact_file (booster, output_path, 2U);
        }
      else if (task_mode == "multiclass")
        {
          bellows::stagewise_booster<bellows::multinomial_deviance_loss>
            booster (bellows::with_stage_count{stages},
                      bellows::with_step_rate{rate});
          bellows::fit (booster, table, targets);
          bellows::save_artifact_file (booster, output_path, 3U);
        }
      else if (task_mode == "regression")
        {
          bellows::stagewise_booster<bellows::squared_residual_loss> booster (
            bellows::with_stage_count{stages}, bellows::with_step_rate{rate});
          bellows::fit (booster, table, targets);
          bellows::save_artifact_file (booster, output_path, 1U);
        }
      else
        {
          bellows::emit_cli_error (
            "bellows-train",
            "unsupported task mode '" + task_mode
              + "' (expected 'regression', 'binary', or 'multiclass')");
          return 1;
        }
    }
  catch (const std::exception &ex)
    {
      bellows::emit_cli_error ("bellows-train", ex.what ());
      return 1;
    }

  return 0;
}
