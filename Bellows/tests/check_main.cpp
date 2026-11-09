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

#include <iostream>
#include <string>

#include <bellows/bellows.hpp>

namespace bellows_test
{

int run_core_tests ();
int run_preprocess_tests ();
int run_estimator_tests ();
int run_boosting_tests ();
int run_composition_tests ();
int run_io_and_cli_tests ();

} // namespace bellows_test

int
main (int argc, char **argv)
{
  for (int i = 1; i < argc; ++i)
    {
      std::string arg = argv[i];
      if (arg == "--help" || arg == "-h")
        {
          std::cout
            << "Usage: bellows-check [OPTION]...\n"
            << "Execute the Bellows unit and integration verification suite.\n\n"
            << "Options:\n"
            << "  --help      Display this help and exit\n"
            << "  --version   Output version information and exit\n";
          return 0;
        }
      if (arg == "--version" || arg == "-v")
        {
          std::cout << bellows::package_version_banner ("bellows-check");
          return 0;
        }
    }

  int failures = 0;
  std::cout << "=== Bellows Verification Suite (bellows-check) ===\n";

  failures += bellows_test::run_core_tests ();
  failures += bellows_test::run_preprocess_tests ();
  failures += bellows_test::run_estimator_tests ();
  failures += bellows_test::run_boosting_tests ();
  failures += bellows_test::run_composition_tests ();
  failures += bellows_test::run_io_and_cli_tests ();

  if (failures == 0)
    {
      std::cout << "=== ALL BELLOWS TESTS PASSED ===\n";
      return 0;
    }

  std::cerr << "bellows-check: " << failures << " test suite(s) failed\n";
  return 1;
}
