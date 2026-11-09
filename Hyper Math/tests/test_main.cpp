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

#include "hyper_math/hyper_math.hpp"

#include <iostream>

int run_tensor_tests ();
int run_scientific_tests ();
int run_tabular_tests ();

int
main ()
{
  std::cout << "========================================================\n"
            << " Running " << hyper_math::get_version_string ()
            << " Verification Suite\n"
            << "========================================================\n";

  int failures = 0;
  failures += run_tensor_tests ();
  failures += run_scientific_tests ();
  failures += run_tabular_tests ();

  std::cout << "========================================================\n";
  if (failures == 0)
    {
      std::cout << " PASS: All Hyper Math test suites succeeded (0 failures)\n"
                << "========================================================\n";
      return 0;
    }
  else
    {
      std::cerr << " FAIL: " << failures << " test(s) failed\n"
                << "========================================================\n";
      return 1;
    }
}
