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

#ifndef OBSIDIAN_TEST_HARNESS_HPP
#define OBSIDIAN_TEST_HARNESS_HPP

/**
 * @file test_harness.hpp
 * @brief Minimal header-only C++23 unit test harness for Obsidian.
 */

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace obsidian::testing
{

struct Test_Case
{
  std::string name;
  std::function<void (void)> fn;
};

inline std::vector<Test_Case> &
registry (void)
{
  static std::vector<Test_Case> cases;
  return cases;
}

inline int &
failure_count (void)
{
  static int failures = 0;
  return failures;
}

struct Registrar
{
  Registrar (const char *name, std::function<void (void)> fn)
  {
    registry ().push_back (Test_Case {name, std::move (fn)});
  }
};

inline void
report_failure (const char *file, int line, const char *expr,
                const std::string &detail = "")
{
  ++failure_count ();
  std::cerr << file << ":" << line << ": FAIL: " << expr;
  if (!detail.empty ())
    {
      std::cerr << " (" << detail << ")";
    }
  std::cerr << "\n";
}

inline int
run_all_tests (const char *suite_name)
{
  std::cout << "=== Running suite: " << suite_name << " ===\n";
  size_t passed = 0;
  for (const auto &tc : registry ())
    {
      int before = failure_count ();
      try
        {
          tc.fn ();
        }
      catch (const std::exception &e)
        {
          report_failure (__FILE__, __LINE__, tc.name.c_str (), e.what ());
        }
      if (failure_count () == before)
        {
          ++passed;
          std::cout << "  [PASS] " << tc.name << "\n";
        }
      else
        {
          std::cout << "  [FAIL] " << tc.name << "\n";
        }
    }
  std::cout << "=== Summary: " << passed << "/" << registry ().size ()
            << " passed, " << failure_count () << " failed ===\n";
  return (failure_count () == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

} /* namespace obsidian::testing */

#define OBSIDIAN_TEST(name) \
  static void test_fn_##name (void); \
  static ::obsidian::testing::Registrar reg_##name (#name, test_fn_##name); \
  static void test_fn_##name (void)

#define OBSIDIAN_CHECK(cond) \
  do \
    { \
      if (!(cond)) \
        { \
          ::obsidian::testing::report_failure (__FILE__, __LINE__, #cond); \
          return; \
        } \
    } \
  while (0)

#define OBSIDIAN_CHECK_EQ(a, b) \
  do \
    { \
      if (!((a) == (b))) \
        { \
          ::obsidian::testing::report_failure ( \
            __FILE__, __LINE__, #a " == " #b); \
          return; \
        } \
    } \
  while (0)

#define OBSIDIAN_CHECK_NEAR(a, b, tol) \
  do \
    { \
      double diff_val_ = std::fabs (static_cast<double> (a) \
                                    - static_cast<double> (b)); \
      if (diff_val_ > static_cast<double> (tol)) \
        { \
          ::obsidian::testing::report_failure ( \
            __FILE__, __LINE__, #a " ~= " #b, \
            "diff=" + std::to_string (diff_val_)); \
          return; \
        } \
    } \
  while (0)

#define OBSIDIAN_CHECK_OK(res_expr) \
  do \
    { \
      if (!(res_expr).has_value ()) \
        { \
          ::obsidian::testing::report_failure ( \
            __FILE__, __LINE__, #res_expr, \
            (res_expr).error ().format ()); \
          return; \
        } \
    } \
  while (0)

#endif /* OBSIDIAN_TEST_HARNESS_HPP */
