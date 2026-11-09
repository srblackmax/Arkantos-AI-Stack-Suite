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

#include <sphinx/sphinx.hpp>

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

auto
main () -> int
{
  const std::filesystem::path sb = "/tmp/sphinx_stress_sandbox";
  const std::filesystem::path store_dir = "/tmp/sphinx_stress_yantra";
  std::error_code ec;
  std::filesystem::remove_all (sb, ec);
  std::filesystem::remove_all (store_dir, ec);

  if (!sphinx::ToolEnvironment::set_sandbox_dir (sb))
    {
      return 1;
    }

  auto store_res = yantra::ExperimentStore::open (store_dir);
  if (!store_res)
    {
      return 1;
    }
  auto run_res = store_res->create_run ("stress_run");
  if (!run_res)
    {
      return 1;
    }
  (void)run_res->start ();
  sphinx::ToolEnvironment::bind_yantra_run (&(*run_res), 1U);

  sutra::ToolRegistry reg;
  if (!sphinx::register_all_sphinx_tools (reg))
    {
      return 1;
    }

  std::atomic<bool> failed{ false };
  std::vector<std::jthread> workers;
  workers.reserve (8U);

  for (std::size_t tid = 0U; tid < 8U; ++tid)
    {
      workers.emplace_back ([&, tid] {
        sutra::ToolContext ctx{};
        for (std::size_t iter = 0U; iter < 25U; ++iter)
          {
            const std::string fname = "file_" + std::to_string (tid) + "_"
                                      + std::to_string (iter) + ".txt";
            const std::string w_args
                = "{\"path\":\"" + fname + "\",\"content\":\"stress_test\"}";
            if (!reg.invoke ("write_file", ctx, w_args))
              {
                failed.store (true, std::memory_order_release);
                return;
              }
            const std::string r_args
                = "{\"path\":\"" + fname + "\",\"max_lines\":5}";
            if (!reg.invoke ("read_file", ctx, r_args))
              {
                failed.store (true, std::memory_order_release);
                return;
              }
          }
      });
    }

  for (auto &w : workers)
    {
      w.join ();
    }

  sphinx::ToolEnvironment::bind_yantra_run (nullptr, 0U);
  (void)run_res->finish ();
  (void)store_res->compact ();

  return failed.load (std::memory_order_acquire) ? 1 : 0;
}
