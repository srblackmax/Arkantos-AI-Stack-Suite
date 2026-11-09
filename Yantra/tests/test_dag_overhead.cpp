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

#undef NDEBUG
#include <yantra/dag.hpp>
#include <yantra/observability.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace
{

auto
quantile (std::vector<double> vals, double q) -> double
{
  if (vals.empty ())
    {
      return 0.0;
    }
  std::sort (vals.begin (), vals.end ());
  const double pos = q * static_cast<double> (vals.size () - 1U);
  const auto lo = static_cast<std::size_t> (std::floor (pos));
  const auto hi = static_cast<std::size_t> (std::ceil (pos));
  const double frac = pos - static_cast<double> (lo);
  return vals[lo] * (1.0 - frac) + vals[hi] * frac;
}

} /* namespace */

auto
main () -> int
{
  constexpr std::size_t kDagNodes = 100U;

  yantra::PipelineDefinition pipe_single;
  {
    yantra::Task t;
    t.id = "t_0";
    t.resources.cpu_cores = 1U;
    t.run = [] (yantra::TaskContext &)
        -> yantra::Result<std::vector<std::string>> {
      return std::vector<std::string>{};
    };
    (void)pipe_single.add_task (std::move (t));
  }

  std::array<std::atomic<uint32_t>, kDagNodes> task_exec_counts{};
  yantra::PipelineDefinition pipe_100_indep;
  for (std::size_t i = 0U; i < kDagNodes; ++i)
    {
      yantra::Task t;
      t.id = "t_" + std::to_string (i);
      t.resources.cpu_cores = 1U;
      t.run = [&task_exec_counts, i] (yantra::TaskContext &)
          -> yantra::Result<std::vector<std::string>> {
        task_exec_counts[i].fetch_add (1U, std::memory_order_relaxed);
        return std::vector<std::string>{};
      };
      (void)pipe_100_indep.add_task (std::move (t));
    }

  yantra::DagScheduler sched_w1 (1U, { 20U, 8ULL << 30 });
  yantra::DagScheduler sched_w4 (4U, { 20U, 8ULL << 30 });
  yantra::DagScheduler sched_w8 (8U, { 20U, 8ULL << 30 });
  yantra::DagScheduler sched_w12 (12U, { 20U, 8ULL << 30 });

  for (int w = 0; w < 500; ++w)
    {
      if (!sched_w1.execute (pipe_single).has_value ())
        {
          return 1;
        }
    }

  std::vector<double> single_task_samples_us;
  for (int iter = 0; iter < 20; ++iter)
    {
      constexpr int kSubBatch = 25;
      const uint64_t t0 = yantra::monotonic_time_ns ();
      for (int s = 0; s < kSubBatch; ++s)
        {
          auto r = sched_w1.execute (pipe_single);
          if (!r.has_value () || !r->all_succeeded || r->results.size () != 1U)
            {
              return 1;
            }
        }
      const uint64_t t1 = yantra::monotonic_time_ns ();
      single_task_samples_us.push_back (
          static_cast<double> (t1 - t0) / (1.0e3 * kSubBatch));
    }
  const double single_task_us = quantile (single_task_samples_us, 0.50);

  auto run_100_indep = [&] (yantra::DagScheduler &s) -> double {
    std::vector<double> samples_ms;
    for (int iter = 0; iter < 60; ++iter)
      {
        for (auto &c : task_exec_counts)
          {
            c.store (0U, std::memory_order_relaxed);
          }
        const uint64_t t0 = yantra::monotonic_time_ns ();
        auto rep = s.execute (pipe_100_indep);
        const uint64_t t1 = yantra::monotonic_time_ns ();
        if (!rep.has_value () || !rep->all_succeeded
            || rep->results.size () != kDagNodes)
          {
            return 1.0e9;
          }
        for (std::size_t i = 0U; i < kDagNodes; ++i)
          {
            if (task_exec_counts[i].load (std::memory_order_relaxed) != 1U)
              {
                return 1.0e9;
              }
          }
        if (iter >= 20)
          {
            samples_ms.push_back (static_cast<double> (t1 - t0) / 1.0e6);
          }
      }
    return quantile (samples_ms, 0.50);
  };

  const double w4_ms = run_100_indep (sched_w4);
  const double w8_ms = run_100_indep (sched_w8);
  const double w12_ms = run_100_indep (sched_w12);

  const double ideal_w4_ms = (single_task_us * 25.0) / 1000.0;
  const double ideal_w8_ms = (single_task_us * 13.0) / 1000.0;
  const double ideal_w12_ms = (single_task_us * 9.0) / 1000.0;

  const double ovh_w4 = (w4_ms / ideal_w4_ms) - 1.0;
  const double ovh_w8 = (w8_ms / ideal_w8_ms) - 1.0;
  const double ovh_w12 = (w12_ms / ideal_w12_ms) - 1.0;

  std::fprintf (stderr,
                "[test_dag_overhead] single_us=%.4f w4_ms=%.4f (ovh=%.4f) "
                "w8_ms=%.4f (ovh=%.4f) w12_ms=%.4f (ovh=%.4f)\n",
                single_task_us, w4_ms, ovh_w4, w8_ms, ovh_w8, w12_ms, ovh_w12);

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
  constexpr double kMaxOvhW8 = 15.0;
  constexpr double kMaxOvhOther = 25.0;
#else
  constexpr double kMaxOvhW8 = 0.20;
  constexpr double kMaxOvhOther = 0.30;
#endif
  if (ovh_w8 > kMaxOvhW8 || ovh_w4 > kMaxOvhOther || ovh_w12 > kMaxOvhOther)
    {
      std::fprintf (stderr,
                    "[test_dag_overhead] FAILED (target w8<=%.2f, "
                    "w4/w12<=%.2f)\n",
                    kMaxOvhW8, kMaxOvhOther);
      return 1;
    }

  return 0;
}
