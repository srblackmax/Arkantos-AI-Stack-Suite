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

#include <yantra/yantra.hpp>

#include <atomic>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace
{

auto
read_rss_mb () -> double
{
  std::ifstream in ("/proc/self/statm");
  std::size_t virt_pages = 0U;
  std::size_t rss_pages = 0U;
  if (in >> virt_pages >> rss_pages)
    {
      return static_cast<double> (rss_pages * 4096U) / (1024.0 * 1024.0);
    }
  return 0.0;
}

/* Stress 7.3.1: 1M metric points logged across 1,000 runs with bounded RSS. */
void
stress_1m_metrics_1000_runs ()
{
  const fs::path wal_path
      = fs::temp_directory_path () / "yantra_stress_1m_wal.bin";
  std::error_code ec;
  fs::remove (wal_path, ec);

  auto wal_res = yantra::WalEngine::open (wal_path, true);
  assert (wal_res.has_value ());
  auto &wal = *wal_res;

  constexpr std::size_t kRuns = 1000U;
  constexpr std::size_t kPointsPerRun = 1000U;
  std::vector<yantra::WalRecord> batch (kPointsPerRun);

  const double rss_before = read_rss_mb ();
  for (std::size_t r = 0U; r < kRuns; ++r)
    {
      for (std::size_t s = 0U; s < kPointsPerRun; ++s)
        {
          batch[s].run_id_num = r + 1U;
          batch[s].step = s;
          batch[s].timestamp_ns = s * 1000U;
          batch[s].scalar_value = static_cast<double> (s) * 0.001;
          batch[s].key_id = 1U;
        }
      auto app_res = wal.append_batch (batch, false);
      assert (app_res.has_value ());
    }
  const double rss_after = read_rss_mb ();
  assert ((rss_after - rss_before) < 32.0);

  auto sz = wal.file_size_bytes ();
  assert (sz.has_value () && *sz == 64ULL + 1000000ULL * 64ULL);
  fs::remove (wal_path, ec);
}

/* Stress 7.3.2: 100 concurrent DAG executions across threads with zero
   deadlocks or task loss.  */
void
stress_100_concurrent_dags ()
{
  yantra::DagScheduler sched (8U);
  yantra::PipelineDefinition dag;

  for (int i = 0; i < 12; ++i)
    {
      yantra::Task t{};
      t.id = "task_" + std::to_string (i);
      if (i > 0)
        {
          t.dependencies.push_back ("task_" + std::to_string (i / 2));
        }
      t.run = [] (yantra::TaskContext &)
          -> yantra::Result<std::vector<std::string>> {
        return std::vector<std::string>{};
      };
      assert (dag.add_task (std::move (t)).has_value ());
    }

  std::atomic<int> completed_dags{ 0 };
  constexpr int kThreads = 10;
  constexpr int kDagsPerThread = 10;
  std::vector<std::jthread> callers;
  callers.reserve (kThreads);

  for (int c = 0; c < kThreads; ++c)
    {
      callers.emplace_back ([&] () {
        for (int d = 0; d < kDagsPerThread; ++d)
          {
            auto rep = sched.execute (dag);
            assert (rep.has_value () && rep->all_succeeded);
            assert (rep->results.size () == 12U);
            completed_dags.fetch_add (1, std::memory_order_relaxed);
          }
      });
    }
  callers.clear ();
  assert (completed_dags.load () == 100);
}

/* Stress 7.3.3 & 7.3.4: Concurrent vector search across 8 threads + soak RSS
   stability check.  */
void
stress_concurrent_vector_queries ()
{
  constexpr std::size_t kN = 1024U;
  constexpr std::size_t kD = 32U;
  std::mt19937 rng (777U);
  std::uniform_real_distribution<float> dist (-1.0F, 1.0F);
  std::vector<float> vecs (kN * kD);
  for (float &v : vecs)
    {
      v = dist (rng);
    }

  yantra::HnswIndex hnsw (kD, { 12U, 64U, 32U });
  assert (hnsw.add_batch (vecs, kN).has_value ());

#if defined(__SANITIZE_THREAD__) || defined(__SANITIZE_ADDRESS__)
  constexpr std::size_t kTotalQueries = 20000U;
#else
  constexpr std::size_t kTotalQueries = 100000U;
#endif

  const double rss_start = read_rss_mb ();
  std::atomic<std::size_t> q_counter{ 0U };
  std::vector<std::jthread> readers;
  readers.reserve (8U);
  for (std::size_t t = 0U; t < 8U; ++t)
    {
      readers.emplace_back ([&, t] () {
        for (;;)
          {
            const std::size_t q
                = q_counter.fetch_add (1U, std::memory_order_relaxed);
            if (q >= kTotalQueries)
              {
                break;
              }
            const std::size_t idx = (q + t * 17U) % kN;
            std::span<const float> qvec (vecs.data () + idx * kD, kD);
            auto res = hnsw.search (qvec, 5U, 16U);
            assert (res.has_value () && res->size () == 5U);
          }
      });
    }
  readers.clear ();

  const double rss_end = read_rss_mb ();
  assert (rss_end <= rss_start * 1.10 + 8.0);
}

} /* namespace */

auto
main () -> int
{
  stress_1m_metrics_1000_runs ();
  stress_100_concurrent_dags ();
  stress_concurrent_vector_queries ();
  std::cout << "All Yantra stress and memory stability tests PASSED.\n";
  return 0;
}
